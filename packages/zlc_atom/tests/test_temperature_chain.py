import zou_lab_control_v2

"""The temperature chain, end to end on the virtual bench.

The physics under test IS the product's purpose: the temperature template
probes the traps twice around a variable release (``t_off``), the occupancy
processor turns each cycle's two frames into a survival fraction, and the
scan sweeps ``t_off`` -- STREAMED, from the board's own scan table, which is
the seamless mode the template was designed for.  The world plants a trap-off
lifetime; the measured survival curve must decay with that constant, computed
from the ground truth rather than hard-coded.
"""

import json
from pathlib import Path
import time

import numpy as np
from zlc_pulse import compile_sequence, resolve_api_parameters, sequence_from_tree
from zlc_runtime import NodeHost, SignalDataPlane

from zlc_atom.install import create_installation
from zlc_atom.nodes import (
    ResolvedWorkspaceResource,
    discover_logic_nodes,
    temperature_pulse_template_bytes,
)
from zlc_atom.nodes.scan import PULSE_PARAM_FAMILY, ScanAxis, ScanPlan
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE


def _wait_terminal(host: object, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not host.observation.terminal:
        host.poll()
        time.sleep(0.005)
    observed = host.observation
    assert observed.error is None, observed.error
    assert observed.terminal, "the host never finished"


def test_the_temperature_chain_recovers_the_planted_lifetime(tmp_path: Path) -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {d.api_name: d for d in discover_logic_nodes()}
    sequencer = installation.device("sequencer")
    camera = installation.device("camera")
    monitor = None
    occupancy_host = None
    scan_host = None
    try:
        # --- 1. A real calibration: the occupancy's site map and thresholds.
        calibration_node = descriptors["calibration"].instantiate(
            camera=camera,
            camera_key="camera",
            sequencer=sequencer,
            sequencer_key="sequencer",
            pulse_resource=IMAGING_PULSE_RESOURCE,
            artifact_directory=tmp_path,
            repeats=30,
        )
        calibration_host = NodeHost(calibration_node, plane)
        calibration_host.start()
        _wait_terminal(calibration_host, timeout=90.0)
        calibration_host.shutdown()
        artifact_path = calibration_node.result.artifact_path

        # --- 2. The site camera as a two-frame-per-cycle monitor.
        camera_node = descriptors["camera_measurement"].instantiate(
            camera=camera,
            camera_key="camera",
            signal_plane=plane,
            repeat=0,
            frames_per_cycle=2,
            exposure_seconds=0.005,
        )
        monitor = camera_node.monitor()
        frames_signal = camera_node.signal_key("frames")

        sequence = sequence_from_tree(
            json.loads(temperature_pulse_template_bytes().decode("utf-8"))
        )
        board = sequencer.describe()
        seeded = resolve_api_parameters(sequence)
        sequencer.load(
            compile_sequence(seeded, board.geometry, board.clock_hz),
            source=seeded,
        )
        sequencer.fire()
        sequencer.wait_done(5.0)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            monitor.poll()
            front = plane.freeze()
            if front.value(frames_signal) is not None:
                break
            time.sleep(0.02)
        assert plane.freeze().value(frames_signal) is not None, (
            "the temperature template never produced a two-frame cycle"
        )

        # --- 3. Occupancy as a LIVE processor over the monitor's frames.
        codec = descriptors["occupancy"].input_specs[1].codec
        occupancy_node = descriptors["occupancy"].instantiate(
            calibration=codec.resolve(artifact_path),
            source_signal=frames_signal,
            model_kind="default",
        )
        occupancy_host = NodeHost(occupancy_node, plane)
        occupancy_host.start()
        survival_signal = occupancy_host.signal_key("survival_rate")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            monitor.poll()
            plane.freeze()
            occupancy_host.poll()
            if plane.freeze().value(survival_signal) is not None:
                break
            time.sleep(0.02)
        assert plane.freeze().value(survival_signal) is not None, (
            "occupancy never published a live survival_rate"
        )

        # --- 4. The streamed t_off scan, watched on survival_rate.
        t_offs_ms = (0.2, 0.6, 1.2, 2.4)
        samples = 6
        plan = ScanPlan(
            (ScanAxis(PULSE_PARAM_FAMILY + "t_off", t_offs_ms),)
        )
        scan_node = descriptors["scan"].instantiate(
            sequencer=sequencer,
            signal_plane=plane,
            source_signal=survival_signal,
            pulse_resource=ResolvedWorkspaceResource(
                Path("temperature_template.json"),
                "zlc.pulse/scan-template",
                sequence,
            ),
            plan=plan.to_tree(),
            samples_per_point=samples,
            capture="direct",
            advance="streamed",
        )
        scan_host = NodeHost(scan_node, plane)
        scan_host.start()
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline and not scan_host.observation.terminal:
            monitor.poll()
            plane.freeze()
            occupancy_host.poll()
            scan_host.poll()
        observed = scan_host.observation
        assert observed.error is None, observed.error
        assert observed.terminal

        value = plane.freeze().value(scan_host.signal_key("scan"))
        assert value is not None, "the scan published nothing"
        block = np.asarray(value.block.values, dtype=float)
        # (repeat visits, t_off points, survival pairs): six looks per point,
        # one probe pair per cycle.
        assert block.shape[:2] == (samples, len(t_offs_ms))
        survival = np.nanmean(
            block.reshape(samples, len(t_offs_ms), -1), axis=(0, 2)
        )
        assert np.all(np.isfinite(survival)), (
            f"survival has empty points: {survival.tolist()}"
        )
        assert np.all(np.diff(survival) < 0), (
            f"survival must fall with t_off: {survival.round(3).tolist()}"
        )
        # The decay constant, against the world's OWN planted lifetime.
        lifetime_ms = float(installation.world.trap_off_lifetime_s) * 1e3
        slope = np.polyfit(np.asarray(t_offs_ms), np.log(survival), 1)[0]
        expected = -1.0 / lifetime_ms
        assert abs(slope - expected) <= 0.4 * abs(expected), (
            f"measured decay {slope:.3f}/ms, planted {expected:.3f}/ms; "
            f"survival={survival.round(3).tolist()}"
        )
    finally:
        for host in (scan_host, occupancy_host):
            if host is not None and not host.observation.terminal:
                host.cancel("test cleanup")
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline and not host.observation.terminal:
                    host.poll()
        if scan_host is not None:
            scan_host.shutdown()
        if occupancy_host is not None:
            occupancy_host.shutdown()
        if monitor is not None:
            monitor.close()
        plane.close()
        installation.close()
