import zou_lab_control_v2

"""The temperature Task, end to end on the virtual bench.

The physics under test IS the product's purpose.  A real calibration measures
the sites and their thresholds; the temperature Task then plays the release
template with the board advancing ``t_off`` from its own scan table, pairs the
two probe windows of every fired cycle, and reports how fast the atoms stop
coming back.  The virtual world plants a trap-off lifetime, so the fitted
decay must be that constant -- read from the world's own ground truth, never
written down here.

What this file guards, beyond "it runs":

* the PAIRING is per site and per shot: survival is 1, 0, or "not a fact",
  never a brightness ratio, and the published rate IS the pooled fraction of
  the loaded sites that came back;
* the SITE axis survives into the output, so one dead trap can still be seen;
* the release times reach the dataset as its point axis, with their unit;
* the Task's results are FINAL: they are still on the plane after the run is
  terminal and the host is shut down, which is what keeps a panel alive;
* the saved artifact's fit is the same curve the dataset published, in
  SECONDS, and it matches the planted lifetime.
"""

import json
from pathlib import Path
import time

import numpy as np
from zlc_data import SITE
from zlc_pulse import compile_sequence, resolve_api_parameters, sequence_from_tree
from zlc_runtime import NodeHost, SignalDataPlane

from zlc_atom.install import create_installation
from zlc_atom.nodes import (
    ResolvedWorkspaceResource,
    discover_logic_nodes,
    temperature_pulse_template_bytes,
)
from zlc_atom.nodes.scan import SCAN_PULSE_CONTRACT
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE


#: The release times played, in the template's own unit (ms), and how many
#: whole sweeps of them.  Four points over one and a half lifetimes, eight
#: sweeps: enough loaded sites per point that the fitted slope is the physics
#: and not the counting noise.
T_OFF_MS = (0.5, 1.0, 2.0, 3.0)
REPEATS = 8
#: The readout exposure the calibration is measured at IS the probe window the
#: template opens, so the frames the Task judges are the frames its thresholds
#: were trained on.
EXPOSURE_SECONDS = 0.005


def _wait_terminal(host: object, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not host.observation.terminal:
        host.poll()
        time.sleep(0.005)
    observed = host.observation
    assert observed.error is None, observed.error
    assert observed.terminal, "the host never finished"


def test_the_temperature_task_recovers_the_planted_lifetime(tmp_path: Path) -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    sequencer = installation.device("sequencer")
    camera = installation.device("camera")
    monitor = None
    host = None
    try:
        # --- 1. A real calibration: the site map and thresholds the Task reads.
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
        _wait_terminal(calibration_host, timeout=120.0)
        calibration_host.shutdown()
        artifact_path = calibration_node.result.artifact_path
        calibration = descriptors["temperature"].input_specs[1].codec.resolve(
            artifact_path
        )

        # --- 2. The site camera as a two-frame-per-cycle monitor: the probe
        #        pair of one release, published as one cycle.
        camera_node = descriptors["camera_measurement"].instantiate(
            camera=camera,
            camera_key="camera",
            signal_plane=plane,
            repeat=0,
            frames_per_cycle=2,
            exposure_seconds=EXPOSURE_SECONDS,
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
            if plane.freeze().value(frames_signal) is not None:
                break
            time.sleep(0.02)
        assert plane.freeze().value(frames_signal) is not None, (
            "the temperature template never produced a two-frame cycle"
        )

        # --- 3. The Task itself.
        task = descriptors["temperature"].instantiate(
            sequencer=sequencer,
            signal_plane=plane,
            source_signal=frames_signal,
            calibration=calibration,
            pulse_resource=ResolvedWorkspaceResource(
                Path("temperature_template.json"),
                SCAN_PULSE_CONTRACT,
                sequence,
            ),
            artifact_directory=tmp_path,
            t_off=", ".join(str(value) for value in T_OFF_MS),
            repeats=REPEATS,
        )
        host = NodeHost(task, plane)
        host.start()
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline and not host.observation.terminal:
            monitor.poll()
            plane.freeze()
            host.poll()
        observed = host.observation
        assert observed.error is None, observed.error
        assert observed.terminal

        rate_signal = host.signal_key("survival_rate")
        survival_value = plane.freeze().value(host.signal_key("survival"))
        rate_value = plane.freeze().value(rate_signal)
        assert survival_value is not None and rate_value is not None, (
            "the Task published no survival"
        )

        # --- The axes: repeats x release times x sites, and the site axis is
        #     kept so one dead trap is still visible.
        schema = survival_value.snapshot.block.schema
        assert schema.repeat_axis.size == REPEATS
        assert schema.point_table.row_count == len(T_OFF_MS)
        column = schema.point_table.columns[0]
        assert column.name == "t_off"
        assert column.unit == "ms"
        assert tuple(column.values) == T_OFF_MS
        site_axes = tuple(
            axis for axis in schema.cell_schema.data_axes if axis.role == SITE
        )
        assert len(site_axes) == 1, (
            "survival must keep the site axis, so a per-site answer stays "
            f"readable; the cell axes are {schema.cell_schema.data_axes}"
        )
        assert site_axes[0].size == calibration.value.n_sites

        # --- The pairing: one loaded site either came back or it did not, and
        #     a site that held nothing answers nothing.
        survival = np.asarray(survival_value.block.values, dtype=float)
        assert survival.shape == (REPEATS, len(T_OFF_MS), calibration.value.n_sites)
        judged = survival[np.isfinite(survival)]
        assert judged.size, "no site was loaded anywhere in the run"
        assert set(np.unique(judged).tolist()) <= {0.0, 1.0}, (
            "survival is a per-site recapture, not a ratio of brightness: "
            f"{np.unique(judged)[:8].tolist()}"
        )
        assert np.isnan(survival).any(), (
            "an empty trap answers nothing about recapture and must stay NaN"
        )

        # --- The rate IS the pooled fraction of the loaded sites, per shot.
        rate = np.asarray(rate_value.block.values, dtype=float).reshape(
            REPEATS, len(T_OFF_MS)
        )
        with np.errstate(invalid="ignore"):
            per_shot = np.nanmean(survival, axis=2)
        np.testing.assert_allclose(rate, per_shot, rtol=0, atol=1e-12)

        # --- The physics: recapture falls with the release time, at the rate
        #     the world planted.  Every non-NaN survival entry is one loaded
        #     site, so the mean over repeats and sites IS the pooled fraction.
        pooled = np.nanmean(survival, axis=(0, 2))
        assert np.all(np.isfinite(pooled)), (
            f"a release time recaptured nothing: {pooled.tolist()}"
        )
        assert np.all(np.diff(pooled) < 0), (
            f"survival must fall with t_off: {pooled.round(3).tolist()}"
        )
        lifetime_seconds = float(installation.world.trap_off_lifetime_s)
        slope = float(
            np.polyfit(np.asarray(T_OFF_MS) * 1e-3, np.log(pooled), 1)[0]
        )
        planted = -1.0 / lifetime_seconds
        assert abs(slope - planted) <= 0.35 * abs(planted), (
            f"measured decay {slope:.1f}/s against a planted {planted:.1f}/s; "
            f"survival={pooled.round(3).tolist()}"
        )

        # --- The artifact: the same curve, fitted in SECONDS, with the
        #     temperature that decay implies.
        result = host.final_result
        saved = Path(result["artifact_path"])
        assert saved.is_file() and saved.parent == tmp_path
        payload = json.loads(saved.read_text(encoding="utf-8"))
        assert payload["t_off"] == {"unit": "ms", "values": list(T_OFF_MS)}
        fit = payload["run_record"]["fit"]
        np.testing.assert_allclose(fit["survival_rate"], pooled, rtol=0, atol=1e-12)
        assert sum(fit["loaded_pairs"]) == judged.size
        assert abs(fit["lifetime_seconds"] - lifetime_seconds) <= (
            0.35 * lifetime_seconds
        ), (
            f"the saved lifetime {fit['lifetime_seconds']:.2e}s is not the "
            f"planted {lifetime_seconds:.2e}s"
        )
        assert fit["temperature_kelvin"] > 0
        np.testing.assert_allclose(
            fit["temperature_kelvin"],
            result["temperature_kelvin"],
            rtol=1e-12,
        )

        # --- A Task's results outlive its run: the panel that watched them is
        #     still holding data after the host is gone.
        assert rate_value.coverage is None and not rate_value.transient
        host.shutdown()
        host = None
        assert plane.freeze().value(rate_signal) is not None, (
            "the survival rate was withdrawn when the Task finished, so every "
            "panel watching it went blank"
        )
    finally:
        if host is not None and not host.observation.terminal:
            host.cancel("test cleanup")
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not host.observation.terminal:
                host.poll()
        if host is not None:
            host.shutdown()
        if monitor is not None:
            monitor.close()
        plane.close()
        installation.close()
