"""The general scan node: its plan, its ports, and a real optimisation.

The end-to-end test IS the goal this node was built for: scan the three bias
DACs on the virtual bench and find the MOT optimum the simulation planted.
Nothing here reads the world's ground truth except to say where the answer
should have landed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from zlc_pulse import resolve_api_parameters, sequence_from_tree
from zlc_runtime import NodeHost, SignalDataPlane

from zlc_atom.devices.simulation import DEFAULT_MOT_FIELD_OPTIMUM_DAC
from zlc_atom.install import create_installation, tunable_devices
from zlc_atom.nodes import ResolvedWorkspaceResource, discover_logic_nodes
from zlc_atom.nodes.scan import (
    CAPTURE_MODES,
    DEVICE_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    SCAN_SCHEMA,
    ScanAxis,
    ScanPlan,
    bind_plan,
    scan_ports_for,
    scan_ports_for_devices,
)


from zlc_atom.nodes import scan_pulse_template_bytes

TEMPLATE_NAME = "mot_field_template.json"
BIAS_PORTS = tuple(
    PULSE_PARAM_FAMILY + name
    for name in ("da_bias_x", "da_bias_y", "da_bias_z")
)


def _template_sequence():
    return sequence_from_tree(json.loads(scan_pulse_template_bytes().decode("utf-8")))


def test_the_mot_template_offers_the_three_bias_ports() -> None:
    """Ports come from the pulse's own declarations, nothing invented."""

    ports = scan_ports_for(_template_sequence())
    assert tuple(port.port for port in ports) == BIAS_PORTS
    for port in ports:
        assert port.unit == "", "a DAC code is dimensionless; the unit is empty"
        assert port.lo < 0 < port.hi, "the signed range brackets zero"


def test_the_capture_mode_is_the_operators_authored_choice() -> None:
    """How freshness is taken is DECLARED on the node, never probed.

    The safe default skips the straddler (right for a free-running monitor
    like the MOT camera); a pulse-driven source may keep every publication.
    """

    field = next(f for f in SCAN_SCHEMA.fields if f.name == "capture")
    assert field.value_type == "choice"
    assert field.default == "skip_one"
    assert tuple(choice.value for choice in field.choices) == CAPTURE_MODES
    with pytest.raises(ValueError, match="must be one of"):
        SCAN_SCHEMA.project_values(
            {"pulse_template": "t.json", "plan": "{}", "capture": "guess"}
        )


def test_plan_rows_nest_outer_first_and_round_trip() -> None:
    plan = ScanPlan(
        (
            ScanAxis(BIAS_PORTS[0], (1.0, 2.0)),
            ScanAxis(BIAS_PORTS[1], (10.0, 20.0, 30.0)),
        )
    )
    assert plan.shape == (2, 3)
    assert plan.point_count == 6
    rows = plan.rows()
    # The declared order is the nesting order: the LAST axis advances fastest.
    assert rows == (
        (1.0, 10.0), (1.0, 20.0), (1.0, 30.0),
        (2.0, 10.0), (2.0, 20.0), (2.0, 30.0),
    )
    assert ScanPlan.from_tree(plan.to_tree()) == plan


def test_binding_refuses_unknown_ports_and_out_of_range_values() -> None:
    ports = scan_ports_for(_template_sequence())

    with pytest.raises(ValueError, match="offers no scan port named"):
        bind_plan(ScanPlan((ScanAxis("pulse:param:nonsense", (1.0,)),)), ports)

    with pytest.raises(ValueError, match="outside the port's range"):
        bind_plan(ScanPlan((ScanAxis(BIAS_PORTS[0], (1e9,)),)), ports)

    bound = bind_plan(
        ScanPlan((ScanAxis(BIAS_PORTS[2], (-256.0, 0.0, 256.0)),)), ports
    )
    assert bound[0].label == "da_bias_z"


def test_tunable_devices_project_device_ports() -> None:
    """A device volunteers its runtime knobs as ``device:<key>:<field>`` ports.

    The aggregation is duck-typed off the installation, and only fields with
    BOTH bounds declared become ports -- a plan must be refusable against a
    finite range before anything touches hardware.
    """

    installation = create_installation("virtual")
    try:
        tunables = tunable_devices(installation)
        assert "mot_camera" in tunables and "camera" in tunables
        ports = scan_ports_for_devices(tunables)
        by_name = {port.port: port for port in ports}
        key = DEVICE_PARAM_FAMILY + "mot_camera:exposure_seconds"
        assert key in by_name
        port = by_name[key]
        assert port.label == "mot_camera.exposure_seconds"
        assert 0 < port.lo < port.hi
    finally:
        installation.close()


def test_scanning_a_device_port_moves_the_camera_exposure() -> None:
    """End to end: a ``device:`` axis tunes the camera and the frames show it.

    The plan scans the MOT camera's exposure over a 4x range; the spot's
    photon count is proportional to exposure in the simulated world (the
    read-noise floor is not), so pooled above-floor brightness at the long
    exposure must clearly exceed the short one.  Red if the device family is
    never dispatched -- the exposure then never moves and the two points
    look alike.
    """

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {d.api_name: d for d in discover_logic_nodes()}
    sequencer = installation.device("sequencer")
    monitor = None
    host = None
    try:
        monitor_node = descriptors["camera_measurement"].instantiate(
            camera=installation.device("mot_camera"),
            camera_key="mot_camera",
            signal_plane=plane,
            repeat=0,
        )
        monitor = monitor_node.monitor()

        sequence = _template_sequence()
        board = sequencer.describe()
        from zlc_pulse import compile_sequence

        seeded = resolve_api_parameters(sequence)
        sequencer.load(
            compile_sequence(seeded, board.geometry, board.clock_hz),
            source=seeded,
        )
        sequencer.fire()
        sequencer.wait_done(5.0)
        deadline = time.monotonic() + 10.0
        signal_name = ""
        while time.monotonic() < deadline and not signal_name:
            monitor.poll()
            plane.freeze()
            for name in plane.describe_signals():
                text = str(getattr(name, "name", name))
                if "/frame" in text:
                    signal_name = text
            time.sleep(0.02)
        assert signal_name, "the MOT monitor never published a frame"

        exposures = (0.02, 0.08)
        plan = ScanPlan(
            (
                ScanAxis(
                    DEVICE_PARAM_FAMILY + "mot_camera:exposure_seconds",
                    exposures,
                ),
            )
        )
        scan_node = descriptors["scan"].instantiate(
            sequencer=sequencer,
            signal_plane=plane,
            source_signal=signal_name,
            pulse_resource=ResolvedWorkspaceResource(
                Path(TEMPLATE_NAME), "zlc.pulse/scan-template", sequence
            ),
            plan=plan.to_tree(),
            samples_per_point=1,
            tunable_devices=tunable_devices(installation),
        )
        host = NodeHost(scan_node, plane)
        host.start()
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline and not host.observation.terminal:
            monitor.poll()
            plane.freeze()
            host.poll()
        observed = host.observation
        assert observed.error is None, observed.error
        assert observed.terminal

        frozen = plane.freeze()
        value = frozen.value(host.signal_key("scan"))
        assert value is not None, "the scan published nothing"
        frames = np.asarray(value.block.values, dtype=float)
        assert frames.shape[1] == len(exposures)
        pooled = np.clip(frames - 12.0, 0.0, None)
        brightness = pooled.sum(
            axis=tuple(axis for axis in range(pooled.ndim) if axis != 1)
        )
        assert brightness[1] > 2.0 * brightness[0], (
            "a 4x exposure did not brighten the spot; the device port was "
            f"never applied (brightness={brightness.round(1).tolist()})"
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
        installation.close()


def test_scanning_the_bias_dacs_finds_the_planted_mot_optimum() -> None:
    """The goal, end to end: a 3x3x3 field scan lands on the world's optimum.

    The grid is coarse on purpose -- the assertion is that the brightest scan
    point is the grid point NEAREST the planted optimum, computed from the
    ground truth rather than hard-coded, so re-planting the optimum moves the
    expectation with it.
    """

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {d.api_name: d for d in discover_logic_nodes()}
    sequencer = installation.device("sequencer")
    monitor = None
    host = None
    try:
        monitor_node = descriptors["camera_measurement"].instantiate(
            camera=installation.device("mot_camera"),
            camera_key="mot_camera",
            signal_plane=plane,
            repeat=0,
        )
        monitor = monitor_node.monitor()

        # Seed the live signal: one fire at the authored zeros, one poll.
        sequence = _template_sequence()
        board = sequencer.describe()
        from zlc_pulse import compile_sequence

        seeded = resolve_api_parameters(sequence)
        sequencer.load(
            compile_sequence(seeded, board.geometry, board.clock_hz),
            source=seeded,
        )
        sequencer.fire()
        sequencer.wait_done(5.0)
        deadline = time.monotonic() + 10.0
        signal_name = ""
        while time.monotonic() < deadline and not signal_name:
            monitor.poll()
            # Publications materialise when the plane FREEZES -- in the
            # product that is the console beat's act; here the test plays it.
            plane.freeze()
            for name in plane.describe_signals():
                text = str(getattr(name, "name", name))
                if "/frame" in text:
                    signal_name = text
            time.sleep(0.02)
        assert signal_name, "the MOT monitor never published a frame"

        values = tuple((-256.0, 0.0, 256.0))
        plan = ScanPlan(tuple(ScanAxis(port, values) for port in BIAS_PORTS))
        scan_node = descriptors["scan"].instantiate(
            sequencer=sequencer,
            signal_plane=plane,
            source_signal=signal_name,
            pulse_resource=ResolvedWorkspaceResource(
                Path(TEMPLATE_NAME), "zlc.pulse/scan-template", sequence
            ),
            plan=plan.to_tree(),
            samples_per_point=1,
        )
        host = NodeHost(scan_node, plane)
        host.start()
        scan_signal = host.signal_key("scan")
        live_fill_levels: set[int] = set()
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline and not host.observation.terminal:
            monitor.poll()
            front = plane.freeze()
            # A measurement publishes LIVE while it runs: a panel attaching
            # mid-scan must see the growing dataset, not silence until the
            # end.  This guard is red under an implementation that only
            # publishes a FINAL result.
            live = front.value(scan_signal)
            if live is not None and live.coverage is not None:
                live_fill_levels.add(live.coverage.written_cells)
            host.poll()
        observed = host.observation
        assert observed.error is None, observed.error
        assert observed.terminal
        partial = {
            level
            for level in live_fill_levels
            if 0 < level < plan.point_count
        }
        assert partial, (
            "the scan never published a partially filled live dataset; "
            f"observed fill levels: {sorted(live_fill_levels)}"
        )

        frozen = plane.freeze()
        value = frozen.value(host.signal_key("scan"))
        assert value is not None, "the scan published nothing"

        frames = np.asarray(value.block.values, dtype=float)
        # (repeat, scan points, event, y, x): one MOT cycle per grid point,
        # its frames on the READOUT_EVENT axis.
        assert frames.shape[1] == plan.point_count
        # Brightness above the read-noise floor; position-independent, so the
        # spot moving with the field cannot fool the metric.
        pooled = np.clip(frames - 12.0, 0.0, None)
        brightness = pooled.sum(
            axis=tuple(axis for axis in range(pooled.ndim) if axis != 1)
        )
        best = plan.rows()[int(np.argmax(brightness))]

        expected = tuple(
            min(values, key=lambda value: abs(value - optimum))
            for optimum in DEFAULT_MOT_FIELD_OPTIMUM_DAC
        )
        assert best == expected, (
            f"the scan says {best} but the planted optimum {DEFAULT_MOT_FIELD_OPTIMUM_DAC} "
            f"is nearest {expected}; brightness={brightness.round(1).tolist()}"
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
        installation.close()
