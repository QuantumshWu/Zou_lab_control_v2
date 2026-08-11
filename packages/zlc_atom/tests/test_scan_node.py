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
from zlc_atom.install import create_installation
from zlc_atom.nodes import ResolvedWorkspaceResource, discover_logic_nodes
from zlc_atom.nodes.scan import (
    PULSE_PARAM_FAMILY,
    ScanAxis,
    ScanPlan,
    bind_plan,
    scan_ports_for,
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
        assert "DAC code" in port.unit, "a DAC axis is labelled in signed codes"
        assert port.lo < 0 < port.hi, "the signed range brackets zero"


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
        # (repeat, scan points, y, x): one MOT frame per grid point.
        assert frames.shape[1] == plan.point_count
        # Brightness above the read-noise floor; position-independent, so the
        # spot moving with the field cannot fool the metric.
        brightness = np.clip(frames - 12.0, 0.0, None).sum(axis=(0, 2, 3))
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
