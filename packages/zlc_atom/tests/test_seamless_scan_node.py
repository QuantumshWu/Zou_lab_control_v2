"""The seamless scan node: the BOARD advances the points from its scan table.

Two things have to be true and nothing else matters.  The plan must reach the
board as a scan table whose rows are the plan's rows -- each repeated
``shots_per_point`` times, swept ``repeats`` times -- and the publications
that come back must land on the rows that produced them, in played order.
The first is asserted against the table the board was handed; the second
twice: once against a source whose every publication is named, and once
against the virtual world's own physics, where a wrong order would show up as
a survival curve that does not fall.

A ``device:`` axis is refused here, by name, pointing at the node that can
run it: a host call between two cycles of one fired table is exactly what
does not happen.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from zlc_pulse import compile_sequence, resolve_api_parameters, sequence_from_tree
from zlc_runtime import NodeHost, SignalDataPlane

from zlc_atom.install import create_installation
from zlc_atom.nodes import (
    ResolvedWorkspaceResource,
    discover_logic_nodes,
    scan_pulse_template_bytes,
    temperature_pulse_template_bytes,
)
from zlc_atom.nodes.scan import (
    DEVICE_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    SCAN_PULSE_CONTRACT,
    ScanAxis,
    ScanPlan,
)
from zlc_atom.nodes.seamless_scan import SEAMLESS_SCAN_SCHEMA

from tests.fakes import SCRIPTED_SEED_VALUE, ScriptedScanBench


TEMPLATE_NAME = "mot_field_template.json"
BIAS_X_PORT = PULSE_PARAM_FAMILY + "da_bias_x"
#: Long enough that no scheduling jitter could produce it, short enough to pay.
AUTHORED_SETTLE_SECONDS = 0.37


def _template_sequence():
    return sequence_from_tree(json.loads(scan_pulse_template_bytes().decode("utf-8")))


def _pulse_resource(name: str, sequence):
    return ResolvedWorkspaceResource(Path(name), SCAN_PULSE_CONTRACT, sequence)


def _scripted_run(
    *,
    values: tuple[float, ...],
    shots: int,
    repeats: int,
    settle: float,
) -> tuple[np.ndarray, ScriptedScanBench]:
    """Play the table over a source whose every publication is named.

    Returns the kept shots as (visit, plan row) values -- each cell is the
    index of the publication that landed in it -- and the bench that scripted
    them.  One fire hands over every publication the table plays, which is
    what a board-driven source does.
    """

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    bench = None
    host = None
    try:
        bench = ScriptedScanBench(
            installation.device("sequencer"),
            plane,
            publications_per_fire=repeats * len(values) * shots,
        )
        bench.publish(SCRIPTED_SEED_VALUE)
        plan = ScanPlan((ScanAxis(BIAS_X_PORT, values),))
        node = descriptors["seamless_scan"].instantiate(
            sequencer=bench,
            signal_plane=plane,
            source_signal=bench.signal_name,
            pulse_resource=_pulse_resource(TEMPLATE_NAME, _template_sequence()),
            plan=plan.to_tree(),
            repeats=repeats,
            shots_per_point=shots,
            settle_seconds=settle,
        )
        host = NodeHost(node, plane)
        host.start()
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and not host.observation.terminal:
            host.poll()
        observed = host.observation
        assert observed.error is None, observed.error
        assert observed.terminal, (
            "the seamless scan never finished; it published "
            f"{bench.published} and kept waiting"
        )
        value = plane.freeze().value(host.signal_key("scan"))
        assert value is not None, "the scan published nothing"
        block = np.asarray(value.block.values, dtype=float)
        # (visit, plan row, y, x): every pixel of a scripted frame carries the
        # publication's index, so the cell mean IS the shot that landed there.
        return block.mean(axis=(2, 3)), bench
    finally:
        if host is not None and not host.observation.terminal:
            host.cancel("test cleanup")
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not host.observation.terminal:
                host.poll()
        if host is not None:
            host.shutdown()
        if bench is not None:
            bench.close()
        plane.close()
        installation.close()


def test_a_device_axis_is_refused_and_the_refusal_names_the_stepped_node() -> None:
    """The board cannot make a host call between two cycles of one table."""

    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    plan = ScanPlan(
        (
            ScanAxis(
                DEVICE_PARAM_FAMILY + "mot_camera:exposure_seconds", (0.02, 0.08)
            ),
        )
    )
    plane = SignalDataPlane()
    try:
        with pytest.raises(ValueError, match="stepped_scan") as refusal:
            descriptors["seamless_scan"].instantiate(
                sequencer=None,
                signal_plane=plane,
                source_signal="@logic/monitor/frames",
                pulse_resource=_pulse_resource(
                    TEMPLATE_NAME, _template_sequence()
                ),
                plan=plan.to_tree(),
            )
        assert "mot_camera:exposure_seconds" in str(refusal.value)
    finally:
        plane.close()


def test_the_seamless_node_asks_nothing_about_gating_or_advance() -> None:
    """The fired table drives the frames, so the gating question cannot arise."""

    names = SEAMLESS_SCAN_SCHEMA.field_names
    assert "gating" not in names
    assert "capture" not in names
    assert "advance" not in names
    assert set(names) == {
        "pulse_template",
        "plan",
        "repeats",
        "shots_per_point",
        "settle_seconds",
    }


def test_the_table_carries_every_row_and_shot_and_the_shots_land_in_order() -> None:
    """One load, one fire: the table IS the plan, and order is the assignment.

    Two points, two shots each, two sweeps: the board is handed a table whose
    rows repeat per shot and whose sweep count is the repeats, and the eight
    publications land on (visit, row) in played order -- red the moment a
    shot is credited to the point beside it.
    """

    kept, bench = _scripted_run(
        values=(-256.0, 256.0), shots=2, repeats=2, settle=0.0
    )
    assert kept.tolist() == [
        [0.0, 2.0],  # sweep 0, shot 0
        [1.0, 3.0],  # sweep 0, shot 1
        [4.0, 6.0],  # sweep 1, shot 0
        [5.0, 7.0],  # sweep 1, shot 1
    ]
    assert SCRIPTED_SEED_VALUE not in kept.reshape(-1).tolist()

    assert len(bench.scan_tables) == 1, "the table is written once, not per point"
    table, sweeps = bench.scan_tables[0]
    assert sweeps == 2, "whole-table sweeps are the board's own repeat count"
    assert len(table) == 4, "one wire row per plan row per shot"
    first, second, third, fourth = (tuple(row) for row in np.asarray(table))
    assert first == second and third == fourth, "a shot repeats its row in place"
    assert first != third, "the two plan points must reach the board apart"


def test_the_authored_settle_time_stops_the_board_before_the_table() -> None:
    """The whole table plays from one fire, so the board is stopped once."""

    _kept, bench = _scripted_run(
        values=(-256.0, 256.0), shots=1, repeats=1, settle=AUTHORED_SETTLE_SECONDS
    )
    intervals = bench.stop_intervals()
    assert len(intervals) == 1, (
        f"one stop before the one fire was expected, got {intervals}"
    )
    assert intervals[0] >= AUTHORED_SETTLE_SECONDS, (
        f"the board was stopped for only {intervals[0]:.3f}s, less than the "
        f"authored {AUTHORED_SETTLE_SECONDS}s"
    )
    assert intervals[0] < AUTHORED_SETTLE_SECONDS + 1.0, (
        f"the stop of {intervals[0]:.3f}s is not the authored "
        f"{AUTHORED_SETTLE_SECONDS}s"
    )


def test_the_board_advanced_scan_recovers_the_planted_trap_loss() -> None:
    """End to end on the virtual bench, against the world's own ground truth.

    The temperature template probes the traps twice around a variable release
    (``t_off``) and the site camera is triggered by that same fired program,
    so its cycles arrive in played order -- the source family this node is
    for.  The second probe's brightness over the first IS the survival
    fraction, and it must fall with ``t_off`` at the rate the world planted.

    The metric is a site-box sum with the frame's own floor removed, not the
    occupancy readout, so it keeps some background and UNDER-reports the
    decay (measured ~0.27/ms against a planted 0.5/ms).  The band below is
    that measurement, not a hope; what is exact here is the ORDER: a survival
    curve that does not fall is a scan whose frames landed on the wrong
    points.
    """

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {d.api_name: d for d in discover_logic_nodes()}
    sequencer = installation.device("sequencer")
    monitor = None
    host = None
    try:
        camera_node = descriptors["camera_measurement"].instantiate(
            camera=installation.device("camera"),
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
            compile_sequence(seeded, board.geometry, board.clock_hz), source=seeded
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

        t_offs = (0.2, 0.6, 1.2, 2.4)
        shots = 6
        plan = ScanPlan((ScanAxis(PULSE_PARAM_FAMILY + "t_off", t_offs),))
        scan_node = descriptors["seamless_scan"].instantiate(
            sequencer=sequencer,
            signal_plane=plane,
            source_signal=frames_signal,
            pulse_resource=_pulse_resource("temperature_template.json", sequence),
            plan=plan.to_tree(),
            shots_per_point=shots,
            settle_seconds=0.05,
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

        value = plane.freeze().value(host.signal_key("scan"))
        assert value is not None, "the scan published nothing"
        block = np.asarray(value.block.values, dtype=float)
        # (shots, t_off points x probe frames, y, x).
        assert block.shape[:2] == (shots, 2 * len(t_offs))
        sites = np.zeros(block.shape[2:], dtype=bool)
        for x, y in installation.world.geometry.site_centers_xy:
            row, column = int(round(y)), int(round(x))
            sites[max(0, row - 1) : row + 2, max(0, column - 1) : column + 2] = True
        floor = np.median(block.reshape(*block.shape[:2], -1), axis=2)
        pooled = (block[..., sites] - floor[..., None]).sum(axis=2)
        pooled = pooled.reshape(shots, len(t_offs), 2)
        survival = np.mean(pooled[..., 1] / pooled[..., 0], axis=0)
        assert np.all(np.isfinite(survival)), (
            f"survival has empty points: {survival.tolist()}"
        )
        assert np.all(np.diff(survival) < 0), (
            f"survival must fall with t_off: {survival.round(3).tolist()}"
        )
        lifetime_ms = float(installation.world.trap_off_lifetime_s) * 1e3
        slope = np.polyfit(np.asarray(t_offs), np.log(survival), 1)[0]
        planted = -1.0 / lifetime_ms
        assert 0.25 <= slope / planted <= 1.25, (
            f"measured decay {slope:.3f}/ms against a planted {planted:.3f}/ms; "
            f"survival={survival.round(3).tolist()}"
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
