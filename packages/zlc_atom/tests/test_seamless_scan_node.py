"""The seamless scan node: the BOARD advances the points from its scan table.

Two things have to be true and nothing else matters.  The plan must reach the
board as a scan table whose rows are EXACTLY the plan's rows -- shots play
inside one point as the pulse's outermost repeat bracket, sweeps as extra
fired cycles over the same table -- and the publications that come back must
land on the rows that produced them, in played order.
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
from dataclasses import replace

from zlc_pulse import (
    RepeatRegion,
    compile_sequence,
    resolve_api_parameters,
    sequence_from_tree,
)
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
    MANUAL_AXIS_REQUEST,
    PULSE_PARAM_FAMILY,
    SCAN_PULSE_CONTRACT,
    SCAN_OUTPUT,
    ScanAxis,
    ScanPlan,
    manual_axis,
    scan_ports_for,
    slots_from_plan,
    split_outer_axes,
)
from zlc_atom.nodes.seamless_scan import SEAMLESS_SCAN_SCHEMA

from tests.fakes import SCRIPTED_SEED_VALUE, ScriptedScanBench


TEMPLATE_NAME = "mot_field_template.json"
BIAS_X_PORT = PULSE_PARAM_FAMILY + "da_bias_x"
#: Long enough that no scheduling jitter could produce it, short enough to pay.
AUTHORED_SETTLE_SECONDS = 0.37


def _scan_host(node: object, plane: SignalDataPlane) -> NodeHost:
    return NodeHost(
        node,
        plane,
        instance_id=node.instance_id,
        kind="measurement",
        dataset_output_declarations=(SCAN_OUTPUT,),
        input_signal=node.source.signal_name,
        input_delivery="exact",
    )


def _template_sequence(*scanned: str):
    """The fixture template with its planned parameters compiled to slots.

    A seamless template CARRIES its hardware scan slots; the shared fixture
    authors API parameters, so each test names what it scans and this
    compiles exactly those into the slots the template would have carried.
    """

    raw = sequence_from_tree(
        json.loads(scan_pulse_template_bytes().decode("utf-8"))
    )
    names = set(scanned) or {"da_bias_x"}
    ports = tuple(
        port
        for port in scan_ports_for(raw)
        if port.port[len(PULSE_PARAM_FAMILY):] in names
    )
    assert len(ports) == len(names), (names, [p.port for p in ports])
    return slots_from_plan(raw, ports)


def _pulse_resource(name: str, sequence):
    return ResolvedWorkspaceResource(Path(name), SCAN_PULSE_CONTRACT, sequence)


def _scripted_run(
    *,
    values: tuple[float, ...],
    shots: int,
    repeats: int,
    settle: float,
    sequence: object | None = None,
    readouts: int | None = None,
    seed: bool = True,
) -> tuple[np.ndarray, ScriptedScanBench]:
    """Play the table over a source whose every publication is named.

    Returns the kept shots as (visit, plan row) values -- each cell is the
    index of the publication that landed in it -- and the bench that scripted
    them.  One fire hands over every publication the table plays, which is
    what a board-driven source does.  ``readouts`` is how many that is when a
    template's own bracket multiplies the shots.
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
            publications_per_fire=(
                repeats * len(values) * shots if readouts is None else readouts
            ),
        )
        if seed:
            bench.publish(SCRIPTED_SEED_VALUE)
        plan = ScanPlan((ScanAxis(BIAS_X_PORT, values),))
        node = descriptors["seamless_scan"].instantiate(
            sequencer=bench,
            signal_plane=plane,
            source_signal=bench.signal_name,
            pulse_resource=_pulse_resource(
                TEMPLATE_NAME,
                _template_sequence() if sequence is None else sequence,
            ),
            plan=plan.to_tree(),
            repeats=repeats,
            shots_per_point=shots,
            settle_seconds=settle,
        )
        host = _scan_host(node, plane)
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
        publication = plane.latest_publication(host.signal_key("scan"))
        assert publication is not None
        (parent,) = plane.direct_parent_publications(publication)
        assert parent.value(bench.signal_name) is not None
        value = plane.current_dataset(host.signal_key("scan"))
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


def test_a_device_axis_below_a_board_axis_is_refused_by_name() -> None:
    """The host moves a device knob BETWEEN fires, never inside one.

    The seamless node used to refuse every device axis outright, pointing
    at the stepped executor; a device axis is now an outer loop of this
    node -- the run pauses, the knob is tuned and read back, the next
    segment fires.  What stays refused is the impossible nesting: a device
    axis underneath the board's own table.
    """

    plan = ScanPlan(
        (
            ScanAxis(BIAS_X_PORT, (-256.0, 256.0)),
            ScanAxis(
                DEVICE_PARAM_FAMILY + "rf:frequency_hz", (1e9, 2e9)
            ),
        )
    )
    with pytest.raises(ValueError) as refusal:
        split_outer_axes(plan)
    assert "rf.frequency_hz" in str(refusal.value)
    assert "above the board axes" in str(refusal.value)


def test_a_plan_of_device_axes_alone_has_no_table_to_play() -> None:
    plan = ScanPlan(
        (ScanAxis(DEVICE_PARAM_FAMILY + "rf:frequency_hz", (1e9, 2e9)),)
    )
    with pytest.raises(ValueError, match="no table to play"):
        split_outer_axes(plan)


def test_the_seamless_node_asks_nothing_about_gating_or_advance() -> None:
    """The fired table drives the frames, so the gating question cannot arise."""

    names = SEAMLESS_SCAN_SCHEMA.field_names
    assert "gating" not in names
    assert "capture" not in names
    assert "advance" not in names
    assert set(names) == {
        "pulse_template",
        "plan",
        "api_values",
        "repeats",
        "shots_per_point",
        "settle_seconds",
    }


def test_source_preflight_rejects_before_the_board_is_loaded(monkeypatch) -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    bench = None
    host = None
    try:
        bench = ScriptedScanBench(
            installation.device("sequencer"), plane, publications_per_fire=1
        )
        bench.publish(SCRIPTED_SEED_VALUE)
        descriptor = {
            value.api_name: value for value in discover_logic_nodes()
        }["seamless_scan"]
        node = descriptor.instantiate(
            sequencer=bench,
            signal_plane=plane,
            source_signal=bench.signal_name,
            pulse_resource=_pulse_resource(TEMPLATE_NAME, _template_sequence()),
            plan=ScanPlan((ScanAxis(BIAS_X_PORT, (0.0,)),)).to_tree(),
            repeats=1,
            shots_per_point=1,
            settle_seconds=0.0,
        )

        def reject(*_args, **_kwargs) -> None:
            raise ValueError("invalid camera cadence")

        monkeypatch.setattr(node.source, "validate", reject)
        host = _scan_host(node, plane)
        host.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not host.observation.terminal:
            host.poll()
        assert "invalid camera cadence" in str(host.observation.error)
        assert bench.loads == 0, "the board was mutated before source preflight"
    finally:
        if host is not None:
            host.shutdown()
        if bench is not None:
            bench.close()
        plane.close()
        installation.close()


def test_the_table_is_the_plan_and_the_shots_ride_the_bracket(monkeypatch) -> None:
    """One load, one fire: the table IS the plan, and order is the assignment.

    Two points, two shots each, two sweeps: the board is handed one TWO-row
    table -- the shots live in the pulse's own repeat bracket, not as
    repeated rows -- one fired cycle per point per sweep, and the
    publications land on (visit, row) in played order, red the moment a shot
    is credited to the point beside it.
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
    table = bench.scan_tables[0]
    assert len(table) == 2, "one wire row per plan row; shots are not rows"
    first, second = (tuple(row) for row in np.asarray(table))
    assert first != second, "the two plan points must reach the board apart"
    assert bench.fired_cycles == [4], "one fired cycle per point per sweep"
    assert bench.loaded_loop_counts == [2], "the shot count IS the bracket"

    # A long duration is still a full-width period; only its variation rides
    # the signed slot multiplier.  Make the requested span wider than one
    # 25-bit tick operand so the application must choose scale 2, and retain
    # the canonical schema that the real writer hands Runtime.
    from zlc_atom.nodes.scan.dataset import ScanDatasetWriter
    from zlc_pulse import PulseFieldRef, PulseSlot, scan_columns_for

    canonical = []
    original_write = ScanDatasetWriter.write

    def record_canonical(writer, value, *, row, visit):
        output = original_write(writer, value, row=row, visit=visit)
        canonical.append(output)
        return output

    monkeypatch.setattr(ScanDatasetWriter, "write", record_canonical)
    sequence = _template_sequence()
    period_id = sequence.periods[-1].period_id
    sequence = replace(
        sequence,
        periods=tuple(
            replace(period, duration=700.0, unit="ms")
            if period.period_id == period_id
            else period
            for period in sequence.periods
        ),
        slots=(
            PulseSlot(
                "duration",
                PulseFieldRef("duration", period_id=period_id),
                "ms",
                slot_id="da_bias_x",
            ),
        ),
    )
    requested = (200.00001, 1200.00001)
    _kept, long_bench = _scripted_run(
        values=requested,
        shots=1,
        repeats=1,
        settle=0.0,
        sequence=sequence,
    )

    program = long_bench._loaded_program
    assert program.slot_tick_scales == (2,)
    coefficient = (1 << program.scan_coeff_frac_bits) * 2
    assert {
        abs(value)
        for row in program.tick_slot_coeffs
        for value in row
        if value
    } == {coefficient}, "the compiled affine program did not apply scale 2"

    wire = tuple(
        tuple(int(value) for value in row)
        for row in long_bench.scan_tables[0]
    )
    columns = scan_columns_for(
        long_bench.loaded_sources[0],
        program.slot_tick_scales,
    )
    played = tuple(
        (float(value) - columns[0].wire_offset) / columns[0].wire_scale
        for value, in wire
    )
    assert played == pytest.approx((200.0, 1200.0))
    assert played != requested, "this case must exercise visible tick quantization"

    schema = canonical[0].canonical_schema
    coordinate = next(
        column
        for column in schema.point_table.columns
        if column.name == "da_bias_x"
    )
    assert coordinate.values == pytest.approx(played)
    run_record = canonical[0].run_record
    assert run_record["slot_tick_scales"] == [2]
    assert run_record["named_devices"] == {"sequencer": "sequencer"}
    sequencer = run_record["device_snapshots"]["sequencer"]["description"]
    assert sequencer["clock_hz"] > 0.0
    assert sequencer["layout_fingerprint"] > 0
    assert sequencer["target"]["raw_lanes"]
    assert "package_pins" in sequencer["target"]


def test_an_authored_whole_bracket_multiplies_with_the_shots() -> None:
    """Every iteration of a whole-sequence bracket is one shot.

    The template already plays its whole span twice; the operator asks for
    two shots on top.  One point is then four bracket iterations -- four
    readouts -- and the dataset says so: four visits per sweep, in played
    order, over an unchanged two-row table.
    """

    template = _template_sequence()
    template = replace(
        template,
        repeat=RepeatRegion(
            template.periods[0].period_id, template.periods[-1].period_id, 2
        ),
    )
    kept, bench = _scripted_run(
        values=(-256.0, 256.0),
        shots=2,
        repeats=1,
        settle=0.0,
        sequence=template,
        readouts=8,
    )
    assert kept.tolist() == [
        [0.0, 4.0],  # shot 0 = the bracket's first authored play
        [1.0, 5.0],
        [2.0, 6.0],
        [3.0, 7.0],
    ]
    assert len(bench.scan_tables[0]) == 2
    assert bench.fired_cycles == [2]
    assert bench.loaded_loop_counts == [4], "authored twice x two shots"


def test_a_partial_bracket_refuses_more_than_one_shot_and_plays_at_one() -> None:
    """The board has ONE hardware loop; a partial bracket already spends it.

    Two shots of a partially-bracketed template cannot nest, so the run is
    refused before the board is touched, in words that name the way out.  At
    one shot the authored bracket is the author's own business and plays
    untouched.
    """

    template = _template_sequence()
    partial = replace(
        template,
        repeat=RepeatRegion(
            template.periods[0].period_id, template.periods[1].period_id, 2
        ),
    )

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    bench = None
    host = None
    try:
        bench = ScriptedScanBench(
            installation.device("sequencer"), plane, publications_per_fire=1
        )
        bench.publish(SCRIPTED_SEED_VALUE)
        node = descriptors["seamless_scan"].instantiate(
            sequencer=bench,
            signal_plane=plane,
            source_signal=bench.signal_name,
            pulse_resource=_pulse_resource(TEMPLATE_NAME, partial),
            plan=ScanPlan((ScanAxis(BIAS_X_PORT, (0.0,)),)).to_tree(),
            repeats=1,
            shots_per_point=2,
            settle_seconds=0.0,
        )
        host = _scan_host(node, plane)
        host.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not host.observation.terminal:
            host.poll()
        message = str(host.observation.error)
        assert "whole-sequence bracket" in message, message
        assert "shots_per_point" in message, message
        assert bench.loads == 0, "the board was touched by a refused run"
    finally:
        if host is not None:
            host.shutdown()
        if bench is not None:
            bench.close()
        plane.close()
        installation.close()

    kept, bench = _scripted_run(
        values=(0.0,), shots=1, repeats=1, settle=0.0, sequence=partial
    )
    assert kept.tolist() == [[0.0]]
    assert bench.loaded_loop_counts == [2], "one shot leaves the author's bracket"


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

        # Microseconds, in the template's own unit: where a recapture curve
        # for micro-kelvin atoms in a micron trap actually falls.
        t_offs = (0.004, 0.010, 0.016, 0.024)
        shots = 6
        plan = ScanPlan((ScanAxis(PULSE_PARAM_FAMILY + "t_off", t_offs),))
        scan_node = descriptors["seamless_scan"].instantiate(
            sequencer=sequencer,
            signal_plane=plane,
            source_signal=frames_signal,
            pulse_resource=_pulse_resource(
                "temperature_template.json",
                # The node's contract: the template CARRIES its scan slot.
                slots_from_plan(
                    sequence,
                    tuple(
                        port
                        for port in scan_ports_for(sequence)
                        if port.port == PULSE_PARAM_FAMILY + "t_off"
                    ),
                ),
            ),
            plan=plan.to_tree(),
            shots_per_point=shots,
            settle_seconds=0.05,
        )
        host = _scan_host(scan_node, plane)
        host.start()
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline and not host.observation.terminal:
            monitor.poll()
            plane.freeze()
            host.poll()
        observed = host.observation
        assert observed.error is None, observed.error
        assert observed.terminal

        value = plane.current_dataset(host.signal_key("scan"))
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
        # Against the world's OWN model of what a release does, not against a
        # formula copied here: an atom leaves because it is fast enough to
        # walk out of the trap while the light is off.
        planted = np.asarray(
            [installation.world._release_survival(value * 1e-3, 1.0) for value in t_offs],
            dtype=float,
        )
        planted = planted / planted[0]
        assert np.all(np.abs(survival / survival[0] - planted) <= 0.2), (
            f"measured {(survival / survival[0]).round(3).tolist()} against the "
            f"world's own {planted.round(3).tolist()}"
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


def test_an_armed_silent_chain_is_a_valid_scan_source() -> None:
    """The user's bench flow: camera armed, pulse stopped, then Start scan.

    An externally triggered chain publishes NOTHING until a pulse fires its
    triggers -- and the scan is what fires them.  The scan must accept that
    armed silence (it is the aligned start: frame one is point one), start
    only its own pulse, and land every publication on its played row.  It
    never starts the camera and never judges frame alignment; zero frames
    before the first trigger makes alignment a construction, not a check.
    """

    kept, bench = _scripted_run(
        values=(-256.0, 256.0), shots=1, repeats=1, settle=0.0, seed=False
    )
    assert kept.tolist() == [[0.0, 1.0]]
    assert bench.published == [0, 1], (
        "every frame the chain ever produced was fired by the scan itself"
    )


def test_a_chain_that_is_not_armed_stays_refused_by_name() -> None:
    """No camera measurement running at all: the scan refuses loudly and
    tells the operator to start the chain, not the pulse."""

    from zlc_runtime import SignalDataPlane as _Plane

    plane = _Plane()
    try:
        descriptor = {
            value.api_name: value for value in discover_logic_nodes()
        }["seamless_scan"]
        with pytest.raises(ValueError, match="not armed"):
            descriptor.instantiate(
                sequencer=object(),
                signal_plane=plane,
                source_signal="nobody:frames",
                pulse_resource=_pulse_resource(
                    TEMPLATE_NAME, _template_sequence()
                ),
                plan=ScanPlan(
                    (ScanAxis(BIAS_X_PORT, (0.0,)),)
                ).to_tree(),
                repeats=1,
                shots_per_point=1,
                settle_seconds=0.0,
            )
    finally:
        plane.close()


def _manual_run(
    *,
    manual: tuple[tuple[str, tuple[float, ...]], ...],
    values: tuple[float, ...],
    shots: int = 1,
    repeats: int = 1,
    answer=None,
):
    """Walk a plan whose outer axes only a hand can move.

    Returns the finished dataset value, the questions the run asked, and
    the bench.  ``answer`` overrides what the operator says, so a test can
    refuse on purpose.
    """

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    bench = None
    host = None
    asked: list[object] = []
    try:
        bench = ScriptedScanBench(
            installation.device("sequencer"),
            plane,
            publications_per_fire=len(values) * shots,
        )
        bench.publish(SCRIPTED_SEED_VALUE)
        plan = ScanPlan(
            tuple(manual_axis(name, points) for name, points in manual)
            + (ScanAxis(BIAS_X_PORT, values),)
        )
        node = descriptors["seamless_scan"].instantiate(
            sequencer=bench,
            signal_plane=plane,
            source_signal=bench.signal_name,
            pulse_resource=_pulse_resource(TEMPLATE_NAME, _template_sequence()),
            plan=plan.to_tree(),
            repeats=repeats,
            shots_per_point=shots,
            settle_seconds=0.0,
        )
        host = _scan_host(node, plane)
        host.start()
        deadline = time.monotonic() + 60.0
        served = ""
        while time.monotonic() < deadline and not host.observation.terminal:
            host.poll()
            request = host.operator_request
            if request is None or request.request_id == served:
                continue
            served = request.request_id
            asked.append(request)
            assert request.kind == MANUAL_AXIS_REQUEST
            reply = (
                None
                if answer is None
                else answer(request, len(asked) - 1)
            )
            if reply is None:
                reply = {}
            if reply == "stop":
                host.cancel("operator stopped the manual scan")
                continue
            host.submit_operator_input(request.request_id, reply)
        observed = host.observation
        assert observed.error is None, observed.error
        assert observed.terminal, (
            "the manual scan never finished; it published "
            f"{bench.published} and kept waiting"
        )
        return (
            plane.current_dataset(host.signal_key("scan")),
            tuple(asked),
            bench,
        )
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


def test_a_manual_axis_is_the_outer_loop_and_its_answers_are_the_axis() -> None:
    """A hand walks the outside; the board still plays the inside seamlessly.

    Three power points over two bias points: THREE fires, each playing the
    same two-row table, and the dataset's power coordinate is the axis the
    plan authored -- outermost, advancing slowest.
    """

    value, asked, bench = _manual_run(
        manual=(("power", (1.5, 2.5, 4.0)),),
        values=(-256.0, 256.0),
    )

    assert bench.fired_cycles == [2, 2, 2], (
        "one fire per manual point, each playing the whole inner table"
    )
    assert len(bench.scan_tables) == 3, "the inner table is written per fire"

    schema = value.block.schema
    power = next(
        column for column in schema.point_table.columns
        if column.name == "power"
    )
    bias = next(
        column for column in schema.point_table.columns
        if column.name == "da_bias_x"
    )
    # Outermost first: power advances slowest, exactly as the plan reads.
    assert power.values == pytest.approx((1.5, 1.5, 2.5, 2.5, 4.0, 4.0))
    assert bias.values == pytest.approx((-256.0, 256.0) * 3)
    assert power.unit is None, "a manual axis carries a name, not a unit"

    # Every point captured, in played order: publication k lands on row k.
    block = np.asarray(value.block.values, dtype=float)
    assert block.mean(axis=(2, 3)).tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]

    # One question, one stop, and nothing else asked of the operator.
    assert len(asked) == 3, "a stop per manual point, and no other question"
    assert [
        request.payload["value"] for request in asked
    ] == pytest.approx([1.5, 2.5, 4.0])
    assert [request.payload["point"] for request in asked] == [1, 2, 3]


def test_repeats_walk_the_whole_plan_again_and_stop_again() -> None:
    """``repeats`` means the same sentence it always did.

    A plan the board owns spends it on a longer fire.  A plan with a hand
    in it cannot, so it spends it on a second walk -- the same points, the
    same coordinates, stopped for again because the knob has moved since.
    """

    value, asked, bench = _manual_run(
        manual=(("power", (1.0, 2.0)),),
        values=(-256.0, 256.0),
        repeats=2,
    )

    assert bench.fired_cycles == [2, 2, 2, 2], "two walks of two manual points"
    stops = [request.payload["value"] for request in asked]
    assert stops == pytest.approx([1.0, 2.0, 1.0, 2.0])

    schema = value.block.schema
    assert schema.repeat_axis.size == 2, "a walk is a visit, not a new point"
    assert schema.point_table.row_count == 4
    block = np.asarray(value.block.values, dtype=float)
    # Walk 0 played publications 0-3, walk 1 played 4-7, both over the same
    # four points.
    assert block.mean(axis=(2, 3)).tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
    ]


def test_only_the_axis_that_moves_is_asked_for() -> None:
    """A stop is for a hand, so it names only what the hand must move."""

    _value, asked, _bench = _manual_run(
        manual=(("power", (1.0, 2.0)), ("angle", (10.0, 20.0, 30.0))),
        values=(-256.0,),
    )

    stops = [
        (request.payload["axis"], request.payload["value"])
        for request in asked
    ]
    # power advances once every three angle points, and says so only then.
    assert stops == [
        ("power", 1.0),
        ("angle", 10.0),
        ("angle", 20.0),
        ("angle", 30.0),
        ("power", 2.0),
        ("angle", 10.0),
        ("angle", 20.0),
        ("angle", 30.0),
    ]


def test_stopping_at_the_question_stops_the_run() -> None:
    """The operator is the loop here; declining is an answer, not an error."""

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    bench = ScriptedScanBench(
        installation.device("sequencer"), plane, publications_per_fire=1
    )
    host = None
    try:
        bench.publish(SCRIPTED_SEED_VALUE)
        plan = ScanPlan(
            (manual_axis("power", (1.0, 2.0)), ScanAxis(BIAS_X_PORT, (-256.0,)))
        )
        node = descriptors["seamless_scan"].instantiate(
            sequencer=bench,
            signal_plane=plane,
            source_signal=bench.signal_name,
            pulse_resource=_pulse_resource(TEMPLATE_NAME, _template_sequence()),
            plan=plan.to_tree(),
            repeats=1,
            shots_per_point=1,
            settle_seconds=0.0,
        )
        host = _scan_host(node, plane)
        host.start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and host.operator_request is None:
            host.poll()
        assert host.operator_request is not None
        host.cancel("operator stopped the manual scan")
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not host.observation.terminal:
            host.poll()
        assert host.observation.terminal
        assert host.observation.phase == "cancelled"
        assert bench.fired_cycles == [], "nothing fired before the hand answered"
    finally:
        if host is not None:
            host.shutdown()
        bench.close()
        plane.close()
        installation.close()


def test_a_manual_axis_nested_inside_the_table_is_refused_by_name() -> None:
    """A hand cannot reach into a fired table, so it cannot be nested there."""

    plan = ScanPlan(
        (ScanAxis(BIAS_X_PORT, (-256.0, 256.0)), manual_axis("power", (1.0, 2.0)))
    )
    with pytest.raises(ValueError) as refusal:
        split_outer_axes(plan)
    assert "'power'" in str(refusal.value)
    assert "above the board axes" in str(refusal.value)


def test_a_plan_of_manual_axes_alone_has_no_table_to_play() -> None:
    """The seamless node exists to play a board table; a hand is not one."""

    with pytest.raises(ValueError) as refusal:
        split_outer_axes(ScanPlan((manual_axis("power", (1.0, 2.0)),)))
    assert "no table to play" in str(refusal.value)


def _device_run(
    *,
    frequencies: tuple[float, ...],
    values: tuple[float, ...],
    repeats: int = 1,
    tunables=None,
):
    """Walk a plan whose outer axis is an installed device knob.

    The device is the REAL Vaunix driver over its in-memory library --
    the same code path the hardware brick runs -- unless a test hands in
    its own tunables to stage a refusal.
    """

    from zlc_atom.devices.rf.vaunix_lms import VaunixLmsConfig
    from zlc_atom.devices.simulation.rf import virtual_rf_source

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    bench = None
    host = None
    if tunables is None:
        tunables = {
            "rf": virtual_rf_source(
                VaunixLmsConfig(
                    serial=1001,
                    frequency_low_hz=500e6,
                    frequency_high_hz=8e9,
                    power_low_dbm=-40.0,
                    power_high_dbm=10.0,
                )
            )
        }
    try:
        bench = ScriptedScanBench(
            installation.device("sequencer"),
            plane,
            publications_per_fire=len(values),
        )
        bench.publish(SCRIPTED_SEED_VALUE)
        plan = ScanPlan(
            (
                ScanAxis(
                    DEVICE_PARAM_FAMILY + "rf:frequency_hz", frequencies
                ),
                ScanAxis(BIAS_X_PORT, values),
            )
        )
        node = descriptors["seamless_scan"].instantiate(
            sequencer=bench,
            signal_plane=plane,
            source_signal=bench.signal_name,
            pulse_resource=_pulse_resource(TEMPLATE_NAME, _template_sequence()),
            plan=plan.to_tree(),
            tunable_devices=tunables,
            repeats=repeats,
            shots_per_point=1,
            settle_seconds=0.0,
        )
        host = _scan_host(node, plane)
        host.start()
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and not host.observation.terminal:
            host.poll()
            assert host.operator_request is None, (
                "a device axis is moved by a call, never by a question"
            )
            time.sleep(0.002)
        observation = host.poll()
        assert observation.terminal, observation
        if observation.phase != "done":
            raise RuntimeError(str(observation.error))
        value = plane.current_dataset(host.signal_key(SCAN_OUTPUT.name))
        record = dict(node.last_run_record or {})
        claims = node.resolved_device_claims()
        return value, record, bench, tunables["rf"], claims
    finally:
        if host is not None:
            host.shutdown()
        if bench is not None:
            bench.close()
        plane.close()
        installation.close()


def test_a_device_axis_is_the_outer_loop_and_the_device_is_verified() -> None:
    """Three frequencies over two bias points: three fires, no questions.

    The dataset's frequency coordinate is the axis the plan authored --
    outermost, advancing slowest, in HERTZ -- and the run record carries
    the device's identity and its final settings, because a coordinate
    without provenance is a number nobody can trust next month.
    """

    frequencies = (1_000_000_000.0, 1_500_000_000.0, 2_000_000_000.0)
    value, record, bench, source, claims = _device_run(
        frequencies=frequencies,
        values=(-256.0, 256.0),
    )

    assert bench.fired_cycles == [2, 2, 2], (
        "one fire per device point, each playing the whole inner table"
    )

    schema = value.block.schema
    frequency = next(
        column
        for column in schema.point_table.columns
        if column.name == "rf.frequency_hz"
    )
    assert frequency.values == pytest.approx(
        (1e9, 1e9, 1.5e9, 1.5e9, 2e9, 2e9)
    )
    assert frequency.unit == "Hz", "a device axis publishes its knob's unit"

    # The instrument itself ends on the last coordinate -- tune() really ran.
    assert source.tunable_values()["frequency_hz"] == pytest.approx(2e9)

    assert record["named_devices"]["tunable:rf"] == "rf"
    # The snapshot is the device's state at run START: the swept field's
    # truth lives in the axis values above, and what provenance needs
    # beyond it is the UNSWEPT context -- the power and output the whole
    # scan ran at -- plus the identity and epoch to match it to a session.
    snapshot = record["device_snapshots"]["tunable:rf"]
    # The run-start snapshot carries the whole surface, the safety window
    # included: provenance should say what fence the scan ran inside.
    assert set(snapshot["settings"]) == {
        "frequency_hz",
        "power_dbm",
        "output_enabled",
        "frequency_low_hz",
        "frequency_high_hz",
        "power_low_dbm",
        "power_high_dbm",
    }
    assert snapshot["device_session_id"] == "vaunix-lms:1001"
    assert "settings_epoch" in snapshot

    # The console converts these into runtime claims, so a control-panel
    # tune of the swept field is blocked while the scan owns it.
    (claim,) = claims
    assert claim.device_key == "rf"
    assert claim.device is source
    assert claim.protected_fields == ("frequency_hz",)


def test_an_off_grid_device_value_fails_the_run_with_the_grid_named() -> None:
    """The brick holds 10 Hz units; a coordinate between them is refused.

    Refused at APPLY, before the segment fires -- the dataset must never
    contain a frequency the hardware did not actually stand at.
    """

    with pytest.raises(RuntimeError, match="10.*Hz grid"):
        _device_run(
            frequencies=(1_000_000_005.0,),
            values=(-256.0,),
        )


def test_a_device_that_answers_differently_fails_the_run() -> None:
    """tune() returns the read-back, and any difference is a refusal."""

    class _DriftingKnob:
        def tunable_fields(self):
            from zlc_atom.authoring import AuthoringField, TunableField

            return (
                TunableField(
                    metadata=AuthoringField(
                        "frequency_hz",
                        "float",
                        "Frequency (Hz)",
                        0.0,
                        minimum=0.0,
                        maximum=1e10,
                        unit="Hz",
                    ),
                    current=0.0,
                    live_write=True,
                    dependency_group=("frequency_hz",),
                ),
            )

        def tune(self, name, value):
            del name
            return float(value) + 7.0

        def tunable_values(self):
            return {"frequency_hz": 0.0}

        def settings_provenance(self):
            return {"device_session_id": "drift", "settings_epoch": 0}

    with pytest.raises(RuntimeError, match="applied"):
        _device_run(
            frequencies=(1e9,),
            values=(-256.0,),
            tunables={"rf": _DriftingKnob()},
        )
