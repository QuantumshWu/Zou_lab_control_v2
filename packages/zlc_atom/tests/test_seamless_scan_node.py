"""The seamless scan node: the BOARD advances the points from its scan table.

Two things have to be true and nothing else matters.  The plan must reach the
board as a scan table whose rows are EXACTLY the plan's rows -- Run repeats
hold each point for its shots, Scan repeats re-walk the table, and the
independent PulseBracket stays inside every shot -- and the publications that come back must
land on the rows that produced them, in played order.
The first is asserted against the table the board was handed; the second
twice: once against a source whose every publication is named, and once
against the virtual world's own physics, where a wrong order would show up as
a survival curve that does not fall.

Manual and device axes advance between fires. With no board axes, each
fire repeats the fixed pulse without a scan table or a fabricated slot.
"""

from __future__ import annotations

import time
from pathlib import Path

from types import SimpleNamespace

import numpy as np
import pytest
from dataclasses import replace

from zlc_data import owned_snapshot_from_arrays
from zlc_pulse import (
    PulseBracket,
    compile_sequence,
    load_streamer_config,
    pulse_field_value,
    resolve_api_parameters,
)
from zlc_pulse.device import BoardDescription, ConfigValueHolder
from zlc_runtime import MonitorCoverage, NodeHost, SignalDataPlane, SignalValue

from zlc_atom.authoring import AuthoringField, TunableField
from zlc_atom.install import create_installation
from zlc_atom.nodes import (
    ResolvedWorkspaceResource,
    discover_logic_nodes,
)
from tests.pulse_fixture import pulse_sequence
from zlc_atom.nodes.scan import (
    DEVICE_PARAM_FAMILY,
    MANUAL_AXIS_REQUEST,
    PULSE_PARAM_FAMILY,
    SCAN_PULSE_CONTRACT,
    SCAN_OUTPUT,
    ScanAxis,
    ScanPlan,
    ScanPort,
    SeamlessScanMeasurement,
    check_cancelled,
    hardware_scan_ports_for,
    manual_axis,
    scan_ports_for,
    slots_from_plan,
    split_outer_axes,
)
from zlc_atom.nodes.seamless_scan import SEAMLESS_SCAN_SCHEMA

from tests.fakes import SCRIPTED_SEED_VALUE, ScriptedScanBench
from test_scan_repeat_domain import _source_schema


class _FakeSequencer(ConfigValueHolder):
    """The board's surface with no board behind it: every command counted.

    ``on_safe`` runs inside SAFE, which is where an operator's Stop lands
    while the board is acknowledging: the one moment the engines used to
    read too late.
    """

    def __init__(self, sequence) -> None:
        self._init_config_values()
        settings = load_streamer_config()
        self.board = BoardDescription(
            sequence.target, settings["params"], settings["clock_hz"]
        )
        self.load_config_values(
            {
                parameter.parameter_id: (
                    pulse_field_value(sequence, parameter.field_ref, parameter.unit),
                    parameter.unit,
                )
                for parameter in sequence.config_parameters
            }
        )
        self.fires = 0
        self.loads = 0
        self.safe_calls = 0
        self.on_safe = None

    def describe(self):
        return self.board

    def safe(self) -> None:
        self.safe_calls += 1
        if self.on_safe is not None:
            self.on_safe()

    def load(self, program, **_kwargs) -> None:
        self.loads += 1
        self.program = program

    def fire(self, **_kwargs) -> None:
        self.fires += 1

    def wait_done(self, _timeout):
        return SimpleNamespace(fault=None)


class _FakeSource:
    """A point's value on demand; ``fail_at`` names the take that fails and
    ``on_take`` sees every take, so a test can press Stop at a moment."""

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.taken = 0
        self.fail_at = fail_at
        self.on_take = None

    def open(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    def validate(self, *_args, **_kwargs) -> None:
        pass

    def arm(self) -> None:
        pass

    def discard_pending(self) -> None:
        pass

    def describe(self) -> dict:
        return {"source_signal": "fake"}

    def next_value(self, context):
        self.taken += 1
        check_cancelled(context)
        if self.taken == self.fail_at:
            raise RuntimeError("scripted source failed")
        schema = _source_schema(shots=1)
        snapshot = owned_snapshot_from_arrays(
            schema,
            np.zeros(schema.physical_shape),
            self.taken,
            stream_generation="fake-source",
        )
        if self.on_take is not None:
            self.on_take(self.taken)
        return SignalValue("fake-source", snapshot, MonitorCoverage(1, 1)), None


class _Knob:
    """One installed device with one field, remembering every tune.

    ``refuse_restore`` raises the device's refusal to accept the restore.
    """

    def __init__(self, level: float = 0.25, *, refuse_restore: bool = False) -> None:
        self.level = level
        self.tunes: list[float] = []
        self.refuse_restore = refuse_restore

    def tune(self, field: str, value: float) -> float:
        assert field == "level"
        self.tunes.append(float(value))
        if self.refuse_restore and value == 0.25:
            raise RuntimeError("scripted device refused restore")
        self.level = float(value)
        return self.level

    def tunable_fields(self) -> tuple[TunableField, ...]:
        return (
            TunableField(
                AuthoringField("level", "float", "level", 0.25, minimum=0.0, maximum=3.0),
                self.level,
                True,
                ("level",),
            ),
        )

    def tunable_values(self) -> dict:
        return {"level": self.level}

    def settings_provenance(self) -> dict:
        return {"device_session_id": "knob", "settings_epoch": 0}


class _Context:
    """The host's surface: Stop is a flag, commits are counted."""

    def __init__(self) -> None:
        self.cancelled = False
        self.commits = 0
        self.progress = []

    def cancel_requested(self) -> bool:
        return self.cancelled

    def commit_live(self, outputs, *, source_publication=None) -> None:
        del outputs, source_publication
        self.commits += 1

    def report_progress(self, message, *, current=None, total=None) -> None:
        self.progress.append((message, current, total, self.commits))

    def current_dataset(self, name: str) -> str:
        return name


TEMPLATE_NAME = "mot_field_template.json"
BIAS_X_PORT = PULSE_PARAM_FAMILY + "da_bias_x"


def _point_axis_values(schema, name: str) -> tuple[object, ...]:
    axis = next(axis for axis in schema.point_domain.axes if axis.name == name)
    return tuple(
        axis.coordinate_at(code) for code in schema.point_domain.codes(axis.axis_id)
    )


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

    raw = pulse_sequence("mot_field_template.json")
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
    sequence: object | None = None,
    seed: bool = True,
) -> tuple[np.ndarray, ScriptedScanBench]:
    """Play the table over a source whose every publication is named.

    Returns the kept shots as (scan/run repeat, plan row) values -- each cell is the
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
        # (scan repeat x run repeat, plan row, y, x): every pixel of a
        # scripted frame carries the publication's index, so the cell mean IS
        # the shot that landed there.
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


def test_device_axes_alone_repeat_a_fixed_pulse_and_restore_the_device() -> None:
    value, record, bench, source, claims = _device_run(
        frequencies=(1.0, 1.5, 2.0), values=None, unit="GHz", shots=2, repeats=2,
    )
    assert bench.fired_repeats == [(2, 1)] * 6
    assert bench.loads == 1 and bench.scan_tables == []
    assert all(not sequence.slots for sequence in bench.loaded_sources)
    assert bench._loaded_rows == () and bench._loaded_program.slot_count == 0
    assert bench.published[1:] == list(range(12))
    schema = value.block.schema
    axes = tuple(axis for axis in schema.point_domain.axes if axis.axis_id.value.startswith("scan."))
    assert tuple(axis.name for axis in axes) == ("rf.frequency",)
    assert axes[0].unit == "GHz" and tuple(axes[0].coordinates) == (1.0, 1.5, 2.0)
    assert tuple(axis.size for axis in schema.repeat_domain.axes) == (2, 2)
    assert np.asarray(value.block.values).mean(axis=(2, 3)).tolist() == [
        [0.0, 2.0, 4.0], [1.0, 3.0, 5.0],
        [6.0, 8.0, 10.0], [7.0, 9.0, 11.0],
    ]
    assert source.tunable_values()["frequency"] == 600e6
    field = next(field for field in source.tunable_fields() if field.metadata.name == "frequency")
    assert field.metadata.unit == "Hz" and field.current == 600e6
    assert record["plan"]["axes"] == [
        {"port": "device:rf:frequency", "values": [1.0, 1.5, 2.0], "unit": "GHz"},
    ]
    assert claims[0].protected_fields == ("frequency",)


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


def test_the_table_is_the_plan_and_the_shots_are_run_repeats(monkeypatch) -> None:
    """One load, one fire: the table IS the plan, and order is the assignment.

    Two points, two shots each, two sweeps: the board is handed one TWO-row
    table -- shots are Run repeats at one row, not Bracket iterations or
    repeated rows -- and the
    publications land on (scan repeat, run repeat, row) in played order, read
    the moment a shot is credited to the point beside it.
    """

    kept, bench = _scripted_run(
        values=(-256.0, 256.0), shots=2, repeats=2
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
    assert bench.fired_repeats == [(2, 2)]
    assert bench.loaded_loop_counts == [1], "shots must not rewrite the Bracket"

    # A long duration is still a full-width period; only its variation rides
    # the signed slot multiplier.  Make the requested span wider than one
    # 25-bit tick operand so the application must choose scale 2, and retain
    # the canonical schema that the real writer hands Runtime.
    from zlc_atom.nodes.scan.dataset import ScanDatasetWriter
    from zlc_pulse import PulseFieldRef, PulseSlot, scan_columns_for

    canonical = []
    original_write = ScanDatasetWriter.write

    def record_canonical(
        writer, value, *, row, scan_repeat, run_repeat
    ):
        output = original_write(
            writer,
            value,
            row=row,
            scan_repeat=scan_repeat,
            run_repeat=run_repeat,
        )
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
    assert _point_axis_values(schema, "da_bias_x") == pytest.approx(played)
    run_record = canonical[0].run_record
    assert run_record["slot_tick_scales"] == [2]
    assert run_record["named_devices"] == {"sequencer": "sequencer"}
    sequencer = run_record["device_snapshots"]["sequencer"]["description"]
    assert sequencer["clock_hz"] > 0.0
    assert sequencer["layout_fingerprint"] > 0
    assert sequencer["target"]["raw_lanes"]
    assert "package_pins" in sequencer["target"]
    # The pulse is the FILE the operator chose, and the board's snapshot
    # carries what it played: the program the board loaded and the filled
    # document it was compiled from.
    assert run_record["pulse"] == {
        "name": Path(TEMPLATE_NAME).stem,
        "path": str(Path(TEMPLATE_NAME)),
    }
    played_program = run_record["device_snapshots"]["sequencer"]["program"]
    assert played_program["digest"] == long_bench._loaded_program.digest
    assert played_program["rows"] == [list(row) for row in long_bench._loaded_rows]
    assert (played_program["run_repeats"], played_program["scan_repeats"]) == (1, 1)
    document = run_record["device_snapshots"]["sequencer"]["pulse"]
    assert [period["name"] for period in document["periods"]] == [
        period.name for period in long_bench.loaded_sources[-1].periods
    ]


def test_an_authored_whole_bracket_stays_independent_of_run_repeats() -> None:
    """A whole Bracket remains internal while Run repeats supplies shots."""

    template = _template_sequence()
    template = replace(
        template,
        bracket=PulseBracket(
            template.periods[0].period_id, template.periods[-1].period_id, 2
        ),
    )
    kept, bench = _scripted_run(
        values=(-256.0, 256.0),
        shots=2,
        repeats=1,
        sequence=template,
    )
    assert kept.tolist() == [
        [0.0, 2.0],
        [1.0, 3.0],
    ]
    assert len(bench.scan_tables[0]) == 2
    assert bench.fired_repeats == [(2, 1)]
    assert bench.loaded_loop_counts == [2]


def test_a_partial_bracket_and_multiple_run_repeats_are_independent() -> None:
    """A partial Bracket no longer consumes the per-point Run-repeat layer."""

    template = _template_sequence()
    partial = replace(
        template,
        bracket=PulseBracket(
            template.periods[0].period_id, template.periods[1].period_id, 2
        ),
    )

    kept, bench = _scripted_run(
        values=(0.0,), shots=2, repeats=1, sequence=partial
    )
    assert kept.tolist() == [[0.0], [1.0]]
    assert bench.fired_repeats == [(2, 1)]
    assert bench.loaded_loop_counts == [2]


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

        sequence = pulse_sequence("temperature_template.json")
        board = sequencer.describe()
        seeded = resolve_api_parameters(sequence)
        sequencer.load(
            compile_sequence(seeded, board.geometry, board.clock_hz), source=seeded
        )
        sequencer.fire(run_repeats=1, scan_repeats=1)
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
        values=(-256.0, 256.0), shots=1, repeats=1, seed=False
    )
    assert kept.tolist() == [[0.0, 1.0]]
    assert bench.published == [0, 1], (
        "every frame the chain ever produced was fired by the scan itself"
    )


@pytest.mark.parametrize("sealed_before_open", (False, True))
def test_a_chain_that_is_not_armed_waits_for_its_first_real_publication(sealed_before_open) -> None:
    """Pending fit-like outputs open without a fake value or a new registry.

    The arrival callback must attach before another exact event can be lost;
    a prior sealed generation is never replayed as the new scan's first value.
    """
    from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput
    from zlc_atom.nodes.scan import watched_signal_source

    plane = SignalDataPlane()
    declaration = DatasetOutputDeclaration("parameter", "test.parameter")
    producer = SimpleNamespace(instance_id="late-fit", dataset_output_declarations=(declaration,),
                               signal_key=lambda name: f"@logic/late-fit/{name}")
    signal = producer.signal_key("parameter")
    source = None

    def publish(number):
        schema = _source_schema(shots=1)
        snapshot = owned_snapshot_from_arrays(schema, np.full(schema.physical_shape, float(number)),
                                              number, stream_generation="source-input")
        plane.commit_live(producer, {"parameter": LiveDatasetOutput(
            declaration, snapshot, MonitorCoverage(1, 1))})
        return plane.latest_publication(signal)

    try:
        if sealed_before_open:
            plane.begin_generation(producer)
            publish(999)
            plane.seal_committed(producer)
        callbacks = len(plane._publication_callbacks)
        source = watched_signal_source(plane, signal)
        source.open(_Context(), cycles=2)
        source.arm()  # Still no new producer: return so the scan may FIRE.
        checks = iter((False, True))
        waiting = _Context()
        waiting.cancel_requested = lambda: next(checks)
        with pytest.raises(RuntimeError, match="cancelled"):
            source.next_value(waiting)
        plane.begin_generation(producer)
        first, second = publish(1), publish(2)
        for expected in (first, second):
            value, publication = source.next_value(_Context())
            assert publication is expected
            assert value is expected.value(signal)
        plane.seal_committed(producer)
        with pytest.raises(RuntimeError, match="restarted during the scan"):
            source.next_value(_Context())
        source.close()
        assert len(plane._publication_callbacks) == callbacks
        plane.begin_generation(producer)
        publish(3)
        with pytest.raises(RuntimeError, match="not opened"):
            source.next_value(_Context())
    finally:
        if source is not None:
            source.close()
        plane.close()


def _manual_run(
    *,
    manual: tuple[tuple[str, tuple[float, ...]], ...],
    values: tuple[float, ...] | None,
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
            publications_per_fire=(1 if values is None else len(values)) * shots,
        )
        bench.publish(SCRIPTED_SEED_VALUE)
        plan = ScanPlan(
            tuple(manual_axis(name, points) for name, points in manual)
            + (() if values is None else (ScanAxis(BIAS_X_PORT, values),))
        )
        node = descriptors["seamless_scan"].instantiate(
            sequencer=bench,
            signal_plane=plane,
            source_signal=bench.signal_name,
            pulse_resource=_pulse_resource(TEMPLATE_NAME,
                pulse_sequence("mot_field_template.json") if values is None else _template_sequence()),
            plan=plan.to_tree(),
            repeats=repeats,
            shots_per_point=shots,
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

    assert bench.fired_repeats == [(1, 1)] * 3, (
        "one fire per manual point, each playing the whole inner table"
    )
    assert len(bench.scan_tables) == 3, "the inner table is written per fire"

    schema = value.block.schema
    power = next(axis for axis in schema.point_domain.axes if axis.name == "power")
    # Outermost first: power advances slowest, exactly as the plan reads.
    assert _point_axis_values(schema, "power") == pytest.approx(
        (1.5, 1.5, 2.5, 2.5, 4.0, 4.0)
    )
    assert _point_axis_values(schema, "da_bias_x") == pytest.approx(
        (-256.0, 256.0) * 3
    )
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

    assert bench.fired_repeats == [(1, 1)] * 4, "two walks of two manual points"
    stops = [request.payload["value"] for request in asked]
    assert stops == pytest.approx([1.0, 2.0, 1.0, 2.0])

    schema = value.block.schema
    assert schema.repeat_domain.size == 2
    assert tuple(axis.name for axis in schema.repeat_domain.axes) == (
        "repeat",
        "shots per point",
    )
    assert tuple(axis.size for axis in schema.repeat_domain.axes) == (2, 1)
    assert schema.point_domain.size == 4
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
        assert bench.fired_repeats == [], "nothing fired before the hand answered"
    finally:
        if host is not None:
            host.shutdown()
        bench.close()
        plane.close()
        installation.close()


@pytest.mark.parametrize("port", ("manual:power", "device:rf:power"))
def test_a_host_axis_is_moved_outside_the_board_table(port) -> None:
    """Place host knobs outside the table without changing their coordinates."""

    outer_axis = ScanAxis(port, (135.0, 247.0), "mVpp")
    board_axis = ScanAxis(BIAS_X_PORT, (-256.0, 256.0))
    plan = ScanPlan(
        (board_axis, outer_axis)
    )
    outer, board = split_outer_axes(plan)
    assert plan.axes == (outer_axis, board_axis)
    assert outer == (outer_axis,) and board == (board_axis,)
    assert ScanPlan.from_tree(plan.to_tree()) == plan


def test_manual_axes_alone_repeat_a_fixed_pulse_at_each_confirmation() -> None:
    value, asked, bench = _manual_run(
        manual=(("power", (1.0, 2.0)),), values=None, shots=2, repeats=2,
    )
    assert bench.fired_repeats == [(2, 1)] * 4
    assert bench.loads == 1 and bench.scan_tables == []
    assert all(not sequence.slots for sequence in bench.loaded_sources)
    assert bench._loaded_rows == () and bench._loaded_program.slot_count == 0
    assert bench.published[1:] == list(range(8))
    assert [request.payload["value"] for request in asked] == [1.0, 2.0, 1.0, 2.0]
    assert len({request.request_id for request in asked}) == 4
    schema = value.block.schema
    assert tuple(axis.name for axis in schema.point_domain.axes
                 if axis.axis_id.value.startswith("scan.")) == ("power",)
    assert tuple(axis.size for axis in schema.repeat_domain.axes) == (2, 2)
    assert np.asarray(value.block.values).mean(axis=(2, 3)).tolist() == [
        [0.0, 2.0], [1.0, 3.0], [4.0, 6.0], [5.0, 7.0],
    ]


def _device_run(
    *,
    frequencies: tuple[float, ...],
    values: tuple[float, ...] | None,
    repeats: int = 1,
    shots: int = 1,
    tunables=None,
    unit="",
    device_field="frequency",
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
        source = virtual_rf_source(
            VaunixLmsConfig(
                serial=1001,
                frequency_low_hz=500e6,
                frequency_high_hz=8e9,
                power_low_dbm=-40.0,
                power_high_dbm=10.0,
            )
        )
        # As an operator leaves a brick before scanning it: standing inside
        # the window, where the scan can put it back.
        source.tune("frequency", 600e6)
        tunables = {"rf": source}
    try:
        bench = ScriptedScanBench(
            installation.device("sequencer"),
            plane,
            publications_per_fire=(1 if values is None else len(values)) * shots,
        )
        bench.publish(SCRIPTED_SEED_VALUE)
        plan = ScanPlan(
            (
                ScanAxis(
                    DEVICE_PARAM_FAMILY + "rf:" + device_field, frequencies, unit
                ),
            )
            + (() if values is None else (ScanAxis(BIAS_X_PORT, values),))
        )
        node = descriptors["seamless_scan"].instantiate(
            sequencer=bench,
            signal_plane=plane,
            source_signal=bench.signal_name,
            pulse_resource=_pulse_resource(TEMPLATE_NAME,
                pulse_sequence("mot_field_template.json") if values is None else _template_sequence()),
            plan=plan.to_tree(),
            tunable_devices=tunables,
            repeats=repeats,
            shots_per_point=shots,
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

    assert bench.fired_repeats == [(1, 1)] * 3, (
        "one fire per device point, each playing the whole inner table"
    )

    schema = value.block.schema
    frequency = next(
        axis
        for axis in schema.point_domain.axes
        if axis.name == "rf.frequency"
    )
    assert _point_axis_values(schema, "rf.frequency") == pytest.approx(
        (1e9, 1e9, 1.5e9, 1.5e9, 2e9, 2e9)
    )
    assert frequency.unit == "Hz", "a device axis publishes its knob's unit"

    # The instrument is handed back where the operator left it, and that
    # tune() really ran shows in the epoch: three moves and the way back.
    assert source.tunable_values()["frequency"] == pytest.approx(600e6)
    assert source.settings_provenance()["settings_epoch"] == (
        record["device_snapshots"]["tunable:rf"]["settings_epoch"] + 4
    )

    assert record["named_devices"]["tunable:rf"] == "rf"
    # The snapshot is the device's state at run START: the swept field's
    # truth lives in the axis values above, and what provenance needs
    # beyond it is the UNSWEPT context -- the power and output the whole
    # scan ran at -- plus the identity and epoch to match it to a session.
    snapshot = record["device_snapshots"]["tunable:rf"]
    # The run-start snapshot carries the whole surface, the safety window
    # included: provenance should say what fence the scan ran inside.
    assert set(snapshot["settings"]) == {
        "frequency",
        "power",
        "output_enabled",
        "frequency_low",
        "frequency_high",
        "power_low",
        "power_high",
    }
    assert snapshot["device_session_id"] == (
        source.settings_provenance()["device_session_id"]
    ), "the session the scan ran under, not the instrument's label"
    assert "settings_epoch" in snapshot

    # The console converts these into runtime claims, so a control-panel
    # tune of the swept field is blocked while the scan owns it.
    (claim,) = claims
    assert claim.device_key == "rf"
    assert claim.device is source
    assert claim.protected_fields == ("frequency",)


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


def _device_seamless(
    knob: _Knob, sequencer: _FakeSequencer, source: _FakeSource, *,
    shots: int = 1, acquisition_logic: str = "", restart_logic=None,
    repeats: int = 1,
):
    sequence = _template_sequence()
    pulse_port = hardware_scan_ports_for(sequence)[0]
    device_port = ScanPort(
        DEVICE_PARAM_FAMILY + "knob:level", "knob.level", "1", 0.0, 3.0
    )
    return SeamlessScanMeasurement(
        sequencer=sequencer,
        source=source,
        sequence=sequence,
        pulse_path=Path(TEMPLATE_NAME),
        plan=ScanPlan(
            (
                ScanAxis(device_port.port, (1.0, 2.0)),
                ScanAxis(pulse_port.port, (-256.0, 256.0)),
            )
        ),
        ports=(device_port, pulse_port),
        tunables={"knob": knob},
        repeats=repeats,
        shots_per_point=shots,
        acquisition_logic=acquisition_logic,
        restart_logic=restart_logic,
    )


def test_a_device_axis_is_put_back_however_the_table_ends() -> None:
    """The outer device knob goes back to its pre-run value: complete,
    stopped or failed. Each ready/fire segment reports its committed shots
    while it runs, replacing the preceding acquisition preparation."""

    knob, sequencer, source = _Knob(), _FakeSequencer(_template_sequence()), _FakeSource()
    context = _Context()
    prepared = []
    readout_progress = []
    source.on_take = lambda _taken: readout_progress.append(context.progress[-1])
    _device_seamless(
        knob, sequencer, source, shots=3, repeats=2, acquisition_logic="selected_acquisition",
        restart_logic=lambda name, _context: prepared.append((name, context.commits)),
    ).execute(context)
    assert knob.tunes == [1.0, 2.0, 1.0, 2.0, 0.25] and knob.level == 0.25
    assert sequencer.fires == 4 and sequencer.loads == 1
    assert sequencer.safe_calls == 1, "normal DONE is already safe; only initial SAFE is sent"
    assert prepared == [("selected_acquisition", 0)], "prepare the selected logic once per Scan Start"
    assert all(message.startswith("Scanning point") for message, *_rest in readout_progress)
    scanning = [entry for entry in context.progress if entry[1] is not None]
    assert [(current, total, committed) for _message, current, total, committed in scanning] == [
        (current, 24, current) for base in (0, 6, 12, 18) for current in range(base, base + 7)
    ], "Fire starts at the retained shot count; every committed shot advances it"
    assert scanning[0][0] == "Scanning point 1/8; shots"
    assert scanning[7][0] == "Scanning point 3/8; shots"
    assert scanning[-1][0] == "Scanning point 8/8; shots"

    knob, sequencer = _Knob(), _FakeSequencer(_template_sequence())
    with pytest.raises(RuntimeError, match="scripted source failed"):
        _device_seamless(knob, sequencer, _FakeSource(fail_at=2)).execute(_Context())
    assert knob.tunes == [1.0, 0.25] and knob.level == 0.25

    knob, sequencer, source, context = (
        _Knob(), _FakeSequencer(_template_sequence()), _FakeSource(), _Context()
    )
    source.on_take = lambda taken: setattr(context, "cancelled", taken == 2)
    with pytest.raises(RuntimeError, match="cancelled"):
        _device_seamless(knob, sequencer, source).execute(context)
    assert knob.tunes == [1.0, 0.25] and knob.level == 0.25
    assert sequencer.fires == 1

    knob, sequencer = _Knob(refuse_restore=True), _FakeSequencer(_template_sequence())
    with pytest.raises(RuntimeError, match="scripted device refused restore"):
        _device_seamless(knob, sequencer, _FakeSource()).execute(_Context())


def test_a_stop_received_while_the_board_goes_safe_fires_no_table() -> None:
    """Stop during SAFE's acknowledgement: no load, no fire, knob put back."""

    knob, sequencer, source, context = (
        _Knob(), _FakeSequencer(_template_sequence()), _FakeSource(), _Context()
    )
    sequencer.on_safe = lambda: setattr(context, "cancelled", True)
    with pytest.raises(RuntimeError, match="cancelled"):
        _device_seamless(knob, sequencer, source).execute(context)
    assert sequencer.fires == 0
    assert knob.tunes == [] and knob.level == 0.25, "do not change a device before initial SAFE completes"


def test_a_device_readback_does_not_replace_the_authored_scan_coordinates() -> None:
    """Setpoints stay in the selected unit; readback rounding is not refusal."""

    class _DriftingKnob:
        def tunable_fields(self):
            return (
                TunableField(
                    metadata=AuthoringField(
                        "power",
                        "float",
                        "Power",
                        0.0,
                        minimum=-40.0,
                        maximum=20.0,
                        unit="dBm",
                    ),
                    current=0.0,
                    live_write=True,
                    dependency_group=("power",),
                ),
            )

        def tune(self, name, value):
            del name
            return float(value) + 0.000003

        def tunable_values(self):
            return {"power": 0.0}

        def settings_provenance(self):
            return {"device_session_id": "drift", "settings_epoch": 0}

    wanted = tuple(float(value) for value in np.linspace(135.0, 247.0, 10))
    value, record, _bench, _device, _claims = _device_run(
        frequencies=wanted, values=(-256.0,),
        tunables={"rf": _DriftingKnob()}, unit="mVpp", device_field="power",
    )
    axis = next(axis for axis in value.block.schema.point_domain.axes
                if axis.name == "rf.power")
    assert axis.unit == "mVpp"
    assert tuple(axis.coordinates) == wanted
    assert record["plan"]["axes"][0]["values"] == list(wanted)
    assert record["plan"]["axes"][0]["unit"] == "mVpp"
