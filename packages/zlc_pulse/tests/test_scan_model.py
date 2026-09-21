"""One scan table, one column per bound slot, seeded by that slot's kind."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from zlc_pulse import PulsePeriod, PulseSequence, pulse_target_from_xdc
from zlc_pulse.model import PulseFieldRef, PulseBinding
from zlc_pulse.scan import (
    ScanColumnSpec,
    prepare_scan_application,
    scan_columns_for,
    scan_rows_from_wire,
    scan_rows_to_wire,
    scan_table_template,
    validate_scan_table,
)


def _sequence(bindings=(), *, probe_duration=5.0, probe_unit="ms"):
    target = pulse_target_from_xdc()
    lanes = len(target.raw_lanes)
    return PulseSequence(
        name="scan",
        target=target,
        time_step_ns=20.0,
        periods=(
            PulsePeriod("load", 2.0, "ms", (0,) * lanes),
            PulsePeriod("probe", probe_duration, probe_unit, (0,) * lanes),
        ),
        bindings=bindings,
    )


def test_a_column_is_seeded_by_its_slot_kind() -> None:
    """A DAC column must not inherit a duration's nanosecond range.

    It did once, and every point of a +-512 code sweep came back clamped to
    +511 -- a scan that ran, reported nothing wrong, and measured one value.
    """

    dac_port = next(port for port in pulse_target_from_xdc().ports if port.kind == "dac")
    sequence = _sequence(
        (
            PulseBinding(PulseFieldRef('duration', period_id='probe'), 'ns', scan=True),
            PulseBinding(PulseFieldRef('dac', period_id='probe', port=dac_port.key), 'value', scan=True),
        )
    )
    duration, dac = scan_columns_for(sequence)

    # The SIGNED code, which is what the operator types into that same DAC's
    # box on the Edit page.  The wire holds offset-binary, and the column says
    # how to get there rather than making the author do it: describing the
    # column in wire units gave one output two number systems, and only the
    # window knew which one it was showing.
    low, high = dac_port.signed_range
    assert dac.is_dac and (dac.lo, dac.hi) == (float(low), float(high))
    assert (dac.wire_scale, dac.wire_offset) == (1.0, float(-low))
    with pytest.raises(ValueError, match="DAC code"):
        validate_scan_table(((float(high) + 0.5,),), (dac,))
    # API and Scan share the binding unit, even when the period stores ms.
    assert not duration.is_dac and duration.unit == "ns"
    assert duration.wire_scale == 1.0 / 20.0
    assert duration.lo < 5e6 < duration.hi

    from zlc_pulse import compile_sequence
    from zlc_pulse.binding import field_label
    from zlc_pulse.scan import api_parameter_columns_for
    from zlc_pulse.wire import StreamerParams

    from zlc_pulse import resolve_api_parameters, resolve_scan_point
    microseconds = replace(sequence, bindings=(
        PulseBinding(PulseFieldRef("duration", "probe"), "us", scan=True, source="api"),
    ))
    scan_column, = scan_columns_for(microseconds)
    api_column, = api_parameter_columns_for(microseconds)
    assert scan_column.unit == api_column.unit == "µs"
    selected = resolve_scan_point(microseconds, (50.0,))
    assert selected.period_by_id["probe"].duration == pytest.approx(0.05)
    assert selected.period_by_id["probe"].unit == "ms"
    assert not selected.api_bindings and not selected.scan_bindings
    effective, wire = prepare_scan_application(microseconds, ((50.0,), (100.0,)))
    assert effective == ((50.0,), (100.0,))
    compiled = compile_sequence(resolve_api_parameters(microseconds), StreamerParams(), 50e6)
    # The wire holds the row's whole tick count.
    assert wire == ((2500,), (5000,))
    assert compiled.resolved_durations(wire[0]) == compile_sequence(
        selected, StreamerParams(), 50e6
    ).durations

    renamed = replace(sequence, periods=(
        sequence.periods[0], replace(sequence.periods[1], name="MOT"),
    ))
    named_duration, named_dac = scan_columns_for(renamed)
    assert named_duration.name == duration.name == sequence.scan_bindings[0].field_id
    assert named_duration.label == "MOT.duration"
    assert named_dac.label == f"MOT.{dac_port.label or dac_port.key}"
    assert renamed.scan_bindings == sequence.scan_bindings
    original_time = replace(sequence, bindings=sequence.scan_bindings[:1])
    renamed_time = replace(renamed, bindings=renamed.scan_bindings[:1])
    assert compile_sequence(original_time, StreamerParams(), 50e6) == compile_sequence(renamed_time, StreamerParams(), 50e6)
    digital = next(port for port in renamed.target.ports if port.kind == "digital")
    delayed = replace(renamed, bindings=(PulseBinding(PulseFieldRef('delay', port=digital.key), 'ns', source='api'),))
    delay_column, = api_parameter_columns_for(delayed)
    assert delay_column.name == f"delay:{digital.key}"
    assert delay_column.label == field_label(delayed, delayed.api_bindings[0].field_ref) == f"{digital.label or digital.key}.delay"
    assert ScanColumnSpec("bare_variable", 1, 2).label == "bare_variable"
    for periods in (
        (replace(sequence.periods[0], name="MOT"), renamed.periods[1]),
        (sequence.periods[0], replace(sequence.periods[1], name="load")),
    ):
        with pytest.raises(ValueError, match="period names must be unique"):
            replace(sequence, periods=periods)


def test_a_duration_column_spans_one_tick_to_the_whole_row_counter() -> None:
    """A duration slot is the row's tick count: one tick up to 2^32-1 of them."""

    sequence = _sequence(
        (PulseBinding(PulseFieldRef('duration', period_id='probe'), 's', scan=True),),
        probe_duration=1.0,
        probe_unit="s",
    )
    column, = scan_columns_for(sequence)

    assert column.limit_lo == pytest.approx(20.0e-9)
    assert column.limit_hi == pytest.approx(((1 << 32) - 1) * 20.0e-9)
    assert column.lo < 1.0 < column.hi
    assert column.wire_offset == 0.0 and column.wire_scale == pytest.approx(50e6)

    authored = ((0.4,), (1.0,), (1.6,))
    effective, wire = prepare_scan_application(sequence, authored)
    assert wire == ((20_000_000,), (50_000_000,), (80_000_000,))
    assert effective == authored
    assert scan_rows_from_wire(wire, (column,)) == authored

    edge_rows = ((20.0e-9,), (((1 << 32) - 1) * 20.0e-9,))
    edge_effective, edge_wire = prepare_scan_application(sequence, edge_rows)
    assert edge_wire == ((1,), ((1 << 32) - 1,))
    np.testing.assert_allclose(edge_effective, edge_rows)
    with pytest.raises(ValueError, match="become identical"):
        prepare_scan_application(
            sequence,
            ((0.4,), (0.4 + 1e-10,), (1.6,)),
        )

    with pytest.raises(ValueError, match="s must be within"):
        validate_scan_table(((100.0,),), (column,))
    with pytest.raises(ValueError, match="s must be within"):
        validate_scan_table(((0.0,),), (column,))


def test_the_starter_program_builds_one_table_of_the_right_width() -> None:
    """scan_table is the contract with whoever runs the program."""

    columns = (
        ScanColumnSpec("a", 20.0, 200.0),
        ScanColumnSpec("b", -512.0, 511.0, True, "value"),
    )
    namespace: dict = {}
    exec(scan_table_template("column_stack", columns), namespace)  # noqa: S102

    table = np.asarray(namespace["scan_table"])
    assert table.ndim == 2 and table.shape[1] == len(columns)
    assert table.shape[0] > 1
    # The DAC column is integer codes, not fractions of one.
    assert np.all(table[:, 1] == np.round(table[:, 1]))


def test_a_slot_id_is_never_a_name_in_the_starter_program() -> None:
    """Slot ids live in the pulse's namespace; the program has its own.

    ``np``, ``for`` and ``N`` are all legal slot ids.  A starter that spelled
    its axes after them shadowed its own import, was a syntax error, or made
    the point count an array -- the product's own template could not run.
    """

    for names in (("np",), ("for",), ("N", "other")):
        columns = tuple(ScanColumnSpec(name, 20.0, 200.0) for name in names)
        for kind in ("column_stack", "grid"):
            source = scan_table_template(kind, columns)
            # The id is still said, where a person reads it.
            assert all(name in source for name in names), source
            namespace: dict = {}
            exec(source, namespace)  # noqa: S102
            table = np.asarray(namespace["scan_table"])
            assert table.ndim == 2 and table.shape[1] == len(columns), (kind, names)
            assert table.shape[0] > 1


def test_a_grid_sweeps_every_combination_and_says_its_shape() -> None:
    columns = (ScanColumnSpec("a", 0.0, 4.0), ScanColumnSpec("b", 0.0, 3.0))
    namespace: dict = {}
    exec(scan_table_template("grid", columns), namespace)  # noqa: S102

    table = np.asarray(namespace["scan_table"])
    shape = namespace["scan_shape"]
    assert table.shape == (int(np.prod(shape)), len(columns))
    assert len({tuple(row) for row in table}) == table.shape[0], "a grid repeated a point"


def test_a_column_needs_a_range_to_sweep() -> None:
    with pytest.raises(ValueError):
        ScanColumnSpec("a", 1.0, 1.0)


def test_scan_rows_are_packed_once_without_materializing_repeats() -> None:

    from zlc_pulse import load_streamer_config
    from zlc_pulse.wire import pack_scan_rows

    geometry = load_streamer_config()["params"]
    rows = [(10,), (20,), (30,)]

    words = pack_scan_rows(rows, geometry, 0, 0)
    ordered = [words[key] for key in sorted(words)]
    assert ordered[:: geometry.num_slots] == [10, 20, 30]


def test_scan_sweeps_stream_table_local_chunks_into_alternating_banks() -> None:
    """A partial final chunk wraps to chunk zero without copying the table."""

    from dataclasses import replace

    from zlc_pulse import load_streamer_config
    from zlc_pulse.wire import pack_scan_rows

    geometry = replace(load_streamer_config()["params"], bank_size=4)
    rows = [(value,) for value in range(10)]
    chunks_per_sweep = 3

    played: list[int] = []
    for stream_chunk in range(chunks_per_sweep * 3):
        table_chunk = stream_chunk % chunks_per_sweep
        words = pack_scan_rows(
            rows,
            geometry,
            stream_chunk & 1,
            table_chunk,
        )
        ordered = [words[key] for key in sorted(words)]
        played.extend(ordered[:: geometry.num_slots])

    assert played == list(range(10)) * 3


def test_a_resolved_scan_point_is_a_plain_pulse_carrying_that_row() -> None:
    """A held point is an ordinary pulse, not a scan of length one.

    Saying it the other way round is what broke holding on hardware: the board
    was handed a one-point table looping forever -- a state nothing else ever
    asks it for -- and its DAC segments were never re-applied while the digital
    edges kept playing.  Resolve the point into the document and run a plain
    pulse; this is that.
    """

    from zlc_pulse import AnalogStep, resolve_scan_point

    target = pulse_target_from_xdc()
    lanes = len(target.raw_lanes)
    dac = next(port for port in target.ports if port.kind == "dac")
    sequence = PulseSequence(
        name="scan",
        target=target,
        time_step_ns=20.0,
        periods=(
            PulsePeriod("load", 2.0, "ms", (0,) * lanes),
            PulsePeriod("probe", 5.0, "ms", (0,) * lanes,
                        (AnalogStep(dac.key, "edge", 0),)),
        ),
        bindings=(PulseBinding(PulseFieldRef('duration', period_id='probe'), 'ms', scan=True), PulseBinding(PulseFieldRef('dac', period_id='probe', port=dac.key), 'value', scan=True)),
    )

    resolved = resolve_scan_point(sequence, (7.5, 300))

    assert resolved.scan_bindings == (), "a resolved point has nothing left to sweep"
    probe = next(item for item in resolved.periods if item.period_id == "probe")
    assert probe.duration == pytest.approx(7.5), "the duration slot did not land"
    assert probe.analog_steps[0].value == 300, "the DAC slot did not land"
    assert probe.analog_steps[0].mode == "edge", "resolving changed what it does"
    # Everything the row did not name is untouched.
    load = next(item for item in resolved.periods if item.period_id == "load")
    assert load.duration == pytest.approx(2.0)
    assert resolved.target is sequence.target


def test_a_scan_point_must_have_one_value_per_slot() -> None:
    """A short row would silently leave a field on yesterday's number."""

    from zlc_pulse import resolve_scan_point

    sequence = _sequence(
        (PulseBinding(PulseFieldRef('duration', period_id='probe'), 'ms', scan=True),)
    )
    with pytest.raises(ValueError, match="one value per slot"):
        resolve_scan_point(sequence, (1.0, 2.0))


def test_a_template_for_nothing_is_refused_rather_than_invented() -> None:
    """With no bound field there is no column, so there is no starter program.

    A stand-in column made the file state two falsehoods -- a slot named s0
    that does not exist, and, because a stand-in carries no limits, a legal
    range of "0 .. 0" under a comment saying that is what the board accepts.
    The table it built was then refused, so the operator read the truth one
    click later than the file had already implied it.
    """

    for kind in ("column_stack", "grid"):
        with pytest.raises(ValueError, match="at least one bound field"):
            scan_table_template(kind, ())
