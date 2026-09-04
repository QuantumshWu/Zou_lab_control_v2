"""One scan table, one column per bound slot, seeded by that slot's kind."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_pulse import PulsePeriod, PulseSequence, pulse_target_from_xdc
from zlc_pulse.model import PulseFieldRef, PulseSlot
from zlc_pulse.scan import (
    ScanColumnSpec,
    prepare_scan_application,
    scan_columns_for,
    scan_rows_from_wire,
    scan_rows_to_wire,
    scan_table_template,
    validate_scan_table,
)


def test_binding_cycle_is_owned_by_canonical_pulse_field_kinds() -> None:
    from zlc_pulse.model import (
        FIELD_DAC,
        FIELD_DELAY,
        FIELD_DURATION,
        cycle_binding_kind,
    )

    # Three ways a field can be supplied, so three stops before off again:
    # the board per point, a caller per run, and the pulse's own config.
    for field_kind in (FIELD_DURATION, FIELD_DAC):
        assert cycle_binding_kind(None, field_kind=field_kind) == "scan"
        assert cycle_binding_kind("scan", field_kind=field_kind) == "api"
        assert cycle_binding_kind("api", field_kind=field_kind) == "config"
        assert cycle_binding_kind("config", field_kind=field_kind) is None

    # A delay has no scan-table column, and every other stop is the same.
    assert cycle_binding_kind(None, field_kind=FIELD_DELAY) == "api"
    assert cycle_binding_kind("api", field_kind=FIELD_DELAY) == "config"
    assert cycle_binding_kind("config", field_kind=FIELD_DELAY) is None
    with pytest.raises(ValueError, match="not valid"):
        cycle_binding_kind("scan", field_kind=FIELD_DELAY)
    with pytest.raises(ValueError, match="unknown pulse field kind"):
        cycle_binding_kind(None, field_kind="analog")


def _sequence(slots=(), *, probe_duration=5.0, probe_unit="ms"):
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
        slots=slots,
    )


def test_a_column_is_seeded_by_its_slot_kind() -> None:
    """A DAC column must not inherit a duration's nanosecond range.

    It did once, and every point of a +-512 code sweep came back clamped to
    +511 -- a scan that ran, reported nothing wrong, and measured one value.
    """

    dac_port = next(port for port in pulse_target_from_xdc().ports if port.kind == "dac")
    sequence = _sequence(
        (
            PulseSlot("duration", PulseFieldRef("duration", period_id="probe"), "ns"),
            PulseSlot("dac", PulseFieldRef("dac", period_id="probe", port=dac_port.key), "value"),
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
    # A time column is in the unit its period is written in, and carries the
    # ticks-per-unit that gets it onto the wire.  Generated in ticks, it asked
    # the author to convert; generated in nanoseconds and labelled "ns" while
    # meaning ticks, it ran twenty times long at 50 MHz.
    # This period is written in ms, so its column is too -- one unit for one
    # field, whichever page it is edited on.
    assert not duration.is_dac and duration.unit == "ms"
    assert duration.wire_scale == 1e6 / 20.0
    # The sweep brackets the value the field actually holds: 5 ms.
    assert duration.lo < 5 < duration.hi


def test_a_long_duration_uses_a_full_width_base_and_signed_scan_delta() -> None:
    """The 25-bit DSP operand limits variation, not absolute period length."""

    from zlc_pulse.compile import slot_operand_width

    sequence = _sequence(
        (PulseSlot("duration", PulseFieldRef("duration", period_id="probe"), "s"),),
        probe_duration=1.0,
        probe_unit="s",
    )
    column, = scan_columns_for(sequence)

    delta = (1 << (slot_operand_width() - 1)) * 20.0e-9
    assert column.limit_lo == pytest.approx(20.0e-9)
    assert column.limit_hi > 40.0
    assert column.lo < 1.0 < column.hi
    assert column.wire_offset == -50_000_000.0

    authored = ((0.4,), (1.0,), (1.6,))
    effective, scales, wire = prepare_scan_application(sequence, authored)
    assert scales == (2,)
    assert wire == ((-15_000_000,), (0,), (15_000_000,))
    assert effective == authored
    execution_column, = scan_columns_for(sequence, scales)
    assert scan_rows_from_wire(wire, (execution_column,)) == authored
    assert execution_column.wire_scale == column.wire_scale / 2.0
    assert delta * scales[0] > 0.6

    negative_limit = 1 << (slot_operand_width() - 1)
    positive_limit = negative_limit - 1
    edge_rows = (
        (1.0 - negative_limit * 20.0e-9,),
        (1.0 + positive_limit * 20.0e-9,),
    )
    edge_effective, edge_scales, edge_wire = prepare_scan_application(
        sequence,
        edge_rows,
    )
    assert edge_scales == (1,)
    assert edge_wire == ((-negative_limit,), (positive_limit,))
    np.testing.assert_allclose(edge_effective, edge_rows)

    widest = _sequence(
        (PulseSlot("duration", PulseFieldRef("duration", period_id="probe"), "s"),),
        probe_duration=43.0,
        probe_unit="s",
    )
    widest_rows = (
        (43.0 - negative_limit * 127 * 20.0e-9,),
        (43.0 + positive_limit * 127 * 20.0e-9,),
    )
    widest_effective, widest_scales, widest_wire = prepare_scan_application(
        widest,
        widest_rows,
    )
    assert widest_scales == (127,)
    assert widest_wire == ((-negative_limit,), (positive_limit,))
    np.testing.assert_allclose(widest_effective, widest_rows)

    coarse_effective, coarse_scales, _coarse_wire = prepare_scan_application(
        sequence,
        ((20.0e-9,), (43.5,)),
    )
    assert coarse_scales == (127,)
    assert coarse_effective[0][0] >= 20.0e-9
    assert coarse_effective[1][0] <= column.limit_hi
    with pytest.raises(ValueError, match="become identical"):
        prepare_scan_application(
            sequence,
            ((0.4,), (0.40000002,), (1.6,)),
        )

    with pytest.raises(ValueError, match="s must be within"):
        validate_scan_table(((50.0,),), (column,))

    dac_port = next(
        port for port in pulse_target_from_xdc().ports if port.kind == "dac"
    )
    with pytest.raises(ValueError, match="DAC slot tick scale"):
        scan_columns_for(
            _sequence(
                (PulseSlot("dac", PulseFieldRef("dac", "probe", dac_port.key), "value"),)
            ),
            (2,),
        )


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
        slots=(
            PulseSlot("duration", PulseFieldRef("duration", period_id="probe"), "ms"),
            PulseSlot("dac", PulseFieldRef("dac", period_id="probe", port=dac.key), "value"),
        ),
    )

    resolved = resolve_scan_point(sequence, (7.5, 300))

    assert resolved.slots == (), "a resolved point has nothing left to sweep"
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
        (PulseSlot("duration", PulseFieldRef("duration", period_id="probe"), "ms"),)
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
