"""One scan table, one column per bound slot, seeded by that slot's kind."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_pulse import PulsePeriod, PulseSequence, pulse_target_from_xdc
from zlc_pulse.model import PulseFieldRef, PulseSlot
from zlc_pulse.scan import (
    ScanColumnSpec,
    scan_columns_for,
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

    for field_kind in (FIELD_DURATION, FIELD_DAC):
        assert cycle_binding_kind(None, field_kind=field_kind) == "scan"
        assert cycle_binding_kind("scan", field_kind=field_kind) == "api"
        assert cycle_binding_kind("api", field_kind=field_kind) is None

    assert cycle_binding_kind(None, field_kind=FIELD_DELAY) == "api"
    assert cycle_binding_kind("api", field_kind=FIELD_DELAY) is None
    with pytest.raises(ValueError, match="not valid"):
        cycle_binding_kind("scan", field_kind=FIELD_DELAY)
    with pytest.raises(ValueError, match="unknown pulse field kind"):
        cycle_binding_kind(None, field_kind="analog")


def _sequence(slots=()):
    target = pulse_target_from_xdc()
    lanes = len(target.raw_lanes)
    return PulseSequence(
        name="scan",
        target=target,
        time_step_ns=20.0,
        periods=(
            PulsePeriod("load", 2.0, "ms", (0,) * lanes),
            PulsePeriod("probe", 5.0, "ms", (0,) * lanes),
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


def test_a_column_says_what_the_board_will_accept_not_only_what_to_try() -> None:
    """The seeded sweep and the legal range are different questions.

    They were one pair of numbers, so the only thing that knew a duration slot
    cannot exceed a 25-bit multiplier operand was the device -- which said so
    in ticks, at load time, having already compiled:

        scan slot 0 value 50000000000 does not fit the board's 25-bit signed
        multiplier operand ([-16777216, 16777215])

    ...for a table whose author had written milliseconds.
    """

    from zlc_pulse.compile import slot_operand_width

    sequence = _sequence(
        (PulseSlot("duration", PulseFieldRef("duration", period_id="probe"), "ms"),)
    )
    column, = scan_columns_for(sequence)

    longest = ((1 << (slot_operand_width() - 1)) - 1) * 20.0 / 1e6
    assert column.limit_hi == pytest.approx(longest)
    # Seeded around the 5 ms the field holds, not across the board's reach.
    assert column.lo < 5 < column.hi < column.limit_hi

    with pytest.raises(ValueError, match="ms must be within"):
        validate_scan_table(np.linspace(1e6, 10e6, 5).reshape(-1, 1), (column,))


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


def test_cycle_count_reuses_one_value_table_without_copying_rows() -> None:

    from zlc_pulse import load_streamer_config
    from zlc_pulse.wire import pack_scan_rows

    geometry = load_streamer_config()["params"]
    rows = [(10,), (20,), (30,)]

    def slot0(cycles):
        words = pack_scan_rows(rows, geometry, 0, 0, cycles)
        ordered = [words[key] for key in sorted(words)]
        return ordered[:: geometry.num_slots]

    assert slot0(3) == [10, 20, 30]
    assert slot0(9) == [10, 20, 30, 10, 20, 30, 10, 20, 30]


def test_a_repeated_table_is_seamless_across_a_bank_boundary() -> None:
    """The hard case: a table length that does NOT divide the bank size.

    The end of one table traversal then falls in the MIDDLE of a chunk.  It
    stays seamless because a finite cycle count does not re-arm anything: the
    board is given one stream of cycles and the host fills chunks 0,1,2,... into
    alternating banks, exactly as it does for any long scan.  The boundary
    between two traversals is the boundary between two points and nothing else --
    no re-arm, no bank-parity wrap, no gap.

    (repeat_forever is the other path, where the RTL repeats the application
    until SAFE.  That one is untouched.)
    """

    from dataclasses import replace

    from zlc_pulse import load_streamer_config
    from zlc_pulse.wire import pack_scan_rows

    geometry = replace(load_streamer_config()["params"], bank_size=4)
    rows = [(10,), (20,), (30,)]

    played: list[int] = []
    for chunk in range(4):
        words = pack_scan_rows(rows, geometry, chunk & 1, chunk, 9)
        if not words:
            break
        ordered = [words[key] for key in sorted(words)]
        played.extend(ordered[:: geometry.num_slots])

    assert played == [10, 20, 30] * 3


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
