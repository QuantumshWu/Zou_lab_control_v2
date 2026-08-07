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


def test_a_table_is_played_many_times_without_being_sent_many_times() -> None:
    """The RTL has no sweep register.

    ``scan_count`` is a point count, ``repeat_forever`` is a bit, and
    ``loop_count`` is the pulse timeline's own loop INSIDE each point.  So a
    finite number of sweeps is N*len(rows) points -- but which row a point
    takes is decided when a bank is refilled, so the rows cross the wire once.
    """

    from zlc_pulse import load_streamer_config
    from zlc_pulse.wire import pack_scan_rows

    geometry = load_streamer_config()["params"]
    rows = [(10,), (20,), (30,)]

    def slot0(sweeps):
        words = pack_scan_rows(rows, geometry, 0, 0, sweeps)
        ordered = [words[key] for key in sorted(words)]
        return ordered[:: geometry.num_slots]

    assert slot0(1) == [10, 20, 30]
    assert slot0(3) == [10, 20, 30, 10, 20, 30, 10, 20, 30]
