"""One scan table, one column per bound slot, seeded by that slot's kind."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_pulse import PulsePeriod, PulseSequence, pulse_target_from_xdc
from zlc_pulse.model import PulseFieldRef, PulseSlot
from zlc_pulse.scan import ScanColumnSpec, scan_columns_for, scan_table_template


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

    # Offset-binary, which is what a DAC slot value on the wire IS: the
    # compiler writes value - signed_range[0].  Offering the SIGNED range and
    # calling 0 "0 V" put the most negative voltage where zero was meant.
    assert dac.is_dac and (dac.lo, dac.hi) == (0.0, float((1 << dac_port.width) - 1))
    assert str(dac_port.safe_value) in dac.unit
    # A time column is in DEVICE TICKS, which is what a slot value on the wire
    # IS.  It was generated in nanoseconds and labelled "ns", so every sweep
    # ran twenty times long at 50 MHz and the "one tick" floor was fifty.
    assert not duration.is_dac and duration.unit == "ticks"
    # The sweep brackets the value the field actually holds: 5 ms = 250 000
    # ticks at the sequence's 20 ns step.
    assert duration.lo < 250_000 < duration.hi


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
