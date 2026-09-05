"""A number box steps in decimal, on its step's grid, inside the owner's bound.

Stepped in binary, 0.1 three times was 0.30000000000000004 and the honest
formatter printed every digit of it; stepped down to a floor that was not
on the grid, 0.1 became 0.001 and could not come back up the way it went
down.  Both are the box deciding arithmetic it was never asked to decide.
"""

from __future__ import annotations

from decimal import Decimal

import pytest


@pytest.fixture
def box():
    pytest.importorskip("PyQt5")
    from zlc_ui.qt import ensure_qt_app
    from zlc_ui.fluent import FluentDoubleSpinBox

    ensure_qt_app(["decimal-stepping"])
    return FluentDoubleSpinBox()


def _step(box, notches: int) -> str:
    box.stepBy(notches)
    return box.text()


def test_stepping_is_decimal_arithmetic(box) -> None:
    box.setRange(0.0, 10.0)
    box.setSingleStep(0.1)
    box.setValue(0.0)
    assert [_step(box, 1) for _ in range(3)] == ["0.1", "0.2", "0.3"]
    assert box.decimalValue() == Decimal("0.3")
    assert box.value() == 0.3
    assert [_step(box, -1) for _ in range(3)] == ["0.2", "0.1", "0"]


def test_a_notch_off_the_grid_goes_to_the_next_grid_point(box) -> None:
    box.setRange(0.0, 10.0)
    box.setSingleStep(0.1)
    box.setValue(0.25)
    assert _step(box, 1) == "0.3"
    box.setValue(0.25)
    assert _step(box, -1) == "0.2"


def test_a_step_stops_at_the_last_grid_point_inside_the_bound(box) -> None:
    """Down from 0.1 in steps of 0.1 with a floor of 0.001 is still 0.1."""

    box.setRange(0.001, 1.0)
    box.setSingleStep(0.1)
    box.setValue(0.2)
    assert _step(box, -1) == "0.1"
    assert _step(box, -1) == "0.1", "the floor is off the grid; a step never lands on it"
    assert _step(box, 1) == "0.2", "and the way back up is the way down"
    box.setValue(0.95)
    assert _step(box, 1) == "1"
    assert _step(box, 1) == "1"


def test_a_typed_number_is_clamped_to_the_owners_bound(box) -> None:
    box.setRange(0.001, 1.0)
    box.setSingleStep(0.1)
    box.lineEdit().setText("0.0002")
    box.interpretText()
    assert box.text() == "0.001"
    assert box.value() == 0.001


def test_the_box_invents_no_bound_of_its_own(box) -> None:
    box.setSingleStep(1)
    box.setValue(1e-7)
    assert box.text() == "0.0000001"
    # Off the integer grid, the first notch down is the grid point below;
    # nothing declared a floor, so the next one is negative.
    assert _step(box, -1) == "0"
    assert _step(box, -1) == "-1"
    box.setValue(123456789012.0)
    assert box.text() == "123456789012"


def test_a_prefix_shift_is_a_decimal_point_moving(box) -> None:
    box.setRange(0.0, 1e12)
    box.setValueUnit("Hz")
    box.setValue(6834700000.0)
    box.setShownUnit("GHz")
    assert box.text() == "6.8347"
    # 6.8347 is not on the 0.001 grid, so the first notch up is 6.835.
    box.setSingleStep(0.001)
    assert _step(box, 1) == "6.835"
    assert box.decimalValue() == Decimal("6835000000")
    assert _step(box, 1) == "6.836"
    box.lineEdit().setText("6.83471")
    box.interpretText()
    assert box.decimalValue() == Decimal("6834710000")
    assert box.value() == 6834710000.0


def test_a_declared_resolution_is_honoured_when_typed(box) -> None:
    box.setRange(0.0, 10.0)
    box.setDecimals(2)
    box.setSingleStep(0.05)
    box.setValue(0.123)
    assert box.text() == "0.12"
    assert _step(box, 1) == "0.15"


def test_an_integer_box_stays_integral(box) -> None:
    box.setRange(0.0, 1e9)
    box.setDecimals(0)
    box.setSingleStep(1)
    box.setValue(41)
    assert _step(box, 1) == "42"
    assert box.decimalValue() == Decimal("42")


def test_a_unit_redeclared_keeps_the_spelling_on_screen(box) -> None:
    """Every projection re-configures the box it keeps; "s" said again is
    not a change, and the "ms" the operator chose to read it in is theirs."""

    box.setRange(0.0, 10.0)
    box.setValueUnit("s")
    box.setValue(0.02)
    box.setShownUnit("ms")
    assert box.text() == "20"
    box.setValueUnit("s")
    assert box.shownUnit() == "ms"
    assert box.text() == "20"
    # A DIFFERENT owner unit is a change: the number is now in hertz, and
    # milliseconds are no spelling of that.
    box.setValueUnit("Hz")
    assert box.shownUnit() == "Hz"
    assert box.text() == "0.02"


def test_a_count_box_counts_in_32_bits() -> None:
    pytest.importorskip("PyQt5")
    from zlc_ui.qt import ensure_qt_app
    from zlc_ui.fluent import fluent_count_box

    ensure_qt_app(["count-box"])
    box = fluent_count_box()
    assert box._step_btn.isHidden(), "a count moves one at a time"
    box.setValue(4294967295)
    assert box.text() == "4294967295"
    assert _step(box, 1) == "4294967295", "uint32 is the board's ceiling"
    box.setValue(41)
    assert _step(box, 1) == "42"
    assert _step(box, -42) == "0"
    assert _step(box, -1) == "0", "and zero is its floor"
    bounded = fluent_count_box(minimum=2)
    bounded.setValue(1)
    assert bounded.text() == "2"
