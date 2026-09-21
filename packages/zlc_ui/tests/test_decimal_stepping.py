"""Decimal stepping, actual bounds, and visible precision share one value."""

from __future__ import annotations

from decimal import Decimal
import math

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


def test_a_step_clamps_to_the_bound_and_keeps_interior_decimal_grid(box) -> None:
    """Exceeded limits clamp, while interior steps still use their own grid."""

    box.setRange(0.001, 1.0)
    box.setSingleStep(0.1)
    box.setValue(0.2)
    assert _step(box, -1) == "0.1"
    assert _step(box, -1) == "0.001"
    assert _step(box, 1) == "0.1"
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


def test_a_narrowed_range_moves_the_value_before_the_first_step(box) -> None:
    """Qt clamped its shadow and the decimal stayed at 7: the box showed 5,
    stepped from the hidden 7 and was clamped straight back -- the first
    notch down went nowhere.  Every entry that can move the value goes
    through the one decimal authority, and a slot hearing valueChanged from
    inside setRange reads the same number the signal carries."""

    heard: list[tuple[float, Decimal]] = []
    box.valueChanged.connect(lambda value: heard.append((value, box.decimalValue())))
    box.setSingleStep(1)
    box.setValue(7)
    box.setRange(0.0, 5.0)
    assert box.decimalValue() == Decimal(5)
    assert (box.value(), box.text()) == (5.0, "5")
    assert heard[-1] == (5.0, Decimal(5)), heard
    assert _step(box, -1) == "4"
    box.setMinimum(4.5)
    assert (box.decimalValue(), box.text()) == (Decimal("4.5"), "4.5")
    box.setMinimum(0.0)
    box.setMaximum(2.0)
    assert (box.decimalValue(), box.text()) == (Decimal(2), "2")
    box.setValue(1.123)
    box.setDecimals(2)
    assert (box.decimalValue(), box.text()) == (Decimal("1.12"), "1.12")
    box.setSingleStep(0.05)
    assert _step(box, 1) == "1.15"


def test_a_box_with_no_bound_steps_in_a_logarithmic_unit(box) -> None:
    """Qt says "no bound" with the whole double line, and the top of that
    line read in milliwatts is 10**(DBL_MAX / 10): a step that converted
    both ends before moving could not move at all."""

    box.setValueUnit("dBm")
    box.setValue(0.0)
    box.setShownUnit("mW")
    box.setSingleStep(0.1)
    assert box.text() == "1"
    assert _step(box, 1) == "1.1"
    assert abs(box.value() - 10 * math.log10(1.1)) < 1e-12
    # A floor of 0 W is no floor in dBm: there is no dBm for it to be.
    power = type(box)()
    power.setValueUnit("W")
    power.setRange(0.0, 1.0)
    power.setValue(0.001)
    power.setShownUnit("dBm")
    power.setSingleStep(1)
    assert power.text() == "0"
    assert _step(power, -1) == "-1"
    assert abs(power.value() - 10 ** (-0.1) / 1000) < 1e-15


def test_visible_precision_is_the_value_and_resize_is_not_a_user_edit(box) -> None:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from zlc_ui.fluent import signals_blocked

    app = QtWidgets.QApplication.instance()
    normalized, edited = [], []
    box.valueNormalized.connect(lambda: normalized.append(box.decimalValue()))
    box.valueChanged.connect(edited.append)
    box.resize(110, 32)
    box.show()
    try:
        app.processEvents()
        with signals_blocked(box):
            box.setValue(123.456789123456)
        app.processEvents()
        assert Decimal(box.text()) == box.decimalValue()
        assert box.value() == float(box.text())
        assert box.value() != 123.456789123456
        assert QtGui.QFontMetrics(box.lineEdit().font()).horizontalAdvance(box.text()) + 2 <= box._text_width
        assert normalized[-1] == box.decimalValue()
        edited.clear()
        box.resize(90, 32)
        app.processEvents()
        assert not edited, "a resize is normalization, never a user valueChanged"
        assert Decimal(box.text()) == box.decimalValue()
        box.setValueUnit("Hz")
        box.setValue(123456789.123456)
        box.setShownUnit("MHz")
        assert box.decimalValue() == Decimal(box.text()) * Decimal(1000000)
        assert box.validate("1e-", 3)[0] == QtGui.QValidator.Intermediate
        box.setValueUnit("1")
        previous = box.value(), box.width(), box.minimum(), box.maximum()
        box.setRange(1.23456789012345, 1.23456789012346, reject_unrepresentable=True)
        app.processEvents()
        assert (box.value(), box.width(), box.minimum(), box.maximum()) == previous
        assert "too narrow" in box.property("numericError")
        box.setRange(0, 10)
        assert not box.property("numericError")
        from PyQt5 import QtTest
        app.clipboard().setText("9.999999999999999e-16")
        box.lineEdit().setFocus()
        box.lineEdit().selectAll()
        QtTest.QTest.keyClick(box.lineEdit(), QtCore.Qt.Key_V, QtCore.Qt.ControlModifier)
        assert QtGui.QFontMetrics(box.lineEdit().font()).horizontalAdvance(box.text()) + 2 <= box._text_width
        QtTest.QTest.keyClick(box.lineEdit(), QtCore.Qt.Key_Return)
        assert box.value() == 0 and box.text() == "0"
        assert box.setRange(0.001, 2.0, value=-1, unit="ms")
        assert (box.value(), box.text(), box.valueUnit()) == (0.001, "0.001", "ms")
        assert box.setRange(1, 2000, value=3000, unit="µs")
        assert (box.value(), box.text(), box.valueUnit()) == (2000, "2000", "µs")
        assert box.setRange(0, 1.23456789012345, value=2, unit="s")
        assert 1.23 <= box.value() <= box.maximum()
        assert float(box.text()) == box.value() and "e" not in box.text()
        accepted = box.value(), box.minimum(), box.maximum(), box.valueUnit(), box.width()
        assert not box.setRange(1.23456789012345, 1.23456789012346, value=0,
                                unit="ms", reject_unrepresentable=True)
        assert (box.value(), box.minimum(), box.maximum(), box.valueUnit(), box.width()) == accepted
        assert "too narrow" in box.property("numericError")
        # An authoritative device range cannot be refused by a formatter.
        assert not box.setRange(1.23456789012345, 1.23456789012346, value=0, unit="ms")
        assert box.minimum() == 1.23456789012345 and box.maximum() == 1.23456789012346
        assert box.valueUnit() == "ms" and box.value() == box.minimum()
        assert box.text() == "" and not box.hasAcceptableInput()
        app.processEvents()
        assert box.text() == "" and box.property("numericError")
        assert box.setRange(0, 2, value=1, unit="ms")
        assert box.text() == "1" and not box.property("numericError")
        from zlc_ui.fluent import FluentLineEdit
        plain = FluentLineEdit()
        plain.set_numeric_validator("float", bottom=0, top=10)
        plain.resize(110, 32)
        plain.show()
        try:
            app.processEvents()
            plain.setFocus()
            QtTest.QTest.keyClicks(plain, "0.00")
            assert plain.text() == "0.00"
            QtTest.QTest.keyClicks(plain, "5")
            assert plain.text() == "0.005"
            QtTest.QTest.keyClick(plain, QtCore.Qt.Key_Return)
            assert float(plain.text()) == 0.005
        finally:
            plain.close()

        passive = FluentLineEdit("123.456789123456")
        passive.set_numeric_validator("float", bottom=-1e6, top=1e6)
        passive.resize(150, 32)
        passive.show()
        app.processEvents()
        changed, passive_normalized = [], []
        passive.textChanged.connect(changed.append)
        passive.valueNormalized.connect(
            lambda: passive_normalized.append(passive.text())
        )
        passive.resize(70, 32)
        app.processEvents()
        assert passive_normalized and not changed, (
            "a resize was reported as a user edit",
            changed,
            passive_normalized,
        )

        narrow = "1.23456789012345"
        passive.set_numeric_validator(
            "float",
            bottom=float(narrow),
            top=1.23456789012346,
        )
        passive.setText(narrow)
        passive.resize(48, 32)
        app.processEvents()
        assert passive.property("numericError")
        passive.hide()
        passive.setEnabled(False)
        app.processEvents()
        assert not passive.property("numericError")
        passive.setEnabled(True)
        passive.show()
        app.processEvents()
        assert passive.property("numericError"), (
            "re-entering editable mode did not restore real validation"
        )
        passive.close()

        from zlc_ui.fluent import FluentSpinBox
        integer = FluentSpinBox()
        integer.setFixedWidth(80)
        integer.setValue(12)
        integer.show()
        try:
            app.processEvents()
            integer.setRange(1_000_000_000, 2_000_000_000)
            assert integer.minimum() == integer.value() == 1_000_000_000
            assert integer.maximum() == 2_000_000_000
            assert integer.text() == "" and integer.property("numericError")
            app.processEvents()
            assert integer.text() == ""
            integer.setRange(0, 100)
            assert integer.value() == 100 and integer.text() == "100"
            assert not integer.property("numericError")
        finally:
            integer.close()

        # Read-only projections keep the exact text and canonical number;
        # their pixel width is not an authoring rule.
        # The editable cases above still reject a genuinely unrepresentable
        # value and never retain a hidden authored digit.
        readonly = type(box)()
        readonly.resize(76, 32)
        readonly.show()
        app.processEvents()
        exact = 142.1234567890123
        readonly.setEnabled(False)
        assert readonly.setRange(-1e12, 1e12, value=exact, unit="mVpp")
        assert readonly.value() == exact
        assert Decimal(readonly.text()) == readonly.decimalValue()
        assert not readonly.property("numericError")
        readonly.setReadOnly(True)
        readonly.setEnabled(True)
        other_exact = 153.12345678395613
        assert readonly.setRange(-1e12, 1e12, value=other_exact, unit="mVpp")
        assert readonly.value() == other_exact
        assert readonly.text() and not readonly.property("numericError")
        assert readonly.setRange(0, 1e12, value=123456789.123456, unit="Hz")
        readonly.setShownUnit("MHz")
        assert readonly.value() == 123456789.123456
        assert 100 < float(readonly.text()) < 1000
        readonly.interpretText()
        assert readonly.value() == 123456789.123456
        readonly.close()

        readonly_integer = FluentSpinBox()
        integer_normalized = []
        readonly_integer.valueNormalized.connect(lambda: integer_normalized.append(True))
        readonly_integer.setFixedWidth(60)
        readonly_integer.show()
        app.processEvents()
        readonly_integer.setEnabled(False)
        readonly_integer.setRange(1_000_000_000, 2_000_000_000)
        readonly_integer.setValue(1_500_000_000)
        assert readonly_integer.value() == 1_500_000_000
        assert readonly_integer.text()
        assert not readonly_integer.property("numericError")
        app.processEvents()
        assert not integer_normalized, "a read-only projection wrote normalization back"
        readonly_integer.close()

        # Hidden tabs still carry readable form values: changing editability
        # must clear stale width errors before Show, for both spin types.
        for control, bounds in (
            (type(box)(), (1.23456789012345, 1.23456789012346)),
            (FluentSpinBox(), (1_000_000_000, 2_000_000_000)),
        ):
            control.resize(60, 32)
            control.show()
            app.processEvents()
            control.setRange(*bounds)
            app.processEvents()
            assert control.property("numericError")
            for method, locked, unlocked in (
                (control.setEnabled, False, True),
                (control.setReadOnly, True, False),
            ):
                control.hide()
                method(locked)
                assert not control.property("numericError")
                assert float(control.text()) == control.value()
                method(unlocked)
                control.show()
                app.processEvents()
                assert control.property("numericError")
            control.close()
    finally:
        box.close()
        app.processEvents()
