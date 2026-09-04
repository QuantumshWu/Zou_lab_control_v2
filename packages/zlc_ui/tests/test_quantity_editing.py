"""A number on screen is digits; which unit they are in is said beside them."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Every snippet starts here: the product bootstrap is what puts THIS
#: checkout's layers on the path.
_BOOTSTRAP = "import zou_lab_control" + chr(10)


def _run_qt(code: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        ""
        if environment.get("ZLC_TEST_INSTALLED") == "1"
        else os.pathsep.join((str(ROOT.parents[1]), str(SRC)))
    )
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", _BOOTSTRAP + code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_a_field_holds_digits_and_says_its_unit_beside_them() -> None:
    """The box is for the number; the row says what the number is in.

    Printing the symbol inside as well put the same fact in two places and
    left no room to type in the box it was crowding.  A prefix is not typed
    either: it is a way of SHOWING a number, and the picker beside the field
    is the one place it is asked for -- so there is exactly one spelling of
    the value on screen and one place that decides it.
    """

    _run_qt(
        """
from PyQt5 import QtGui
from zlc_ui.qt import ensure_qt_app
from zlc_ui.form.form import FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FluentParameterForm

app = ensure_qt_app(['quantity'])
spec = FormSpec((
    # required -> a spin box; optional -> a blank-or-number line edit.  Both
    # hold a quantity, and every RF bound in this project is the second kind.
    FormFieldProps(key='drive', kind='float', label='Drive', unit='Hz',
                   default=0.0, required=True, minimum=0.0, maximum=1e12),
    FormFieldProps(key='floor', kind='float', label='Power floor', unit='dBm',
                   minimum=-120.0, maximum=30.0),
    FormFieldProps(key='window', kind='int', label='Window', unit='count',
                   default=0, required=True, minimum=0, maximum=99),
    FormFieldProps(key='ratio', kind='float', label='Ratio',
                   default=0.0, required=True, minimum=-10.0, maximum=10.0),
))
form = FluentParameterForm(
    spec, {'drive': 120000000.0, 'floor': -3.5, 'window': 3, 'ratio': 0.5}
)

drive = form.widget_for('drive')
assert drive.text() == '120000000', drive.text()
assert drive.valueFromText('1050000') == 1050000.0
assert drive.validate('120', 3)[0] == QtGui.QValidator.Acceptable
assert drive.validate('1.05M', 5)[0] == QtGui.QValidator.Invalid, 'a prefix is picked'
assert drive.validate('1.05 MHz', 8)[0] == QtGui.QValidator.Invalid
assert drive.validate('-', 1)[0] == QtGui.QValidator.Intermediate, 'still typing'

# The picker is the ladder, and choosing on it moves the DISPLAY only.
picker = form.unit_picker_for('drive')
assert picker is not None, 'a unit with a ladder is offered'
assert picker.unit() == 'Hz'
drive.setShownUnit('MHz')
assert drive.text() == '120', drive.text()
assert drive.value() == 120000000.0, 'the value never moved'
assert drive.valueFromText('121') == 121000000.0, 'typed in the unit on screen'

floor = form.widget_for('floor')
assert floor.text() == '-3.5', floor.text()
assert form.read_value('floor') == -3.5
floor.setText('-6')
assert form.read_value('floor') == -6.0
floor.setText('')
assert form.read_value('floor') is None, 'blank still means no bound'

# dBm and mW are the same power spelled two ways, and the field is workable
# in either without anybody doing that arithmetic by hand.
floor.setText('-3.0')
floor.setShownUnit('mW')
assert abs(form.read_value('floor') - -3.0) < 1e-9, form.read_value('floor')
assert abs(float(floor.text()) - 0.5011872336272722) < 1e-9, floor.text()

window = form.widget_for('window')
assert window.suffix() == '', repr(window.suffix())
assert form.unit_picker_for('window') is None, 'a count has no ladder'

ratio = form.widget_for('ratio')
assert ratio.text() == '0.5', ratio.text()
assert form.unit_picker_for('ratio') is None, 'a bare number has no unit'
print('ok')
"""
    )


def test_an_unreadable_keystroke_never_ends_the_process() -> None:
    """Qt calls both of these from inside its own event handling.

    An exception out of a Qt slot is not a traceback, it is the end of the
    process -- and the fallback must not ask the box for its value either,
    because value() interprets the text and would come straight back here.
    """

    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentDoubleSpinBox

app = ensure_qt_app(['quantity-safety'])
box = FluentDoubleSpinBox()
box.setRange(-1e18, 1e18)
box.setValueUnit('Hz')
box.setValue(1.0)
assert box.valueFromText('nonsense') == 1.0
assert box.valueFromText('') == 1.0
assert box.valueFromText('5 pixel') == 1.0, 'a unit is not typed into a box'

# A unit nobody registered is a defect where it was declared, not a reason
# for the window to die while painting.  It simply has no ladder to offer.
odd = FluentDoubleSpinBox()
odd.setRange(0.0, 10.0)
odd.setValueUnit('DAC code')
odd.setValue(4.0)
assert odd.text() == '4', odd.text()
print('ok')
"""
    )


def test_a_device_reading_is_shown_the_way_its_editable_twin_is() -> None:
    """Device Control printed str(value) in the one column made to be read."""

    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.form.form import FormFieldProps, FormSpec
from zlc_ui.device_manager.view import DeviceControlView

app = ensure_qt_app(['device-current'])
spec = FormSpec((
    FormFieldProps(key='drive', kind='float', label='Drive', unit='Hz',
                   minimum=0.0, maximum=1e12),
    FormFieldProps(key='mode', kind='text', label='Mode'),
))
projection = {
    'fields': {
        'drive': {'current': 120000000.0, 'desired': 120000000.0, 'editable': True,
                  'live_apply': False, 'live_enabled': True, 'apply_enabled': True,
                  'status': '', 'severity': 'info', 'reason': ''},
        'mode': {'current': 'holding', 'desired': 'holding', 'editable': False,
                 'live_apply': False, 'live_enabled': False, 'apply_enabled': False,
                 'status': '', 'severity': 'info', 'reason': ''},
    },
    'owners': (), 'reason': '', 'risk_accepted': False, 'risk_enabled': False,
}
view = DeviceControlView(spec, projection)
current = {key: widgets[0].text() for key, widgets in view._field_rows.items()}
assert current['drive'] == '120 MHz', current
assert current['mode'] == 'holding', 'a device may report a word, not a number'
print('ok')
"""
    )
