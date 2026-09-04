"""A number on screen says what it is in, and can be typed back that way."""

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


def test_a_field_is_shown_and_read_in_the_unit_it_declared() -> None:
    """``FormFieldProps.unit`` was declared by owners and read by nobody.

    The word "unit" did not appear once in the whole Qt form layer, so every
    box in this project showed a bare repr of a float and refused anything
    that was not one -- the box that means megahertz being exactly the box
    into which megahertz could not be typed.
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
assert drive.text() == '120.0000000 MHz', drive.text()
assert drive.valueFromText('1.05M') == 1050000.0
assert drive.valueFromText('1.05 MHz') == 1050000.0
assert drive.valueFromText('1050 kHz') == 1050000.0
assert drive.validate('1.05M', 5)[0] == QtGui.QValidator.Acceptable

floor = form.widget_for('floor')
assert floor.text() == '-3.5 dBm', floor.text()
assert floor.validator().validate('-6 dBm', 6)[0] == QtGui.QValidator.Acceptable
assert form.read_value('floor') == -3.5
floor.setText('-6 dBm')
assert form.read_value('floor') == -6.0, 'a unit may be typed into an optional field'
floor.setText('')
assert form.read_value('floor') is None, 'blank still means no bound'

window = form.widget_for('window')
assert window.suffix() == ' count', repr(window.suffix())

ratio = form.widget_for('ratio')
assert ratio.text() == '0.5', ratio.text()
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
box.setDisplayUnit('Hz')
box.setValue(1.0)
assert box.valueFromText('nonsense') == 1.0
assert box.valueFromText('') == 1.0
assert box.valueFromText('5 pixel') == 1.0, 'a wrong dimension is not a value'

# A unit nobody registered is a defect where it was declared, not a reason
# for the window to die while painting.
odd = FluentDoubleSpinBox()
odd.setRange(0.0, 10.0)
odd.setDisplayUnit('DAC code')
odd.setValue(4.0)
assert odd.text() == '4.0 DAC code', odd.text()
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
assert current['drive'] == '120.0000000 MHz', current
assert current['mode'] == 'holding', 'a device may report a word, not a number'
print('ok')
"""
    )
