from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO = ROOT.parents[1]


#: Every snippet starts here.  Without it the subprocess resolves the
#: layers through whatever the editable install points at -- on this
#: machine, sibling checkouts of the same package names -- so the suite
#: silently tested a DIFFERENT zlc_plot than the one beside it.  The
#: product bootstrap is what puts this checkout's layers on the path,
#: and it is the same one every launcher uses.
_BOOTSTRAP = "import zou_lab_control; print(zou_lab_control.__file__)" + chr(10)


def _run_qt(code: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        "" if environment.get("ZLC_TEST_INSTALLED") == "1"
        else os.pathsep.join(
            value
            for value in (str(REPO), str(SRC), environment.get("PYTHONPATH", ""))
            if value
        )
    )
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", _BOOTSTRAP + code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_device_manager_construct_and_plain_data_setters() -> None:
    _run_qt(
        """import zou_lab_control
import zlc_ui.device_manager.view as tested_module
print(tested_module.__file__)
from zlc_ui.device_manager import DeviceManagerView
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['test'])
view = DeviceManagerView()
view.set_device_choices((
    ('Sensor', 'sensor.fake', 'sensor'),
    ('Camera', 'camera.fake', 'camera'),
))
view.set_devices((('id-1', 'input', 'sensor.fake', 'sensor'),))
spec = FormSpec((FormFieldProps('count', 'int', 'Count', default=2, minimum=0, maximum=99),))
view.set_form_spec('id-1', spec, (('count', 7),))
assert view.read_values('id-1') == (('count', 7),)
form = view._cards['id-1'].form
field = form.widget_for('count')
view.set_form_spec('id-1', spec, (('count', 8),))
assert view._cards['id-1'].form is form
assert form.widget_for('count') is field
assert view.read_values('id-1') == (('count', 8),)
view.show_status('ready', 'idle')
assert view.status_strip.text() == 'ready'
assert view.status_strip.current_severity == 'idle'
assert tuple(view.domain_groups) == ('sensor', 'camera')
assert view._cards['id-1'].role_edit.text() == 'input'
card = view._cards['id-1']
view.set_devices(())
assert not card.isWindow(), 'retiring a card briefly promoted it to a top-level window'
assert card.isHidden()
"""
    )


def test_device_manager_qtest_signal_payloads() -> None:
    _run_qt(
        """import zou_lab_control
import zlc_ui.device_manager.view as tested_module
print(tested_module.__file__)
from PyQt5 import QtCore, QtTest
from zlc_ui.device_manager import DeviceManagerView
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['test'])
view = DeviceManagerView()
view.set_device_choices((
    ('Sensor', 'sensor.fake', 'sensor'),
    ('Other sensor', 'sensor.other', 'sensor'),
    ('Camera', 'camera.fake', 'camera'),
))
view.set_devices((('id-1', 'input', 'sensor.fake', 'sensor'),))
view.set_form_spec('id-1', FormSpec((FormFieldProps('count', 'int', 'Count', default=2, minimum=0, maximum=99),)), (('count', 2),))
view.show(); app.processEvents()
events = []
view.device_add_requested.connect(lambda value: events.append(('add', value)))
view.device_remove_requested.connect(lambda value: events.append(('remove', value)))
view.role_committed.connect(lambda instance_id, value: events.append(('role', instance_id, value)))
view.type_picked.connect(lambda instance_id, value: events.append(('type', instance_id, value)))
view.parameter_committed.connect(lambda instance_id, key: events.append(('parameter', instance_id, key)))
card = view._cards['id-1']
QtTest.QTest.mouseClick(view.domain_add_buttons['sensor'], QtCore.Qt.LeftButton)
card.role_edit.setFocus()
card.role_edit.selectAll()
QtTest.QTest.keyClicks(card.role_edit, 'output')
QtTest.QTest.keyClick(card.role_edit, QtCore.Qt.Key_Return)
card.type_combo.setCurrentIndex(1)
card.form.widget_for('count').setValue(3)
QtTest.QTest.mouseClick(card.remove_button, QtCore.Qt.LeftButton)
assert ('add', 'sensor.fake') in events
assert ('role', 'id-1', 'output') in events
assert ('type', 'id-1', 'sensor.other') in events
assert ('parameter', 'id-1', 'count') in events
assert ('remove', 'id-1') in events
"""
    )


def test_device_manager_demo_is_a_reusable_human_entry() -> None:
    _run_qt(
        """import zou_lab_control
import zlc_ui.device_manager.view as tested_module
print(tested_module.__file__)
from examples.demo_device_manager import create_window
from zlc_ui.device_manager import DeviceManagerHandle
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['device-demo'])
handle = create_window(window_ratio=0.4)
assert isinstance(handle, DeviceManagerHandle)
assert not isinstance(handle, QtWidgets.QWidget), "a widget escaped zlc_ui"
# Reaching the view is this package's own business; the demo cannot.
view = handle._view
assert tuple(view._cards) == ('sensor-1', 'camera-1')
assert view._cards['sensor-1'].form.widget_for('count').value() == 4
assert view._cards['camera-1'].form.widget_for('count').value() == 2
assert view.status_strip.text() == 'Offline fake devices · edit only'
"""
    )


def test_device_manager_keeps_the_compact_config_surface_and_lifecycle_verbs() -> None:
    _run_qt(
        """import zou_lab_control
import zlc_ui.device_manager.view as tested_module
print(tested_module.__file__)
from zlc_ui.device_manager import DeviceManagerView
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['device-manager-config-surface'])
view = DeviceManagerView()
view.set_device_choices((
    ('Virtual camera', 'camera.virtual', 'camera'),
    ('Hardware camera', 'camera.dcam', 'camera'),
    ('Virtual sequencer', 'sequencer.virtual', 'sequencer'),
))
view.set_templates((('Virtual', 'virtual'), ('Hardware', 'hardware')))
view.resize(1100, 700); view.show(); app.processEvents()
assert view.tabs.tabText(0) == 'Config'
assert view.heading_label.text() == 'Devices'
assert view.document_name.text() == 'untitled'
assert tuple(view.domain_groups) == ('camera', 'sequencer')
assert tuple(group.title() for group in view.domain_groups.values()) == ('Camera', 'Sequencer')
assert all(button.text() == 'Add device' for button in view.domain_add_buttons.values())
assert view.discovered_group.title() == 'Discovered hardware'
assert view.discover_button.text() == 'Scan hardware'
assert not view.discover_button.isEnabled()
assert view.discover_button.toolTip() == 'No installed device type declares discovery'
assert view.loaded_group.title() == 'Loaded session'
group_titles = {group.title() for group in view.findChildren(type(view.loaded_group))}
assert not {'Installation', 'Configured devices', 'Available', 'Loaded (session)'} & group_titles
assert view.new_combo.itemText(0) == 'New…'
assert tuple(view.new_combo.itemText(index) for index in range(1, view.new_combo.count())) == ('Virtual', 'Hardware')
# FluentComboBox paints its own collapsed text, so its natural width must use
# the same chrome budget as that painter.  Depending on QComboBox's unrelated
# native size hint leaves less room than the widest item and visibly elides it.
from zlc_ui.fluent import scaled_px
from zlc_ui.fluent.style import COMBO_WIDTH, EDIT_PADDING_H
pad = scaled_px(EDIT_PADDING_H)
text_inset = pad + scaled_px(2)
text_width = view.new_combo.width() - scaled_px(COMBO_WIDTH) - text_inset - pad
measure = getattr(view.new_combo.fontMetrics(), 'horizontalAdvance', view.new_combo.fontMetrics().width)
assert text_width >= max(
    measure(view.new_combo.itemText(index))
    for index in range(view.new_combo.count())
)
assert view.load_button.text() == 'Load…'
assert view.save_button.text() == 'Save'
assert view.save_as_button.text() == 'Save as…'
assert not hasattr(view, 'cancel_button')
assert view.lifecycle_button.text() == 'Init devices'
assert not hasattr(view, 'test_button')
events = []
view.load_requested.connect(lambda: events.append('load'))
view.save_as_requested.connect(lambda: events.append('save-as'))
view.discovery_requested.connect(lambda: events.append('discover'))
view.lifecycle_requested.connect(lambda: events.append('lifecycle'))
view.template_selected.connect(lambda name: events.append(('template', name)))
for button in (
    view.load_button,
    view.save_as_button,
):
    button.click()
view.set_discovery_enabled(True)
view.discover_button.click()
view.new_combo.setCurrentIndex(1)
view.new_combo.activated[int].emit(1)
view.set_lifecycle('Init devices', enabled=True, active=False)
view.lifecycle_button.click()
assert events == ['load', 'save-as', 'discover', ('template', 'virtual'), 'lifecycle']
view.set_lifecycle('Apply device changes', enabled=True, active=True, changed=True)
assert view.status_dot.toolTip() == 'Configuration differs from the active installation'
"""
    )


def test_loaded_device_opens_one_independent_generic_control_surface() -> None:
    _run_qt(
        """import zou_lab_control
print(zou_lab_control.__file__)
import zlc_ui.device_manager.view as tested_module
print(tested_module.__file__)
from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui import open_device_control
from zlc_ui.device_manager import DeviceManagerHandle, DeviceManagerView
from zlc_ui.device_manager.handle import DeviceControlHandle
from zlc_ui.fluent import WINDOW_SCREEN_FRACTION, screen_fit_window_size
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FluentParameterForm
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['device-control'])

manager = DeviceManagerView()
manager_handle = DeviceManagerHandle(None, manager)
opened = []
closed_devices = []
manager_handle.device_open_requested.connect(opened.append)
manager_handle.device_close_requested.connect(closed_devices.append)
manager.set_loaded_devices((('camera', 'qCMOS', 'camera.dcam'),))
manager.resize(900, 600); manager.show(); app.processEvents()
card = manager._loaded_cards['camera']
assert card.role_label.text() == 'qCMOS'
assert card.control_button.text() == 'Control'
assert card.close_button.text() == 'Close'
assert card.findChildren(FluentParameterForm) == []
manager.set_lifecycle('Applying device changes', enabled=True, active=True, busy=True)
assert not card.isEnabled()
manager.set_lifecycle('Shutdown devices', enabled=True, active=True, busy=False)
assert card.isEnabled()
QtTest.QTest.mouseClick(card.control_button, QtCore.Qt.LeftButton)
assert opened == ['camera']
QtTest.QTest.mouseClick(card.close_button, QtCore.Qt.LeftButton)
assert closed_devices == ['camera']

spec = FormSpec((
    FormFieldProps('gain', 'int', 'Gain', default=2, minimum=0, maximum=20),
))
control = open_device_control(
    title='qCMOS control',
    spec=spec,
    projection={
        'owners': ('Camera Measurement',),
        'reason': 'gain is unclaimed but requires operator risk acceptance',
        'risk_enabled': True,
        'risk_accepted': False,
        'fields': {
            'gain': {
                'current': 2,
                'desired': 2,
                'editable': False,
                'live_apply': False,
                'live_enabled': False,
                'apply_enabled': False,
                'status': 'Risk acceptance required',
                'severity': 'warning',
                'reason': 'Accept risk to edit this unclaimed live-safe field',
            },
        },
    },
)
assert isinstance(control, DeviceControlHandle)
assert not isinstance(control, QtWidgets.QWidget), 'a QWidget escaped zlc_ui'
# Snug fit (user decree): the control opens at half the standard width and
# its content's own height -- a fixed screen fraction full of blank space
# is exactly what it must not be.  It stays resizable.
standard = screen_fit_window_size(WINDOW_SCREEN_FRACTION)
size = control._window.size()
view_hint = control._view.sizeHint()
assert size.width() <= max(standard.width() // 2, view_hint.width()) + 1, (
    size.width(), standard.width(), view_hint.width(),
)
assert size.height() < standard.height(), (size.height(), standard.height())
assert size.height() >= view_hint.height(), (size.height(), view_hint.height())
refresh_events = []
risk_events = []
desired_events = []
live_events = []
apply_events = []
control.refresh_requested.connect(lambda: refresh_events.append(True))
control.risk_toggled.connect(risk_events.append)
control.field_desired_changed.connect(lambda key, value: desired_events.append((key, value)))
control.field_live_apply_toggled.connect(lambda key, value: live_events.append((key, value)))
control.field_apply_requested.connect(lambda key, value: apply_events.append((key, value)))
assert not hasattr(control, 'field_committed')
assert not hasattr(control, 'set_form')
assert not hasattr(control, 'read_values')
assert control._view.owner_label.text() == 'Owner: Camera Measurement'
assert 'operator risk acceptance' in control._view.reason_label.text()
assert control._view.current_heading.text() == 'Current'
assert control._view.desired_heading.text() == 'Desired'
assert control._view.live_heading.text() == 'Live'
# The headings and the rows are two layouts pretending to be one table, so
# every column must be at the same x and the same width in both -- Field,
# Current and Desired were kept in step by hand while Live, Apply and Status
# were given round numbers no widget had a reason to match.
control._window.resize(1000, 400)
control._window.show()
for _ in range(4):
    app.processEvents()
control._view._align_headings()
app.processEvents()
row = next(iter(control._view.form._rows.values()))
cells = [row.layout().itemAt(i).widget() for i in range(row.layout().count())]
headings = [
    control._view.field_heading, control._view.current_heading,
    control._view.desired_heading, control._view.limits_heading,
    control._view.live_heading, control._view.apply_heading,
    control._view.status_heading,
]
assert len(cells) == len(headings), (len(cells), len(headings))
for heading, cell in zip(headings, cells):
    assert (heading.x(), heading.width()) == (cell.x(), cell.width()), (
        heading.text(), heading.x(), heading.width(), cell.x(), cell.width()
    )
control._view.refresh_button.click()
assert refresh_events == [True]
QtTest.QTest.mouseClick(control._view.risk_switch, QtCore.Qt.LeftButton)
assert risk_events == [True]
form = control._view.form
widget = form.widget_for('gain')
assert not widget.isEnabled()
assert control._view._field_rows['gain'][0].text() == '2'

control.set_projection(spec, {
    'owners': ('Camera Measurement',),
    'reason': 'risk accepted for unclaimed fields',
    'risk_enabled': True,
    'risk_accepted': True,
    'fields': {
        'gain': {
            'current': 2, 'desired': 2, 'editable': True,
            'live_apply': False, 'live_enabled': True, 'apply_enabled': True,
            'status': 'Ready', 'severity': 'ready', 'reason': '',
        },
    },
})
assert control._view.form is form
assert form.widget_for('gain') is widget
app.processEvents()
row = form._rows['gain']
current, limits, live, apply, _dot, status = control._view._field_rows['gain']
ordered = (row._label, current, widget, limits, live, apply, status.parentWidget())
for left, right in zip(ordered, ordered[1:]):
    assert left.mapTo(row, left.rect().topRight()).x() <= right.mapTo(row, right.rect().topLeft()).x()
widget.setValue(4); app.processEvents()
assert desired_events == [('gain', 4)]
assert apply_events == []
control._view._field_rows['gain'][3].click()
assert apply_events == [('gain', 4)]

live = control._view._field_rows['gain'][2]
QtTest.QTest.mouseClick(live, QtCore.Qt.LeftButton)
assert live_events[-1] == ('gain', True)
widget.setValue(5); widget.setValue(6)
QtTest.QTest.qWait(30)
control.set_projection(spec, {
    'owners': ('Camera Measurement',),
    'reason': 'unchanged policy projection',
    'risk_enabled': True,
    'risk_accepted': True,
    'fields': {
        'gain': {
            'current': 2, 'desired': 6, 'editable': True,
            'live_apply': True, 'live_enabled': True, 'apply_enabled': True,
            'status': 'Ready', 'severity': 'ready', 'reason': '',
        },
    },
})
QtTest.QTest.qWait(110); app.processEvents()
assert apply_events[-1] == ('gain', 6)
assert apply_events.count(('gain', 5)) == 0
assert control._view._field_rows['gain'][0].text() == '2'

control.set_projection(spec, {
    'owners': ('Camera Measurement',),
    'reason': 'exposure working point is claimed',
    'risk_enabled': True,
    'risk_accepted': True,
    'fields': {
        'gain': {
            'current': 6, 'desired': 6, 'editable': False,
            'live_apply': False, 'live_enabled': False, 'apply_enabled': False,
            'status': 'Protected', 'severity': 'warning',
            'reason': 'Camera Measurement claims this field',
        },
    },
})
assert control._view._field_rows['gain'][0].text() == '6'
assert control._view._field_rows['gain'][5].text() == 'Protected'
assert not widget.isEnabled()
assert widget.toolTip() == 'Camera Measurement claims this field'
assert form.read_value('gain') == 6
control.show_status('ready', 'idle')
assert control._view.status_strip.text() == 'ready'

window = control._window
window.hide(); app.processEvents()
assert not control.is_visible()
control.restore(); app.processEvents()
assert control._window is window
assert control.is_visible()
closed = []
control.closed.connect(lambda: closed.append(True))
control.close(); app.processEvents()
assert closed == [True]
assert not control.is_visible()
manager.close(); app.processEvents()
"""
    )


def test_a_projected_type_is_not_a_pick() -> None:
    """``type_picked`` is the operator choosing; ``set_devices`` projecting
    the host's own record selected the type with signals live and told the
    host the operator had just asked for it."""

    _run_qt(
        """import zlc_ui.device_manager.view as tested_module
print(zou_lab_control.__file__)
print(tested_module.__file__)
from zlc_ui.device_manager import DeviceManagerView
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['device-projection'])
view = DeviceManagerView()
view.set_device_choices((('First', 'sensor.first', 'sensor'), ('Second', 'sensor.second', 'sensor')))
picks = []
view.type_picked.connect(lambda key, value: picks.append((key, value)))
view.set_devices((('sensor', 'Sensor', 'sensor.second', 'sensor'),))
view.set_devices((('sensor', 'Sensor', 'sensor.first', 'sensor'),))
assert picks == [], picks
card = view._cards['sensor']
assert card.type_combo.currentData() == 'sensor.first'
card.type_combo.setCurrentIndex(1)
assert picks == [('sensor', 'sensor.second')], 'a real pick still reaches the host'
"""
    )


def test_a_closed_log_window_stops_polling_and_is_forgotten() -> None:
    """A log window that was closed kept reading the snapshot every 500 ms
    for the life of the manager; the clock follows the widget's own
    visibility, and the closed window is retired rather than kept."""

    _run_qt(
        """import zlc_ui.device_manager.view as tested_module
print(tested_module.__file__)
from PyQt5 import QtCore, QtTest, sip
from zlc_ui.device_manager import DeviceManagerView
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['device-log'])
view = DeviceManagerView()
calls = []

def snapshot():
    calls.append(True)
    return len(calls), ('line',)

view.open_device_log('sensor', snapshot, label='Science sensor')
window = view._device_log_windows['sensor']
assert window.windowTitle() == 'Science sensor log@Zou lab'
body = window.loaded
assert body._timer.isActive(), 'shown, it polls'
window.close()
assert not body._timer.isActive(), 'hidden, it stops'
assert 'sensor' not in view._device_log_windows, 'a closed window is over'
before = len(calls)
QtTest.QTest.qWait(700)
assert len(calls) == before, 'a closed log asks for nothing'
assert sip.isdeleted(window), 'and it is retired, not kept'
view.open_device_log('sensor', snapshot, label='Science sensor')
again = view._device_log_windows['sensor']
assert again is not window and again.loaded._timer.isActive()
again.close(); app.processEvents()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
"""
    )


def test_the_control_shows_the_devices_own_limits_beside_the_window() -> None:
    """An operator setting a bench window has to see which fence bites: the
    instrument's own range is shown, read-only, in its own column beside
    the editable window, in the spelling the row is read in; a field whose
    device states none shows nothing there."""

    _run_qt(
        """import zlc_ui.device_manager.view as tested_module
print(tested_module.__file__)
from zlc_ui.qt import ensure_qt_app
from zlc_ui.form.form import FormFieldProps, FormSpec
from zlc_ui.device_manager.view import DeviceControlView
from dataclasses import replace
from zlc_data.units import DEFAULT_UNITS
from PyQt5 import QtTest
app = ensure_qt_app(['device-limits'])
spec = FormSpec((
    FormFieldProps(key='power', kind='float', label='Power', unit='dBm',
                   minimum=-20.0, maximum=10.0),
    FormFieldProps(key='output', kind='bool', label='Output', default=False),
    FormFieldProps(key='frequency', kind='float', label='Frequency', unit='Hz',
                   default=1000.0, minimum=1e-6, maximum=160e6),
))
def state(current, limits):
    return {'current': current, 'desired': current, 'editable': True,
            'live_apply': False, 'live_enabled': True, 'apply_enabled': False,
            'status': '', 'severity': 'info', 'reason': '', 'device_limits': limits}
view = DeviceControlView(spec, {
    'fields': {'power': state(-3.0, (-120.0, 30.0)), 'output': state(False, None),
               'frequency': state(1000.0, (1e-6, 160e6))},
    'owners': (), 'reason': '', 'risk_accepted': False, 'risk_enabled': False,
})
limits = {key: row[1] for key, row in view._field_rows.items()}
assert limits['power'].text() == '-120 dBm to 30 dBm', limits['power'].text()
assert limits['output'].text() == '', 'a device that states no range shows none'
assert view.form.widget_for('power') is not limits['power'], 'a value, not a control'
def project_unit(key, unit):
    # Stand in for the owner's complete read-only response, not a tune.
    global spec, limits
    old = next(field for field in spec.fields if field.key == key)
    converted = lambda value: None if value is None else float(DEFAULT_UNITS.convert(value, old.unit, unit))
    fields = {name: dict(value) for name, value in view._field_states.items()}
    fields[key].update(current=converted(fields[key]['current']),
        desired=converted(fields[key]['desired']), desired_unit=unit,
        device_limits=tuple(converted(value) for value in fields[key]['device_limits']))
    spec = FormSpec(tuple(replace(field, unit=unit, minimum=converted(field.minimum),
        maximum=converted(field.maximum)) if field.key == key else field for field in spec.fields))
    view.set_projection(spec, {'fields': fields, 'owners': (), 'reason': '',
                              'risk_accepted': False, 'risk_enabled': False})
    limits = {name: row[1] for name, row in view._field_rows.items()}
view.field_unit_requested.connect(project_unit)
view.form._shown_unit_picked('power', 'mW')
assert limits['power'].text() == '0.000000000001 mW to 1000 mW', limits['power'].text()
view.form._shown_unit_picked('power', 'dBm')
assert limits['power'].text() == '-120 dBm to 30 dBm'
view.set_projection(spec, {
    'fields': {'power': state(-3.0, (-110.0, 20.0)), 'output': state(False, None),
               'frequency': state(1000.0, (1e-6, 160e6))},
    'owners': (), 'reason': '', 'risk_accepted': False, 'risk_enabled': False,
})
assert limits['power'].text() == '-110 dBm to 20 dBm'
# The unit gesture changes only the authored pair, even while Live is on.
applies = []
fields = {'power': state(-3.0, (-110.0, 20.0)), 'output': state(False, None),
          'frequency': state(1000.0, (1e-6, 160e6))}
fields['power']['live_apply'] = True
projection = {'fields': fields, 'owners': (), 'reason': '',
              'risk_accepted': False, 'risk_enabled': False}
def desired(key, value, unit):
    current = {name: dict(state) for name, state in view._field_states.items()}
    current[key].update(desired=value, desired_unit=unit, apply_enabled=True)
    view.set_projection(spec, {**projection, 'fields': current})
view.field_desired_changed.connect(desired)
view.field_apply_requested.connect(lambda key, value, unit: applies.append((key, value, unit)))
view.set_projection(spec, projection)
view.form._shown_unit_picked('power', 'mVpp')
QtTest.QTest.qWait(110)
assert applies == [], 'choosing a display unit is not a hardware command'
assert not view._live_timers['power'].isActive()
assert view.form.widget_for('power').valueUnit() == 'mVpp'
view._field_rows['power'][2].setChecked(False)
view.form.widget_for('power').setText('135')
view._field_rows['power'][3].click()
assert applies == [('power', 135.0, 'mVpp')], applies
headings = [view.field_heading, view.current_heading, view.desired_heading,
            view.limits_heading, view.live_heading, view.apply_heading, view.status_heading]
from PyQt5.QtCore import QPoint
for width in (view.sizeHint().width(), view.sizeHint().width() + 200):
    view.resize(width, 300); view.show()
    for _ in range(3):
        app.processEvents()
    view._align_headings(); app.processEvents()
    for row in view.form._rows.values():
        cells = [row.layout().itemAt(i).widget() for i in range(row.layout().count())]
        assert len(cells) == len(headings), (len(cells), len(headings))
        for heading, cell in zip(headings, cells):
            assert (heading.mapTo(view, QPoint()).x(), heading.width()) == (cell.mapTo(view, QPoint()).x(), cell.width())
    pickers = [(picker.mapTo(view, QPoint()).x(), picker.width()) for picker in view.form._unit_pickers.values()]
    assert len(set(pickers)) == 1, pickers
assert view.form._rows['power'].layout().itemAt(3).widget() is limits['power']
view.close()
"""
    )
