from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO = ROOT.parents[1]


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
        [sys.executable, "-c", code],
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
target = screen_fit_window_size(WINDOW_SCREEN_FRACTION)
assert control._window.size() == target, (
    control._window.size().width(), control._window.size().height(),
    target.width(), target.height(),
)
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
assert control._view.live_heading.text() == 'Live apply'
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
current, live, apply, _dot, status = control._view._field_rows['gain']
ordered = (row._label, current, widget, live, apply, status.parentWidget())
for left, right in zip(ordered, ordered[1:]):
    assert left.mapTo(row, left.rect().topRight()).x() <= right.mapTo(row, right.rect().topLeft()).x()
widget.setValue(4); app.processEvents()
assert desired_events == [('gain', 4)]
assert apply_events == []
control._view._field_rows['gain'][2].click()
assert apply_events == [('gain', 4)]

live = control._view._field_rows['gain'][1]
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
assert control._view._field_rows['gain'][4].text() == 'Protected'
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
