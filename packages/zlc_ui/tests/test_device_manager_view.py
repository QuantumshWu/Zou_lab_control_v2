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
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(REPO), str(SRC), environment.get("PYTHONPATH", ""))
        if value
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
        """import zou_lab_control_v2
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
view.show_status('ready', 'idle')
assert view.status_strip.text() == 'ready'
assert view.status_strip.current_severity == 'idle'
assert tuple(view.domain_groups) == ('sensor', 'camera')
assert view._cards['id-1'].role_edit.text() == 'input'
"""
    )


def test_device_manager_qtest_signal_payloads() -> None:
    _run_qt(
        """import zou_lab_control_v2
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
        """import zou_lab_control_v2
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


def test_device_manager_keeps_the_v1_config_surface_and_lifecycle_verbs() -> None:
    _run_qt(
        """import zou_lab_control_v2
import zlc_ui.device_manager.view as tested_module
print(tested_module.__file__)
from zlc_ui.device_manager import DeviceManagerView
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['device-manager-v1-surface'])
view = DeviceManagerView()
view.set_device_choices((
    ('Virtual camera', 'camera.virtual', 'camera'),
    ('Hardware camera', 'camera.dcam', 'camera'),
    ('Virtual sequencer', 'sequencer.virtual', 'sequencer'),
))
view.set_templates((('Virtual', 'virtual'), ('Hardware', 'hardware')))
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
"""
    )
