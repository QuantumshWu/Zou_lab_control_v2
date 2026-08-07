from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run_qt(code: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
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
        """
from zlc_ui.device_manager import DeviceManagerView
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['test'])
view = DeviceManagerView()
view.set_device_choices((('Sensor', 'sensor'), ('Camera', 'camera')))
view.set_devices((('id-1', 'input', 'sensor'),))
spec = FormSpec((FormFieldProps('count', 'int', 'Count', default=2, minimum=0, maximum=99),))
view.set_form_spec('id-1', spec, (('count', 7),))
assert view.read_values('id-1') == (('count', 7),)
view.show_status('ready', 'idle')
assert view.status_strip.text() == 'ready'
assert view.status_strip.current_severity == 'idle'
assert view.add_type_combo.count() == 2
assert view._cards['id-1'].role_edit.text() == 'input'
"""
    )


def test_device_manager_qtest_signal_payloads() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.device_manager import DeviceManagerView
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['test'])
view = DeviceManagerView()
view.set_device_choices((('Sensor', 'sensor'), ('Camera', 'camera')))
view.set_devices((('id-1', 'input', 'sensor'),))
view.set_form_spec('id-1', FormSpec((FormFieldProps('count', 'int', 'Count', default=2, minimum=0, maximum=99),)), (('count', 2),))
view.show(); app.processEvents()
events = []
view.device_add_requested.connect(lambda value: events.append(('add', value)))
view.device_remove_requested.connect(lambda value: events.append(('remove', value)))
view.role_committed.connect(lambda instance_id, value: events.append(('role', instance_id, value)))
view.type_picked.connect(lambda instance_id, value: events.append(('type', instance_id, value)))
view.parameter_committed.connect(lambda instance_id, key: events.append(('parameter', instance_id, key)))
card = view._cards['id-1']
QtTest.QTest.mouseClick(view.add_button, QtCore.Qt.LeftButton)
card.role_edit.setFocus()
card.role_edit.selectAll()
QtTest.QTest.keyClicks(card.role_edit, 'output')
QtTest.QTest.keyClick(card.role_edit, QtCore.Qt.Key_Return)
card.type_combo.setCurrentIndex(1)
card.form.widget_for('count').setValue(3)
QtTest.QTest.mouseClick(card.remove_button, QtCore.Qt.LeftButton)
assert ('add', 'sensor') in events
assert ('role', 'id-1', 'output') in events
assert ('type', 'id-1', 'camera') in events
assert ('parameter', 'id-1', 'count') in events
assert ('remove', 'id-1') in events
"""
    )


def test_device_manager_demo_is_a_reusable_human_entry() -> None:
    _run_qt(
        """
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
