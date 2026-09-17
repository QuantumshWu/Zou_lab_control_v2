from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _environment() -> dict[str, str]:
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    environment["PYTHONPATH"] = (
        ""
        if environment.get("ZLC_TEST_INSTALLED") == "1"
        # The repository root too: an example bootstraps through
        # zou_lab_control so it demonstrates THIS checkout, and the
        # bootstrap lives there.
        else os.pathsep.join((str(ROOT.parents[1]), str(SRC)))
    )
    return environment


def test_gallery_offscreen_smoke_only() -> None:
    environment = _environment()
    completed = subprocess.run(
        [sys.executable, "examples/gallery.py"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_console_demo_imports_as_a_package_and_has_mixed_two_row_cards() -> None:
    environment = _environment()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import zou_lab_control
from examples.demo_console import create_window
from zlc_ui.console import TaskConsoleHandle
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['demo-import'])
console = create_window()
app.processEvents()
assert isinstance(console, TaskConsoleHandle)
# Reaching the view is this package's own business; the demo cannot.
view = console._view
cards = tuple(view.board._cards.values())
assert tuple(card.panel_size for card in cards) == ('1x4', '2x2', '4x2')
assert len({card.geometry().width() for card in cards}) >= 2
assert len({card.geometry().y() for card in cards}) >= 2
console.close()
""",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_gallery_has_named_layers_scan_api_and_all_complete_demo_tabs() -> None:
    environment = _environment()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import zou_lab_control
from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtTest
from zlc_ui.fluent import FluentTabWidget
from zlc_ui.qt import ensure_qt_app
from examples.gallery import create_window
app = ensure_qt_app(['gallery-structure'])
window = create_window()
app.processEvents()
body = window.body
assert body.findChild(QtWidgets.QWidget, 'GalleryHeading1') is not None
assert body.findChild(QtWidgets.QWidget, 'GalleryHeading2') is not None
assert body.findChild(QtWidgets.QWidget, 'GalleryHeading3') is not None
visible_names = {label.text() for label in body.findChildren(QtWidgets.QLabel) if label.text()}
assert 'FluentStatusStrip' in visible_names
assert 'ConsoleBoardView' in visible_names
assert 'FluentScanLineEdit · Duration' in visible_names
assert 'FluentScanLineEdit · Scan · duration' in visible_names
assert 'FluentScanLineEdit · Scan + API · duration' in visible_names
assert 'FluentScanLineEdit · Scan + Config · da_bias_y' in visible_names
assert 'FluentScanLineEdit · Delay' in visible_names

duration = body.binding_examples['duration']
assert duration.source == 'default' and not duration.scan
QtTest.QTest.mouseClick(duration.field.binding_button, QtCore.Qt.LeftButton)
app.processEvents()
assert duration.source == 'default' and duration.field.text() == '0'
duration.field._popup.hide()

dac = body.binding_examples['dac']
assert dac.scan and dac.source == 'config' and dac.field.text() == '0'
QtTest.QTest.mouseClick(dac.field.binding_button, QtCore.Qt.LeftButton)
app.processEvents()
assert dac.scan and dac.field.text() == '0'
dac.field._popup.hide()

delay = body.binding_examples['delay']
QtTest.QTest.mouseClick(delay.field.binding_button, QtCore.Qt.LeftButton)
app.processEvents()
assert delay.source == 'default'
delay.field._popup.hide()
tab_sets = [
    {tab.tabText(index) for index in range(tab.count())}
    for tab in body.findChildren(FluentTabWidget)
]
assert {'TaskConsole', 'PulseEditor', 'FigureViewer', 'DeviceManager'} in tab_sets
window.close()
window.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
app.quit()
""",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
