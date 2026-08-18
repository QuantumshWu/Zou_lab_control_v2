from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_gallery_offscreen_smoke_only() -> None:
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONPATH"] = str(SRC)
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
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
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
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONPATH"] = str(SRC)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
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
assert 'FluentScanLineEdit · Duration intent' in visible_names
assert 'FluentScanLineEdit · Scan slot 1 · duration' in visible_names
assert 'FluentScanLineEdit · API slot 1 · duration' in visible_names
assert 'FluentScanLineEdit · DAC slot 2 · da_bias_y' in visible_names
assert 'FluentScanLineEdit · Delay intent · off' in visible_names

duration = body.binding_examples['duration_cycle']
assert duration.binding is None
QtTest.QTest.mouseClick(duration.field.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert duration.binding is None and duration.field.text() == '0'

dac = body.binding_examples['dac_cycle']
assert dac.binding == 'scan' and dac.field.text() == 's1'
QtTest.QTest.mouseClick(dac.field.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert dac.binding == 'scan' and dac.field.text() == 's1'

delay = body.binding_examples['delay_cycle']
QtTest.QTest.mouseClick(delay.field.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert delay.binding is None
tab_sets = [
    {tab.tabText(index) for index in range(tab.count())}
    for tab in body.findChildren(FluentTabWidget)
]
assert {'TaskConsole', 'PulseEditor', 'FigureViewer', 'DeviceManager'} in tab_sets
window.close()
app.processEvents()
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


def test_usage_notebook_has_one_direct_cell_per_complete_gui() -> None:
    notebook = json.loads((ROOT / "notebooks" / "usage.ipynb").read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", ())) for cell in notebook["cells"]}
    expected = {
        "gallery": "create_gallery_window()",
        "console-demo": "create_console_window()",
        "pulse-demo": "create_pulse_window()",
        "figure-demo": "create_figure_window()",
        "device-demo": "create_device_window()",
    }
    for cell_id, call in expected.items():
        assert cell_id in cells, cell_id
        assert call in cells[cell_id], (cell_id, call)
        assert "close_demo_windows()" in cells[cell_id], cell_id

    assert "capture_window(" in cells["capture"]
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
