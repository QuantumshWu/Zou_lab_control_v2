from __future__ import annotations


def test_console_demo_imports_as_a_package_and_has_mixed_two_row_cards(run_qt) -> None:
    run_qt(
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
"""
    )


def test_gallery_has_named_layers_scan_api_and_all_complete_demo_tabs(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtTest
from zlc_ui.fluent import FluentTabWidget
from zlc_ui.qt import ensure_qt_app
from examples.gallery import create_window
app = ensure_qt_app(['gallery-structure'])
window = create_window()
app.processEvents()
body = window.loaded
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

# Each binding button opens the production popup over its field.
for key in ('duration', 'dac', 'delay'):
    example = body.binding_examples[key]
    QtTest.QTest.mouseClick(example.field.binding_button, QtCore.Qt.LeftButton)
    app.processEvents()
    assert example.field._popup is not None and example.field._popup.isVisible(), key
    example.field._popup.hide()
assert body.binding_examples['duration'].field.text() == '0'
assert body.binding_examples['dac'].field.text() == '0'
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
    )
