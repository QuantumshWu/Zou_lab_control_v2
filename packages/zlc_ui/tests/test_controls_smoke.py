from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

from zlc_ui.board import BoardMetrics, nearest_anchor, pack


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO_ROOT = ROOT.parents[1]


def _run_qt_smoke(code: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        ""
        if environment.get("ZLC_TEST_INSTALLED") == "1"
        else os.pathsep.join((str(REPO_ROOT), str(SRC)))
    )
    environment["QT_QPA_PLATFORM"] = "offscreen"
    verified = """
import zou_lab_control
import zlc_ui
print(zou_lab_control.__file__)
print(zlc_ui.__file__)
""" + code
    completed = subprocess.run(
        [sys.executable, "-c", verified],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_fluent_controls_and_shared_scale() -> None:
    _run_qt_smoke(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentButton, FluentLabel, scaled_px, set_fluent_scale, window_pad
ensure_qt_app(['zlc-ui-tests'])
set_fluent_scale(1.0)
assert window_pad() > 0
assert scaled_px(10) == 10
assert isinstance(FluentButton('Run'), QtWidgets.QPushButton)
assert isinstance(FluentLabel('status'), QtWidgets.QLabel)
"""
    )


def test_point_review_is_one_fluent_view_and_dialog() -> None:
    _run_qt_smoke(
        """
import zou_lab_control
print(zou_lab_control.__file__)
from PyQt5 import QtCore, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import PointReviewView
from zlc_ui.fluent import (
    FluentButton, FluentCheckBox, FluentDialogWindow, FluentFrame,
    FluentLineEdit, FluentScrollArea, FluentWindow,
)
app = ensure_qt_app(['zlc-ui-tests'])
parent = FluentWindow(
    widget=QtWidgets.QWidget(), title='TaskConsole parent probe'
)
parent.show()
surface = QtWidgets.QWidget()
view = PointReviewView(
    surface,
    (('site-a', '1', 2.0, 3.0), ('site-b', '2', 7.0, 5.0)),
    message='Exclude unwanted sites.',
    confirm_label='Continue calibration',
)
assert isinstance(view, FluentFrame)
assert isinstance(view.search, FluentLineEdit)
assert isinstance(view.point_scroll, FluentScrollArea)
assert len(view.findChildren(FluentCheckBox)) == 2
assert not view.findChildren(QtWidgets.QListWidget)
for button in (
    view.exclude_selected_button, view.restore_selected_button,
    view.reset_button, view.stop_button, view.confirm_button,
):
    assert isinstance(button, FluentButton), type(button)
view.select_points(('site-b',))
view.exclude_selected_button.click()
assert view.excluded_ids == ('site-b',)
def verify_real_dialog_and_accept():
    dialog = view.window()
    assert isinstance(dialog, FluentDialogWindow), type(dialog)
    assert dialog is not parent
    assert dialog.isWindow()
    assert QtWidgets.QApplication.activeModalWidget() is dialog
    view.confirm_button.click()
QtCore.QTimer.singleShot(0, verify_real_dialog_and_accept)
assert view.exec_(parent, title='Review detected sites') == FluentDialogWindow.Accepted
parent.close()
"""
    )


def test_plain_choice_picker_and_legend() -> None:
    _run_qt_smoke(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentComboBox, PublishedItemsLegend, fill_grouped_choice_combo, read_editable_combo
ensure_qt_app(['zlc-ui-tests'])
combo = FluentComboBox()
fill_grouped_choice_combo(combo, names=('temperature', 'count'), sources={'temperature': ('sensor',), 'count': ()}, metadata={'temperature': 'float', 'count': 'int'}, current='temperature', none_label='(none)', labels={'temperature': 'Temperature'}, state_labels=('ready', 'waiting', 'unassigned'), empty_source_label='(no source)')
assert read_editable_combo(combo) == 'temperature'
assert combo.count() >= 3
legend = PublishedItemsLegend()
legend.set_rows((('temperature', '1', 'ambient temperature'),))
assert 'temperature' in legend.text()
assert 'ambient temperature' in legend.toolTip()
"""
    )


def test_fluent_combo_popup_is_lazy_single_owned_and_keyboard_selectable() -> None:
    _run_qt_smoke(
        """
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentComboBox, FluentTreeComboBox, scaled_px
app = ensure_qt_app(['combo-lifecycle'])
flat = FluentComboBox()
flat.addItems((
    'short', 'a considerably longer choice',
    'Measurement: Camera Measurement',
    *(f'choice {index}' for index in range(9)),
))
tree = FluentTreeComboBox()
tree.set_choice_tree((
    ('camera', (('frames', '@logic/camera/frames', 'camera · frames'),)),
), current='@logic/camera/frames')
assert not flat.findChildren(QtWidgets.QAbstractItemView)
assert not tree.findChildren(QtWidgets.QAbstractItemView)

initial_width = flat.sizeHint().width()
flat.addItem('new widest choice of the complete model')
assert flat.sizeHint().width() > initial_width
font = flat.font(); font.setPointSize(font.pointSize() + 2); flat.setFont(font)
assert flat.sizeHint().width() > initial_width
flat.setFixedWidth(scaled_px(170, minimum=130))

flat.show(); tree.show(); app.processEvents()
assert not flat.findChildren(QtWidgets.QAbstractItemView)
assert not tree.findChildren(QtWidgets.QAbstractItemView)
flat.showPopup(); app.processEvents()
flat_view = flat.view()
assert type(flat_view).__name__ == 'QListView'
assert flat_view.verticalScrollBar().maximum() > 0
assert flat_view.sizeHintForColumn(0) <= flat_view.viewport().width()
assert flat_view.horizontalScrollBar().maximum() == 0
flat_events = []
flat.activated.connect(flat_events.append)
flat_view.setCurrentIndex(flat.model().index(1, 0))
QtTest.QTest.keyClick(flat_view, QtCore.Qt.Key_Return)
assert flat.currentIndex() == 1 and flat_events == [1]
tree.showPopup(); app.processEvents()
tree_view = tree.view()
assert type(tree_view).__name__ == '_ExpandableTreeView'
assert not tree.findChildren(QtWidgets.QListView)
assert flat_view.styleSheet() == tree_view.styleSheet() != ''
assert tree.current_choice_key() == '@logic/camera/frames'
tree_index = tree.model().index(0, 0, tree.model().index(0, 0))
tree_view.setExpanded(tree_index.parent(), True)
app.processEvents()
assert tree_view.sizeHintForColumn(0) <= tree_view.viewport().width()
assert tree_view.horizontalScrollBar().maximum() == 0
tree_view.setCurrentIndex(tree_index)
QtTest.QTest.keyClick(tree_view, QtCore.Qt.Key_Return)
assert tree.current_choice_key() == '@logic/camera/frames'
flat.showPopup(); app.processEvents(); flat.hidePopup()
tree.showPopup(); app.processEvents()
assert flat.view() is flat_view and tree.view() is tree_view

# A model change while the popup is open re-measures the real delegate and
# viewport rather than leaving a stale width owner behind.
before = tree._popup.width()
tree.model().item(0).appendRow(QtGui.QStandardItem(
    'a newly inserted and significantly wider tree choice'))
app.processEvents()
assert tree._popup.width() >= before
assert tree_view.sizeHintForColumn(0) <= tree_view.viewport().width()
assert tree_view.horizontalScrollBar().maximum() == 0
expanded_width = tree._popup.width()
tree_view.setExpanded(tree.model().index(0, 0), False); app.processEvents()
assert tree._popup.width() <= expanded_width
assert tree_view.horizontalScrollBar().maximum() == 0
tree_view.setExpanded(tree.model().index(0, 0), True); app.processEvents()
assert tree_view.sizeHintForColumn(0) <= tree_view.viewport().width()

# Exactly the shared visible-row limit needs neither scrollbar.
twelve = FluentComboBox(); twelve.addItems(tuple(f'row {i}' for i in range(12)))
twelve.setFixedWidth(scaled_px(170, minimum=130))
twelve.show(); app.processEvents(); twelve.showPopup(); app.processEvents()
assert twelve.view().verticalScrollBar().maximum() == 0
assert twelve.view().horizontalScrollBar().maximum() == 0

# Horizontal scrolling remains valid at the physical screen boundary: the
# popup grows to the available width, then and only then exposes overflow.
capped = FluentComboBox(); capped.addItem('x' * 5000)
capped.show(); app.processEvents(); capped.showPopup(); app.processEvents()
assert capped._popup.width() == capped.screen().availableGeometry().width()
assert capped.view().sizeHintForColumn(0) > capped.view().viewport().width()
assert capped.view().horizontalScrollBar().maximum() > 0
assert capped.view().horizontalScrollBar().isVisible()

flat.hidePopup(); tree.hidePopup(); twelve.hidePopup(); capped.hidePopup()
flat.close(); tree.close(); twelve.close(); capped.close()
"""
    )


def test_form_runtime_context_and_qt_projection() -> None:
    _run_qt_smoke(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.form import FormFieldProps, FormRuntimeContext, FormSpec, FluentParameterForm
ensure_qt_app(['zlc-ui-tests'])
spec = FormSpec((FormFieldProps('name', 'text', 'Name', default='demo', required=True), FormFieldProps('count', 'int', 'Count', default=2, minimum=0)))
form = FluentParameterForm(spec, spec.default_values(), runtime=FormRuntimeContext(choice_names=lambda key: (key,)))
assert form.read_all() == {'name': 'demo', 'count': 2}
form.populate({'name': 'changed', 'count': 4})
assert form.read_all() == {'name': 'changed', 'count': 4}
"""
    )


def test_board_geometry_values() -> None:
    metrics = BoardMetrics(4)
    cards = [
        SimpleNamespace(width=40, height=30, col=0, row=0),
        SimpleNamespace(width=40, height=30, col=0, row=0),
    ]
    assert pack(cards, metrics, board_w=100)
    assert cards[0].col <= cards[1].col
    assert nearest_anchor(cards[0], cards[1:], metrics, board_w=100) == (4, 4)


def test_nearest_anchor_uses_probe_position_without_mutating_layout_record() -> None:
    metrics = BoardMetrics(8)
    others = [
        SimpleNamespace(width=500, height=275, col=0, row=0) for _ in range(3)
    ]
    pack(others, metrics, board_w=1200)
    probe = SimpleNamespace(width=500, height=275, col=600, row=0)
    assert nearest_anchor(probe, others, metrics, 1200) == (516, 8)
    assert (probe.col, probe.row) == (600, 0)

def test_drop_chooses_the_nearest_two_dimensional_gravity_anchor() -> None:
    metrics = BoardMetrics(10)
    others = [
        SimpleNamespace(width=100, height=80, col=0, row=0) for _ in range(2)
    ]
    pack(others, metrics, board_w=350)
    probe = SimpleNamespace(width=100, height=80, col=12, row=96)
    assert nearest_anchor(probe, others, metrics, board_w=350) == (10, 100)
    probe.col, probe.row = 12, 14
    assert nearest_anchor(probe, others, metrics, board_w=350) == (10, 10)


def test_figure_info_construct() -> None:
    _run_qt_smoke(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentReadoutMultiline, InfoPane
app = ensure_qt_app(['zlc-ui-tests'])
long_value = 'C:/' + 'very-long-segment/' * 12 + 'figure.npz'
explicit = 'first\\nsecond\\nthird'
pane = InfoPane(
    label_names=('Name', 'Description'),
    tabs=(('Summary', (
        ('Short', 'demo'), ('Long', long_value), ('Explicit', explicit),
    )),),
)
pane.resize(560, 600); pane.show(); app.processEvents()
fields = [
    pane._tab_layouts['Summary'].itemAt(index).widget().findChild(
        FluentReadoutMultiline
    )
    for index in range(3)
]
assert [field.toPlainText() for field in fields] == ['demo', long_value, explicit]
assert fields[1].height() > fields[0].height()
assert fields[2].height() > fields[0].height()
for field in fields:
    assert field.horizontalScrollBar().maximum() == 0
    assert field.verticalScrollBar().maximum() == 0
"""
    )
