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
