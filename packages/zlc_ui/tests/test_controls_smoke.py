from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

from zlc_ui.board import BoardMetrics, nearest_anchor, pack


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run_qt_smoke(code: str) -> None:
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
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import InfoPane
app = ensure_qt_app(['zlc-ui-tests'])
pane = InfoPane(
    label_names=('Name', 'Description'),
    tabs=(('Summary', (('Name', 'demo'),)), ('Raw', (('Payload', '{}'),))),
)
assert pane.info_tabs.count() == 2
assert pane._tab_layouts['Raw'].itemAt(0).widget().findChild(QtWidgets.QPlainTextEdit).toPlainText() == '{}'
"""
    )
