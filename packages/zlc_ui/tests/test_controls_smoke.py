from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

from zlc_ui.board import BoardMetrics, drop_index, pack
from zlc_ui.graph import FlowGraphNode, describe_shape_text, flow_graph_from_tree


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


def test_board_graph_and_shape_values() -> None:
    metrics = BoardMetrics(4, lambda size: (40, 30) if size == "small" else (60, 30))
    cards = [
        SimpleNamespace(size="small", col=0, row=0),
        SimpleNamespace(size="small", col=0, row=0),
    ]
    assert pack(cards, metrics, board_w=100)
    assert cards[0].col <= cards[1].col
    assert drop_index(cards[0], cards[1:], metrics, board_w=100) in {0, 1}


def test_drop_index_uses_probe_position_without_mutating_layout_record() -> None:
    metrics = BoardMetrics(8, lambda size: (500, 275))
    others = [SimpleNamespace(size="small", col=0, row=0) for _ in range(3)]
    pack(others, metrics, board_w=1200)
    probe = SimpleNamespace(size="small", col=600, row=0)
    assert drop_index(probe, others, metrics, 1200) == 1
    assert (probe.col, probe.row) == (600, 0)

    graph = flow_graph_from_tree(
        {
            "nodes": (
                {"id": "source", "name": "Source", "role": "producer"},
                {"id": "view", "name": "View", "role": "consumer"},
            ),
            "edges": ({"from": "source", "to": "view", "signal": "temperature"},),
        }
    )
    assert isinstance(graph.nodes[0], FlowGraphNode)
    assert describe_shape_text("1 × 2") == "1 × 2"


def test_gravity_drop_index_accepts_many_north_west_slots() -> None:
    metrics = BoardMetrics(10, lambda size: (100, 80))
    others = [SimpleNamespace(size="small", col=0, row=0) for _ in range(5)]
    pack(others, metrics, board_w=350)
    # The v1 rule is not a two-panel swap: the raw release point is compared
    # with every trial order produced by the same north-west packer.
    indices = {
        drop_index(
            SimpleNamespace(size="small", col=point[0], row=point[1]),
            others,
            metrics,
            board_w=350,
        )
        for point in ((10, 10), (120, 10), (230, 10), (10, 100), (120, 100))
    }
    assert indices == {0, 1, 2, 3, 4}


def test_figure_info_and_owner_wake_construct() -> None:
    _run_qt_smoke(
        """
import time
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.concurrency import QtOwnerWake
from zlc_ui.fluent import InfoPane
app = ensure_qt_app(['zlc-ui-tests'])
pane = InfoPane(
    label_names=('Name', 'Description'),
    tabs=(('Summary', (('Name', 'demo'),)), ('Raw', (('Payload', '{}'),))),
)
assert pane.info_tabs.count() == 2
assert pane._tab_layouts['Raw'].itemAt(0).widget().findChild(QtWidgets.QPlainTextEdit).toPlainText() == '{}'
wake = QtOwnerWake()
called = []
wake.bind(lambda: called.append(True))
wake.request_owner_wake()
deadline = time.monotonic() + 1.0
while not called and time.monotonic() < deadline:
    app.processEvents()
assert called == [True]
"""
    )


def test_flow_graph_view_paints_edges_and_arrow_heads() -> None:
    # The gallery grabs the top of its scroll area, so FlowGraphView's edge
    # painting never ran there; a widget-level grab exercises it directly.
    _run_qt_smoke(
        """
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['zlc-ui-tests'])
from zlc_ui.graph import FlowGraph, FlowGraphEdge, FlowGraphNode, FlowGraphView
view = FlowGraphView()
view.set_role_styles({'producer': ('#123456', '#654321')}, compact_roles=frozenset({'view'}))
assert view._role_style('producer') == ('#123456', '#654321')
assert view._is_compact(FlowGraphNode('panel', 'Fake panel', 'view'))
view.set_graph(
    FlowGraph(
        nodes=(
            FlowGraphNode('source', 'Fake source', 'producer'),
            FlowGraphNode('transform', 'Fake transform', 'processor'),
            FlowGraphNode('panel', 'Fake panel', 'view'),
        ),
        edges=(
            FlowGraphEdge('source', 'transform', 'temperature', (1,)),
            FlowGraphEdge('transform', 'panel', 'smoothed', (1,)),
        ),
    )
)
view.resize(900, 500)
view.show()
app.processEvents()
arrows = []
original = FlowGraphView._draw_arrow_head
FlowGraphView._draw_arrow_head = staticmethod(
    lambda painter, p1, p2: (arrows.append((p1, p2)), original(painter, p1, p2))
)
pixmap = view.grab()
assert not pixmap.isNull()
assert len(arrows) >= 2, f'expected >=2 painted arrow heads, got {len(arrows)}'
"""
    )
