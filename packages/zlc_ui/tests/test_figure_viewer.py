from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO_ROOT = ROOT.parents[1]


def _run_qt(code: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        ""
        if environment.get("ZLC_TEST_INSTALLED") == "1"
        else os.pathsep.join((str(REPO_ROOT), str(SRC)))
    )
    environment["QT_QPA_PLATFORM"] = "offscreen"
    verified = """
import zou_lab_control
import zlc_ui.figure_viewer.view as tested_module
print(zou_lab_control.__file__)
print(tested_module.__file__)
""" + code
    completed = subprocess.run(
        [sys.executable, "-c", verified], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_figure_viewer_mount_reconcile_and_open_intent() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from pathlib import Path
from types import SimpleNamespace
from zlc_ui.figure_viewer import FigureViewerHandle, FigureViewerView
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(["figure-test"])
viewer_folder = str(Path.cwd())
view = FigureViewerView(path_base_dir=viewer_folder); view.set_archive_info(
    (("Summary", (("Name", "fake"),)), ("Flow", ())),
    {'nodes': (), 'edges': ()},
)
assert view.info_pane.path_edit._filter == "Saved figure archives (*.npz)"
assert view.info_pane.path_edit._base_dir == viewer_folder
view.set_panel_sizes(('2x2',), '2x2')
view.set_panel_intervals((100, 400, 1000), 400)
view.set_grid_cell_kinds(('curve', 'image', 'histogram'))
view.set_panel_kinds((('image', 'Image'), ('curve', 'Curve')))
view.add_panel('panel-1', 'Camera frame')
view.set_panel_signal_choices(
    'panel-1', (('this archive', (('Camera frame', '@figure/1/data'),)),),
    current='@figure/1/data',
)
state = {
    'signal': '@figure/1/data', 'kind': 'image', 'cell_kind': '', 'size': '2x2',
    'interval_ms': 400, 'title': 'Camera frame', 'semantic': {}, 'display': {},
    'fit': {}, 'overlay_signal': '',
}
surface = {
    'semantic': (), 'display': (), 'fit': (),
    'data_structure': (), 'data_scope': (), 'paints_images': True,
}
view.set_panel_projection('panel-1', state, surface)
first = QtWidgets.QLabel("first"); second = QtWidgets.QLabel("second")
view.set_panel_surface('panel-1', first); view.set_panel_surface('panel-1', second)
assert view._cards['panel-1'].surface is second and first.parentWidget() is None
view.resize(1200, 700); view.show(); app.processEvents()
assert view._panel_bar.height() == view._panel_bar.sizeHint().height()
assert view.board._cards['panel-1'] is view._cards['panel-1']
assert view.scroll.isVisible() and not view._placeholder.isVisible()

# Showing which file is open must not re-ask for it to be opened.
committed = []; view.path_committed.connect(committed.append)
view.set_path("D:/data/2026_08_05/run.npz")
assert view.info_pane.path_edit.text() == "D:/data/2026_08_05/run.npz"
assert committed == []

handle = FigureViewerHandle(None, view)
graph = {
    'nodes': (
        {'id': 'device:camera', 'kind': 'device', 'title': 'camera', 'subtitle': 'Device · camera', 'root': False, 'tooltip': 'camera'},
        {'id': 'device:sequencer', 'kind': 'device', 'title': 'sequencer', 'subtitle': 'Device · sequencer', 'root': False, 'tooltip': 'sequencer'},
        {'id': 'device:slm', 'kind': 'device', 'title': 'slm', 'subtitle': 'Device · slm', 'root': False, 'tooltip': 'slm'},
        {'id': 'logic:camera', 'kind': 'logic', 'title': 'camera measurement', 'subtitle': 'frames', 'root': False, 'tooltip': 'camera measurement'},
        {'id': 'logic:left', 'kind': 'logic', 'title': 'left processor', 'subtitle': 'left', 'root': False, 'tooltip': 'left'},
        {'id': 'logic:right', 'kind': 'logic', 'title': 'right processor', 'subtitle': 'right', 'root': False, 'tooltip': 'right'},
        {'id': 'logic:fit', 'kind': 'logic', 'title': 'fit', 'subtitle': 'amplitude', 'root': True, 'tooltip': 'fit'},
    ),
    'edges': (
        {'source': 'device:camera', 'target': 'logic:camera', 'kind': 'device', 'label': 'camera'},
        {'source': 'device:sequencer', 'target': 'logic:camera', 'kind': 'device', 'label': 'sequencer'},
        {'source': 'device:slm', 'target': 'logic:fit', 'kind': 'device', 'label': 'slm'},
        {'source': 'logic:camera', 'target': 'logic:left', 'kind': 'causal', 'label': ''},
        {'source': 'logic:camera', 'target': 'logic:right', 'kind': 'causal', 'label': ''},
        {'source': 'logic:left', 'target': 'logic:fit', 'kind': 'causal', 'label': ''},
        {'source': 'logic:right', 'target': 'logic:fit', 'kind': 'causal', 'label': ''},
    ),
}
handle.set_archive_info(view._info_tabs, graph)
assert view._flow_graph == graph
flow = view.info_pane._graph_tabs['Flow']
assert flow._flow_edge_count == 7
assert set(flow._flow_node_rects) == {
    'device:camera', 'device:sequencer', 'device:slm',
    'logic:camera', 'logic:left', 'logic:right', 'logic:fit'
}
rects = tuple(flow._flow_node_rects.values())
assert all(not left.intersects(right) for i, left in enumerate(rects) for right in rects[i + 1:])
scene = flow.sceneRect()
assert all(scene.contains(rect) for rect in rects)
for source, target, path in flow._flow_edge_paths:
    stroker = QtGui.QPainterPathStroker(); stroker.setWidth(3.0)
    stroke = stroker.createStroke(path)
    crossings = tuple(
        node_id
        for node_id, rect in flow._flow_node_rects.items()
        if node_id not in {source, target}
        and stroke.intersects(rect)
    )
    assert not crossings, (source, target, crossings)

projection = {
    'state': state,
    'parameter_surface': surface,
    'signal_options': (('this archive', (('Camera frame', 'data'),)),),
    'overlay_signal_options': (),
    'frozen_signal': '@figure/1/data',
    'frozen_snapshot': SimpleNamespace(
        ref=SimpleNamespace(revision=SimpleNamespace(value=7)),
        block=SimpleNamespace(values=SimpleNamespace(shape=(1, 16, 1))),
    ),
    'save_directory': viewer_folder,
}
view.open_panel_editor(
    'panel-1', projection, 'Edit · Camera frame'
)
editor = view._editors['panel-1']
assert view.tabs.currentWidget() is editor
assert 'interval_ms' in editor.panel_form.spec.keys
assert not editor.snapshot_group.isHidden()
assert not editor.interaction_group.isHidden()
assert not editor.producer_group.isHidden()
assert not editor.open_producer_button.isEnabled()
assert not editor.save_group.isHidden()
assert 'revision 7' in editor.snapshot_label.text()
assert 'shape (1, 16, 1)' in editor.snapshot_label.text()
refreshes = []; saves = []
handle.panel_snapshot_refresh_requested.connect(refreshes.append)
handle.panel_save_figure_requested.connect(lambda panel, path: saves.append((panel, path)))
editor.snapshot_refresh_requested.emit()
editor.save_figure_requested.emit('D:/data/copied.npz')
assert refreshes == ['panel-1']
assert saves == [('panel-1', 'D:/data/copied.npz')]
view.close(); app.processEvents()
"""
    )


def test_figure_viewer_reuses_task_console_permanent_navigation_tabs() -> None:
    _run_qt(
        """
from zlc_ui.console import TaskConsoleView
from zlc_ui.figure_viewer import FigureViewerView
from zlc_ui.fluent import FluentTabWidget
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['shared-tabs'])
figure = FigureViewerView()
console = TaskConsoleView()
assert type(figure.info_pane.info_tabs) is FluentTabWidget
assert type(console.tabs) is FluentTabWidget
assert type(figure.info_pane.info_tabs) is type(console.tabs)
assert [figure.info_pane.info_tabs.tabText(i) for i in range(figure.info_pane.info_tabs.count())] == [
    'Plot', 'Logic', 'Devices', 'Flow', 'Raw'
]
assert [console.tabs.tabText(i) for i in range(console.tabs.count())] == ['Monitor', 'Logic']
"""
    )


def test_manual_data_editor_is_virtual_and_emits_plain_intents() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest, QtWidgets
from collections.abc import Sequence
import zlc_ui.figure_viewer.view as viewer_module
from zlc_ui.figure_viewer import FigureViewerView
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['manual-data-editor'])

class LazyMatrix:
    def __init__(self, rows, columns):
        self.shape = (rows, columns); self.reads = 0
    def __getitem__(self, index):
        self.reads += 1
        row, column = index
        return row * 1000 + column

class LazyScope(Sequence):
    def __init__(self, size): self.size = size
    def __len__(self): return self.size
    def __getitem__(self, index): return (('scope-value', int(index)), str(index))

values = LazyMatrix(100_000, 1_000)
validity = LazyMatrix(100_000, 1_000)
projection = {
    'dataset': {
        'name': 'manual image', 'dtype': '<f8', 'unit': 'count',
        'dtype_choices': (('<f8', 'Float 64'), ('<u2', 'Unsigned 16')),
        'note': '', 'source': 'New manual Dataset',
    },
    'domain_choices': (
        ('repeat', 'Repeat'), ('point', 'Point'), ('cell_data', 'Cell data'),
    ),
    'axes': (
        {'id': 'repeat', 'domain': 'repeat', 'domain_label': 'Repeat',
         'name': 'repeat', 'size': 2_000_000, 'unit': ''},
        {'id': 'spatial-y', 'domain': 'point', 'domain_label': 'Point',
         'name': 'spatial-y', 'size': 100_000, 'unit': 'pixel'},
        {'id': 'spatial-x', 'domain': 'cell_data', 'domain_label': 'Cell data',
         'name': 'spatial-x', 'size': 1_000, 'unit': 'pixel'},
    ),
    'selected_axis': 'spatial-y',
    'axis_values': {
        'shape': (1, 100_000), 'values': (range(100_000),),
        'row_headers': ('Value',), 'column_headers': range(100_000),
        'editable': True,
    },
    'table': {
        'component': 'values',
        'component_choices': (('values', 'Values'), ('validity', 'Validity')),
        'sigma_enabled': False,
        'blank_help': 'Blank removes the value', 'blank_hint': 'No value',
        'shape': values.shape, 'values': values, 'validity': validity,
        'row_headers': range(100_000), 'column_headers': range(1_000),
        'structure': (
            (('repeat', 2_000_000),),
            (('spatial-y', 100_000),),
            (('spatial-x', 1_000),),
        ),
        'axes': (
            {'axis_id': 'repeat', 'name': 'repeat', 'size': 2_000_000,
             'unit': '', 'mode': 'scope', 'index': 0,
             'scope_choices': LazyScope(2_000_000)},
            {'axis_id': 'spatial-y', 'name': 'spatial-y', 'size': 100_000,
             'unit': 'pixel', 'mode': 'rows', 'index': 0,
             'scope_choices': LazyScope(100_000)},
            {'axis_id': 'spatial-x', 'name': 'spatial-x', 'size': 1_000,
             'unit': 'pixel', 'mode': 'columns', 'index': 0,
             'scope_choices': LazyScope(1_000)},
        ),
    },
    'dirty': True, 'can_apply': True, 'can_save': True,
    'save_suggested': 'manual-image.npz', 'message': '',
}

view = FigureViewerView(); view.set_panel_sizes(('2x2',), '2x2')
intents = []; data_closed = []; panel_closed = []
view.data_editor_intent.connect(lambda editor_id, intent: intents.append((editor_id, intent)))
view.data_editor_closed.connect(data_closed.append)
view.panel_editor_closed.connect(panel_closed.append)
view.open_data_editor('manual-1', projection, 'Data · manual image')
editor = view._data_editors['manual-1']
view.resize(1500, 900); view.show(); app.processEvents()
assert editor.dataset_group.width() == editor.axes_group.width() == editor.data_group.width()
assert not hasattr(editor, 'role_combo')
assert not hasattr(editor, 'coordinate_table')
assert not hasattr(editor, 'axis_up_button')
assert editor.axis_value_model.rowCount() == 1
assert editor.axis_value_model.columnCount() == 100_000
assert editor.axis_value_table.horizontalScrollBar().maximum() > 0
assert editor.axis_value_table.sizeAdjustPolicy() == QtWidgets.QAbstractScrollArea.AdjustIgnored
assert 'spatial-y' in editor.row_axis_label.text()
assert 'spatial-x' in editor.column_axis_label.text()
assert editor.value_model.rowCount() == 100_000
assert editor.value_model.columnCount() == 1_000
assert editor.value_table.verticalScrollBar().maximum() > 0
assert editor.value_table.horizontalScrollBar().maximum() > 0
assert values.reads < 500, values.reads
assert editor.value_table.indexWidget(editor.value_model.index(0, 0)) is None
editor.value_table.setCurrentIndex(editor.value_model.index(10, 10))
editor.value_table.setFocus()
QtTest.QTest.keyClick(editor.value_table, QtCore.Qt.Key_Right)
QtTest.QTest.keyClick(editor.value_table, QtCore.Qt.Key_Down)
assert editor.value_table.currentIndex() == editor.value_model.index(11, 11)

repeat_scope = editor._axis_view_widgets['repeat'][2]
repeat_scope.setCyclePosition(1)
repeat_scope.activated.emit(repeat_scope.currentIndex())
assert intents[-1] == (
    'manual-1',
    {'op': 'set_scope', 'axis_id': 'repeat', 'index': 1},
)

editor.value_model.setData(editor.value_model.index(1, 0), '', QtCore.Qt.EditRole)
assert intents[-1] == (
    'manual-1',
    {'op': 'set_cells', 'component': 'values', 'cells': ((1, 0, ''),)},
)

editor.axis_name_edit.setText('detuning')
editor.axis_size_spin.setValue(3)
editor.axis_unit_edit.setText('MHz')
QtTest.QTest.mouseClick(editor.apply_axis_button, QtCore.Qt.LeftButton)
assert intents[-1] == (
    'manual-1',
    {'op': 'edit_axis', 'axis_id': 'spatial-y', 'name': 'detuning',
     'length': 3, 'unit': 'MHz', 'domain': 'point'},
)
editor.axis_value_model.setData(
    editor.axis_value_model.index(0, 1), '0.5', QtCore.Qt.EditRole
)
assert intents[-1] == (
    'manual-1',
    {'op': 'set_axis_values', 'axis_id': 'spatial-y',
     'cells': ((0, 1, '0.5'),)},
)

QtTest.QTest.mouseClick(editor.add_axis_button, QtCore.Qt.LeftButton)
editor.axis_name_edit.setText('shot')
editor.axis_size_spin.setValue(2)
domain = editor.domain_combo.findData('repeat')
editor.domain_combo.setCurrentIndex(domain)
QtTest.QTest.mouseClick(editor.apply_axis_button, QtCore.Qt.LeftButton)
assert intents[-1] == (
    'manual-1',
    {'op': 'add_axis', 'name': 'shot', 'length': 2, 'unit': '',
     'domain': 'repeat'},
)

# One rectangular paste remains one presenter intent.
editor.value_table.setCurrentIndex(editor.value_model.index(2, 3))
QtWidgets.QApplication.clipboard().setText('1\\t2\\n3\\t4')
editor.value_table.paste_clipboard()
assert intents[-1] == (
    'manual-1',
    {'op': 'set_cells', 'component': 'values',
     'cells': ((2, 3, '1'), (2, 4, '2'), (3, 3, '3'), (3, 4, '4'))},
)

# A same-shape projection updates data without resetting Qt's current index.
current = editor.value_model.index(4, 5)
editor.value_table.setCurrentIndex(current)
editor.update_projection(projection)
assert editor.value_table.currentIndex() == current

# A real editor commit may synchronously re-project the same shape.  Tab must
# still commit and enter the next data cell instead of losing the Qt index.
def echo_projection(editor_id, intent):
    if editor_id == 'manual-1' and intent.get('op') == 'set_cells':
        editor.update_projection(projection)
view.data_editor_intent.connect(echo_projection)
current = editor.value_model.index(6, 7)
editor.value_table.setCurrentIndex(current)
editor.value_table.setFocus()
QtTest.QTest.keyClick(editor.value_table, QtCore.Qt.Key_F2)
app.processEvents()
cell_editor = QtWidgets.QApplication.focusWidget()
assert isinstance(cell_editor, QtWidgets.QLineEdit)
cell_editor.selectAll(); QtTest.QTest.keyClicks(cell_editor, '42')
QtTest.QTest.keyClick(cell_editor, QtCore.Qt.Key_Tab); app.processEvents()
assert editor.value_table.currentIndex() == editor.value_model.index(6, 8)
assert editor.value_table.state() == QtWidgets.QAbstractItemView.EditingState
cell_editor = QtWidgets.QApplication.focusWidget()
QtTest.QTest.keyClick(
    cell_editor, QtCore.Qt.Key_Backtab
)
app.processEvents()
assert editor.value_table.currentIndex() == editor.value_model.index(6, 7)
assert editor.value_table.state() == QtWidgets.QAbstractItemView.EditingState

editor.note_edit.setFocus(); app.processEvents()
editor.note_edit.setText('corrected camera background')
viewer_module.fluent_save_path = lambda *_args, **_kwargs: 'D:/data/manual-edited.npz'
QtTest.QTest.mouseClick(editor.save_button, QtCore.Qt.LeftButton)
assert intents[-1] == (
    'manual-1',
    {'op': 'save_as', 'path': 'D:/data/manual-edited.npz',
     'note': 'corrected camera background'},
)
view.tabs.tab_close_requested.emit(editor)
assert data_closed == ['manual-1'] and panel_closed == []
view.close(); app.processEvents()
"""
    )


def test_figure_viewer_demo_smoke() -> None:
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    command = (
        "import runpy, sys, zou_lab_control, zlc_ui; "
        "print(zou_lab_control.__file__); print(zlc_ui.__file__); "
        "sys.argv=['demo_figure_viewer.py', '--once']; "
        "runpy.run_path('examples/demo_figure_viewer.py', run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    # What the demo shows now is a HOST filling the page through the handle;
    # the File field's own intent is checked against the widget above, which is
    # where a widget may be poked at all.
    assert "filled: 2 signals" in completed.stdout


def test_the_info_pane_width_is_stable_across_loaded_content() -> None:
    """Loading metadata must not move the Viewer split or resize the window."""

    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.figure_viewer import FigureViewerView
app = ensure_qt_app(['pane-width'])
view = FigureViewerView(); view.resize(1200, 700); view.show(); app.processEvents()
pane = view.info_pane
bar = pane.info_tabs.tabBar()

assert bar.count() == 5
assert not pane.info_tabs.cornerWidget().isVisible(), 'the tab bar overflowed its own pane'
assert sum(bar.tabRect(i).width() for i in range(bar.count())) >= bar.natural_width(), (
    'a tab title was elided to fit'
)

long_label = 'required_external_trigger_interval_seconds'
before = pane.width()
pane.set_tabs((('Device', ((long_label, '0.02'),)),))
assert pane.width() == before, 'loaded labels changed the fixed Viewer split'
"""
    )


def test_an_info_refresh_keeps_the_tab_the_operator_is_reading() -> None:
    """A refresh replaces the CONTENT of the tabs, not which one is open.

    Every tab is destroyed and rebuilt on each ``set_tabs`` -- the rows
    change, the five titles never do -- and the rebuilt stack starts at the
    first tab, so anyone reading Devices was thrown back to Plot by a
    refresh they did not ask for.
    """

    _run_qt(
        """
import zou_lab_control
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent.info_pane import InfoPane
app = ensure_qt_app(['info-tab'])

TITLES = ('Plot', 'Logic', 'Devices', 'Flow', 'Raw')
pane = InfoPane(label_names=TITLES)
pane.set_tabs(tuple((title, (('a', '1'),)) for title in TITLES))
app.processEvents()
pane.info_tabs.setCurrentIndex(2)
app.processEvents()
assert pane.info_tabs.tabText(pane.info_tabs.currentIndex()) == 'Devices'

pane.set_tabs(tuple((title, (('a', '9'),)) for title in TITLES))
app.processEvents()
assert pane.info_tabs.tabText(pane.info_tabs.currentIndex()) == 'Devices', (
    pane.info_tabs.tabText(pane.info_tabs.currentIndex()))

# A tab that no longer exists cannot be kept; falling back to the first is
# the only honest answer, and it must not raise.
pane.set_tabs((('Plot', (('a', '9'),)), ('Logic', (('a', '9'),))))
app.processEvents()
assert pane.info_tabs.tabText(pane.info_tabs.currentIndex()) == 'Plot'
"""
    )
