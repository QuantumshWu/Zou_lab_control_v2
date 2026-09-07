from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO_ROOT = ROOT.parents[1]


#: Every snippet starts here.  Without it the subprocess resolves the
#: layers through whatever the editable install points at -- on this
#: machine, sibling checkouts of the same package names -- so the suite
#: silently tested a DIFFERENT zlc_plot than the one beside it.  The
#: product bootstrap is what puts this checkout's layers on the path,
#: and it is the same one every launcher uses.
_BOOTSTRAP = "import zou_lab_control" + chr(10)


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

# The board's panel-widget policy reaches the viewer's cards through the
# handle, and one widget serves one host.  A host asked for its own widget
# presents every render the moment it lands, before the board accepted it.
staged = []
def staging_policy(host):
    widget = QtWidgets.QLabel('staged'); widget.host = host
    staged.append(widget); return widget
class SelfPresentingHost:
    def qt_widget(self):
        raise AssertionError('the viewer asked the host for its own widget')
host = SelfPresentingHost()
staged_handle = FigureViewerHandle(None, view, plot_surface=staging_policy)
staged_handle.show_panel('panel-1', host)
staged_handle.show_panel('panel-1', host)
assert len(staged) == 1 and view._cards['panel-1'].surface is staged[0]
staged_handle.show_panel('panel-1', None)
assert view._cards['panel-1'].surface is None
view.set_panel_surface('panel-1', second)

graph = {
    'nodes': (
        {'id': 'device:camera', 'kind': 'device', 'title': 'camera', 'subtitle': 'Device · camera', 'root': False, 'tooltip': 'camera', 'row': None},
        {'id': 'device:sequencer', 'kind': 'device', 'title': 'sequencer', 'subtitle': 'Device · sequencer', 'root': False, 'tooltip': 'sequencer', 'row': None},
        {'id': 'device:slm', 'kind': 'device', 'title': 'slm', 'subtitle': 'Device · slm', 'root': False, 'tooltip': 'slm', 'row': None},
        {'id': 'logic:camera', 'kind': 'logic', 'title': 'camera measurement', 'subtitle': 'frames', 'root': False, 'tooltip': 'camera measurement', 'row': None},
        {'id': 'logic:left', 'kind': 'logic', 'title': 'left processor', 'subtitle': 'left', 'root': False, 'tooltip': 'left', 'row': None},
        {'id': 'logic:right', 'kind': 'logic', 'title': 'right processor', 'subtitle': 'right', 'root': False, 'tooltip': 'right', 'row': None},
        {'id': 'logic:fit', 'kind': 'logic', 'title': 'fit', 'subtitle': 'amplitude', 'root': True, 'tooltip': 'fit', 'row': None},
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


def test_a_played_pulse_gets_a_read_only_tab_beside_board_and_edit() -> None:
    """A Devices row's action opens one preview tab per played pulse -- the
    editor's preview page, controls and all, each control speaking with the
    tab's key; a second open focuses it, and closing the tab tells the
    presenter which one went."""

    _run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.figure_viewer import FigureViewerHandle, FigureViewerView
from zlc_ui.fluent import FluentButton
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['pulse-tab'])
view = FigureViewerView()
handle = FigureViewerHandle(None, view)
actions, closed, controls = [], [], []
handle.info_action_requested.connect(actions.append)
handle.pulse_tab_closed.connect(closed.append)
handle.pulse_include_off_toggled.connect(lambda key, on: controls.append(('off', key, on)))
handle.pulse_selectors_toggled.connect(lambda key, on: controls.append(('selectors', key, on)))
handle.pulse_size_committed.connect(lambda key, size: controls.append(('size', key, size)))
handle.pulse_save_requested.connect(lambda key: controls.append(('save', key)))
view.set_archive_info(
    (('Devices', (('sequencer pulse (scan)', {'text': 'Open scan', 'action': 'pulse:k'}),)), ('Flow', ())),
    {'nodes': (), 'edges': ()},
)
buttons = [b for b in view.info_pane.findChildren(FluentButton) if b.text() == 'Open scan']
assert len(buttons) == 1, [b.text() for b in view.info_pane.findChildren(FluentButton)]
buttons[0].click()
assert actions == ['pulse:k']
before = view.tabs.count()
handle.open_pulse_tab('k', 'Pulse · scan')
assert view.tabs.count() == before + 1
assert view.tabs.tabText(view.tabs.currentIndex()) == 'Pulse · scan'
assert handle.has_pulse_tab('k')
page = view._pulse_tabs['k']
assert page.preview_size_combo.isVisibleTo(page) and page.preview_selectors_switch.isVisibleTo(page)
assert handle.set_pulse_size_names('k', ('2x2', '4x4')) and handle.set_pulse_size('k', '4x4')
assert page.preview_size == '4x4'
assert handle.set_pulse_status('k', '4x4 · 3 periods') and page.preview_status.text() == '4x4 · 3 periods'
page.preview_include_off.setChecked(True)
page.preview_selectors_switch.setChecked(True)
page.preview_size_combo.setCurrentText('2x2')
page.preview_size_combo.activated[int].emit(page.preview_size_combo.currentIndex())
page.preview_save_figure_button.click()
assert controls == [('off', 'k', True), ('selectors', 'k', True), ('size', 'k', '2x2'), ('save', 'k')], controls
class _Host:
    logical_size = (120, 80)
    def __init__(self):
        self.widget = QtWidgets.QWidget()
    def qt_widget(self):
        return self.widget
    def wheel_target(self):
        return self.widget
host = _Host()
assert handle.show_pulse('k', host)
assert page.preview_placeholder.isHidden()
assert host.widget.parent() is page.preview_body
handle.open_pulse_tab('k', 'Pulse · scan')
assert view.tabs.count() == before + 1, 'a second open focuses, never duplicates'
view._tab_close_clicked(page)
assert closed == ['k']
assert handle.close_pulse_tab('k')
assert view.tabs.count() == before and not handle.has_pulse_tab('k')
assert handle.show_pulse_placeholder('k', 'gone') is False
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
# The role controls ARE the statement of what rows and columns are; a
# second line repeating the choice just made is not a second fact.
assert not hasattr(editor, 'row_axis_label')
assert not hasattr(editor, 'column_axis_label')
roles = {
    label.text().split(' (')[0]: mode.currentText()
    for _holder, label, mode in editor._axis_view_widgets.values()
}
assert roles.get('spatial-y') == 'Rows ↓', roles
assert roles.get('spatial-x') == 'Columns →', roles
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
# Add is a mode of the same control group, entered without a presenter
# round trip: nothing of the selected axis may still be clickable, and the
# New axis form must be writable even when nothing was selected before.
assert not editor.remove_axis_button.isEnabled()
assert not editor.axis_value_table.isEnabled()
assert editor.axis_name_edit.isEnabled() and editor.apply_axis_button.isEnabled()
before_delete = len(intents)
QtTest.QTest.mouseClick(editor.remove_axis_button, QtCore.Qt.LeftButton)
assert len(intents) == before_delete, 'Delete fired while adding an axis'
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

# A Dataset with no named axis left can still get its first one.
scalar = dict(projection)
scalar['axes'] = (); scalar['selected_axis'] = ''
scalar['axis_values'] = {'shape': (1, 0), 'values': ((),), 'row_headers': ('Value',),
                         'column_headers': (), 'editable': True}
scalar['table'] = dict(projection['table'])
scalar['table'].update({'shape': (1, 1), 'values': ((0,),), 'validity': ((True,),),
                        'row_headers': (0,), 'column_headers': (0,),
                        'structure': ((), (), ()), 'axes': ()})
editor.update_projection(scalar)
assert not editor.axis_name_edit.isEnabled() and not editor.apply_axis_button.isEnabled()
QtTest.QTest.mouseClick(editor.add_axis_button, QtCore.Qt.LeftButton)
assert editor.axis_name_edit.isEnabled() and editor.apply_axis_button.isEnabled()
assert editor.domain_combo.count() == 3 and not editor.remove_axis_button.isEnabled()
editor.axis_name_edit.setText('again')
QtTest.QTest.mouseClick(editor.apply_axis_button, QtCore.Qt.LeftButton)
assert intents[-1] == (
    'manual-1',
    {'op': 'add_axis', 'name': 'again', 'length': 1, 'unit': '',
     'domain': 'repeat'},
)
editor.update_projection(projection)

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


def test_a_record_is_read_as_a_tree_and_the_flow_is_a_map_of_it() -> None:
    """A run's record opens under the run, a device's snapshot under the
    device; a filter finds a name or a value anywhere in the tab; Copy
    takes the whole value; and a click on a flow card opens the row it
    stands for.  The pane used to print each record as a Python literal."""

    _run_qt(
        """
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from zlc_ui.fluent.info_pane import InfoPane, copy_text, value_text
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['info-tree'])

TITLES = ('Plot', 'Logic', 'Devices', 'Flow', 'Raw')
record = {
    'outputs': ['frames'],
    'parameters': {'exposure_seconds': 0.02, 'frames_per_cycle': 3, 'photoelectrons': False, 'roi_xywh': None},
}
device = {
    'roles': ['camera'], 'used_by': ['cm'],
    'snapshots': [{'logic': 'cm', 'scope': 'run', 'snapshot': {
        'exposure_seconds': 0.02, 'roi_shape_yx': [96, 128],
        'coordinates': list(range(96)),
    }}],
}
pane = InfoPane(label_names=TITLES, graph_tabs=('Flow',))
pane.set_tabs((
    ('Plot', (('data', '1x3x96x128 uint16'),)),
    ('Logic', (('cm', record),)),
    ('Devices', (('camera', device), ('sequencer pulse 1', {'text': 'imaging', 'action': 'pulse:k'}))),
    ('Flow', ()),
    ('Raw', (('source', {'signal': '@logic/cm/frames', 'title': 'camera'}),)),
))
pane.resize(520, 640); pane.show(); app.processEvents()

# The record is a tree: the run's name, then its fields, then the fields
# of its fields -- and a summary of the scalars beside every branch.
logic = pane._rows_tabs['Logic'].tree
cm = logic.topLevelItem(0)
assert cm.text(0) == 'cm' and cm.isExpanded()
assert [cm.child(i).text(0) for i in range(cm.childCount())] == ['outputs', 'parameters']
assert cm.child(0).text(1) == 'frames'
parameters = cm.child(1)
assert not parameters.isExpanded()
assert parameters.text(1) == 'exposure_seconds: 0.02; frames_per_cycle: 3; photoelectrons: false; roi_xywh: none'
assert [parameters.child(i).text(1) for i in range(4)] == ['0.02', '3', 'false', 'none']
assert value_text(list(range(96))) == '96 numbers, 0 to 95'
assert copy_text(list(range(3))) == '0, 1, 2'
assert copy_text(record).splitlines()[0:3] == ['outputs: frames', 'parameters:', '  exposure_seconds: 0.02']

# A filter finds a value deep in a device's snapshot and opens the way to it.
devices = pane._rows_tabs['Devices']
devices.filter_edit.setText('roi_shape')
app.processEvents()
camera = devices.tree.topLevelItem(0)
assert not camera.isHidden() and camera.isExpanded()
snapshots = next(camera.child(i) for i in range(camera.childCount()) if camera.child(i).text(0) == 'snapshots')
assert snapshots.isExpanded() and not snapshots.isHidden()
roles = next(camera.child(i) for i in range(camera.childCount()) if camera.child(i).text(0) == 'roles')
assert roles.isHidden()
assert devices.tree.topLevelItem(1).isHidden(), 'the pulse row does not mention roi_shape'
devices.filter_edit.clear(); app.processEvents()
assert not roles.isHidden() and not devices.tree.topLevelItem(1).isHidden()
assert camera.isExpanded() and not snapshots.isExpanded()

# Copy takes the whole value, not the summary on screen.
logic.setCurrentItem(parameters)
QtTest.QTest.keyClick(logic, QtCore.Qt.Key_C, QtCore.Qt.ControlModifier)
assert QtWidgets.QApplication.clipboard().text() == copy_text(record['parameters'])
assert logic.row_name() == 'cm.parameters'

# A pressed action row still asks for its action.
actions = []; pane.action_requested.connect(actions.append)
button = devices.tree.itemWidget(devices.tree.topLevelItem(1), 1)
button.click()
assert actions == ['pulse:k']

# A flow card names its row; a click on it opens that tab on that row.
pane.set_graph('Flow', {
    'nodes': (
        {'id': 'device:camera', 'kind': 'device', 'title': 'camera', 'subtitle': 'camera', 'root': False, 'tooltip': 'camera', 'row': ('Devices', 'camera')},
        {'id': 'logic:cm', 'kind': 'logic', 'title': 'cm', 'subtitle': 'frames', 'root': True, 'tooltip': 'cm', 'row': ('Logic', 'cm')},
    ),
    'edges': ({'source': 'device:camera', 'target': 'logic:cm', 'kind': 'device', 'label': 'camera'},),
})
pane.info_tabs.setCurrentWidget(pane._graph_tabs['Flow']); app.processEvents()
flow = pane._graph_tabs['Flow']
centre = flow.mapFromScene(flow._flow_node_rects['device:camera'].center())
QtTest.QTest.mouseClick(flow.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, centre)
app.processEvents()
assert pane.info_tabs.currentWidget() is devices
assert devices.tree.currentItem() is camera and camera.isExpanded()
assert pane.show_row('Logic', 'cm') and logic.currentItem() is cm
assert not pane.show_row('Logic', 'nobody')
"""
    )
