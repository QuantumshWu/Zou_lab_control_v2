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
    'live': False,
}
view.open_panel_editor(
    'panel-1', projection, 'Edit · Camera frame'
)
editor = view._editors['panel-1']
assert view.tabs.currentWidget() is editor
assert 'interval_ms' not in editor.panel_form.spec.keys
assert editor.snapshot_group.isHidden()
assert editor.interaction_group.isHidden()
assert editor.producer_group.isHidden()
assert not editor.open_producer_button.isEnabled()
assert editor.save_group.isHidden()
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
import zlc_ui.figure_viewer.view as viewer_module
from zlc_ui.figure_viewer import FigureViewerView
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['manual-data-editor'])

class LazyMatrix:
    def __init__(self, rows, columns):
        self.shape = (rows, columns)
        self.reads = 0
    def __getitem__(self, index):
        self.reads += 1
        row, column = index
        return row * 1000 + column

class LazyLabels:
    def __init__(self, size):
        self.size = size
        self.reads = 0
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        self.reads += 1
        return '' if index % 2 else f'row-{index}'

class LazyGridIndices:
    def __init__(self, rows, rank):
        self.shape = (rows, rank)
        self.reads = 0
    def __getitem__(self, index):
        self.reads += 1
        row, dimension = index
        return (row // 10, row % 10)[dimension]

values = LazyMatrix(100_000, 1_000)
validity = LazyMatrix(100_000, 1_000)
coordinate_labels = LazyLabels(100_000)
row_grid_indices = LazyGridIndices(100_000, 2)
projection = {
    'dataset': {
        'name': 'manual image', 'dtype': 'float64', 'unit': 'count',
        'dtype_choices': (('float64', 'Float 64'), ('uint16', 'Unsigned 16')),
        'note': '', 'source': 'New manual Dataset',
    },
    'domain_choices': (
        ('repeat', 'Repeat'), ('point', 'Point / Grid'), ('cell', 'Cell'),
    ),
    'axes': (
        {
            'id': 'spatial-y', 'domain': 'cell', 'domain_label': 'Cell',
            'name': 'spatial-y', 'size': 100_000,
            'role': 'spatial-y', 'unit': 'pixel',
            'coordinate_frame': 'camera',
            'role_choices': (('spatial-y', 'Spatial Y'), ('data', 'Data')),
        },
        {
            'id': 'spatial-x', 'domain': 'cell', 'domain_label': 'Cell',
            'name': 'spatial-x', 'size': 1_000,
            'role': 'spatial-x', 'unit': 'pixel',
            'coordinate_frame': 'camera',
            'role_choices': (('spatial-x', 'Spatial X'), ('data', 'Data')),
        },
        {
            'id': 'scan', 'domain': 'point', 'domain_label': 'Point / Grid',
            'name': 'detuning', 'size': 1_000, 'role': 'scan', 'unit': '',
            'coordinate_frame': '', 'value_kind': 'TEXT',
            'value_kind_choices': (('NUMERIC', 'Numeric'), ('TEXT', 'Text')),
            'show_value_kind': True, 'unit_enabled': False,
            'role_choices': (('scan', 'Scan'),),
        },
    ),
    'selected_axis': 'spatial-y',
    'coordinates': {
        'shape': (100_000, 2),
        'column_values': (range(100_000), coordinate_labels),
        'row_headers': range(100_000),
        'column_headers': ('Coordinate', 'Label'),
    },
    'slices': (
        {'axis_id': 'repeat', 'label': 'Repeat', 'size': 2_000_000,
         'index': 0, 'current_label': '0 · first'},
    ),
    'table': {
        'component': 'value',
        'component_choices': (
            ('value', 'Value'), ('validity', 'Validity'), ('sigma', 'Sigma', False),
        ),
        'sigma_enabled': False,
        'blank_help': 'Blank removes the value · Ctrl+C / Ctrl+V supported',
        'blank_hint': 'No value',
        'shape': values.shape, 'values': values, 'validity': validity,
        'row_header_grid': {
            'cell_indices': row_grid_indices,
            'coordinates': (range(10_000), range(900, 910)),
            'labels': (coordinate_labels, None),
        },
        'column_header_grid': {
            'cell_indices': tuple((index,) for index in range(1_000)),
            'coordinates': (range(1_000),),
            'labels': None,
        },
    },
    'dirty': True, 'can_apply': True, 'can_save': True,
    'save_suggested': 'manual-image.npz', 'message': '',
}

view = FigureViewerView(); view.set_panel_sizes(('2x2',), '2x2')
created = []; edited = []; intents = []; data_closed = []; panel_closed = []
view.new_data_requested.connect(lambda: created.append(True))
view.edit_data_requested.connect(edited.append)
view.data_editor_intent.connect(lambda editor_id, intent: intents.append((editor_id, intent)))
view.data_editor_closed.connect(data_closed.append)
view.panel_editor_closed.connect(panel_closed.append)
view.set_editable_data_choices((('dataset/image', 'Image data'),), current='dataset/image')
QtTest.QTest.mouseClick(view.new_data_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(view.edit_data_button, QtCore.Qt.LeftButton)
assert created == [True] and edited == ['dataset/image']

view.open_data_editor('manual-1', projection, 'Data · manual image')
editor = view._data_editors['manual-1']
view.resize(1500, 850); view.show(); app.processEvents()
assert view.tabs.currentWidget() is editor
assert view.tabs.tabText(view.tabs.indexOf(editor)).endswith(' *')
renamed_projection = dict(projection)
renamed_projection['dataset'] = {
    **projection['dataset'], 'name': 'renamed manual image',
}
assert view.update_data_editor('manual-1', renamed_projection)
assert view.tabs.tabText(view.tabs.indexOf(editor)) == 'Data · renamed manual image *'
assert editor.discard_button.isEnabled()
assert editor.value_model.rowCount() == 100_000
assert editor.value_model.columnCount() == 1_000
assert values.reads < 500, values.reads
assert coordinate_labels.reads < 5_000, coordinate_labels.reads
assert row_grid_indices.reads < 5_000, row_grid_indices.reads
assert editor.value_table.indexWidget(editor.value_model.index(0, 0)) is None
assert editor.value_model.headerData(20, QtCore.Qt.Vertical) == '(row-2, 900)'
assert editor.value_model.headerData(
    20, QtCore.Qt.Vertical, QtCore.Qt.ToolTipRole
) == '(row-2, 900)'
assert editor.value_model.headerData(3, QtCore.Qt.Horizontal) == '(3)'
assert editor.value_table.verticalHeader().minimumWidth() >= 46
slice_spin = editor._slice_widgets['repeat'][1]
assert slice_spin.maximum() == 1_999_999
slice_spin.setValue(1)
assert intents[-1] == (
    'manual-1', {'op': 'set_slice', 'axis_id': 'repeat', 'index': 1}
)

# A blank is passed through as pending text; the UI does not guess a dtype or validity.
editor.value_model.setData(editor.value_model.index(0, 0), '', QtCore.Qt.EditRole)
assert intents[-1] == (
    'manual-1',
    {'op': 'set_cells', 'component': 'value', 'cells': ((0, 0, ''),)},
)

editor.coordinate_model.setData(
    editor.coordinate_model.index(1, 1), '', QtCore.Qt.EditRole
)
assert intents[-1] == (
    'manual-1',
    {'op': 'set_coordinates', 'axis_id': 'spatial-y',
     'cells': ((1, 1, ''),)},
)

# One rectangular paste is one presenter intent, even for a large virtual table.
editor.value_table.setCurrentIndex(editor.value_model.index(2, 3))
QtWidgets.QApplication.clipboard().setText('1\\t\\n3\\t4')
editor.value_table.paste_clipboard()
assert intents[-1] == (
    'manual-1',
    {'op': 'set_cells', 'component': 'value',
     'cells': ((2, 3, '1'), (2, 4, ''), (3, 3, '3'), (3, 4, '4'))},
)

before_overflow = len(intents)
editor.value_table.setCurrentIndex(editor.value_model.index(99_999, 999))
QtWidgets.QApplication.clipboard().setText('1\\t2')
editor.value_table.paste_clipboard()
assert len(intents) == before_overflow
assert editor.message_label.text() == 'Paste exceeds the current table; resize axes first'

QtTest.QTest.mouseClick(editor.discard_button, QtCore.Qt.LeftButton)
assert intents[-1] == ('manual-1', {'op': 'discard'})

# Save intent carries the operator's chosen target and optional note.
editor.note_edit.setText('corrected camera background')
viewer_module.fluent_save_path = lambda *_args, **_kwargs: 'D:/data/manual-edited.npz'
QtTest.QTest.mouseClick(editor.save_button, QtCore.Qt.LeftButton)
assert intents[-1] == (
    'manual-1',
    {'op': 'save_as', 'path': 'D:/data/manual-edited.npz',
     'note': 'corrected camera background'},
)

# Point/Grid coordinate kind is explicit; TEXT does not leave a fake unit editable.
point_projection = dict(projection)
point_projection['selected_axis'] = 'scan'
editor.update_projection(point_projection)
assert editor._axis_rows['value_kind'].isVisible()
assert editor.value_kind_combo.currentData() == 'TEXT'
assert not editor.axis_unit_edit.isEnabled()
before_size_edit = len(intents)
size_line = editor.axis_size_spin.lineEdit()
size_line.setFocus(); size_line.selectAll()
QtTest.QTest.keyClicks(size_line, '1200'); app.processEvents()
assert len(intents) == before_size_edit
QtTest.QTest.keyClick(size_line, QtCore.Qt.Key_Return); app.processEvents()
assert intents[before_size_edit:] == [(
    'manual-1',
    {'op': 'set_axis_field', 'axis_id': 'scan',
     'field': 'size', 'value': 1_200},
)]
numeric = editor.value_kind_combo.findData('NUMERIC')
editor.value_kind_combo.setCurrentIndex(numeric)
editor.value_kind_combo.activated.emit(numeric)
assert intents[-1] == (
    'manual-1',
    {'op': 'set_axis_field', 'axis_id': 'scan',
     'field': 'value_kind', 'value': 'NUMERIC'},
)

# A Data tab close is not misrouted as a Panel editor close.
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
