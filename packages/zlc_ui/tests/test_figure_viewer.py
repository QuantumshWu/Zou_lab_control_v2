from __future__ import annotations


def test_figure_viewer_mount_reconcile_and_open_intent(run_qt) -> None:
    run_qt(
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
# The two panes are built alike: the data band as tall as the path band,
# its row on the path row's centre line, the two tab strips on one top.
def centre(widget):
    return widget.mapTo(view, QtCore.QPoint()).y() + widget.height() / 2
band = view.info_pane.path_edit.parentWidget()
assert view._panel_bar.height() == band.height()
assert abs(centre(view.data_combo) - centre(view.info_pane.path_edit.edit)) <= 1
assert (view.tabs.mapTo(view, QtCore.QPoint()).y()
        == view.info_pane.info_tabs.mapTo(view, QtCore.QPoint()).y())
# Adding a panel and saving the board's picture head the board's own tab.
assert view.tabs.widget(0).isAncestorOf(view.add_panel_button)
assert view.tabs.widget(0).isAncestorOf(view.save_image_button)
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
view.finish_close()
assert not view.isVisible()
view.deleteLater(); app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
"""
    )


def test_a_played_pulse_offers_its_open_action_on_the_devices_page(run_qt) -> None:
    """The pure Viewer shell emits the recorded Pulse identity for its owner."""

    run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.figure_viewer import FigureViewerHandle, FigureViewerView
from zlc_ui.fluent import FluentButton
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['pulse-open'])
view = FigureViewerView()
handle = FigureViewerHandle(None, view)
actions = []
handle.info_action_requested.connect(actions.append)
view.set_archive_info(
    (('Devices', (('sequencer pulse (scan)', {'text': 'Open scan', 'action': 'pulse:k'}),)), ('Flow', ())),
    {'nodes': (), 'edges': ()},
)
# A tab's rows exist once a reader turns to it, so turn to it: pressing
# the button is what this is about, and nobody presses one they cannot see.
tabs = view.info_pane.info_tabs
tabs.setCurrentIndex(next(
    index for index in range(tabs.count()) if tabs.tabText(index) == 'Devices'
))
buttons = [b for b in view.info_pane.findChildren(FluentButton) if b.text() == 'Open scan']
assert len(buttons) == 1, [b.text() for b in view.info_pane.findChildren(FluentButton)]
buttons[0].click()
assert actions == ['pulse:k']
"""
    )


def test_manual_data_editor_is_virtual_and_emits_plain_intents(run_qt) -> None:
    run_qt(
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
        'shape': (100_000, 1), 'values': range(100_000),
        'row_headers': range(100_000), 'column_headers': ('Value',),
        'editable': True,
    },
    'table': {
        'component': 'values',
        'component_choices': (
            ('values', 'Values', True), ('validity', 'Validity', True), ('sigma', 'Sigma', False),
        ),
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
# The editor opens on its Dataset and Axes page; the Data card is the
# other page of the choice at its top.
assert editor.page_choice.current() == 'dataset'
assert editor.axes_group.isVisible() and not editor.data_group.isVisible()
assert editor.dataset_group.width() == editor.axes_group.width()
for first, second in ((editor.name_edit, editor.axis_name_edit),
                      (editor.unit_edit, editor.axis_unit_edit),
                      (editor.dtype_combo, editor.axis_size_spin)):
    assert first.mapTo(editor, QtCore.QPoint()).x() == second.mapTo(editor, QtCore.QPoint()).x()
    assert first.width() == second.width()
# The domain is a segmented choice as wide as its labels, on the same edge.
assert (editor.note_edit.mapTo(editor, QtCore.QPoint()).x()
        == editor.domain_choice.mapTo(editor, QtCore.QPoint()).x())
# Sigma keeps its place in the components, offered once the draft has one.
assert not editor.component_choice.button('sigma').isEnabled()
assert editor.component_choice.button('validity').isEnabled()
assert editor.axis_value_model.rowCount() == 100_000
assert editor.axis_value_model.columnCount() == 1
assert editor.axis_value_table.verticalScrollBar().maximum() > 0
assert editor.axis_value_table.sizeAdjustPolicy() == QtWidgets.QAbstractScrollArea.AdjustIgnored
roles = {
    label.text().split(' (')[0]: mode.currentText()
    for _holder, label, mode in editor._axis_view_widgets.values()
}
assert roles.get('spatial-y') == 'Rows ↓', roles
assert roles.get('spatial-x') == 'Columns →', roles
# The three cards share one label column, wide enough for each axis row's
# name, size and unit: the size is what the suffix is there to show.
from zlc_ui.fluent import FluentSettingRow
assert len({row._label.width() for row in editor.findChildren(FluentSettingRow)}) == 1
for _holder, label, _mode in editor._axis_view_widgets.values():
    assert label.fontMetrics().horizontalAdvance(label.text()) < label.width(), label.text()
# Add axis is the Axes card's command on its title line; the values end on
# the first column of the fields above them.
def right(widget):
    return widget.mapTo(editor, QtCore.QPoint(widget.width(), 0)).x()
from zlc_ui.fluent import CARD_PAD, scaled_px
assert editor.add_axis_button.mapTo(editor.axes_group, QtCore.QPoint()).y() == scaled_px(CARD_PAD)
assert right(editor.axis_value_table) == right(editor.axis_name_edit)
card_width = editor.axes_group.width()
QtTest.QTest.mouseClick(editor.page_choice.button('data'), QtCore.Qt.LeftButton)
app.processEvents()
assert editor.page_choice.current() == 'data'
assert editor.data_group.isVisible() and not editor.axes_group.isVisible()
assert editor.data_group.width() == card_width
assert editor.value_model.rowCount() == 100_000
assert editor.value_model.columnCount() == 1_000
assert editor.value_table.verticalScrollBar().maximum() > 0
assert editor.value_table.horizontalScrollBar().maximum() > 0
assert values.reads < 500, values.reads
assert editor.value_table.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
small = dict(projection)
small_values = LazyMatrix(12, 1)
small['table'] = dict(projection['table'], shape=(12, 1), values=small_values)
editor.update_projection(small); app.processEvents()
assert editor.value_model.rowCount() == 12
assert editor.value_table.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff
assert editor.value_table.verticalScrollBar().maximum() == 0
assert small_values.reads < 500, small_values.reads
editor.update_projection(projection); app.processEvents()
assert editor.value_table.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
assert editor.value_table.indexWidget(editor.value_model.index(0, 0)) is None
removed = []; headers = []
for combo in (editor.dtype_combo, editor.axis_combo,
              *(parts[2] for parts in editor._axis_view_widgets.values())):
    combo.model().rowsRemoved.connect(lambda *_args: removed.append(True))
editor.axis_value_model.headerDataChanged.connect(lambda *_args: headers.append(True))
editor.value_model.headerDataChanged.connect(lambda *_args: headers.append(True))
editor.update_projection(projection)
assert not removed, 'a data-only projection rebuilt unchanged choice rows'
assert not headers, 'unchanged axis coordinates were announced as new headers'
changed_regions = []
editor.value_model.dataChanged.connect(lambda first, last, _roles:
    changed_regions.append((first.row(), first.column(), last.row(), last.column())))
changed = dict(projection)
changed['table'] = dict(projection['table'], changed_cells=((3, 4),))
changed['axis_values'] = dict(projection['axis_values'], changed_cells=())
editor.update_projection(changed)
assert changed_regions == [(3, 4, 3, 4)]
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

QtTest.QTest.mouseClick(editor.page_choice.button('dataset'), QtCore.Qt.LeftButton)
app.processEvents()
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
    editor.axis_value_model.index(1, 0), '0.5', QtCore.Qt.EditRole
)
assert intents[-1] == (
    'manual-1',
    {'op': 'set_axis_values', 'axis_id': 'spatial-y',
     'cells': ((1, 0, '0.5'),)},
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
sent = len(intents)
QtTest.QTest.mouseClick(editor.domain_choice.button('cell_data'), QtCore.Qt.LeftButton)
assert len(intents) == sent, 'a domain is part of the axis draft, not an edit of its own'
QtTest.QTest.mouseClick(editor.apply_axis_button, QtCore.Qt.LeftButton)
assert intents[-1] == (
    'manual-1',
    {'op': 'add_axis', 'name': 'shot', 'length': 2, 'unit': '',
     'domain': 'cell_data'},
)
# A refused Create re-projects the axes that were already there.  The New
# axis draft, and the Create that sends it, stay the operator's: cleared on
# the press, the draft became the selected axis and its Apply renamed it.
editor.update_projection(projection)
assert editor.apply_axis_button.text() == 'Create axis'
assert editor.axis_name_edit.text() == 'shot' and editor.axis_size_spin.value() == 2
QtTest.QTest.mouseClick(editor.apply_axis_button, QtCore.Qt.LeftButton)
assert intents[-1][1]['op'] == 'add_axis', intents[-1]
# Create lands as a projection that selects the axis it made.
landed = dict(projection, selected_axis='shot', axes=projection['axes'] + (
    {'id': 'shot', 'domain': 'repeat', 'domain_label': 'Repeat',
     'name': 'shot', 'size': 2, 'unit': ''},
))
editor.update_projection(landed)
assert editor.apply_axis_button.text() == 'Apply axis'
assert editor.remove_axis_button.isEnabled()

# A Dataset with no named axis left can still get its first one.
scalar = dict(projection)
scalar['axes'] = (); scalar['selected_axis'] = ''
scalar['axis_values'] = {'shape': (0, 1), 'values': (), 'row_headers': (),
                         'column_headers': ('Value',), 'editable': True}
scalar['table'] = dict(projection['table'])
scalar['table'].update({'shape': (1, 1), 'values': ((0,),), 'validity': ((True,),),
                        'row_headers': (0,), 'column_headers': (0,),
                        'structure': ((), (), ()), 'axes': ()})
editor.update_projection(scalar)
assert not editor.axis_name_edit.isEnabled() and not editor.apply_axis_button.isEnabled()
QtTest.QTest.mouseClick(editor.add_axis_button, QtCore.Qt.LeftButton)
assert editor.axis_name_edit.isEnabled() and editor.apply_axis_button.isEnabled()
assert editor.domain_choice.isEnabled() and not editor.remove_axis_button.isEnabled()
editor.axis_name_edit.setText('again')
QtTest.QTest.mouseClick(editor.apply_axis_button, QtCore.Qt.LeftButton)
assert intents[-1] == (
    'manual-1',
    {'op': 'add_axis', 'name': 'again', 'length': 1, 'unit': '',
     'domain': 'repeat'},
)
editor.update_projection(projection)
# A projection keeps the page the operator chose.
assert editor.page_choice.current() == 'dataset'
QtTest.QTest.mouseClick(editor.page_choice.button('data'), QtCore.Qt.LeftButton)
app.processEvents()

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
view.finish_close()
assert not view.isVisible()
view.deleteLater(); app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
"""
    )


def test_the_right_surface_has_one_left_edge_one_right_edge_and_one_page_gap(run_qt) -> None:
    """The data bar, the tab strip and every tab page of the right surface
    -- the board, the data editor and a panel's Edit page -- start on one
    card-pad edge and end on one right edge; a tab page starts one card pad
    under its strip in both panes, and the data editor's first card stands
    where the board's does; the axis values stand under a whole heading
    cell a section gap below the fields; and each page of the data editor
    scrolls only its value table, which is whole."""

    run_qt(
        """
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.figure_viewer import FigureViewerView
from zlc_ui.fluent import (
    ACCENT, FluentLabel, FluentSectionLabel, card_pad, control_height,
    fluent_scrollbar_thickness, row_gap, section_gap,
)
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['figure-surface-edges'])

def settle():
    for _ in range(6):
        app.processEvents()

def box(widget, top):
    origin = widget.mapTo(top, QtCore.QPoint())
    return origin.x(), origin.y(), origin.x() + widget.width(), origin.y() + widget.height()

def underline_left(tabs, top):
    # Where the current tab's ACCENT underline starts, on the strip's last row.
    bar = tabs.tabBar()
    image = top.grab().toImage()
    ratio = image.devicePixelRatio() or 1.0
    left, _y, right, bottom = box(bar, top)
    y = int(bottom * ratio) - 1
    accent = QtGui.QColor(ACCENT).name()
    return next(x / ratio for x in range(int(left * ratio), int(right * ratio))
                if image.pixelColor(x, y).name() == accent)

class Matrix:
    def __init__(self, rows, columns): self.shape = (rows, columns)
    def __getitem__(self, index): return index[0] + index[1]

rows = 5_000
projection = {
    'dataset': {'name': 'frame', 'dtype': '<f8', 'unit': '', 'note': '', 'source': '',
                'dtype_choices': (('<f8', 'Float 64'),)},
    'domain_choices': (('point', 'Point'),),
    'axes': ({'id': 'y', 'domain': 'point', 'domain_label': 'Point', 'name': 'y',
              'size': rows, 'unit': ''},),
    'selected_axis': 'y',
    'axis_values': {'shape': (rows, 1), 'values': range(rows), 'row_headers': range(rows),
                    'column_headers': ('Value',), 'editable': True},
    'table': {
        'component': 'values', 'component_choices': (('values', 'Values', True),),
        'sigma_enabled': False, 'blank_help': '', 'blank_hint': '',
        'shape': (rows, 60), 'values': Matrix(rows, 60), 'validity': None,
        'row_headers': range(rows), 'column_headers': range(60),
        'structure': ((('y', rows),), (), ()),
        'axes': ({'axis_id': 'y', 'name': 'y', 'size': rows, 'unit': '', 'mode': 'rows',
                  'index': 0, 'scope_choices': tuple((index, str(index)) for index in range(rows))},),
    },
    'dirty': False, 'can_apply': True, 'can_save': False,
    'save_suggested': 'frame.npz', 'message': '',
}

view = FigureViewerView()
view.set_panel_sizes(('2x2',), '2x2')
view.set_panel_kinds((('image', 'Image'),))
view.add_panel('panel-1', 'Camera frame')
view.resize(1200, 640); view.show(); settle()
surface = view.tabs.parentWidget()
pad = card_pad()
inset = max(pad, fluent_scrollbar_thickness())
edge, end = pad, surface.width() - inset
bar = view.tabs.tabBar()

# The tab strip runs from side to side: the first tab's own card-pad inset
# is what puts its underline on the surface's edge.
assert box(view.tabs, surface)[0] == 0 and box(view.tabs, surface)[2] == surface.width()
label = view.data_combo.parentWidget().layout().itemAt(0).widget()
assert isinstance(label, FluentLabel) and not isinstance(label, FluentSectionLabel)
assert box(label, surface)[0] == edge
assert box(view.new_data_button, surface)[2] == end
view.open_data_editor('frame', projection, 'Data · frame')
editor = view._data_editors['frame']
settle()
# The Monitor page: its commands and its first card on the same two edges,
# one card pad under the strip a second page has shown.
view.tabs.setCurrentIndex(0); settle()
strip_bottom = box(bar, surface)[3]
assert bar.isVisible()
# The first tab's underline starts on that edge too (the strip's page inset).
assert underline_left(view.tabs, surface) == edge
assert box(view.kind_combo, surface)[0] == edge
assert box(view.save_image_button, surface)[2] == end
assert box(view.kind_combo, surface)[1] - strip_bottom == pad
card = view._cards['panel-1']
assert box(card, surface)[0] == edge
assert box(card, surface)[1] - box(view.kind_combo, surface)[3] == pad
board_commands, board_card = box(view.kind_combo, surface), box(card, surface)
view.tabs.setCurrentWidget(editor); settle()
for page, cards in (('dataset', (editor.dataset_group, editor.axes_group)),
                    ('data', (editor.data_group,))):
    editor.page_choice.button(page).click(); settle()
    for widget in (*cards, editor.discard_button):
        assert box(widget, surface)[0] == edge, widget
    for widget in (*cards, editor.save_button):
        assert box(widget, surface)[2] == end, widget
    # The page's choice heads it where the board's commands head the
    # board, and its first card stands where the board's first card does.
    assert box(editor.page_choice, surface)[:2] == board_commands[:2]
    assert box(cards[0], surface)[1] == board_card[1]
editor.page_choice.button('dataset').click(); settle()
assert box(editor.page_choice, surface)[1] - strip_bottom == pad
# The left pane's pages start as far under their own strip.
pane = view.info_pane
page = pane.info_tabs.currentWidget()
filter_edit = page.findChildren(QtWidgets.QLineEdit)[0]
assert (box(filter_edit, view)[1] - box(pane.info_tabs.tabBar(), view)[3]) == pad
assert underline_left(pane.info_tabs, view) == box(filter_edit, view)[0]
# The two panes' last lines read on one line: the pane's status on the
# line of the editor's action bar.
def centre(widget):
    top = box(widget, view)
    return (top[1] + top[3]) / 2.0
assert box(pane.status, view)[3] == box(surface, view)[3]
assert centre(pane.status.message) == centre(pane.status.dot) == centre(editor.discard_button)
assert box(filter_edit, view)[1] == box(editor.page_choice, view)[1]
# File names its path as Data names its chooser: a row label, one gap from it.
header = pane.path_edit.parentWidget().layout()
assert isinstance(header.itemAt(0).widget(), FluentLabel)
assert not isinstance(header.itemAt(0).widget(), FluentSectionLabel)
assert header.spacing() == row_gap()

# Axis values: a whole control-high heading cell, a section gap under the
# fields, and the table one row gap under it.
heading = next(w for w in editor.axes_group.findChildren(FluentSectionLabel) if w.text() == 'Axis values')
assert heading.contentsMargins().top() == 0 and heading.height() == control_height()
unit_row = editor._axis_rows['unit']
assert box(heading, editor)[1] - box(unit_row, editor)[3] == section_gap()
assert box(editor.axis_value_table, editor)[1] - box(heading, editor)[3] == row_gap()

# Each page scrolls one thing, its long value table: the page has nothing
# to scroll, and its card -- and so the table, its column names and its
# horizontal bar -- is whole.
viewport = editor.editor_scroll.viewport()
page_bar = editor.editor_scroll.verticalScrollBar()
for page, table, card in (('dataset', editor.axis_value_table, editor.axes_group),
                          ('data', editor.value_table, editor.data_group)):
    editor.page_choice.button(page).click(); settle()
    assert page_bar.maximum() == 0 and not page_bar.isVisible(), page
    assert table.verticalScrollBar().maximum() > 0
    assert box(card, viewport)[3] <= viewport.height(), (page, box(card, viewport), viewport.height())
# A value is never shown in part: the table scrolls by whole columns, and
# the column the viewport's right edge cuts shows nothing.
table = editor.value_table
assert table.horizontalScrollMode() == QtWidgets.QAbstractItemView.ScrollPerItem
cut = table.cut_column_left()
if cut is not None:
    image = table.viewport().grab().toImage()
    ratio = image.devicePixelRatio() or 1.0
    assert not any(image.pixelColor(x, y).lightness() < 150
                   for x in range(int(cut * ratio), image.width()) for y in range(image.height()))
# A taller window gives the table the rows it gains, and the page still
# has nothing to scroll.
before = editor.value_table.height()
view.resize(1200, 900); settle()
assert editor.value_table.height() > before
assert page_bar.maximum() == 0
assert box(editor.data_group, viewport)[3] <= viewport.height()

# A panel's Edit page lays its content on this surface's page inset, as
# the board and the data editor do: its first line one card pad under the
# strip, its cards on the left and right edges and padded by a card pad.
view.resize(1200, 640); settle()
state ={'signal': '@figure/1/data', 'kind': 'image', 'cell_kind': '', 'size': '2x2',
         'interval_ms': 400, 'title': 'Camera frame', 'semantic': {}, 'display': {},
         'fit': {}, 'overlay_signal': ''}
panel_surface = {'semantic': (), 'display': (), 'fit': (), 'data_structure': (),
                 'data_scope': (), 'paints_images': True}
view.set_panel_intervals((100, 400), 400)
view.set_panel_projection('panel-1', state, panel_surface)
view.open_panel_editor('panel-1', {
    'state': state, 'parameter_surface': panel_surface,
    'signal_options': (('this archive', (('Camera frame', 'data'),)),),
    'overlay_signal_options': (), 'frozen_signal': '@figure/1/data',
    'frozen_snapshot': None, 'save_directory': '.',
}, 'Edit · Camera frame')
settle()
panel_editor = view._editors['panel-1']
assert view.tabs.currentWidget() is panel_editor
assert box(panel_editor.title_label, surface)[0] == edge
assert box(panel_editor.title_label, surface)[1] - strip_bottom == pad
for group in (panel_editor.snapshot_group, panel_editor.save_group):
    assert box(group, surface)[0] == edge and box(group, surface)[2] == end, group.title()
assert box(panel_editor.refresh_button, panel_editor.snapshot_group)[0] == pad
saved = panel_editor.save_group
assert box(panel_editor.save_button, saved)[2] == saved.width() - pad
assert box(panel_editor.save_button, saved)[3] == saved.height() - pad
view.finish_close()
"""
    )


def test_figure_viewer_demo_smoke(run_qt) -> None:
    completed = run_qt(
        "import runpy, sys; "
        "sys.argv = ['demo_figure_viewer.py', '--once']; "
        "runpy.run_path('examples/demo_figure_viewer.py', run_name='__main__')"
    )
    # What the demo shows now is a HOST filling the page through the handle;
    # the File field's own intent is checked against the widget above, which is
    # where a widget may be poked at all.
    assert "filled: 2 signals" in completed.stdout


def test_the_info_pane_width_is_stable_across_loaded_content(run_qt) -> None:
    """Loading metadata must not move the Viewer split or resize the window.

    The Viewer's five info tabs and the console's two are permanent
    navigation, named the same way on both windows.
    """

    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import TaskConsoleView
from zlc_ui.figure_viewer import FigureViewerView
app = ensure_qt_app(['pane-width'])
view = FigureViewerView(); view.resize(1200, 700); view.show(); app.processEvents()
pane = view.info_pane
bar = pane.info_tabs.tabBar()

assert [pane.info_tabs.tabText(i) for i in range(bar.count())] == [
    'Plot', 'Logic', 'Devices', 'Flow', 'Raw'
]
console = TaskConsoleView()
assert [console.tabs.tabText(i) for i in range(console.tabs.count())] == ['Monitor', 'Logic']
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


def test_a_record_is_read_as_a_tree_and_the_flow_is_a_map_of_it(run_qt) -> None:
    """A run's record opens under the run, a device's snapshot under the
    device; a filter finds a name or a value anywhere in the tab; Copy
    takes the whole value; and a click on a flow card opens the row it
    stands for.  The pane used to print each record as a Python literal.

    A refresh replaces the CONTENT of the tabs, not which one is open.
    Every tab is destroyed and rebuilt on each ``set_tabs`` -- the rows
    change, the five titles never do -- and the rebuilt stack starts at the
    first tab, so anyone reading Devices was thrown back to Plot by a
    refresh they did not ask for.
    """

    run_qt(
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
# of its fields -- every one on a row of its own, open from the start, a
# branch saying only how many rows are under it.  Nothing is repeated in
# a tooltip: a wide value is reached by scrolling sideways.
logic = pane._rows_tabs['Logic'].tree
pane.info_tabs.setCurrentWidget(pane._rows_tabs['Logic'])
app.processEvents()
assert logic.sizeHintForRow(0) > logic.fontMetrics().height()
tab = pane._rows_tabs['Logic']
assert tab.height() - logic.geometry().bottom() - 1 > 0
cm = logic.topLevelItem(0)
assert cm.text(0) == 'cm' and cm.isExpanded()
assert [cm.child(i).text(0) for i in range(cm.childCount())] == ['outputs', 'parameters']
assert cm.child(0).text(1) == 'frames'
parameters = cm.child(1)
assert parameters.isExpanded()
assert parameters.text(1) == '4 fields'
assert [parameters.child(i).text(1) for i in range(4)] == ['0.02', '3', 'false', 'none']
assert parameters.toolTip(1) == '' and parameters.child(0).toolTip(1) == ''
assert logic.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
assert not logic.header().stretchLastSection()
assert value_text(list(range(96))) == '96 numbers, 0 to 95'
assert copy_text(list(range(3))) == '0, 1, 2'
assert copy_text(record).splitlines()[0:3] == ['outputs: frames', 'parameters:', '  exposure_seconds: 0.02']

# A filter finds a value deep in a device's snapshot, opens the way to
# it, and tints the cell it found -- and only that cell.
from zlc_ui.fluent.info_pane import _MATCH_ROLE
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
found = snapshots.child(0).child(2).child(1)
assert found.text(0) == 'roi_shape_yx' and not found.isHidden()
assert found.data(0, _MATCH_ROLE) and not found.data(1, _MATCH_ROLE)
assert not camera.data(0, _MATCH_ROLE) and not snapshots.data(1, _MATCH_ROLE)
devices.filter_edit.clear(); app.processEvents()
assert not roles.isHidden() and not devices.tree.topLevelItem(1).isHidden()
assert camera.isExpanded() and snapshots.isExpanded(), 'an empty filter opens the whole tree again'
assert not found.data(0, _MATCH_ROLE)

# Copy takes the whole value, not the summary on screen.
logic.setCurrentItem(parameters)
QtTest.QTest.keyClick(logic, QtCore.Qt.Key_C, QtCore.Qt.ControlModifier)
assert QtWidgets.QApplication.clipboard().text() == copy_text(record['parameters'])
assert logic.row_name() == 'cm.parameters'

# A pressed action row still asks for its action.
actions = []; pane.action_requested.connect(actions.append)
holder = devices.tree.itemWidget(devices.tree.topLevelItem(1), 1)
button = holder.findChild(QtWidgets.QPushButton)
pane.info_tabs.setCurrentWidget(devices)
app.processEvents()
devices.tree.verticalScrollBar().setValue(devices.tree.verticalScrollBar().maximum())
app.processEvents()
bottom = button.mapTo(devices.tree.viewport(), QtCore.QPoint()).y() + button.height()
assert bottom < devices.tree.viewport().height(), 'action content must fit above scrollbar'
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

refreshed = InfoPane(label_names=TITLES)
refreshed.set_tabs(tuple((title, (('a', '1'),)) for title in TITLES))
app.processEvents()
refreshed.info_tabs.setCurrentIndex(2)
app.processEvents()
assert refreshed.info_tabs.tabText(refreshed.info_tabs.currentIndex()) == 'Devices'

refreshed.set_tabs(tuple((title, (('a', '9'),)) for title in TITLES))
app.processEvents()
assert refreshed.info_tabs.tabText(refreshed.info_tabs.currentIndex()) == 'Devices', (
    refreshed.info_tabs.tabText(refreshed.info_tabs.currentIndex()))

# A tab that no longer exists cannot be kept; falling back to the first is
# the only honest answer, and it must not raise.
refreshed.set_tabs((('Plot', (('a', '9'),)), ('Logic', (('a', '9'),))))
app.processEvents()
assert refreshed.info_tabs.tabText(refreshed.info_tabs.currentIndex()) == 'Plot'
"""
    )
