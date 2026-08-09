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
    environment["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), str(SRC)))
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


def test_panel_card_construct_and_setters() -> None:
    _run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import PanelCardView
app = ensure_qt_app(['test'])
card = PanelCardView('panel-1', 'Card')
card.set_signal_choices((('source', (('Temperature', 'temperature'),)),))
state = {
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 100,
    'title': 'Card', 'semantic': {}, 'display': {}, 'fit': {},
    'site_overlay': 'off',
}
surface_projection = {
    'semantic': (), 'display': (), 'fit': (), 'site_overlay': None,
}
card.set_panel_projection(state, surface_projection)
class _GatedSurface(QtWidgets.QLabel):
    # What a plot widget is, as far as this card is concerned: something you
    # can stop the operator dragging on.
    interaction = True
    def set_interaction_enabled(self, enabled):
        self.interaction = bool(enabled)
surface = _GatedSurface('fake surface')
card.set_surface(surface)
card.set_status('ready', error=False)
card.set_selectors_enabled(False)
assert card.panel_id == 'panel-1'
signal_field = next(field for field in card._form_spec().fields if field.key == 'signal')
assert len(signal_field.choices) == 1
assert surface.parentWidget() is not None
# The switch is about dragging on the PLOT, not about the card's own controls:
# turning it off used to grey out this combo and never reach the plot at all.
assert surface.interaction is False
card.set_selectors_enabled(True)
assert surface.interaction is True
card.set_surface(None)
assert not card._placeholder.isHidden()
card.set_panel_projection(dict(state, size='1x4'), surface_projection)
assert card.panel_size == '1x4'
try:
    card.set_panel_projection(dict(state, size='not-a-v1-size'), surface_projection)
except ValueError:
    pass
else:
    raise AssertionError('invalid panel size was accepted')
"""
    )


def test_panel_card_qtest_signal_payloads() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import PanelCardView
app = ensure_qt_app(['test'])
card = PanelCardView('panel-1', 'Card')
card.set_interval_choices((100, 200, 400, 800))
card.set_panel_projection({
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 400,
    'title': 'Card', 'semantic': {}, 'display': {}, 'fit': {},
    'site_overlay': 'off',
}, {
    'semantic': (), 'display': (), 'fit': (), 'site_overlay': None,
    'semantic_unavailable': '', 'display_unavailable': '',
    'fit_unavailable': '',
})
card.resize(340, 180); card.show(); app.processEvents()
events = []
card.remove_requested.connect(lambda: events.append(('remove',)))
card.edit_requested.connect(lambda: events.append(('edit',)))
card.state_changed.connect(lambda patch: events.append(('state', patch)))
# The operator's own route: Edit and Remove live in the Setting popup, beside
# every other per-panel decision.  Reached any other way they were hidden
# widgets nothing ever showed, so a panel could not be removed at all.
top_levels = {widget for widget in app.topLevelWidgets() if widget.isVisible()}
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
assert card._settings_frame.parentWidget() is card
assert not card._settings_frame.isWindow()
assert card.rect().contains(card._settings_frame.geometry())
assert {widget for widget in app.topLevelWidgets() if widget.isVisible()} == top_levels
assert not hasattr(card, 'apply_button')
assert card._settings_scroll.verticalScrollBar().maximum() > 0
title = card._settings_form.widget_for('title')
title.setText('Card changed')
title.editingFinished.emit()
interval = card._settings_form.widget_for('interval_ms')
interval.setCurrentIndex(interval.findData(800))
interval.activated.emit(interval.currentIndex())
app.processEvents()
assert card._settings_frame.isVisible()
QtTest.QTest.mouseClick(card.edit_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
QtTest.QTest.mouseClick(card.remove_button, QtCore.Qt.LeftButton)
assert ('edit',) in events
assert ('remove',) in events
assert any(event[0] == 'state' and event[1]['title'] == 'Card changed' for event in events)
assert ('state', {'interval_ms': 800}) in events
"""
    )


def test_board_constructs_and_packs_from_metrics() -> None:
    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.board import BoardMetrics
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['test'])
card = PanelCardView('panel-1')
board = ConsoleBoardView(metrics=BoardMetrics(12, lambda size: (240, 160)))
board.set_cards((card,))
board.resize(300, 220); board.show(); app.processEvents()
assert card.geometry().getRect() == (12, 12, 240, 160)
assert not board.grab_board().isNull()
"""
    )


def test_board_qtest_drop_intent_payload() -> None:
    _run_qt(
        """
from PyQt5 import QtCore
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['test'])
card = PanelCardView('panel-1')
board = ConsoleBoardView(); board.set_cards((card,))
events = []
board.order_committed.connect(lambda value: events.append(value))
card.dropped.emit((40, 50))
assert events == [('panel-1',)]
"""
    )


def test_board_qtest_drag_reorders_and_matches_packer() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.board import BoardMetrics, GeomProxy, pack
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['test'])
metrics = BoardMetrics(10, lambda size: (100, 80))
cards = tuple(PanelCardView(f'panel-{index}') for index in range(3))
board = ConsoleBoardView(metrics=metrics); board.resize(260, 220); board.set_cards(cards); board.show(); app.processEvents()
events = []
board.order_committed.connect(events.append)
target = cards[2].geometry().center()
local_target = cards[0].mapFrom(board, target)
QtTest.QTest.mousePress(cards[0], QtCore.Qt.LeftButton, pos=QtCore.QPoint(20, 20))
QtTest.QTest.mouseMove(cards[0], local_target)
QtTest.QTest.mouseRelease(cards[0], QtCore.Qt.LeftButton, pos=local_target)
assert events
order = events[-1]
assert order[0] != 'panel-0'
proxies = [GeomProxy(board._size_key(board._cards[panel_id])) for panel_id in order]
pack(proxies, metrics, board.width())
for panel_id, proxy in zip(order, proxies):
    assert board._cards[panel_id].geometry().getRect()[:2] == (proxy.col, proxy.row)
"""
    )


def test_board_qtest_drag_has_free_positions_without_ghost_or_live_reflow() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.board import BoardMetrics
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['test'])
metrics = BoardMetrics(10, lambda size: (100, 80))
cards = tuple(PanelCardView(f'panel-{index}') for index in range(4))
board = ConsoleBoardView(metrics=metrics); board.resize(350, 260); board.set_cards(cards); board.show(); app.processEvents()
before = tuple(card.geometry().getRect() for card in cards[1:])
assert not hasattr(board, '_ghost')

def drag_to(card, board_point):
    start = card.geometry().center()
    local_target = card.mapFrom(board, QtCore.QPoint(*board_point))
    QtTest.QTest.mousePress(card, QtCore.Qt.LeftButton, pos=QtCore.QPoint(18, 18))
    QtTest.QTest.mouseMove(card, local_target)
    app.processEvents()
    return local_target

first_target = drag_to(cards[0], (275, 150))
assert cards[0].pos() == QtCore.QPoint(257, 132)
assert tuple(card.geometry().getRect() for card in cards[1:]) == before
QtTest.QTest.mouseRelease(cards[0], QtCore.Qt.LeftButton, pos=first_target)
app.processEvents()
assert board._order != ('panel-0', 'panel-1', 'panel-2', 'panel-3')

second_target = drag_to(cards[0], (35, 215))
assert cards[0].pos() == QtCore.QPoint(17, 197)
QtTest.QTest.mouseRelease(cards[0], QtCore.Qt.LeftButton, pos=second_target)
app.processEvents()
assert board._order == ('panel-1', 'panel-2', 'panel-3', 'panel-0')
"""
    )


def test_board_drag_raises_the_active_card_above_an_overlapping_sibling() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.board import BoardMetrics
from zlc_ui.console import ConsoleBoardView, PanelCardView
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['drag-stack'])
metrics = BoardMetrics(10, lambda size: (100, 80))
first, second = PanelCardView('first'), PanelCardView('second')
board = ConsoleBoardView(metrics=metrics); board.resize(240, 180); board.set_cards((first, second)); board.show(); app.processEvents()
first.move(second.pos())
second.raise_()
def top_card_at(point):
    widget = board.childAt(point)
    while widget is not None and widget.parentWidget() is not board:
        widget = widget.parentWidget()
    return widget
assert top_card_at(first.geometry().center()) is second
QtTest.QTest.mousePress(first, QtCore.Qt.LeftButton, pos=QtCore.QPoint(18, 18))
app.processEvents()
assert top_card_at(first.geometry().center()) is first
QtTest.QTest.mouseRelease(first, QtCore.Qt.LeftButton, pos=QtCore.QPoint(18, 18))
"""
    )


def test_board_reuses_cards_and_reflows_on_resize() -> None:
    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.board import BoardMetrics
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['test'])
metrics = BoardMetrics(10, lambda size: (100, 80))
cards = tuple(PanelCardView(f'panel-{index}') for index in range(3))
board = ConsoleBoardView(metrics=metrics); board.resize(260, 220); board.set_cards(cards); board.show(); app.processEvents()
identities = tuple(board._cards.values())
assert cards[0].geometry().x() == 10 and cards[1].geometry().x() == 120
board.set_cards(cards); app.processEvents()
assert tuple(board._cards.values()) == identities
board.resize(120, 400); app.processEvents()
assert all(card.geometry().x() == 10 for card in cards)
"""
    )


def test_logic_row_construct_and_setters() -> None:
    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import LogicRowView
app = ensure_qt_app(['test'])
row = LogicRowView('Processor', 'processor')
assert not row.start_button.isEnabled()
assert not row.stop_button.isEnabled()
row.set_commands(can_start=True, can_stop=True)
row.set_state('running', 'processing fake input')
row.set_publishes((('out', '1', 'fake output'),))
assert row.status_label.text() == 'processing fake input'
assert row.stop_button.isEnabled()
assert row.start_button.isEnabled()
assert row.start_button.text() == 'Restart'
assert 'out' in row.publishes_label.text()
"""
    )


def test_logic_row_qtest_signal_payloads() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import LogicRowView
app = ensure_qt_app(['test'])
row = LogicRowView('Processor', 'processor'); row.show(); app.processEvents()
events = []
row.start_requested.connect(lambda: events.append('start'))
row.edit_requested.connect(lambda: events.append('edit'))
row.set_commands(can_start=True, can_stop=False)
QtTest.QTest.mouseClick(row.start_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(row.edit_button, QtCore.Qt.LeftButton)
assert events == ['start', 'edit']
"""
    )


def test_logic_editor_is_a_live_closable_draft_projection() -> None:
    _run_qt(
        """
import zou_lab_control_v2
import zlc_ui.console.logic_editor_view as tested_module
print(tested_module.__file__)
from PyQt5 import QtCore, QtTest
from zlc_ui.console import TaskConsoleHandle, TaskConsoleView
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['logic-editor'])
view = TaskConsoleView()
handle = TaskConsoleHandle(None, view)
handle.add_logic_row('camera-1', 'measurement')
projection = {
    'node_id': 'camera-1',
    'api_name': 'camera_measurement',
    'kind': 'measurement',
    'form_spec': FormSpec((FormFieldProps(key='repeat', kind='int', label='Repeat', default=0, minimum=0),)),
    'form_values': {'repeat': 0},
    'artifact_form_spec': FormSpec((
        FormFieldProps(
            key='calibration_path', kind='path', label='Calibration Path',
            required=True, file_filter='JSON files (*.json)', base_dir='C:/data',
        ),
    )),
    'artifact_values': {'calibration_path': 'C:/data/calibration.json'},
    'artifact_results': ({
        'name': 'artifact_path', 'contract_id': 'calibration.readout.v1',
        'path': 'C:/data/calibration-2.json',
    },),
    'source_required': True,
    'source_signal': '',
    'source_options': ('camera-1.frames',),
    'source_labels': {'camera-1.frames': 'frames  [1 × 1 × (3×96×128)]'},
    'device_keys': {'camera': 'camera'},
    'device_options': {'camera': ('camera', 'mot_camera')},
    'running': False,
    'pending': False,
    'can_start': False,
    'can_stop': False,
    'issues': ('select a compatible source',),
    'error': '',
}
patches = []
def accept_and_refresh(node_id, patch):
    patches.append((node_id, patch))
    if 'values' in patch:
        projection['form_values'].update(patch['values'])
    if 'artifact_inputs' in patch:
        projection['artifact_values'].update(patch['artifact_inputs'])
    if 'source_signal' in patch:
        projection['source_signal'] = patch['source_signal']
    if 'device_keys' in patch:
        projection['device_keys'].update(patch['device_keys'])
    handle.update_logic_editor(node_id, projection)
handle.logic_draft_changed.connect(accept_and_refresh)
handle.open_logic_editor('camera-1', projection)
editor = handle._logic_editors['camera-1']
assert editor.source_combo.itemText(1) == 'frames  [1 × 1 × (3×96×128)]'
handle.set_logic_commands('camera-1', can_start=False, can_stop=False)
view.show()
app.processEvents()
assert view.tabs.count() == 3
assert view.tabs.currentWidget() is editor
assert not editor.start_button.isEnabled()
assert not editor.stop_button.isEnabled()
assert not handle._rows['camera-1'].start_button.isEnabled()
editor.form.widget_for('repeat').setValue(3)
artifact_picker = editor.artifact_form.widget_for('calibration_path')
assert artifact_picker.browse.text() == 'Browse…'
artifact_picker.setText('C:/data/manual.json')
def pick_from_popup(combo, value):
    index = combo.findData(value)
    assert index >= 0
    combo.showPopup()
    app.processEvents()
    model_index = combo.model().index(index, combo.modelColumn())
    rect = combo.view().visualRect(model_index)
    assert rect.isValid() and not rect.isEmpty()
    QtTest.QTest.mouseClick(
        combo.view().viewport(), QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier, rect.center(),
    )
source_combo = editor.source_combo
source_model_events = []
source_combo.model().rowsRemoved.connect(
    lambda *_args: source_model_events.append('removed')
)
source_combo.model().rowsInserted.connect(
    lambda *_args: source_model_events.append('inserted')
)
pick_from_popup(source_combo, 'camera-1.frames')
assert editor.source_combo is source_combo
assert not source_model_events
camera_combo = editor._device_combos['camera']
camera_model_events = []
camera_combo.model().rowsRemoved.connect(
    lambda *_args: camera_model_events.append('removed')
)
camera_combo.model().rowsInserted.connect(
    lambda *_args: camera_model_events.append('inserted')
)
pick_from_popup(camera_combo, 'mot_camera')
assert editor._device_combos['camera'] is camera_combo
assert not camera_model_events
app.processEvents()
assert ('camera-1', {'values': {'repeat': 3}}) in patches
assert ('camera-1', {'artifact_inputs': {'calibration_path': 'C:/data/manual.json'}}) in patches
assert ('camera-1', {'source_signal': 'camera-1.frames'}) in patches
assert ('camera-1', {'device_keys': {'camera': 'mot_camera'}}) in patches
readout = editor._artifact_result_readouts['artifact_path']
assert readout.isReadOnly()
assert readout.text() == 'C:/data/calibration-2.json'

patches.clear()
running = dict(
    projection,
    form_values={'repeat': 3},
    running=True,
    can_start=True,
    can_stop=True,
    issues=(),
)
handle.set_logic_commands('camera-1', can_start=True, can_stop=True)
handle.update_logic_editor('camera-1', running)
assert not patches, patches
assert editor.start_button.text() == 'Restart'
assert editor.start_button.isEnabled()
assert editor.stop_button.isEnabled()
view.editor_close_requested.emit(editor)
app.processEvents()
assert view.tabs.count() == 2
assert 'camera-1' not in handle._logic_editors
assert handle.logic_row_ids() == ('camera-1',)
"""
    )


def test_panel_editor_and_setting_are_views_of_the_same_projection() -> None:
    _run_qt(
        """
import zou_lab_control_v2
import zlc_ui.console.panel_editor_view as tested_module
print(tested_module.__file__)
from PyQt5 import QtWidgets
from zlc_ui.console import TaskConsoleHandle, TaskConsoleView
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['panel-editor'])
view = TaskConsoleView()
handle = TaskConsoleHandle(None, view)
handle.set_panel_intervals((100, 200, 400, 800))
handle.add_panel('panel-1', 'Camera')
groups = (('camera', (('frames', '@logic/cm/frames'),)),)
handle.set_panel_signal_choices(
    'panel-1', groups, current='@logic/cm/frames'
)
state = {
    'signal': '@logic/cm/frames', 'kind': 'image', 'size': '2x2',
    'interval_ms': 100, 'title': 'Camera',
    'semantic': {}, 'display': {}, 'fit': {}, 'site_overlay': 'off',
}
surface = {
    'semantic': ({
        'key': 'x', 'label': 'X axis', 'kind': 'choice', 'value': 'sensor_x',
        'allow_none': False,
        'choices': (('Sensor X', 'sensor_x'), ('Point row', 'point_row')),
        'minimum': None, 'maximum': None, 'step': None,
    },),
    'display': (
        {'key': 'colormap', 'label': 'Colormap', 'kind': 'choice',
         'value': 'viridis', 'allow_none': False,
         'choices': (('Viridis', 'viridis'), ('Magma', 'magma')),
         'minimum': None, 'maximum': None, 'step': None},
        {'key': 'show_colorbar', 'label': 'Colorbar', 'kind': 'boolean',
         'value': True, 'allow_none': False, 'choices': (),
         'minimum': None, 'maximum': None, 'step': None},
        {'key': 'interpolation', 'label': 'Interpolation', 'kind': 'choice',
         'value': 'nearest', 'allow_none': False,
         'choices': (('Nearest', 'nearest'), ('Bilinear', 'bilinear')),
         'minimum': None, 'maximum': None, 'step': None},
    ),
    'fit': ({
        'key': 'model', 'label': 'Fit model', 'kind': 'choice', 'value': None,
        'allow_none': True,
        'choices': (('2-D Gaussian', 'anisotropic_gaussian_center'),),
        'minimum': None, 'maximum': None, 'step': None,
    },),
    'site_overlay': {
        'key': 'site_overlay', 'label': 'Site overlay', 'kind': 'choice',
        'value': 'off', 'allow_none': False,
        'choices': (('Off', 'off'), ('Centers', 'centers'), ('Occupancy', 'occupancy')),
        'minimum': None, 'maximum': None, 'step': None,
    },
}
handle.set_panel_projection('panel-1', state, surface)
card = handle._cards['panel-1']
card._open_settings()
assert card._settings_form.read_all()['kind'] == 'image'
assert not card._settings_form.widget_for('kind').isEnabled()
assert card._settings_form.read_all()['display__show_colorbar'] is True
assert card._settings_form.read_all()['display__colormap'] == 'viridis'
assert 'display__interpolation' in card._settings_form.spec.keys
interval_combo = card._settings_form.widget_for('interval_ms')
assert isinstance(interval_combo, QtWidgets.QComboBox)
assert tuple(interval_combo.itemData(index) for index in range(interval_combo.count())) == (
    100, 200, 400, 800,
)
facet_state = dict(state, kind='facet_grid', cell_kind='curve', title='Site grid')
handle.set_panel_projection('panel-1', facet_state, surface)
assert card._settings_form.read_all()['cell_kind'] == 'curve'
assert not card._settings_form.widget_for('cell_kind').isEnabled()
handle.set_panel_projection('panel-1', state, surface)

producer = {
    'node_id': 'cm', 'api_name': 'camera_measurement', 'kind': 'measurement',
    'form_spec': FormSpec((
        FormFieldProps('repeat', 'int', 'Repeat', default=0, minimum=0),
    )),
    'form_values': {'repeat': 0},
    'source_required': False, 'source_signal': '', 'source_options': (),
    'device_keys': {'camera': 'camera'},
    'device_options': {'camera': ('camera', 'mot_camera')},
    'running': True, 'pending': False, 'error': '',
}
projection = {
    'panel_id': 'panel-1', 'state': state, 'signal_options': groups,
    'interval_choices': (100, 200, 400, 800),
    'parameter_surface': surface,
    'kind_read_only': True, 'frozen_signal': '@logic/cm/frames',
    'frozen_publication': object(), 'frozen_snapshot': object(),
    'stale': False, 'producer_node_id': 'cm', 'producer_logic': producer,
}
events = []
handle.panel_state_changed.connect(
    lambda panel_id, patch: events.append(('state', panel_id, patch))
)
handle.logic_draft_changed.connect(
    lambda node_id, patch: events.append(('draft', node_id, patch))
)
handle.panel_snapshot_refresh_requested.connect(
    lambda panel_id: events.append(('refresh', panel_id))
)
handle.panel_producer_restart_requested.connect(
    lambda panel_id: events.append(('restart', panel_id))
)
handle.panel_save_figure_requested.connect(
    lambda panel_id: events.append(('save', panel_id))
)
handle.panel_editor_closed.connect(
    lambda panel_id: events.append(('closed', panel_id))
)
handle.open_panel_editor('panel-1', projection)
editor = handle._panel_editors['panel-1']
class _PlotHost:
    def __init__(self, text):
        self.widget = QtWidgets.QLabel(text)
        self.qt_widget_calls = 0
    def qt_widget(self):
        self.qt_widget_calls += 1
        return self.widget
first_host = _PlotHost('frozen plot one')
second_host = _PlotHost('frozen plot two')
handle.show_panel_editor('panel-1', first_host)
assert editor._surface is first_host.widget
assert first_host.qt_widget_calls == 1
assert first_host.widget.parentWidget() is editor.surface_holder
handle.show_panel_editor('panel-1', second_host)
assert editor._surface is second_host.widget
assert first_host.widget.parentWidget() is None
assert second_host.widget.parentWidget() is editor.surface_holder
assert view.tabs.count() == 3 and view.tabs.currentWidget() is editor
assert editor.kind_label.text() == 'image'
facet_projection = dict(projection, state=facet_state)
assert handle.update_panel_editor('panel-1', facet_projection)
assert editor.kind_label.text() == 'facet grid · curve cells'
assert handle.update_panel_editor('panel-1', projection)
assert not editor._producer_editor.start_button.isVisible()
assert editor.parameter_forms['semantic'].spec.keys == ('x',)
assert editor.parameter_forms['display'].spec.keys == (
    'colormap', 'show_colorbar', 'interpolation'
)
assert editor.parameter_forms['fit'].spec.keys == ('model',)
semantic_combo = editor.parameter_forms['semantic'].widget_for('x')
semantic_combo.setCurrentIndex(1)
semantic_combo.activated.emit(1)
display_combo = editor.parameter_forms['display'].widget_for('colormap')
display_combo.setCurrentIndex(1)
display_combo.activated.emit(1)
fit_combo = editor.parameter_forms['fit'].widget_for('model')
fit_combo.setCurrentIndex(1)
fit_combo.activated.emit(1)
editor_interval = editor.panel_form.widget_for('interval_ms')
assert isinstance(editor_interval, QtWidgets.QComboBox)
editor_interval.setCurrentIndex(editor_interval.findData(800))
editor_interval.activated.emit(editor_interval.currentIndex())
editor._producer_editor.form.widget_for('repeat').setValue(2)
editor.refresh_button.click()
editor.producer_restart_button.click()
editor.save_button.click()
app.processEvents()
assert ('state', 'panel-1', {'interval_ms': 800}) in events
assert ('state', 'panel-1', {'semantic': {'x': 'point_row'}}) in events
assert ('state', 'panel-1', {'display': {'colormap': 'magma'}}) in events
assert ('state', 'panel-1', {
    'fit': {'model': 'anisotropic_gaussian_center'}
}) in events
assert ('draft', 'cm', {'values': {'repeat': 2}}) in events
assert ('refresh', 'panel-1') in events
assert ('restart', 'panel-1') in events
assert ('save', 'panel-1') in events

changed = dict(state, title='Retitled', interval_ms=800)
handle.set_panel_projection('panel-1', changed, surface)
handle.update_panel_editor('panel-1', dict(projection, state=changed, stale=True))
assert card.title() == 'Retitled'
assert editor.panel_form.read_all()['title'] == 'Retitled'
assert not editor.save_button.isEnabled()
view.editor_close_requested.emit(editor)
app.processEvents()
assert view.tabs.count() == 2 and 'panel-1' not in handle._panel_editors
assert ('closed', 'panel-1') in events
assert second_host.widget.parentWidget() is None
"""
    )


def test_status_strip_construct_and_priority() -> None:
    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import StatusStrip
app = ensure_qt_app(['test'])
strip = StatusStrip()
strip.show_status('warning text', 'warning')
assert strip.current_severity == 'warning'
strip.show_status('task text', 'task')
assert strip.current_severity == 'task'
strip.show_status('error text', 'error')
assert strip.current_severity == 'error'
assert strip.text() == 'error text'
# And the next thing that happens is what the operator sees.  An error used to
# latch forever, so every later success was written into a slot the strip
# would not display: fix the problem, press the button, see the old error.
strip.show_status('saved 3 panel(s)', 'task')
assert strip.current_severity == 'task'
assert strip.text() == 'saved 3 panel(s)'
"""
    )


def test_status_strip_idle_fallback() -> None:
    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import StatusStrip
app = ensure_qt_app(['test'])
strip = StatusStrip()
strip.show_status('task text', 'task')
strip.show_status('', 'task')
strip.show_status('idle text', 'idle')
assert strip.current_severity == 'idle'
assert strip.text() == 'idle text'
"""
    )


def test_task_console_construct_and_setters() -> None:
    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import LogicRowView, PanelCardView, TaskConsoleView
from zlc_ui.fluent import scaled_px
app = ensure_qt_app(['test'])
view = TaskConsoleView()
card = PanelCardView('panel-1')
row = LogicRowView('Logic')
view.set_cards((card,))
view.set_logic_rows((row,))
view.set_summary('one fake card')
view.show_status('ready', 'idle')
view.set_panel_kinds((('image', 'Image'),))
view.resize(1500, 422); view.show(); app.processEvents()
assert view.summary_label.text() == 'one fake card'
assert view.status_strip.current_severity == 'idle'
assert view.board._cards['panel-1'] is card
assert view.selectors_switch.width() > 0
# The v1 header is one semantic row: identity on the left, telemetry as the
# only flexible middle cell, and the command cluster pinned to the right.
# A wrapping row top-aligned every sizeHint and left the spare width after the
# final button, making both the grey summary and the action cluster visibly
# wrong while every control still existed.
header = view.summary_label.parentWidget()
layout = header.layout()
right_edge = header.rect().right() - layout.contentsMargins().right()
assert view.load_layout_button.geometry().right() == right_edge
center_y = header.rect().center().y()
for widget in (
    view.status_dot,
    view.name_edit,
    view.summary_label,
    view.kind_combo,
    view.add_panel_button,
    view.selectors_switch,
    view.pause_switch,
    view.save_screenshot_button,
    view.save_layout_button,
    view.load_layout_button,
):
    assert abs(widget.geometry().center().y() - center_y) <= 1
"""
    )


def test_task_console_reuses_logic_rows_after_repeated_projection() -> None:
    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import LogicRowView, TaskConsoleView
app = ensure_qt_app(['test'])
view = TaskConsoleView()
row_a = LogicRowView('A')
row_b = LogicRowView('B')
view.set_logic_rows((row_a, row_b))
view.set_logic_rows((row_a, row_b))
app.processEvents()
row_a.set_state('running', 'still usable')
row_b.set_state('idle', 'also usable')
assert row_a.parentWidget() is view.logic_body
assert row_b.parentWidget() is view.logic_body
assert view._logic_rows == (row_a, row_b)
assert view.logic_layout.count() == 4  # hidden hint + two rows + permanent stretch
assert not view.logic_hint.isVisible()
assert row_a.status_label.text() == 'still usable'
assert row_b.status_label.text() == 'also usable'
"""
    )


def test_task_console_qtest_signal_payloads() -> None:
    _run_qt(
        """import zou_lab_control_v2
from zlc_ui.console import task_console_view as tested_module
print(tested_module.__file__)
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import TaskConsoleHandle
app = ensure_qt_app(['test'])
view = tested_module.TaskConsoleView()
handle = TaskConsoleHandle(None, view)
view.show(); app.processEvents()
events = []
handle.set_panel_kinds((('curve', 'Curve'), ('image', '2D image')), 'image')
handle.set_logic_kinds((
    ('calibration', 'task', 'nothing', ''),
    ('occupancy', 'processor', 'occupied', ''),
    ('camera_measurement', 'measurement', 'frames', ''),
))
handle.add_panel_requested.connect(lambda kind: events.append(('panel', kind)))
handle.add_logic_requested.connect(lambda api_name: events.append(('logic', api_name)))
handle.pause_toggled.connect(lambda value: events.append(('pause', value)))
handle.save_screenshot_requested.connect(lambda: events.append(('screenshot',)))
handle.save_layout_requested.connect(lambda: events.append(('layout',)))
handle.stop_task_requested.connect(lambda: events.append(('stop-task',)))
assert view.kind_combo.count() == 5
assert view.kind_combo.itemData(0) == ('plot', 'curve')
assert view.kind_combo.itemData(1) == ('plot', 'image')
assert view.kind_combo.itemData(2) == ('logic', 'camera_measurement')
assert view.kind_combo.itemData(3) == ('logic', 'occupancy')
assert view.kind_combo.itemData(4) == ('logic', 'calibration')
assert view.kind_combo.itemText(0) == 'Plot: Curve'
assert view.kind_combo.itemText(1) == 'Plot: 2D image'
assert view.kind_combo.itemText(2) == 'Measurement: Camera Measurement'
assert view.kind_combo.itemText(3) == 'Processor: Occupancy'
assert view.kind_combo.itemText(4) == 'Task: Calibration'
assert view.kind_combo.currentData() == ('plot', 'image')
assert not hasattr(view, 'add_logic_button')
QtTest.QTest.mouseClick(view.add_panel_button, QtCore.Qt.LeftButton)
view.kind_combo.setCurrentIndex(2)
QtTest.QTest.mouseClick(view.add_panel_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(view.pause_switch, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(view.save_screenshot_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(view.save_layout_button, QtCore.Qt.LeftButton)
assert ('panel', 'image') in events
assert ('logic', 'camera_measurement') in events
assert ('pause', True) in events
assert ('screenshot',) in events
assert ('layout',) in events

# A running Task owns the console through one status-strip command surface.
# Drive the actual widgets: disabled controls must not emit around the
# presenter gate, while Stop task must still emit its one dedicated intent.
handle.add_panel('task-preview', 'Capture preview')
handle.add_logic_row('calibration', 'task')
handle.add_logic_row('camera', 'measurement')
handle.set_logic_commands('calibration', can_start=True, can_stop=False)
handle.set_logic_commands('camera', can_start=True, can_stop=False)
preview = handle._cards['task-preview']
task_row = handle._rows['calibration']
camera_row = handle._rows['camera']
handle.set_task_takeover(True)
assert view.status_strip.action_button is not None
assert view.status_strip.action_button.isVisible()
assert not view.name_edit.isEnabled()
assert not view.kind_combo.isEnabled()
assert not view.add_panel_button.isEnabled()
assert not view.save_layout_button.isEnabled()
assert not view.load_layout_button.isEnabled()
assert view.selectors_switch.isEnabled()
assert view.pause_switch.isEnabled()
assert view.save_screenshot_button.isEnabled()
assert not preview.settings_button.isEnabled()
for row in (task_row, camera_row):
    assert not row.start_button.isEnabled()
    assert not row.stop_button.isEnabled()
    assert not row.edit_button.isEnabled()
    assert not row.remove_button.isEnabled()
assert not task_row.stop_button.isVisible()
before = list(events)
QtTest.QTest.mouseClick(view.add_panel_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(camera_row.start_button, QtCore.Qt.LeftButton)
assert events == before
QtTest.QTest.mouseClick(view.status_strip.action_button, QtCore.Qt.LeftButton)
assert events[-1] == ('stop-task',)

handle.set_task_takeover(False)
assert not view.status_strip.action_button.isVisible()
assert view.name_edit.isEnabled()
assert view.kind_combo.isEnabled()
assert view.add_panel_button.isEnabled()
assert preview.settings_button.isEnabled()
assert camera_row.start_button.isEnabled()
assert camera_row.edit_button.isEnabled()
assert camera_row.remove_button.isEnabled()

"""
    )


def test_pause_is_reversible_and_says_which_way_it_goes() -> None:
    """A one-way pause is a stopped console with no way back.

    The button carries the command and its label carries the state, so it must
    ask for the state it is NOT in -- and the presenter, which owns the answer,
    is what tells it which that is.
    """

    _run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import TaskConsoleView
app = ensure_qt_app(['pause'])
view = TaskConsoleView(); view.show(); app.processEvents()
asked = []
view.pause_toggled.connect(asked.append)
QtTest.QTest.mouseClick(view.pause_switch, QtCore.Qt.LeftButton)
assert asked == [True], asked
assert view.pause_switch.text() == 'Pause', 'the label moved before the presenter agreed'
view.set_paused(True)
assert view.pause_switch.text() == 'Resume'
QtTest.QTest.mouseClick(view.pause_switch, QtCore.Qt.LeftButton)
assert asked == [True, False], asked
"""
    )


def test_the_signal_chooser_lists_rows_under_their_producer() -> None:
    """It is asked for at the moment it is needed, so it can be readable.

    A picker squeezed into the v1 header collapsed to an ellipsis and clipped
    the control beside it; a chooser you cannot read is not a chooser.
    """

    _run_qt(
        """
from PyQt5 import QtCore
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import SignalChooser
app = ensure_qt_app(['chooser'])
rows = (
    ('@logic/cm/frames', 'frames', 'live', 'cm', ''),
    ('@logic/panel-1/roi_value', 'roi_value', 'finished', 'panel-1', '@logic/cm/frames'),
)
dialog = SignalChooser(rows)
texts = [dialog.list.item(i).text() for i in range(dialog.list.count())]
assert texts == ['cm', '    frames', 'panel-1', '    roi_value  (finished)'], texts

# A producer heading is a heading, not a choice.
assert not dialog.list.item(0).flags() & QtCore.Qt.ItemIsSelectable
assert dialog.chosen() == '@logic/cm/frames', dialog.chosen()

dialog.list.setCurrentRow(3)
assert dialog.chosen() == '@logic/panel-1/roi_value'
assert 'cut from @logic/cm/frames' in dialog.list.item(3).toolTip()
assert dialog.accept_button.isEnabled()
"""
    )


def test_the_signal_chooser_cannot_accept_an_empty_offer() -> None:
    """Nothing has published: say so, and refuse to pretend otherwise."""

    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import SignalChooser
app = ensure_qt_app(['chooser-empty'])
dialog = SignalChooser(())
assert dialog.chosen() is None
assert not dialog.accept_button.isEnabled()
"""
    )


def test_task_console_acceptance_launcher_keeps_v1_target_and_left_anchored_empty_status() -> None:
    _run_qt(
        """
from PyQt5 import QtCore
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import TaskConsoleView
from zlc_ui.fluent import WINDOW_SCREEN_FRACTION, launch_fluent_window, screen_fit_window_size
app = ensure_qt_app(['acceptance'])
body = TaskConsoleView()
window = launch_fluent_window(body, title='TaskConsole@Zou lab', fixed_size=False)
app.processEvents()
target = screen_fit_window_size(WINDOW_SCREEN_FRACTION)
assert window.size() == target, (window.size().width(), window.size().height(), target.width(), target.height())
dot_left = body.status_strip.dot.geometry().left()
assert dot_left == body.status_strip.layout().contentsMargins().left(), (dot_left, body.status_strip.layout().contentsMargins().left())
window.close()
"""
    )


def test_a_window_hosting_a_close_guarding_body_can_actually_be_closed() -> None:
    """The X did nothing at all.

    Several bodies refuse their own close and raise close_requested so a host
    can tear down first.  The host half of that handshake lived nowhere, so an
    application that did not reinvent it produced a window that could not be
    closed -- the body ignored the event and nothing answered the request.
    """

    _run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import launch_fluent_window
from zlc_ui.pulse.editor_view import PulseEditorView
app = ensure_qt_app(['close-handshake'])

window = launch_fluent_window(PulseEditorView, title='PulseGUI@Zou lab')
body = window.findChild(PulseEditorView)
assert body is not None and window.isVisible()

# The body asking to close closes the window...
torn = []
body.close_requested.connect(lambda: torn.append('asked'))
body.close_requested.emit()
app.processEvents()
assert not window.isVisible(), 'the body asked to close and the window stayed'

# ...and a window closing lets the body commit its own close.
second = launch_fluent_window(PulseEditorView, title='PulseGUI@Zou lab')
inner = second.findChild(PulseEditorView)
second.close()
app.processEvents()
assert not second.isVisible()
assert inner._closing, 'the body never got to run its own teardown'
"""
    )


def test_the_launcher_leaves_a_body_without_the_protocol_alone() -> None:
    """Binding must not invent a close guard for a widget that has none."""

    _run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import launch_fluent_window
app = ensure_qt_app(['plain-body'])
window = launch_fluent_window(lambda: QtWidgets.QLabel('plain'), title='Plain')
assert window.isVisible()
window.close()
app.processEvents()
assert not window.isVisible()
"""
    )


def test_the_connection_control_is_one_complete_presenter_projection() -> None:

    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse.models import ConnectionChoiceVM, ConnectionVM
from zlc_ui.pulse.schedule_view import PulseScheduleView
app = ensure_qt_app(['endpoint'])
view = PulseScheduleView(); view.show(); app.processEvents()

choices = (
    ConnectionChoiceVM('Virtual (sim)', 'virtual'),
    ConnectionChoiceVM('Remote server', 'remote', endpoint_editable=True),
    ConnectionChoiceVM('Offline (edit only)', 'offline'),
)
view.set_connection(ConnectionVM(choices, 'offline', '127.0.0.1:18861', 'not connected'))
assert view.connection_endpoint.text() == '127.0.0.1:18861'
assert 'host:port' in view.connection_endpoint.toolTip()
assert not view.connection_endpoint.isEnabled()

view.set_connection(ConnectionVM(choices, 'remote', '', 'remote selected'))
assert view.connection_endpoint.text() == ''
assert view.connection_endpoint.isEnabled()

given = (ConnectionChoiceVM('Experiment session', 'given'),)
view.set_connection(ConnectionVM(given, 'given', '', 'connected', locked=True))
assert view.connection_combo.currentText() == 'Experiment session'
assert not view.connection_combo.isEnabled()
assert not view.connection_button.isEnabled()
assert not view.connection_endpoint.isEnabled()
"""
    )


def test_a_schedule_rebuild_keeps_the_operator_where_they_were() -> None:
    """Every update tore the cards out of their layout and reset the scroll.

    So pressing On Pulse -- or editing anything at all -- threw the board back
    to the first period and the leftmost time, losing the place of anyone
    working on period nine.
    """

    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse.schedule_view import PulseScheduleView
from zlc_ui.pulse.models import FieldVM, PeriodVM, PortRowVM, ScheduleVM
app = ensure_qt_app(['scroll'])
view = PulseScheduleView(); view.resize(900, 500); view.show()
for _ in range(5): app.processEvents()

def vm(revision, name='p'):
    ports = tuple(PortRowVM(key=f'ch{i}', kind='digital', label=f'ch{i}', endpoint_text=f'P{i}') for i in range(20))
    periods = tuple(
        PeriodVM(period_id=f'period{i}', name=f'{name}{i}', duration=FieldVM(text='1'), unit='us',
                 digital=tuple((p.key, False) for p in ports))
        for i in range(12)
    )
    return ScheduleVM(document_generation=0, revision=revision, document_name='probe',
                      clock_text='20 ns/tick', total_text='12 us', total_tooltip='', period_count=12,
                      visible_text='20/20', summary_text='', ports=ports, periods=periods)

view.set_schedule(vm(1))
for _ in range(5): app.processEvents()
bars = [view.timeline_scroll.horizontalScrollBar(), view.timeline_scroll.verticalScrollBar(),
        view.left_scroll.verticalScrollBar()]
for bar in bars: bar.setValue(bar.maximum())
before = [bar.value() for bar in bars]
assert all(before), before

view.set_schedule(vm(2, name='q'))
for _ in range(5): app.processEvents()
assert [bar.value() for bar in bars] == before, [bar.value() for bar in bars]

# A rebuild that shortens the pulse cannot restore past the new end.
def short(revision):
    full = vm(revision)
    return ScheduleVM(document_generation=0, revision=revision, document_name='probe',
                      clock_text=full.clock_text, total_text='1 us', total_tooltip='', period_count=1,
                      visible_text=full.visible_text, summary_text='', ports=full.ports,
                      periods=full.periods[:1])

view.set_schedule(short(3))
for _ in range(5): app.processEvents()
assert all(bar.value() <= bar.maximum() for bar in bars)
"""
    )
