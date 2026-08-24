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
        "" if environment.get("ZLC_TEST_INSTALLED") == "1"
        else os.pathsep.join((str(REPO_ROOT), str(SRC)))
    )
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
import zou_lab_control
print(zou_lab_control.__file__)
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentTreeComboBox
from zlc_ui.console import panel_card_view as tested_module
print(tested_module.__file__)
PanelCardView = tested_module.PanelCardView
app = ensure_qt_app(['test'])
card = PanelCardView('panel-1', 'Card')
card.set_size_choices(('1x2', '2x2', '1x4'), '2x2')
card.set_signal_choices((('source', (('Temperature', 'temperature'),)),))
state = {
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 100,
    'title': 'Card', 'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '',
}
surface_projection = {
    'semantic': (), 'display': (), 'fit': (),
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
# The status mark: a dot overlaying the title strip beside Setting, so the
# card body reserves no status row and the message rides the tooltip.
from zlc_ui.fluent import GREY, RED
card.set_status('ready', error=False)
assert not card.status_dot.isHidden(), 'a status did not surface its dot'
assert card.status_dot.toolTip() == 'ready'
assert GREY in card.status_dot.styleSheet()
card.set_status('the renderer refused this frame', error=True)
assert not card.status_dot.isHidden()
assert card.status_dot.toolTip() == 'the renderer refused this frame'
assert RED in card.status_dot.styleSheet()
# Geometry is only true once the layout has run: the dot and the Setting
# button are items in the title strip's row, not free-floating overlays.
card.show(); app.processEvents()
settings_x = card.settings_button.mapTo(card._title_band, QtCore.QPoint()).x()
assert card.status_dot.x() < settings_x, 'the dot must sit beside Setting'
assert card.status_dot.y() >= 0 and card.status_dot.y() < card._title_band.height()
card.set_status('', error=False)
assert card.status_dot.isHidden(), 'clearing the status must clear the mark'
card.set_status('ready', error=False)
assert card.panel_id == 'panel-1'
signal_field = next(field for field in card._form_spec().fields if field.key == 'signal')
assert signal_field.kind == 'keyed_choice'
card._open_settings()
signal_picker = card._settings_form.widget_for('signal')
assert isinstance(signal_picker, FluentTreeComboBox)
assert signal_picker.current_choice_key() == ''
assert signal_picker.model().item(0).text() == 'source'
picked = []
card.state_changed.connect(picked.append)
card.show()
signal_picker.showPopup()
source_index = signal_picker.model().index(0, 0)
signal_picker.view().expand(source_index)
app.processEvents()
leaf_index = signal_picker.model().index(0, 0, source_index)
QtTest.QTest.mouseClick(
    signal_picker.view().viewport(), QtCore.Qt.LeftButton,
    pos=signal_picker.view().visualRect(leaf_index).center(),
)
assert {'signal': 'temperature'} in picked
assert surface.parentWidget() is not None
# A bare card is interactive until TaskConsole projects its global switch.
assert surface.interaction is True
card.set_selectors_enabled(True)
assert surface.interaction is True
card.set_selectors_enabled(False)
assert surface.interaction is False
body = QtWidgets.QWidget()
body.setMinimumSize(420, 1200)
card.setParent(body)
card.setGeometry(10, 10, 340, 240)
scroll = QtWidgets.QScrollArea()
scroll.setWidget(body)
scroll.resize(380, 300)
scroll.show()
app.processEvents()
bar = scroll.verticalScrollBar()
assert bar.maximum() > 0
wheel = QtGui.QWheelEvent(
    QtCore.QPointF(surface.rect().center()),
    QtCore.QPointF(surface.mapToGlobal(surface.rect().center())),
    QtCore.QPoint(),
    QtCore.QPoint(0, -240),
    QtCore.Qt.NoButton,
    QtCore.Qt.NoModifier,
    QtCore.Qt.NoScrollPhase,
    False,
)
QtWidgets.QApplication.sendEvent(surface, wheel)
assert bar.value() > 0, 'a wheel the surface ignores must still reach the board scroll'
card.set_surface(None)
assert not card._placeholder.isHidden()
card.set_panel_projection(dict(state, size='1x4'), surface_projection)
assert card.panel_size == '1x4'
try:
    card.set_panel_projection(dict(state, size='not-a-panel-size'), surface_projection)
except ValueError:
    pass
else:
    raise AssertionError('invalid panel size was accepted')
"""
    )


def test_panel_card_qtest_signal_payloads() -> None:
    _run_qt(
        """
import zou_lab_control
print(zou_lab_control.__file__)
from zlc_ui.console import panel_card_view as tested_module
print(tested_module.__file__)
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from zlc_ui.fluent import GREY, RED, FluentPopup
from zlc_ui.qt import ensure_qt_app
PanelCardView = tested_module.PanelCardView
app = ensure_qt_app(['test'])
card = PanelCardView('panel-1', 'Card')
card.set_size_choices(('2x2',), '2x2')
card.set_interval_choices((100, 200, 400, 800), 400)
card.set_panel_projection({
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 400,
    'title': 'Card', 'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '',
}, {
    'semantic': (), 'display': (), 'fit': (),
    'semantic_unavailable': '', 'display_unavailable': '',
    'fit_unavailable': '',
})
card.resize(340, 180); card.show(); app.processEvents()
events = []
card.remove_requested.connect(lambda: events.append(('remove',)))
card.edit_requested.connect(lambda: events.append(('edit',)))
card.state_changed.connect(lambda patch: events.append(('state', patch)))
# The header × is guarded against a stray click: first click turns red, the
# second click inside the platform double-click interval removes the panel.
QtTest.QTest.mouseClick(card.close_button, QtCore.Qt.LeftButton)
assert events == []
assert RED.lower() in card.close_button.styleSheet().lower()
QtTest.QTest.mouseClick(card.close_button, QtCore.Qt.LeftButton)
assert events == [('remove',)]
assert GREY.lower() in card.close_button.styleSheet().lower()
# Edit and the explicit text Remove remain available in Setting as well.
top_levels = {widget for widget in app.topLevelWidgets() if widget.isVisible()}
shown_top_levels = []
class _TopLevelShowSpy(QtCore.QObject):
    def eventFilter(self, watched, event):
        if (
            event.type() == QtCore.QEvent.Show
            and isinstance(watched, QtWidgets.QWidget)
            and watched.isWindow()
        ):
            shown_top_levels.append(type(watched).__name__)
        return False
show_spy = _TopLevelShowSpy()
app.installEventFilter(show_spy)
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
popup = card._settings_popup
assert isinstance(popup, FluentPopup)
assert popup.parentWidget() is card
assert popup.isWindow()
assert popup.windowFlags() & QtCore.Qt.Popup
assert shown_top_levels == ['FluentPopup']
assert {
    widget for widget in app.topLevelWidgets() if widget.isVisible()
} == top_levels | {popup}
assert 240 <= popup.width() <= 320
assert popup.height() <= card.height()
assert popup.mapToGlobal(popup.rect().bottomRight()).y() <= card.mapToGlobal(
    card.rect().bottomRight()
).y()
assert card._settings_scroll.geometry().right() == popup.rect().right()
assert not hasattr(card, 'apply_button')
app.processEvents()
# Whatever the form's length, all of it is reachable inside the popup the
# card clamps: the scrollbar's range is exactly the overflow.
scroll = card._settings_scroll
overflow = max(0, scroll.widget().sizeHint().height() - scroll.viewport().height())
assert scroll.verticalScrollBar().maximum() == overflow
title = card._settings_form.widget_for('title')
title.setText('Card changed')
title.editingFinished.emit()
interval = card._settings_form.widget_for('interval_ms')
interval.setCurrentIndex(interval.findData(800))
interval.activated.emit(interval.currentIndex())
app.processEvents()
assert popup.isVisible()
start = popup.pos()
press = QtCore.QPoint(5, 5)
handle = card._settings_drag_handle
origin = handle.mapToGlobal(press)
target = origin + QtCore.QPoint(24, 18)
QtWidgets.QApplication.sendEvent(
    handle,
    QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, press, origin,
        QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    ),
)
QtWidgets.QApplication.sendEvent(
    handle,
    QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove, press + QtCore.QPoint(24, 18), target,
        QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    ),
)
QtWidgets.QApplication.sendEvent(
    handle,
    QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease, press + QtCore.QPoint(24, 18), target,
        QtCore.Qt.LeftButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
    ),
)
app.processEvents()
assert popup.pos() != start
popup.hide()  # Qt.Popup's outside-click dismissal reaches the same hide path.
app.processEvents()
assert not popup.isVisible()
QtTest.QTest.qWait(300)
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
QtTest.QTest.mouseClick(card.edit_button, QtCore.Qt.LeftButton)
QtTest.QTest.qWait(300)
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
QtTest.QTest.mouseClick(card.remove_button, QtCore.Qt.LeftButton)
assert ('edit',) in events
assert ('remove',) in events
assert any(event[0] == 'state' and event[1]['title'] == 'Card changed' for event in events)
assert ('state', {'interval_ms': 800}) in events
app.removeEventFilter(show_spy)
popup.close(); card.close(); card.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete); app.processEvents()
"""
    )


def test_fluent_popup_retires_with_its_owner_window() -> None:
    _run_qt(
        """
import zou_lab_control
print(zou_lab_control.__file__)
from zlc_ui.fluent import fluent as tested_module
print(tested_module.__file__)
from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui import open_task_console
from zlc_ui.console import PanelCardView
from zlc_ui.qt import ensure_qt_app
FluentPopup = tested_module.FluentPopup
app = ensure_qt_app(['test'])
top_levels = set(app.topLevelWidgets())
console = open_task_console(window_ratio=0.4)
console.set_panel_sizes(('2x2',), '2x2')
console.set_panel_intervals((100, 200, 400, 800), 100)
console.add_panel('panel-1', 'Panel')
console.set_panel_projection('panel-1', {
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 100,
    'title': 'Panel', 'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '',
}, {
    'semantic': (), 'display': (), 'fit': (),
    'semantic_unavailable': '', 'display_unavailable': '',
    'fit_unavailable': '',
})
app.processEvents()
owner = next(
    widget for widget in set(app.topLevelWidgets()) - top_levels
    if isinstance(widget, tested_module.FluentWindow)
)
card = owner.findChild(PanelCardView)
assert card is not None

QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
popup = owner.findChild(FluentPopup)
assert popup is not None
assert popup.isVisible()
assert {
    widget for widget in app.topLevelWidgets() if widget.isVisible()
} == top_levels | {owner, popup}
QtWidgets.QApplication.sendEvent(
    owner, QtCore.QEvent(QtCore.QEvent.WindowDeactivate)
)
app.processEvents()
assert not popup.isVisible(), 'an inactive owner left its popup above another window'

# Owner activation/restoration must never resurrect an explicitly retired popup.
QtWidgets.QApplication.sendEvent(
    owner, QtCore.QEvent(QtCore.QEvent.WindowActivate)
)
owner.showNormal()
app.processEvents()
assert not popup.isVisible()

QtTest.QTest.qWait(300)
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
assert popup.isVisible()
owner.setWindowState(owner.windowState() | QtCore.Qt.WindowMinimized)
app.processEvents()
assert not popup.isVisible(), 'a minimized owner left its popup visible'

owner.showNormal()
app.processEvents()
assert not popup.isVisible()
QtTest.QTest.qWait(300)
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
owner.hide()
app.processEvents()
assert not popup.isVisible(), 'a hidden owner left its popup visible'

owner.show()
app.processEvents()
QtTest.QTest.qWait(300)
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
owner.close()
app.processEvents()
assert not popup.isVisible(), 'a closed owner left its popup visible'
assert not any(
    widget.isVisible() and isinstance(widget, FluentPopup)
    for widget in app.topLevelWidgets()
)
popup.close()
console.deleteLater()
owner.deleteLater()
QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
"""
    )


def test_public_panel_removal_retires_its_popup_and_card() -> None:
    _run_qt(
        """
import zou_lab_control
print(zou_lab_control.__file__)
from PyQt5 import QtCore, sip
from zlc_ui import open_task_console
from zlc_ui.console import board_view as board_module
from zlc_ui.console import handle as handle_module
print(board_module.__file__)
print(handle_module.__file__)
from zlc_ui.qt import ensure_qt_app

app = ensure_qt_app(['panel-remove'])
console = open_task_console(window_ratio=0.4)
console.set_panel_sizes(('2x2',), '2x2')
console.set_panel_intervals((100,), 100)
console.add_panel('panel-1', 'Panel')
console.set_panel_projection('panel-1', {
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 100,
    'title': 'Panel', 'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '',
}, {
    'semantic': (), 'display': (), 'fit': (),
    'semantic_unavailable': '', 'display_unavailable': '',
    'fit_unavailable': '',
})
card = console._cards['panel-1']
card._open_settings()
app.processEvents()
popup = card._settings_popup
assert popup is not None and popup.isVisible()

try:
    console.remove_panel('panel-1')
    assert not popup.isVisible(), 'public remove_panel left Setting visible'
    assert console.panel_ids() == ()
    assert not console._view.board._wired_cards
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.DeferredDelete
    )
    app.processEvents()
    assert sip.isdeleted(card), 'removed card was not deferred-deleted'
finally:
    console.close()
    app.processEvents()
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
# The card's own rectangle is the packer's input, so the expectation is
# derived from it -- a number typed here would be a second answer to the
# question this change exists to give one answer to.
# The rectangle the card RESERVED -- the picture its preset plans plus its
# own strip and margins -- is what the packer is given.
hint = card.size()
board = ConsoleBoardView(metrics=BoardMetrics(12))
board.set_cards((card,))
board.resize(hint.width() + 40, hint.height() + 40); board.show(); app.processEvents()
assert card.geometry().getRect() == (12, 12, hint.width(), hint.height())
assert not board.grab_board().isNull()
"""
    )


def test_board_does_not_capture_ancestor_update_suppression() -> None:
    _run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['ancestor-update-suppression'])

class CountingCard(PanelCardView):
    def __init__(self, panel_id):
        self.paint_count = 0
        super().__init__(panel_id)

    def paintEvent(self, event):
        self.paint_count += 1
        super().paintEvent(event)

host = QtWidgets.QWidget()
board = ConsoleBoardView(host)
card = CountingCard('panel-1')
host.resize(640, 480)
board.setGeometry(host.rect())

# QTabWidget/QScrollArea may suppress their descendants while changing the
# visible page.  Packing during that interval must not record the inherited
# state as a permanent board-local disable.
host.setUpdatesEnabled(False)
board.set_cards((card,))
assert not board.updatesEnabled()
host.setUpdatesEnabled(True)
host.show()
app.processEvents()

assert board.updatesEnabled()
assert card.updatesEnabled()
assert card.paint_count > 0
assert card.geometry().width() > 0 and card.geometry().height() > 0
host.close()
app.processEvents()
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


def test_board_qtest_drop_uses_the_nearest_anchor_before_gravity() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.board import BoardMetrics
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['test'])
metrics = BoardMetrics(10)
cards = tuple(PanelCardView(f'panel-{index}') for index in range(3))
w, h = cards[0].size().width(), cards[0].size().height()
board = ConsoleBoardView(metrics=metrics)
board.resize(3 * w + 4 * 10, 2 * h + 3 * 10)
board.set_cards(cards); board.show(); app.processEvents()
events = []
board.order_committed.connect(events.append)
row = ((10, 10), (20 + w, 10), (30 + 2 * w, 10))
assert tuple(card.geometry().getRect()[:2] for card in cards) == row

def drop_top_left_at(card, top_left):
    grab = QtCore.QPoint(18, 18)
    target = card.mapFrom(board, QtCore.QPoint(top_left[0] + 18, top_left[1] + 18))
    QtTest.QTest.mousePress(card, QtCore.Qt.LeftButton, pos=grab)
    QtTest.QTest.mouseMove(card, target)
    QtTest.QTest.mouseRelease(card, QtCore.Qt.LeftButton, pos=target)
    app.processEvents()

drop_top_left_at(cards[2], (12, h + 16))
assert board._order == ('panel-0', 'panel-1', 'panel-2')
second_row = ((10, 10), (20 + w, 10), (10, 20 + h))
assert tuple(card.geometry().getRect()[:2] for card in cards) == second_row

board.resize(w + 20, 4 * h); app.processEvents()
assert all(card.geometry().x() == 10 for card in cards)
board.resize(3 * w + 4 * 10, 2 * h + 3 * 10); app.processEvents()
assert tuple(card.geometry().getRect()[:2] for card in cards) == second_row

drop_top_left_at(cards[2], (12, 14))
assert board._order == ('panel-2', 'panel-0', 'panel-1')
assert tuple(board._cards[panel_id].geometry().getRect()[:2] for panel_id in board._order) == row
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
metrics = BoardMetrics(10)
cards = tuple(PanelCardView(f'panel-{index}') for index in range(4))
w, h = cards[0].size().width(), cards[0].size().height()
board = ConsoleBoardView(metrics=metrics)
board.resize(2 * w + 3 * 10, 2 * h + 3 * 10)
board.set_cards(cards); board.show(); app.processEvents()
before = tuple(card.geometry().getRect() for card in cards[1:])
assert not hasattr(board, '_ghost')

def drag_to(card, board_point):
    start = card.geometry().center()
    local_target = card.mapFrom(board, QtCore.QPoint(*board_point))
    QtTest.QTest.mousePress(card, QtCore.Qt.LeftButton, pos=QtCore.QPoint(18, 18))
    QtTest.QTest.mouseMove(card, local_target)
    app.processEvents()
    return local_target

first_target = drag_to(cards[0], (w + 60, h + 60))
assert cards[0].pos() == QtCore.QPoint(w + 42, h + 42)
assert tuple(card.geometry().getRect() for card in cards[1:]) == before
QtTest.QTest.mouseRelease(cards[0], QtCore.Qt.LeftButton, pos=first_target)
app.processEvents()
assert board._order != ('panel-0', 'panel-1', 'panel-2', 'panel-3')

second_target = drag_to(cards[0], (35, 2 * h))
assert cards[0].pos() == QtCore.QPoint(17, 2 * h - 18)
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
metrics = BoardMetrics(10)
first, second = PanelCardView('first'), PanelCardView('second')
w, h = first.size().width(), first.size().height()
board = ConsoleBoardView(metrics=metrics); board.resize(w + 20, 2 * h + 30)
board.set_cards((first, second)); board.show(); app.processEvents()
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
metrics = BoardMetrics(10)
cards = tuple(PanelCardView(f'panel-{index}') for index in range(3))
w, h = cards[0].size().width(), cards[0].size().height()
board = ConsoleBoardView(metrics=metrics); board.resize(2 * w + 3 * 10, 2 * h + 3 * 10)
board.set_cards(cards); board.show(); app.processEvents()
identities = tuple(board._cards.values())
assert cards[0].geometry().x() == 10 and cards[1].geometry().x() == 20 + w
board.set_cards(cards); app.processEvents()
assert tuple(board._cards.values()) == identities
board.resize(w + 20, 4 * h); app.processEvents()
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
import zou_lab_control
import zlc_ui.console.logic_editor_view as tested_module
print(tested_module.__file__)
from PyQt5 import QtCore, QtTest
from zlc_ui.console import TaskConsoleHandle, TaskConsoleView
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['logic-editor'])
view = TaskConsoleView()
handle = TaskConsoleHandle(None, view)
handle.set_panel_sizes(('2x2',), '2x2')
handle.add_logic_row('camera-1', 'measurement')
handle.set_panel_publishers((('panel-1', (
    ('center_x', '35', 'held · @logic/panel-1/center_x'),
    ('roi', '4', 'live · @logic/panel-1/roi'),
)),))
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
    'auto_preview': True,
    'preview_offered': True,
    'artifact_results': ({
        'name': 'artifact_path', 'contract_id': 'calibration.readout',
        'path': 'C:/data/calibration-2.json',
    },),
    'source_required': True,
    'source_signal': '',
    'source_options': ('camera-1.frames',),
    'source_labels': {'camera-1.frames': 'frames  [1 × 1 × (3×96×128)]'},
    'source_groups': {'camera-1.frames': 'camera-1'},
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
# The frame-signal picker is the SAME grouped tree every signal chooser uses:
# a producer header with keyed leaves, not a flat list.
from zlc_ui.fluent import FluentTreeComboBox
assert isinstance(editor.source_combo, FluentTreeComboBox)
assert editor.source_combo.model().item(0).text() == '(not selected)'
source_header = editor.source_combo.model().item(1)
assert source_header.text() == 'camera-1'
assert source_header.child(0).text() == 'frames  [1 × 1 × (3×96×128)]'
handle.set_logic_commands('camera-1', can_start=False, can_stop=False)
view.show()
app.processEvents()
publisher = handle._panel_publisher_rows['panel-1']
assert tuple(view._logic_rows) == (handle._rows['camera-1'], publisher)
assert not publisher.publishes_label.isHidden()
assert all(button.isHidden() for button in (
    publisher.start_button, publisher.preview_switch, publisher.stop_button,
    publisher.remove_button,
))
assert not publisher.edit_button.isHidden()
publisher_edits = []
handle.panel_publisher_edit_requested.connect(publisher_edits.append)
QtTest.QTest.mouseClick(publisher.edit_button, QtCore.Qt.LeftButton)
assert publisher_edits == ['panel-1']
publisher_projection = {
    'node_id': 'panel-1',
    'api_name': 'Image outputs',
    'kind': 'panel publisher',
    'form_spec': FormSpec((
        FormFieldProps(key='roi_mean', kind='bool', label='Mean', default=True),
        FormFieldProps(key='roi_max', kind='bool', label='Max', default=True),
    )),
    'form_values': {'roi_mean': True, 'roi_max': True},
    'source_required': False,
    'device_options': {}, 'device_keys': {},
    'preview_offered': False, 'auto_preview': False,
    'running': False, 'pending': False,
    'can_start': False, 'can_stop': False,
    'issues': (), 'error': '', 'status': '',
}
publisher_patches = []
handle.panel_publisher_draft_changed.connect(
    lambda panel_id, patch: publisher_patches.append((panel_id, patch))
)
handle.open_panel_publisher_editor('panel-1', publisher_projection)
publisher_editor = handle._panel_publisher_editors['panel-1']
assert publisher_editor.source_combo is None
assert publisher_editor.start_button.isHidden()
publisher_editor.form.widget_for('roi_max').click()
assert publisher_patches == [('panel-1', {'values': {'roi_max': False}})]
assert view.tabs.indexOf(editor) >= 0
assert view.tabs.indexOf(publisher_editor) >= 0
assert view.tabs.currentWidget() is publisher_editor
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
source_combo.showPopup()
app.processEvents()
header_index = source_combo.model().index(1, 0)
source_combo.view().expand(header_index)
app.processEvents()
leaf_index = source_combo.model().index(0, 0, header_index)
leaf_rect = source_combo.view().visualRect(leaf_index)
assert leaf_rect.isValid() and not leaf_rect.isEmpty()
QtTest.QTest.mouseClick(
    source_combo.view().viewport(), QtCore.Qt.LeftButton,
    QtCore.Qt.NoModifier, leaf_rect.center(),
)
assert editor.source_combo is source_combo
assert source_combo.current_choice_key() == 'camera-1.frames'
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
assert view.tabs.indexOf(editor) < 0
assert view.tabs.indexOf(publisher_editor) >= 0
assert 'camera-1' not in handle._logic_editors
assert handle.logic_row_ids() == ('camera-1',)
view.editor_close_requested.emit(publisher_editor)
app.processEvents()
assert 'panel-1' not in handle._panel_publisher_editors
"""
    )


def test_panel_editor_and_setting_are_views_of_the_same_projection() -> None:
    _run_qt(
        """
import zou_lab_control
import zlc_ui.console.panel_editor_view as tested_module
print(zou_lab_control.__file__)
print(tested_module.__file__)
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from zlc_ui.console import TaskConsoleHandle, TaskConsoleView
from zlc_ui.fluent import FluentTreeComboBox
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['panel-editor'])
temporary = TemporaryDirectory()
save_directory = temporary.name
view = TaskConsoleView()
handle = TaskConsoleHandle(None, view)
handle.set_panel_sizes(('1x2', '2x2', '1x4'), '2x2')
handle.set_panel_intervals((100, 200, 400, 800), 400)
handle.add_panel('panel-1', 'Camera')
groups = (('camera', (('frames', '@logic/cm/frames'),)),)
overlay_groups = (('occupancy', (('sites', '@logic/occ/site_overlay'),)),)
handle.set_panel_signal_choices(
    'panel-1', groups, current='@logic/cm/frames',
    overlay_groups=overlay_groups,
    overlay_current='@logic/occ/site_overlay',
)
state = {
    'signal': '@logic/cm/frames', 'kind': 'image', 'size': '2x2',
    'interval_ms': 100, 'title': 'Camera',
    'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '@logic/occ/site_overlay',
}
surface = {
    'semantic': ({
        'key': 'x', 'label': 'X axis', 'kind': 'choice', 'value': 'sensor_x',
        'allow_none': False,
        'choices': (('Sensor X', 'sensor_x'), ('Point row', 'point_row')),
        'minimum': None, 'maximum': None, 'step': None,
    },),
    'display': (
        {'key': 'title', 'label': 'Title', 'kind': 'text',
         'value': None, 'allow_none': True, 'choices': (),
         'minimum': None, 'maximum': None, 'step': None,
         'automatic': True, 'unavailable_reason': ''},
        {'key': 'x_label', 'label': 'X label', 'kind': 'text',
         'value': None, 'allow_none': True, 'choices': (),
         'minimum': None, 'maximum': None, 'step': None,
         'automatic': True, 'unavailable_reason': ''},
        {'key': 'x_display_unit', 'label': 'X unit', 'kind': 'choice',
         'value': None, 'allow_none': True,
         'choices': (('Pixel', 'pixel'), ('Millimetre', 'mm')),
         'minimum': None, 'maximum': None, 'step': None,
         'automatic': True, 'unavailable_reason': ''},
        {'key': 'color_min', 'label': 'Color minimum', 'kind': 'number',
         'value': None, 'allow_none': True, 'choices': (),
         'minimum': None, 'maximum': None, 'step': None,
         'automatic': False,
         'unavailable_reason': 'Choose Fixed color limits to edit.'},
        {'key': 'colormap', 'label': 'Colormap', 'kind': 'choice',
         'value': 'viridis', 'allow_none': False,
         'choices': (('Viridis', 'viridis'), ('Magma', 'magma')),
         'minimum': None, 'maximum': None, 'step': None},
        {'key': 'show_colorbar', 'label': 'Colorbar', 'kind': 'boolean',
         'value': True, 'allow_none': False, 'choices': (),
         'minimum': None, 'maximum': None, 'step': None},
    ),
    'fit': ({
        'key': 'model', 'label': 'Fit model', 'kind': 'choice', 'value': None,
        'allow_none': True,
        'choices': (('2-D Gaussian', 'anisotropic_gaussian_center'),),
        'minimum': None, 'maximum': None, 'step': None,
    },),
}
handle.set_panel_projection('panel-1', state, surface)
card = handle._cards['panel-1']
blank_surface = {
    'semantic': (), 'display': (), 'fit': (),
    'semantic_unavailable': '', 'display_unavailable': '',
    'fit_unavailable': '',
}
handle.set_panel_projection('panel-1', state, blank_surface)
card.resize(340, 180)
authored_size = card.size()
card._open_settings()
assert card._settings_popup.isVisible()
blank_popup_width = card._settings_popup.width()
handle.set_panel_projection('panel-1', state, surface)
app.processEvents()
assert card.size() == authored_size, 'same-size projection reset the card geometry'
assert card._settings_popup.isVisible()
card._settings_body.layout().activate()
bound_editor = card._settings_form.widget_for('display__colormap')
bound_editor_right = bound_editor.mapTo(
    card._settings_scroll.viewport(),
    QtCore.QPoint(bound_editor.width() - 1, 0),
).x()
assert bound_editor_right < card._settings_scroll.viewport().width(), (
    'an already-open Setting popup kept its blank-schema width after binding: '
    f'blank={blank_popup_width}, bound={card._settings_popup.width()}, '
    f'editor-right={bound_editor_right}, '
    f'viewport={card._settings_scroll.viewport().width()}'
)
for key, row in card._settings_form._rows.items():
    assert row.height() >= row.minimumSizeHint().height(), (
        f'Setting row {key!r} was vertically cut off: '
        f'{row.height()} < {row.minimumSizeHint().height()}'
    )
parameter_left_edges = {
    key: card._settings_form.widget_for(key).mapTo(
        card._settings_form, QtCore.QPoint(0, 0)
    ).x()
    for key in ('display__title', 'display__x_label', 'display__colormap')
}
assert len(set(parameter_left_edges.values())) == 1, parameter_left_edges
label_slot_widths = {
    key: card._settings_form._rows[key].layout().itemAt(0).widget().width()
    for key in ('display__title', 'display__x_label', 'display__colormap')
}
assert len(set(label_slot_widths.values())) == 1, label_slot_widths
stable_geometry = {
    key: (
        card._settings_form._rows[key].geometry(),
        card._settings_form.widget_for(key).geometry(),
    )
    for key in card._settings_form.spec.keys
}
interval_control = card._settings_form.widget_for('interval_ms')
interval_control.setCurrentIndex(interval_control.findData(800))
interval_control.activated.emit(interval_control.currentIndex())
state = dict(state, interval_ms=800)
handle.set_panel_projection('panel-1', state, surface)
app.processEvents()
projected_geometry = {
    key: (
        card._settings_form._rows[key].geometry(),
        card._settings_form.widget_for(key).geometry(),
    )
    for key in card._settings_form.spec.keys
}
assert projected_geometry == stable_geometry, {
    key: (stable_geometry[key], projected_geometry[key])
    for key in stable_geometry
    if stable_geometry[key] != projected_geometry[key]
}
class LayoutProbe(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.requests = 0
    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.LayoutRequest:
            self.requests += 1
        return False
layout_probe = LayoutProbe()
for watched in (
    card._settings_form,
    *card._settings_form._rows.values(),
):
    watched.installEventFilter(layout_probe)
card._settings_form.reconcile(
    card._settings_form.spec,
    card._settings_form.read_all(),
)
app.processEvents()
assert layout_probe.requests == 0, (
    'a value-only Setting projection rebuilt the form layout'
)
compact_surface = dict(
    surface,
    semantic=(),
    display=(surface['display'][0], surface['display'][1], surface['display'][4]),
    fit=(),
)
handle.set_panel_projection('panel-1', state, compact_surface)
card._settings_popup.hide()
QtTest.QTest.qWait(300)
card._open_settings()
app.processEvents()
compact_left_edges = {
    key: card._settings_form.widget_for(key).mapTo(
        card._settings_form, QtCore.QPoint(0, 0)
    ).x()
    for key in ('display__title', 'display__x_label', 'display__colormap')
}
assert len(set(compact_left_edges.values())) == 1, compact_left_edges
handle.set_panel_projection('panel-1', state, surface)
title_widget = card._settings_form.widget_for('title')
natural_title_width = title_widget.minimumWidth()
title_widget.setMinimumWidth(520)
card._settings_popup.hide()
QtTest.QTest.qWait(300)
card._open_settings()
card._settings_body.layout().activate()
editor_right = title_widget.mapTo(
    card._settings_scroll.viewport(),
    QtCore.QPoint(title_widget.width() - 1, 0),
).x()
assert editor_right < card._settings_scroll.viewport().width(), (
    'Setting cut off the editor: '
    f'popup={card._settings_popup.width()} '
    f'viewport={card._settings_scroll.viewport().width()} '
    f'editor-right={editor_right}'
)
title_widget.setMinimumWidth(natural_title_width)
assert isinstance(card._settings_form.widget_for('signal'), FluentTreeComboBox)
assert isinstance(card._settings_form.widget_for('overlay_signal'), FluentTreeComboBox)
assert card._settings_form.widget_for('signal').current_choice_key() == '@logic/cm/frames'
assert card._settings_form.widget_for('overlay_signal').current_choice_key() == '@logic/occ/site_overlay'
assert card._settings_form.widget_for('signal')._display_text() == 'camera · frames'
assert card._settings_form.widget_for('overlay_signal')._display_text() == 'occupancy · sites'
assert card._settings_form.widget_for('overlay_signal').model().item(0).text() == 'Off'
replacement_groups = (('camera', (('reference', '@logic/cm/reference'),)),)
replacement_state = dict(state, signal='@logic/cm/reference')
handle.set_panel_signal_choices(
    'panel-1', replacement_groups, current='@logic/cm/reference',
    overlay_groups=overlay_groups, overlay_current='@logic/occ/site_overlay',
)
handle.set_panel_projection('panel-1', replacement_state, surface)
assert card._settings_form.widget_for('signal').current_choice_key() == '@logic/cm/reference'
handle.set_panel_signal_choices(
    'panel-1', groups, current='@logic/cm/frames',
    overlay_groups=overlay_groups,
    overlay_current='@logic/occ/site_overlay',
)
handle.set_panel_projection('panel-1', state, surface)
assert 'kind' not in card._settings_form.read_all()
assert card._settings_form.read_all()['display__show_colorbar'] is True
assert card._settings_form.read_all()['display__colormap'] == 'viridis'
assert card._settings_form.read_all()['display__title'] is None
assert card._settings_form.auto_switch_for('display__title').isChecked()
assert not card._settings_form.widget_for('display__title').isEnabled()
assert card._settings_form.read_all()['semantic__x'] == 'sensor_x'
assert card._settings_form.widget_for('fit__model').currentText() == '(none)'
assert card._settings_form.read_all()['overlay_signal'] == '@logic/occ/site_overlay'
interval_combo = card._settings_form.widget_for('interval_ms')
assert isinstance(interval_combo, QtWidgets.QComboBox)
assert tuple(interval_combo.itemData(index) for index in range(interval_combo.count())) == (
    100, 200, 400, 800,
)
# The cell kind is a grid panel PARAMETER: an enabled choice whose empty
# entry means the data decides, emitting the same cell_kind state patch the
# presenter accepts.
handle.set_grid_cell_kinds(('curve', 'image', 'histogram'))
facet_state = dict(state, kind='facet_grid', cell_kind='curve', title='Site grid')
handle.set_panel_projection('panel-1', facet_state, surface)
cell_combo = card._settings_form.widget_for('cell_kind')
assert isinstance(cell_combo, QtWidgets.QComboBox)
assert cell_combo.isEnabled()
assert tuple(cell_combo.itemData(index) for index in range(cell_combo.count())) == (
    '', 'curve', 'image', 'histogram',
)
assert card._settings_form.read_all()['cell_kind'] == 'curve'
cell_patches = []
capture_cell_patch = lambda patch: cell_patches.append(dict(patch))
card.state_changed.connect(capture_cell_patch)
cell_combo.setCurrentIndex(cell_combo.findData('image'))
cell_combo.activated.emit(cell_combo.currentIndex())
card.state_changed.disconnect(capture_cell_patch)
assert {'cell_kind': 'image'} in cell_patches
handle.set_panel_projection('panel-1', state, surface)
assert 'cell_kind' not in card._settings_form.spec.keys, (
    'only a facet grid offers a cell kind')

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
    'auto_preview': True,
    'preview_offered': True,
}
projection = {
    'panel_id': 'panel-1', 'state': state, 'signal_options': groups,
    'overlay_signal_options': overlay_groups,
    'interval_choices': (100, 200, 400, 800),
    'parameter_surface': surface,
    'save_directory': save_directory,
    'frozen_signal': '@logic/cm/frames',
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
    lambda panel_id, path: events.append(('save', panel_id, path))
)
handle.panel_editor_closed.connect(
    lambda panel_id: events.append(('closed', panel_id))
)
card_semantic = card._settings_form.widget_for('semantic__x')
card_semantic.setCurrentIndex(1)
card_semantic.activated.emit(1)
card_fit = card._settings_form.widget_for('fit__model')
card_fit.setCurrentIndex(1)
card_fit.activated.emit(1)
handle.open_panel_editor('panel-1', projection)
editor = handle._panel_editors['panel-1']
assert isinstance(editor.panel_form.widget_for('signal'), FluentTreeComboBox)
assert isinstance(editor.panel_form.widget_for('overlay_signal'), FluentTreeComboBox)
assert editor.panel_form.widget_for('signal').current_choice_key() == '@logic/cm/frames'
assert editor.panel_form.widget_for('overlay_signal').current_choice_key() == '@logic/occ/site_overlay'
assert editor.panel_form.widget_for('signal')._display_text() == 'camera · frames'
assert editor.panel_form.widget_for('overlay_signal')._display_text() == 'occupancy · sites'
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
    'title', 'x_label', 'x_display_unit', 'color_min',
    'colormap', 'show_colorbar'
)
assert editor.parameter_forms['fit'].spec.keys == ('model',)
locked_surface = dict(surface, science_locked=True)
handle.set_panel_projection('panel-1', state, locked_surface)
assert handle.update_panel_editor(
    'panel-1', dict(projection, parameter_surface=locked_surface)
)
assert not card._settings_form.widget_for('signal').isEnabled()
assert not card._settings_form.widget_for('overlay_signal').isEnabled()
assert not card._settings_form.widget_for('semantic__x').isEnabled()
assert card._settings_form.widget_for('size').isEnabled()
assert card._settings_form.widget_for('display__colormap').isEnabled()
assert card._settings_form.widget_for('fit__model').isEnabled()
assert not editor.panel_form.widget_for('signal').isEnabled()
assert not editor.panel_form.widget_for('overlay_signal').isEnabled()
assert not editor.parameter_forms['semantic'].isEnabled()
assert editor.parameter_forms['display'].isEnabled()
assert editor.parameter_forms['fit'].isEnabled()
assert not editor._producer_editor.form.isEnabled()
handle.set_panel_projection('panel-1', state, surface)
assert handle.update_panel_editor('panel-1', projection)
assert editor.parameter_forms['semantic'].isEnabled()
title_auto = editor.parameter_forms['display'].auto_switch_for('title')
title_edit = editor.parameter_forms['display'].widget_for('title')
assert title_auto.isChecked()
assert title_auto.text() == 'Auto Title'
switch_image = QtGui.QImage(
    title_auto.sizeHint(), QtGui.QImage.Format_ARGB32_Premultiplied
)
switch_image.fill(QtCore.Qt.transparent)
title_auto.resize(title_auto.sizeHint())
title_auto.render(switch_image)
middle_y = switch_image.height() // 2
assert all(
    QtGui.qAlpha(switch_image.pixel(x, middle_y)) > 0
    for x in range(2, title_auto._content_width() - 2)
), 'Auto/Manual text is painted outside rather than inside the switch track'
title_row = editor.parameter_forms['display']._rows['title']
assert title_row.layout().itemAt(0).widget() is title_auto
assert title_row.layout().itemAt(1).widget() is title_edit
assert not title_edit.isEnabled()
title_auto.setChecked(False)
assert title_auto.text() == 'Manual Title'
assert title_edit.isEnabled()
title_edit.setText('Camera image')
title_edit.editingFinished.emit()
title_auto.setChecked(True)
assert not title_edit.isEnabled()
x_unit_auto = editor.parameter_forms['display'].auto_switch_for('x_display_unit')
assert x_unit_auto.isChecked()
x_unit_edit = editor.parameter_forms['display'].widget_for('x_display_unit')
assert not x_unit_edit.isEnabled()
x_unit_auto.setChecked(False)
assert x_unit_edit.isEnabled() and x_unit_edit.currentData() == 'pixel'
x_unit_auto.setChecked(True)
color_min_edit = editor.parameter_forms['display'].widget_for('color_min')
try:
    editor.parameter_forms['display'].auto_switch_for('color_min')
except KeyError:
    pass
else:
    raise AssertionError('fixed color limits must not have independent Auto switches')
assert not color_min_edit.isEnabled()
fixed_display = tuple(
    dict(field, value=0, unavailable_reason='')
    if field['key'] == 'color_min' else field
    for field in surface['display']
)
fixed_surface = dict(surface, display=fixed_display)
handle.set_panel_projection('panel-1', state, fixed_surface)
handle.update_panel_editor(
    'panel-1', dict(projection, parameter_surface=fixed_surface)
)
color_min_edit = editor.parameter_forms['display'].widget_for('color_min')
assert color_min_edit.isEnabled() and color_min_edit.text() == '0'
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
assert editor.save_directory.text() == save_directory
assert tuple(
    editor.save_format.itemData(index)
    for index in range(editor.save_format.count())
) == ('png', 'pdf', 'svg')
assert editor.save_auto_name.isChecked()
assert Path(editor.save_preview.text()).parent == Path(save_directory)
editor.save_auto_name.setChecked(False)
editor.save_name.setText('manual-name')
editor.save_format.setCurrentIndex(editor.save_format.findData('svg'))
editor.save_button.click()
app.processEvents()
assert ('state', 'panel-1', {'interval_ms': 800}) in events
assert ('state', 'panel-1', {'semantic': {'x': 'point_row'}}) in events
assert ('state', 'panel-1', {
    'fit': {'model': 'anisotropic_gaussian_center'}
}) in events
assert ('state', 'panel-1', {'display': {'colormap': 'magma'}}) in events
assert ('state', 'panel-1', {'display': {'title': 'Camera image'}}) in events
assert ('state', 'panel-1', {'display': {'title': None}}) in events
assert ('state', 'panel-1', {'display': {'x_display_unit': 'pixel'}}) in events
assert ('state', 'panel-1', {'display': {'x_display_unit': None}}) in events
assert ('state', 'panel-1', {
    'fit': {'model': 'anisotropic_gaussian_center'}
}) in events
assert ('draft', 'cm', {'values': {'repeat': 2}}) in events
assert ('refresh', 'panel-1') in events
assert ('restart', 'panel-1') in events
assert ('save', 'panel-1', str(Path(save_directory) / 'manual-name.svg')) in events

changed = dict(state, title='Retitled', interval_ms=800)
handle.set_panel_projection('panel-1', changed, surface)
handle.update_panel_editor('panel-1', dict(projection, state=changed, stale=True))
assert 'Retitled' in card._title_label.toolTip()
assert editor.panel_form.read_all()['title'] == 'Retitled'
assert not editor.save_button.isEnabled()
view.editor_close_requested.emit(editor)
app.processEvents()
assert view.tabs.count() == 2 and 'panel-1' not in handle._panel_editors
assert ('closed', 'panel-1') in events
assert second_host.widget.parentWidget() is None
temporary.cleanup()
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
# The header is one semantic row: identity on the left, telemetry as the
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
        """import zou_lab_control
from zlc_ui.console import task_console_view as tested_module
print(tested_module.__file__)
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import TaskConsoleHandle
app = ensure_qt_app(['test'])
view = tested_module.TaskConsoleView()
handle = TaskConsoleHandle(None, view)
handle.set_panel_sizes(('2x2',), '2x2')
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

# A running Task freezes Logic identity, not the monitor window.
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
assert view.name_edit.isEnabled()
assert view.kind_combo.isEnabled()
assert not view.add_panel_button.isEnabled()
assert view.save_layout_button.isEnabled()
assert not view.load_layout_button.isEnabled()
assert view.selectors_switch.isEnabled()
assert view.pause_switch.isEnabled()
assert view.save_screenshot_button.isEnabled()
assert preview.settings_button.isEnabled()
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
view.kind_combo.setCurrentIndex(0)
assert view.kind_combo.currentData()[0] == 'plot'
assert view.add_panel_button.isEnabled()
QtTest.QTest.mouseClick(view.add_panel_button, QtCore.Qt.LeftButton)
assert events[-1] == ('panel', 'curve')
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

    A picker squeezed into the header collapsed to an ellipsis and clipped
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
    ('@logic/panel-1/roi_mean', 'roi_mean', 'finished', 'panel-1', '@logic/cm/frames'),
)
dialog = SignalChooser(rows)
texts = [dialog.list.item(i).text() for i in range(dialog.list.count())]
assert texts == ['cm', '    frames', 'panel-1', '    roi_mean  (finished)'], texts

# A producer heading is a heading, not a choice.
assert not dialog.list.item(0).flags() & QtCore.Qt.ItemIsSelectable
assert dialog.chosen() == '@logic/cm/frames', dialog.chosen()

dialog.list.setCurrentRow(3)
assert dialog.chosen() == '@logic/panel-1/roi_mean'
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


def test_task_console_acceptance_launcher_keeps_target_size_and_left_anchored_empty_status() -> None:
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


def test_every_panel_kind_opens_its_setting_form_with_exact_keys() -> None:
    """The Setting form's SPEC and its VALUES are one decision, not two lists.

    ``FluentParameterForm`` demands exact keys, and it is built inside a Qt
    slot -- so a spec that declares a row the values do not supply does not
    raise, it ABORTS the process the moment an operator presses Setting.  The
    only test that can catch it is one that compares the two key sets for
    every panel kind a card can be put in.
    """

    _run_qt(
        """
import zou_lab_control
print(zou_lab_control.__file__)
from PyQt5 import QtCore, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import panel_card_view as tested_module
from zlc_ui.console import panel_editor_view as editor_module
print(tested_module.__file__)
print(editor_module.__file__)

app = ensure_qt_app(['test'])
surface = {'semantic': (), 'display': (), 'fit': ()}
groups = (('camera', (('frames', '@logic/cm/frames'),)),)
overlays = (('occupancy', (('sites', '@logic/occ/occupied'),)),)
owner = QtWidgets.QWidget()
owner.show()
app.processEvents()

try:
    for kind, cell_kind, overlay_signal in (
        ('image', '', '@logic/occ/occupied'),
        ('image', '', ''),
        ('facet_grid', '', '@logic/occ/occupied'),
        ('facet_grid', 'image', '@logic/occ/occupied'),
        ('facet_grid', 'curve', ''),
        ('facet_grid', 'histogram', ''),
        ('curve', '', ''),
        ('histogram', '', ''),
    ):
        card = None
        editor = None
        try:
            state = {
                'signal': '@logic/cm/frames', 'kind': kind,
                'cell_kind': cell_kind, 'size': '2x2',
                'interval_ms': 100, 'title': 'Card', 'semantic': {},
                'display': {}, 'fit': {}, 'overlay_signal': overlay_signal,
            }
            card = tested_module.PanelCardView('panel-1', 'Card', owner)
            card.set_size_choices(('1x2', '2x2', '1x4'), '2x2')
            card.set_cell_kind_choices(('curve', 'image', 'histogram'))
            card.set_signal_choices(
                groups, current=state['signal'],
                overlay_groups=overlays, overlay_current=overlay_signal,
            )
            card.set_panel_projection(state, surface)
            card.show()
            app.processEvents()
            spec_keys = set(card._form_spec().keys)
            value_keys = set(card._form_values())
            assert spec_keys == value_keys, (
                kind, cell_kind, sorted(spec_keys ^ value_keys)
            )
            # The real crash path: this is what pressing Setting runs.
            card._open_settings()
            assert card._settings_form is not None
            assert set(card._settings_form.spec.keys) == value_keys

            editor = editor_module.PanelEditorView('panel-1', {
                'panel_id': 'panel-1', 'state': state,
                'signal_options': groups,
                'overlay_signal_options': overlays,
                'interval_choices': (100, 200, 400, 800),
                'size_choices': ('1x2', '2x2', '1x4'),
                'parameter_surface': surface, 'save_directory': '.',
                'frozen_signal': state['signal'],
                'frozen_publication': None, 'frozen_snapshot': None,
                'stale': False, 'producer_node_id': '',
                'producer_logic': None,
            }, owner)
            assert ('overlay_signal' in editor.panel_form.spec.keys) == (
                'overlay_signal' in spec_keys
            )
        finally:
            if card is not None:
                card.retire_settings_popup()
                card.deleteLater()
            if editor is not None:
                editor.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(
                None, QtCore.QEvent.DeferredDelete
            )
            app.processEvents()
finally:
    owner.close()
    owner.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.DeferredDelete
    )
    app.processEvents()
"""
    )


def test_the_auto_preview_control_is_a_switch_that_looks_different_when_on() -> None:
    """A checkable FluentButton drew NO checked state: on and off were the
    same pixels, in the same accent as Edit beside it, so the control read as
    a fourth command button and an operator could not find the toggle at all.
    A boolean is a switch in this chrome; a command is a button.
    """

    _run_qt(
        """
import zou_lab_control
print(zou_lab_control.__file__)
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentSwitch
from zlc_ui.console import logic_row_view as tested_module
print(tested_module.__file__)

app = ensure_qt_app(['test'])
row = tested_module.LogicRowView('camera_measurement', 'measurement')
row.resize(760, 48)
row.show()
app.processEvents()

switch = row.preview_switch
assert isinstance(switch, FluentSwitch), type(switch)

seen = []
switch.toggled.connect(seen.append)
row.set_auto_preview(False)
app.processEvents()
off = switch.grab().toImage()
row.set_auto_preview(True)
app.processEvents()
on = switch.grab().toImage()
assert not seen, 'showing the owner state must not re-raise it: %r' % (seen,)
assert on != off, 'the switch must look different when it is on'

order = []
layout = row.layout().itemAt(0).layout()
for index in range(layout.count()):
    widget = layout.itemAt(index).widget()
    if widget is not None and widget in (
        row.start_button, row.preview_switch, row.stop_button,
        row.edit_button, row.remove_button,
    ):
        order.append(widget)
assert order.index(row.preview_switch) == order.index(row.start_button) + 1, (
    'the switch belongs immediately beside Start')

row.set_task_takeover(True)
assert not switch.isEnabled(), 'a row that cannot Start cannot re-aim Start'

# A node that declares nothing to open has no preference to express, so it
# has no switch either -- a control that does nothing is worse than absent.
row.set_preview_offered(False)
app.processEvents()
assert not switch.isVisible(), 'no switch where the node opens nothing'
row.set_preview_offered(True)
app.processEvents()
assert switch.isVisible()
"""
    )


def test_the_editor_shows_the_same_preview_switch_beside_its_own_start() -> None:
    """Adding a node opens its Edit tab and focuses it, so THAT Start is the
    one an operator presses.  The switch is a projection of the node's stored
    preference in both places -- the editor keeps no default of its own.
    """

    _run_qt(
        """
import zou_lab_control
print(zou_lab_control.__file__)
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentSwitch
from zlc_ui.form import FormSpec
from zlc_ui.console import logic_editor_view as tested_module
print(tested_module.__file__)

app = ensure_qt_app(['test'])
projection = {
    'node_id': 'camera_measurement',
    'api_name': 'camera_measurement',
    'kind': 'measurement',
    'form_spec': FormSpec(()),
    'form_values': {},
    'artifact_form_spec': FormSpec(()),
    'artifact_values': {},
    'artifact_results': (),
    'auto_preview': False,
    'preview_offered': True,
    'can_start': True,
    'can_stop': False,
}
editor = tested_module.LogicEditorView('camera_measurement', projection)
editor.resize(900, 420)
editor.show()
app.processEvents()
switch = editor.preview_switch
assert isinstance(switch, FluentSwitch), type(switch)
assert switch.isChecked() is False, 'the editor shows the stored preference'

# Beside Start, in pixels: same row, immediately to its right.  A switch that
# merely EXISTS somewhere is what an operator could not find.
start = editor.start_button
assert switch.isVisible() and start.isVisible()
assert abs(switch.y() - start.y()) <= 2, (switch.geometry(), start.geometry())
gap = switch.x() - (start.x() + start.width())
assert 0 <= gap <= 40, (switch.geometry(), start.geometry())

raised = []
switch.toggled.connect(raised.append)
editor.update_projection(dict(projection, auto_preview=True))
assert switch.isChecked() is True
assert not raised, 'projecting the owner state must not raise intent: %r' % (raised,)

intents = []
editor.auto_preview_changed.connect(intents.append)
switch.setChecked(False)
assert intents == [False], intents

editor.set_mutation_enabled(False)
assert not switch.isEnabled(), 'a Task owns the console: nothing here re-aims Start'

editor.update_projection(dict(projection, preview_offered=False))
assert not switch.isVisible(), 'no switch where the node opens nothing'
"""
    )
