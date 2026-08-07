from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run_qt(code: str) -> None:
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


def test_panel_card_construct_and_setters() -> None:
    _run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import PanelCardView
app = ensure_qt_app(['test'])
card = PanelCardView('panel-1', 'Card')
card.set_signal_choices((('source', (('Temperature', 'temperature'),)),))
surface = QtWidgets.QLabel('fake surface')
card.set_surface(surface)
card.set_status('ready', error=False)
card.set_selectors_enabled(False)
assert card.panel_id == 'panel-1'
assert card.signal_combo.count() == 2
assert surface.parentWidget() is not None
assert not card.signal_combo.isEnabled()
card.set_surface(None)
assert not card._placeholder.isHidden()
card.set_panel_size('1x4')
assert card.size_combo.currentData() == '1x4'
try:
    card.set_panel_size('not-a-v1-size')
except ValueError:
    pass
else:
    raise AssertionError('invalid panel size was accepted')
"""
    )


def test_panel_card_qtest_signal_payloads() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import PanelCardView
app = ensure_qt_app(['test'])
card = PanelCardView('panel-1', 'Card')
card.show(); app.processEvents()
events = []
card.remove_requested.connect(lambda: events.append(('remove',)))
card.edit_requested.connect(lambda: events.append(('edit',)))
card.title_committed.connect(lambda value: events.append(('title', value)))
# The operator's own route: Edit and Remove live in the Setting popup, beside
# every other per-panel decision.  Reached any other way they were hidden
# widgets nothing ever showed, so a panel could not be removed at all.
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
QtTest.QTest.mouseClick(card.edit_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
QtTest.QTest.mouseClick(card.remove_button, QtCore.Qt.LeftButton)
card.title_edit.setFocus()
QtTest.QTest.keyClicks(card.title_edit, ' changed')
QtTest.QTest.keyClick(card.title_edit, QtCore.Qt.Key_Return)
assert ('edit',) in events
assert ('remove',) in events
assert ('title', 'Card changed') in events
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
row.set_state('running', 'processing fake input')
row.set_publishes((('out', '1', 'fake output'),))
assert row.status_label.text() == 'processing fake input'
assert row.stop_button.isEnabled()
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
QtTest.QTest.mouseClick(row.start_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(row.edit_button, QtCore.Qt.LeftButton)
assert events == ['start', 'edit']
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
view.resize(1096, 422); view.show(); app.processEvents()
assert view.summary_label.text() == 'one fake card'
assert view.status_strip.current_severity == 'idle'
assert view.board._cards['panel-1'] is card
assert view.selectors_switch.width() > 0
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
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import TaskConsoleView
app = ensure_qt_app(['test'])
view = TaskConsoleView(); view.show(); app.processEvents()
events = []
view.add_panel_requested.connect(lambda: events.append(('panel',)))
view.pause_toggled.connect(lambda value: events.append(('pause', value)))
view.save_requested.connect(lambda: events.append(('save',)))
QtTest.QTest.mouseClick(view.add_panel_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(view.pause_switch, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(view.save_button, QtCore.Qt.LeftButton)
assert ('panel',) in events
assert ('pause', True) in events
assert ('save',) in events
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


def test_the_connection_field_arrives_filled_in_and_is_never_wiped() -> None:
    """An empty address box makes every operator retype the same string.

    It was seeded and then cleared by the first status update, which reports
    "offline" with no address to offer -- so the seed lasted until the window
    finished opening.
    """

    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse.schedule_view import PulseScheduleView
app = ensure_qt_app(['endpoint'])
view = PulseScheduleView(); view.show(); app.processEvents()

# Whoever opens the window seeds the address it offers; this package does not
# know where a pulse server listens and must not write one down.
view.set_connection('offline', '127.0.0.1:18861', 'not connected')
assert view.connection_endpoint.text() == '127.0.0.1:18861'
assert 'host:port' in view.connection_endpoint.toolTip()

# Reporting offline again offers no address, and must leave the seed alone.
view.set_connection('offline', '', 'not connected')
assert view.connection_endpoint.text() == '127.0.0.1:18861'

# A real endpoint replaces it, and the status keeps the whole message.
view.set_connection('remote', '10.0.0.9:18861', '10.0.0.9:18861 - 26 ports, 62 lanes, 50 MHz')
assert view.connection_endpoint.text() == '10.0.0.9:18861'
assert '26 ports' in view.connection_status.toolTip()
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


def test_scan_page_shows_the_program_it_is_given_and_keeps_a_half_typed_one() -> None:
    """The source was the one field set_page dropped.

    So the editor opened blank -- the presenter generates a starter template for
    exactly that moment -- and then went stale while the revision it carries
    advanced.  And an operator mid-edit owns the box: a page catching up must
    not take what they were writing.
    """

    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse.models import ScanPageRecord
from zlc_ui.pulse.scan_view import PulseScanView
app = ensure_qt_app(['scan'])
view = PulseScanView(); view.show(); app.processEvents()

view.set_page(ScanPageRecord(source_text='scan_table = None', source_revision=1))
assert 'scan_table' in view.scan_code.toPlainText()

# Typing makes it the operator's, and a later page must not take it back.
view.scan_code.setPlainText('half typed')
assert view.code_dirty
view.set_page(ScanPageRecord(source_text='something else', source_revision=2))
assert view.scan_code.toPlainText() == 'half typed'
"""
    )


def test_one_more_control_cannot_widen_the_console_window() -> None:
    """Adding a button must not be a window-geometry decision.

    A plain horizontal row's minimum width is the sum of everything in it and
    nothing in it may shrink, so nine controls set the window's minimum and the
    tenth pushed it past the shared screen-fit size -- the window grew, the size
    rule broke, and the acceptance check failed on a change that had nothing to
    do with geometry.  The header wraps now: its minimum is the widest single
    control, and a second line appears when one is needed.
    """

    _run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import (
    ACCENT, FluentButton, WINDOW_SCREEN_FRACTION, screen_fit_window_size,
)
from zlc_ui.console import TaskConsoleView

app = ensure_qt_app(["header-wrap"])
view = TaskConsoleView(); view.show(); app.processEvents()
header = view.name_edit.parentWidget()
target = screen_fit_window_size(WINDOW_SCREEN_FRACTION).width()

before = header.minimumSizeHint().width()
assert before < target, (before, target)

# Five more, which no toolbar would survive if the row could not wrap.
for index in range(5):
    header.layout().addWidget(FluentButton(f"Extra {index}", color=ACCENT))
app.processEvents()
after = header.minimumSizeHint().width()
assert after == before, "a new control must not raise the window's minimum"
assert header.layout().heightForWidth(target) > header.layout().heightForWidth(4000), (
    "a narrow row has to take more lines than a wide one"
)
"""
    )
