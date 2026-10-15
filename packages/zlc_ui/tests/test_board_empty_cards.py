"""A card with no picture yet is laid out, held and dropped like a drawn one.

Drawn cards mount REAL zlc_plot raster surfaces, sized by the plotting
package's own plan, and every gesture goes through the window the way the
operator's pointer does -- to whatever child is under it, then to the card
that grabs it -- not to the card directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PLOT_TESTS = Path(__file__).resolve().parents[3] / "packages" / "zlc_plot" / "tests"


@pytest.fixture
def run_qt(run_qt):
    """Real zlc_plot surfaces: their test data factory and time to compile."""

    return lambda code: run_qt(code, timeout=180, extra_path=(PLOT_TESTS,))


_PROLOGUE = """
import time

import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets
from data_factory import make_dataset_schema, make_snapshot, mapped_domain_from_columns, repeat_domain
from zlc_plot import AxisRef, CurvePlot, Qt5PlotWidget, ensure_qt5_application
from zlc_plot.raster import RasterPlotHost
from zlc_ui.board import BoardMetrics
from zlc_ui.console import ConsoleBoardView, PanelCardView
from zlc_workbench.panel_sizes import install

app = ensure_qt5_application([])
# The composition layer's one size answer: an empty card is the frame the
# plotting package plans for its preset, which is what a drawn one frames.
install()

def wait_until(condition, timeout=15.0, message='condition did not settle'):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError(message)

schema = make_dataset_schema(
    repeat_domain(size=1),
    mapped_domain_from_columns({'x': [0.0, 1.0, 2.0, 3.0]}),
    dtype=np.float64,
)
host = RasterPlotHost.from_plot(
    make_snapshot(schema, np.arange(4.0).reshape(1, 4), revision=0),
    CurvePlot(AxisRef.point('x')),
    size='2x2',
)
# A cold checkout compiles the renderer first.
wait_until(lambda: host.front is not None, timeout=120.0, message='the plot never drew')
STATE = {
    'signal': '', 'kind': 'curve', 'size': '2x2', 'interval_ms': 100,
    'title': 'Plot', 'semantic': {}, 'display': {}, 'fit': {}, 'overlay_signal': '',
}
SHAPE = {'data_structure': ((('repeat', 1),), (('x', 4),), ())}

def new_card(panel_id, drawn):
    card = PanelCardView(panel_id, 'Plot')
    card.set_size_choices(('2x2',), '2x2')
    if drawn:
        card.set_panel_projection(dict(STATE, signal='@logic/cm/frames'), SHAPE)
        # Over a host that already drew: it shows that picture at once.
        card.set_surface(Qt5PlotWidget(host))
    # The console's default: Selectors off, the plot takes no gesture.
    card.set_selectors_enabled(False)
    return card

def strip_line_count(card):
    return card._title_label.text().count('<br>') + 1
"""


def test_an_empty_card_is_laid_out_like_a_drawn_one(run_qt) -> None:
    """Same frame, same strip, same caption line; the second line says why.

    The empty card centred a one-line caption in its strip -- half a line
    below every drawn neighbour's -- and said "Pick a signal in Setting" in
    its body even once a signal was picked.
    """

    run_qt(
        _PROLOGUE
        + """
drawn = new_card('panel-1', True)
empty = new_card('panel-2', False)
for card in (drawn, empty):
    card.show()
wait_until(lambda: drawn._surface_framed, message='the drawn card never framed its picture')
app.processEvents()

assert empty.size() == drawn.size(), (empty.size(), drawn.size())
for part in ('_title_band', '_title_label', '_title_commands'):
    assert getattr(empty, part).geometry() == getattr(drawn, part).geometry(), part
for button in ('settings_button', 'close_button'):
    assert getattr(empty, button).geometry() == getattr(drawn, button).geometry(), button
# Two lines each, and the second whole: the strip is as tall as its rich
# text stacks two lines, so no descender is cut by the label's edge.
for card in (drawn, empty):
    assert strip_line_count(card) == 2, card._title_label.text()
    assert card._title_label.height() >= card._title_label.sizeHint().height()
assert 'Choose&nbsp;a&nbsp;signal&nbsp;in&nbsp;Setting' in empty._title_label.text()
# The body is the blank frame; nothing in it claims the room.
assert not [child for child in empty.findChildren(QtWidgets.QLabel)
            if child.isVisible() and child is not empty._title_label]

# A picked signal with nothing published yet is not "pick a signal".
empty.set_panel_projection(dict(STATE, signal='@logic/cm/frames'), {})
assert 'No&nbsp;data&nbsp;yet' in empty._title_label.text()
assert strip_line_count(empty) == 2

# A card whose picture leaves is the empty frame again, saying why.
widget = drawn.surface
drawn.set_surface(None)
widget.close_adapter()
assert drawn.size() == empty.size()
assert drawn._title_label.geometry() == empty._title_label.geometry()
host.close(timeout=10)
"""
    )


def test_an_empty_card_is_held_and_inserted_like_a_drawn_one(run_qt) -> None:
    """The strip carries every card; an empty card's blank body carries it too.

    A drawn card's body is its plot's, so it is held by its strip alone.
    An empty card's body is a blank frame, nine tenths of the card, so the
    whole card carries it, under the open hand.  Carried by the strip to
    each insert position, an empty card lands exactly as a drawn one
    carried the same way.  Carried by a point low in its body, it lands
    where it is seen -- the row its strip is in -- exactly as when held by
    its strip at the same x: aimed at the pointer, a card held low and
    shown mostly over a row was previewed and put down a row below it.
    """

    run_qt(
        _PROLOGUE
        + """
GAP = 10

def new_board(kinds):
    window = QtWidgets.QWidget()
    board = ConsoleBoardView(window, metrics=BoardMetrics(GAP))
    cards = tuple(new_card(f'panel-{index}', kind == 'drawn') for index, kind in enumerate(kinds))
    w, h = cards[0].width(), cards[0].height()
    # A row of room under the cards, for a card carried by its body below
    # the last row.
    window.resize(2 * w + 3 * GAP, 3 * h + 4 * GAP)
    board.setGeometry(window.rect())
    board.set_cards(cards)
    window.show()
    wait_until(lambda: all(card._surface_framed for card, kind in zip(cards, kinds) if kind == 'drawn'))
    events = []
    board.order_committed.connect(events.append)
    starts = []
    for card in cards:
        card.drag_started.connect(lambda _point, card=card: starts.append(card.panel_id))
    return window, board, cards, events, starts

def carry(board, card, held, slot):
    # Through the window, as the pointer is: the press reaches whatever child
    # is under it first, and only a card that took it follows the moves.
    window = board.window()
    handle = window.windowHandle()
    start = card.mapTo(window, held)
    end = board.mapTo(window, QtCore.QPoint(*slot) + held)
    QtTest.QTest.mouseMove(handle, start)
    QtTest.QTest.mousePress(handle, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, start)
    for step in range(1, 9):
        QtTest.QTest.mouseMove(handle, start + (end - start) * step / 8)
        app.processEvents()
    QtTest.QTest.mouseRelease(handle, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, end)
    app.processEvents()

def retire(window, cards):
    for card in cards:
        surface = card.surface
        card.set_surface(None)
        if surface is not None:
            surface.close_adapter()
    window.close()
    app.processEvents()

# A press in a drawn card's body carries nothing, wherever the pointer then
# goes; an empty card's body is its grip, under the open hand.
window, board, cards, events, starts = new_board(('drawn', 'empty', 'drawn', 'empty'))
w, h = cards[0].width(), cards[0].height()
slots = ((GAP, GAP), (2 * GAP + w, GAP), (GAP, 2 * GAP + h), (2 * GAP + w, 2 * GAP + h))
assert tuple(card.pos() for card in cards) == tuple(QtCore.QPoint(*slot) for slot in slots)
body = lambda card: QtCore.QPoint(card.width() // 2, card.height() // 2 + card._title_band.height() // 2)
drawn, empty = cards[0], cards[1]
assert drawn.cursor().shape() == QtCore.Qt.ArrowCursor
assert empty.cursor().shape() == QtCore.Qt.OpenHandCursor
assert empty.handle_rect() == empty.rect() and drawn.handle_rect() == drawn.strip_rect()
before = drawn.pos()
carry(board, drawn, body(drawn), (before.x() + w // 2, before.y() + h // 2))
assert drawn.pos() == before, (drawn.pos(), before)
assert starts == [] and events == [], (starts, events)
carry(board, empty, body(empty), slots[2])
assert starts == ['panel-1'], starts
assert empty.pos() == QtCore.QPoint(*slots[2]), empty.pos()
assert empty.cursor().shape() == QtCore.Qt.OpenHandCursor
retire(window, cards)

# The same strip gesture to each insert position: the empty card and the
# drawn card it stands in for end in the same place, in the same order.
# The empty card held 80 % down its body and shown a third of a card below
# the slot, its strip still in the slot's row, lands there too: the hand
# is then in the row below.
for target in (0, 2, 3):
    outcomes = []
    for kinds, by_body in (
        (('drawn', 'empty', 'drawn', 'empty'), False),
        (('empty', 'drawn', 'empty', 'drawn'), False),
        (('drawn', 'empty', 'drawn', 'empty'), True),
    ):
        window, board, cards, events, starts = new_board(kinds)
        mover = cards[1]
        if by_body:
            held = QtCore.QPoint(GAP + 2, mover.height() * 4 // 5)
            seen = (slots[target][0], slots[target][1] + h // 3)
            assert seen[1] + held.y() >= slots[target][1] + h + GAP
        else:
            held = QtCore.QPoint(GAP + 2, mover._title_band.geometry().center().y())
            seen = slots[target]
        carry(board, mover, held, seen)
        assert starts == ['panel-1'], (kinds, target, starts)
        # The carried card lands in the slot it was shown in.
        assert mover.pos() == QtCore.QPoint(*slots[target]), (kinds, target, mover.pos())
        assert events, (kinds, target)
        outcomes.append((events[-1], tuple(card.geometry().getRect() for card in cards)))
        retire(window, cards)
    assert outcomes[0] == outcomes[1] == outcomes[2], (target, outcomes)
host.close(timeout=10)
"""
    )


def test_the_header_remover_is_a_quiet_ghost_and_the_setting_close_a_plain_one(run_qt) -> None:
    """The header x is the console's one-glyph remover: no face at rest, the
    white-on-red confirm once armed, and quiet again when it disarms.  The
    Setting frame's x only closes the frame: the quiet look of a tab's
    close x -- no face, MUTED_INK, never red -- square at a control's
    height, ending where the frame's fields end."""

    run_qt(
        """
from PyQt5 import QtCore, QtGui, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import BG, MUTED_INK, RED_DARK, SURFACE, control_height
from zlc_ui.console import PanelCardView

app = ensure_qt_app(['header-remover'])
card = PanelCardView('panel-1', 'Plot')
card.set_size_choices(('2x2',), '2x2')
card.show(); app.processEvents()

def face(widget, surface):
    # A pixel of the button's face, off its glyph, over the surface it sits on.
    image = surface.grab().toImage()
    return image.pixelColor(widget.mapTo(surface, QtCore.QPoint(3, 3))).name()

remover = card.close_button
band = card._title_band
assert remover.role() == 'destructive' and remover.property('glyph')
assert face(remover, band) == QtGui.QColor(BG).name(), face(remover, band)
QtTest.QTest.mouseClick(remover, QtCore.Qt.LeftButton)
assert remover.role() == 'destructive_primary'
assert face(remover, band) == QtGui.QColor(RED_DARK).name()
card._disarm_header_remove()
app.processEvents()
assert remover.role() == 'destructive'
assert face(remover, band) == QtGui.QColor(BG).name()

card._open_settings(); app.processEvents()
closer = card._settings_close_button
popup = card._settings_popup
assert closer.role() == 'quiet' and closer._glyph_ink() == MUTED_INK
assert closer.size() == QtCore.QSize(control_height(), control_height())
assert face(closer, popup) == QtGui.QColor(SURFACE).name(), face(closer, popup)
field = card._settings_form.widget_for('title')
assert closer.mapTo(popup, QtCore.QPoint(closer.width(), 0)).x() == (
    field.mapTo(popup, QtCore.QPoint(field.width(), 0)).x())
card.retire_settings_popup(); card.close(); app.processEvents()
"""
    )


# A Monitor page without the plotting package: empty cards, sized as the
# frame this package plans for them, on a board inside the Fluent scroll
# area the console and the FigureViewer both mount it in.  Every gesture and
# every resize goes through the window.
_MONITOR = """
import time

from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.board import BoardMetrics
from zlc_ui.fluent import ACCENT_TINT, FluentButton, FluentScrollArea
from zlc_ui.console import ConsoleBoardView, PanelCardView

app = ensure_qt_app(['monitor-board'])
GAP = 10

def wait_until(condition, timeout=10.0, message='condition did not settle'):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError(message)

def settle():
    for _ in range(5):
        app.processEvents()

def empty_card(panel_id):
    card = PanelCardView(panel_id, 'Plot')
    card.set_size_choices(('2x2',), '2x2')
    return card

def monitor(count):
    window = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    scroll = FluentScrollArea()
    board = ConsoleBoardView(metrics=BoardMetrics(GAP))
    scroll.setWidget(board)
    layout.addWidget(scroll)
    cards = tuple(empty_card(f'panel-{index}') for index in range(count))
    return window, scroll, board, cards

def places(cards):
    return [(card.x(), card.y()) for card in cards]

def grid(w, h, count, columns=2):
    return [
        (GAP + (index % columns) * (w + GAP), GAP + (index // columns) * (h + GAP))
        for index in range(count)
    ]

def strip_point(card, x):
    return QtCore.QPoint(x, card._title_band.geometry().center().y())

def press_and_carry(window, card, held, end, steps=8):
    handle = window.windowHandle()
    start = card.mapTo(window, held)
    QtTest.QTest.mouseMove(handle, start)
    QtTest.QTest.mousePress(handle, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, start)
    for step in range(1, steps + 1):
        QtTest.QTest.mouseMove(handle, start + (end - start) * step / steps)
        app.processEvents()

def release(window, end):
    QtTest.QTest.mouseRelease(window.windowHandle(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, end)
    settle()
"""


def test_panels_added_before_the_board_is_laid_out_tile_its_real_width(run_qt) -> None:
    """Cards nobody has put anywhere follow the width the board has NOW.

    A card's place was recorded when it arrived, at whatever width the board
    had -- before the window's first layout that is Qt's default, one card
    wide -- and gravity, which only moves cards up and left, kept that column
    on a board two cards wide for good: three empty panels stacked down the
    left with the rest of the Monitor empty.
    """

    run_qt(
        _MONITOR
        + """
window, scroll, board, cards = monitor(3)
w, h = cards[0].width(), cards[0].height()
board.set_cards(cards)
assert board.width() < 2 * w + 3 * GAP, 'the board was already wide enough before its layout'
assert places(cards) == [(GAP, GAP), (GAP, 2 * GAP + h), (GAP, 3 * GAP + 2 * h)]
wide = (2 * w + 3 * GAP + 40, 3 * h)
narrow = (w + 2 * GAP + 40, 3 * h)
window.resize(*wide)
window.show()
settle()
assert places(cards) == grid(w, h, 3), places(cards)

# A narrow window packs one column; widening restores the tiling.
window.resize(*narrow); settle()
assert {x for x, _y in places(cards)} == {GAP}, places(cards)
window.resize(*wide); settle()
assert places(cards) == grid(w, h, 3), places(cards)

# A panel added while the window is narrow tiles once it widens.
window.resize(*narrow); settle()
more = cards + (empty_card('panel-3'),)
board.set_cards(more); settle()
assert {x for x, _y in places(more)} == {GAP}, places(more)
window.resize(*wide); settle()
assert places(more) == grid(w, h, 4), places(more)
window.close(); settle()
"""
    )


def test_a_carried_card_lands_in_the_pointers_column_wherever_its_strip_is_held(run_qt) -> None:
    """Where the pointer lets go decides the column, and the board shows it first.

    The slot was the one nearest the card's TOP-LEFT, so the same release
    just inside the second column sent a card held near the left of its
    strip there and a card held near Setting back to the first column, and
    nothing on screen said which until it landed.  Now, while it is carried,
    the others stand where the drop will leave them and the slot is drawn
    in the accent tint wherever the card does not cover it.
    """

    run_qt(
        _MONITOR
        + """
outcomes = []
for hold in ('left', 'right'):
    window, scroll, board, cards = monitor(4)
    w, h = cards[0].width(), cards[0].height()
    window.resize(2 * w + 3 * GAP + 40, 3 * h)
    window.show(); settle()
    board.set_cards(cards); settle()
    slots = grid(w, h, 4)
    assert places(cards) == slots
    mover = cards[0]
    near_setting = mover.settings_button.mapTo(mover, QtCore.QPoint(0, 0)).x() - 12
    held = strip_point(mover, 14 if hold == 'left' else near_setting)
    # Just inside the last slot, whichever part of the strip is held.
    end = board.mapTo(window, QtCore.QPoint(slots[3][0] + 12, slots[3][1] + 12))
    press_and_carry(window, mover, held, end)
    assert mover.mapTo(window, held) == end, (hold, mover.pos())
    # The others fell north-west into the hole it left ...
    assert places(cards[1:]) == [slots[0], slots[2], slots[1]], (hold, places(cards[1:]))
    # ... and the slot is drawn wherever the carried card leaves it showing.
    target = QtCore.QRect(*slots[3], w, h)
    picture = board.grab().toImage()
    inside = [
        point for point in (
            QtCore.QPoint(target.right() - 8, target.bottom() - 4),
            QtCore.QPoint(target.left() + 4, target.bottom() - 8),
            QtCore.QPoint(target.right() - 4, target.center().y()),
            QtCore.QPoint(target.left() + 4, target.center().y()),
        )
        if not mover.geometry().adjusted(-1, -1, 1, 1).contains(point)
    ]
    assert inside, (hold, mover.geometry())
    for point in inside:
        colour = picture.pixelColor(point).name().upper()
        assert colour == ACCENT_TINT.upper(), (hold, point, colour)
    events = []
    board.order_committed.connect(events.append)
    release(window, end)
    assert events and events[-1] == ('panel-1', 'panel-3', 'panel-2', 'panel-0'), (hold, events)
    assert places(cards) == [slots[3], slots[0], slots[2], slots[1]], (hold, places(cards))
    # The presenter hands the cards back in the committed order: nothing moves.
    by_id = {card.panel_id: card for card in cards}
    before = places(cards)
    board.set_cards(tuple(by_id[panel_id] for panel_id in events[-1])); settle()
    assert places(cards) == before
    outcomes.append(before)
    window.close(); settle()
assert outcomes[0] == outcomes[1], outcomes
"""
    )


def test_a_card_carried_to_the_viewport_edge_scrolls_the_board_under_it(run_qt) -> None:
    """A card can be carried to a row that is off screen.

    Nothing scrolled the board while a card was carried, so the furthest a
    card could go in one gesture was the bottom of the viewport.  Held at the
    viewport's edge, the board scrolls under the pointer; the card stays in
    the operator's hand and lands in the row the pointer reached.
    """

    run_qt(
        _MONITOR
        + """
window, scroll, board, cards = monitor(6)
w, h = cards[0].width(), cards[0].height()
window.resize(2 * w + 3 * GAP + 40, h + h // 2)
window.show(); settle()
board.set_cards(cards); settle()
viewport = scroll.viewport()
seen = viewport.height()
assert places(cards) == grid(w, h, 6)
bar = scroll.verticalScrollBar()
assert bar.value() == 0
mover = cards[0]
held = strip_point(mover, 14)
edge = viewport.mapTo(window, QtCore.QPoint(mover.mapTo(viewport, held).x(), viewport.height() - 4))
press_and_carry(window, mover, held, edge)
wait_until(lambda: bar.value() >= h, message='the board did not scroll under the carried card')
# Still in the operator's hand: the point it is held by is under the pointer.
assert mover.mapTo(window, held) == edge, (mover.pos(), bar.value())
release(window, edge)
assert mover.y() > seen, (mover.pos(), seen)
assert mover.y() in {y for _x, y in grid(w, h, 8)}, mover.pos()
window.close(); settle()
"""
    )


def test_the_strip_commands_are_page_buttons_and_its_words_start_at_the_board_text_edge(run_qt) -> None:
    """Setting is as tall as every other button and the remover is square at
    that height, centred in the strip, and both stay on screen on a card
    wider than its viewport; and the board says where a card's words start,
    so an empty board's note can start there too."""

    run_qt(
        _MONITOR
        + """
from zlc_ui.fluent import set_fluent_scale

# At the full scale, where a button's own height and the old strip's
# 26-px command are different numbers (the offscreen screen's automatic
# scale floors both at 22).
set_fluent_scale(1.0)
window, scroll, board, cards = monitor(1)
window.resize(800, 600)
board.set_cards(cards)
page_button = FluentButton('Add Panel')
window.layout().addWidget(page_button)
window.show(); settle()
card = cards[0]
setting, remover = card.settings_button, card.close_button
assert setting.height() == page_button.height(), (setting.height(), page_button.height())
assert remover.width() == remover.height() == setting.height(), remover.size()
band = card._title_band
for button in (setting, remover):
    # Centred to the logical pixel: an odd remainder cannot split evenly.
    top = button.mapTo(band, QtCore.QPoint(0, 0)).y()
    assert abs(top - (band.height() - top - button.height())) <= 1, (button.text(), top, band.height())
assert card._title_label.mapTo(board, QtCore.QPoint(0, 0)).x() == board.text_left()
# The two commands stand the row's one gap apart.
from zlc_ui.fluent import card_pad, row_gap
assert remover.x() - (setting.x() + setting.width()) == row_gap()
# On a card wider than what the viewport shows, the two ride the viewport's
# right edge, as far in from it as from the card's own end; scrolled to
# that end, or with room for the whole card again, they are back in its
# corner.
def right_in(widget, top):
    return widget.mapTo(top, QtCore.QPoint(widget.width(), 0)).x()
corner = card.width() - right_in(remover, card)
viewport, sideways = scroll.viewport(), scroll.horizontalScrollBar()
window.resize(card.width() // 2, 600); settle()
assert sideways.maximum() > 0
assert viewport.width() - right_in(remover, viewport) == corner
assert right_in(setting, viewport) < right_in(remover, viewport)
sideways.setValue(sideways.maximum()); settle()
assert card.width() - right_in(remover, card) == corner
sideways.setValue(0); settle()
assert viewport.width() - right_in(remover, viewport) == corner
window.resize(800, 600); settle()
assert card.width() - right_in(remover, card) == corner
window.close(); settle()
# A board's own gap is the card pad: its first card starts where a white
# page's content does, and its words one more card pad in.
default = ConsoleBoardView()
lone = empty_card('lone')
default.resize(800, 600)
default.set_cards((lone,))
default.show(); settle()
assert (lone.x(), lone.y()) == (card_pad(), card_pad())
assert default.text_left() == card_pad() + PanelCardView.strip_text_inset()
default.close(); settle()
"""
    )
