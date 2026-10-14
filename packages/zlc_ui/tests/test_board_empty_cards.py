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
assert 'Pick&nbsp;a&nbsp;signal&nbsp;in&nbsp;Setting' in empty._title_label.text()
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
    """The strip carries every card; a body carries none.

    An empty card's body let its presses through to the card, so it was
    picked up mid-body -- half a card below its top-left, which is what a
    drop snaps -- while a drawn card's body is its plot's and it could only
    be held by its strip.  Aimed with the pointer at a slot, the empty card
    then landed a row up.  Carried by the strip to each insert position,
    an empty card must land exactly as a drawn one carried the same way.
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
    window.resize(2 * w + 3 * GAP, 2 * h + 3 * GAP)
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

# A press in the body carries neither card, wherever the pointer then goes.
window, board, cards, events, starts = new_board(('drawn', 'empty', 'drawn', 'empty'))
w, h = cards[0].width(), cards[0].height()
slots = ((GAP, GAP), (2 * GAP + w, GAP), (GAP, 2 * GAP + h), (2 * GAP + w, 2 * GAP + h))
assert tuple(card.pos() for card in cards) == tuple(QtCore.QPoint(*slot) for slot in slots)
for card in cards[:2]:
    body = QtCore.QPoint(card.width() // 2, card.height() // 2 + card._title_band.height() // 2)
    before = card.pos()
    carry(board, card, body, (before.x() + w // 2, before.y() + h // 2))
    assert card.pos() == before, (card.panel_id, card.pos(), before)
assert starts == [] and events == [], (starts, events)
retire(window, cards)

# The same strip gesture to each insert position: the empty card and the
# drawn card it stands in for end in the same place, in the same order.
for target in (0, 2, 3):
    outcomes = []
    for kinds in (('drawn', 'empty', 'drawn', 'empty'), ('empty', 'drawn', 'empty', 'drawn')):
        window, board, cards, events, starts = new_board(kinds)
        mover = cards[1]
        held = QtCore.QPoint(GAP + 2, mover._title_band.geometry().center().y())
        carry(board, mover, held, slots[target])
        assert starts == ['panel-1'], (kinds, target, starts)
        # The place the drop chose IS the slot the pointer carried it to.
        assert board._authored['panel-1'] == slots[target], (kinds, target, board._authored)
        assert events, (kinds, target)
        outcomes.append((events[-1], tuple(card.geometry().getRect() for card in cards)))
        retire(window, cards)
    assert outcomes[0] == outcomes[1], (target, outcomes)
host.close(timeout=10)
"""
    )


def test_the_header_remover_is_a_quiet_ghost_and_the_setting_close_a_plain_one(run_qt) -> None:
    """The header x is the console's one-glyph remover: no face at rest, the
    white-on-red confirm once armed, and quiet again when it disarms.  The
    Setting frame's x only closes the frame: no face, and never red."""

    run_qt(
        """
from PyQt5 import QtCore, QtGui, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import BG, RED_DARK, SURFACE
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
assert closer.role() == 'standard'
assert face(closer, popup) == QtGui.QColor(SURFACE).name(), face(closer, popup)
card.retire_settings_popup(); card.close(); app.processEvents()
"""
    )
