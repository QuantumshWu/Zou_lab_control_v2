"""Cards of MIXED sizes carried on the real Monitor board.

Every gesture goes through the window, the way the operator's pointer does,
on a board mounted in the Fluent scroll area as the Task Console mounts it,
with cards framed as the composition layer sizes the presets.  After every
drag the board must stand where its preview stood -- the slot drawn under
the carried card and every other card -- and must still stand there once
the presenter hands the cards back in the order the drop committed.

Board drag tests use mixed sizes: on a board of one size every edge is a
column or a row line and every card covers one cell, which hides every
defect these tests exist for.
"""

from __future__ import annotations

_MONITOR = """
import random

from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_plot import PANEL_SIZE_NAMES
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentScrollArea, card_pad
from zlc_ui.console import ConsoleBoardView, PanelCardView
from zlc_workbench.panel_sizes import install

app = ensure_qt_app(['mixed-board'])
# The composition layer's one size answer: each card is the frame the
# plotting package plans for its preset.
install()
GAP = card_pad()

def settle():
    for _ in range(5):
        app.processEvents()

def new_card(panel_id, size):
    card = PanelCardView(panel_id, 'Plot')
    card.set_size_choices(PANEL_SIZE_NAMES, size)
    return card

def measured(size):
    card = new_card('probe', size)
    answer = (card.width(), card.height())
    card.deleteLater()
    return answer

SIZE = {size: measured(size) for size in ('1x2', '2x2', '4x2', '2x4', '4x4')}
settle()

def col_after(*sizes):
    # The column a card stands in to the right of these cards, side by side.
    return GAP + sum(SIZE[size][0] + GAP for size in sizes)

def row_after(*sizes):
    # The row a card stands in under these cards, stacked.
    return GAP + sum(SIZE[size][1] + GAP for size in sizes)

# Three widths: a 4x4 beside two 2x2; a 4x4 beside one; two 2x2 side by side
# but never a 4x4 beside a 2x2.
WIDE = col_after('4x4', '2x2', '2x2') + 24
MID = col_after('4x4', '2x2') + 35
NARROW = col_after('2x2', '2x2') + 75
assert NARROW < col_after('4x4', '2x2') - GAP


class Monitor:
    # One Monitor page: the board in a Fluent scroll area that keeps the
    # vertical bar's gutter, as the console mounts it.

    def __init__(self, sizes, board_w, height=4000):
        self.window = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.window)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scroll = FluentScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.board = ConsoleBoardView()
        self.scroll.setWidget(self.board)
        self.scroll.reserve_vertical_gutter()
        layout.addWidget(self.scroll)
        self.gutter = self.scroll.verticalScrollBar().sizeHint().width()
        self.cards = {f'p{index}': new_card(f'p{index}', size) for index, size in enumerate(sizes)}
        self.window.resize(board_w + self.gutter, height)
        self.window.show()
        settle()
        self.board.set_cards(tuple(self.cards.values()))
        settle()
        assert self.board.width() == board_w, (self.board.width(), board_w)
        self.commits = []
        self.board.order_committed.connect(self.commits.append)

    def places(self):
        return {panel_id: (card.x(), card.y()) for panel_id, card in self.cards.items()}

    def resize(self, board_w):
        self.window.resize(board_w + self.gutter, self.window.height())
        settle()
        assert self.board.width() == board_w, (self.board.width(), board_w)

    def held(self, card, hold):
        strip = card._title_band.geometry().center().y()
        if hold == 'strip_left':
            return QtCore.QPoint(14, strip)
        if hold == 'strip_right':
            return QtCore.QPoint(card.settings_button.mapTo(card, QtCore.QPoint(0, 0)).x() - 12, strip)
        assert hold == 'body' and card.handle_rect() == card.rect(), 'only an empty card is held by its body'
        return QtCore.QPoint(card.width() // 2, card.height() // 2)

    def drag(self, panel_id, hold, seen, via=()):
        # Carry one card until it is drawn with its top-left at ``seen`` (by
        # way of ``via``), and let go.  The board must stand as its preview
        # stood, and stay so when the presenter commits the new order.
        card = self.cards[panel_id]
        handle = self.window.windowHandle()
        held = self.held(card, hold)
        start = card.mapTo(self.window, held)
        widths = []
        QtTest.QTest.mouseMove(handle, start)
        QtTest.QTest.mousePress(handle, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, start)
        for target in (*via, seen):
            end = self.board.mapTo(self.window, QtCore.QPoint(*target) + held)
            for step in range(1, 9):
                QtTest.QTest.mouseMove(handle, start + (end - start) * step / 8)
                app.processEvents()
                widths.append(self.board.width())
            start = end
        assert card.pos() == QtCore.QPoint(*seen), (card.pos(), seen)
        drop = QtCore.QRect(self.board._drop_rect)
        preview = {pid: place for pid, place in self.places().items() if pid != panel_id}
        QtTest.QTest.mouseRelease(handle, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, start)
        settle()
        widths.append(self.board.width())
        result = self.places()
        assert drop == card.geometry(), (drop, card.geometry())
        assert {pid: place for pid, place in result.items() if pid != panel_id} == preview, (preview, result)
        rects = [card.geometry().adjusted(-GAP, -GAP, GAP, GAP) for card in self.cards.values()]
        for index, rect in enumerate(rects):
            for other in list(self.cards.values())[index + 1:]:
                assert not rect.intersects(other.geometry()), result
        committed = self.commits[-1]
        assert sorted(committed) == sorted(self.cards)
        self.board.set_cards(tuple(self.cards[pid] for pid in committed))
        settle()
        assert self.places() == result, (self.places(), result)
        self.widths = widths
        return result

    def close(self):
        self.window.close()
        self.window.deleteLater()
        settle()
"""


def test_a_card_let_go_onto_another_swaps_with_it(run_qt) -> None:
    """Let go onto a card, the two change places: that card takes the place
    the carried one left and the carried one takes the slot -- on a grid of
    one size, and on mixed sizes either way round, where the card further up
    and left keeps its place and the other moves just clear of it.

    The cards in the way used to make room by hand-made cases -- back to
    where they stood, along the row, into the room left beside them -- so a
    card let go onto a 4x4 was put back where it came from, and a 4x4 let
    go onto the 2x4 above it changed nothing.
    """

    run_qt(
        _MONITOR
        + """
# A 2x2 grid: bottom-left onto bottom-right, then top-left onto bottom-right.
for carried in ('p2', 'p0'):
    board = Monitor(('2x2',) * 4, NARROW)
    before = board.places()
    assert before == {
        'p0': (GAP, GAP), 'p1': (col_after('2x2'), GAP),
        'p2': (GAP, row_after('2x2')), 'p3': (col_after('2x2'), row_after('2x2')),
    }, before
    board.drag(carried, 'strip_left', (before['p3'][0] + 10, before['p3'][1] + 10))
    assert board.places() == {**before, carried: before['p3'], 'p3': before[carried]}, board.places()
    board.close()

# A 2x2 beside a 4x4, a 2x4 under them: either let go onto the other, the
# 4x4 ends on the left and the 2x2 right of it.
for carried, onto, hold in (('p0', 'p1', 'strip_left'), ('p1', 'p0', 'strip_right')):
    board = Monitor(('2x2', '4x4', '2x4'), MID)
    before = board.places()
    assert before == {'p0': (GAP, GAP), 'p1': (col_after('2x2'), GAP), 'p2': (GAP, row_after('4x4'))}, before
    board.drag(carried, hold, (before[onto][0] + 10, before[onto][1] + 10))
    assert board.places() == {
        'p1': (GAP, GAP), 'p0': (col_after('4x4'), GAP), 'p2': before['p2'],
    }, (carried, board.places())
    board.close()

# A 2x4 above a 4x4 on a board one column wide for them: either let go onto
# the other, the 4x4 ends on top and the 2x4 under it.
for carried, onto in (('p0', 'p1'), ('p1', 'p0')):
    board = Monitor(('2x4', '4x4', '2x2'), NARROW)
    before = board.places()
    assert before == {
        'p0': (GAP, GAP), 'p1': (GAP, row_after('2x4')), 'p2': (GAP, row_after('2x4', '4x4')),
    }, before
    board.drag(carried, 'strip_left', (before[onto][0] + 10, before[onto][1] + 10))
    assert board.places() == {
        'p1': (GAP, GAP), 'p0': (GAP, row_after('4x4')), 'p2': (GAP, row_after('4x4', '2x4')),
    }, (carried, board.places())
    board.close()
"""
    )


def test_cards_nobody_let_go_onto_stay_where_they_are(run_qt) -> None:
    """Only the card let go onto and the cards in the way move; a 4x4 beside
    them stays put.

    A card whose place another covered took the first free hole anywhere on
    the board, and every card no drop had placed was list-flowed again on
    every drop: a 2x2 dropped by a 1x2 landed in the far top corner past a
    4x4, and a 4x4 nobody touched travelled from one corner to the other.
    """

    run_qt(
        _MONITOR
        + """
# Onto the 2x2 under a 1x2: the two change places; the 4x4 and the 2x4
# under it stay.
board = Monitor(('2x2', '1x2', '4x4', '2x2', '2x4'), WIDE)
before = board.places()
assert before == {
    'p0': (GAP, GAP), 'p1': (col_after('2x2'), GAP), 'p2': (col_after('2x2', '1x2'), GAP),
    'p3': (col_after('2x2'), row_after('1x2')), 'p4': (col_after('2x2', '1x2'), row_after('4x4')),
}, before
board.drag('p0', 'strip_left', (before['p3'][0] + 10, before['p3'][1] + 10))
assert board.places() == {**before, 'p0': before['p3'], 'p3': before['p0']}, board.places()
board.close()

# The 4x4 in the top-right corner is not in the way, so it does not move.
board = Monitor(('2x4', '4x4', '2x2'), WIDE)
before = board.places()
assert before == {'p0': (GAP, GAP), 'p1': (col_after('2x4'), GAP), 'p2': (GAP, row_after('2x4'))}, before
board.drag('p2', 'strip_left', (GAP + 3, GAP + 3))
assert board.places() == {'p2': (GAP, GAP), 'p0': (GAP, row_after('2x2')), 'p1': before['p1']}, board.places()
board.close()
"""
    )


def test_where_a_card_is_held_does_not_change_where_it_lands(run_qt) -> None:
    """The same drawn place gives the same landing, held anywhere.

    The column was the pointer's: a 2x4 held near Setting, drawn at the
    top-left, landed a column to the right of one held at the left of its
    strip, and an empty 4x4 carried by its body landed a column right of
    where it was seen.
    """

    run_qt(
        _MONITOR
        + """
landed = []
for hold in ('strip_left', 'strip_right'):
    board = Monitor(('2x2', '4x4', '2x4'), MID)
    board.drag('p2', hold, (GAP + 3, GAP + 3))
    landed.append(board.places())
    board.close()
assert landed[0] == landed[1] == {
    'p2': (GAP, GAP), 'p0': (GAP, row_after('2x4')), 'p1': (col_after('2x2'), row_after('2x4')),
}, landed

landed = []
for hold in ('body', 'strip_left'):
    board = Monitor(('2x2', '1x2', '4x4'), MID)
    assert board.places()['p2'] == (col_after('2x2'), row_after('1x2'))
    # Drawn over the first column, nine tenths of it: onto the 2x2, which
    # takes the 4x4's place and moves just clear of it, under the 1x2.
    board.drag('p2', hold, (SIZE['4x4'][0] // 8, GAP))
    landed.append(board.places())
    board.close()
assert landed[0] == landed[1] == {
    'p2': (GAP, GAP), 'p1': (col_after('4x4'), GAP), 'p0': (col_after('4x4'), row_after('1x2')),
}, landed
"""
    )


def test_a_card_let_go_on_empty_space_under_a_tall_card_stays_under_it(run_qt) -> None:
    """Let go under a card, a card stays under it: it does not slide along
    that card's bottom and rise into a hole beside it.

    A 2x2 released under a 4x4 slid along the 4x4's bottom and rose into
    the hole under the shorter 2x4 -- back where it came from; a 1x2 let go
    over the lower half of a 4x4 was refused outright.
    """

    run_qt(
        _MONITOR
        + """
board = Monitor(('2x4', '4x4', '2x2'), WIDE)
before = board.places()
assert before == {'p0': (GAP, GAP), 'p1': (col_after('2x4'), GAP), 'p2': (GAP, row_after('2x4'))}, before
board.drag('p2', 'strip_left', (col_after('2x4') + 10, row_after('4x4') + 10))
assert board.places() == {**before, 'p2': (col_after('2x4'), row_after('4x4'))}, board.places()
board.close()

board = Monitor(('1x2', '4x4', '1x2'), WIDE)
before = board.places()
assert before == {
    'p0': (GAP, GAP), 'p1': (col_after('1x2'), GAP), 'p2': (col_after('1x2', '4x4'), GAP),
}, before
# Over the lower half of the 4x4.
board.drag('p2', 'strip_left', (col_after('1x2') + 49, GAP + SIZE['4x4'][1] * 3 // 4))
assert board.places() == {**before, 'p2': (col_after('1x2'), row_after('4x4'))}, board.places()
board.close()
"""
    )


def test_each_drop_leaves_every_card_where_it_showed_it(run_qt) -> None:
    """Drops in a row: each one records where it left every card, so a later
    drop starts from the board on screen.

    A drop recorded only the carried card's place.  The others' places went
    stale, and a later drop that nothing tied to an earlier card sent that
    card flying across the board, or was refused.
    """

    run_qt(
        _MONITOR
        + """
board = Monitor(('2x4', '4x4', '2x2'), WIDE)
# The 2x4 onto the 2x2: the two change places.
board.drag('p0', 'strip_left', (GAP, row_after('2x4')))
assert board.places() == {
    'p2': (GAP, GAP), 'p1': (col_after('2x4'), GAP), 'p0': (GAP, row_after('2x2')),
}, board.places()
# The 4x4 onto the 2x2 at the top-left: the two change places, and the 2x4
# in the 4x4's way moves along its row, just clear of it.
board.drag('p1', 'strip_left', (GAP, GAP))
assert board.places() == {
    'p1': (GAP, GAP), 'p2': (col_after('4x4'), GAP), 'p0': (col_after('4x4'), row_after('2x2')),
}, board.places()
# The 2x2 on the empty space under the 4x4: nothing else moves but the 2x4
# rising into its place.
board.drag('p2', 'strip_left', (GAP + 5, row_after('4x4') + 5))
assert board.places() == {
    'p1': (GAP, GAP), 'p0': (col_after('4x4'), GAP), 'p2': (GAP, row_after('4x4')),
}, board.places()
board.close()
"""
    )


def test_putting_a_card_back_changes_nothing_and_the_board_keeps_its_width(run_qt) -> None:
    """Picked up and put back, a card leaves the board as it was -- including
    on a board just wide enough for a 2x4 beside a 2x2.

    Carrying a card reserves a row for it, which brought the vertical bar in;
    the board lost the bar's width under the drag and committed a narrower
    layout on release.  The Monitor keeps the bar's gutter, so the board is
    as wide with the bar as without it.
    """

    run_qt(
        _MONITOR
        + """
width = col_after('2x4', '2x2') + 5
# Tall enough for the board, not for the row a carried card reserves.
height = row_after('2x4', '4x4') + 2 * GAP
board = Monitor(('2x4', '2x2', '4x4', '2x2'), width, height)
before = board.places()
assert before == {
    'p0': (GAP, GAP), 'p1': (col_after('2x4'), GAP), 'p2': (GAP, row_after('2x4')),
    'p3': (col_after('4x4'), row_after('2x4')),
}, before
assert not board.scroll.verticalScrollBar().isVisible()
for panel_id in ('p1', 'p3', 'p0', 'p2'):
    at = before[panel_id]
    board.drag(panel_id, 'strip_left', at, via=((at[0] + 60, at[1] + 60),))
    assert set(board.widths) == {width}, (panel_id, set(board.widths))
    assert board.places() == before, (panel_id, board.places())
board.close()
"""
    )


def test_narrowing_packs_the_arrangement_and_widening_restores_it(run_qt) -> None:
    """Each column count keeps the arrangement its last drop left.  A drop
    made in a narrow window does not overwrite the wide arrangement, and a
    panel added afterwards takes the first free slot."""

    run_qt(
        _MONITOR
        + """
board = Monitor(('4x4', '2x2', '2x2', '2x4', '2x2'), WIDE)
# The last 2x2 onto the 4x4: the two change places, and the rest fall.
board.drag('p4', 'strip_left', (GAP, GAP))
wide = board.places()
assert wide == {
    'p4': (GAP, GAP), 'p1': (col_after('2x2'), GAP), 'p2': (col_after('2x2', '2x2'), GAP),
    'p0': (GAP, row_after('2x2')), 'p3': (col_after('4x4'), row_after('2x2')),
}, wide
board.resize(NARROW)
assert board.places() != wide
board.drag('p3', 'strip_left', (GAP, GAP))
narrow = board.places()
assert narrow['p3'] == (GAP, GAP), narrow
board.resize(WIDE)
assert board.places() == wide, board.places()
board.resize(NARROW)
assert board.places() == narrow, board.places()
board.resize(WIDE)
# Added after the arrangement: the first free slot, and nothing else moves.
extra = new_card('p5', '1x2')
board.cards['p5'] = extra
board.board.set_cards(tuple(board.cards.values()))
settle()
assert {pid: place for pid, place in board.places().items() if pid != 'p5'} == wide
# The top row is full; the first hole it fits is under the 2x4, beside the 4x4.
assert board.places()['p5'] == (col_after('4x4'), row_after('2x2', '2x4')), board.places()
board.close()
"""
    )


def test_random_drags_on_mixed_boards_show_what_they_do(run_qt) -> None:
    """Random mixed boards and drags, through the window: preview equals
    result, no two cards overlap, the committed order changes nothing, and a
    card put back where it stood leaves every card where it was."""

    run_qt(
        _MONITOR
        + """
rng = random.Random(5)
presets = ('1x2', '2x2', '4x2', '2x4', '4x4')
for width in (WIDE, MID, NARROW):
    sizes = [rng.choice(presets) for _ in range(5)]
    sizes[0], sizes[1] = '4x4', '2x2'
    rng.shuffle(sizes)
    board = Monitor(tuple(sizes), width)
    for _drag in range(5):
        panel_id = rng.choice(sorted(board.cards))
        card = board.cards[panel_id]
        hold = rng.choice(('strip_left', 'strip_right', 'body'))
        before = board.places()
        if rng.random() < 0.3:
            at = before[panel_id]
            board.drag(panel_id, hold, at, via=((at[0] + 40, at[1] + 40),))
            assert board.places() == before, (sizes, panel_id, hold, before, board.places())
            continue
        target = before[rng.choice([pid for pid in board.cards if pid != panel_id])]
        seen = (
            min(max(0, target[0] + rng.randint(-12, 12)), width - card.width()),
            max(0, target[1] + rng.randint(-12, 12)),
        )
        board.drag(panel_id, hold, seen)
    board.close()
"""
    )
