"""The board's placement rules on MIXED card sizes, without Qt.

Every board drag test uses mixed sizes.  On a board of one size every card
edge is a column or a row line and every card covers one cell, so a slot
cut through a tall card by a short neighbour's edge, a card flung past
another to a far hole, or a card that slides along a tall card's bottom and
rises into the hole beside it all pass there -- and fail the moment a 4x4
stands beside a 2x2.

The rectangles are the presets' frames at the offscreen scale, and the gap
is the card pad there; they are this file's inputs, not the product's
constants.
"""

from __future__ import annotations

import random

import pytest

from zlc_ui.board import BoardMetrics, GeomProxy, first_free_slot, lift, pack, slot_under
from zlc_ui.board import board_layout
from zlc_ui.board.board_layout import gravity_slot

GAP = 7
METRICS = BoardMetrics(GAP)
SIZES = {
    "1x2": (504, 282),
    "2x2": (504, 408),
    "4x2": (504, 660),
    "2x4": (840, 408),
    "4x4": (840, 660),
}
#: Where a carried card's strip line is, from its top: a board aims at the
#: pointer's x on this line.
STRIP = 21


def card(size: str, col: int = 0, row: int = 0) -> GeomProxy:
    return GeomProxy(*SIZES[size], col, row)


def places(cards) -> list[tuple[int, int]]:
    return [(c.col, c.row) for c in cards]


def test_a_card_on_a_taken_place_makes_way_straight_down(monkeypatch) -> None:
    """A card whose place a 4x4 covers goes straight down its own column.

    It used to rise through the 4x4 and then take the first free hole on the
    whole board, from the row it had risen to: on a wide board, up and to the
    right, past the 4x4.
    """

    def search(*_args, **_kwargs):
        raise AssertionError("gravity searched the board for a hole")

    monkeypatch.setattr(board_layout, "first_free_slot", search)
    big = card("4x4", 7, 7)
    small = card("2x2", 7, 422)
    assert gravity_slot(small, [big], 1900, METRICS) == (7, 674)
    assert gravity_slot(small, [big], 1104, METRICS) == (7, 674)


def test_pack_never_passes_one_card_through_another() -> None:
    """A 1x2 let go below the 2x2 rises to the top row; the 4x4 to its right
    stays where it is, and nothing is left overlapping."""

    first, big, dropped = card("2x2", 7, 7), card("4x4", 1029, 7), card("1x2", 518, 422)
    pack([first, big, dropped], METRICS, 1900, dropped=dropped)
    assert places([first, dropped, big]) == [(7, 7), (518, 7), (1029, 7)]


def test_the_slot_is_the_place_the_card_is_seen_over_wherever_it_is_held() -> None:
    """The pointer picks the places the card can be put at, and of those the
    one the card is drawn over wins -- held at either end of its strip, or
    by an empty card's body."""

    frame = [card("2x2", 7, 7), card("4x4", 518, 7)]
    wide = card("2x4")
    for aim in ((24, 31), (745, 31)):
        assert slot_under(aim, wide, frame, METRICS, 1400, seen=(10, 10)) == (7, 7), aim
    # An empty 4x4 held at its body centre, drawn 92 % over the first column.
    assert slot_under((530, 7 + STRIP), card("4x4"), frame, METRICS, 1400, seen=(110, 7)) == (7, 7)


def test_rows_come_only_from_the_cards_in_the_slots_column() -> None:
    """The bottom of a 2x2 in another column does not cut the 4x4 in this one."""

    frame = [card("4x4", 7, 7), card("2x2", 854, 7), card("2x2", 854, 422)]
    assert slot_under((100, 450), card("2x2"), frame, METRICS, 1400, seen=(86, 429)) == (7, 674)


def test_narrowing_packs_without_rewriting_the_places() -> None:
    """A board too narrow for the arrangement packs it, column by column, and
    the same places on the wide board again are the arrangement."""

    sizes = ("4x4", "2x2", "2x2", "2x2")
    arranged = [(7, 7), (854, 7), (854, 422), (7, 674)]

    def packed(board_w: int) -> list[tuple[int, int]]:
        cards = [card(size, *place) for size, place in zip(sizes, arranged)]
        pack(cards, METRICS, board_w)
        return places(cards)

    assert packed(1400) == arranged
    # The 2x2 that stood beside the 4x4 goes below it, and the one under that
    # 2x2 stays under it.
    assert packed(1100) == [(7, 7), (518, 674), (518, 1089), (7, 674)]
    assert packed(1400) == arranged


def test_a_card_stopped_by_a_card_above_it_stays_below_it() -> None:
    """A card a card above has stopped slides left no further than that
    card's left edge -- with several above it, the leftmost one's.

    Rising, then sliding left, then rising again took a 2x2 put under a 4x4
    along the 4x4's bottom and up into the hole under the shorter 2x4 beside
    it: put below one card, it landed beside another.
    """

    # Under the 4x4, with the hole under the 2x4 beside it: it stays.
    wide, big = card("2x4", 7, 7), card("4x4", 854, 7)
    assert gravity_slot(card("2x2", 854, 674), [wide, big], 1900, METRICS) == (854, 674)
    # A 1x2 under a 4x4 that starts left of it slides along under it to the
    # 4x4's left edge, and no further: not on under the 1x2 beside it.
    low, big = card("1x2", 7, 7), card("4x4", 518, 7)
    assert gravity_slot(card("1x2", 1029, 674), [low, big], 1900, METRICS) == (518, 674)
    # Under two cards, the leftmost one bounds it.
    left, right = card("2x2", 518, 7), card("2x2", 1029, 7)
    assert gravity_slot(card("2x4", 854, 422), [left, right], 1900, METRICS) == (518, 422)
    # Nothing above it: it rises, and on the top margin it slides freely.
    assert gravity_slot(card("2x2", 1365, 900), [wide], 1900, METRICS) == (854, 7)


def test_lifting_a_card_opens_its_own_hole_and_not_the_board_above_it() -> None:
    """The others fall north-west into the lifted card's hole; a card that
    stood below its top line stops at that line."""

    # A 2x4 under a 2x2, with two 2x2 under it; the right one is clear of
    # the 2x2 above, so only the lifted card's top line stops it.
    top_row = [card("2x2", 7, 7), card("2x2", 518, 7)]
    beside = card("2x2", 7, 422)
    lifted = card("2x4", 518, 422)
    under = [card("2x2", 518, 837), card("2x2", 1029, 837)]
    lift([*top_row, beside, *under], lifted, METRICS)
    assert places(top_row + [beside]) == [(7, 7), (518, 7), (7, 422)]
    assert places(under) == [(518, 422), (1029, 422)], "they fill the hole, not the top row"
    # A card beside the lifted one on the top row slides into its place, and
    # the card under it slides along with it, still under it.
    lifted = card("1x2", 7, 7)
    wide = card("2x4", 518, 7)
    below = card("2x2", 518, 422)
    lift([wide, below], lifted, METRICS)
    assert places([wide, below]) == [(7, 7), (7, 422)]


def _lifted(board, carried):
    """The frame a carried card's slot is chosen in, and where each other card stood."""

    others = [GeomProxy(c.width, c.height, c.col, c.row) for c in board if c is not carried]
    home = {id(o): (o.col, o.row) for o in others}
    lift(others, carried, METRICS)
    return others, home


def _drop(board, carried, slot, board_w):
    """The board once ``carried`` is let go at ``slot`` -- copies, as a preview
    is -- and where each card stood while it was carried."""

    frame, home = _lifted(board, carried)
    moved = GeomProxy(carried.width, carried.height, *slot)
    homes = {o: home[id(o)] for o in frame}
    homes[moved] = (carried.col, carried.row)
    after = [moved if c is carried else frame.pop(0) for c in board]
    carried_over = places(after)
    pack(after, METRICS, board_w, dropped=moved, home=homes)
    return after, carried_over


def _tiled(sizes, board_w):
    """A board of these cards as Add tiles it."""

    placed = []
    for size in sizes:
        c = card(size)
        c.col, c.row = first_free_slot(c, placed, board_w, METRICS)
        placed.append(c)
    return placed


def _random_board(rng):
    sizes = [rng.choice(tuple(SIZES)) for _ in range(rng.randint(3, 6))]
    if len(set(sizes)) == 1:
        sizes[0] = "4x4" if sizes[0] != "4x4" else "2x2"
    board_w = rng.choice((1104, 1400, 1900))
    return _tiled(sizes, board_w), board_w


def _hold(rng, c):
    return rng.choice(((14, STRIP), (c.width - 110, STRIP), (c.width // 2, c.height // 2)))


def test_a_card_let_go_on_its_row_neighbour_swaps_with_it() -> None:
    """Carried onto the card beside it in a full row, the two trade places:
    the one in the way takes the room the carried card left in that row --
    it is not pushed down a row with a hole left beside it.  Holds for a row
    of one size and for a 4x4 beside a 2x2."""

    for left_size, right_size, board_w in (("2x2", "2x2", 1104), ("2x2", "4x4", 1400), ("4x4", "2x2", 1400)):
        board = _tiled(("2x2", "2x2", left_size, right_size), board_w)
        carried, neighbour = board[2], board[3]
        assert carried.row == neighbour.row, places(board)
        slot = (neighbour.col + neighbour.width - carried.width if carried.width < neighbour.width else neighbour.col,
                neighbour.row)
        after, _over = _drop(board, carried, slot, board_w)
        assert after[3].row == carried.row, (left_size, right_size, places(after))
        assert after[3].col < after[2].col, (left_size, right_size, places(after))
        assert max(c.row + c.height for c in after) == max(c.row + c.height for c in board), places(after)


def test_putting_a_card_back_changes_nothing() -> None:
    """Drawn exactly where it stood, held anywhere, a card is put back there
    and every other card is where it was -- on random mixed boards, after
    random earlier drops."""

    rng = random.Random(7)
    for _board in range(150):
        board, board_w = _random_board(rng)
        for _drag in range(rng.randint(0, 2)):
            carried = rng.choice(board)
            drawn = (rng.randint(0, board_w - carried.width), rng.randint(0, 1600))
            hold = _hold(rng, carried)
            frame, _home = _lifted(board, carried)
            aim = (drawn[0] + hold[0], drawn[1] + STRIP)
            slot = slot_under(aim, carried, frame, METRICS, board_w, seen=drawn)
            board, _carried_over = _drop(board, carried, slot, board_w)
        before = places(board)
        for carried in board:
            hold = _hold(rng, carried)
            seen = (carried.col, carried.row)
            frame, _home = _lifted(board, carried)
            aim = (seen[0] + hold[0], seen[1] + STRIP)
            slot = slot_under(aim, carried, frame, METRICS, board_w, seen=seen)
            assert slot == seen, (before, seen, hold, slot)
            assert places(_drop(board, carried, slot, board_w)[0]) == before


def _overlaps(a, b) -> bool:
    return (
        a.col < b.col + b.width + GAP
        and b.col < a.col + a.width + GAP
        and a.row < b.row + b.height + GAP
        and b.row < a.row + a.height + GAP
    )


def _crosses(start, end, box) -> bool:
    """A horizontal or vertical run from ``start`` to ``end`` crosses the inside of ``box``."""

    x0, y0, x1, y1 = box[0] + 2, box[1] + 2, box[0] + box[2] - 2, box[1] + box[3] - 2
    if start[1] == end[1]:
        low, high = sorted((start[0], end[0]))
        return y0 < start[1] < y1 and low < x1 and x0 < high
    low, high = sorted((start[1], end[1]))
    return x0 < start[0] < x1 and low < y1 and y0 < high


def _passed(start, end, stayed) -> bool:
    """Both L-shaped paths between two centres cross a card that stayed put:
    the move could not have been made without passing it."""

    if start == end:
        return False
    paths = (((end[0], start[1]),), ((start[0], end[1]),))
    return all(
        any(_crosses(start, corner, box) or _crosses(corner, end, box) for box in stayed)
        for (corner,) in paths
    )


def _centre(c, place) -> tuple[int, int]:
    return (place[0] + c.width // 2, place[1] + c.height // 2)


@pytest.mark.parametrize("seed", range(4))
def test_random_mixed_drags_keep_every_rule(seed, monkeypatch) -> None:
    """Random mixed boards, widths and drags -- onto a card, anywhere, held by
    either end of the strip or by the body:

    - no two cards overlap;
    - the board a drop leaves is the board that packing its places gives
      again: what was previewed is what stays;
    - every card is at north-west rest among the others;
    - no card the operator did not carry passes a card that stayed put, in
      any of the moves it makes: falling into the carried card's hole,
      making way, and falling again.  (Making way along its row and then
      rising up an empty column is two moves; their sum is no L.)
    """

    made_way: dict[int, tuple[int, int]] = {}
    fall = board_layout.gravity_slot

    def falling(cfg, placed, board_w, metrics):
        # The first place pack lets a card fall from is where it made way to.
        made_way.setdefault(id(cfg), (cfg.col, cfg.row))
        return fall(cfg, placed, board_w, metrics)

    monkeypatch.setattr(board_layout, "gravity_slot", falling)
    rng = random.Random(1000 + seed)
    for _board in range(60):
        board, board_w = _random_board(rng)
        for _drag in range(rng.randint(1, 3)):
            carried = rng.choice(board)
            if rng.random() < 0.5:
                target = rng.choice([c for c in board if c is not carried])
                drawn = (target.col + rng.randint(-12, 12), target.row + rng.randint(-12, 12))
            else:
                bottom = max(c.row + c.height for c in board)
                drawn = (rng.randint(0, board_w - carried.width), rng.randint(0, bottom + 100))
            hold = _hold(rng, carried)
            frame, _home = _lifted(board, carried)
            slot = slot_under(
                (drawn[0] + hold[0], drawn[1] + STRIP), carried, frame, METRICS, board_w, seen=drawn
            )
            made_way.clear()
            after, carried_over = _drop(board, carried, slot, board_w)
            moves = {
                id(a): ((b.col, b.row), lifted, made_way.get(id(a), (a.col, a.row)), (a.col, a.row))
                for b, a, lifted in zip(board, after, carried_over)
                if b is not carried
            }
            for index, a in enumerate(after):
                for b in after[index + 1:]:
                    assert not _overlaps(a, b), places(after)
            again = [GeomProxy(c.width, c.height, c.col, c.row) for c in after]
            pack(again, METRICS, board_w)
            assert places(again) == places(after)
            for c in after:
                rest = gravity_slot(c, [o for o in after if o is not c], board_w, METRICS)
                assert rest == (c.col, c.row), (places(after), places([c]))
            stayed = [
                (a.col, a.row, a.width, a.height)
                for b, a in zip(board, after)
                if b is not carried and (a.col, a.row) == (b.col, b.row)
            ]
            for b, a in zip(board, after):
                if b is carried or (a.col, a.row) == (b.col, b.row):
                    continue
                steps = moves[id(a)]
                for start, end in zip(steps, steps[1:]):
                    assert not _passed(_centre(a, start), _centre(a, end), stayed), (
                        places(board), places(after), steps,
                    )
            board = after
