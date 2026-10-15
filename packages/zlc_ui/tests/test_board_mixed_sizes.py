"""The board's placement rules on MIXED card sizes, without Qt.

Every board drag test uses mixed sizes.  On a board of one size every card
edge is a column or a row line and every card covers one cell, so a slot
cut through a tall card by a short neighbour's edge, a card flung past
another to a far hole, or a card that slides along a tall card's bottom and
rises into the hole beside it all pass there -- and fail the moment a 4x4
stands beside a 2x2.

The rectangles are the presets' frames at the offscreen scale, and the gap
is the card pad there; they are this file's inputs, not the product's
constants.  Every expected place is derived from them.
"""

from __future__ import annotations

import random

from zlc_ui.board import BoardMetrics, GeomProxy, drop, first_free_slot, pack, slot_under
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


def col_after(*sizes: str) -> int:
    """The column a card stands in to the right of these cards, side by side."""

    return GAP + sum(SIZES[size][0] + GAP for size in sizes)


def row_after(*sizes: str) -> int:
    """The row a card stands in under these cards, stacked."""

    return GAP + sum(SIZES[size][1] + GAP for size in sizes)


def _tiled(sizes, board_w):
    """A board of these cards as Add tiles it."""

    placed = []
    for size in sizes:
        c = card(size)
        c.col, c.row = first_free_slot(c, placed, board_w, METRICS)
        placed.append(c)
    return placed


def _slot(board, carried, drawn, hold, board_w):
    """The slot a card carried by ``hold`` and drawn at ``drawn`` is aimed at,
    on the board as it stood when it was picked up."""

    others = [c for c in board if c is not carried]
    aim = (drawn[0] + hold[0], drawn[1] + STRIP)
    return slot_under(aim, carried, others, METRICS, board_w, seen=drawn)


def _let_go(board, carried, slot, board_w):
    """The board once ``carried`` is let go at ``slot`` -- on copies, as a
    preview is."""

    after = [GeomProxy(c.width, c.height, c.col, c.row) for c in board]
    drop(after, after[board.index(carried)], slot, METRICS, board_w)
    return after


def test_a_card_let_go_onto_another_swaps_with_it() -> None:
    """The card let go onto takes the carried card's place and the carried
    card takes the slot, held by either end of its strip -- on a grid of one
    size and on mixed sizes, both ways round."""

    w2, w4 = SIZES["2x2"][0], SIZES["4x4"][0]
    cases = (
        # A 2x2 grid: bottom-left onto bottom-right, top-left onto bottom-right.
        (("2x2",) * 4, 1104, 2, 3),
        (("2x2",) * 4, 1104, 0, 3),
        # A 2x2 beside a 4x4, a 2x4 under them: either onto the other.
        (("2x2", "4x4", "2x4"), 1400, 0, 1),
        (("2x2", "4x4", "2x4"), 1400, 1, 0),
        # A 2x4 above a 4x4 in one column: either onto the other.
        (("2x4", "4x4", "2x2"), 1104, 0, 1),
        (("2x4", "4x4", "2x2"), 1104, 1, 0),
    )
    landed = {}
    for sizes, board_w, carried_at, onto in cases:
        board = _tiled(sizes, board_w)
        carried, target = board[carried_at], board[onto]
        for hold in ((14, STRIP), (carried.width - 110, STRIP)):
            slot = _slot(board, carried, (target.col + 10, target.row + 10), hold, board_w)
            assert slot == (target.col, target.row), (sizes, carried_at, hold, slot)
            landed[sizes, carried_at, hold[0] > 14] = places(_let_go(board, carried, slot, board_w))
    grid = [(GAP, GAP), (col_after("2x2"), GAP), (GAP, row_after("2x2")), (col_after("2x2"), row_after("2x2"))]
    for hold in (False, True):
        assert landed[("2x2",) * 4, 2, hold] == [grid[0], grid[1], grid[3], grid[2]]
        assert landed[("2x2",) * 4, 0, hold] == [grid[3], grid[1], grid[2], grid[0]]
        # The 4x4 on the left and the 2x2 right of it, whichever was carried.
        beside = [(col_after("4x4"), GAP), (GAP, GAP), (GAP, row_after("4x4"))]
        assert landed[("2x2", "4x4", "2x4"), 0, hold] == beside
        assert landed[("2x2", "4x4", "2x4"), 1, hold] == beside
        # The 4x4 on top and the 2x4 under it, whichever was carried.
        stacked = [(GAP, row_after("4x4")), (GAP, GAP), (GAP, row_after("4x4", "2x4"))]
        assert landed[("2x4", "4x4", "2x2"), 0, hold] == stacked
        assert landed[("2x4", "4x4", "2x2"), 1, hold] == stacked
    assert col_after("4x4") + w2 <= 1400 - GAP < col_after("2x2", "2x2") + w4


def test_a_card_let_go_on_empty_space_takes_it_and_one_under_a_tall_card_stays_under_it() -> None:
    """Let go where no card stands, the card takes the slot and falls; let go
    under a 4x4 beside the hole under a shorter 2x4, it stays under the 4x4."""

    board = [card("2x4", GAP, GAP), card("4x4", col_after("2x4"), GAP), card("2x2", GAP, row_after("2x4"))]
    small = board[2]
    under = (col_after("2x4"), row_after("4x4"))
    slot = _slot(board, small, (under[0] + 10, under[1] + 10), (14, STRIP), 1900)
    assert slot == under
    assert places(_let_go(board, small, slot, 1900)) == [*places(board[:2]), under]


def test_putting_a_card_back_changes_nothing() -> None:
    """Drawn exactly where it stood, a card is aimed back there, and let go
    there it leaves every card where it was."""

    board = _tiled(("4x4", "2x2", "1x2", "2x4", "2x2", "4x2"), 1400)
    for carried in board:
        for hold in ((14, STRIP), (carried.width - 110, STRIP)):
            seen = (carried.col, carried.row)
            slot = _slot(board, carried, seen, hold, 1400)
            assert slot == seen, (places(board), seen, hold, slot)
            assert places(_let_go(board, carried, slot, 1400)) == places(board)


def test_a_card_in_the_way_moves_right_when_its_row_has_room_else_straight_down(monkeypatch) -> None:
    """A card a settled card overlaps moves right just clear of it if its row
    has the room, and otherwise straight down its own column.  It never
    searches the board for a hole."""

    def search(*_args, **_kwargs):
        raise AssertionError("packing searched the board for a hole")

    monkeypatch.setattr(board_layout, "first_free_slot", search)
    for board_w, rest in ((1900, (col_after("4x4"), GAP)), (1104, (GAP, row_after("4x4")))):
        big, small = card("4x4", GAP, GAP), card("2x2", GAP, row_after("2x4"))
        pack([big, small], METRICS, board_w)
        assert places([big, small]) == [(GAP, GAP), rest], board_w
    # Right only if the way there passes no card.  A 2x4 put over the top of
    # a 4x2, with a 1x2 beside the 4x2 under the 2x4: the 4x2 would have to
    # pass the 1x2 to clear the 2x4 rightward, so it goes down, under it.
    wide = card("2x4", SIZES["4x2"][0] // 2, GAP)
    tall = card("4x2", GAP, GAP)
    low = card("1x2", col_after("4x2"), row_after("2x4"))
    pack([wide, tall, low], METRICS, 1900, first=(wide,))
    assert places([wide, tall, low]) == [(GAP, GAP), (GAP, row_after("2x4")), (col_after("4x2"), row_after("2x4"))]


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
    assert gravity_slot(card("2x2", 854, 674), [wide, big], METRICS) == (854, 674)
    # A 1x2 under a 4x4 that starts left of it slides along under it to the
    # 4x4's left edge, and no further: not on under the 1x2 beside it.
    low, big = card("1x2", 7, 7), card("4x4", 518, 7)
    assert gravity_slot(card("1x2", 1029, 674), [low, big], METRICS) == (518, 674)
    # Under two cards, the leftmost one bounds it.
    left, right = card("2x2", 518, 7), card("2x2", 1029, 7)
    assert gravity_slot(card("2x4", 854, 422), [left, right], METRICS) == (518, 422)
    # Nothing above it: it rises, and on the top margin it slides freely.
    assert gravity_slot(card("2x2", 1365, 900), [wide], METRICS) == (854, 7)


def _random_board(rng):
    sizes = [rng.choice(tuple(SIZES)) for _ in range(rng.randint(3, 6))]
    if len(set(sizes)) == 1:
        sizes[0] = "4x4" if sizes[0] != "4x4" else "2x2"
    board_w = rng.choice((1104, 1400, 1900))
    return _tiled(sizes, board_w), board_w


def _hold(rng, c):
    return rng.choice(((14, STRIP), (c.width - 110, STRIP), (c.width // 2, c.height // 2)))


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


def test_the_users_case_a_card_let_go_on_a_bigger_card_takes_its_place() -> None:
    """The case the user drew: a 2x2 top-left, a 4x4 beside it, a 2x2 under
    the first.  The bottom-left card let go on the 4x4 takes the 4x4's place
    and the 4x4 takes the place it left; the top-left card stays."""

    board_w = 2 * GAP + SIZES["2x2"][0] + GAP + SIZES["4x4"][0] + 35
    board = _tiled(("2x2", "4x4", "2x2"), board_w)
    top_left, big, bottom_left = board
    assert places(board) == [(GAP, GAP), (2 * GAP + SIZES["2x2"][0], GAP), (GAP, 2 * GAP + SIZES["2x2"][1])]
    for drawn in ((big.col + 10, big.row + 10), (big.col + 200, big.row + 150), (big.col + 320, big.row + 40)):
        for hold in ((14, STRIP), (bottom_left.width - 60, STRIP), (bottom_left.width // 2, bottom_left.height // 2)):
            slot = _slot(board, bottom_left, drawn, hold, board_w)
            after = _let_go(board, bottom_left, slot, board_w)
            assert places(after) == [(top_left.col, top_left.row), (bottom_left.col, bottom_left.row),
                                     (big.col, big.row)], (drawn, hold, places(after))


def test_a_card_in_the_way_is_never_carried_across_the_card_it_makes_way_for() -> None:
    """A wide card swapped into a narrow card's place that overlaps the card
    let go beside it makes way DOWN, under it -- never rightward across it,
    which read as an insert and could fling a card that was not touched."""

    board = [card("1x2", GAP, GAP), card("2x4", 2 * GAP + SIZES["1x2"][0], GAP),
             card("2x2", GAP, 2 * GAP + SIZES["1x2"][1])]
    small, wide, carried = board
    after = _let_go(board, carried, (wide.col, wide.row), 1900)
    assert places(after) == [(small.col, small.row), (GAP, 2 * GAP + SIZES["2x2"][1]), (wide.col, wide.row)], (
        places(after))


def test_a_column_makes_way_as_one_so_its_cards_keep_their_order() -> None:
    """Making way down carries the cards below in that column along: a card
    that stood above another in its column still stands above it."""

    right = 2 * GAP + SIZES["4x2"][0]
    board = [card("4x2", GAP, GAP), card("2x4", right, GAP), card("1x2", right, 2 * GAP + SIZES["2x4"][1]),
             card("1x2", GAP, 674), card("2x4", right, 711), card("4x2", GAP, 963)]
    pack(board, METRICS, 1400)
    upper, lower = board[1], board[2]
    assert upper.row < lower.row
    after = _let_go(board, board[4], (board[0].col, board[0].row), 1400)
    assert after[1].row < after[2].row, places(after)


def test_random_mixed_drags_keep_every_rule(monkeypatch) -> None:
    """Random mixed boards, widths and drags -- onto a card, anywhere, held by
    either end of the strip or by the body:

    - a card drawn where it stands is aimed there, and let go there changes
      nothing;
    - no two cards overlap;
    - the board a drop leaves is the board that packing its places gives
      again: what was previewed is what stays;
    - every card is at north-west rest among the others;
    - every card but the two that swapped makes way only right along its row
      or straight down its column, then falls only up and left; making way,
      either way, and falling pass no card the drop left untouched.  (Making
      way along its row and then rising up an empty column is two moves;
      their sum is no L.)
    """

    made_way: dict[int, tuple[int, int]] = {}
    fall = board_layout.gravity_slot

    def falling(cfg, placed, metrics):
        # The first place pack lets a card fall from is where it made way to.
        made_way.setdefault(id(cfg), (cfg.col, cfg.row))
        return fall(cfg, placed, metrics)

    monkeypatch.setattr(board_layout, "gravity_slot", falling)
    rng = random.Random(1000)
    for _board in range(120):
        board, board_w = _random_board(rng)
        for _drag in range(rng.randint(1, 3)):
            carried = rng.choice(board)
            seen = (carried.col, carried.row)
            back = _slot(board, carried, seen, _hold(rng, carried), board_w)
            assert back == seen, (places(board), seen, back)
            assert places(_let_go(board, carried, back, board_w)) == places(board)
            if rng.random() < 0.5:
                target = rng.choice([c for c in board if c is not carried])
                drawn = (target.col + rng.randint(-12, 12), target.row + rng.randint(-12, 12))
            else:
                bottom = max(c.row + c.height for c in board)
                drawn = (rng.randint(0, board_w - carried.width), rng.randint(0, bottom + 100))
            slot = _slot(board, carried, drawn, _hold(rng, carried), board_w)
            swapped = {
                id(c)
                for c in board
                if c is carried or (c.col <= slot[0] < c.col + c.width and c.row <= slot[1] < c.row + c.height)
            }
            made_way.clear()
            after = _let_go(board, carried, slot, board_w)
            for index, a in enumerate(after):
                for b in after[index + 1:]:
                    assert not _overlaps(a, b), places(after)
            again = [GeomProxy(c.width, c.height, c.col, c.row) for c in after]
            pack(again, METRICS, board_w)
            assert places(again) == places(after)
            for c in after:
                rest = fall(c, [o for o in after if o is not c], METRICS)
                assert rest == (c.col, c.row), (places(after), places([c]))
            moves = {
                id(a): ((b.col, b.row), made_way.get(id(a), (a.col, a.row)), (a.col, a.row))
                for b, a in zip(board, after)
                if id(b) not in swapped
            }
            untouched = [
                (a.col, a.row, a.width, a.height) for a in after if id(a) in moves and len(set(moves[id(a)])) == 1
            ]
            for a in after:
                if id(a) not in moves:
                    continue
                start, way, end = moves[id(a)]
                assert (
                    way == start
                    or (way[1] == start[1] and way[0] > start[0])
                    or (way[0] == start[0] and way[1] > start[1])
                ), ("made way neither right nor down", places(board), places(after), moves[id(a)])
                assert end[0] <= way[0] and end[1] <= way[1], ("fell other than north-west", moves[id(a)])
                for run in ((start, way), (way, end)):
                    assert not _passed(_centre(a, run[0]), _centre(a, run[1]), untouched), (
                        places(board), places(after), moves[id(a)],
                    )
            board = after
