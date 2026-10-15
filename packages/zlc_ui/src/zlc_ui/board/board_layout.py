"""Pure north-west-gravity placement for a panel board.

The packer depends only on rectangle geometry: it is GIVEN each card's
rectangle and the clear gap between them, and imports neither UI nor
rendering backends.

It used to be given a preset NAME per card and a callable that turned one
into a size, which made "how big is a card" a second answer beside the
card's own layout -- and one that cannot be right, since two cards of the
same preset differ by whatever is mounted in them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = ["BoardMetrics", "GeomProxy", "slot_under", "first_free_slot",
           "min_board_width", "pack", "lift"]


@dataclass(frozen=True)
class BoardMetrics:
    """The one fact the packer cannot derive: the clear gap between cards."""

    gap: int

    def __post_init__(self) -> None:
        if int(self.gap) != self.gap or self.gap < 0:
            raise ValueError("gap must be a non-negative whole number of pixels")


class GeomProxy:
    """A placement-only stand-in, so a trial pack never mutates a real card.

    It carries the card's own rectangle.  Whoever builds one asks the card
    how big it is, which is the only answer there is.
    """

    __slots__ = ("width", "height", "col", "row")

    def __init__(self, width: int, height: int, col: int = 0, row: int = 0) -> None:
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("a card rectangle must be positive")
        self.width = int(width)
        self.height = int(height)
        self.col = col
        self.row = row


def _aabb(cfg) -> tuple[int, int, int, int]:
    """The card's pixel AABB ``(x0, y0, x1, y1)`` -- top-left ``(col, row)`` plus its size."""
    return (cfg.col, cfg.row, cfg.col + cfg.width, cfg.row + cfg.height)


def _overlaps_with_gap(box: tuple[int, int, int, int], placed, metrics: BoardMetrics) -> bool:
    """True when ``box`` (an ``(x0, y0, x1, y1)`` AABB), EXPANDED by the gap on all sides,
    intersects any already-placed card.  Equivalently: the clear distance between ``box`` and a
    placed card is < gap on the axis where they overlap, so leaving a card exactly gap away
    counts as clear."""
    gap = metrics.gap
    x0, y0, x1, y1 = box
    for p in placed:
        px0, py0, px1, py1 = _aabb(p)
        if x0 < px1 + gap and px0 < x1 + gap and y0 < py1 + gap and py0 < y1 + gap:
            return True
    return False


def first_free_slot(cfg, placed, board_w: int, metrics: BoardMetrics) -> tuple[int, int]:
    """The TOP-MOST then LEFT-MOST free ``(col, row)`` where ``cfg`` fits clear of every ``placed``
    card (gap apart, inside ``board_w``).  This is where a card with no place of its own GOES -- a
    panel just added tiles the top row left-to-right, wraps to the next shelf, and never leaves a
    middle hole.  Candidate points are the gap (origin) plus each placed card's right/bottom edge
    (``+gap``) and its left/top edge (so a card can tuck under a wider one); swept by y then x,
    first feasible wins.
    """
    gap = metrics.gap
    w, _h = cfg.width, cfg.height
    xs = {gap}
    ys = {gap}
    for p in placed:
        px0, py0, px1, py1 = _aabb(p)
        xs.add(px1 + gap)
        ys.add(py1 + gap)
        xs.add(px0)            # also align left edges, so a card can tuck under a wider one
        ys.add(py0)
    max_x = max(gap, board_w - gap - w)
    cand_x = sorted(x for x in xs if gap <= x <= max_x) or [gap]
    for y in sorted(y for y in ys if y >= gap):
        for x in cand_x:
            if not _overlaps_with_gap((x, y, x + w, y + _h), placed, metrics):
                return (x, y)
    # No candidate fit (should not happen -- placing past the lowest card always clears).
    bottom = max((py1 for *_rest, py1 in (_aabb(p) for p in placed)), default=0)
    return (gap, bottom + gap if placed else gap)


def _rise_to(x: int, y: int, width: int, boxes, gap: int) -> int:
    """The row a card at ``(x, y)`` rises to: the margin, or one gap under
    the lowest card above it in its column band."""

    top = gap
    for px0, _py0, px1, py1 in boxes:
        if x < px1 + gap and px0 < x + width + gap and py1 + gap <= y:
            top = max(top, py1 + gap)
    return top


def _slide_to(x: int, y: int, height: int, boxes, gap: int) -> int:
    """The column a card at ``(x, y)`` slides to: the margin, or one gap
    right of the nearest card to its left in its row band."""

    left = gap
    for px0, py0, px1, py1 in boxes:
        if y < py1 + gap and py0 < y + height + gap and px1 + gap <= x:
            left = max(left, px1 + gap)
    return left


def _below(x: int, y: int, width: int, height: int, boxes, gap: int) -> int:
    """The first row at or below ``y`` where a card at ``x`` is clear of every
    box: straight DOWN in its own column, never up and never sideways."""

    while True:
        hits = [
            py1
            for px0, py0, px1, py1 in boxes
            if x < px1 + gap and px0 < x + width + gap and y < py1 + gap and py0 < y + height + gap
        ]
        if not hits:
            return y
        y = max(hits) + gap


def _fall(x: int, y: int, width: int, height: int, boxes, gap: int, floor: int) -> tuple[int, int]:
    """Where a clear card at ``(x, y)`` comes to rest under north-west gravity.

    It rises until a card above it, the top margin or ``floor`` stops it, then slides left until a
    card beside it or the left margin stops it, and repeats until neither move is possible.

    A card that a card above has stopped RESTS UNDER it, and slides left no further than the left
    edge of the leftmost card it rests under: it stays below that card, and never slides out from
    under it to rise into a hole beside it.  A card resting on nothing -- on the top margin, or held
    at ``floor`` -- slides left freely, and a card with nothing above it rises.
    """

    while True:
        top = max(_rise_to(x, y, width, boxes, gap), floor)
        left = _slide_to(x, top, height, boxes, gap)
        under = [
            px0
            for px0, _py0, px1, py1 in boxes
            if py1 + gap == top and x < px1 + gap and px0 < x + width + gap
        ]
        if under:
            left = max(left, min(x, *under))
        if (left, top) == (x, y):
            return (x, y)
        x, y = left, top


def gravity_slot(cfg, placed, board_w: int, metrics: BoardMetrics) -> tuple[int, int]:
    """Where one card comes to rest, falling NORTH-WEST from where it already is.

    It rises until a card above it -- or the top margin -- stops it, then slides left until a card
    beside it, or the left margin, stops it, and repeats until neither move is possible.  It only
    ever moves up and left and it stops at the FIRST thing in the way.

    A card stopped by a card above it does not slide left past that card's left edge -- with several
    cards above it, past the leftmost one's: a card put below another stays below it, and does not
    slide along its bottom and rise into a hole beside it.  While nothing above stops it, it slides
    left freely, and while nothing is above it, it rises.

    A card whose own place is taken first makes way straight DOWN in its own column, to the first
    row where it is clear, and falls from there.  It never searches the board for a hole: from a
    clear start, rising and sliding stop at the first card in the way and cannot pass through one.
    """

    gap = metrics.gap
    width, height = cfg.width, cfg.height
    x = min(max(int(cfg.col), gap), max(gap, board_w - gap - width))
    boxes = [_aabb(p) for p in placed]
    y = _below(x, max(int(cfg.row), gap), width, height, boxes, gap)
    return _fall(x, y, width, height, boxes, gap, gap)


def min_board_width(configs: Sequence, metrics: BoardMetrics) -> int:
    """The NARROWEST a board may pack to: one WIDEST card plus both gap margins.  A viewport thinner
    than this still has to fit the widest card, so we clamp up to it -- but NOT to the cards' current
    right-extent: clamping to the extent would RATCHET (once cards spread wide the board could never
    pack narrower), so narrowing the window would never reflow into a single column.  At one-card
    width the gravity packer simply stacks every card in one column, which is the correct reflow."""
    widest = max((c.width for c in configs), default=0)
    return widest + 2 * metrics.gap


def _along_row(cfg, resolved, pending, board_w: int, gap: int) -> dict[int, int] | None:
    """Columns that make room for ``cfg`` further along its row.

    ``cfg`` moves right just clear of the resolved cards it covers, and every
    pending card a moved card then covers moves right just clear of it, in
    turn.  None -- ``cfg`` makes way downward instead -- when any of them would
    run past the board or land on a resolved card, which it would have to
    pass.  Keyed by ``id`` of the card.
    """

    def box(card, x):
        return (x, card.row, x + card.width, card.row + card.height)

    def covers(b, o):
        return b[0] < o[2] + gap and o[0] < b[2] + gap and b[1] < o[3] + gap and o[1] < b[3] + gap

    fixed = [_aabb(p) for p in resolved]
    start = [o for o in fixed if covers(_aabb(cfg), o)]
    moved: dict[int, int] = {}
    queue = [(cfg, max(o[2] for o in start) + gap)]
    while queue:
        card, x = queue.pop(0)
        x = max(x, moved.get(id(card), card.col))
        if x > board_w - gap - card.width or any(covers(box(card, x), o) for o in fixed):
            return None
        if moved.get(id(card)) == x:
            continue
        moved[id(card)] = x
        for other in pending:
            if other is not card and covers(box(card, x), box(other, moved.get(id(other), other.col))):
                queue.append((other, x + card.width + gap))
    return moved


def pack(
    order: Sequence,
    metrics: BoardMetrics,
    board_w: int,
    *,
    dropped=None,
    home=None,
) -> bool:
    """Settle every card of ``order`` under north-west gravity, from where it is.

    Each proxy arrives carrying where its card starts from and leaves carrying
    where it comes to rest on a board this wide.  ``home`` maps a proxy to
    where it stood before the drag.  Let go where it stood itself, ``dropped``
    finds every card where it stood: putting a card back changes nothing.

    1. ``dropped`` -- the card the operator lets go -- rises straight up among
       the cards it does not cover and keeps that place: it is where the
       operator put it.
    2. Every other card, in the reading order of where it starts, makes way for
       the cards already resolved: back to its ``home`` when that is clear;
       else along its row, if the row has room for it and every card it
       pushes; else straight DOWN in its own column.  A card a narrower board
       squeezed onto another does the same.
    3. Every card falls north-west (:func:`gravity_slot`) -- the others
       first, around the dropped one, then all of them -- with EVERY other
       card an obstacle, until nothing moves.  No card passes through another
       or searches the board for a hole, none is left floating, and a card
       stopped by a card above it stays below it.

    ``board_w`` is the live viewport width, always clamped to fit one card.
    Returns whether any proxy moved.
    """

    order = list(order)
    if dropped is not None and not any(cfg is dropped for cfg in order):
        raise ValueError("the dropped card must belong to the packed board")
    board_w = max(board_w, min_board_width(order, metrics))
    gap = metrics.gap
    start = [(cfg.col, cfg.row) for cfg in order]
    rank = {id(cfg): index for index, cfg in enumerate(order)}
    home = {} if home is None else {id(cfg): place for cfg, place in home.items()}
    if dropped is not None and home.get(id(dropped)) == (dropped.col, dropped.row):
        # Not every card that fell into its hole stands in its way again --
        # one that slid along under a card that fell there does not -- so
        # making way alone would not bring them all back.
        for cfg in order:
            cfg.col, cfg.row = home.get(id(cfg), (cfg.col, cfg.row))
    for cfg in order:
        cfg.col = min(max(int(cfg.col), gap), max(gap, board_w - gap - cfg.width))
        cfg.row = max(int(cfg.row), gap)
    resolved: list = []
    if dropped is not None:
        clear = [
            _aabb(p)
            for p in order
            if p is not dropped and not _overlaps_with_gap(_aabb(dropped), [p], metrics)
        ]
        dropped.row = _rise_to(dropped.col, dropped.row, dropped.width, clear, gap)
        resolved.append(dropped)
    for cfg in sorted(
        (cfg for cfg in order if cfg is not dropped),
        key=lambda cfg: (start[rank[id(cfg)]][1], start[rank[id(cfg)]][0], rank[id(cfg)]),
    ):
        if _overlaps_with_gap(_aabb(cfg), resolved, metrics):
            place = home.get(id(cfg))
            if place is not None and not _overlaps_with_gap(
                (place[0], place[1], place[0] + cfg.width, place[1] + cfg.height), resolved, metrics
            ):
                cfg.col, cfg.row = place
            else:
                pending = [p for p in order if p is not cfg and all(p is not r for r in resolved)]
                shift = _along_row(cfg, resolved, pending, board_w, gap)
                if shift is not None:
                    for card in (cfg, *pending):
                        card.col = shift.get(id(card), card.col)
        cfg.row = _below(cfg.col, cfg.row, cfg.width, cfg.height, [_aabb(p) for p in resolved], gap)
        resolved.append(cfg)
    others = [cfg for cfg in order if cfg is not dropped]
    for movers in (others, order):
        moved = True
        while moved:
            moved = False
            for cfg in sorted(movers, key=lambda cfg: (cfg.row, cfg.col, rank[id(cfg)])):
                rest = gravity_slot(cfg, [p for p in order if p is not cfg], board_w, metrics)
                if rest != (cfg.col, cfg.row):
                    cfg.col, cfg.row = rest
                    moved = True
    return any((cfg.col, cfg.row) != place for cfg, place in zip(order, start))


def lift(others: Sequence, lifted, metrics: BoardMetrics) -> None:
    """The board with ``lifted`` picked up, in place: the other cards fall
    north-west into its hole as :func:`gravity_slot` falls them, every card
    an obstacle -- but a card that stood at or below the line its top stood
    on never rises above that line.  Lifting a card opens its own hole, not
    the board above it: a card under it fills the hole and does not fly on
    past it to the top of the board."""

    gap = metrics.gap
    others = list(others)
    floor = {id(cfg): lifted.row if cfg.row >= lifted.row else gap for cfg in others}
    moved = True
    while moved:
        moved = False
        for cfg in sorted(others, key=lambda cfg: (cfg.row, cfg.col)):
            boxes = [_aabb(p) for p in others if p is not cfg]
            rest = _fall(cfg.col, cfg.row, cfg.width, cfg.height, boxes, gap, floor[id(cfg)])
            if rest != (cfg.col, cfg.row):
                cfg.col, cfg.row = rest
                moved = True


def _nearest(anchors, at: int, seen: int, size: int) -> int:
    """The anchor that keeps ``at`` inside a card ``size`` long, nearest ``seen``;
    with none, the anchor nearest ``seen``.  Ties go to the smaller anchor."""

    inside = [a for a in anchors if a <= at < a + size]
    return min(inside or anchors, key=lambda a: (abs(a - seen), a))


def slot_under(
    point: tuple[int, int],
    cfg,
    others: Sequence,
    metrics: BoardMetrics,
    board_w: int,
    *,
    seen: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """The place a carried card is put at: under the pointer, nearest where it is seen.

    Anchors are the margin and the other cards' edges (a gap past each far
    edge).  On each axis the slot is the anchor that keeps ``point`` -- the
    pointer -- inside the carried card and lies nearest ``seen``, the carried
    card's top-left as drawn; with no such anchor, the one nearest ``seen``.
    So the pointer picks the column wherever along the strip the card is
    held, and of the places under it the one the card is drawn over wins,
    whatever part of a big card -- or of an empty card's body -- is held.

    The column comes first; the row is then chosen only among the cards in
    that column band, so the edge of a short card in another column never
    cuts a tall card in this one.  An occupied place stays a candidate:
    choosing it puts the card there and the occupant makes way.  ``cfg`` only
    says how big the carried card is.  ``seen`` defaults to ``point``.
    """

    gap = metrics.gap
    board_w = max(board_w, min_board_width(list(others) + [cfg], metrics))
    seen = point if seen is None else seen
    max_x = max(gap, board_w - gap - cfg.width)
    xs = {gap}
    for other in others:
        left, _top, right, _bottom = _aabb(other)
        xs.update((left, right + gap))
    xs = sorted(x for x in xs if gap <= x <= max_x) or [gap]
    x = _nearest(xs, int(point[0]), int(seen[0]), cfg.width)
    ys = {gap}
    for other in others:
        left, top, right, bottom = _aabb(other)
        if x < right + gap and left < x + cfg.width + gap:
            ys.update((top, bottom + gap))
    ys = sorted(y for y in ys if y >= gap)
    return (x, _nearest(ys, int(point[1]), int(seen[1]), cfg.height))
