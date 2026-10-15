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
           "min_board_width", "pack", "drop"]


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


def gravity_slot(cfg, placed, metrics: BoardMetrics) -> tuple[int, int]:
    """Where a card standing clear of every ``placed`` card comes to rest under north-west gravity.

    It rises until a card above it -- or the top margin -- stops it, then slides left until a card
    beside it, or the left margin, stops it, and repeats until neither move is possible.  It only
    ever moves up and left and it stops at the FIRST thing in the way, so from a clear place it
    cannot pass through a card.

    A card stopped by a card above it RESTS UNDER it, and slides left no further than the left edge
    of the leftmost card it rests under: a card put below another stays below it, and does not slide
    along its bottom and rise into a hole beside it.  While nothing above stops it, it slides left
    freely, and while nothing is above it, it rises.
    """

    gap = metrics.gap
    width, height = cfg.width, cfg.height
    boxes = [_aabb(p) for p in placed]
    x, y = cfg.col, cfg.row
    while True:
        top = _rise_to(x, y, width, boxes, gap)
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


def min_board_width(configs: Sequence, metrics: BoardMetrics) -> int:
    """The NARROWEST a board may pack to: one WIDEST card plus both gap margins.  A viewport thinner
    than this still has to fit the widest card, so we clamp up to it -- but NOT to the cards' current
    right-extent: clamping to the extent would RATCHET (once cards spread wide the board could never
    pack narrower), so narrowing the window would never reflow into a single column.  At one-card
    width the gravity packer simply stacks every card in one column, which is the correct reflow."""
    widest = max((c.width for c in configs), default=0)
    return widest + 2 * metrics.gap


def _make_way(cfg, settled, order, board_w: int, metrics: BoardMetrics) -> None:
    """Move ``cfg`` clear of the ``settled`` cards it overlaps.

    RIGHT, just clear of them, when its row has the room: the place is on the
    board, clear of every settled card, and the way there passes no card -- it
    meets none that neither its start nor that place touches.  Otherwise
    straight DOWN its own column, to the first row clear of every settled card.
    A card that place touches and that has not settled yet makes way in its
    own turn.
    """

    gap = metrics.gap
    start = _aabb(cfg)
    x = max(p.col + p.width for p in settled if _overlaps_with_gap(start, [p], metrics)) + gap
    place = (x, cfg.row, x + cfg.width, cfg.row + cfg.height)
    way = (cfg.col, cfg.row, x + cfg.width, cfg.row + cfg.height)
    passed = [
        p
        for p in order
        if p is not cfg
        and _overlaps_with_gap(way, [p], metrics)
        and not _overlaps_with_gap(start, [p], metrics)
        and not _overlaps_with_gap(place, [p], metrics)
    ]
    if x <= board_w - gap - cfg.width and not passed and not _overlaps_with_gap(place, settled, metrics):
        cfg.col = x
    else:
        cfg.row = _below(cfg.col, cfg.row, cfg.width, cfg.height, [_aabb(p) for p in settled], gap)


def pack(order: Sequence, metrics: BoardMetrics, board_w: int, *, first: Sequence = ()) -> None:
    """Settle every card of ``order`` under north-west gravity, from where it stands.

    Each proxy arrives carrying where its card stands and leaves carrying
    where it comes to rest on a board this wide (``board_w``, the live
    viewport width, clamped to fit the widest card).

    1. Overlaps settle in priority: the cards of ``first``, then every other
       card, each group in the reading order of where its cards stand.  A card
       that overlaps one already settled moves RIGHT just clear of it if its
       row has the room -- on the board, passing no card -- otherwise straight
       DOWN its own column to the first clear place.  A card a narrower board
       squeezed onto another makes way the same way.
    2. Every card falls north-west (:func:`gravity_slot`), every other card an
       obstacle, until nothing moves.  No card passes through another or
       searches the board for a hole, and a card stopped by a card above it
       stays below it.
    """

    order = list(order)
    board_w = max(board_w, min_board_width(order, metrics))
    gap = metrics.gap
    for cfg in order:
        cfg.col = min(max(int(cfg.col), gap), board_w - gap - cfg.width)
        cfg.row = max(int(cfg.row), gap)
    rank = {id(cfg): index for index, cfg in enumerate(order)}
    ahead = {id(cfg) for cfg in first}

    def reading(cfg) -> tuple[int, int, int]:
        return (cfg.row, cfg.col, rank[id(cfg)])

    settled: list = []
    for cfg in sorted(order, key=lambda cfg: (id(cfg) not in ahead, *reading(cfg))):
        if _overlaps_with_gap(_aabb(cfg), settled, metrics):
            _make_way(cfg, settled, order, board_w, metrics)
        settled.append(cfg)
    moved = True
    while moved:
        moved = False
        for cfg in sorted(order, key=reading):
            rest = gravity_slot(cfg, [p for p in order if p is not cfg], metrics)
            if rest != (cfg.col, cfg.row):
                cfg.col, cfg.row = rest
                moved = True


def drop(order: Sequence, carried, slot: tuple[int, int], metrics: BoardMetrics, board_w: int) -> None:
    """The board once ``carried`` is let go at ``slot``, in place.

    ``order`` is every card where it stood when ``carried`` was picked up,
    ``carried`` among them, settled (:func:`pack`).

    1. The TARGET is the card the carried one is let go onto: the one whose
       place the slot is in -- standing over the slot's top-left corner.  Let
       go on empty space, there is none.
    2. The two swap: the target takes the place the carried card stood in,
       and the carried card takes the slot.  Let go on empty space, the
       carried card takes the slot.
    3. :func:`pack` settles the board with those two first: where their new
       places overlap, the one further up and left keeps its place and the
       other makes way, and every other card makes way for both.  Then
       everything falls north-west.

    Let go where it was picked up, it finds every card where it stood.
    """

    if not any(cfg is carried for cfg in order):
        raise ValueError("the carried card must belong to the board")
    x, y = slot
    target = next(
        (
            cfg
            for cfg in order
            if cfg is not carried
            and cfg.col <= x < cfg.col + cfg.width
            and cfg.row <= y < cfg.row + cfg.height
        ),
        None,
    )
    if target is not None:
        target.col, target.row = carried.col, carried.row
    carried.col, carried.row = x, y
    pack(order, metrics, board_w, first=(carried,) if target is None else (carried, target))


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
    choosing it lets the card go onto the card there, and the two swap
    (:func:`drop`).  ``cfg`` only says how big the carried card is.
    ``seen`` defaults to ``point``.
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
