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

__all__ = ["BoardMetrics", "GeomProxy", "board_width", "nearest_anchor",
           "first_free_slot", "gravity_slot", "min_board_width", "pack"]


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


def _aabb(cfg, metrics: BoardMetrics) -> tuple[int, int, int, int]:
    """The card's pixel AABB ``(x0, y0, x1, y1)`` -- top-left ``(col, row)`` plus its size."""
    del metrics
    return (cfg.col, cfg.row, cfg.col + cfg.width, cfg.row + cfg.height)


def _overlaps_with_gap(box: tuple[int, int, int, int], placed, metrics: BoardMetrics) -> bool:
    """True when ``box`` (an ``(x0, y0, x1, y1)`` AABB), EXPANDED by the gap on all sides,
    intersects any already-placed card.  Equivalently: the clear distance between ``box`` and a
    placed card is < gap on the axis where they overlap, so leaving a card exactly gap away
    counts as clear."""
    gap = metrics.gap
    x0, y0, x1, y1 = box
    for p in placed:
        px0, py0, px1, py1 = _aabb(p, metrics)
        if x0 < px1 + gap and px0 < x1 + gap and y0 < py1 + gap and py0 < y1 + gap:
            return True
    return False


def first_free_slot(
    cfg, placed, board_w: int, metrics: BoardMetrics, *, floor: int | None = None
) -> tuple[int, int]:
    """The TOP-MOST then LEFT-MOST free ``(col, row)`` where ``cfg`` fits clear of every ``placed``
    card (gap apart, inside ``board_w``).  This is where a card with no place of its own GOES -- a
    panel just added tiles the top row left-to-right, wraps to the next shelf, and never leaves a
    middle hole.  Candidate points are the gap (origin) plus each placed card's right/bottom edge
    (``+gap``) and its left/top edge (so a card can tuck under a wider one); swept by y then x,
    first feasible wins.

    ``floor`` refuses every row above it.  A card whose own place is taken has to yield, and it
    yields DOWNWARD: searching the whole board would send it up to a hole somewhere else, which is
    the one thing gravity never does.
    """
    gap = metrics.gap
    w, _h = cfg.width, cfg.height
    lowest = gap if floor is None else max(gap, int(floor))
    xs = {gap}
    ys = {lowest}
    for p in placed:
        px0, py0, px1, py1 = _aabb(p, metrics)
        xs.add(px1 + gap)
        ys.add(py1 + gap)
        xs.add(px0)            # also align left edges, so a card can tuck under a wider one
        ys.add(py0)
    max_x = max(gap, board_w - gap - w)
    cand_x = sorted(x for x in xs if gap <= x <= max_x) or [gap]
    for y in sorted(y for y in ys if y >= lowest):
        for x in cand_x:
            if not _overlaps_with_gap((x, y, x + w, y + _h), placed, metrics):
                return (x, y)
    # No candidate fit (should not happen -- placing past the lowest card always clears).
    bottom = max((py1 for *_rest, py1 in (_aabb(p, metrics) for p in placed)), default=0)
    return (gap, max(lowest, bottom + gap) if placed else lowest)


def gravity_slot(cfg, placed, board_w: int, metrics: BoardMetrics) -> tuple[int, int]:
    """Where one card comes to rest, falling NORTH-WEST from where it already is.

    It rises until a card above it -- or the top margin -- stops it, then slides left until a card
    beside it, or the left margin, stops it, and repeats until neither move is possible.  It only
    ever moves up and left and it stops at the FIRST thing in the way: a card put below a wide one
    stays below it, and does not fly off to a free slot beside it.  That is the whole difference
    between gravity and a flow layout, and the reason a board can hold more than one placement.

    A card whose own place is already taken cannot rise or slide out of the overlap, so it yields
    downward instead, to the first free row at or below its own.
    """

    gap = metrics.gap
    width, height = cfg.width, cfg.height
    x = min(max(int(cfg.col), gap), max(gap, board_w - gap - width))
    y = max(int(cfg.row), gap)
    boxes = [_aabb(p, metrics) for p in placed]
    # Each pass strictly lowers x or y, and both take values from a finite set
    # (the margin, and each placed card's right or bottom edge), so this ends.
    for _pass in range(2 * len(boxes) + 2):
        moved = False
        top = gap
        for px0, _py0, px1, py1 in boxes:
            if x < px1 + gap and px0 < x + width + gap and py1 + gap <= y:
                top = max(top, py1 + gap)
        if top < y:
            y, moved = top, True
        left = gap
        for px0, py0, px1, py1 in boxes:
            if y < py1 + gap and py0 < y + height + gap and px1 + gap <= x:
                left = max(left, px1 + gap)
        if left < x:
            x, moved = left, True
        if not moved:
            break
    if _overlaps_with_gap((x, y, x + width, y + height), placed, metrics):
        return first_free_slot(cfg, placed, board_w, metrics, floor=y)
    return (x, y)


def min_board_width(configs: Sequence, metrics: BoardMetrics) -> int:
    """The NARROWEST a board may pack to: one WIDEST card plus both gap margins.  A viewport thinner
    than this still has to fit the widest card, so we clamp up to it -- but NOT to the cards' current
    right-extent: clamping to the extent would RATCHET (once cards spread wide the board could never
    pack narrower), so narrowing the window would never reflow into a single column.  At one-card
    width the gravity packer simply stacks every card in one column, which is the correct reflow."""
    widest = max((c.width for c in configs), default=0)
    return widest + 2 * metrics.gap


def board_width(configs: Sequence, metrics: BoardMetrics) -> int:
    """A fallback packing width for callers without a live viewport (the pure-function tests): two
    of the WIDEST card side by side plus the gap margins, so cards CAN pack side by side.  The real
    GUI passes the scroll viewport width to :func:`pack` instead, so the board wraps at the edge."""
    widest = max((c.width for c in configs), default=0)
    return max(2 * widest + 3 * metrics.gap, min_board_width(configs, metrics))


def pack(
    order: Sequence,
    metrics: BoardMetrics,
    board_w: int | None = None,
    *,
    dropped=None,
) -> bool:
    """Settle every card of ``order`` under north-west gravity, from where it is.

    Each proxy arrives carrying the place its card was PUT -- seeded when the
    panel was added, or authored by the operator's last drop -- and leaves
    carrying where that place comes to rest on a board this wide.  Cards
    settle north-west first, so whatever is above or to the left of a card has
    already taken its place by the time that card falls past it.

    ``dropped`` names the card the operator has just released.  It wins ties
    in that order, and nothing else: its intent decides who yields when two
    cards want the same place, and gravity then treats it like any other.

    ``board_w`` defaults to a two-wide headless width and is always clamped to
    fit one card.  A narrow board clamps positions but the caller keeps the
    authored ones, so widening restores the arrangement rather than leaving
    the operator with the single column the narrow board packed.  Returns
    whether any proxy moved.
    """
    order = list(order)
    if dropped is not None and not any(cfg is dropped for cfg in order):
        raise ValueError("the dropped card must belong to the packed board")
    board_w = (board_width(order, metrics) if board_w is None
               else max(board_w, min_board_width(order, metrics)))
    settling = sorted(
        order,
        key=lambda cfg: (int(cfg.row), int(cfg.col), 0 if cfg is dropped else 1),
    )
    placed: list = []
    moved = False
    for cfg in settling:
        col, row = gravity_slot(cfg, placed, board_w, metrics)
        if (cfg.col, cfg.row) != (col, row):
            cfg.col, cfg.row = col, row
            moved = True
        placed.append(cfg)
    return moved


def nearest_anchor(
    cfg,
    others: Sequence,
    metrics: BoardMetrics,
    board_w: int | None = None,
) -> tuple[int, int]:
    """Nearest two-dimensional gravity anchor to a dropped card's top-left.

    Anchors come from the settled board itself: its origin and every sibling's
    left/right and top/bottom grid lines.  Occupied anchors remain candidates;
    choosing one means that the dropped card displaces the card there.  The
    chosen anchor is resolved *before* gravity, so vertical intent is never
    flattened into a trial list order.  Equal distances prefer the northern,
    then western anchor.
    """

    configs = list(others) + [cfg]
    board_w = board_width(configs, metrics) if board_w is None else board_w
    board_w = max(board_w, min_board_width(configs, metrics))
    drop_x, drop_y = int(round(cfg.col)), int(round(cfg.row))
    width = cfg.width
    xs = {metrics.gap}
    ys = {metrics.gap}
    for other in others:
        left, top, right, bottom = _aabb(other, metrics)
        xs.update((left, right + metrics.gap))
        ys.update((top, bottom + metrics.gap))
    max_x = max(metrics.gap, board_w - metrics.gap - width)
    candidates = (
        (x, y)
        for x in xs
        for y in ys
        if metrics.gap <= x <= max_x and y >= metrics.gap
    )
    return min(
        candidates,
        key=lambda point: (
            (point[0] - drop_x) ** 2 + (point[1] - drop_y) ** 2,
            point[1],
            point[0],
        ),
    )
