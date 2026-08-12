"""Pure north-west-gravity placement for a panel board.

The packer depends only on rectangle geometry.  Qt chrome and renderer size
policy enter through :class:`BoardMetrics`; this module imports neither UI nor
rendering backends.  ``card_size`` remains callable because the answer depends
on the current display scale and must not be captured as stale construction
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

__all__ = ["BoardMetrics", "GeomProxy", "board_width", "nearest_anchor",
           "first_free_slot", "min_board_width", "pack"]


@dataclass(frozen=True)
class BoardMetrics:
    """The two facts the packer cannot derive: the clear gap, and how big a card is.

    ``card_size`` maps a panel-size preset to the card's outer ``(width, height)`` in
    pixels.  It stays a callable so the answer always reflects the CURRENT display
    scale; see the module docstring for why a mapping would be wrong.
    """

    gap: int
    card_size: Callable[[str], "tuple[int, int]"]

    def __post_init__(self) -> None:
        if int(self.gap) != self.gap or self.gap < 0:
            raise ValueError("gap must be a non-negative whole number of pixels")
        if not callable(self.card_size):
            raise TypeError("card_size must be callable: a snapshot would go stale")


class GeomProxy:
    """A placement-only stand-in, so a trial pack never mutates a real card."""

    __slots__ = ("size", "col", "row")

    def __init__(self, size: str, col: int = 0, row: int = 0) -> None:
        self.size = size
        self.col = col
        self.row = row


def _aabb(cfg, metrics: BoardMetrics) -> tuple[int, int, int, int]:
    """The card's pixel AABB ``(x0, y0, x1, y1)`` -- top-left ``(col, row)`` plus its size."""
    w, h = metrics.card_size(cfg.size)
    return (cfg.col, cfg.row, cfg.col + w, cfg.row + h)


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


def first_free_slot(cfg, placed, board_w: int, metrics: BoardMetrics) -> tuple[int, int]:
    """The TOP-MOST then LEFT-MOST free ``(col, row)`` where ``cfg`` fits clear of every ``placed``
    card (gap apart, inside ``board_w``) -- the per-card north-west placement :func:`pack` applies to
    EVERY card in order (so the board tiles the top row left-to-right, wraps to the next shelf, and
    never leaves a middle hole).  Candidate points are the gap (origin) plus each placed card's
    right/bottom edge (``+gap``) and its left/top edge (so a card can tuck under a wider one); swept
    by y then x, first feasible wins."""
    gap = metrics.gap
    w, _h = metrics.card_size(cfg.size)
    xs = {gap}
    ys = {gap}
    for p in placed:
        px0, py0, px1, py1 = _aabb(p, metrics)
        xs.add(px1 + gap)
        ys.add(py1 + gap)
        xs.add(px0)            # also align left edges, so a card can tuck under a wider one
        ys.add(py0)
    max_x = max(gap, board_w - gap - w)
    cand_x = sorted(x for x in xs if gap <= x <= max_x) or [gap]
    for y in sorted(ys):
        for x in cand_x:
            if not _overlaps_with_gap((x, y, x + w, y + _h), placed, metrics):
                return (x, y)
    # No candidate fit (should not happen -- placing past the lowest card always clears).
    bottom = max((py1 for *_rest, py1 in (_aabb(p, metrics) for p in placed)), default=0)
    return (gap, bottom + gap if placed else gap)


def min_board_width(configs: Sequence, metrics: BoardMetrics) -> int:
    """The NARROWEST a board may pack to: one WIDEST card plus both gap margins.  A viewport thinner
    than this still has to fit the widest card, so we clamp up to it -- but NOT to the cards' current
    right-extent: clamping to the extent would RATCHET (once cards spread wide the board could never
    pack narrower), so narrowing the window would never reflow into a single column.  At one-card
    width the gravity packer simply stacks every card in one column, which is the correct reflow."""
    widest = max((metrics.card_size(c.size)[0] for c in configs),
                 default=metrics.card_size("1x2")[0])
    return widest + 2 * metrics.gap


def board_width(configs: Sequence, metrics: BoardMetrics) -> int:
    """A fallback packing width for callers without a live viewport (the pure-function tests): two
    of the WIDEST card side by side plus the gap margins, so cards CAN pack side by side.  The real
    GUI passes the scroll viewport width to :func:`pack` instead, so the board wraps at the edge."""
    widest = max((metrics.card_size(c.size)[0] for c in configs),
                 default=metrics.card_size("1x2")[0])
    return max(2 * widest + 3 * metrics.gap, min_board_width(configs, metrics))


def pack(
    order: Sequence,
    metrics: BoardMetrics,
    board_w: int | None = None,
    *,
    pinned=None,
) -> bool:
    """Apply deterministic north-west gravity to ``order``.

    With no ``pinned`` card, every card takes the first north-west free slot in
    sequence.  A drop supplies one pinned card whose authored anchor remains
    fixed while every other card falls around it in the same way.  This is the
    single distinction needed between ordinary reflow and an operator's
    explicit two-dimensional placement; it does not introduce a second
    packing algorithm.

    ``board_w`` defaults to a two-wide headless width and is always clamped to
    fit one card.  A pinned x-coordinate is likewise clamped for a narrow
    viewport, but its authored coordinate is not mutated by the caller, so
    widening the viewport can restore it.  Returns whether any proxy moved.
    """
    order = list(order)
    board_w = (board_width(order, metrics) if board_w is None
               else max(board_w, min_board_width(order, metrics)))
    placed: list = []
    moved = False
    if pinned is not None:
        if not any(cfg is pinned for cfg in order):
            raise ValueError("the pinned card must belong to the packed board")
        width, _height = metrics.card_size(pinned.size)
        col = min(
            max(int(pinned.col), metrics.gap),
            max(metrics.gap, board_w - metrics.gap - width),
        )
        row = max(int(pinned.row), metrics.gap)
        if (pinned.col, pinned.row) != (col, row):
            pinned.col, pinned.row = col, row
            moved = True
        placed.append(pinned)
    for cfg in order:
        if cfg is pinned:
            continue
        col, row = first_free_slot(cfg, placed, board_w, metrics)
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
    width, _height = metrics.card_size(cfg.size)
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
