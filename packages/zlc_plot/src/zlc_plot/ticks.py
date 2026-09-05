"""Tick labels: one law, one ladder, for every axis-like surface.

Every labelled axis prints at least two DISTINCT labels, and every label it
prints lies inside the ROOM its surface owns.  The room is geometry the
layout states once per axes (:class:`~zlc_plot.layout.Room`): the axis's
own extent plus, at each end, the free margin when nothing labelled lies
beyond it, or half the gap to the neighbour that labels the same row --
the image and its distribution rail split their gap, the rolling history
and its rail split theirs, and facet cells all get half a gap so their
marks stay identical.  A bare figure with no declared room owns its
margins.

One ladder decides what fills the room, in one order.  At the size the
caller draws, the finest lattice whose labels all lie in the room a digit
apart; when none does, fewer labels; when the count is at its floor,
smaller labels down to :data:`MIN_TICK_LABEL_PT`; for a surface with a
declared vocabulary, shorter spellings; and only then may the survivors
touch.  An edge label that does not fit centred on its tick is anchored
inward -- a label box need not straddle its tick -- which is what lets a
range end at "2000" without "2000" hanging into the rail beside it, and
lets the rail's own zero stand at its left edge without leaning out.  A
tick whose label cannot be placed even so is not one of that axis's
ticks: it is never refilled, never centred over the neighbour, and never
blanked with its neighbour left alone.

A count rail is the same ladder with a declared candidate list: zero and
the bound when both fit, the bound alone otherwise.  The bound is the
label that carries the information and is never dropped; the zero is the
only optional label on any surface.  Two identical labels are one
statement, so a declared pair spells itself longer until its ends differ.
Enumerated axes -- a pulse's channel rows, a categorical x -- are not tick
axes and stand outside this law, and a degenerate interval has one
statement to make.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from numbers import Integral
import math
import types
from typing import Any

from matplotlib import rcParams
import matplotlib.ticker as ticker
import numpy as np


#: How small a tick label may be shrunk before crowding is preferred to
#: illegibility.  Below this a number is a smudge, and two smudges apart say
#: less than two touching digits.
MIN_TICK_LABEL_PT = 3.0

#: Each rung of the size ladder draws the labels this much smaller.
_SHRINK = 0.8

#: Where the compact offset text is written -- BOTH its parts, the common
#: scale and the common constant, which modify the same labels and so belong
#: in the same place.
#:
#: In the figure's own corners: the far end of the row of x tick labels, and
#: the top of the column of y tick labels.  Those are where a reader of a
#: scientific plot looks, and they are the two corners nothing else uses --
#: the axis labels are centred on their sides and a title is centred on top.
#:
#: Not in axes fractions, which is what this was.  Below a panel's axes,
#: ``-0.1`` is outside the figure once the tick labels and the axis label have
#: taken the bottom margin, and the two-line form -- a scale AND a constant --
#: was printed off the canvas entirely.  Beside a grid CELL, the margin an
#: axes fraction reaches into is not margin at all: it is the next cell.
_OFFSET_PLACEMENT = {
    "x": ((0.995, 0.008), "figure", "right", "bottom"),
    "y": ((0.008, 0.995), "figure", "left", "top"),
}


def _label_size_pt(text: str, size_pt: float) -> tuple[float, float]:
    """Price one label, in points, at the size it will be drawn.

    The size is passed IN.  It used to be read from ``rcParams``, which is
    the PANEL's tick size: a grid cell draws at its own smaller one, so every
    candidate layout was priced against labels twice the width of the ones
    that would appear, and the axis carried half the labels it had room for.
    """

    from matplotlib import rcParams

    from .layout import _text_size_pt

    families = tuple(rcParams.get("font.sans-serif", ("DejaVu Sans",)))
    return _text_size_pt(str(text), families or ("DejaVu Sans",), float(size_pt))


def _label_line_pt(size_pt: float) -> float:
    """The height of one line of labels as the renderer lays it out.

    Not the ink of any label: matplotlib gives every single-line text the
    line of "lp" -- an ascender over a descender -- so that baselines agree,
    and that is the box a label occupies whatever digits it holds.  Priced
    by the ink of "Ay" instead, seven labels were predicted to clear on a
    histogram's y axis and drew 0.06 pt too close.
    """

    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    families = tuple(rcParams.get("font.sans-serif", ("DejaVu Sans",))) or ("DejaVu Sans",)
    _width, height, descent = TextToPath().get_text_width_height_descent(
        "lp", FontProperties(family=list(families), size=float(size_pt)), False
    )
    return float(height) + float(descent)


def compact_number(value: float, *, length: int | None = None) -> str:
    """As many significant digits of one number as ``length`` characters allow.

    NOT engineering notation: it produces %g and falls back to %e, so it
    prints 1.2e+06 where an engineering form would print 1.2 M.

    This is also not :func:`zlc_data.units.format_quantity`, and must not
    become it.  That one shows a QUANTITY -- every digit the value has, with
    the unit it is in -- because a box an operator reads a device through may
    not round.  This one is for a label with a character budget in a crowded
    plot (a colour-bar end, a count rail's bound, a fit parameter, a
    threshold), where fitting is the whole point and the unit is already
    written elsewhere.  Two jobs, two rules; the mistake would be to give
    either one the other's.
    """

    if not math.isfinite(float(value)):
        return "nan"
    numeric = float(value)

    def normalized(text: str) -> str:
        if "e" not in text.lower():
            return text
        mantissa, exponent = text.lower().split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"

    if length is None:
        general = f"{numeric:.4g}"
        return general if "e" not in general.lower() else normalized(f"{numeric:.1e}")
    maximum = max(1, int(length))
    for significant in range(4, 0, -1):
        general = normalized(f"{numeric:.{significant}g}")
        if len(general) <= maximum:
            return general
        scientific = normalized(f"{numeric:.{significant - 1}e}")
        if len(scientific) <= maximum:
            return scientific
    return normalized(f"{numeric:.0e}")


def count_label(value: float, length: int | None) -> str:
    """A count rail's label: every digit, or as many as ``length`` allows."""

    return f"{float(value):.0f}" if length is None else compact_number(value, length=length)


# --------------------------------------------------------------------- room


def declare_room(axes: Any, room: Any) -> None:
    """Stamp the room the layout gave this axes; read at tick time."""

    axes._zlc_room = room


def _room_pt(axes: Any, axis: Any) -> tuple[float, float, float | None] | None:
    """Room in points along ``axis``: before its low end, past its high end,
    and beyond the side its labels hang on (None when that side is unknown).

    From the layout's declaration when there is one; a bare figure's axes
    owns its distance to the figure's edges.  In points, like the extent,
    so DPR, resize and export dpi cannot change the answer.
    """

    figure = getattr(axes, "figure", None)
    if figure is None:
        return None
    dots_per_point = float(figure.dpi) / 72.0
    if dots_per_point <= 0.0:
        return None
    horizontal = axis is getattr(axes, "xaxis", None)
    side = str(axis.get_ticks_position())
    room = getattr(axes, "_zlc_room", None)
    if room is None:
        bbox, frame = axes.bbox, figure.bbox
        if horizontal:
            before, after = bbox.x0 - frame.x0, frame.x1 - bbox.x1
            across = bbox.y0 - frame.y0 if side != "top" else frame.y1 - bbox.y1
        else:
            before, after = bbox.y0 - frame.y0, frame.y1 - bbox.y1
            across = bbox.x0 - frame.x0 if side != "right" else frame.x1 - bbox.x1
        before, after, across = (
            before / dots_per_point, after / dots_per_point, across / dots_per_point
        )
    else:
        width = float(figure.bbox.width) / dots_per_point
        height = float(figure.bbox.height) / dots_per_point
        if horizontal:
            across = (room.top if side == "top" else room.bottom) * height
            before, after = room.left * width, room.right * width
        else:
            across = (room.right if side == "right" else room.left) * width
            before, after = room.bottom * height, room.top * height
    # THE CORNER BELONGS TO THE OTHER AXIS.  The margin past an axis's low
    # end is where its partner's labels hang -- the y labels' column left of
    # the x axis, the x labels' row below the y axis -- so at that end the
    # room is nothing, and the corner label of each axis is anchored inward
    # instead of the two zeros printing over each other in the corner.
    other = getattr(axes, "yaxis" if horizontal else "xaxis", None)
    if other is not None and _labels_shown(other):
        other_side = str(other.get_ticks_position())
        if horizontal:
            if other_side == "right":
                after = 0.0
            else:
                before = 0.0
        elif other_side == "top":
            after = 0.0
        else:
            before = 0.0
    return before, after, across


def _labels_shown(axis: Any) -> bool:
    """Whether this axis is a labelled one at all.

    Decided by its formatter, not by whether a given cell happens to show
    its labels: a grid gates labels per cell, and the cells must still lay
    identical ticks, so the corner is the partner's in every cell alike.
    """

    return not isinstance(axis.get_major_formatter(), ticker.NullFormatter)


@dataclass(frozen=True)
class _Candidate:
    """One way an axis could be labelled: its ticks and what they say."""

    ticks: tuple[float, ...]
    texts: tuple[str, ...]
    payload: Any = None


@dataclass(frozen=True)
class _Placement:
    """A candidate judged fit to draw: where each label sits, at what size."""

    ticks: tuple[float, ...]
    texts: tuple[str, ...]
    #: Per tick: ``center`` on it, or anchored inward -- ``start`` for the
    #: lowest tick (the box begins AT the tick), ``end`` for the highest.
    aligns: tuple[str, ...]
    size_pt: float
    payload: Any = None


_EMPTY = _Placement((), (), (), MIN_TICK_LABEL_PT)


class _MeasuredLocator(ticker.Locator):
    """The ladder, once: room -> fit -> clear -> inward -> fewer -> smaller -> touch.

    A subclass only says which candidates an axis has, in tiers from the
    most preferred spelling to the least, each tier from fewest labels to
    most.  This class prices them against the extent the axis is painted at
    and the room the layout gave it, and decides.
    """

    #: No axis is ever given one label.  A single label names a point, not a
    #: scale, and an axis that cannot afford two labels is better crowded
    #: than mute.  A count rail lowers this to one: its bound alone is the
    #: information, and its zero is the one optional label there is.
    FLOOR = 2

    def __init__(
        self,
        *,
        max_ticks: int = 8,
        label_pt: float = 10.0,
        measure: "Callable[[str, float], tuple[float, float]] | None" = None,
    ) -> None:
        super().__init__()
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, Integral) or max_ticks <= 0:
            raise ValueError("max_ticks must be a positive integer")
        if measure is not None and not callable(measure):
            raise TypeError("measure must be callable or None")
        if not float(label_pt) > 0.0:
            raise ValueError("label_pt must be positive")
        self.max_ticks = max(int(max_ticks), 2)
        #: How a label is measured, and at what size it is drawn.  Given
        #: these, the locator spends the axis's ACTUAL extent on the labels
        #: that will ACTUALLY appear.
        self.measure = measure
        self.label_pt = float(label_pt)
        #: The size the labels are drawn at now.  Equal to ``label_pt`` until
        #: the ladder had to shrink them, which is the only thing that does.
        self.drawn_pt = float(label_pt)
        self.ticks: list[float] = []
        self.texts: tuple[str, ...] = ()
        self.aligns: tuple[str, ...] = ()
        self._tick_cache_key: tuple[object, ...] | None = None

    # ------------------------------------------------------------ geometry

    def _geometry(self) -> tuple[float, float, float, float | None, bool] | None:
        """(extent, before, after, across, horizontal), in points, or None
        for a bare locator with nothing to measure against."""

        if self.measure is None or self.axis is None:
            return None
        axes = getattr(self.axis, "axes", None)
        figure = getattr(axes, "figure", None)
        if axes is None or figure is None:
            return None
        dots_per_point = float(figure.dpi) / 72.0
        if dots_per_point <= 0.0:
            return None
        horizontal = self.axis is getattr(axes, "xaxis", None)
        extent = float(axes.bbox.width if horizontal else axes.bbox.height)
        room = _room_pt(axes, self.axis)
        if room is None:
            return None
        before, after, across = room
        return extent / dots_per_point, before, after, across, horizontal

    def _positions(
        self, lower: float, upper: float, ticks: Sequence[float], extent: float
    ) -> list[float]:
        """Where each tick is drawn along the axis, in points from its low end.

        Through the axis's own scale, so a logarithmic axis is priced where
        its labels land and not where their values would fall on a ruler.
        """

        values = np.asarray([lower, upper, *ticks], dtype=float)
        mapped = values
        transform = getattr(self.axis, "get_transform", None)
        if callable(transform):
            try:
                mapped = np.asarray(transform().transform(values), dtype=float).reshape(-1)
            except (ValueError, TypeError, FloatingPointError):
                mapped = values
        if mapped.shape != values.shape or not np.isfinite(mapped).all():
            mapped = values
        low, high = float(mapped[0]), float(mapped[1])
        if high <= low:
            return [0.0 for _tick in ticks]
        return [(float(value) - low) / (high - low) * extent for value in mapped[2:]]

    def _size_ladder(self) -> list[float]:
        sizes = [self.label_pt]
        while sizes[-1] > MIN_TICK_LABEL_PT:
            sizes.append(max(MIN_TICK_LABEL_PT, sizes[-1] * _SHRINK))
        return sizes

    # --------------------------------------------------------------- judge

    #: Whether the width of a label across the axis -- into the margin its
    #: labels hang in -- is priced.  Only a declared surface asks: the
    #: colorbar's ends grow longer to differ, and the margin is what stops
    #: them.  A coordinate axis takes an offset instead of growing.
    PRICE_ACROSS = False

    def _judge(
        self,
        lower: float,
        upper: float,
        candidate: _Candidate,
        size_pt: float,
        *,
        room: bool = True,
        clear: bool = True,
    ) -> _Placement | None:
        """This candidate at this size, placed -- or None if it cannot be.

        Every label sits in the room: centred on its tick, or -- for the
        lowest and the highest -- anchored inward when centred would leave.
        Neighbours stay one digit apart.  Identical labels count once: two
        that say the same thing are one statement.
        """

        ticks, texts = candidate.ticks, candidate.texts
        count = len(ticks)
        floor = min(self.FLOOR, count) if count else self.FLOOR
        if count < self.FLOOR or len(set(texts)) < floor:
            return None
        geometry = self._geometry()
        centred = tuple("center" for _tick in ticks)
        if geometry is None or upper <= lower:
            return _Placement(ticks, texts, centred, size_pt, candidate.payload)
        extent, before, after, across, horizontal = geometry
        if extent <= 0.0:
            return _Placement(ticks, texts, centred, size_pt, candidate.payload)
        assert self.measure is not None
        widths = [self.measure(text, size_pt)[0] for text in texts]
        line = _label_line_pt(size_pt)
        # Side by side, what a label costs along the axis is its own width;
        # stacked, it is its LINE, whatever it says -- an ascender over a
        # descender, which is the box the renderer reserves.
        along = widths if horizontal else [line] * count
        if (
            room
            and self.PRICE_ACROSS
            and across is not None
            and max(line if horizontal else width for width in widths) > across + 1e-9
        ):
            return None
        positions = self._positions(lower, upper, ticks, extent)
        order = sorted(range(count), key=lambda item: positions[item])
        aligns = ["center"] * count
        boxes: list[tuple[float, float]] = []
        for rank, index in enumerate(order):
            position, size = positions[index], along[index]
            options = [("center", position - size / 2.0, position + size / 2.0)]
            if rank == 0:
                options.append(("start", position, position + size))
            if rank == count - 1:
                options.append(("end", position - size, position))
            chosen = None
            for name, start, end in options:
                if not room or (
                    start >= -before - 1e-9 and end <= extent + after + 1e-9
                ):
                    chosen = (name, start, end)
                    break
            if chosen is None:
                return None
            aligns[index] = chosen[0]
            boxes.append((chosen[1], chosen[2]))
        if clear:
            digit = self.measure("0", size_pt)[0]
            for (_start, end), (start, _end) in zip(boxes, boxes[1:]):
                if start - end < digit - 1e-9:
                    return None
        return _Placement(ticks, texts, tuple(aligns), size_pt, candidate.payload)

    def _settle(
        self,
        lower: float,
        upper: float,
        tiers: Sequence[Sequence[_Candidate]],
    ) -> _Placement:
        """Walk the ladder over these candidates and return what is drawn.

        ``tiers`` run from the most preferred spelling to the least; within a
        tier the candidates run from fewest labels to most.  At the drawn
        size the finest admissible candidate wins; below it, fewer labels
        cost nothing, so each smaller size takes the coarsest that fits.
        """

        sizes = self._size_ladder()
        ranked_tiers = [
            [
                candidate
                for candidate in tier
                if self.FLOOR <= len(candidate.ticks) <= self.max_ticks
            ]
            for tier in tiers
        ]
        for ranked in ranked_tiers:
            chosen = None
            for candidate in ranked:
                placement = self._judge(lower, upper, candidate, sizes[0])
                if placement is not None:
                    chosen = placement
            if chosen is not None:
                return chosen
            for size in sizes[1:]:
                for candidate in ranked:
                    placement = self._judge(lower, upper, candidate, size)
                    if placement is not None:
                        return placement
        # Nothing fits its room at the readable floor: the axis says what it
        # can.  The most preferred spelling, the fewest labels, as small as
        # is still readable, touching if they must -- over the edge only if
        # even that is not enough, which the guard tests prove the product's
        # layouts never ask for.
        for ranked in ranked_tiers:
            if not ranked:
                continue
            candidate = ranked[0]
            return (
                self._judge(lower, upper, candidate, sizes[-1], clear=False)
                or self._judge(lower, upper, candidate, sizes[-1], room=False, clear=False)
                or _EMPTY
            )
        return _EMPTY

    # --------------------------------------------------------------- state

    def _store(self, placement: _Placement, reverse: bool) -> None:
        ticks = list(placement.ticks)
        texts = tuple(placement.texts)
        aligns = tuple(placement.aligns)
        if reverse:
            ticks.reverse()
            texts = tuple(reversed(texts))
            aligns = tuple(reversed(aligns))
        self.ticks = ticks
        self.texts = texts
        self.aligns = aligns
        self._apply_drawn_size(placement.size_pt)

    def _apply_drawn_size(self, size_pt: float) -> None:
        """Draw the labels at the size this layout was priced at.

        The policy owns both halves -- how many labels, and how big -- because
        they are one decision taken in one order: fewer first, smaller only
        when there are no fewer to give.  Two owners is how an axis came to be
        measured at one size and drawn at another.
        """

        if self.axis is None or abs(size_pt - self.drawn_pt) < 1.0e-9:
            return
        self.drawn_pt = float(size_pt)
        name = "x" if self.axis is getattr(self.axis.axes, "xaxis", None) else "y"
        self.axis.axes.tick_params(axis=name, labelsize=float(size_pt))

    def _nearest(self, value: float) -> int | None:
        if not self.ticks:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return int(np.argmin([abs(numeric - tick) for tick in self.ticks]))

    def text_for(self, value: float) -> str:
        """What the tick nearest ``value`` says."""

        index = self._nearest(value)
        return "" if index is None or index >= len(self.texts) else self.texts[index]

    def align_for(self, value: float) -> str:
        """How the label nearest ``value`` sits on its tick."""

        index = self._nearest(value)
        return "center" if index is None or index >= len(self.aligns) else self.aligns[index]

    def _cache_key(self, vmin: float, vmax: float) -> tuple[object, ...]:
        axis = self.axis
        axes = getattr(axis, "axes", None)
        return (
            float(vmin), float(vmax), id(axis),
            id(getattr(axes, "figure", None)), getattr(axis, "axis_name", None),
            self._geometry(), self.max_ticks, self.label_pt, id(self.measure),
            tuple(rcParams.get("font.sans-serif", ())),
        )

    def __call__(self) -> list[float]:
        if self.axis is None:
            return []
        vmin, vmax = self.axis.get_view_interval()
        return [] if vmax == vmin else self.tick_values(vmin, vmax)


class _AlignedFormatter(ticker.Formatter):
    """A formatter that says what its locator decided, where it decided it.

    The ladder may anchor an edge label inward instead of centring it on
    its tick; the tick artists are told so here, at the one moment
    matplotlib hands them over with their values, before any of them is
    drawn.
    """

    def __init__(self, locator: _MeasuredLocator) -> None:
        super().__init__()
        self.locator = locator

    def __call__(self, value: float, pos: int | None = None) -> str:
        del pos
        return self.locator.text_for(value)

    def format_ticks(self, values: Sequence[float]) -> list[str]:
        axis = self.axis
        if axis is not None:
            horizontal = axis is getattr(axis.axes, "xaxis", None)
            for tick, value in zip(axis.get_major_ticks(len(values)), values):
                align = self.locator.align_for(value)
                for label in (tick.label1, tick.label2):
                    if horizontal:
                        label.set_horizontalalignment(
                            {"start": "left", "end": "right"}.get(align, "center")
                        )
                    else:
                        label.set_verticalalignment(
                            {"start": "bottom", "end": "top"}.get(align, "center")
                        )
        return [self(value, index) for index, value in enumerate(values)]


# -------------------------------------------------------- coordinate axes


class SmartOffsetLocator(_MeasuredLocator):
    """Separate a large common offset from short coordinate tick labels.

    The candidates are the (1, 2, 5) x 10^k lattice, coarse to fine; on a
    logarithmic axis the decade families come first, coarsest to finest,
    with the same lattice behind them for a narrow range.
    """

    def __init__(
        self,
        steps: Sequence[int] = (1, 2, 5),
        max_ticks: int = 8,
        oom: int = 3,
        label_pt: float = 10.0,
        measure: "Callable[[str, float], tuple[float, float]] | None" = None,
    ) -> None:
        super().__init__(max_ticks=max_ticks, label_pt=label_pt, measure=measure)
        selected_steps = tuple(steps)
        if not selected_steps or any(
            isinstance(step, bool) or not isinstance(step, Integral) or step <= 0
            for step in selected_steps
        ):
            raise ValueError("steps must contain positive integers")
        if isinstance(oom, bool) or not isinstance(oom, Integral) or oom <= 0:
            raise ValueError("oom must be a positive integer")
        self.steps = tuple(int(step) for step in selected_steps)
        self.oom = int(oom)
        self._settled: tuple[int, int] | None = None
        self.k = 0
        self.m = 0
        self.C = 0.0
        self.C_int = 0
        self.C_exp = 0
        self.step = 1
        self.n_array: list[int] = []

    def _logarithmic(self) -> bool:
        scale = getattr(self.axis, "get_scale", None)
        return callable(scale) and str(scale()) == "log"

    def _lattice_candidate(
        self, lower: float, upper: float, step: int, decade: int
    ) -> _Candidate | None:
        ticks, indices, offset_int, exponent, scale, label_decade = self._lay_out(
            lower, upper, step, decade, False, max_count=self.max_ticks
        )
        if len(ticks) < self.FLOOR:
            return None
        # Plain unless the labels would mostly repeat one another: an axis
        # whose labels ARE its coordinates is easier to read than one
        # carrying a "+2080" in the corner, and an offset that shortens
        # nothing is a second thing to read for no saving at all.
        if self._carries_a_common_part(indices, step):
            offset = self._lay_out(
                lower, upper, step, decade, True, max_count=self.max_ticks
            )
            if len(offset[0]) >= self.FLOOR:
                ticks, indices, offset_int, exponent, scale, label_decade = offset
        texts = tuple(
            SmartOffsetFormatter._fmt_scaled_int(index * step, label_decade)
            for index in indices
        )
        return _Candidate(
            tuple(ticks),
            texts,
            payload=(step, decade, tuple(indices), offset_int, exponent, scale, label_decade),
        )

    def _lattice(self, lower: float, upper: float) -> list[_Candidate]:
        """The (1, 2, 5) x 10^k candidates, coarse to fine.

        Laid out by the one function that lays ticks out, not counted by a
        second rule beside it.  How many multiples of a unit lie inside a
        view depends on where the view sits, and in binary floats 1e-6 /
        1e-6 is not always 1 -- an arithmetic estimate said two where there
        was one, and one where there were two.
        """

        span = upper - lower
        exponent = int(np.floor(np.log10(span)))
        lattice = [
            (step, decade)
            for decade in range(exponent + 1, exponent - 4, -1)
            for step in self.steps
        ]
        lattice.sort(key=lambda pair: -float(pair[0]) * 10.0 ** pair[1])
        candidates: list[_Candidate] = []
        for step, decade in lattice:
            unit = float(step) * 10.0**decade
            if unit <= 0.0 or not np.isfinite(unit):
                continue
            candidate = self._lattice_candidate(lower, upper, step, decade)
            if candidate is None:
                if candidates:
                    # Past the finest that still fits the count: every
                    # finer unit only adds ticks.
                    break
                continue
            candidates.append(candidate)
        return candidates

    def _decades(self, lower: float, upper: float) -> list[_Candidate]:
        """A logarithmic axis's candidates: its decades, coarsest family first."""

        if lower <= 0.0:
            return []
        first = int(math.floor(math.log10(lower)))
        last = int(math.ceil(math.log10(upper)))
        decades = [
            k for k in range(first, last + 1)
            if lower - 1e-12 * upper <= 10.0**k <= upper + 1e-12 * upper
        ]
        families: list[tuple[int, tuple[int, ...]]] = [
            (3, (1,)), (2, (1,)), (1, (1,)), (1, (1, 2, 5)), (1, tuple(range(1, 10))),
        ]
        candidates: list[_Candidate] = []
        for stride, subs in families:
            ticks: list[float] = []
            texts: list[str] = []
            for k in range(first, last + 1):
                for sub in subs:
                    value = float(sub) * 10.0**k
                    if not lower <= value <= upper:
                        continue
                    if sub == 1 and k not in decades[::stride]:
                        continue
                    if sub != 1 and stride != 1:
                        continue
                    ticks.append(value)
                    texts.append(SmartOffsetFormatter._fmt_scaled_int(sub, k))
            if len(ticks) < self.FLOOR or len(ticks) > self.max_ticks:
                continue
            candidate = _Candidate(tuple(ticks), tuple(texts), payload=None)
            if candidates and candidates[-1].ticks == candidate.ticks:
                continue
            candidates.append(candidate)
        return candidates

    def _unit(self, lower: float, upper: float) -> _Placement:
        """The unit and placement these labels are drawn with.

        Hysteresis first: keep the unit already in force while it still
        works.  Zoom is continuous and the choice of unit is not, so a view
        drifting across the boundary between two units would otherwise
        re-label the whole axis on every jitter of the wheel.  "Works" is the
        same admissibility the search uses, so a held unit and a chosen one
        are judged by one rule.
        """

        logarithmic = self._logarithmic()
        settled = self._settled
        if settled is not None and not logarithmic:
            held = self._lattice_candidate(lower, upper, *settled)
            if held is not None:
                placement = self._judge(lower, upper, held, self.label_pt)
                if placement is not None:
                    return placement
        tiers = [self._decades(lower, upper) + self._lattice(lower, upper)] if logarithmic else [
            self._lattice(lower, upper)
        ]
        return self._settle(lower, upper, tiers)

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        cache_key = (*self._cache_key(vmin, vmax), self.steps, self.oom)
        if self._tick_cache_key == cache_key:
            return self.ticks
        lower, upper = sorted((float(vmin), float(vmax)))
        if not np.isfinite((lower, upper)).all() or lower == upper:
            self._store(_EMPTY, False)
            self.k = self.m = self.C_int = self.C_exp = 0
            self.C = 0.0
            self.step = 1
            self.n_array = []
            self._tick_cache_key = cache_key
            return self.ticks
        placement = self._unit(lower, upper)
        payload = placement.payload
        if payload is None:
            self._settled = None
            self.step, self.m, self.k, self.C_int, self.C_exp = 1, 0, 0, 0, 0
            self.C = 0.0
            self.n_array = []
        else:
            step, decade, indices, offset_int, exponent, scale, label_decade = payload
            self._settled = (step, decade)
            self.step = step
            self.m = label_decade
            self.k = scale
            self.C_int = offset_int
            self.C_exp = exponent
            self.C = float(Decimal(offset_int).scaleb(exponent))
            self.n_array = list(indices)
            if vmin > vmax:
                self.n_array.reverse()
        self._store(placement, vmin > vmax)
        self._tick_cache_key = cache_key
        return self.ticks

    def _lay_out(
        self,
        lower: float,
        upper: float,
        step: int,
        decade: int,
        use_offset: bool,
        *,
        max_count: int | None = None,
    ) -> tuple[list[float], list[int], int, int, int, int]:
        """One complete tick layout: ticks, their indices, and the offset.

        ``max_count`` prices a unit without enumerating a lattice that is
        known to be inadmissibly dense for this axis.
        """

        with localcontext() as context:
            context.prec = 32
            lower_decimal = Decimal.from_float(lower)
            upper_decimal = Decimal.from_float(upper)
            unit = Decimal(step).scaleb(decade)
            # The offset is a TICK -- the one at or below the data -- not a
            # rounding of the range's centre at some coarser decade.  Rounding
            # the centre could place it outside the range entirely, which is
            # how an axis over 5180..5220 printed labels from -4820 to -4780
            # beside "+10000", and how one over -1..2 announced "-100".
            # Anchored on the lattice, every label is a distance forward from
            # a tick, and the anchor moves only when the view crosses a tick,
            # so a small pan renumbers nothing.
            exponent = decade
            offset_int = 0
            offset = Decimal(0)
            if use_offset:
                multiple = int(
                    (lower_decimal / unit).to_integral_value(rounding=ROUND_FLOOR)
                )
                offset_int = multiple * step
                offset = Decimal(offset_int).scaleb(exponent)
                if not np.isfinite(float(offset)):
                    offset_int, exponent = 0, 0
                    offset = Decimal(0)
            first = int(
                ((lower_decimal - offset) / unit).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
            last = int(
                ((upper_decimal - offset) / unit).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )
            if max_count is not None and last - first + 1 > max_count:
                return [], [], 0, 0, 0, decade
            distinct: dict[float, tuple[Decimal, int]] = {}
            for n in range(first, last + 1):
                target = Decimal(n) * unit + offset
                tick = float(target)
                error = abs(Decimal.from_float(tick) - target)
                if tick not in distinct or error < distinct[tick][0]:
                    distinct[tick] = (error, n)
            ticks = list(distinct)
            indices = [value[1] for value in distinct.values()]
            residual = (
                Decimal(max(map(abs, indices), default=0) * step).scaleb(decade)
            )
        scale = 0
        if indices and (
            decade <= -self.oom or residual >= Decimal(1).scaleb(self.oom + 1)
        ):
            scale, decade = decade, 0
        return ticks, indices, offset_int, exponent, scale, decade

    def _carries_a_common_part(self, indices: Sequence[int], step: int) -> bool:
        """Whether these labels would mostly repeat each other.

        A label should say what distinguishes its tick from the next one.  On
        a view that sits far from zero in units of its own tick, most of every
        label is the part they all share -- ``10000000, 10000001, ...`` for a
        window four tenths wide -- and that shared part belongs in the offset,
        once, not in every label.

        "Far" is :attr:`oom` decades above the tick unit, the same measure
        this module already uses to decide when a common SCALE factor comes
        out; both questions are "do these labels carry more digits than they
        distinguish".
        """

        largest = max((abs(index) for index in indices), default=0) * int(step)
        return largest >= 10**self.oom


class SmartOffsetFormatter(_AlignedFormatter):
    """Formatter paired with :class:`SmartOffsetLocator`."""

    def __init__(
        self,
        locator: SmartOffsetLocator,
        axis_type: str = "y",
        offset_xy: tuple[float, float] | None = None,
        offset_coords: str = "axes",
        offset_ha: str | None = None,
        offset_va: str | None = None,
    ) -> None:
        super().__init__(locator)
        if axis_type not in {"x", "y"}:
            raise ValueError("axis_type must be 'x' or 'y'")
        if offset_coords not in {"axes", "data", "figure"}:
            raise ValueError("offset_coords must be 'axes', 'data', or 'figure'")
        self.axis_type = axis_type
        self._offset_xy = offset_xy
        self._offset_coords = offset_coords
        self._offset_ha = offset_ha
        self._offset_va = offset_va

    def set_axis(self, axis: Any) -> None:
        super().set_axis(axis)

        def apply_offset(offset: Any) -> None:
            if self._offset_xy is None:
                return
            transforms = {
                "axes": axis.axes.transAxes,
                "data": axis.axes.transData,
                "figure": axis.axes.figure.transFigure,
            }
            offset.set_transform(transforms[self._offset_coords])
            offset.set_position(self._offset_xy)
            if self._offset_ha is not None:
                offset.set_ha(self._offset_ha)
            if self._offset_va is not None:
                offset.set_va(self._offset_va)
            offset.set_clip_on(False)

        if (
            getattr(axis, "_smart_offset_patched_by", None) is not self
            and hasattr(axis, "_update_offset_text_position")
        ):
            if not hasattr(axis, "_smart_offset_original_update_position"):
                axis._smart_offset_original_update_position = (
                    axis._update_offset_text_position
                )

            def patched(target: Any, *args: object, **kwargs: object) -> Any:
                result = target._smart_offset_original_update_position(
                    *args,
                    **kwargs,
                )
                apply_offset(target.get_offset_text())
                return result

            axis._update_offset_text_position = types.MethodType(patched, axis)
            axis._smart_offset_patched_by = self

    @staticmethod
    def _fmt_scaled_int(
        value_int: int,
        exp10: int,
        force_sign: bool = False,
    ) -> str:
        value = int(value_int)
        if value == 0:
            return "+0" if force_sign else "0"
        sign = "-" if value < 0 else ("+" if force_sign else "")
        base = abs(value)
        if exp10 >= 0:
            return sign + str(base * 10**exp10)
        denominator = 10 ** (-exp10)
        quotient, remainder = divmod(base, denominator)
        fraction = f"{remainder:0{-exp10}d}".rstrip("0")
        return f"{sign}{quotient}.{fraction}" if fraction else f"{sign}{quotient}"

    def _format_C(self) -> str:
        plain = self._fmt_scaled_int(
            self.locator.C_int,
            int(self.locator.C_exp),
            force_sign=True,
        )
        if plain in ("", "+0", "-0"):
            return ""
        max_length = 8
        if len(plain) <= max_length:
            return plain
        value = int(self.locator.C_int)
        sign = "-" if value < 0 else "+"
        digits = str(abs(value))
        if digits == "0":
            return ""
        exponent = int(self.locator.C_exp) + len(digits) - 1
        suffix = f"e{exponent:d}"
        keep = max(0, max_length - 2 - len(suffix))
        fraction = digits[1:keep]
        return sign + digits[0] + (("." + fraction) if fraction else "") + suffix

    def get_offset(self) -> str:
        parts = []
        if self.locator.k != 0:
            parts.append(f"×1e{self.locator.k}")
        constant = self._format_C()
        if constant:
            parts.append(constant)
        if not parts:
            return ""
        if len(parts) == 2:
            # A scale and an offset are two separate statements about the same
            # labels, and run together -- "x1e4+3.84e11" -- they read as one
            # arithmetic expression that means nothing.  The x axis has the
            # width for two lines under it; the y axis has one line above it.
            return "\n".join(parts) if self.axis_type == "x" else " ".join(parts)
        return "".join(parts)


# ----------------------------------------------------------- declared axes


class DeclaredLocator(_MeasuredLocator):
    """An axis whose ticks are declared -- evenly spaced ends -- not searched.

    The distribution rail beside an image counts pixels per bin from zero,
    and two ticks, the floor and the top, say everything there is; the
    colorbar's two ends bound its scale.  What the ladder decides for such
    an axis is which of the declared labels fit, how they are spelled and
    how big they are drawn -- never which values to tick.
    """

    PRICE_ACROSS = True

    def __init__(
        self,
        count: int,
        *,
        text: Callable[[float, int | None], str],
        text_lengths: Sequence[int | None] = (None,),
        zero_optional: bool = True,
        label_pt: float,
        measure: "Callable[[str, float], tuple[float, float]] | None" = None,
    ) -> None:
        super().__init__(max_ticks=max(2, int(count)), label_pt=label_pt, measure=measure)
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("a declared tick count must be at least two")
        if not callable(text):
            raise TypeError("text must be callable")
        self.count = int(count)
        self.text = text
        #: Spellings from the most preferred to the least: a rail prefers
        #: every digit and falls back to fewer; a colorbar prefers the
        #: shortest that tells its two ends apart.
        self.text_lengths = tuple(text_lengths) or (None,)
        self.zero_optional = bool(zero_optional)
        #: The bound alone is the information a count rail carries, so it
        #: may stand alone; a pair of ends is nothing without both.
        self.FLOOR = 1 if self.zero_optional else 2

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        cache_key = (*self._cache_key(vmin, vmax), self.count, self.text_lengths, self.zero_optional)
        if self._tick_cache_key == cache_key:
            return self.ticks
        lower, upper = sorted((float(vmin), float(vmax)))
        if not np.isfinite((lower, upper)).all() or lower == upper:
            self._store(_EMPTY, False)
            self._tick_cache_key = cache_key
            return self.ticks
        ticks = tuple(float(value) for value in np.linspace(lower, upper, self.count))
        tiers: list[list[_Candidate]] = []
        for length in self.text_lengths:
            texts = tuple(self.text(tick, length) for tick in ticks)
            tier: list[_Candidate] = []
            if self.zero_optional and ticks[0] == 0.0:
                tier.append(_Candidate(ticks[1:], texts[1:]))
            tier.append(_Candidate(ticks, texts))
            tiers.append(tier)
        self._store(self._settle(lower, upper, tiers), vmin > vmax)
        self._tick_cache_key = cache_key
        return self.ticks


class DeclaredFormatter(_AlignedFormatter):
    """Formatter paired with :class:`DeclaredLocator`."""


# ------------------------------------------------------------- installers


def _measured(axis: Any) -> _MeasuredLocator | None:
    locator = axis.get_major_locator()
    return locator if isinstance(locator, _MeasuredLocator) else None


def apply_smart_ticks(
    axis: Any,
    which: str = "both",
    *,
    label_pt: float,
) -> None:
    """Install the shared tick policy on a coordinate axis.

    THE one place a tick decision is made for a coordinate axis.  How many
    labels an axis carries is not a caller's preference: it follows from the
    extent the axis is painted at, the room the layout gave it and the size
    of the labels themselves, all of which this module can read.  A caller
    says where it draws and how big -- ``label_pt`` is required because the
    policy prices its candidates against the labels that will ACTUALLY
    appear, and it may SHRINK that size, so the caller must not set the
    size again afterwards -- and nothing else.

    A logarithmic y takes the same policy: its candidates are its decade
    families first and the linear lattice, priced where a log axis draws
    them, behind.
    """

    if which not in {"x", "y", "both"}:
        raise ValueError("which must be 'x', 'y', or 'both'")
    size_pt = float(label_pt)
    if not size_pt > 0.0:
        raise ValueError("label_pt must be positive")
    for name in ("x", "y"):
        if which not in (name, "both"):
            continue
        target = axis.xaxis if name == "x" else axis.yaxis
        scale = str(axis.get_xscale() if name == "x" else axis.get_yscale())
        # Reinstalling a locator resets the axis' tick artists, which both
        # reallocates them every frame and leaves them unpositioned until the
        # next full Axis draw.  Install once per configuration.
        signature = (f"smart-{name}", scale, size_pt)
        if getattr(target, "_zlc_tick_signature", None) == signature:
            continue
        locator = SmartOffsetLocator(measure=_label_size_pt, label_pt=size_pt)
        offset_xy, offset_coords, offset_ha, offset_va = _OFFSET_PLACEMENT[name]
        target.set_major_locator(locator)
        target.set_major_formatter(
            SmartOffsetFormatter(
                locator,
                axis_type=name,
                offset_xy=offset_xy,
                offset_coords=offset_coords,
                offset_ha=offset_ha,
                offset_va=offset_va,
            )
        )
        if scale == "log":
            # The 2..9 ticks stay as unlabelled visual guides on a log axis.
            target.set_minor_locator(
                ticker.LogLocator(
                    base=10.0,
                    subs=tuple(float(value) for value in range(2, 10)),
                    numticks=16,
                )
            )
        else:
            # A linear axis has no minor labels or extra grid from this policy.
            target.set_minor_locator(ticker.NullLocator())
        target.set_minor_formatter(ticker.NullFormatter())
        target.get_offset_text().set_visible(True)
        axis.tick_params(axis=name, labelsize=size_pt)
        target._zlc_tick_signature = signature


def apply_declared_ticks(
    axis: Any,
    which: str,
    count: int,
    *,
    label_pt: float,
    text: Callable[[float, int | None], str] = count_label,
    text_lengths: Sequence[int | None] = (None, 4, 3),
    zero_optional: bool = True,
) -> None:
    """Give one axis ``count`` evenly spaced declared ticks, on the ladder.

    For a surface whose numbers are not a coordinate to read off but a bound
    to know: the distribution rail beside an image counts pixels per bin,
    it always starts at zero, and two ticks -- the floor and the top -- say
    everything there is.  Which of them fit, how they are spelled and how
    big they are drawn is the shared ladder's decision, so the same rail
    says the same thing at every preset.  The bound is never dropped; the
    zero is the one label the ladder may leave out (``zero_optional``).
    A colorbar declares both ends as required and spells them longer until
    they differ.
    """

    if which not in {"x", "y"}:
        raise ValueError("which must be 'x' or 'y'")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("a declared tick count must be at least two")
    size_pt = float(label_pt)
    if not size_pt > 0.0:
        raise ValueError("label_pt must be positive")
    target = axis.xaxis if which == "x" else axis.yaxis
    lengths = tuple(text_lengths)
    signature = ("declared", count, size_pt, lengths, bool(zero_optional), text)
    if getattr(target, "_zlc_tick_signature", None) == signature:
        return
    locator = DeclaredLocator(
        count,
        text=text,
        text_lengths=lengths,
        zero_optional=zero_optional,
        label_pt=size_pt,
        measure=_label_size_pt,
    )
    target.set_major_locator(locator)
    target.set_major_formatter(DeclaredFormatter(locator))
    target.set_minor_locator(ticker.NullLocator())
    target.set_minor_formatter(ticker.NullFormatter())
    target.axes.tick_params(axis=which, labelsize=size_pt)
    target.get_offset_text().set_visible(False)
    target._zlc_tick_signature = signature


def declare_colorbar_ticks(
    colorbar: Any,
    *,
    label_pt: float,
    label_chars: int,
) -> None:
    """The colorbar's two ends, on the ladder: the shortest spelling that
    tells them apart and fits the margin they hang in.

    Through the Colorbar's own locator slot, because a colorbar reinstalls
    its axis's locator whenever its norm moves; installed on the axis
    directly, the policy was gone by the second frame.
    """

    size_pt = float(label_pt)
    if not size_pt > 0.0:
        raise ValueError("label_pt must be positive")
    shortest = max(1, int(label_chars))
    lengths = tuple(range(shortest, shortest + 4))
    signature = ("colorbar", size_pt, lengths)
    if getattr(colorbar, "_zlc_tick_signature", None) != signature:
        colorbar._zlc_tick_locator = DeclaredLocator(
            2,
            text=lambda value, length: compact_number(value, length=length),
            text_lengths=lengths,
            zero_optional=False,
            label_pt=size_pt,
            measure=_label_size_pt,
        )
        colorbar._zlc_tick_signature = signature
    locator = colorbar._zlc_tick_locator
    colorbar.locator = locator
    colorbar.formatter = DeclaredFormatter(locator)
    colorbar.minorlocator = ticker.NullLocator()
    colorbar.update_ticks()
    colorbar.ax.tick_params(labelsize=size_pt)


__all__ = [
    "MIN_TICK_LABEL_PT",
    "DeclaredFormatter",
    "DeclaredLocator",
    "SmartOffsetFormatter",
    "SmartOffsetLocator",
    "apply_declared_ticks",
    "apply_smart_ticks",
    "compact_number",
    "count_label",
    "declare_colorbar_ticks",
    "declare_room",
]
