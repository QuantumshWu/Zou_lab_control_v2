"""Smart tick locator and formatter for compact scientific plot axes.

The paired objects keep short tick labels around a large common offset and
place the scale/offset text at fixed plot-relative positions.  This
module owns no Figure lifecycle and can be applied identically by notebook,
GUI, and export renderers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
from numbers import Integral
import types
from typing import Any

import matplotlib.ticker as ticker
import numpy as np


#: Clear space beside a tick label, as a multiple of its own width.  The one
#: taste decision in this module; everything else is measured.
TICK_LABEL_GAP_RATIO = 1.3


def _rendered_width_pt(text: str) -> float:
    """Price one label in points, in the font matplotlib will actually use.

    Read from rcParams rather than taken as a parameter: the style config
    writes the tick size and family THERE, so this is the same number the
    renderer draws with, and no caller has to pass its typography down.
    """

    from matplotlib import rcParams

    from .layout import _text_width_pt

    try:
        size = float(rcParams["xtick.labelsize"])
    except (TypeError, ValueError):
        size = float(rcParams.get("font.size", 10.0))
    families = tuple(rcParams.get("font.sans-serif", ("DejaVu Sans",)))
    return _text_width_pt(str(text), families or ("DejaVu Sans",), size)


class SmartOffsetLocator(ticker.Locator):
    """Separate a large common offset from short coordinate tick labels."""

    #: No axis is ever given one tick.  A single label names a point, not a
    #: scale, and an axis that cannot afford two labels is better crowded than
    #: mute.
    FLOOR = 2

    def __init__(
        self,
        steps: Sequence[int] = (1, 2, 5),
        max_ticks: int = 8,
        oom: int = 3,
        measure: "Callable[[str], float] | None" = None,
        gap_ratio: float = 1.3,
        prune_edges: bool = False,
    ) -> None:
        super().__init__()
        selected_steps = tuple(steps)
        if not selected_steps or any(
            isinstance(step, bool) or not isinstance(step, Integral) or step <= 0
            for step in selected_steps
        ):
            raise ValueError("steps must contain positive integers")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            for value in (max_ticks, oom)
        ):
            raise ValueError("max_ticks and oom must be positive integers")
        if measure is not None and not callable(measure):
            raise TypeError("measure must be callable or None")
        if not float(gap_ratio) > 0.0:
            raise ValueError("gap_ratio must be positive")
        self.steps = tuple(int(step) for step in selected_steps)
        self.max_ticks = max(int(max_ticks), self.FLOOR)
        self.oom = int(oom)
        #: How a label's width is obtained, and how much clear space one needs
        #: beside it.  Given these, the locator spends the axis's ACTUAL width
        #: instead of a constant nobody compared against anything.
        self.measure = measure
        self.gap_ratio = float(gap_ratio)
        #: Drop a label that sits on the very edge of the axis.  A panel has a
        #: figure margin to overflow into; a grid cell has its NEIGHBOUR, and
        #: two cells' edge labels then print on top of each other -- which is
        #: what the cells' own locator used ``prune="both"`` for before they
        #: shared this policy.  Only the caller knows which it is.
        self.prune_edges = bool(prune_edges)
        self._settled: tuple[int, int, int] | None = None
        self.k = 0
        self.m = 0
        self.C = 0
        self.C_int = 0
        self.C_exp = 0
        self.step = 1
        self.n_array: list[int] = []
        self.ticks: list[float] = []

    def _unit(self, lower: float, upper: float, budget: int) -> tuple[int, int]:
        """The finest (step, decade) whose ticks fit the budget.

        A budget that is not a cap is not a budget: the old search tried three
        steps at two decades and, when none of them landed inside a min/max
        window, fell back to step 1 -- which is the FINEST option and produced
        up to ten labels however few were asked for.  Walking the whole
        (1, 2, 5) x 10^k lattice from coarse to fine and stopping at the last
        one that still fits makes the number mean what it says.
        """

        span = upper - lower
        exponent = int(np.floor(np.log10(span)))
        lattice = [
            (step, decade)
            for decade in range(exponent + 1, exponent - 4, -1)
            for step in sorted(self.steps, reverse=True)
        ]
        lattice.sort(key=lambda pair: -float(pair[0]) * 10.0 ** pair[1])
        fitting = None
        for step, decade in lattice:  # coarse first: each step adds labels
            unit = float(step) * 10.0**decade
            if unit <= 0.0 or not np.isfinite(unit):
                continue
            # Counted by the one function that lays ticks out, not by a
            # second rule beside it.  How many multiples of the unit lie
            # inside a view depends on where the view sits, and in binary
            # floats 1e-6 / 1e-6 is not always 1 -- an arithmetic estimate
            # said two where there was one, and one where there were two.
            count = len(self._lay_out(lower, upper, step, decade, False)[0])
            if count > budget:
                # Finer only adds more, so the previous one is the finest fit.
                if fitting is not None:
                    return fitting
                # ...unless nothing has reached the floor yet, in which case
                # the floor wins: an axis with one label names a point, not a
                # scale, and a crowded axis is still readable.
                if count >= self.FLOOR:
                    return step, decade
                continue
            if count >= self.FLOOR:
                fitting = (step, decade)
        return fitting if fitting is not None else (min(self.steps), exponent - 4)

    def _budget(self) -> int:
        """How many labels this axis can afford, from its painted width.

        The count used to be a constant that no code compared against the
        space it had, which is why eight four-character labels were asked to
        share a facet cell an inch wide.  ``measure`` prices a label in points
        and the axis knows its own extent without a draw, so the budget is
        arithmetic.  Never below :attr:`FLOOR`.
        """

        if self.measure is None or self.axis is None:
            return self.max_ticks
        axes = getattr(self.axis, "axes", None)
        figure = getattr(axes, "figure", None)
        if axes is None or figure is None:
            return self.max_ticks
        dots_per_point = float(figure.dpi) / 72.0
        if dots_per_point <= 0.0:
            return self.max_ticks
        horizontal = self.axis is getattr(axes, "xaxis", None)
        extent = float(axes.bbox.width if horizontal else axes.bbox.height)
        available = extent / dots_per_point
        # A y label is separated by its line, an x label by its own width; the
        # widest label is the one that has to fit either way.
        widest = self._widest_label_pt() if horizontal else self.measure("0")
        room = available / max(widest * self.gap_ratio, 1e-6)
        return max(self.FLOOR, min(self.max_ticks, int(room)))

    def _widest_label_pt(self) -> float:
        """The width of the longest label this layout would print."""

        assert self.measure is not None
        formatter = getattr(self.axis, "major", None)
        formatter = getattr(formatter, "formatter", None)
        widths = []
        for index, value in enumerate(self.ticks):
            if formatter is not None:
                widths.append(self.measure(formatter(value, index)))
            else:
                widths.append(self.measure(f"{value:g}"))
        return max(widths, default=self.measure("0"))

    def _lay_out(
        self,
        lower: float,
        upper: float,
        step: int,
        decade: int,
        use_offset: bool,
    ) -> tuple[list[float], list[int], int, int, int, int]:
        """One complete tick layout: ticks, their indices, and the offset."""

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

    def _fits(self, ticks: list[float], indices: list[int], step: int, decade: int) -> bool:
        """Whether these labels and their gaps fit the axis's painted width."""

        if self.measure is None or self.axis is None or not ticks:
            return True
        axes = getattr(self.axis, "axes", None)
        figure = getattr(axes, "figure", None)
        if axes is None or figure is None:
            return True
        if self.axis is not getattr(axes, "xaxis", None):
            return True  # stacked vertically, a label's width is not the limit
        dots_per_point = float(figure.dpi) / 72.0
        if dots_per_point <= 0.0:
            return True
        available = float(axes.bbox.width) / dots_per_point
        widest = max(
            (
                self.measure(
                    SmartOffsetFormatter._fmt_scaled_int(index * step, decade)
                )
                for index in indices
            ),
            default=0.0,
        )
        return len(ticks) * widest * self.gap_ratio <= available

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        lower, upper = sorted((float(vmin), float(vmax)))
        if not np.isfinite((lower, upper)).all() or lower == upper:
            self.k = self.m = self.C_int = self.C_exp = 0
            self.C = 0.0
            self.step = 1
            self.ticks = []
            self.n_array = []
            return self.ticks

        budget = self._budget()
        step, decade = self._unit(lower, upper, budget)
        # Hysteresis: keep the unit already in force while it still fits.
        #
        # Zoom is continuous and the choice of unit is not, so a view drifting
        # across the boundary between two units would otherwise re-label the
        # whole axis on every jitter of the wheel.  Holding the settled unit
        # until it genuinely stops fitting means a threshold is crossed once,
        # in one direction, when the range really has changed.
        settled = self._settled
        if settled is not None:
            held_step, held_decade = settled
            held = float(held_step) * 10.0**held_decade
            span = upper - lower
            if held > 0.0 and self.FLOOR - 1 <= span / held <= budget - 1:
                step, decade = held_step, held_decade
        self._settled = (step, decade)
        # Plain first: an axis whose labels ARE its coordinates is easier to
        # read than one carrying a "+2080" in the corner, so the offset is
        # taken only when the plain labels do not fit the width.
        chosen = None
        for use_offset in (False, True):
            layout = self._lay_out(lower, upper, step, decade, use_offset)
            if len(layout[0]) < self.FLOOR:
                continue
            chosen = layout
            if self._fits(layout[0], layout[1], step, layout[5]):
                break
        if chosen is None:
            chosen = self._lay_out(lower, upper, step, decade, False)

        ticks, indices, offset_int, exponent, scale, label_decade = chosen
        if self.prune_edges and len(ticks) > self.FLOOR:
            quarter = 0.25 * float(step) * 10.0**decade
            keep = [
                position
                for position, tick in enumerate(ticks)
                if lower + quarter <= tick <= upper - quarter
            ]
            if len(keep) >= self.FLOOR:
                ticks = [ticks[position] for position in keep]
                indices = [indices[position] for position in keep]
        self.step = step
        self.m = label_decade
        self.k = scale
        self.C_int = offset_int
        self.C_exp = exponent
        self.C = float(Decimal(offset_int).scaleb(exponent))
        self.ticks = ticks
        self.n_array = indices
        if vmin > vmax:
            self.n_array.reverse()
            self.ticks.reverse()
        return self.ticks

    def __call__(self) -> list[float]:
        if self.axis is None:
            return []
        vmin, vmax = self.axis.get_view_interval()
        return [] if vmax == vmin else self.tick_values(vmin, vmax)


class SmartOffsetFormatter(ticker.Formatter):
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
        super().__init__()
        if axis_type not in {"x", "y"}:
            raise ValueError("axis_type must be 'x' or 'y'")
        self.locator = locator
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
            offset.set_transform(
                axis.axes.transAxes
                if self._offset_coords == "axes"
                else axis.axes.transData
            )
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

    def __call__(self, value: float, pos: int | None = None) -> str:
        del pos
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(numeric) or not self.locator.ticks:
            return ""
        index = int(
            np.argmin([abs(numeric - tick) for tick in self.locator.ticks])
        )
        return self._fmt_scaled_int(
            int(self.locator.n_array[index] * self.locator.step),
            int(self.locator.m),
        )

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
        if self.axis_type == "x" and len(parts) == 2:
            return parts[0] + "\n" + parts[1]
        return "".join(parts)


def apply_smart_ticks(
    axis: Any, which: str = "both", *, prune_edges: bool = False
) -> None:
    """Install the shared tick policy on ``axis``.

    THE one place a tick decision is made.  How many labels an axis carries
    is not a caller's preference: it follows from the width the axis is
    painted at and the width of the labels themselves, both of which this
    module can read.  Callers used to hand in a number instead -- eight for a
    panel, four for an image, three for a facet cell through a locator that
    bypassed this policy entirely -- and none of those numbers had ever been
    compared against the space they were spent in.

    The compact offset locator is only meaningful on a linear coordinate.
    Applying it to a logarithmic count axis treats log-space as linear data,
    which produces labels such as ``200, 400, 600`` at visually logarithmic
    positions.  Log y therefore has its own decade-aware locator/formatter;
    x and all linear axes continue to use the existing compact policy.
    """

    if which not in {"x", "y", "both"}:
        raise ValueError("which must be 'x', 'y', or 'both'")
    if which in ("x", "both"):
        # Reinstalling a locator resets the axis' tick artists, which both
        # reallocates them every frame and leaves them unpositioned until the
        # next full Axis draw.  Install once per configuration.
        signature = ("smart-x", prune_edges)
        if getattr(axis.xaxis, "_zlc_tick_signature", None) != signature:
            locator = SmartOffsetLocator(
                measure=_rendered_width_pt,
                gap_ratio=TICK_LABEL_GAP_RATIO,
                prune_edges=prune_edges,
            )
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(
                SmartOffsetFormatter(
                    locator,
                    axis_type="x",
                    offset_xy=(0.9, -0.1),
                    offset_ha="left",
                    offset_va="top",
                )
            )
            axis.xaxis._zlc_tick_signature = signature
    if which in ("y", "both"):
        if axis.get_yscale() == "log":
            low, high = sorted(map(float, axis.get_ylim()))
            if low <= 0.0 or not np.isfinite((low, high)).all():
                raise ValueError("logarithmic y limits must be finite and positive")
            decades = np.log10(high) - np.log10(low)
            # For a narrow range, 1/2/5 subdivisions are useful major labels;
            # over several decades only decade ticks are labelled and the
            # 2..9 ticks remain visual minor guides.
            if decades <= 1.5:
                major_subs = (1.0, 2.0, 5.0)
            else:
                major_subs = (1.0,)
            signature = ("smart-y-log", major_subs)
            if getattr(axis.yaxis, "_zlc_tick_signature", None) != signature:
                if decades <= 1.5:
                    formatter = ticker.LogFormatter(
                        base=10.0,
                        labelOnlyBase=False,
                    )
                else:
                    formatter = ticker.LogFormatterMathtext(
                        base=10.0,
                        labelOnlyBase=True,
                    )
                axis.yaxis.set_major_locator(
                    ticker.LogLocator(
                        base=10.0,
                        subs=major_subs,
                        numticks=SmartOffsetLocator().max_ticks,
                    )
                )
                axis.yaxis.set_major_formatter(formatter)
                axis.yaxis.set_minor_locator(
                    ticker.LogLocator(
                        base=10.0,
                        subs=tuple(float(value) for value in range(2, 10)),
                        numticks=16,
                    )
                )
                axis.yaxis.set_minor_formatter(ticker.NullFormatter())
                axis.yaxis.get_offset_text().set_visible(False)
                axis.yaxis._zlc_tick_signature = signature
        else:
            signature = ("smart-y-lin", prune_edges)
            if getattr(axis.yaxis, "_zlc_tick_signature", None) != signature:
                locator = SmartOffsetLocator(
                    measure=_rendered_width_pt,
                    gap_ratio=TICK_LABEL_GAP_RATIO,
                    prune_edges=prune_edges,
                )
                axis.yaxis.set_major_locator(locator)
                axis.yaxis.set_major_formatter(
                    SmartOffsetFormatter(
                        locator,
                        axis_type="y",
                        offset_xy=(0.0, 1.005),
                        offset_ha="left",
                        offset_va="bottom",
                    )
                )
                # Do not retain logarithmic minor ticks after switching back
                # to a linear histogram.  The linear style deliberately has no
                # minor labels or extra grid generated by this policy.
                axis.yaxis.set_minor_locator(ticker.NullLocator())
                axis.yaxis.set_minor_formatter(ticker.NullFormatter())
                axis.yaxis.get_offset_text().set_visible(True)
                axis.yaxis._zlc_tick_signature = signature


__all__ = [
    "SmartOffsetFormatter",
    "SmartOffsetLocator",
    "apply_smart_ticks",
]
