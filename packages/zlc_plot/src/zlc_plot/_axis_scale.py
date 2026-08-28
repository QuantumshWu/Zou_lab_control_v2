"""How an axis divides the space between its ends.  One owner, no cycle.

Every mapping between a position and a value is an interpolation between the
axis' limits, and an interpolation is only a straight line in the space the
axis actually draws.  On a log axis that space is the decimal exponent.

This lives on its own because both sides of the pointer path need it and they
already point at each other: :mod:`_axis_transform` imports ``CrosshairPoint``
from :mod:`selectors`, so the helpers cannot live in either without a cycle --
and a second copy in the other is exactly how the two would come to disagree.
"""

from __future__ import annotations

import math

#: The scales this product draws.  Matplotlib has more; an unknown one must
#: not be quietly treated as one of these.
LINEAR = "linear"
LOG = "log"


def axis_space(value: float, scale: str) -> float:
    """The coordinate in which this axis is LINEAR on screen.

    Without it, a press at the vertical middle of a count axis limited to
    (0.8, 1200) read 600.4 where the truth is 30.98 -- and Matplotlib, which
    draws selectors in data coordinates and so IS log-aware, then faithfully
    drew the corner at 9.5 per cent from the top.  The box did not follow the
    pointer because the number under the pointer was wrong, not because the
    drawing was.
    """

    number = float(value)
    if scale != LOG:
        return number
    # A log axis cannot show a non-positive value, and its limits are refused
    # if they are not positive, so this only guards a value carried over from
    # before the scale changed.
    return math.log10(number) if number > 0.0 else -math.inf


def axis_value(position: float, scale: str) -> float:
    """Undo :func:`axis_space`."""

    number = float(position)
    return number if scale != LOG else float(10.0**number)


def interpolate(low: float, high: float, fraction: float, scale: str) -> float:
    """The value ``fraction`` of the way from ``low`` to ``high`` ON SCREEN."""

    start = axis_space(low, scale)
    stop = axis_space(high, scale)
    return axis_value(start + fraction * (stop - start), scale)


def fraction_of(value: float, low: float, high: float, scale: str) -> float:
    """Where ``value`` sits between the limits, as a fraction of the box."""

    start = axis_space(low, scale)
    stop = axis_space(high, scale)
    if stop == start:
        return 0.0
    return (axis_space(value, scale) - start) / (stop - start)


def midpoint(low: float, high: float, scale: str) -> float:
    """The value halfway between two others ON SCREEN.

    ``(low + high) / 2`` is the middle of the box only on a linear axis.  The
    selector scene puts its grab handles at edge midpoints, and on a count
    axis limited to (0.8, 1200) the arithmetic mean renders at 90.5 per cent
    of the box height -- a handle nowhere near the edge it belongs to.
    """

    return interpolate(low, high, 0.5, scale)


__all__ = [
    "LINEAR",
    "LOG",
    "axis_space",
    "axis_value",
    "fraction_of",
    "interpolate",
    "midpoint",
]
