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

import numpy as np
from zlc_data.units import Unit

#: The scales this product draws.  Matplotlib has more; an unknown one must
#: not be quietly treated as one of these.
LINEAR = "linear"
LOG = "log"
Scale = str | tuple[float, ...] | tuple[str | tuple[float, ...], Unit, Unit]


def _piecewise(value, coordinates, positions):
    """Interpolate and linearly extend the two endpoint segments."""
    values = np.asarray(value, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if coordinates.size == 1:
        result = values - coordinates[0] + positions[0]
    else:
        if coordinates[-1] < coordinates[0]:
            coordinates, positions = coordinates[::-1], positions[::-1]
        index = np.clip(np.searchsorted(coordinates, values, side="right") - 1,
                        0, coordinates.size - 2)
        fraction = (values - coordinates[index]) / (coordinates[index + 1] - coordinates[index])
        result = positions[index] + fraction * (positions[index + 1] - positions[index])
    return float(result) if result.ndim == 0 else result


def axis_space(value, scale: Scale):
    """The coordinate in which this axis is LINEAR on screen.

    Without it, a press at the vertical middle of a count axis limited to
    (0.8, 1200) read 600.4 where the truth is 30.98 -- and Matplotlib, which
    draws selectors in data coordinates and so IS log-aware, then faithfully
    drew the corner at 9.5 per cent from the top.  The box did not follow the
    pointer because the number under the pointer was wrong, not because the
    drawing was.
    """

    if isinstance(scale, tuple):
        if len(scale) == 3 and isinstance(scale[1], Unit):
            mapping, canonical, display = scale
            return axis_space(canonical.convert_value_to(value, display), mapping)
        positions = np.arange(len(scale))
        if scale[-1] < scale[0]:
            positions = positions[::-1]
        return _piecewise(value, scale, positions)
    if np.ndim(value):
        if scale != LOG:
            return value
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.asarray(value) > 0.0, np.log10(value), -np.inf)
    number = float(value)
    if scale != LOG:
        return number
    # A log axis cannot show a non-positive value, and its limits are refused
    # if they are not positive, so this only guards a value carried over from
    # before the scale changed.
    return math.log10(number) if number > 0.0 else -math.inf


def axis_value(position, scale: Scale):
    """Undo :func:`axis_space`."""

    if isinstance(scale, tuple):
        if len(scale) == 3 and isinstance(scale[1], Unit):
            mapping, canonical, display = scale
            result = display.convert_value_to(axis_value(position, mapping), canonical)
            return float(result) if result.ndim == 0 else result
        positions = np.arange(len(scale))
        if scale[-1] < scale[0]:
            positions = positions[::-1]
        return _piecewise(position, positions, scale)
    if np.ndim(position):
        return position if scale != LOG else np.power(10.0, position)
    number = float(position)
    return number if scale != LOG else float(10.0**number)


def interpolate(low: float, high: float, fraction: float, scale: Scale) -> float:
    """The value ``fraction`` of the way from ``low`` to ``high`` ON SCREEN."""

    start = axis_space(low, scale)
    stop = axis_space(high, scale)
    return axis_value(start + fraction * (stop - start), scale)


def fraction_of(value: float, low: float, high: float, scale: Scale) -> float:
    """Where ``value`` sits between the limits, as a fraction of the box."""

    start = axis_space(low, scale)
    stop = axis_space(high, scale)
    if stop == start:
        return 0.0
    return (axis_space(value, scale) - start) / (stop - start)


def midpoint(low: float, high: float, scale: Scale) -> float:
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
    "Scale",
    "axis_space",
    "axis_value",
    "fraction_of",
    "interpolate",
    "midpoint",
]
