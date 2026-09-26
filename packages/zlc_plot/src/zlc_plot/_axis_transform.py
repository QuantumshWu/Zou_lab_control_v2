"""One immutable axes transform shared by native and raster interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

from ._axis_scale import (
    LINEAR,
    LOG,
    Scale,
    axis_space,
    axis_value,
    fraction_of as _fraction,
    interpolate as _interpolate,
)
from .selectors import CrosshairPoint


def canvas_physical_size(canvas: Any) -> tuple[float, float]:
    """Return the physical-pixel extent used by Matplotlib pointer events."""

    get_width_height = getattr(canvas, "get_width_height", None)
    if not callable(get_width_height):
        raise TypeError("interactive canvas has no pixel-size query")
    try:
        size = get_width_height(physical=True)
    except TypeError:
        width, height = get_width_height()
        ratio = float(getattr(canvas, "device_pixel_ratio", 1.0))
        size = (float(width) * ratio, float(height) * ratio)
    try:
        width, height = map(float, size)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "interactive canvas pixel size must contain two numbers"
        ) from error
    if not math.isfinite(width) or not math.isfinite(height):
        raise ValueError("interactive canvas physical size must be positive and finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("interactive canvas physical size must be positive and finite")
    return width, height


def _read(
    fraction: float,
    canonical: tuple[float, float],
    display: tuple[float, float],
    scale: Scale,
    canonical_scale: Scale | None,
) -> float:
    """The canonical value ``fraction`` of the way along one drawn axis."""

    if canonical_scale is None:
        return _interpolate(*canonical, fraction, scale)
    # Drawn straight in its display unit, so interpolated between the
    # limits it is DRAWN with and only the value under the pointer
    # converted.  Between the converted limits instead, an axis panned
    # below 0 mW on a dBm value had no lower limit to start from, and
    # every reading on it was NaN.
    start, stop = (axis_space(value, scale) for value in display)
    return axis_value(start + fraction * (stop - start), canonical_scale)


@dataclass(frozen=True, slots=True)
class AxisTransform:
    """Exact top-origin plot box plus display and canonical coordinates."""

    role: str
    cell_index: int | None
    bounds: tuple[float, float, float, float]
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    canonical_x_limits: tuple[float, float]
    canonical_y_limits: tuple[float, float]
    #: How each axis maps value to position.  It is a field because it is a
    #: fact about the drawn axis that nothing downstream can recover: the
    #: limits alone say where the ends are, never how the space between them
    #: is divided.  Defaulted so a caller that does not know cannot silently
    #: claim an axis is logarithmic.
    x_scale: Scale = LINEAR
    y_scale: Scale = LINEAR
    canonical_x_scale: Scale | None = None
    canonical_y_scale: Scale | None = None

    def _axis(
        self, name: str
    ) -> tuple[tuple[float, float], tuple[float, float], Scale, Scale | None]:
        """One axis' drawn limits, canonical limits, scale and unit mapping."""

        if name == "x":
            return self.x_limits, self.canonical_x_limits, self.x_scale, self.canonical_x_scale
        return self.y_limits, self.canonical_y_limits, self.y_scale, self.canonical_y_scale

    def interaction_scale(self, name: str) -> Scale:
        """How a canonical value on one axis is placed where it is straight.

        A display-unit axis converts it into the unit it is drawn in first
        (``canonical_*_scale``); any other places it as drawn.  Hitting,
        moving and letting go of a selector all measure in this one space.
        """

        _display, _canonical, scale, canonical_scale = self._axis(name)
        return scale if canonical_scale is None else canonical_scale

    def drawn_span(self, name: str) -> float:
        """One drawn axis' length in :meth:`interaction_scale`'s space.

        Between the limits :func:`_read` interpolates between: a display-unit
        axis' drawn ones, which are numbers where its canonical ones may not
        be (0 mW is -inf dBm); any other's canonical ones, which a linear
        unit change (MHz shown in kHz) scales.
        """

        display, canonical, scale, canonical_scale = self._axis(name)
        start, stop = (
            axis_space(value, scale)
            for value in (canonical if canonical_scale is None else display)
        )
        return abs(stop - start)

    def display_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        """Map display-space axes data into top-origin widget coordinates."""

        left, top, right, bottom = self.bounds
        x0, x1 = self.x_limits
        y0, y1 = self.y_limits
        nx = left + _fraction(x, x0, x1, self.x_scale) * (right - left)
        ny = top + _fraction(y, y1, y0, self.y_scale) * (bottom - top)
        return nx, ny

    def canonical_from_normalized(self, nx: float, ny: float) -> CrosshairPoint:
        """Map top-origin normalized widget coordinates into canonical data."""

        return CrosshairPoint(*self._canonical_values(nx, ny))

    def _canonical_values(self, nx: float, ny: float) -> tuple[float, float]:
        left, top, right, bottom = self.bounds
        tx = (float(nx) - left) / (right - left)
        ty = (float(ny) - top) / (bottom - top)
        x0, x1 = self.canonical_x_limits
        y0, y1 = self.canonical_y_limits
        if self.role == "distribution":
            # The rail is the value axis stood on its side: its vertical
            # extent is the X pair, so it is the X scale that divides it.
            x_scale = self.x_scale if self.canonical_x_scale is None else self.canonical_x_scale
            return _interpolate(x1, x0, ty, x_scale), 0.0
        return (
            _read(tx, (x0, x1), self.x_limits, self.x_scale, self.canonical_x_scale),
            _read(ty, (y1, y0), self.y_limits[::-1], self.y_scale, self.canonical_y_scale),
        )

    def display_from_normalized(self, nx: float, ny: float) -> CrosshairPoint:
        """Map top-origin normalized widget coordinates into display data."""

        left, top, right, bottom = self.bounds
        tx = (float(nx) - left) / (right - left)
        ty = (float(ny) - top) / (bottom - top)
        x0, x1 = self.x_limits
        y0, y1 = self.y_limits
        if self.role == "distribution":
            return CrosshairPoint(_interpolate(y1, y0, ty, self.y_scale), 0.0)
        return CrosshairPoint(
            _interpolate(x0, x1, tx, self.x_scale),
            _interpolate(y1, y0, ty, self.y_scale),
        )

    @staticmethod
    def _event_normalized(event: Any, canvas: Any) -> tuple[float, float] | None:
        pixel_x = getattr(event, "x", None)
        pixel_y = getattr(event, "y", None)
        if pixel_x is None or pixel_y is None:
            return None
        width, height = canvas_physical_size(canvas)
        return float(pixel_x) / width, 1.0 - float(pixel_y) / height

    def canonical_reading(self, event: Any, canvas: Any) -> tuple[float, float] | None:
        """Map one Matplotlib event into canonical data coordinates.

        None for an event without a position.  Over the part of an axis its
        canonical unit cannot express -- 0 mW and below, on a dBm value
        shown in mW -- that coordinate reads NaN or infinite, and what that
        means is the gesture's to say: a drag keeps the coordinate it last
        read there.
        """

        normalized = self._event_normalized(event, canvas)
        if normalized is None:
            return None
        if self.role == "distribution":
            # The colour rail reads as it always has: what its limits allow
            # is the colour-limit owner's to decide.
            point = self.canonical_from_normalized(*normalized)
            return point.x, point.y
        return self._canonical_values(*normalized)

    def display_point(self, event: Any, canvas: Any) -> CrosshairPoint | None:
        """Map one Matplotlib event into display-space data coordinates."""

        normalized = self._event_normalized(event, canvas)
        return None if normalized is None else self.display_from_normalized(*normalized)

    def display_to_pixel(
        self,
        x: float,
        y: float,
        canvas: Any,
    ) -> tuple[float, float]:
        """Map display-space data into bottom-origin physical canvas pixels."""

        width, height = canvas_physical_size(canvas)
        nx, ny = self.display_to_normalized(x, y)
        return nx * width, (1.0 - ny) * height


__all__ = [
    "LINEAR",
    "LOG",
    "AxisTransform",
    "axis_space",
    "axis_value",
    "canvas_physical_size",
]
