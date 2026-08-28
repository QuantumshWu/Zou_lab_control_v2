"""One immutable axes transform shared by native and raster interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

from ._axis_scale import (
    LINEAR,
    LOG,
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
    x_scale: str = LINEAR
    y_scale: str = LINEAR

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

        left, top, right, bottom = self.bounds
        tx = (float(nx) - left) / (right - left)
        ty = (float(ny) - top) / (bottom - top)
        x0, x1 = self.canonical_x_limits
        y0, y1 = self.canonical_y_limits
        if self.role == "distribution":
            # The rail is the value axis stood on its side: its vertical
            # extent is the X pair, so it is the X scale that divides it.
            return CrosshairPoint(_interpolate(x1, x0, ty, self.x_scale), 0.0)
        return CrosshairPoint(
            _interpolate(x0, x1, tx, self.x_scale),
            _interpolate(y1, y0, ty, self.y_scale),
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

    def screen_span(self, low: float, high: float, *, axis: str) -> float:
        """How far apart two values are ON SCREEN, in axis space."""

        scale = self.x_scale if axis == "x" else self.y_scale
        return abs(axis_space(high, scale) - axis_space(low, scale))

    @staticmethod
    def _event_normalized(event: Any, canvas: Any) -> tuple[float, float] | None:
        pixel_x = getattr(event, "x", None)
        pixel_y = getattr(event, "y", None)
        if pixel_x is None or pixel_y is None:
            return None
        width, height = canvas_physical_size(canvas)
        return float(pixel_x) / width, 1.0 - float(pixel_y) / height

    def canonical_point(self, event: Any, canvas: Any) -> CrosshairPoint | None:
        """Map one Matplotlib event into canonical data coordinates."""

        normalized = self._event_normalized(event, canvas)
        return (
            None
            if normalized is None
            else self.canonical_from_normalized(*normalized)
        )

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
