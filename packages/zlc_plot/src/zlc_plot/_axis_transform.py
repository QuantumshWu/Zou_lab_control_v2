"""One immutable axes transform shared by native and raster interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

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

    def display_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        """Map display-space axes data into top-origin widget coordinates."""

        left, top, right, bottom = self.bounds
        x0, x1 = self.x_limits
        y0, y1 = self.y_limits
        nx = left + (float(x) - x0) / (x1 - x0) * (right - left)
        ny = top + (float(y) - y1) / (y0 - y1) * (bottom - top)
        return nx, ny

    def canonical_from_normalized(self, nx: float, ny: float) -> CrosshairPoint:
        """Map top-origin normalized widget coordinates into canonical data."""

        left, top, right, bottom = self.bounds
        tx = (float(nx) - left) / (right - left)
        ty = (float(ny) - top) / (bottom - top)
        x0, x1 = self.canonical_x_limits
        y0, y1 = self.canonical_y_limits
        if self.role == "distribution":
            return CrosshairPoint(x1 + ty * (x0 - x1), 0.0)
        return CrosshairPoint(
            x0 + tx * (x1 - x0),
            y1 + ty * (y0 - y1),
        )

    def display_from_normalized(self, nx: float, ny: float) -> CrosshairPoint:
        """Map top-origin normalized widget coordinates into display data."""

        left, top, right, bottom = self.bounds
        tx = (float(nx) - left) / (right - left)
        ty = (float(ny) - top) / (bottom - top)
        x0, x1 = self.x_limits
        y0, y1 = self.y_limits
        if self.role == "distribution":
            return CrosshairPoint(y1 + ty * (y0 - y1), 0.0)
        return CrosshairPoint(
            x0 + tx * (x1 - x0),
            y1 + ty * (y0 - y1),
        )

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


__all__ = ["AxisTransform", "canvas_physical_size"]
