"""Backend-neutral display-space values for presenting fitted models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import readonly_copy
from .fit import FitParameterDisplay


@dataclass(frozen=True, slots=True)
class FitPolyline:
    x: np.ndarray
    y: np.ndarray
    role: str = "primary"
    component_index: int = 0

    def __post_init__(self) -> None:
        x = readonly_copy(self.x, dtype=float).reshape(-1)
        y = readonly_copy(self.y, dtype=float).reshape(-1)
        if x.shape != y.shape or not x.size:
            raise ValueError("fit polyline x/y must be non-empty equal-length arrays")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


@dataclass(frozen=True, slots=True)
class FitEllipseGlyph:
    """One indivisible center-and-two-radii glyph in display coordinates."""

    center_x: float
    center_y: float
    radius_x: float
    radius_y: float


@dataclass(frozen=True, slots=True)
class FitOverlay:
    """Complete display-space fit scene."""

    polylines: tuple[FitPolyline, ...] = ()
    ellipse_glyph: FitEllipseGlyph | None = None
    success: bool = True
    formula: str = ""
    parameter_display: tuple[FitParameterDisplay, ...] = ()
    diagnostic: str = ""
    facet_index: int | None = None
    headline_parameter: FitParameterDisplay | None = None


__all__ = ["FitOverlay"]
