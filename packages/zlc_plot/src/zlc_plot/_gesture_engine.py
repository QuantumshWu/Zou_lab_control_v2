"""Backend-neutral pointer gesture state.

The engine stores only immutable axis transforms and selector values.  Frontends
translate their native pointer messages into this state through PlotSession;
Matplotlib artists and widget objects never enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Any, Callable, TypeAlias

import numpy as np

from ._axis_transform import AxisTransform
from .selectors import (
    CrosshairPoint,
    DragHandle,
    NumericRange,
    RectangleRange,
    SelectorKind,
    _drag_numeric_range,
)


def range_endpoint_hit(
    value: NumericRange,
    coordinate: float,
    tolerance: float,
) -> tuple[float, DragHandle] | None:
    """Resolve a bounded numeric drag against its nearest endpoint."""

    endpoints = (
        (abs(coordinate - value.low) / tolerance, DragHandle.LOW),
        (abs(coordinate - value.high) / tolerance, DragHandle.HIGH),
    )
    score, handle = min(endpoints, key=lambda item: item[0])
    return (score, handle) if score <= 1.0 else None


def area_drag_handle(
    value: RectangleRange,
    mouse: np.ndarray,
    pixel_point: Callable[[tuple[float, float]], np.ndarray],
    *,
    handle_radius: float,
) -> tuple[float, DragHandle] | None:
    """Choose the area body/handle under one pixel-space pointer.

    ``pixel_point`` is the only frontend-supplied operation.  The geometry
    engine itself knows no Matplotlib axes, canvas, Qt object, or notebook
    widget and can therefore be exercised with a tiny affine test transform.
    """

    x0, x1 = value.x.low, value.x.high
    y0, y1 = value.y.low, value.y.high
    xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def distance(coordinate: tuple[float, float]) -> float:
        return float(np.linalg.norm(pixel_point(coordinate) - mouse))

    handles = (
        ((x0, y0), DragHandle.BOTTOM_LEFT),
        ((xm, y0), DragHandle.BOTTOM),
        ((x1, y0), DragHandle.BOTTOM_RIGHT),
        ((x1, ym), DragHandle.RIGHT),
        ((x1, y1), DragHandle.TOP_RIGHT),
        ((xm, y1), DragHandle.TOP),
        ((x0, y1), DragHandle.TOP_LEFT),
        ((x0, ym), DragHandle.LEFT),
    )
    handle_hit = min(
        (
            (distance(coordinate) / handle_radius, handle)
            for coordinate, handle in handles
        ),
        key=lambda item: item[0],
    )
    if handle_hit[0] <= 1.0:
        return handle_hit

    corners = np.asarray(
        tuple(pixel_point(coordinate) for coordinate, _handle in handles[::2]),
        dtype=float,
    )
    low = np.min(corners, axis=0)
    high = np.max(corners, axis=0)
    if bool(np.all(low <= mouse) and np.all(mouse <= high)):
        return 1.0, DragHandle.BODY
    return None


def pan_rectangle(
    origin: CrosshairPoint,
    point: CrosshairPoint,
    x: NumericRange,
    y: NumericRange,
    *,
    image_like: bool,
) -> RectangleRange | None:
    """Return the translated viewport for one pointer position."""

    dx = origin.x - point.x
    dy = origin.y - point.y
    if np.isclose(dx, 0.0, rtol=0.0, atol=1.0e-15) and (
        not image_like or np.isclose(dy, 0.0, rtol=0.0, atol=1.0e-15)
    ):
        return None
    return RectangleRange(
        x.shifted(dx),
        y.shifted(dy) if image_like else y,
    )


@dataclass(frozen=True, slots=True)
class _ColorLimitDrag:
    """One canonical color-scale gesture, separate from data selectors."""

    original: NumericRange
    candidate: NumericRange
    handle: DragHandle
    origin: float
    bounds: NumericRange
    minimum_span: float

    @property
    def changed(self) -> bool:
        return not np.allclose(
            (self.candidate.low, self.candidate.high),
            (self.original.low, self.original.high),
            rtol=1.0e-12,
            atol=1.0e-15,
        )

    def moved(self, position: float) -> "_ColorLimitDrag":
        return replace(
            self,
            candidate=_drag_numeric_range(
                self.original,
                handle=self.handle,
                origin=self.origin,
                position=position,
                minimum_span=self.minimum_span,
                bounds=self.bounds,
            ),
        )


@dataclass(slots=True)
class _PointerGestureBase:
    axes: Any
    transform: AxisTransform
    _cadence_at: dict[str, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def lane_due(self, lane: str, interval_ms: int) -> bool:
        now = monotonic()
        previous = self._cadence_at.get(lane)
        if previous is not None and now - previous < float(interval_ms) / 1000.0:
            return False
        self._cadence_at[lane] = now
        return True


@dataclass(slots=True)
class _SelectorGesture(_PointerGestureBase):
    kind: SelectorKind


@dataclass(slots=True)
class _ColorGesture(_PointerGestureBase):
    drag: _ColorLimitDrag


@dataclass(slots=True)
class _PanGesture(_PointerGestureBase):
    origin: CrosshairPoint
    x: NumericRange
    y: NumericRange
    candidate: RectangleRange | None = None


_PointerGesture: TypeAlias = _SelectorGesture | _ColorGesture | _PanGesture


__all__ = [
    "area_drag_handle",
    "_ColorGesture",
    "_ColorLimitDrag",
    "_PanGesture",
    "_PointerGesture",
    "_PointerGestureBase",
    "_SelectorGesture",
    "pan_rectangle",
    "range_endpoint_hit",
]
