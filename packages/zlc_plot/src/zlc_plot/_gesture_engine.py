"""Backend-neutral pointer gesture state.

A gesture holds the axes the session started it on -- a routing fact that
says which surface receives the pointer -- together with immutable axis
transforms and selector values.  Every numeric helper here works on the
transforms and values alone: frontends translate their native pointer
messages into this state through PlotSession, and nothing in this module
reads or draws through an artist or a widget.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, TypeAlias

import numpy as np

from ._axis_transform import AxisTransform
from ._axis_scale import LINEAR, Scale, axis_space, axis_value
from .selectors import (
    CrosshairPoint,
    DragHandle,
    NumericRange,
    RectangleRange,
    Viewport,
    SelectorKind,
    SelectorState,
    _drag_numeric_range,
)


def range_endpoint_hit(
    value: NumericRange,
    coordinate: float,
    tolerance: float,
    scale: Scale = LINEAR,
) -> tuple[float, DragHandle] | None:
    """Resolve a bounded numeric drag against its nearest endpoint.

    ``tolerance`` is measured where the axis is straight (``scale``), so an
    end grabs as far off on screen wherever it sits on the axis.
    """

    position = axis_space(coordinate, scale)
    endpoints = (
        (abs(position - axis_space(value.low, scale)) / tolerance, DragHandle.LOW),
        (abs(position - axis_space(value.high, scale)) / tolerance, DragHandle.HIGH),
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

    # Every handle is placed in PIXELS.  The four corners are the box's
    # own values; the four edge middles are the middles of those corners
    # ON SCREEN, which is the average of their pixels whatever scale the
    # axis has.  Averaging the DATA values named the visual middle only
    # on a linear axis -- between counts 0.8 and 1200 the arithmetic
    # mean sits at 90.5 per cent of the box height, so a side handle was
    # nowhere near the side it belongs to.  This keeps the engine free
    # of Matplotlib exactly as the docstring above requires:
    # ``pixel_point`` is still the only injected operation.
    bottom_left = pixel_point((x0, y0))
    bottom_right = pixel_point((x1, y0))
    top_right = pixel_point((x1, y1))
    top_left = pixel_point((x0, y1))

    def middle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return (first + second) / 2.0

    handles = (
        (bottom_left, DragHandle.BOTTOM_LEFT),
        (middle(bottom_left, bottom_right), DragHandle.BOTTOM),
        (bottom_right, DragHandle.BOTTOM_RIGHT),
        (middle(bottom_right, top_right), DragHandle.RIGHT),
        (top_right, DragHandle.TOP_RIGHT),
        (middle(top_left, top_right), DragHandle.TOP),
        (top_left, DragHandle.TOP_LEFT),
        (middle(bottom_left, top_left), DragHandle.LEFT),
    )
    handle_hit = min(
        (
            (float(np.linalg.norm(point - mouse)) / handle_radius, handle)
            for point, handle in handles
        ),
        key=lambda item: item[0],
    )
    if handle_hit[0] <= 1.0:
        return handle_hit

    corners = np.asarray(
        (bottom_left, bottom_right, top_right, top_left),
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
    x_scale: Scale = LINEAR,
    y_scale: Scale = LINEAR,
) -> RectangleRange | None:
    """Return the translated viewport for one pointer position."""

    dx = axis_space(origin.x, x_scale) - axis_space(point.x, x_scale)
    dy = axis_space(origin.y, y_scale) - axis_space(point.y, y_scale)
    if np.isclose(dx, 0.0, rtol=0.0, atol=1.0e-15) and (
        not image_like or np.isclose(dy, 0.0, rtol=0.0, atol=1.0e-15)
    ):
        return None
    return RectangleRange(
        NumericRange(*(axis_value(axis_space(value, x_scale) + dx, x_scale)
                       for value in (x.low, x.high))),
        NumericRange(*(axis_value(axis_space(value, y_scale) + dy, y_scale)
                       for value in (y.low, y.high))) if image_like else y,
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
    """What every pointer gesture holds.

    No pacing here: a move renders as it arrives, and the host's pointer
    coalescing is the flow control -- a burst that outruns the frames
    reaches the session as its latest position.
    """

    axes: Any
    transform: AxisTransform


@dataclass(slots=True)
class _SelectorGesture(_PointerGestureBase):
    kind: SelectorKind
    state: SelectorState
    handle: DragHandle
    origin: CrosshairPoint
    origin_px: tuple[float, float]
    #: Where the hand was last read, the press to begin with.  A coordinate
    #: it cannot read (below 0 mW on a dBm value) is taken from here.
    reading: CrosshairPoint
    started: bool = False
    #: A move has left the press pixel.  Not ``started``: a new box's hand
    #: over what nothing can be read from moves nothing, and is no click.
    moved: bool = False


@dataclass(slots=True)
class _ColorGesture(_PointerGestureBase):
    drag: _ColorLimitDrag


@dataclass(slots=True)
class _PanGesture(_PointerGestureBase):
    origin: CrosshairPoint
    x: NumericRange
    y: NumericRange
    candidate: Viewport | None = None


@dataclass(slots=True)
class _OrbitGesture(_PointerGestureBase):
    """A height-bar camera drag: pixels in, orbit angles out."""

    origin_px: tuple[float, float]
    start: Any
    current: Any


@dataclass(slots=True)
class _PickGesture(_PointerGestureBase):
    """A left press on the 3D scene: a click picks a bar, a drag is inert."""

    origin_px: tuple[float, float]


_PointerGesture: TypeAlias = (
    _SelectorGesture | _ColorGesture | _PanGesture | _OrbitGesture
    | _PickGesture
)


__all__ = [
    "area_drag_handle",
    "_ColorGesture",
    "_ColorLimitDrag",
    "_OrbitGesture",
    "_PickGesture",
    "_PanGesture",
    "_PointerGesture",
    "_PointerGestureBase",
    "_SelectorGesture",
    "pan_rectangle",
    "range_endpoint_hit",
]
