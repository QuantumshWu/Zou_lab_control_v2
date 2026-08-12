from __future__ import annotations

import random

import numpy as np
import pytest

from zlc_plot._gesture_engine import area_drag_handle, pan_rectangle, range_endpoint_hit
from zlc_plot.selectors import (
    CrosshairPoint,
    DragHandle,
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorState,
    _SelectorController,
    _clamp_range,
    _drag_numeric_range,
)


def test_drag_numeric_range_and_clamp_never_escape_bounds() -> None:
    bounds = NumericRange(0.0, 10.0)
    original = NumericRange(2.0, 5.0)
    rng = random.Random(9)
    for handle in (DragHandle.NEW, DragHandle.BODY, DragHandle.LOW, DragHandle.HIGH):
        for _ in range(200):
            position = rng.uniform(-50.0, 50.0)
            if handle is DragHandle.NEW:
                origin = rng.uniform(0.0, 10.0)
            else:
                origin = 3.0
            result = _drag_numeric_range(
                original,
                handle=handle,
                origin=origin,
                position=position,
                minimum_span=0.5,
                bounds=bounds,
            )
            assert bounds.low <= result.low <= result.high <= bounds.high
            assert result.span >= 0.5 - 1e-12
    assert _clamp_range(NumericRange(-10.0, 20.0), bounds) == bounds


def test_new_area_drag_is_sorted_and_controller_revisions_are_monotonic() -> None:
    controller = _SelectorController()
    initial = SelectorState(
        SelectorKind.AREA,
        RectangleRange(NumericRange(2.0, 8.0), NumericRange(2.0, 8.0)),
    )
    _, installed = controller.install(initial)
    controller.pointer_down(
        SelectorKind.AREA,
        8.0,
        7.0,
        handle=DragHandle.BODY,
    )
    first = controller.pointer_move(7.0, 6.0, x_bounds=NumericRange(0, 10), y_bounds=NumericRange(0, 10))
    second = controller.pointer_move(6.0, 5.0, x_bounds=NumericRange(0, 10), y_bounds=NumericRange(0, 10))
    assert first is not None and second is not None
    assert second.revision > first.revision
    assert second.value.x.low >= 0 and second.value.y.low >= 0
    finished = controller.finish_gesture(1.0, 1.0, x_bounds=NumericRange(0, 10), y_bounds=NumericRange(0, 10))
    assert finished is not None
    assert controller.active_gesture is None
    _, committed = controller.install(finished)
    assert committed.revision > installed.revision

    draft = SelectorState(SelectorKind.X_RANGE, NumericRange(7.0, 2.0))
    controller.pointer_down(SelectorKind.X_RANGE, 7.0, 0.0, handle=DragHandle.NEW, draft=draft)
    candidate = controller.pointer_move(1.0, 0.0, x_bounds=NumericRange(0, 10))
    assert candidate is not None
    assert candidate.value.low <= candidate.value.high


def test_controller_cancel_restores_committed_and_allows_one_kind_each() -> None:
    controller = _SelectorController()
    area = SelectorState(
        SelectorKind.AREA,
        RectangleRange(NumericRange(1, 2), NumericRange(3, 4)),
    )
    crosshair = SelectorState(SelectorKind.CROSSHAIR, CrosshairPoint(5, 6))
    controller.install(area)
    controller.install(crosshair)
    assert len(controller.states()) == 2
    controller.pointer_down(SelectorKind.AREA, 1.5, 3.5, handle=DragHandle.BODY)
    moved = controller.pointer_move(4, 5, x_bounds=NumericRange(0, 10), y_bounds=NumericRange(0, 10))
    assert moved is not None
    cancelled = controller.cancel()
    assert cancelled is not None
    assert controller.state(SelectorKind.AREA).value == area.value
    assert controller.candidate_state() is None
    with pytest.raises(KeyError):
        controller.state(SelectorKind.THRESHOLD)


def test_backend_neutral_gesture_geometry_has_one_authority() -> None:
    value = NumericRange(2.0, 8.0)
    hit_endpoint = range_endpoint_hit(value, 2.1, 0.2)
    assert hit_endpoint is not None
    score, handle = hit_endpoint
    assert score == pytest.approx(0.5)
    assert handle is DragHandle.LOW
    rectangle = RectangleRange(value, NumericRange(3.0, 7.0))

    def pixel(point: tuple[float, float]):
        return np.asarray(point, dtype=float)

    hit = area_drag_handle(
        rectangle,
        np.asarray((2.0, 3.0)),
        pixel,
        handle_radius=1.0,
    )
    assert hit == (0.0, DragHandle.BOTTOM_LEFT)
    assert area_drag_handle(
        rectangle,
        np.asarray((4.0, 4.0)),
        pixel,
        handle_radius=1.0,
    ) == (1.0, DragHandle.BODY)
    assert pan_rectangle(
        CrosshairPoint(5.0, 5.0),
        CrosshairPoint(4.0, 6.0),
        NumericRange(0.0, 10.0),
        NumericRange(0.0, 10.0),
        image_like=True,
    ) == RectangleRange(NumericRange(1.0, 11.0), NumericRange(-1.0, 9.0))
