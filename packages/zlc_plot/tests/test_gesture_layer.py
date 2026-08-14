"""A pointer move is not a data frame, and must not be composed like one.

A move used to re-enter the whole compose -- every dynamic artist of every
axes, the image included -- at the same priority as a camera frame, so
interaction cost scaled with the scene and interaction latency was additive
with whatever frame the worker happened to be composing.  What these fix on
is that the cheap path must paint EXACTLY what the expensive one paints; a
faster wrong picture is not a faster picture.
"""

from __future__ import annotations

import time
from threading import Event

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, HistogramPlot, ImagePlot, SelectorKind
from zlc_plot.raster import RasterPlotHost
from zlc_plot.selectors import NumericRange, SelectorState
from zlc_plot.session import PlotSession

SIZE = 256


def _image_snapshot(revision: int = 0) -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"shot": [0.0]}),
        data_axes=(
            Axis.create(
                "sy", values=tuple(float(v) for v in range(SIZE)), role=SPATIAL_Y
            ),
            Axis.create(
                "sx", values=tuple(float(v) for v in range(SIZE)), role=SPATIAL_X
            ),
        ),
        dtype=np.uint16,
    )
    values = (
        np.random.default_rng(revision)
        .integers(180, 900, (1, 1, SIZE, SIZE))
        .astype(np.uint16)
    )
    return DatasetSnapshot(schema, values, revision=revision)


def test_a_move_paints_the_same_pixels_as_a_compose() -> None:
    """The whole licence for the cheap path."""

    session = PlotSession(
        _image_snapshot(), ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")), size="4x4"
    )
    try:
        renderer = session._renderer
        session.rgba()
        renderer.begin_selector_gesture(SelectorKind.X_RANGE)
        state = SelectorState(SelectorKind.X_RANGE, NumericRange(60.0, 190.0))
        renderer.preview_selector(state)
        assert renderer._gesture_region is not None, "the gesture was never captured"
        cheap = np.array(session.rgba(), copy=True)

        renderer._forget_gesture_region()
        renderer.preview_selector(state)
        expensive = np.array(session.rgba(), copy=True)
        assert np.array_equal(cheap, expensive)
    finally:
        session.close()


def test_a_move_costs_a_fraction_of_a_compose() -> None:
    """Not a wall-clock promise: the ratio is the architecture being right."""

    session = PlotSession(
        _image_snapshot(), ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")), size="4x4"
    )
    try:
        renderer = session._renderer
        session.rgba()

        def timed(call, count: int) -> float:
            call(0)
            start = time.perf_counter()
            for index in range(count):
                call(index)
            return (time.perf_counter() - start) / count

        compose = timed(lambda _index: renderer._compose_frame(chrome_stable=True), 8)
        renderer.begin_selector_gesture(SelectorKind.X_RANGE)
        moves = [
            SelectorState(SelectorKind.X_RANGE, NumericRange(60.0 + index, 190.0 + index))
            for index in range(16)
        ]
        move = timed(lambda index: renderer.preview_selector(moves[index % 16]), 16)
        assert move < compose / 2.0, (move, compose)
    finally:
        session.close()


def test_the_gesture_capture_dies_with_anything_that_could_stale_it() -> None:
    session = PlotSession(
        _image_snapshot(), ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")), size="4x4"
    )
    try:
        renderer = session._renderer
        session.rgba()
        renderer.begin_selector_gesture(SelectorKind.X_RANGE)
        renderer.preview_selector(
            SelectorState(SelectorKind.X_RANGE, NumericRange(60.0, 190.0))
        )
        assert renderer._gesture_region is not None
        renderer.draw()
        assert renderer._gesture_region is None

        renderer.begin_selector_gesture(SelectorKind.X_RANGE)
        renderer.preview_selector(
            SelectorState(SelectorKind.X_RANGE, NumericRange(70.0, 200.0))
        )
        assert renderer._gesture_region is not None
        renderer.end_selector_gesture()
        assert renderer._gesture_region is None
        assert renderer._paint_gesture_overlay() is False
    finally:
        session.close()


def _histogram_host() -> RasterPlotHost:
    rng = np.random.default_rng(3)
    samples = np.concatenate(
        [rng.normal(100.0, 9.0, 400), rng.normal(180.0, 13.0, 400)]
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=samples.size),
        PointTable.from_columns({"site": [0.0]}),
        dtype=np.float64,
    )
    return RasterPlotHost.from_plot(
        DatasetSnapshot(schema, samples.reshape(-1, 1), revision=0),
        HistogramPlot(),
        size="4x4",
    )


def test_the_threshold_follows_the_pointer_and_commits_on_release() -> None:
    """Every selector tracks the hand; release only settles what it tracked."""

    host = _histogram_host()
    try:
        front = host.wait_for_front(timeout=20)
        host.configure(parameters={"threshold_classifier": True}).result(timeout=60)

        def pointer(action: str, x: float) -> object:
            current = host.front or front
            return host._pointer_event(
                action,
                x,
                0.5,
                button=1,
                identity=current.identity,
                axes=next(
                    item
                    for item in current.interaction.axes
                    if item.role == "main"
                ),
                interaction=current.interaction,
            ).result(timeout=20)

        def candidate() -> float | None:
            state = host.dispatch_control(
                lambda: host._worker_adapter._session()
                ._selector_controller.candidate_state()
            ).result(timeout=20).value
            return None if state is None else float(state.value)

        settled = host.selector_state(SelectorKind.THRESHOLD, display=False).result(
            timeout=20
        ).value
        assert settled is not None
        pointer("press", 0.456)
        assert candidate() == pytest.approx(float(settled.value))

        tracked = []
        for step in range(6):
            pointer("move", 0.47 + 0.03 * step)
            tracked.append(candidate())
        assert len(set(tracked)) == len(tracked), tracked
        assert tracked == sorted(tracked)

        pointer("release", 0.47 + 0.03 * 5)
        committed = host.selector_state(SelectorKind.THRESHOLD, display=False).result(
            timeout=20
        ).value
        assert float(committed.value) == pytest.approx(tracked[-1])
    finally:
        host.close(timeout=20)


def test_a_pointer_runs_before_data_frames_already_queued_behind_it() -> None:
    """Interaction is a person's hand; a frame is worth serving whenever.

    Also: a task coalesces against the newest one with its key ANYWHERE in
    the queue.  Checking only the tail meant one interleaved task of any
    other kind made every following pointer move its own frame.
    """

    from zlc_plot.raster import _DispatchMode

    host = _histogram_host()
    try:
        host.wait_for_front(timeout=20)
        gate = Event()
        started = Event()
        order: list[str] = []

        def block() -> None:
            started.set()
            gate.wait(10.0)

        blocker = host._submit(block, mode=_DispatchMode.CONTROL)
        assert started.wait(5.0)

        first_move = host._submit(
            lambda: order.append("move"),
            mode=_DispatchMode.ADAPTIVE,
            coalesce_key="move",
        )
        frame = host._submit(
            lambda: order.append("frame"), mode=_DispatchMode.PUBLISH
        )
        newest_move = host._submit(
            lambda: order.append("move"),
            mode=_DispatchMode.ADAPTIVE,
            coalesce_key="move",
        )
        gate.set()
        blocker.result(timeout=20)
        newest_move.result(timeout=20)
        frame.result(timeout=20)

        assert first_move.cancelled(), "a queued move must coalesce, not pile up"
        assert order == ["move", "frame"], order
    finally:
        gate.set()
        host.close(timeout=20)
