"""A pointer move is not a data frame, and must not be composed like one.

A move used to re-enter the whole compose -- every dynamic artist of every
axes, the image included -- at the same priority as a camera frame, so
interaction cost scaled with the scene and interaction latency was additive
with whatever frame the worker happened to be composing.  What these fix on
is that the cheap path must paint EXACTLY what the expensive one paints; a
faster wrong picture is not a faster picture.
"""

from __future__ import annotations

from threading import Event

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot, READOUT_EVENT, REPEAT, SITE, SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, HistogramPlot, ImagePlot, SelectorKind
from zlc_plot.raster import RasterPlotHost
from zlc_plot.selectors import NumericRange, RectangleRange, SelectorState
from zlc_plot.session import PlotSession

SIZE = 256


def _image_snapshot(revision: int = 0) -> OwnedSnapshot:
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"shot": [0.0]}),
        cell_axes=(
            axis(
                "sy", values=tuple(float(v) for v in range(SIZE)), role=SPATIAL_Y
            ),
            axis(
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
    return make_snapshot(schema, values, revision=revision)


def test_a_move_paints_the_same_pixels_as_a_compose() -> None:
    """The whole licence for the cheap path."""

    session = PlotSession(
        _image_snapshot(), ImagePlot(AxisRef.cell_data("sx"), AxisRef.cell_data("sy")), size="4x4"
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


def test_a_move_reuses_the_gesture_capture_without_composing(monkeypatch) -> None:
    """The cheap path is structural, not a noisy wall-clock ratio."""

    session = PlotSession(
        _image_snapshot(), ImagePlot(AxisRef.cell_data("sx"), AxisRef.cell_data("sy")), size="4x4"
    )
    try:
        renderer = session._renderer
        session.rgba()

        renderer.begin_selector_gesture(SelectorKind.X_RANGE)
        moves = [
            SelectorState(SelectorKind.X_RANGE, NumericRange(60.0 + index, 190.0 + index))
            for index in range(16)
        ]
        # The first candidate creates the selector artist and recaptures the
        # scene. Subsequent pointer moves must reuse that exact topology.
        assert renderer.preview_selector(moves[0]) is True

        def forbidden_compose(*_args, **_kwargs) -> None:
            raise AssertionError("a pointer move re-entered full frame composition")

        monkeypatch.setattr(renderer, "_compose_frame", forbidden_compose)
        generation = renderer.raster_generation
        for move in moves[1:]:
            assert renderer.preview_selector(move) is True
        assert renderer.raster_generation == generation + len(moves) - 1
    finally:
        session.close()


def test_the_gesture_capture_dies_with_anything_that_could_stale_it() -> None:
    session = PlotSession(
        _image_snapshot(), ImagePlot(AxisRef.cell_data("sx"), AxisRef.cell_data("sy")), size="4x4"
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
    schema = make_dataset_schema(
        repeat_domain(size=samples.size),
        mapped_domain_from_columns({"site": [0.0]}, roles={"site": SITE}),
        dtype=np.float64,
    )
    return RasterPlotHost.from_plot(
        make_snapshot(schema, samples.reshape(-1, 1), revision=0),
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
            return host.pointer_event(
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

        settled = host.selector_state(SelectorKind.THRESHOLD, display=False).result(
            timeout=20
        ).value
        assert settled is not None
        pressed = pointer("press", 0.456)

        fronts = [pressed.front]
        for step in range(6):
            fronts.append(pointer("move", 0.47 + 0.03 * step).front)
        assert [front.identity.sequence for front in fronts] == sorted(
            {front.identity.sequence for front in fronts}
        )
        # One distinct image per accepted front.  A front's pixels are a
        # memoryview over its own shared block, and a memoryview of this
        # format cannot go in a set, so compare the bytes they carry.
        assert len({bytes(front.buffer.pixels) for front in fronts}) == len(fronts)

        pointer("release", 0.47 + 0.03 * 5)
        committed = host.selector_state(SelectorKind.THRESHOLD, display=False).result(
            timeout=20
        ).value
        assert float(committed.value) != pytest.approx(float(settled.value))
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


def test_a_focused_facet_can_be_moved_to_another_cell() -> None:
    """The case a geometry handle could not express.

    A focused front publishes geometry for exactly one cell -- the one it is
    showing -- so a caller that named a cell by its axes could ask for any
    cell EXCEPT while one was open, which is precisely when the operator asks.
    """

    from zlc_plot import CurvePlot, FacetGridPlot

    settings = [0.0, 1.0, 2.0]
    points = [0.0, 1.0, 2.0, 3.0]
    rows = [(setting, point) for setting in settings for point in points]
    table = mapped_domain_from_columns(
        {
            "setting": [row[0] for row in rows],
            "shot": [row[1] for row in rows],
        }
    )
    schema = make_dataset_schema(
        repeat_domain(size=1),
        table,
        dtype=np.float64,
    )
    values = np.asarray(
        [[10.0 * row[0] + row[1] for row in rows]], dtype=np.float64
    )
    host = RasterPlotHost.from_plot(
        make_snapshot(schema, values, revision=0),
        FacetGridPlot(
            AxisRef.point("setting"), CurvePlot(AxisRef.point("shot"))
        ),
        size="4x4",
    )
    try:
        host.wait_for_front(timeout=20)

        def shown() -> int | None:
            axes = [
                item
                for item in host.front.interaction.axes
                if item.role == "facet_cell"
            ]
            return axes[0].cell_index if len(axes) == 1 else None

        host.focus_facet(0).result(timeout=20)
        assert shown() == 0
        host.focus_facet(2).result(timeout=20)
        assert shown() == 2
        host.focus_facet(1).result(timeout=20)
        assert shown() == 1
    finally:
        host.close(timeout=20)


def test_a_gesture_frame_is_composed_under_the_same_style() -> None:
    """A drag draws the same picture as a data frame, in the same fonts.

    The selector previews entered the style context to UPDATE the artists and
    left it again to COMPOSE, so the first drag on an image rendered a whole
    frame under matplotlib's defaults: its font list, not ours.  On this bench
    that meant a burst of "font family not found" for families nobody here has
    ever had -- the visible half of a frame drawn in the wrong style.
    """

    import logging

    session = PlotSession(
        _image_snapshot(), ImagePlot(AxisRef.cell_data("sx"), AxisRef.cell_data("sy")), size="4x4"
    )
    messages: list[str] = []

    class _Catch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Catch()
    font_log = logging.getLogger("matplotlib.font_manager")
    font_log.addHandler(handler)
    try:
        session.rgba()
        messages.clear()
        renderer = session._renderer
        renderer.begin_selector_gesture(SelectorKind.AREA)
        renderer.preview_selector(
            SelectorState(
                SelectorKind.AREA,
                RectangleRange(NumericRange(60.0, 190.0), NumericRange(40.0, 150.0)),
            )
        )
        session.rgba()
        missing = [text for text in messages if "not found" in text]
        assert not missing, f"the gesture frame left our style: {missing[:4]}"
    finally:
        font_log.removeHandler(handler)
        session.close()


def _cycle_snapshot(frames: int = 3, revision: int = 0) -> OwnedSnapshot:
    """One camera cycle: its frames are the point rows a grid faces."""

    side = 32
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns(
            {"frame": [float(i) for i in range(frames)]},
            roles={"frame": READOUT_EVENT},
        ),
        cell_axes=(
            axis(
                "sy", values=tuple(float(v) for v in range(side)), role=SPATIAL_Y
            ),
            axis(
                "sx", values=tuple(float(v) for v in range(side)), role=SPATIAL_X
            ),
        ),
        dtype=np.uint16,
    )
    values = (
        np.random.default_rng(revision)
        .integers(180, 900, (1, frames, side, side))
        .astype(np.uint16)
    )
    return make_snapshot(schema, values, revision=revision)


def _cell_transform(session: PlotSession, index: int):
    """The painted geometry of one grid cell, as a frontend sees it."""

    return next(
        axis
        for axis in session._raster_axes_snapshot()
        if axis.role == "facet_cell" and axis.cell_index == index
    )


def _inside(transform, fraction_x: float, fraction_y: float) -> tuple[float, float]:
    left, top, right, bottom = transform.bounds
    return (
        left + (right - left) * fraction_x,
        1.0 - (top + (bottom - top) * fraction_y),
    )


@pytest.mark.parametrize("cell", (0, 1, 2))
def test_an_overview_grid_only_focuses_then_the_focused_cell_accepts_an_area(
    cell: int,
) -> None:
    """Overview only enters a cell; selector gestures belong to that cell."""

    from zlc_plot import FacetGridPlot

    session = PlotSession(
        _cycle_snapshot(),
        FacetGridPlot(
            AxisRef.point("frame"),
            ImagePlot(AxisRef.cell_data("sx"), AxisRef.cell_data("sy")),
        ),
        size="4x4",
    )
    try:
        session.rgba()
        transform = _cell_transform(session, cell)
        before = session.viewport
        before_pointer_cell = session._focused_facet_index
        session._raster_pointer_event(
            "scroll", *_inside(transform, 0.5, 0.5), step=-1.0,
            axes_snapshot=transform,
        )
        assert session.viewport == before
        assert session._focused_facet_index == before_pointer_cell

        transform = _cell_transform(session, cell)
        session._raster_pointer_event(
            "press", *_inside(transform, 0.3, 0.3), button=1,
            axes_snapshot=transform,
        )
        session._raster_pointer_event(
            "move", *_inside(transform, 0.7, 0.7), button=1,
            axes_snapshot=transform,
        )
        session._raster_pointer_event(
            "release", *_inside(transform, 0.7, 0.7), button=1,
            axes_snapshot=transform,
        )
        assert tuple(session.selectors) == ()
        assert session._focused_facet_index == before_pointer_cell

        # The one allowed overview gesture enters the cell.
        transform = _cell_transform(session, cell)
        session._raster_pointer_event(
            "press", *_inside(transform, 0.5, 0.5), button=1, double=True,
            axes_snapshot=transform,
        )
        assert session._facet_focus_index == cell

        # Once focused, the same area gesture has one unambiguous cell owner.
        transform = _cell_transform(session, cell)
        session._raster_pointer_event(
            "press", *_inside(transform, 0.3, 0.3), button=1,
            axes_snapshot=transform,
        )
        session._raster_pointer_event(
            "move", *_inside(transform, 0.7, 0.7), button=1,
            axes_snapshot=transform,
        )
        session._raster_pointer_event(
            "release", *_inside(transform, 0.7, 0.7), button=1,
            axes_snapshot=transform,
        )
        committed = tuple(session.selectors)
        assert [state.facet_index for state in committed] == [cell]
        assert committed[0].kind is SelectorKind.AREA
    finally:
        session.close()


def test_a_wheel_notch_zooms_the_committed_view_not_the_drawn_one() -> None:
    """The wheel compounds on what the session HOLDS, not on the last frame.

    Renders lag commitments, and once rendering moved into its own process
    they lag by whole frames -- so a notch that arrives before the next frame
    read limits that still showed the step already taken and re-derived it.
    The operator turned the wheel and the picture kept landing back where it
    was.  The camera branch of the same handler was anchored on its committed
    zoom for this exact reason; the viewport branch was not.
    """

    from zlc_plot import DEFAULTS, CurvePlot

    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"shot": [0.0, 1.0, 2.0, 3.0]}),
        dtype=np.float64,
    )
    snapshot = make_snapshot(schema, np.arange(4.0).reshape(1, 4), revision=0)
    session = PlotSession(snapshot, CurvePlot(AxisRef.point("shot")), size="2x2")
    try:
        session.rgba()
        transform = next(
            item
            for item in session._raster_axes_snapshot()
            if item.role == "main"
        )
        session.set_viewport(NumericRange(0.0, 100.0), NumericRange(0.0, 10.0))
        # A frame that has not caught up: the drawn limits are a previous
        # commitment's, which during a wheel burst is the ordinary state.
        session._renderer.primary_axes.set_xlim(0.0, 1000.0)
        factor = float(DEFAULTS.interaction.wheel_zoom_factor)

        session._raster_pointer_event(
            "scroll",
            *_inside(transform, 0.5, 0.5),
            step=1.0,
            axes_snapshot=transform,
        )
        first = session.viewport
        assert first is not None
        width = float(first.x.high) - float(first.x.low)
        assert width == pytest.approx(100.0 * factor, rel=1e-9)

        # And a second notch compounds on the first, whatever the axes say.
        session._renderer.primary_axes.set_xlim(0.0, 1000.0)
        session._raster_pointer_event(
            "scroll",
            *_inside(transform, 0.5, 0.5),
            step=1.0,
            axes_snapshot=transform,
        )
        second = session.viewport
        assert second is not None
        assert float(second.x.high) - float(second.x.low) == pytest.approx(
            100.0 * factor * factor, rel=1e-9
        )
    finally:
        session.close()
