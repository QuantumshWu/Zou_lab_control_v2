"""The exact blit is a SPEED path; Matplotlib's draw is the specification.

``_blit_exact_rgba_image`` copies a precomposed RGBA front straight into
the Agg buffer instead of letting the artist draw it.  That is only ever
allowed to be faster, never different, so every surface that takes the fast
path is composed twice here -- once with the copy, once with the copy
refused so the artist draws -- and the two frames are compared pixel by
pixel.

The device pixel ratios matter: a panel margin of 16.8 design pixels lands
on a whole pixel at ratio 1 and on 50.4 at ratio 3, and it is exactly the
fractional boxes where a copy could land somewhere the draw would not.
"""
from __future__ import annotations

import numpy as np
import pytest

from zlc_plot import AxisRef, CurvePlot, ImagePlot, PlotSession
from zlc_plot import rendering as rendering_module

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable


def _camera_snapshot(height: int, width: int, revision: int = 1, scale: float = 1.0):
    rng = np.random.default_rng(4)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"shot": np.asarray([0.0])}),
        data_axes=(
            Axis.create("y", values=[float(i) for i in range(height)]),
            Axis.create("x", values=[float(i) for i in range(width)]),
        ),
        dtype=np.uint16,
        generation="blit-parity",
    )
    yy, xx = np.mgrid[0:height, 0:width]
    blob = 3000.0 * np.exp(
        -(((yy - height / 2) ** 2) + ((xx - width / 2) ** 2))
        / (2 * (height / 6.0) ** 2)
    )
    values = np.clip(
        (blob + rng.normal(scale=90.0, size=(height, width)) + 300.0) * scale,
        0,
        65535,
    ).astype(np.uint16)[None, None]
    return DatasetSnapshot(schema, values, revision=revision)


def _compose_both_ways(session: PlotSession) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(with_blit, with_artist_draw)`` for the current scene."""

    renderer = rendering_module.MatplotlibRenderer
    fast = np.array(session.rgba(), copy=True)
    original = renderer._blit_exact_rgba_image
    try:
        renderer._blit_exact_rgba_image = lambda self, artist, canvas: False
        session._renderer._composed_generation = -1
        drawn = np.array(session.rgba(), copy=True)
    finally:
        renderer._blit_exact_rgba_image = original
        session._renderer._composed_generation = -1
    return fast, drawn


@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("preset", ["2x2", "4x4"])
def test_the_exact_blit_paints_what_the_artist_would(ratio, preset) -> None:
    """A camera image, whose front is composed to the box and copied."""

    session = PlotSession(
        _camera_snapshot(256, 256),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
        device_pixel_ratio=ratio,
    )
    try:
        session.set_size(preset)
        session.update_data(_camera_snapshot(256, 256, revision=2))
        fast, drawn = _compose_both_ways(session)
    finally:
        session.close()
    np.testing.assert_array_equal(fast, drawn)


@pytest.mark.parametrize("ratio", [1.0, 1.5, 3.0])
def test_the_height_bar_scene_is_copied_exactly(ratio) -> None:
    """The 3D scene: a full-panel front on a box that ratio 3 makes fractional.

    Its frame is finished over its own background, so the copy and the
    draw must agree -- and this is the surface the fractional box used to
    exclude from the fast path entirely.
    """

    session = PlotSession(
        _camera_snapshot(64, 64),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
        device_pixel_ratio=ratio,
    )
    try:
        session.set_size("4x4")
        session.set_parameters({"presentation": "height_bars"})
        session.update_data(_camera_snapshot(64, 64, revision=2))
        fast, drawn = _compose_both_ways(session)
    finally:
        session.close()
    np.testing.assert_array_equal(fast, drawn)


@pytest.mark.parametrize("ratio", [1.0, 3.0])
def test_a_curve_panel_is_unaffected_by_the_fast_path(ratio) -> None:
    """A surface with no image front composes identically either way."""

    rng = np.random.default_rng(2)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=4),
        PointTable.from_columns({"x": np.arange(64.0)}),
        generation="blit-parity-curve",
    )
    values = rng.normal(size=(4, 64))
    session = PlotSession(
        DatasetSnapshot(schema, values, revision=1),
        CurvePlot(AxisRef.point("x")),
        device_pixel_ratio=ratio,
    )
    try:
        session.set_size("4x4")
        fast, drawn = _compose_both_ways(session)
    finally:
        session.close()
    np.testing.assert_array_equal(fast, drawn)


def _states(session, feed_shape):
    """Drive one surface through the states an operator drives it through.

    Steady-state revisions alone were not enough.  The height-bar scene took
    the fallback only DURING a drag -- the preview renders at a fraction of
    the box, and a front that is not the size of its rectangle cannot be
    copied -- and a zoom left the picture a seventh of a pixel inside its
    own axes.  Both were invisible to a test that only pushed data through.
    """

    height, width = feed_shape
    revision = [2]

    def revisions(count=3):
        for _ in range(count):
            session.update_data(_camera_snapshot(height, width, revision[0]))
            revision[0] += 1
            session.rgba()

    def bounds_point(transform, fx, fy):
        left, top, right, bottom = transform.bounds
        return (left + (right - left) * fx, top + (bottom - top) * fy)

    def gesture(transform, actions, button):
        for action, fx, fy in actions:
            x, y = bounds_point(transform, fx, fy)
            session._raster_pointer_event(
                action, x, y, button=button, axes_snapshot=transform
            )
            session.rgba()

    revisions()
    main = session._raster_axes_snapshot()[0]
    drag = [("press", .35, .35)]
    drag += [("move", .35 + .02 * step, .35 + .02 * step) for step in range(1, 9)]
    drag += [("release", .51, .51)]
    gesture(main, drag, 1)
    pan = [("press", .5, .5)]
    pan += [("move", .5 + .02 * step, .5 + .01 * step) for step in range(1, 9)]
    pan += [("release", .66, .58)]
    gesture(main, pan, 2)
    for direction in (-1.0, -1.0, -1.0, 1.0, 1.0):
        x, y = bounds_point(main, 0.5, 0.5)
        session._raster_pointer_event(
            "scroll", x, y, step=direction, axes_snapshot=main
        )
        session.rgba()
        revisions(1)
    revisions()


@pytest.mark.parametrize(
    "presentation", ["heatmap", "height_bars"], ids=["heatmap", "3d"]
)
@pytest.mark.parametrize(
    "shape",
    [(1200, 1920), (1920, 1200), (1200, 1200), (40, 60), (3, 2)],
    ids=lambda shape: "%dx%d" % shape,
)
@pytest.mark.parametrize("ratio", [1.0, 1.5, 3.0])
def test_no_image_front_is_ever_left_to_matplotlib(shape, ratio, presentation) -> None:
    """The copy is not a lucky case: a refused front is a REGRESSION.

    The copy used to require the front to fill its axes, which a square
    field with a non-square frame letterboxed in it never does -- so the
    operator's 1200x1920 camera silently paid Matplotlib's whole image
    machinery on every frame, twenty milliseconds of it, while the bench
    matrix (square frames, every one) reported the fast path working.  The
    front is composed AT THE BOX now, with the picture placed on whole
    pixels and the band beside it taking the axes' own background, so the
    extent, the limits and the box are one rectangle whatever the view.

    So this asserts the absence of the fallback, not the presence of the
    fast path.  Any state that lands back on Matplotlib fails here.
    """

    height, width = shape
    renderer_class = rendering_module.MatplotlibRenderer
    original = renderer_class._blit_exact_rgba_image
    verdicts: list[bool] = []

    def watched(self, artist, canvas):
        from matplotlib.image import AxesImage

        answer = original(self, artist, canvas)
        if isinstance(artist, AxesImage):
            verdicts.append(bool(answer))
        return answer

    renderer_class._blit_exact_rgba_image = watched
    try:
        session = PlotSession(
            _camera_snapshot(height, width),
            ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
            device_pixel_ratio=ratio,
        )
        try:
            session.set_size("4x4")
            session.set_parameters({"presentation": presentation})
            session.rgba()
            verdicts.clear()
            _states(session, shape)
        finally:
            session.close()
    finally:
        renderer_class._blit_exact_rgba_image = original
    assert verdicts, "no image front reached the compose at all"
    assert all(verdicts), (
        "%d of %d image fronts fell back to Matplotlib's image machinery"
        % (verdicts.count(False), len(verdicts))
    )


@pytest.mark.parametrize("ratio", [1.0, 3.0])
def test_a_confined_gesture_composes_what_a_full_draw_would(ratio) -> None:
    """A turning camera moves the scene; the frame is still the whole frame.

    For the length of a camera drag the colorbar and the distribution rail
    are treated as chrome rather than dynamics -- they cannot change, and
    repainting them cost seven and a half milliseconds a move.  That is only
    sound while the composed frame stays identical to a full draw, which is
    what this measures: the same scene, composed and drawn, pixel for pixel.
    """

    session = PlotSession(
        _camera_snapshot(48, 64),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
        device_pixel_ratio=ratio,
    )
    try:
        session.set_size("4x4")
        session.set_parameters({"presentation": "height_bars"})
        session.rgba()
        renderer = session._renderer
        renderer.set_height_bars_dragging(True)
        try:
            for index, azimuth in enumerate((-40.0, -25.0, -10.0)):
                session.set_parameter("camera_azimuth", azimuth)
                if index == 1:
                    # A shot lands mid-drag.  Its colour limits move the
                    # colorbar's labels and the rail's range -- both on axes
                    # the confinement is treating as chrome -- so this is
                    # where a background captured before it would be stale.
                    session.update_data(
                        # A different RANGE, or the colour limits do not move
                        # and this proves nothing: the whole question is
                        # whether chrome baked before the shot goes stale.
                        _camera_snapshot(48, 64, revision=index + 2, scale=0.4)
                    )
                composed = np.array(session.rgba(), copy=True)
                canvas = renderer.figure.canvas
                renderer._native_draw(canvas)
                drawn = np.asarray(canvas.buffer_rgba()).copy()
                assert composed.shape == drawn.shape
                difference = np.abs(
                    composed.astype(np.int16) - drawn.astype(np.int16)
                )
                assert int(difference.max()) == 0, (
                    "a confined-gesture compose differs from a full draw on "
                    "%d pixels at azimuth %s"
                    % (int((difference.max(axis=2) > 0).sum()), azimuth)
                )
                renderer._composed_generation = -1
        finally:
            renderer.set_height_bars_dragging(False)
    finally:
        session.close()
