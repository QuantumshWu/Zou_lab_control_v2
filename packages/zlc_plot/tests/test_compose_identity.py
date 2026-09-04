"""Composed frames are bit-identical to full draws, including tick churn.

The renderer's chrome-background compose repaints boundary tick marks and
grid lines as dynamics.  Those must be the exact position-fresh, view-clipped
subset a full ``Axis.draw`` paints: the raw ``majorTicks`` lists keep stale
instances parked at out-of-view locations after a limit change, and painting
those leaked mark segments outside the axes box (the "stray tick lines"
beside a 2D image's distribution pane).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot, REPEAT, SPATIAL_X, SPATIAL_Y

from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotLabels,
    PlotSession,
    RollingPlot,
)
from zlc_plot import _raster_kernels as kernels
from zlc_plot.rendering import MatplotlibRenderer
from zlc_plot._selector_scene import ColorLimitCandidate
from zlc_plot.selectors import (
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorState,
)
from zlc_plot.style import style_context


def _image_contract(size: int):
    repeat = repeat_domain(values=np.array([0], dtype=np.int64))
    y_axis = axis("camera_y", values=np.arange(size, dtype=np.int32), role=SPATIAL_Y)
    x_axis = axis("camera_x", values=np.arange(size, dtype=np.int32), role=SPATIAL_X)
    points = mapped_domain_from_columns({"sample": np.array([0], dtype=np.int64)})
    return make_dataset_schema(
        repeat,
        points,
        cell_axes=(y_axis, x_axis),
        value_unit="1",
        dtype=np.uint16,
    )


def _snapshot(schema, size: int, scale: float, revision: int, seed: int):
    rng = np.random.default_rng(seed)
    frame = (
        rng.normal(0.5, 0.15, size=(size, size)).clip(0.01, 1.0) * scale
    ).astype(np.uint16)
    return make_snapshot(
        schema, frame[np.newaxis, np.newaxis, :, :], revision=revision
    )


def _composed_matches_full_draw(session) -> int:
    renderer = session._renderer
    composed = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
    renderer.draw()
    full = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
    return int(np.count_nonzero(np.any(composed != full, axis=-1)))


def _composed_matches_owned_recompose(session) -> int:
    """Compare two passes through the renderer's selected consumer."""

    renderer = session._renderer
    composed = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
    with style_context(renderer.style):
        renderer._compose_frame(chrome_stable=True)
    repeated = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
    return int(np.count_nonzero(np.any(composed != repeated, axis=-1)))


def test_composed_frame_is_full_draw_exact_across_a_tick_shrink() -> None:
    size = 128
    schema = _image_contract(size)
    session = PlotSession(
        _snapshot(schema, size, 40000.0, 1, seed=3),
        ImagePlot(
            AxisRef.cell_data("camera_x"),
            AxisRef.cell_data("camera_y"),
            labels=PlotLabels("compose", "x", "y", value="Counts"),
        ),
    )
    try:
        session.update_data(_snapshot(schema, size, 40000.0, 2, seed=4))
        assert _composed_matches_full_draw(session) == 0

        # The value range collapses 1000x: the distribution/colorbar tick
        # sets shrink, stranding stale tick instances at wide positions.
        session.update_data(_snapshot(schema, size, 35.0, 3, seed=5))
        assert _composed_matches_full_draw(session) == 0

        # Steady state composes over the cached chrome background.
        session.update_data(_snapshot(schema, size, 30.0, 4, seed=6))
        assert _composed_matches_full_draw(session) == 0
    finally:
        session.close()


def test_color_limit_preview_composes_without_touching_chrome() -> None:
    size = 128
    schema = _image_contract(size)
    session = PlotSession(
        _snapshot(schema, size, 200.0, 1, seed=9),
        ImagePlot(
            AxisRef.cell_data("camera_x"),
            AxisRef.cell_data("camera_y"),
            labels=PlotLabels("preview", "x", "y", value="Counts"),
        ),
    )
    try:
        session.update_data(_snapshot(schema, size, 200.0, 2, seed=10))
        session.set_viewport(NumericRange(-64.5, 191.5), NumericRange(-0.5, 127.5))
        renderer = session._renderer
        session.rgba()

        def pixels_of(axis):
            pixels = np.asarray(renderer.figure.canvas.buffer_rgba())
            height = pixels.shape[0]
            box = axis.bbox
            left = max(0, int(np.floor(box.x0)))
            right = min(pixels.shape[1], int(np.ceil(box.x1)))
            top = max(0, int(np.floor(height - box.y1)))
            bottom = min(height, int(np.ceil(height - box.y0)))
            return np.array(pixels[top:bottom, left:right], copy=True)

        colorbar_axis = renderer._artists["image:colorbar"].ax
        colorbar_before = pixels_of(colorbar_axis)
        before_front = np.array(renderer._artists["image:applied_front"], copy=True)
        current = renderer._resolved_color_limit_state()
        assert current is not None
        with renderer.raster_transaction():
            renderer.begin_color_limit_gesture(ColorLimitCandidate(current.value))
        np.testing.assert_array_equal(
            renderer._artists["image:applied_front"], before_front
        )
        np.testing.assert_array_equal(pixels_of(colorbar_axis), colorbar_before)

        before = renderer._background_signature
        with renderer.raster_transaction():
            renderer.preview_color_limit_candidate(
                ColorLimitCandidate(NumericRange(20.0, 150.0))
            )
        np.testing.assert_array_equal(pixels_of(colorbar_axis), colorbar_before)
        preview_pixels = np.array(
            renderer.figure.canvas.buffer_rgba(), copy=True
        )
        renderer._forget_gesture_region()
        with style_context(renderer.style):
            renderer._compose_frame(chrome_stable=True)
        np.testing.assert_array_equal(
            renderer.figure.canvas.buffer_rgba(), preview_pixels
        )
        session.update_data(_snapshot(schema, size, 200.0, 3, seed=11))
        np.testing.assert_array_equal(pixels_of(colorbar_axis), colorbar_before)
        with renderer.raster_transaction():
            renderer.preview_color_limit_candidate(
                ColorLimitCandidate(NumericRange(20.0, 145.0))
            )
        np.testing.assert_array_equal(pixels_of(colorbar_axis), colorbar_before)
        # The preview repainted pixels without invalidating the chrome
        # background: no colorbar label rewrite, no full recapture.
        assert not renderer._chrome_dirty_axes
        assert renderer._background_signature == before
        image = renderer._artists["image"]
        assert tuple(map(float, image.get_clim())) == (20.0, 145.0)
        preview_front = renderer._artists["image:applied_front"]
        assert preview_front.shape == before_front.shape
        background = renderer._axes_background_rgba(image.axes)
        _rows, columns = renderer._artists["image:view_sampling"][1]
        column, column_stop, _column_map = columns
        assert column > 0 and column_stop < preview_front.shape[1]
        assert np.all(preview_front[:, :column] == background)
        assert np.all(preview_front[:, column_stop:] == background)
        renderer.end_selector_gesture()
        session.set_color_limits(20.0, 145.0, fixed=True)
        assert renderer._artists["image:colorbar_state"][1] == (20.0, 145.0)
        assert renderer._artists["image:colorbar"].outline.get_visible()
    finally:
        session.close()


@pytest.mark.parametrize("device_pixel_ratio", (1.0, 2.0))
def test_tight_chrome_reuses_one_exact_background_without_residue(
    monkeypatch,
    device_pixel_ratio: float,
) -> None:
    size = 128
    schema = _image_contract(size)
    spec = ImagePlot(
        AxisRef.cell_data("camera_x"),
        AxisRef.cell_data("camera_y"),
        labels=PlotLabels("tight", "x", "y", value="Counts"),
    )
    session = PlotSession(
        _snapshot(schema, size, 40000.0, 1, seed=12),
        spec,
        parameters={"relim_mode": "tight"},
        device_pixel_ratio=device_pixel_ratio,
    )
    try:
        session.update_data(_snapshot(schema, size, 35000.0, 2, seed=13))
        renderer = session._renderer
        native_draw = MatplotlibRenderer._native_draw
        native_draws = 0

        def counted_draw(canvas) -> None:
            nonlocal native_draws
            native_draws += 1
            native_draw(canvas)

        monkeypatch.setattr(
            MatplotlibRenderer,
            "_native_draw",
            staticmethod(counted_draw),
        )
        session.update_data(_snapshot(schema, size, 35.0, 3, seed=14))
        small = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
        _cmap, limits, _label = renderer._artists["image:colorbar_state"]
        assert tuple(renderer._artists["image:colorbar_mappable"].get_clim()) == limits
        assert tuple(renderer._artists["image:colorbar"].get_ticks()) == limits
        renderer._composed_generation = -1
        repeated = np.array(session.rgba(), copy=True)
        np.testing.assert_array_equal(repeated, small)
        session.update_data(_snapshot(schema, size, 39000.0, 4, seed=15))

        assert native_draws <= 1
        composed = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
        renderer.draw()
        full = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
        np.testing.assert_array_equal(composed, full)
    finally:
        session.close()


@pytest.mark.parametrize("device_pixel_ratio", (1.0, 2.0))
def test_tight_fit_and_area_selector_remain_full_draw_exact(
    device_pixel_ratio: float,
) -> None:
    size = 128
    schema = _image_contract(size)
    coordinate = np.arange(size, dtype=float)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")

    def fitted_snapshot(revision: int, shift: float) -> OwnedSnapshot:
        values = 100.0 + 4000.0 * np.exp(
            -(
                (xx - (64.0 + shift)) ** 2
                + (yy - (64.0 - shift)) ** 2
            )
            / (2.0 * 7.0**2)
        )
        return make_snapshot(
            schema,
            values.astype(np.uint16)[np.newaxis, np.newaxis, :, :],
            revision=revision,
        )

    session = PlotSession(
        fitted_snapshot(1, 0.0),
        ImagePlot(AxisRef.cell_data("camera_x"), AxisRef.cell_data("camera_y")),
        device_pixel_ratio=device_pixel_ratio,
    )
    try:
        session.set_area_selector(
            NumericRange(50.5, 76.5),
            NumericRange(50.5, 76.5),
            display=False,
        )
        session.fit("radial_gaussian_center", live=True)
        renderer = session._renderer
        background = renderer._background_region

        session.update_data(fitted_snapshot(2, 0.4))
        assert renderer._background_region is background
        assert _composed_matches_full_draw(session) == 0
    finally:
        session.close()


def test_tight_colorbar_updates_its_proxy_once_per_frame(monkeypatch) -> None:
    """A stable colormap must not rebuild the Colorbar before its new clim."""

    from matplotlib.colorbar import Colorbar

    size = 64
    schema = _image_contract(size)
    session = PlotSession(
        _snapshot(schema, size, 4000.0, 1, seed=31),
        ImagePlot(AxisRef.cell_data("camera_x"), AxisRef.cell_data("camera_y")),
    )
    try:
        draws = 0
        native = Colorbar._draw_all

        def counted(colorbar) -> None:
            nonlocal draws
            draws += 1
            native(colorbar)

        monkeypatch.setattr(Colorbar, "_draw_all", counted)
        session.update_data(_snapshot(schema, size, 3000.0, 2, seed=32))
        assert draws <= 1
        renderer = session._renderer
        _cmap, limits, _label = renderer._artists["image:colorbar_state"]
        assert tuple(renderer._artists["image:colorbar_mappable"].get_clim()) == limits
        assert tuple(renderer._artists["image:colorbar"].get_ticks()) == limits
        assert _composed_matches_full_draw(session) == 0
    finally:
        session.close()


def _generic_kind_pair(kind: str):
    if kind == "facet":
        schema = make_dataset_schema(
            repeat_domain(size=1),
            mapped_domain_from_columns({"facet": (0.0, 1.0)}),
            cell_axes=(
                axis("y", values=np.arange(8.0), role=SPATIAL_Y),
                axis("x", values=np.arange(12.0), role=SPATIAL_X),
            ),
            dtype=np.float64,
        )
        first = np.arange(192.0).reshape(1, 2, 8, 12)
        second = first[::-1].copy() + 3.0
        return (
            make_snapshot(schema, first, revision=1),
            make_snapshot(schema, second, revision=2),
            FacetGridPlot(
                AxisRef.point("facet"),
                ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
            ),
        )
    if kind == "rolling":
        schema = make_dataset_schema(
            repeat_domain(size=12),
            mapped_domain_from_columns({"sample": (0.0,)}),
            dtype=np.float64,
        )
        first = np.arange(12.0).reshape(12, 1)
        return (
            make_snapshot(schema, first, revision=1),
            make_snapshot(schema, first + 1.0, revision=2),
            RollingPlot(),
        )
    points = np.arange(16.0)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": points}),
        dtype=np.float64,
    )
    first = make_snapshot(schema, np.sin(points)[np.newaxis, :], revision=1)
    second = make_snapshot(
        schema,
        np.cos(points)[np.newaxis, :],
        revision=2,
    )
    spec = (
        CurvePlot(AxisRef.point("x"))
        if kind == "curve"
        else HistogramPlot()
    )
    return first, second, spec


@pytest.mark.parametrize("kind", ("curve", "histogram", "rolling", "facet"))
@pytest.mark.parametrize("device_pixel_ratio", (1.0, 2.0))
def test_generic_kind_updates_remain_stable_under_their_owned_draw_path(
    kind: str,
    device_pixel_ratio: float,
) -> None:
    first, second, spec = _generic_kind_pair(kind)
    session = PlotSession(first, spec, device_pixel_ratio=device_pixel_ratio)
    try:
        session.update_data(second)
        if kind == "curve":
            # Dense Curve deliberately uses the native raster consumer; its
            # anti-aliasing is close to, but not byte-identical with, Agg.
            # What must be exact is that the accepted prepared scene produces
            # the same pixels again, rather than alternating consumers across
            # revisions.  Pixel proximity to Agg is measured separately.
            assert _composed_matches_owned_recompose(session) == 0
        else:
            assert _composed_matches_full_draw(session) == 0
    finally:
        session.close()


def test_the_side_frames_stay_above_their_own_content() -> None:
    """The rail's spines and the colorbar's outline draw ABOVE their fills.

    The native-image compose used to pull ALL boundary chrome into its
    first pass, so the colorbar gradient and the rail histogram painted
    over the inner half of their own black frames -- on the console the
    borders looked partly missing.  Only chrome the image raster can
    overwrite comes forward; everything else keeps full-draw z order.
    """

    schema = _image_contract(96)
    session = PlotSession(
        _snapshot(schema, 96, 30000.0, 1, seed=3),
        ImagePlot(
            AxisRef.cell_data("camera_x"),
            AxisRef.cell_data("camera_y"),
            labels=PlotLabels("frames", "x", "y", value="Counts"),
        ),
    )
    try:
        session.set_size("2x2")
        pixels = np.asarray(session.rgba())
        height = pixels.shape[0]
        gray = pixels[..., :3].min(axis=2)
        renderer = session._renderer

        def edge_dark(role, side):
            box = renderer._axes[role][0].bbox
            x0, x1 = int(box.x0), int(box.x1)
            top = int(height - box.y1)
            bottom = int(height - box.y0)
            if side == "top":
                band = gray[max(top - 2, 0) : top + 3, x0:x1]
                return float(np.mean((band < 128).any(axis=0)))
            band = gray[top:bottom, max(x0 - 2, 0) : x0 + 3]
            return float(np.mean((band < 128).any(axis=1)))

        assert edge_dark("colorbar", "top") > 0.9, "colorbar outline top missing"
        assert edge_dark("colorbar", "left") > 0.9, "colorbar outline left missing"
        assert edge_dark("distribution", "left") > 0.9, "rail left spine missing"
    finally:
        session.close()


def _side_axis_ids(renderer) -> dict[str, set[int]]:
    """The colour scale's long axis and the rail's two axes, by role."""

    roles: dict[str, set[int]] = {}
    for role in ("colorbar", "distribution"):
        roles[role] = {
            id(axis)
            for axes in renderer._axes.get(role, ())
            for axis in (axes.xaxis, axes.yaxis)
        }
    return roles


def _replayed_axis_ids(renderer) -> set[int]:
    return {
        axis_id
        for axis_id, (_key, commands) in renderer._dynamic_axis_commands.items()
        if commands
    }


def test_held_side_chrome_is_replayed_and_stays_full_draw_exact() -> None:
    """Limits that hold: the colour scale and rail axes replay their draw.

    Four revisions of the same picture under the default hysteresis keep
    every side limit in place, so from the third frame on the colour scale's
    long axis and the rail's axes are replayed rather than redrawn -- and
    the replayed frame is still the full draw, pixel for pixel.
    """

    size = 128
    schema = _image_contract(size)
    session = PlotSession(
        _snapshot(schema, size, 40000.0, 1, seed=31),
        ImagePlot(
            AxisRef.cell_data("camera_x"),
            AxisRef.cell_data("camera_y"),
            labels=PlotLabels("replay", "x", "y", value="Counts"),
        ),
        parameters={"relim_mode": "normal"},
    )
    try:
        renderer = session._renderer
        for revision, seed in ((2, 32), (3, 33), (4, 34), (5, 35)):
            session.update_data(_snapshot(schema, size, 40000.0, revision, seed=seed))
        roles = _side_axis_ids(renderer)
        replayed = _replayed_axis_ids(renderer)
        assert replayed & roles["colorbar"], "the held colour scale was never replayed"
        assert replayed & roles["distribution"], "the held rail was never replayed"
        assert replayed <= roles["colorbar"] | roles["distribution"]
        assert _composed_matches_full_draw(session) == 0
    finally:
        session.close()


def test_moving_colour_scale_is_never_recorded() -> None:
    """A key that changes every frame is drawn plainly, never recorded.

    TIGHT re-fits the colour scale to each revision of a range that moves,
    so its endpoint labels are new every frame and the long axis' key never
    repeats.  The frame stays full-draw exact either way.
    """

    size = 128
    schema = _image_contract(size)
    session = PlotSession(
        _snapshot(schema, size, 40000.0, 1, seed=41),
        ImagePlot(
            AxisRef.cell_data("camera_x"),
            AxisRef.cell_data("camera_y"),
            labels=PlotLabels("tight-replay", "x", "y", value="Counts"),
        ),
        parameters={"relim_mode": "tight"},
    )
    try:
        renderer = session._renderer
        scales = (40000.0, 41000.0, 39500.0, 40500.0)
        for revision, scale in enumerate(scales, start=2):
            session.update_data(_snapshot(schema, size, scale, revision, seed=40 + revision))
        roles = _side_axis_ids(renderer)
        replayed = _replayed_axis_ids(renderer)
        assert not (replayed & roles["colorbar"]), "a re-fitted scale must not replay"
        assert _composed_matches_full_draw(session) == 0
    finally:
        session.close()


def test_replayed_side_chrome_follows_a_limit_change_exactly() -> None:
    """A replay that no longer matches its key is dropped, not reused.

    After the side chrome has settled into replay, the value range collapses
    a thousandfold: every key changes, the recorded draws are stale, and the
    frame must come from a fresh draw -- full-draw exact, with no stale mark
    or label from the replayed frames.
    """

    size = 128
    schema = _image_contract(size)
    session = PlotSession(
        _snapshot(schema, size, 40000.0, 1, seed=51),
        ImagePlot(
            AxisRef.cell_data("camera_x"),
            AxisRef.cell_data("camera_y"),
            labels=PlotLabels("collapse", "x", "y", value="Counts"),
        ),
        parameters={"relim_mode": "normal"},
    )
    try:
        renderer = session._renderer
        for revision in (2, 3, 4):
            session.update_data(_snapshot(schema, size, 40000.0, revision, seed=50 + revision))
        assert _replayed_axis_ids(renderer), "the scene never settled into replay"
        session.update_data(_snapshot(schema, size, 35.0, 5, seed=55))
        assert _composed_matches_full_draw(session) == 0
    finally:
        session.close()


def _curve_contract(points: int, repeats: int):
    repeat = repeat_domain(values=np.arange(repeats, dtype=np.int64))
    table = mapped_domain_from_columns({"x": np.linspace(-3.0, 3.0, points)})
    return make_dataset_schema(
        repeat,
        table,
        value_unit="1",
        dtype=np.float64,
    )


def _curve_snapshot(schema, points: int, repeats: int, revision: int, seed: int):
    rng = np.random.default_rng(seed)
    x = np.linspace(-3.0, 3.0, points)
    # The peak height moves with the revision so TIGHT limits re-fit on
    # every frame, which is what keeps the chrome background missing.
    peak = 40.0 + 15.0 * np.sin(revision)
    values = peak * np.exp(-0.5 * (x / 0.8) ** 2) + rng.normal(0.0, 1.5, (repeats, points))
    return make_snapshot(schema, values[..., None], revision=revision)


def _live_advance(session, snapshot) -> None:
    prepared = session.prepare_live_frame(snapshot).result()
    solved = session.solve_live_frame(prepared)
    finalization = session.commit_live_frame(
        prepared, None if solved is None else solved.result()
    )
    assert finalization is not None, "the live frame was not committed"
    session.publish_live_frame(finalization)


@pytest.mark.parametrize("spec_kind", ["curve", "facet_curve"])
def test_a_re_fitting_curve_stays_full_draw_exact_frame_after_frame(spec_kind: str) -> None:
    """Limits that re-fit every shot keep the chrome background missing.

    Two misses in a row used to send the compose down a bare full draw --
    complete for a scene of artists, an empty axes for a curve whose data
    is a prepared scene stroked by the kernels.  A Curve in TIGHT mode, or
    a Facet grid of curve cells that inherited TIGHT from its image cells,
    went blank from its second live frame on and stayed blank while the
    data kept moving.  Every frame here must be the full draw, pixel for
    pixel, and must actually contain the data.
    """

    points, repeats = 160, 6
    schema = _curve_contract(points, repeats)
    cell = CurvePlot(AxisRef.point("x"), labels=PlotLabels("tight-curve", "x", "y"))
    spec = (
        cell
        if spec_kind == "curve"
        else FacetGridPlot(AxisRef.repeat("repeat"), CurvePlot(AxisRef.point("x")))
    )
    session = PlotSession(
        _curve_snapshot(schema, points, repeats, 1, seed=61),
        spec,
        parameters={"relim_mode": "tight", "uncertainty": spec_kind == "curve"},
    )
    try:
        session.configure(selectors=(), fit={}, fit_live=True)
        renderer = session._renderer
        session.rgba()
        first = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
        chrome_only = None
        for revision in range(2, 7):
            _live_advance(session, _curve_snapshot(schema, points, repeats, revision, seed=60 + revision))
            if revision >= 4:
                # The oracle draw below resets the count; read it here.
                assert renderer._chrome_churn > 1, "the scene was meant to keep missing its background"
            composed = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
            renderer.draw()
            full = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
            differing = int(np.count_nonzero(np.any(composed != full, axis=-1)))
            assert differing == 0, f"revision {revision}: {differing} pixels differ from the full draw"
            ink = int(np.count_nonzero(np.any(composed != composed[0, 0], axis=-1)))
            if chrome_only is None:
                chrome_only = ink
            # The frame carries the curve, not just the axes: its ink is
            # within a third of the first frame's, never a fraction of it.
            first_ink = int(np.count_nonzero(np.any(first != first[0, 0], axis=-1)))
            assert ink > 0.66 * first_ink, f"revision {revision}: {ink} ink pixels against {first_ink} on the first frame"
    finally:
        session.close()


def _bimodal_snapshot(schema, points: int, repeats: int, revision: int, seed: int):
    """Two populations per cell, so every cell's bimodal fit converges."""

    rng = np.random.default_rng(seed)
    bright = rng.random((repeats, points)) < 0.5
    values = np.where(
        bright,
        rng.normal(40.0 + revision, 6.0, (repeats, points)),
        rng.normal(8.0, 3.0, (repeats, points)),
    )
    return make_snapshot(schema, np.clip(values, 0.0, None)[..., None], revision=revision)


def test_a_histogram_grid_with_live_cell_fits_stays_full_draw_exact_and_parses_once() -> None:
    """A cell label is a constant mathtext symbol and a plain live value, so
    a shot parses no MathText at all, and the composed frame of a grid of
    histogram cells -- Matplotlib collections, not a native raster -- is the
    full draw, pixel for pixel."""

    from zlc_plot import FacetGridPlot, PlotSession

    points, repeats = 160, 6
    schema = _curve_contract(points, repeats)
    spec = FacetGridPlot(AxisRef.repeat("repeat"), HistogramPlot())
    session = PlotSession(_bimodal_snapshot(schema, points, repeats, 1, seed=71), spec)
    try:
        renderer = session._renderer
        session.rgba()
        result = session.fit("bimodal_gaussian")
        assert len(tuple(result.results)) == repeats
        from matplotlib.mathtext import MathTextParser

        for revision in range(2, 6):
            misses_before = MathTextParser._parse_cached.cache_info().misses
            _live_advance(session, _bimodal_snapshot(schema, points, repeats, revision, seed=70 + revision))
            composed = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
            if revision >= 3:
                assert MathTextParser._parse_cached.cache_info().misses == misses_before, (
                    f"revision {revision}: a cell label was parsed as MathText again"
                )
            renderer.draw()
            full = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
            differing = int(np.count_nonzero(np.any(composed != full, axis=-1)))
            assert differing == 0, f"revision {revision}: {differing} pixels differ from the full draw"
            labels = tuple(
                artist.get_text()
                for artist in renderer._fit_artists
                if hasattr(artist, "get_text") and artist.get_visible()
            )
            symbols = tuple(label for label in labels if "$" in label)
            values = tuple(label for label in labels if "$" not in label)
            assert len(symbols) == repeats and len(set(symbols)) == 1
            assert len(values) == repeats and all("±" in value for value in values)
    finally:
        session.close()


def _ink(buffer) -> int:
    """Pixels that are not the frame's background colour."""

    flat = np.asarray(buffer).reshape(-1, np.asarray(buffer).shape[-1])
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    background = colours[int(np.argmax(counts))]
    return int(np.count_nonzero(np.any(flat != background, axis=-1)))


def _gesture_candidate(kind: SelectorKind) -> SelectorState:
    if kind is SelectorKind.AREA:
        return SelectorState(
            kind, RectangleRange(NumericRange(-2.0, -1.0), NumericRange(5.0, 20.0))
        )
    return SelectorState(kind, NumericRange(-2.0, -1.0))


@pytest.mark.parametrize("gesture", (SelectorKind.AREA, SelectorKind.X_RANGE))
@pytest.mark.parametrize("spec_kind", ("curve", "facet_curve"))
def test_a_selector_gesture_never_erases_the_scene_below_it(
    spec_kind: str, gesture: SelectorKind
) -> None:
    """Opening a gesture may add ink to the frame; it may never take any away.

    The compose splits the z-order where the gesture's own artists begin, so
    a pointer move repaints only the tail over a captured frame.  That split
    says WHERE the frame is cut -- never WHAT is drawn.  Every native pass
    was gated on ``split is None`` instead, so with a gesture open the
    kernels painted nothing, and a kind whose data is a prepared scene has
    no artist to fall back on: the install hides the real lines and bars and
    hands the picture to the kernels.  A curve panel therefore went blank
    for the whole length of an area drag -- chrome and the rubber band, no
    data -- which is exactly what an operator sees while dragging a region
    on a live trace.

    Ink is the operator's own measure, so it is the one asserted here, and
    it holds for every kind and every gesture rather than for the one that
    was reported.
    """

    points, repeats = 160, 6
    schema = _curve_contract(points, repeats)
    spec = (
        CurvePlot(AxisRef.point("x"))
        if spec_kind == "curve"
        else FacetGridPlot(AxisRef.repeat("repeat"), CurvePlot(AxisRef.point("x")))
    )
    session = PlotSession(
        _curve_snapshot(schema, points, repeats, 1, seed=71),
        spec,
        parameters={"uncertainty": spec_kind == "curve"},
    )
    try:
        renderer = session._renderer
        before = np.array(session.rgba(), copy=True)
        prepared = renderer._has_prepared_scene()
        if kernels.engaged():
            assert prepared, (
                "this guard is only worth anything over a kernel-drawn scene, "
                "and this session did not produce one"
            )

        renderer.begin_selector_gesture(gesture)
        renderer.preview_selector(_gesture_candidate(gesture))
        during = np.array(session.rgba(), copy=True)

        assert _ink(during) >= _ink(before), (
            f"the {gesture.value} gesture erased the {spec_kind} scene: "
            f"{_ink(before)} inked pixels before, {_ink(during)} during"
        )
    finally:
        session.close()


@pytest.mark.parametrize("spec_kind", ("curve", "facet_curve"))
def test_a_gesture_move_repaints_the_scene_a_compose_paints(spec_kind: str) -> None:
    """The captured frame a move restores is the frame a compose would draw.

    The cheap move path restores the capture and repaints only the tail, so
    the capture has to be everything below it -- the kernel-stroked data
    included.  Composing the same candidate with the capture thrown away
    must therefore land on the same pixels.
    """

    points, repeats = 160, 6
    schema = _curve_contract(points, repeats)
    spec = (
        CurvePlot(AxisRef.point("x"))
        if spec_kind == "curve"
        else FacetGridPlot(AxisRef.repeat("repeat"), CurvePlot(AxisRef.point("x")))
    )
    session = PlotSession(
        _curve_snapshot(schema, points, repeats, 1, seed=72),
        spec,
        parameters={"uncertainty": spec_kind == "curve"},
    )
    try:
        renderer = session._renderer
        session.rgba()
        renderer.begin_selector_gesture(SelectorKind.AREA)
        candidate = _gesture_candidate(SelectorKind.AREA)
        renderer.preview_selector(candidate)
        moved = SelectorState(
            SelectorKind.AREA,
            RectangleRange(NumericRange(-2.0, -0.5), NumericRange(5.0, 25.0)),
        )
        renderer.preview_selector(moved)
        cheap = np.array(session.rgba(), copy=True)

        renderer._forget_gesture_region()
        renderer.preview_selector(moved)
        expensive = np.array(session.rgba(), copy=True)
        assert np.array_equal(cheap, expensive)
    finally:
        session.close()

