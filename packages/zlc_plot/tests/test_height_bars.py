"""The Image kind's height-bar presentation: kernel, session and gestures.

The scene must be a PRESENTATION of the same image surface: identical
snapshot, payload, clim, colormap and pipeline -- so the contract here is
mostly about what must NOT change: a heatmap->bars->heatmap roundtrip is
bit-identical, committed selectors survive the trip, and the camera is
display state that never touches the projection.
"""

from __future__ import annotations

from time import perf_counter

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest

from data_factory import (
    Axis,
    DatasetSchema,
    DatasetSnapshot,
    PointTable,
    PointTopology,
)
from zlc_data import AxisId
from zlc_plot import AxisRef, FacetGridPlot, ImagePlot, PlotSession
from zlc_plot._height3d_raster import (
    HeightBarCamera,
    render_height_bars,
)
from zlc_plot.selectors import (
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorState,
)

MAX_STRESS_RENDER_SECONDS = 0.5


def _scan_snapshot(
    side: int = 10, *, repeats: int = 4, revision: int = 1, seed: int = 7
):
    rows = side * side
    cells = [(i % side, i // side) for i in range(rows)]
    schema = DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        PointTable.from_columns({
            "ax": np.asarray([float(c[0]) for c in cells]),
            "ay": np.asarray([float(c[1]) for c in cells]),
        }),
        data_axes=(Axis.create("site", values=[0.0, 1.0]),),
        dtype=np.float64,
        point_topology=PointTopology(
            (AxisId("ax"), AxisId("ay")),
            (tuple(float(i) for i in range(side)),) * 2,
            tuple(cells),
        ),
    )
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(side), np.arange(side))
    profile = np.exp(
        -((xx - side / 2) ** 2 + (yy - side / 2) ** 2) / (side / 1.5)
    )
    values = profile.reshape(-1)[None, :, None] + rng.normal(
        scale=0.05, size=(repeats, rows, 2)
    )
    return DatasetSnapshot(schema, values, revision=revision)


def _session(side: int = 10) -> PlotSession:
    session = PlotSession(
        _scan_snapshot(side),
        ImagePlot(AxisRef.point_dimension("ax"), AxisRef.point_dimension("ay")),
    )
    session.set_size("2x2")
    return session


# --------------------------------------------------------------- kernel
@pytest.mark.parametrize("azimuth", [-55.0, 40.0, 130.0, 220.0, 305.0])
def test_pick_inverts_projection_in_every_quadrant(azimuth) -> None:
    """Projecting a bar's top-face centre and picking it must return the
    same bar, whatever fold quadrant the azimuth lands in."""

    heights = np.full((6, 9), 0.5)
    colors = np.ones((6, 9, 3), dtype=np.float32) * 0.5
    camera = HeightBarCamera(azimuth_deg=azimuth, elevation_deg=30.0)
    _frame, scene = render_height_bars(
        heights, colors, camera=camera, value_limits=(0.0, 1.0),
        width=320, height=240,
    )
    for row, column in ((0, 0), (2, 5), (5, 8), (3, 3)):
        a, b = scene.fold_cell(row, column)
        x, y = scene.project(a + 0.5, b + 0.5, 0.5)
        picked = scene.pick(x, y)
        assert picked == (row, column), (azimuth, row, column, picked)


def test_bars_clip_to_the_value_limits() -> None:
    """A value beyond the colour limits saturates in HEIGHT exactly as it
    saturates in colour: the z axis and the colorbar are one scale."""

    heights = np.asarray([[0.5, 2.0]])
    colors = np.ones((1, 2, 3), dtype=np.float32) * 0.5
    camera = HeightBarCamera()
    _frame, scene = render_height_bars(
        heights, colors, camera=camera, value_limits=(0.0, 1.0),
        width=320, height=240,
    )
    a0, b0 = scene.fold_cell(0, 0)
    a1, b1 = scene.fold_cell(0, 1)
    top_full = scene.project(a1 + 0.5, b1 + 0.5, 1.0)
    picked = scene.pick(*top_full)
    assert picked == (0, 1)


def test_absent_bars_leave_the_floor() -> None:
    # Near-flat neighbours, so the hole shows FLOOR rather than the side
    # face of the bar behind it (which a deep hole correctly reveals).
    heights = np.full((4, 4), 0.02)
    heights[1, 2] = np.nan
    colors = np.ones((4, 4, 3), dtype=np.float32) * 0.5
    frame, scene = render_height_bars(
        heights, colors, camera=HeightBarCamera(), value_limits=(0.0, 1.0),
        width=320, height=240,
    )
    a, b = scene.fold_cell(1, 2)
    x, y = scene.project(a + 0.5, b + 0.5, 0.0)
    assert scene.pick(x, y) is None


def test_dense_grids_pool_to_display_resolution() -> None:
    rng = np.random.default_rng(0)
    heights = rng.random((400, 800))
    colors = np.ones((400, 800, 3), dtype=np.float32) * 0.5
    _frame, scene = render_height_bars(
        heights, colors, camera=HeightBarCamera(), value_limits=(0.0, 1.0),
        width=300, height=220,
    )
    assert scene.pool_x > 1 and scene.pool_y > 1
    assert scene.nx <= 300 and scene.ny <= 300
    # Picks still speak SOURCE indices (of the pooled block's origin).
    a, b = scene.fold_cell(100, 200)
    x, y = scene.project(a + 0.5, b + 0.5, 0.5)
    picked = scene.pick(x, y)
    assert picked is not None
    assert picked[0] % scene.pool_y == 0 and picked[1] % scene.pool_x == 0


def test_stress_grid_renders_inside_the_guard() -> None:
    rng = np.random.default_rng(1)
    heights = rng.random((96, 128))
    colors = np.ones((96, 128, 3), dtype=np.float32) * 0.5
    start = perf_counter()
    render_height_bars(
        heights, colors, camera=HeightBarCamera(), value_limits=(0.0, 1.0),
        width=600, height=440,
    )
    assert perf_counter() - start < MAX_STRESS_RENDER_SECONDS


def test_camera_clamps_its_angles() -> None:
    camera = HeightBarCamera(azimuth_deg=10.0, elevation_deg=89.0, zoom=99.0)
    assert camera.elevation_deg == 80.0
    assert camera.zoom == 6.0


# --------------------------------------------------------------- session
def test_presentation_roundtrip_is_bit_identical_and_keeps_selectors() -> None:
    session = _session()
    try:
        session.set_area_selector(
            NumericRange(2.0, 6.0), NumericRange(2.0, 6.0)
        )
        heatmap_before = session.rgba().copy()
        session.set_parameter("presentation", "height_bars")
        bars = session.rgba()
        assert np.abs(
            bars.astype(int) - heatmap_before.astype(int)
        ).max() > 0, "the scene must actually change the pixels"
        assert [s.kind for s in session.selectors] == [SelectorKind.AREA]
        session.set_parameter("presentation", "heatmap")
        heatmap_after = session.rgba()
        np.testing.assert_array_equal(heatmap_after, heatmap_before)
        assert [s.kind for s in session.selectors] == [SelectorKind.AREA]
    finally:
        session.close()


def test_camera_parameters_are_display_state_not_projection() -> None:
    session = _session()
    try:
        session.set_parameter("presentation", "height_bars")
        revision_before = session.data_revision
        first = session.rgba().copy()
        session.set_parameter("camera_azimuth", -20.0)
        second = session.rgba()
        assert session.data_revision == revision_before
        assert np.abs(second.astype(int) - first.astype(int)).max() > 0
    finally:
        session.close()


def test_live_revision_rerenders_the_scene() -> None:
    session = _session()
    try:
        session.set_parameter("presentation", "height_bars")
        first = session.rgba().copy()
        session.update_data(_scan_snapshot(revision=2, seed=11))
        second = session.rgba()
        assert np.abs(second.astype(int) - first.astype(int)).max() > 0
    finally:
        session.close()


def _pointer(session, action, axis, fx, fy, **kwargs):
    left, top, right, bottom = axis.bounds
    return session._raster_pointer_event(
        action,
        left + fx * (right - left),
        top + fy * (bottom - top),
        axes_snapshot=axis,
        **kwargs,
    )


def test_orbit_drag_commits_camera_and_creates_no_selector() -> None:
    session = _session()
    try:
        session.set_parameter("presentation", "height_bars")
        session.rgba()
        axis = next(
            t for t in session._raster_axes_snapshot() if t.role == "image"
        )
        _pointer(session, "press", axis, 0.5, 0.5, button=1)
        _pointer(session, "move", axis, 0.7, 0.6, button=1)
        _pointer(session, "release", axis, 0.7, 0.6, button=1)
        state = session.display_state
        assert float(state["camera_azimuth"]) != -55.0
        assert session.selectors == ()
    finally:
        session.close()


def test_click_picks_the_bar_as_a_crosshair() -> None:
    session = _session()
    try:
        session.set_parameter("presentation", "height_bars")
        session.rgba()
        renderer = session._renderer
        scene = renderer._height_bars_scene_map
        # Aim at the TALLEST bar's top-face centre: the peak cannot be
        # occluded by anything nearer, so the pick must return it.
        payload = session._payload
        z_grid = np.asarray(payload.z.canonical)
        row, column = np.unravel_index(np.nanargmax(z_grid), z_grid.shape)
        row, column = int(row), int(column)
        a, b = scene.fold_cell(row, column)
        px, py = scene.project(
            a + 0.5, b + 0.5, float(z_grid[row, column])
        )
        box = renderer.primary_axes.bbox
        canvas_x = float(box.x0) + px
        canvas_y = float(box.y1) - py
        width, height_px = renderer.figure.canvas.get_width_height(
            physical=True
        )
        image_axis = next(
            t for t in session._raster_axes_snapshot() if t.role == "image"
        )
        session._raster_pointer_event(
            "press",
            canvas_x / width,
            1.0 - canvas_y / height_px,
            button=1,
            axes_snapshot=image_axis,
        )
        session._raster_pointer_event(
            "release",
            canvas_x / width,
            1.0 - canvas_y / height_px,
            button=1,
            axes_snapshot=image_axis,
        )
        kinds = [s.kind for s in session.selectors]
        assert kinds == [SelectorKind.CROSSHAIR], kinds
        crosshair = session.selectors[0]
        cell = renderer._height_bars_cell_of(
            float(crosshair.value.x), float(crosshair.value.y)
        )
        assert cell == (row, column)
    finally:
        session.close()


def test_facet_overview_stays_heatmap_and_focus_honours_the_scene() -> None:
    rows = 100
    cells = [(i % 10, i // 10) for i in range(rows)]
    schema = DatasetSchema.create(
        Axis.create("repeat", size=4),
        PointTable.from_columns({
            "ax": np.asarray([float(c[0]) for c in cells]),
            "ay": np.asarray([float(c[1]) for c in cells]),
        }),
        data_axes=(Axis.create("site", values=[0.0, 1.0, 2.0]),),
        dtype=np.float64,
        point_topology=PointTopology(
            (AxisId("ax"), AxisId("ay")),
            (tuple(float(i) for i in range(10)),) * 2,
            tuple(cells),
        ),
    )
    rng = np.random.default_rng(0)
    session = PlotSession(
        DatasetSnapshot(schema, rng.random((4, rows, 3)), revision=1),
        FacetGridPlot(
            AxisRef.data("site"),
            ImagePlot(
                AxisRef.point_dimension("ax"), AxisRef.point_dimension("ay")
            ),
        ),
    )
    session.set_size("2x2")
    try:
        overview = session.rgba().copy()
        session.set_parameter("presentation", "height_bars")
        np.testing.assert_array_equal(session.rgba(), overview)
        session.focus_facet(1)
        focused = session.rgba()
        assert np.abs(focused.astype(int) - overview.astype(int)).max() > 0
    finally:
        session.close()


def test_occlusion_is_a_box_test_not_a_centre_depth_proxy() -> None:
    """The exact-occlusion semantics the outline sampler must keep.

    A LOWER neighbour can never hide a higher edge (the centre-depth
    proxy carved dashes there); an equal-height shared rim stays whole;
    and a floor-level line under a plate is covered by it.
    """

    from zlc_plot.rendering import MatplotlibRenderer

    heights = np.asarray([[0.02, 0.03]])
    colors = np.tile(
        np.asarray([0.5, 0.7, 0.9], dtype=np.float32), (1, 2, 1)
    )
    frame, scene = render_height_bars(
        heights, colors, camera=HeightBarCamera(), value_limits=(0.0, 1.0),
        width=400, height=300, bar_edges=False,
    )
    sampler = MatplotlibRenderer._height_bars_occluded_polyline

    def visible_fraction(edge):
        xs, _ = sampler(None, scene, np.asarray([edge], dtype=np.float64))
        finite = np.isfinite(xs[:-1])
        return finite.mean()

    a0, b0 = scene.fold_cell(0, 0)   # the 0.02 plate
    a1, b1 = scene.fold_cell(0, 1)   # the 0.03 plate
    shared_a = max(a0, a1)           # the plane between them

    # The higher plate's top edge along the shared plane: its LOWER
    # neighbour must not eat it.
    high_edge = ((shared_a, b1, 0.03), (shared_a, b1 + 1.0, 0.03))
    assert visible_fraction(high_edge) == 1.0

    # A floor-level segment across the 0.02 plate's footprint is covered.
    across = ((a0 + 0.2, b0 + 0.5, 0.0), (a0 + 0.8, b0 + 0.5, 0.0))
    assert visible_fraction(across) < 0.2

    # An equal-height shared rim stays whole.
    frame, scene = render_height_bars(
        np.asarray([[0.4, 0.4]]), colors,
        camera=HeightBarCamera(), value_limits=(0.0, 1.0),
        width=400, height=300, bar_edges=False,
    )
    a0, b0 = scene.fold_cell(0, 0)
    a1, b1 = scene.fold_cell(0, 1)
    rim = ((max(a0, a1), b0, 0.4), (max(a0, a1), b0 + 1.0, 0.4))
    assert visible_fraction(rim) == 1.0


def test_pane_grid_leaves_the_right_wall_open() -> None:
    """The reference (MATLAB) pane-grid construction: every rule sits at
    a TICK position and runs the full display limits.  The limits carry
    no tick, so horizontals reach the pane border but NO VERTICAL rule
    ever lands on it -- that absence IS the open boundary."""

    side = 4
    rows = side * side
    cells = [(i % side, i // side) for i in range(rows)]
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({
            "ax": np.asarray([float(c[0]) for c in cells]),
            "ay": np.asarray([float(c[1]) for c in cells]),
        }),
        data_axes=(Axis.create("site", values=[0.0]),),
        dtype=np.float64,
        point_topology=PointTopology(
            (AxisId("ax"), AxisId("ay")),
            (tuple(float(i) for i in range(side)),) * 2,
            tuple(cells),
        ),
    )
    # Deterministically tall bars: every floor cell is covered, so the
    # only visible grid geometry is the pane grid under test.
    session = PlotSession(
        DatasetSnapshot(schema, np.full((1, rows, 1), 0.5), revision=1),
        ImagePlot(AxisRef.point_dimension("ax"), AxisRef.point_dimension("ay")),
    )
    session.set_size("2x2")
    try:
        session.set_parameter("presentation", "height_bars")
        session.rgba()
        renderer = session._renderer
        scene = renderer._height_bars_scene_map
        chrome = renderer._artists["image:h3d_chrome"]
        xs, ys = chrome["grid"].get_data()
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        assert np.isfinite(xs).any(), "the scene must draw a pane grid"
        border_x = scene.project(
            float(scene.nx), float(scene.ny), 0.0
        )[0] / scene.width
        # Horizontal rules run the FULL limits: some grid geometry must
        # reach the pane border...
        near_border = np.isfinite(xs) & (
            np.abs(xs - border_x) < 3.0 / scene.width
        )
        assert near_border.any()
        # ...but no VERTICAL rule may land there: split the polyline at
        # NaNs and demand every constant-x (vertical) run keeps its
        # distance from the border.
        breaks = np.nonzero(~np.isfinite(xs))[0]
        start = 0
        for stop in list(breaks) + [len(xs)]:
            run_x = xs[start:stop]
            start = stop + 1
            run_x = run_x[np.isfinite(run_x)]
            if run_x.size < 4:
                continue
            if run_x.max() - run_x.min() < 1.0 / scene.width:
                assert abs(float(run_x.mean()) - border_x) > 10.0 / scene.width
    finally:
        session.close()


def test_render_is_deterministic_at_exact_crossing_ties() -> None:
    """The crossing merge must be a permutation BY CONSTRUCTION.

    At azimuth 45 the two crossing ladders tie exactly; a second float
    formula for the b positions could collide with the a positions and
    leave a slot holding uninitialized memory -- frames then differed
    run to run.  Fresh allocations between renders shake the heap so a
    regression cannot hide behind recycled garbage.
    """

    rng = np.random.default_rng(11)
    heights = rng.uniform(0.0, 1.0, size=(12, 12))
    colors = np.tile(
        np.asarray([0.4, 0.6, 0.9], dtype=np.float32), (12, 12, 1)
    )
    camera = HeightBarCamera(azimuth_deg=-45.0)
    frames = []
    for _ in range(4):
        noise = [
            np.random.default_rng(i).random(size)
            for i, size in enumerate((317, 4093, 65537))
        ]
        frame, _ = render_height_bars(
            heights.copy(), colors.copy(), camera=camera,
            value_limits=(0.0, 1.0), width=360, height=280,
            supersample=2, bar_edges=False,
        )
        frames.append(frame)
        del noise
    for frame in frames[1:]:
        np.testing.assert_array_equal(frames[0], frame)


def test_scanline_engine_matches_the_reference_bit_for_bit() -> None:
    """The numba scanline engine is a SPEED path, never a semantics path.

    The numpy kernel is the specification; the compiled engine must
    reproduce it bit for bit -- frame and pick map both -- across folds,
    exact crossing ties (azimuth 45), clipping limits, NaN holes,
    hanging negative bars and the pooled dense surface.
    """

    pytest.importorskip("numba")
    from zlc_plot import _height3d_raster as raster

    rng = np.random.default_rng(9)
    xx, yy = np.meshgrid(np.arange(16), np.arange(16))
    gauss = np.exp(-((xx - 8.0) ** 2 + (yy - 7.0) ** 2) / 18.0)
    holes = gauss.copy()
    holes[3, 4] = np.nan
    holes[9:11, 12] = np.nan
    cases = (
        (gauss, (0.0, 1.0), dict(azimuth_deg=-55.0), 3, 360, 300),
        (gauss, (0.0, 1.0), dict(azimuth_deg=-45.0), 2, 320, 260),
        (gauss, (0.2, 0.8), dict(azimuth_deg=130.0, elevation_deg=12.0),
         2, 300, 240),
        (holes, (0.0, 1.0), dict(azimuth_deg=220.0, elevation_deg=70.0),
         2, 280, 280),
        (rng.normal(scale=0.5, size=(8, 8)), (-1.0, 1.0),
         dict(azimuth_deg=40.0), 2, 300, 260),
        (rng.random((300, 500)), (0.0, 1.0), dict(azimuth_deg=-55.0),
         2, 240, 200),
        (0.2 + 0.8 * rng.random((32, 32)), (0.0, 1.0),
         dict(azimuth_deg=220.0, elevation_deg=60.0), 1, 300, 240),
    )
    previous = raster._ENGINE
    try:
        for heights, limits, camera_kwargs, ss, width, height in cases:
            colors = np.ascontiguousarray(
                np.stack([
                    np.clip(np.nan_to_num(heights), 0.0, 1.0),
                    np.full(heights.shape, 0.4),
                    1.0 - np.clip(np.nan_to_num(heights), 0.0, 1.0),
                ], axis=-1).astype(np.float32)
            )
            frames = {}
            for engine in ("numpy", "numba"):
                raster._ENGINE = engine
                frame, scene = render_height_bars(
                    heights, colors,
                    camera=HeightBarCamera(**camera_kwargs),
                    value_limits=limits, width=width, height=height,
                    supersample=ss, zero_rgb=(0.9, 0.9, 1.0),
                )
                frames[engine] = (frame, scene.id_plane)
            np.testing.assert_array_equal(
                frames["numpy"][0], frames["numba"][0]
            )
            np.testing.assert_array_equal(
                frames["numpy"][1], frames["numba"][1]
            )
    finally:
        raster._ENGINE = previous


def test_composed_camera_frames_equal_a_full_draw() -> None:
    """The compose fast lane must be invisible: after any run of camera
    commits (buffer reused, background preserved, chrome repainted as
    dynamic artists), the published pixels equal a forced full draw."""

    session = _session(6)
    try:
        session.set_parameter("presentation", "height_bars")
        session.rgba()
        # two commits WITHOUT reading frames in between: the second
        # compose must not lean on a full draw having wiped the stale
        # background between them.
        session.set_parameter("camera_azimuth", -40.0)
        session.set_parameter("camera_elevation", 44.0)
        composed = session.rgba().copy()
        renderer = session._renderer
        renderer.draw()
        full = renderer._rgba_buffer()
        np.testing.assert_array_equal(composed, full)
    finally:
        session.close()
