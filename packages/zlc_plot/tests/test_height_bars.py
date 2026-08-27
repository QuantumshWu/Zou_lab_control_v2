"""The Image kind's height-bar presentation: kernel, session and gestures.

The scene must be a PRESENTATION of the same image surface: identical
snapshot, payload, clim, colormap and pipeline -- so the contract here is
mostly about what must NOT change: a heatmap->bars->heatmap roundtrip is
bit-identical, committed selectors survive the trip, and the camera is
display state that never touches the projection.
"""

from __future__ import annotations

from time import perf_counter, sleep

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
from zlc_plot.specs import RenderEffect
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


def test_the_scene_is_an_oblique_view_of_the_same_heatmap() -> None:
    """Tipped nearly flat, the 3D grid must READ as the heatmap.

    The scene showed the array's own row order regardless of the origin
    the heatmap draws under, so the picture was mirrored top for bottom:
    the cell at the heatmap's bottom right stood at the scene's far
    corner, directly above the near one.
    """

    from zlc_plot.config import DEFAULTS

    heights = np.zeros((4, 5))
    colors = np.full((4, 5, 3), 0.5, dtype=np.float32)
    # azimuth 0, elevation at its ceiling: as close to looking straight
    # down at the picture as the camera goes.
    camera = HeightBarCamera(azimuth_deg=0.0, elevation_deg=80.0)

    def centres(origin):
        _frame, scene = render_height_bars(
            heights, colors, camera=camera, value_limits=(0.0, 1.0),
            width=320, height=240, origin=origin,
        )

        def centre(row, column):
            a, b = scene.fold_cell(row, column)
            return scene.project(a + 0.5, b + 0.5, 0.0)

        return centre

    # The panel draws its heatmap with THIS origin, and the scene follows
    # it: row 0 is the top of the picture, so it is farther up the screen
    # (scene pixel y grows downward), and column 0 is on the left.
    origin = DEFAULTS.style.render.image_origin
    assert origin == "upper"
    centre = centres(origin)
    assert centre(0, 0)[1] < centre(3, 0)[1]
    assert centre(0, 0)[0] < centre(0, 4)[0]
    # The kernel's own default keeps the array's order, which is the
    # opposite picture -- the parameter is what carries the fact.
    plain = centres("lower")
    assert plain(0, 0)[1] > plain(3, 0)[1]


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


def test_the_scene_turns_as_one_rigid_object() -> None:
    """A wall stands on the picture's own edge, whichever way it faces.

    The rasterizer folds the grid so it only ever walks one octant, and
    the chrome used to hang its wall rules, its vertical axis and its base
    labels on FOLDED sides.  A fold is a fact about the walk, not about
    the data, so crossing one jumped every piece of chrome a quarter turn
    around a picture that had not moved: the folded corner (0, 0) projects
    to the top of the frame at azimuth -0.5 degrees and to the bottom at
    +0.5, while the four projected SOURCE corners move by less than one
    and a half pixels.

    So the test is on the data: the walls stand on the heatmap's ab and
    ad edges, the axes run along cd and bc, and the z axis rises at d --
    and each must move continuously with the camera.
    """

    from zlc_plot._height3d_raster import HeightBarCamera, render_height_bars
    from zlc_plot.rendering import MatplotlibRenderer

    rng = np.random.default_rng(3)
    heights = rng.random((9, 13)) * 100.0
    colors = np.repeat(
        (heights / 100.0)[..., None].astype(np.float32), 3, axis=-1
    )

    def anchors(azimuth):
        _frame, scene = render_height_bars(
            heights, colors,
            camera=HeightBarCamera(azimuth_deg=azimuth, elevation_deg=30.0),
            value_limits=(0.0, 100.0), width=320, height=240,
            supersample=1, rim_width_px=3.3)
        anchor = MatplotlibRenderer._height_bars_ground_anchors(scene)
        return scene, anchor

    def screen(azimuth):
        scene, anchor = anchors(azimuth)
        return {
            "axis corner": scene.project(
                anchor["axis_a"], anchor["axis_b"], 0.0),
            "wall corner": scene.project(
                anchor["wall_a"], anchor["wall_b"], 0.0),
        }

    # Either side of a fold boundary the camera has moved one degree; no
    # anchor may move further than the picture does.
    for boundary in (0.0, 90.0, 180.0, -90.0):
        before = screen(boundary - 0.5)
        after = screen(boundary + 0.5)
        for name in before:
            moved = max(abs(before[name][index] - after[name][index])
                        for index in (0, 1))
            assert moved < 4.0, (
                "the %s jumped %.1f px across the fold at %g degrees"
                % (name, moved, boundary)
            )

    # And the anchors really are the picture's own corners: a carries the
    # walls, c the axes and d the z axis, at every camera.
    for azimuth in (-55.0, 5.0, 95.0, 185.0, 275.0):
        scene, anchor = anchors(azimuth)
        top_row = 0 if scene.flip_rows else scene.source_ny - 1
        bottom_row = scene.source_ny - 1 - top_row
        for corner, prefix in (
            ((top_row, 0), "wall"),                          # a
            ((bottom_row, scene.source_nx - 1), "axis"),     # c
            ((bottom_row, 0), "z"),                          # d
        ):
            cell_a, cell_b = scene.fold_cell(*corner)
            assert abs(cell_a - anchor[prefix + "_a"]) <= 1.0
            assert abs(cell_b - anchor[prefix + "_b"]) <= 1.0


def test_a_drag_renders_the_resolution_it_leaves_behind() -> None:
    """One resolution, all the way through the gesture.

    A drag used to render at half resolution and the release repaint at
    full, so the picture changed character under the hand and changed back
    when it let go.  That crutch was worth having while every rim was
    vector chrome priced per bar; once the rims became pixels it bought
    11 ms of a 93 ms move -- measured on the console's own panel -- and a
    second look is not worth 11 ms.
    """

    session = _session(16)
    try:
        session.set_parameter("presentation", "height_bars")
        session.rgba()
        renderer = session._renderer
        committed = renderer._height_bars_scene_map.width
        axis = next(
            t for t in session._raster_axes_snapshot() if t.role == "image"
        )
        _pointer(session, "press", axis, 0.5, 0.5, button=2)
        for step in range(6):
            _pointer(
                session, "move", axis,
                0.5 + 0.02 * (step + 1), 0.5 + 0.01 * (step + 1),
                button=2,
            )
            assert renderer._height_bars_scene_map.width == committed, (
                "a drag frame is drawn at another resolution"
            )
        _pointer(session, "release", axis, 0.62, 0.56, button=2)
        assert renderer._height_bars_scene_map.width == committed
    finally:
        session.close()


def test_middle_drag_orbits_and_left_drag_is_inert() -> None:
    """The scene keeps the 2D button grammar: MIDDLE navigates the view.

    A middle drag commits the camera orbit in one display revision; a
    left drag says nothing -- no camera change, no selector -- because
    the left button speaks data (a click picks) and the 2D selector
    gestures wait for the heatmap to return.
    """

    session = _session()
    try:
        session.set_parameter("presentation", "height_bars")
        session.rgba()
        axis = next(
            t for t in session._raster_axes_snapshot() if t.role == "image"
        )
        _pointer(session, "press", axis, 0.5, 0.5, button=2)
        _pointer(session, "move", axis, 0.7, 0.6, button=2)
        _pointer(session, "release", axis, 0.7, 0.6, button=2)
        state = session.display_state
        assert float(state["camera_azimuth"]) != HeightBarCamera().azimuth_deg
        assert session.selectors == ()
        committed = float(state["camera_azimuth"])
        _pointer(session, "press", axis, 0.5, 0.5, button=1)
        _pointer(session, "move", axis, 0.2, 0.3, button=1)
        _pointer(session, "release", axis, 0.2, 0.3, button=1)
        state = session.display_state
        assert float(state["camera_azimuth"]) == committed
        assert session.selectors == ()
    finally:
        session.close()


def test_a_hand_still_holding_the_scene_already_owns_the_view() -> None:
    """The camera has ONE owner, and the moving hand writes to it.

    A drag that keeps its turn to itself until the button comes up is a
    view that nothing else can see: a live generation arriving mid-drag
    mounts a replacement surface from the panel's record, and the record
    still held the pre-drag camera -- so an operator who had not let go
    yet watched the scene home itself.  Every frame of the drag is the
    committed view; only the render RESOLUTION is transient.
    """

    session = _session()
    try:
        session.set_parameter("presentation", "height_bars")
        session.rgba()
        axis = next(
            t for t in session._raster_axes_snapshot() if t.role == "image"
        )
        start = float(session.display_state["camera_azimuth"])
        _pointer(session, "press", axis, 0.5, 0.5, button=2)
        _pointer(session, "move", axis, 0.72, 0.34, button=2)
        turned = float(session.display_state["camera_azimuth"])
        assert turned != start, "the hand's turn is not the panel's view"
        shown = session._renderer.height_bars_camera
        assert shown is not None and shown.azimuth_deg == turned
        assert session._renderer.height_bars_dragging, (
            "a drag in flight still holds the frame around the scene"
        )
        # Whatever takes the pointer away -- a replacement surface, a
        # layout rebuild, a window losing focus -- ends the DRAG.  It
        # does not get to move the view back.
        session.cancel_interaction()
        assert float(session.display_state["camera_azimuth"]) == turned
        assert not session._renderer.height_bars_dragging
    finally:
        session.close()


def test_orbit_holds_the_camera_distance_still() -> None:
    """The scale is azimuth-invariant: an orbit must not breathe.

    A fit against the current projected footprint widens and narrows
    with the azimuth, which read as the scene lurching nearer and
    farther during a drag.  The scale fits the invariant envelope
    instead, so every azimuth at one elevation shares one scale.
    """

    from zlc_plot._height3d_raster import HeightBarCamera, render_height_bars

    rng = np.random.default_rng(21)
    heights = rng.random((24, 32))
    colors = np.repeat(
        heights[..., None].astype(np.float32).clip(0.0, 1.0), 3, axis=-1
    )
    scales = []
    for azimuth in range(-175, 185, 10):
        _, scene = render_height_bars(
            heights, colors,
            camera=HeightBarCamera(azimuth_deg=float(azimuth)),
            value_limits=(0.0, 1.0), width=420, height=320,
            supersample=1, zero_rgb=(1.0, 1.0, 1.0),
        )
        scales.append(scene.scale)
    assert len(set(scales)) == 1, scales


def test_color_limit_preview_rerenders_the_scene() -> None:
    """A clim drag previews the 3D scene itself, never a stale heatmap.

    The 2D preview fast path re-gathers heatmap pixels into the shared
    artist; in scene mode that painted an old heatmap over the boxes
    (or, with no cached planes, changed nothing until release).  The
    preview must re-render the scene with the candidate limits, and the
    preview frame must equal the frame those limits commit to.
    """

    session = _session()
    try:
        session.set_parameters({
            "presentation": "height_bars",
            "color_min": 0.0,
            "color_max": 1.0,
        })
        session.rgba()
        renderer = session._renderer
        image = renderer._active_image_artist()
        before = np.array(image.get_array())
        renderer.preview_color_limits(0.0, 0.4)
        preview = np.array(renderer._active_image_artist().get_array())
        assert not np.array_equal(before, preview)
        session.set_parameter("color_max", 0.4)
        session.rgba()
        committed = np.array(renderer._active_image_artist().get_array())
        np.testing.assert_array_equal(preview, committed)
    finally:
        session.close()


def test_the_artist_is_handed_the_runs_and_nothing_else() -> None:
    """Hidden samples are dropped, and the picture does not change.

    Sampling answers per sample, so a hidden stretch arrives as one
    blank per sample -- more than half of every edge on a crowded scene
    -- and a blank is still a vertex the renderer walks in order to
    draw nothing.  A hole needs exactly one gap to be a hole.  What
    must survive is every pixel: same runs, same order, same line.
    """

    from zlc_plot.rendering import MatplotlibRenderer

    session = _session(10)
    try:
        session.set_parameter("presentation", "height_bars")
        session.rgba()
        renderer = session._renderer
        scene = renderer._height_bars_scene_map
        cells = [(row, column) for row in range(4) for column in range(4)]
        edges = renderer._height_bars_box_edges(
            scene, cells,
            np.zeros(len(cells)), np.full(len(cells), 0.6),
        )
        outline = renderer._artists["image:h3d_chrome"]["grid"]

        raw = renderer._height_bars_sampled_polyline(scene, edges, 64)
        compact = MatplotlibRenderer._height_bars_visible_runs(*raw)
        blank = np.isnan(raw[0])
        assert blank.sum() > np.isfinite(raw[0]).sum() * 0.2, (
            "this scene must actually hide something"
        )
        assert compact[0].size < raw[0].size
        doubled = np.isnan(compact[0][1:]) & np.isnan(compact[0][:-1])
        assert not doubled.any(), "a hole was left wider than one gap"

        def painted(xs, ys):
            outline.set_data(xs, ys)
            renderer._composed_generation = -1
            return renderer.rgba().copy()

        assert np.array_equal(painted(*compact), painted(*raw)), (
            "dropping the blanks moved a pixel"
        )
    finally:
        session.close()


def test_a_visible_run_is_handed_over_as_its_two_ends() -> None:
    """The compaction itself, on a line whose holes are known.

    Every sample of one run lies on the line through its ends -- the
    edge is straight and the projection is affine -- so the points
    between them say nothing the ends do not.
    """

    from zlc_plot.rendering import MatplotlibRenderer

    xs = np.asarray([np.nan, 1.0, 2.0, 3.0, np.nan, np.nan, 7.0, np.nan])
    ys = np.asarray([np.nan, 4.0, 5.0, 6.0, np.nan, np.nan, 8.0, np.nan])
    out_x, out_y = MatplotlibRenderer._height_bars_visible_runs(xs, ys)
    # run one: 1 -> 3.  run two: the lone 7, a segment of no length,
    # which is what a one-sample run has always drawn.
    assert out_x.tolist()[:2] == [1.0, 3.0]
    assert out_y.tolist()[:2] == [4.0, 6.0]
    assert np.isnan(out_x[2])
    assert out_x.tolist()[3:5] == [7.0, 7.0]
    assert np.isnan(out_x[5])
    assert out_x.size == 6
    # Nothing visible: nothing to draw.
    blank = np.full(4, np.nan)
    empty_x, empty_y = MatplotlibRenderer._height_bars_visible_runs(blank, blank)
    assert empty_x.size == 0 and empty_y.size == 0


def test_one_mechanism_draws_every_edge_the_scene_has() -> None:
    """Edges are vector chrome, and that is the only kind there is.

    A second, raster mechanism darkened box boundaries inside the
    scanline kernel for the grids the vector path declined -- two looks
    for one fact, and they swapped mid-gesture, because the vector path
    was gated on the drag's reduced resolution too.  So: the same scene
    turned under the hand changed the character of its own lines.

    What the kernel paints is therefore FACES ONLY.  Every line -- bar
    outlines, pane grids, floor rules, the cage -- is an artist.
    """

    from zlc_plot._height3d_raster import HeightBarCamera, render_height_bars

    rng = np.random.default_rng(11)
    heights = rng.random((12, 12))
    colors = np.repeat(
        heights[..., None].astype(np.float32).clip(0.0, 1.0), 3, axis=-1
    )
    frame, scene = render_height_bars(
        heights, colors, camera=HeightBarCamera(), value_limits=(0.0, 1.0),
        width=240, height=200, zero_rgb=(1.0, 1.0, 1.0),
    )
    # Every drawn pixel is a face colour or the background: a darkened
    # boundary would show up as a value darker than the darkest face.
    # The frame is opaque by construction -- it is already composited over
    # its background -- so "drawn" is "differs from that background", which
    # is the same set the coverage used to name.
    background = np.array([255, 255, 255], dtype=np.uint8)
    drawn = (frame[..., :3] != background).any(axis=-1)
    painted = frame[..., :3][drawn].astype(np.float64) / 255.0
    darkest_face = float(np.min(np.clip(heights, 0.0, 1.0)))
    assert painted.min() >= darkest_face - 0.02, (
        "the kernel darkened something that is not a face"
    )


def test_a_crease_is_stroked_once_and_only_where_the_faces_change() -> None:
    """The rims belong to the picture, so the picture draws them.

    Coverage is the STRONGEST stamp reaching a pixel, never a sum, so two
    creases meeting at a corner cannot darken it twice -- the darkest rim
    pixel is exactly the rim colour.  And a stroked pixel must sit within
    a stamp radius of a place where the visible face changes: anywhere
    else is the kernel darkening something that is not a crease.
    """

    from zlc_plot._height3d_raster import _rim_boundary, _rim_stamp

    rng = np.random.default_rng(5)
    heights = np.round(rng.random((9, 11)) * 3.0) / 3.0
    heights[2, 3] = np.nan
    colors = np.repeat(
        np.nan_to_num(heights)[..., None].astype(np.float32), 3, axis=-1
    )
    common = dict(
        camera=HeightBarCamera(), value_limits=(0.0, 1.0),
        width=360, height=280, zero_rgb=(1.0, 1.0, 1.0),
    )
    plain, scene = render_height_bars(heights, colors, **common)
    rimmed, _scene = render_height_bars(
        heights, colors, rim_rgb=(0.0, 0.0, 0.0), rim_width_px=3.3, **common
    )
    changed = np.any(plain != rimmed, axis=-1)
    assert changed.any(), "no rim was drawn at all"
    darkest = rimmed[..., :3][changed].min()
    assert darkest == 0, "a crease was darkened past its own colour"

    _weights, radius = _rim_stamp(3.3)
    boundary = _rim_boundary(scene.id_plane)
    reach = np.zeros_like(boundary)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            rolled = np.roll(np.roll(boundary, dy, axis=0), dx, axis=1)
            reach |= rolled
    assert not (changed & ~reach).any(), (
        "the kernel darkened a pixel that is not beside a crease"
    )


def test_a_drag_draws_the_same_rims_as_the_frame_it_leaves() -> None:
    """Turning the scene must not change what its lines ARE.

    The drag holds the frame around the scene so the colorbar and the
    rail beside it are not repainted per move.  Holding pixels still is
    not permission to draw a different picture: the rims the hand turns
    are the rims it lets go of.
    """

    from zlc_plot._height3d_raster import _rim_stamp

    session = _session(12)
    try:
        session.set_parameter("presentation", "height_bars")
        committed = session.rgba().copy()
        renderer = session._renderer
        axis = next(
            t for t in session._raster_axes_snapshot() if t.role == "image"
        )
        _pointer(session, "press", axis, 0.5, 0.5, button=2)
        _pointer(session, "move", axis, 0.56, 0.52, button=2)
        assert renderer.height_bars_dragging
        preview = session.rgba().copy()
        _pointer(session, "release", axis, 0.56, 0.52, button=2)
        settled = session.rgba().copy()
        assert preview.shape == committed.shape == settled.shape

        def rim_share(frame):
            dark = frame[..., :3].max(axis=-1) < 64
            return float(dark.mean())

        # Same picture, same amount of line in it: a preview that dropped
        # its rims, or drew them at the raster's own width, would not be
        # within a factor of two of the frame it becomes.
        assert rim_share(preview) > 0.0
        assert 0.5 < rim_share(preview) / max(rim_share(settled), 1e-9) < 2.0
        _weights, radius = _rim_stamp(3.3)
        assert radius >= 1
    finally:
        session.close()


def test_the_bar_count_is_the_data_and_nothing_else() -> None:
    """As many boxes as the heatmap has cells, at every camera and size.

    The scene is the heatmap in another form, so what divides the ground
    is what divides the picture.  A level of detail here answered a
    question the data already answers, and it answered it invisibly: an
    ROI shrinking from 849 columns to 200 kept drawing the same hundred
    bars under a range that moved, so the operator saw the extent change
    and the structure stand still.
    """

    from zlc_plot._height3d_raster import HeightBarCamera, render_height_bars

    rng = np.random.default_rng(33)

    def drawn(rows, columns, width, height, taps=3, zoom=1.0, rim=3.3):
        heights = rng.random((rows, columns))
        colors = np.repeat(
            heights[..., None].astype(np.float32).clip(0.0, 1.0), 3, axis=-1
        )
        _frame, scene = render_height_bars(
            heights, colors,
            camera=HeightBarCamera(azimuth_deg=-55.0, zoom=zoom),
            value_limits=(0.0, 1.0), width=width, height=height,
            supersample=taps, zero_rgb=(1.0, 1.0, 1.0),
            rim_rgb=(0.0, 0.0, 0.0), rim_width_px=rim,
        )
        # Odd fold quadrants swap the folded axes; the pair is the count.
        return tuple(sorted((scene.nx, scene.ny)))

    assert drawn(300, 400, 320, 240) == (300, 400)
    # Neither the camera, the taps, the line width nor the panel size is
    # allowed a say in it.
    assert drawn(300, 400, 320, 240, zoom=2.0) == (300, 400)
    assert drawn(300, 400, 320, 240, rim=9.9) == (300, 400)
    assert drawn(300, 400, 120, 90) == (300, 400)
    assert drawn(300, 400, 1200, 900, taps=1) == (300, 400)
    # And an ROI that shrinks draws fewer bars, which is the whole point.
    assert drawn(75, 100, 320, 240) == (75, 100)


def test_drag_preview_keeps_the_chrome_typography_in_place() -> None:
    """The half-resolution drag preview must not move the scene chrome.

    Tick lengths and label gaps are POINT metrics on the canvas; dividing
    them by the reduced preview raster inflated every gap by the drag
    divisor, so the labels flew outward the moment a drag began.  With
    the same camera, preview and committed chrome may differ only by the
    raster's coarser position quantization.
    """

    session = _session()
    try:
        session.set_parameters({
            "presentation": "height_bars",
            "color_min": 0.0,
            "color_max": 1.0,
        })
        session.rgba()
        renderer = session._renderer
        def chrome_positions():
            artists = renderer._artists["image:h3d_chrome"]
            return {
                text.get_text(): text.get_position()
                for text in artists["texts"]
            }
        committed = chrome_positions()
        renderer.set_height_bars_dragging(True)
        try:
            session._render_current(RenderEffect.BASE_GEOMETRY)
            preview = chrome_positions()
        finally:
            renderer.set_height_bars_dragging(False)
            session._render_current(RenderEffect.BASE_GEOMETRY)
        assert set(preview) == set(committed)
        for label, (px, py) in preview.items():
            cx, cy = committed[label]
            assert abs(px - cx) <= 0.008 and abs(py - cy) <= 0.008, (
                label, (px, py), (cx, cy)
            )
    finally:
        session.close()


def test_middle_double_click_restores_the_home_camera() -> None:
    session = _session()
    try:
        session.set_parameter("presentation", "height_bars")
        session.set_parameters({
            "camera_azimuth": 130.0,
            "camera_elevation": 62.0,
            "camera_zoom": 2.5,
        })
        session.rgba()
        axis = next(
            t for t in session._raster_axes_snapshot() if t.role == "image"
        )
        _pointer(session, "press", axis, 0.5, 0.5, button=2)
        _pointer(session, "release", axis, 0.5, 0.5, button=2)
        _pointer(session, "press", axis, 0.5, 0.5, button=2)
        _pointer(session, "release", axis, 0.5, 0.5, button=2)
        state = session.display_state
        home = HeightBarCamera()
        assert float(state["camera_azimuth"]) == home.azimuth_deg
        assert float(state["camera_elevation"]) == 30.0
        assert float(state["camera_zoom"]) == 1.0
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
        width=400, height=300,
    )
    renderer = object.__new__(MatplotlibRenderer)

    def visible_fraction(edge):
        # The SAMPLER is the unit that decides what hides what; what the
        # artist is handed afterwards is the visible runs, which no
        # longer carries a sample-by-sample answer.
        xs, _ = renderer._height_bars_sampled_polyline(
            scene, np.asarray([edge], dtype=np.float64), 64
        )
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
        width=400, height=300,
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
            supersample=2,
        )
        frames.append(frame)
        del noise
    for frame in frames[1:]:
        np.testing.assert_array_equal(frames[0], frame)


def test_the_occlusion_sampler_mirrors_its_reference_bit_for_bit() -> None:
    """The compiled sampler is a SPEED path for the outlines, like the
    materializer is for the pixels -- and until now only the materializer
    had a contract holding it to its reference.

    The numpy walk is the specification.  Compare them on a scene that
    really hides things, at several sample counts and from several
    cameras, so a change to either one cannot drift quietly.
    """

    pytest.importorskip("numba")
    from zlc_plot import _height3d_raster as raster

    session = _session(12)
    try:
        session.set_parameter("presentation", "height_bars")
        for azimuth, elevation in ((-55.0, 30.0), (40.0, 12.0), (220.0, 62.0)):
            session.set_parameter("camera_azimuth", azimuth)
            session.set_parameter("camera_elevation", elevation)
            session.rgba()
            renderer = session._renderer
            scene = renderer._height_bars_scene_map
            cells = [(row, column) for row in range(4)
                     for column in range(4)]
            edges = renderer._height_bars_box_edges(
                scene, cells,
                np.zeros(len(cells)), np.full(len(cells), 0.6),
            )
            assert edges.shape[0] > 0
            previous = raster._ENGINE
            try:
                walks = {}
                for engine in ("numpy", "numba"):
                    raster._ENGINE = engine
                    walks[engine] = tuple(
                        renderer._height_bars_sampled_polyline(scene, edges, count)
                        for count in (4, 16, 64)
                    )
            finally:
                raster._ENGINE = previous
            for reference, compiled in zip(walks["numpy"], walks["numba"]):
                np.testing.assert_array_equal(reference[0], compiled[0])
                np.testing.assert_array_equal(reference[1], compiled[1])
            hidden = np.isnan(walks["numpy"][2][0])
            assert hidden.any(), "this camera must hide something"
    finally:
        session.close()


def test_the_rim_stroke_mirrors_its_reference_bit_for_bit() -> None:
    """The creases are pixels now, so the compiled path that draws them is
    held to the numpy one exactly as the materializer and the occlusion
    sampler are: same frame in, same frame out, byte for byte."""

    pytest.importorskip("numba")
    from zlc_plot import _height3d_raster as raster

    rng = np.random.default_rng(17)
    for cells, side, width in ((7, 200, 3.3), (23, 480, 2.0), (50, 815, 1.65)):
        ids = np.ascontiguousarray(np.repeat(np.repeat(
            rng.integers(0, cells * cells, size=(cells, cells)).astype(np.int32)
            * 4 + 4, side // cells + 1, axis=0), side // cells + 1,
            axis=1)[:side, :side])
        # A few background and floor faces, so the "not the room" rule is
        # exercised rather than assumed.
        ids[: side // 8, :] = 1
        ids[:, : side // 9] = 2
        base = rng.integers(0, 255, size=(side, side, 4)).astype(np.uint8)
        frames = {}
        previous = raster._ENGINE
        try:
            for engine in ("numpy", "numba"):
                raster._ENGINE = engine
                frame = base.copy()
                raster._stroke_rims(frame, ids, (0.1, 0.2, 0.3), width)
                frames[engine] = frame
        finally:
            raster._ENGINE = previous
        np.testing.assert_array_equal(frames["numpy"], frames["numba"])
        assert not np.array_equal(frames["numpy"], base), "nothing was drawn"


def test_scanline_engine_matches_the_reference_bit_for_bit() -> None:
    """The numba scanline engine is a SPEED path, never a semantics path.

    The numpy kernel is the specification; the compiled engine must
    reproduce it bit for bit -- frame and pick map both -- across folds,
    exact crossing ties (azimuth 45), clipping limits, NaN holes,
    hanging negative bars and a pooled grid.
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


def test_the_orbit_is_continuous_across_quadrant_boundaries() -> None:
    """The fold is a ROTATION: crossing a quadrant boundary must change
    the picture no more than the same step inside a quadrant.  The
    flip-only fold kept the axes in place and the scene snapped ninety
    degrees at every boundary."""

    from zlc_plot._height3d_raster import HeightBarCamera, render_height_bars

    xx, yy = np.meshgrid(np.arange(24), np.arange(12))
    heights = np.exp(-((xx - 6.0) ** 2 + (yy - 3.0) ** 2) / 12.0)
    colors = np.repeat(
        heights[..., None].astype(np.float32).clip(0.0, 1.0), 3, axis=-1
    )

    def frame(azimuth: float) -> np.ndarray:
        rendered, _ = render_height_bars(
            heights, colors,
            camera=HeightBarCamera(azimuth_deg=azimuth, elevation_deg=30.0),
            value_limits=(0.0, 1.0), width=360, height=280,
            supersample=1, zero_rgb=(1.0, 1.0, 1.0),
        )
        return rendered[..., :3].astype(np.int64)

    def step(low: float, high: float) -> int:
        return int(np.abs(frame(low) - frame(high)).sum())

    control = step(44.9, 45.1)
    for boundary in (90.0, 180.0, 270.0, 360.0):
        jump = step(boundary - 0.1, boundary + 0.1)
        assert jump < 4 * control, (boundary, jump, control)


def test_a_display_commit_mid_drag_does_not_undo_the_drag() -> None:
    """A pan or orbit reads the AXES; a colour limit does not move them.

    Cancelling every gesture on INTERACTION_REPROJECT threw away the
    in-flight candidate, so a clim commit -- or any mirrored display
    parameter -- landing mid-drag snapped the scene back to where the
    drag began and left the rest of the drag dead.
    """

    session = _session()
    try:
        session.set_parameters({
            "presentation": "height_bars",
            "color_min": 0.0,
            "color_max": 1.0,
        })
        session.rgba()
        axis = next(
            t for t in session._raster_axes_snapshot() if t.role == "image"
        )
        start = float(session.display_state["camera_azimuth"])
        _pointer(session, "press", axis, 0.5, 0.5, button=2)
        _pointer(session, "move", axis, 0.62, 0.55, button=2)
        # a display commit lands while the hand is still down
        session.set_parameter("color_max", 0.75)
        _pointer(session, "move", axis, 0.74, 0.60, button=2)
        _pointer(session, "release", axis, 0.74, 0.60, button=2)
        assert float(session.display_state["camera_azimuth"]) != start
    finally:
        session.close()


def test_every_move_of_a_fast_hand_reaches_the_screen() -> None:
    """A gesture lane paces itself by what its own frames cost.

    The lane was a fixed 30 ms, and a lane that is not due DROPS the
    motion rather than deferring it -- so a hand moving at mouse rate
    lost two thirds of its updates, and whenever the last move before
    release fell in a closed window the picture only caught up when the
    button came up.  A scene preview costs a few milliseconds, so at a
    realistic 125 Hz hand every move must reach the screen.
    """

    session = _session()
    try:
        session.set_parameters({
            "presentation": "height_bars",
            "color_min": 0.0,
            "color_max": 1.0,
        })
        session.rgba()
        axis = next(
            t for t in session._raster_axes_snapshot() if t.role == "image"
        )
        _pointer(session, "press", axis, 0.5, 0.5, button=2)
        rendered = 0
        for step in range(20):
            state = _pointer(
                session,
                "move",
                axis,
                0.5 + 0.02 * (step + 1),
                0.5 + 0.01 * (step + 1),
                button=2,
            )
            if getattr(state, "publish_front", False):
                rendered += 1
            sleep(0.008)
        _pointer(session, "release", axis, 0.9, 0.7, button=2)
        assert rendered == 20, f"{20 - rendered} moves never reached pixels"
    finally:
        session.close()


def _long_coordinate_session(side: int = 10):
    """A scan whose coordinates are long numbers -- an optical frequency.

    Long labels are what makes a rotated scene crowd: the text boxes are
    wide enough to meet however the projection lays the ticks out.
    """

    base = 384227900000.0
    coords = tuple(base + index for index in range(side))
    cells = [(i % side, i // side) for i in range(side * side)]
    schema = DatasetSchema.create(
        Axis.create("repeat", size=4),
        PointTable.from_columns({
            "ax": np.asarray([coords[c[0]] for c in cells]),
            "ay": np.asarray([coords[c[1]] for c in cells]),
        }),
        data_axes=(Axis.create("site", values=[0.0, 1.0]),),
        dtype=np.float64,
        point_topology=PointTopology(
            (AxisId("ax"), AxisId("ay")), (coords,) * 2, tuple(cells)
        ),
    )
    rng = np.random.default_rng(7)
    session = PlotSession(
        DatasetSnapshot(schema, rng.random((4, side * side, 2)), revision=1),
        ImagePlot(AxisRef.point_dimension("ax"), AxisRef.point_dimension("ay")),
    )
    session.set_size("2x2")
    session.set_parameter("presentation", "height_bars")
    return session


@pytest.mark.parametrize("azimuth", [-55.0, 40.0, 130.0, 220.0])
def test_a_rotated_scene_never_prints_a_label_across_another(azimuth) -> None:
    """Rotation moves the labels, so no fixed stride can keep them apart.

    Whether two labels collide is measurable, so that is what decides --
    and the ends survive, because an axis whose extremes are legible still
    says what it spans.
    """

    session = _long_coordinate_session()
    try:
        session.set_parameters({
            "camera_azimuth": azimuth,
            "camera_elevation": 25.0,
        })
        session.rgba()
        renderer = session._renderer
        chrome = renderer._artists["image:h3d_chrome"]
        shown = [text for text in chrome["texts"] if text.get_visible()]
        assert len(shown) >= 2
        canvas_renderer = renderer.figure.canvas.get_renderer()
        boxes = [text.get_window_extent(canvas_renderer) for text in shown]
        for first in range(len(boxes)):
            for second in range(first + 1, len(boxes)):
                assert not boxes[first].overlaps(boxes[second]), (
                    f"{shown[first].get_text()!r} prints across "
                    f"{shown[second].get_text()!r} at azimuth {azimuth}"
                )
    finally:
        session.close()


def test_the_scene_and_its_labels_share_one_padded_region() -> None:
    """A scene is laid out as a scene: ONE region, for picture and labels.

    A heatmap can reserve margins because its chrome has fixed places --
    ticks under the bottom spine, labels left of the left one.  Turn a
    camera and a label that hung under the floor is beside the colorbar,
    so no margin can be reserved for a place that moves.  The scene
    therefore gets the whole picture side of the panel: down to one
    padding from the figure's left and bottom edges, out to one padding
    from the rail beside it, and no further up than the picture already
    reached -- the title's room is not the scene's to take.  The padding
    is the gap the layout already leaves between picture and rail.
    """

    def boxes(presentation):
        session = _session()
        try:
            session.set_parameter("presentation", presentation)
            session.rgba()
            plan = session.surface_plan
            return (
                next(a for a in plan.axes if a.role == "image").box,
                next(a for a in plan.axes if a.role == "distribution").box,
                plan.figure_size_inches,
            )
        finally:
            session.close()

    picture, rail, figure_size = boxes("heatmap")
    scene, scene_rail, _ = boxes("height_bars")
    # The rails do not move: toggling the presentation re-rooms the
    # picture, it does not re-lay the panel.
    assert (scene_rail.left, scene_rail.right) == (rail.left, rail.right)
    pad = rail.left - picture.right
    assert pad > 0.0
    figure_width, figure_height = figure_size
    vertical = pad * figure_width / figure_height
    assert scene.left == pytest.approx(pad)
    assert scene.right == pytest.approx(rail.left - pad)
    assert scene.bottom == pytest.approx(1.0 - vertical)
    assert scene.top == pytest.approx(picture.top)


def test_scene_labels_are_cut_off_at_the_room_the_scene_owns() -> None:
    """A 3D label goes where the projection puts it, which is not a place
    any layout reserved.  It gets the scene's own room -- out to whatever
    bounds it on the right -- and is cut there rather than printing over a
    neighbour that means something else."""

    session = _long_coordinate_session()
    try:
        session.set_parameters({
            "camera_azimuth": 40.0,
            "camera_elevation": 25.0,
        })
        session.rgba()
        renderer = session._renderer
        chrome = renderer._artists["image:h3d_chrome"]
        shown = [text for text in chrome["texts"] if text.get_visible()]
        rails = [
            axes
            for role in ("distribution", "colorbar")
            for axes in renderer._axes.get(role, ())
        ]
        assert rails, "this panel has the neighbours the region is bounded by"
        limit = min(float(axes.get_window_extent().x0) for axes in rails)
        canvas_renderer = renderer.figure.canvas.get_renderer()
        reaching = [
            text
            for text in shown
            if text.get_window_extent(canvas_renderer).x1 > limit
        ]
        assert reaching, (
            "this arrangement is the one where labels reach the rail; "
            "without that the clip proves nothing"
        )
        for text in reaching:
            assert text.get_clip_on(), text.get_text()
            assert float(text.get_clip_box().x1) <= limit + 1.0, (
                f"{text.get_text()!r} may paint past the room it owns"
            )
    finally:
        session.close()
