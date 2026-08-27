"""A focused FacetGrid image cell IS the standalone Image surface.

The focused facet cell must show
everything the standalone image kind shows -- the ``_split_image`` layout
(data + distribution + colorbar axes), the colorbar with its value label,
the image spatial tick budget, the authored figure title, and a parameter
surface that matches the standalone schema modulo the facet's own
``facet_display_unit``.  One chrome authority paints both paths.

The twin admits-vs-build contradiction is guarded here too: the image kind
``admits`` multi-point scans of frames (point rows pool under the declared
reduction), so ``PlotSession`` construction must build and render them --
including the frames-on-point-axis camera-cycle shape ``(1, F, y, x)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from data_factory import (
    Axis,
    DatasetSchema,
    DatasetSnapshot,
    PointTable,
    PointTopology,
)
from zlc_data import SITE, SPATIAL_X, SPATIAL_Y, ValidityContract, ValueSchema
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, ImagePlot, PlotSession
from zlc_plot._kinds.image import default_spec as image_default
from zlc_plot.config import DEFAULTS
from zlc_plot.selectors import CrosshairPoint, NumericRange


def _frame_axes() -> tuple:
    return (
        Axis.create(
            "sy", values=tuple(float(v) for v in range(40)), role=SPATIAL_Y
        ),
        Axis.create(
            "sx", values=tuple(float(v) for v in range(60)), role=SPATIAL_X
        ),
    )


def _frames_scan_snapshot(*, repeats: int = 1) -> DatasetSnapshot:
    """(repeats, 3 bias points, y, x) frames under a declared scan topology."""

    table = PointTable.from_columns({"bias": [-1.0, 0.0, 1.0]})
    outer = Axis.create("bias", values=[-1.0, 0.0, 1.0])
    topology = PointTopology.from_cartesian((outer,), point_table=table)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        table,
        data_axes=_frame_axes(),
        point_topology=topology,
        dtype=np.float64,
    )
    frame = np.add.outer(np.arange(40.0), np.arange(60.0))
    values = np.broadcast_to(frame, (repeats, 3, 40, 60)).copy()
    values += np.arange(3.0)[None, :, None, None]
    return DatasetSnapshot(schema, values, revision=1)


def _occupancy_overlay(
    snapshot: DatasetSnapshot,
    occupied: np.ndarray,
    valid: np.ndarray,
    *,
    acquired: np.ndarray | None = None,
):
    from zlc_plot.primitives import ImagePointOverlay

    source = snapshot.block.schema
    site = Axis.create("site", size=2, role=SITE)
    schema = DatasetSchema(
        source.repeat_axis,
        source.point_table,
        source.grid_topology,
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            np.dtype(np.bool_),
            "1",
        ),
    )
    if acquired is None:
        acquired = np.ones(occupied.shape[:2], dtype=np.bool_)
    status_validity = (
        np.asarray(valid, dtype=np.bool_)
        & np.asarray(acquired, dtype=np.bool_)[..., None]
    )
    return ImagePointOverlay(
        coordinates=((10.0, 12.0), (30.0, 24.0)),
        revision=1,
        point_ids=("s0", "s1"),
        status=DatasetSnapshot(
            schema,
            np.asarray(occupied, dtype=np.bool_),
            revision=1,
            validity=status_validity,
        ),
    )


def _single_frame_snapshot() -> DatasetSnapshot:
    """One point, one frame: the standalone twin of one focused facet cell."""

    table = PointTable.from_columns({"bias": [0.0]})
    outer = Axis.create("bias", values=[0.0])
    topology = PointTopology.from_cartesian((outer,), point_table=table)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        table,
        data_axes=_frame_axes(),
        point_topology=topology,
        dtype=np.float64,
    )
    frame = np.add.outer(np.arange(40.0), np.arange(60.0))
    return DatasetSnapshot(schema, frame[None, None], revision=1)


def _grid_frames_snapshot() -> DatasetSnapshot:
    table = PointTable.from_columns(
        {
            "bias": [-1.0, -1.0, 1.0, 1.0],
            "grad": [10.0, 20.0, 10.0, 20.0],
        }
    )
    topology = PointTopology.from_cartesian(
        (
            Axis.create("bias", values=[-1.0, 1.0]),
            Axis.create("grad", values=[10.0, 20.0]),
        ),
        point_table=table,
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        table,
        data_axes=_frame_axes(),
        point_topology=topology,
        dtype=np.float64,
    )
    values = np.zeros((1, 4, 40, 60), dtype=np.float64)
    return DatasetSnapshot(schema, values, revision=1)


def _cycle_frames_on_point_axis_snapshot() -> DatasetSnapshot:
    """(1, 3 frame-points, y, x): a camera cycle publishing frames as points."""

    table = PointTable.from_columns({"frame": [0.0, 1.0, 2.0]})
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        table,
        data_axes=_frame_axes(),
        dtype=np.float64,
    )
    values = np.stack(
        [np.full((40, 60), float(v)) for v in range(3)], axis=0
    )[None]
    return DatasetSnapshot(schema, values, revision=1)


_FACET_SPEC = FacetGridPlot(
    AxisRef.point_dimension("bias"),
    ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")),
)


def test_focused_image_cell_matches_the_standalone_image_surface() -> None:
    standalone = PlotSession(
        _single_frame_snapshot(),
        image_default(_single_frame_snapshot().block.schema),
        size="4x4",
    )
    facet = PlotSession(_frames_scan_snapshot(), _FACET_SPEC, size="4x4")
    try:
        facet.set_parameter("title", "authored title")
        facet.focus_facet(1)
        alone = standalone._renderer
        focused = facet._renderer
        alone.figure.canvas.draw()
        focused.figure.canvas.draw()

        # The focused cell resolves through the SAME image split: data axes
        # plus one distribution and one colorbar axes, all visible.
        for renderer in (alone, focused):
            assert len(renderer.axes.get("distribution", ())) == 1
            assert len(renderer.axes.get("colorbar", ())) == 1
            assert renderer.axes["distribution"][0].get_visible()
            assert renderer.axes["colorbar"][0].get_visible()
        assert (
            sum(1 for axis in focused.figure.axes if axis.get_visible())
            == sum(1 for axis in alone.figure.axes if axis.get_visible())
        )

        # Colorbar artist and value label, from the one chrome authority.
        alone_colorbar = alone._artists["image:colorbar"]
        focused_colorbar = focused._artists["facet:1:colorbar"]
        assert focused_colorbar.ax.get_ylabel() == alone_colorbar.ax.get_ylabel()

        # One tick policy, and the cell is on it: how many labels each axis
        # carries follows from the width it is painted at, so the signature
        # says WHICH policy, not how many.
        for axis_obj in (
            focused.primary_axes.xaxis,
            focused.primary_axes.yaxis,
            alone.primary_axes.xaxis,
            alone.primary_axes.yaxis,
        ):
            assert axis_obj._zlc_tick_signature[0].startswith("smart-")

        # The parameter surface matches modulo the facet's own axis unit.
        alone_names = set(standalone.describe_display().parameter_schema)
        focused_names = set(facet.describe_display().parameter_schema)
        assert focused_names - alone_names == {"facet_display_unit"}
        assert alone_names - focused_names == set()
        assert "interpolation" not in alone_names | focused_names

        from matplotlib.image import AxesImage

        for renderer in (alone, focused):
            images = tuple(
                artist
                for artist in renderer._artists.values()
                if isinstance(artist, AxesImage)
            )
            assert images
            assert {image.get_interpolation() for image in images} == {"nearest"}

        # The authored figure title stays visible alongside the cell title.
        title_artist = focused._artists["figure:title"]
        assert title_artist.get_visible()
        assert title_artist.get_text() == "authored title"
        assert focused.primary_axes.get_title() == "bias=0"

        # SAME GEOMETRY, not merely the same chrome.  The cell used to be
        # drawn with the squaring carved out (``square_view=False``), so a
        # 60x40 frame filled a 3:2 box in the facet and a square one alone --
        # the same picture in two shapes, and the overview slots were shaped
        # by a THIRD rule that agreed with neither.
        # To the pixel, not to the last bit: the focused cell measures its
        # box from the union of the overview's cells and the standalone from
        # the data region, and those two arrive at the same rectangle by
        # slightly different arithmetic.
        assert tuple(focused.primary_axes.get_window_extent().bounds) == pytest.approx(
            tuple(alone.primary_axes.get_window_extent().bounds), abs=1e-6
        )
        assert focused.primary_axes.get_xlim() == alone.primary_axes.get_xlim()
        assert focused.primary_axes.get_ylim() == alone.primary_axes.get_ylim()
    finally:
        standalone.close()
        facet.close()


def test_focused_cell_colour_limit_drag_moves_the_cell_clim() -> None:
    """A driven press/move/release on the focused cell's rail must land.

    The preview path looked up a hardcoded ``"image"`` artist key and an
    outer-spec branch, so the very first motion event of a facet-cell drag
    raised ``color-limit preview requires an Image`` out of a pointer slot.
    """

    session = PlotSession(_frames_scan_snapshot(), _FACET_SPEC, size="4x4")
    try:
        session.focus_facet(1)
        renderer = session._renderer
        distribution = renderer.axes["distribution"][0]
        transform = session._axis_transform_for_axis(distribution)
        low, high = renderer.resolved_color_limits()
        left, top, right, bottom = transform.bounds
        y_low, y_high = transform.y_limits
        middle_x = (left + right) / 2.0

        def pointer_y(value: float) -> float:
            # The rail is vertical: a value maps down the distribution box,
            # and the raster pointer contract is bottom-origin normalized.
            return 1.0 - (top + (value - y_high) / (y_low - y_high) * (bottom - top))

        target = high - 0.25 * (high - low)
        session._raster_pointer_event(
            "press", middle_x, pointer_y(high), button=1, axes_snapshot=transform
        )
        session._raster_pointer_event("move", middle_x, pointer_y(target), button=1)
        session._raster_pointer_event("release", middle_x, pointer_y(target), button=1)

        moved_low, moved_high = renderer.resolved_color_limits()
        assert (moved_low, moved_high) != (low, high)
        assert moved_high < high or moved_low > low
    finally:
        session.close()


def test_focused_cell_crosshair_keeps_its_value_rail() -> None:
    """The crosshair's sample rail is a fact of the IMAGE, so a cell has it."""

    from zlc_plot.selectors import SelectorKind, SelectorState

    session = PlotSession(_frames_scan_snapshot(), _FACET_SPEC, size="4x4")
    try:
        session.focus_facet(1)
        renderer = session._renderer
        snapshot = type(renderer._last_selectors)(
            (
                SelectorState(
                    SelectorKind.CROSSHAIR, CrosshairPoint(30.0, 20.0), facet_index=1
                ),
            )
        )
        context = renderer._make_selector_scene_owner(snapshot).contexts[0]
        assert context.sample_target is not None
        assert context.sample_target.role == "distribution"
        assert context.sample_x_limits is not None
        assert context.sample_rgba is not None
    finally:
        session.close()


def test_overview_slots_are_shaped_for_what_the_renderer_draws() -> None:
    """The layout's declared cell aspect IS the aspect the cell is drawn at.

    The layout used to answer "what shape is an image" from the pixel counts
    (60x40 -> 1.5) while the renderer squared the same cell, so every slot
    kept a third of its width as dead space around the drawn cell.

    Height over width, and the field says so: the same number used to be
    read as width/height here and as height/width by the split, which agreed
    only while every image declared a square 1.0.
    """

    session = PlotSession(_frames_scan_snapshot(), _FACET_SPEC, size="4x4")
    try:
        session._renderer.figure.canvas.draw()
        declared = session.surface_plan.facet_topology.cell_height_over_width
        drawn = [
            axis.get_window_extent().bounds
            for axis in session._renderer.axes["facet_cell"]
            if axis.get_visible()
        ]
        assert len(drawn) == 3
        for _x, _y, width, height in drawn:
            assert math.isclose(
                declared, float(height) / float(width), rel_tol=1e-9
            ), (declared, width, height)
    finally:
        session.close()


def test_overview_view_limits_apply_to_every_visible_cell() -> None:
    """An overview shows N views of ONE picture; zooming one is not a view.

    The requested rectangle grows to the source pixel grid -- an image view
    is a whole number of pixels, and the request stays wholly inside it.
    Every cell moves the same way, which is the point here.
    """

    session = PlotSession(_frames_scan_snapshot(), _FACET_SPEC, size="4x4")
    try:
        session.set_view_limits(x=(10.0, 30.0), y=(5.0, 25.0))
        cells = session._renderer.axes["facet_cell"]
        visible = [axis for axis in cells if axis.get_visible()]
        assert len(visible) == 3
        for axis in visible:
            assert tuple(map(float, axis.get_xlim())) == (9.5, 30.5)
            assert tuple(map(float, sorted(axis.get_ylim()))) == (4.5, 25.5)
        # ...and every cell prepares its RASTER for the view it shows, which
        # is what the standalone image does.  Honouring the request on the
        # selected cell only left the rest showing a full-extent front cropped
        # to a zoom: the same pixels, at a fraction of the resolution.
        for index in range(len(visible)):
            prepared = session._renderer._artists[f"facet:{index}:prepared_current"]
            assert tuple(map(float, prepared.extent)) == (9.5, 30.5, 25.5, 4.5)
    finally:
        session.close()

    # A curve cell reaches the same limits through the render frame alone --
    # it has no image artist to also honour the request, which is what kept
    # the image half of this passing while the general rule was still broken.
    curves = PlotSession(
        _frames_scan_snapshot(repeats=3),
        FacetGridPlot(AxisRef.repeat(), CurvePlot(AxisRef.point_dimension("bias"))),
        size="4x4",
    )
    try:
        curves.set_view_limits(x=(-0.5, 0.5), y=(10.0, 40.0))
        visible = [
            axis
            for axis in curves._renderer.axes["facet_cell"]
            if axis.get_visible()
        ]
        assert len(visible) == 3
        for axis in visible:
            assert tuple(map(float, axis.get_xlim())) == (-0.5, 0.5)
            assert tuple(map(float, axis.get_ylim())) == (10.0, 40.0)
    finally:
        curves.close()


def test_facet_image_cells_carry_the_point_overlay() -> None:
    """The site overlay is a fact of the IMAGE, so a grid of images shows it.

    The painter gated on the outer spec and namespaced its artists as the
    bare ``image:points``, so a FacetGrid of frames could not carry one at
    all -- the session refused the overlay before the renderer ever saw it.
    """

    from zlc_plot.primitives import ImagePointOverlay

    session = PlotSession(_frames_scan_snapshot(), _FACET_SPEC, size="4x4")
    try:
        overlay = ImagePointOverlay(
            coordinates=((10.0, 12.0), (30.0, 24.0)),
            revision=1,
            point_ids=("s0", "s1"),
        )
        session.update_image_overlay(overlay)
        artists = session._renderer._artists
        for index in range(3):
            assert f"facet:{index}:points" in artists, sorted(artists)
            assert artists[f"facet:{index}:points"].get_visible()
        assert "image:points" not in artists
    finally:
        session.close()


def test_overview_destroys_the_focused_chrome_again() -> None:
    session = PlotSession(_frames_scan_snapshot(), _FACET_SPEC, size="4x4")
    try:
        before = session.rgba().copy()
        session.focus_facet(1)
        assert session._renderer.axes.get("distribution")
        session.show_facet_overview()
        # The side chrome is a fact of the focused presentation: unfocusing
        # destroys the axes and their cached artists rather than parking
        # them invisibly.
        assert not session._renderer.axes.get("distribution")
        assert not session._renderer.axes.get("colorbar")
        assert not any(
            key.startswith("facet:1:colorbar")
            for key in session._renderer._artists
        )
        np.testing.assert_array_equal(session.rgba(), before)
    finally:
        session.close()


def test_direct_focus_switch_leaves_no_chrome_ghost() -> None:
    """Switching the focus between two cells rebuilds the side chrome.

    The first ghost found here: the color-rail selector scene cached its
    artists on the REMOVED distribution axes, and the compose path kept
    painting them at the dead box while a full draw skipped them -- 3033
    differing pixels.  Compose must stay full-draw-exact through every
    focus transition, so the scene dies with its axes and the dynamic
    collector honours figure membership.
    """

    session = PlotSession(_frames_scan_snapshot(), _FACET_SPEC, size="4x4")
    try:
        renderer = session._renderer
        for index in (0, 2, 1):
            session.focus_facet(index)
            assert f"facet:{index}:colorbar" in renderer._artists
            other_keys = [
                key
                for key in renderer._artists
                if key.endswith(":colorbar")
                and not key.startswith(f"facet:{index}:")
            ]
            assert other_keys == [], other_keys
            raw, height, width = session._raster_capture_rgba_bytes()
            composed = np.frombuffer(raw, dtype=np.uint8).reshape(
                height, width, 4
            ).copy()
            renderer.draw()
            raw, height, width = session._raster_capture_rgba_bytes()
            reference = np.frombuffer(raw, dtype=np.uint8).reshape(
                height, width, 4
            )
            np.testing.assert_array_equal(composed, reference)
    finally:
        session.close()


def test_color_limit_pointer_gates_on_the_cell_spec() -> None:
    """The focused distribution must own the color-limit drag: the gesture
    gate reads the SEMANTIC spec (the cell), not the outer FacetGrid."""

    session = PlotSession(_frames_scan_snapshot(), _FACET_SPEC, size="4x4")
    try:
        session.focus_facet(1)
        distribution = session._renderer.axes["distribution"][0]
        low, high = session._renderer.resolved_color_limits()
        middle = (low + high) / 2.0
        # A press away from both handles is accepted (claimed, no gesture);
        # before the fix the outer-spec gate returned False outright.
        assert session._begin_color_limit_pointer(
            CrosshairPoint(middle, 0.0), distribution, None
        )
    finally:
        session.close()

    curve_facet = PlotSession(
        _frames_scan_snapshot(repeats=3),
        FacetGridPlot(
            AxisRef.repeat(),
            CurvePlot(AxisRef.point_dimension("bias")),
        ),
        size="4x4",
    )
    try:
        curve_facet.focus_facet(0)
        # A curve cell has no color surface: the gate must refuse.
        assert not curve_facet._begin_color_limit_pointer(
            CrosshairPoint(0.0, 0.0), object(), None
        )
    finally:
        curve_facet.close()


def test_image_admits_scans_of_frames_and_builds() -> None:
    """admits == buildable: the projections image.default_spec admits must
    construct and render, pooling point rows under the declared reduction."""

    scan = _frames_scan_snapshot(repeats=2)
    spec = image_default(scan.block.schema)
    assert isinstance(spec, ImagePlot)
    session = PlotSession(scan, spec, size="4x4")
    try:
        assert session.rgba().size
        payload = session._payload
        expected = np.asarray(scan.block.values).mean(axis=(0, 1))
        np.testing.assert_allclose(np.asarray(payload.z.canonical), expected)
    finally:
        session.close()


def test_frames_on_point_axis_cycle_draws_as_a_standalone_image() -> None:
    cycle = _cycle_frames_on_point_axis_snapshot()
    spec = image_default(cycle.block.schema)
    assert isinstance(spec, ImagePlot)
    session = PlotSession(cycle, spec, size="4x4")
    try:
        assert session.rgba().size
        payload = session._payload
        np.testing.assert_allclose(
            np.asarray(payload.z.canonical),
            np.full((40, 60), 1.0),  # frames 0, 1, 2 meaned
        )
    finally:
        session.close()


def test_a_scoped_image_draws_the_point_it_shows() -> None:
    """A scoped surface selects the same exact leading cell for its rings."""

    snapshot = _frames_scan_snapshot()
    overlay = _occupancy_overlay(
        snapshot,
        occupied=np.asarray([[[True, False], [False, True], [True, False]]]),
        valid=np.ones((1, 3, 2), dtype=np.bool_),
    )
    scoped = PlotSession(
        snapshot,
        ImagePlot(
            AxisRef.data("sx"),
            AxisRef.data("sy"),
            scope=((AxisRef.point_dimension("bias"), 0.0),),
        ),
        size="4x4",
    )
    try:
        selection_events = []
        scoped.subscribe_selection(selection_events.append)
        scoped.update_image_overlay(overlay)
        scoped.set_area_selector(
            NumericRange(1.0, 3.0),
            NumericRange(1.0, 3.0),
        )
        scoped.rgba()
        assert _drawn_statuses(scoped, "image:points") == (
            "EMPTY",
            "OCCUPIED",
        )
        from zlc_data import owned_snapshot_from_arrays

        restarted = owned_snapshot_from_arrays(
            snapshot.block.schema,
            snapshot.block.values + 100.0,
            snapshot.ref.revision,
            block_id=snapshot.ref.block_id,
            stream_generation="scoped-restart",
        )
        scoped.update_data(restarted)
        assert scoped.data_generation == "scoped-restart"
        # The overlay is DATA-derived and dies with its run; the area
        # selector is the OPERATOR's and survives a same-geometry
        # restart -- the axes still name the same world, and wiping the
        # table per run erased the marker every shot boundary.
        assert scoped.image_overlay is None
        from zlc_plot.selectors import SelectorKind

        assert [state.kind for state in scoped.selectors] == [
            SelectorKind.AREA
        ]
        assert not any(
            event.change.value == "removed" for event in selection_events
        )
    finally:
        scoped.close()


def test_overlay_geometry_refuses_a_reordered_same_count_status_axis() -> None:
    from zlc_plot import (
        image_point_overlay_from_signal,
        image_point_overlay_geometry,
    )

    image = _frames_scan_snapshot()
    overlay = _occupancy_overlay(
        image,
        occupied=np.zeros((1, 3, 2), dtype=np.bool_),
        valid=np.ones((1, 3, 2), dtype=np.bool_),
    )
    status = overlay.status
    assert status is not None
    status_axis = status.block.schema.cell_schema.data_axes[0]
    geometry = image_point_overlay_geometry(
        image,
        overlay.coordinates,
        overlay.point_ids,
        status_axis=status_axis,
    )
    reversed_axis = Axis.create("site", values=(1, 0), role=SITE)
    bad_schema = DatasetSchema(
        status.block.schema.repeat_axis,
        status.block.schema.point_table,
        status.block.schema.grid_topology,
        ValueSchema(
            (reversed_axis,),
            ValidityContract.components(reversed_axis.axis_id),
            np.dtype(np.bool_),
            "1",
        ),
    )
    bad = DatasetSnapshot(
        bad_schema,
        status.block.values,
        revision=1,
        validity=status.expanded_validity(),
    )
    with pytest.raises(ValueError, match="status axis"):
        image_point_overlay_from_signal(
            geometry,
            bad,
            image,
            revision=1,
        )

    for incompatible_image in (
        _frames_scan_snapshot(repeats=2),
        _single_frame_snapshot(),
    ):
        incompatible_geometry = image_point_overlay_geometry(
            incompatible_image,
            overlay.coordinates,
            overlay.point_ids,
            status_axis=status_axis,
        )
        with pytest.raises(ValueError, match="leading geometry"):
            image_point_overlay_from_signal(
                incompatible_geometry,
                status,
                incompatible_image,
                revision=1,
            )


def test_single_pixel_overlay_axis_never_invents_an_affine_step() -> None:
    from zlc_plot import image_point_overlay_geometry

    y_axis = Axis.create("sy", values=(7.0,), role=SPATIAL_Y)
    x_axis = Axis.create("sx", values=(5.0,), role=SPATIAL_X)
    image = DatasetSnapshot(
        DatasetSchema.create(
            Axis.create("repeat", size=1),
            PointTable.from_columns({"point": [0.0]}),
            data_axes=(y_axis, x_axis),
            dtype=np.float64,
        ),
        np.zeros((1, 1, 1, 1), dtype=np.float64),
        revision=1,
    )
    status_axis = Axis.create("site", size=1, role=SITE)
    with pytest.raises(ValueError, match="single-pixel"):
        image_point_overlay_geometry(
            image,
            ((0.25, 0.0),),
            ("site-0",),
            status_axis=status_axis,
            coordinates_are_indices=True,
        )


def _drawn_statuses(session: PlotSession, key: str) -> tuple[str, ...]:
    """Which judgement one painted surface's rings actually say."""

    from matplotlib.colors import to_rgba

    from zlc_plot.primitives import PointStatus

    tokens = session._renderer.style.artists
    by_colour = {
        tuple(round(value, 5) for value in to_rgba(token.color, token.alpha)): status
        for status, token in (
            (PointStatus.UNKNOWN, tokens.point_unknown),
            (PointStatus.EMPTY, tokens.point_empty),
            (PointStatus.OCCUPIED, tokens.point_occupied),
            (PointStatus.INVALID, tokens.point_invalid),
        )
    }
    collection = session._renderer._artists[key]
    return tuple(
        by_colour[tuple(round(float(value), 5) for value in colour)].name
        for colour in collection.get_edgecolors()
    )


def test_the_rings_a_surface_draws_are_the_ones_it_pinned_that_axis_to() -> None:
    """Only an exact repeat/point cell owns a site judgement.

    Scope and facet are the two ways a surface pins a leading axis.  An axis
    left to the image reduction is deliberately UNKNOWN even when every shot
    happened to agree; consensus is a scientific product, not a renderer
    default.  The final future cell is invalid in the canonical prefix and is
    likewise UNKNOWN rather than a false INVALID measurement.
    """

    snapshot = _frames_scan_snapshot(repeats=2)
    overlay = _occupancy_overlay(
        snapshot,
        occupied=np.asarray(
            [
                [[True, False], [False, True], [True, True]],
                [[False, True], [False, True], [False, False]],
            ]
        ),
        valid=np.asarray(
            [
                [[True, True], [True, True], [False, True]],
                [[True, True], [True, True], [False, False]],
            ]
        ),
        acquired=np.asarray(
            [[True, True, True], [True, True, False]],
        ),
    )
    unknown = ("UNKNOWN", "UNKNOWN")

    cases = (
        (
            "pooling both leading axes has no single-shot judgement",
            ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")),
            {"image:points": unknown},
        ),
        (
            "scoping only point still pools repeats",
            ImagePlot(
                AxisRef.data("sx"),
                AxisRef.data("sy"),
                scope=((AxisRef.point_dimension("bias"), 0.0),),
            ),
            {"image:points": unknown},
        ),
        (
            "scoping repeat and point selects the exact cell",
            ImagePlot(
                AxisRef.data("sx"),
                AxisRef.data("sy"),
                scope=(
                    (AxisRef.repeat(), 0.0),
                    (AxisRef.point_dimension("bias"), 1.0),
                ),
            ),
            {"image:points": ("INVALID", "OCCUPIED")},
        ),
        (
            "point facets with repeat scoped draw each point cell",
            FacetGridPlot(
                AxisRef.point_dimension("bias"),
                ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")),
                scope=((AxisRef.repeat(), 0.0),),
            ),
            {
                "facet:0:points": ("OCCUPIED", "EMPTY"),
                "facet:1:points": ("EMPTY", "OCCUPIED"),
                "facet:2:points": ("INVALID", "OCCUPIED"),
            },
        ),
        (
            "repeat facets with point scoped draw each repeat cell",
            FacetGridPlot(
                AxisRef.repeat(),
                ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")),
                scope=((AxisRef.point_dimension("bias"), -1.0),),
            ),
            {
                "facet:0:points": ("OCCUPIED", "EMPTY"),
                "facet:1:points": ("EMPTY", "OCCUPIED"),
            },
        ),
        (
            "a future invalid canonical cell is not a measured INVALID site",
            ImagePlot(
                AxisRef.data("sx"),
                AxisRef.data("sy"),
                scope=(
                    (AxisRef.repeat(), 1.0),
                    (AxisRef.point_dimension("bias"), 1.0),
                ),
            ),
            {"image:points": unknown},
        ),
    )
    for title, spec, expected in cases:
        session = PlotSession(snapshot, spec, size="4x4")
        try:
            session.update_image_overlay(overlay)
            session.rgba()
            for key, statuses in expected.items():
                assert _drawn_statuses(session, key) == statuses, title
        finally:
            session.close()


def test_multidimensional_grid_status_requires_every_point_dimension_to_resolve() -> None:
    snapshot = _grid_frames_snapshot()
    overlay = _occupancy_overlay(
        snapshot,
        occupied=np.asarray(
            [[[True, False], [False, True], [True, True], [False, False]]]
        ),
        valid=np.ones((1, 4, 2), dtype=np.bool_),
    )
    exact = FacetGridPlot(
        AxisRef.point_dimension("bias"),
        ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")),
        scope=(
            (AxisRef.repeat(), 0.0),
            (AxisRef.point_dimension("grad"), 20.0),
        ),
    )
    pooled = FacetGridPlot(
        AxisRef.point_dimension("bias"),
        ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")),
        scope=((AxisRef.repeat(), 0.0),),
    )
    cases = (
        (
            exact,
            {
                "facet:0:points": ("EMPTY", "OCCUPIED"),
                "facet:1:points": ("EMPTY", "EMPTY"),
            },
        ),
        (
            pooled,
            {
                "facet:0:points": ("UNKNOWN", "UNKNOWN"),
                "facet:1:points": ("UNKNOWN", "UNKNOWN"),
            },
        ),
    )
    for spec, expected in cases:
        session = PlotSession(snapshot, spec, size="4x4")
        try:
            session.update_image_overlay(overlay)
            session.rgba()
            for key, statuses in expected.items():
                assert _drawn_statuses(session, key) == statuses
        finally:
            session.close()
