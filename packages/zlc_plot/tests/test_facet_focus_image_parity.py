"""A focused FacetGrid image cell IS the standalone Image surface.

Contract from the migration review: the focused facet cell must show
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

import numpy as np

from data_factory import (
    Axis,
    DatasetSchema,
    DatasetSnapshot,
    PointTable,
    PointTopology,
)
from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, ImagePlot, PlotSession
from zlc_plot._kinds.image import default_spec as image_default
from zlc_plot.config import DEFAULTS
from zlc_plot.selectors import CrosshairPoint


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
        focused_colorbar = focused._artists["facet:1:image:colorbar"]
        assert focused_colorbar.ax.get_ylabel() == alone_colorbar.ax.get_ylabel()

        # The standalone image's spatial tick budget applies to the cell.
        budget = DEFAULTS.style.render.image_spatial_max_ticks
        for axis_obj in (
            focused.primary_axes.xaxis,
            focused.primary_axes.yaxis,
            alone.primary_axes.xaxis,
            alone.primary_axes.yaxis,
        ):
            assert axis_obj._zlc_tick_signature[1] == budget

        # The parameter surface matches modulo the facet's own axis unit.
        alone_names = set(standalone.describe_display().parameter_schema)
        focused_names = set(facet.describe_display().parameter_schema)
        assert focused_names - alone_names == {"facet_display_unit"}
        assert alone_names - focused_names == set()

        # The authored figure title stays visible alongside the cell title.
        title_artist = focused._artists["figure:title"]
        assert title_artist.get_visible()
        assert title_artist.get_text() == "authored title"
        assert focused.primary_axes.get_title() == "bias=0"
    finally:
        standalone.close()
        facet.close()


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
            key.startswith("facet:1:image:colorbar")
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
            assert f"facet:{index}:image:colorbar" in renderer._artists
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
