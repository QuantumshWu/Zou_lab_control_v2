"""Every facet cell carries the SAME tick marks; only labels are gated.

Regression for the facet tick audit: non-first-column cells used to get
``set_yticks([])`` and non-bottom-row cells ``set_xticks([])`` -- the tick
MARKS were deleted, so eight of nine cells drew as unscaled thumbnails and
the bottom row kept a single centred x tick.  The product rule is one
shared locator/formatter on BOTH axes of EVERY cell, with tick LABELS
appearing only on the boundary (column 0 for y, bottom row for x) via
``tick_params`` -- never via ``set_ticks``.
"""

from __future__ import annotations

import numpy as np
from zlc_plot.ticks import SmartOffsetLocator

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot, REPEAT, SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, FacetGridPlot, ImagePlot
from zlc_plot.session import PlotSession

def _frames_snapshot(points: int = 9, *, revision: int = 1) -> OwnedSnapshot:
    """(repeat=1, points, y, x) camera frames faceted over a scan dimension."""

    table = mapped_domain_from_columns({"bias": [float(v) for v in range(points)]})
    schema = make_dataset_schema(
        repeat_domain(size=1),
        table,
        cell_axes=(
            axis(
                "sy", values=tuple(float(v) for v in range(30)), role=SPATIAL_Y
            ),
            axis(
                "sx", values=tuple(float(v) for v in range(40)), role=SPATIAL_X
            ),
        ),
        dtype=np.uint8,
    )
    values = (
        np.random.default_rng(0)
        .integers(0, 255, (1, points, 30, 40))
        .astype(np.uint8)
    )
    return make_snapshot(schema, values, revision=revision)

_FRAMES_SPEC = FacetGridPlot(
    AxisRef.point("bias"),
    ImagePlot(AxisRef.cell_data("sx"), AxisRef.cell_data("sy")),
)

def _scalar_scan_snapshot(size: int = 3, *, partial: bool = False) -> tuple[FacetGridPlot, OwnedSnapshot]:
    """3D scalar scan whose facet holds scan-heatmap (ImagePlot) cells."""

    from dataclasses import replace

    rows = np.indices((size, size, size)).reshape(3, -1).T
    coordinates = np.linspace(-500.0, 500.0, size) if partial else np.arange(size, dtype=float)
    table = mapped_domain_from_columns(
        {
            "va": rows[:, 0].astype(float),
            "vb": coordinates[rows[:, 1]],
            "vc": coordinates[rows[:, 2]],
        }
    )
    if partial:
        table = replace(table, axes=(
            replace(table.axes[0], name="pgcwaiting.da_bias_x"), *table.axes[1:],
        ))
    schema = make_dataset_schema(
        repeat_domain(size=2),
        table,
        dtype=np.float64,
    )
    values = np.random.default_rng(1).normal(size=(2, len(rows)))
    spec = FacetGridPlot(
        AxisRef.point("va"),
        ImagePlot(
            AxisRef.point("vc"), AxisRef.point("vb")
        ),
    )
    valid = None
    if partial:
        valid = np.zeros(values.shape, dtype=bool)
        valid[0, 0] = True
    return spec, make_snapshot(schema, values, revision=1, validity=valid)

def _visible_cells(session: PlotSession) -> list[tuple[int, object]]:
    renderer = session._renderer
    renderer.draw()  # settle tick artists at final positions
    return [
        (index, axis)
        for index, axis in enumerate(renderer.axes["facet_cell"])
        if axis.get_visible()
    ]

def _marked_ticks(axis_obj) -> list:
    return [
        tick
        for tick in axis_obj._update_ticks()
        if tick.tick1line.get_visible()
    ]

def _labelled_ticks(axis_obj) -> list:
    return [
        tick
        for tick in axis_obj._update_ticks()
        if tick.label1.get_visible() and tick.label1.get_text()
    ]

def _assert_shared_marks_boundary_labels(session: PlotSession) -> None:
    renderer = session._renderer
    rows, columns = renderer.plan.facet_shape
    cells = _visible_cells(session)
    assert cells, "the facet drew no cells"
    x_values: set[tuple[float, ...]] = set()
    y_values: set[tuple[float, ...]] = set()
    for index, axis in cells:
        row, column = divmod(index, columns)
        # Marks: every cell, both axes, from the ONE shared locator -- the
        # same policy a full-size panel gets, spending the width a cell has.
        assert isinstance(axis.xaxis.get_major_locator(), SmartOffsetLocator)
        assert isinstance(axis.yaxis.get_major_locator(), SmartOffsetLocator)
        assert len(_marked_ticks(axis.xaxis)) > 0, f"cell {index} has no x ticks"
        assert len(_marked_ticks(axis.yaxis)) > 0, f"cell {index} has no y ticks"
        x_values.add(tuple(map(float, axis.get_xticks())))
        y_values.add(tuple(map(float, axis.get_yticks())))
        # Labels: boundary cells only.
        label_left = column == 0
        label_bottom = row == rows - 1 or index + columns >= len(cells)
        assert bool(_labelled_ticks(axis.yaxis)) == label_left, (
            f"cell {index}: y labels must appear exactly on column 0"
        )
        assert bool(_labelled_ticks(axis.xaxis)) == label_bottom, (
            f"cell {index}: x labels must appear exactly on the bottom row"
        )
        for tick in _labelled_ticks(axis.xaxis):
            assert tick.label1.get_horizontalalignment() == "center"
            assert tick.label1.get_position()[0] == tick.get_loc()
        for tick in _labelled_ticks(axis.yaxis):
            assert tick.label1.get_verticalalignment() == "center_baseline"
            assert tick.label1.get_position()[1] == tick.get_loc()
    # Identical data domains per fixture, so the shared locator must place
    # the SAME tick values in every cell -- the marks are comparable.
    assert len(x_values) == 1
    assert len(y_values) == 1

def test_the_ladder_is_walked_once_for_the_whole_grid() -> None:
    """One answer per question, not one per cell.

    Every cell of a facet grid carries the same span in the same size of
    box, so the ladder has one answer for the grid's x and one for its y --
    two more where a boundary cell's labels change the room it has.  Each
    cell owning its own locator meant walking the same ladder once per
    cell: measured at 88 ms of a sixty-four cell first frame, 1536 calls
    for four distinct answers.
    """

    original = SmartOffsetLocator._unit

    def walks_for(points: int) -> int:
        walked: list[tuple[float, float]] = []

        def counted(self, lower, upper):
            walked.append((float(lower), float(upper)))
            return original(self, lower, upper)

        SmartOffsetLocator._unit = counted
        try:
            session = PlotSession(
                _frames_snapshot(points=points), _FRAMES_SPEC, size="8x8"
            )
            try:
                assert len(_visible_cells(session)) == points
                _assert_shared_marks_boundary_labels(session)
            finally:
                session.close()
        finally:
            SmartOffsetLocator._unit = original
        return len(walked)

    nine = walks_for(9)
    twenty_five = walks_for(25)
    assert twenty_five == nine, (nine, twenty_five)
    # An answer per axis, per distinct room -- a boundary cell carries
    # labels and an interior one does not -- and nothing per cell.
    assert nine <= 8, nine


def test_frames_facet_cells_share_tick_marks_and_gate_labels() -> None:
    session = PlotSession(_frames_snapshot(), _FRAMES_SPEC, size="8x8")
    try:
        _assert_shared_marks_boundary_labels(session)
    finally:
        session.close()

def test_scan_heatmap_facet_cells_share_tick_marks_and_gate_labels() -> None:
    spec, snapshot = _scalar_scan_snapshot()
    session = PlotSession(snapshot, spec, size="8x8")
    try:
        _assert_shared_marks_boundary_labels(session)
    finally:
        session.close()

def test_overview_cell_ticks_have_one_owner_across_frames(monkeypatch) -> None:
    """A cell's tick configuration is installed ONCE, not once per authority.

    Routing facet cells through the standalone image render brought the image
    kind's own spatial tick budget with it, and the grid's shared 3-tick
    locator is applied right after -- two authorities writing the same
    ``_zlc_tick_signature``.  Each then saw the other's value and reinstalled
    its locator on every frame, resetting every cell's tick artists twice per
    revision, which is exactly what the signature guard exists to prevent.
    """

    session = PlotSession(_frames_snapshot(), _FRAMES_SPEC, size="8x8")
    try:
        cells = [
            axis
            for axis in session._renderer.axes["facet_cell"]
            if axis.get_visible()
        ]
        assert cells
        before = [
            (id(axis.xaxis.get_major_locator()), id(axis.yaxis.get_major_locator()))
            for axis in cells
        ]
        session.update_data(_frames_snapshot(revision=2))
        after = [
            (id(axis.xaxis.get_major_locator()), id(axis.yaxis.get_major_locator()))
            for axis in cells
        ]
        assert after == before
        # ...and the owner is still the grid: shared marks, gated labels.
        _assert_shared_marks_boundary_labels(session)
    finally:
        session.close()

    # The reported 50-cell scan: only the first point is valid. Final Image
    # squares are narrower than their grid slots, so titles cannot spend the
    # slot's extra width over the Y-label gutter.
    spec, snapshot = _scalar_scan_snapshot(50, partial=True)
    session = PlotSession(snapshot, spec, size="4x4", device_pixel_ratio=3)
    try:
        renderer = session._renderer

        def consistent():
            cells = [axis for axis in renderer.axes["facet_cell"] if axis.get_visible()]
            for name in ("xaxis", "yaxis"):
                assert len({getattr(axis, name).get_major_locator().drawn_pt for axis in cells}) == 1
            assert len({round(float(axis.bbox.width), 6) for axis in cells}) == 1
            assert all(abs(axis.bbox.width - axis.bbox.height) < 1e-6 for axis in cells)
            for axis in cells:
                transform = session._axis_transform_for_axis(axis)
                nx, ny = transform.display_to_normalized(0.0, 0.0)
                point = transform.canonical_from_normalized(nx, ny)
                assert abs(point.x) < 1e-8 and abs(point.y) < 1e-8
            titles = renderer._artists["facet:chrome_titles"]
            assert len({title.get_fontsize() for title in titles}) == 1
            draw = renderer.figure.canvas.get_renderer()
            boxes = [title.get_window_extent(draw) for title in titles]
            labels = renderer._artists["facet:chrome_labels"].texts
            assert not any(
                box.overlaps(label.get_window_extent(draw))
                for box in boxes for label in labels
                if label.get_visible() and label.get_text()
            )
            assert not any(box.overlaps(other) for i, box in enumerate(boxes) for other in boxes[i + 1:])

        consistent()
        assert any(title.get_text().endswith("\N{HORIZONTAL ELLIPSIS}")
                   for title in renderer._artists["facet:chrome_titles"])
        # A lane can have different limits. A common font must remain common
        # when its own cached placement is queried on the next draw.
        cells = renderer.axes["facet_cell"]
        cells[0].set_ylim(30.0, -5.0)
        renderer.draw()
        consistent()
        renderer.draw()
        consistent()
        for axis in cells:
            axis.set_ylim(30.0, -5.0)
        renderer.draw()
        consistent()
        assert cells[0].yaxis.get_major_locator().drawn_pt > 3.0
        session.focus_facet(0)
        session.show_facet_overview()
        consistent()
        session.set_size("8x8")
        consistent()
        # Restored ±500 endpoints have a real X/Y text collision at the
        # bottom-left corner; larger data boxes do not move those anchors.
        assert renderer.axes["facet_cell"][0].yaxis.get_major_locator().drawn_pt >= 3.0
        refreshed = []
        original = renderer._refresh_facet_cell_chrome

        def refresh(*args):
            refreshed.append(True)
            return original(*args)

        monkeypatch.setattr(renderer, "_refresh_facet_cell_chrome", refresh)
        session.update_data(make_snapshot(
            snapshot.block.schema, snapshot.block.values, revision=2,
            validity=snapshot.expanded_validity(),
        ))
        assert not refreshed, "unchanged live geometry must not remeasure collisions"
        consistent()
    finally:
        session.close()

def test_boundary_label_gating_refires_after_focus_round_trip() -> None:
    """Focus installs its own locators and full labels; the overview pass
    must re-fire the shared-marks/boundary-labels configuration through the
    signature mechanism instead of trusting a stale cached tuple."""

    session = PlotSession(_frames_snapshot(), _FRAMES_SPEC, size="8x8")
    try:
        session.focus_facet(4)  # an interior cell: no labels in overview
        session.show_facet_overview()
        _assert_shared_marks_boundary_labels(session)
    finally:
        session.close()

def test_declared_coordinate_names_survive_the_overview_and_the_focus() -> None:
    """A producer's coordinate labels tick the x axis BY NAME on every surface.

    The native overview cell skipped the cell painter and the grid's tick
    pass then put numbers where the dataset declared ``zero, one, two``; the
    focused cell showed the names.  One tick entry serves the standalone
    curve, the overview cell -- natively painted or not -- and the focused
    cell, and no later pass replaces the names.
    """

    from dataclasses import replace

    from zlc_plot import CurvePlot

    points = mapped_domain_from_columns(
        {"facet": np.repeat([0, 1], 3), "x": np.tile(np.arange(3), 2)}
    )
    points = replace(
        points,
        axes=tuple(
            replace(item, coordinate_labels=("zero", "one", "two"))
            if item.name == "x"
            else item
            for item in points.axes
        ),
    )
    schema = make_dataset_schema(repeat_domain(size=1), points)
    session = PlotSession(
        make_snapshot(schema, np.arange(6.0)[None, :], revision=0),
        FacetGridPlot(AxisRef.point("facet"), CurvePlot(AxisRef.point("x"))),
    )
    try:

        def names(index: int) -> list[str]:
            session._renderer.draw()
            axis = session._renderer.axes["facet_cell"][index]
            return [text.get_text() for text in axis.get_xticklabels()]

        assert names(0) == ["zero", "one", "two"]
        session.update_data(
            make_snapshot(schema, np.arange(6.0)[None, :] + 1.0, revision=1)
        )
        assert names(0) == ["zero", "one", "two"]
        session.focus_facet(0)
        assert names(0) == ["zero", "one", "two"]
        session.show_facet_overview()
        assert names(1) == ["zero", "one", "two"]
    finally:
        session.close()
