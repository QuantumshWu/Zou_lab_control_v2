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
from matplotlib.ticker import MaxNLocator

from data_factory import (
    Axis,
    DatasetSchema,
    DatasetSnapshot,
    PointTable,
    PointTopology,
)
from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, FacetGridPlot, ImagePlot
from zlc_plot.session import PlotSession


def _frames_snapshot(points: int = 9, *, revision: int = 1) -> DatasetSnapshot:
    """(repeat=1, points, y, x) camera frames faceted over a scan dimension."""

    bias = Axis.create("bias", values=[float(v) for v in range(points)])
    table = PointTable.from_columns({"bias": [float(v) for v in range(points)]})
    topology = PointTopology.from_cartesian((bias,), point_table=table)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        table,
        data_axes=(
            Axis.create(
                "sy", values=tuple(float(v) for v in range(30)), role=SPATIAL_Y
            ),
            Axis.create(
                "sx", values=tuple(float(v) for v in range(40)), role=SPATIAL_X
            ),
        ),
        point_topology=topology,
        dtype=np.uint8,
    )
    values = (
        np.random.default_rng(0)
        .integers(0, 255, (1, points, 30, 40))
        .astype(np.uint8)
    )
    return DatasetSnapshot(schema, values, revision=revision)


_FRAMES_SPEC = FacetGridPlot(
    AxisRef.point_dimension("bias"),
    ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")),
)


def _scalar_scan_snapshot() -> tuple[FacetGridPlot, DatasetSnapshot]:
    """3D scalar scan whose facet holds scan-heatmap (ImagePlot) cells."""

    import itertools

    a = [0.0, 1.0, 2.0]
    rows = list(itertools.product(a, a, a))
    table = PointTable.from_columns(
        {
            "va": [r[0] for r in rows],
            "vb": [r[1] for r in rows],
            "vc": [r[2] for r in rows],
        }
    )
    dims = tuple(Axis.create(name, values=a) for name in ("va", "vb", "vc"))
    topology = PointTopology.from_cartesian(dims, point_table=table)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=2),
        table,
        point_topology=topology,
        dtype=np.float64,
    )
    values = np.random.default_rng(1).normal(size=(2, len(rows)))
    spec = FacetGridPlot(
        AxisRef.point_dimension("va"),
        ImagePlot(
            AxisRef.point_dimension("vc"), AxisRef.point_dimension("vb")
        ),
    )
    return spec, DatasetSnapshot(schema, values, revision=1)


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
        # Marks: every cell, both axes, from the ONE shared locator.
        assert isinstance(axis.xaxis.get_major_locator(), MaxNLocator)
        assert isinstance(axis.yaxis.get_major_locator(), MaxNLocator)
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
    # Identical data domains per fixture, so the shared locator must place
    # the SAME tick values in every cell -- the marks are comparable.
    assert len(x_values) == 1
    assert len(y_values) == 1


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


def test_overview_cell_ticks_have_one_owner_across_frames() -> None:
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
