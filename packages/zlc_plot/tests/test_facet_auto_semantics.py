"""What an unauthored panel shows: the auto ladder for grids and images.

The rules under test ARE the product decision: the cell shows the densest
structure the data offers (frame > scan heatmap > curve), degenerate axes
are invisible to inference, and the facet axis follows the cell.  Repeat is
measurement history, so only an explicitly authored grid may facet it.
"""

from __future__ import annotations

import numpy as np
from zlc_data import READOUT_EVENT, REPEAT, SITE

from zlc_plot._kinds.facet_grid import default_spec as facet_default
from zlc_plot._kinds.image import default_spec as image_default
from zlc_plot.data_contract import classify_axes
from zlc_plot.kinds import AxisRef
from zlc_plot.specs import CurvePlot, FacetGridPlot, ImagePlot, Reduction

from data_factory import (
    axis,
    make_dataset_schema,
    mapped_domain_from_columns,
    repeat_domain,
)


def _scan_schema(dim_sizes: dict[str, int], *, repeats: int = 1, cell_axes=()):
    """One Cartesian scan dataset: named dimensions, optional cell axes."""

    rows = list(np.ndindex(*(size for size in dim_sizes.values())))
    columns = {
        name: [row[index] for row in rows]
        for index, name in enumerate(dim_sizes)
    }
    table = mapped_domain_from_columns(columns)
    return make_dataset_schema(
        repeat_domain(size=repeats),
        table,
        cell_axes=tuple(cell_axes),
    )


def _frame_axes(height: int = 4, width: int = 6):
    from zlc_data import SPATIAL_X, SPATIAL_Y

    return (
        axis("y", size=height, role=SPATIAL_Y),
        axis("x", size=width, role=SPATIAL_X),
    )


def test_degenerate_dimensions_are_invisible_to_inference() -> None:
    schema = _scan_schema({"a": 1, "b": 4, "c": 5})
    families = classify_axes(schema)
    assert [ref.axis_id for ref, _size in families.scan] == ["a", "b", "c"]
    assert [ref.axis_id for ref, _size in families.live_scan()] == ["b", "c"]


def test_a_scalar_multi_dimension_scan_cells_its_two_innermost_dims() -> None:
    """Three scan axes: the outermost facets, the inner two ARE the cell.

    This is the scan the plan editor authors every day; a curve cell here
    walked one dimension and averaged the other away.
    """

    spec = facet_default(_scan_schema({"a": 3, "b": 4, "c": 5}))
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.point("a")
    cell = spec.cell
    assert isinstance(cell, ImagePlot)
    assert cell.x == AxisRef.point("c")
    assert cell.y == AxisRef.point("b")
    assert cell.reduction is Reduction.MEAN


def test_a_two_dimension_heatmap_does_not_automatically_facet_repeat() -> None:
    """The heatmap consumes both scan dimensions; repeat remains reduced.

    Having many repeats must not turn acquisition history into a layout axis.
    With no non-repeat dimension left the FacetGrid is one heatmap cell.
    """

    for repeats in (1, 5):
        schema = _scan_schema({"a": 3, "b": 4}, repeats=repeats)
        spec = facet_default(schema)
        assert isinstance(spec, FacetGridPlot)
        assert spec.facet is None
        assert isinstance(spec.cell, ImagePlot)
        image = image_default(schema)
        assert isinstance(image, ImagePlot)
        assert image.x == AxisRef.point("b")
        assert image.y == AxisRef.point("a")


def test_repeats_reduce_into_the_heatmap_when_an_outer_dimension_exists() -> None:
    """The averaged map is the measurement; per-sweep stays one edit away."""

    spec = facet_default(_scan_schema({"a": 3, "b": 4, "c": 5}, repeats=6))
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.point("a")
    assert isinstance(spec.cell, ImagePlot)


def test_a_degenerate_data_axis_still_counts_as_a_scalar_point() -> None:
    """A degenerate cell axis does not justify faceting repeat."""

    pairs = (axis("pair", size=1),)
    spec = facet_default(
        _scan_schema({"a": 3, "b": 4}, repeats=5, cell_axes=pairs)
    )
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet is None
    assert isinstance(spec.cell, ImagePlot)


def test_frame_cells_facet_the_scan_and_leave_repeats_to_the_reduction() -> None:
    """A scanned frame belongs to its scan point, not to its sweep.

    Pooling repeats is the reduction the operator declared and says so on
    the panel; pooling a scan point silently averages seven different
    physical settings into one picture, and the scan becomes invisible in
    the plot of the scan.  So the scan facets and the repeats reduce.
    """

    frames_with_repeats = _scan_schema(
        {"a": 3}, repeats=4, cell_axes=_frame_axes()
    )
    spec = facet_default(frames_with_repeats)
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.point("a")
    assert isinstance(spec.cell, ImagePlot)
    assert spec.cell.x == AxisRef.cell_data("x")
    assert spec.cell.y == AxisRef.cell_data("y")

    frames_scanned = _scan_schema(
        {"a": 1, "b": 3}, cell_axes=_frame_axes()
    )
    spec = facet_default(frames_scanned)
    assert isinstance(spec, FacetGridPlot)
    # The degenerate "a" is invisible: the LIVE dimension facets the frames.
    assert spec.facet == AxisRef.point("b")

    # A grid has ONE facet axis, so a multidimensional scan defaults to its
    # outermost real axis.  The flattened row ordinal is not a fourth axis;
    # the remaining dimensions stay explicit in the fate table and reduce
    # until the operator assigns them differently.
    frames_scanned_twice = _scan_schema(
        {"a": 2, "b": 3}, repeats=4, cell_axes=_frame_axes()
    )
    spec = facet_default(frames_scanned_twice)
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.point("a")
    assert isinstance(spec.cell, ImagePlot)

    # With nothing else live, repeat is still history rather than an automatic
    # layout axis.
    frames_repeated_only = _scan_schema(
        {"a": 1}, repeats=4, cell_axes=_frame_axes()
    )
    spec = facet_default(frames_repeated_only)
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet is None
    assert isinstance(spec.cell, ImagePlot)


def test_a_camera_cycle_facets_its_frames_from_the_point_axis() -> None:
    """(cycles, frames, y, x): frames ARE points, and the grid shows them.

    A cycle with no scan topology publishes its frames on the point axis;
    the facet walks that axis so frame_0 | frame_1 sit side by side --
    with repeats averaged into each frame's cell, because the cycle's
    structure, not the shot count, is what the acquisition authored.
    """

    frames = mapped_domain_from_columns(
        {"frame": [0, 1]}, roles={"frame": READOUT_EVENT}
    )
    single = make_dataset_schema(
        repeat_domain(size=1),
        frames,
        cell_axes=_frame_axes(),
    )
    spec = facet_default(single)
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.point("frame")
    assert isinstance(spec.cell, ImagePlot)
    assert spec.cell.x == AxisRef.cell_data("x")
    assert spec.cell.y == AxisRef.cell_data("y")

    with_cycles = make_dataset_schema(
        repeat_domain(size=30),
        frames,
        cell_axes=_frame_axes(),
    )
    spec = facet_default(with_cycles)
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.point("frame")

    one_frame = make_dataset_schema(
        repeat_domain(size=30),
        mapped_domain_from_columns(
            {"frame": [0]}, roles={"frame": READOUT_EVENT}
        ),
        cell_axes=_frame_axes(),
    )
    spec = facet_default(one_frame)
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.point("frame")


def test_a_site_resolved_scan_spends_position_before_content() -> None:
    """Two live scan dimensions are the heatmap; the sites are reduced.

    Position is spent before content: a field map of a per-site quantity
    opens as the scan's heatmap with the sites averaged, which is what the
    scan was measured for.  The per-site view is one cell-kind switch away
    -- a named curve cell walks the inner dimension and groups the sites.
    """

    from zlc_plot import PlotKind, fitting_spec

    sites = (axis("site", size=7, role=SITE),)
    schema = _scan_schema({"a": 3, "b": 4}, cell_axes=sites)
    spec = facet_default(schema)
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet is None
    cell = spec.cell
    assert isinstance(cell, ImagePlot)
    assert cell.x == AxisRef.point("b")
    assert cell.y == AxisRef.point("a")
    curves = fitting_spec(schema, PlotKind.FACET_GRID, cell=PlotKind.CURVE)
    assert curves.facet == AxisRef.point("a")
    assert curves.cell.x == AxisRef.point("b")
    assert curves.cell.group == AxisRef.cell_data("site")


def test_the_image_kind_ignores_degenerate_grid_dimensions() -> None:
    """A one-value dimension must not become a one-pixel-thick grid image."""

    schema = _scan_schema({"a": 1, "b": 3}, cell_axes=_frame_axes())
    image = image_default(schema)
    assert isinstance(image, ImagePlot)
    assert image.x == AxisRef.cell_data("x")
    assert image.y == AxisRef.cell_data("y")
