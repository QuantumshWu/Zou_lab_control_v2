"""What an unauthored panel shows: the auto ladder for grids and images.

The rules under test ARE the product decision: the cell shows the densest
structure the data offers (frame > scan heatmap > curve), degenerate axes
are invisible to inference, and the facet axis follows the cell -- repeats
for frames, outer dimensions for heatmaps.
"""

from __future__ import annotations

import numpy as np

from zlc_plot._kinds.facet_grid import default_spec as facet_default
from zlc_plot._kinds.image import default_spec as image_default
from zlc_plot.data_contract import live_grid_dimensions
from zlc_plot.kinds import AxisRef
from zlc_plot.specs import CurvePlot, FacetGridPlot, ImagePlot, Reduction

from data_factory import Axis, DatasetSchema, PointTable, PointTopology


def _scan_schema(dim_sizes: dict[str, int], *, repeats: int = 1, data_axes=()):
    """One Cartesian scan dataset: named dimensions, optional cell axes."""

    dims = tuple(
        Axis.create(name, values=list(range(size)))
        for name, size in dim_sizes.items()
    )
    rows = list(np.ndindex(*(size for size in dim_sizes.values())))
    columns = {
        name: [row[index] for row in rows]
        for index, name in enumerate(dim_sizes)
    }
    table = PointTable.from_columns(columns)
    return DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        table,
        data_axes=tuple(data_axes),
        point_topology=PointTopology.from_cartesian(dims, point_table=table),
    )


def _frame_axes(height: int = 4, width: int = 6):
    from zlc_data import SPATIAL_X, SPATIAL_Y

    return (
        Axis.create("y", size=height, role=SPATIAL_Y),
        Axis.create("x", size=width, role=SPATIAL_X),
    )


def test_degenerate_dimensions_are_invisible_to_inference() -> None:
    schema = _scan_schema({"a": 1, "b": 4, "c": 5})
    assert live_grid_dimensions(schema) == ("b", "c")


def test_a_scalar_multi_dimension_scan_cells_its_two_innermost_dims() -> None:
    """Three scan axes: the outermost facets, the inner two ARE the cell.

    This is the scan the plan editor authors every day; a curve cell here
    walked one dimension and averaged the other away.
    """

    spec = facet_default(_scan_schema({"a": 3, "b": 4, "c": 5}))
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.point_dimension("a")
    cell = spec.cell
    assert isinstance(cell, ImagePlot)
    assert cell.x == AxisRef.point_dimension("c")
    assert cell.y == AxisRef.point_dimension("b")
    assert cell.reduction is Reduction.MEAN


def test_a_scalar_two_dimension_scan_with_repeats_facets_them() -> None:
    """No outer dimension left: repeats are the only lawful facet."""

    spec = facet_default(_scan_schema({"a": 3, "b": 4}, repeats=5))
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.repeat()
    assert isinstance(spec.cell, ImagePlot)
    assert spec.cell.x == AxisRef.point_dimension("b")
    assert spec.cell.y == AxisRef.point_dimension("a")


def test_a_scalar_two_dimension_scan_without_repeats_is_no_grid() -> None:
    """The plain image kind already IS that picture; a grid adds nothing."""

    schema = _scan_schema({"a": 3, "b": 4})
    assert facet_default(schema) is None
    image = image_default(schema)
    assert isinstance(image, ImagePlot)
    assert image.x == AxisRef.point_dimension("b")
    assert image.y == AxisRef.point_dimension("a")


def test_repeats_reduce_into_the_heatmap_when_an_outer_dimension_exists() -> None:
    """The averaged map is the measurement; per-sweep stays one edit away."""

    spec = facet_default(_scan_schema({"a": 3, "b": 4, "c": 5}, repeats=6))
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.point_dimension("a")
    assert isinstance(spec.cell, ImagePlot)


def test_a_degenerate_data_axis_still_counts_as_a_scalar_point() -> None:
    """survival_rate carries a one-pair axis; the heatmap must still win."""

    pairs = (Axis.create("pair", size=1),)
    spec = facet_default(
        _scan_schema({"a": 3, "b": 4}, repeats=5, data_axes=pairs)
    )
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.repeat()
    assert isinstance(spec.cell, ImagePlot)
    assert spec.cell.x == AxisRef.point_dimension("b")


def test_frame_cells_keep_their_repeat_facet_and_live_dimension_fallback() -> None:
    frames_with_repeats = _scan_schema(
        {"a": 3}, repeats=4, data_axes=_frame_axes()
    )
    spec = facet_default(frames_with_repeats)
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.repeat()
    assert isinstance(spec.cell, ImagePlot)
    assert spec.cell.x == AxisRef.data("x")
    assert spec.cell.y == AxisRef.data("y")

    frames_scanned = _scan_schema(
        {"a": 1, "b": 3}, data_axes=_frame_axes()
    )
    spec = facet_default(frames_scanned)
    assert isinstance(spec, FacetGridPlot)
    # The degenerate "a" is invisible: the LIVE dimension facets the frames.
    assert spec.facet == AxisRef.point_dimension("b")


def test_a_site_resolved_scan_keeps_the_grouped_curve_cell() -> None:
    """One live data axis is per-site structure, not a scalar: curves stay.

    Imaging the scan dimensions here would average the sites away inside
    every pixel -- the exact information a site-resolved signal exists for.
    """

    sites = (Axis.create("site", size=7),)
    spec = facet_default(_scan_schema({"a": 3, "b": 4}, data_axes=sites))
    assert isinstance(spec, FacetGridPlot)
    assert spec.facet == AxisRef.point_dimension("a")
    cell = spec.cell
    assert isinstance(cell, CurvePlot)
    assert cell.x == AxisRef.point_dimension("b")
    assert cell.group == AxisRef.data("site")


def test_the_image_kind_ignores_degenerate_grid_dimensions() -> None:
    """A one-value dimension must not become a one-pixel-thick grid image."""

    schema = _scan_schema({"a": 1, "b": 3}, data_axes=_frame_axes())
    image = image_default(schema)
    assert isinstance(image, ImagePlot)
    assert image.x == AxisRef.data("x")
    assert image.y == AxisRef.data("y")
