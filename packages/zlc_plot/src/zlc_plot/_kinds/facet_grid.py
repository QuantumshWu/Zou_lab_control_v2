"""FacetGrid kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..data_contract import image_axes, live_grid_dimensions
from ..kinds import AxisRef, PlotKind
from zlc_data import DatasetSchema
from ..specs import FacetGridPlot, HistogramPlot, ImagePlot, Reduction
from .base import KindHandler
from .curve import default_spec as curve_default_spec


def render(renderer: Any, payload: Any, state: Any) -> None:
    renderer._update_facets(payload, state)


def build_payload(projection: Any, view: Any, state: Any) -> None:
    spec = projection._spec
    cell = spec.cell
    bins = projection._histogram_bins(view, state) if isinstance(cell, HistogramPlot) else None
    projection._payload = view.facet(spec, bins=bins)


def admits(schema: Any) -> bool:
    # Any dataset can host an authored facet layout; whether one facet axis
    # is *unambiguous* is the narrower default_spec question below.  Keeping
    # the two apart is what lets a frontend show the kind greyed out instead
    # of silently dropping it from the choice list.
    return isinstance(schema, DatasetSchema)


def validate(view: Any, spec: Any) -> None:
    view.validate_facet(spec)


def label_roles(spec: Any) -> tuple[tuple[str, tuple], ...]:
    """Facet labels describe the cell; the grid itself only adds a title."""

    from . import handler_for

    cell_roles = tuple(
        (slot, role)
        for slot, role in handler_for(spec.cell).label_roles(spec.cell)
        if slot != "title"
    )
    return (("title", ("title",)), *cell_roles)


def _data_axes_cell(schema: DatasetSchema) -> ImagePlot | None:
    """The cell a point's own data axes admit: two of them make an image.

    The DATA decides what a cell is, here and only here; a curve cell would
    ACCEPT a camera frame and draw it as a millions-of-points polyline.
    """

    data_axes = tuple(schema.cell_schema.data_axes)
    if len(data_axes) < 2:
        return None
    pair = image_axes(schema)
    if pair is not None:
        x_axis, y_axis = pair
    else:
        # Declared slowest-first, so the last data axis is horizontal.
        x_axis, y_axis = data_axes[-1], data_axes[-2]
    return ImagePlot(
        AxisRef.data(str(x_axis.axis_id)),
        AxisRef.data(str(y_axis.axis_id)),
        reduction=Reduction.MEAN,
    )


def default_spec(schema: Any) -> FacetGridPlot | None:
    """Facet the repeat axis or the outer scan loops over a data-decided cell.

    The cell shows the DENSEST structure the data offers.  Two data axes
    make an image cell (a camera frame).  A SCALAR point over two or more
    live scan dimensions images its two innermost dimensions instead -- the
    scan heatmap, which is what a field map was measured for.  Anything less
    falls back to the curve default.  Degenerate axes (one value) are real
    provenance but not structure, so inference never sees them.

    The facet axis differs by cell on purpose:

    * frame cells: the repeat axis wins whenever it is non-trivial -- each
      shot's frame is the thing to compare; without repeats, the outermost
      live dimension facets the frames.
    * scan-heatmap cells: the OUTER dimensions win and repeats reduce into
      the cell's mean -- the averaged map is the measurement, and a
      per-sweep facet stays one authored change away.  With no outer
      dimension left, repeats are the only lawful facet; without either,
      the plain image kind already IS the picture, so there is no grid.
    * curve cells: the repeat axis, since the curve itself walks the
      innermost dimension.
    """

    if not isinstance(schema, DatasetSchema):
        return None
    live = live_grid_dimensions(schema)
    repeats = schema.repeat_axis.size > 1
    cell = _data_axes_cell(schema)
    if cell is not None:
        if repeats:
            return FacetGridPlot(AxisRef.repeat(), cell)
        if live:
            return FacetGridPlot(AxisRef.point_dimension(live[0]), cell)
        return None
    live_data_axes = tuple(
        axis for axis in schema.cell_schema.data_axes if axis.size > 1
    )
    if not live_data_axes and len(live) >= 2:
        # Declared slowest-first, so the last live dimension is the innermost
        # loop: the horizontal image axis, exactly as the dense image kind
        # reads the same grid.
        heatmap = ImagePlot(
            AxisRef.point_dimension(live[-1]),
            AxisRef.point_dimension(live[-2]),
            reduction=Reduction.MEAN,
        )
        if len(live) >= 3:
            return FacetGridPlot(AxisRef.point_dimension(live[0]), heatmap)
        if repeats:
            return FacetGridPlot(AxisRef.repeat(), heatmap)
        return None
    cell = curve_default_spec(schema)
    if cell is None:
        return None
    if repeats:
        return FacetGridPlot(AxisRef.repeat(), cell)
    if len(live) >= 2:
        return FacetGridPlot(AxisRef.point_dimension(live[0]), cell)
    return None


HANDLER = KindHandler(
    PlotKind.FACET_GRID,
    "Facet grid",
    FacetGridPlot,
    None,
    render,
    build_payload,
    ("kind", "facet"),
    admits,
    default_spec,
    label_roles,
    validate,
)
