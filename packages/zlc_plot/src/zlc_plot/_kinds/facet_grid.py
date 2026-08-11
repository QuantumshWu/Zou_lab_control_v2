"""FacetGrid kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..data_contract import image_axes
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
    """Facet the repeat axis, or the outer scan loop, over a data-decided cell.

    The cell shows what is INSIDE one point: two data axes make an image
    cell, anything less falls back to the curve default.  The repeat axis is
    the one facet dimension every acquisition shares, so it wins whenever it
    is non-trivial.  Without repeats, a Cartesian scan facets its outermost
    (slowest) dimension -- for image cells even a SINGLE scan dimension is a
    legitimate facet, because the cell needs nothing left to walk; a curve
    cell walks a dimension itself, so it still needs a second one.
    """

    if not isinstance(schema, DatasetSchema):
        return None
    cell = _data_axes_cell(schema) or curve_default_spec(schema)
    if cell is None:
        return None
    dimensions = (
        ()
        if schema.grid_topology is None
        else schema.grid_topology.dimension_ids
    )
    if schema.repeat_axis.size > 1:
        facet = AxisRef.repeat()
    elif len(dimensions) >= 2 or (
        len(dimensions) == 1 and isinstance(cell, ImagePlot)
    ):
        facet = AxisRef.point_dimension(str(dimensions[0]))
    else:
        return None
    return FacetGridPlot(facet, cell)


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
