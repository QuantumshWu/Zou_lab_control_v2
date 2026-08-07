"""FacetGrid kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import AxisRef, PlotKind
from zlc_data import DatasetSchema
from ..specs import FacetGridPlot, HistogramPlot
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


def default_spec(schema: Any) -> FacetGridPlot | None:
    """Facet the repeat axis, or the outer scan loop, over a default cell.

    The repeat axis is the one facet dimension every acquisition shares, so
    it wins whenever it is non-trivial.  Without repeats, a Cartesian scan
    faceted by its outermost (slowest) dimension complements the curve
    default, which walks the innermost dimension as x.  Anything else needs
    an authored choice.
    """

    if not isinstance(schema, DatasetSchema):
        return None
    cell = curve_default_spec(schema)
    if cell is None:
        return None
    if schema.repeat_axis.size > 1:
        facet = AxisRef.repeat()
    elif (
        schema.grid_topology is not None
        and len(schema.grid_topology.dimension_ids) >= 2
    ):
        facet = AxisRef.point_dimension(str(schema.grid_topology.dimension_ids[0]))
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
