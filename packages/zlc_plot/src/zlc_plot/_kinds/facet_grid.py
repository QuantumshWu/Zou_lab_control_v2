"""FacetGrid kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import PlotKind
from zlc_data import DatasetSchema
from ..specs import CurvePlot, FacetGridPlot, HistogramPlot, Reduction
from .base import KindHandler
from . import defaults


def render(renderer: Any, payload: Any, state: Any, *, axes: Any, key: str) -> None:
    # A grid is not one surface: it resolves its cells' axes and keys from
    # the renderer's painted surfaces and calls the CELL kind's own render
    # once per cell.  The selected cell arrives here as ``axes``/``key``.
    del axes, key
    renderer._update_facets(payload, state)


def build_payload(projection: Any, view: Any, state: Any) -> None:
    spec = projection._spec
    cell = spec.cell
    bins = None
    window = 1
    if isinstance(cell, HistogramPlot):
        # The cell's own vocabulary, honoured: a window pools the last N
        # shots into each cell, and a reduced fate collapses the named axes
        # inside each cell before its values are binned.  The shared edges
        # are then taken from the pools the cells will actually bin, which is
        # what keeps them comparable AND on scale.
        window = int(state["window"])
        pool, pool_valid = view.facet_histogram_pool(spec, window=window)
        bins = projection._histogram_bins(
            view, state, binned_values=pool, binned_valid=pool_valid
        )
    uncertainty = bool(
        isinstance(cell, CurvePlot)
        and state["uncertainty"]
        and cell.reduction is Reduction.MEAN
    )
    projection._payload = view.facet(
        spec, bins=bins, uncertainty=uncertainty, window=window
    )


def admits(schema: Any) -> bool:
    # Any dataset can host an authored facet layout; what the grid shows
    # unasked is the table's answer (``default_spec``).  Keeping the two apart
    # is what lets a frontend show the kind greyed out instead of silently
    # dropping it from the choice list.
    return isinstance(schema, DatasetSchema)


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
    """The grid this dataset shows unasked: one reading of the table.

    The cell shows the densest structure the data offers and the grid
    faces the first live position axis the cell leaves free; with nothing
    left to face it is one cell.  See :mod:`zlc_plot._kinds.defaults`.
    """

    return defaults.default_spec(schema, PlotKind.FACET_GRID)


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
)
