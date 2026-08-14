"""FacetGrid kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..data_contract import image_axes, live_grid_dimensions
from ..kinds import AxisRef, PlotKind
from zlc_data import DatasetSchema
from ..specs import CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot, Reduction
from .base import KindHandler
from .curve import default_spec as curve_default_spec


def render(renderer: Any, payload: Any, state: Any, *, axes: Any, key: str) -> None:
    # A grid is not one surface: it resolves its cells' axes and keys from
    # the renderer's painted surfaces and calls the CELL kind's own render
    # once per cell.  The selected cell arrives here as ``axes``/``key``.
    del axes, key
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


def _facet_axis(schema: Any, cell: Any) -> AxisRef | None:
    """What varies from cell to cell: the thing one plot would have to pool.

    One rule, and it depends on what the CELL already consumes, because a
    grid has exactly one facet axis and the two must not claim the same
    thing.  A curve cell walks a scan dimension itself, so the grid faces
    the dimension outside it; an image cell consumes the data axes, leaving
    every scan point free to have its own cell.

    Frames of a cycle and points of a scan are different measurements and
    get a cell each; repeats are the same measurement again, and pooling
    them is the reduction the operator declared, so they face a grid only
    when nothing else varies.  When nothing varies at all, the point axis is
    faced anyway and the grid holds one cell: a grid is never inferred, so
    reaching here means somebody asked for one, and a cycle of one frame is
    still a cycle.  ``None`` is kept for data with no point table to face.
    """

    live = live_grid_dimensions(schema)
    repeats = schema.repeat_axis.size > 1
    dense = tuple(axis for axis in schema.cell_schema.data_axes if axis.size > 1)
    # No scan topology and several point rows: the point axis IS the thing
    # measured several times -- the frames of a camera cycle, the frames
    # occupancy judged.
    points_vary = schema.grid_topology is None and schema.point_table.row_count > 1
    point_axis = (
        AxisRef.point(str(schema.point_table.columns[0].coordinate_id))
        if points_vary
        else None
    )
    if isinstance(cell, CurvePlot):
        # A curve walks something itself.  It can hand the point axis to the
        # grid only when the dense data gives it another one to walk; with a
        # scalar per point the curve IS the walk over those points, and what
        # is left to face are the sweeps.
        if point_axis is not None and dense:
            return point_axis
        if len(live) >= 2:
            # The curve walks the innermost dimension; the grid takes the
            # outermost.
            return AxisRef.point_dimension(live[0])
        return AxisRef.repeat() if repeats else None
    if point_axis is not None:
        return point_axis
    if len(live) == 1:
        return AxisRef.point_dimension(live[0])
    if len(live) >= 2:
        # One facet axis, two or more scanned dimensions: every point
        # measured gets its cell rather than one dimension being folded
        # into the other.
        return AxisRef.point_rows()
    if repeats:
        return AxisRef.repeat()
    # Nothing VARIES -- but a grid was asked for, and this function is only
    # ever reached by an explicit request (inference never offers a grid).
    # A cycle of one frame is still a cycle: face its point axis and draw
    # the one cell, so a camera measurement opens the same way whatever its
    # frames per cycle, instead of silently becoming a bare image at one.
    if schema.grid_topology is None and schema.point_table.columns:
        return AxisRef.point(str(schema.point_table.columns[0].coordinate_id))
    return None


def cell_within_one_cell(schema: Any, facet: Any, cell: Any) -> Any | None:
    """The cell spec as ONE cell of a grid faceted by ``facet`` sees it.

    Every cell kind defaulted in isolation reads the whole dataset, so it
    reaches for the very structure the facets walk ACROSS: an image cell
    over a scan claims the scan grid as its surface, a curve cell walks the
    point axis that is now one value per cell.  Both are right alone and
    wrong inside a cell, and the composition is one rule -- so it lives once,
    here, rather than being re-derived by whoever composes next.

    What remains inside a cell is the dense data.  A cell that cannot be
    moved off the facet axis is a refusal, not a guess.
    """

    dense = tuple(axis for axis in schema.cell_schema.data_axes if axis.size > 1)
    if isinstance(cell, ImagePlot) and len(dense) >= 2:
        # Data axes are declared slowest-first, so the last is horizontal.
        return ImagePlot(
            AxisRef.data(str(dense[-1].axis_id)),
            AxisRef.data(str(dense[-2].axis_id)),
            reduction=cell.reduction,
            labels=cell.labels,
        )
    if isinstance(cell, CurvePlot):
        if facet in (cell.x, cell.group):
            if not dense:
                return None
            x = AxisRef.data(str(dense[0].axis_id))
            # Whatever the curve grouped BY is either the facet (the grid
            # owns it now) or the very axis it just started walking.
            group = cell.group if cell.group not in (facet, x) else None
            return CurvePlot(
                x, group=group, reduction=cell.reduction, labels=cell.labels
            )
    if facet in (
        getattr(cell, "x", None),
        getattr(cell, "y", None),
        getattr(cell, "group", None),
    ):
        return None
    return cell


def default_spec(schema: Any) -> FacetGridPlot | None:
    """One cell per thing measured, showing the densest structure it holds.

    Two questions, asked separately because they are separate: WHAT varies
    from cell to cell (see :func:`_facet_axis`), and what one cell shows.

    The cell shows the densest structure the data offers.  Two data axes
    make an image cell (a camera frame).  A SCALAR point over two or more
    live scan dimensions images its two innermost dimensions instead -- the
    scan heatmap, which is what a field map was measured for, and which
    consumes those dimensions, so that branch keeps its own facet.  Anything
    else falls back to the curve default: one curve per cell, which is how
    thirty-five sites judged in each of three frames become three cells with
    thirty-five points, instead of a panel that refuses to draw at all.

    Degenerate axes (one value) are real provenance but not structure, so
    inference never sees them.
    """

    if not isinstance(schema, DatasetSchema):
        return None
    live = live_grid_dimensions(schema)
    repeats = schema.repeat_axis.size > 1
    cell = _data_axes_cell(schema)
    if cell is None:
        live_data_axes = tuple(
            axis for axis in schema.cell_schema.data_axes if axis.size > 1
        )
        if not live_data_axes and len(live) >= 2:
            # Declared slowest-first, so the last live dimension is the
            # innermost loop: the horizontal image axis, exactly as the dense
            # image kind reads the same grid.  The heatmap IS the scan, so
            # what is left to face is an outer loop or the sweeps.
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
    facet = _facet_axis(schema, cell)
    if facet is None:
        return None
    cell = cell_within_one_cell(schema, facet, cell)
    return None if cell is None else FacetGridPlot(facet, cell)


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
