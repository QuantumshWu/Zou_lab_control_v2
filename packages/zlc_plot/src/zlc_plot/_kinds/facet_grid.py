"""FacetGrid kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..data_contract import image_axes, live_grid_dimensions
from ..kinds import AxisRef, PlotKind
from zlc_data import DatasetSchema
from ..specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    Reduction,
)
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
    # Any dataset can host an authored facet layout; whether one facet axis
    # is *unambiguous* is the narrower default_spec question below.  Keeping
    # the two apart is what lets a frontend show the kind greyed out instead
    # of silently dropping it from the choice list.
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
    the declared scan dimensions as facet candidates.

    Frames of an unscanned cycle use their real point coordinate and get a
    cell each.  A multidimensional scan cannot be represented by one hidden
    flattened axis: its outermost real dimension is the default facet and its
    other dimensions remain reducible/reassignable in Setting.  Repeats are
    the same measurement again, so the default reduction keeps pooling them;
    only an explicitly authored specification may make repeat a facet.
    ``None`` means the cell already consumes every non-repeat axis and there
    is no honest automatic FacetGrid layout.
    """

    live = live_grid_dimensions(schema)
    dense = tuple(axis for axis in schema.cell_schema.data_axes if axis.size > 1)
    # Without scan topology the point coordinate IS the authored cell identity
    # -- the frames of a camera cycle, including a one-frame cycle.  Its size
    # does not change that identity; using repeat instead made panel meaning
    # depend on the number of shots.
    point_axis = (
        AxisRef.point(str(schema.point_table.columns[0].coordinate_id))
        if schema.grid_topology is None and schema.point_table.columns
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
        return None
    if point_axis is not None:
        return point_axis
    if live:
        # A FacetGrid owns exactly one facet role.  A multidimensional scan
        # therefore defaults to its outermost REAL dimension and reduces the
        # others; the Setting table lets the operator reassign them.  The
        # flattened point-row ordinal is not another scientific axis.  Using
        # it here created a phantom ``point`` fate, turned a 10x10x10 scan into
        # 1000 cells, and left a phantom point-row restriction behind when
        # the operator tried to reduce it.
        return AxisRef.point_dimension(live[0])
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

    if facet is None:
        return cell
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
    """One cell per value of one real facet axis, showing dense cell data.

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
    they never decide what a cell holds.  A no-topology point coordinate may
    still identify a single authored frame.  Repeat is never selected here:
    an operator may author it explicitly, but acquisition history is reduced
    by default.
    """

    if not isinstance(schema, DatasetSchema):
        return None
    live = live_grid_dimensions(schema)
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
            return FacetGridPlot(None, heatmap)
        # A scalar measured repeatedly at authored point coordinates is a
        # distribution per point.  A curve would consume that point axis and
        # leave only repeat, which must not become the automatic facet.  The
        # histogram cell consumes values instead, leaving the authored point
        # coordinate as the honest facet (and also seeds persisted explicit
        # point-facet Histogram panels on reopen).
        cell = (
            HistogramPlot()
            if not live_data_axes
            and not live
            and schema.point_table.columns
            and schema.repeat_axis.size > 1
            else curve_default_spec(schema)
        )
    if cell is None:
        return None
    facet = _facet_axis(schema, cell)
    if facet is None:
        return FacetGridPlot(None, cell)
    cell = cell_within_one_cell(schema, facet, cell)
    return None if cell is None else FacetGridPlot(facet, cell)


def chosen_spec(schema: Any, current: Any) -> FacetGridPlot | None:
    """A grid for an operator who ASKED for one, default or not.

    ``default_spec`` answers a narrower question -- is there one
    unambiguous grid this dataset obviously wants -- and a dataset that
    has no obvious answer still has legal grids the operator may want to
    build.  Choosing the kind must therefore land somewhere they can
    edit: the plot they were already looking at becomes the cell, and
    the facet is simply the first axis that cell does not consume.  It
    is a starting point, not a recommendation; the fate table is where
    the operator says what they actually meant.
    """

    if not isinstance(schema, DatasetSchema):
        return None
    automatic = default_spec(schema)
    if automatic is not None:
        return automatic
    cell = (
        current
        if isinstance(current, (CurvePlot, ImagePlot, HistogramPlot))
        else curve_default_spec(schema)
    )
    if cell is None:
        return None
    consumed = {
        ref.physical_identity
        for ref in (
            getattr(cell, "x", None),
            getattr(cell, "y", None),
            getattr(cell, "group", None),
        )
        if isinstance(ref, AxisRef)
    }
    from ..semantics import axis_choices_for_schema

    offered = axis_choices_for_schema(schema)
    facet = next(
        (ref for ref in offered if ref.physical_identity not in consumed),
        offered[0] if offered else None,
    )
    if facet is None:
        return FacetGridPlot(None, cell)
    within = cell_within_one_cell(schema, facet, cell)
    return FacetGridPlot(facet, cell if within is None else within)


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
