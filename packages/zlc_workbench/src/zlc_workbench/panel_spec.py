"""Resolve a saved Workbench panel identity through zlc_plot's public API."""

from __future__ import annotations

from dataclasses import replace

from zlc_plot import PlotKind, fitting_spec, updated_spec
from zlc_plot.semantics import axis_choices_for_schema, axis_size


__all__ = ["fitting_panel_spec"]


_FACET_CELL_KINDS = {
    PlotKind.CURVE,
    PlotKind.IMAGE,
    PlotKind.HISTOGRAM,
}


def _dense_series_x(schema: object, spec: object) -> object:
    """Land a defaulted series x on an axis that can actually carry a series.

    A camera frame's point table has one row, and the library's curve default
    walks the point domain -- so "1D vector on a camera signal" opened as one
    invisible point.  The size authority is ``zlc_plot.semantics.axis_size``
    (the same one the semantic editor's choices use), and the re-point goes
    through ``updated_spec``, the one semantic composition authority, so no
    second spec-editing path exists here.
    """

    cell = getattr(spec, "cell", None)
    semantic = cell if cell is not None else spec
    if getattr(semantic, "kind", None) is not PlotKind.CURVE:
        return spec
    x = getattr(semantic, "x", None)
    if x is None or axis_size(schema, x) > 1:
        return spec
    taken = {
        value
        for value in (
            getattr(semantic, "y", None),
            getattr(semantic, "group", None),
            getattr(spec, "facet", None),
        )
        if value is not None
    }
    for candidate in axis_choices_for_schema(schema):
        if candidate in taken or axis_size(schema, candidate) <= 1:
            continue
        try:
            return updated_spec(schema, spec, "x", candidate)
        except Exception:
            continue
    return spec


def fitting_panel_spec(
    schema: object,
    kind: object = "",
    cell_kind: object = "",
) -> object | None:
    """Return the fixed outer/cell spec a generic saved panel describes."""

    resolved = None if kind in (None, "") else PlotKind(kind)
    cell_text = cell_kind.value if isinstance(cell_kind, PlotKind) else str(cell_kind)
    if resolved is not PlotKind.FACET_GRID:
        if cell_text:
            raise ValueError("only a FacetGrid panel has a cell kind")
        spec = fitting_spec(schema, resolved)
        return None if spec is None else _dense_series_x(schema, spec)

    if cell_text:
        cell = PlotKind(cell_text)
        if cell not in _FACET_CELL_KINDS:
            raise ValueError("FacetGrid cell kind must be curve, image, or histogram")
    else:
        # An empty cell kind means the DATA decides, here and only here.  Two
        # data axes inside one point make an image cell; anything less is a
        # curve.  Probing "whichever does not raise" is not enough -- a curve
        # cell ACCEPTS an image and draws it as a 2.3-million-point polyline.
        data_axes = tuple(getattr(schema.cell_schema, "data_axes", ()))
        cell = PlotKind.IMAGE if len(data_axes) >= 2 else PlotKind.CURVE
    outer_spec = fitting_spec(schema, PlotKind.FACET_GRID)
    cell_spec = fitting_spec(schema, cell)
    if outer_spec is None or cell_spec is None:
        return None
    cell_spec = _cell_inside_one_point(schema, cell_spec)
    cell_spec = _cell_clear_of_facet(schema, outer_spec.facet, cell_spec)
    if cell_spec is None:
        return None
    return _dense_series_x(schema, replace(outer_spec, cell=cell_spec))


def _cell_inside_one_point(schema: object, cell_spec: object) -> object:
    """A facet cell shows what is INSIDE one point: the data axes.

    Alone, an image defaults to the scan grid as its surface -- the right map
    for scalar cells.  As a facet CELL that default is inside-out: the grid
    dimensions are what the facets walk ACROSS, and what remains within one
    cell is the dense data.  A scan of camera frames faceted this way drew
    each 2.3-megapixel frame as a one-pixel stripe over the scan dimension.
    """

    from zlc_plot import AxisRef

    data_axes = tuple(
        getattr(axis, "axis_id", None)
        for axis in getattr(schema.cell_schema, "data_axes", ())
    )
    if getattr(cell_spec, "kind", None) is PlotKind.IMAGE and len(data_axes) >= 2:
        # Data axes are declared slowest-first, so the last is horizontal.
        for slot, axis_id in (("x", data_axes[-1]), ("y", data_axes[-2])):
            try:
                cell_spec = updated_spec(
                    schema, cell_spec, slot, AxisRef.data(str(axis_id))
                )
            except Exception:
                return cell_spec
    return cell_spec


def _cell_clear_of_facet(schema: object, facet: object, cell_spec: object) -> object | None:
    """Re-point cell axes off the facet axis; the grid OWNS that dimension.

    The outer default facets the outermost scan dimension, and a cell kind
    defaulted in isolation may claim the same axis -- an image cell over a
    2D scan of camera frames picked the scan dimensions as its own x/y, and
    composing the two then failed ("facet source cannot also be a cell axis
    or group").  Both defaults are right alone; the COMPOSITION is where the
    disjointness rule lives, so it is enforced here, through updated_spec,
    the one semantic composition authority.  A cell that cannot be moved off
    the facet axis is a refusal, not a guess.
    """

    taken = {
        value
        for value in (
            getattr(cell_spec, "x", None),
            getattr(cell_spec, "y", None),
            getattr(cell_spec, "group", None),
        )
        if value is not None
    }
    taken.add(facet)
    for slot in ("x", "y", "group"):
        if getattr(cell_spec, slot, None) != facet:
            continue
        for candidate in axis_choices_for_schema(schema):
            if candidate in taken:
                continue
            try:
                cell_spec = updated_spec(schema, cell_spec, slot, candidate)
            except Exception:
                continue
            taken.add(candidate)
            break
        else:
            return None
    return cell_spec
