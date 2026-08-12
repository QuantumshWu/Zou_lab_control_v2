"""Resolve a saved Workbench panel identity through zlc_plot's public API."""

from __future__ import annotations

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

    cell = None
    if cell_text:
        cell = PlotKind(cell_text)
        if cell not in _FACET_CELL_KINDS:
            raise ValueError("FacetGrid cell kind must be curve, image, or histogram")
    # An empty cell kind means the DATA decides, and either way the grid and
    # its cell are composed once, in zlc_plot: this layer only says which.
    outer_spec = fitting_spec(schema, PlotKind.FACET_GRID, cell=cell)
    if outer_spec is None:
        return None
    return _dense_series_x(schema, outer_spec)


