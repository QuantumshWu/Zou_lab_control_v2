"""Resolve a saved Workbench panel identity through zlc_plot's public API."""

from __future__ import annotations

from dataclasses import replace

from zlc_plot import PlotKind, fitting_spec


__all__ = ["fitting_panel_spec"]


_FACET_CELL_KINDS = {
    PlotKind.CURVE,
    PlotKind.IMAGE,
    PlotKind.HISTOGRAM,
}


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
        return fitting_spec(schema, resolved)

    cell = PlotKind(cell_text)
    if cell not in _FACET_CELL_KINDS:
        raise ValueError("FacetGrid cell kind must be curve, image, or histogram")
    outer_spec = fitting_spec(schema, PlotKind.FACET_GRID)
    cell_spec = fitting_spec(schema, cell)
    if outer_spec is None or cell_spec is None:
        return None
    return replace(outer_spec, cell=cell_spec)
