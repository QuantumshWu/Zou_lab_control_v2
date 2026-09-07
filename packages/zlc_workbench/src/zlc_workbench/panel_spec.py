"""Resolve a saved Workbench panel identity through zlc_plot's public API.

This layer fixes WHICH kind (and, for a grid, which cell kind) a saved
panel is; what that kind shows for the data -- every axis choice -- is
the one default table in ``zlc_plot._kinds.defaults``, read unaltered.  A
second choice made here, however small, is a second answer to the same
question, and the library's standalone picture and the console's diverged
by exactly that much.
"""

from __future__ import annotations

from zlc_plot import GRID_CELL_KINDS, PlotKind, fitting_spec


__all__ = ["fitting_panel_spec"]


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

    cell = None
    if cell_text:
        cell = PlotKind(cell_text)
        if cell not in GRID_CELL_KINDS:
            raise ValueError(
                "FacetGrid cell kind must be one of "
                + ", ".join(kind.value for kind in GRID_CELL_KINDS)
            )
    # An empty cell kind means the DATA decides, and either way the grid and
    # its cell are composed once, in zlc_plot: this layer only says which.
    return fitting_spec(schema, PlotKind.FACET_GRID, cell=cell)
