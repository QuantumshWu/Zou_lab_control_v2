"""The product vocabulary for plots an operator may add to TaskConsole.

``zlc_plot`` owns every renderer it can provide, including PulseTimeline.
TaskConsole is a narrower product surface: its catalog names which of those
renderers the live board offers.  Keeping that policy here prevents a
renderer registry from silently becoming a menu definition.

There is ONE naming scheme: the plot kind's own name.  A menu row is
``image``, ``curve``, ``facet_grid`` -- the standard vocabulary and nothing
invented beside it.  A FacetGrid's CELL kind is not a menu row at all: it is
a parameter of the grid panel, chosen in the panel's settings, where empty
means the data decides.  Which cell kinds a grid may hold is ``PanelState``'s
rule; this module only says which kinds the board offers.

Every axis choice is the one default table in ``zlc_plot._kinds.defaults``,
read unaltered: this layer fixes WHICH kind (and, for a grid, which cell
kind) a panel is, and a second choice made here, however small, is a second
answer to the same question.
"""

from __future__ import annotations

from zlc_plot import PlotKind, fitting_spec
from zlc_plot.specs import semantic_spec


__all__ = [
    "TASK_CONSOLE_PANEL_KINDS",
    "task_console_fitting_spec",
    "task_console_panel_identity_for_spec",
    "task_console_panel_kind",
]


TASK_CONSOLE_PANEL_KINDS: tuple[PlotKind, ...] = (
    PlotKind.IMAGE,
    PlotKind.CURVE,
    PlotKind.ROLLING,
    PlotKind.HISTOGRAM,
    PlotKind.FACET_GRID,
)


def task_console_panel_kind(kind: object) -> PlotKind:
    """Resolve one TaskConsole kind or reject a renderer-only vocabulary item."""

    key = kind.value if isinstance(kind, PlotKind) else str(kind)
    for offered in TASK_CONSOLE_PANEL_KINDS:
        if offered.value == key:
            return offered
    raise ValueError(f"plot kind {key!r} is not available on TaskConsole")


def task_console_panel_identity_for_spec(spec: object) -> tuple[str, str]:
    """The complete TaskConsole identity of one accepted Plot specification."""

    kind = task_console_panel_kind(getattr(spec, "kind", None))
    cell_key = (
        semantic_spec(spec).kind.value if kind is PlotKind.FACET_GRID else ""
    )
    return kind.value, cell_key


def task_console_fitting_spec(
    schema: object,
    kind: object = "",
    cell_kind: object = "",
) -> object | None:
    """The spec this data admits under one fixed TaskConsole identity.

    With no kind the data decides, and the answer must still be a kind the
    board offers.  An empty cell kind on a grid means the data decides the
    cell too; a named one is the operator's choice, and the grid and its cell
    are composed once, in zlc_plot.
    """

    if kind in (None, ""):
        spec = fitting_spec(schema)
        if spec is not None:
            task_console_panel_kind(spec.kind)
        return spec
    resolved = task_console_panel_kind(kind)
    requested_cell = (
        cell_kind.value if isinstance(cell_kind, PlotKind) else str(cell_kind)
    )
    spec = fitting_spec(
        schema,
        resolved,
        cell=PlotKind(requested_cell) if requested_cell else None,
    )
    if spec is not None and requested_cell:
        if semantic_spec(spec).kind.value != requested_cell:
            raise ValueError("FacetGrid resolver returned another cell kind")
    return spec
