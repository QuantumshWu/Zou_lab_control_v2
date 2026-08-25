"""The product vocabulary for plots an operator may add to TaskConsole.

``zlc_plot`` owns every renderer it can provide, including PulseTimeline.
TaskConsole is a narrower product surface: its catalog names which of those
renderers the live board offers.  Keeping that policy here prevents a
renderer registry from silently becoming a menu definition.

There is ONE naming scheme: the plot kind's own name.  A menu row is
``image``, ``curve``, ``facet_grid`` -- the standard vocabulary and nothing
invented beside it.  A FacetGrid's CELL kind is not a menu row at all: it is
a parameter of the grid panel, chosen in the panel's settings, where empty
means the data decides.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_plot import PlotKind
from zlc_plot.specs import semantic_spec

from .panel_spec import fitting_panel_spec


__all__ = [
    "GRID_CELL_KINDS",
    "TASK_CONSOLE_PANEL_CATALOG",
    "TaskConsolePanelKind",
    "panel_kind_choices",
    "task_console_fitting_spec",
    "task_console_panel_identity",
    "task_console_panel_kind",
]


# What a grid cell may be.  The identity gate, the spec resolver, and the
# panel settings control all speak this one vocabulary.
GRID_CELL_KINDS = (PlotKind.CURVE, PlotKind.IMAGE, PlotKind.HISTOGRAM)


@dataclass(frozen=True, slots=True)
class TaskConsolePanelKind:
    """One immutable panel identity selected by the combined Add control."""

    kind: PlotKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PlotKind):
            raise TypeError("panel kind must be PlotKind")

    @property
    def key(self) -> str:
        return self.kind.value

    @property
    def label(self) -> str:
        """The plot kind's own name; there is no second naming scheme."""

        return self.kind.value


TASK_CONSOLE_PANEL_CATALOG: tuple[TaskConsolePanelKind, ...] = (
    TaskConsolePanelKind(PlotKind.IMAGE),
    TaskConsolePanelKind(PlotKind.CURVE),
    TaskConsolePanelKind(PlotKind.ROLLING),
    TaskConsolePanelKind(PlotKind.HISTOGRAM),
    TaskConsolePanelKind(PlotKind.FACET_GRID),
)

_BY_KEY = {entry.key: entry for entry in TASK_CONSOLE_PANEL_CATALOG}


def task_console_panel_kind(kind: object) -> TaskConsolePanelKind:
    """Resolve one TaskConsole kind or reject a renderer-only vocabulary item."""

    key = kind.value if isinstance(kind, PlotKind) else str(kind)
    try:
        return _BY_KEY[key]
    except KeyError as error:
        raise ValueError(f"plot kind {key!r} is not available on TaskConsole") from error


def panel_kind_choices() -> tuple[tuple[str, str], ...]:
    """Return the exact plain rows consumed by the toolkit-neutral view port."""

    return tuple((entry.key, entry.label) for entry in TASK_CONSOLE_PANEL_CATALOG)


def task_console_panel_identity(
    kind: object,
    cell_kind: object = "",
) -> TaskConsolePanelKind:
    """Validate the complete immutable identity stored by TaskConsole.

    An empty cell kind on a FacetGrid means the data decides; a named one
    must be a legal grid cell.  What a grid cell can BE is a fact of the
    data in it, so nothing here pins a cell to a catalog default.
    """

    definition = task_console_panel_kind(kind)
    cell_key = cell_kind.value if isinstance(cell_kind, PlotKind) else str(cell_kind)
    if definition.kind is not PlotKind.FACET_GRID:
        if cell_key:
            raise ValueError(f"{definition.label} does not take a cell kind")
        return definition
    if not cell_key:
        return definition
    if PlotKind(cell_key) not in GRID_CELL_KINDS:
        raise ValueError(f"a grid cell cannot be a {cell_key}")
    return definition


def task_console_fitting_spec(
    schema: object,
    kind: object = "",
    cell_kind: object = "",
) -> object | None:
    """Resolve data through the same fixed identity selected at Add Panel."""

    if kind in (None, ""):
        spec = fitting_panel_spec(schema)
        if spec is None:
            return None
        task_console_panel_kind(spec.kind)
        return spec
    definition = task_console_panel_kind(kind)
    requested_cell = (
        cell_kind.value if isinstance(cell_kind, PlotKind) else str(cell_kind)
    )
    task_console_panel_identity(definition.kind, requested_cell)
    spec = fitting_panel_spec(
        schema,
        definition.kind,
        requested_cell if definition.kind is PlotKind.FACET_GRID else "",
    )
    if spec is not None and requested_cell:
        if semantic_spec(spec).kind.value != requested_cell:
            raise ValueError("FacetGrid resolver returned another cell kind")
    return spec
