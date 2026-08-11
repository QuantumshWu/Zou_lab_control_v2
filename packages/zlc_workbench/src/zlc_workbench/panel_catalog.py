"""The product vocabulary for plots an operator may add to TaskConsole.

``zlc_plot`` owns every renderer it can provide, including PulseTimeline.
TaskConsole is a narrower product surface: its catalog is the five live-board
entries from the authoritative v1 UI.  Keeping that policy here prevents a
renderer registry from silently becoming a menu definition.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_plot import PlotKind

from .panel_spec import fitting_panel_spec


__all__ = [
    "TASK_CONSOLE_PANEL_CATALOG",
    "TaskConsolePanelKind",
    "panel_kind_choices",
    "task_console_fitting_spec",
    "task_console_panel_identity",
    "task_console_panel_kind",
]


@dataclass(frozen=True, slots=True)
class TaskConsolePanelKind:
    """One immutable panel identity selected by the combined Add control."""

    kind: PlotKind
    label: str
    cell_kind: PlotKind | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PlotKind):
            raise TypeError("panel kind must be PlotKind")
        label = str(self.label).strip()
        if not label:
            raise ValueError("panel kind label must not be empty")
        object.__setattr__(self, "label", label)
        if self.kind is PlotKind.FACET_GRID:
            if self.cell_kind is not PlotKind.CURVE:
                raise ValueError("TaskConsole Site grid has fixed curve cells")
        elif self.cell_kind is not None:
            raise ValueError("only a FacetGrid panel has a cell kind")

    @property
    def key(self) -> str:
        return self.kind.value

    @property
    def cell_key(self) -> str:
        return "" if self.cell_kind is None else self.cell_kind.value


# Exact authoritative v1 order and labels.  PulseTimeline belongs to PulseGUI;
# Image/Histogram FacetGrid specs remain available to authored reports and task
# previews, but are not additional TaskConsole Add entries.
TASK_CONSOLE_PANEL_CATALOG: tuple[TaskConsolePanelKind, ...] = (
    TaskConsolePanelKind(PlotKind.IMAGE, "2D image"),
    TaskConsolePanelKind(PlotKind.CURVE, "1D vector"),
    TaskConsolePanelKind(PlotKind.ROLLING, "Rolling trace"),
    TaskConsolePanelKind(PlotKind.HISTOGRAM, "Distribution"),
    TaskConsolePanelKind(PlotKind.FACET_GRID, "Site grid", PlotKind.CURVE),
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
    """Validate the complete immutable identity stored by TaskConsole."""

    definition = task_console_panel_kind(kind)
    cell_key = cell_kind.value if isinstance(cell_kind, PlotKind) else str(cell_kind)
    if definition.cell_kind is None:
        if cell_key:
            raise ValueError(f"{definition.label} does not take a cell kind")
        return definition
    if not cell_key:
        return definition
    # The catalog's cell kind is the DEFAULT, not a constraint: what a grid
    # cell can be is a fact of the DATA in it.  Pinning it to curve drew a
    # scan of camera frames as nine 2.3-million-point polylines -- the blank,
    # lagging card this replaces.
    parsed = PlotKind(cell_key)
    if parsed in (PlotKind.FACET_GRID, PlotKind.PULSE_TIMELINE, PlotKind.ROLLING):
        raise ValueError(f"a grid cell cannot be a {parsed.value}")
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
    effective = requested_cell or definition.cell_key
    spec = fitting_panel_spec(
        schema,
        definition.kind,
        effective if definition.cell_kind is not None else "",
    )
    if spec is not None and definition.cell_kind is not None and effective:
        actual = getattr(getattr(spec, "cell", None), "kind", None)
        if str(getattr(actual, "value", actual)) != effective:
            raise ValueError("FacetGrid resolver returned another cell kind")
    return spec
