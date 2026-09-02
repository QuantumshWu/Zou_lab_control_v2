"""A limit describes one quantity; it does not travel to a kind that plots another.

Appearance -- a title, a colormap, the grid -- is carried across a panel's
identity change.  A limit is not appearance: colour limits are not y limits,
and a curve cell that inherited an image cell's TIGHT colour re-fit re-fitted
its y axis on every shot.  Each spec says which it is, and the one answer
the console reads is derived from those declarations.
"""

from __future__ import annotations

from zlc_plot import DEFAULTS
from zlc_plot.kinds import PlotKind
from zlc_plot.specs import (
    GRID_CELL_KINDS,
    non_portable_display_names,
    parameter_schema_for_kind,
)

LIMITS = frozenset(
    {
        "relim_mode",
        "x_relim_mode",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "color_min",
        "color_max",
    }
)


def test_the_non_portable_display_names_are_exactly_the_limits() -> None:
    assert non_portable_display_names() == LIMITS


def test_every_kind_marks_its_limits_and_nothing_else() -> None:
    seen: set[str] = set()
    for kind in PlotKind:
        cells = GRID_CELL_KINDS if kind is PlotKind.FACET_GRID else (None,)
        for cell in cells:
            schema = parameter_schema_for_kind(
                kind, style=DEFAULTS.style, facet_cell_kind=cell
            )
            for name in schema:
                seen.add(name)
                assert schema[name].portable == (name not in LIMITS), (
                    kind,
                    cell,
                    name,
                )
    assert LIMITS <= seen, "every limit is declared by some kind"
