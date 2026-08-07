"""Tell the window package how big each named panel's picture is.

Two packages need the same nine numbers and neither may import the other:
zlc_plot decides them, because a figure's size is a fact about the figure, and
zlc_ui frames them into cards while carrying no domain dependency at all.
This layer imports both, which makes it the only place that can carry the
number across without either side reaching past its seam.

It replaces a copy.  zlc_ui restated the margins, the cell unit and the display
scale, and a test stood over the two copies checking they still agreed -- a
guard whose existence admits there are two of something.  When the panel
margins were corrected against the instrument these figures descend from, the
copy drifted and that test was what noticed.  A guard that has to catch a
class of bug is worse than that class being impossible.
"""

from __future__ import annotations

from functools import lru_cache


def install() -> None:
    """Supply the sizes once, from the plan the drawing package computes."""

    from zlc_ui import use_panel_display_sizes

    use_panel_display_sizes(_display_size)


@lru_cache(maxsize=None)
def _display_size(preset: str) -> tuple[int, int]:
    """The logical size zlc_plot plans for one preset, as (width, height).

    Cached because it is a pure function of a constant layout and is asked for
    on every card resize.  The kind is CURVE deliberately: presets size the
    CANVAS, and only the pulse timeline widens its own left margin, which is a
    property of that drawing rather than of the card holding it.
    """

    from zlc_plot import DEFAULTS
    from zlc_plot.kinds import PlotKind
    from zlc_plot.layout import resolve_surface

    plan = resolve_surface(
        preset, PlotKind.CURVE, layout=DEFAULTS.layout, style=DEFAULTS.style
    )
    width, height = plan.logical_size
    return int(width), int(height)
