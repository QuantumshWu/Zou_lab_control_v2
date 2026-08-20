"""The panel-size vocabulary the board lays cards out with.

A card is a frame around a picture, so the card's size is the picture's size
plus chrome.  Who knows the picture's size is the package that draws it, and
this package may not import that one -- zlc_ui carries no domain dependency at
all.  So the sizes are SUPPLIED: whoever composes a window owns both halves and
tells this one what the drawing package said.

They used to be copied.  Margins, cell units and a display scale were all
restated here, with a test standing over the two copies checking they still
agreed -- which is a guard admitting there are two of something.  It caught
the drift the moment the margins were corrected against the instrument these
figures descend from, and a guard that has to catch a class of bug is worse
than that class of bug being impossible.  A card asks for the size; with
nobody drawing -- a gallery, a demo -- it sizes itself as the empty frame it
is, from a cell of its own that copies nothing.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = [
    "panel_display_size",
    "PLACEHOLDER_CELL_PX",
    "panel_size_cells",
    "use_panel_display_sizes",
]


#: What an EMPTY card measures, in logical pixels per grid cell.  This is
#: this package's own number and not a copy of anything: a frame with no
#: picture in it still needs a size, and how big that is is a UI decision.
#: What is deliberately NOT here is the figure geometry -- margins, the design
#: cell and the display scale were all restated in this file, kept in step
#: with the drawing package by a test, and they drifted the moment those
#: margins were corrected.
PLACEHOLDER_CELL_PX = (168, 126)  # width, height

_measure: Callable[[str], tuple[int, int]] | None = None


def use_panel_display_sizes(measure: Callable[[str], tuple[int, int]]) -> None:
    """Tell this package how big each named panel's picture is.

    Called by whoever composes a window, from the drawing package's own plan.
    Repeating the same callable is idempotent; a different callable is a second
    truth source and is rejected.
    """

    global _measure
    if not callable(measure):
        raise TypeError("panel display sizes must come from a callable")
    if _measure is measure:
        return
    if _measure is not None:
        raise RuntimeError(
            "panel display sizes are already installed by a different owner"
        )
    _measure = measure


def panel_size_cells(size: str) -> tuple[int, int]:
    """Parse one positive ``rows x columns`` geometry name.

    Which names the product offers is a plotting policy projected by the
    composition layer.  This UI helper only interprets the chosen geometry.
    """

    key = str(size).strip().lower().replace(" ", "")
    parts = key.split("x")
    if len(parts) != 2:
        raise ValueError(f"panel size must be rows x columns, got {size!r}")
    try:
        rows, columns = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"panel size must be rows x columns, got {size!r}") from error
    if rows < 1 or columns < 1:
        raise ValueError("panel size rows and columns must be positive")
    return rows, columns


def panel_display_size(size: str) -> tuple[int, int]:
    """The logical Qt data region for a named panel, as the drawer measured it."""

    key = str(size).strip().lower().replace(" ", "")
    rows, columns = panel_size_cells(key)
    if _measure is None:
        # Nobody is drawing: a gallery, a demo, this package's own tests.  The
        # card is sized as the frame it is, from this package's own cell.
        cell_width, cell_height = PLACEHOLDER_CELL_PX
        return columns * cell_width, rows * cell_height
    width, height = _measure(key)
    return int(width), int(height)
