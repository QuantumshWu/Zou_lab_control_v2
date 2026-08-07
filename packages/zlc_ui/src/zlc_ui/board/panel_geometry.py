"""The v1 panel-size vocabulary and display geometry used by the board.

This is the small, Qt-free portion of v1's ``panel_size`` and ``plot_layout``
that the UI board needs.  The names and constants are kept here so the board,
cards, and any future preview do not invent separate size presets.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_PANEL_SIZE",
    "PANEL_SIZES",
    "panel_display_size",
    "panel_size_cells",
]


PANEL_SIZES = ("1x2", "2x2", "4x2", "1x4", "2x4", "4x4", "4x8", "8x4", "8x8")
DEFAULT_PANEL_SIZE = "2x2"

# v1 plot-layout facts.  ``panel_display_size`` is the logical Qt data region;
# Fluent chrome is added by ``ConsoleBoardView.card_size``.
_PANEL_DISPLAY_SCALE = 0.7
_PANEL_UNIT_PX = (180, 240)  # row height, column width in design pixels
_PANEL_MARGINS_PX = (110, 96, 80, 70)  # left, right, bottom, title


def panel_size_cells(size: str) -> tuple[int, int]:
    """Parse one v1 ``rows x columns`` panel preset."""

    key = str(size).strip().lower().replace(" ", "")
    if key not in PANEL_SIZES:
        raise ValueError(
            f"unknown panel size {size!r}; choose from {', '.join(PANEL_SIZES)}"
        )
    rows, columns = key.split("x")
    return int(rows), int(columns)


def panel_display_size(size: str = DEFAULT_PANEL_SIZE) -> tuple[int, int]:
    """Return v1's logical Qt data-region size for a named panel."""

    rows, columns = panel_size_cells(size)
    left, right, bottom, top = _PANEL_MARGINS_PX
    design_width = columns * _PANEL_UNIT_PX[1] + left + right
    design_height = rows * _PANEL_UNIT_PX[0] + bottom + top
    return (
        round(design_width * _PANEL_DISPLAY_SCALE),
        round(design_height * _PANEL_DISPLAY_SCALE),
    )
