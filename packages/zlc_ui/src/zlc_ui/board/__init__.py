"""Pure rectangle packers for board views."""

from .board_layout import (
    BoardMetrics,
    GeomProxy,
    board_width,
    drop_index,
    first_free_slot,
    min_board_width,
    pack,
)
from .panel_geometry import (
    DEFAULT_PANEL_SIZE,
    PANEL_SIZES,
    panel_display_size,
    panel_size_cells,
)

__all__ = [
    "BoardMetrics",
    "GeomProxy",
    "board_width",
    "drop_index",
    "first_free_slot",
    "min_board_width",
    "pack",
    "DEFAULT_PANEL_SIZE",
    "PANEL_SIZES",
    "panel_display_size",
    "panel_size_cells",
]
