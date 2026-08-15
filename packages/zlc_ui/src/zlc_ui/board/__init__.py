"""Pure rectangle packers for board views."""

from .board_layout import (
    BoardMetrics,
    GeomProxy,
    board_width,
    nearest_anchor,
    first_free_slot,
    min_board_width,
    pack,
)
from .panel_geometry import (
    PLACEHOLDER_CELL_PX,
    panel_display_size,
    use_panel_display_sizes,
    panel_size_cells,
)

__all__ = [
    "PLACEHOLDER_CELL_PX",
    "BoardMetrics",
    "GeomProxy",
    "board_width",
    "nearest_anchor",
    "first_free_slot",
    "min_board_width",
    "pack",
    "panel_display_size",
    "use_panel_display_sizes",
    "panel_size_cells",
]
