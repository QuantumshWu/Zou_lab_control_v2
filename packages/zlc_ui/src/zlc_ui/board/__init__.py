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
    panel_display_size,
    use_panel_display_sizes,
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
    "panel_display_size",
    "use_panel_display_sizes",
    "panel_size_cells",
]
