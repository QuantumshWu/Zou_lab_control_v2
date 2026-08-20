"""The process-global panel-size seam has one explicit composition owner."""

from __future__ import annotations

import pytest


def test_panel_size_owner_is_idempotent_but_cannot_be_replaced(monkeypatch) -> None:
    import zlc_ui.board.panel_geometry as geometry

    monkeypatch.setattr(geometry, "_measure", None)

    def first_owner(_size: str) -> tuple[int, int]:
        return 320, 240

    def second_owner(_size: str) -> tuple[int, int]:
        return 640, 480

    geometry.use_panel_display_sizes(first_owner)
    geometry.use_panel_display_sizes(first_owner)
    assert geometry.panel_display_size("2x2") == (320, 240)
    with pytest.raises(RuntimeError, match="already installed"):
        geometry.use_panel_display_sizes(second_owner)
    assert geometry.panel_display_size("2x2") == (320, 240)
