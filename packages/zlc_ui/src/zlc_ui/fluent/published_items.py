"""One reusable Qt renderer for published typed-item inventories."""

from __future__ import annotations

from PyQt5 import QtWidgets

from .fluent import FluentLabel
from .style import GREY

__all__ = ["PublishedItemsLegend"]


class PublishedItemsLegend(FluentLabel):
    """Render ``(name, shape, description)`` rows with one shared style."""

    def __init__(self, parent=None) -> None:
        super().__init__("", parent)
        self.setWordWrap(True)
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none; "
            "font-family: Consolas, 'DejaVu Sans Mono', monospace;"
        )

    def set_rows(self, rows) -> None:
        rows = tuple((str(name), str(shape), str(description)) for name, shape, description in rows)
        if rows:
            name_width = max(len(name) for name, _shape, _description in rows)
            shape_width = max(len(shape) for _name, shape, _description in rows)
            lines = [
                f"  {name:<{name_width}}  {shape:<{shape_width}}".rstrip()
                for name, shape, _description in rows
            ]
            text = "items:\n" + "\n".join(lines)
            tooltip = "\n".join(
                f"{name} {shape} — {description}"
                for name, shape, description in rows
                if description
            )
        else:
            text, tooltip = "items: (none)", ""
        if text == self.text() and tooltip == self.toolTip():
            return
        self.setText(text)
        self.setToolTip(tooltip)
