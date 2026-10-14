"""One reusable Qt renderer for published typed-item inventories."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from .fluent import fluent_font_size, scaled_px, _radius
from .style import DIVIDER, FONT, MUTED_INK, SURFACE, TEXT

__all__ = ["PublishedItemsLegend"]


class PublishedItemsLegend(QtWidgets.QWidget):
    """Render ``(name, shape, description)`` rows as small outlined chips.

    Each item is a hairline chip in the body font -- its name, then its shape
    in graphite -- and the chips wrap to the width they are given.  It was a
    monospace ``items:`` block that read like debug output.  The descriptions
    are the tooltip; :meth:`text` is the same inventory as plain words.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._rows: tuple[tuple[str, str, str], ...] = ()
        self._text = ""
        font = QtGui.QFont(FONT)
        font.setPointSize(fluent_font_size())
        self.setFont(font)
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMinimumWidth(0)
        self.set_rows(())

    def set_rows(self, rows) -> None:
        rows = tuple((str(name), str(shape), str(description)) for name, shape, description in rows)
        tooltip = "\n".join(
            f"{name} {shape} — {description}" for name, shape, description in rows if description
        )
        text = (
            "outputs: " + ", ".join(f"{name} {shape}".rstrip() for name, shape, _ in rows)
            if rows else "no outputs"
        )
        if rows == self._rows and text == self._text and tooltip == self.toolTip():
            return
        self._rows, self._text = rows, text
        self.setToolTip(tooltip)
        self.updateGeometry()
        self.update()

    def text(self) -> str:
        return self._text

    # -- geometry: chips flow left to right and wrap -----------------------
    def _chips(self, width: int) -> list[tuple[QtCore.QRect, str, str]]:
        metrics = self.fontMetrics()
        pad_x, gap = scaled_px(8, minimum=6), scaled_px(6, minimum=4)
        height = metrics.height() + scaled_px(4, minimum=3)
        placed, x, y = [], 0, 0
        for name, shape, _description in self._rows:
            shown = f"  {shape}" if shape else ""
            chip = metrics.horizontalAdvance(name + shown) + 2 * pad_x
            if x and x + chip > width:
                x, y = 0, y + height + gap
            placed.append((QtCore.QRect(x, y, chip, height), name, shown))
            x += chip + gap
        return placed

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API
        chips = self._chips(max(1, width))
        if not chips:
            return self.fontMetrics().height()
        return max(rect.bottom() for rect, _name, _shape in chips) + 1

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API
        width = max(1, self.width())
        return QtCore.QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API
        return QtCore.QSize(0, self.heightForWidth(max(1, self.width())))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self.updateGeometry()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setFont(self.font())
        chips = self._chips(max(1, self.width()))
        if not chips:
            painter.setPen(QtGui.QColor(MUTED_INK))
            painter.drawText(self.rect(), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self._text)
            painter.end()
            return
        metrics = self.fontMetrics()
        pad_x = scaled_px(8, minimum=6)
        radius = float(_radius())
        for rect, name, shape in chips:
            painter.setPen(QtGui.QPen(QtGui.QColor(DIVIDER), 1.0))
            painter.setBrush(QtGui.QColor(SURFACE))
            painter.drawRoundedRect(QtCore.QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
            text_rect = rect.adjusted(pad_x, 0, -pad_x, 0)
            painter.setPen(QtGui.QColor(TEXT))
            painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, name)
            if shape:
                painter.setPen(QtGui.QColor(MUTED_INK))
                painter.drawText(
                    text_rect.adjusted(metrics.horizontalAdvance(name), 0, 0, 0),
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, shape,
                )
        painter.end()
