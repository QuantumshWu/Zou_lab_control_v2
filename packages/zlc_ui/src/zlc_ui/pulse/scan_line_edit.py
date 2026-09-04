"""A numeric field with an injected Scan/API binding state."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.fluent import (
    API_VIOLET, API_VIOLET_DARK, BG, CONFIG_GREEN, CONFIG_GREEN_DARK,
    EDIT_PADDING_H, FONT, ORANGE, ORANGE_DARK,
    ORANGE_TINT, PADDING_V, PLACEHOLDER, RADIUS, SURFACE, FluentLineEdit, Metrics,
    fluent_font_size, scaled_px,
)


class _FluentScanDot(QtWidgets.QAbstractButton):
    def __init__(self, parent=None, *, tooltip: str) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._number: int | None = None
        self._kind: str | None = None
        diameter = Metrics.dot()
        self.setFixedSize(diameter, diameter)
        self.setToolTip(tooltip)

    def set_number(self, number: int | None) -> None:
        self._number = None if number is None else int(number)
        self.update()

    def set_binding(self, kind: str | None) -> None:
        """Which of the three ways this field is supplied, or none.

        One setter rather than a flag per kind: the three are alternatives,
        and a pair of booleans can say a thing the model cannot mean.
        """

        self._kind = None if kind is None else str(kind)
        self.update()

    def nextCheckState(self) -> None:  # model owns state
        pass

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        center = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        radius = min(self.width(), self.height()) / 2.0 - max(1.0, scaled_px(1))
        if self._kind is not None:
            painter.setBrush(QtGui.QColor(_BINDING_FILL[self._kind]))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(center, radius, radius)
            if self._number is not None:
                painter.setPen(QtGui.QColor(SURFACE))
                font = QtGui.QFont(FONT, max(6, fluent_font_size() - 5))
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(self.rect(), QtCore.Qt.AlignCenter, str(self._number))
        else:
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(PLACEHOLDER), max(1, scaled_px(1))))
            painter.drawEllipse(center, radius, radius)
            painter.setBrush(QtGui.QColor(PLACEHOLDER))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(center, radius * 0.42, radius * 0.42)
        painter.end()


#: What each binding paints as.  The board's own orange for a scan column,
#: the violet a caller writes in for an API parameter, and slate for a config
#: number, which is nobody's to supply because it is already the pulse's.
_BINDING_FILL = {
    "scan": ORANGE,
    "api": API_VIOLET,
    "config": CONFIG_GREEN,
}


def _bound_field_style(*, text: str, border: str, fill: str | None) -> str:
    background = f"background: {fill}; " if fill else ""
    return (
        f"QLineEdit {{ {background}color: {text}; border: 1px solid {border}; "
        f"border-radius: {scaled_px(RADIUS)}px; padding: {scaled_px(PADDING_V)}px "
        f"{scaled_px(EDIT_PADDING_H)}px; font: {fluent_font_size()}pt \"{FONT}\"; }}"
    )


def _muted_line_style() -> str:
    return (
        f"QLineEdit {{ background: {BG}; color: {PLACEHOLDER}; border: 1px solid {PLACEHOLDER}; "
        f"border-radius: {scaled_px(RADIUS)}px; padding: {scaled_px(PADDING_V)}px "
        f"{scaled_px(EDIT_PADDING_H)}px; font: {fluent_font_size()}pt \"{FONT}\"; }}"
    )


class FluentScanLineEdit(FluentLineEdit):
    """Numeric field whose dot displays unbound, Scan, or API state."""

    scan_clicked = QtCore.pyqtSignal()

    def __init__(self, text: str = "", parent=None, *, tooltip: str = "Click the dot to cycle this field binding") -> None:
        super().__init__(text, parent)
        self._base_style = self.styleSheet()
        self._dot = _FluentScanDot(self, tooltip=tooltip)
        self._dot.clicked.connect(self.scan_clicked)
        self._field_state: tuple[bool, str | None, int | None] | None = None
        self._reserve_right()

    @property
    def dot(self) -> _FluentScanDot:
        return self._dot

    @property
    def binding_kind(self) -> str | None:
        """Current injected binding kind, or ``None`` for a literal field."""

        return None if self._field_state is None else self._field_state[1]

    @property
    def binding_number(self) -> int | None:
        """Current injected binding number, or ``None`` when unbound."""

        return None if self._field_state is None else self._field_state[2]

    def _dot_size(self) -> int:
        return Metrics.dot()

    def _reserve_right(self) -> None:
        self.setTextMargins(0, 0, self._dot_size() + scaled_px(3), 0)

    def _place_dot(self) -> None:
        diameter = self._dot_size()
        self._dot.setGeometry(
            int(self.width() - diameter - scaled_px(4)),
            int((self.height() - diameter) // 2), diameter, diameter,
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place_dot()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._place_dot()

    def set_field_state(self, *, editable: bool, binding: str | None = None, number: int | None = None) -> None:
        normalized = None if binding is None else str(binding).strip().lower()
        if normalized is not None and normalized not in _BINDING_FILL:
            raise ValueError(
                f"binding must be one of {sorted(_BINDING_FILL)}, or None"
            )
        if normalized is None and number is not None:
            raise ValueError("an unbound field cannot have a binding number")
        state = (bool(editable), normalized, number)
        if state == self._field_state:
            return
        self._field_state = state
        is_scan = normalized == "scan"
        self._dot.set_binding(normalized)
        # A scan column is the one binding whose number this field cannot
        # hold: the board writes it per point.  An API or config field still
        # shows -- and edits -- the number it carries.
        self._dot.setChecked(is_scan)
        self._dot.set_number(number if normalized is not None else None)
        self.setReadOnly(is_scan or not state[0])
        if is_scan:
            style = _bound_field_style(text=ORANGE_DARK, border=ORANGE, fill=ORANGE_TINT)
        elif normalized == "api":
            style = _bound_field_style(text=API_VIOLET_DARK, border=API_VIOLET, fill=None)
        elif normalized == "config":
            style = _bound_field_style(
                text=CONFIG_GREEN_DARK, border=CONFIG_GREEN, fill=None
            )
        else:
            style = self._base_style if state[0] else _muted_line_style()
        self.setStyleSheet(style)
        self._reserve_right()
        self.update()


__all__ = ["FluentScanLineEdit"]
