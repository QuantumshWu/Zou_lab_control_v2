"""Editable default value with independently projected Scan and source bindings."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.fluent import (
    API_VIOLET, API_VIOLET_DARK, CONFIG_GREEN, CONFIG_GREEN_DARK,
    EDIT_PADDING_H, FONT, ORANGE, ORANGE_DARK, PADDING_V, PLACEHOLDER,
    RADIUS, SURFACE, FluentCheckBox, FluentComboBox, FluentLabel,
    FluentLineEdit, FluentPopup, fluent_font_size, scaled_px, signals_blocked,
    show_fluent_popup_for_anchor,
)


class _BindingButton(QtWidgets.QAbstractButton):
    """A fixed-width pair of badges, never a cycling or numbered control."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scan = False
        self.source = "default"
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(scaled_px(18, minimum=16), scaled_px(28, minimum=24))

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        badges = []
        if self.scan:
            badges.append(("S", ORANGE))
        if self.source != "default":
            badges.append((self.source[0].upper(), API_VIOLET if self.source == "api" else CONFIG_GREEN))
        if not badges:
            badges.append(("", PLACEHOLDER))
        diameter = min(self.width() - 2, self.height() / len(badges) - 1)
        font = QtGui.QFont(FONT, max(6, fluent_font_size() - 5))
        font.setBold(True)
        painter.setFont(font)
        for index, (text, color) in enumerate(badges):
            y = (self.height() - len(badges) * diameter) / 2 + index * diameter
            rect = QtCore.QRectF((self.width() - diameter) / 2, y, diameter, diameter)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(color).darker(120 if self.underMouse() else 100))
            painter.drawEllipse(rect.adjusted(0.5, 0.5, -0.5, -0.5))
            painter.setPen(QtGui.QColor(SURFACE))
            painter.drawText(rect, QtCore.Qt.AlignCenter, text)


def _bound_style(color: str, border: str, *, applied: bool = False) -> str:
    background = f"background: {QtGui.QColor(border).lighter(175).name()}; " if applied else ""
    return (
        f'QLineEdit {{ {background}color: {color}; border: 1px solid {border}; '
        f'border-radius: {scaled_px(RADIUS)}px; padding: {scaled_px(PADDING_V)}px '
        f'{scaled_px(EDIT_PADDING_H)}px; font: {fluent_font_size()}pt "{FONT}"; }}'
    )


class FluentScanLineEdit(FluentLineEdit):
    """One default value; the popup emits binding intent, never a value override."""

    binding_committed = QtCore.pyqtSignal(bool, str)

    def __init__(self, text: str = "", parent=None, *, tooltip: str = "Edit Scan and value source") -> None:
        super().__init__(text, parent)
        self._base_style = self.styleSheet()
        self.binding_button = _BindingButton(self)
        self.binding_button.clicked.connect(self._show_binding)
        self._tooltip = tooltip
        self._field_state = None
        self._popup = None
        self._reserve_right()

    def _reserve_right(self) -> None:
        self.setTextMargins(0, 0, self.binding_button.width() + scaled_px(3), 0)

    def _place_button(self) -> None:
        button = self.binding_button
        button.move(self.width() - button.width() - scaled_px(4), (self.height() - button.height()) // 2)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place_button()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._place_button()

    def _show_binding(self) -> None:
        if self._popup is None:
            self._popup = FluentPopup(self)
            layout = QtWidgets.QVBoxLayout(self._popup)
            margin = scaled_px(10)
            layout.setContentsMargins(margin, margin, margin, margin)
            layout.setSpacing(scaled_px(6))
            self.scan_toggle = FluentCheckBox("Scan")
            layout.addWidget(self.scan_toggle)
            source_row = QtWidgets.QHBoxLayout()
            source_row.addWidget(FluentLabel("Source"))
            self.source_combo = FluentComboBox()
            for label, value in (("Default", "default"), ("API", "api"), ("Config", "config")):
                self.source_combo.addItem(label, value)
            source_row.addWidget(self.source_combo, 1)
            layout.addLayout(source_row)
            self.source_info = FluentLabel("")
            self.source_info.setWordWrap(True)
            layout.addWidget(self.source_info)
            self.scan_toggle.toggled.connect(self._commit_binding)
            self.source_combo.currentIndexChanged.connect(self._commit_binding)
        self._project_popup()
        self._place_popup()

    def _place_popup(self) -> None:
        self._popup.ensurePolished()
        self._popup.layout().activate()
        show_fluent_popup_for_anchor(
            self._popup, self.binding_button, self._popup,
            minimum_width=210, minimum_height=1,
            maximum_height=self._popup.sizeHint().height(),
        )

    def _project_popup(self) -> None:
        if self._popup is None or self._field_state is None:
            return
        _editable, scan, source, can_scan, effective, source_text = self._field_state
        with signals_blocked(self.scan_toggle, self.source_combo):
            self.scan_toggle.setChecked(scan)
            self.scan_toggle.setEnabled(can_scan)
            self.source_combo.setCurrentIndex(self.source_combo.findData(source))
        info = "Names: Config tab" if source == "config" else ""
        if effective:
            info = f"Effective: {effective}\n{info}".strip()
        self.source_info.setText(info)
        self.source_info.setToolTip(source_text)
        self.source_info.setVisible(bool(info))
        if self._popup.isVisible():
            self._place_popup()

    def _commit_binding(self, *_args) -> None:
        self.binding_committed.emit(self.scan_toggle.isChecked(), str(self.source_combo.currentData()))

    def set_field_state(self, *, editable: bool, scan: bool = False, source: str = "default",
                        can_scan: bool = True, effective_text: str = "", source_text: str = "") -> None:
        if source not in ("default", "api", "config"):
            raise ValueError("source must be default, api or config")
        state = (bool(editable), bool(scan), source, bool(can_scan), effective_text, source_text)
        if state == self._field_state:
            return
        self._field_state = state
        self.binding_button.scan = bool(scan)
        self.binding_button.source = source
        summary = " + ".join((["Scan"] if scan else []) + ([source.upper()] if source != "default" else [])) or "Default"
        self.binding_button.setToolTip(f"{self._tooltip}\n{summary}\n{source_text}".strip())
        self.binding_button.update()
        self.setReadOnly(not editable)
        if source == "config":
            style = _bound_style(CONFIG_GREEN_DARK, CONFIG_GREEN, applied=bool(effective_text))
        elif source == "api":
            style = _bound_style(API_VIOLET_DARK, API_VIOLET, applied=bool(effective_text))
        elif scan:
            style = _bound_style(ORANGE_DARK, ORANGE)
        else:
            style = self._base_style
        self.setStyleSheet(style)
        self._reserve_right()
        self._project_popup()


__all__ = ["FluentScanLineEdit"]
