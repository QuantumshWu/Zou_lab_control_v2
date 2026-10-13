"""Editable default value with independently projected Scan and source bindings."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.fluent import (
    API_VIOLET, API_VIOLET_DARK, CONFIG_GREEN, CONFIG_GREEN_DARK, CONFIG_GREEN_TINT,
    EDIT_PADDING_H, FONT, ORANGE, ORANGE_DARK, PADDING_V, PLACEHOLDER,
    RADIUS, SURFACE, FluentCheckBox, FluentComboBox, FluentTriSwitch, FluentLabel,
    FluentLineEdit, FluentPopup, FluentSettingsPopupAnchor, fluent_font_size, scaled_px, signals_blocked,
    read_editable_combo, show_fluent_popup_for_anchor,
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
            painter.setBrush(QtGui.QColor(color if self.isEnabled() else PLACEHOLDER).darker(
                120 if self.isEnabled() and self.underMouse() else 100))
            painter.drawEllipse(rect.adjusted(0.5, 0.5, -0.5, -0.5))
            painter.setPen(QtGui.QColor(SURFACE))
            painter.drawText(rect, QtCore.Qt.AlignCenter, text)


def _bound_style(color: str, border: str, *, background: str = SURFACE) -> str:
    return (
        f'QLineEdit {{ background: {background}; color: {color}; border: 1px solid {border}; '
        f'border-radius: {scaled_px(RADIUS)}px; padding: {scaled_px(PADDING_V)}px '
        f'{scaled_px(EDIT_PADDING_H)}px; font: {fluent_font_size()}pt "{FONT}"; }}'
    )


class FluentScanLineEdit(FluentLineEdit):
    """One default value; the popup emits binding intent, never a value override."""

    binding_committed = QtCore.pyqtSignal(bool, str)
    config_key_committed = QtCore.pyqtSignal(str)

    def __init__(self, text: str = "", parent=None, *, tooltip: str = "Edit Scan and value source") -> None:
        super().__init__(text, parent)
        self._base_style = self.styleSheet()
        self.binding_button = _BindingButton(self)
        self.binding_button.clicked.connect(self._toggle_binding)
        self._tooltip = tooltip
        self._field_state = None
        self._config_names: tuple[str, ...] = ()
        self._popup = None
        self.textChanged.connect(self._project_popup)
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

    def _toggle_binding(self) -> None:
        if self._popup is None:
            self._popup = FluentPopup(self)
            self._popup_anchor = FluentSettingsPopupAnchor(self._popup, self.binding_button)
            layout = QtWidgets.QGridLayout(self._popup)
            margin = scaled_px(10)
            layout.setContentsMargins(margin, margin, margin, margin)
            layout.setSpacing(scaled_px(6))
            layout.addWidget(FluentLabel("Scan"), 0, 0)
            self.scan_toggle = FluentCheckBox("")
            layout.addWidget(self.scan_toggle, 0, 1, QtCore.Qt.AlignLeft)
            layout.addWidget(FluentLabel("Source"), 1, 0)
            self.source_switch = FluentTriSwitch(("Default", "API", "Config"))
            layout.addWidget(self.source_switch, 1, 1, QtCore.Qt.AlignLeft)
            self.config_name_label = FluentLabel("Config name")
            self.config_name = FluentComboBox()
            self.config_name.setEditable(True)
            self.config_name.lineEdit().setPlaceholderText("Choose or type a name")
            layout.addWidget(self.config_name_label, 2, 0)
            layout.addWidget(self.config_name, 2, 1)
            layout.addWidget(FluentLabel("Default"), 3, 0)
            self.default_value = FluentLabel()
            layout.addWidget(self.default_value, 3, 1)
            layout.addWidget(FluentLabel("Effective"), 4, 0)
            self.effective_value = FluentLabel()
            layout.addWidget(self.effective_value, 4, 1)
            self.source_note = FluentLabel()
            self.source_note.setWordWrap(True)
            self.source_note.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            layout.addWidget(self.source_note, 5, 0, 1, 2)
            content_width = max(self.source_switch.sizeHint().width(), scaled_px(230))
            for widget in (self.config_name, self.default_value, self.effective_value):
                widget.setFixedWidth(content_width)
            for widget in (self.default_value, self.effective_value):
                widget.setFixedHeight(widget.fontMetrics().height())
            self.source_note.setFixedWidth(
                self.config_name_label.sizeHint().width() + layout.horizontalSpacing() + content_width)
            self.source_note.setFixedHeight(self.source_note.fontMetrics().lineSpacing() * 3)
            self.scan_toggle.toggled.connect(self._commit_binding)
            self.source_switch.stateChanged.connect(self._commit_binding)
            self.config_name.activated.connect(self._commit_config_key)
            self.config_name.editingFinished.connect(self._commit_config_key)
        self._popup_anchor.toggle(self._popup, prepare=self._project_popup, present=self._place_popup)

    def _place_popup(self) -> None:
        self._popup.ensurePolished()
        self._popup.layout().activate()
        show_fluent_popup_for_anchor(
            self._popup, self.binding_button, self._popup,
            minimum_width=1, minimum_height=1,
            maximum_height=self._popup.sizeHint().height(),
        )

    def _project_popup(self, *_args) -> None:
        if self._popup is None or self._field_state is None:
            return
        _editable, scan, source, can_scan, effective, source_text, config_key, can_api = self._field_state
        with signals_blocked(self.scan_toggle, self.source_switch):
            self.scan_toggle.setChecked(scan)
            self.scan_toggle.setEnabled(can_scan)
            self.source_switch.setState(("default", "api", "config").index(source))
            self.source_switch.set_position_offered(1, can_api)
        named = source == "config"
        combo = self.config_name
        draft = combo.currentText() if named and combo.lineEdit().hasFocus() and combo.lineEdit().isModified() else None
        with signals_blocked(combo, combo.lineEdit()):
            if tuple(combo.itemText(index) for index in range(combo.count())) != self._config_names:
                combo.clear()
                combo.addItems(self._config_names)
            combo.setCurrentIndex(combo.findText(config_key) if named else -1)
            combo.setEditText((config_key if draft is None else draft) if named else "")
            combo.lineEdit().setModified(draft is not None)
            combo.setEnabled(named)
        self.config_name.setToolTip(config_key if named else "")
        if not source_text:
            if named:
                source_text = (
                    f"Saved Config '{config_key}' supplies this value." if effective else
                    "Config name is unassigned; using the Pulse default." if not config_key else
                    f"Config '{config_key}' is not supplied; using the Pulse default."
                )
            elif source == "api":
                source_text = "An API caller may override this field; otherwise its Pulse default is used."
            else:
                source_text = "Using the Pulse default."
        for widget, value in ((self.default_value, self.text()),
                              (self.effective_value, effective or f"{self.text()} · default")):
            widget.setText(widget.fontMetrics().elidedText(value, QtCore.Qt.ElideMiddle, widget.width()))
            widget.setToolTip(value)
        self.source_note.setText(source_text)
        self.source_note.setToolTip(source_text)

    def set_config_names(self, names: tuple[str, ...]) -> None:
        if names != self._config_names:
            self._config_names = tuple(names)
            self._project_popup()

    def _commit_config_key(self, *_args) -> None:
        if self._field_state is None or self._field_state[2] != "config" or self.source_switch.state() != 2:
            return
        wanted = read_editable_combo(self.config_name)
        with signals_blocked(self.config_name, self.config_name.lineEdit()):
            self.config_name.lineEdit().setModified(False)
            if wanted != self._field_state[6]:
                self.config_key_committed.emit(wanted)
            # The same owner synchronously projects either acceptance or rejection.
            self._project_popup()

    def _commit_binding(self, *_args) -> None:
        self.binding_committed.emit(self.scan_toggle.isChecked(), ("default", "api", "config")[self.source_switch.state()])

    def set_field_state(self, *, editable: bool, scan: bool = False, source: str = "default",
                        can_scan: bool = True, effective_text: str = "", source_text: str = "",
                        config_key: str = "", can_api: bool = True) -> None:
        if source not in ("default", "api", "config"):
            raise ValueError("source must be default, api or config")
        state = (bool(editable), bool(scan), source, bool(can_scan), effective_text, source_text, config_key,
                 bool(can_api))
        if state == self._field_state:
            self._project_popup()
            return
        self._field_state = state
        self.binding_button.scan = bool(scan)
        self.binding_button.source = source
        summary = " + ".join((["Scan"] if scan else []) + ([source.upper()] if source != "default" else [])) or "Default"
        self.binding_button.setToolTip(f"{self._tooltip}\n{summary}\n{source_text}".strip())
        self.binding_button.update()
        self.setReadOnly(not editable)
        self.setEnabled(editable)
        if not editable:
            if self._popup is not None:
                self._popup.hide()
            style = self._base_style
        elif source == "config":
            style = _bound_style(CONFIG_GREEN_DARK, CONFIG_GREEN,
                                 background=CONFIG_GREEN_TINT if effective_text else SURFACE)
        elif source == "api":
            style = _bound_style(SURFACE if effective_text else API_VIOLET_DARK, API_VIOLET,
                                 background=API_VIOLET_DARK if effective_text else SURFACE)
        elif scan:
            style = _bound_style(ORANGE_DARK, ORANGE)
        else:
            style = self._base_style
        self.setStyleSheet(style)
        self._reserve_right()
        self._project_popup()
        if self._popup is not None and self._popup.isVisible():
            self._place_popup()


__all__ = ["FluentScanLineEdit"]
