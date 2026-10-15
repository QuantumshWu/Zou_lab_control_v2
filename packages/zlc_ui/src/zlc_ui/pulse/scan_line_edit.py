"""Editable default value with independently projected Scan and source bindings."""

from __future__ import annotations

import math

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.fluent import (
    API_VIOLET, API_VIOLET_DARK, CONFIG_GREEN, CONFIG_GREEN_DARK, CONFIG_GREEN_TINT,
    EDIT_PADDING_H, FIELD_BORDER_PX, FONT, ORANGE, ORANGE_DARK, PADDING_V, PLACEHOLDER,
    RADIUS, SURFACE, FluentComboBox, FluentDoubleSpinBox, FluentLabel, FluentLineEdit,
    FluentPopup, FluentSegmentedButtons, FluentSettingsPopupAnchor, FluentSwitch,
    device_snap, field_room_end, field_text_inset, fluent_font_size, fluent_inner_edit_stylesheet,
    frame_inner_rect, line_edit_text_right, paint_centred_ink, scaled_even_px, scaled_px,
    setting_label_width, signals_blocked, read_editable_combo, show_fluent_popup_for_anchor,
)

from ._layout import row_height

#: Where a value comes from, in the order the popup offers them, by the
#: words every surface says them in (the popup, a badge's tooltip, the Scan
#: page's list of bound fields).
_SOURCES = (("default", "Default"), ("api", "API"), ("config", "Config"))


def popup_label_width() -> int:
    """The label column of a binding popup: a value's (Scan, Source, Config
    name, Default, Effective) and a Subpulse's (Source) alike, so the
    Source choice stands on one x in both."""

    return setting_label_width(("Scan", "Source", "Config name", "Default", "Effective"), minimum=0)


class _BindingButton(QtWidgets.QAbstractButton):
    """A column of binding badges, never a cycling or numbered control.

    Seated (:meth:`seat`) in the frame the operator SEES -- the field's
    own, or that of a spin box that adopted the field -- rather than in the
    line edit's rect, which inside a spin box is shorter than the frame
    and sits padding away from the stepper.  The column ends one ``gap``
    before what ends the field's room (:func:`field_room_end`: the frame's
    inner edge, or the stepper's chevron ink), so a badge stands as far
    from the chevrons in a repeat box as from the edge in a plain field.
    The badges are centred on the frame's inner rect, never taller
    together than it with at least a device pixel clear above and below,
    and each letter is centred on its own ink (:func:`paint_centred_ink`).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scan = False
        self.source = "default"
        #: The badge column and the frame's inner height, in this button.
        self._column = QtCore.QRectF()
        self.setCursor(QtCore.Qt.PointingHandCursor)

    @staticmethod
    def gap() -> int:
        """The clear room between the badges and what ends the field's room."""

        return scaled_px(4, minimum=3)

    @staticmethod
    def diameter() -> int:
        """A badge's diameter where the frame has room for it."""

        return scaled_even_px(16, minimum=12)

    def seat(self, frame: QtWidgets.QWidget) -> None:
        """Seat the column in ``frame`` (this button's parent): one gap
        before the end of its room, as tall as its inner rect.  The button
        is the column plus a pixel each side, to take the pointer."""

        inner = frame_inner_rect(frame)
        right = device_snap(frame, field_room_end(frame) - self.gap())
        left = right - self.diameter()
        pad = scaled_px(1)
        x0, x1 = int(math.floor(left)) - pad, int(math.ceil(right)) + pad
        y0, y1 = int(math.ceil(inner.top())), int(math.floor(inner.bottom()))
        self.setGeometry(x0, y0, x1 - x0, max(1, y1 - y0))
        self._column = QtCore.QRectF(left - x0, inner.top() - y0, right - left, inner.height())
        self.update()

    def stand_alone(self, height: int) -> None:
        """Size the button for a place with no field frame around it (a
        Subpulse card's duration row): one badge column, centred in a box
        ``height`` tall the way a field's inner rect centres it."""

        pad = scaled_px(1)
        self.setFixedSize(self.diameter() + 2 * pad, int(height))
        self._column = QtCore.QRectF(
            pad, FIELD_BORDER_PX, self.diameter(), int(height) - 2 * FIELD_BORDER_PX)
        self.update()

    def column_left(self) -> int:
        """Where the badge column begins, in the parent's coordinates (whole
        px, rounded outward): what the text keeps its gap from."""

        return self.x() + int(math.floor(self._column.left()))

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        badges = []
        # Every badge wears its binding's colour at reading strength, so its
        # white letter reads: S on ORANGE_DARK 7.2:1 (on ORANGE 2.4:1), A on
        # API_VIOLET_DARK 7.8:1 (on API_VIOLET 3.2:1), C on CONFIG_GREEN_DARK
        # 6.2:1 (on CONFIG_GREEN 2.0:1).
        if self.scan:
            badges.append(("S", ORANGE_DARK))
        if self.source != "default":
            badges.append((self.source[0].upper(), API_VIOLET_DARK if self.source == "api" else CONFIG_GREEN_DARK))
        if not badges:
            badges.append(("", PLACEHOLDER))
        # In whole device pixels: the frame's inner height, the badges'
        # size and the space between two.  The stack leaves at least that
        # space clear above and below (a logical pixel, never less than a
        # device one), and takes the inner height's parity -- a size, or
        # the space between two, one pixel smaller or larger -- so it is
        # centred on whole pixels: half a pixel low is visible.
        ratio = float(self.devicePixelRatioF())
        column = self._column
        room = int(round(column.height() * ratio))
        count = len(badges)
        between = max(1, int(round(scaled_px(1) * ratio)))
        size = min(int(round(self.diameter() * ratio)), (room - (count + 1) * between) // count)
        if (room - count * size - (count - 1) * between) & 1:
            if count & 1:
                size -= 1
            else:
                between += 1
        size = max(2, size)
        stack = count * size + (count - 1) * between
        across = int(round(column.width() * ratio))
        left = column.left() + (across - size) // 2 / ratio
        font = QtGui.QFont(FONT, max(6, fluent_font_size() - 5))
        font.setBold(True)
        for index, (text, color) in enumerate(badges):
            top = column.top() + ((room - stack) // 2 + index * (size + between)) / ratio
            rect = QtCore.QRectF(left, top, size / ratio, size / ratio)
            painter.setPen(QtCore.Qt.NoPen)
            # A badge says what supplies the value whether or not it can be
            # opened here: on a read-only card it keeps its binding's colour
            # and only stops answering the pointer.
            painter.setBrush(QtGui.QColor(color).darker(
                120 if self.isEnabled() and self.underMouse() else 100))
            painter.drawEllipse(rect)
            if text:
                paint_centred_ink(painter, self, rect, font, text, SURFACE)


def _bound_style(color: str, border: str, *, background: str = SURFACE) -> str:
    return (
        f'QLineEdit {{ background: {background}; color: {color}; border: {FIELD_BORDER_PX}px solid {border}; '
        f'border-radius: {scaled_px(RADIUS)}px; padding: {scaled_px(PADDING_V)}px '
        f'{scaled_px(EDIT_PADDING_H)}px; font: {fluent_font_size()}pt "{FONT}"; }}'
    )


class FluentScanLineEdit(FluentLineEdit):
    """One default value; the popup emits binding intent, never a value override.

    The popup says Scan with an on/off :class:`FluentSwitch` and the value's
    source with a :class:`FluentSegmentedButtons` of Default / API / Config
    (API withheld where a caller cannot supply the field): a click on a
    source chooses it, and the owner projects back what it accepted.
    """

    binding_committed = QtCore.pyqtSignal(bool, str)
    config_key_committed = QtCore.pyqtSignal(str)
    #: A class attribute until __init__ makes the badges: Qt may deliver a
    #: parent change or a resize while the base class is still being built.
    binding_button: _BindingButton | None = None

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
        self._place_button()

    def _frame(self) -> QtWidgets.QWidget:
        """The widget whose frame the operator sees: a spin box that adopted
        this edit (it wears the binding's colours too), else the edit."""

        host = self.parentWidget()
        return host if isinstance(host, FluentDoubleSpinBox) and host.lineEdit() is self else self

    def _place_button(self) -> None:
        """Seat the badges in the frame that is drawn, and end the text a
        gap before them."""

        button = self.binding_button
        if button is None:
            return
        frame = self._frame()
        if button.parentWidget() is not frame:
            # A child of the frame it sits in, over this edit: inside a spin
            # box the edit is shorter than the frame and stops short of the
            # stepper, and a child of it was clipped to it.  It goes with
            # the edit, whoever its parent is.
            button.setParent(frame)
            if frame is not self:
                self.destroyed.connect(button.deleteLater)
            button.show()
        button.seat(frame)
        button.raise_()
        # Where the text may end, in the frame's coordinates: this edit's
        # right edge (as the box lays it out, when a box adopted it) less
        # its own inset.  The digits end as far before the badges as they
        # end before a step chip.
        edge = self.width() if frame is self else frame.edit_field().right() + 1
        text_end = edge - (self.width() - line_edit_text_right(self))
        room = max(0, text_end - (button.column_left() - scaled_px(4, minimum=2)))
        margins = self.textMargins()
        if margins.right() != room:
            self.setTextMargins(margins.left(), margins.top(), room, margins.bottom())

    def event(self, event) -> bool:
        if event.type() == QtCore.QEvent.ParentChange:
            # Adopted by a spin box (or released by one): its frame is now
            # the one the badges sit in.
            self._place_button()
        return super().event(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place_button()

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
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
            # Every row is a control's height, so the five rows keep one
            # pitch and each label reads level with what it names: the read
            # values were a text line tall and the rows came 31, 31, 27, 23
            # apart.
            for row in range(5):
                layout.setRowMinimumHeight(row, row_height())
            # One label column, as every form sizes its own (the widest
            # label and a pad), not the labels' own widths: the Config
            # name box stood 6 px from its label.
            label_width = popup_label_width()
            self.config_name_label = FluentLabel("Config name")
            labels = (FluentLabel("Scan"), FluentLabel("Source"), self.config_name_label,
                      FluentLabel("Default"), FluentLabel("Effective"))
            for row, label in enumerate(labels):
                label.setFixedWidth(label_width)
                layout.addWidget(label, row, 0)
            self.scan_toggle = FluentSwitch("")
            layout.addWidget(self.scan_toggle, 0, 1, QtCore.Qt.AlignLeft)
            self.source_switch = FluentSegmentedButtons(_SOURCES)
            layout.addWidget(self.source_switch, 1, 1, QtCore.Qt.AlignLeft)
            self.config_name = FluentComboBox()
            self.config_name.setEditable(True)
            self.config_name.lineEdit().setPlaceholderText("Choose or type a name")
            layout.addWidget(self.config_name, 2, 1)
            self.default_value = FluentLabel()
            layout.addWidget(self.default_value, 3, 1)
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
                widget.setFixedHeight(row_height())
                # The values read in the column of the Config name's TEXT,
                # not from the edge of its box.
                widget.setIndent(field_text_inset())
            self.source_note.setFixedWidth(label_width + layout.horizontalSpacing() + content_width)
            # Two lines, whatever the source says: the popup keeps its shape
            # as the source changes, and every note is worded to fit two
            # (the name and the value are on the rows above it).  Three
            # left a blank band under the usual two-line note.  Measured as
            # the label lays out two lines, which is a pixel taller than two
            # line spacings -- that cut the descenders of the second line.
            self.source_note.ensurePolished()
            self.source_note.setFixedHeight(self.source_note.fontMetrics().boundingRect(
                QtCore.QRect(0, 0, self.source_note.width(), 1 << 16), QtCore.Qt.TextWordWrap, "Ag\nAg",
            ).height())
            self.scan_toggle.toggled.connect(self._commit_binding)
            self.source_switch.activated.connect(self._choose_source)
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
        (_editable, scan, source, can_scan, effective, source_text, config_key, can_api,
         config_active, config_enabled, default_text, binding_editable) = self._field_state
        with signals_blocked(self.scan_toggle, self.source_switch):
            self.scan_toggle.setChecked(scan)
            self.scan_toggle.setEnabled(can_scan and binding_editable)
            self.source_switch.set_current(source)
            self.source_switch.button("api").setEnabled(can_api)
            self.source_switch.setEnabled(binding_editable)
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
            combo.setEnabled(named and binding_editable)
        self.config_name.setToolTip(config_key if named else "")
        if not source_text:
            if named:
                source_text = (
                    "The saved Config supplies this value." if effective else
                    "Config name is unassigned; using the Pulse default." if not config_key else
                    "That name is not supplied by Config; using the Pulse default."
                )
            elif source == "api":
                source_text = "An API caller may override it; otherwise the Pulse default is used."
            else:
                source_text = "Using the Pulse default."
        default = default_text or self.text()
        for widget, value in ((self.default_value, default),
                              (self.effective_value, (effective or self.text()) if config_active else
                               effective if source != "config" and effective else f"{default} · default")):
            widget.setText(widget.fontMetrics().elidedText(value, QtCore.Qt.ElideMiddle, widget.width()))
            widget.setToolTip(value)
        if named and not config_enabled:
            source_text = "Default mode; the saved Config reference is not applied."
        self.source_note.setText(source_text)
        self.source_note.setToolTip(source_text)

    def set_config_names(self, names: tuple[str, ...]) -> None:
        if names != self._config_names:
            self._config_names = tuple(names)
            self._project_popup()

    def _commit_config_key(self, *_args) -> None:
        if self._field_state is None or self._field_state[2] != "config" or self.source_switch.current() != "config":
            return
        wanted = read_editable_combo(self.config_name)
        with signals_blocked(self.config_name, self.config_name.lineEdit()):
            self.config_name.lineEdit().setModified(False)
            if wanted != self._field_state[6]:
                self.config_key_committed.emit(wanted)
            # The same owner synchronously projects either acceptance or rejection.
            self._project_popup()

    def _choose_source(self, source: str) -> None:
        """A click on a source: shown at once, and asked of the owner, which
        projects back what it accepts.  The source already chosen asks
        nothing."""

        if source == self.source_switch.current():
            return
        self.source_switch.set_current(source)
        self._commit_binding()

    def _commit_binding(self, *_args) -> None:
        self.binding_committed.emit(self.scan_toggle.isChecked(), self.source_switch.current())

    def set_field_state(self, *, editable: bool, scan: bool = False, source: str = "default",
                        can_scan: bool = True, effective_text: str = "", source_text: str = "",
                        config_key: str = "", can_api: bool = True,
                        config_active: bool = False, config_enabled: bool = False,
                        default_text: str = "", binding_editable: bool = True) -> None:
        if source not in ("default", "api", "config"):
            raise ValueError("source must be default, api or config")
        state = (bool(editable), bool(scan), source, bool(can_scan), effective_text, source_text, config_key,
                 bool(can_api), bool(config_active), bool(config_enabled), default_text, bool(binding_editable))
        if state == self._field_state:
            self._project_popup()
            return
        self._field_state = state
        self.binding_button.scan = bool(scan)
        self.binding_button.source = source
        # The note under the popup leaves the Config name to its own row;
        # the tooltip, read alone, names it.
        named = f"Config '{config_key}'" if source == "config" and config_key else dict(_SOURCES)[source]
        summary = " + ".join((["Scan"] if scan else []) + ([named] if source != "default" else [])) or "Default"
        self.binding_button.setToolTip(f"{self._tooltip}\n{summary}\n{source_text}".strip())
        self.binding_button.update()
        self.setReadOnly(not editable or config_active)
        self.setEnabled(editable or config_active)
        self.binding_button.setEnabled(binding_editable and (editable or config_active))
        if not self.binding_button.isEnabled() and self._popup is not None:
            self._popup.hide()
        # The binding's colours: (ink, edge, fill), or None for a plain field.
        if not editable and not config_active:
            if self._popup is not None:
                self._popup.hide()
            tone = None
        elif config_active:
            tone = (CONFIG_GREEN_DARK, CONFIG_GREEN, CONFIG_GREEN_TINT)
        elif source == "config":
            tone = (CONFIG_GREEN_DARK, CONFIG_GREEN, SURFACE)
        elif source == "api":
            tone = (SURFACE if effective_text else API_VIOLET_DARK, API_VIOLET,
                    API_VIOLET_DARK if effective_text else SURFACE)
        elif scan:
            tone = (ORANGE_DARK, ORANGE, SURFACE)
        else:
            tone = None
        host = self.parentWidget()
        if isinstance(host, FluentDoubleSpinBox) and host.lineEdit() is self:
            # Inside a spin box the frame is the box's: it wears the colours.
            if tone is None:
                host.set_tone()
                self.setStyleSheet(fluent_inner_edit_stylesheet())
            else:
                ink, edge, fill = tone
                host.set_tone(edge=edge, fill=fill, ink=ink)
                self.setStyleSheet(fluent_inner_edit_stylesheet(ink))
        elif tone is None:
            self.setStyleSheet(self._base_style)
        else:
            ink, edge, fill = tone
            self.setStyleSheet(_bound_style(ink, edge, background=fill))
        self._place_button()
        self._project_popup()
        if self._popup is not None and self._popup.isVisible():
            self._place_popup()


__all__ = ["FluentScanLineEdit"]
