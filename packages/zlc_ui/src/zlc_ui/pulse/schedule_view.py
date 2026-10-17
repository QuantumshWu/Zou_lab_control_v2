"""Pure Qt projection for the pulse schedule page; it accepts only
:mod:`zlc_ui.pulse.models` records, never a domain document.
"""

from __future__ import annotations

import json
import weakref
from dataclasses import replace

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.form import FormChoice, being_edited
from zlc_ui.fluent import (
    retire_widget,
    ACCENT, ACCENT_TINT, BG, DERIVED_INK, DISABLED_INK, FIELD_BORDER_PX, HOVER, MUTED_INK,
    SELECTION_EDGE, TEXT, CONFIG_GREEN,
    CONFIG_GREEN_DARK, CONFIG_GREEN_TINT, FluentButton, FluentCheckBox, FluentSegmentedButtons,
    FluentSwitch, FluentComboBox, FluentFrame, FluentGroupBox, fluent_count_box,
    FluentLabel, FluentLineEdit, FluentPageBody, FluentReadoutEdit, FluentScrollArea, LinkedScrollPanes,
    ElidedLabel, FluentPopup, FluentSettingRow, FluentSettingsPopupAnchor, field_text_inset,
    set_status_ink, show_fluent_popup_for_anchor, signals_blocked,
)

from ._layout import (
    card_caption, channel_name_edit_width, count_box_width,
    label_column_width, panel_top_height, period_card_width, period_control_width, row_spacing,
    spacer_card_width, spacer_control_width, time_unit_width,
)
from zlc_ui.fluent import card_pad, control_height, row_gap, scaled_px, section_gap
from .models import (
    PERIOD_KIND_SPACER,
    VALIDATOR_FLOAT,
    VALIDATOR_INT,
    ConnectionVM,
    ComponentVM,
    DelayRowVM,
    FieldVM,
    PeriodVM,
    PortRowVM,
    BracketVM,
    bracket_gap_bounds,
    bracket_post_key,
    ScheduleVM,
)

#: The address the pulse server prints for a client on the same machine.  One
#: constant so the seeded field, the tooltip and any host default agree.
from .scan_line_edit import FluentScanLineEdit, _BindingButton, popup_label_width
from zlc_ui.status import counted


def _apply_field(widget: FluentScanLineEdit, field: FieldVM) -> None:
    with signals_blocked(widget):
        widget.setReadOnly(not field.editable or field.config_active)
        widget.setProperty("pulse_field_id", field.field_id)
        widget.set_config_names(field.config_names)
        if field.validator_kind in (VALIDATOR_INT, VALIDATOR_FLOAT):
            widget.set_numeric_validator(
                field.validator_kind,
                bottom=field.validator_lo,
                top=field.validator_hi if field.validator_kind == VALIDATOR_INT else None,
            )
            widget.set_resolution(field.resolution or None)
            widget.set_allow_any(field.allow_any)
        else:
            widget.setValidator(None)
        widget.setText(field.text)
        widget.set_field_state(editable=field.editable, scan=field.scan, source=field.source,
                               can_scan=field.can_scan, effective_text=field.effective_text,
                               source_text=field.source_text, config_key=field.config_key,
                               can_api=field.can_api, config_active=field.config_active,
                               config_enabled=field.config_enabled, default_text=field.default_text,
                               binding_editable=field.binding_editable)


class _PortToggleIcon(QtGui.QIconEngine):
    """A Delay row's every-period on/off command: one ring, with a dot in
    it for "on", both drawn: the same ring, the same stroke, in the
    command's own ink -- its button's :meth:`FluentButton.glyph_ink`, so
    the ring follows the button's state exactly as a one-glyph label does.
    The button is held weakly (it owns its icon); a copy Qt paints after
    the button is gone takes the standard command's ink."""

    def __init__(self, on: bool, button: "weakref.ref[FluentButton]") -> None:
        super().__init__()
        self._on = bool(on)
        self._button = button

    def paint(self, painter, rect, mode, state) -> None:  # noqa: N802 - Qt API name
        del state
        button = self._button()
        if button is not None:
            ink = QtGui.QColor(button.glyph_ink())
        else:
            ink = QtGui.QColor(DISABLED_INK if mode == QtGui.QIcon.Disabled else HOVER)
        side = float(min(rect.width(), rect.height()))
        stroke = max(1.0, side / 9.0)
        centre = QtCore.QRectF(rect).center()
        radius = (side - stroke) / 2.0
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(ink, stroke))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(centre, radius, radius)
        if self._on:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(ink)
            dot = radius * 0.5
            painter.drawEllipse(centre, dot, dot)
        painter.restore()

    def pixmap(self, size, mode, state):  # noqa: N802 - Qt API name
        image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32_Premultiplied)
        image.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(image)
        self.paint(painter, QtCore.QRect(QtCore.QPoint(0, 0), size), mode, state)
        painter.end()
        return QtGui.QPixmap.fromImage(image)

    def clone(self):
        return _PortToggleIcon(self._on, self._button)


def _show_port_ring(button: FluentButton, on: bool) -> None:
    """Draw ``button`` as the ring (``on``: with its dot), in its own ink."""

    button.setIcon(QtGui.QIcon(_PortToggleIcon(on, weakref.ref(button))))


def _port_toggle_button(on: bool, tooltip: str) -> FluentButton:
    button = FluentButton("")
    _show_port_ring(button, on)
    side = control_height()
    # About half the face, of the face's parity: an odd ring in an even
    # face has no centred place.
    glyph = side // 2
    glyph = max(8 + side % 2, glyph - (side - glyph) % 2)
    button.setIconSize(QtCore.QSize(glyph, glyph))
    button.setFixedSize(side, side)
    button.setToolTip(tooltip)
    return button


def _read_only(field: FieldVM) -> FieldVM:
    """A field shown on a card whose content cannot be edited here (a
    member of a Subpulse that comes from Config): not editable, its binding
    not either.  It is then drawn the way every switched-off control on
    that card is -- grey, like its unit combo and its checkboxes -- unless
    Config supplies it, which keeps the config look it has on every card.
    """

    return replace(field, editable=False, binding_editable=False)


#: Past any viewport's far edge: a mask that only cuts its leading gap.
_UNBOUNDED = 1 << 20


def _place_rows_in_order(layout: QtWidgets.QVBoxLayout, holders) -> None:
    """Stack a column's row holders under its top block, in port order.

    The columns are read across, so row N of each is the same output, even
    for a port shown again or inserted between two others.
    """

    for index, holder in enumerate(holders, start=1):
        if layout.indexOf(holder) != index:
            layout.removeWidget(holder)
            layout.insertWidget(index, holder)


def _card_column(card: QtWidgets.QWidget) -> tuple[QtWidgets.QVBoxLayout, QtWidgets.QWidget,
                                                   QtWidgets.QVBoxLayout]:
    """The column every card and side panel of the strip is laid in -- the
    card pad round it, the port rows' pitch down it -- and the header block
    it opens with (:func:`panel_top_height`) with that block's layout."""

    column = QtWidgets.QVBoxLayout(card)
    pad = card_pad()
    column.setContentsMargins(pad, pad, pad, pad)
    column.setSpacing(row_spacing())
    top = QtWidgets.QWidget()
    top.setStyleSheet("background: transparent;")
    top.setFixedHeight(panel_top_height())
    top_layout = QtWidgets.QVBoxLayout(top)
    top_layout.setContentsMargins(0, 0, 0, 0)
    top_layout.setSpacing(row_gap())
    return column, top, top_layout


def _header_rows(layout: QtWidgets.QVBoxLayout, rows) -> list[ElidedLabel]:
    """A side panel's header rows, ``label | control`` each, a control's
    height; their labels, which the panel sizes to the ones it shows."""

    labels = []
    for text, widget in rows:
        labels.append(ElidedLabel(text))
        widget.setFixedHeight(control_height())
        layout.addWidget(FluentSettingRow(labels[-1], widget))
    return labels


class PeriodCard(FluentGroupBox):
    """One stable period card keyed by ``period_id``.

    A SPACER is a period of another kind and takes the same card, narrowed:
    the page's grey with a dashed edge and a muted "Spacer" pill, its
    duration, unit and name on the same lines as every other card's -- the
    name given by the editor and not editable -- its channel circles
    without labels (the neighbours' rows say which is which; the label is
    on hover), and each DAC as a disabled "Hold": a spacer holds every DAC.
    """

    period_name_committed = QtCore.pyqtSignal(str, str)
    duration_committed = QtCore.pyqtSignal(str, object, str)
    digital_committed = QtCore.pyqtSignal(str, str, bool)
    analog_committed = QtCore.pyqtSignal(str, str, str, object)
    binding_committed = QtCore.pyqtSignal(str, object, object, bool, str)
    config_binding_committed = QtCore.pyqtSignal(str, str)
    feedback_requested = QtCore.pyqtSignal(str)

    def __init__(self, period: PeriodVM, *, index: int = 0, total_periods: int = 1,
                 ports: tuple[PortRowVM, ...] = (),
                 analog_mode_choices: tuple[FormChoice, ...] = (), card_width: int | None = None,
                 parent=None) -> None:
        if not isinstance(period, PeriodVM):
            raise TypeError("period must be PeriodVM")
        self.kind = period.kind
        spacer = period.kind == PERIOD_KIND_SPACER
        super().__init__("", parent, title_color=MUTED_INK if spacer else TEXT)
        self.period_id = period.period_id
        self._period = period
        self._ports: dict[str, PortRowVM] = {}
        self.checks: dict[str, FluentCheckBox] = {}
        self.bus_mode_combos: dict[str, FluentComboBox] = {}
        self.bus_value_edits: dict[str, FluentScanLineEdit] = {}
        self.port_rows: dict[str, QtWidgets.QWidget] = {}

        width = spacer_card_width() if spacer else (
            card_width if card_width is not None else period_card_width(
                ports=ports, analog_mode_choices=analog_mode_choices))
        self.setFixedWidth(width)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        if spacer:
            self.set_surface(BG, dashed=True)
        column, top, top_layout = _card_column(self)
        # The duration box sits on the same line on every card: a spacer
        # leaves the label's line empty rather than moving the box up into it.
        top_layout.addWidget(self._blank_line() if spacer else card_caption("Duration"))
        self.duration_edit = FluentScanLineEdit(
            "",
            tooltip="Edit value source" if spacer else "Edit Scan and value source",
        )
        control_width = spacer_control_width(width) if spacer else period_control_width(width)
        self.duration_edit.setFixedWidth(control_width)
        top_layout.addWidget(self.duration_edit)
        self.unit_combo = FluentComboBox()
        self.unit_combo.setFixedWidth(control_width)
        top_layout.addWidget(self.unit_combo)
        self.name_edit = FluentLineEdit("")
        # No label repeats it on the card: the placeholder is the field's
        # only name, so it reads in MUTED_INK, not as a faint prompt.
        self.name_edit.setPlaceholderText("Name")
        self.name_edit.setFixedWidth(control_width)
        # A spacer is named by the editor, not the operator: the name is
        # shown where every card shows its name and cannot be typed into
        # (set_period switches it off).
        top_layout.addWidget(self.name_edit)
        # Each header line one row tall, whatever height a control would
        # take at this scale, so row N stands on one y across the strip.
        for widget in (self.duration_edit, self.unit_combo, self.name_edit):
            widget.setFixedHeight(control_height())
        top_layout.addStretch(1)
        column.addWidget(top)
        self._column = column
        self.duration_edit.editingFinished.connect(self._commit_duration)
        self.duration_edit.valueNormalized.connect(self._commit_duration)
        self.duration_edit.binding_committed.connect(
            lambda scan, source: self.binding_committed.emit("duration", self.period_id, None, scan, source)
        )
        self.duration_edit.config_key_committed.connect(
            lambda key: self.config_binding_committed.emit(self.duration_edit.property("pulse_field_id"), key)
        )
        self.unit_combo.currentTextChanged.connect(self._commit_duration)
        self.name_edit.editingFinished.connect(self._commit_name)
        self.set_period(
            period,
            index=index,
            total_periods=total_periods,
            ports=ports,
            analog_mode_choices=analog_mode_choices,
        )
        column.addStretch(1)

    @staticmethod
    def _blank_line() -> QtWidgets.QWidget:
        blank = QtWidgets.QWidget()
        blank.setStyleSheet("background: transparent;")
        blank.setFixedHeight(control_height())
        return blank

    def set_period(self, period: PeriodVM, *, index: int, total_periods: int,
                   ports: tuple[PortRowVM, ...],
                   analog_mode_choices: tuple[FormChoice, ...], card_width: int | None = None) -> None:
        if period.period_id != self.period_id:
            raise ValueError("period identity cannot change")
        if period.kind != self.kind:
            raise ValueError("a period cannot change kind under its card")
        if card_width is not None and self.kind != PERIOD_KIND_SPACER:
            self.setFixedWidth(card_width)
            for widget in (self.duration_edit, self.unit_combo, self.name_edit):
                widget.setFixedWidth(period_control_width(card_width))
        self._period = period
        # Spacers are not counted: "Period 2/5" numbers the periods someone
        # wrote, and a spacer is the gap between two of them.
        self.setTitle(
            "Spacer" if self.kind == PERIOD_KIND_SPACER
            else f"Period {int(index) + 1}/{max(1, int(total_periods))}"
        )
        with signals_blocked(self.unit_combo):
            choices = period.unit_choices or (period.unit,)
            if tuple(self.unit_combo.itemText(i) for i in range(self.unit_combo.count())) != choices:
                self.unit_combo.clear()
                self.unit_combo.addItems([str(value) for value in choices])
            self.unit_combo.setCurrentText(period.unit)
        _apply_field(self.duration_edit, _read_only(period.duration) if period.readonly else period.duration)
        self.unit_combo.setEnabled(not period.readonly and not period.duration.config_active)
        self.name_edit.setText(period.name)
        # One look for what cannot be edited here: switched off, like the
        # unit combo and the checkboxes beside it.
        self.name_edit.setEnabled(self.kind != PERIOD_KIND_SPACER and not period.readonly)
        self._analog_mode_choices = tuple(analog_mode_choices)
        self._reconcile_ports(tuple(ports), period)
        # Switched off like everything else on a read-only card, not left
        # live and inert: a box that looks tickable and is not is the one
        # thing this card must not show.  A ticked box keeps a fill of its
        # own while switched off (FluentCheckBox), so which outputs the
        # member drives still reads at a glance.
        for check in self.checks.values():
            check.setEnabled(not period.readonly)
        for key, combo in self.bus_mode_combos.items():
            field = next((field for port, _mode, field in period.analog if port == key), None)
            combo.setEnabled(self.kind != PERIOD_KIND_SPACER and not period.readonly
                             and not (field and field.config_active))
        self.setProperty("readonly_content", period.readonly)

    def _set_analog_mode(self, combo: FluentComboBox, mode: str) -> None:
        with signals_blocked(combo):
            existing = tuple((combo.itemText(i), combo.itemData(i)) for i in range(combo.count()))
            desired = tuple((choice.label, choice.value) for choice in self._analog_mode_choices)
            if existing != desired:
                combo.clear()
                for label, value in desired:
                    combo.addItem(label, value)
            selected = combo.findData(mode)
            if selected < 0:
                raise ValueError(f"analog mode {mode!r} was not supplied to the view")
            combo.setCurrentIndex(selected)

    def _reconcile_ports(self, ports: tuple[PortRowVM, ...], period: PeriodVM) -> None:
        desired = {port.key: port for port in ports if port.visible}
        for key in tuple(self.port_rows):
            if key not in desired:
                widget = self.port_rows.pop(key)
                self._column.removeWidget(widget)
                retire_widget(widget)
                self.checks.pop(key, None)
                self.bus_mode_combos.pop(key, None)
                self.bus_value_edits.pop(key, None)
        digital = dict(period.digital)
        analog = {key: (mode, field) for key, mode, field in period.analog}
        for port in ports:
            if not port.visible:
                continue
            known = self._ports.get(port.key)
            if known is not None and known.kind != port.kind:
                # The same key with another KIND is another row: a digital
                # checkbox cannot become a DAC mode-and-value editor by
                # being told a new value.
                widget = self.port_rows.pop(port.key)
                self._column.removeWidget(widget)
                retire_widget(widget)
                self.checks.pop(port.key, None)
                self.bus_mode_combos.pop(port.key, None)
                self.bus_value_edits.pop(port.key, None)
            if port.key in self.port_rows:
                if port.kind == "digital":
                    check = self.checks.get(port.key)
                    if check is not None:
                        with signals_blocked(check):
                            check.setChecked(bool(digital.get(port.key, False)))
                else:
                    combo = self.bus_mode_combos.get(port.key)
                    edit = self.bus_value_edits.get(port.key)
                    if combo is not None and edit is not None:
                        if port.key not in analog:
                            raise ValueError(
                                f"period {period.period_id!r} has no analog row for {port.key!r}"
                            )
                        mode, field = analog[port.key]
                        self._set_analog_mode(combo, mode)
                        _apply_field(edit, _read_only(field) if period.readonly else field)
                continue
            if port.kind == "digital":
                widget = FluentCheckBox("" if self.kind == PERIOD_KIND_SPACER else port.label)
                if self.kind == PERIOD_KIND_SPACER:
                    widget.setToolTip(port.label)
                widget.setChecked(bool(digital.get(port.key, False)))
                widget.setFixedHeight(control_height())
                widget.toggled.connect(lambda checked, key=port.key: self.digital_committed.emit(self.period_id, key, bool(checked)))
                self.checks[port.key] = widget
            else:
                if port.key not in analog:
                    raise ValueError(
                        f"period {period.period_id!r} has no analog row for {port.key!r}"
                    )
                mode, field = analog[port.key]
                widget = QtWidgets.QWidget()
                widget.setFixedHeight(control_height())
                row_layout = QtWidgets.QHBoxLayout(widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(row_gap())
                combo = FluentComboBox()
                if self.kind == PERIOD_KIND_SPACER:
                    # A spacer holds every DAC: the one mode it has is shown,
                    # cannot be changed, and there is no value to type.
                    label = next(
                        (choice.label for choice in self._analog_mode_choices if choice.value == mode),
                        mode,
                    )
                    combo.addItem(label, mode)
                    combo.setEnabled(False)
                    combo.setToolTip(f"{port.label}: a spacer holds the DAC")
                    row_layout.addWidget(combo, 1)
                    self.bus_mode_combos[port.key] = combo
                    self.port_rows[port.key] = widget
                    continue
                self._set_analog_mode(combo, mode)
                combo.setSizePolicy(
                    QtWidgets.QSizePolicy.Fixed,
                    QtWidgets.QSizePolicy.Fixed,
                )
                # The port's own code range, enforced where it is typed.  It
                # was carried all the way here on the row and then read by
                # nobody, so a 10-bit signed DAC accepted 700: refused later by
                # the model, on a path the operator had already left.  A limit
                # that exists must be the limit the box has.
                edit = FluentScanLineEdit(
                    field.text,
                    tooltip=(
                        f"{port.label}: signed integer {port.lo}..{port.hi} "
                        "(0 = 0 V)\n"
                        "Edit Scan and value source"
                    ),
                )
                edit.set_numeric_validator("int", bottom=port.lo, top=port.hi)
                edit.setFixedHeight(control_height())
                _apply_field(edit, _read_only(field) if period.readonly else field)
                combo.currentIndexChanged[int].connect(
                    lambda _index, key=port.key: self._commit_analog(key)
                )
                edit.editingFinished.connect(lambda key=port.key: self._commit_analog(key))
                edit.valueNormalized.connect(lambda key=port.key: self._commit_analog(key))
                edit.binding_committed.connect(lambda scan, source, key=port.key: self.binding_committed.emit("analog", self.period_id, key, scan, source))
                edit.config_key_committed.connect(
                    lambda key, field=edit: self.config_binding_committed.emit(field.property("pulse_field_id"), key)
                )
                row_layout.addWidget(combo)
                row_layout.addWidget(edit, 1)
                self.bus_mode_combos[port.key] = combo
                self.bus_value_edits[port.key] = edit
            self.port_rows[port.key] = widget
        ordered = [port.key for port in ports if port.visible]
        for index, key in enumerate(ordered):
            widget = self.port_rows[key]
            self._column.removeWidget(widget)
            self._column.insertWidget(1 + index, widget)
            widget.show()
        self._ports = {port.key: port for port in ports}

    def _commit_name(self) -> None:
        if self._period.readonly:
            return
        value = self.name_edit.text()
        if value == self._period.name:
            return
        self.period_name_committed.emit(self.period_id, value)

    def _commit_duration(self, _unit: str | None = None) -> None:
        if not self.unit_combo.isEnabled():
            return
        if self.duration_edit.property("numericError"):
            self.feedback_requested.emit(str(self.duration_edit.property("numericError")))
            return
        try:
            value = float(self.duration_edit.text())
        except ValueError:
            return
        unit = self.unit_combo.currentText()
        try:
            if (value, unit) == (float(self._period.duration.text), self._period.unit):
                return
        except ValueError:
            pass  # The previous projection can be a non-numeric binding label.
        self.duration_committed.emit(self.period_id, value, unit)

    def _commit_analog(self, port: str) -> None:
        combo = self.bus_mode_combos.get(port)
        edit = self.bus_value_edits.get(port)
        if combo is None or edit is None:
            return
        if not combo.isEnabled():
            return
        mode = combo.currentData()
        if not isinstance(mode, str) or not mode:
            raise ValueError("the selected analog mode has no domain value")
        previous_mode, field = next(
            (previous_mode, field) for key, previous_mode, field in self._period.analog if key == port
        )
        if edit.isReadOnly():
            if mode == previous_mode:
                return
            # Hold's inherited level is read-only; its mode is not. Start
            # the new step from the accepted held level, not widget edits.
            text = field.text
        else:
            if edit.property("numericError"):
                self.feedback_requested.emit(str(edit.property("numericError")))
                return
            text = edit.text()
        try:
            value = int(float(text))
        except ValueError:
            return
        # Only the accepted projection says what is unchanged. A previous
        # intent may have been refused, or superseded by Open/Sync/Clear all.
        try:
            if (mode, value) == (previous_mode, int(float(field.text))):
                return
        except ValueError:
            pass
        self.analog_committed.emit(self.period_id, port, mode, value)

    def set_port_label(self, port: str, label: str) -> None:
        check = self.checks.get(port)
        if check is None:
            return
        if self.kind == PERIOD_KIND_SPACER:
            check.setToolTip(str(label))
        else:
            check.setText(str(label))


class ChannelNamesPanel(FluentGroupBox):
    """The pulse's name and totals, then one row per port: its endpoint
    code and its label.

    ONE label column (its widest shown label and the air after it, the
    names and the codes alike) and ONE field column: the Name field and
    every port's label field share both edges, and a total read beside them
    starts where their typed text starts.

    The Name is where the schedule is renamed, on every page: the Pulse on
    Edit, the Subpulse on its own page.  The port labels are the board's,
    named on Edit; with ``ports_editable`` off they are read here, as
    readouts in the same column as the totals above them.
    """

    document_name_committed = QtCore.pyqtSignal(str)
    port_label_committed = QtCore.pyqtSignal(str, str)

    def __init__(self, parent=None, *, ports_editable: bool = True) -> None:
        super().__init__("Port catalog", parent)
        self._ports_editable = ports_editable
        self.name_edit = FluentLineEdit("")
        self.name_edit.setPlaceholderText("Pulse name")
        self.name_edit.set_placeholder_prompt()
        self.name_edit.editingFinished.connect(
            lambda: self.document_name_committed.emit(self.name_edit.text()))
        # What the pulse adds up to is read, not typed: readouts, not
        # switched-off fields.
        self.total_label = FluentReadoutEdit("")
        self.periods_label = FluentReadoutEdit("")
        self.visible_label = FluentReadoutEdit("")
        self._layout, top, top_layout = _card_column(self)
        self._labels = _header_rows(top_layout, (
            ("Name", self.name_edit), ("Total", self.total_label),
            ("Periods", self.periods_label), ("Visible", self.visible_label)))
        top_layout.addStretch(1)
        self._layout.addWidget(top)
        self._rows: dict[str, FluentLineEdit] = {}
        self._row_holders: dict[str, QtWidgets.QWidget] = {}
        self._row_labels: dict[str, ElidedLabel] = {}
        self._layout.addStretch(1)
        self._fit_label_column()

    def _fit_label_column(self) -> None:
        """Size the label column to the labels shown and the panel to it."""

        shown = [label.text() for label in self._labels]
        shown += [label.text() for key, label in self._row_labels.items()
                  if not self._row_holders[key].isHidden()]
        width = label_column_width(shown)
        for label in (*self._labels, *self._row_labels.values()):
            label.setFixedWidth(width)
        self.setFixedWidth(2 * card_pad() + width + row_gap() + channel_name_edit_width())

    def set_ports(self, document_name: str, ports: tuple[PortRowVM, ...]) -> None:
        self.name_edit.setText(str(document_name))
        existing = self._rows
        self._rows = {}
        for port in ports:
            field = existing.pop(port.key, None)
            if field is None:
                if self._ports_editable:
                    field = FluentLineEdit(port.label)
                    field.editingFinished.connect(
                        lambda key=port.key, edit=field: self.port_label_committed.emit(key, edit.text()))
                else:
                    field = FluentReadoutEdit(port.label)
                # A name longer than the column reads from its head, cut
                # with an ellipsis like the delay column's labels.
                field.set_display_elide(QtCore.Qt.ElideRight)
                endpoint = ElidedLabel(port.endpoint_text or port.key)
                field.setFixedHeight(control_height())
                holder = FluentSettingRow(endpoint, field)
                self._layout.insertWidget(self._layout.count() - 1, holder)
                self._row_holders[port.key] = holder
                self._row_labels[port.key] = endpoint
            field.setText(port.label)
            self._row_holders[port.key].setVisible(bool(port.visible))
            self._rows[port.key] = field
        for key, field in existing.items():
            holder = self._row_holders.pop(key, field)
            self._row_labels.pop(key, None)
            self._layout.removeWidget(holder)
            retire_widget(holder)
        _place_rows_in_order(self._layout, [self._row_holders[port.key] for port in ports])
        self._fit_label_column()

    def set_port_label(self, key: str, label: str) -> None:
        if key in self._rows:
            self._rows[key].setText(label)

    def set_summary(self, total_text: str, total_tooltip: str, period_count: int, visible_text: str) -> None:
        self.total_label.setText(str(total_text))
        self.total_label.setToolTip(str(total_tooltip))
        self.periods_label.setText(str(period_count))
        self.visible_label.setText(str(visible_text))


class ChannelPanel(FluentGroupBox):
    """The pulse's clock, scan and repeat count, then one delay row per port.

    The rows are a form: ONE label column (its widest shown label, a port's
    or "Repeats", and the air after it) and ONE control column, where the
    clock and scan readouts start their text, the repeat box and the Config
    switch start, and every delay field starts.
    """

    feedback_requested = QtCore.pyqtSignal(str)
    delay_committed = QtCore.pyqtSignal(str, object, str)
    binding_committed = QtCore.pyqtSignal(str, object, object, bool, str)
    config_binding_committed = QtCore.pyqtSignal(str, str)
    #: One output in EVERY period at once: on (a digital port high) or off
    #: (a digital port low, an analog port without steps).
    fill_port_requested = QtCore.pyqtSignal(str)
    clear_port_requested = QtCore.pyqtSignal(str)
    run_repeats_committed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__("Delay / Scan", parent)
        self._layout, top, top_layout = _card_column(self)
        self.clock_label = FluentReadoutEdit("")
        self.scan_summary_label = FluentReadoutEdit("")
        # How many times the whole thing plays is a number the PULSE carries,
        # like its clock and its scan table -- not an action, which is all the
        # Control group beside it holds.  Reading it directly under the scan
        # summary says the one sentence they make together: what will be
        # played, and how many times.
        # A count of 0 runs forever, and the box says so where it is typed:
        # it shows ∞.
        self.run_repeats_spin = fluent_count_box()
        self.run_repeats_spin.setSpecialValueText("∞")
        self.run_repeats_spin.setToolTip(
            "Complete Pulse runs per scan point (or per On Pulse without a "
            "scan); ∞ (0) runs indefinitely"
        )
        # A count, not a stretch of the column: as wide as every repeat box
        # (its stepper beside its digits), from the column's left edge, and
        # wider only while it shows or takes a longer count.
        self._size_run_repeats()
        self.run_repeats_spin.lineEdit().textChanged.connect(self._size_run_repeats)
        self.run_repeats_spin.editingFinished.connect(
            lambda: self.run_repeats_committed.emit(
                int(self.run_repeats_spin.value())
            )
        )
        mode_row = QtWidgets.QWidget()
        mode_layout = QtWidgets.QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(row_gap())
        # The row's "Config" label names it; a switch saying "Config" again
        # beside it read "Config: Config".
        self.config_mode = FluentSwitch("")
        self.config_refresh_button = FluentButton("Refresh")
        self.config_refresh_button.setEnabled(False)
        mode_layout.addWidget(self.config_mode)
        mode_layout.addStretch(1)
        mode_layout.addWidget(self.config_refresh_button)
        self._labels = _header_rows(top_layout, (
            ("Clock", self.clock_label), ("Scan", self.scan_summary_label),
            ("Repeats", self.run_repeats_spin), ("Config", mode_row)))
        top_layout.addStretch(1)
        self._layout.addWidget(top)
        self._rows: dict[
            str, tuple[FluentScanLineEdit, FluentComboBox, FluentButton, FluentButton]
        ] = {}
        self._row_labels: dict[str, ElidedLabel] = {}
        self._delay_models: dict[str, DelayRowVM] = {}
        self._layout.addStretch(1)
        self._fit_label_column()

    def _size_run_repeats(self, text: str = "") -> None:
        self.run_repeats_spin.setFixedWidth(count_box_width(self.run_repeats_spin, text))

    def set_run_repeats(self, count: int) -> None:
        """Show the Pulse's run count, the box first made wide enough to
        hold it whole: a count wider than the box is refused by it."""

        self._size_run_repeats(str(int(count)))
        with signals_blocked(self.run_repeats_spin):
            self.run_repeats_spin.setValue(float(count))

    @staticmethod
    def _delay_width() -> int:
        return scaled_px(70, minimum=60)

    def _fit_label_column(self) -> None:
        """Size the label column to the labels shown and the panel to it:
        the delay row's cells, at the one inline gap, fill it exactly."""

        width = label_column_width(
            [label.text() for label in (*self._labels, *self._row_labels.values())]
        )
        for label in (*self._labels, *self._row_labels.values()):
            label.setFixedWidth(width)
        cells = (width, self._delay_width(), time_unit_width(), control_height(), control_height())
        self.setFixedWidth(2 * card_pad() + sum(cells) + row_gap() * (len(cells) - 1))

    def set_delay_rows(
        self,
        rows: tuple[DelayRowVM, ...],
        ports: tuple[PortRowVM, ...],
    ) -> None:
        """One row per delay, beside the port it belongs to.

        The port says what the row can do to it: a digital output can be
        turned on or off in every period, an analog one only cleared of its
        steps -- there is no level to fill it with.
        """

        by_key = {port.key: port for port in ports}
        self._delay_models = {row.port_key: row for row in rows}
        existing = self._rows
        self._rows = {}
        for row in rows:
            port = by_key[row.port_key]
            current = existing.pop(row.port_key, None)
            if current is None:
                edit = FluentScanLineEdit(
                    row.value.text,
                    tooltip="Edit value source",
                )
                edit.setFixedWidth(self._delay_width())
                combo = FluentComboBox()
                combo.setFixedWidth(time_unit_width())
                fill = _port_toggle_button(True, "Turn this output on in every period")
                clear = _port_toggle_button(False, "Turn this output off in every period")
                edit.binding_committed.connect(lambda scan, source, key=row.port_key: self.binding_committed.emit("delay", None, key, scan, source))
                edit.config_key_committed.connect(
                    lambda key, field=edit: self.config_binding_committed.emit(field.property("pulse_field_id"), key)
                )
                edit.editingFinished.connect(lambda key=row.port_key, field=edit, units=combo: self._emit_delay(key, field, units))
                edit.valueNormalized.connect(lambda key=row.port_key, field=edit, units=combo: self._emit_delay(key, field, units))
                combo.currentTextChanged.connect(lambda _text, key=row.port_key, field=edit, units=combo: self._emit_delay(key, field, units))
                fill.clicked.connect(lambda _checked=False, key=row.port_key: self.fill_port_requested.emit(key))
                clear.clicked.connect(lambda _checked=False, key=row.port_key: self.clear_port_requested.emit(key))
                label = ElidedLabel(port.label)
                holder = FluentSettingRow(label, edit, after=((combo, 0), (fill, 0), (clear, 0)))
                self._layout.insertWidget(self._layout.count() - 1, holder)
                self._row_labels[row.port_key] = label
                current = (edit, combo, fill, clear)
            edit, combo, fill, clear = current
            digital = port.kind == "digital"
            # Inapplicable actions keep their column in every port row.
            fill.setEnabled(digital)
            # On an analog row there is no "off" to turn it to: the button
            # takes the output's steps away, so it is the row remover every
            # row has -- the ghost x, red under the pointer, one click.
            if digital and clear.text():
                clear.setText("")
                _show_port_ring(clear, False)
                clear.set_role("standard")
            elif not digital and not clear.text():
                clear.setIcon(QtGui.QIcon())
                clear.setText("×")
                clear.set_role("destructive")
            clear.setToolTip(
                "Turn this output off in every period"
                if digital
                else "Remove this output's steps from every period"
            )
            self._row_labels[row.port_key].setText(port.label)
            _apply_field(edit, row.value)
            with signals_blocked(combo):
                combo.clear()
                combo.addItems(list(row.units) or [row.unit])
                combo.setCurrentText(row.unit)
                combo.setEnabled(not row.value.config_active and row.value.editable)
            self._rows[row.port_key] = current
        for key, (edit, _combo, _fill, _clear) in existing.items():
            holder = edit.parentWidget()
            if holder is not None:
                self._layout.removeWidget(holder)
                retire_widget(holder)
            self._row_labels.pop(key, None)
        _place_rows_in_order(
            self._layout, [self._rows[row.port_key][0].parentWidget() for row in rows]
        )
        self._fit_label_column()

    def set_port_label(self, key: str, label: str) -> None:
        widget = self._row_labels.get(str(key))
        if widget is not None:
            widget.setText(str(label))
            self._fit_label_column()

    def _emit_delay(self, key: str, field: FluentScanLineEdit, units: FluentComboBox) -> None:
        if field.isReadOnly():
            return
        if field.property("numericError"):
            self.feedback_requested.emit(str(field.property("numericError")))
            return
        try:
            value = float(field.text())
        except ValueError:
            return
        previous = self._delay_models.get(key)
        if previous is not None and units.currentText() == previous.unit:
            try:
                if value == float(previous.value.text):
                    return
            except ValueError:
                pass
        self.delay_committed.emit(str(key), value, units.currentText())

    def set_clock(self, text: str) -> None:
        self.clock_label.setText(str(text))

    def set_scan_summary(self, text: str) -> None:
        self.scan_summary_label.setText(str(text))


def _reading_ink(stroke: str) -> str:
    """A bracket's stroke colour as words: mixed toward TEXT, a twentieth
    at a time, until it reads 4.5:1 on the ACCENT_TINT body of a selected
    post, the paler of a post's two faces.  The stroke itself edges the
    post and draws the preview's loop; as a title it is too pale (3.1:1)."""

    def luminance(color: QtGui.QColor) -> float:
        channels = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
                    for value in color.getRgbF()[:3]]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    face = luminance(QtGui.QColor(ACCENT_TINT)) + 0.05
    start, end = QtGui.QColor(stroke), QtGui.QColor(TEXT)
    for step in range(21):
        ink = QtGui.QColor(*(round(a + (b - a) * step / 20) for a, b in zip(
            start.getRgb()[:3], end.getRgb()[:3])))
        if face / (luminance(ink) + 0.05) >= 4.5:
            break
    return ink.name()


class BracketPost(FluentGroupBox):
    """One draggable post framing a bracketed run of periods.

    The shape is the point: a column built to the same height and header
    block as a period card, titled "Bracket N" in the pill where a card says
    "Period k/N", with its count on the cards' first control line.  The
    bracket then reads as a frame drawn around cards rather than a widget
    wedged between them, and the number and the ink say WHICH frame: both
    posts of a bracket wear its colour on the title and the edge, and the
    preview draws that bracket's loop in the same colour.  A period card is
    white with a black title, a spacer grey and dashed with a grey title, a
    bracket post white with a coloured title and edge: three kinds, three
    looks.

    It had become a titled box with a glyph in it, at whatever height the
    layout happened to give -- the thing marking a span lined up with nothing
    in the span it marked.

    As wide as every repeat box (:func:`count_box_width`) plus its source
    badge: a longer authored count grows the post without rounding.
    """

    count_committed = QtCore.pyqtSignal(object)
    binding_committed = QtCore.pyqtSignal(bool, str)
    config_binding_committed = QtCore.pyqtSignal(str, str)
    feedback_requested = QtCore.pyqtSignal(str)

    def __init__(
        self, bracket_id: str, kind: str, *, count: int = 2, minimum: int = 1,
        ordinal: int = 1, color: str = "", count_field: FieldVM | None = None, parent=None,
    ) -> None:
        super().__init__("", parent)
        self.bracket_id = str(bracket_id)
        self.kind = str(kind)
        #: The item key this post answers to in the strip's order.
        self.key = bracket_post_key(self.bracket_id, self.kind)
        self.ordinal = 1
        self.color = ""
        self.set_bracket_style(ordinal, color)
        self.count_spin = fluent_count_box(minimum=minimum)
        self.count_edit = FluentScanLineEdit("", tooltip="Edit repeat value source")
        # The count starts where every number in the strip starts: at a
        # field's text inset, not centred in its box.
        self.count_spin.setLineEdit(self.count_edit)
        self.count_edit.textChanged.connect(self._size_count_editor)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        column, top, top_layout = _card_column(self)
        # The post's name is its title pill, where a card says "Period 3/6".
        # The header line under it names the count on the end post and stays
        # empty on the start post, so both posts stay level with the cards.
        top_layout.addWidget(card_caption("Repeats" if kind == "end" else ""))
        # The count belongs to the end post: a span is closed by saying how
        # many times.  The start post keeps an empty line of the same height so
        # the two posts stay level with each other and with the cards.
        self.count_spin.setFixedHeight(control_height())
        self.set_count_field(count_field or FieldVM(
            str(count), can_scan=False, validator_kind=VALIDATOR_INT,
            validator_lo=minimum, validator_hi=(1 << 32) - 1,
        ))
        self.count_spin.valueChanged.connect(self._commit_count)
        self.count_edit.binding_committed.connect(self.binding_committed)
        self.count_edit.config_key_committed.connect(
            lambda key: self.config_binding_committed.emit(self.count_edit.property("pulse_field_id"), key)
        )
        if kind == "start":
            self.count_spin.hide()
            spacer = QtWidgets.QWidget()
            spacer.setStyleSheet("background: transparent;")
            spacer.setFixedHeight(control_height())
            top_layout.addWidget(spacer)
        else:
            top_layout.addWidget(self.count_spin)
        top_layout.addStretch(1)
        column.addWidget(top)
        column.addStretch(1)

    def set_count_field(self, field: FieldVM) -> None:
        self._count_field = field
        self._size_count_editor(field.text)
        with signals_blocked(self.count_spin, self.count_edit):
            _apply_field(self.count_edit, field)
            self.count_spin.setReadOnly(not field.editable or field.config_active)
            self.count_spin.setEnabled(field.editable or field.config_active)
            self.count_spin.setRange(float(field.validator_lo), float(field.validator_hi))
            self.count_spin.setValue(float(int(field.text)))

    def _size_count_editor(self, text: str) -> None:
        width = count_box_width(self.count_spin, text)
        self.count_spin.setFixedWidth(width)
        self.setFixedWidth(width + 2 * card_pad())

    def _commit_count(self) -> None:
        if self.count_spin.isReadOnly():
            return
        error = self.count_spin.property("numericError") or self.count_edit.property("numericError")
        if error:
            self.feedback_requested.emit(str(error))
            return
        value = int(self.count_spin.value())
        if value != int(self._count_field.text):
            self.count_committed.emit(value)

    def set_bracket_style(self, ordinal: int, color: str) -> None:
        """Say which bracket this post frames.

        Its number goes in the title pill and its colour on the edge; both
        posts of a bracket wear the same, and the preview draws that
        bracket's loop in it.  The title wears that hue at reading strength
        (:func:`_reading_ink`).  Renumbered in place when a bracket is added
        or removed before it, the way a card's "Period k/N" is.
        """

        self.ordinal = int(ordinal)
        self.color = str(color or "")
        self.setTitle(f"Bracket {self.ordinal}")
        self.set_title_color(_reading_ink(self.color) if self.color else TEXT)
        self.set_surface(edge=self.color or None)
        self.setToolTip(f"Bracket {self.ordinal}: drag to move this boundary")


def bracket_spans_of(order: list[tuple[str, str]]) -> dict[str, tuple[int, int]] | None:
    """Each bracket's (start gap, end gap) in a visual order, or None when it is not well formed.

    Not well formed: a post before its own partner, or two brackets that
    overlap without one lying inside the other.  An empty bracket lies inside
    another only when its gap is strictly inside -- the model's rule, so what
    the strip refuses while dragging is what the document would refuse.
    """

    positions: dict[str, dict[str, int]] = {}
    gaps: dict[str, dict[str, int]] = {}
    periods_before = 0
    for index, (kind, key) in enumerate(order):
        if kind == "period":
            periods_before += 1
            continue
        bracket_id, _colon, side = key.rpartition(":")
        positions.setdefault(bracket_id, {})[side] = index
        gaps.setdefault(bracket_id, {})[side] = periods_before
    spans: dict[str, tuple[int, int]] = {}
    for bracket_id, sides in positions.items():
        if set(sides) != {"start", "end"} or sides["end"] < sides["start"]:
            return None
        spans[bracket_id] = (gaps[bracket_id]["start"], gaps[bracket_id]["end"])

    def contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
        if not (outer[0] <= inner[0] and inner[1] <= outer[1]):
            return False
        return inner[0] < inner[1] or outer[0] < inner[0] < outer[1]

    listed = tuple(spans.values())
    for index, first in enumerate(listed):
        for second in listed[index + 1:]:
            disjoint = first[1] <= second[0] or second[1] <= first[0]
            if not (disjoint or contains(first, second) or contains(second, first)):
                return None
    return spans


class ComponentCard(FluentGroupBox):
    """A compact group header, not another implementation of period editing.

    Built on the cards' own grid: its header block has the Period card's
    four lines (Duration, the total where the duration field is, a summary
    where the unit is, the name), and its commands start on the first port
    row, one per row.  The total reads like a field's value -- text at the
    field's inset, its badge where a field seats its badge -- with no box.
    A Subpulse wears its own blue (DERIVED_INK title, SELECTION_EDGE edge):
    ACCENT is for small indicators.
    """

    action_requested = QtCore.pyqtSignal(str, object)

    def __init__(self, component: ComponentVM, parent=None, *, mode: str = "closed",
                 card_width: int | None = None) -> None:
        super().__init__("Subpulse" if mode == "closed" else "Start" if mode == "start" else "End", parent,
                         title_color=DERIVED_INK)
        self.component_id = component.component_id
        self.mode = mode
        self.key = ("component" if mode == "closed" else f"component-{mode}", component.component_id)
        self._component = component
        self.set_surface(edge=SELECTION_EDGE)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.expand_button = FluentButton("Expand" if mode == "closed" else "Collapse")
        self.expand_button.clicked.connect(lambda: self.action_requested.emit("expand", self.component_id))
        if mode != "closed":
            width = spacer_card_width()
        else:
            width = card_width if card_width is not None else period_card_width()
        self.setFixedWidth(width)
        column, top, top_layout = _card_column(self)
        self.duration_label = card_caption("Duration")
        top_layout.addWidget(self.duration_label)
        self.total_label = ElidedLabel("")
        self.source_button = _BindingButton()
        # No field frame here: the badge column stands in the row itself,
        # one badge gap inside where a field's inner edge would be.
        self.source_button.stand_alone(control_height())
        self.source_button.setAccessibleName("Subpulse source")
        self.source_button.clicked.connect(self._toggle_source)
        top_layout.addWidget(self._value_line(self.total_label, self.source_button))
        # The unit's line says what the Subpulse holds (closed); expanded,
        # its members say it themselves and the line stays empty.
        self.summary_label = ElidedLabel("")
        set_status_ink(self.summary_label, "idle")
        top_layout.addWidget(self._value_line(self.summary_label))
        self.name_edit = FluentLineEdit(component.name)
        # A field like every other: its text starts at the field's inset.
        # The card shows no Name label: the placeholder carries it.
        self.name_edit.setPlaceholderText("Name")
        self.name_edit.editingFinished.connect(self._commit_name)
        # Expanded, the block's two ends carry its name alike: read, at the
        # field's text inset -- the closed card and the Subpulse tab rename
        # it.  A field on one end and bare text on the other would read as
        # two kinds of thing.
        self.name_label = ElidedLabel(component.name)
        self.name_edit.setFixedHeight(control_height())
        name_line = self._value_line(self.name_label)
        top_layout.addWidget(self.name_edit if mode == "closed" else name_line)
        if mode == "closed":
            name_line.setParent(self)
            name_line.hide()
        else:
            self.name_edit.setParent(self)
            self.name_edit.hide()
        top_layout.addStretch(1)
        column.addWidget(top)
        self.edit_button = FluentButton("Edit")
        self.edit_button.setToolTip("Edit this Subpulse")
        self.edit_button.clicked.connect(lambda: self.action_requested.emit("edit", self.component_id))
        # Ungroup dissolves the Subpulse: red like every command that takes
        # something away, and one click.
        self.ungroup_button = FluentButton("Ungroup", role="destructive")
        self.ungroup_button.setToolTip(
            "Dissolve this Subpulse: its embedded (Default) periods become "
            "ordinary Pulse items; a Config source link is dropped"
        )
        self.ungroup_button.clicked.connect(lambda: self.action_requested.emit("ungroup", self.component_id))
        # One command per port row, from the first: as tall as a row and
        # the rows' spacing apart, so none straddles two rows of the cards
        # beside it.
        for button in (self.expand_button, self.edit_button, self.ungroup_button):
            button.setFixedHeight(control_height())
            column.addWidget(button)
        column.addStretch(1)
        self._source_popup = None
        self.set_component(component)

    @staticmethod
    def _value_line(label: ElidedLabel, badge: _BindingButton | None = None) -> QtWidgets.QWidget:
        """A read-only value in a field's place: the text at the field's
        text inset, a badge where a field seats its badge (the column one
        badge gap before the inner edge, the button the column and a pixel
        each side)."""

        line = QtWidgets.QWidget()
        line.setStyleSheet("background: transparent;")
        line.setFixedHeight(control_height())
        layout = QtWidgets.QHBoxLayout(line)
        right = 0 if badge is None else FIELD_BORDER_PX + _BindingButton.gap() - scaled_px(1)
        layout.setContentsMargins(field_text_inset(), 0, right, 0)
        layout.setSpacing(_BindingButton.gap())
        layout.addWidget(label, 1)
        if badge is not None:
            layout.addWidget(badge)
        return line

    def _toggle_source(self) -> None:
        if self._source_popup is None:
            self._source_popup = FluentPopup(self)
            self._source_anchor = FluentSettingsPopupAnchor(self._source_popup, self.source_button)
            layout = QtWidgets.QHBoxLayout(self._source_popup)
            layout.setContentsMargins(card_pad(), card_pad(), card_pad(), card_pad())
            layout.setSpacing(row_gap())
            # The same choice, in the same control and at the same x, as a
            # value slot's Source: the two popups share one label column.
            label = FluentLabel("Source")
            label.setFixedWidth(popup_label_width())
            layout.addWidget(label)
            self.source_switch = FluentSegmentedButtons((("default", "Default"), ("config", "Config")))
            layout.addWidget(self.source_switch)
            self.source_switch.activated.connect(self._commit_source)
        self._source_anchor.toggle(self._source_popup, prepare=self._project_source,
                                   present=self._place_source)

    def _place_source(self) -> None:
        show_fluent_popup_for_anchor(
            self._source_popup, self.source_button, self._source_popup,
            minimum_width=1, minimum_height=1,
            maximum_height=self._source_popup.sizeHint().height(),
        )

    def _project_source(self) -> None:
        if self._source_popup is not None:
            self.source_switch.set_current(self._component.source)

    def _commit_source(self, source: str) -> None:
        if source != self._component.source:
            self.action_requested.emit("source", (self.component_id, source, self._component.config_path))
        self._project_source()

    def _commit_name(self) -> None:
        name = self.name_edit.text().strip()
        if name != self._component.name:
            self.action_requested.emit("rename", (self.component_id, name))
            # The synchronous authoring owner either projected the accepted
            # name or retained this VM. Never leave a refused draft on the card.
            self.name_edit.setText(self._component.name)

    def _show_command_faces(self) -> None:
        """The commands' face for the body under them.  On a tinted body --
        the green of a Config Subpulse, the selection wash -- the commands
        wear a white face, no edge (:meth:`FluentButton.set_on_tint`).
        Ungroup keeps its red face on any body, and an expanded Subpulse's
        Collapse its active face."""

        tinted = self._component.config_active or self.is_selected()
        for button in (self.expand_button, self.edit_button):
            button.set_on_tint(tinted)

    def set_selected(self, selected: bool) -> None:
        super().set_selected(selected)
        self._show_command_faces()

    def set_component(self, component: ComponentVM) -> None:
        self._component = component
        self.set_surface(fill=CONFIG_GREEN_TINT if component.config_active else None,
                         edge=CONFIG_GREEN if component.config_active else SELECTION_EDGE)
        self.set_title_color(CONFIG_GREEN_DARK if component.config_active else DERIVED_INK)
        self._show_command_faces()
        self.source_button.source = component.source
        self.source_button.setToolTip(component.error or f"Subpulse source: {component.source.title()}")
        self.source_button.update()
        self.total_label.setText(component.total_text)
        controls = (self.expand_button, self.edit_button, self.ungroup_button)
        if self.mode != "closed":
            # As wide as its widest command or its whole total, whichever
            # needs more.
            line = self.total_label.parentWidget().layout()
            margins = line.contentsMargins()
            total = (margins.left() + self.total_label.fontMetrics().horizontalAdvance(component.total_text)
                     + scaled_px(2) + line.spacing() + self.source_button.width() + margins.right())
            self.setFixedWidth(max(spacer_card_width(), 2 * card_pad() + max(
                total, *(button.sizeHint().width() for button in controls))))
        width = period_control_width(self.width())
        for button in controls:
            button.setFixedWidth(width)
        self.name_edit.setFixedWidth(width)
        self.name_edit.setText(component.name)
        self.name_edit.setToolTip(component.name)
        self.name_label.setText(component.name)
        self.summary_label.setText(component.summary_text if self.mode == "closed" else "")
        self.expand_button.setText("Expand" if self.mode == "closed" else "Collapse")
        # A toggle: held down while it is expanded, its label the next press.
        self.expand_button.set_active(self.mode != "closed")
        self._project_source()


class PulseDragContainer(FluentPageBody):
    """Horizontal card strip whose drag releases are proposal-only."""

    reorder_items_requested = QtCore.pyqtSignal(object)
    remove_items_requested = QtCore.pyqtSignal(object)
    #: (bracket id, count) from that bracket's end post.
    bracket_count_committed = QtCore.pyqtSignal(str, object)
    bracket_binding_committed = QtCore.pyqtSignal(str, bool, str)
    config_binding_committed = QtCore.pyqtSignal(str, str)
    feedback_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.layout_main = QtWidgets.QHBoxLayout(self)
        self.layout_main.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        # Cards stand one card gap apart, and one gap in from the strip's
        # ends (the first gap is the one after the side panels), level with
        # the side panels at the top.
        gap = row_gap()
        self.layout_main.setContentsMargins(gap, 0, gap, 0)
        self.layout_main.setSpacing(gap)
        self.layout_main.setAlignment(QtCore.Qt.AlignLeft)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._cards: tuple[PeriodCard, ...] = ()
        self._components: tuple[ComponentCard, ...] = ()
        self._item_blocks: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
        self._posts: tuple["BracketPost", ...] = ()
        self._pressed: tuple[tuple[str, str], QtCore.QPoint, bool] | None = None
        self._dragging = False
        # The insertion caret: a rounded accent bar the height of the cards,
        # centred in the gap it marks.
        self._indicator = QtWidgets.QFrame(self)
        self._indicator_width = 0
        self._indicator.hide()
        self._selection: tuple[tuple[str, str], ...] | int | None = None
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def set_items(
        self,
        cards: tuple[PeriodCard, ...],
        brackets: tuple[BracketVM, ...],
        *,
        order: tuple[tuple[str, str], ...],
        minimum_bracket: int = 1,
        components: tuple[ComponentCard, ...] = (),
        item_blocks: dict[tuple[str, str], tuple[tuple[str, str], ...]] | None = None,
    ) -> None:
        previous = self.items()
        self._cards = tuple(cards)
        self._components = tuple(components)
        self._item_blocks = item_blocks or {}
        widgets = {("period", card.period_id): card for card in cards}
        widgets.update({card.key: card for card in components})
        # One start and one end post per bracket, kept across re-projections
        # by their key so a post being dragged or edited is the same widget.
        existing = {post.key: post for post in self._posts}
        posts: list[BracketPost] = []
        for bracket in brackets:
            for side in ("start", "end"):
                key = bracket_post_key(bracket.bracket_id, side)
                post = existing.get(key)
                if post is None:
                    post = BracketPost(
                        bracket.bracket_id, side, count=bracket.count, minimum=minimum_bracket,
                        ordinal=bracket.ordinal, color=bracket.color,
                        count_field=bracket.count_field,
                    )
                    if side == "end":
                        post.count_committed.connect(
                            lambda count, bracket_id=bracket.bracket_id:
                                self.bracket_count_committed.emit(bracket_id, int(count))
                        )
                        post.binding_committed.connect(
                            lambda scan, source, bid=bracket.bracket_id:
                                self.bracket_binding_committed.emit(bid, scan, source)
                        )
                        post.feedback_requested.connect(self.feedback_requested)
                        post.config_binding_committed.connect(self.config_binding_committed)
                else:
                    # A kept post may have become another number: a bracket
                    # added or removed before it renumbers the rest.
                    post.set_bracket_style(bracket.ordinal, bracket.color)
                posts.append(post)
                widgets[("bracket", key)] = post
                count_field = bracket.count_field or FieldVM(
                    str(bracket.count), can_scan=False, validator_kind=VALIDATOR_INT,
                    validator_lo=minimum_bracket, validator_hi=(1 << 32) - 1,
                )
                if bracket.readonly:
                    count_field = replace(count_field, editable=False, binding_editable=False)
                post.set_count_field(count_field)
                post.setProperty("readonly_content", bracket.readonly)
        self._posts = tuple(posts)
        desired = tuple(widgets[key] for key in order)
        for widget in previous:
            if widget not in desired:
                self.layout_main.removeWidget(widget)
                widget.removeEventFilter(self)
                retire_widget(widget)
        for index, widget in enumerate(desired):
            if self.layout_main.indexOf(widget) != index:
                self.layout_main.removeWidget(widget)
                self.layout_main.insertWidget(index, widget)
            # A retained card may have new channel rows after Hide/Show.
            self.watch_item_chrome(widget)
        if desired == previous:
            return
        self._pressed = None
        gap = self.selected_gap
        self.show_selection(
            tuple(key for key in self.selected_items() if key in order),
            gap=gap if gap is not None and gap <= len(order) else None,
        )

    def event(self, event):
        handled = super().event(event)
        if event.type() == QtCore.QEvent.LayoutRequest and self.selected_gap is not None:
            self._show_indicator_at_gap(self.selected_gap)
        return handled

    #: Widgets that need their own clicks: typing in them, ticking them and
    #: opening them ARE the click.  Everything else on a card is chrome, and a
    #: click on chrome belongs to the card.
    _INTERACTIVE = (
        QtWidgets.QAbstractButton,
        QtWidgets.QLineEdit,
        QtWidgets.QAbstractSpinBox,
        QtWidgets.QComboBox,
    )

    def watch_item_chrome(self, item: QtWidgets.QWidget) -> None:
        """Let a click anywhere that is not an editable control select the card.

        A card is almost entirely covered by its own children -- checkboxes,
        value boxes, a name field -- so a filter installed on the card alone
        would see a press only in the few pixels of padding around them.

        Only what is drawn IN the card: a card's popups (a slot's binding
        popup, the Subpulse source popup) are its children but separate
        windows, whose own outside-press handling must see their presses.
        """

        item.installEventFilter(self)
        for child in item.findChildren(QtWidgets.QWidget):
            if child.window() is item.window():
                child.installEventFilter(self)

    @staticmethod
    def _item_key(item: QtWidgets.QWidget) -> tuple[str, str]:
        if isinstance(item, ComponentCard):
            return item.key
        return ("period", item.period_id) if isinstance(item, PeriodCard) else ("bracket", item.key)

    def flat_order(self, order=None) -> tuple[tuple[str, str], ...]:
        keys = tuple(self._item_key(item) for item in self.items()) if order is None else order
        return tuple(item for key in keys for item in (
            () if key[0] in ("component-start", "component-end")
            else self._item_blocks.get(key, (key,))
        ))

    def items(self) -> tuple[QtWidgets.QWidget, ...]:
        return tuple(self.layout_main.itemAt(index).widget() for index in range(self.layout_main.count()))

    def _item_of(self, widget: object) -> QtWidgets.QWidget | None:
        while isinstance(widget, QtWidgets.QWidget):
            if isinstance(widget, (PeriodCard, BracketPost, ComponentCard)):
                return widget
            widget = widget.parentWidget()
        return None

    # ------------------------------------------------------- where edits land
    def selected_items(self) -> tuple[tuple[str, str], ...]:
        """Exactly the selected visible items, in timeline order."""
        return self._selection if isinstance(self._selection, tuple) else ()

    @property
    def selected_gap(self) -> int | None:
        return self._selection if isinstance(self._selection, int) else None

    def selection_payload(self) -> tuple[tuple[str, str], ...]:
        """Keep component identity instead of silently promoting its contents."""
        selected = self.selected_items()
        components = {key for kind, key in selected if kind.startswith("component")}
        covered = {item for key in components for item in self._item_blocks.get(("component", key), ())}
        result = []
        for kind, key in selected:
            item = ("component", key) if kind.startswith("component") else (kind, key)
            if item not in covered and item not in result:
                result.append(item)
        return tuple(result)

    def show_selection(self, items=(), *, gap: int | None = None) -> None:
        if gap is not None and items:
            raise ValueError("select timeline items or a gap, not both")
        if gap is not None:
            self._selection = int(gap)
        else:
            chosen = set(items)
            self._selection = tuple(self._item_key(item) for item in self.items() if self._item_key(item) in chosen)
        self._paint_selection(self.selected_items())

    def _paint_selection(self, selected) -> None:
        chosen = set(selected)
        for kind, component_id in tuple(chosen):
            if kind in ("component-start", "component-end"):
                chosen.update(self._item_blocks[("component", component_id)])
                chosen.update((("component-start", component_id), ("component-end", component_id)))
        for widget in self.items():
            key = self._item_key(widget)
            selected_here = key in chosen
            if key[0] in ("component-start", "component-end"):
                block = self._item_blocks[("component", key[1])]
                selected_here = selected_here or bool(block) and set(block).issubset(chosen)
            widget.set_selected(selected_here)
        if self.selected_gap is None:
            self._indicator.hide()
        else:
            self._show_indicator_at_gap(self.selected_gap)

    def _clicked(self, key: tuple[str, str], *, append: bool) -> None:
        current = self.selected_items()
        if append:
            self.show_selection(tuple(item for item in current if item != key) if key in current else (*current, key))
        else:
            self.show_selection(()) if current == (key,) else self.show_selection((key,))

    def clear_selection(self) -> None:
        """Forget it.  Anything that changes the period LIST invalidates it."""

        self.show_selection()

    def _gap_at(self, x: int) -> int:
        """One gap per visible adjacency, including BOTH sides of each post."""
        items = self.items()
        for index, item in enumerate(items):
            if x < item.geometry().center().x():
                return index
        return len(items)

    def _show_indicator_at_gap(self, position: int) -> None:
        """Centre the caret in the gap and give it the cards' own height."""

        items = self.items()
        if not items:
            self._indicator.hide()
            return
        clamped = max(0, min(int(position), len(items)))
        margins = self.layout_main.contentsMargins()
        # The gap as the pixels between two painted edges, [left, right).
        if clamped == 0:
            right = items[0].geometry().left()
            left = right - margins.left()
        elif clamped == len(items):
            left = items[-1].geometry().right() + 1
            right = left + margins.right()
        else:
            left = items[clamped - 1].geometry().right() + 1
            right = items[clamped].geometry().left()
        # The caret keeps one pixel clear of each neighbour and takes the
        # rest, so a four-pixel gap holds a two-pixel caret dead centre.  A
        # caret of fixed width sat off-centre whenever the parities differed.
        clearance = scaled_px(1, minimum=1)
        width = max(scaled_px(2, minimum=2), (right - left) - 2 * clearance)
        if width != self._indicator_width:
            self._indicator_width = width
            self._indicator.setStyleSheet(
                f"background: {ACCENT}; border-radius: {width // 2}px;"
            )
        top = min(item.geometry().top() for item in items)
        bottom = max(item.geometry().bottom() for item in items)
        self._indicator.setGeometry(
            left + ((right - left) - width) // 2, top, width, bottom - top + 1
        )
        self._indicator.raise_()
        self._indicator.show()

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt name
        """A click on empty strip selects the gap it fell in, so Add period
        can insert there."""

        if event.button() == QtCore.Qt.LeftButton and not event.modifiers() & QtCore.Qt.ShiftModifier and not any(
            widget.geometry().contains(event.pos())
            for widget in self._cards + self._posts + self._components
        ):
            # A release inside something this row HOLDS belongs to that thing,
            # even when it let the release through; only the strip between
            # them is a gap.
            gap = self._gap_at(event.pos().x())
            self.show_selection(gap=None if self.selected_gap == gap else gap)
        super().mouseReleaseEvent(event)

    ITEM_MIME = "application/x-zlc-pulse-item"

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            selected = self.selection_payload()
            if selected:
                self.remove_items_requested.emit(selected)
                event.accept()
                return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event):  # noqa: N802 - Qt name
        if event.mimeData().hasFormat(self.ITEM_MIME):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def _proposal_at(self, data: QtCore.QMimeData, gap: int) -> object | None:
        """Move the exact selection as one proposal; the document still owns it."""
        if not data.hasFormat(self.ITEM_MIME):
            return None
        try:
            key = tuple(json.loads(bytes(data.data(self.ITEM_MIME))))
        except (TypeError, ValueError):
            return None
        order = [self._item_key(item) for item in self.items()]
        if key not in order or not 0 <= gap <= len(order):
            return None
        moving = self._drag_items(key)
        moved_components = {identifier for kind, identifier in moving if kind.startswith("component")}
        for item in self.items():
            item_key = self._item_key(item)
            if item_key in moving and item.property("readonly_content") and not any(
                item_key in self._item_blocks.get(("component", identifier), ())
                for identifier in moved_components
            ):
                return None
        destination = gap - sum(item in moving for item in order[:gap])
        remaining = [item for item in order if item not in moving]
        proposed = remaining[:destination] + list(moving) + remaining[destination:]
        if proposed == order:
            return None
        # Empty brackets remain editable; only crossing a post's own partner or
        # making two brackets overlap without one inside the other is refused
        # while dragging.  Execution validates the content.
        flat_order = self.flat_order(proposed)
        if bracket_spans_of(flat_order) is None:
            return None
        return flat_order

    def _drag_items(self, grabbed) -> tuple[tuple[str, str], ...]:
        chosen = set(self.selected_items()) if grabbed in self.selected_items() else {grabbed}
        for (_kind, component_id), block in self._item_blocks.items():
            caps = (("component-start", component_id), ("component-end", component_id))
            if any(cap in chosen for cap in caps) or block and set(block).issubset(chosen):
                chosen.update((*caps, *block))
        return tuple(self._item_key(item) for item in self.items() if self._item_key(item) in chosen)

    def dragMoveEvent(self, event):  # noqa: N802 - Qt name
        """The marker and drop consume exactly the same ordering proposal."""
        if not event.mimeData().hasFormat(self.ITEM_MIME):
            return super().dragMoveEvent(event)
        gap = self._gap_at(event.pos().x())
        if self._proposal_at(event.mimeData(), gap) is None:
            event.ignore()
            self._indicator.hide()
            return
        event.acceptProposedAction()
        self._show_indicator_at_gap(gap)

    def dragLeaveEvent(self, event):  # noqa: N802 - Qt name
        self._restore_selection()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):  # noqa: N802 - Qt name
        """Commit exactly what the marker was offering when the button came up.

        Decide, then accept: accepting first and working out afterwards
        whether there was anything to do is how a drop came to be accepted
        and then dropped.
        """

        data = event.mimeData()
        if not data.hasFormat(self.ITEM_MIME):
            return super().dropEvent(event)
        proposal = self._proposal_at(data, self._gap_at(event.pos().x()))
        self._restore_selection()
        if proposal is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.reorder_items_requested.emit(proposal)

    def _restore_selection(self) -> None:
        """Put back whatever was picked before the drag moved the marker."""
        self._paint_selection(self.selected_items())

    def _begin_drag(self, key: tuple[str, str]) -> None:
        item = next((item for item in self.items() if self._item_key(item) == key), None)
        if item is None:
            return
        data = QtCore.QMimeData()
        data.setData(
            self.ITEM_MIME, QtCore.QByteArray(json.dumps(key).encode("utf-8")),
        )
        drag = QtGui.QDrag(self)
        drag.setMimeData(data)
        drag.setPixmap(item.grab())
        drag.setHotSpot(QtCore.QPoint(item.width() // 2, 0))
        drag.exec_(QtCore.Qt.MoveAction)

    def eventFilter(self, obj, event):  # noqa: N802
        item = self._item_of(obj)
        # A popup opened from a card (and everything in it) is another
        # window: its presses are its own, never a card selection or drag.
        if item is None or obj.window() is not item.window():
            return super().eventFilter(obj, event)
        key = self._item_key(item)
        mouse_types = (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseMove, QtCore.QEvent.MouseButtonRelease)
        if event.type() not in mouse_types:
            return super().eventFilter(obj, event)
        interactive = obj
        while interactive is not item and not isinstance(interactive, self._INTERACTIVE):
            interactive = interactive.parentWidget()
        shift = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
        # Shift selects the containing item even over an editor or a checkbox.
        # Normal editing is left completely to that control.
        if interactive is not item and not shift and self._pressed is None:
            return super().eventFilter(obj, event)
        if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
            self._pressed = (key, event.globalPos(), shift)
            self._dragging = False
            self.setFocus(QtCore.Qt.MouseFocusReason)
            if not shift and key not in self.selected_items():
                self._paint_selection((key,))
                self._indicator.hide()
            return True
        elif event.type() == QtCore.QEvent.MouseMove and self._pressed is not None:
            if not event.buttons() & QtCore.Qt.LeftButton:
                self._pressed = None
                self._restore_selection()
            elif not self._dragging and (
                event.globalPos() - self._pressed[1]
            ).manhattanLength() >= QtWidgets.QApplication.startDragDistance():
                moving, _position, append = self._pressed
                self._pressed = None
                if moving not in self.selected_items():
                    self.show_selection((*self.selected_items(), moving) if append else (moving,))
                self.show_selection(self._drag_items(moving))
                self._dragging = True
                try:
                    self._begin_drag(moving)
                finally:
                    self._dragging = False
                    self._restore_selection()
                return True
            return True
        elif event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
            pressed = self._pressed
            self._pressed = None
            if not self._dragging and pressed is not None and pressed[0] == key:
                self._clicked(key, append=pressed[2])
            else:
                self._restore_selection()
            return pressed is not None or shift
        return super().eventFilter(obj, event)


class PulseScheduleView(QtWidgets.QWidget):
    component_action_requested = QtCore.pyqtSignal(str, object)
    document_name_committed = QtCore.pyqtSignal(str)
    port_label_committed = QtCore.pyqtSignal(str, str)
    period_name_committed = QtCore.pyqtSignal(str, str)
    duration_committed = QtCore.pyqtSignal(str, object, str)
    digital_committed = QtCore.pyqtSignal(str, str, bool)
    analog_committed = QtCore.pyqtSignal(str, str, str, object)
    delay_committed = QtCore.pyqtSignal(str, object, str)
    binding_committed = QtCore.pyqtSignal(str, object, object, bool, str)
    config_binding_committed = QtCore.pyqtSignal(str, str)
    insert_period_requested = QtCore.pyqtSignal(object)
    insert_spacer_requested = QtCore.pyqtSignal(object)
    reorder_items_requested = QtCore.pyqtSignal(object)
    remove_items_requested = QtCore.pyqtSignal(object)
    #: (bracket id, start period id, end period id, count): move or recount one bracket.
    bracket_committed = QtCore.pyqtSignal(str, object, object, object)
    #: (start period id, end period id, count): a new bracket; the presenter names it.
    bracket_add_requested = QtCore.pyqtSignal(object, object, object)
    run_repeats_committed = QtCore.pyqtSignal(int)
    visible_ports_committed = QtCore.pyqtSignal(object)
    fill_port_requested = QtCore.pyqtSignal(str)
    clear_port_requested = QtCore.pyqtSignal(str)
    run_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    sync_requested = QtCore.pyqtSignal()
    save_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()
    connection_requested = QtCore.pyqtSignal(str, str)
    feedback_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, *, embedded: bool = False) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._schedule: ScheduleVM | None = None
        self._connection: ConnectionVM | None = None
        self._version = (-1, -1)
        self._cards: dict[str, PeriodCard] = {}
        self._component_cards: dict[tuple[str, str], ComponentCard] = {}
        self._expanded_components: set[str] = set()
        # Set by Add; the next schedule that brings exactly one new card is
        # what that Add produced, and the new card is picked.
        self._expect_new_card = False
        #: Embedded: the edit and port commands, for the host page to stand
        #: above its scrolling body.
        self.toolbar: QtWidgets.QWidget | None = None
        layout = QtWidgets.QVBoxLayout(self)
        # The page owns its edges: every tab's content starts and ends one
        # section gap in from the tab pane.  Embedded in another page, the
        # host page owns them and this view adds none of its own.  The
        # panes stand in the right inset, their vertical bar at the tab's
        # edge as on every tab whose page scrolls, and the inset is kept
        # whether or not the bar shows: the strip and the bottom bar under
        # it end on the content edge, bar or no bar.
        page = 0 if embedded else section_gap()
        layout.setContentsMargins(page, page, 0, page)
        layout.setSpacing(section_gap())
        gap = row_gap()

        # The fixed-width operator columns and the timeline are two panes of
        # ONE scrolling body: their rows line up, so they move together, and
        # LinkedScrollPanes owns the single vertical bar that appears only
        # while some pane actually overflows.  Nothing here decides how far
        # the operator can reach -- the taller pane's own range does.
        dataset_frame = LinkedScrollPanes(gutter=page)

        self.left_scroll = FluentScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.left_scroll.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.left_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.left_body = FluentPageBody()
        left_body = self.left_body
        left = QtWidgets.QHBoxLayout(left_body)
        left.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)

        # Each side panel stands at the page's left edge or one card gap
        # after its neighbour, and is as tall as its rows: it ends a card
        # pad below its last row, where every card of the strip ends, so
        # side by side they share the top AND the bottom.
        def hold(panel: QtWidgets.QWidget) -> QtWidgets.QWidget:
            holder = QtWidgets.QWidget()
            holder_layout = QtWidgets.QVBoxLayout(holder)
            holder_layout.setContentsMargins(0, 0, 0, 0)
            holder_layout.setSpacing(0)
            holder_layout.addWidget(panel, 0, QtCore.Qt.AlignTop)
            left.addWidget(holder)
            return holder

        self.names_panel = ChannelNamesPanel(ports_editable=not embedded)
        self.names_panel_holder = hold(self.names_panel)
        self.names_panel_holder.layout().setContentsMargins(0, 0, gap, 0)
        self.channel_panel = ChannelPanel()
        self.channel_panel_holder = hold(self.channel_panel)

        self.left_panel_stub = FluentFrame()
        self.left_panel_stub.setFixedWidth(scaled_px(82, minimum=68))
        stub_layout = QtWidgets.QVBoxLayout(self.left_panel_stub)
        pad = card_pad()
        stub_layout.setContentsMargins(pad, pad, pad, pad)
        stub_layout.setSpacing(row_gap())
        stub_label = FluentLabel("Name\nDelay")
        stub_label.setAlignment(QtCore.Qt.AlignCenter)
        stub_layout.addWidget(stub_label)
        self.stub_show_button = FluentButton("Show")
        self.stub_show_button.setFixedHeight(control_height())
        # As wide as the stub's column, under its centred label.
        self.stub_show_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.stub_show_button.clicked.connect(self._show_left_panels)
        stub_layout.addWidget(self.stub_show_button)
        self.left_panel_stub_holder = hold(self.left_panel_stub)
        self.left_panel_stub_holder.hide()

        self.left_scroll.setWidget(left_body)
        dataset_frame.add_pane(self.left_scroll)
        self._settle_left_pane_width()

        self.timeline_scroll = FluentScrollArea()
        self.timeline_scroll.setWidgetResizable(False)
        self.timeline_scroll.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.timeline_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.timeline_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.drag_container = PulseDragContainer()
        self.timeline_scroll.setWidget(self.drag_container)
        self.timeline_scroll.horizontalScrollBar().valueChanged.connect(self._clip_scrolled_gutter)
        dataset_frame.add_pane(self.timeline_scroll, stretch=1)
        # THE PAGE OWNS ITS SURFACE: a white region of its own behind the
        # panes, not Qt's private stacked widget painted white by one CSS
        # rule aimed at its internal object name.  Edgeless: the tab pane is
        # already the white page.
        self.dataset_surface = FluentFrame(bordered=False)
        surface_layout = QtWidgets.QVBoxLayout(self.dataset_surface)
        surface_layout.setContentsMargins(0, 0, 0, 0)
        surface_layout.setSpacing(0)
        surface_layout.addWidget(dataset_frame, 1)
        layout.addWidget(self.dataset_surface, 1)
        self.dataset_panes = dataset_frame

        # Bottom bar: three titled cards.  Control keeps its compact button
        # grid and the Pulse's one complete-run count; Connection and Ports
        # retain fixed widths so the timeline gets the remaining space.  The
        # bar spans the page's own edges, like the panes above it.
        self.button_frame = FluentFrame(bordered=False)
        self.button_frame.setObjectName("zlcPulseButtonBar")
        self.button_frame.setStyleSheet(
            "QFrame#zlcPulseButtonBar { background: transparent; border: none; }"
        )
        bar = QtWidgets.QHBoxLayout(self.button_frame)
        bar.setContentsMargins(0, 0, page, 0)
        bar.setSpacing(gap)
        # The bar is ONE table of rows read across three cards: row 2 of
        # Control sits beside row 2 of Connection and row 2 of Ports.  So the
        # row height, the gap between rows and the top inset are single
        # numbers here rather than three cards' private choices.  Each card
        # keeps every card's rhythm: the title band, the card pad, then the
        # first row.
        card_margins = (pad, pad, pad, pad)
        control_area = FluentGroupBox("Control")
        control_layout = QtWidgets.QVBoxLayout(control_area)
        control_layout.setContentsMargins(*card_margins)
        control_layout.setSpacing(gap)
        controls = QtWidgets.QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(row_gap())
        controls.setVerticalSpacing(gap)
        self.run_button = FluentButton("On Pulse")
        # Stop takes the board safe: it ends what the hardware is doing, and
        # every Stop in the suite wears the one destructive face for it.
        self.stop_button = FluentButton("Stop Pulse", role="destructive")
        self.sync_button = FluentButton("Sync")
        self.add_button = FluentButton("Add period")
        self.spacer_button = FluentButton("Add spacer")
        self.remove_button = FluentButton("Remove", role="destructive")
        self.bracket_button = FluentButton("Add bracket")
        self.subpulse_button = FluentButton("Add subpulse")
        # One primary at a time: On Pulse while it would put something new
        # on the board and the document is saved; Save Pulse, with its star,
        # while the file is behind the editor -- On Pulse is then a standard
        # command.
        self.save_button = FluentButton("Save Pulse")
        # Asks for the file to read: the ellipsis says more comes first.
        self.load_button = FluentButton("Open…")
        self.collapse_button = FluentButton("Hide")
        self.collapse_button.setToolTip("Hide side panels")
        self.group_button = FluentButton("Group Subpulse")
        # Three compact rows: execute; edit the timeline; save and organize.
        control_buttons = (
            (self.run_button, 0, 0, 2), (self.stop_button, 0, 2, 1), (self.remove_button, 0, 3, 1),
            (self.add_button, 1, 0, 1), (self.spacer_button, 1, 1, 1),
            (self.bracket_button, 1, 2, 1), (self.subpulse_button, 1, 3, 1),
            (self.save_button, 2, 0, 1), (self.load_button, 2, 1, 1),
            (self.group_button, 2, 2, 1),
        )
        for button, row, column, span in control_buttons:
            button.setFixedHeight(control_height())
            button.setMinimumWidth(scaled_px(74, minimum=62))
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            controls.addWidget(button, row, column, 1, span)
        view_controls = QtWidgets.QHBoxLayout()
        view_controls.setContentsMargins(0, 0, 0, 0)
        view_controls.setSpacing(gap)
        for button in (self.sync_button, self.collapse_button):
            button.setFixedHeight(control_height())
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            view_controls.addWidget(button, 1)
        controls.addLayout(view_controls, 2, 3)
        for column in range(4):
            controls.setColumnStretch(column, 1)
        control_layout.addLayout(controls)
        control_layout.addStretch(1)
        bar.addWidget(control_area, 1)

        connection_area = FluentGroupBox("Connection")
        connection_area.setFixedWidth(scaled_px(252, minimum=212))
        connection_layout = QtWidgets.QVBoxLayout(connection_area)
        connection_layout.setContentsMargins(*card_margins)
        connection_layout.setSpacing(gap)
        self.connection_combo = FluentComboBox()
        self.connection_combo.setFixedHeight(control_height())
        # The presenter projects the complete endpoint value before the window
        # is shown.  This widget neither seeds nor preserves a second copy.
        self.connection_endpoint = FluentLineEdit("")
        self.connection_endpoint.setToolTip(
            "Endpoint for a connection choice that accepts one, as host:port."
        )
        self.connection_combo.setToolTip(
            "Connection choices are supplied by the host that opened this editor."
        )
        self.connection_endpoint.setFixedHeight(control_height())
        self.connection_button = FluentButton("Connect")
        self.connection_button.setFixedHeight(control_height())
        # Read under two boxes: its words start where theirs do.
        self.connection_status = FluentReadoutEdit("")
        self.connection_status.setFixedHeight(control_height())
        connection_layout.addWidget(self.connection_combo)
        connection_row = QtWidgets.QHBoxLayout()
        connection_row.setContentsMargins(0, 0, 0, 0)
        connection_row.setSpacing(row_gap())
        connection_row.addWidget(self.connection_endpoint, 1)
        connection_row.addWidget(self.connection_button)
        connection_layout.addLayout(connection_row)
        connection_layout.addWidget(self.connection_status)
        connection_layout.addStretch(1)
        bar.addWidget(connection_area)

        ports_area = FluentGroupBox("Ports")
        ports_area.setFixedWidth(scaled_px(286, minimum=246))
        ports_layout = QtWidgets.QVBoxLayout(ports_area)
        ports_layout.setContentsMargins(*card_margins)
        ports_layout.setSpacing(gap)
        self.hidden_port_combo = FluentComboBox()
        self.hidden_port_combo.setFixedHeight(control_height())
        self.add_port_button = FluentButton("Add")
        self.hide_off_button = FluentButton("Hide off")
        self.hide_off_button.setToolTip("Hide the ports no period drives")
        self.show_all_button = FluentButton("Show all")
        # How many ports the combo above holds.  How many are shown is the
        # Port catalog's Visible line, said there once for the page.
        self.hidden_count_label = FluentReadoutEdit("")
        for button in (self.add_port_button, self.hide_off_button, self.show_all_button):
            button.setFixedHeight(control_height())
        self.hidden_count_label.setFixedHeight(control_height())
        ports_layout.addWidget(self.hidden_port_combo)
        ports_row = QtWidgets.QHBoxLayout()
        ports_row.setContentsMargins(0, 0, 0, 0)
        ports_row.setSpacing(row_gap())
        for button in (self.add_port_button, self.hide_off_button, self.show_all_button):
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            ports_row.addWidget(button, 1)
        ports_layout.addLayout(ports_row)
        ports_layout.addWidget(self.hidden_count_label)
        ports_layout.addStretch(1)
        bar.addWidget(ports_area)
        self.button_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        layout.addWidget(self.button_frame)

        self._wire_signals()
        if embedded:
            self.channel_panel_holder.hide()
            self.names_panel_holder.layout().setContentsMargins(0, 0, 0, 0)
            # Reuse the editing controls, but not the full Pulse execution bar.
            # Hidden execution buttons must not leave a three-row Ports panel
            # determining the height of this embedded editor.  The strip is
            # the schedule's header: its edit commands from the page's left
            # edge and its port commands ending at the right.
            compact = FluentFrame(bordered=False)
            toolbar = QtWidgets.QHBoxLayout(compact)
            toolbar.setContentsMargins(0, 0, 0, 0)
            toolbar.setSpacing(row_gap())
            for widget in (self.add_button, self.spacer_button, self.bracket_button, self.remove_button):
                widget.setParent(compact)
                widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
                toolbar.addWidget(widget)
            toolbar.addStretch(1)
            for widget in (self.hidden_port_combo, self.add_port_button, self.hide_off_button, self.show_all_button):
                widget.setParent(compact)
                toolbar.addWidget(widget)
            self.hidden_port_combo.setMaximumWidth(scaled_px(170))
            self.hidden_port_combo.setMinimumWidth(scaled_px(100))
            layout.removeWidget(self.button_frame)
            # The unused controls remain owned and hidden by the original bar;
            # ordinary projection methods still address their existing widgets.
            self.button_frame.hide()
            # The host page scrolls, and it is the one vertical scroller: the
            # panes are as tall as their rows, so they never scroll inside it
            # with a second bar beside its own.  The commands are the host's
            # to stand above that scrolling body, where they stay in reach.
            self.toolbar = compact
            self._fit_timer = QtCore.QTimer(self)
            self._fit_timer.setSingleShot(True)
            self._fit_timer.setInterval(0)
            self._fit_timer.timeout.connect(self._fit_embedded_height)
            self.left_body.installEventFilter(self)
            self.drag_container.installEventFilter(self)
            self.timeline_scroll.horizontalScrollBar().rangeChanged.connect(
                lambda *_range: self._fit_timer.start()
            )
            self._settle_left_pane_width()

    def eventFilter(self, watched, event):  # noqa: N802 - Qt name
        """Embedded: refit the panes to their rows when the rows change."""

        if (watched is self.left_body and event.type() == QtCore.QEvent.LayoutRequest
                or watched is self.drag_container and event.type() == QtCore.QEvent.Resize):
            self._fit_timer.start()
        return super().eventFilter(watched, event)

    def _fit_embedded_height(self) -> None:
        """As tall as the taller column of rows, and the timeline's own bar
        when it scrolls sideways: nothing is left below the fold of the
        panes, so their shared vertical bar never shows."""

        rows = max(self.left_body.sizeHint().height(), self.drag_container.sizeHint().height())
        bar = self.timeline_scroll.horizontalScrollBar()
        height = rows + 2 * self.timeline_scroll.frameWidth() + (
            bar.sizeHint().height() if bar.maximum() > bar.minimum() else 0
        )
        if self.dataset_panes.minimumHeight() != height or self.dataset_panes.maximumHeight() != height:
            self.dataset_panes.setFixedHeight(height)

    def _clip_scrolled_gutter(self, value: int) -> None:
        """Keep the card gap after the side panels clear while the timeline
        is scrolled.

        The gap is the strip's own leading margin -- where gap 0's caret
        stands and a click picks that gap -- so it scrolls away with the
        cards, and a scrolled card's sliver would stand against the panel's
        edge and read as part of it.  Scrolled, the viewport is masked by
        that gap: the cards clip one card gap from the panel, where they
        stand unscrolled.
        """

        viewport = self.timeline_scroll.viewport()
        if value > 0:
            gap = self.drag_container.layout_main.contentsMargins().left()
            viewport.setMask(QtGui.QRegion(gap, 0, _UNBOUNDED, _UNBOUNDED))
        else:
            viewport.clearMask()

    def _wire_signals(self) -> None:
        self.names_panel.document_name_committed.connect(self.document_name_committed)
        self.names_panel.port_label_committed.connect(self.port_label_committed)
        self.channel_panel.delay_committed.connect(self.delay_committed)
        self.channel_panel.feedback_requested.connect(self.feedback_requested)
        self.channel_panel.binding_committed.connect(self.binding_committed)
        self.channel_panel.config_binding_committed.connect(self.config_binding_committed)
        self.drag_container.config_binding_committed.connect(self.config_binding_committed)
        self.channel_panel.fill_port_requested.connect(self.fill_port_requested)
        self.channel_panel.clear_port_requested.connect(self.clear_port_requested)
        self.channel_panel.run_repeats_committed.connect(self.run_repeats_committed)
        self.drag_container.reorder_items_requested.connect(self.reorder_items_requested)
        self.drag_container.remove_items_requested.connect(self._request_remove_items)
        self.drag_container.bracket_count_committed.connect(self._commit_bracket_count)
        self.drag_container.bracket_binding_committed.connect(
            lambda bid, scan, source: self.binding_committed.emit("bracket_count", bid, None, scan, source)
        )
        self.drag_container.feedback_requested.connect(self.feedback_requested)
        self.run_button.clicked.connect(self.run_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.sync_button.clicked.connect(self.sync_requested)
        self.save_button.clicked.connect(self.save_requested)
        self.load_button.clicked.connect(self.load_requested)
        self.add_button.clicked.connect(lambda: self._request_insert("insert_period"))
        self.spacer_button.clicked.connect(lambda: self._request_insert("insert_spacer"))
        self.remove_button.clicked.connect(self._request_remove_items)
        self.bracket_button.clicked.connect(self._request_add_bracket)
        self.subpulse_button.clicked.connect(
            lambda: self.component_action_requested.emit("insert_file", None)
        )
        self.collapse_button.clicked.connect(self._toggle_left_panels)
        self.group_button.clicked.connect(self._request_group)
        self.add_port_button.clicked.connect(self._request_add_port)
        self.hide_off_button.clicked.connect(self._request_hide_off_ports)
        self.show_all_button.clicked.connect(self._request_show_all_ports)
        self.connection_button.clicked.connect(self._request_connection)
        self.connection_combo.currentIndexChanged[int].connect(
            self._sync_endpoint_enabled
        )
        self._sync_endpoint_enabled()

    def set_schedule(self, vm: ScheduleVM) -> bool:
        if not isinstance(vm, ScheduleVM):
            raise TypeError("vm must be ScheduleVM")
        incoming = (int(vm.document_generation), int(vm.revision))
        if incoming < self._version:
            return False
        if incoming == self._version and self._schedule is not None:
            if vm != self._schedule:
                raise ValueError("one schedule revision cannot identify two view models")
            return False
        self._schedule = vm
        self._version = incoming
        self._reconcile(vm)
        return True

    def _commit_bracket_count(self, bracket_id: str, count: int) -> None:
        bracket = self._bracket(bracket_id)
        if bracket is not None:
            self.bracket_committed.emit(
                bracket.bracket_id, bracket.start_period_id, bracket.end_period_id, count,
            )

    def _bracket(self, bracket_id: str) -> BracketVM | None:
        if self._schedule is None:
            return None
        return next(
            (item for item in self._schedule.brackets if item.bracket_id == bracket_id), None,
        )

    @staticmethod
    def _visible_delay_rows(vm: ScheduleVM) -> tuple[DelayRowVM, ...]:
        """The delay column shows exactly the rows the cards show.

        Which ports have rows is ONE fact: the names column and every period
        card filter on ``port.visible``, and the delay rows arrive as "every
        output the board can delay", so they are filtered the same way -- the
        columns are read across, and a delay beside the wrong channel is worse
        than no delay.

        Taken here, once, because both paths that fill the column pass through
        it: the full rebuild and the single-row push.
        """

        visible = {port.key for port in vm.ports if port.visible}
        return tuple(row for row in vm.delay_rows if row.port_key in visible)

    def _reconcile(self, vm: ScheduleVM) -> None:
        """Update existing rows and move timeline items only when their order changes."""
        card_width = period_card_width(ports=vm.ports, analog_mode_choices=vm.analog_mode_choices)
        self.names_panel.set_ports(vm.document_name, vm.ports)
        self.names_panel.set_summary(vm.total_text, vm.total_tooltip, vm.period_count, vm.visible_text)
        self.channel_panel.set_delay_rows(self._visible_delay_rows(vm), vm.ports)
        self.channel_panel.set_clock(vm.clock_text)
        self.channel_panel.set_scan_summary(vm.scan_summary_text)
        previous = set(self._cards)
        desired: dict[str, PeriodCard] = {}
        self._expanded_components.intersection_update(item.component_id for item in vm.components)
        grouped = {period_id for component in vm.components if component.component_id not in self._expanded_components for period_id in component.period_ids}
        internal_brackets = {key for component in vm.components if component.component_id not in self._expanded_components for key in component.bracket_ids}
        authored = [period for period in vm.periods if period.kind != PERIOD_KIND_SPACER]
        authored_indices = {period.period_id: index for index, period in enumerate(authored)}
        flat_items = vm.item_order
        for period in vm.periods:
            if period.period_id in grouped:
                continue
            index = authored_indices.get(period.period_id, 0)
            card = self._cards.get(period.period_id)
            if card is None:
                card = PeriodCard(
                    period,
                    index=index,
                    total_periods=len(authored),
                    ports=vm.ports,
                    analog_mode_choices=vm.analog_mode_choices,
                    card_width=card_width,
                )
                self._connect_card(card)
            else:
                card.set_period(
                    period,
                    index=index,
                    total_periods=len(authored),
                    ports=vm.ports,
                    analog_mode_choices=vm.analog_mode_choices,
                    card_width=card_width,
                )
            desired[period.period_id] = card
        self._cards = desired
        component_cards = {}
        item_blocks = {}
        item_owner = {}
        for component in vm.components:
            modes = ("start", "end") if component.component_id in self._expanded_components else ("closed",)
            for mode in modes:
                card_key = ("component" if mode == "closed" else f"component-{mode}", component.component_id)
                card = self._component_cards.get(card_key)
                if card is None:
                    card = ComponentCard(component, mode=mode, card_width=card_width)
                    card.action_requested.connect(self._component_action)
                elif mode == "closed":
                    card.setFixedWidth(card_width)
                    card.name_edit.setFixedWidth(period_control_width(card_width))
                card.set_component(component)
                component_cards[card_key] = card
            key = ("component", component.component_id)
            members = set(component.period_ids)
            bracket_ids = set(component.bracket_ids)
            block = tuple(item for item in flat_items if (
                item[0] == "period" and item[1] in members
                or item[0] == "bracket" and item[1].rpartition(":")[0] in bracket_ids
            ))
            item_blocks[key] = block
            item_owner.update({item: key for item in block})
        self._component_cards = component_cards
        order = []
        seen = set()
        for item in flat_items:
            key = item_owner.get(item, item)
            if key[0] == "component":
                expanded = key[1] in self._expanded_components
                if key not in seen:
                    order.append(("component-start", key[1]) if expanded else key)
                    seen.add(key)
                if expanded:
                    order.append(item)
                    if item == item_blocks[key][-1]:
                        order.append(("component-end", key[1]))
            else:
                order.append(key)
        self.drag_container.set_items(
            tuple(desired.values()),
            tuple(bracket for bracket in vm.brackets if bracket.bracket_id not in internal_brackets),
            order=tuple(order),
            minimum_bracket=vm.min_bracket_count,
            components=tuple(component_cards.values()),
            item_blocks=item_blocks,
        )
        arrived = set(desired) - previous
        if self._expect_new_card and len(arrived) == 1:
            self.drag_container.show_selection((("period", next(iter(arrived))),))
        self._expect_new_card = False
        if not being_edited(self.channel_panel.run_repeats_spin):
            self.channel_panel.set_run_repeats(vm.run_repeats)
        self.channel_panel.run_repeats_spin.setEnabled(bool(vm.periods))
        self._rebuild_hidden_ports(vm)
        # A renamed port or a different set of delay rows changes how wide the
        # operator columns want to be, and the pane's cached hint is just as
        # stale for that as it is for hiding the side panels.
        self._settle_left_pane_width()

    def _connect_card(self, card: PeriodCard) -> None:
        card.period_name_committed.connect(self.period_name_committed)
        card.duration_committed.connect(self.duration_committed)
        card.digital_committed.connect(self.digital_committed)
        card.analog_committed.connect(self.analog_committed)
        card.binding_committed.connect(self.binding_committed)
        card.config_binding_committed.connect(self.config_binding_committed)
        card.feedback_requested.connect(self.feedback_requested)

    def component_insert_before(self) -> tuple[str, str] | None:
        return self._selected_before_item()

    def _component_action(self, action: str, component_id: object) -> None:
        if action == "expand":
            key = str(component_id)
            opening = key not in self._expanded_components
            if opening:
                self._expanded_components.add(key)
            else:
                self._expanded_components.remove(key)
            if self._schedule is not None:
                self._reconcile(self._schedule)
            self.drag_container.show_selection(
                self.drag_container._item_blocks[("component", key)] if opening else (("component", key),)
            )
        else:
            self.component_action_requested.emit(action, component_id)

    def _request_group(self, _checked=False) -> None:
        if self._readonly_selection():
            return
        selected = self.drag_container.selection_payload()
        if selected:
            self.component_action_requested.emit("group", selected)
        else:
            self.feedback_requested.emit("Select the timeline items to group; Shift-click adds or removes one item.")

    def focus_component_name(self, component_id: str) -> None:
        """Put the cursor in a new block's name: the closed card's field,
        the one place on the strip a block is named."""

        card = self._component_cards.get(("component", component_id))
        if card is not None:
            self.timeline_scroll.ensureWidgetVisible(card)
            self.drag_container.show_selection((card.key,))
            card.name_edit.setFocus(QtCore.Qt.OtherFocusReason)
            card.name_edit.selectAll()

    def _rebuild_hidden_ports(self, vm: ScheduleVM) -> None:
        visible = {port.key for port in vm.ports if port.visible}
        hidden = [port for port in vm.ports if port.key not in visible]
        with signals_blocked(self.hidden_port_combo):
            self.hidden_port_combo.clear()
            for port in hidden:
                self.hidden_port_combo.addItem(port.label, port.key)
            if not hidden:
                self.hidden_port_combo.addItem("All ports shown", None)
        self.hidden_port_combo.setEnabled(bool(hidden))
        self.add_port_button.setEnabled(bool(hidden))
        # Counted from the list the combo holds; with none hidden the combo
        # already says "All ports shown".
        self.hidden_count_label.setText(f"{counted(len(hidden), 'port')} hidden" if hidden else "")

    def set_period(self, period: PeriodVM) -> None:
        if self._schedule is None:
            return
        periods = tuple(
            period if item.period_id == period.period_id else item
            for item in self._schedule.periods
        )
        self._schedule = replace(self._schedule, periods=periods)
        if period.period_id not in self._cards:
            return
        card = self._cards[period.period_id]
        authored = [item for item in periods if item.kind != PERIOD_KIND_SPACER]
        card.set_period(
            period,
            index=authored.index(period) if period in authored else 0,
            total_periods=len(authored),
            ports=self._schedule.ports,
            analog_mode_choices=self._schedule.analog_mode_choices,
        )
        # That call rebuilds any port row whose shape changed, and a row built
        # after the card was adopted is one the strip has never watched -- so a
        # click on it would select nothing.  Re-walked here, where the rebuild
        # is known to have happened, rather than guessed at from ChildAdded:
        # a child is not fully constructed when that fires, and touching it
        # there segfaults.
        self.drag_container.watch_item_chrome(card)

    def set_delay_row(self, row: DelayRowVM) -> None:
        if self._schedule is None:
            return
        rows = tuple(row if item.port_key == row.port_key else item for item in self._schedule.delay_rows)
        self._schedule = replace(self._schedule, delay_rows=rows)
        self.channel_panel.set_delay_rows(
            self._visible_delay_rows(self._schedule), self._schedule.ports
        )

    def set_port_label(self, key: str, label: str) -> None:
        """Rename one output everywhere it is shown, and in the model the
        next rebuild (a Hide/Show) reads."""

        if self._schedule is None:
            return
        self._schedule = replace(
            self._schedule,
            ports=tuple(
                replace(port, label=str(label)) if port.key == key else port
                for port in self._schedule.ports
            ),
        )
        self.names_panel.set_port_label(key, label)
        self.channel_panel.set_port_label(key, label)
        for card in self._cards.values():
            card.set_port_label(key, label)
        # A longer name widens the delay column's labels, and the pane with it.
        self._settle_left_pane_width()

    def set_visible_ports(self, ports: tuple[str, ...]) -> None:
        if self._schedule is None:
            return
        visible = set(str(key) for key in ports)
        updated = tuple(replace(port, visible=port.key in visible) for port in self._schedule.ports)
        # The rows only: the "N/M ports" text is the presenter's to word, and
        # it arrives with set_summary.
        self._schedule = replace(self._schedule, ports=updated)
        self._reconcile(self._schedule)

    def set_summary(self, total_text: str, total_tooltip: str, period_count: int, visible_text: str, summary_text: str, scan_summary_text: str) -> None:
        if self._schedule is None:
            return
        self._schedule = replace(self._schedule, total_text=str(total_text), total_tooltip=str(total_tooltip), period_count=int(period_count), visible_text=str(visible_text), summary_text=str(summary_text), scan_summary_text=str(scan_summary_text))
        self.names_panel.set_summary(str(total_text), str(total_tooltip), int(period_count), str(visible_text))
        self.channel_panel.set_scan_summary(str(scan_summary_text))

    def set_connection(self, vm: ConnectionVM) -> None:
        """Project the presenter's complete connection domain and authority."""

        if not isinstance(vm, ConnectionVM):
            raise TypeError("connection must be ConnectionVM")
        self._connection = vm
        with signals_blocked(self.connection_combo):
            self.connection_combo.clear()
            for choice in vm.choices:
                self.connection_combo.addItem(choice.label, choice.value)
            selected = self.connection_combo.findData(vm.selected)
            if selected < 0:
                raise ValueError(
                    f"selected connection {vm.selected!r} was not supplied to the view"
                )
            self.connection_combo.setCurrentIndex(selected)
        self.connection_endpoint.setText(vm.endpoint)
        self._sync_endpoint_enabled()
        self.connection_status.setText(vm.status)
        self.connection_status.setToolTip(vm.status)
        set_status_ink(self.connection_status, vm.status_word)

    def _request_connection(self) -> None:
        """Emit the selected presenter-supplied connection intent."""

        if self._connection is None or self._connection.locked:
            raise RuntimeError("this connection control is not operator-owned")
        mode = self.connection_combo.currentData()
        if not isinstance(mode, str) or not mode:
            raise ValueError("the selected connection has no domain value")
        self.connection_requested.emit(mode, self.connection_endpoint.text())

    def _sync_endpoint_enabled(self, _index: int | None = None) -> None:
        """An address that will not be dialled must not look like an input.

        The selected presenter-owned choice says whether it accepts an
        endpoint.  The widget does not infer that authority from a mode name.
        """

        operator_owned = self._connection is not None and not self._connection.locked
        current = self.connection_combo.currentData()
        choice = next(
            (
                item
                for item in (() if self._connection is None else self._connection.choices)
                if item.value == current
            ),
            None,
        )
        self.connection_combo.setEnabled(operator_owned)
        self.connection_endpoint.setEnabled(
            operator_owned and choice is not None and choice.endpoint_editable
        )
        self.connection_button.setEnabled(operator_owned)

    def set_control_state(
        self,
        running: bool,
        synchronized: bool,
        file_dirty: bool,
        *,
        can_run: bool,
        can_stop: bool,
    ) -> None:
        """What is happening, and which controls that leaves possible.

        Neither flag has a default on purpose: a control that looks available
        and cannot be is the failure a default would let a caller reintroduce.

        On Pulse is NOT gated on running: On Pulse means "put this on the
        board and play it", which is off-then-on, and it is how an operator
        applies an edit to a pulse that is already running.

        Its star means what every star means (Save*, Run*): pressing it now
        would put something new in place, and it is the one command to
        press.  So it wears one only while it can run, the board is not
        playing what is on screen, and the file is saved (otherwise Save
        Pulse is the one); the difference itself is said in its tooltip.

        Stop is gated on a board alone.  Going safe must work from any state,
        including one this window has misread, so it does not additionally
        require a pulse to be open or the board to look busy.
        """

        playing_this = bool(running) and bool(synchronized)
        self.run_button.setEnabled(bool(can_run))
        self.run_button.set_dirty(bool(can_run) and not bool(file_dirty) and not playing_this)
        self.run_button.setToolTip(
            "The board is playing this pulse" if playing_this
            else "The board is not playing what is on screen; On Pulse puts it there"
        )
        self.stop_button.setEnabled(bool(can_stop))
        self.save_button.set_dirty(bool(file_dirty))

    def set_capabilities(self, can_sync: bool, can_hold: bool, can_step: bool) -> None:
        """Apply presenter-advertised command availability without probing a controller.

        Sync is the one command here it gates; hold and step are the Scan
        page's, which the shell gates beside this call.  Sync's availability
        has this ONE writer, so a refresh never toggles it (and its focus)
        off and on again.
        """

        # Its own answer, not Run's.  Sync READS the board, so what it needs
        # is a board; ANDing it with Run meant it also needed a pulse, which
        # is exactly the case where reading one back is worth doing.
        self.sync_button.setEnabled(bool(can_sync))

    def _request_insert(self, action: str) -> None:
        """Add where the selection says, and pick what was added once it arrives."""

        if self._readonly_selection():
            return
        self._expect_new_card = True
        before = self._selected_before_item()
        selected = set(self.drag_container.selected_items())
        gap = self.drag_container.selected_gap
        visual = tuple(self.drag_container._item_key(item) for item in self.drag_container.items())
        for key in self._expanded_components:
            block = self.drag_container._item_blocks[("component", key)]
            first, last = visual.index(("component-start", key)), visual.index(("component-end", key))
            if selected and selected.issubset(block) or gap is not None and first < gap <= last:
                self.component_action_requested.emit(action, (key, before if before in block else None))
                return
        getattr(self, f"{action}_requested").emit(before)

    def _selected_before_item(self) -> tuple[str, str] | None:
        """Add in the selected visual gap, or immediately after the selection."""
        visual = tuple(self.drag_container._item_key(item) for item in self.drag_container.items())
        gap = self.drag_container.selected_gap
        if gap is not None:
            after = gap
        else:
            selected = set(self.drag_container.selected_items())
            if not selected:
                return None
            for kind, key in tuple(selected):
                if kind in ("component-start", "component-end"):
                    selected.add(("component-end", key))
            after = max(visual.index(key) for key in selected) + 1
        if after < len(visual):
            following = self.drag_container.flat_order(visual[after:])
            return following[0] if following else None
        return None

    def _readonly_selection(self) -> bool:
        if self._schedule is None:
            return False
        selected = self.drag_container.selection_payload()
        locked = {("period", item.period_id) for item in self._schedule.periods if item.readonly}
        locked.update(("bracket", bracket_post_key(item.bracket_id, side))
                      for item in self._schedule.brackets if item.readonly for side in ("start", "end"))
        gap = self.drag_container.selected_gap
        visual = tuple(self.drag_container._item_key(item) for item in self.drag_container.items())
        inside = gap is not None and any(
            component.config_active and component.component_id in self._expanded_components
            and visual.index(("component-start", component.component_id)) < gap
            <= visual.index(("component-end", component.component_id))
            for component in self._schedule.components
        )
        if locked.intersection(selected) or inside:
            self.feedback_requested.emit("This Subpulse comes from Config. Open its source file to edit it.")
            return True
        return False

    def _request_remove_items(self, _items=None) -> None:
        """One atomic removal request; no intermediate broken bracket state."""
        if self._readonly_selection():
            return
        selected = self.drag_container.selection_payload()
        if selected:
            self.remove_items_requested.emit(selected)

    def _request_add_bracket(self) -> None:
        """A new bracket around whatever is picked: a period, a bracket, a gap, or all.

        Around a selected bracket it is the enclosing loop -- that is how
        brackets nest.  In a selected gap it is empty, to be filled by
        dragging periods in.  With nothing picked it frames the whole pulse.
        """

        if self._readonly_selection() or self._schedule is None or not self._schedule.periods:
            return
        periods = self._schedule.periods
        selected = self.drag_container.selection_payload()
        gap = self.drag_container.selected_gap
        count = self._schedule.default_bracket_count
        flat = self.drag_container.flat_order(selected)
        bracket = (self._bracket(flat[0][1].rpartition(":")[0])
                   if len(flat) == 1 and flat[0][0] == "bracket" else None)
        if bracket is not None:
            start, end = bracket.start_period_id, bracket.end_period_id
        elif selected:
            ids = tuple(period.period_id for period in periods)
            chosen = {key for kind, key in flat if kind == "period"}
            positions = [index for index, key in enumerate(ids) if key in chosen]
            if not positions or len(positions) != positions[-1] - positions[0] + 1:
                self.feedback_requested.emit("Select a contiguous set of periods for the bracket; unselected periods are not added.")
                return
            for kind, key in flat:
                if kind == "bracket":
                    bracket = self._bracket(key.rpartition(":")[0])
                    first, stop = bracket_gap_bounds(ids, bracket.start_period_id, bracket.end_period_id)
                    if not set(ids[first:stop]).issubset(chosen):
                        self.feedback_requested.emit("The selected bracket extends beyond the selected periods.")
                        return
            start, end = ids[positions[0]], ids[positions[-1]]
        elif gap is not None:
            ids = tuple(period.period_id for period in periods)
            periods_before = sum(
                kind == "period" for kind, _key in self.drag_container.flat_order(
                    tuple(self.drag_container._item_key(item) for item in self.drag_container.items())[:gap]
                )
            )
            start = ids[periods_before] if periods_before < len(ids) else None
            end = ids[periods_before - 1] if periods_before > 0 else None
        else:
            start, end = periods[0].period_id, periods[-1].period_id
        visual = tuple(self.drag_container._item_key(item) for item in self.drag_container.items())
        for component in self._schedule.components:
            key = component.component_id
            if key not in self._expanded_components:
                continue
            block = self.drag_container._item_blocks[("component", key)]
            first, last = visual.index(("component-start", key)), visual.index(("component-end", key))
            if selected and set(selected).issubset(block) or gap is not None and first < gap <= last:
                self.component_action_requested.emit("bracket_add", (key, (
                    start if start in component.period_ids else None,
                    end if end in component.period_ids else None, count,
                )))
                return
        self.bracket_add_requested.emit(start, end, count)

    def _request_add_port(self) -> None:
        key = self.hidden_port_combo.currentData()
        if self._schedule is None or key is None:
            return
        visible = tuple(port.key for port in self._schedule.ports if port.visible)
        self.visible_ports_committed.emit((*visible, str(key)))

    def _driven_keys(self) -> set[str]:
        """Which outputs the pulse on screen actually drives.

        Computed from the periods this view is holding, not read off a flag on
        the port rows.  Whether a lane is high is edited constantly and the
        port rows are only re-pushed on a STRUCTURAL change, so a flag would be
        stale for every edit in between -- and "Hide off" reads exactly it.
        """

        driven: set[str] = set()
        for period in (self._schedule.periods if self._schedule else ()):
            driven.update(key for key, on in period.digital if on)
            driven.update(key for key, _mode, field in period.analog if field.text.strip())
        return driven

    def _request_hide_off_ports(self) -> None:
        if self._schedule is None:
            return
        driven = self._driven_keys()
        if not driven:
            # Hiding every row is not a view of the board.  Say so instead.
            self.feedback_requested.emit(
                "Nothing is driven yet, so there is nothing to hide."
            )
            return
        self.visible_ports_committed.emit(
            tuple(port.key for port in self._schedule.ports if port.key in driven)
        )

    def _request_show_all_ports(self) -> None:
        if self._schedule is not None:
            self.visible_ports_committed.emit(tuple(port.key for port in self._schedule.ports))

    def _settle_left_pane_width(self) -> None:
        """The operator columns are exactly as wide as the columns showing.

        A QScrollArea does NOT shrink-wrap.  ``AdjustToContents`` computes the
        area's size hint once and caches it, and neither hiding the widget's
        children, nor a LayoutRequest, nor re-setting ``widgetResizable``
        clears that cache.

        The width is not the scroll area's to remember: it is the width of the
        panels inside it, set here whenever it may have changed.  Set, not
        capped: the panels size their label column to the labels they show,
        so a renamed port or another set of rows can WIDEN them, and a cap
        can only ever take width away from the hint the area cached when it
        was built.
        """

        layout = self.left_body.layout()
        if layout is not None:
            # A panel sized to its labels tells only its holder, whose own
            # hint the body's layout keeps cached until the holder says it
            # changed; and hidden children invalidate the layout.  The hint
            # is only current once both are recomputed, and this is read in
            # the same turn as the change that caused it.
            for holder in (self.names_panel_holder, self.channel_panel_holder,
                           self.left_panel_stub_holder):
                holder.updateGeometry()
            layout.activate()
        self.left_scroll.setFixedWidth(self.left_body.sizeHint().width())

    def _toggle_left_panels(self) -> None:
        visible = self.names_panel_holder.isVisible()
        self.names_panel_holder.setVisible(not visible)
        self.channel_panel_holder.setVisible(not visible)
        self.left_panel_stub_holder.setVisible(visible)
        self.collapse_button.setText("Show" if visible else "Hide")
        self.collapse_button.setToolTip("Show side panels" if visible else "Hide side panels")
        self.collapse_button.set_active(visible)
        self._settle_left_pane_width()

    def _show_left_panels(self) -> None:
        self.names_panel_holder.show()
        self.channel_panel_holder.show()
        self.left_panel_stub_holder.hide()
        self.collapse_button.setText("Hide")
        self.collapse_button.setToolTip("Hide side panels")
        self.collapse_button.set_active(False)
        self._settle_left_pane_width()


__all__ = [
    "ChannelNamesPanel", "ChannelPanel", "PeriodCard", "PulseDragContainer",
    "PulseScheduleView", "BracketPost", "bracket_spans_of",
]
