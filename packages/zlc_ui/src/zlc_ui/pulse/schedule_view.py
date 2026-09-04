"""Pure Qt projection for the pulse schedule page.

The original schedule page was a useful visual reference, but its public
entry point accepted a domain document and performed projection in the
widget.  This version keeps the extracted interaction shape while accepting
only :mod:`zlc_ui.pulse.models` records.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.form import FormChoice
from zlc_ui.fluent import (
    ACCENT, GREEN, GREY, ORANGE, RED, YELLOW, FluentButton, FluentCheckBox,
    FluentComboBox, FluentDoubleSpinBox, FluentFrame, FluentGroupBox,
    FluentLabel, FluentLineEdit, FluentScrollArea, LinkedScrollPanes,
    signals_blocked,
)

from ._layout import (
    add_labeled_widget, bus_mode_combo_width, channel_label_width,
    card_gutter, channel_name_edit_width, channel_row_height, hide_button_width,
    panel_top_height, px, period_card_width, period_control_width, row_height,
    row_region_vmetrics, time_unit_width,
)
from .models import (
    VALIDATOR_FLOAT,
    VALIDATOR_INT,
    ConnectionVM,
    DelayRowVM,
    FieldVM,
    PeriodVM,
    PortRowVM,
    BracketVM,
    ScheduleVM,
)

#: The address the pulse server prints for a client on the same machine.  One
#: constant so the seeded field, the tooltip and any host default agree.
from .scan_line_edit import FluentScanLineEdit


def _field_binding(field: FieldVM) -> tuple[str | None, int | None]:
    kind = str(field.binding_kind or "").strip().lower()
    return (kind or None, int(field.binding_number) if kind else None)


def _apply_field(widget: FluentScanLineEdit, field: FieldVM) -> None:
    binding, number = _field_binding(field)
    with signals_blocked(widget):
        widget.setText(field.text)
        widget.set_field_state(editable=field.editable, binding=binding, number=number)
        if field.validator_kind in (VALIDATOR_INT, VALIDATOR_FLOAT) and not binding == "scan":
            widget.set_numeric_validator(
                field.validator_kind,
                bottom=field.validator_lo,
                top=field.validator_hi if field.validator_kind == VALIDATOR_INT else None,
            )
            widget.set_resolution(field.resolution or None)
            widget.set_allow_any(field.allow_any)
        else:
            widget.setValidator(None)


class PeriodCard(FluentGroupBox):
    """One stable period card keyed by ``period_id``."""

    period_name_committed = QtCore.pyqtSignal(str, str)
    duration_committed = QtCore.pyqtSignal(str, object, str)
    digital_committed = QtCore.pyqtSignal(str, str, bool)
    analog_committed = QtCore.pyqtSignal(str, str, str, object)
    binding_cycle_requested = QtCore.pyqtSignal(str, object, object)

    def __init__(self, period: PeriodVM, *, index: int = 0, total_periods: int = 1,
                 ports: tuple[PortRowVM, ...] = (),
                 analog_mode_choices: tuple[FormChoice, ...] = (), parent=None) -> None:
        if not isinstance(period, PeriodVM):
            raise TypeError("period must be PeriodVM")
        super().__init__("", parent)
        self.period_id = period.period_id
        self._period = period
        self._ports: dict[str, PortRowVM] = {}
        self.checks: dict[str, FluentCheckBox] = {}
        self.bus_mode_combos: dict[str, FluentComboBox] = {}
        self.bus_value_edits: dict[str, FluentScanLineEdit] = {}
        self.port_rows: dict[str, QtWidgets.QWidget] = {}
        self._last_duration: tuple[float, str] = (0.0, "")
        self._last_name = ""

        width = period_card_width()
        self.setFixedWidth(width)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(px(7), px(7), px(7), px(7))
        column.setSpacing(px(4, minimum=3))
        top = QtWidgets.QWidget()
        top.setFixedHeight(panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(px(6, minimum=4))
        top_layout.addWidget(self._center_label("Duration"))
        self.duration_edit = FluentScanLineEdit(
            "",
            tooltip="Click the dot to cycle off → Scan → API → Config → off",
        )
        control_width = period_control_width(width)
        self.duration_edit.setFixedWidth(control_width)
        top_layout.addWidget(self.duration_edit)
        self.duration_dot = self.duration_edit.dot
        self.unit_combo = FluentComboBox()
        self.unit_combo.setFixedWidth(control_width)
        top_layout.addWidget(self.unit_combo)
        self.name_edit = FluentLineEdit("")
        self.name_edit.setPlaceholderText("name")
        self.name_edit.setFixedWidth(control_width)
        top_layout.addWidget(self.name_edit)
        top_layout.addStretch(1)
        column.addWidget(top)
        row_top, _row_gap = row_region_vmetrics()
        column.addSpacing(max(0, row_top - px(7)))
        self._column = column
        self.duration_edit.editingFinished.connect(self._commit_duration)
        self.duration_dot.clicked.connect(self._cycle_duration_binding)
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
    def _center_label(text: str) -> FluentLabel:
        label = FluentLabel(text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setFixedHeight(row_height())
        return label

    def set_period(self, period: PeriodVM, *, index: int, total_periods: int,
                   ports: tuple[PortRowVM, ...],
                   analog_mode_choices: tuple[FormChoice, ...]) -> None:
        if period.period_id != self.period_id:
            raise ValueError("period identity cannot change")
        self._period = period
        self.setTitle(f"Period {int(index) + 1}/{max(1, int(total_periods))}")
        with signals_blocked(self.unit_combo):
            self.unit_combo.clear()
            choices = period.unit_choices or (period.unit,)
            self.unit_combo.addItems([str(value) for value in choices])
            self.unit_combo.setCurrentText(period.unit)
        _apply_field(self.duration_edit, period.duration)
        self.name_edit.setText(period.name)
        self._last_name = period.name
        try:
            self._last_duration = (float(period.duration.text), period.unit)
        except ValueError:
            self._last_duration = (0.0, period.unit)
        self._analog_mode_choices = tuple(analog_mode_choices)
        self._reconcile_ports(tuple(ports), period)

    def _set_analog_mode(self, combo: FluentComboBox, mode: str) -> None:
        with signals_blocked(combo):
            combo.clear()
            for choice in self._analog_mode_choices:
                combo.addItem(choice.label, choice.value)
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
                widget.hide()
                widget.deleteLater()
                self.checks.pop(key, None)
                self.bus_mode_combos.pop(key, None)
                self.bus_value_edits.pop(key, None)
        digital = dict(period.digital)
        analog = {key: (mode, field) for key, mode, field in period.analog}
        for port in ports:
            if not port.visible:
                continue
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
                        _apply_field(edit, field)
                continue
            if port.kind == "digital":
                widget = FluentCheckBox(port.label)
                widget.setChecked(bool(digital.get(port.key, False)))
                widget.setFixedHeight(channel_row_height())
                widget.toggled.connect(lambda checked, key=port.key: self.digital_committed.emit(self.period_id, key, bool(checked)))
                self.checks[port.key] = widget
            else:
                if port.key not in analog:
                    raise ValueError(
                        f"period {period.period_id!r} has no analog row for {port.key!r}"
                    )
                mode, field = analog[port.key]
                widget = QtWidgets.QWidget()
                widget.setFixedHeight(channel_row_height())
                row_layout = QtWidgets.QHBoxLayout(widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(px(4, minimum=3))
                combo = FluentComboBox()
                self._set_analog_mode(combo, mode)
                combo.setFixedWidth(
                    bus_mode_combo_width(
                        tuple(choice.label for choice in self._analog_mode_choices)
                    )
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
                        "Click the dot to cycle off → Scan → API → Config → off"
                    ),
                )
                edit.set_numeric_validator("int", bottom=port.lo, top=port.hi)
                edit.setFixedHeight(channel_row_height())
                _apply_field(edit, field)
                combo.currentIndexChanged[int].connect(
                    lambda _index, key=port.key: self._commit_analog(key)
                )
                edit.editingFinished.connect(lambda key=port.key: self._commit_analog(key))
                edit.dot.clicked.connect(lambda _checked=False, key=port.key: self.binding_cycle_requested.emit("analog", self.period_id, key))
                row_layout.addWidget(combo)
                row_layout.addWidget(edit, 1)
                self.bus_mode_combos[port.key] = combo
                self.bus_value_edits[port.key] = edit
            self.port_rows[port.key] = widget
        ordered = [port.key for port in ports if port.visible]
        for index, key in enumerate(ordered):
            widget = self.port_rows[key]
            self._column.removeWidget(widget)
            self._column.insertWidget(2 + index, widget)
            widget.show()
        self._ports = {port.key: port for port in ports}

    def _cycle_duration_binding(self, _checked: bool = False) -> None:
        self.binding_cycle_requested.emit("duration", self.period_id, None)

    def _commit_name(self) -> None:
        value = self.name_edit.text()
        if value == self._last_name:
            return
        self._last_name = value
        self.period_name_committed.emit(self.period_id, value)

    def _commit_duration(self, _unit: str | None = None) -> None:
        if not self.unit_combo.isEnabled():
            return
        try:
            value = float(self.duration_edit.text())
        except ValueError:
            return
        unit = self.unit_combo.currentText()
        if (value, unit) == self._last_duration:
            return
        self._last_duration = (value, unit)
        self.duration_committed.emit(self.period_id, value, unit)

    def _commit_analog(self, port: str) -> None:
        combo = self.bus_mode_combos.get(port)
        edit = self.bus_value_edits.get(port)
        if combo is None or edit is None:
            return
        mode = combo.currentData()
        if not isinstance(mode, str) or not mode:
            raise ValueError("the selected analog mode has no domain value")
        try:
            value = int(float(edit.text()))
        except ValueError:
            return
        self.analog_committed.emit(self.period_id, port, mode, value)

    def set_port_label(self, port: str, label: str) -> None:
        check = self.checks.get(port)
        if check is not None:
            check.setText(str(label))


class ChannelNamesPanel(FluentGroupBox):
    document_name_committed = QtCore.pyqtSignal(str)
    port_label_committed = QtCore.pyqtSignal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__("Port Catalog", parent)
        panel_width = (
            channel_label_width()
            + channel_name_edit_width()
            + px(5)
            + px(20)
        )
        self.setMinimumWidth(panel_width)
        self.setMaximumWidth(panel_width)
        self.name_edit = FluentLineEdit("")
        self.document_name_edit = self.name_edit
        self.name_edit.setPlaceholderText("pulse name")
        self.name_edit.editingFinished.connect(lambda: self.document_name_committed.emit(self.name_edit.text()))
        self.total_label = FluentLineEdit("")
        self.periods_label = FluentLineEdit("")
        self.visible_label = FluentLineEdit("")
        for field in (self.total_label, self.periods_label, self.visible_label):
            field.setEnabled(False)
        self._layout = QtWidgets.QVBoxLayout(self)
        top_margin, row_gap = row_region_vmetrics()
        self._layout.setContentsMargins(px(8), top_margin, px(8), px(8))
        self._layout.setSpacing(row_gap)
        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(px(6, minimum=4))
        add_labeled_widget(top_layout, "Name:", self.name_edit)
        add_labeled_widget(top_layout, "Total:", self.total_label)
        add_labeled_widget(top_layout, "Periods:", self.periods_label)
        add_labeled_widget(top_layout, "Visible:", self.visible_label)
        top_layout.addStretch(1)
        self._layout.addWidget(top)
        self._rows: dict[str, FluentLineEdit] = {}
        self._row_holders: dict[str, QtWidgets.QWidget] = {}
        self._layout.addStretch(1)

    def set_ports(self, document_name: str, ports: tuple[PortRowVM, ...]) -> None:
        self.name_edit.setText(str(document_name))
        existing = self._rows
        self._rows = {}
        for port in ports:
            field = existing.pop(port.key, None)
            if field is None:
                field = FluentLineEdit(port.label)
                field.editingFinished.connect(lambda key=port.key, edit=field: self.port_label_committed.emit(key, edit.text()))
                holder = QtWidgets.QWidget()
                holder.setStyleSheet("background: transparent;")
                row = QtWidgets.QHBoxLayout(holder)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(px(5, minimum=3))
                endpoint = FluentLabel(port.endpoint_text or port.key)
                endpoint.setAlignment(QtCore.Qt.AlignCenter)
                endpoint.setFixedSize(channel_label_width(), channel_row_height())
                field.setFixedWidth(channel_name_edit_width())
                field.setFixedHeight(channel_row_height())
                row.addWidget(endpoint)
                row.addWidget(field, 1)
                self._layout.insertWidget(self._layout.count() - 1, holder)
                self._row_holders[port.key] = holder
            field.setText(port.label)
            field.setEnabled(port.visible)
            self._row_holders[port.key].setVisible(bool(port.visible))
            self._rows[port.key] = field
        for key, field in existing.items():
            holder = self._row_holders.pop(key, field)
            self._layout.removeWidget(holder)
            holder.deleteLater()

    def set_port_label(self, key: str, label: str) -> None:
        if key in self._rows:
            self._rows[key].setText(label)

    def set_summary(self, total_text: str, period_count: int, visible_text: str) -> None:
        self.total_label.setText(str(total_text))
        self.periods_label.setText(str(period_count))
        self.visible_label.setText(str(visible_text))


class ChannelPanel(FluentGroupBox):
    delay_committed = QtCore.pyqtSignal(str, object, str)
    binding_cycle_requested = QtCore.pyqtSignal(str, object, object)
    clear_port_requested = QtCore.pyqtSignal(str)
    scan_array_load_requested = QtCore.pyqtSignal()
    run_repeats_committed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__("Delay / Scan", parent)
        label_width = channel_label_width()
        delay_width = px(70, minimum=60)
        gap = px(4, minimum=3)
        panel_width = (
            label_width
            + delay_width
            + time_unit_width()
            + hide_button_width()
            + gap * 3
            + px(16)
        )
        self.setMinimumWidth(panel_width)
        self.setMaximumWidth(panel_width)
        self._layout = QtWidgets.QVBoxLayout(self)
        top_margin, row_gap = row_region_vmetrics()
        self._layout.setContentsMargins(px(8), top_margin, px(8), px(8))
        self._layout.setSpacing(row_gap)
        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(px(6, minimum=4))
        self.clock_label = FluentLineEdit("")
        self.clock_label.setEnabled(False)
        self.scan_summary_label = FluentLineEdit("")
        self.scan_summary_label.setEnabled(False)
        # How many times the whole thing plays is a number the PULSE carries,
        # like its clock and its scan table -- not an action, which is all the
        # Control group beside it holds.  Reading it directly under the scan
        # summary says the one sentence they make together: what will be
        # played, and how many times.
        self.run_repeats_spin = FluentDoubleSpinBox()
        self.run_repeats_spin._step_btn.hide()
        self.run_repeats_spin.setDecimals(0)
        self.run_repeats_spin.setSingleStep(1.0)
        self.run_repeats_spin.setRange(0.0, float((1 << 32) - 1))
        self.run_repeats_spin.setToolTip(
            "Complete Pulse runs per scan point (or per On Pulse without a "
            "scan); 0 runs indefinitely"
        )
        self.run_repeats_spin.editingFinished.connect(
            lambda: self.run_repeats_committed.emit(
                int(self.run_repeats_spin.value())
            )
        )
        add_labeled_widget(top_layout, "Clock:", self.clock_label)
        add_labeled_widget(top_layout, "Scan:", self.scan_summary_label)
        add_labeled_widget(top_layout, "Repeat:", self.run_repeats_spin)
        self.load_array_button = FluentButton("Load Array", color=ACCENT)
        self.load_array_button.clicked.connect(self.scan_array_load_requested)
        top_layout.addWidget(self.load_array_button)
        top_layout.addStretch(1)
        self._layout.addWidget(top)
        self._rows: dict[str, tuple[FluentScanLineEdit, FluentComboBox, FluentButton]] = {}
        self._row_labels: dict[str, FluentLabel] = {}
        self._layout.addStretch(1)

    def set_delay_rows(
        self,
        rows: tuple[DelayRowVM, ...],
        labels: dict[str, str] | None = None,
    ) -> None:
        labels = {} if labels is None else {
            str(key): str(value) for key, value in labels.items()
        }
        existing = self._rows
        self._rows = {}
        for row in rows:
            current = existing.pop(row.port_key, None)
            if current is None:
                edit = FluentScanLineEdit(
                    row.value.text,
                    tooltip="Click the dot to cycle off → API → Config → off",
                )
                edit.setFixedWidth(px(70, minimum=60))
                combo = FluentComboBox()
                combo.setFixedWidth(time_unit_width())
                clear = FluentButton("X", color=ORANGE)
                clear.setFixedWidth(hide_button_width())
                edit.dot.clicked.connect(lambda _checked=False, key=row.port_key: self.binding_cycle_requested.emit("delay", None, key))
                edit.editingFinished.connect(lambda key=row.port_key, field=edit, units=combo: self._emit_delay(key, field, units))
                combo.currentTextChanged.connect(lambda _text, key=row.port_key, field=edit, units=combo: self._emit_delay(key, field, units))
                clear.clicked.connect(lambda _checked=False, key=row.port_key: self.clear_port_requested.emit(key))
                holder = QtWidgets.QWidget()
                holder_layout = QtWidgets.QHBoxLayout(holder)
                holder_layout.setContentsMargins(0, 0, 0, 0)
                label = FluentLabel(labels.get(row.port_key, row.port_key))
                label.setFixedSize(channel_label_width(), channel_row_height())
                label.setAlignment(QtCore.Qt.AlignCenter)
                holder_layout.addWidget(label)
                holder_layout.addWidget(edit)
                holder_layout.addWidget(combo)
                holder_layout.addWidget(clear)
                self._layout.insertWidget(self._layout.count() - 1, holder)
                self._row_labels[row.port_key] = label
                current = (edit, combo, clear)
            edit, combo, clear = current
            label = self._row_labels.get(row.port_key)
            if label is not None:
                label.setText(labels.get(row.port_key, label.text()))
            _apply_field(edit, row.value)
            with signals_blocked(combo):
                combo.clear()
                combo.addItems([unit for unit, _quantum in row.unit_quantums] or [row.unit])
                combo.setCurrentText(row.unit)
            self._rows[row.port_key] = current
        for key, (edit, combo, clear) in existing.items():
            holder = edit.parentWidget()
            if holder is not None:
                self._layout.removeWidget(holder)
                holder.deleteLater()
            self._row_labels.pop(key, None)

    def set_port_label(self, key: str, label: str) -> None:
        widget = self._row_labels.get(str(key))
        if widget is not None:
            widget.setText(str(label))

    def _emit_delay(self, key: str, field: FluentScanLineEdit, units: FluentComboBox) -> None:
        try:
            value = float(field.text())
        except ValueError:
            return
        self.delay_committed.emit(str(key), value, units.currentText())

    def set_clock(self, text: str) -> None:
        self.clock_label.setText(str(text))

    def set_scan_summary(self, text: str) -> None:
        self.scan_summary_label.setText(str(text))

BRACKET_WIDTH = 78


class BracketPost(FluentGroupBox):
    """One draggable post framing a bracketed run of periods.

    The shape is the point: an untitled column built to the same height and
    header block as a period card, so its "Bracket" sits on the cards' own
    header line and its count on their first control line.  The
    bracket then reads as a frame drawn around cards rather than a widget
    wedged between them.

    It had become a titled box with a glyph in it, at whatever height the
    layout happened to give -- the thing marking a span lined up with nothing
    in the span it marked.
    """

    count_committed = QtCore.pyqtSignal(int)

    def __init__(self, kind: str, *, count: int = 2, minimum: int = 2, parent=None) -> None:
        super().__init__("", parent)
        self.kind = str(kind)
        width = px(BRACKET_WIDTH, minimum=60)
        self.setFixedWidth(width)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(px(7), px(7), px(7), px(7))
        column.setSpacing(px(4, minimum=3))
        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(px(6, minimum=4))
        label = FluentLabel("Bracket")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setFixedHeight(row_height())
        top_layout.addWidget(label)
        # The count belongs to the end post: a span is closed by saying how
        # many times.  The start post keeps an empty line of the same height so
        # the two posts stay level with each other and with the cards.
        self.setCursor(QtCore.Qt.SizeHorCursor)
        self.setToolTip("Drag this post to any valid period gap")
        self.count_spin = FluentDoubleSpinBox()
        self.count_spin._step_btn.hide()
        self.count_spin.setDecimals(0)
        self.count_spin.setSingleStep(1.0)
        self.count_spin.setRange(float(minimum), float((1 << 32) - 1))
        self.count_spin.setValue(float(max(int(minimum), int(count))))
        self.count_spin.setFixedSize(width - 2 * px(7), row_height())
        self.count_spin.valueChanged.connect(
            lambda value: self.count_committed.emit(int(value))
        )
        if kind == "start":
            self.count_spin.hide()
            spacer = QtWidgets.QWidget()
            spacer.setStyleSheet("background: transparent;")
            spacer.setFixedHeight(row_height())
            top_layout.addWidget(spacer)
        else:
            top_layout.addWidget(self.count_spin)
        top_layout.addStretch(1)
        column.addWidget(top)
        row_top, _row_gap = row_region_vmetrics()
        column.addSpacing(max(0, row_top - px(7)))
        column.addStretch(1)


class PulseDragContainer(QtWidgets.QWidget):
    """Horizontal card strip whose drag releases are proposal-only."""

    period_clicked = QtCore.pyqtSignal(str)
    bracket_clicked = QtCore.pyqtSignal(str)
    gap_clicked = QtCore.pyqtSignal(int)
    move_period_requested = QtCore.pyqtSignal(str, object)
    bracket_committed = QtCore.pyqtSignal(object, object, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.layout_main = QtWidgets.QHBoxLayout(self)
        self.layout_main.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        pad = card_gutter()
        self.layout_main.setContentsMargins(pad, pad, pad, pad)
        self.layout_main.setSpacing(px(5, minimum=3))
        self.layout_main.setAlignment(QtCore.Qt.AlignLeft)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._cards: tuple[PeriodCard, ...] = ()
        self._posts: tuple["BracketPost", ...] = ()
        self._pressed: tuple[str, str, QtCore.QPoint] | None = None
        self._dragging = False
        self._indicator = QtWidgets.QFrame(self)
        self._indicator.setFixedWidth(px(3, minimum=2))
        self._indicator.setStyleSheet(f"background: {ACCENT};")
        self._indicator.hide()
        # WHERE the next Add lands and WHICH period Remove takes: a card, a gap,
        # or neither.  The two are mutually exclusive because they answer the
        # same question, and clicking the current one again clears it.
        self._selected_card: str | None = None
        self._selected_post: str | None = None
        self._selected_gap: int | None = None

    def set_items(
        self,
        cards: tuple[PeriodCard, ...],
        bracket: BracketVM | None,
        *,
        minimum_bracket: int = 2,
    ) -> None:
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            widget.removeEventFilter(self)
            if widget in cards:
                # About to be re-added below.  Taking it out of the LAYOUT is
                # the whole job; setParent(None) additionally makes it a
                # top-level window, which is a different thing and one this
                # widget spends the rest of the function undoing.
                continue
            # Not wanted any more.  Hiding an orphan leaves it alive for the
            # life of the process: every reopened pulse left its predecessor's
            # cards behind, six of them per load, still holding their text.
            widget.setParent(None)
            widget.deleteLater()
        self._cards = tuple(cards)
        for card in self._cards:
            self.watch_card_chrome(card)
            self.layout_main.addWidget(card)
        if bracket is not None:
            start = BracketPost("start", minimum=minimum_bracket)
            end = BracketPost(
                "end", count=bracket.count, minimum=minimum_bracket
            )
            payload = "\0".join(
                (
                    bracket.start_period_id,
                    bracket.end_period_id,
                    str(bracket.count),
                )
            )
            for post in (start, end):
                post.setProperty("zlcBracketPayload", payload)
                self.watch_bracket_chrome(post)
            end.count_committed.connect(
                lambda count,
                s=bracket.start_period_id,
                e=bracket.end_period_id: self.bracket_committed.emit(
                    s, e, int(count)
                )
            )
            start_index = next(
                (
                    i
                    for i, card in enumerate(self._cards)
                    if card.period_id == bracket.start_period_id
                ),
                0,
            )
            end_index = next(
                (
                    i
                    for i, card in enumerate(self._cards)
                    if card.period_id == bracket.end_period_id
                ),
                len(self._cards) - 1,
            )
            self.layout_main.insertWidget(start_index, start)
            self.layout_main.insertWidget(end_index + 2, end)
            self._posts = (start, end)
        else:
            self._posts = ()
        self.setMinimumSize(0, 0)
        self.adjustSize()
        # A rebuild drops every outline, and a selection that names a period
        # this strip no longer holds would aim the next edit at nothing.  A gap
        # keeps its meaning only while the count it counted into is unchanged.
        card = self._selected_card
        if card is not None and not any(item.period_id == card for item in self._cards):
            card = None
        post = self._selected_post
        if post is not None and not any(item.kind == post for item in self._posts):
            post = None
        gap = self._selected_gap
        if gap is not None and gap > len(self._cards):
            gap = None
        if card is not None:
            self.show_selection(card=card)
        elif post is not None:
            self.show_selection(post=post)
        else:
            self.show_selection(gap=gap)

    #: Widgets that need their own clicks: typing in them, ticking them and
    #: opening them ARE the click.  Everything else on a card is chrome, and a
    #: click on chrome belongs to the card.
    _INTERACTIVE = (
        QtWidgets.QAbstractButton,
        QtWidgets.QLineEdit,
        QtWidgets.QAbstractSpinBox,
        QtWidgets.QComboBox,
    )

    def watch_card_chrome(self, card: "PeriodCard") -> None:
        """Let a click anywhere that is not an editable control select the card.

        A card is almost entirely covered by its own children -- checkboxes,
        value boxes, a name field -- so a filter installed on the card alone
        saw a press only in the few pixels of padding around them.  Clicking a
        period therefore selected nothing at all, or fell through to the strip
        underneath and selected a GAP instead.
        """

        card.installEventFilter(self)
        for child in card.findChildren(QtWidgets.QWidget):
            self._watch_chrome(child)

    def watch_bracket_chrome(self, post: "BracketPost") -> None:
        """Make the post itself draggable while leaving its count editable."""

        post.installEventFilter(self)
        for child in post.findChildren(QtWidgets.QWidget):
            if not isinstance(child, self._INTERACTIVE):
                child.setCursor(QtCore.Qt.SizeHorCursor)
                child.installEventFilter(self)

    def _watch_chrome(self, widget: QtWidgets.QWidget) -> None:
        if not isinstance(widget, self._INTERACTIVE):
            widget.installEventFilter(self)

    def _card_of(self, widget: object) -> "PeriodCard | None":
        """Which card a watched widget belongs to, if any."""

        while isinstance(widget, QtWidgets.QWidget):
            if isinstance(widget, PeriodCard):
                return widget
            widget = widget.parentWidget()
        return None

    def _post_of(self, widget: object) -> "BracketPost | None":
        while isinstance(widget, QtWidgets.QWidget):
            if isinstance(widget, BracketPost):
                return widget
            widget = widget.parentWidget()
        return None

    def pulse_cards(self) -> tuple[PeriodCard, ...]:
        return self._cards

    # ------------------------------------------------------- where edits land
    def selection(self) -> tuple[str | None, str | None, int | None]:
        """(period id, bracket post kind, gap position).  At most one is set."""

        return self._selected_card, self._selected_post, self._selected_gap

    def show_selection(
        self,
        *,
        card: str | None = None,
        post: str | None = None,
        gap: int | None = None,
    ) -> None:
        """Mark the period, the bracket post, or the gap an edit will act on.

        A post is outlined exactly as a card is, by this same call: it is a
        FluentGroupBox too, so the mechanism has been there since the day it
        was written and simply never reached.  Clicking a card drew a border
        and clicking a post drew nothing, which made the two look like
        different kinds of thing when they are the same kind of thing --
        something you pick, and then drag if you want to.

        The outline goes on the widget's own outer edge through
        FluentGroupBox.set_outline; a stylesheet border set here would
        cascade into everything inside it.
        """

        chosen = tuple(
            name
            for name, value in (("card", card), ("post", post), ("gap", gap))
            if value is not None
        )
        if len(chosen) > 1:
            raise ValueError(
                f"only one of these can be selected: {', '.join(chosen)}"
            )
        self._selected_card = None if card is None else str(card)
        self._selected_post = None if post is None else str(post)
        self._selected_gap = None if gap is None else int(gap)
        for widget in self._cards:
            widget.set_outline(
                ACCENT if widget.period_id == self._selected_card else None
            )
        for widget in self._posts:
            widget.set_outline(ACCENT if widget.kind == self._selected_post else None)
        if self._selected_gap is None:
            self._indicator.hide()
        else:
            self._show_indicator_at_gap(self._selected_gap)

    def clear_selection(self) -> None:
        """Forget it.  Anything that changes the period LIST invalidates it."""

        self.show_selection()

    def _gap_at(self, x: int) -> int:
        """Which gap an empty-space click at ``x`` means, in period positions.

        0 is before the first card, len(cards) is after the last.
        """

        for index, card in enumerate(self._cards):
            if x < card.geometry().center().x():
                return index
        return len(self._cards)

    def _show_indicator_at_gap(self, position: int) -> None:
        if not self._cards:
            self._indicator.hide()
            return
        clamped = max(0, min(int(position), len(self._cards)))
        if clamped < len(self._cards):
            x = self._cards[clamped].geometry().left() - self.layout_main.spacing() // 2
        else:
            x = self._cards[-1].geometry().right() + self.layout_main.spacing() // 2
        self._indicator.setGeometry(x, 0, self._indicator.width(), self.height())
        self._indicator.raise_()
        self._indicator.show()

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt name
        """A click on empty strip selects the gap it fell in.

        Without this the signal existed and could never fire, so Add Period
        could only ever append: the operator had no way to say "here".
        """

        if event.button() == QtCore.Qt.LeftButton and not any(
            widget.geometry().contains(event.pos())
            for widget in self._cards + self._posts
        ):
            # A release inside something this row HOLDS belongs to that thing,
            # even when it let the release through; only the strip between
            # them is a gap.  Cards were excluded and posts were not, so
            # clicking a post picked the gap underneath it -- the click landed
            # somewhere the operator had not clicked.
            self.gap_clicked.emit(self._gap_at(event.pos().x()))
        super().mouseReleaseEvent(event)

    def request_move(self, period_id: str, before_period_id: str | None) -> None:
        self.move_period_requested.emit(str(period_id), before_period_id)

    #: What a dragged period carries.  Typed, so a drop from anywhere else is
    #: not mistaken for one of ours.
    CARD_MIME = "application/x-zlc-pulse-card"
    BRACKET_MIME = "application/x-zlc-pulse-bracket"

    def dragEnterEvent(self, event):  # noqa: N802 - Qt name
        if event.mimeData().hasFormat(self.CARD_MIME) or event.mimeData().hasFormat(
            self.BRACKET_MIME
        ):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def _proposal_at(self, data: QtCore.QMimeData, gap: int) -> object | None:
        """What dropping THIS payload in THIS gap would do, or None for nothing.

        One question, asked by the two events that must never disagree about
        it.  While the cursor moves it decides whether to offer the gap at
        all; when the button comes up it decides what to emit.  A card and a
        post are different payloads and the same gesture, so they differ in
        the payload and nowhere else.
        """

        if data.hasFormat(self.BRACKET_MIME):
            return self._bracket_proposal(data, gap)
        if data.hasFormat(self.CARD_MIME):
            return self._card_proposal(data, gap)
        return None

    def dragMoveEvent(self, event):  # noqa: N802 - Qt name
        """Offer only the gaps that would actually do something.

        Without this the whole gesture was invisible: the drop position was
        decided from where the button happened to come up, with nothing on
        screen to aim at, so a drag that clearly went somewhere did nothing.
        It arrived for the bracket post alone, and the card kept half of the
        old problem: it accepted every gap and showed the marker in all of
        them, including the two that mean "where it already is", and then
        discarded the drop in silence.  A gesture that says yes and does
        nothing is worse than one that says no.
        """

        if not (
            event.mimeData().hasFormat(self.CARD_MIME)
            or event.mimeData().hasFormat(self.BRACKET_MIME)
        ):
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
        if not (data.hasFormat(self.CARD_MIME) or data.hasFormat(self.BRACKET_MIME)):
            return super().dropEvent(event)
        proposal = self._proposal_at(data, self._gap_at(event.pos().x()))
        self._restore_selection()
        if proposal is None:
            event.ignore()
            return
        event.acceptProposedAction()
        if data.hasFormat(self.BRACKET_MIME):
            self.bracket_committed.emit(*proposal)
        else:
            self.move_period_requested.emit(*proposal)

    def _restore_selection(self) -> None:
        """Put back whatever was picked before the drag moved the marker."""

        self.show_selection(
            card=self._selected_card,
            post=self._selected_post,
            gap=self._selected_gap,
        )

    def _card_proposal(
        self, data: QtCore.QMimeData, gap: int
    ) -> tuple[str, str | None] | None:
        """Translate a card drop into one proposed move: what, and before what.

        ``None`` where the move would change nothing.  The gap just before a
        card and the gap just after it are both where that card already is,
        so proposing either is a rebuild of the whole board for no change.
        """

        period_id = bytes(data.data(self.CARD_MIME)).decode("utf-8")
        order = [card.period_id for card in self._cards]
        if period_id not in order:
            return None
        here = order.index(period_id)
        if gap in (here, here + 1):
            return None
        return period_id, (order[gap] if gap < len(order) else None)

    def _bracket_proposal(
        self, data: QtCore.QMimeData, gap: int
    ) -> tuple[str, str, int] | None:
        """Translate a post drop into one proposed inclusive bracket span."""

        parts = bytes(data.data(self.BRACKET_MIME)).decode("utf-8").split("\0")
        if len(parts) != 4:
            return None
        kind, start, end, count_text = parts
        order = [card.period_id for card in self._cards]
        if start not in order or end not in order:
            return None
        count = int(count_text)
        start_index = order.index(start)
        end_index = order.index(end)
        if kind == "start" and 0 <= gap <= end_index:
            candidate = (order[gap], end, count)
        elif kind == "end" and start_index < gap <= len(order):
            candidate = (start, order[gap - 1], count)
        else:
            return None
        return None if candidate == (start, end, count) else candidate

    def _begin_card_drag(self, period_id: str) -> None:
        """Carry one period under the cursor, Qt's own way.

        A QDrag is what makes the gesture real: the cursor changes, the widget
        under it is told, and the drop is a decision taken where the operator
        let go rather than inferred afterwards from two coordinates.
        """

        data = QtCore.QMimeData()
        data.setData(self.CARD_MIME, QtCore.QByteArray(str(period_id).encode("utf-8")))
        drag = QtGui.QDrag(self)
        drag.setMimeData(data)
        card = next((item for item in self._cards if item.period_id == period_id), None)
        if card is not None:
            drag.setPixmap(card.grab())
            drag.setHotSpot(QtCore.QPoint(card.width() // 2, 0))
        drag.exec_(QtCore.Qt.MoveAction)

    def _begin_bracket_drag(self, post: BracketPost) -> None:
        payload = str(post.property("zlcBracketPayload") or "")
        data = QtCore.QMimeData()
        data.setData(
            self.BRACKET_MIME,
            QtCore.QByteArray(f"{post.kind}\0{payload}".encode("utf-8")),
        )
        drag = QtGui.QDrag(self)
        drag.setMimeData(data)
        drag.setPixmap(post.grab())
        drag.setHotSpot(QtCore.QPoint(post.width() // 2, 0))
        drag.exec_(QtCore.Qt.MoveAction)

    def eventFilter(self, obj, event):  # noqa: N802
        card = self._card_of(obj)
        if card is not None:
            obj = card
            if (
                event.type() == QtCore.QEvent.MouseButtonPress
                and event.button() == QtCore.Qt.LeftButton
            ):
                self._pressed = ("card", obj.period_id, event.pos())
                self._dragging = False
            elif event.type() == QtCore.QEvent.MouseMove and self._pressed is not None:
                if not self._dragging and (
                    event.pos() - self._pressed[2]
                ).manhattanLength() >= QtWidgets.QApplication.startDragDistance():
                    self._dragging = True
                    period_id = self._pressed[1]
                    self._pressed = None
                    self._begin_card_drag(period_id)
                    return True
            elif (
                event.type() == QtCore.QEvent.MouseButtonRelease
                and event.button() == QtCore.Qt.LeftButton
            ):
                # A drag was carried out by the QDrag loop and ended there, so
                # a press that reaches release without one IS a click.
                if not self._dragging and self._pressed is not None:
                    self.period_clicked.emit(obj.period_id)
                self._pressed = None
                self._dragging = False
            return super().eventFilter(obj, event)

        post = self._post_of(obj)
        if post is not None:
            if (
                event.type() == QtCore.QEvent.MouseButtonPress
                and event.button() == QtCore.Qt.LeftButton
            ):
                self._pressed = ("bracket", post.kind, event.pos())
                self._dragging = False
            elif event.type() == QtCore.QEvent.MouseMove and self._pressed is not None:
                if not self._dragging and (
                    event.pos() - self._pressed[2]
                ).manhattanLength() >= QtWidgets.QApplication.startDragDistance():
                    self._dragging = True
                    self._pressed = None
                    self._begin_bracket_drag(post)
                    return True
            elif (
                event.type() == QtCore.QEvent.MouseButtonRelease
                and event.button() == QtCore.Qt.LeftButton
            ):
                # Same rule as a card's: a press that reaches release
                # without a drag IS a click, and a click picks the thing.
                if not self._dragging and self._pressed is not None:
                    self.bracket_clicked.emit(post.kind)
                self._pressed = None
                self._dragging = False
        return super().eventFilter(obj, event)


class PulseScheduleView(QtWidgets.QWidget):
    document_name_committed = QtCore.pyqtSignal(str)
    port_label_committed = QtCore.pyqtSignal(str, str)
    period_name_committed = QtCore.pyqtSignal(str, str)
    duration_committed = QtCore.pyqtSignal(str, object, str)
    digital_committed = QtCore.pyqtSignal(str, str, bool)
    analog_committed = QtCore.pyqtSignal(str, str, str, object)
    delay_committed = QtCore.pyqtSignal(str, object, str)
    binding_cycle_requested = QtCore.pyqtSignal(str, object, object)
    insert_period_requested = QtCore.pyqtSignal(object)
    move_period_requested = QtCore.pyqtSignal(str, object)
    remove_period_requested = QtCore.pyqtSignal(str)
    bracket_committed = QtCore.pyqtSignal(object, object, int)
    run_repeats_committed = QtCore.pyqtSignal(int)
    visible_ports_committed = QtCore.pyqtSignal(object)
    clear_port_requested = QtCore.pyqtSignal(str)
    run_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    sync_requested = QtCore.pyqtSignal()
    save_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()
    values_save_requested = QtCore.pyqtSignal()
    values_load_requested = QtCore.pyqtSignal()
    connection_requested = QtCore.pyqtSignal(str, str)
    scan_array_load_requested = QtCore.pyqtSignal()
    left_panels_collapsed = QtCore.pyqtSignal(bool)
    feedback_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._schedule: ScheduleVM | None = None
        self._connection: ConnectionVM | None = None
        self._version = (-1, -1)
        self._capabilities = {"can_sync": True, "can_hold": True, "can_step": True}
        self._cards: dict[str, PeriodCard] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(px(8, minimum=5), px(8, minimum=5), px(8, minimum=5), px(8, minimum=5))
        layout.setSpacing(px(8, minimum=5))

        # The document title and summary belong to PulseEditorView's single
        # top header.  Keep these two labels as non-owning presenter seams so
        # the schedule API remains stable, but do not render a second header.
        self.title_label = FluentLabel("")
        self.title_label.hide()
        self.summary_label = FluentLabel("")
        self.summary_label.hide()

        # The fixed-width operator columns and the timeline are two panes of
        # ONE scrolling body: their rows line up, so they move together, and
        # LinkedScrollPanes owns the single vertical bar that appears only
        # while some pane actually overflows.  Nothing here decides how far
        # the operator can reach -- the taller pane's own range does.
        dataset_frame = LinkedScrollPanes()
        gutter = card_gutter()

        self.left_scroll = FluentScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.left_scroll.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.left_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Expanding,
        )
        left_body = QtWidgets.QWidget()
        left = QtWidgets.QHBoxLayout(left_body)
        left.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)

        self.names_panel_holder = QtWidgets.QWidget()
        names_holder_layout = QtWidgets.QVBoxLayout(self.names_panel_holder)
        names_holder_layout.setContentsMargins(gutter, gutter, gutter, gutter)
        names_holder_layout.setSpacing(0)
        self.names_panel = ChannelNamesPanel()
        names_holder_layout.addWidget(self.names_panel)
        left.addWidget(self.names_panel_holder)

        self.channel_panel_holder = QtWidgets.QWidget()
        channel_holder_layout = QtWidgets.QVBoxLayout(self.channel_panel_holder)
        channel_holder_layout.setContentsMargins(gutter, gutter, gutter, gutter)
        channel_holder_layout.setSpacing(0)
        self.channel_panel = ChannelPanel()
        channel_holder_layout.addWidget(self.channel_panel)
        left.addWidget(self.channel_panel_holder)

        self.left_panel_stub_holder = QtWidgets.QWidget()
        stub_holder_layout = QtWidgets.QVBoxLayout(self.left_panel_stub_holder)
        stub_holder_layout.setContentsMargins(gutter, gutter, gutter, gutter)
        stub_holder_layout.setSpacing(0)
        self.left_panel_stub = FluentFrame()
        self.left_panel_stub.setFixedWidth(px(82, minimum=68))
        stub_layout = QtWidgets.QVBoxLayout(self.left_panel_stub)
        stub_layout.setContentsMargins(px(6), px(8), px(6), px(8))
        stub_layout.setSpacing(px(6, minimum=4))
        stub_label = FluentLabel("Name\nDelay")
        stub_label.setAlignment(QtCore.Qt.AlignCenter)
        stub_layout.addWidget(stub_label)
        self.stub_show_button = FluentButton("Show", color=ACCENT)
        self.stub_show_button.setFixedHeight(row_height())
        self.stub_show_button.clicked.connect(self._show_left_panels)
        stub_layout.addWidget(self.stub_show_button)
        stub_layout.addStretch(1)
        stub_holder_layout.addWidget(self.left_panel_stub)
        self.left_panel_stub_holder.hide()
        left.addWidget(self.left_panel_stub_holder)

        self.left_scroll.setWidget(left_body)
        dataset_frame.add_pane(self.left_scroll)

        self.timeline_scroll = FluentScrollArea()
        self.timeline_scroll.setWidgetResizable(False)
        self.timeline_scroll.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.timeline_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.timeline_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.drag_container = PulseDragContainer()
        self.drag_container.period_clicked.connect(self._period_clicked)
        self.drag_container.bracket_clicked.connect(self._bracket_clicked)
        self.drag_container.gap_clicked.connect(self._gap_clicked)
        self.timeline_scroll.setWidget(self.drag_container)
        dataset_frame.add_pane(self.timeline_scroll, stretch=1)
        layout.addWidget(dataset_frame, 1)
        self.dataset_panes = dataset_frame

        # Bottom bar: three titled cards.  Control keeps its compact button
        # grid and the Pulse's one complete-run count; Connection and Ports
        # retain fixed widths so the timeline gets the remaining space.
        self.button_frame = FluentFrame(bordered=False)
        self.button_frame.setObjectName("zlcPulseButtonBar")
        self.button_frame.setStyleSheet(
            "QFrame#zlcPulseButtonBar { background: transparent; border: none; }"
        )
        bar = QtWidgets.QHBoxLayout(self.button_frame)
        bar.setContentsMargins(gutter, gutter + px(2), gutter, gutter)
        bar.setSpacing(px(10, minimum=8))
        control_height = px(30, minimum=26)
        control_area = FluentGroupBox("Control")
        control_layout = QtWidgets.QVBoxLayout(control_area)
        control_layout.setContentsMargins(px(8), px(2), px(8), px(6))
        control_layout.setSpacing(px(4, minimum=3))
        controls = QtWidgets.QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(px(6, minimum=4))
        self.run_button = FluentButton("On Pulse*", color=GREEN)
        self.stop_button = FluentButton("Stop Pulse", color=RED)
        self.sync_button = FluentButton("Sync", color=ORANGE)
        self.add_button = FluentButton("Add Period", color=ACCENT)
        self.remove_button = FluentButton("Remove", color=ORANGE)
        self.bracket_button = FluentButton("Add Bracket", color=ACCENT)
        self.save_button = FluentButton("Save*", color=YELLOW)
        self.load_button = FluentButton("Load", color=ORANGE)
        self.collapse_button = FluentButton("Collapse", color=GREY)
        # The board's calibrated numbers -- a channel delay, a DAC bias --
        # which it then fills into every pulse it plays.  They belong to the
        # apparatus, so these carry the whole set on and off the BOARD rather
        # than editing the document that is open.
        self.load_values_button = FluentButton("Load config", color=ORANGE)
        self.save_values_button = FluentButton("Save config", color=YELLOW)
        self.load_values_button.setToolTip(
            "Give the connected board a set of calibrated values.  It holds "
            "them until the next load and fills them into every pulse it "
            "compiles, including ones fired from outside this window."
        )
        self.save_values_button.setToolTip(
            "Write what the connected board is holding now as a set it can be "
            "given again."
        )
        control_buttons = (
            self.run_button, self.stop_button, self.sync_button,
            self.add_button, self.remove_button, self.bracket_button,
            self.save_button, self.load_button, self.collapse_button,
            self.load_values_button, self.save_values_button,
        )
        for index, button in enumerate(control_buttons):
            button.setFixedHeight(control_height)
            button.setMinimumWidth(px(74, minimum=62))
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            controls.addWidget(button, index // 3, index % 3)
        for column in range(3):
            controls.setColumnStretch(column, 1)
        control_layout.addStretch(1)
        control_layout.addLayout(controls)
        control_layout.addStretch(1)
        bar.addWidget(control_area, 1)

        connection_area = FluentGroupBox("Connection")
        connection_area.setFixedWidth(px(252, minimum=212))
        connection_layout = QtWidgets.QVBoxLayout(connection_area)
        connection_layout.setContentsMargins(px(8), px(2), px(8), px(6))
        connection_layout.setSpacing(px(4, minimum=3))
        self.connection_combo = FluentComboBox()
        self.connection_combo.setFixedHeight(control_height)
        # The presenter projects the complete endpoint value before the window
        # is shown.  This widget neither seeds nor preserves a second copy.
        self.connection_endpoint = FluentLineEdit("")
        self.connection_endpoint.setToolTip(
            "Endpoint for a connection choice that accepts one, as host:port."
        )
        self.connection_combo.setToolTip(
            "Connection choices are supplied by the host that opened this editor."
        )
        self.connection_endpoint.setFixedHeight(control_height)
        self.connection_button = FluentButton("Connect", color=ACCENT)
        self.connection_button.setFixedHeight(control_height)
        self.connection_status = FluentLineEdit("")
        self.connection_status.setEnabled(False)
        self.connection_status.setFixedHeight(row_height())
        # Which calibrated set the board is filling config parameters from.
        # It belongs here rather than beside the pulse's own fields: it is a
        # fact about the board, and it outlives whatever pulse is open.
        self.config_source_line = FluentLineEdit("")
        self.config_source_line.setEnabled(False)
        self.config_source_line.setFixedHeight(row_height())
        connection_layout.addWidget(self.connection_combo)
        connection_row = QtWidgets.QHBoxLayout()
        connection_row.setContentsMargins(0, 0, 0, 0)
        connection_row.setSpacing(px(6, minimum=4))
        connection_row.addWidget(self.connection_endpoint, 1)
        connection_row.addWidget(self.connection_button)
        connection_layout.addLayout(connection_row)
        connection_layout.addWidget(self.connection_status)
        connection_layout.addWidget(self.config_source_line)
        connection_layout.addStretch(1)
        bar.addWidget(connection_area)

        ports_area = FluentGroupBox("Ports")
        ports_area.setFixedWidth(px(286, minimum=246))
        ports_layout = QtWidgets.QVBoxLayout(ports_area)
        ports_layout.setContentsMargins(px(8), px(2), px(8), px(6))
        ports_layout.setSpacing(px(4, minimum=3))
        self.hidden_port_combo = FluentComboBox()
        self.hidden_port_combo.setFixedHeight(control_height)
        self.add_port_button = FluentButton("Add", color=ACCENT)
        self.hide_off_button = FluentButton("Hide Off", color=ORANGE)
        self.show_all_button = FluentButton("Show All", color=ACCENT)
        self.visible_label = FluentLineEdit("")
        self.visible_label.setEnabled(False)
        for button in (self.add_port_button, self.hide_off_button, self.show_all_button):
            button.setFixedHeight(control_height)
        self.visible_label.setFixedHeight(row_height())
        ports_layout.addWidget(self.hidden_port_combo)
        ports_row = QtWidgets.QHBoxLayout()
        ports_row.setContentsMargins(0, 0, 0, 0)
        ports_row.setSpacing(px(6, minimum=4))
        for button in (self.add_port_button, self.hide_off_button, self.show_all_button):
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            ports_row.addWidget(button, 1)
        ports_layout.addLayout(ports_row)
        ports_layout.addWidget(self.visible_label)
        ports_layout.addStretch(1)
        bar.addWidget(ports_area)
        self.button_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        layout.addWidget(self.button_frame)

        self._wire_signals()

    def _wire_signals(self) -> None:
        self.names_panel.document_name_committed.connect(self.document_name_committed)
        self.names_panel.port_label_committed.connect(self.port_label_committed)
        self.channel_panel.delay_committed.connect(self.delay_committed)
        self.channel_panel.binding_cycle_requested.connect(self.binding_cycle_requested)
        self.channel_panel.clear_port_requested.connect(self.clear_port_requested)
        self.channel_panel.scan_array_load_requested.connect(self.scan_array_load_requested)
        self.channel_panel.run_repeats_committed.connect(self.run_repeats_committed)
        self.drag_container.move_period_requested.connect(self.move_period_requested)
        self.drag_container.bracket_committed.connect(self.bracket_committed)
        self.run_button.clicked.connect(self.run_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.sync_button.clicked.connect(self.sync_requested)
        self.save_button.clicked.connect(self.save_requested)
        self.load_button.clicked.connect(self.load_requested)
        self.load_values_button.clicked.connect(self.values_load_requested)
        self.save_values_button.clicked.connect(self.values_save_requested)
        self.add_button.clicked.connect(lambda: self.insert_period_requested.emit(self._selected_before_id()))
        self.remove_button.clicked.connect(self._request_remove_period)
        self.bracket_button.clicked.connect(self._request_toggle_bracket)
        self.collapse_button.clicked.connect(self._toggle_left_panels)
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

    @contextmanager
    def _kept_scroll(self):
        """Rebuild without throwing the operator back to the top.

        Every schedule update tears the period cards out of their layout and
        re-adds them, which resets each scroll area to zero.  So pressing On
        Pulse -- or editing anything at all -- jumped the board back to the
        first period and the leftmost time, losing the place of anyone working
        on period nine.

        The position is restored clamped to the new range: a rebuild that
        removed a period cannot restore a scroll past the end.  It is restored
        twice -- now, and once the layout has settled -- because a scroll bar's
        range is recomputed when the rebuilt content is laid out, which is a
        turn later, and restoring only now would clamp against a stale range of
        zero.
        """

        bars = tuple(
            widget
            for area in (self.timeline_scroll, self.left_scroll)
            for widget in (area.horizontalScrollBar(), area.verticalScrollBar())
        ) + (self.dataset_panes.scrollbar,)
        before = tuple(bar.value() for bar in bars)

        def _restore() -> None:
            for bar, value in zip(bars, before):
                if value:
                    bar.setValue(min(int(value), bar.maximum()))

        try:
            yield
        finally:
            _restore()
            if any(before):
                QtCore.QTimer.singleShot(0, _restore)

    @staticmethod
    def _visible_delay_rows(vm: ScheduleVM) -> tuple[DelayRowVM, ...]:
        """The delay column shows exactly the rows the cards show.

        Which ports have rows is ONE fact, and it was written down twice: the
        names column and every period card filter on ``port.visible``, while
        the delay rows arrive as "every output the board can delay" and were
        handed over whole.  So Hide Off dropped a row from the cards and left
        its delay sitting there, lined up with nothing -- the columns are read
        across, and a delay beside the wrong channel is worse than no delay.

        Taken here, once, because both paths that fill the column pass through
        it: the full rebuild and the single-row push.
        """

        visible = {port.key for port in vm.ports if port.visible}
        return tuple(row for row in vm.delay_rows if row.port_key in visible)

    def _reconcile(self, vm: ScheduleVM) -> None:
        """Rebuild the page from one model, keeping the operator's place.

        The scroll keeper wraps THIS rather than its callers: rebuilding is
        what resets the scroll areas, so whoever rebuilds is who has to
        restore.  It used to sit around one call site, so the value-level
        setters -- set_visible_ports among them -- threw the board back to the
        top of an 22-row list every time.
        """

        with self._kept_scroll():
            self._reconcile_now(vm)

    def _reconcile_now(self, vm: ScheduleVM) -> None:
        self.title_label.setText(vm.document_name)
        self.summary_label.setText(vm.summary_text)
        visible_count = sum(1 for port in vm.ports if port.visible)
        self.visible_label.setText(
            f"Visible {visible_count}/{len(vm.ports)} ports | "
            f"Hidden {max(0, len(vm.ports) - visible_count)}"
        )
        self.names_panel.set_ports(vm.document_name, vm.ports)
        self.names_panel.set_summary(vm.total_text, len(vm.periods), vm.visible_text)
        self.channel_panel.set_delay_rows(
            self._visible_delay_rows(vm),
            {port.key: port.label for port in vm.ports},
        )
        self.channel_panel.set_clock(vm.clock_text)
        self.channel_panel.set_scan_summary(vm.scan_summary_text)
        desired: dict[str, PeriodCard] = {}
        for index, period in enumerate(vm.periods):
            card = self._cards.get(period.period_id)
            if card is None:
                card = PeriodCard(
                    period,
                    index=index,
                    total_periods=len(vm.periods),
                    ports=vm.ports,
                    analog_mode_choices=vm.analog_mode_choices,
                )
                self._connect_card(card)
            else:
                card.set_period(
                    period,
                    index=index,
                    total_periods=len(vm.periods),
                    ports=vm.ports,
                    analog_mode_choices=vm.analog_mode_choices,
                )
            desired[period.period_id] = card
        for key, card in self._cards.items():
            if key not in desired:
                card.setParent(None)
                card.deleteLater()
        self._cards = desired
        self.drag_container.set_items(
            tuple(desired[p.period_id] for p in vm.periods),
            vm.bracket,
            minimum_bracket=vm.min_bracket_count,
        )
        with signals_blocked(self.channel_panel.run_repeats_spin):
            self.channel_panel.run_repeats_spin.setValue(float(vm.run_repeats))
        self.channel_panel.run_repeats_spin.setEnabled(bool(vm.periods))
        self._rebuild_hidden_ports(vm)
        self.bracket_button.setText(
            "Del Bracket" if vm.bracket is not None else "Add Bracket"
        )

    def _connect_card(self, card: PeriodCard) -> None:
        card.period_name_committed.connect(self.period_name_committed)
        card.duration_committed.connect(self.duration_committed)
        card.digital_committed.connect(self.digital_committed)
        card.analog_committed.connect(self.analog_committed)
        card.binding_cycle_requested.connect(self.binding_cycle_requested)

    def _rebuild_hidden_ports(self, vm: ScheduleVM) -> None:
        visible = {port.key for port in vm.ports if port.visible}
        hidden = [port for port in vm.ports if port.key not in visible]
        with signals_blocked(self.hidden_port_combo):
            self.hidden_port_combo.clear()
            for port in hidden:
                self.hidden_port_combo.addItem(port.label, port.key)

    def set_period(self, period: PeriodVM) -> None:
        if self._schedule is None or period.period_id not in self._cards:
            return
        periods = tuple(
            period if item.period_id == period.period_id else item
            for item in self._schedule.periods
        )
        self._schedule = replace(self._schedule, periods=periods)
        card = self._cards[period.period_id]
        card.set_period(
            period,
            index=next(i for i, item in enumerate(periods) if item.period_id == period.period_id),
            total_periods=len(periods),
            ports=self._schedule.ports,
            analog_mode_choices=self._schedule.analog_mode_choices,
        )
        # That call rebuilds any port row whose shape changed, and a row built
        # after the card was adopted is one the strip has never watched -- so a
        # click on it would select nothing.  Re-walked here, where the rebuild
        # is known to have happened, rather than guessed at from ChildAdded:
        # a child is not fully constructed when that fires, and touching it
        # there segfaults.
        self.drag_container.watch_card_chrome(card)

    def set_delay_row(self, row: DelayRowVM) -> None:
        if self._schedule is None:
            return
        rows = tuple(row if item.port_key == row.port_key else item for item in self._schedule.delay_rows)
        self._schedule = replace(self._schedule, delay_rows=rows)
        self.channel_panel.set_delay_rows(
            self._visible_delay_rows(self._schedule),
            {port.key: port.label for port in self._schedule.ports},
        )

    def set_port_label(self, key: str, label: str) -> None:
        self.names_panel.set_port_label(key, label)
        self.channel_panel.set_port_label(key, label)
        for card in self._cards.values():
            card.set_port_label(key, label)

    def set_visible_ports(self, ports: tuple[str, ...]) -> None:
        if self._schedule is None:
            return
        visible = set(str(key) for key in ports)
        updated = tuple(replace(port, visible=port.key in visible) for port in self._schedule.ports)
        self._schedule = replace(self._schedule, ports=updated, visible_text=f"{len(visible)}/{len(updated)}")
        self._reconcile(self._schedule)

    def set_summary(self, total_text: str, total_tooltip: str, period_count: int, visible_text: str, summary_text: str, scan_summary_text: str) -> None:
        if self._schedule is None:
            return
        self._schedule = replace(self._schedule, total_text=str(total_text), total_tooltip=str(total_tooltip), period_count=int(period_count), visible_text=str(visible_text), summary_text=str(summary_text), scan_summary_text=str(scan_summary_text))
        self.title_label.setToolTip(str(total_tooltip))
        self.summary_label.setText(str(summary_text))
        visible_count = sum(1 for port in self._schedule.ports if port.visible)
        self.visible_label.setText(
            f"Visible {visible_count}/{len(self._schedule.ports)} ports | "
            f"Hidden {max(0, len(self._schedule.ports) - visible_count)}"
        )
        self.names_panel.set_summary(str(total_text), int(period_count), str(visible_text))
        self.channel_panel.set_scan_summary(str(scan_summary_text))

    def set_scan_busy(self, busy: bool) -> None:
        self.channel_panel.load_array_button.setEnabled(not bool(busy))

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
        self.connection_combo.setEnabled(not vm.locked)
        self.connection_button.setEnabled(not vm.locked)
        self._sync_endpoint_enabled()
        self.connection_status.setText(vm.status)
        held = vm.config_source
        self.config_source_line.setText(
            f"Config: {Path(held).name}" if held else "Config: none loaded"
        )
        # The whole path, because the box is 252px and a calibration set is
        # identified by which folder it came from as much as by its name.
        self.config_source_line.setToolTip(held or "no config values loaded")
        self.connection_status.setToolTip(vm.status)

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

        Neither flag has a default on purpose.  Enablement used to follow only
        from ``running``, so an editor with no pulse open and no board attached
        still offered On Pulse -- a control that looks available and cannot be
        is the failure this shell keeps being audited for, and a default would
        let the next caller reintroduce it silently.

        On Pulse is NOT gated on running, and the correction matters: On Pulse
        means "put this on the board and play it", which is off-then-on.  It is
        how an operator applies an edit to a pulse that is already running --
        the single most frequent thing done at the bench -- and disabling it
        while running removed exactly that.  What running changes is the
        LABEL: the asterisk says the board is playing something other than
        what is on screen.

        Stop is gated on a board alone.  Going safe must work from any state,
        including one this window has misread, so it does not additionally
        require a pulse to be open or the board to look busy.
        """

        self.run_button.setEnabled(bool(can_run))
        self.run_button.setText(
            "On Pulse" if bool(running) and bool(synchronized) else "On Pulse*"
        )
        self.stop_button.setEnabled(bool(can_stop))
        self.sync_button.setEnabled(bool(can_run) and self._capabilities["can_sync"])
        self.save_button.set_dirty(bool(file_dirty))

    def set_capabilities(self, can_sync: bool, can_hold: bool, can_step: bool) -> None:
        """Apply presenter-advertised command availability without probing a controller."""

        self._capabilities = {
            "can_sync": bool(can_sync),
            "can_hold": bool(can_hold),
            "can_step": bool(can_step),
        }
        # Its own answer, not Run's.  Sync READS the board, so what it needs
        # is a board; ANDing it with Run meant it also needed a pulse, which
        # is exactly the case where reading one back is worth doing.
        self.sync_button.setEnabled(bool(can_sync))

    def _period_clicked(self, period_id: str) -> None:
        """Select it, or clear it if it was already the selected one."""

        current, _post, _gap = self.drag_container.selection()
        self.drag_container.show_selection(
            card=None if current == str(period_id) else str(period_id)
        )

    def _bracket_clicked(self, kind: str) -> None:
        """Same rule as a period's, because it is the same kind of act."""

        _card, current, _gap = self.drag_container.selection()
        self.drag_container.show_selection(
            post=None if current == str(kind) else str(kind)
        )

    def _gap_clicked(self, position: int) -> None:
        _card, _post, current = self.drag_container.selection()
        self.drag_container.show_selection(
            gap=None if current == int(position) else int(position)
        )

    def _selected_before_id(self) -> str | None:
        """Which period the next Add goes BEFORE, or None to append.

        A selected gap inserts there, a selected card inserts AFTER it, and
        with neither the new period is appended.  This returned None
        unconditionally, so a
        period could only ever be added to the end.
        """

        periods = self._schedule.periods if self._schedule else ()
        card, _post, gap = self.drag_container.selection()
        if gap is not None:
            return periods[gap].period_id if gap < len(periods) else None
        if card is not None:
            after = next(
                (index for index, item in enumerate(periods) if item.period_id == card),
                None,
            )
            if after is not None and after + 1 < len(periods):
                return periods[after + 1].period_id
        return None

    def _request_remove_period(self) -> None:
        """Remove the selected period, or the last one when none is selected."""

        if not (self._schedule and self._schedule.periods):
            return
        card, _post, _gap = self.drag_container.selection()
        target = card if card is not None else self._schedule.periods[-1].period_id
        self.remove_period_requested.emit(target)

    def _request_toggle_bracket(self) -> None:
        """Bracket the whole pulse, or take the bracket off."""

        if self._schedule is None:
            return
        if self._schedule.bracket is not None:
            self.bracket_committed.emit(None, None, 0)
            return
        if not self._schedule.periods:
            return
        self.bracket_committed.emit(
            self._schedule.periods[0].period_id,
            self._schedule.periods[-1].period_id,
            self._schedule.default_bracket_count,
        )

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
        stale for every edit in between -- and "Hide Off" reads exactly it.
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
                "nothing is driven yet, so there is nothing to hide"
            )
            return
        self.visible_ports_committed.emit(
            tuple(port.key for port in self._schedule.ports if port.key in driven)
        )

    def _request_show_all_ports(self) -> None:
        if self._schedule is not None:
            self.visible_ports_committed.emit(tuple(port.key for port in self._schedule.ports))

    def _toggle_left_panels(self) -> None:
        visible = self.names_panel_holder.isVisible()
        self.names_panel_holder.setVisible(not visible)
        self.channel_panel_holder.setVisible(not visible)
        self.left_panel_stub_holder.setVisible(visible)
        self.collapse_button.setText("Show Left" if visible else "Collapse")
        self.left_panels_collapsed.emit(visible)

    def _show_left_panels(self) -> None:
        self.names_panel_holder.show()
        self.channel_panel_holder.show()
        self.left_panel_stub_holder.hide()
        self.collapse_button.setText("Collapse")
        self.left_panels_collapsed.emit(False)


__all__ = [
    "ChannelNamesPanel", "ChannelPanel", "PeriodCard", "PulseDragContainer",
    "PulseScheduleView", "BracketPost",
]
