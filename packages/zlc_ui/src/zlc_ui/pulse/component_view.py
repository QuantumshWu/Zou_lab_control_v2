"""Subpulse editing with the same Period/Bracket surface as a full Pulse."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT, GREY, YELLOW, ElidedLabel, FluentButton, FluentComboBox,
    FluentFrame, FluentGroupBox, FluentLabel, FluentLineEdit, FluentScrollArea,
    read_editable_combo, retire_widget, signals_blocked,
)

from ._layout import px, row_height
from .models import ScheduleVM
from .scan_line_edit import FluentScanLineEdit
from .schedule_view import PulseScheduleView, _apply_field


class PulseComponentView(QtWidgets.QWidget):
    """Only editor intents and view records cross this page's boundary."""

    action_requested = QtCore.pyqtSignal(str, object)
    edit_requested = QtCore.pyqtSignal(str, object)
    feedback_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, *, compact: bool = False) -> None:
        super().__init__(parent)
        self._compact = bool(compact)
        self._context_id = ""
        self._accepted_name = ""
        self._binding_rows: dict[str, tuple] = {}
        self._parameter_rows: dict[tuple[str, str, str], tuple] = {}
        self._compact_schedule: ScheduleVM | None = None
        self.schedule_view: PulseScheduleView | None = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(px(8), px(8), px(8), px(8))
        outer.setSpacing(px(8))
        header = FluentFrame()
        head = QtWidgets.QVBoxLayout(header)
        head.setContentsMargins(px(12), px(10), px(12), px(10))
        top = QtWidgets.QHBoxLayout()
        top.addWidget(FluentLabel("Component parameters" if compact else "Subpulse file"))
        self.context_combo = FluentComboBox()
        self.context_combo.setMinimumWidth(px(130 if compact else 190))
        self.context_combo.currentIndexChanged.connect(self._context_changed)
        top.addWidget(self.context_combo, 1)
        self.context_combo.hide()
        self.name_edit = FluentLineEdit("")
        self.name_edit.setPlaceholderText("Component name")
        self.name_edit.editingFinished.connect(self._name_committed)
        if compact:
            head.addLayout(top)
            head.addWidget(self.name_edit)
        else:
            top.addWidget(self.name_edit, 1)
        self._buttons = {}
        actions = (
            ("expand", "Expand", GREY),
            ("insert", "Duplicate", ACCENT), ("export", "Export", GREY),
        ) if compact else (
            ("new", "New", GREY), ("open", "Open", ACCENT),
            ("save", "Save Subpulse", YELLOW), ("save_as", "Save as", YELLOW),
            ("insert", "Insert into Pulse", ACCENT),
        )
        if compact:
            top = QtWidgets.QHBoxLayout()
        for action, text, color in actions:
            button = FluentButton(text, color=color)
            button.clicked.connect(lambda _checked=False, a=action: self.action_requested.emit(a, self._context_id))
            self._buttons[action] = button
            top.addWidget(button)
        head.addLayout(top)
        self.path_label = ElidedLabel("")
        head.addWidget(self.path_label)
        self.scope_label = FluentLabel("")
        self.scope_label.setWordWrap(True)
        head.addWidget(self.scope_label)
        outer.addWidget(header)
        self.placeholder = FluentLabel(
            "Select a component in the timeline to edit its parameters."
            if compact else "Create or open a reusable Subpulse file."
        )
        self.placeholder.setWordWrap(True)
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
        outer.addWidget(self.placeholder, 1)
        if compact:
            self._parameter_scroll = FluentScrollArea()
            body = QtWidgets.QWidget()
            self._parameter_layout = QtWidgets.QVBoxLayout(body)
            self._parameter_layout.setContentsMargins(0, 0, 0, 0)
            self._parameter_layout.setSpacing(px(8))
            self._parameter_layout.setAlignment(QtCore.Qt.AlignTop)
            self._parameter_scroll.set_width_bounded_widget(body)
            outer.addWidget(self._parameter_scroll, 1)
            self._parameter_scroll.hide()
        self.bindings_box = FluentGroupBox("Component Config references")
        bindings_layout = QtWidgets.QVBoxLayout(self.bindings_box)
        bindings_layout.setContentsMargins(px(10), px(8), px(10), px(10))
        scroll = FluentScrollArea()
        scroll.setMaximumHeight(px(180))
        self._bindings_scroll = scroll
        body = QtWidgets.QWidget()
        self._binding_layout = QtWidgets.QGridLayout(body)
        self._binding_layout.setContentsMargins(0, 0, 0, 0)
        self._binding_layout.setColumnStretch(0, 2)
        self._binding_layout.setColumnStretch(1, 1)
        self._binding_layout.setAlignment(QtCore.Qt.AlignTop)
        scroll.set_width_bounded_widget(body)
        bindings_layout.addWidget(scroll)
        outer.addWidget(self.bindings_box)
        self.bindings_box.hide()

    def _context_changed(self) -> None:
        key = self.context_combo.currentData()
        if key is not None and key != self._context_id:
            self.action_requested.emit("context", str(key))

    def _name_committed(self) -> None:
        name = self.name_edit.text().strip()
        if name != self._accepted_name:
            self.action_requested.emit("rename", (self._context_id, name))

    def _make_schedule(self) -> PulseScheduleView:
        schedule = PulseScheduleView(embedded=True)
        schedule.setObjectName("componentSchedule")
        for action, signal in (
            ("period_name", "period_name_committed"), ("duration", "duration_committed"),
            ("digital", "digital_committed"), ("analog", "analog_committed"),
            ("binding", "binding_committed"), ("insert_period", "insert_period_requested"),
            ("insert_spacer", "insert_spacer_requested"), ("reorder_items", "reorder_items_requested"),
            ("remove_items", "remove_items_requested"), ("bracket", "bracket_committed"),
            ("bracket_add", "bracket_add_requested"),
            ("visible_ports", "visible_ports_committed"),
            ("config_binding", "config_binding_committed"),
        ):
            getattr(schedule, signal).connect(lambda *args, a=action: self.edit_requested.emit(a, args))
        schedule.feedback_requested.connect(self.feedback_requested)
        self.layout().insertWidget(1, schedule, 1)
        return schedule

    def _parameter_fields(self) -> tuple:
        schedule = self._compact_schedule
        if schedule is None:
            return ()
        labels = {port.key: port.label or port.key for port in schedule.ports}
        rows = []
        for period in schedule.periods:
            name = period.name or period.period_id
            rows.append((("duration", period.period_id, ""), f"{name} · Duration",
                         period.duration, period.unit, period.unit_choices))
            for port, mode, field in period.analog:
                if mode != "hold":
                    rows.append((("analog", period.period_id, port),
                                 f"{name} · {labels.get(port, port)}", field, mode, ()))
        for bracket in schedule.brackets:
            if bracket.count_field is not None:
                rows.append((("bracket_count", bracket.bracket_id, ""),
                             f"Bracket {bracket.ordinal} · Repeats", bracket.count_field, "count", ()))
        return tuple(rows)

    def _set_parameters(self, schedule: ScheduleVM | None, busy: bool) -> None:
        self._compact_schedule = schedule
        fields = self._parameter_fields()
        wanted = {key for key, *_rest in fields}
        for key in tuple(self._parameter_rows):
            if key not in wanted:
                holder = self._parameter_rows.pop(key)[0]
                self._parameter_layout.removeWidget(holder)
                retire_widget(holder)
        for position, (key, text, field, unit, choices) in enumerate(fields):
            widgets = self._parameter_rows.get(key)
            if widgets is None:
                holder = FluentFrame()
                column = QtWidgets.QVBoxLayout(holder)
                column.setContentsMargins(px(10), px(8), px(10), px(8))
                column.setSpacing(px(4))
                label = FluentLabel("")
                label.setWordWrap(True)
                column.addWidget(label)
                line = QtWidgets.QHBoxLayout()
                edit = FluentScanLineEdit()
                edit.setMinimumWidth(px(90))
                edit.setFixedHeight(row_height())
                combo = FluentComboBox()
                combo.setMinimumWidth(px(62))
                line.addWidget(edit, 1)
                line.addWidget(combo)
                column.addLayout(line)
                detail = FluentLabel("")
                detail.setWordWrap(True)
                column.addWidget(detail)
                edit.editingFinished.connect(lambda k=key: self._commit_parameter(k))
                edit.valueNormalized.connect(lambda k=key: self._commit_parameter(k))
                combo.currentIndexChanged.connect(lambda _index, k=key: self._commit_parameter(k))
                edit.binding_committed.connect(lambda scan, source, k=key: self.edit_requested.emit(
                    "binding", (k[0], k[1], k[2] or None, scan, source),
                ))
                edit.config_key_committed.connect(lambda name, control=edit: self.edit_requested.emit(
                    "config_binding", (control.property("pulse_field_id"), name),
                ))
                widgets = self._parameter_rows[key] = (holder, label, edit, combo, detail)
            holder, label, edit, combo, detail = widgets
            label.setText(text)
            _apply_field(edit, field)
            with signals_blocked(combo):
                offered = choices or (unit,)
                if tuple(combo.itemText(index) for index in range(combo.count())) != offered:
                    combo.clear()
                    combo.addItems(offered)
                combo.setCurrentText(unit)
                combo.setEnabled(bool(choices) and field.editable and not busy)
            edit.setEnabled(field.editable and not busy)
            source = "Using Pulse default" if field.source == "default" else "API may override the default"
            if field.source == "config":
                source = f"Config: {field.config_key or 'unassigned'}"
            if field.effective_text:
                source += f" → {field.effective_text}"
            elif field.source == "config":
                source += " · using default"
            if field.scan:
                source = f"Scan + {source}"
            label.setText(text + " · default")
            detail.setText(source)
            detail.setToolTip(field.source_text)
            edit.setToolTip(field.source_text)
            if self._parameter_layout.indexOf(holder) != position:
                self._parameter_layout.removeWidget(holder)
                self._parameter_layout.insertWidget(position, holder)
        self._parameter_scroll.setVisible(schedule is not None)

    def _commit_parameter(self, key: tuple[str, str, str]) -> None:
        _holder, _label, edit, combo, _detail = self._parameter_rows[key]
        if not edit.isEnabled() or self._compact_schedule is None:
            return
        error = edit.property("numericError")
        if error:
            self.feedback_requested.emit(str(error))
            return
        field = next(value for row_key, _label, value, _unit, _choices
                     in self._parameter_fields() if row_key == key)
        try:
            value = float(edit.text()) if key[0] == "duration" else int(edit.text())
            accepted = float(field.text) if key[0] == "duration" else int(field.text)
        except ValueError:
            return
        if key[0] == "duration":
            period = next(period for period in self._compact_schedule.periods if period.period_id == key[1])
            unit = combo.currentText()
            if (value, unit) != (accepted, period.unit):
                self.edit_requested.emit("duration", (key[1], value, unit))
        elif key[0] == "analog":
            if value != accepted:
                self.edit_requested.emit("analog", (key[1], key[2], combo.currentText(), value))
        elif value != accepted:
            bracket = next(item for item in self._compact_schedule.brackets if item.bracket_id == key[1])
            self.edit_requested.emit("bracket", (
                key[1], bracket.start_period_id, bracket.end_period_id, value,
            ))

    def set_document(
        self,
        schedule: ScheduleVM | None,
        *,
        contexts: tuple[tuple[str, str], ...] = (),
        context_id: str = "",
        path: str = "",
        dirty: bool = False,
        bindings: tuple[tuple[str, str, str], ...] = (),
        config_names: tuple[str, ...] = (),
        busy: bool = False,
    ) -> None:
        self._context_id = str(context_id)
        choices = (("", "Select component" if self._compact else "Independent Subpulse file"),) + tuple(
            (key, label) for key, label in contexts if key
        )
        with signals_blocked(self.context_combo):
            existing = tuple((self.context_combo.itemData(i), self.context_combo.itemText(i)) for i in range(self.context_combo.count()))
            if existing != choices:
                self.context_combo.clear()
                for key, label in choices:
                    self.context_combo.addItem(label, key)
            self.context_combo.setCurrentIndex(self.context_combo.findData(self._context_id))
        if self._compact and not self._context_id:
            schedule = None
            bindings = ()
        self._accepted_name = schedule.document_name if schedule else ""
        self.name_edit.setText(self._accepted_name)
        self.name_edit.setEnabled(schedule is not None and not busy)
        instance = bool(self._context_id)
        if not self._compact:
            self._buttons["save"].setText("Save Pulse" if instance else "Save Subpulse")
            self._buttons["save_as"].setText("Export Subpulse" if instance else "Save as")
            self._buttons["insert"].setText("Duplicate in Pulse" if instance else "Insert into Pulse")
        for action, button in self._buttons.items():
            button.setEnabled(not busy and (schedule is not None or action in ("new", "open")))
        if not self._compact:
            self._buttons["save"].set_dirty(dirty)
        self.path_label.setText(
            f"{schedule.total_text} · Pulse defaults" if self._compact and schedule else
            "" if self._compact else
            path or ("Current Pulse" if instance else "Unsaved Subpulse file")
        )
        self.path_label.setToolTip(path)
        self.scope_label.setText(
            "Only this component changes. Save with Pulse."
            if instance else "" if self._compact else
            "Independent file. Insert creates a new component in the Pulse."
        )
        self.placeholder.setVisible(schedule is None)
        if self._compact:
            self._set_parameters(schedule, busy)
        elif schedule is not None:
            if self.schedule_view is None:
                self.schedule_view = self._make_schedule()
            self.schedule_view.set_schedule(schedule)
            self.schedule_view.setEnabled(not busy)
            self.schedule_view.show()
        elif self.schedule_view is not None:
            self.schedule_view.hide()
        if self._compact:
            bindings = ()
        wanted = {field_id for field_id, _label, _key in bindings}
        for field_id in tuple(self._binding_rows):
            if field_id not in wanted:
                for widget in self._binding_rows.pop(field_id):
                    self._binding_layout.removeWidget(widget)
                    retire_widget(widget)
        for row, (field_id, label, key) in enumerate(bindings):
            widgets = self._binding_rows.get(field_id)
            if widgets is None:
                label_widget = FluentLabel(label)
                combo = FluentComboBox()
                combo.setEditable(True)
                combo.lineEdit().setPlaceholderText("Config name")
                combo.lineEdit().editingFinished.connect(lambda fid=field_id: self._commit_binding(fid))
                combo.activated.connect(lambda _index, fid=field_id: self._commit_binding(fid))
                widgets = self._binding_rows[field_id] = (label_widget, combo)
            label_widget, combo = widgets
            label_widget.setText(label)
            with signals_blocked(combo, combo.lineEdit()):
                values = tuple(dict.fromkeys(config_names))
                if tuple(combo.itemData(i) for i in range(combo.count())) != values:
                    combo.clear()
                    for value in values:
                        combo.addItem(value, value)
                combo.setCurrentIndex(combo.findData(key))
                combo.setEditText(key)
                combo.setProperty("accepted_key", key)
                combo.setEnabled(not busy)
            self._binding_layout.addWidget(label_widget, row, 0)
            self._binding_layout.addWidget(combo, row, 1)
        self.bindings_box.setVisible(bool(bindings) and not self._compact)
        self._bindings_scroll.setFixedHeight(min(px(180), len(bindings) * (row_height() + px(6)) + px(4)))

    def _commit_binding(self, field_id: str) -> None:
        combo = self._binding_rows[field_id][1]
        key = read_editable_combo(combo)
        if key != (combo.property("accepted_key") or ""):
            self.edit_requested.emit("config_binding", (field_id, key))


__all__ = ["PulseComponentView"]
