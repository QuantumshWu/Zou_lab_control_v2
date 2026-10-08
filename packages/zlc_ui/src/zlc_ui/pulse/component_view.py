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
from .schedule_view import PulseScheduleView


class PulseComponentView(QtWidgets.QWidget):
    """Only editor intents and view records cross this page's boundary."""

    action_requested = QtCore.pyqtSignal(str, object)
    edit_requested = QtCore.pyqtSignal(str, object)
    feedback_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._context_id = ""
        self._accepted_name = ""
        self._binding_rows: dict[str, tuple] = {}
        self.schedule_view: PulseScheduleView | None = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(px(8), px(8), px(8), px(8))
        outer.setSpacing(px(8))
        header = FluentFrame()
        head = QtWidgets.QVBoxLayout(header)
        head.setContentsMargins(px(12), px(10), px(12), px(10))
        top = QtWidgets.QHBoxLayout()
        top.addWidget(FluentLabel("Editing"))
        self.context_combo = FluentComboBox()
        self.context_combo.setMinimumWidth(px(190))
        self.context_combo.currentIndexChanged.connect(self._context_changed)
        top.addWidget(self.context_combo, 1)
        self.name_edit = FluentLineEdit("")
        self.name_edit.setPlaceholderText("Component name")
        self.name_edit.editingFinished.connect(self._name_committed)
        top.addWidget(self.name_edit, 1)
        self._buttons = {}
        for action, text, color in (
            ("new", "New", GREY), ("open", "Open", ACCENT),
            ("save", "Save Subpulse", YELLOW), ("save_as", "Save as", YELLOW),
            ("insert", "Insert into Pulse", ACCENT),
        ):
            button = FluentButton(text, color=color)
            button.clicked.connect(lambda _checked=False, a=action: self.action_requested.emit(a, self._context_id))
            self._buttons[action] = button
            top.addWidget(button)
        head.addLayout(top)
        self.path_label = ElidedLabel("Create or open a Subpulse, or select an instance from the current Pulse.")
        head.addWidget(self.path_label)
        self.scope_label = ElidedLabel("Subpulses are reusable fragments. They do not run independently.")
        head.addWidget(self.scope_label)
        outer.addWidget(header)
        self.placeholder = FluentLabel("Create a Subpulse with New, open an existing file, or choose a Pulse component above.")
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
        outer.addWidget(self.placeholder, 1)
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
            ("remove_period", "remove_period_requested"), ("bracket", "bracket_committed"),
            ("bracket_add", "bracket_add_requested"), ("bracket_remove", "bracket_remove_requested"),
            ("visible_ports", "visible_ports_committed"),
        ):
            getattr(schedule, signal).connect(lambda *args, a=action: self.edit_requested.emit(a, args))
        schedule.feedback_requested.connect(self.feedback_requested)
        self.layout().insertWidget(1, schedule, 1)
        return schedule

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
        choices = (("", "Subpulse file"),) + tuple((key, label) for key, label in contexts if key)
        with signals_blocked(self.context_combo):
            existing = tuple((self.context_combo.itemData(i), self.context_combo.itemText(i)) for i in range(self.context_combo.count()))
            if existing != choices:
                self.context_combo.clear()
                for key, label in choices:
                    self.context_combo.addItem(label, key)
            self.context_combo.setCurrentIndex(self.context_combo.findData(self._context_id))
        self._accepted_name = schedule.document_name if schedule else ""
        self.name_edit.setText(self._accepted_name)
        self.name_edit.setEnabled(schedule is not None and not busy)
        instance = bool(self._context_id)
        self._buttons["save"].setText("Save Pulse" if instance else "Save Subpulse")
        self._buttons["save_as"].setText("Export Subpulse" if instance else "Save as")
        self._buttons["insert"].setText("Duplicate in Pulse" if instance else "Insert into Pulse")
        for action, button in self._buttons.items():
            button.setEnabled(not busy and (schedule is not None or action in ("new", "open")))
        self._buttons["save"].set_dirty(dirty)
        self.path_label.setText(path or ("Embedded in the current Pulse" if instance else "Unsaved Subpulse"))
        self.path_label.setToolTip(path)
        self.scope_label.setText(
            "Editing this instance changes the current Pulse. Other instances and Subpulse files are unchanged."
            if instance else "Independent copy; Hold/Ramp inherits the preceding output when inserted. Standalone values do not define its entry state."
        )
        self.placeholder.setVisible(schedule is None)
        if schedule is not None:
            if self.schedule_view is None:
                self.schedule_view = self._make_schedule()
            self.schedule_view.set_schedule(schedule)
            self.schedule_view.setEnabled(not busy)
            self.schedule_view.show()
        elif self.schedule_view is not None:
            self.schedule_view.hide()
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
                values = tuple(dict.fromkeys(("", *config_names, *([key] if key else []))))
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
        self.bindings_box.setVisible(bool(bindings))
        self._bindings_scroll.setFixedHeight(min(px(180), len(bindings) * (row_height() + px(6)) + px(4)))

    def _commit_binding(self, field_id: str) -> None:
        combo = self._binding_rows[field_id][1]
        key = read_editable_combo(combo)
        if key != (combo.property("accepted_key") or ""):
            self.edit_requested.emit("config_binding", (field_id, key))


__all__ = ["PulseComponentView"]
