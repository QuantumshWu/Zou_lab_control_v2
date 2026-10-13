"""Subpulse editing with the same Period/Bracket surface as a full Pulse."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT, GREY, YELLOW, ElidedLabel, FluentButton, FluentComboBox,
    FluentGroupBox, FluentLabel, FluentLineEdit, FluentScrollArea,
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
        self._source_rows: dict[str, tuple] = {}
        self.schedule_view: PulseScheduleView | None = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = FluentScrollArea(self)
        body = QtWidgets.QWidget()
        self.scroll.set_width_bounded_widget(body)
        outer.addWidget(self.scroll)
        content = QtWidgets.QVBoxLayout(body)
        content.setContentsMargins(px(8), px(8), px(8), px(8))
        content.setSpacing(px(8))
        self.sources_box = FluentGroupBox("Pulse bindings")
        sources_layout = QtWidgets.QVBoxLayout(self.sources_box)
        sources_layout.setContentsMargins(px(10), px(8), px(10), px(10))
        self.source_mode_label = FluentLabel("Pulse mode: Default")
        sources_layout.addWidget(self.source_mode_label)
        hint = FluentLabel("Browse to bind a file to each Subpulse. Save the Pulse to keep these bindings.")
        hint.setWordWrap(True)
        sources_layout.addWidget(hint)
        self._source_grid = QtWidgets.QGridLayout()
        self._source_grid.setHorizontalSpacing(px(8))
        self._source_grid.setVerticalSpacing(px(6))
        for column, (title, stretch) in enumerate(zip(
            ("Subpulse", "Source", "Subpulse file", "", "", "Status"), (2, 0, 5, 0, 0, 3),
        )):
            label = FluentLabel(title)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            self._source_grid.addWidget(label, 0, column)
            self._source_grid.setColumnStretch(column, stretch)
        sources_layout.addLayout(self._source_grid)
        self.sources_placeholder = FluentLabel("No Subpulses yet. Group periods or insert a Subpulse in Edit.")
        self.sources_placeholder.setWordWrap(True)
        sources_layout.addWidget(self.sources_placeholder)
        content.addWidget(self.sources_box)

        self.editor_box = FluentGroupBox("Subpulse editor")
        head = QtWidgets.QVBoxLayout(self.editor_box)
        self._editor_layout = head
        head.setContentsMargins(px(10), px(8), px(10), px(10))
        toolbar = QtWidgets.QHBoxLayout()
        self._buttons = {}
        for action, text, color in (
            ("new", "New file", GREY), ("open", "Open file", ACCENT),
            ("save", "Save", YELLOW), ("save_as", "Save as", YELLOW),
            ("insert", "Insert into Pulse", ACCENT),
        ):
            button = FluentButton(text, color=color)
            button.setFixedHeight(row_height())
            button.clicked.connect(lambda _checked=False, a=action: self.action_requested.emit(a, self._context_id))
            self._buttons[action] = button
            toolbar.addWidget(button)
            button.setVisible(action in ("new", "open"))
        toolbar.addStretch(1)
        head.addLayout(toolbar)
        self.document_controls = QtWidgets.QWidget()
        top = QtWidgets.QHBoxLayout(self.document_controls)
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(FluentLabel("Editing"))
        self.context_combo = FluentComboBox()
        self.context_combo.setMinimumWidth(px(190))
        self.context_combo.currentIndexChanged.connect(self._context_changed)
        top.addWidget(self.context_combo, 1)
        self.context_combo.hide()
        self.name_edit = FluentLineEdit("")
        self.name_edit.setPlaceholderText("Subpulse name")
        self.name_edit.editingFinished.connect(self._name_committed)
        top.addWidget(self.name_edit, 1)
        head.addWidget(self.document_controls)
        self.document_controls.hide()
        self.path_label = ElidedLabel("")
        head.addWidget(self.path_label)
        self.path_label.hide()
        self.scope_label = FluentLabel("")
        self.scope_label.setWordWrap(True)
        head.addWidget(self.scope_label)
        self.scope_label.hide()
        self.placeholder = FluentLabel("Click a Subpulse name above to edit its default, or open an independent file.")
        self.placeholder.setWordWrap(True)
        head.addWidget(self.placeholder)
        self.bindings_box = FluentGroupBox("Numeric Config references")
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
        head.addWidget(self.bindings_box)
        self.bindings_box.hide()
        content.addWidget(self.editor_box)
        content.addStretch(1)

    def set_sources(
        self, rows: tuple[tuple[str, str, str, str, str], ...], *,
        enabled: bool, busy: bool = False, error: str = "",
    ) -> None:
        """Project every Pulse instance's file binding, independently of the editor below."""
        self.source_mode_label.setText(
            error or ("Config is on. Saved sources are applied."
                      if enabled else "Config is off. Pulse defaults are used.")
        )
        self.source_mode_label.setWordWrap(True)
        wanted = {key for key, _name, _source, _path, _status in rows}
        for key in tuple(self._source_rows):
            if key not in wanted:
                for widget in self._source_rows.pop(key):
                    self._source_grid.removeWidget(widget)
                    retire_widget(widget)
        for row, (key, name, source, path, status) in enumerate(rows, 1):
            widgets = self._source_rows.get(key)
            if widgets is None:
                name_button = FluentButton(name, color=GREY)
                name_button.setToolTip("Edit this Subpulse's embedded default")
                name_button.clicked.connect(lambda _checked=False, item=key: self.action_requested.emit("edit", item))
                source_combo = FluentComboBox()
                source_combo.addItem("Default", "default")
                source_combo.addItem("Config", "config")
                source_combo.currentIndexChanged.connect(lambda _index, item=key: self._commit_source(item))
                path_edit = FluentLineEdit("")
                path_edit.setMinimumWidth(px(160))
                path_edit.setPlaceholderText("Choose a Subpulse file")
                path_edit.setToolTip("Enter a file to bind it as Config. Clear the path to use the embedded default.")
                path_edit.editingFinished.connect(lambda item=key: self._commit_source(item, path_changed=True))
                browse = FluentButton("Browse…", color=ACCENT)
                browse.clicked.connect(lambda _checked=False, item=key: self.action_requested.emit("browse_source", item))
                open_file = FluentButton("Open file", color=GREY)
                open_file.clicked.connect(lambda _checked=False, item=key: self.action_requested.emit("open_source", item))
                state = FluentLabel("")
                state.setWordWrap(True)
                widgets = self._source_rows[key] = (name_button, source_combo, path_edit, browse, open_file, state)
                for widget in widgets[:-1]:
                    widget.setFixedHeight(row_height())
            name_button, source_combo, path_edit, browse, open_file, state = widgets
            name_button.setText(name)
            name_button.setToolTip(f"Select {name}")
            with signals_blocked(source_combo, path_edit):
                draft = path_edit.hasFocus() and path_edit.isModified()
                source_combo.setCurrentIndex(source_combo.findData(source))
                if not draft:
                    path_edit.setText(path)
                source_combo.setProperty("accepted_source", source)
                path_edit.setProperty("accepted_path", path)
            for widget in widgets[:-1]:
                widget.setEnabled(not busy)
            open_file.setEnabled(bool(path) and not busy)
            path_edit.setToolTip(path or "Choose a Subpulse file")
            state.setText(status or ("Choose file" if source == "config" and not path else "Embedded default"))
            state.setToolTip(status)
            for column, widget in enumerate(widgets):
                self._source_grid.addWidget(widget, row, column)
        self.sources_placeholder.setVisible(not rows)

    def _commit_source(self, key: str, *, path_changed: bool = False) -> None:
        _name, combo, path_edit, _browse, _open, _status = self._source_rows[key]
        path = path_edit.text().strip()
        source = combo.currentData()
        if path_changed and path != (path_edit.property("accepted_path") or ""):
            source = "config" if path else "default"
        path_edit.setModified(False)
        if (source, path) != (combo.property("accepted_source"), path_edit.property("accepted_path") or ""):
            self.action_requested.emit("source", (key, source, path))
        # The owner projects acceptance or rejection synchronously. Use that
        # accepted projection even when a refused edit left the same record.
        with signals_blocked(combo, path_edit):
            combo.setCurrentIndex(combo.findData(combo.property("accepted_source")))
            path_edit.setText(path_edit.property("accepted_path") or "")

    def focus_source(self, component_id: str) -> None:
        widgets = self._source_rows.get(str(component_id))
        if widgets is not None:
            path_edit = widgets[2]
            self.scroll.ensureWidgetVisible(path_edit)
            path_edit.setFocus(QtCore.Qt.OtherFocusReason)
            path_edit.selectAll()

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
        schedule.setMinimumHeight(px(340))
        self._editor_layout.insertWidget(self._editor_layout.indexOf(self.bindings_box), schedule, 1)
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
        self.editor_box.setTitle("Embedded default" if instance else "Subpulse file")
        self._buttons["save"].setText("Save Pulse" if instance else "Save")
        self._buttons["save_as"].setText("Export file" if instance else "Save as")
        self._buttons["insert"].setText("Duplicate in Pulse" if instance else "Insert into Pulse")
        for action, button in self._buttons.items():
            button.setEnabled(not busy and (schedule is not None or action in ("new", "open")))
            button.setVisible(schedule is not None or action in ("new", "open"))
        self._buttons["save"].set_dirty(dirty)
        self.path_label.setText(path or ("Embedded in the current Pulse" if instance else "Unsaved Subpulse"))
        self.path_label.setToolTip(path)
        self.scope_label.setText(
            "Editing the embedded default. Save the Pulse to keep these changes."
            if instance else "Save this file, then Refresh to apply it to its Subpulse references."
        )
        self.document_controls.setVisible(schedule is not None)
        self.path_label.setVisible(schedule is not None)
        self.scope_label.setVisible(schedule is not None)
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
        self.bindings_box.setVisible(bool(bindings))
        self._bindings_scroll.setFixedHeight(min(px(180), len(bindings) * (row_height() + px(6)) + px(4)))

    def _commit_binding(self, field_id: str) -> None:
        combo = self._binding_rows[field_id][1]
        key = read_editable_combo(combo)
        if key != (combo.property("accepted_key") or ""):
            self.edit_requested.emit("config_binding", (field_id, key))


__all__ = ["PulseComponentView"]
