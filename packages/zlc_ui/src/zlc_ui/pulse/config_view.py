"""Named Config values and this Pulse's references; all file I/O is external."""

from __future__ import annotations

from pathlib import Path
from difflib import SequenceMatcher
from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.fluent import (
    ACCENT, GREY, ORANGE, YELLOW, FluentButton, FluentComboBox,
    ElidedLabel, FluentGroupBox, FluentLabel, FluentLineEdit, FluentScrollArea,
    FluentTableView, read_editable_combo, retire_widget, signals_blocked,
)

from ._layout import px, row_height
from .models import ConfigPageRecord


class PulseConfigView(QtWidgets.QWidget):
    new_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()
    refresh_requested = QtCore.pyqtSignal()
    save_requested = QtCore.pyqtSignal()
    save_as_requested = QtCore.pyqtSignal()
    unload_requested = QtCore.pyqtSignal()
    entries_edited = QtCore.pyqtSignal(object)
    binding_committed = QtCore.pyqtSignal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._projecting_entries = False
        self._binding_rows: dict[str, tuple] = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = FluentScrollArea(self)
        body = QtWidgets.QWidget()
        self.scroll.set_width_bounded_widget(body)
        outer.addWidget(self.scroll)
        content = QtWidgets.QVBoxLayout(body)
        content.setContentsMargins(px(8), px(8), px(8), px(8))
        content.setSpacing(px(8))
        file_box = FluentGroupBox("Config file")
        file_layout = QtWidgets.QVBoxLayout(file_box)
        file_layout.setContentsMargins(px(10), px(8), px(10), px(10))
        toolbar = QtWidgets.QHBoxLayout()
        self._file_buttons = []
        for name, text, color in (
            ("new", "New", GREY), ("load", "Load", ACCENT),
            ("refresh", "Refresh", GREY), ("save", "Save", YELLOW),
            ("save_as", "Save as", YELLOW), ("unload", "Unload", ORANGE),
        ):
            button = FluentButton(text, color=color)
            button.setFixedHeight(row_height())
            button.clicked.connect(getattr(self, f"{name}_requested"))
            setattr(self, f"{name}_button", button)
            self._file_buttons.append(button)
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        file_layout.addLayout(toolbar)
        self.path_text = FluentLineEdit("")
        self.path_text.setReadOnly(True)
        self.path_text.setPlaceholderText("Unsaved Config")
        file_layout.addWidget(self.path_text)
        self.file_status = ElidedLabel("Save the Config file before using its values.")
        file_layout.addWidget(self.file_status)
        self.values_table = FluentTableView()
        self.values_model = QtGui.QStandardItemModel(0, 3, self)
        self.values_model.setHorizontalHeaderLabels(("Name", "Value", "Unit"))
        self.values_table.setModel(self.values_model)
        self.values_table.set_content_rows_limit(2000)
        self.values_table.verticalHeader().hide()
        self.values_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.values_model.itemChanged.connect(lambda _item: self._emit_entries())
        file_layout.addWidget(self.values_table)
        actions = QtWidgets.QHBoxLayout()
        self.add_button = FluentButton("Add parameter", color=ACCENT)
        self.remove_button = FluentButton("Remove selected", color=ORANGE)
        self.add_button.clicked.connect(self._add_entry)
        self.remove_button.clicked.connect(self._remove_entries)
        actions.addWidget(self.add_button)
        actions.addWidget(self.remove_button)
        actions.addStretch(1)
        file_layout.addLayout(actions)
        content.addWidget(file_box)

        bindings_box = FluentGroupBox("This Pulse bindings")
        bindings_layout = QtWidgets.QVBoxLayout(bindings_box)
        bindings_layout.setContentsMargins(px(10), px(8), px(10), px(10))
        hint = ElidedLabel("Assign Config names here. Unassigned fields keep their Pulse default.")
        bindings_layout.addWidget(hint)
        self._binding_grid = QtWidgets.QGridLayout()
        self._binding_grid.setHorizontalSpacing(px(8))
        self._binding_grid.setVerticalSpacing(px(6))
        for column, (title, stretch) in enumerate(zip(
            ("Pulse field", "Config name", "Pulse default", "Saved value", "Status"),
            (2, 2, 1, 1, 2),
        )):
            header = FluentLabel(title)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            self._binding_grid.addWidget(header, 0, column)
            self._binding_grid.setColumnStretch(column, stretch)
        bindings_layout.addLayout(self._binding_grid)
        content.addWidget(bindings_box)
        content.addStretch(1)

    def _entries(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            tuple(self.values_model.index(row, column).data() or "" for column in range(3))
            for row in range(self.values_model.rowCount())
        )

    def _emit_entries(self) -> None:
        if not self._projecting_entries:
            self.entries_edited.emit(self._entries())

    def _add_entry(self) -> None:
        self.entries_edited.emit((*self._entries(), ("", "", "")))
        index = self.values_model.index(self.values_model.rowCount() - 1, 0)
        self.values_table.setCurrentIndex(index)
        self.values_table.edit(index, QtWidgets.QAbstractItemView.AllEditTriggers, None)

    def _remove_entries(self) -> None:
        rows = {index.row() for index in self.values_table.selectedIndexes()}
        if rows:
            self.entries_edited.emit(tuple(entry for row, entry in enumerate(self._entries()) if row not in rows))

    def _commit_binding(self, field_id: str) -> None:
        widgets = self._binding_rows.get(field_id)
        if widgets is None:
            return
        combo = widgets[1]
        wanted = read_editable_combo(combo)
        with signals_blocked(combo, combo.lineEdit()):
            combo.lineEdit().setModified(False)
            if wanted != (combo.property("accepted_key") or ""):
                self.binding_committed.emit(field_id, wanted)
            # The synchronous owner accepts and projects, or reports rejection.
            accepted = str(combo.property("accepted_key") or "")
            combo.setCurrentIndex(combo.findData(accepted))
            combo.setEditText(accepted)

    def set_page(self, record: ConfigPageRecord) -> None:
        self.path_text.setText(record.file_path)
        self.path_text.setToolTip(record.file_path)
        status = f"Active file: {Path(record.active_path).name}" if record.active_path else "No active file; Pulse defaults are used."
        if record.dirty:
            status += "  Unsaved edits do not participate in Fire."
        self.file_status.setText(status)
        self.file_status.setToolTip(record.active_path)
        self.save_button.set_dirty(record.dirty)
        for button in self._file_buttons:
            button.setEnabled(not record.busy)
        self.refresh_button.setEnabled(bool(record.file_path) and not record.busy)
        self.unload_button.setEnabled(bool(record.active_path or record.file_path) and not record.busy)
        self.values_table.setEnabled(not record.busy)
        self.add_button.setEnabled(not record.busy)
        self.remove_button.setEnabled(not record.busy)
        previous = self._entries()
        if previous != record.entries:
            self._projecting_entries = True
            try:
                for tag, start, stop, new_start, new_stop in reversed(
                    SequenceMatcher(a=previous, b=record.entries, autojunk=False).get_opcodes()
                ):
                    if tag == "equal":
                        continue
                    kept = min(stop - start, new_stop - new_start)
                    if stop - start > kept:
                        self.values_model.removeRows(start + kept, stop - start - kept)
                    for offset, entry in enumerate(record.entries[new_start:new_stop]):
                        row = start + offset
                        if offset >= kept:
                            self.values_model.insertRow(row, [QtGui.QStandardItem(value) for value in entry])
                        else:
                            for column, value in enumerate(entry):
                                self.values_model.item(row, column).setText(value)
            finally:
                self._projecting_entries = False
        fields = tuple(row[0] for row in record.bindings)
        for field_id in tuple(self._binding_rows):
            if field_id not in fields:
                for widget in self._binding_rows.pop(field_id):
                    self._binding_grid.removeWidget(widget)
                    retire_widget(widget)
        keys = record.available_names
        for row, (field_id, label, key, default, effective, state) in enumerate(record.bindings):
            widgets = self._binding_rows.get(field_id)
            if widgets is None:
                combo = FluentComboBox()
                combo.setEditable(True)
                combo.lineEdit().setPlaceholderText("Config name")
                combo.activated.connect(lambda _index, field=field_id: self._commit_binding(field))
                combo.editingFinished.connect(lambda field=field_id: self._commit_binding(field))
                widgets = (ElidedLabel(), combo, FluentLineEdit(), FluentLineEdit(), ElidedLabel())
                for value_widget in widgets[2:4]:
                    value_widget.setReadOnly(True)
                for widget in widgets:
                    widget.setFixedHeight(row_height())
                self._binding_rows[field_id] = widgets
            for column, widget in enumerate(widgets):
                index = self._binding_grid.indexOf(widget)
                if index < 0 or self._binding_grid.getItemPosition(index) != (row + 1, column, 1, 1):
                    self._binding_grid.addWidget(widget, row + 1, column)
            for column, value in ((0, label), (2, default), (3, effective), (4, state)):
                if widgets[column].text() != value:
                    widgets[column].setText(value)
                    widgets[column].setToolTip(value)
            combo = widgets[1]
            draft = combo.currentText() if combo.lineEdit().hasFocus() and combo.lineEdit().isModified() else None
            combo.setProperty("accepted_key", key)
            with signals_blocked(combo):
                if tuple(combo.itemData(index) for index in range(combo.count())) != keys:
                    combo.clear()
                    for name in keys:
                        combo.addItem(name, name)
                combo.setCurrentIndex(combo.findData(key))
                combo.setEditText(key if draft is None else draft)
                if draft is not None:
                    combo.lineEdit().setModified(True)
            combo.setEnabled(not record.busy)


__all__ = ["PulseConfigView"]
