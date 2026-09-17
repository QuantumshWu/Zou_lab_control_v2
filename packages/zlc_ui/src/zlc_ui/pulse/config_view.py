"""Named Config values and this Pulse's references; all file I/O is external."""

from __future__ import annotations

from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.fluent import (
    ACCENT, GREY, ORANGE, YELLOW, FluentButton, FluentComboBox,
    ElidedLabel, FluentGroupBox, FluentLineEdit, FluentTableView, signals_blocked,
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
        self._record = ConfigPageRecord()
        self._projecting_entries = False
        self._binding_combos: dict[str, FluentComboBox] = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(px(8), px(8), px(8), px(8))
        outer.setSpacing(px(8))
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
        self.values_table.ensurePolished()
        self.values_table.setMinimumHeight(
            self.values_table.horizontalHeader().sizeHint().height()
            + 2 * self.values_table.verticalHeader().defaultSectionSize()
            + 2 * self.values_table.frameWidth()
        )
        self.values_table.verticalHeader().hide()
        self.values_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.values_model.itemChanged.connect(lambda _item: self._emit_entries())
        file_layout.addWidget(self.values_table, 1)
        actions = QtWidgets.QHBoxLayout()
        self.add_button = FluentButton("Add parameter", color=ACCENT)
        self.remove_button = FluentButton("Remove selected", color=ORANGE)
        self.add_button.clicked.connect(self._add_entry)
        self.remove_button.clicked.connect(self._remove_entries)
        actions.addWidget(self.add_button)
        actions.addWidget(self.remove_button)
        actions.addStretch(1)
        file_layout.addLayout(actions)
        outer.addWidget(file_box, 3)

        bindings_box = FluentGroupBox("This Pulse bindings")
        bindings_layout = QtWidgets.QVBoxLayout(bindings_box)
        bindings_layout.setContentsMargins(px(10), px(8), px(10), px(10))
        hint = ElidedLabel("Assign Config names here. Unassigned fields keep their Pulse default.")
        bindings_layout.addWidget(hint)
        self.bindings_table = FluentTableView()
        self.bindings_model = QtGui.QStandardItemModel(0, 5, self)
        self.bindings_model.setHorizontalHeaderLabels(("Pulse field", "Config name", "Default", "Saved value", "Status"))
        self.bindings_table.setModel(self.bindings_model)
        self.bindings_table.verticalHeader().hide()
        self.bindings_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        bindings_layout.addWidget(self.bindings_table, 1)
        outer.addWidget(bindings_box, 2)

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

    def set_page(self, record: ConfigPageRecord) -> None:
        self._record = record
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
        if self._entries() != record.entries:
            self._projecting_entries = True
            try:
                self.values_model.setRowCount(len(record.entries))
                for row, entry in enumerate(record.entries):
                    for column, value in enumerate(entry):
                        self.values_model.setItem(row, column, QtGui.QStandardItem(str(value)))
            finally:
                self._projecting_entries = False
        fields = tuple(row[0] for row in record.bindings)
        if fields != tuple(self._binding_combos):
            self.bindings_model.setRowCount(0)
            self._binding_combos.clear()
            self.bindings_model.setRowCount(len(fields))
            for row, field_id in enumerate(fields):
                combo = FluentComboBox()
                combo.currentIndexChanged.connect(
                    lambda _index, field=field_id, widget=combo: self.binding_committed.emit(field, str(widget.currentData() or ""))
                )
                self._binding_combos[field_id] = combo
                self.bindings_table.setIndexWidget(self.bindings_model.index(row, 1), combo)
                self.bindings_table.setRowHeight(row, combo.sizeHint().height() + px(4))
            self.bindings_table.ensurePolished()
            self.bindings_table.setMinimumHeight(
                self.bindings_table.horizontalHeader().sizeHint().height()
                + sum(self.bindings_table.rowHeight(row) for row in range(min(2, len(fields))))
                + 2 * self.bindings_table.frameWidth()
            )
        keys = tuple(dict.fromkeys(name for name, _value, _unit in record.entries if name))
        for row, (field_id, label, key, default, effective, state) in enumerate(record.bindings):
            for column, value in ((0, label), (2, default), (3, effective), (4, state)):
                item = self.bindings_model.item(row, column)
                if item is None:
                    item = QtGui.QStandardItem()
                    item.setEditable(False)
                    self.bindings_model.setItem(row, column, item)
                item.setText(value)
                item.setToolTip(value)
            combo = self._binding_combos[field_id]
            choices = ("", *keys, *((key,) if key and key not in keys else ()))
            with signals_blocked(combo):
                if tuple(combo.itemData(index) for index in range(combo.count())) != choices:
                    combo.clear()
                    for name in choices:
                        combo.addItem(name or "Default (unassigned)", name)
                combo.setCurrentIndex(combo.findData(key))
            combo.setEnabled(not record.busy)


__all__ = ["PulseConfigView"]
