"""Named Config values and this Pulse's references; all file I/O is external."""

from __future__ import annotations

from pathlib import Path
from difflib import SequenceMatcher
from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.fluent import (
    FluentButton, FluentComboBox,
    ElidedLabel, FluentGroupBox, FluentLabel, FluentReadoutEdit, FluentScrollArea,
    FluentSectionLabel, FluentTableView, column_header_label, field_text_inset,
    fluent_scrollbar_thickness,
    read_editable_combo, retire_widget, signals_blocked,
)

from ._layout import (
    bound_field_width, config_name_width, px, row_height,
)
from zlc_ui.fluent import card_pad, row_gap, section_gap
from .models import ConfigPageRecord
from zlc_ui.status import SUMMARY_SEPARATOR

#: The binding grid's columns, left to right: what it binds, the Config name
#: chosen for it, its default and the Config value (columns 2 and 3), and the
#: status, which takes the rest of the row.
_FIELD, _NAME, _DEFAULT, _STATUS = 0, 1, 2, 4


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
        self._binding_headers: dict[str, FluentLabel] = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = FluentScrollArea(self)
        body = QtWidgets.QWidget()
        self.scroll.set_width_bounded_widget(body)
        outer.addWidget(self.scroll)
        content = QtWidgets.QVBoxLayout(body)
        # The cards end on every tab's right edge whether or not the bar
        # shows: the scroll area keeps the bar's gutter at all times, and it
        # stands inside the page's right inset rather than added to it.
        page = section_gap()
        content.setContentsMargins(page, page, max(0, page - fluent_scrollbar_thickness()), page)
        content.setSpacing(section_gap())
        pad = card_pad()
        file_box = FluentGroupBox("Value file")
        file_layout = QtWidgets.QVBoxLayout(file_box)
        file_layout.setContentsMargins(pad, pad, pad, pad)
        file_layout.setSpacing(row_gap())
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(row_gap())
        self._file_buttons = []
        for name, text, role in (
            ("new", "New file", "standard"), ("load", "Open file", "standard"),
            ("refresh", "Refresh", "standard"), ("save", "Save", "standard"),
            ("save_as", "Save as", "standard"), ("unload", "Unload", "destructive"),
        ):
            button = FluentButton(text, role=role)
            button.setFixedHeight(row_height())
            button.clicked.connect(getattr(self, f"{name}_requested"))
            setattr(self, f"{name}_button", button)
            self._file_buttons.append(button)
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        file_layout.addLayout(toolbar)
        # The chosen file is read, not typed: Load and Save as choose it.
        # It stands on its own line under the commands, flush with the
        # status below it -- no field above it to share a text column with.
        self.path_text = FluentReadoutEdit("", flush=True)
        self.path_text.setPlaceholderText("No value file selected")
        file_layout.addWidget(self.path_text)
        self.file_status = ElidedLabel("Save the Config file before using its values.")
        file_layout.addWidget(self.file_status)
        self.values_table = FluentTableView()
        self.values_model = QtGui.QStandardItemModel(0, 3, self)
        self.values_model.setHorizontalHeaderLabels(("Name", "Value", "Unit"))
        self.values_table.setModel(self.values_model)
        self.values_table.set_content_rows_limit(12)
        self.values_table.verticalHeader().hide()
        # Name and value take what they need, the number set right like any
        # number and so beside its name; the unit closes the row.  Equal
        # thirds gave the unit as much room as the name, and a stretched
        # name put every value a table's width away from what it names.
        header = self.values_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.values_model.setHeaderData(
            1, QtCore.Qt.Horizontal, int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter),
            QtCore.Qt.TextAlignmentRole,
        )
        for column in (0, 2):
            self.values_model.setHeaderData(
                column, QtCore.Qt.Horizontal, int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter),
                QtCore.Qt.TextAlignmentRole,
            )
        self.values_model.itemChanged.connect(lambda _item: self._emit_entries())
        file_layout.addWidget(self.values_table)
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(row_gap())
        self.add_button = FluentButton("Add parameter")
        self.remove_button = FluentButton("Remove selected", role="destructive")
        self.add_button.clicked.connect(self._add_entry)
        self.remove_button.clicked.connect(self._remove_entries)
        actions.addWidget(self.add_button)
        actions.addWidget(self.remove_button)
        actions.addStretch(1)
        file_layout.addLayout(actions)
        content.addWidget(file_box)

        bindings_box = FluentGroupBox("Pulse bindings")
        bindings_layout = QtWidgets.QVBoxLayout(bindings_box)
        bindings_layout.setContentsMargins(pad, pad, pad, pad)
        bindings_layout.setSpacing(row_gap())
        self.mode_label = FluentLabel("Config is off. Pulse defaults are used.")
        self.mode_label.setWordWrap(True)
        bindings_layout.addWidget(self.mode_label)
        hint = ElidedLabel("Choose a value name for each field. Save the Pulse to keep these bindings.")
        bindings_layout.addWidget(hint)
        # Each column as wide as what it holds, so a value name sits beside
        # the field it binds: the field column is its widest label, the name
        # and the two values have fixed widths, and the status takes the
        # rest.  Stretch shares put a page width between them.
        self._binding_grid = QtWidgets.QGridLayout()
        self._binding_grid.setHorizontalSpacing(row_gap())
        self._binding_grid.setVerticalSpacing(row_gap())
        self._binding_grid.setColumnStretch(_STATUS, 1)
        self._grid_headers = tuple(
            column_header_label(title)
            for title in ("Pulse field", "Value name", "Default", "Config value", "Status")
        )
        # A header stands over its column's TEXT: the value names start a
        # field's text inset inside their boxes; the read values start at
        # their own edge.
        self._grid_headers[_NAME].setIndent(field_text_inset())
        for column, header in enumerate(self._grid_headers):
            self._binding_grid.addWidget(header, 0, column)
        bindings_layout.addLayout(self._binding_grid)
        self.bindings_placeholder = FluentLabel("No value bindings yet. Choose Config from a field's source control in Edit.")
        self.bindings_placeholder.setWordWrap(True)
        bindings_layout.addWidget(self.bindings_placeholder)
        content.insertWidget(0, bindings_box)
        content.addStretch(1)

    def set_config_mode(self, enabled: bool, error: str = "") -> None:
        self.mode_label.setText(error or ("Config is on. Saved sources are applied."
                                        if enabled else "Config is off. Pulse defaults are used."))

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
        self.bindings_placeholder.setVisible(not record.bindings)
        self.path_text.setText(record.file_path)
        self.path_text.setToolTip(record.file_path)
        status = f"Loaded file: {Path(record.active_path).name}" if record.active_path else "No value file selected"
        if record.dirty:
            status += f"{SUMMARY_SEPARATOR}unsaved edits do not participate in Fire"
        self.file_status.setText(status)
        self.file_status.setToolTip(record.active_path)
        # Primary with its star only while there is something to save.
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
                            items = [QtGui.QStandardItem(value) for value in entry]
                            items[1].setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                            self.values_model.insertRow(row, items)
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
        group_facts = {
            field_id: (group_id, label)
            for field_id, group_id, label in record.binding_groups
        }
        grouped = {}
        for binding in record.bindings:
            group_id, label = group_facts.get(binding[0], ("", "Pulse"))
            grouped.setdefault(group_id, (label, []))[1].append(binding)
        groups = sorted(grouped, key=lambda group_id: bool(group_id))
        show_groups = any(groups)
        for group_id in tuple(self._binding_headers):
            if not show_groups or group_id not in grouped:
                header = self._binding_headers.pop(group_id)
                self._binding_grid.removeWidget(header)
                retire_widget(header)
        ordered_rows = []
        row = 1
        for group_id in groups:
            label, bindings = grouped[group_id]
            if show_groups:
                header = self._binding_headers.get(group_id)
                if header is None:
                    header = FluentSectionLabel()
                    self._binding_headers[group_id] = header
                header.setText(label or "Pulse")
                index = self._binding_grid.indexOf(header)
                if index < 0 or self._binding_grid.getItemPosition(index) != (row, 0, 1, 5):
                    self._binding_grid.addWidget(header, row, 0, 1, 5)
                row += 1
            for binding in bindings:
                ordered_rows.append((row, binding))
                row += 1
        keys = record.available_names
        for row, (field_id, label, key, default, effective, state) in ordered_rows:
            widgets = self._binding_rows.get(field_id)
            if widgets is None:
                combo = FluentComboBox()
                combo.setEditable(True)
                combo.lineEdit().setPlaceholderText("Config name")
                combo.activated.connect(lambda _index, field=field_id: self._commit_binding(field))
                combo.editingFinished.connect(lambda field=field_id: self._commit_binding(field))
                # Default and Config value are read, not typed: readouts, flush
                # under their column headings like every other column here.
                widgets = (ElidedLabel(), combo, FluentReadoutEdit(flush=True), FluentReadoutEdit(flush=True),
                           ElidedLabel())
                for widget in widgets:
                    widget.setFixedHeight(row_height())
                combo.setFixedWidth(config_name_width())
                for widget in widgets[_DEFAULT:_STATUS]:
                    widget.setFixedWidth(px(110, minimum=90))
                self._binding_rows[field_id] = widgets
            for column, widget in enumerate(widgets):
                index = self._binding_grid.indexOf(widget)
                if index < 0 or self._binding_grid.getItemPosition(index) != (row, column, 1, 1):
                    self._binding_grid.addWidget(widget, row, column)
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
        labels = [widgets[_FIELD].text() for widgets in self._binding_rows.values()]
        width = bound_field_width((self._grid_headers[_FIELD].text(), *labels))
        for widget in (self._grid_headers[_FIELD], *(widgets[_FIELD] for widgets in self._binding_rows.values())):
            widget.setFixedWidth(width)


__all__ = ["PulseConfigView"]
