"""Pure saved-figure browser shell.

The presenter owns archive IO, metadata projection and the plot widget.  This
view only provides the file/path intent, generic info projection, and an
atomic QWidget mount point for the presenter-owned surface.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.console.board_view import ConsoleBoardView
from zlc_ui.console.panel_card_view import PanelCardView
from zlc_ui.console.panel_editor_view import PanelEditorView
from zlc_ui.fluent import (
    ACCENT,
    GREY,
    ORANGE_TINT,
    FluentButton,
    FluentCheckBox,
    FluentComboBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentReadoutEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentSpinBox,
    FluentTableView,
    FluentTabWidget,
    InfoPane,
    fluent_save_path,
    muted_note_label,
    scaled_px,
    setting_label_width,
    signals_blocked,
    window_pad,
)


def _plain_scalar(value: object) -> str:
    """Format one already-projected table scalar without interpreting it."""

    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):
            pass
    return "" if value is None else str(value)


def _matrix_item(values: object, row: int, column: int) -> object:
    """Read a two-dimensional plain/array projection lazily."""

    try:
        return values[row, column]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        row_value = values[row]  # type: ignore[index]
        if isinstance(row_value, (str, bytes)):
            return row_value if column == 0 else ""
        try:
            return row_value[column]
        except (IndexError, KeyError, TypeError):
            return row_value if column == 0 else ""


def _grid_header_text(projection: object, section: int) -> str:
    """Format one sparse-grid address without materializing every header."""

    grid = dict(projection or {})
    cell_indices = grid.get("cell_indices", ())
    coordinates = tuple(grid.get("coordinates", ()))
    labels = tuple(grid.get("labels") or ())
    values = []
    for dimension, coordinate_values in enumerate(coordinates):
        try:
            coordinate_index = int(cell_indices[section, dimension])
        except (IndexError, KeyError, TypeError):
            coordinate_index = int(cell_indices[section][dimension])
        label_values = labels[dimension] if dimension < len(labels) else None
        label = (
            ""
            if label_values is None
            else _plain_scalar(label_values[coordinate_index])
        )
        values.append(
            label
            or _plain_scalar(coordinate_values[coordinate_index])
        )
    return f"({', '.join(values)})"


class _VirtualTextTableModel(QtCore.QAbstractTableModel):
    """Visible-cell-only editable view over a presenter-owned 2-D projection."""

    edits_requested = QtCore.pyqtSignal(object)
    rejected = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._shape = (0, 0)
        self._values: object = ()
        self._column_values: object | None = None
        self._validity: object | None = None
        self._row_headers: object = ()
        self._column_headers: object = ()
        self._row_header_grid: object | None = None
        self._column_header_grid: object | None = None
        self._editable = True
        self._blank_hint = ""
        self._pending: dict[tuple[int, int], str] = {}

    def set_projection(self, projection: object) -> None:
        data = dict(projection or {})
        shape = tuple(int(value) for value in tuple(data.get("shape", (0, 0))))
        if len(shape) != 2 or any(value < 0 for value in shape):
            raise ValueError("table projection shape must contain two nonnegative sizes")
        self.beginResetModel()
        self._shape = shape
        self._values = data.get("values", ())
        self._column_values = data.get("column_values")
        self._validity = data.get("validity")
        self._row_headers = data.get("row_headers", ())
        self._column_headers = data.get("column_headers", ())
        self._row_header_grid = data.get("row_header_grid")
        self._column_header_grid = data.get("column_header_grid")
        self._editable = bool(data.get("editable", True))
        self._blank_hint = str(data.get("blank_hint", ""))
        self._pending.clear()
        self.endResetModel()

    def rowCount(self, _parent=QtCore.QModelIndex()) -> int:  # noqa: N802
        return self._shape[0]

    def columnCount(self, _parent=QtCore.QModelIndex()) -> int:  # noqa: N802
        return self._shape[1]

    def _projected_text(self, row: int, column: int) -> str:
        pending = self._pending.get((row, column))
        if pending is not None:
            return pending
        if self._validity is not None and not bool(
            _matrix_item(self._validity, row, column)
        ):
            return ""
        if self._column_values is not None:
            values = self._column_values[column]  # type: ignore[index]
            return "" if values is None else _plain_scalar(values[row])
        return _plain_scalar(_matrix_item(self._values, row, column))

    def data(self, index, role=QtCore.Qt.DisplayRole):  # noqa: A003
        if not index.isValid():
            return None
        row, column = index.row(), index.column()
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            return self._projected_text(row, column)
        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        if role == QtCore.Qt.BackgroundRole and (row, column) in self._pending:
            return QtGui.QBrush(QtGui.QColor(ORANGE_TINT))
        if (
            role == QtCore.Qt.ToolTipRole
            and self._blank_hint
            and self._projected_text(row, column) == ""
        ):
            return self._blank_hint
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):  # noqa: N802
        if role not in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole):
            return None
        grid = (
            self._column_header_grid
            if orientation == QtCore.Qt.Horizontal
            else self._row_header_grid
        )
        if grid is not None:
            return _grid_header_text(grid, int(section))
        labels = self._column_headers if orientation == QtCore.Qt.Horizontal else self._row_headers
        try:
            return _plain_scalar(labels[section])  # type: ignore[index]
        except (IndexError, KeyError, TypeError):
            return str(int(section))

    def flags(self, index):
        flags = super().flags(index)
        return flags | QtCore.Qt.ItemIsEditable if self._editable else flags

    def setData(self, index, value, role=QtCore.Qt.EditRole):  # noqa: N802
        if role != QtCore.Qt.EditRole or not index.isValid() or not self._editable:
            return False
        self.set_block_data(index.row(), index.column(), ((str(value),),))
        return True

    def set_block_data(self, row: int, column: int, values: object) -> None:
        rows = tuple(tuple(fields) for fields in tuple(values))
        first_row = int(row)
        first_column = int(column)
        if (
            first_row < 0
            or first_column < 0
            or first_row + len(rows) > self._shape[0]
            or any(first_column + len(fields) > self._shape[1] for fields in rows)
        ):
            self.rejected.emit(
                "Paste exceeds the current table; resize axes first"
            )
            return
        edits = []
        for row_offset, fields in enumerate(rows):
            for column_offset, value in enumerate(fields):
                target = (first_row + row_offset, first_column + column_offset)
                text = str(value)
                self._pending[target] = text
                edits.append((target[0], target[1], text))
        self._publish_edits(edits)

    def clear_indices(self, indices: object) -> None:
        edits = []
        for row, column in tuple(indices):
            target = (int(row), int(column))
            if 0 <= target[0] < self._shape[0] and 0 <= target[1] < self._shape[1]:
                self._pending[target] = ""
                edits.append((target[0], target[1], ""))
        self._publish_edits(edits)

    def _publish_edits(self, edits: object) -> None:
        cells = tuple(edits)
        if not cells:
            return
        rows = tuple(value[0] for value in cells)
        columns = tuple(value[1] for value in cells)
        self.dataChanged.emit(
            self.index(min(rows), min(columns)),
            self.index(max(rows), max(columns)),
            (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.BackgroundRole),
        )
        self.edits_requested.emit(cells)


def _fill_choice_combo(combo: FluentComboBox, rows: object, current: object) -> None:
    with signals_blocked(combo):
        combo.clear()
        for row in tuple(rows or ()):
            values = tuple(row)
            if len(values) not in (2, 3):
                raise ValueError("choice rows must be (key, label[, enabled])")
            key, label = values[:2]
            combo.addItem(str(label), key)
            if len(values) == 3:
                item = combo.model().item(combo.count() - 1)
                if item is not None:
                    item.setEnabled(bool(values[2]))
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else (-1 if not combo.count() else 0))


class _DataEditorView(QtWidgets.QWidget):
    """Fluent working-copy editor; all Dataset meaning remains in its presenter."""

    intent_requested = QtCore.pyqtSignal(object)

    def __init__(self, projection: object, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._updating = False
        self._selected_axis = ""
        self._save_suggested = "figure.npz"
        self._slice_widgets: dict[
            str, tuple[QtWidgets.QWidget, FluentSpinBox, FluentReadoutEdit]
        ] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(window_pad(0.5), window_pad(0.5), window_pad(0.5), window_pad(0.5))
        root.setSpacing(window_pad(0.5))

        dataset_group = FluentGroupBox("Dataset", self)
        dataset_layout = QtWidgets.QGridLayout(dataset_group)
        dataset_layout.setContentsMargins(window_pad(0.75), window_pad(0.75), window_pad(0.75), window_pad(0.6))
        dataset_layout.setHorizontalSpacing(window_pad(0.5))
        dataset_layout.setVerticalSpacing(window_pad(0.35))
        label_width = setting_label_width(("Name", "Type", "Value unit", "Note"), minimum=66)
        self.name_edit = FluentLineEdit()
        self.dtype_combo = FluentComboBox()
        self.dtype_combo.setMinimumContentsLength(9)
        self.unit_edit = FluentLineEdit()
        self.note_edit = FluentLineEdit()
        self.note_edit.setPlaceholderText("Optional reason for this manual edit")
        dataset_layout.addWidget(FluentSettingRow("Name", self.name_edit, label_width=label_width), 0, 0)
        dataset_layout.addWidget(FluentSettingRow("Type", self.dtype_combo, label_width=label_width), 0, 1)
        dataset_layout.addWidget(FluentSettingRow("Value unit", self.unit_edit, label_width=label_width), 1, 0)
        dataset_layout.addWidget(FluentSettingRow("Note", self.note_edit, label_width=label_width), 1, 1)
        action_row = QtWidgets.QHBoxLayout()
        action_row.setSpacing(window_pad(0.5))
        self.source_note = muted_note_label("")
        action_row.addWidget(self.source_note, 1)
        self.apply_button = FluentButton("Apply && Preview", color=ACCENT)
        self.save_button = FluentButton("Save As…", color=ACCENT)
        self.discard_button = FluentButton("Discard", color=GREY)
        action_row.addWidget(self.apply_button)
        action_row.addWidget(self.save_button)
        action_row.addWidget(self.discard_button)
        dataset_layout.addLayout(action_row, 2, 0, 1, 2)
        root.addWidget(dataset_group)

        body = QtWidgets.QWidget(self)
        body.setStyleSheet("background: transparent;")
        body_layout = QtWidgets.QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(window_pad(0.5))

        axes_group = FluentGroupBox("Axes", body)
        axes_group.setMinimumWidth(scaled_px(450, minimum=380))
        axes_group.setMaximumWidth(scaled_px(560, minimum=460))
        axes_layout = QtWidgets.QVBoxLayout(axes_group)
        axes_layout.setContentsMargins(window_pad(0.75), window_pad(0.75), window_pad(0.75), window_pad(0.6))
        axes_layout.setSpacing(window_pad(0.35))
        axis_choice_row = QtWidgets.QHBoxLayout()
        axis_choice_row.setSpacing(window_pad(0.25))
        self.axis_combo = FluentComboBox()
        self.axis_combo.setMinimumContentsLength(13)
        axis_choice_row.addWidget(self.axis_combo, 1)
        self.remove_axis_button = FluentButton("−", color=GREY)
        self.remove_axis_button.setToolTip("Remove selected axis")
        self.axis_up_button = FluentButton("↑", color=GREY)
        self.axis_up_button.setToolTip("Move selected axis up")
        self.axis_down_button = FluentButton("↓", color=GREY)
        self.axis_down_button.setToolTip("Move selected axis down")
        compact = scaled_px(30, minimum=24)
        for button in (self.remove_axis_button, self.axis_up_button, self.axis_down_button):
            button.setFixedWidth(compact)
            axis_choice_row.addWidget(button)
        axes_layout.addLayout(axis_choice_row)
        add_axis_row = QtWidgets.QHBoxLayout()
        add_axis_row.setSpacing(window_pad(0.25))
        self.add_axis_domain_combo = FluentComboBox()
        self.add_axis_domain_combo.setMinimumContentsLength(12)
        self.add_axis_button = FluentButton("Add axis", color=ACCENT)
        add_axis_row.addWidget(self.add_axis_domain_combo, 1)
        add_axis_row.addWidget(self.add_axis_button)
        axes_layout.addLayout(add_axis_row)
        axis_label_width = setting_label_width(
            (
                "Domain", "Size", "Name", "Role", "Coord. type", "Unit",
                "Coord. frame",
            ),
            minimum=68,
        )
        self.domain_readout = FluentReadoutEdit()
        self.axis_size_spin = FluentSpinBox()
        self.axis_size_spin.setRange(1, 2_147_483_647)
        self.axis_size_spin.setKeyboardTracking(False)
        self.axis_name_edit = FluentLineEdit()
        self.role_combo = FluentComboBox()
        self.value_kind_combo = FluentComboBox()
        self.axis_unit_edit = FluentLineEdit()
        self.frame_edit = FluentLineEdit()
        self._axis_rows: dict[str, FluentSettingRow] = {}
        axis_form = QtWidgets.QGridLayout()
        axis_form.setContentsMargins(0, 0, 0, 0)
        axis_form.setHorizontalSpacing(window_pad(0.35))
        axis_form.setVerticalSpacing(window_pad(0.25))
        for field, label, control, row_index, column_index in (
            ("domain", "Domain", self.domain_readout, 0, 0),
            ("size", "Size", self.axis_size_spin, 0, 1),
            ("name", "Name", self.axis_name_edit, 1, 0),
            ("role", "Role", self.role_combo, 1, 1),
            ("unit", "Unit", self.axis_unit_edit, 2, 0),
            ("coordinate_frame", "Coord. frame", self.frame_edit, 2, 1),
            ("value_kind", "Coord. type", self.value_kind_combo, 3, 0),
        ):
            row = FluentSettingRow(label, control, label_width=axis_label_width)
            self._axis_rows[field] = row
            axis_form.addWidget(row, row_index, column_index)
        axes_layout.addLayout(axis_form)
        coordinate_header = QtWidgets.QHBoxLayout()
        coordinate_header.setSpacing(window_pad(0.25))
        coordinate_header.addWidget(FluentSectionLabel("Coordinates"))
        coordinate_header.addStretch(1)
        self.insert_coordinate_button = FluentButton("+", color=ACCENT)
        self.insert_coordinate_button.setToolTip("Insert coordinate")
        self.remove_coordinate_button = FluentButton("−", color=GREY)
        self.remove_coordinate_button.setToolTip("Remove selected coordinates")
        self.coordinate_up_button = FluentButton("↑", color=GREY)
        self.coordinate_up_button.setToolTip("Move selected coordinate up")
        self.coordinate_down_button = FluentButton("↓", color=GREY)
        self.coordinate_down_button.setToolTip("Move selected coordinate down")
        for button in (
            self.insert_coordinate_button,
            self.remove_coordinate_button,
            self.coordinate_up_button,
            self.coordinate_down_button,
        ):
            button.setFixedWidth(compact)
            coordinate_header.addWidget(button)
        axes_layout.addLayout(coordinate_header)
        self.coordinate_model = _VirtualTextTableModel(self)
        self.coordinate_table = FluentTableView(self)
        self.coordinate_table.setModel(self.coordinate_model)
        self.coordinate_table.horizontalHeader().setStretchLastSection(True)
        axes_layout.addWidget(self.coordinate_table, 1)
        body_layout.addWidget(axes_group, 0)

        data_group = FluentGroupBox("Data", body)
        data_layout = QtWidgets.QVBoxLayout(data_group)
        data_layout.setContentsMargins(window_pad(0.75), window_pad(0.75), window_pad(0.75), window_pad(0.6))
        data_layout.setSpacing(window_pad(0.35))
        self._slice_holder = QtWidgets.QWidget(data_group)
        self._slice_holder.setStyleSheet("background: transparent;")
        self._slice_layout = QtWidgets.QGridLayout(self._slice_holder)
        self._slice_layout.setContentsMargins(0, 0, 0, 0)
        self._slice_layout.setHorizontalSpacing(window_pad(0.5))
        self._slice_layout.setVerticalSpacing(window_pad(0.25))
        data_layout.addWidget(self._slice_holder)
        component_row = QtWidgets.QHBoxLayout()
        component_row.addWidget(FluentLabel("Table"))
        self.component_combo = FluentComboBox()
        self.component_combo.setMinimumContentsLength(10)
        component_row.addWidget(self.component_combo)
        self.sigma_check = FluentCheckBox("Enable sigma")
        component_row.addWidget(self.sigma_check)
        component_row.addStretch(1)
        data_layout.addLayout(component_row)
        self.blank_help_label = muted_note_label("")
        data_layout.addWidget(self.blank_help_label)
        self.value_model = _VirtualTextTableModel(self)
        self.value_table = FluentTableView(self)
        self.value_table.setModel(self.value_model)
        self.value_table.horizontalHeader().setStretchLastSection(True)
        data_layout.addWidget(self.value_table, 1)
        self.message_label = muted_note_label("")
        self.message_label.setWordWrap(True)
        data_layout.addWidget(self.message_label)
        body_layout.addWidget(data_group, 1)
        root.addWidget(body, 1)

        self.name_edit.editingFinished.connect(lambda: self._dataset_field("name", self.name_edit.text()))
        self.dtype_combo.activated.connect(lambda _index: self._dataset_field("dtype", self.dtype_combo.currentData()))
        self.unit_edit.editingFinished.connect(lambda: self._dataset_field("unit", self.unit_edit.text()))
        self.note_edit.editingFinished.connect(lambda: self._dataset_field("note", self.note_edit.text()))
        self.axis_combo.activated.connect(lambda _index: self._emit("select_axis", axis_id=self.axis_combo.currentData()))
        self.add_axis_button.clicked.connect(
            lambda: self._emit("add_axis", domain=self.add_axis_domain_combo.currentData())
        )
        self.remove_axis_button.clicked.connect(lambda: self._emit("remove_axis", axis_id=self._selected_axis))
        self.axis_up_button.clicked.connect(lambda: self._emit("move_axis", axis_id=self._selected_axis, delta=-1))
        self.axis_down_button.clicked.connect(lambda: self._emit("move_axis", axis_id=self._selected_axis, delta=1))
        self.axis_name_edit.editingFinished.connect(lambda: self._axis_field("name", self.axis_name_edit.text()))
        self.role_combo.activated.connect(lambda _index: self._axis_field("role", self.role_combo.currentData()))
        self.axis_size_spin.valueChanged.connect(
            lambda value: self._axis_field("size", int(value))
        )
        self.value_kind_combo.activated.connect(
            lambda _index: self._axis_field(
                "value_kind", self.value_kind_combo.currentData()
            )
        )
        self.axis_unit_edit.editingFinished.connect(lambda: self._axis_field("unit", self.axis_unit_edit.text()))
        self.frame_edit.editingFinished.connect(lambda: self._axis_field("coordinate_frame", self.frame_edit.text()))
        self.insert_coordinate_button.clicked.connect(self._insert_coordinate)
        self.remove_coordinate_button.clicked.connect(self._remove_coordinates)
        self.coordinate_up_button.clicked.connect(lambda: self._move_coordinate(-1))
        self.coordinate_down_button.clicked.connect(lambda: self._move_coordinate(1))
        self.coordinate_model.edits_requested.connect(
            lambda cells: self._emit("set_coordinates", axis_id=self._selected_axis, cells=cells)
        )
        self.coordinate_model.rejected.connect(self._show_local_rejection)
        self.component_combo.activated.connect(
            lambda _index: self._emit("set_component", component=self.component_combo.currentData())
        )
        self.sigma_check.toggled.connect(lambda enabled: self._emit("toggle_sigma", enabled=bool(enabled)))
        self.value_model.edits_requested.connect(
            lambda cells: self._emit("set_cells", component=self.component_combo.currentData(), cells=cells)
        )
        self.value_model.rejected.connect(self._show_local_rejection)
        self.apply_button.clicked.connect(lambda: self._emit("apply_preview", note=self.note_edit.text()))
        self.save_button.clicked.connect(self._save_as)
        self.discard_button.clicked.connect(lambda: self._emit("discard"))
        self.update_projection(projection)

    def is_dirty(self) -> bool:
        return bool(getattr(self, "_dirty", False))

    def _emit(self, operation: str, **values: object) -> None:
        if not self._updating:
            self.intent_requested.emit({"op": str(operation), **values})

    def _dataset_field(self, field: str, value: object) -> None:
        self._emit("set_dataset_field", field=str(field), value=value)

    def _show_local_rejection(self, message: str) -> None:
        self.message_label.setText(str(message))

    def _axis_field(self, field: str, value: object) -> None:
        if self._selected_axis:
            self._emit("set_axis_field", axis_id=self._selected_axis, field=str(field), value=value)

    def _coordinate_rows(self) -> tuple[int, ...]:
        return tuple(sorted({index.row() for index in self.coordinate_table.selectionModel().selectedIndexes()}))

    def _insert_coordinate(self) -> None:
        rows = self._coordinate_rows()
        self._emit("insert_coordinate", axis_id=self._selected_axis, after=(rows[-1] if rows else -1))

    def _remove_coordinates(self) -> None:
        self._emit("remove_coordinates", axis_id=self._selected_axis, indices=self._coordinate_rows())

    def _move_coordinate(self, delta: int) -> None:
        rows = self._coordinate_rows()
        if len(rows) == 1:
            self._emit("move_coordinate", axis_id=self._selected_axis, index=rows[0], delta=int(delta))

    def _save_as(self) -> None:
        path = fluent_save_path(
            self,
            "Save edited figure",
            self._save_suggested,
            "Saved figure archives (*.npz)",
        )
        if path:
            self._emit("save_as", path=path, note=self.note_edit.text())

    def _set_slice_controls(self, rows: object) -> None:
        while self._slice_layout.count():
            item = self._slice_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._slice_widgets.clear()
        for position, raw in enumerate(tuple(rows or ())):
            row = dict(raw)
            axis_id = str(row.get("axis_id", ""))
            label = FluentLabel(str(row.get("label", axis_id)))
            size = max(0, int(row.get("size", 0)))
            spin = FluentSpinBox()
            spin.setRange(0, max(0, size - 1))
            spin.setValue(min(max(0, int(row.get("index", 0))), max(0, size - 1)))
            spin.setEnabled(size > 1)
            coordinate = FluentReadoutEdit(str(row.get("current_label", "")))
            coordinate.setToolTip("Coordinate / label at this slice index")
            spin.valueChanged.connect(
                lambda index, aid=axis_id: self._emit(
                    "set_slice", axis_id=aid, index=int(index)
                )
            )
            pair = QtWidgets.QWidget(self._slice_holder)
            pair.setStyleSheet("background: transparent;")
            pair_layout = QtWidgets.QHBoxLayout(pair)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            pair_layout.setSpacing(window_pad(0.25))
            pair_layout.addWidget(label)
            pair_layout.addWidget(spin)
            pair_layout.addWidget(coordinate, 1)
            self._slice_layout.addWidget(pair, position, 0)
            self._slice_widgets[axis_id] = (pair, spin, coordinate)
        self._slice_holder.setVisible(bool(self._slice_widgets))

    def update_projection(self, projection: object) -> None:
        data = dict(projection or {})
        dataset = dict(data.get("dataset", {}))
        axes = tuple(dict(axis) for axis in tuple(data.get("axes", ())))
        selected_axis = str(data.get("selected_axis", ""))
        selected = next((axis for axis in axes if str(axis.get("id", "")) == selected_axis), None)
        if selected is None and axes:
            selected = axes[0]
            selected_axis = str(selected.get("id", ""))
        self._updating = True
        try:
            self.name_edit.setText(str(dataset.get("name", "")))
            _fill_choice_combo(self.dtype_combo, dataset.get("dtype_choices", ()), dataset.get("dtype"))
            self.unit_edit.setText(str(dataset.get("unit", "")))
            self.note_edit.setText(str(dataset.get("note", "")))
            self.source_note.setText(str(dataset.get("source", "")))
            self._save_suggested = str(data.get("save_suggested", "figure.npz"))

            axis_rows = tuple(
                (
                    str(axis.get("id", "")),
                    f"{axis.get('domain_label', axis.get('domain', 'Axis'))} · {axis.get('name', axis.get('id', ''))}",
                )
                for axis in axes
            )
            _fill_choice_combo(self.axis_combo, axis_rows, selected_axis)
            self._selected_axis = selected_axis
            domain_choices = data.get("domain_choices", ())
            current_domain = "" if selected is None else str(
                selected.get("domain_label", selected.get("domain", ""))
            )
            self.domain_readout.setText(current_domain)
            _fill_choice_combo(
                self.add_axis_domain_combo,
                domain_choices,
                data.get("new_axis_domain", "cell"),
            )
            role_choices = () if selected is None else selected.get("role_choices", ())
            _fill_choice_combo(self.role_combo, role_choices, None if selected is None else selected.get("role"))
            value_kind_choices = (
                () if selected is None else selected.get("value_kind_choices", ())
            )
            _fill_choice_combo(
                self.value_kind_combo,
                value_kind_choices,
                None if selected is None else selected.get("value_kind"),
            )
            self._axis_rows["value_kind"].setVisible(
                bool(selected is not None and selected.get("show_value_kind", False))
            )
            self.axis_name_edit.setText("" if selected is None else str(selected.get("name", "")))
            self.axis_size_spin.setValue(
                1 if selected is None else max(1, int(selected.get("size", 1)))
            )
            self.axis_unit_edit.setText("" if selected is None else str(selected.get("unit", "")))
            self.frame_edit.setText("" if selected is None else str(selected.get("coordinate_frame", "")))
            coordinates = dict(data.get("coordinates", {}))
            self.coordinate_model.set_projection(coordinates)
            if self.coordinate_model.columnCount() >= 2:
                self.coordinate_table.setColumnWidth(
                    0, scaled_px(120, minimum=90)
                )
            self._set_slice_controls(data.get("slices", ()))
            table = dict(data.get("table", {}))
            _fill_choice_combo(
                self.component_combo,
                table.get("component_choices", ()),
                table.get("component"),
            )
            with signals_blocked(self.sigma_check):
                self.sigma_check.setChecked(bool(table.get("sigma_enabled", False)))
            self.value_model.set_projection(table)
            self.blank_help_label.setText(str(table.get("blank_help", "")))
            self.message_label.setText(str(data.get("message", "")))
            self._dirty = bool(data.get("dirty", False))
            self.save_button.set_dirty(self._dirty)
            self.discard_button.setEnabled(self._dirty)
            has_axis = selected is not None
            for control in (
                self.remove_axis_button,
                self.axis_up_button,
                self.axis_down_button,
                self.axis_name_edit,
                self.axis_size_spin,
                self.role_combo,
                self.value_kind_combo,
                self.frame_edit,
                self.coordinate_table,
                self.insert_coordinate_button,
                self.remove_coordinate_button,
                self.coordinate_up_button,
                self.coordinate_down_button,
            ):
                control.setEnabled(has_axis)
            self.axis_unit_edit.setEnabled(
                bool(has_axis and selected.get("unit_enabled", True))
                if selected is not None
                else False
            )
            self.apply_button.setEnabled(bool(data.get("can_apply", True)))
            self.save_button.setEnabled(bool(data.get("can_save", False)))
        finally:
            self._updating = False


class FigureViewerView(QtWidgets.QWidget):
    path_committed = QtCore.pyqtSignal(str)
    new_data_requested = QtCore.pyqtSignal()
    edit_data_requested = QtCore.pyqtSignal(str)
    data_editor_intent = QtCore.pyqtSignal(str, object)
    data_editor_closed = QtCore.pyqtSignal(str)
    add_panel_requested = QtCore.pyqtSignal(str)
    panel_state_changed = QtCore.pyqtSignal(str, object)
    panel_remove_requested = QtCore.pyqtSignal(str)
    panel_edit_requested = QtCore.pyqtSignal(str)
    panel_order_committed = QtCore.pyqtSignal(tuple)
    panel_editor_closed = QtCore.pyqtSignal(str)
    panel_plot_error = QtCore.pyqtSignal(str, str)
    save_image_requested = QtCore.pyqtSignal()
    close_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None, *, path_base_dir: str = "") -> None:
        super().__init__(parent)
        self.setObjectName("FigureViewerView")
        self.setStyleSheet("background: transparent;")
        self._cards: dict[str, PanelCardView] = {}
        self._editors: dict[str, QtWidgets.QWidget] = {}
        self._data_editors: dict[str, _DataEditorView] = {}
        self._panel_sizes: tuple[str, ...] = ()
        self._panel_default_size = ""
        self._panel_intervals: tuple[int, ...] = ()
        self._panel_default_interval = 0
        self._grid_cell_kinds: tuple[str, ...] = ()
        self._closing = False
        self._info_tabs: tuple = ()
        self._flow_graph: object = {"nodes": (), "edges": ()}
        root = QtWidgets.QHBoxLayout(self)
        # InfoPane owns the left/right window inset.  The host supplies only
        # the top/bottom frame.
        root.setContentsMargins(0, window_pad(1), 0, window_pad(1))
        root.setSpacing(window_pad(0.5))

        self.info_pane = InfoPane(
            # These formal projection keys also determine the fixed left-pane
            # width, so using invented labels
            # changes the entire FigureViewer split even while the pane looks
            # otherwise correct.
            label_names=("schema_fingerprint", "coordinate_frame"),
            tabs=(
                ("Plot", ()),
                ("Logic", ()),
                ("Devices", ()),
                ("Flow", ()),
                ("Raw", ()),
            ),
            path_label="File",
            path_caption="Open a saved figure archive (.npz)",
            file_filter="Saved figure archives (*.npz)",
            path_base_dir=str(path_base_dir),
            initial_status="Open a current saved Figure (.npz).",
            graph_tabs=("Flow",),
            parent=self,
        )
        self.info_pane.path_committed.connect(self.path_committed)
        root.addWidget(self.info_pane, 0)

        # The right half is one white Fluent work surface.  Cards remain
        # visibly separate inside it instead of dissolving into the grey
        # top-level background.
        holder = FluentFrame(parent=self, bordered=False)
        holder.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._surface_layout = QtWidgets.QVBoxLayout(holder)
        self._surface_layout.setContentsMargins(
            window_pad(0.75), window_pad(0.5), window_pad(0.75), window_pad(0.75)
        )
        self._surface_layout.setSpacing(window_pad(0.5))

        self._panel_bar = QtWidgets.QWidget(holder)
        self._panel_bar.setStyleSheet("background: transparent;")
        self._panel_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        bar_layout = QtWidgets.QVBoxLayout(self._panel_bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(window_pad(0.25))
        data_row = QtWidgets.QHBoxLayout()
        data_row.setContentsMargins(0, 0, 0, 0)
        data_row.setSpacing(window_pad(0.5))
        data_row.addWidget(FluentSectionLabel("Data"))
        self.data_combo = FluentComboBox()
        self.data_combo.setMinimumContentsLength(15)
        self.data_combo.setEnabled(False)
        data_row.addWidget(self.data_combo, 1)
        self.new_data_button = FluentButton("New data", color=ACCENT)
        self.new_data_button.clicked.connect(self.new_data_requested)
        data_row.addWidget(self.new_data_button)
        self.edit_data_button = FluentButton("Edit data", color=ACCENT)
        self.edit_data_button.setEnabled(False)
        self.edit_data_button.clicked.connect(self._edit_selected_data)
        data_row.addWidget(self.edit_data_button)
        bar_layout.addLayout(data_row)
        panel_row = QtWidgets.QHBoxLayout()
        panel_row.setContentsMargins(0, 0, 0, 0)
        panel_row.setSpacing(window_pad(0.5))
        panel_row.addWidget(FluentSectionLabel("Panels"))
        self.kind_combo = FluentComboBox()
        self.kind_combo.setMinimumContentsLength(12)
        panel_row.addWidget(self.kind_combo, 1)
        self.add_panel_button = FluentButton("Add panel", color=ACCENT)
        self.add_panel_button.clicked.connect(self._add_selected_panel)
        self.add_panel_button.setEnabled(False)
        panel_row.addWidget(self.add_panel_button)
        self.save_image_button = FluentButton("Save image", color=ACCENT)
        self.save_image_button.clicked.connect(self.save_image_requested)
        self.save_image_button.setEnabled(False)
        panel_row.addWidget(self.save_image_button)
        bar_layout.addLayout(panel_row)
        self._surface_layout.addWidget(self._panel_bar)

        self.tabs = FluentTabWidget(holder)
        self.tabs.tab_close_requested.connect(self._tab_close_clicked)
        monitor = QtWidgets.QWidget()
        monitor.setStyleSheet("background: white;")
        monitor_layout = QtWidgets.QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(0, 0, 0, 0)
        self.board = ConsoleBoardView()
        self.board.order_committed.connect(self.panel_order_committed)
        self.scroll = FluentScrollArea()
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setWidget(self.board)
        monitor_layout.addWidget(self.scroll, 1)
        self.tabs.add_permanent_tab(monitor, "Monitor")
        self._surface_layout.addWidget(self.tabs, 1)

        self._placeholder = FluentFrame(parent=holder)
        placeholder_layout = QtWidgets.QVBoxLayout(self._placeholder)
        placeholder_layout.addStretch(1)
        placeholder_label = FluentLabel("Open a saved figure to begin", self._placeholder)
        placeholder_label.setAlignment(QtCore.Qt.AlignCenter)
        placeholder_layout.addWidget(placeholder_label)
        placeholder_layout.addStretch(1)
        monitor_layout.addWidget(self._placeholder, 1)
        root.addWidget(holder, 1)
        self._sync_monitor_empty()

    def set_archive_info(
        self,
        tabs: tuple[tuple[str, tuple[tuple[str, object], ...]], ...],
        graph: object,
    ) -> None:
        """Replace one archive's rows and Flow in the same owner turn."""

        previous_tabs = self._info_tabs
        previous_graph = self._flow_graph
        self._info_tabs = tuple(tabs)
        self._flow_graph = graph
        try:
            self._render_info()
        except BaseException:
            self._info_tabs = previous_tabs
            self._flow_graph = previous_graph
            self._render_info()
            raise

    def _render_info(self) -> None:
        rows = dict(self._info_tabs)
        self.info_pane.set_tabs(tuple(
            (title, () if title == "Flow" else tuple(rows.get(title, ())))
            for title in ("Plot", "Logic", "Devices", "Flow", "Raw")
        ))
        self.info_pane.set_graph("Flow", self._flow_graph)

    def set_panel_kinds(self, kinds: object, default_kind: str = "") -> None:
        rows = tuple((str(key), str(label or key)) for key, label in tuple(kinds))
        current = self.kind_combo.currentData()
        self.kind_combo.clear()
        for key, label in rows:
            self.kind_combo.addItem(label, key)
        if current is not None:
            index = self.kind_combo.findData(current)
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
        if self.kind_combo.currentIndex() < 0 and default_kind:
            index = self.kind_combo.findData(str(default_kind))
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
        self.add_panel_button.setEnabled(bool(rows))

    def set_editable_data_choices(
        self, choices: object, *, current: str = ""
    ) -> None:
        """Project archive/manual Dataset choices without interpreting them."""

        rows = tuple((str(key), str(label or key)) for key, label in tuple(choices))
        selected = str(current or self.data_combo.currentData() or "")
        _fill_choice_combo(self.data_combo, rows, selected)
        enabled = bool(rows)
        self.data_combo.setEnabled(enabled)
        self.edit_data_button.setEnabled(enabled)

    def _edit_selected_data(self) -> None:
        key = self.data_combo.currentData()
        if isinstance(key, str) and key:
            self.edit_data_requested.emit(key)

    def set_panel_sizes(self, sizes: object, default_size: str) -> None:
        self._panel_sizes = tuple(str(value) for value in tuple(sizes))
        self._panel_default_size = str(default_size)
        for card in self._cards.values():
            card.set_size_choices(self._panel_sizes, self._panel_default_size)

    def set_panel_intervals(
        self, intervals: object, default_interval: int
    ) -> None:
        self._panel_intervals = tuple(int(value) for value in tuple(intervals))
        self._panel_default_interval = int(default_interval)
        for card in self._cards.values():
            card.set_interval_choices(
                self._panel_intervals,
                self._panel_default_interval,
            )

    def set_grid_cell_kinds(self, kinds: object) -> None:
        self._grid_cell_kinds = tuple(str(value) for value in tuple(kinds))
        for card in self._cards.values():
            card.set_cell_kind_choices(self._grid_cell_kinds)

    def _add_selected_panel(self) -> None:
        kind = self.kind_combo.currentData()
        if isinstance(kind, str):
            self.add_panel_requested.emit(kind)

    def add_panel(self, panel_id: str, title: str) -> None:
        key = str(panel_id)
        if key in self._cards:
            return
        if not self._panel_sizes:
            raise RuntimeError("FigureViewer panel sizes were not projected")
        card = PanelCardView(key, str(title))
        card.set_size_choices(self._panel_sizes, self._panel_default_size)
        if self._panel_intervals:
            card.set_interval_choices(
                self._panel_intervals,
                self._panel_default_interval,
            )
        if self._grid_cell_kinds:
            card.set_cell_kind_choices(self._grid_cell_kinds)
        card.remove_requested.connect(
            lambda _=None, pid=key: self.panel_remove_requested.emit(pid)
        )
        card.edit_requested.connect(
            lambda _=None, pid=key: self.panel_edit_requested.emit(pid)
        )
        card.state_changed.connect(
            lambda patch, pid=key: self.panel_state_changed.emit(pid, patch)
        )
        card.plot_error.connect(
            lambda message, pid=key: self.panel_plot_error.emit(pid, str(message))
        )
        self._cards[key] = card
        self.board.set_cards(tuple(self._cards.values()))
        self._sync_monitor_empty()

    def remove_panel(self, panel_id: str) -> None:
        key = str(panel_id)
        self.close_panel_editor(key)
        self._cards.pop(key, None)
        self.board.set_cards(tuple(self._cards.values()))
        self._sync_monitor_empty()

    def set_panel_order(self, order: object) -> None:
        wanted = [str(key) for key in tuple(order) if str(key) in self._cards]
        wanted += [key for key in self._cards if key not in wanted]
        self._cards = {key: self._cards[key] for key in wanted}
        self.board.set_cards(tuple(self._cards.values()))

    def set_panel_signal_choices(
        self,
        panel_id: str,
        groups: object,
        *,
        current: str = "",
        overlay_groups: object = (),
        overlay_current: str = "",
    ) -> None:
        self._cards[str(panel_id)].set_signal_choices(
            groups,
            current=str(current),
            overlay_groups=overlay_groups,
            overlay_current=str(overlay_current),
        )

    def set_panel_publishers(self, _publishers: object) -> None:
        """FigureViewer has no Logic-row chrome; signals remain in Setting."""

    def panel_ids(self) -> tuple[str, ...]:
        return tuple(self._cards)

    def set_panel_selectors_enabled(self, panel_id: str, enabled: bool) -> None:
        self._cards[str(panel_id)].set_selectors_enabled(bool(enabled))

    def set_panel_mutation_enabled(self, panel_id: str, enabled: bool) -> None:
        self._cards[str(panel_id)].set_editing_enabled(bool(enabled))

    def present_panel_front(self, panel_id: str, front: object) -> bool:
        surface = self._cards[str(panel_id)].surface
        present = getattr(surface, "present_front", None)
        return bool(callable(present) and present(front))

    def set_panel_projection(
        self, panel_id: str, state: object, surface: object
    ) -> None:
        self._cards[str(panel_id)].set_panel_projection(state, surface)

    def set_panel_surface(
        self, panel_id: str, widget: QtWidgets.QWidget | None
    ) -> None:
        if widget is not None and not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("figure surface must be QWidget or None")
        self._cards[str(panel_id)].set_surface(widget)

    def set_panel_status(self, panel_id: str, text: str, *, error: bool) -> None:
        self._cards[str(panel_id)].set_status(str(text), error=bool(error))

    def _sync_monitor_empty(self) -> None:
        has_panels = bool(self._cards)
        self.scroll.setVisible(has_panels)
        self._placeholder.setVisible(not has_panels)
        self.save_image_button.setEnabled(has_panels)

    def open_panel_editor(
        self, panel_id: str, projection: object, title: str
    ) -> None:
        key = str(panel_id)
        existing = self._editors.get(key)
        if existing is not None:
            self.tabs.setCurrentWidget(existing)
            return
        incoming = dict(projection)
        incoming["size_choices"] = self._panel_sizes
        incoming["interval_choices"] = self._panel_intervals
        editor = PanelEditorView(key, incoming)
        editor.state_changed.connect(
            lambda patch, pid=key: self.panel_state_changed.emit(pid, patch)
        )
        self._editors[key] = editor
        self.tabs.add_closable_tab(editor, str(title), focus=True)

    def open_data_editor(
        self, editor_id: str, projection: object, title: str
    ) -> None:
        key = str(editor_id)
        existing = self._data_editors.get(key)
        if existing is not None:
            existing.update_projection(projection)
            self.tabs.setCurrentWidget(existing)
            return
        editor = _DataEditorView(projection, self)
        editor.intent_requested.connect(
            lambda intent, eid=key: self.data_editor_intent.emit(eid, intent)
        )
        self._data_editors[key] = editor
        editor.setProperty("data_editor_title", str(title))
        self.tabs.add_closable_tab(editor, str(title), focus=True)
        self._sync_data_editor_title(editor)

    def close_data_editor(self, editor_id: str) -> bool:
        key = str(editor_id)
        editor = self._data_editors.pop(key, None)
        if editor is None:
            return False
        index = self.tabs.indexOf(editor)
        if index >= 0:
            self.tabs.removeTab(index)
        editor.setParent(None)
        editor.deleteLater()
        return True

    def update_data_editor(self, editor_id: str, projection: object) -> bool:
        editor = self._data_editors.get(str(editor_id))
        if editor is None:
            return False
        document = dict(projection or {})
        dataset = dict(document.get("dataset") or {})
        name = str(dataset.get("name") or "").strip()
        if name:
            editor.setProperty("data_editor_title", f"Data · {name}")
        editor.update_projection(projection)
        self._sync_data_editor_title(editor)
        return True

    def _sync_data_editor_title(self, editor: _DataEditorView) -> None:
        index = self.tabs.indexOf(editor)
        if index < 0:
            return
        base = str(editor.property("data_editor_title") or "Data")
        self.tabs.setTabText(index, base + (" *" if editor.is_dirty() else ""))

    def has_data_editor(self, editor_id: str) -> bool:
        return str(editor_id) in self._data_editors

    def focus_data_editor(self, editor_id: str) -> bool:
        editor = self._data_editors.get(str(editor_id))
        if editor is None:
            return False
        self.tabs.setCurrentWidget(editor)
        return True

    def close_panel_editor(self, panel_id: str) -> bool:
        key = str(panel_id)
        editor = self._editors.pop(key, None)
        if editor is None:
            return False
        index = self.tabs.indexOf(editor)
        if index >= 0:
            self.tabs.removeTab(index)
        editor.setParent(None)
        editor.deleteLater()
        return True

    def update_panel_editor(self, panel_id: str, projection: object) -> bool:
        editor = self._editors.get(str(panel_id))
        if not isinstance(editor, PanelEditorView):
            return False
        incoming = dict(projection)
        incoming["size_choices"] = self._panel_sizes
        incoming["interval_choices"] = self._panel_intervals
        editor.update_projection(incoming)
        return True

    def has_panel_editor(self, panel_id: str) -> bool:
        return str(panel_id) in self._editors

    def show_panel_editor(
        self, panel_id: str, widget: QtWidgets.QWidget | None
    ) -> None:
        editor = self._editors[str(panel_id)]
        assert isinstance(editor, PanelEditorView)
        editor.set_surface(widget)

    def focus_panel_editor(self, panel_id: str) -> bool:
        editor = self._editors.get(str(panel_id))
        if editor is None:
            return False
        self.tabs.setCurrentWidget(editor)
        return True

    def _tab_close_clicked(self, editor: QtWidgets.QWidget) -> None:
        data_editor_id = next(
            (key for key, value in self._data_editors.items() if value is editor),
            "",
        )
        if data_editor_id:
            self.data_editor_closed.emit(data_editor_id)
            return
        panel_id = next(
            (key for key, value in self._editors.items() if value is editor),
            "",
        )
        if panel_id:
            self.panel_editor_closed.emit(panel_id)

    def set_status(self, text: str, *, error: bool = False) -> None:
        if error:
            self.info_pane.status.show_message(str(text), severity="error")
        else:
            self.info_pane.set_status(str(text))

    def set_title(self, text: str) -> None:
        self.setWindowTitle(str(text))

    def set_path(self, path: str) -> None:
        """Show which file is open.

        A viewer whose File field stays empty after opening something is a
        viewer you cannot tell apart from one that opened nothing -- and the
        first thing anyone checks when a figure looks wrong is which file it is.
        """

        self.info_pane.path_edit.blockSignals(True)
        self.info_pane.path_edit.setText(str(path))
        self.info_pane.path_edit.blockSignals(False)

    def finish_close(self) -> None:
        self._closing = True
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._closing:
            event.ignore()
            self.close_requested.emit()
            return
        super().closeEvent(event)


__all__ = ["FigureViewerView"]
