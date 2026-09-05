"""Pure saved-figure browser shell.

The presenter owns archive IO, metadata projection and the plot widget.  This
view only provides the file/path intent, generic info projection, and an
atomic QWidget mount point for the presenter-owned surface.
"""

from __future__ import annotations

import math

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.console.board_view import ConsoleBoardView
from zlc_ui.console.panel_card_view import PanelCardView, data_structure_fragments
from zlc_ui.console.panel_editor_view import PanelEditorView
from zlc_ui.fluent import (
    retire_widget,
    ACCENT,
    GREY,
    ORANGE_TINT,
    FluentButton,
    FluentCheckBox,
    FluentComboBox,
    FluentCycleComboBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
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


class _VirtualTextTableModel(QtCore.QAbstractTableModel):
    """Visible-cell-only editable view over a presenter-owned 2-D projection."""

    edits_requested = QtCore.pyqtSignal(object)
    rejected = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._shape = (0, 0)
        self._values: object = ()
        self._validity: object | None = None
        self._row_headers: object = ()
        self._column_headers: object = ()
        self._editable = True
        self._finite_values = False
        self._blank_hint = ""
        self._pending: dict[tuple[int, int], str] = {}

    def set_projection(self, projection: object) -> None:
        data = dict(projection or {})
        shape = tuple(int(value) for value in tuple(data.get("shape", (0, 0))))
        if len(shape) != 2 or any(value < 0 for value in shape):
            raise ValueError("table projection shape must contain two nonnegative sizes")
        reset = shape != self._shape
        if reset:
            self.beginResetModel()
        self._shape = shape
        self._values = data.get("values", ())
        self._validity = data.get("validity")
        self._row_headers = data.get("row_headers", ())
        self._column_headers = data.get("column_headers", ())
        self._editable = bool(data.get("editable", True))
        self._finite_values = bool(data.get("finite_values", False))
        self._blank_hint = str(data.get("blank_hint", ""))
        self._pending.clear()
        if reset:
            self.endResetModel()
            return
        if shape[0] and shape[1]:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(shape[0] - 1, shape[1] - 1),
                (
                    QtCore.Qt.DisplayRole,
                    QtCore.Qt.EditRole,
                    QtCore.Qt.BackgroundRole,
                    QtCore.Qt.ToolTipRole,
                ),
            )
        self.headerDataChanged.emit(QtCore.Qt.Horizontal, 0, max(0, shape[1] - 1))
        self.headerDataChanged.emit(QtCore.Qt.Vertical, 0, max(0, shape[0] - 1))

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
        value = _matrix_item(self._values, row, column)
        if self._finite_values:
            try:
                if not math.isfinite(float(value)):
                    return ""
            except (TypeError, ValueError):
                return ""
        return _plain_scalar(value)

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
        if str(value) == self._projected_text(index.row(), index.column()):
            self._pending.pop((index.row(), index.column()), None)
            return True
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
        self._adding_axis = False
        self._domain_choices: tuple = ()
        self._save_suggested = "figure.npz"
        self._axis_view_widgets: dict[
            str,
            tuple[
                QtWidgets.QWidget,
                FluentLabel,
                FluentCycleComboBox,
            ],
        ] = {}

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.editor_scroll = FluentScrollArea(self)
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setMinimumSize(0, 0)
        content = QtWidgets.QWidget()
        content.setStyleSheet("background: transparent;")
        root = QtWidgets.QVBoxLayout(content)
        root.setContentsMargins(
            window_pad(0.5), window_pad(0.5), window_pad(0.5), window_pad(0.5)
        )
        root.setSpacing(window_pad(0.5))
        self.editor_scroll.setWidget(content)
        outer.addWidget(self.editor_scroll)

        dataset_group = FluentGroupBox("Dataset", self)
        self.dataset_group = dataset_group
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
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(window_pad(0.5))

        axes_group = FluentGroupBox("Axes", body)
        self.axes_group = axes_group
        axes_layout = QtWidgets.QVBoxLayout(axes_group)
        axes_layout.setContentsMargins(window_pad(0.75), window_pad(0.75), window_pad(0.75), window_pad(0.6))
        axes_layout.setSpacing(window_pad(0.35))
        axis_label_width = setting_label_width(
            ("Name", "Length", "Unit", "Domain", "Editing"),
            minimum=68,
        )
        # Creating an axis and choosing which one to edit are two different
        # sentences.  Sharing a row said they were one control with three
        # buttons, so the row above is the CREATE action and the row below
        # is the chooser -- labelled, because every other control in this
        # form says what it is.
        add_axis_row = QtWidgets.QHBoxLayout()
        add_axis_row.setSpacing(window_pad(0.25))
        self.add_axis_button = FluentButton("Add axis", color=ACCENT)
        self.add_axis_button.setToolTip("Create a new axis")
        add_axis_row.addStretch(1)
        add_axis_row.addWidget(self.add_axis_button)
        axes_layout.addLayout(add_axis_row)
        axis_choice_row = QtWidgets.QHBoxLayout()
        axis_choice_row.setSpacing(window_pad(0.25))
        self.axis_combo = FluentComboBox()
        self.axis_combo.setMinimumContentsLength(13)
        self.remove_axis_button = FluentButton("Delete", color=GREY)
        self.remove_axis_button.setToolTip(
            "Delete selected axis and keep its current Scope slice"
        )
        self.axis_choice_row = FluentSettingRow(
            "Editing", self.axis_combo, label_width=axis_label_width
        )
        axis_choice_row.addWidget(self.axis_choice_row, 1)
        axis_choice_row.addWidget(self.remove_axis_button)
        axes_layout.addLayout(axis_choice_row)
        self.domain_combo = FluentComboBox()
        self.axis_size_spin = FluentSpinBox()
        self.axis_size_spin.setRange(1, 2_147_483_647)
        self.axis_size_spin.setKeyboardTracking(False)
        self.axis_name_edit = FluentLineEdit()
        self.axis_unit_edit = FluentLineEdit()
        self._axis_rows: dict[str, FluentSettingRow] = {}
        axis_form = QtWidgets.QGridLayout()
        axis_form.setContentsMargins(0, 0, 0, 0)
        axis_form.setHorizontalSpacing(window_pad(0.35))
        axis_form.setVerticalSpacing(window_pad(0.25))
        for field, label, control, row_index, column_index in (
            ("name", "Name", self.axis_name_edit, 0, 0),
            ("size", "Length", self.axis_size_spin, 0, 1),
            ("unit", "Unit", self.axis_unit_edit, 1, 0),
            ("domain", "Domain", self.domain_combo, 1, 1),
        ):
            row = FluentSettingRow(label, control, label_width=axis_label_width)
            self._axis_rows[field] = row
            axis_form.addWidget(row, row_index, column_index)
        axes_layout.addLayout(axis_form)
        axes_layout.addWidget(FluentSectionLabel("Axis values"))
        self.axis_value_model = _VirtualTextTableModel(self)
        self.axis_value_table = FluentTableView(self)
        self.axis_value_table.setModel(self.axis_value_model)
        self.axis_value_table.setTabKeyNavigation(True)
        self.axis_value_table.horizontalHeader().setStretchLastSection(False)
        self.axis_value_table.setMinimumHeight(scaled_px(72, minimum=62))
        self.axis_value_table.setMaximumHeight(scaled_px(82, minimum=68))
        axes_layout.addWidget(self.axis_value_table)
        axis_actions = QtWidgets.QHBoxLayout()
        self.axis_mode_label = muted_note_label("Edit selected axis")
        axis_actions.addWidget(self.axis_mode_label, 1)
        self.apply_axis_button = FluentButton("Apply axis", color=ACCENT)
        axis_actions.addWidget(self.apply_axis_button)
        axes_layout.addLayout(axis_actions)
        body_layout.addWidget(axes_group, 0)

        data_group = FluentGroupBox("Data", body)
        self.data_group = data_group
        data_layout = QtWidgets.QVBoxLayout(data_group)
        data_layout.setContentsMargins(window_pad(0.75), window_pad(0.75), window_pad(0.75), window_pad(0.6))
        data_layout.setSpacing(window_pad(0.35))
        self.structure_label = FluentLabel("")
        self.structure_label.setTextFormat(QtCore.Qt.RichText)
        self.structure_label.setWordWrap(True)
        data_layout.addWidget(self.structure_label)
        self._axis_view_holder = QtWidgets.QWidget(data_group)
        self._axis_view_holder.setStyleSheet("background: transparent;")
        self._axis_view_layout = QtWidgets.QGridLayout(self._axis_view_holder)
        self._axis_view_layout.setContentsMargins(0, 0, 0, 0)
        self._axis_view_layout.setHorizontalSpacing(window_pad(0.35))
        self._axis_view_layout.setVerticalSpacing(window_pad(0.2))
        data_layout.addWidget(self._axis_view_holder)
        # One sentence, one row: what the table below is showing, and the
        # switch that decides whether "sigma" is even on offer.  A setting
        # row is a width CONSUMER, so the pair travels inside its control
        # slot rather than beside it in a box that would collapse it.
        self.component_combo = FluentComboBox()
        self.component_combo.setMinimumContentsLength(10)
        self.sigma_check = FluentCheckBox("Enable sigma")
        component_controls = QtWidgets.QWidget(data_group)
        component_controls.setStyleSheet("background: transparent;")
        component_layout = QtWidgets.QHBoxLayout(component_controls)
        component_layout.setContentsMargins(0, 0, 0, 0)
        component_layout.setSpacing(window_pad(0.5))
        component_layout.addWidget(self.component_combo)
        component_layout.addWidget(self.sigma_check)
        component_layout.addStretch(1)
        data_layout.addWidget(
            FluentSettingRow(
                "Table shows",
                component_controls,
                label_width=setting_label_width(
                    ("Table shows",), minimum=68
                ),
            )
        )
        self.blank_help_label = muted_note_label("")
        data_layout.addWidget(self.blank_help_label)
        self.value_model = _VirtualTextTableModel(self)
        self.value_table = FluentTableView(self)
        self.value_table.setModel(self.value_model)
        self.value_table.setTabKeyNavigation(True)
        self.value_table.horizontalHeader().setStretchLastSection(True)
        self.value_table.setMinimumHeight(scaled_px(240, minimum=180))
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
        self.axis_combo.activated.connect(self._select_axis)
        self.add_axis_button.clicked.connect(self._begin_add_axis)
        self.remove_axis_button.clicked.connect(
            lambda: self._emit("delete_axis", axis_id=self._selected_axis)
        )
        self.apply_axis_button.clicked.connect(self._commit_axis)
        self.axis_value_model.edits_requested.connect(
            lambda cells: self._emit(
                "set_axis_values", axis_id=self._selected_axis, cells=cells
            )
        )
        self.axis_value_model.rejected.connect(self._show_local_rejection)
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

    def _select_axis(self, _index: int) -> None:
        self._adding_axis = False
        self._emit("select_axis", axis_id=self.axis_combo.currentData())

    def _begin_add_axis(self) -> None:
        self._adding_axis = True
        self._selected_axis = ""
        with signals_blocked(self.axis_combo):
            self.axis_combo.setCurrentIndex(-1)
        self.axis_name_edit.clear()
        self.axis_size_spin.setValue(1)
        self.axis_unit_edit.clear()
        self.axis_value_model.set_projection(
            {"shape": (1, 0), "values": ((),), "row_headers": ("Value",)}
        )
        if self.domain_combo.count():
            self.domain_combo.setCurrentIndex(0)
        self.axis_mode_label.setText("New axis")
        self.apply_axis_button.setText("Create axis")
        self.axis_name_edit.setFocus()

    def _commit_axis(self) -> None:
        name = self.axis_name_edit.text().strip()
        if not name:
            self._show_local_rejection("Axis name cannot be blank")
            return
        values = {
            "name": name,
            "length": int(self.axis_size_spin.value()),
            "unit": self.axis_unit_edit.text().strip(),
            "domain": self.domain_combo.currentData(),
        }
        if self._adding_axis:
            self._adding_axis = False
            self._emit("add_axis", **values)
        elif self._selected_axis:
            self._emit("edit_axis", axis_id=self._selected_axis, **values)

    def _save_as(self) -> None:
        path = fluent_save_path(
            self,
            "Save edited figure",
            self._save_suggested,
            "Saved figure archives (*.npz)",
        )
        if path:
            self._emit("save_as", path=path, note=self.note_edit.text())

    @staticmethod
    def _rich_text(value: object) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ", "&nbsp;")
        )

    def _set_structure(self, structure: object) -> None:
        shape, names = data_structure_fragments(structure)

        def html(fragments: object) -> str:
            return "".join(
                self._rich_text(text)
                if colour is None
                else f'<span style="color:{colour}">{self._rich_text(text)}</span>'
                for text, colour, _elide in tuple(fragments)
            )

        self.structure_label.setText(
            html(shape) + "<br>" + html(names)
        )

    def _axis_view_mode_changed(
        self, axis_id: str, combo: FluentCycleComboBox
    ) -> None:
        if combo.isCycleSelected():
            self._emit(
                "set_scope",
                axis_id=str(axis_id),
                index=combo.cyclePosition(),
            )
        else:
            self._emit(
                "set_table_axis",
                axis_id=str(axis_id),
                mode=combo.currentData(),
            )

    def _set_axis_view_controls(self, rows: object) -> None:
        projected = tuple(dict(raw) for raw in tuple(rows or ()))
        wanted = {str(row.get("axis_id", "")) for row in projected}
        for axis_id in tuple(self._axis_view_widgets):
            if axis_id in wanted:
                continue
            widget, *_unused = self._axis_view_widgets.pop(axis_id)
            self._axis_view_layout.removeWidget(widget)
            widget.deleteLater()
        for position, row in enumerate(projected):
            axis_id = str(row.get("axis_id", ""))
            controls = self._axis_view_widgets.get(axis_id)
            if controls is None:
                holder = QtWidgets.QWidget(self._axis_view_holder)
                holder.setStyleSheet("background: transparent;")
                layout = QtWidgets.QHBoxLayout(holder)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(window_pad(0.25))
                label = FluentLabel("")
                label.setMinimumWidth(scaled_px(150, minimum=110))
                mode = FluentCycleComboBox()
                mode.setMinimumContentsLength(9)
                layout.addWidget(label, 1)
                layout.addWidget(mode)
                mode.activated.connect(
                    lambda _index, aid=axis_id, control=mode: self._axis_view_mode_changed(
                        aid, control
                    )
                )
                controls = (holder, label, mode)
                self._axis_view_widgets[axis_id] = controls
            holder, label, mode = controls
            self._axis_view_layout.addWidget(holder, position, 0)
            unit = str(row.get("unit") or "")
            suffix = f" ({int(row.get('size', 1))})" + (f" [{unit}]" if unit else "")
            label.setText(str(row.get("name", axis_id)) + suffix)
            with signals_blocked(mode):
                mode.clear()
                mode.addItem("Rows ↓", "rows")
                mode.addItem("Columns →", "columns")
                mode.setCycleChoices("Scope", row["scope_choices"])
                selected_mode = str(row.get("mode", "scope"))
                if selected_mode == "scope":
                    mode.setCyclePosition(int(row.get("index", 0)))
                else:
                    mode.setCurrentIndex(mode.findData(selected_mode))
        self._axis_view_holder.setVisible(bool(projected))

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
            if self._adding_axis:
                self.axis_combo.setCurrentIndex(-1)
            self._selected_axis = selected_axis
            self._domain_choices = tuple(data.get("domain_choices", ()))
            if not self._adding_axis:
                _fill_choice_combo(
                    self.domain_combo,
                    self._domain_choices,
                    None if selected is None else selected.get("domain"),
                )
                self.axis_name_edit.setText(
                    "" if selected is None else str(selected.get("name", ""))
                )
                self.axis_size_spin.setValue(
                    1 if selected is None else max(1, int(selected.get("size", 1)))
                )
                self.axis_unit_edit.setText(
                    "" if selected is None else str(selected.get("unit", ""))
                )
                self.axis_mode_label.setText("Edit selected axis")
                self.apply_axis_button.setText("Apply axis")
            self.axis_value_model.set_projection(data.get("axis_values", {}))
            table = dict(data.get("table", {}))
            self._set_structure(table.get("structure", ()))
            self._set_axis_view_controls(table.get("axes", ()))
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
                self.axis_name_edit,
                self.axis_size_spin,
                self.axis_unit_edit,
                self.domain_combo,
                self.apply_axis_button,
            ):
                control.setEnabled(has_axis or self._adding_axis)
            self.remove_axis_button.setEnabled(has_axis and not self._adding_axis)
            self.axis_value_table.setEnabled(has_axis and not self._adding_axis)
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
    panel_snapshot_refresh_requested = QtCore.pyqtSignal(str)
    panel_save_figure_requested = QtCore.pyqtSignal(str, str)
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
        # The button beside a chooser acts ON what the chooser holds; the
        # one after it does something else entirely.  Both rows of this bar
        # read that way now -- Edit data / Add panel answer the box to their
        # left, New data and Save image do not -- so a creating action never
        # sits between a chooser and the thing that reads it.
        self.edit_data_button = FluentButton("Edit data", color=ACCENT)
        self.edit_data_button.setEnabled(False)
        self.edit_data_button.clicked.connect(self._edit_selected_data)
        data_row.addWidget(self.edit_data_button)
        self.new_data_button = FluentButton("New data", color=ACCENT)
        self.new_data_button.clicked.connect(self.new_data_requested)
        data_row.addWidget(self.new_data_button)
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
        key = str(panel_id)
        self._cards[key].set_selectors_enabled(bool(enabled))
        editor = self._editors.get(key)
        if isinstance(editor, PanelEditorView):
            editor.set_selectors_enabled(bool(enabled))

    def set_panel_mutation_enabled(self, panel_id: str, enabled: bool) -> None:
        key = str(panel_id)
        self._cards[key].set_editing_enabled(bool(enabled))
        editor = self._editors.get(key)
        if isinstance(editor, PanelEditorView):
            editor.set_mutation_enabled(bool(enabled))

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
            incoming = dict(projection)
            incoming["size_choices"] = self._panel_sizes
            incoming["interval_choices"] = self._panel_intervals
            existing.update_projection(incoming)
            self.tabs.setCurrentWidget(existing)
            return
        incoming = dict(projection)
        incoming["size_choices"] = self._panel_sizes
        incoming["interval_choices"] = self._panel_intervals
        editor = PanelEditorView(key, incoming)
        editor.state_changed.connect(
            lambda patch, pid=key: self.panel_state_changed.emit(pid, patch)
        )
        editor.snapshot_refresh_requested.connect(
            lambda _=None, pid=key: self.panel_snapshot_refresh_requested.emit(pid)
        )
        editor.save_figure_requested.connect(
            lambda path, pid=key: self.panel_save_figure_requested.emit(pid, str(path))
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
        retire_widget(editor)
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
        retire_widget(editor)
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

    def show_status(self, text: str, severity: str) -> None:
        """Present the same ConsolePresenter status channel as TaskConsole."""

        level = str(severity or "info")
        if level == "idle":
            level = "info"
        self.info_pane.status.show_message(str(text), severity=level)

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
