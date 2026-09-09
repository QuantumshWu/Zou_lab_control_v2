"""Derive's authored programs and the schemas they actually receive/publish.

This leaf edits plain draft values. It neither evaluates programs nor asks a
device or Runtime to materialize data on the Qt thread.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

from PyQt5 import QtCore, QtWidgets

from zlc_plot.semantics import schema_structure
from zlc_ui.fluent import (
    ACCENT, AXIS_GROUP_COLORS, GREY, FluentButton, FluentCodeEdit, FluentComboBox, FluentLineEdit, FluentLabel, FluentFrame,
    FluentSpinBox, retire_widget, scaled_px, signals_blocked,
)


class DeriveEditor(QtWidgets.QWidget):
    """The concrete Derive contribution to the existing Logic editor."""

    draft_changed = QtCore.pyqtSignal(object)
    managed_fields = ("expressions", "input_view", "window")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from .expression import HELP_TEXT

        self._loading = False
        self._rows = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.controls = QtWidgets.QWidget(self)
        controls = QtWidgets.QHBoxLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(FluentLabel("Input range"))
        self.input_view = FluentComboBox()
        for label, value in (("Event", "event"), ("Run", "run"),
                             ("Window", "window")):
            self.input_view.addItem(label, value)
        controls.addWidget(self.input_view)
        self.window_label = FluentLabel("Window")
        self.window = FluentSpinBox()
        self.window.setRange(1, 2_147_483_647)
        self.window.setValue(50)
        controls.addWidget(self.window_label)
        controls.addWidget(self.window)
        controls.addStretch(1)
        layout.addWidget(self.controls)

        self.input_summary = FluentLabel()
        self.input_summary.setWordWrap(True)
        self.input_summary.setTextFormat(QtCore.Qt.RichText)
        self.input_summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        input_frame = FluentFrame()
        input_layout = QtWidgets.QVBoxLayout(input_frame)
        input_layout.addWidget(self.input_summary)
        layout.addWidget(input_frame)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(FluentLabel("Outputs · Assign each answer to result"))
        header.addStretch(1)
        self.add_button = FluentButton("Add output", color=ACCENT)
        header.addWidget(self.add_button)
        layout.addLayout(header)
        self.rows_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.rows_layout)
        self.output_summary = FluentLabel()
        self.output_summary.setWordWrap(True)
        self.output_summary.setTextFormat(QtCore.Qt.RichText)
        self.output_summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        output_frame = FluentFrame()
        output_layout = QtWidgets.QVBoxLayout(output_frame)
        output_layout.addWidget(self.output_summary)
        layout.addWidget(output_frame)
        self.help = FluentLabel(HELP_TEXT)
        self.help.setWordWrap(True)
        self.help.setTextFormat(QtCore.Qt.PlainText)
        self.help.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.help)

        self.input_view.currentIndexChanged.connect(self._range_changed)
        self.window.valueChanged.connect(self._emit_draft)
        self.add_button.clicked.connect(self._add_output)
        self._range_changed()

    def _range_changed(self, *_args) -> None:
        visible = self.input_view.currentData() == "window"
        self.window_label.setVisible(visible)
        self.window.setVisible(visible)
        self._emit_draft()

    def _append_row(self, name: str, code: str) -> None:
        frame = QtWidgets.QWidget(self)
        column = QtWidgets.QVBoxLayout(frame)
        column.setContentsMargins(0, 0, 0, scaled_px(6))
        header = QtWidgets.QHBoxLayout()
        header.addWidget(FluentLabel("Name"))
        name_edit = FluentLineEdit(name)
        name_edit.setPlaceholderText("Output name, e.g. contrast")
        header.addWidget(name_edit, 1)
        remove = FluentButton("×", color=GREY)
        remove.setToolTip("Remove this output")
        header.addWidget(remove)
        column.addLayout(header)
        code_edit = FluentCodeEdit(code)
        code_edit.setPlaceholderText("result = a.<output>")
        code_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        code_edit.setTabStopDistance(code_edit.fontMetrics().horizontalAdvance("    "))
        code_edit.setFixedHeight(scaled_px(136))
        column.addWidget(code_edit)
        self._rows.append((frame, name_edit, code_edit, remove))
        self.rows_layout.addWidget(frame)
        name_edit.textChanged.connect(self._emit_draft)
        code_edit.textChanged.connect(self._emit_draft)
        remove.clicked.connect(self._remove_output)

    def _add_output(self) -> None:
        taken = {name.text() for _frame, name, _code, _remove in self._rows}
        ordinal = 1
        while f"signal_{ordinal}" in taken:
            ordinal += 1
        self._append_row(f"signal_{ordinal}", "")
        self._rows[-1][2].setFocus()
        self._emit_draft()

    def _remove_output(self) -> None:
        for index, row in enumerate(self._rows):
            if row[3] is self.sender():
                self._rows.pop(index)
                retire_widget(row[0])
                self._emit_draft()
                break

    def _emit_draft(self, *_args) -> None:
        if not self._loading:
            self.draft_changed.emit({"values": {
                "input_view": self.input_view.currentData(),
                "window": self.window.value(),
                "expressions": tuple({"name": name.text(), "code": code.toPlainText()}
                    for _frame, name, code, _remove in self._rows),
            }})

    @staticmethod
    def _bundle_text(title: str, bundle: object) -> str:
        lines = [f"<b>{escape(title)}</b>"]
        for name, schema, snapshot in tuple(bundle or ()):
            if schema is None:
                lines.append(escape(f"a.{name}: Waiting for data"))
                continue
            groups = []
            for color, group in zip(AXIS_GROUP_COLORS, schema_structure(schema), strict=True):
                sizes = " × ".join(str(size) for _name, size in group) or "1"
                groups.append(f'<span style="color: {color}">({escape(sizes)})</span>')
            lines.append(f"<b>{escape(str(name))}: {' × '.join(groups)}</b>")
            lines.append(escape(f"Storage (NumPy): {schema.physical_shape} · {schema.value_schema.dtype} · {schema.value_schema.value_unit or '1'}"))
            lines.append("Repeat and Point each use one storage dimension; their named axes stay separate.")
            if snapshot is not None and snapshot.block.schema != schema:
                lines.append(escape(f"Event storage: {snapshot.block.schema.physical_shape}; Run storage shown above"))
            for number, (label, domain) in enumerate((("Repeat", schema.repeat_domain), ("Point", schema.point_domain),
                                  ("Cell-data", schema.cell_domain))):
                color = AXIS_GROUP_COLORS[number]
                physical = str(number) if number < 2 else ", ".join(str(2+i) for i in range(len(domain.shape)))
                if not domain.axes:
                    detail = f"{label} · No named axis, length 1 · storage dim {physical}"
                    lines.append(f'<span style="color: {color}">{escape(detail)}</span>')
                for index, axis in enumerate(domain.axes):
                    if axis.size <= 6:
                        coordinates = repr(tuple(axis.coordinate_at(i) for i in range(axis.size)))
                    else:
                        coordinates = f"({axis.coordinate_at(0)!r}, …, {axis.coordinate_at(axis.size - 1)!r})"
                    physical = 2 + index if number == 2 else number
                    detail = (f"{label} · {axis.name} [{axis.axis_id.value}] "
                              f"· size {axis.size} · unit {axis.unit or '1'} · {coordinates} · storage dim {physical}")
                    lines.append(f'<span style="color: {color}">{escape(detail)}</span>')
        if len(lines) == 1:
            lines.append("No published data")
        return "<br>".join(lines)

    def update_projection(self, projection: Mapping[str, object]) -> None:
        values = projection.get("form_values") or {}
        self._loading = True
        try:
            mode = str(values.get("input_view", "event"))
            with signals_blocked(self.input_view, self.window):
                self.input_view.setCurrentIndex(self.input_view.findData(mode))
                self.window.setValue(int(values.get("window", 50)))
            self._range_changed()
            incoming = tuple(values.get("expressions") or ())
            while len(self._rows) > len(incoming):
                retire_widget(self._rows.pop()[0])
            for index, entry in enumerate(incoming):
                name, code = str(entry["name"]), str(entry["code"])
                if index == len(self._rows):
                    self._append_row(name, code)
                else:
                    _frame, name_edit, code_edit, _remove = self._rows[index]
                    if name_edit.text() != name:
                        name_edit.setText(name)
                    if code_edit.toPlainText() != code:
                        code_edit.setPlainText(code)
            input_text = self._bundle_text(
                "Inputs · a.<output>",
                projection.get("input_bundle"))
            output_text = self._bundle_text(
                "Outputs · Published data", projection.get("output_bundle"))
            for widget, text in ((self.input_summary, input_text), (self.output_summary, output_text)):
                if widget.text() != text:
                    widget.setText(text)
        finally:
            self._loading = False

    def set_mutation_enabled(self, enabled: bool) -> None:
        self.controls.setEnabled(enabled)
        self.add_button.setEnabled(enabled)
        for frame, _name, _code, _remove in self._rows:
            frame.setEnabled(enabled)


def derive_editor_factory(parent=None) -> DeriveEditor:
    return DeriveEditor(parent)
