"""Domain-free target-port editor.

The presenter supplies records and width rules.  This view only performs
text-level checks that keep the editor usable while a person is typing; it
never constructs or validates a domain manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5 import QtCore, QtWidgets

from zlc_ui.form import being_edited
from zlc_ui.fluent import (
    ACCENT, GREY, FluentButton, FluentFrame, FluentGroupBox, FluentLabel,
    FluentLineEdit, FluentScrollArea, FluentSpinBox, signals_blocked,
)

from ._layout import px, row_height
from .models import TargetPortRecord, TargetWidthRule


@dataclass
class _TargetRowWidgets:
    record: TargetPortRecord
    signal: FluentLineEdit
    endpoints: FluentLineEdit
    width: FluentSpinBox
    clock_endpoint: FluentLineEdit
    remove_button: FluentButton


_QT_SPINBOX_MAXIMUM = (1 << 31) - 1


class PulseTargetView(QtWidgets.QWidget):
    apply_requested = QtCore.pyqtSignal(tuple)
    feedback_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, *, endpoint_template: str = "endpoint:{key}[{bit}]") -> None:
        super().__init__(parent)
        self._editable = False
        self._rows: list[_TargetRowWidgets] = []
        self._width_rules = {
            "digital": TargetWidthRule(1, 1, 1),
            "dac": TargetWidthRule(1, 2, None),
        }
        self._endpoint_template = str(endpoint_template)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(px(10), px(10), px(10), px(10))
        layout.setSpacing(px(8))
        header = FluentFrame()
        header.setFixedHeight(px(58, minimum=48))
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(px(12), px(8), px(12), px(8))
        self.status_label = FluentLabel("")
        self.status_label.setWordWrap(True)
        header_layout.addWidget(self.status_label, 1)
        self.add_digital_button = FluentButton("Add Digital", color=ACCENT)
        self.add_dac_button = FluentButton("Add DAC", color=ACCENT)
        self.apply_button = FluentButton("Apply target", color=ACCENT)
        for button in (self.add_digital_button, self.add_dac_button, self.apply_button):
            header_layout.addWidget(button)
        layout.addWidget(header)

        self.cards_scroll = FluentScrollArea()
        self.cards_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # A bar that is always there but cannot move says "there is more below"
        # when there is not.  Let the content decide.
        self.cards_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.cards_body = QtWidgets.QWidget()
        self.cards_layout = QtWidgets.QVBoxLayout(self.cards_body)
        self.cards_layout.setContentsMargins(px(4), px(4), px(4), px(8))
        self.cards_layout.setSpacing(px(8))
        self.digital_card = FluentGroupBox("Digital outputs")
        self.digital_layout = QtWidgets.QGridLayout(self.digital_card)
        self.digital_layout.setContentsMargins(px(12), px(10), px(12), px(12))
        self.digital_layout.setHorizontalSpacing(px(10, minimum=6))
        self.digital_layout.setVerticalSpacing(px(6, minimum=4))
        self.digital_layout.setColumnStretch(0, 1)
        self.digital_layout.setColumnStretch(1, 2)
        for column, text in enumerate(("signal name", "endpoint", "")):
            self.digital_layout.addWidget(self._header_label(text), 0, column)
        self.cards_layout.addWidget(self.digital_card)
        self.dac_card = FluentGroupBox("DAC outputs")
        self.dac_layout = QtWidgets.QGridLayout(self.dac_card)
        self.dac_layout.setContentsMargins(px(12), px(10), px(12), px(12))
        self.dac_layout.setHorizontalSpacing(px(10, minimum=6))
        self.dac_layout.setVerticalSpacing(px(6, minimum=4))
        self.dac_layout.setColumnStretch(0, 1)
        self.dac_layout.setColumnStretch(2, 2)
        self.dac_layout.setColumnStretch(3, 1)
        for column, text in enumerate(("signal name", "width", "data endpoints (bit order)", "latch endpoint", "")):
            self.dac_layout.addWidget(self._header_label(text), 0, column)
        self.cards_layout.addWidget(self.dac_card)
        self.cards_layout.addStretch(1)
        self.cards_scroll.setWidget(self.cards_body)
        layout.addWidget(self.cards_scroll, 1)
        self.add_digital_button.clicked.connect(self._add_digital)
        self.add_dac_button.clicked.connect(self._add_dac)
        self.apply_button.clicked.connect(self._emit_apply)

    @staticmethod
    def _header_label(text: str) -> FluentLabel:
        label = FluentLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        label.setFixedHeight(row_height())
        return label

    def set_width_rules(self, digital: TargetWidthRule, dac: TargetWidthRule) -> None:
        self._width_rules = {"digital": digital, "dac": dac}
        for row in self._rows:
            rule = self._width_rules[row.record.kind]
            row.width.setRange(rule.minimum, _QT_SPINBOX_MAXIMUM if rule.maximum is None else rule.maximum)

    def set_ports(self, records: tuple[TargetPortRecord, ...], editable: bool, status_text: str) -> None:
        incoming = tuple(records)
        if any(not isinstance(record, TargetPortRecord) for record in incoming):
            raise TypeError("records must contain TargetPortRecord values")
        existing = {row.record.key: row for row in self._rows}
        reconciled: list[_TargetRowWidgets] = []
        for record in incoming:
            row = existing.pop(record.key, None)
            if row is None or row.record.kind != record.kind:
                if row is not None:
                    self._destroy_row(row)
                row = self._make_row(record)
            else:
                self._reconcile_row(row, record)
            reconciled.append(row)
        for row in existing.values():
            self._destroy_row(row)
        self._rows = reconciled
        self._place_rows()
        self._editable = bool(editable)
        self.status_label.setText(str(status_text))
        self._set_editable(self._editable)
        self.digital_card.setTitle(f"Digital outputs · {sum(r.record.kind == 'digital' for r in self._rows)}")
        self.dac_card.setTitle(f"DAC outputs · {sum(r.record.kind == 'dac' for r in self._rows)}")

    def set_feedback(self, text: str) -> None:
        self.feedback_requested.emit(str(text))
        self.status_label.setText(str(text))

    def _make_row(self, record: TargetPortRecord) -> _TargetRowWidgets:
        signal = FluentLineEdit(record.signal)
        signal.setFixedHeight(row_height())
        endpoints = FluentLineEdit(", ".join(record.endpoints))
        endpoints.setFixedHeight(row_height())
        width = FluentSpinBox()
        rule = self._width_rules[record.kind]
        width.setRange(rule.minimum, _QT_SPINBOX_MAXIMUM if rule.maximum is None else rule.maximum)
        width.setValue(max(rule.minimum, len(record.endpoints)))
        width.setFixedSize(px(82, minimum=70), row_height())
        clock_endpoint = FluentLineEdit(record.clock_endpoint or "")
        clock_endpoint.setFixedHeight(row_height())
        remove_button = FluentButton("Remove", color=GREY)
        remove_button.setFixedSize(px(86, minimum=74), row_height())
        remove_button.clicked.connect(lambda _checked=False, key=record.key: self._remove_key(key))
        if record.kind == "dac":
            width.valueChanged.connect(lambda value, field=endpoints: self._resize_dac_endpoints(field, int(value)))
        return _TargetRowWidgets(record, signal, endpoints, width, clock_endpoint, remove_button)

    def _reconcile_row(self, row: _TargetRowWidgets, record: TargetPortRecord) -> None:
        with signals_blocked(row.signal, row.endpoints, row.width, row.clock_endpoint):
            self._set_line_text(row.signal, record.signal)
            self._set_line_text(row.endpoints, ", ".join(record.endpoints))
            width = max(1, len(record.endpoints))
            if row.width.value() != width and not being_edited(row.width):
                row.width.setValue(width)
            self._set_line_text(row.clock_endpoint, record.clock_endpoint or "")
        row.record = record

    @staticmethod
    def _set_line_text(field: FluentLineEdit, text: str) -> None:
        """A projected value lands in a field the operator is not inside."""

        text = str(text)
        if field.text() == text or being_edited(field):
            return
        field.setText(text)

    def _placed_widgets(self, row: _TargetRowWidgets) -> tuple[QtWidgets.QWidget, ...]:
        if row.record.kind == "digital":
            return row.signal, row.endpoints, row.remove_button
        return row.signal, row.width, row.endpoints, row.clock_endpoint, row.remove_button

    def _destroy_row(self, row: _TargetRowWidgets) -> None:
        layout = self.digital_layout if row.record.kind == "digital" else self.dac_layout
        for widget in self._placed_widgets(row):
            layout.removeWidget(widget)
            widget.hide()
            widget.deleteLater()

    def _place_rows(self) -> None:
        positions = {"digital": 1, "dac": 1}
        desired = []
        for row in self._rows:
            index = positions[row.record.kind]
            layout = self.digital_layout if row.record.kind == "digital" else self.dac_layout
            if row.record.kind == "digital":
                desired.extend(((layout, row.signal, index, 0), (layout, row.endpoints, index, 1), (layout, row.remove_button, index, 2)))
            else:
                desired.extend(((layout, row.signal, index, 0), (layout, row.width, index, 1), (layout, row.endpoints, index, 2), (layout, row.clock_endpoint, index, 3), (layout, row.remove_button, index, 4)))
            positions[row.record.kind] += 1
        for layout, widget, row_index, column_index in desired:
            current = layout.indexOf(widget)
            if current >= 0:
                current_row, current_column, row_span, column_span = layout.getItemPosition(current)
                if (current_row, current_column, row_span, column_span) == (row_index, column_index, 1, 1):
                    continue
                layout.removeWidget(widget)
            layout.addWidget(widget, row_index, column_index)

    def _set_editable(self, editable: bool) -> None:
        """Open the page for authoring, or lock the WIRING and keep the names.

        Two different permissions, not one.  A board owns its topology -- which
        lanes exist, how wide a bus is, which pins they reach -- and none of
        that may move while the editor is pointed at it.  What a signal is
        CALLED is display metadata that belongs to the person reading it, and
        the page says so in its own status line: "rename freely, the topology
        is the board's".  One flag drove both, so attached the rename box was
        greyed out and Apply with it: the page told the operator to do
        something it had just disabled.
        """

        for row in self._rows:
            # Names: always the operator's.
            row.signal.setReadOnly(False)
            row.signal.setEnabled(True)
            # Wiring: the board's whenever there is one.
            for field in (row.endpoints, row.clock_endpoint):
                field.setReadOnly(not editable)
                field.setEnabled(editable)
            row.width.setEnabled(editable)
            row.remove_button.setEnabled(editable)
            row.remove_button.setVisible(True)
        for button in (self.add_digital_button, self.add_dac_button):
            button.setEnabled(editable)
        # Apply commits whatever IS editable, and a rename always is.
        self.apply_button.setEnabled(True)

    def _used_keys(self) -> set[str]:
        return {key for row in self._rows for key in (row.record.key, row.record.clock_key) if key}

    def _allocate_key(self, stem: str) -> str:
        used = self._used_keys()
        index = 1
        while f"{stem}_{index}" in used:
            index += 1
        return f"{stem}_{index}"

    @staticmethod
    def _endpoint_values(text: str) -> tuple[str, ...]:
        return tuple(value.strip() for value in str(text).split(",") if value.strip())

    def _resize_dac_endpoints(self, field: FluentLineEdit, width: int) -> None:
        width = max(self._width_rules["dac"].minimum, int(width))
        endpoints = list(self._endpoint_values(field.text()))
        key = next((row.record.key for row in self._rows if row.endpoints is field), "dac")
        if len(endpoints) < width:
            endpoints.extend(self._endpoint_template.format(key=key, bit=bit) for bit in range(len(endpoints), width))
        else:
            endpoints = endpoints[:width]
        field.setText(", ".join(endpoints))

    def _current_records(self) -> tuple[TargetPortRecord, ...]:
        values = []
        for row in self._rows:
            endpoints = self._endpoint_values(row.endpoints.text())
            width = int(row.width.value()) if row.record.kind == "dac" else 1
            if len(endpoints) != width:
                raise ValueError(f"{row.signal.text().strip() or row.record.key}: width is {width} but {len(endpoints)} endpoints are listed")
            values.append(TargetPortRecord(
                key=row.record.key,
                kind=row.record.kind,
                signal=row.signal.text().strip(),
                endpoints=endpoints,
                clock_key=row.record.clock_key,
                clock_endpoint=row.clock_endpoint.text().strip() if row.record.kind == "dac" else None,
                lane_order=row.record.lane_order if len(row.record.lane_order) == len(endpoints) else tuple(range(len(endpoints))),
            ))
        return tuple(values)

    def _queue_reveal_row(self, key: str) -> None:
        QtCore.QTimer.singleShot(0, lambda: next((self.cards_scroll.ensureWidgetVisible(row.signal) for row in self._rows if row.record.key == key), None))

    def _add_digital(self) -> None:
        try:
            current = self._current_records()
        except ValueError as error:
            self.set_feedback(str(error))
            return
        key = self._allocate_key("digital")
        record = TargetPortRecord(key, "digital", key, (f"endpoint:{key}",), lane_order=(0,))
        split = next((index for index, row in enumerate(current) if row.kind == "dac"), len(current))
        self.set_ports(current[:split] + (record,) + current[split:], self._editable, self.status_label.text())
        self._queue_reveal_row(key)

    def _add_dac(self) -> None:
        try:
            current = self._current_records()
        except ValueError as error:
            self.set_feedback(str(error))
            return
        key = self._allocate_key("dac")
        clock_key = self._allocate_key(f"{key}_clock")
        width = self._width_rules["dac"].default
        record = TargetPortRecord(key, "dac", key, tuple(self._endpoint_template.format(key=key, bit=bit) for bit in range(width)), clock_key, f"endpoint:{clock_key}", tuple(range(width)))
        self.set_ports(current + (record,), self._editable, self.status_label.text())
        self._queue_reveal_row(key)

    def _remove_key(self, key: str) -> None:
        try:
            current = self._current_records()
        except ValueError as error:
            self.set_feedback(str(error))
            return
        self.set_ports(tuple(row for row in current if row.key != key), self._editable, self.status_label.text())

    def _emit_apply(self) -> None:
        try:
            records = self._current_records()
        except (TypeError, ValueError) as error:
            self.set_feedback(str(error))
            return
        self.apply_requested.emit(records)


__all__ = ["PulseTargetView"]
