"""Concrete ScanTab contribution for the Pulse API scan plugin."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PyQt5 import QtCore, QtWidgets
from zlc_pulse import api_parameter_columns_for, scan_table_template
from zlc_ui.pulse import PulseScanView, ScanPageRecord

from .logic_node import (
    _accepted_scan_rows,
    _evaluate_scan_source,
    _scan_rows_text,
)


class PulseApiScanEditor(QtWidgets.QWidget):
    """Use the existing Pulse Scan page to author this Measurement's table."""

    draft_changed = QtCore.pyqtSignal(object)
    managed_fields = ("scan_source", "scan_rows")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scan = PulseScanView()
        layout.addWidget(self.scan)

        info = self.scan.scan_slots_label.parentWidget()
        for label in info.findChildren(QtWidgets.QLabel):
            if label is not self.scan.scan_slots_label:
                label.hide()
        for control in (
            self.scan.scan_repeats_spin,
            self.scan.scan_hold_button,
            self.scan.scan_step_back_button,
            self.scan.scan_step_forward_button,
            self.scan.scan_progress_label,
            self.scan.scan_load_program_button,
            self.scan.scan_save_array_button,
        ):
            control.hide()

        self._sequence = None
        self._resource_path = ""
        self._source_text = ""
        self._rows_text = ""
        self._table_text = ""
        self._status = "Choose a Pulse template with API parameters."
        self._dirty = False
        self.scan.source_edited.connect(self._source_edited)
        self.scan.template_requested.connect(self._template_requested)
        self.scan.run_requested.connect(self._run)

    def update_projection(self, projection: Mapping[str, object]) -> None:
        resources = projection.get("workspace_resources") or {}
        values = projection.get("form_values") or {}
        if not isinstance(resources, Mapping) or not isinstance(values, Mapping):
            raise TypeError("Pulse API scan projection needs resource and form mappings")
        resource = resources.get("pulse_template")
        sequence = getattr(resource, "value", None)
        resource_path = str(getattr(resource, "path", "") or "")
        if resource_path != self._resource_path:
            self._resource_path = resource_path
            self._table_text = ""
            self._rows_text = ""
            self._dirty = bool(values.get("scan_source"))
        self._sequence = sequence
        source = str(values.get("scan_source") or "")
        rows_text = str(values.get("scan_rows") or "")
        if source != self._source_text:
            self._source_text = source
        if rows_text != self._rows_text:
            self._rows_text = rows_text
            if rows_text and self._sequence is not None:
                try:
                    rows, shape = _accepted_scan_rows(rows_text, self._sequence)
                except Exception:
                    self._table_text = ""
                else:
                    self._table_text = self._format_rows(rows, shape)
        columns = () if self._sequence is None else api_parameter_columns_for(self._sequence)
        self._status = (
            "Choose a Pulse template with API parameters."
            if not columns
            else "API parameters: "
            + ", ".join(f"{column.name} ({column.unit})" for column in columns)
        )
        self.scan.set_page(
            ScanPageRecord(
                slots_text=self._status,
                table_text=self._table_text,
                source_text=self._source_text,
                source_dirty=self._dirty,
                repeats=1,
            )
        )

    def set_mutation_enabled(self, enabled: bool) -> None:
        self.scan.set_workspace_busy(not bool(enabled))
        self.scan.scan_code.setReadOnly(not bool(enabled))

    def _source_edited(self, source: str) -> None:
        self._source_text = str(source)
        self._dirty = True
        self._rows_text = ""
        self._table_text = ""
        self.scan.set_scan_table_text("")
        self.scan.set_run_dirty(True)
        self.draft_changed.emit(
            {
                "values": {
                    "scan_source": self._source_text,
                    "scan_rows": "",
                }
            }
        )

    def _template_requested(self, kind: str) -> None:
        if self._sequence is None:
            self.scan.set_slots_text("Choose a valid Pulse template first.")
            return
        self.scan.scan_code.setPlainText(
            scan_table_template(
                str(kind),
                api_parameter_columns_for(self._sequence),
            )
        )

    def _run(self) -> None:
        if self._sequence is None:
            self.scan.set_slots_text("Choose a valid Pulse template first.")
            return
        try:
            rows, shape = _evaluate_scan_source(
                self.scan.scan_code.toPlainText(),
                self._sequence,
            )
        except Exception as error:
            self.scan.set_slots_text(f"Scan program failed: {error}")
            return
        self._rows_text = _scan_rows_text(rows, shape, self._sequence)
        self._table_text = self._format_rows(rows, shape)
        self._dirty = False
        self.scan.set_scan_table_text(self._table_text)
        self.scan.set_slots_text(
            f"{len(rows)} point(s) ready · {self._status}"
        )
        self.scan.set_run_dirty(False)
        self.draft_changed.emit({"values": {"scan_rows": self._rows_text}})

    @staticmethod
    def _format_rows(rows, shape: tuple[int, ...]) -> str:
        text = np.array2string(np.asarray(rows), separator=", ")
        return f"shape = {shape}\n{text}" if shape else text


__all__ = ["PulseApiScanEditor"]
