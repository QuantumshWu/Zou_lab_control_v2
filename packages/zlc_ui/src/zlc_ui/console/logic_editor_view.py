"""One non-modal TaskConsole Logic Edit tab.

The widget consumes a UI form projection and plain selector options.  It does
not know which package declared the node, how devices are resolved, or what a
Start does; those remain Workbench responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentLabel,
    FluentReadoutEdit,
    FluentScrollArea,
    scaled_px,
)
from zlc_ui.form import FluentParameterForm, FormSpec


class LogicEditorView(QtWidgets.QWidget):
    """Editable projection of one stopped/running Logic row draft."""

    draft_changed = QtCore.pyqtSignal(object)
    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        node_id: str,
        projection: Mapping[str, object],
        parent=None,
        *,
        show_actions: bool = True,
    ) -> None:
        super().__init__(parent)
        self.node_id = str(node_id)
        self._mutation_enabled = True
        self._projection: dict[str, object] = {}
        self._device_combos: dict[str, FluentComboBox] = {}
        self._artifact_result_readouts: dict[str, FluentReadoutEdit] = {}

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        body.setStyleSheet("background: transparent;")
        self._body_layout = QtWidgets.QVBoxLayout(body)
        margin = scaled_px(14, minimum=8)
        self._body_layout.setContentsMargins(margin, margin, margin, margin)
        self._body_layout.setSpacing(scaled_px(10, minimum=6))

        header = FluentFrame(bordered=False)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = FluentLabel(self.node_id)
        self.kind_label = FluentLabel("")
        self.kind_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.status_label = FluentLabel("stopped")
        self.status_label.setWordWrap(True)
        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(self.kind_label)
        header_layout.addWidget(self.status_label, 2)
        self._body_layout.addWidget(header)

        self._selector_frame = FluentFrame()
        self._selector_layout = QtWidgets.QFormLayout(self._selector_frame)
        self._selector_layout.setContentsMargins(margin, margin, margin, margin)
        self._selector_layout.setSpacing(scaled_px(8, minimum=5))
        self.source_combo: FluentComboBox | None = None
        self._body_layout.addWidget(self._selector_frame)

        empty = FormSpec(())
        self.artifact_form = FluentParameterForm(empty, {})
        self._body_layout.addWidget(self.artifact_form)
        self.artifact_form.changed.connect(self._artifact_changed)

        self.form = FluentParameterForm(empty, {})
        self._body_layout.addWidget(self.form)
        self.form.changed.connect(self._parameter_changed)

        self._artifact_result_frame = FluentFrame()
        self._artifact_result_layout = QtWidgets.QFormLayout(
            self._artifact_result_frame
        )
        self._artifact_result_layout.setContentsMargins(
            margin, margin, margin, margin
        )
        self._artifact_result_layout.setSpacing(scaled_px(8, minimum=5))
        self._body_layout.addWidget(self._artifact_result_frame)

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        self.start_button = FluentButton("Start", color=GREEN)
        self.stop_button = FluentButton("Stop", color=ORANGE)
        self.remove_button = FluentButton("Remove", color=ACCENT)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.remove_button.clicked.connect(self.remove_requested.emit)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.remove_button)
        self._body_layout.addLayout(actions)
        self.start_button.setVisible(bool(show_actions))
        self.stop_button.setVisible(bool(show_actions))
        self.remove_button.setVisible(bool(show_actions))
        self._body_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        self.update_projection(projection)

    def update_projection(self, projection: Mapping[str, object]) -> None:
        incoming = dict(projection)
        spec = incoming.get("form_spec")
        values = incoming.get("form_values")
        if not isinstance(spec, FormSpec):
            raise TypeError("logic editor projection needs form_spec: FormSpec")
        if not isinstance(values, Mapping):
            raise TypeError("logic editor projection needs form_values: mapping")
        artifact_spec = incoming.get("artifact_form_spec", FormSpec(()))
        artifact_values = incoming.get("artifact_values", {})
        if not isinstance(artifact_spec, FormSpec):
            raise TypeError(
                "logic editor projection needs artifact_form_spec: FormSpec"
            )
        if not isinstance(artifact_values, Mapping):
            raise TypeError(
                "logic editor projection needs artifact_values: mapping"
            )

        self._projection = incoming
        self.title_label.setText(str(incoming.get("api_name") or self.node_id))
        kind = str(incoming.get("kind") or "logic")
        self.kind_label.setText(f"({kind})")
        self.artifact_form.reconcile(artifact_spec, dict(artifact_values))
        self.artifact_form.setVisible(bool(artifact_spec.keys))
        self.form.reconcile(spec, dict(values))
        self._reconcile_selectors(incoming)
        self._rebuild_artifact_results(incoming.get("artifact_results", ()))

        running = bool(incoming.get("running"))
        pending = bool(incoming.get("pending"))
        error = str(incoming.get("error") or "")
        self.start_button.setText("Starting…" if pending else ("Restart" if running else "Start"))
        self.start_button.setEnabled(self._mutation_enabled and not pending)
        self.stop_button.setEnabled(self._mutation_enabled and (running or pending))
        self.remove_button.setEnabled(self._mutation_enabled)
        state = "waiting for device release" if pending else ("running" if running else "stopped")
        status = str(incoming.get("status") or state)
        self.status_label.setText(error or status)
        self.status_label.setStyleSheet(
            f"color: {'#D13438' if error else GREY}; background: transparent; border: none;"
        )

    def set_mutation_enabled(self, enabled: bool) -> None:
        """Disable authored and lifecycle commands while a Task owns the console."""

        self._mutation_enabled = bool(enabled)
        self._selector_frame.setEnabled(self._mutation_enabled)
        self.artifact_form.setEnabled(self._mutation_enabled)
        self.form.setEnabled(self._mutation_enabled)
        running = bool(self._projection.get("running"))
        pending = bool(self._projection.get("pending"))
        self.start_button.setEnabled(self._mutation_enabled and not pending)
        self.stop_button.setEnabled(
            self._mutation_enabled and (running or pending)
        )
        self.remove_button.setEnabled(self._mutation_enabled)

    def _reconcile_selectors(self, projection: Mapping[str, object]) -> None:
        device_options = projection.get("device_options") or {}
        device_keys = projection.get("device_keys") or {}
        if not isinstance(device_options, Mapping) or not isinstance(device_keys, Mapping):
            raise TypeError("logic editor device selectors must be mappings")

        has_source = bool(projection.get("source_required"))
        if has_source:
            current = str(projection.get("source_signal") or "")
            options = tuple(str(item) for item in projection.get("source_options", ()))
            if self.source_combo is None:
                self.source_combo = FluentComboBox()
                self.source_combo.currentIndexChanged[int].connect(
                    self._source_changed
                )
                self._selector_layout.insertRow(
                    0, "Frames signal", self.source_combo
                )
            self._fill_combo(self.source_combo, current, options, blank=True)
        elif self.source_combo is not None:
            self._retire_selector(self.source_combo)
            self.source_combo = None

        for argument_name, offered in device_options.items():
            name = str(argument_name)
            combo = self._device_combos.get(name)
            if combo is None:
                combo = FluentComboBox()
                combo.currentIndexChanged[int].connect(
                    lambda _index, key=name, control=combo: self._device_changed(
                        key, control
                    )
                )
                self._device_combos[name] = combo
                self._selector_layout.addRow(
                    name.replace("_", " ").title(), combo
                )
            current = str(device_keys.get(name, ""))
            self._fill_combo(combo, current, tuple(str(item) for item in offered), blank=True)

        retired = set(self._device_combos).difference(
            str(name) for name in device_options
        )
        for name in retired:
            self._retire_selector(self._device_combos.pop(name))

        self._selector_frame.setVisible(has_source or bool(self._device_combos))

    def _retire_selector(self, combo: FluentComboBox) -> None:
        row = self._selector_layout.takeRow(combo)
        for item in (row.labelItem, row.fieldItem):
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

    @staticmethod
    def _fill_combo(
        combo: FluentComboBox,
        current: str,
        options: tuple[str, ...],
        *,
        blank: bool,
    ) -> None:
        ordered: list[str] = []
        if blank:
            ordered.append("")
        if current and current not in options:
            ordered.append(current)
        ordered.extend(item for item in options if item not in ordered)
        desired = tuple((value or "(not selected)", value) for value in ordered)
        existing = tuple(
            (combo.itemText(index), combo.itemData(index))
            for index in range(combo.count())
        )
        combo.blockSignals(True)
        if existing != desired:
            combo.clear()
            for label, value in desired:
                combo.addItem(label, value)
        index = combo.findData(current)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)

    def _parameter_changed(self, key: str) -> None:
        try:
            value = self.form.read_value(str(key))
        except (TypeError, ValueError) as error:
            self.status_label.setText(str(error))
            return
        self.draft_changed.emit({"values": {str(key): value}})

    def _artifact_changed(self, key: str) -> None:
        try:
            value = self.artifact_form.read_value(str(key))
        except (TypeError, ValueError) as error:
            self.status_label.setText(str(error))
            return
        self.draft_changed.emit({"artifact_inputs": {str(key): value}})

    def _rebuild_artifact_results(self, rows: object) -> None:
        while self._artifact_result_layout.rowCount():
            self._artifact_result_layout.removeRow(0)
        self._artifact_result_readouts.clear()
        for raw in tuple(rows or ()):  # type: ignore[arg-type]
            if not isinstance(raw, Mapping):
                raise TypeError("logic artifact results must be mappings")
            name = str(raw.get("name") or "")
            path = str(raw.get("path") or "")
            if not name or not path:
                raise ValueError("logic artifact results need name and path")
            readout = FluentReadoutEdit(path)
            readout.setToolTip(str(raw.get("contract_id") or ""))
            self._artifact_result_readouts[name] = readout
            self._artifact_result_layout.addRow(
                name.replace("_", " ").title(),
                readout,
            )
        self._artifact_result_frame.setVisible(
            bool(self._artifact_result_readouts)
        )

    def _source_changed(self, _index: int) -> None:
        if self.source_combo is None:
            return
        self.draft_changed.emit(
            {"source_signal": str(self.source_combo.currentData() or "")}
        )

    def _device_changed(self, argument_name: str, combo: FluentComboBox) -> None:
        self.draft_changed.emit(
            {"device_keys": {str(argument_name): str(combo.currentData() or "")}}
        )


__all__ = ["LogicEditorView"]
