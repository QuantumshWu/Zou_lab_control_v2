"""One non-modal TaskConsole Panel Edit tab.

The widget consumes a plain Workbench projection.  It deliberately knows
nothing about signal publications, plot specs, measurement descriptors, or
devices: it shows the state it is given and emits named operator intents.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentPageBody,
    FluentPathEdit,
    FluentReadoutEdit,
    FluentScrollArea,
    FluentSettingRow,
    FluentSwitch,
    GREY,
    ORANGE,
    scaled_px,
    setting_label_width,
    stamped_file_name,
)
from zlc_ui.form import (
    FluentParameterForm,
    FormChoice,
    FormFieldProps,
    FormSpec,
)

from ._panel_projection import (
    interval_form_field,
    panel_state_document,
    parameter_edit_values,
    parameter_fields,
    parameter_form_spec,
    parameter_form_values,
    signal_form_runtime,
)
from .panel_card_view import _set_interaction


_IMAGE_FORMATS = ("png", "pdf", "svg")


class PanelEditorView(QtWidgets.QWidget):
    """Editable projection of one Workbench-owned panel state and snapshot."""

    state_changed = QtCore.pyqtSignal(object)
    snapshot_refresh_requested = QtCore.pyqtSignal()
    producer_edit_requested = QtCore.pyqtSignal(str)
    save_figure_requested = QtCore.pyqtSignal(str)

    def __init__(self, panel_id: str, projection: Mapping[str, object], parent=None) -> None:
        super().__init__(parent)
        self.panel_id = str(panel_id)
        self._mutation_enabled = True
        self._science_locked = False
        self._projection: dict[str, object] = {}
        self._state: dict[str, Any] = {}
        self._parameter_fields: dict[str, dict[str, dict[str, object]]] = {
            "semantic": {},
            "display": {},
            "fit": {},
        }
        self._surface: QtWidgets.QWidget | None = None
        self._selectors_on = True
        self._save_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_directory_initialized = False
        self._save_name_initialized = False
        self._snapshot_can_save = False
        self._signal_groups = ()
        self._overlay_groups = ()
        self._signal_runtime = signal_form_runtime(self._choice_groups)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.scroll = FluentScrollArea()
        self.scroll.setWidgetResizable(True)
        # Opaque, so a scroll blits the page instead of repainting it whole.
        body = FluentPageBody()
        body_layout = QtWidgets.QVBoxLayout(body)
        margin = scaled_px(14, minimum=8)
        body_layout.setContentsMargins(margin, margin, margin, margin)
        body_layout.setSpacing(scaled_px(10, minimum=6))

        header = FluentFrame(bordered=False)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = FluentLabel(self.panel_id)
        self.kind_label = FluentLabel("")
        self.kind_label.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(FluentLabel("Plot kind (fixed):"))
        header_layout.addWidget(self.kind_label)
        body_layout.addWidget(header)

        panel_group = FluentGroupBox("Panel")
        panel_layout = QtWidgets.QVBoxLayout(panel_group)
        panel_layout.setContentsMargins(margin, margin, margin, margin)
        self.panel_form = FluentParameterForm(
            FormSpec(()), {}, runtime=self._signal_runtime
        )
        self.panel_form.changed.connect(self._panel_value_changed)
        panel_layout.addWidget(self.panel_form)
        body_layout.addWidget(panel_group)

        self.parameter_groups: dict[str, FluentGroupBox] = {}
        self.parameter_forms: dict[str, FluentParameterForm] = {}
        self.parameter_unavailable: dict[str, FluentLabel] = {}
        for section, title in (
            ("semantic", "Semantic"),
            ("display", "Display"),
            ("fit", "Fit"),
        ):
            group = FluentGroupBox(title)
            layout = QtWidgets.QVBoxLayout(group)
            layout.setContentsMargins(margin, margin, margin, margin)
            unavailable = FluentLabel("")
            unavailable.setWordWrap(True)
            unavailable.setStyleSheet(
                f"color: {GREY}; background: transparent; border: none;"
            )
            layout.addWidget(unavailable)
            form = FluentParameterForm(FormSpec(()), {})
            form.changed.connect(
                lambda key, name=section: self._mapping_value_changed(name, str(key))
            )
            layout.addWidget(form)
            self.parameter_groups[section] = group
            self.parameter_forms[section] = form
            self.parameter_unavailable[section] = unavailable
            body_layout.addWidget(group)

        snapshot_group = FluentGroupBox("Frozen snapshot")
        self.snapshot_group = snapshot_group
        snapshot_layout = QtWidgets.QVBoxLayout(snapshot_group)
        snapshot_layout.setContentsMargins(margin, margin, margin, margin)
        self.surface_holder = FluentFrame(bordered=False)
        self.surface_holder.setMinimumHeight(scaled_px(360, minimum=280))
        self.surface_layout = QtWidgets.QVBoxLayout(self.surface_holder)
        self.surface_layout.setContentsMargins(0, 0, 0, 0)
        self.surface_layout.setSpacing(0)
        self.surface_placeholder = FluentLabel("No frozen plot surface")
        self.surface_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.surface_layout.addWidget(self.surface_placeholder)
        snapshot_layout.addWidget(self.surface_holder, 1)
        self.snapshot_label = FluentLabel("No frozen snapshot")
        self.snapshot_label.setWordWrap(True)
        snapshot_layout.addWidget(self.snapshot_label)
        snapshot_actions = QtWidgets.QHBoxLayout()
        self.refresh_button = FluentButton("Refresh snapshot", color=ORANGE)
        self.refresh_button.clicked.connect(self.snapshot_refresh_requested.emit)
        snapshot_actions.addWidget(self.refresh_button)
        snapshot_actions.addStretch(1)
        snapshot_layout.addLayout(snapshot_actions)
        body_layout.addWidget(snapshot_group)

        interaction_group = FluentGroupBox("Interaction")
        self.interaction_group = interaction_group
        interaction_layout = QtWidgets.QVBoxLayout(interaction_group)
        interaction_layout.setContentsMargins(margin, margin, margin, margin)
        self.interaction_note = FluentLabel(
            "Area controls the direct producer ROI when present; otherwise "
            "the frozen plot's zoom/pan viewport controls it."
        )
        self.interaction_note.setWordWrap(True)
        self.interaction_note.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        interaction_layout.addWidget(self.interaction_note)
        body_layout.addWidget(interaction_group)

        self.producer_group = FluentGroupBox("Direct producer")
        producer_layout = QtWidgets.QVBoxLayout(self.producer_group)
        producer_layout.setContentsMargins(margin, margin, margin, margin)
        self.producer_summary = FluentLabel(
            "This signal has no editable direct producer."
        )
        self.producer_summary.setWordWrap(True)
        self.producer_summary.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        producer_layout.addWidget(self.producer_summary)
        producer_actions = QtWidgets.QHBoxLayout()
        producer_actions.addStretch(1)
        self.open_producer_button = FluentButton("Open Producer", color=ACCENT)
        self.open_producer_button.setToolTip(
            "Open this node's existing Logic Edit tab"
        )
        self.open_producer_button.clicked.connect(self._open_producer)
        producer_actions.addWidget(self.open_producer_button)
        producer_layout.addLayout(producer_actions)
        body_layout.addWidget(self.producer_group)

        save_group = FluentGroupBox("Save figure")
        self.save_group = save_group
        save_layout = QtWidgets.QVBoxLayout(save_group)
        save_layout.setContentsMargins(margin, margin, margin, margin)
        save_layout.setSpacing(scaled_px(8, minimum=5))
        label_width = setting_label_width(("Directory", "Name", "Format", "File"))
        self.save_directory = FluentPathEdit(
            "",
            mode="dir",
            caption="Choose where to save",
            parent=save_group,
        )
        save_layout.addWidget(FluentSettingRow(
            "Directory", self.save_directory, label_width=label_width, parent=save_group
        ))
        name_controls = QtWidgets.QWidget(save_group)
        name_layout = QtWidgets.QHBoxLayout(name_controls)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.setSpacing(scaled_px(8, minimum=5))
        self.save_auto_name = FluentSwitch("Auto-name", name_controls)
        self.save_auto_name.setChecked(True)
        self.save_name = FluentLineEdit()
        self.save_name.setPlaceholderText("File name")
        name_layout.addWidget(self.save_auto_name)
        name_layout.addWidget(self.save_name, 1)
        save_layout.addWidget(FluentSettingRow(
            "Name", name_controls, label_width=label_width, parent=save_group
        ))
        self.save_format = FluentComboBox(save_group)
        for extension in _IMAGE_FORMATS:
            self.save_format.addItem(extension.upper(), extension)
        save_layout.addWidget(FluentSettingRow(
            "Format", self.save_format, label_width=label_width, parent=save_group
        ))
        self.save_preview = FluentReadoutEdit("")
        save_layout.addWidget(FluentSettingRow(
            "File", self.save_preview, label_width=label_width, parent=save_group
        ))
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        self.save_button = FluentButton("Save Fig", color=ACCENT)
        actions.addWidget(self.save_button)
        save_layout.addLayout(actions)
        body_layout.addWidget(save_group)
        self.save_directory.changed.connect(self._update_save_controls)
        self.save_auto_name.toggled.connect(self._update_save_controls)
        self.save_name.textChanged.connect(self._update_save_controls)
        self.save_format.currentIndexChanged[int].connect(self._update_save_controls)
        self.save_button.clicked.connect(self._save_figure)
        body_layout.addStretch(1)

        self.scroll.setWidget(body)
        outer.addWidget(self.scroll)
        self.update_projection(projection)

    def update_projection(self, projection: Mapping[str, object]) -> None:
        incoming = dict(projection)
        state = panel_state_document(incoming.get("state") or {})
        self._projection = incoming
        self._state = state
        self._signal_groups = tuple(incoming.get("signal_options") or ())
        self._overlay_groups = tuple(incoming.get("overlay_signal_options") or ())
        surface = incoming.get("parameter_surface")
        paints_images = bool(
            isinstance(surface, Mapping) and surface.get("paints_images")
        )
        self.title_label.setText(state["title"] or self.panel_id)
        kind_text = state["kind"].replace("_", " ") or "automatic"
        if state["cell_kind"]:
            kind_text += f" · {state['cell_kind'].replace('_', ' ')} cells"
        self.kind_label.setText(kind_text)
        if not self._save_directory_initialized:
            self.save_directory.setText(str(incoming.get("save_directory") or ""))
            self._save_directory_initialized = True
        if not self._save_name_initialized:
            self.save_name.setText(state["title"] or self.panel_id)
            self._save_name_initialized = True

        fields: list[FormFieldProps] = [
            FormFieldProps(
                "title", "text", "Panel name", default=state["title"]
            ),
            FormFieldProps(
                "signal",
                "keyed_choice",
                "Signal",
                default=state["signal"],
                required=True,
            ),
        ]
        if paints_images:
            fields.append(FormFieldProps(
                "overlay_signal",
                "keyed_choice",
                "Overlay",
                default=state["overlay_signal"],
            ))
        fields.append(
            FormFieldProps(
                "size",
                "choice",
                "Panel size",
                default=state["size"],
                choices=self._size_choices(
                    incoming.get("size_choices"), state["size"]
                ),
            )
        )
        live = bool(incoming.get("live", True))
        self.snapshot_group.setVisible(live)
        self.interaction_group.setVisible(live)
        self.producer_group.setVisible(live)
        self.save_group.setVisible(live)
        if live:
            fields.append(interval_form_field(
                incoming.get("interval_choices"),
                state["interval_ms"],
            ))
        values: dict[str, object] = {
            "title": state["title"],
            "signal": state["signal"],
            "size": state["size"],
        }
        if live:
            values["interval_ms"] = state["interval_ms"]
        if paints_images:
            values["overlay_signal"] = state["overlay_signal"]
        self._science_locked = bool(
            isinstance(surface, Mapping) and surface.get("science_locked")
        )
        self.panel_form.reconcile(FormSpec(tuple(fields)), values)
        self.panel_form.widget_for("signal").setEnabled(bool(self._signal_groups))

        for section in ("semantic", "display", "fit"):
            declared = parameter_fields(surface, section)
            self._parameter_fields[section] = {
                str(field["key"]): field for field in declared
            }
            if declared:
                spec = parameter_form_spec(declared)
                form_values = parameter_form_values(declared)
            else:
                spec = FormSpec(())
                form_values = {}
            unavailable = str(
                surface.get(f"{section}_unavailable") or ""
            ) if isinstance(surface, Mapping) else ""
            self.parameter_unavailable[section].setText(unavailable)
            self.parameter_unavailable[section].setVisible(bool(unavailable))
            self.parameter_forms[section].reconcile(spec, form_values)
            self.parameter_groups[section].setVisible(
                bool(spec.fields) or bool(unavailable)
            )

        stale = bool(incoming.get("stale"))
        snapshot = incoming.get("frozen_snapshot")
        self.snapshot_label.setText(self._snapshot_text(incoming, stale=stale))
        self.snapshot_label.setStyleSheet(
            f"color: {ORANGE if stale else GREY}; background: transparent; border: none;"
        )
        self._snapshot_can_save = snapshot is not None and not stale
        self.save_button.setToolTip(
            "Refresh the stale snapshot before saving" if stale else ""
        )
        self.interaction_note.setText(
            (
                "Zoom and pan remain available. This frozen signal is stale, so "
                "committed selections are ignored until Refresh snapshot."
            )
            if stale
            else (
                "Area controls the direct producer ROI when present; otherwise "
                "the frozen plot's zoom/pan viewport controls it."
            )
        )
        producer_node_id = str(incoming.get("producer_node_id") or "")
        self.producer_summary.setText(
            f"Logic node: {producer_node_id}"
            if producer_node_id
            else "This signal has no editable direct producer."
        )
        self.open_producer_button.setEnabled(live and bool(producer_node_id))
        self.set_mutation_enabled(self._mutation_enabled)

    def set_surface(self, widget: QtWidgets.QWidget | None) -> None:
        """Mount or replace the plot widget supplied by the console handle."""

        if widget is not None and not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("surface must be QWidget or None")
        if widget is self._surface:
            _set_interaction(widget, self._selectors_on)
            return
        previous = self._surface
        if previous is not None:
            previous.removeEventFilter(self)
            self.surface_layout.removeWidget(previous)
            previous.hide()
            previous.setParent(None)
        self._surface = widget
        if widget is None:
            self.surface_placeholder.show()
            return
        self.surface_placeholder.hide()
        widget.setParent(self.surface_holder)
        widget.installEventFilter(self)
        _set_interaction(widget, self._selectors_on)
        self.surface_layout.addWidget(widget)
        widget.show()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if watched is self._surface and not self._selectors_on:
            if event.type() == QtCore.QEvent.Wheel:
                QtWidgets.QApplication.sendEvent(self.scroll.viewport(), event)
                return True
            if event.type() in {
                QtCore.QEvent.MouseButtonPress,
                QtCore.QEvent.MouseButtonRelease,
                QtCore.QEvent.MouseButtonDblClick,
                QtCore.QEvent.MouseMove,
            }:
                return True
        return super().eventFilter(watched, event)

    def set_selectors_enabled(self, enabled: bool) -> None:
        self._selectors_on = bool(enabled)
        _set_interaction(self._surface, self._selectors_on)

    def set_mutation_enabled(self, enabled: bool) -> None:
        """Gate saved-state changes while leaving the frozen plot inspectable."""

        self._mutation_enabled = bool(enabled)
        self.panel_form.setEnabled(self._mutation_enabled)
        if self._mutation_enabled:
            for key in ("signal", "overlay_signal", "cell_kind"):
                if key in self.panel_form.spec.keys:
                    self.panel_form.widget_for(key).setEnabled(
                        not self._science_locked
                    )
            if "signal" in self.panel_form.spec.keys:
                self.panel_form.widget_for("signal").setEnabled(
                    bool(self._signal_groups) and not self._science_locked
                )
        for section, form in self.parameter_forms.items():
            form.setEnabled(
                self._mutation_enabled
                and not (section == "semantic" and self._science_locked)
            )
        self.refresh_button.setEnabled(self._mutation_enabled)
        self._update_save_controls()

    def _save_path(self) -> Path | None:
        directory_text = self.save_directory.text().strip()
        if not directory_text:
            return None
        extension = str(self.save_format.currentData() or "png")
        if self.save_auto_name.isChecked():
            name = stamped_file_name(
                str(self._state.get("title") or self.panel_id),
                extension,
                stamp=self._save_stamp,
            )
        else:
            typed = self.save_name.text().strip().replace("/", "_").replace("\\", "_")
            if not typed:
                return None
            name = f"{Path(typed).stem}.{extension}"
        return Path(directory_text).expanduser() / name

    def _update_save_controls(self, *_args: object) -> None:
        editable = self._mutation_enabled
        self.save_directory.setEnabled(editable)
        self.save_auto_name.setEnabled(editable)
        self.save_name.setEnabled(editable and not self.save_auto_name.isChecked())
        self.save_format.setEnabled(editable)
        path = self._save_path()
        self.save_preview.setText("" if path is None else str(path))
        self.save_button.setEnabled(
            editable and self._snapshot_can_save and path is not None
        )

    def _save_figure(self) -> None:
        path = self._save_path()
        if path is None or not self.save_button.isEnabled():
            return
        self.save_figure_requested.emit(str(path))
        self._save_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._update_save_controls()

    @staticmethod
    def _size_choices(values: object, current: str) -> tuple[FormChoice, ...]:
        choices = tuple(str(value) for value in tuple(values or ()))
        if not choices:
            raise RuntimeError("panel size choices were not projected")
        if current not in choices:
            raise ValueError(
                f"unknown panel size {current!r}; choose from {', '.join(choices)}"
            )
        return tuple(FormChoice(value, value) for value in choices)

    def _panel_value_changed(self, key: str) -> None:
        try:
            value = self.panel_form.read_value(str(key))
        except (TypeError, ValueError) as error:
            self.snapshot_label.setText(str(error))
            return
        self.state_changed.emit({str(key): value})

    def _choice_groups(self, field: str):
        if str(field) == "signal":
            return self._signal_groups
        if str(field) == "overlay_signal":
            return self._overlay_groups
        return ()

    def _mapping_value_changed(self, section: str, key: str) -> None:
        try:
            selected = self._parameter_fields[section][key]
            owner = str(selected.get("edit_section") or section)
            owned = tuple(
                field
                for field in self._parameter_fields[section].values()
                if str(field.get("edit_section") or section) == owner
            )
            values = parameter_edit_values(
                owned,
                key,
                self.parameter_forms[section].read_value,
            )
        except (KeyError, TypeError, ValueError) as error:
            self.snapshot_label.setText(str(error))
            return
        self.state_changed.emit({owner: values})

    def _open_producer(self) -> None:
        node_id = str(self._projection.get("producer_node_id") or "")
        if node_id and bool(self._projection.get("live", True)):
            self.producer_edit_requested.emit(node_id)

    @staticmethod
    def _snapshot_text(projection: Mapping[str, object], *, stale: bool) -> str:
        snapshot = projection.get("frozen_snapshot")
        if snapshot is None:
            return "No frozen snapshot is displayed."
        pieces = [f"Signal: {str(projection.get('frozen_signal') or '(unknown)')}"]
        revision = getattr(snapshot, "revision", None)
        if revision is not None:
            pieces.append(f"revision {revision}")
        data = getattr(snapshot, "data", None)
        shape = getattr(data, "shape", None)
        if shape is not None:
            pieces.append(f"shape {tuple(shape)}")
        if stale:
            # Two different facts wore one sentence, and for the common one it
            # was false: a panel whose SIGNAL was retargeted, and a panel whose
            # frozen picture belongs to a run that has since been replaced.
            state = projection.get("state")
            signal = ""
            if isinstance(state, Mapping):
                signal = str(state.get("signal") or "")
            frozen_signal = str(projection.get("frozen_signal") or "")
            pieces.append(
                "STALE: this panel now shows another signal; refresh before Save Fig"
                if signal and frozen_signal and signal != frozen_signal
                else "STALE: frozen from an earlier run; refresh before Save Fig"
            )
        else:
            pieces.append("frozen and ready to save")
        return " · ".join(pieces)


__all__ = ["PanelEditorView"]
