"""One non-modal TaskConsole Panel Edit tab.

The widget consumes a plain Workbench projection.  It deliberately knows
nothing about signal publications, plot specs, measurement descriptors, or
devices: it shows the state it is given and emits named operator intents.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PyQt5 import QtCore, QtWidgets

from zlc_ui.board import PANEL_SIZES
from zlc_ui.fluent import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    FluentButton,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentScrollArea,
    scaled_px,
)
from zlc_ui.form import (
    FluentParameterForm,
    FormChoice,
    FormFieldProps,
    FormSpec,
)

from ._panel_projection import (
    decode_parameter_value,
    interval_form_field,
    panel_state_document,
    parameter_fields,
    parameter_form_spec,
    parameter_form_values,
    signal_form_choices,
)
from .logic_editor_view import LogicEditorView


class PanelEditorView(QtWidgets.QWidget):
    """Editable projection of one Workbench-owned panel state and snapshot."""

    state_changed = QtCore.pyqtSignal(object)
    snapshot_refresh_requested = QtCore.pyqtSignal()
    producer_draft_changed = QtCore.pyqtSignal(str, object)
    producer_restart_requested = QtCore.pyqtSignal()
    save_figure_requested = QtCore.pyqtSignal()

    def __init__(self, panel_id: str, projection: Mapping[str, object], parent=None) -> None:
        super().__init__(parent)
        self.panel_id = str(panel_id)
        self._mutation_enabled = True
        self._projection: dict[str, object] = {}
        self._state: dict[str, Any] = {}
        self._parameter_fields: dict[str, dict[str, dict[str, object]]] = {
            "semantic": {},
            "display": {},
            "fit": {},
        }
        self._producer_editor: LogicEditorView | None = None
        self._surface: QtWidgets.QWidget | None = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        body.setStyleSheet("background: transparent;")
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
        self.panel_form = FluentParameterForm(FormSpec(()), {})
        self.panel_form.changed.connect(self._panel_value_changed)
        panel_layout.addWidget(self.panel_form)
        body_layout.addWidget(panel_group)

        self.parameter_groups: dict[str, FluentGroupBox] = {}
        self.parameter_forms: dict[str, FluentParameterForm] = {}
        self.parameter_unavailable: dict[str, FluentLabel] = {}
        for section, title in (
            ("semantic", "Semantic parameters"),
            ("display", "Display / interaction parameters"),
            ("fit", "Fit parameters / result"),
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
        interaction_layout = QtWidgets.QVBoxLayout(interaction_group)
        interaction_layout.setContentsMargins(margin, margin, margin, margin)
        self.interaction_note = FluentLabel(
            "Zoom and pan remain display-only. Committed Area/Range selections "
            "are routed to the direct producer draft shown below."
        )
        self.interaction_note.setWordWrap(True)
        self.interaction_note.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        interaction_layout.addWidget(self.interaction_note)
        body_layout.addWidget(interaction_group)

        self.producer_group = FluentGroupBox("Direct producer")
        self.producer_layout = QtWidgets.QVBoxLayout(self.producer_group)
        self.producer_layout.setContentsMargins(margin, margin, margin, margin)
        self.producer_empty = FluentLabel("This signal has no editable direct producer.")
        self.producer_empty.setWordWrap(True)
        self.producer_empty.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        self.producer_layout.addWidget(self.producer_empty)
        producer_actions = QtWidgets.QHBoxLayout()
        producer_actions.addStretch(1)
        self.producer_restart_button = FluentButton("Producer Restart", color=GREEN)
        self.producer_restart_button.setToolTip(
            "Restart the direct producer once with its current shared draft"
        )
        self.producer_restart_button.clicked.connect(
            self.producer_restart_requested.emit
        )
        producer_actions.addWidget(self.producer_restart_button)
        self.producer_layout.addLayout(producer_actions)
        body_layout.addWidget(self.producer_group)

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        self.save_button = FluentButton("Save Fig", color=ACCENT)
        self.save_button.clicked.connect(self.save_figure_requested.emit)
        actions.addWidget(self.save_button)
        body_layout.addLayout(actions)
        body_layout.addStretch(1)

        scroll.setWidget(body)
        outer.addWidget(scroll)
        self.update_projection(projection)

    def update_projection(self, projection: Mapping[str, object]) -> None:
        incoming = dict(projection)
        state = panel_state_document(incoming.get("state") or {})
        self._projection = incoming
        self._state = state
        self.title_label.setText(state["title"] or self.panel_id)
        kind_text = state["kind"].replace("_", " ") or "automatic"
        if state["cell_kind"]:
            kind_text += f" · {state['cell_kind'].replace('_', ' ')} cells"
        self.kind_label.setText(kind_text)

        choices = signal_form_choices(incoming.get("signal_options"), state["signal"])
        fields: list[FormFieldProps] = [
            FormFieldProps("title", "text", "Title", default=state["title"]),
            FormFieldProps(
                "signal",
                "choice",
                "Signal",
                default=state["signal"] if choices else None,
                choices=choices,
                required=not bool(choices),
                unavailable_reason="no signals available" if not choices else "",
            ),
            FormFieldProps(
                "size",
                "choice",
                "Panel size",
                default=state["size"],
                choices=self._size_choices(state["size"]),
            ),
            interval_form_field(
                incoming.get("interval_choices"),
                state["interval_ms"],
            ),
        ]
        values: dict[str, object] = {
            "title": state["title"],
            "signal": state["signal"] if choices else None,
            "size": state["size"],
            "interval_ms": state["interval_ms"],
        }
        surface = incoming.get("parameter_surface")
        overlay = (
            dict(surface.get("site_overlay") or {})
            if isinstance(surface, Mapping)
            else {}
        )
        if overlay:
            overlay["value"] = state["site_overlay"]
            fields.append(parameter_form_spec((overlay,)).fields[0])
            values["site_overlay"] = state["site_overlay"]
        self.panel_form.reconcile(FormSpec(tuple(fields)), values)

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
        self.save_button.setEnabled(
            self._mutation_enabled and snapshot is not None and not stale
        )
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
                "Zoom and pan remain display-only. Committed Area/Range "
                "selections are routed to the direct producer draft shown below."
            )
        )
        self._update_producer(incoming.get("producer_logic"))
        self.set_mutation_enabled(self._mutation_enabled)

    def set_surface(self, widget: QtWidgets.QWidget | None) -> None:
        """Mount or replace the plot widget supplied by the console handle."""

        if widget is not None and not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("surface must be QWidget or None")
        if widget is self._surface:
            return
        previous = self._surface
        if previous is not None:
            self.surface_layout.removeWidget(previous)
            previous.hide()
            previous.setParent(None)
        self._surface = widget
        if widget is None:
            self.surface_placeholder.show()
            return
        self.surface_placeholder.hide()
        widget.setParent(self.surface_holder)
        self.surface_layout.addWidget(widget)
        widget.show()

    def set_mutation_enabled(self, enabled: bool) -> None:
        """Gate saved-state changes while leaving the frozen plot inspectable."""

        self._mutation_enabled = bool(enabled)
        self.panel_form.setEnabled(self._mutation_enabled)
        for form in self.parameter_forms.values():
            form.setEnabled(self._mutation_enabled)
        self.refresh_button.setEnabled(self._mutation_enabled)
        self.save_button.setEnabled(
            self._mutation_enabled
            and self._projection.get("frozen_snapshot") is not None
            and not bool(self._projection.get("stale"))
        )
        self.producer_restart_button.setEnabled(
            self._mutation_enabled and self._producer_editor is not None
        )
        if self._producer_editor is not None:
            self._producer_editor.set_mutation_enabled(self._mutation_enabled)

    @staticmethod
    def _size_choices(current: str) -> tuple[FormChoice, ...]:
        values = list(PANEL_SIZES)
        if current and current not in values:
            values.insert(0, current)
        return tuple(FormChoice(value, value) for value in values)

    def _panel_value_changed(self, key: str) -> None:
        try:
            value = self.panel_form.read_value(str(key))
        except (TypeError, ValueError) as error:
            self.snapshot_label.setText(str(error))
            return
        self.state_changed.emit({str(key): value})

    def _mapping_value_changed(self, section: str, key: str) -> None:
        try:
            raw = self.parameter_forms[section].read_value(key)
            declared = self._parameter_fields[section].get(key)
            if declared is None:
                raise KeyError(key)
            value = decode_parameter_value(declared, raw)
        except (KeyError, TypeError, ValueError) as error:
            self.snapshot_label.setText(str(error))
            return
        self.state_changed.emit({section: {key: value}})

    def _update_producer(self, projection: object) -> None:
        if not isinstance(projection, Mapping):
            if self._producer_editor is not None:
                self.producer_layout.removeWidget(self._producer_editor)
                self._producer_editor.setParent(None)
                self._producer_editor.deleteLater()
                self._producer_editor = None
            self.producer_empty.show()
            self.producer_restart_button.setEnabled(False)
            return

        node_id = str(projection.get("node_id") or "")
        if self._producer_editor is None or self._producer_editor.node_id != node_id:
            if self._producer_editor is not None:
                self.producer_layout.removeWidget(self._producer_editor)
                self._producer_editor.setParent(None)
                self._producer_editor.deleteLater()
            editor = LogicEditorView(node_id, projection, show_actions=False)
            editor.set_mutation_enabled(self._mutation_enabled)
            editor.setMinimumHeight(scaled_px(330, minimum=260))
            editor.draft_changed.connect(
                lambda patch, nid=node_id: self.producer_draft_changed.emit(nid, patch)
            )
            self._producer_editor = editor
            self.producer_layout.insertWidget(1, editor)
        else:
            self._producer_editor.update_projection(projection)
        self.producer_empty.hide()
        self.producer_restart_button.setEnabled(self._mutation_enabled)

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
            pieces.append("STALE: panel signal changed; refresh before Save Fig")
        else:
            pieces.append("frozen and ready to save")
        return " · ".join(pieces)


__all__ = ["PanelEditorView"]
