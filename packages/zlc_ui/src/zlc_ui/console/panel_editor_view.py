"""One non-modal Panel Edit tab, in the TaskConsole or the FigureViewer.

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
    detach_widget,
    FluentButton,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentPageBody,
    FluentPathEdit,
    FluentReadoutEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSegmentedButtons,
    FluentSettingRow,
    FluentSwitch,
    MUTED_LABEL_STYLE,
    card_pad,
    card_title_row,
    muted_note_label,
    row_gap,
    scaled_px,
    section_gap,
    set_status_ink,
    setting_label_width,
    stamped_file_name,
    retire_widget,
    window_pad,
)
from zlc_ui.form import (
    FluentParameterForm,
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
    size_form_field,
)
from .logic_editor_view import _contribution_label_width
from .panel_card_view import (
    PARAMETER_SECTIONS,
    _set_interaction,
    _shows_unrecorded_pick,
    panel_section,
    relay_surface_errors,
    release_surface_errors,
    share_label_column,
)


_IMAGE_FORMATS = ("png", "pdf", "svg")


class PanelEditorView(QtWidgets.QWidget):
    """Editable projection of one Workbench-owned panel state and snapshot."""

    state_changed = QtCore.pyqtSignal(object)
    snapshot_refresh_requested = QtCore.pyqtSignal()
    producer_edit_requested = QtCore.pyqtSignal(str)
    save_figure_requested = QtCore.pyqtSignal(str)
    #: A refusal the mounted Edit plot surface reported (``errorOccurred``),
    #: relayed exactly as the card relays its own: the console reports it
    #: on the same channel.  Unconnected, an Edit surface's refusals were
    #: the one plot error the console never heard.
    plot_error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        panel_id: str,
        projection: Mapping[str, object],
        parent=None,
        *,
        page_inset: int | None = None,
    ) -> None:
        """``page_inset`` is where the host's tab pages start their content
        (its tab widget's page inset): the Task Console's window pad unless
        the host says otherwise -- the Figure Viewer lays its pages on one
        card pad."""

        super().__init__(parent)
        self.panel_id = str(panel_id)
        self._science_locked = False
        self._projection: dict[str, object] = {}
        #: The projection minus what the snapshot half applies, as last
        #: reconciled from.  A shot that changes neither rebuilds nothing.
        self._applied_structure: dict[str, object] | None = None
        self._applied_producer: dict[str, object] | None = None
        self._state: dict[str, Any] = {}
        self._parameter_fields: dict[str, dict[str, dict[str, object]]] = {
            "semantic": {},
            "display": {},
            "fit": {},
        }
        self._surface: QtWidgets.QWidget | None = None
        self._surface_error_connected = False
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
        # Opaque, so a scroll blits the page instead of repainting it whole.
        body = FluentPageBody()
        body_layout = QtWidgets.QVBoxLayout(body)
        # The host's page inset on every side, so this page starts where the
        # host's other pages and its first tab's underline start.  The bar's
        # gutter is held whether or not the bar shows (a width-bounded
        # mount) and stands inside the right inset: the cards end on the
        # host's right edge with or without a bar.
        inset = window_pad() if page_inset is None else int(page_inset)
        gutter = self.scroll.verticalScrollBar().sizeHint().width()
        body_layout.setContentsMargins(inset, inset, max(0, inset - gutter), inset)
        # The cards stack a row gap apart, a section gap under the header.
        body_layout.setSpacing(row_gap())
        pad = card_pad()

        # The page's name as its heading, the panel's fixed kind after it in
        # muted ink -- the Logic editor's "title (kind)".
        header = FluentFrame(bordered=False)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(row_gap())
        self.title_label = FluentSectionLabel(self.panel_id)
        self.kind_label = FluentLabel("")
        self.kind_label.setStyleSheet(MUTED_LABEL_STYLE)
        self.kind_label.setToolTip("The plot kind is fixed when the panel is added")
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.kind_label)
        header_layout.addStretch(1)
        body_layout.addWidget(header)
        body_layout.addSpacing(section_gap() - row_gap())

        # Every card lays its title out as its first row (card_title_row),
        # as the Figure Viewer's data cards and the Device Manager's do; the
        # four setting sections are built as the Setting frame builds them.
        self.panel_form = FluentParameterForm(
            FormSpec(()), {}, runtime=self._signal_runtime
        )
        self.panel_form.changed.connect(self._panel_value_changed)
        self.panel_form.value_normalized.connect(self._panel_value_changed)
        body_layout.addWidget(_section_card("Panel", self.panel_form))

        self.parameter_groups: dict[str, FluentGroupBox] = {}
        self.parameter_forms: dict[str, FluentParameterForm] = {}
        self.parameter_unavailable: dict[str, QtWidgets.QLabel] = {}
        for section, title in PARAMETER_SECTIONS:
            unavailable = muted_note_label("")
            unavailable.setWordWrap(True)
            form = FluentParameterForm(FormSpec(()), {})
            form.changed.connect(
                lambda key, name=section: self._mapping_value_changed(name, str(key))
            )
            form.value_normalized.connect(
                lambda key, name=section: self._mapping_value_changed(name, str(key))
            )
            group = _section_card(title, form, note=unavailable)
            self.parameter_groups[section] = group
            self.parameter_forms[section] = form
            self.parameter_unavailable[section] = unavailable
            body_layout.addWidget(group)

        snapshot_group = FluentGroupBox("Frozen snapshot", title_strip_px=0)
        self.snapshot_group = snapshot_group
        snapshot_layout = QtWidgets.QVBoxLayout(snapshot_group)
        snapshot_layout.setContentsMargins(pad, pad, pad, pad)
        snapshot_layout.setSpacing(row_gap())
        snapshot_layout.addLayout(card_title_row("Frozen snapshot")[0])
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
        snapshot_actions.setSpacing(row_gap())
        self.refresh_button = FluentButton("Refresh snapshot")
        self.refresh_button.clicked.connect(self.snapshot_refresh_requested.emit)
        snapshot_actions.addWidget(self.refresh_button)
        snapshot_actions.addStretch(1)
        snapshot_layout.addLayout(snapshot_actions)
        body_layout.addWidget(snapshot_group)

        self.producer_group = FluentGroupBox("Direct producer", title_strip_px=0)
        producer_layout = QtWidgets.QVBoxLayout(self.producer_group)
        producer_layout.setContentsMargins(pad, pad, pad, pad)
        producer_layout.setSpacing(row_gap())
        producer_layout.addLayout(card_title_row("Direct producer")[0])
        self.producer_summary = muted_note_label(
            "This signal has no editable direct producer."
        )
        self.producer_summary.setWordWrap(True)
        producer_layout.addWidget(self.producer_summary)
        self.producer_form = FluentParameterForm(FormSpec(()), {})
        self.producer_form.setEnabled(False)
        producer_layout.addWidget(self.producer_form)
        self._producer_contributions: dict[object, QtWidgets.QWidget] = {}
        producer_actions = QtWidgets.QHBoxLayout()
        producer_actions.setSpacing(row_gap())
        producer_actions.addStretch(1)
        self.open_producer_button = FluentButton("Open producer")
        self.open_producer_button.setToolTip(
            "Open this node's existing Logic Edit tab"
        )
        self.open_producer_button.clicked.connect(self._open_producer)
        producer_actions.addWidget(self.open_producer_button)
        producer_layout.addLayout(producer_actions)
        body_layout.addWidget(self.producer_group)

        save_group = FluentGroupBox("Save figure", title_strip_px=0)
        self.save_group = save_group
        save_layout = QtWidgets.QVBoxLayout(save_group)
        save_layout.setContentsMargins(pad, pad, pad, pad)
        save_layout.setSpacing(row_gap())
        save_layout.addLayout(card_title_row("Save figure")[0])
        label_width = setting_label_width(("Directory", "Name", "Format", "File"))
        #: The Save figure rows and their words: they take the page's one
        #: label column with the forms above (see _share_label_column).
        self._save_rows: list[tuple[FluentSettingRow, str]] = []
        self.save_directory = FluentPathEdit(
            "",
            mode="dir",
            caption="Choose where to save",
            parent=save_group,
        )
        row = FluentSettingRow("Directory", self.save_directory, label_width=label_width, parent=save_group)
        self._save_rows.append((row, "Directory"))
        save_layout.addWidget(row)
        name_controls = QtWidgets.QWidget(save_group)
        name_layout = QtWidgets.QHBoxLayout(name_controls)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.setSpacing(row_gap())
        self.save_auto_name = FluentSwitch("Auto-name", name_controls)
        self.save_auto_name.setChecked(True)
        self.save_name = FluentLineEdit()
        self.save_name.setPlaceholderText("File name")
        self.save_name.set_placeholder_prompt()
        name_layout.addWidget(self.save_auto_name)
        name_layout.addWidget(self.save_name, 1)
        row = FluentSettingRow("Name", name_controls, label_width=label_width, parent=save_group)
        self._save_rows.append((row, "Name"))
        save_layout.addWidget(row)
        self.save_format = FluentSegmentedButtons(
            tuple((extension, extension.upper()) for extension in _IMAGE_FORMATS), save_group
        )
        self._save_extension = _IMAGE_FORMATS[0]
        self.save_format.set_current(self._save_extension)
        row = FluentSettingRow("Format", self.save_format, label_width=label_width, parent=save_group)
        self._save_rows.append((row, "Format"))
        save_layout.addWidget(row)
        self.save_preview = FluentReadoutEdit("")
        row = FluentSettingRow("File", self.save_preview, label_width=label_width, parent=save_group)
        self._save_rows.append((row, "File"))
        save_layout.addWidget(row)
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(row_gap())
        actions.addStretch(1)
        self.save_button = FluentButton("Save figure", role="primary")
        actions.addWidget(self.save_button)
        save_layout.addLayout(actions)
        body_layout.addWidget(save_group)
        self.save_directory.changed.connect(self._update_save_controls)
        self.save_auto_name.toggled.connect(self._update_save_controls)
        self.save_name.textChanged.connect(self._update_save_controls)
        self.save_format.activated.connect(self._save_format_picked)
        self.save_button.clicked.connect(self._save_figure)
        body_layout.addStretch(1)

        self.scroll.set_width_bounded_widget(body)
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
        self.kind_label.setText(f"({kind_text})")
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
        fields.append(size_form_field(incoming.get("size_choices"), state["size"]))
        fields.append(interval_form_field(
            incoming.get("interval_choices"),
            state["interval_ms"],
        ))
        values: dict[str, object] = {
            "title": state["title"],
            "signal": state["signal"],
            "size": state["size"],
        }
        values["interval_ms"] = state["interval_ms"]
        if paints_images:
            values["overlay_signal"] = state["overlay_signal"]
        self._science_locked = bool(
            isinstance(surface, Mapping) and surface.get("science_locked")
        )
        # FOUR FORMS AND TWO SIGNAL TREES, on a projection that arrives with
        # every landed shot and almost always says what it said last time.
        # The panel form declares the signal and overlay as keyed choices,
        # so reconciling it walks the whole grouped signal list; the three
        # parameter forms rebuild their specs from the surface.  All of it
        # reads only the projection members compared here -- the snapshot
        # age and the producer summary, which DO change every shot, are
        # applied by the half below and are excluded for that reason.
        volatile = ("stale", "frozen_snapshot", "data_advanced",
                    "source_status", "snapshot_error", "producer_node_id", "producer_projection")
        structure = {
            key: value for key, value in incoming.items() if key not in volatile
        }
        if structure != self._applied_structure:
            self._applied_structure = structure
            self.panel_form.reconcile(FormSpec(tuple(fields)), values)
            self.panel_form.widget_for("signal").setEnabled(
                bool(self._signal_groups)
            )

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
            self._share_label_column()
        elif _shows_unrecorded_pick(self.panel_form, state):
            # A refused pick comes back as the same projection; the picker
            # that sent it goes back to what the panel draws.
            self.panel_form.reconcile(FormSpec(tuple(fields)), values)

        self._update_snapshot_status()
        producer_node_id = str(incoming.get("producer_node_id") or "")
        self.producer_summary.setText(
            f"Logic node: {producer_node_id}"
            if producer_node_id
            else "This signal has no editable direct producer."
        )
        self.open_producer_button.setEnabled(bool(producer_node_id))
        self.set_producer_projection(incoming.get("producer_projection") or {})
        self._apply_science_lock()

    def set_producer_projection(self, projection: Mapping[str, object]) -> None:
        """Show linked draft fields without creating a second Logic editor."""

        incoming = dict(projection)
        if incoming == self._applied_producer:
            return
        self._applied_producer = incoming
        self._projection["producer_projection"] = incoming
        fields = incoming.get("read_only_fields", ())
        factories = tuple(incoming.get("ui_contributions") or ())
        for factory in tuple(self._producer_contributions):
            if factory not in factories:
                retire_widget(self._producer_contributions.pop(factory))
        managed = set()
        for factory in factories:
            widget = self._producer_contributions.get(factory)
            if widget is None:
                widget = factory(self.producer_group)
                if not set(widget.managed_fields).intersection(fields):
                    retire_widget(widget)
                    continue
                widget.setEnabled(False)
                self._producer_contributions[factory] = widget
                layout = self.producer_group.layout()
                layout.insertWidget(layout.count() - 1, widget)
            managed.update(widget.managed_fields)
            widget.update_projection(incoming)
        spec = incoming.get("form_spec", FormSpec(()))
        visible = FormSpec(tuple(field for field in spec.fields if field.key not in managed))
        values = incoming.get("form_values") or {}
        self.producer_form.reconcile(visible, {key: values[key] for key in visible.keys})
        self.producer_form.setVisible(bool(visible.fields))
        self._share_label_column()

    def _share_label_column(self) -> None:
        """One control column down the whole page.

        Each card's form sizes a label column to its own words, so Panel,
        Semantic, Display and Save figure would start their controls at as
        many x.  Every card takes the column the widest shown label of any
        card needs: the forms', the Save figure rows' and the producer
        contributions' that take the column.
        """

        forms = (self.panel_form, *self.parameter_forms.values(), self.producer_form)
        spec = (self._applied_producer or {}).get("form_spec", FormSpec(()))
        takers = tuple(
            widget for widget in self._producer_contributions.values()
            if callable(getattr(widget, "set_label_width", None))
        )
        column = share_label_column(forms, widths=(
            setting_label_width(text for _row, text in self._save_rows),
            *(_contribution_label_width(widget, spec) for widget in takers),
        ))
        for row, text in self._save_rows:
            row.set_label(text, width=column)
        for widget in takers:
            widget.set_label_width(column)

    def set_snapshot_status(self, status: Mapping[str, object]) -> None:
        """Update frozen age without reconciling the editor's forms."""

        if all(self._projection.get(key) == value for key, value in status.items()):
            return
        self._projection.update(status)
        self._update_snapshot_status()
        self._update_save_controls()

    def _update_snapshot_status(self) -> None:
        stale = bool(self._projection.get("stale"))
        snapshot = self._projection.get("frozen_snapshot")
        aged = bool(self._projection.get("data_advanced") or self._projection.get("source_status"))
        self.snapshot_label.setText(self._snapshot_text(self._projection, stale=stale))
        set_status_ink(
            self.snapshot_label,
            "error" if self._projection.get("snapshot_error")
            else "warning" if stale or aged else "idle",
        )
        self._snapshot_can_save = snapshot is not None and not stale
        self.save_button.setToolTip(
            "Refresh the stale snapshot before saving" if stale else ""
        )

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
            release_surface_errors(
                previous, self.plot_error, self._surface_error_connected
            )
            self._surface_error_connected = False
            self.surface_layout.removeWidget(previous)
            detach_widget(previous)
        self._surface = widget
        if widget is None:
            self.surface_placeholder.show()
            return
        self.surface_placeholder.hide()
        widget.setParent(self.surface_holder)
        widget.installEventFilter(self)
        self._surface_error_connected = relay_surface_errors(
            widget, self.plot_error
        )
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

    def _apply_science_lock(self) -> None:
        """A Task-protected panel keeps its signal, overlay and semantics."""

        if "overlay_signal" in self.panel_form.spec.keys:
            self.panel_form.widget_for("overlay_signal").setEnabled(
                not self._science_locked
            )
        if "signal" in self.panel_form.spec.keys:
            self.panel_form.widget_for("signal").setEnabled(
                bool(self._signal_groups) and not self._science_locked
            )
        self.parameter_forms["semantic"].setEnabled(not self._science_locked)
        self._update_save_controls()

    def _save_path(self) -> Path | None:
        directory_text = self.save_directory.text().strip()
        if not directory_text:
            return None
        extension = self._save_extension
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

    def _save_format_picked(self, extension: str) -> None:
        self._save_extension = str(extension)
        self.save_format.set_current(self._save_extension)
        self._update_save_controls()

    def _update_save_controls(self, *_args: object) -> None:
        self.save_name.setEnabled(not self.save_auto_name.isChecked())
        path = self._save_path()
        self.save_preview.setText("" if path is None else str(path))
        self.save_button.setEnabled(self._snapshot_can_save and path is not None)

    def _save_figure(self) -> None:
        path = self._save_path()
        if path is None or not self.save_button.isEnabled():
            return
        self.save_figure_requested.emit(str(path))
        self._save_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._update_save_controls()

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
            patch = parameter_edit_values(
                self._parameter_fields[section].values(),
                section,
                key,
                self.parameter_forms[section].read_value,
            )
        except (KeyError, TypeError, ValueError) as error:
            self.snapshot_label.setText(str(error))
            return
        self.state_changed.emit(patch)

    def _open_producer(self) -> None:
        node_id = str(self._projection.get("producer_node_id") or "")
        if node_id:
            self.producer_edit_requested.emit(node_id)

    @staticmethod
    def _snapshot_text(projection: Mapping[str, object], *, stale: bool) -> str:
        snapshot = projection.get("frozen_snapshot")
        if snapshot is None:
            return "No frozen snapshot is displayed."
        pieces = [f"Signal: {str(projection.get('frozen_signal') or '(unknown)')}"]
        ref = getattr(snapshot, "ref", None)
        revision = getattr(getattr(ref, "revision", None), "value", None)
        if revision is not None:
            pieces.append(f"revision {revision}")
        block = getattr(snapshot, "block", None)
        shape = getattr(getattr(block, "values", None), "shape", None)
        if shape is not None:
            pieces.append(f"shape {tuple(shape)}")
        if stale:
            state = projection.get("state")
            signal = ""
            if isinstance(state, Mapping):
                signal = str(state.get("signal") or "")
            frozen_signal = str(projection.get("frozen_signal") or "")
            pieces.append(
                "Stale: this panel now shows another signal; refresh before saving"
                if signal and frozen_signal and signal != frozen_signal
                else "Stale: panel configuration changed; refresh before saving"
            )
        elif projection.get("source_status") == "earlier_run":
            pieces.append("Frozen from an earlier run; Refresh to update; Save keeps this snapshot")
        elif projection.get("source_status") == "no_longer_current":
            pieces.append("Source is no longer current; Refresh to update; Save keeps this snapshot")
        elif projection.get("data_advanced"):
            pieces.append("Live data advanced; Refresh to update; Save keeps this snapshot")
        else:
            pieces.append("frozen and ready to save")
        if projection.get("snapshot_error"):
            pieces.append(f"Refresh failed: {projection['snapshot_error']}; Save keeps this snapshot")
        return " · ".join(pieces)


def _section_card(
    title: str, form: FluentParameterForm, *, note: QtWidgets.QWidget | None = None
) -> FluentGroupBox:
    """One setting section as a card of the page: :func:`panel_section`, a
    card pad inside the card's edge."""

    card = FluentGroupBox(title, title_strip_px=0)
    layout = QtWidgets.QVBoxLayout(card)
    pad = card_pad()
    layout.setContentsMargins(pad, pad, pad, pad)
    layout.addLayout(panel_section(title, form, note=note)[0])
    return card


def panel_editor_projection(owner: Any, projection: object) -> dict[str, object]:
    """An Edit projection carrying ``owner``'s projected size and interval choices."""

    incoming = dict(projection)
    incoming["interval_choices"] = owner._panel_intervals
    incoming["size_choices"] = owner._panel_sizes
    return incoming


def new_panel_editor(
    owner: Any,
    panel_id: str,
    projection: object,
    *,
    page_inset: int | None = None,
) -> PanelEditorView:
    """One Edit page with its intents relayed to ``owner``'s panel signals.

    The console handle and the FigureViewer both open these; made and wired
    here, once, the two Edit pages stay one page.  ``page_inset`` is the
    host's tab-page inset (:class:`PanelEditorView`).
    """

    key = str(panel_id)
    editor = PanelEditorView(
        key, panel_editor_projection(owner, projection), page_inset=page_inset
    )
    editor.state_changed.connect(
        lambda patch, pid=key: owner.panel_state_changed.emit(pid, patch)
    )
    editor.snapshot_refresh_requested.connect(
        lambda _=None, pid=key: owner.panel_snapshot_refresh_requested.emit(pid)
    )
    editor.save_figure_requested.connect(
        lambda path, pid=key: owner.panel_save_figure_requested.emit(pid, str(path))
    )
    # The Edit surface's failures travel the same channel as the card's:
    # one relay rule, one place a plot is reported.
    editor.plot_error.connect(
        lambda message, pid=key: owner.panel_plot_error.emit(pid, str(message))
    )
    return editor


__all__ = ["PanelEditorView", "new_panel_editor", "panel_editor_projection"]
