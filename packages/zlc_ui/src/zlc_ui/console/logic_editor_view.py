"""One non-modal TaskConsole Logic Edit tab.

The widget consumes a UI form projection and plain selector options.  It does
not know which package declared the node, how devices are resolved, or what a
Start does; those remain Workbench responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    GREEN,
    GREY,
    ORANGE,
    ORANGE_DARK,
    RED,
    RED_DARK,
    TEXT,
    FluentButton,
    FluentComboBox,
    FluentLabel,
    FluentReadoutEdit,
    FluentScrollArea,
    FluentSettingRow,
    FluentStatusDot,
    FluentTreeComboBox,
    FONT,
    fill_grouped_choice_combo,
    fluent_font_size,
    retire_widget,
    scaled_px,
    setting_label_width,
)
from zlc_ui.form import FluentParameterForm, FormSpec
from zlc_ui.form.qt_form import _form_label_width


class LogicEditorView(QtWidgets.QWidget):
    """Editable projection of one stopped/running Logic row draft."""

    draft_changed = QtCore.pyqtSignal(object)
    #: A file this node reads was edited elsewhere; re-read it.
    refresh_requested = QtCore.pyqtSignal()
    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal()
    auto_preview_changed = QtCore.pyqtSignal(bool)

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
        #: The projection minus what the status half applies, as last
        #: rebuilt from.  A beat that changes neither rebuilds nothing.
        self._applied_structure: dict[str, object] | None = None
        self._status_colours: tuple[str, str] = ("", "")
        #: The editor's ONE label column, shared by the selectors, both forms,
        #: the results and every contribution, so their fields start at one x.
        self._label_width = setting_label_width(())
        self._selector_rows: dict[QtWidgets.QWidget, tuple[FluentSettingRow, str]] = {}
        self._device_combos: dict[str, FluentComboBox] = {}
        self._acquisition_combo: FluentComboBox | None = None
        self._artifact_result_readouts: dict[str, FluentReadoutEdit] = {}
        self._contributions: dict[object, QtWidgets.QWidget] = {}

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

        # What this is on one line, how it stands on the next: the kind sat
        # a third of the way across and a long draft error wrapped into a
        # block beside it.  The status line is a dot and words, the way the
        # console's status strip says things.
        header = QtWidgets.QWidget()
        header.setStyleSheet("background: transparent;")
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(scaled_px(4, minimum=3))
        title_row = QtWidgets.QHBoxLayout()
        title_row.setSpacing(scaled_px(8, minimum=5))
        self.title_label = FluentLabel(self.node_id)
        self.title_label.setStyleSheet(
            f'QLabel {{ color: {TEXT}; font: {fluent_font_size()}pt "{FONT}"; '
            "font-weight: 600; background: transparent; }"
        )
        self.kind_label = FluentLabel("")
        self.kind_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        title_row.addWidget(self.title_label)
        title_row.addWidget(self.kind_label)
        title_row.addStretch(1)
        header_layout.addLayout(title_row)
        status_row = QtWidgets.QHBoxLayout()
        status_row.setSpacing(scaled_px(8, minimum=5))
        self.status_dot = FluentStatusDot(size=10, color=GREY)
        self.status_label = FluentLabel("stopped")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumWidth(0)
        # The dot sits on the FIRST line of a status that wraps.
        dot_column = QtWidgets.QVBoxLayout()
        dot_column.setContentsMargins(
            0, max(0, (self.status_label.fontMetrics().height() - self.status_dot.height()) // 2), 0, 0
        )
        dot_column.addWidget(self.status_dot)
        dot_column.addStretch(1)
        status_row.addLayout(dot_column)
        status_row.addWidget(self.status_label, 1)
        header_layout.addLayout(status_row)
        self._body_layout.addWidget(header)

        # The source and device choices are rows of the form, in its label
        # column -- not a separately boxed card with its own label style.
        self._selector_frame = QtWidgets.QWidget()
        self._selector_frame.setStyleSheet("background: transparent;")
        self._selector_layout = QtWidgets.QVBoxLayout(self._selector_frame)
        self._selector_layout.setContentsMargins(0, 0, 0, 0)
        self._selector_layout.setSpacing(scaled_px(6, minimum=4))
        self.source_combo: FluentComboBox | None = None
        self._body_layout.addWidget(self._selector_frame)

        empty = FormSpec(())
        self.artifact_form = FluentParameterForm(empty, {})
        self._body_layout.addWidget(self.artifact_form)
        self.artifact_form.changed.connect(self._artifact_changed)
        self.artifact_form.value_normalized.connect(self._artifact_changed)
        self.artifact_form.refresh_requested.connect(
            lambda _key: self.refresh_requested.emit()
        )

        self.form = FluentParameterForm(empty, {})
        self._body_layout.addWidget(self.form)
        self.form.changed.connect(self._parameter_changed)
        self.form.value_normalized.connect(
            lambda key: self._parameter_changed(key, normalized=True)
        )
        self.form.refresh_requested.connect(
            lambda _key: self.refresh_requested.emit()
        )

        self._artifact_result_frame = QtWidgets.QWidget()
        self._artifact_result_frame.setStyleSheet("background: transparent;")
        self._artifact_result_layout = QtWidgets.QVBoxLayout(
            self._artifact_result_frame
        )
        self._artifact_result_layout.setContentsMargins(0, 0, 0, 0)
        self._artifact_result_layout.setSpacing(scaled_px(6, minimum=4))
        self._artifact_result_rows: dict[str, FluentSettingRow] = {}
        self._body_layout.addWidget(self._artifact_result_frame)

        # The lifecycle in one button, as on the node's row: Start while it is
        # stopped, Stop while it runs.  Restart shows only while it runs, and
        # Remove keeps to the far side, away from the button pressed most.
        actions = QtWidgets.QHBoxLayout()
        self.remove_button = FluentButton("Remove", role="destructive")
        self.remove_button.clicked.connect(self.remove_requested.emit)
        actions.addWidget(self.remove_button)
        actions.addStretch(1)
        self.restart_button = FluentButton("Restart")
        self.restart_button.clicked.connect(self.start_requested.emit)
        self.run_button = FluentButton("Stop", role="primary")
        stop_width = self.run_button.sizeHint().width()
        self.run_button.setText("Starting…")
        starting_width = self.run_button.sizeHint().width()
        self.run_button.setText("Start")
        self.run_button.setMinimumWidth(max(stop_width, starting_width, self.run_button.sizeHint().width()))
        self.run_button.clicked.connect(self._run_clicked)
        # This is the Start an operator actually presses -- adding a node
        # opens its Edit and focuses it -- so the switch that says what Start
        # will do belongs here too.  Both views show the SAME preference: it
        # lives on the node's binding, and neither widget keeps a default.
        self.preview_switch = FluentButton("Plot")
        self.preview_switch.setCheckable(True)
        self.preview_switch.setToolTip(
            "Open this node's plot panel when it starts"
        )
        self.preview_switch.toggled.connect(self.auto_preview_changed.emit)
        actions.addWidget(self.restart_button)
        actions.addWidget(self.run_button)
        actions.addWidget(self.preview_switch)
        self._body_layout.addLayout(actions)
        self._show_actions = bool(show_actions)
        for button in (self.remove_button, self.restart_button, self.run_button, self.preview_switch):
            button.setVisible(bool(show_actions))
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
        self.title_label.setToolTip(str(incoming.get("description") or ""))
        kind = str(incoming.get("kind") or "logic")
        self.kind_label.setText(f"({kind})")
        managed_fields = self._reconcile_contributions(incoming)
        acquisition_key = str(incoming.get("acquisition_input") or "")
        visible_spec = FormSpec(
            tuple(field for field in spec.fields
                  if field.key not in managed_fields and field.key != acquisition_key)
        )
        # THE PROJECTION ARRIVES ON EVERY BEAT, and almost always says
        # what it said last time: a status word changed, or nothing did.
        # These four rebuild three forms, refill a grouped tree over the
        # whole signal list, re-parse the scan plan and re-measure a
        # six-column table -- ten times a second, for an editor the
        # operator is reading.  They read nothing but the projection, so
        # an unchanged projection makes them all no-ops, and the fields
        # below are excluded from the comparison precisely because the
        # cheap half further down is what applies them.  ``error`` is not
        # among them: it changes only when a run fails or recovers, and a
        # contribution shows a failure beside the row that caused it.
        volatile = ("running", "pending", "issues", "status",
                    "auto_preview", "can_start", "can_stop")
        structure = {
            key: value for key, value in incoming.items() if key not in volatile
        }
        if structure != self._applied_structure:
            self._applied_structure = structure
            label_width = max(
                _form_label_width((*spec.fields, *artifact_spec.fields)),
                setting_label_width(_row_labels(incoming)),
            )
            self._label_width = label_width
            self.artifact_form.set_label_width(label_width)
            self.form.set_label_width(label_width)
            for widget in self._contributions.values():
                set_width = getattr(widget, "set_label_width", None)
                if callable(set_width):
                    set_width(label_width)
                widget.update_projection(incoming)
            self.artifact_form.reconcile(artifact_spec, dict(artifact_values))
            self.artifact_form.setVisible(bool(artifact_spec.keys))
            self.form.reconcile(
                visible_spec,
                {key: values[key] for key in visible_spec.keys},
            )
            self.form.setVisible(bool(visible_spec.keys))
            self._reconcile_selectors(incoming)
            self._rebuild_artifact_results(incoming.get("artifact_results", ()))

        offered = bool(incoming["preview_offered"])
        self.preview_switch.setVisible(offered and bool(self._show_actions))
        blocked = self.preview_switch.blockSignals(True)
        try:
            self.preview_switch.setChecked(bool(incoming["auto_preview"]))
        finally:
            self.preview_switch.blockSignals(blocked)

        running = bool(incoming.get("running"))
        pending = bool(incoming.get("pending"))
        error = str(incoming.get("error") or "")
        issues = tuple(str(issue) for issue in incoming.get("issues", ()) or ())
        self._project_commands()
        state = "waiting for device release" if pending else ("running" if running else "stopped")
        status = str(incoming.get("status") or state)
        issue = issues[0] if issues else ""
        self.status_label.setText(error or (f"Draft: {issue}" if issue else status))
        # Written when they CHANGE: the sheet is re-parsed and the widget
        # repolished on every write.
        colours = (
            (RED, RED_DARK) if error
            else (ORANGE, ORANGE_DARK) if issue
            else (GREEN, TEXT) if running
            else (GREY, GREY)
        )
        if colours != self._status_colours:
            self._status_colours = colours
            self.status_dot.set_color(colours[0])
            self.status_label.setStyleSheet(
                f"color: {colours[1]}; background: transparent; border: none;"
            )

    def set_mutation_enabled(self, enabled: bool) -> None:
        """Disable authored and lifecycle commands while a Task owns the console."""

        self._mutation_enabled = bool(enabled)
        self._selector_frame.setEnabled(self._mutation_enabled)
        self.artifact_form.setEnabled(self._mutation_enabled)
        self.form.setEnabled(self._mutation_enabled)
        for widget in self._contributions.values():
            setter = getattr(widget, "set_mutation_enabled", None)
            if callable(setter):
                setter(self._mutation_enabled)
        self._project_commands()

    def _reconcile_contributions(
        self,
        projection: Mapping[str, object],
    ) -> frozenset[str]:
        """Create, retire and enable this projection's contribution widgets.

        LIFECYCLE ONLY.  Which fields a contribution manages decides what the
        form below may show, so this has to run before the structure guard --
        but handing each widget the projection does not, and doing it here put
        the contribution's own work (a scan plan re-parsed, its template ports
        walked, a tunable-device range read dispatched) outside the guard that
        exists to stop exactly that on a projection that did not change.
        Projecting happens with the rest of the structure, below.
        """

        factories = tuple(projection.get("ui_contributions", ()) or ())
        for factory in factories:
            if factory in self._contributions:
                continue
            if not callable(factory):
                raise TypeError("logic UI contributions must be widget factories")
            widget = factory(self)
            if not isinstance(widget, QtWidgets.QWidget):
                raise TypeError("logic UI contribution factory must return QWidget")
            changed = getattr(widget, "draft_changed", None)
            updater = getattr(widget, "update_projection", None)
            if changed is None or not callable(getattr(changed, "connect", None)):
                raise TypeError("logic UI contribution must emit draft_changed")
            if not callable(updater):
                raise TypeError("logic UI contribution must update_projection")
            changed.connect(self.draft_changed.emit)
            self._contributions[factory] = widget
            self._body_layout.insertWidget(
                self._body_layout.indexOf(self._artifact_result_frame),
                widget,
            )
        for factory in tuple(self._contributions):
            if factory in factories:
                continue
            widget = self._contributions.pop(factory)
            retire_widget(widget)
        managed: set[str] = set()
        for factory in factories:
            widget = self._contributions[factory]
            fields = tuple(getattr(widget, "managed_fields", ()) or ())
            if any(not isinstance(name, str) or not name for name in fields):
                raise TypeError("logic UI contribution managed_fields must be names")
            managed.update(fields)
            setter = getattr(widget, "set_mutation_enabled", None)
            if callable(setter):
                setter(self._mutation_enabled)
        return frozenset(managed)

    def _run_clicked(self) -> None:
        if bool(self._projection.get("running")):
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def _project_commands(self) -> None:
        pending = bool(self._projection.get("pending"))
        running = bool(self._projection.get("running"))
        can_start = bool(self._projection.get("can_start"))
        can_stop = bool(self._projection.get("can_stop"))
        self.run_button.setText("Starting…" if pending else "Stop" if running else "Start")
        self.run_button.set_role("standard" if running else "primary")
        self.run_button.setEnabled(
            self._mutation_enabled and not pending and (can_stop if running else can_start)
        )
        self.restart_button.setVisible(self._show_actions and running)
        self.restart_button.setEnabled(self._mutation_enabled and can_start and not pending)
        # It says what the next Start will do, so it follows Start's own gate.
        self.preview_switch.setEnabled(self._mutation_enabled)
        self.remove_button.setEnabled(self._mutation_enabled)

    def _add_selector_row(self, label: str, combo: FluentComboBox, *, first: bool = False) -> None:
        row = FluentSettingRow(label, combo, label_width=self._label_width, parent=self._selector_frame)
        self._selector_rows[combo] = (row, str(label))
        if first:
            self._selector_layout.insertWidget(0, row)
        else:
            self._selector_layout.addWidget(row)

    def _reconcile_selectors(self, projection: Mapping[str, object]) -> None:
        for row, label in self._selector_rows.values():
            row.set_label(label, width=self._label_width)
        device_options = projection.get("device_options") or {}
        device_keys = projection.get("device_keys") or {}
        if not isinstance(device_options, Mapping) or not isinstance(device_keys, Mapping):
            raise TypeError("logic editor device selectors must be mappings")

        has_source = bool(projection.get("source_required"))
        if has_source:
            current = str(projection.get("source_signal") or "")
            options = tuple(str(item) for item in projection.get("source_options", ()))
            source_labels = projection.get("source_labels") or {}
            source_groups = projection.get("source_groups") or {}
            if not isinstance(source_labels, Mapping):
                raise TypeError("logic editor source labels must be a mapping")
            if not isinstance(source_groups, Mapping):
                raise TypeError("logic editor source groups must be a mapping")
            bundle = bool(projection.get("source_bundle"))
            combo_type = FluentComboBox if bundle else FluentTreeComboBox
            if self.source_combo is not None and type(self.source_combo) is not combo_type:
                self._retire_selector(self.source_combo)
                self.source_combo = None
            if self.source_combo is None:
                self.source_combo = combo_type()
                # ``activated`` covers both pick paths -- the combo's native
                # activation and the tree's own leaf-click re-emission --
                # exactly as the shared keyed-choice form handler wires it.
                self.source_combo.activated[int].connect(self._source_changed)
                self._add_selector_row(
                    str(projection.get("source_label") or "Signal"), self.source_combo, first=True
                )
            # The one grouped signal chooser every other picker already uses:
            # a producer tree with keyed leaves, not a second flat list.
            if bundle:
                self._fill_combo(
                    self.source_combo, current, options, blank=True,
                    labels=source_labels,
                )
            else:
                fill_grouped_choice_combo(
                    self.source_combo,
                    names=options,
                    sources={
                        str(key): (str(value),)
                        for key, value in source_groups.items()
                    },
                    metadata={},
                    labels={str(key): str(value) for key, value in source_labels.items()},
                    current=current,
                    none_label="(not selected)",
                    empty_source_label="signals",
                )
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
                self._add_selector_row(_device_label(name), combo)
            current = str(device_keys.get(name, ""))
            self._fill_combo(
                combo, current, tuple(str(item) for item in offered), blank=True,
                labels=projection.get("device_labels") or {},
            )

        retired = set(self._device_combos).difference(
            str(name) for name in device_options
        )
        for name in retired:
            self._retire_selector(self._device_combos.pop(name))

        acquisition_key = str(projection.get("acquisition_input") or "")
        if acquisition_key:
            field = next(field for field in projection["form_spec"].fields
                         if field.key == acquisition_key)
            combo = self._acquisition_combo
            if combo is None:
                combo = FluentComboBox()
                combo.activated[int].connect(
                    lambda _index, control=combo: self.draft_changed.emit({
                        "values": {str(self._projection["acquisition_input"]):
                                   str(control.currentData() or "")}
                    })
                )
                self._acquisition_combo = combo
                self._add_selector_row(field.label, combo)
            combo.setToolTip(field.description)
            combo.setEnabled(bool(projection.get("acquisition_enabled", True)) and not field.unavailable)
            self._fill_combo(
                combo, str(projection["form_values"][acquisition_key] or ""),
                tuple(str(choice.value) for choice in field.choices), blank=False,
                labels={str(choice.value): choice.label for choice in field.choices},
            )
        elif self._acquisition_combo is not None:
            self._retire_selector(self._acquisition_combo)
            self._acquisition_combo = None

        self._selector_frame.setVisible(
            has_source or bool(self._device_combos) or bool(acquisition_key)
        )

    def _retire_selector(self, combo: FluentComboBox) -> None:
        row, _label = self._selector_rows.pop(combo)
        self._selector_layout.removeWidget(row)
        retire_widget(row)

    @staticmethod
    def _fill_combo(
        combo: FluentComboBox,
        current: str,
        options: tuple[str, ...],
        *,
        blank: bool,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        ordered: list[str] = []
        if blank:
            ordered.append("")
        if current and current not in options:
            ordered.append(current)
        ordered.extend(item for item in options if item not in ordered)
        desired = tuple(
            (str((labels or {}).get(value, value or "(not selected)")), value)
            for value in ordered
        )
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

    def _parameter_changed(self, key: str, *, normalized: bool = False) -> None:
        try:
            value = self.form.read_value(str(key))
        except (TypeError, ValueError) as error:
            self.status_label.setText(str(error))
            return
        patch = {"values": {str(key): value}}
        if normalized:
            patch["normalized"] = True
        self.draft_changed.emit(patch)

    def _artifact_changed(self, key: str) -> None:
        try:
            value = self.artifact_form.read_value(str(key))
        except (TypeError, ValueError) as error:
            self.status_label.setText(str(error))
            return
        self.draft_changed.emit({"artifact_inputs": {str(key): value}})

    def _rebuild_artifact_results(self, rows: object) -> None:
        """Project the result rows, keyed by name.

        A readout whose name is still listed is kept and re-told its path;
        the rows are rebuilt only when the SET of names, or their order,
        changed.  Torn down and rebuilt on every projection, the same
        results destroyed every readout on every beat for nothing.
        """

        results: list[tuple[str, str, str]] = []
        for raw in tuple(rows or ()):  # type: ignore[arg-type]
            if not isinstance(raw, Mapping):
                raise TypeError("logic artifact results must be mappings")
            name = str(raw.get("name") or "")
            path = str(raw.get("path") or "")
            if not name or not path:
                raise ValueError("logic artifact results need name and path")
            results.append((name, path, str(raw.get("contract_id") or "")))
        if [name for name, _path, _contract in results] != list(
            self._artifact_result_readouts
        ):
            for row in self._artifact_result_rows.values():
                self._artifact_result_layout.removeWidget(row)
                retire_widget(row)
            self._artifact_result_rows.clear()
            self._artifact_result_readouts.clear()
            for name, path, contract in results:
                readout = FluentReadoutEdit(path)
                self._artifact_result_readouts[name] = readout
                row = FluentSettingRow(
                    _device_label(name), readout, label_width=self._label_width,
                    parent=self._artifact_result_frame,
                )
                self._artifact_result_rows[name] = row
                self._artifact_result_layout.addWidget(row)
        for name, row in self._artifact_result_rows.items():
            row.set_label(_device_label(name), width=self._label_width)
        for name, path, contract in results:
            readout = self._artifact_result_readouts[name]
            readout.setText(path)
            readout.setToolTip(contract)
        self._artifact_result_frame.setVisible(
            bool(self._artifact_result_readouts)
        )

    def _source_changed(self, _index: int) -> None:
        if self.source_combo is None:
            return
        value = (
            self.source_combo.current_choice_key()
            if isinstance(self.source_combo, FluentTreeComboBox)
            else self.source_combo.currentData()
        )
        self.draft_changed.emit(
            {"source_signal": str(value or "")}
        )

    def _device_changed(self, argument_name: str, combo: FluentComboBox) -> None:
        self.draft_changed.emit(
            {"device_keys": {str(argument_name): str(combo.currentData() or "")}}
        )


def _device_label(name: str) -> str:
    return str(name).replace("_", " ").title()


def _row_labels(projection: Mapping[str, object]) -> tuple[str, ...]:
    """Every label the editor shows outside its two forms."""

    labels = [_device_label(name) for name in (projection.get("device_options") or {})]
    if projection.get("source_required"):
        labels.append(str(projection.get("source_label") or "Signal"))
    labels.extend(
        _device_label(dict(raw).get("name") or "")
        for raw in tuple(projection.get("artifact_results", ()) or ())
    )
    return tuple(labels)


__all__ = ["LogicEditorView"]
