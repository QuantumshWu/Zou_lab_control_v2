"""Pure view for editing a plain device-instance presentation."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    ElidedLabel,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentGroupBox,
    FluentLineEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentStatusDot,
    FluentTabWidget,
    muted_note_label,
    setting_label_width,
    window_pad,
)
from zlc_ui.form.form import FormSpec
from zlc_ui.form.qt_form import FluentParameterForm
from zlc_ui.console.status_strip import StatusStrip


class _DeviceCard(FluentFrame):
    type_picked = QtCore.pyqtSignal(str, str)
    role_committed = QtCore.pyqtSignal(str, str)
    remove_requested = QtCore.pyqtSignal(str)
    parameter_committed = QtCore.pyqtSignal(str, str)

    def __init__(self, instance_id: str, parent=None) -> None:
        super().__init__(parent)
        self.instance_id = str(instance_id)
        self.form: FluentParameterForm | None = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(window_pad(0.6), window_pad(0.6), window_pad(0.6), window_pad(0.6))
        outer.setSpacing(window_pad(0.35))
        header = QtWidgets.QHBoxLayout()
        self.name_label = ElidedLabel(self.instance_id)
        self.role_edit = FluentLineEdit()
        self.role_edit.setPlaceholderText("role")
        self.type_combo = FluentComboBox()
        self.remove_button = FluentButton("Remove", color=ORANGE)
        header.addWidget(self.name_label)
        header.addWidget(self.role_edit, 1)
        header.addWidget(self.type_combo, 1)
        header.addWidget(self.remove_button)
        outer.addLayout(header)
        self.form_host = QtWidgets.QVBoxLayout()
        outer.addLayout(self.form_host)
        self.role_edit.editingFinished.connect(
            lambda: self.role_committed.emit(self.instance_id, self.role_edit.text())
        )
        self.type_combo.currentIndexChanged[int].connect(self._type_changed)
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self.instance_id))

    def set_record(self, role: str, type_key: str) -> None:
        self.role_edit.setText(str(role))
        index = self.type_combo.findData(str(type_key))
        if index >= 0:
            self.type_combo.setCurrentIndex(index)
        self.name_label.setToolTip(self.instance_id)

    def set_choices(self, choices: tuple[tuple[str, str], ...]) -> None:
        current = self.type_combo.currentData()
        self.type_combo.blockSignals(True)
        try:
            self.type_combo.clear()
            for label, key in choices:
                self.type_combo.addItem(str(label), str(key))
            index = self.type_combo.findData(current)
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
        finally:
            self.type_combo.blockSignals(False)

    def set_form(self, spec: FormSpec, values: tuple[tuple[str, object], ...]) -> None:
        if self.form is not None:
            self.form_host.removeWidget(self.form)
            self.form.setParent(None)
            self.form.deleteLater()
        self.form = FluentParameterForm(spec, dict(values), parent=self)
        self.form.changed.connect(lambda key: self.parameter_committed.emit(self.instance_id, key))
        self.form_host.addWidget(self.form)

    def _type_changed(self, index: int) -> None:
        value = self.type_combo.itemData(index)
        if isinstance(value, str):
            self.type_picked.emit(self.instance_id, value)


class DeviceManagerView(QtWidgets.QWidget):
    """The v1-shaped Config surface for one plain-data apparatus draft."""

    device_add_requested = QtCore.pyqtSignal(str)
    save_requested = QtCore.pyqtSignal()
    template_selected = QtCore.pyqtSignal(str)
    load_requested = QtCore.pyqtSignal()
    save_as_requested = QtCore.pyqtSignal()
    cancel_requested = QtCore.pyqtSignal()
    lifecycle_requested = QtCore.pyqtSignal()
    device_remove_requested = QtCore.pyqtSignal(str)
    role_committed = QtCore.pyqtSignal(str, str)
    type_picked = QtCore.pyqtSignal(str, str)
    parameter_committed = QtCore.pyqtSignal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._choices: tuple[tuple[str, str], ...] = ()
        self._cards: dict[str, _DeviceCard] = {}
        self._loaded_widgets: list[QtWidgets.QWidget] = []
        self._dirty = False
        self._busy = False
        self._lifecycle_enabled = False
        outer = QtWidgets.QVBoxLayout(self)
        pad = window_pad()
        outer.setContentsMargins(pad, pad, pad, pad)
        outer.setSpacing(window_pad(0.5))

        self.tabs = FluentTabWidget(self)
        self.config_page = QtWidgets.QWidget(self.tabs)
        self.tabs.add_permanent_tab(self.config_page, "Config")
        outer.addWidget(self.tabs, 1)

        page = QtWidgets.QVBoxLayout(self.config_page)
        page.setContentsMargins(pad, pad, pad, pad)
        page.setSpacing(window_pad(0.5))

        header = QtWidgets.QHBoxLayout()
        self.heading_label = FluentSectionLabel("Devices")
        header.addWidget(self.heading_label)
        header.addStretch(1)
        # Match the v1 chrome: the left-hand title never changes; runtime state
        # and the edited document are a compact pair on the right.
        self.status_dot = FluentStatusDot(size=12)
        self.status_dot.set_color(GREY)
        self.document_name = ElidedLabel("untitled")
        self.document_name.setMinimumWidth(window_pad(9.5))
        header.addWidget(self.status_dot)
        header.addWidget(self.document_name)
        page.addLayout(header)

        columns = QtWidgets.QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(window_pad(0.65))

        self.left_scroll = FluentScrollArea()
        self.left_body = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(self.left_body)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(window_pad(0.45))
        self.installation_group = FluentGroupBox("Installation", self.left_body)
        installation = QtWidgets.QVBoxLayout(self.installation_group)
        installation.setContentsMargins(pad, pad, pad, pad)
        installation.setSpacing(window_pad(0.35))
        self.backend_combo = FluentComboBox()
        self.backend_combo.addItem("Custom", None)
        self.backend_row = FluentSettingRow(
            "Backend",
            self.backend_combo,
            label_width=setting_label_width(("Backend",)),
            parent=self.installation_group,
        )
        self.installation_detail = muted_note_label(
            "An apparatus is a set of independently named devices."
        )
        self.installation_detail.setWordWrap(True)
        installation.addWidget(self.backend_row)
        installation.addWidget(self.installation_detail)
        left.addWidget(self.installation_group)

        self.configured_group = FluentGroupBox("Configured devices", self.left_body)
        self.cards_layout = QtWidgets.QVBoxLayout(self.configured_group)
        self.cards_layout.setContentsMargins(pad, pad, pad, pad)
        self.cards_layout.setSpacing(window_pad(0.4))
        self.cards_host = QtWidgets.QVBoxLayout()
        self.cards_host.setContentsMargins(0, 0, 0, 0)
        self.cards_host.setSpacing(window_pad(0.4))
        self.cards_layout.addLayout(self.cards_host)

        add_control = QtWidgets.QWidget(self.configured_group)
        add_control.setMinimumWidth(0)
        add_control_layout = QtWidgets.QHBoxLayout(add_control)
        add_control_layout.setContentsMargins(0, 0, 0, 0)
        add_control_layout.setSpacing(window_pad(0.3))
        self.add_type_combo = FluentComboBox(add_control)
        self.add_button = FluentButton("Add", color=ACCENT)
        add_control_layout.addWidget(self.add_type_combo, 1)
        add_control_layout.addWidget(self.add_button)
        self.add_device_row = FluentSettingRow(
            "Add device",
            add_control,
            label_width=setting_label_width(("Add device",)),
            parent=self.configured_group,
        )
        self.cards_layout.addWidget(self.add_device_row)
        self.unavailable_note = muted_note_label("")
        self.unavailable_note.setWordWrap(True)
        self.cards_layout.addWidget(self.unavailable_note)
        self.cards_layout.addStretch(1)
        left.addWidget(self.configured_group)
        left.addStretch(1)
        self.left_scroll.set_width_bounded_widget(self.left_body)
        columns.addWidget(self.left_scroll, 3)

        self.right_scroll = FluentScrollArea()
        self.right_body = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(self.right_body)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(window_pad(0.45))

        self.available_group = FluentGroupBox("Available", self.right_body)
        self.available_layout = QtWidgets.QVBoxLayout(self.available_group)
        self.available_layout.setContentsMargins(pad, pad, pad, pad)
        self.available_layout.setSpacing(window_pad(0.4))
        self.template_buttons: dict[str, FluentButton] = {}
        self.available_empty = muted_note_label("No installation templates available")
        self.available_empty.setWordWrap(True)
        self.available_layout.addWidget(self.available_empty)
        right.addWidget(self.available_group)

        self.loaded_group = FluentGroupBox("Loaded (session)", self.right_body)
        self.loaded_layout = QtWidgets.QVBoxLayout(self.loaded_group)
        self.loaded_layout.setContentsMargins(pad, pad, pad, pad)
        self.loaded_layout.setSpacing(window_pad(0.35))
        self.loaded_empty = muted_note_label("No active installation")
        self.loaded_empty.setWordWrap(True)
        self.loaded_layout.addWidget(self.loaded_empty)
        right.addWidget(self.loaded_group)
        self.runtime_note = muted_note_label(
            "Live device controls belong to DeviceViewer; this window edits "
            "installation configuration."
        )
        self.runtime_note.setWordWrap(True)
        right.addWidget(self.runtime_note)
        right.addStretch(1)
        self.right_scroll.set_width_bounded_widget(self.right_body)
        columns.addWidget(self.right_scroll, 2)
        page.addLayout(columns, 1)

        actions = QtWidgets.QHBoxLayout()
        self.new_combo = FluentComboBox()
        self.new_combo.addItem("New…", None)
        self.load_button = FluentButton("Load…", color=ORANGE)
        self.save_button = FluentButton("Save", color=ACCENT)
        self.save_as_button = FluentButton("Save as…", color=ACCENT)
        self.cancel_button = FluentButton("Cancel", color=GREY)
        self.lifecycle_button = FluentButton("Init devices", color=GREEN)
        actions.addWidget(self.new_combo)
        actions.addWidget(self.load_button)
        actions.addWidget(self.save_button)
        actions.addWidget(self.save_as_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch(1)
        actions.addWidget(self.lifecycle_button)
        page.addLayout(actions)

        self.status_strip = StatusStrip()
        page.addWidget(self.status_strip)
        self.add_button.clicked.connect(self._add_clicked)
        self.save_button.clicked.connect(self.save_requested.emit)
        self.new_combo.activated[int].connect(self._template_chosen)
        self.backend_combo.activated[int].connect(self._backend_chosen)
        self.load_button.clicked.connect(self.load_requested.emit)
        self.save_as_button.clicked.connect(self.save_as_requested.emit)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.lifecycle_button.clicked.connect(self.lifecycle_requested.emit)
        self._refresh_controls()

    def set_apparatus(self, name: str, dirty: bool, saved: bool) -> None:
        """Project the document name and its ordinary unsaved ``*`` marker.

        The adjacent dot deliberately remains a session-state indicator, as
        it was in v1; :meth:`set_lifecycle` is its sole writer.
        """

        self._dirty = bool(dirty)
        shown = f"{name}{'*' if dirty else ''}"
        self.document_name.setText(shown)
        self.document_name.setToolTip(
            f"{name} ({'saved' if saved and not dirty else 'unsaved draft'})"
        )
        self._refresh_controls()

    def set_device_choices(
        self,
        choices: tuple[tuple[str, str], ...],
        unavailable: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """What may be added, and what this machine cannot offer today.

        A lab machine is normally missing at least one vendor runtime, and a
        family that cannot import used to be simply absent -- indistinguishable
        from one that does not exist.  Listed and greyed WITH the reason, the
        window answers the question an operator actually has: not "where is my
        camera type" but "why".
        """

        self._choices = tuple((str(label), str(key)) for label, key in choices)
        self.add_type_combo.clear()
        for label, key in self._choices:
            self.add_type_combo.addItem(label, key)
        for family, reason in unavailable:
            self.add_type_combo.addItem(f"{family} — {reason}", None)
            item = self.add_type_combo.model().item(self.add_type_combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
        self.unavailable_note.setText(
            "\n".join(f"{family}: {reason}" for family, reason in unavailable)
        )
        self.unavailable_note.setVisible(bool(unavailable))
        for card in self._cards.values():
            card.set_choices(self._choices)

    def set_devices(self, devices: tuple[tuple[str, str, str], ...]) -> None:
        wanted = tuple((str(instance_id), str(role), str(type_key)) for instance_id, role, type_key in devices)
        wanted_ids = {instance_id for instance_id, _role, _type_key in wanted}
        for instance_id in tuple(self._cards):
            if instance_id not in wanted_ids:
                card = self._cards.pop(instance_id)
                self.cards_host.removeWidget(card)
                card.setParent(None)
                card.deleteLater()
        for instance_id, role, type_key in wanted:
            card = self._cards.get(instance_id)
            if card is None:
                card = _DeviceCard(instance_id, self.configured_group)
                card.type_picked.connect(self.type_picked.emit)
                card.role_committed.connect(self.role_committed.emit)
                card.remove_requested.connect(self._remove_requested)
                card.parameter_committed.connect(self.parameter_committed.emit)
                self._cards[instance_id] = card
                self.cards_host.addWidget(card)
                card.set_choices(self._choices)
            card.set_record(role, type_key)
        self._refresh_controls()

    def set_templates(self, templates: tuple[tuple[str, str], ...]) -> None:
        current = self.new_combo.currentData()
        backend = self.backend_combo.currentData()
        self.new_combo.blockSignals(True)
        self.backend_combo.blockSignals(True)
        try:
            self.new_combo.clear()
            self.new_combo.addItem("New…", None)
            self.backend_combo.clear()
            self.backend_combo.addItem("Custom", None)
            for label, key in templates:
                self.new_combo.addItem(str(label), str(key))
                self.backend_combo.addItem(str(label), str(key))
            index = self.new_combo.findData(current)
            self.new_combo.setCurrentIndex(max(0, index))
            backend_index = self.backend_combo.findData(backend)
            self.backend_combo.setCurrentIndex(max(0, backend_index))
        finally:
            self.new_combo.blockSignals(False)
            self.backend_combo.blockSignals(False)

        for button in self.template_buttons.values():
            self.available_layout.removeWidget(button)
            button.setParent(None)
            button.deleteLater()
        self.template_buttons.clear()
        for index, (label, key) in enumerate(templates):
            name = str(key)
            button = FluentButton(str(label), color=ACCENT)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed,
            )
            button.clicked.connect(
                lambda _checked=False, template=name: self.template_selected.emit(template)
            )
            self.available_layout.insertWidget(index, button)
            self.template_buttons[name] = button
        self.available_empty.setVisible(not self.template_buttons)
        self._refresh_controls()

    def set_installation_source(self, source: str | None, detail: str) -> None:
        index = self.backend_combo.findData(source)
        self.backend_combo.blockSignals(True)
        try:
            self.backend_combo.setCurrentIndex(max(0, index))
        finally:
            self.backend_combo.blockSignals(False)
        self.installation_detail.setText(str(detail))

    def set_loaded_devices(
        self,
        devices: tuple[tuple[str, str, str], ...],
    ) -> None:
        for widget in self._loaded_widgets:
            self.loaded_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self._loaded_widgets.clear()
        for instance_id, role, type_id in devices:
            card = FluentFrame(self.loaded_group, bordered=True)
            row = QtWidgets.QHBoxLayout(card)
            row.setContentsMargins(window_pad(0.45), window_pad(0.35), window_pad(0.45), window_pad(0.35))
            role_label = ElidedLabel(str(role))
            detail = muted_note_label(f"{type_id} · {instance_id}")
            row.addWidget(role_label)
            row.addWidget(detail, 1)
            self.loaded_layout.insertWidget(len(self._loaded_widgets), card)
            self._loaded_widgets.append(card)
        self.loaded_empty.setVisible(not devices)

    def set_lifecycle(
        self,
        text: str,
        *,
        enabled: bool,
        active: bool,
        busy: bool = False,
    ) -> None:
        self._busy = bool(busy)
        self._lifecycle_enabled = bool(enabled)
        self.lifecycle_button.setText(str(text))
        restart = active and "restart" in str(text).lower()
        self.status_dot.set_color(ORANGE if restart else GREEN if active else GREY)
        self.status_dot.setToolTip(
            "Configuration differs from the active installation"
            if restart
            else "Active installation"
            if active
            else "No active installation"
        )
        self._refresh_controls()

    def set_form_spec(
        self,
        instance_id: str,
        spec: FormSpec,
        values: tuple[tuple[str, object], ...],
    ) -> None:
        card = self._cards.get(str(instance_id))
        if card is None:
            raise KeyError(f"unknown device instance id: {instance_id!r}")
        card.set_form(spec, values)

    def read_values(self, instance_id: str) -> tuple[tuple[str, object], ...]:
        card = self._cards.get(str(instance_id))
        if card is None or card.form is None:
            return ()
        return tuple(card.form.read_all().items())

    def show_status(self, text: str, severity: str) -> None:
        self.status_strip.show_status(text, severity)

    def _add_clicked(self) -> None:
        value = self.add_type_combo.currentData()
        self.device_add_requested.emit(str(value or ""))

    def _template_chosen(self, index: int) -> None:
        value = self.new_combo.itemData(index)
        self.new_combo.blockSignals(True)
        try:
            self.new_combo.setCurrentIndex(0)
        finally:
            self.new_combo.blockSignals(False)
        if isinstance(value, str):
            self.template_selected.emit(value)

    def _backend_chosen(self, index: int) -> None:
        value = self.backend_combo.itemData(index)
        if isinstance(value, str):
            self.template_selected.emit(value)

    def _remove_requested(self, instance_id: str) -> None:
        self.device_remove_requested.emit(str(instance_id))

    def _refresh_controls(self) -> None:
        enabled = not self._busy
        self.new_combo.setEnabled(enabled)
        self.backend_combo.setEnabled(enabled)
        self.load_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.save_as_button.setEnabled(enabled)
        self.cancel_button.setEnabled(enabled and self._dirty)
        self.lifecycle_button.setEnabled(enabled and self._lifecycle_enabled)
        self.add_type_combo.setEnabled(enabled)
        self.add_button.setEnabled(enabled and bool(self._choices))
        for button in self.template_buttons.values():
            button.setEnabled(enabled)
        for card in self._cards.values():
            card.setEnabled(enabled)


__all__ = ["DeviceManagerView"]
