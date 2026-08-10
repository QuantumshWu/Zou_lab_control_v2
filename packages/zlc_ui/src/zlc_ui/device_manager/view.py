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
    FluentStatusDot,
    FluentTabWidget,
    muted_note_label,
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
        self._collapsed = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(window_pad(0.6), window_pad(0.6), window_pad(0.6), window_pad(0.6))
        outer.setSpacing(window_pad(0.35))
        header = QtWidgets.QHBoxLayout()
        self.collapse_button = FluentButton("−", color=GREY)
        self.collapse_button.setFixedWidth(window_pad(2.0))
        self.name_label = ElidedLabel(self.instance_id)
        self.role_edit = FluentLineEdit()
        self.role_edit.setPlaceholderText("role")
        self.type_combo = FluentComboBox()
        self.remove_button = FluentButton("Remove", color=ORANGE)
        header.addWidget(self.collapse_button)
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
        self.collapse_button.clicked.connect(self._toggle_collapsed)
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
        projected = dict(values)
        if self.form is None:
            self.form = FluentParameterForm(spec, projected, parent=self)
            self.form.changed.connect(
                lambda key: self.parameter_committed.emit(self.instance_id, key)
            )
            self.form.setVisible(not self._collapsed)
            self.form_host.addWidget(self.form)
            return
        # A card is the stable owner of one form.  Reconcile already performs
        # the keyed schema/value diff, preserving compatible controls, focus,
        # and the sole changed-signal connection.  Rebuilding here detached a
        # visible QWidget into a transient top-level window before deleteLater.
        self.form.reconcile(spec, projected)

    def _type_changed(self, index: int) -> None:
        value = self.type_combo.itemData(index)
        if isinstance(value, str):
            self.type_picked.emit(self.instance_id, value)

    def _toggle_collapsed(self) -> None:
        self._collapsed = not self._collapsed
        if self.form is not None:
            self.form.setVisible(not self._collapsed)
        self.collapse_button.setText("+" if self._collapsed else "−")


class DeviceManagerView(QtWidgets.QWidget):
    """The v1-shaped Config surface for one plain-data apparatus draft."""

    device_add_requested = QtCore.pyqtSignal(str)
    save_requested = QtCore.pyqtSignal()
    template_selected = QtCore.pyqtSignal(str)
    discovery_requested = QtCore.pyqtSignal()
    discovered_add_requested = QtCore.pyqtSignal(str)
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
        self._choices: tuple[tuple[str, str, str], ...] = ()
        self._choices_by_domain: dict[str, tuple[tuple[str, str], ...]] = {}
        self._cards: dict[str, _DeviceCard] = {}
        self._card_domains: dict[str, str] = {}
        self._discovered_widgets: dict[str, tuple[FluentFrame, ElidedLabel, QtWidgets.QLabel, FluentButton]] = {}
        self._configured_discoveries: set[str] = set()
        self._loaded_widgets: list[QtWidgets.QWidget] = []
        self._dirty = False
        self._busy = False
        self._lifecycle_enabled = False
        self._discovery_enabled = False
        self._discovery_disabled_reason = (
            "No installed device type declares discovery"
        )
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
        self.domain_column = QtWidgets.QVBoxLayout(self.left_body)
        self.domain_column.setContentsMargins(0, 0, 0, 0)
        self.domain_column.setSpacing(window_pad(0.45))
        self.domain_groups: dict[str, FluentGroupBox] = {}
        self.domain_card_layouts: dict[str, QtWidgets.QVBoxLayout] = {}
        self.domain_add_buttons: dict[str, FluentButton] = {}
        self.unavailable_note = muted_note_label("")
        self.unavailable_note.setWordWrap(True)
        self.unavailable_note.setVisible(False)
        self.domain_column.addWidget(self.unavailable_note)
        self.domain_column.addStretch(1)
        self.left_scroll.set_width_bounded_widget(self.left_body)
        columns.addWidget(self.left_scroll, 3)

        self.right_scroll = FluentScrollArea()
        self.right_body = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(self.right_body)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(window_pad(0.45))

        self.discovered_group = FluentGroupBox("Discovered hardware", self.right_body)
        self.discovered_layout = QtWidgets.QVBoxLayout(self.discovered_group)
        self.discovered_layout.setContentsMargins(pad, pad, pad, pad)
        self.discovered_layout.setSpacing(window_pad(0.4))
        self.discover_button = FluentButton("Scan hardware", color=ACCENT)
        self.discovered_layout.addWidget(self.discover_button)
        self.discovered_cards_layout = QtWidgets.QVBoxLayout()
        self.discovered_cards_layout.setSpacing(window_pad(0.4))
        self.discovered_layout.addLayout(self.discovered_cards_layout)
        self.discovered_empty = muted_note_label("No discovered hardware")
        self.discovered_empty.setWordWrap(True)
        self.discovered_layout.addWidget(self.discovered_empty)
        right.addWidget(self.discovered_group)

        self.loaded_group = FluentGroupBox("Loaded session", self.right_body)
        self.loaded_layout = QtWidgets.QVBoxLayout(self.loaded_group)
        self.loaded_layout.setContentsMargins(pad, pad, pad, pad)
        self.loaded_layout.setSpacing(window_pad(0.35))
        self.loaded_empty = muted_note_label("No active installation")
        self.loaded_empty.setWordWrap(True)
        self.loaded_layout.addWidget(self.loaded_empty)
        right.addWidget(self.loaded_group)
        self.runtime_note = muted_note_label(
            "Live controls belong to Device Viewer; this window authors the "
            "device graph."
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
        self.lifecycle_button = FluentButton("Init devices", color=GREEN)
        actions.addWidget(self.new_combo)
        actions.addWidget(self.load_button)
        actions.addWidget(self.save_button)
        actions.addWidget(self.save_as_button)
        actions.addStretch(1)
        actions.addWidget(self.lifecycle_button)
        page.addLayout(actions)

        self.status_strip = StatusStrip()
        page.addWidget(self.status_strip)
        self.discover_button.clicked.connect(self.discovery_requested.emit)
        self.save_button.clicked.connect(self.save_requested.emit)
        self.new_combo.activated[int].connect(self._template_chosen)
        self.load_button.clicked.connect(self.load_requested.emit)
        self.save_as_button.clicked.connect(self.save_as_requested.emit)
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
        choices: tuple[tuple[str, str, str], ...],
        unavailable: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """What may be added, and what this machine cannot offer today.

        A lab machine is normally missing at least one vendor runtime, and a
        family that cannot import used to be simply absent -- indistinguishable
        from one that does not exist.  Listed and greyed WITH the reason, the
        window answers the question an operator actually has: not "where is my
        camera type" but "why".
        """

        self._choices = tuple(
            (str(label), str(key), str(domain))
            for label, key, domain in choices
        )
        grouped: dict[str, list[tuple[str, str]]] = {}
        for label, key, domain in self._choices:
            grouped.setdefault(domain, []).append((label, key))
        self._choices_by_domain = {
            domain: tuple(values) for domain, values in grouped.items()
        }
        for domain in self._choices_by_domain:
            self._ensure_domain(domain)
        self.unavailable_note.setText(
            "\n".join(f"{family}: {reason}" for family, reason in unavailable)
        )
        self.unavailable_note.setVisible(bool(unavailable))
        for instance_id, card in self._cards.items():
            type_key = card.type_combo.currentData()
            if not isinstance(type_key, str):
                continue
            card.set_choices(
                self._choices_for_type(type_key, self._card_domains[instance_id])
            )
        self._refresh_controls()

    def set_devices(self, devices: tuple[tuple[str, str, str, str], ...]) -> None:
        wanted = tuple(
            (str(instance_id), str(role), str(type_key), str(domain))
            for instance_id, role, type_key, domain in devices
        )
        wanted_ids = {
            instance_id for instance_id, _role, _type_key, _domain in wanted
        }
        for instance_id in tuple(self._cards):
            if instance_id not in wanted_ids:
                card = self._cards.pop(instance_id)
                domain = self._card_domains.pop(instance_id)
                self.domain_card_layouts[domain].removeWidget(card)
                card.hide()
                card.deleteLater()
        for instance_id, role, type_key, domain in wanted:
            self._ensure_domain(domain)
            card = self._cards.get(instance_id)
            if card is None:
                card = _DeviceCard(instance_id, self.domain_groups[domain])
                card.type_picked.connect(self.type_picked.emit)
                card.role_committed.connect(self.role_committed.emit)
                card.remove_requested.connect(self._remove_requested)
                card.parameter_committed.connect(self.parameter_committed.emit)
                self._cards[instance_id] = card
                self._card_domains[instance_id] = domain
                self.domain_card_layouts[domain].addWidget(card)
            elif self._card_domains[instance_id] != domain:
                previous = self._card_domains[instance_id]
                self.domain_card_layouts[previous].removeWidget(card)
                card.setParent(self.domain_groups[domain])
                self.domain_card_layouts[domain].addWidget(card)
                self._card_domains[instance_id] = domain
            card.set_choices(self._choices_for_type(type_key, domain))
            card.set_record(role, type_key)
        self._refresh_controls()

    def set_templates(self, templates: tuple[tuple[str, str], ...]) -> None:
        current = self.new_combo.currentData()
        self.new_combo.blockSignals(True)
        try:
            self.new_combo.clear()
            self.new_combo.addItem("New…", None)
            for label, key in templates:
                self.new_combo.addItem(str(label), str(key))
            index = self.new_combo.findData(current)
            self.new_combo.setCurrentIndex(max(0, index))
        finally:
            self.new_combo.blockSignals(False)
        self._refresh_controls()

    def set_installation_source(self, source: str | None, detail: str) -> None:
        # Kept as a plain-data compatibility seam for the current presenter.
        # Installation/template provenance is not a visible Device Manager
        # section; templates are selected only through the ordinary New menu.
        self._installation_source = None if source is None else str(source)
        self._installation_detail = str(detail)

    def set_loaded_devices(
        self,
        devices: tuple[tuple[str, str, str], ...],
    ) -> None:
        for widget in self._loaded_widgets:
            self.loaded_layout.removeWidget(widget)
            widget.hide()
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

    def set_discovery_enabled(
        self,
        enabled: bool,
        reason: str = "No installed device type declares discovery",
    ) -> None:
        self._discovery_enabled = bool(enabled)
        self._discovery_disabled_reason = str(reason)
        self._refresh_controls()

    def set_discovered_devices(
        self,
        devices: tuple[tuple[str, str, str, bool], ...],
    ) -> None:
        wanted = {str(instance_id) for instance_id, _role, _type_id, _added in devices}
        for instance_id in tuple(self._discovered_widgets):
            if instance_id not in wanted:
                frame, _title, _detail, _button = self._discovered_widgets.pop(instance_id)
                self.discovered_cards_layout.removeWidget(frame)
                frame.hide()
                frame.deleteLater()
        self._configured_discoveries = {
            str(instance_id)
            for instance_id, _role, _type_id, added in devices
            if added
        }
        for index, (instance_id, role, type_id, _added) in enumerate(devices):
            instance_id = str(instance_id)
            widgets = self._discovered_widgets.get(instance_id)
            if widgets is None:
                frame = FluentFrame(self.discovered_group, bordered=True)
                row = QtWidgets.QHBoxLayout(frame)
                row.setContentsMargins(
                    window_pad(0.45), window_pad(0.35),
                    window_pad(0.45), window_pad(0.35),
                )
                title = ElidedLabel("")
                detail = muted_note_label("")
                button = FluentButton("Add", color=ACCENT)
                row.addWidget(title)
                row.addWidget(detail, 1)
                row.addWidget(button)
                button.clicked.connect(
                    lambda _checked=False, value=instance_id:
                    self.discovered_add_requested.emit(value)
                )
                widgets = (frame, title, detail, button)
                self._discovered_widgets[instance_id] = widgets
            frame, title, detail, _button = widgets
            title.setText(str(role))
            detail.setText(f"{type_id} · {instance_id}")
            self.discovered_cards_layout.insertWidget(index, frame)
        self.discovered_empty.setVisible(not devices)
        self._refresh_controls()

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

    def _ensure_domain(self, domain: str) -> None:
        if domain in self.domain_groups:
            return
        group = FluentGroupBox(domain.replace("_", " ").title(), self.left_body)
        group_layout = QtWidgets.QVBoxLayout(group)
        pad = window_pad(0.7)
        group_layout.setContentsMargins(pad, pad, pad, pad)
        group_layout.setSpacing(window_pad(0.4))
        cards = QtWidgets.QVBoxLayout()
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setSpacing(window_pad(0.4))
        group_layout.addLayout(cards)
        button = FluentButton("Add device", color=ACCENT)
        button.clicked.connect(
            lambda _checked=False, value=domain: self._add_domain_clicked(value)
        )
        group_layout.addWidget(button)
        self.domain_groups[domain] = group
        self.domain_card_layouts[domain] = cards
        self.domain_add_buttons[domain] = button
        self.domain_column.insertWidget(
            self.domain_column.indexOf(self.unavailable_note),
            group,
        )

    def _choices_for_type(
        self,
        type_key: str,
        domain: str,
    ) -> tuple[tuple[str, str], ...]:
        choices = self._choices_by_domain.get(domain, ())
        if any(key == type_key for _label, key in choices):
            return choices
        return ((type_key, type_key), *choices)

    def _add_domain_clicked(self, domain: str) -> None:
        choices = self._choices_by_domain.get(str(domain), ())
        if choices:
            self.device_add_requested.emit(choices[0][1])

    def _template_chosen(self, index: int) -> None:
        value = self.new_combo.itemData(index)
        self.new_combo.blockSignals(True)
        try:
            self.new_combo.setCurrentIndex(0)
        finally:
            self.new_combo.blockSignals(False)
        if isinstance(value, str):
            self.template_selected.emit(value)

    def _remove_requested(self, instance_id: str) -> None:
        self.device_remove_requested.emit(str(instance_id))

    def _refresh_controls(self) -> None:
        enabled = not self._busy
        self.new_combo.setEnabled(enabled)
        self.load_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.save_as_button.setEnabled(enabled)
        self.lifecycle_button.setEnabled(enabled and self._lifecycle_enabled)
        self.discover_button.setEnabled(enabled and self._discovery_enabled)
        self.discover_button.setToolTip(
            "" if self._discovery_enabled else self._discovery_disabled_reason
        )
        for domain, button in self.domain_add_buttons.items():
            button.setEnabled(enabled and bool(self._choices_by_domain.get(domain)))
        for card in self._cards.values():
            card.setEnabled(enabled)
        for instance_id, (_frame, _title, _detail, button) in self._discovered_widgets.items():
            button.setEnabled(enabled and instance_id not in self._configured_discoveries)


__all__ = ["DeviceManagerView"]
