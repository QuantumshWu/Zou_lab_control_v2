"""Pure view for editing a plain device-instance presentation."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    GREEN,
    GREY,
    ORANGE,
    FluentStatusDot,
    ACCENT,
    GREY,
    ORANGE,
    ElidedLabel,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentLineEdit,
    FluentScrollArea,
    FluentSectionLabel,
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
        self.type_combo.currentIndexChanged.connect(self._type_changed)
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
    """A plain-data device editor surface with injected form projections."""

    device_add_requested = QtCore.pyqtSignal(str)
    save_requested = QtCore.pyqtSignal()
    #: Bring the written apparatus up and say what answered.  A configuration
    #: is a claim; this is where the claim gets checked, rather than ten
    #: minutes into a run.
    test_requested = QtCore.pyqtSignal()
    device_remove_requested = QtCore.pyqtSignal(str)
    role_committed = QtCore.pyqtSignal(str, str)
    type_picked = QtCore.pyqtSignal(str, str)
    parameter_committed = QtCore.pyqtSignal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._choices: tuple[tuple[str, str], ...] = ()
        self._cards: dict[str, _DeviceCard] = {}
        outer = QtWidgets.QVBoxLayout(self)
        pad = window_pad()
        outer.setContentsMargins(pad, pad, pad, pad)
        outer.setSpacing(window_pad(0.5))

        header = QtWidgets.QHBoxLayout()
        # Which file, and whether what is on screen is in it.  An editor that
        # cannot say "you have unsaved changes" is one you close and lose work
        # from -- v1 put a dot and a [*] here and recomputed both on every edit.
        self.status_dot = FluentStatusDot(size=16)
        self.status_dot.set_color(GREY)
        self.title_label = FluentSectionLabel("Devices")
        header.addWidget(self.status_dot)
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.add_type_combo = FluentComboBox()
        self.add_button = FluentButton("Add", color=ACCENT)
        # An apparatus editor that cannot write the apparatus is a viewer.
        self.save_button = FluentButton("Save", color=ACCENT)
        self.test_button = FluentButton("Test devices", color=ORANGE)
        header.addWidget(self.add_type_combo)
        header.addWidget(self.add_button)
        header.addWidget(self.test_button)
        header.addWidget(self.save_button)
        outer.addLayout(header)

        self.scroll = FluentScrollArea()
        self.body = QtWidgets.QWidget()
        self.cards_layout = QtWidgets.QVBoxLayout(self.body)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(window_pad(0.4))
        self.cards_layout.addStretch(1)
        self.scroll.set_width_bounded_widget(self.body)
        outer.addWidget(self.scroll, 1)
        self.status_strip = StatusStrip()
        outer.addWidget(self.status_strip)
        self.add_button.clicked.connect(self._add_clicked)
        self.save_button.clicked.connect(self.save_requested.emit)
        self.test_button.clicked.connect(self.test_requested.emit)

    def set_apparatus(self, name: str, dirty: bool, saved: bool) -> None:
        """Say which apparatus this is and whether it is written down.

        Grey = nothing saved yet, orange = edits the file does not have,
        green = the file and the screen agree.  The colour is the answer to
        "can I close this window", which nothing on screen used to give.
        """

        self.title_label.setText(f"{name}{'*' if dirty else ''}")
        self.status_dot.set_color(
            ORANGE if dirty else (GREEN if saved else GREY)
        )

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
        for card in self._cards.values():
            card.set_choices(self._choices)

    def set_devices(self, devices: tuple[tuple[str, str, str], ...]) -> None:
        wanted = tuple((str(instance_id), str(role), str(type_key)) for instance_id, role, type_key in devices)
        wanted_ids = {instance_id for instance_id, _role, _type_key in wanted}
        for instance_id in tuple(self._cards):
            if instance_id not in wanted_ids:
                card = self._cards.pop(instance_id)
                self.cards_layout.removeWidget(card)
                card.setParent(None)
                card.deleteLater()
        for instance_id, role, type_key in wanted:
            card = self._cards.get(instance_id)
            if card is None:
                card = _DeviceCard(instance_id, self.body)
                card.type_picked.connect(self.type_picked.emit)
                card.role_committed.connect(self.role_committed.emit)
                card.remove_requested.connect(self._remove_requested)
                card.parameter_committed.connect(self.parameter_committed.emit)
                self._cards[instance_id] = card
                self.cards_layout.insertWidget(max(0, self.cards_layout.count() - 1), card)
                card.set_choices(self._choices)
            card.set_record(role, type_key)

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

    def _remove_requested(self, instance_id: str) -> None:
        self.device_remove_requested.emit(str(instance_id))


__all__ = ["DeviceManagerView"]
