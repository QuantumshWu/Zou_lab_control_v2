"""Pure view for editing a plain device-instance presentation."""

from __future__ import annotations

from collections.abc import Mapping

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    ElidedLabel,
    FluentLabel,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentGroupBox,
    FluentLineEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentStatusDot,
    FluentSwitch,
    FluentTabWidget,
    muted_note_label,
    retire_widget,
    signals_blocked,
    window_pad,
)
from zlc_data.units import DEFAULT_UNITS, NO_PREFIX, UnitError, format_quantity

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
        self.name_label = muted_note_label("Role")
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
        """Project the host's record; a projection is never a pick.

        ``type_picked`` is the operator choosing a type.  Selecting the
        projected type here with signals live emitted it too, so every
        ``set_devices`` told the host that the operator had just asked for
        the type the host itself had sent.
        """

        self.role_edit.setText(str(role))
        index = self.type_combo.findData(str(type_key))
        if index >= 0:
            with signals_blocked(self.type_combo):
                self.type_combo.setCurrentIndex(index)
        self.name_label.setToolTip(f"Internal ID: {self.instance_id}")

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


class _LiveDeviceCard(FluentFrame):
    """Identity and entry point for one device in the loaded session."""

    device_open_requested = QtCore.pyqtSignal(str)
    device_close_requested = QtCore.pyqtSignal(str)
    device_remote_toggled = QtCore.pyqtSignal(str)
    device_log_requested = QtCore.pyqtSignal(str)

    def __init__(self, instance_id: str, parent=None) -> None:
        super().__init__(parent, bordered=True)
        self.instance_id = str(instance_id)
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(
            window_pad(0.45), window_pad(0.35), window_pad(0.45), window_pad(0.35)
        )
        outer.setSpacing(window_pad(0.3))
        self.role_label = ElidedLabel("")
        self.detail_label = muted_note_label("")
        self.control_button = FluentButton("Control", color=ACCENT)
        #: One click beside Control publishes this device on the bench
        #: fabric, so the OTHER machine's "Scan hardware" finds it with no
        #: address typed anywhere.  A second click withdraws it.
        self.remote_button = FluentButton("Remote", color=GREY)
        #: The device's own console: who is on its knobs -- local tunes,
        #: its in-process server, and remote clients once published.
        self.log_button = FluentButton("Log", color=GREY)
        self.close_button = FluentButton("Close", color=ORANGE)
        identity = QtWidgets.QVBoxLayout()
        identity.setContentsMargins(0, 0, 0, 0)
        identity.setSpacing(0)
        identity.addWidget(self.role_label)
        identity.addWidget(self.detail_label)
        outer.addLayout(identity, 1)
        outer.addWidget(self.control_button)
        outer.addWidget(self.remote_button)
        outer.addWidget(self.log_button)
        outer.addWidget(self.close_button)
        self.control_button.clicked.connect(
            lambda: self.device_open_requested.emit(self.instance_id)
        )
        self.remote_button.clicked.connect(
            lambda: self.device_remote_toggled.emit(self.instance_id)
        )
        self.log_button.clicked.connect(
            lambda: self.device_log_requested.emit(self.instance_id)
        )
        self.close_button.clicked.connect(
            lambda: self.device_close_requested.emit(self.instance_id)
        )

    def set_record(
        self, role: str, type_id: str, *, remote: bool = False
    ) -> None:
        self.role_label.setText(str(role))
        self.detail_label.setText(str(type_id))
        # Colour alone says published; the caption stays put so the row's
        # layout does not shuffle every time Remote is toggled.
        self.remote_button.set_color(ACCENT if remote else GREY)
        self.setToolTip(f"Internal ID: {self.instance_id}")


def _readable_value(value: object, unit: str, shown: str = "") -> str:
    """One device reading, in the same words its editable twin uses.

    In the spelling the operator chose for the row when they chose one --
    a reading in hertz beside a box being read in megahertz is two numbers
    for one field in two scales -- and otherwise in the readable prefix of
    the field's own unit.  Falls back to the plain text for anything that is
    not a number: a device may report a name, a mode, or a reason it cannot
    answer, and none of them is improved by a number formatter.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    own = (unit or "1").strip() or "1"
    chosen = (shown or "").strip()
    try:
        if chosen and chosen != own:
            return format_quantity(
                float(DEFAULT_UNITS.convert(value, own, chosen)), chosen, prefix=NO_PREFIX
            )
        return format_quantity(value, own)
    except UnitError:
        return str(value)


def _readable_limits(limits: object, unit: str, shown: str = "") -> str:
    """The instrument's own range for a knob, read the way its row is read.

    ``limits`` is the ``(low, high)`` pair a device reports for the field,
    or None when it states none (a switch, a policy edge); it comes in the
    field's own unit and is printed in whichever spelling the operator chose
    for the row, so the fence and the number it fences share one scale.
    """

    if limits is None:
        return ""
    low, high = limits
    return f"{_readable_value(low, unit, shown)} to {_readable_value(high, unit, shown)}"


class DeviceControlView(QtWidgets.QWidget):
    """Projection-only control surface for one loaded device.

    Each field of the projection carries, beside the reading and the desired
    value, ``device_limits``: the instrument's own range as the device
    reports it, or None.  It is SHOWN, read-only, in its own column beside
    the editable window -- an operator setting a bench window has to see
    which of the two fences is biting -- and never edited: the device's
    limit is the device's fact.
    """

    refresh_requested = QtCore.pyqtSignal()
    risk_toggled = QtCore.pyqtSignal(bool)
    field_desired_changed = QtCore.pyqtSignal(str, object, str)
    field_unit_requested = QtCore.pyqtSignal(str, str)
    field_live_apply_toggled = QtCore.pyqtSignal(str, bool)
    field_apply_requested = QtCore.pyqtSignal(str, object, str)

    def __init__(
        self,
        spec: FormSpec,
        projection: Mapping[str, object],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._field_rows: dict[
            str,
            tuple[ElidedLabel, ElidedLabel, FluentSwitch, FluentButton, FluentStatusDot, ElidedLabel],
        ] = {}
        self._field_states: dict[str, Mapping[str, object]] = {}
        self._live_timers: dict[str, QtCore.QTimer] = {}
        outer = QtWidgets.QVBoxLayout(self)
        pad = window_pad()
        outer.setContentsMargins(pad, pad, pad, pad)
        outer.setSpacing(window_pad(0.5))

        header = FluentFrame(self, bordered=True)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(pad, pad, pad, pad)
        owner_stack = QtWidgets.QVBoxLayout()
        self.owner_label = FluentSectionLabel("Owner: none")
        self.reason_label = muted_note_label("No control policy projection")
        self.reason_label.setWordWrap(True)
        owner_stack.addWidget(self.owner_label)
        owner_stack.addWidget(self.reason_label)
        header_layout.addLayout(owner_stack, 1)
        self.refresh_button = FluentButton("Refresh", color=ACCENT)
        self.risk_switch = FluentSwitch("Accept risk")
        header_layout.addWidget(self.refresh_button)
        header_layout.addWidget(self.risk_switch)
        outer.addWidget(header)

        self.columns = QtWidgets.QWidget(self)
        columns = QtWidgets.QHBoxLayout(self.columns)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(window_pad(0.35))
        self.field_heading = muted_note_label("Field")
        self.current_heading = muted_note_label("Current")
        self.desired_heading = muted_note_label("Desired")
        self.limits_heading = muted_note_label("Device limits")
        self.live_heading = muted_note_label("Live")
        self.apply_heading = muted_note_label("Apply")
        self.status_heading = muted_note_label("Status")
        # Only the CURRENT column is a number of our own; every other heading
        # takes its width from the widget under it, in _align_headings.
        self.current_heading.setFixedWidth(window_pad(8.0))
        # These two name a control that is centred in its cell, so the word is
        # centred over it rather than hanging off the column's left edge.
        for heading in (self.live_heading, self.apply_heading):
            heading.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        columns.addWidget(self.field_heading)
        columns.addWidget(self.current_heading)
        columns.addWidget(self.desired_heading, 1)
        columns.addWidget(self.limits_heading)
        columns.addWidget(self.live_heading)
        columns.addWidget(self.apply_heading)
        columns.addWidget(self.status_heading)
        outer.addWidget(self.columns)

        fields = projection.get("fields", {})
        desired = {key: fields[key]["desired"] for key in spec.keys}
        # The form builds each row COMPLETE -- reading, editor, live switch,
        # Apply, status -- through this hook, so no row is ever shown and
        # then taken apart to have its cells added.
        self.form = FluentParameterForm(
            spec, desired, parent=self, row_cells=self._compose_cells
        )
        self.form.changed.connect(self._desired_changed)
        self.form.shown_unit_changed.connect(self._shown_unit_changed)
        outer.addWidget(self.form)
        outer.addStretch(1)
        self.status_strip = StatusStrip(self)
        outer.addWidget(self.status_strip)
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        self.risk_switch.toggled.connect(self.risk_toggled.emit)
        self.set_projection(spec, projection)

    def _retire_field_row(self, key: str) -> None:
        timer = self._live_timers.pop(str(key), None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        # The cells belong to the row the form built them into, and go with it.
        self._field_rows.pop(str(key), None)

    def _compose_cells(self, field):
        """This row's cells, built once as the form builds the row."""

        key = field.key
        self._retire_field_row(key)
        current = ElidedLabel("\u2014")
        current.setFixedWidth(self.current_heading.width())
        # The switch lives in a cell of its own so a field with no live
        # write can leave the CELL EMPTY without the row closing up: the
        # column keeps its width, the rows keep their grid.
        live_host = QtWidgets.QWidget()
        live_layout = QtWidgets.QHBoxLayout(live_host)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(0)
        live = FluentSwitch("", live_host)
        live_layout.addWidget(live, 0, QtCore.Qt.AlignCenter)
        live_host.setFixedWidth(max(1, live.sizeHint().width()))
        # The device's own range, a VALUE beside the editable window, never
        # a control.  A plain label, not an eliding one: a fence the operator
        # cannot read whole tells them nothing, so the text asks for its
        # full width and the column (and the snug window) grow to it.
        limits = FluentLabel("")
        limits.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        limits.setToolTip(
            "The instrument's own range for this field; the editable window "
            "is the tighter of it and the bench policy"
        )
        apply = FluentButton("Apply", color=ACCENT)
        dot = FluentStatusDot(size=12)
        status = ElidedLabel("")
        # Status has one shared width. Long messages elide behind their
        # tooltip, without moving the controls in another row.
        status.setMinimumWidth(
            status.fontMetrics().horizontalAdvance("Applying; latest queued")
        )
        status_host = QtWidgets.QWidget()
        status_layout = QtWidgets.QHBoxLayout(status_host)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(window_pad(0.2))
        status_layout.addWidget(dot)
        status_layout.addWidget(status, 1)
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(75)
        timer.timeout.connect(lambda value=key: self._emit_apply(value))
        live.toggled.connect(
            lambda checked, value=key: self._live_toggled(value, checked)
        )
        apply.clicked.connect(lambda _checked=False, value=key: self._emit_apply(value))
        self._field_rows[key] = current, limits, live, apply, dot, status
        self._live_timers[key] = timer
        return (current,), ((limits, 0), (live_host, 0), (apply, 0), (status_host, 0))

    def _prune_rows(self) -> None:
        """Forget the cells of rows the form no longer has."""

        for key, widgets in tuple(self._field_rows.items()):
            row = self.form._rows.get(key)
            if row is None or widgets[0].parentWidget() is not row:
                self._retire_field_row(key)
        self._align_headings()

    def _align_headings(self) -> None:
        """One column budget for the header and every complete form row.

        Only Desired expands. Two independent expanding columns let each
        row's different editor minimum move the columns after it. Unit
        pickers also share a width so the numeric editors have one right edge.
        """

        rows = tuple(self.form._rows.values())
        if not rows:
            self.field_heading.setFixedWidth(0)
            return
        layout = rows[0].layout()
        headings = self.columns.layout()
        # Only what differs: a layout told its spacing again invalidates
        # itself, and this runs on every projected beat.
        if headings.contentsMargins() != layout.contentsMargins():
            headings.setContentsMargins(layout.contentsMargins())
        if headings.spacing() != layout.spacing():
            headings.setSpacing(layout.spacing())
        if layout.count() < 7:
            return
        pickers = tuple(self.form._unit_pickers.values())
        if pickers:
            width = max(picker.sizeHint().width() for picker in pickers)
            for picker in pickers:
                if picker.minimumWidth() != width or picker.maximumWidth() != width:
                    picker.setFixedWidth(width)
        header_cells = (
            self.field_heading, self.current_heading, self.desired_heading,
            self.limits_heading, self.live_heading, self.apply_heading, self.status_heading,
        )
        row_layouts = tuple(row.layout() for row in rows)
        widths = []
        for index, heading in enumerate(header_cells):
            cells = (heading, *(row.itemAt(index).widget() for row in row_layouts))
            width = max(max(cell.sizeHint().width(), cell.minimumSizeHint().width()) for cell in cells)
            widths.append(width)
            for cell in cells:
                if index == 2:
                    policy = cell.sizePolicy()
                    if policy.horizontalPolicy() != QtWidgets.QSizePolicy.Expanding:
                        cell.setSizePolicy(QtWidgets.QSizePolicy.Expanding, policy.verticalPolicy())
                    if cell.minimumWidth() != width:
                        cell.setMinimumWidth(width)
                elif cell.minimumWidth() != width or cell.maximumWidth() != width:
                    cell.setFixedWidth(width)
            for row in (headings, *row_layouts):
                stretch = 1 if index == 2 else 0
                if row.stretch(index) != stretch:
                    row.setStretch(index, stretch)
        margins = (self.layout().contentsMargins(), self.form.layout().contentsMargins(), layout.contentsMargins())
        minimum = sum(widths) + layout.spacing() * (len(widths) - 1)
        minimum += sum(margin.left() + margin.right() for margin in margins)
        if self.minimumWidth() != minimum:
            self.setMinimumWidth(minimum)

    def align_columns(self) -> None:
        """Give every fixed column its width and let the layout settle.

        The width a control needs is only known once its columns agree:
        before that, a row's size hint is the sum of its widgets' own
        hints, which an eliding label keeps small, and a window opened on
        that hint has its fixed columns overlapping the stretched ones the
        moment they are aligned.  So whoever decides the opening size asks
        for this first.
        """

        self._align_headings()
        layout = self.layout()
        if layout is not None:
            layout.activate()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt naming
        """The width the control needs is its widest aligned row.

        A parameter form is built to take the width it is given and never
        to widen the page it sits on, so its own hint says nothing about
        how wide its rows want to be -- and a window sized by content that
        read it opened at half the standard width whatever the columns
        held, the fixed columns then overlapping the stretched ones.
        """

        hint = super().sizeHint()
        rows = tuple(self.form._rows.values())
        if not rows:
            return hint
        widest = max(
            row.layout().sizeHint().width() for row in rows if row.layout() is not None
        )
        margins = [self.layout().contentsMargins()] if self.layout() else []
        if self.form.layout() is not None:
            margins.append(self.form.layout().contentsMargins())
        width = widest + sum(margin.left() + margin.right() for margin in margins)
        return QtCore.QSize(max(hint.width(), width), hint.height())

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._align_headings()

    def hideEvent(self, event):  # noqa: N802 - Qt naming
        for timer in self._live_timers.values():
            timer.stop()
        super().hideEvent(event)

    def set_projection(self, spec: FormSpec, projection: Mapping[str, object]) -> None:
        fields = projection.get("fields", {})
        self._field_states = dict(fields)
        desired = {key: fields[key]["desired"] for key in spec.keys}
        self.form.reconcile(spec, desired)
        self._prune_rows()
        owners = tuple(str(value) for value in projection.get("owners", ()))
        self.owner_label.setText(f"Owner: {', '.join(owners) if owners else 'none'}")
        self.reason_label.setText(str(projection.get("reason", "")))
        with signals_blocked(self.risk_switch):
            self.risk_switch.setChecked(bool(projection.get("risk_accepted", False)))
        self.risk_switch.setEnabled(bool(projection.get("risk_enabled", False)))
        colours = {
            "info": GREY, "idle": GREY, "ready": GREEN,
            "task": ORANGE, "warning": ORANGE, "error": RED,
        }
        units = {declared.key: declared.unit for declared in spec.fields}
        self._units = units
        for key in spec.keys:
            field = fields[key]
            current, limits, live, apply, dot, status = self._field_rows[key]
            shown = field.get("current")
            # The presenter hands over the device's own number; what it is IN
            # is on the field beside it.  This printed str(value), so a
            # readback of 120000000.0 Hz reached the operator as
            # "120000000.0" -- the one column whose whole job is to be read
            # at a glance, and the hardest thing on the page to read.
            current.setText(
                "—"
                if shown is None
                else _readable_value(shown, units[key], self.form.shown_unit_for(key))
            )
            limits.setText(
                _readable_limits(
                    field.get("device_limits"), units[key], self.form.shown_unit_for(key)
                )
            )
            editor = self.form.widget_for(key)
            self._set_editable(key, bool(field.get("editable", False)))
            with signals_blocked(live):
                live.setChecked(bool(field.get("live_apply", False)))
            # Absent, not disabled: a control that can never be pressed is a
            # question the operator has to answer every time they read the row.
            live.setVisible(bool(field.get("live_capable", True)))
            live.setEnabled(bool(field.get("live_enabled", False)))
            apply.setEnabled(bool(field.get("apply_enabled", False)))
            if not bool(field.get("editable", False)) or not live.isChecked():
                self._live_timers[key].stop()
            status.setText(str(field.get("status", "")))
            dot.set_color(colours.get(str(field.get("severity", "info")), GREY))
            reason = str(field.get("reason", ""))
            for widget in (editor, live, apply, status):
                widget.setToolTip(reason)
        # NOW, not on a timer.  Every width this reads is a size hint, which
        # exists before anything is painted; deferring it to the next pass
        # put the headings on screen at one width and moved them at the
        # first paint, which is the flicker every opened control showed.
        self._align_headings()

    def _shown_unit_changed(self, key: str, symbol: str) -> None:
        """Ask the device owner for a read-only unit projection."""
        self._live_timers[str(key)].stop()
        old_unit = self.form._field_for(str(key)).unit or "1"
        # The shared form's conversion is not device-load-aware. Keep the
        # previous complete row until its owner supplies the converted number,
        # current reading and bounds together; no temporary 50-ohm value.
        self.form.widget_for(str(key)).setShownUnit(old_unit)
        self.form._unit_pickers[str(key)].select_choice_key(old_unit)
        self.field_unit_requested.emit(str(key), symbol)

    def _desired_changed(self, key: str) -> None:
        try:
            value = self.form.read_value(key)
        except (TypeError, ValueError):
            return
        for name, state in self._field_states.items():
            self._set_editable(name, bool(state.get("editable", False)))
        self.field_desired_changed.emit(str(key), value, self.form._field_for(key).unit or "")
        row = self._field_rows.get(str(key))
        live = None if row is None else row[2]
        if live is not None and live.isChecked() and live.isEnabled():
            self._live_timers[str(key)].start()

    def _set_editable(self, key: str, enabled: bool) -> None:
        self.form.widget_for(str(key)).setEnabled(bool(enabled))
        automatic = self.form._auto_switches.get(str(key))
        if automatic is not None:
            automatic.setEnabled(bool(enabled))

    def _live_toggled(self, key: str, checked: bool) -> None:
        if not checked:
            self._live_timers[str(key)].stop()
        self.field_live_apply_toggled.emit(str(key), bool(checked))

    def _emit_apply(self, key: str) -> None:
        try:
            value = self.form.read_value(str(key))
        except (TypeError, ValueError) as error:
            self.show_status(str(error), "error")
            return
        self.field_apply_requested.emit(str(key), value, self.form._field_for(str(key)).unit or "")

    def show_status(self, text: str, severity: str) -> None:
        self.status_strip.show_status(str(text), str(severity))


class _ServerLogView(QtWidgets.QPlainTextEdit):
    """Live tail of every server this bench process runs.

    The widget polls a snapshot callable instead of being pushed lines:
    server threads may narrate at any moment, and a poll on the GUI clock is
    the one crossing that never needs marshalling.

    It polls only while it is on screen.  The clock follows the widget's
    own show and hide, so a log window that was closed -- which hides it --
    stops asking for the snapshot, and one shown again catches up at once.
    """

    def __init__(self, snapshot, parent=None) -> None:
        super().__init__(parent)
        self._snapshot = snapshot
        self._seen = -1
        self.setReadOnly(True)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.setMaximumBlockCount(4200)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().showEvent(event)
        self.refresh()
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self._timer.stop()
        super().hideEvent(event)

    def refresh(self) -> None:
        total, lines = self._snapshot()
        if total == self._seen:
            return
        self._seen = total
        bar = self.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 4
        self.setPlainText(
            "\n".join(lines) if lines else "No interactions recorded yet."
        )
        if follow:
            bar.setValue(bar.maximum())


class DeviceManagerView(QtWidgets.QWidget):
    """The Config surface for one plain-data apparatus draft."""

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
    #: Open the independent control surface for a loaded device.  This view
    #: neither constructs that surface nor touches the device.
    device_open_requested = QtCore.pyqtSignal(str)
    #: Request retirement of exactly one loaded device.  Session ownership,
    #: claims and the actual close remain outside this view.
    device_close_requested = QtCore.pyqtSignal(str)
    #: Toggle publishing one loaded device on the bench fabric.
    device_remote_toggled = QtCore.pyqtSignal(str)
    #: Open the live log of ONE published device (its server narration).
    device_log_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._choices: tuple[tuple[str, str, str], ...] = ()
        self._choices_by_domain: dict[str, tuple[tuple[str, str], ...]] = {}
        self._cards: dict[str, _DeviceCard] = {}
        self._card_domains: dict[str, str] = {}
        self._discovered_widgets: dict[str, tuple[FluentFrame, ElidedLabel, QtWidgets.QLabel, FluentButton]] = {}
        self._configured_discoveries: set[str] = set()
        self._loaded_cards: dict[str, _LiveDeviceCard] = {}
        self._remoted: set[str] = set()
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
        # In the compact chrome, the left-hand title never changes; runtime state
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
            "Control opens each loaded device in its own window; this window "
            "authors the device graph."
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

        The adjacent dot deliberately remains a session-state indicator;
        :meth:`set_lifecycle` is its sole writer.
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

    def set_loaded_devices(
        self,
        devices: tuple[tuple[str, str, str], ...],
    ) -> None:
        wanted = {str(instance_id) for instance_id, _role, _type_id in devices}
        for instance_id in tuple(self._loaded_cards):
            if instance_id in wanted:
                continue
            card = self._loaded_cards.pop(instance_id)
            self.loaded_layout.removeWidget(card)
            card.hide()
            card.deleteLater()
        for index, (instance_id, role, type_id) in enumerate(devices):
            instance_id = str(instance_id)
            card = self._loaded_cards.get(instance_id)
            if card is None:
                card = _LiveDeviceCard(instance_id, self.loaded_group)
                card.device_open_requested.connect(self.device_open_requested)
                card.device_close_requested.connect(self.device_close_requested)
                card.device_remote_toggled.connect(self.device_remote_toggled)
                card.device_log_requested.connect(self.device_log_requested)
                self._loaded_cards[instance_id] = card
                self.loaded_layout.insertWidget(index, card)
            card.set_record(
                str(role),
                str(type_id),
                remote=instance_id in self._remoted,
            )
            log = getattr(self, "_device_log_windows", {}).get(instance_id)
            if log is not None:
                log.setWindowTitle(f"{role} log@Zou lab")
        self.loaded_empty.setVisible(not devices)

    def open_device_log(self, instance_id: str, snapshot, *, label: str) -> None:
        """Open (or re-front) the live log window of ONE published device."""

        key = str(instance_id)
        windows = getattr(self, "_device_log_windows", None)
        if windows is None:
            windows = self._device_log_windows = {}
        window = windows.get(key)
        if window is not None and window.isVisible():
            window.setWindowTitle(f"{label} log@Zou lab")
            window.showNormal()
            window.raise_()
            window.activateWindow()
            return
        from zlc_ui.fluent import open_fluent_window

        window = windows[key] = open_fluent_window(
            lambda: _ServerLogView(snapshot),
            title=f"{label} log@Zou lab",
            window_ratio=0.45,
        )

        def forget() -> None:
            # A closed log window is over: the next request builds a new
            # one, and a window this view no longer names is retired rather
            # than kept, hidden, for the life of the manager.
            if windows.get(key) is window:
                del windows[key]
            retire_widget(window)

        window.closed.connect(forget)

    def set_remoted(self, instance_ids) -> None:
        """Mark which loaded devices are currently published on the fabric."""

        self._remoted = {str(instance_id) for instance_id in instance_ids}
        for instance_id, card in self._loaded_cards.items():
            card.remote_button.set_color(
                ACCENT if instance_id in self._remoted else GREY
            )

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
        changed: bool = False,
    ) -> None:
        self._busy = bool(busy)
        self._lifecycle_enabled = bool(enabled)
        self.lifecycle_button.setText(str(text))
        pending = bool(active and changed)
        self.status_dot.set_color(ORANGE if pending else GREEN if active else GREY)
        self.status_dot.setToolTip(
            "Configuration differs from the active installation"
            if pending
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
        # The card edits a DRAFT: an empty required box is a vacancy to
        # keep, not a reason to refuse the boxes that are filled.
        return tuple(card.form.read_draft().items())

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
        for card in self._loaded_cards.values():
            card.setEnabled(enabled)
        for instance_id, (_frame, _title, _detail, button) in self._discovered_widgets.items():
            button.setEnabled(enabled and instance_id not in self._configured_discoveries)


__all__ = ["DeviceControlView", "DeviceManagerView"]
