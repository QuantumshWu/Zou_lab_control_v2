"""The scan plan, as a table of axes instead of a JSON box.

One row per axis: which port, from, to, how many points.  Row order is the
nesting order, outermost first, which is how the plan itself reads.  The
editor owns exactly one authored field -- ``plan`` -- and says so through
``managed_fields``, so the auto-generated form does not render the raw JSON
beside it.

A MANUAL axis is the other kind of row: a name and a point count, and no
port at all, because nothing here can advance it.  It carries no values
either -- the operator types those when the run asks, which is the whole
reason the axis exists.  Manual rows sit above the machine rows and cannot
be moved below them: an operator walks their points BETWEEN plays of the
inner plan, so they are outside it by construction.

Ports are read from the projection -- the resolved pulse template's API
parameters, plus the bench's tunable devices for a node that can move them --
the same set that node's binding enforces.  A board-advanced scan cannot make
a host call between two rows of one table, so its editor never offers a
device port: the form shows what the node would accept, not what some scan
somewhere could.  An axis whose values are not a uniform grid (authored in a
notebook, say) is shown as "custom" and left untouched until a spin is edited,
at which point it becomes the uniform grid the spins describe.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
from PyQt5 import QtCore, QtWidgets
from zlc_ui.fluent import (
    ACCENT,
    GREY,
    FluentButton,
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentLineEdit,
    FluentSpinBox,
)

from zlc_pulse import apply_api_values, authored_api_entries, field_label

from zlc_atom.pulse_values import (
    CURRENT_API_VALUES,
    current_api_values_path,
    read_api_values,
)

from .plan import (
    MANUAL_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    api_overrides_from_authored,
    api_overrides_to_authored,
    ScanAxis,
    ScanPlan,
    hardware_scan_ports_for,
    manual_axis,
    manual_axis_name,
    scan_ports_for,
    DEVICE_PARAM_FAMILY,
    scan_ports_for_devices,
)


def _uniform(values: tuple[float, ...]) -> bool:
    if len(values) < 3:
        return True
    steps = np.diff(np.asarray(values, dtype=float))
    return bool(np.allclose(steps, steps[0]))


class _AxisRow(QtWidgets.QWidget):
    """One axis: port, from, to, points, and the remove button."""

    edited = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal(object)

    def __init__(self, ports, axis: ScanAxis | None, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.port_combo = FluentComboBox()
        for port in ports:
            self.port_combo.addItem(port.label, port.port)
        self.start_spin = FluentDoubleSpinBox()
        self.stop_spin = FluentDoubleSpinBox()
        self.points_spin = FluentSpinBox()
        self.points_spin.setRange(1, 100_000)
        self.custom_label = QtWidgets.QLabel("")
        remove = FluentButton("×", color=GREY)
        remove.setFixedWidth(32)
        remove.setToolTip("Remove this axis")
        self.remove_button = remove
        layout.addWidget(self.port_combo, 2)
        layout.addWidget(QtWidgets.QLabel("from"))
        layout.addWidget(self.start_spin, 1)
        layout.addWidget(QtWidgets.QLabel("to"))
        layout.addWidget(self.stop_spin, 1)
        layout.addWidget(QtWidgets.QLabel("points"))
        layout.addWidget(self.points_spin)
        layout.addWidget(self.custom_label)
        layout.addWidget(remove)

        self._ports = tuple(ports)
        self._custom_values: tuple[float, ...] | None = None
        self._apply_port_limits()
        if axis is not None:
            at = self.port_combo.findData(axis.port)
            if at < 0:
                # The pulse no longer offers this port -- it was renamed, or
                # the binding was taken off the field.  Keep saying what the
                # operator authored: silently selecting whatever sits at index
                # zero re-points their axis at an unrelated knob, and the run
                # would sweep it without a word.  Left as it is, bind_plan
                # refuses by name before anything arms.
                self.port_combo.addItem(f"{axis.port} (not in this pulse)", axis.port)
                at = self.port_combo.count() - 1
            self.port_combo.setCurrentIndex(at)
            self._apply_port_limits()
            self.start_spin.setValue(axis.values[0])
            self.stop_spin.setValue(axis.values[-1])
            self.points_spin.setValue(len(axis.values))
            if not _uniform(axis.values):
                self._custom_values = axis.values
                self.custom_label.setText("custom values")
        else:
            port = self._ports[0] if self._ports else None
            if port is not None:
                self.start_spin.setValue(port.seed_lo)
                self.stop_spin.setValue(port.seed_hi)
            self.points_spin.setValue(5)

        self.port_combo.currentIndexChanged[int].connect(self._port_changed)
        for spin in (self.start_spin, self.stop_spin, self.points_spin):
            spin.valueChanged.connect(self._spins_edited)
        remove.clicked.connect(lambda: self.remove_requested.emit(self))

    def _apply_port_limits(self) -> None:
        port = next(
            (p for p in self._ports if p.port == self.port_combo.currentData()),
            None,
        )
        for spin in (self.start_spin, self.stop_spin):
            if port is None:
                spin.setRange(-1e12, 1e12)
            else:
                spin.setRange(port.lo, port.hi)
            # The port has said all along what its numbers are in -- a
            # duration sweeps in the period's own unit -- and these two boxes
            # were the one place on the row that never repeated it, so a
            # seamless axis read "from 1 to 40" with nothing saying of what.
            #
            # setDecimals(4) went with it.  It did not make the number
            # readable, it made it four decimals long: an authored 1.00005 us
            # came back as 1.0 in the box that is supposed to be showing what
            # will run.  Readability is the formatter's job now, and the
            # formatter does not round.
            spin.setDisplayUnit("" if port is None else port.unit)

    def _port_changed(self, _index: int) -> None:
        self._custom_values = None
        self.custom_label.setText("")
        self._apply_port_limits()
        port = next(
            (p for p in self._ports if p.port == self.port_combo.currentData()),
            None,
        )
        if port is not None:
            self.start_spin.setValue(port.seed_lo)
            self.stop_spin.setValue(port.seed_hi)
        self.edited.emit()

    def _spins_edited(self) -> None:
        # An edited spin means the operator wants the uniform grid the spins
        # describe; a custom list survives only while it is left alone.
        self._custom_values = None
        self.custom_label.setText("")
        self.edited.emit()

    @property
    def manual(self) -> bool:
        return False

    def axis(self) -> ScanAxis:
        if self._custom_values is not None:
            return ScanAxis(str(self.port_combo.currentData()), self._custom_values)
        points = int(self.points_spin.value())
        values = np.linspace(
            float(self.start_spin.value()), float(self.stop_spin.value()), points
        )
        return ScanAxis(
            str(self.port_combo.currentData()),
            tuple(float(value) for value in values),
        )


class _ManualAxisRow(QtWidgets.QWidget):
    """One manual axis: a name, its values, and the remove button.

    The same from/to/points every other row authors, because a manual
    axis IS a scan axis -- its coordinates have to exist before the data
    they describe, whoever turns the knob.  What it has instead of a port
    is a name, since no port here reaches the thing it moves.
    """

    edited = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal(object)

    def __init__(self, axis: ScanAxis | None, name: str = "", parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.name_edit = FluentLineEdit()
        self.name_edit.setPlaceholderText("manual axis name")
        self.start_spin = FluentDoubleSpinBox()
        self.stop_spin = FluentDoubleSpinBox()
        for spin in (self.start_spin, self.stop_spin):
            # No port, so no hard limits to read: a hand's range is
            # whatever the bench's own knob will do.
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(4)
        self.points_spin = FluentSpinBox()
        self.points_spin.setRange(1, 100_000)
        self.custom_label = QtWidgets.QLabel("")
        remove = FluentButton("×", color=GREY)
        remove.setFixedWidth(32)
        remove.setToolTip("Remove this axis")
        self.remove_button = remove
        layout.addWidget(QtWidgets.QLabel("by hand"))
        layout.addWidget(self.name_edit, 2)
        layout.addWidget(QtWidgets.QLabel("from"))
        layout.addWidget(self.start_spin, 1)
        layout.addWidget(QtWidgets.QLabel("to"))
        layout.addWidget(self.stop_spin, 1)
        layout.addWidget(QtWidgets.QLabel("points"))
        layout.addWidget(self.points_spin)
        layout.addWidget(self.custom_label)
        layout.addWidget(remove)

        self._custom_values: tuple[float, ...] | None = None
        if axis is not None:
            self.name_edit.setText(manual_axis_name(axis.port))
            self.start_spin.setValue(axis.values[0])
            self.stop_spin.setValue(axis.values[-1])
            self.points_spin.setValue(len(axis.values))
            if not _uniform(axis.values):
                self._custom_values = axis.values
                self.custom_label.setText("custom values")
        else:
            # A row the operator can run without first naming it: an
            # unnamed axis is not a plan, and an empty box is a form that
            # refuses to be used until it is filled in.
            self.name_edit.setText(str(name))
            self.stop_spin.setValue(1.0)
            self.points_spin.setValue(3)

        self.name_edit.textChanged.connect(lambda _text: self.edited.emit())
        for spin in (self.start_spin, self.stop_spin, self.points_spin):
            spin.valueChanged.connect(self._spins_edited)
        remove.clicked.connect(lambda: self.remove_requested.emit(self))

    def _spins_edited(self) -> None:
        self._custom_values = None
        self.custom_label.setText("")
        self.edited.emit()

    @property
    def manual(self) -> bool:
        return True

    def axis(self) -> ScanAxis:
        name = self.name_edit.text().strip()
        if self._custom_values is not None:
            return manual_axis(name, self._custom_values)
        points = int(self.points_spin.value())
        values = np.linspace(
            float(self.start_spin.value()), float(self.stop_spin.value()), points
        )
        return manual_axis(name, tuple(float(value) for value in values))


class ScanPlanEditor(QtWidgets.QWidget):
    """The ``plan`` field, authored as axes rather than typed as JSON."""

    draft_changed = QtCore.pyqtSignal(object)
    managed_fields = ("plan", "api_values")

    def __init__(
        self,
        parent=None,
        *,
        device_ports: bool = True,
        hardware_slots: bool = False,
        only_port: str | None = None,
        manual_axes: bool = False,
    ) -> None:
        super().__init__(parent)
        self._device_ports = bool(device_ports)
        self._hardware_slots = bool(hardware_slots)
        # Only a node that can STOP between plays can offer one, so the
        # editor shows the button exactly where the node would honour it.
        self._manual_axes = bool(manual_axes)
        # A node whose measurement IS about one knob -- release-recapture is a
        # statement about t_off and nothing else -- offers that knob and no
        # way to add a second.  The row is always there, so the form opens on
        # something to edit instead of on an empty table.
        self._only_port = None if only_port is None else str(only_port)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Scan plan")
        title.setStyleSheet("font-weight: 600;")
        self.add_button = FluentButton("Add axis", color=ACCENT)
        self.add_manual_button = FluentButton("Add manual axis", color=GREY)
        header.addWidget(title)
        header.addStretch(1)
        if self._only_port is None:
            header.addWidget(self.add_button)
            if self._manual_axes:
                header.addWidget(self.add_manual_button)
            else:
                self.add_manual_button.hide()
        else:
            self.add_button.hide()
            self.add_manual_button.hide()
        column.addLayout(header)
        self.rows_layout = QtWidgets.QVBoxLayout()
        column.addLayout(self.rows_layout)
        self.summary = QtWidgets.QLabel("")
        self.summary.setWordWrap(True)
        column.addWidget(self.summary)

        # The pulse's API slots, set once for this run.  They are not axes:
        # nothing sweeps them, they are the numbers the pulse holds while the
        # table plays.  A slot the plan DOES scan is left out -- the table
        # already says what it plays, and two places saying it is one too many.
        values_header = QtWidgets.QHBoxLayout()
        self.values_title = QtWidgets.QLabel("API values")
        self.values_title.setStyleSheet("font-weight: 600;")
        # Explicit, because a run has to be reproducible from what this node
        # says.  The saved set is a place to PULL numbers from, not a second
        # writer reaching into the pulse behind the form.
        self.load_values_button = FluentButton("Load values…", color=GREY)
        values_header.addWidget(self.values_title)
        values_header.addStretch(1)
        values_header.addWidget(self.load_values_button)
        column.addLayout(values_header)
        self.values_grid = QtWidgets.QGridLayout()
        self.values_grid.setContentsMargins(0, 0, 0, 0)
        column.addLayout(self.values_grid)
        self.values_note = QtWidgets.QLabel("")
        self.values_note.setWordWrap(True)
        column.addWidget(self.values_note)

        self._ports: tuple = ()
        self._rows: list[_AxisRow] = []
        self._loading = False
        self._plan_text = ""
        self._values_text = ""
        self._authored: dict[str, tuple[float, str]] = {}
        self._value_rows: dict[str, QtWidgets.QWidget] = {}
        self._sequence: object = None
        self._pulse_path: object = None
        self.load_values_button.clicked.connect(self._load_values)
        self.add_button.clicked.connect(self._add_axis)
        self.add_manual_button.clicked.connect(self._add_manual_axis)
        # Whether a hand can be waited for is a fact about the NODE, known
        # at construction; the projection only ever changes which ports a
        # machine offers.
        self.add_manual_button.setEnabled(self._manual_axes)

    # ------------------------------------------------------- host contract

    def update_projection(self, projection: Mapping[str, object]) -> None:
        resources = projection.get("workspace_resources") or {}
        resource = resources.get("pulse_template") if isinstance(resources, Mapping) else None
        sequence = getattr(resource, "value", None)
        self._pulse_path = getattr(resource, "path", None)
        # The editor offers exactly what this node's build binds against: the
        # pulse's parameters, and the bench's tunable devices only where a
        # host is there to move them.
        extras = projection.get("bench_extras") or {}
        tunables = extras.get("tunable_devices") if isinstance(extras, Mapping) else None
        template_ports = (
            (
                hardware_scan_ports_for(sequence)
                if self._hardware_slots
                else scan_ports_for(sequence)
            )
            if sequence is not None
            else ()
        )
        ports = template_ports + (
            scan_ports_for_devices(tunables) if self._device_ports else ()
        )
        if self._only_port is not None:
            ports = tuple(port for port in ports if port.port == self._only_port)
        values = projection.get("form_values") or {}
        plan_text = str(values.get("plan") or "") if isinstance(values, Mapping) else ""

        values_text = str(values.get("api_values") or "") if isinstance(values, Mapping) else ""

        ports_changed = tuple(ports) != self._ports
        if ports_changed:
            self._ports = tuple(ports)
        if ports_changed or plan_text != self._plan_text:
            self._plan_text = plan_text
            self._rebuild_rows(plan_text)
        self.add_button.setEnabled(bool(self._ports))
        self._refresh_summary()
        self._values_text = values_text
        self._rebuild_values(sequence)

    # ---------------------------------------------------------- API values

    def _scanned_parameters(self) -> set[str]:
        plan = self._current_plan()
        if plan is None:
            return set()
        return {
            axis.port[len(PULSE_PARAM_FAMILY):]
            for axis in plan.axes
            if axis.port.startswith(PULSE_PARAM_FAMILY)
        }

    def _rebuild_values(self, sequence: object) -> None:
        """One box per API slot this run sets but does not sweep."""

        parameters = tuple(getattr(sequence, "api_parameters", ()) or ())
        scanned = self._scanned_parameters()
        offered = tuple(
            parameter
            for parameter in parameters
            if parameter.parameter_id not in scanned
        )
        self._sequence = sequence
        broken = ""
        try:
            self._authored = authored_api_entries(sequence) if parameters else {}
        except ValueError as error:
            # A pulse saved with a binding whose field was later cleared.  Say
            # so, rather than leaving a Qt slot on a raise.
            self._authored = {}
            offered = ()
            broken = str(error)
        overrides = self._current_overrides()

        self._loading = True
        try:
            while self.values_grid.count():
                item = self.values_grid.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
            self._value_rows = {}
            for row, parameter in enumerate(offered):
                name = parameter.parameter_id
                authored, unit = self._authored[name]
                label = QtWidgets.QLabel(field_label(sequence, parameter.field_ref))
                label.setToolTip(name)
                box = FluentDoubleSpinBox()
                box.setDecimals(0 if unit == "value" else 6)
                box.setRange(-1e12, 1e12)
                box.setValue(float(overrides.get(name, authored)))
                box.valueChanged.connect(self._emit_values)
                self.values_grid.addWidget(label, row, 0)
                self.values_grid.addWidget(box, row, 1)
                self.values_grid.addWidget(QtWidgets.QLabel(unit), row, 2)
                self._value_rows[name] = box
        finally:
            self._loading = False
        shown = bool(offered) or bool(broken)
        self.values_title.setVisible(shown)
        self.values_note.setVisible(shown)
        self.load_values_button.setEnabled(bool(offered))
        if broken:
            self.values_note.setText(broken)
            return
        self._refresh_values_note(scanned & set(self._authored))

    def _current_overrides(self) -> dict[str, float]:
        try:
            return api_overrides_from_authored(self._values_text)
        except ValueError:
            return {}

    def _emit_values(self) -> None:
        """Only what this run sets differently is written down.

        A box left at the value the pulse carries is not an override: the
        pulse's own number is already the workspace's current one, and
        freezing a copy of it here would quietly outrank the next
        recalibration.
        """

        if self._loading:
            return
        overrides = {
            name: box.value()
            for name, box in self._value_rows.items()
            if box.value() != self._authored[name][0]
        }
        self._values_text = api_overrides_to_authored(overrides)
        self.draft_changed.emit({"values": {"api_values": self._values_text}})
        self._refresh_values_note(set())

    def _refresh_values_note(self, scanned: set[str]) -> None:
        parts = []
        overridden = sum(
            1
            for name, box in self._value_rows.items()
            if box.value() != self._authored[name][0]
        )
        if self._value_rows:
            parts.append(
                f"{overridden} of {len(self._value_rows)} set for this run; "
                "the rest run what the pulse carries."
            )
        if scanned:
            parts.append(f"swept by the plan: {', '.join(sorted(scanned))}.")
        differing = self._saved_set_differences()
        if differing:
            # Nothing is applied behind the operator's back; this only says
            # that pressing Load would change something, so a recalibration
            # nobody pulled in is visible rather than silently absent.
            parts.append(
                f"{CURRENT_API_VALUES} differs on: {', '.join(sorted(differing))}."
            )
        self.values_note.setText("  ".join(parts))

    def _saved_set(self) -> dict[str, float] | None:
        """The workspace's current set, read in each parameter's own unit."""

        if self._sequence is None or self._pulse_path is None:
            return None
        try:
            path = current_api_values_path(self._pulse_path)
            if not path.is_file():
                return None
            _name, _source, entries = read_api_values(path)
            if not entries:
                return None
            applied, _ids, _absent = apply_api_values(self._sequence, entries)
        except (OSError, ValueError, TypeError):
            return None
        return {
            name: value for name, (value, _unit) in authored_api_entries(applied).items()
        }

    def _saved_set_differences(self) -> set[str]:
        saved = self._saved_set()
        if saved is None:
            return set()
        return {
            name
            for name, box in self._value_rows.items()
            if name in saved and saved[name] != box.value()
        }

    def _load_values(self) -> None:
        """Fill the boxes from one saved set of values.

        The set is a place to pull numbers from.  What runs is what these
        boxes say, so a run is reproducible from this node alone.
        """

        if not self._value_rows or self._sequence is None:
            return
        start = ""
        if self._pulse_path is not None:
            start = str(current_api_values_path(self._pulse_path))
        chosen, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load API values", start, "ZLC API values (*.json);;All files (*)"
        )
        if not chosen:
            return
        try:
            _name, _source, entries = read_api_values(chosen)
            applied, _ids, absent = apply_api_values(self._sequence, entries)
        except (OSError, ValueError, TypeError) as error:
            self.values_note.setText(f"cannot load that set: {error}")
            return
        loaded = authored_api_entries(applied)
        self._loading = True
        try:
            for name, box in self._value_rows.items():
                if name in loaded:
                    box.setValue(loaded[name][0])
        finally:
            self._loading = False
        self._emit_values()
        if absent:
            self.values_note.setText(
                self.values_note.text()
                + f"  Not in this pulse: {', '.join(sorted(absent))}."
            )

    def set_mutation_enabled(self, enabled: bool) -> None:
        self.setEnabled(bool(enabled))

    # ------------------------------------------------------------ internals

    def _rebuild_rows(self, plan_text: str) -> None:
        self._loading = True
        try:
            for row in self._rows:
                row.setParent(None)
                row.deleteLater()
            self._rows = []
            axes: tuple[ScanAxis, ...] = ()
            if plan_text.strip():
                try:
                    axes = ScanPlan.from_tree(json.loads(plan_text)).axes
                except (ValueError, TypeError, json.JSONDecodeError):
                    axes = ()
            for axis in axes:
                if axis.port.startswith(MANUAL_PARAM_FAMILY):
                    self._attach_manual_row(axis)
                else:
                    self._attach_row(axis)
            if self._only_port is not None and not self._rows and self._ports:
                self._attach_row(None)
        finally:
            self._loading = False

    def _attach_row(self, axis: ScanAxis | None) -> None:
        row = _AxisRow(self._ports, axis, self)
        if self._only_port is not None:
            row.remove_button.hide()
        row.edited.connect(self._emit_plan)
        row.remove_requested.connect(self._remove_row)
        self._rows.append(row)
        self.rows_layout.addWidget(row)

    def _default_manual_name(self) -> str:
        """A name the operator can run with, and change if they like."""

        taken = {
            row.name_edit.text().strip() for row in self._rows if row.manual
        }
        ordinal = 1
        while f"manual {ordinal}" in taken:
            ordinal += 1
        return f"manual {ordinal}"

    def _attach_manual_row(self, axis: ScanAxis | None) -> None:
        row = _ManualAxisRow(axis, self._default_manual_name(), self)
        row.edited.connect(self._emit_plan)
        row.remove_requested.connect(self._remove_row)
        # Above every machine row, because that is where it runs: the
        # displayed order IS the nesting order, and a manual axis nested
        # inside a fired table is not a thing the bench can do.
        at = sum(1 for existing in self._rows if existing.manual)
        self._rows.insert(at, row)
        self.rows_layout.insertWidget(at, row)

    def _add_axis(self) -> None:
        if not self._ports:
            return
        self._attach_row(None)
        self._emit_plan()

    def _add_manual_axis(self) -> None:
        self._attach_manual_row(None)
        self._emit_plan()

    def _remove_row(self, row) -> None:
        if row in self._rows:
            self._rows.remove(row)
            row.setParent(None)
            row.deleteLater()
        self._emit_plan()

    def _current_plan(self) -> ScanPlan | None:
        try:
            return ScanPlan(tuple(row.axis() for row in self._rows))
        except (ValueError, TypeError):
            return None

    def _emit_plan(self) -> None:
        if self._loading:
            return
        plan = self._current_plan()
        self._plan_text = "" if plan is None else json.dumps(plan.to_tree())
        # The host's draft contract: a patch under "values", the same shape
        # the auto-generated form emits.
        self.draft_changed.emit({"values": {"plan": self._plan_text}})
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        plan = self._current_plan()
        if plan is None:
            self.summary.setText(
                "No axes yet.  Each row is one axis, outermost first; "
                "every point plays the pulse and captures one measurement."
            )
            return
        shape = " × ".join(str(n) for n in plan.shape)
        # The ordering law lives in split_outer_axes; saying its refusal HERE
        # is what lets the operator fix the order while the rows are still in
        # front of them, instead of at Start.  Nothing is silently reordered.
        try:
            from .plan import split_outer_axes

            split_outer_axes(plan)
        except ValueError as refusal:
            self.summary.setText(str(refusal))
            return
        manual = tuple(
            axis for axis in plan.axes
            if axis.port.startswith(MANUAL_PARAM_FAMILY)
        )
        device = tuple(
            axis for axis in plan.axes
            if axis.port.startswith(DEVICE_PARAM_FAMILY)
        )
        stops = 1
        for axis in manual:
            stops *= len(axis.values)
        by_hand = (
            ""
            if not manual
            else (
                f"  {stops} of those points are reached by hand: the run "
                "stops at each one and waits for you to set it."
            )
        )
        by_call = 1
        for axis in device:
            by_call *= len(axis.values)
        by_device = (
            ""
            if not device
            else (
                f"  {by_call} device settings are applied and read back "
                "between fires."
            )
        )
        self.summary.setText(
            f"{len(plan.axes)} axis(es), {shape} = {plan.point_count} points, "
            "outermost first; each point resolves the template, plays it, and "
            f"captures one measurement.{by_hand}{by_device}"
        )


def scan_plan_editor_factory(
    parent=None,
    *,
    device_ports: bool = True,
    only_port: str | None = None,
    hardware_slots: bool = False,
    manual_axes: bool = False,
) -> ScanPlanEditor:
    return ScanPlanEditor(
        parent,
        device_ports=device_ports,
        only_port=only_port,
        hardware_slots=hardware_slots,
        manual_axes=manual_axes,
    )
