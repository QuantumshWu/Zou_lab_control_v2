"""The scan plan, as a table of axes instead of a JSON box.

One row per axis: which port, from, to, how many points.  Row order is the
nesting order, outermost first, which is how the plan itself reads.  The
editor owns exactly one authored field -- ``plan`` -- and says so through
``managed_fields``, so the auto-generated form does not render the raw JSON
beside it.

Ports are read from the projection -- the resolved pulse template's API
parameters, plus the bench's tunable devices for a node that can move them --
the same set that node's binding enforces.  A board-advanced scan cannot make
a host call between two cycles of one table, so its editor never offers a
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
    FluentSpinBox,
)

from .plan import ScanAxis, ScanPlan, scan_ports_for, scan_ports_for_devices


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
            if at >= 0:
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
                self.start_spin.setValue(port.lo)
                self.stop_spin.setValue(port.hi)
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
            spin.setDecimals(4)

    def _port_changed(self, _index: int) -> None:
        self._custom_values = None
        self.custom_label.setText("")
        self._apply_port_limits()
        self.edited.emit()

    def _spins_edited(self) -> None:
        # An edited spin means the operator wants the uniform grid the spins
        # describe; a custom list survives only while it is left alone.
        self._custom_values = None
        self.custom_label.setText("")
        self.edited.emit()

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


class ScanPlanEditor(QtWidgets.QWidget):
    """The ``plan`` field, authored as axes rather than typed as JSON."""

    draft_changed = QtCore.pyqtSignal(object)
    managed_fields = ("plan",)

    def __init__(self, parent=None, *, device_ports: bool = True) -> None:
        super().__init__(parent)
        self._device_ports = bool(device_ports)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Scan plan")
        title.setStyleSheet("font-weight: 600;")
        self.add_button = FluentButton("Add axis", color=ACCENT)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.add_button)
        column.addLayout(header)
        self.rows_layout = QtWidgets.QVBoxLayout()
        column.addLayout(self.rows_layout)
        self.summary = QtWidgets.QLabel("")
        self.summary.setWordWrap(True)
        column.addWidget(self.summary)

        self._ports: tuple = ()
        self._rows: list[_AxisRow] = []
        self._loading = False
        self._plan_text = ""
        self.add_button.clicked.connect(self._add_axis)

    # ------------------------------------------------------- host contract

    def update_projection(self, projection: Mapping[str, object]) -> None:
        resources = projection.get("workspace_resources") or {}
        resource = resources.get("pulse_template") if isinstance(resources, Mapping) else None
        sequence = getattr(resource, "value", None)
        # The editor offers exactly what this node's build binds against: the
        # pulse's parameters, and the bench's tunable devices only where a
        # host is there to move them.
        extras = projection.get("bench_extras") or {}
        tunables = extras.get("tunable_devices") if isinstance(extras, Mapping) else None
        ports = (
            scan_ports_for(sequence) if sequence is not None else ()
        ) + (scan_ports_for_devices(tunables) if self._device_ports else ())
        values = projection.get("form_values") or {}
        plan_text = str(values.get("plan") or "") if isinstance(values, Mapping) else ""

        ports_changed = tuple(ports) != self._ports
        if ports_changed:
            self._ports = tuple(ports)
        if ports_changed or plan_text != self._plan_text:
            self._plan_text = plan_text
            self._rebuild_rows(plan_text)
        self.add_button.setEnabled(bool(self._ports))
        self._refresh_summary()

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
                self._attach_row(axis)
        finally:
            self._loading = False

    def _attach_row(self, axis: ScanAxis | None) -> None:
        row = _AxisRow(self._ports, axis, self)
        row.edited.connect(self._emit_plan)
        row.remove_requested.connect(self._remove_row)
        self._rows.append(row)
        self.rows_layout.addWidget(row)

    def _add_axis(self) -> None:
        if not self._ports:
            return
        self._attach_row(None)
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
                "every point plays the pulse and captures the watched signal."
            )
            return
        shape = " × ".join(str(n) for n in plan.shape)
        self.summary.setText(
            f"{len(plan.axes)} axis(es), {shape} = {plan.point_count} points, "
            "outermost first; each point resolves the template, plays it, and "
            "captures the watched signal."
        )


def scan_plan_editor_factory(parent=None, *, device_ports: bool = True) -> ScanPlanEditor:
    return ScanPlanEditor(parent, device_ports=device_ports)
