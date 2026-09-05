"""The scan plan editor, authored as rows rather than as JSON.

The manual row is the one that carries no port -- everything else about it
is an ordinary axis -- so the only way to know it authors the right
document is to build the widget, press the button, and read what it
emitted.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from zlc_atom.nodes.scan import ScanPlan, manual_axis_name
from zlc_atom.nodes.scan.editor import scan_plan_editor_factory
from zlc_atom.nodes.scan.plan import ScanPort
from zlc_ui import ensure_qt_app


BIAS = ScanPort("pulse:param:da_bias_x", "da_bias_x", "V", -1.0, 1.0, -0.5, 0.5)


def _editor(manual_axes: bool = True):
    ensure_qt_app()
    editor = scan_plan_editor_factory(
        device_ports=False, hardware_slots=True, manual_axes=manual_axes
    )
    # The projection is what a node hands its editor; only the ports and the
    # authored plan matter here, so the rest of it stays out of the way.
    editor._ports = (BIAS,)
    editor.add_button.setEnabled(True)
    return editor


def _plan(editor) -> ScanPlan:
    return ScanPlan.from_tree(json.loads(editor._plan_text))


def test_a_manual_row_authors_a_name_and_the_values_a_hand_will_set() -> None:
    editor = _editor()
    try:
        editor._add_axis()
        editor._add_manual_axis()
        row = next(row for row in editor._rows if row.manual)
        row.name_edit.setText("power")
        row.start_spin.setValue(1.0)
        row.stop_spin.setValue(4.0)
        row.points_spin.setValue(4)

        plan = _plan(editor)
        assert [axis.port for axis in plan.axes] == [
            "manual:power",
            "pulse:param:da_bias_x",
        ], "a manual axis is authored OUTSIDE the axes a machine advances"
        manual = plan.axes[0]
        assert manual_axis_name(manual.port) == "power"
        # The same from/to/points every other row authors: a coordinate has
        # to exist before the data it describes, whoever turns the knob.
        assert manual.values == (1.0, 2.0, 3.0, 4.0)
    finally:
        editor.deleteLater()


def test_a_fresh_manual_row_is_already_runnable() -> None:
    """An unnamed axis is not a plan, so the button never authors one."""

    editor = _editor()
    try:
        editor._add_axis()
        editor._add_manual_axis()
        editor._add_manual_axis()
        names = [
            manual_axis_name(axis.port)
            for axis in _plan(editor).axes
            if axis.port.startswith("manual:")
        ]
        assert names == ["manual 1", "manual 2"], (
            "a default name, and never the same one twice"
        )
    finally:
        editor.deleteLater()


def test_a_manual_row_cannot_be_authored_inside_the_machine_axes() -> None:
    """Display order IS nesting order, so the row goes where it runs."""

    editor = _editor()
    try:
        editor._add_axis()
        editor._add_axis()
        editor._add_manual_axis()
        editor._add_manual_axis()
        assert [row.manual for row in editor._rows] == [
            True,
            True,
            False,
            False,
        ]
        positions = [
            editor.rows_layout.indexOf(row) for row in editor._rows
        ]
        assert positions == sorted(positions), (
            "the laid-out order must match the authored order"
        )
    finally:
        editor.deleteLater()


def test_a_node_that_cannot_stop_for_a_hand_never_offers_the_button() -> None:
    editor = _editor(manual_axes=False)
    try:
        assert editor.add_manual_button.isHidden()
        assert not editor.add_manual_button.isEnabled()
    finally:
        editor.deleteLater()


def test_the_summary_says_how_many_stops_the_operator_is_signing_up_for() -> None:
    editor = _editor()
    try:
        editor._add_axis()
        editor._add_manual_axis()
        row = next(row for row in editor._rows if row.manual)
        row.name_edit.setText("power")
        row.points_spin.setValue(3)
        editor._refresh_summary()
        assert "3 of those points are reached by hand" in editor.summary.text()
    finally:
        editor.deleteLater()


@pytest.mark.parametrize("points", (1, 7))
def test_a_manual_row_round_trips_through_the_document(points: int) -> None:
    editor = _editor()
    try:
        editor._add_axis()
        editor._add_manual_axis()
        row = next(row for row in editor._rows if row.manual)
        row.name_edit.setText("angle")
        row.points_spin.setValue(points)
        text = editor._plan_text

        reopened = _editor()
        try:
            reopened._reconcile_rows(text)
            restored = next(row for row in reopened._rows if row.manual)
            assert restored.name_edit.text() == "angle"
            assert int(restored.points_spin.value()) == points
        finally:
            reopened.deleteLater()
    finally:
        editor.deleteLater()


def test_a_devices_knobs_hang_under_that_device_not_in_one_flat_list() -> None:
    """The axis chooser is a tree, and its branches are what owns the knobs.

    A flat list put a laser current beside a pulse parameter by accident and
    grew with every device installed, so finding "what can the RF source
    sweep" meant reading the whole thing.  The branch a port hangs under is
    derived from the port itself, beside the label, so where a knob is found
    and what it is called cannot drift apart.
    """

    from zlc_atom.nodes.scan.editor import _AxisRow
    from zlc_atom.nodes.scan.plan import ScanAxis, port_label

    def port(name: str, lo: float, hi: float) -> ScanPort:
        return ScanPort(name, port_label(name), "", lo, hi)

    ports = (
        port("pulse:param:mot_duration", 0.0, 1.0),
        port("device:rf_source:frequency_hz", 1e5, 5e6),
        port("device:rf_source:power_dbm", -30.0, 10.0),
        port("device:slm:tilt_x", -1.0, 1.0),
    )
    row = _AxisRow(ports, ScanAxis("device:rf_source:power_dbm", (0.0, 1.0, 2.0)))
    try:
        model = row.port_combo._model
        tree = {
            model.item(index).text(): [
                model.item(index).child(leaf).text()
                for leaf in range(model.item(index).rowCount())
            ]
            for index in range(model.rowCount())
        }
        assert tree == {
            "pulse": ["mot_duration"],
            "rf_source": ["frequency_hz", "power_dbm"],
            "slm": ["tilt_x"],
        }, tree
        # The authored port is still the selection, and its own limits are
        # what the sweep is bounded by.
        assert row.port_combo.currentData() == "device:rf_source:power_dbm"
        assert (row.start_spin.minimum(), row.start_spin.maximum()) == (-30.0, 10.0)
    finally:
        row.deleteLater()


def test_rebuilding_rows_in_one_pass_never_shows_a_window() -> None:
    """Add axis re-projects the editor in the same pass that built its rows.

    A row added to the visible editor and retired before the loop turns used
    to be shown by the queued show after losing its parent: a window on the
    desktop for one frame, which the operator saw as a flash at Add axis.
    """

    from PyQt5 import QtCore, QtWidgets

    app = ensure_qt_app(["scan-editor-rebuild"])
    editor = _editor()
    editor.show()
    for _ in range(5):
        app.processEvents(QtCore.QEventLoop.AllEvents, 10)

    seen: list[str] = []

    class Filter(QtCore.QObject):
        def eventFilter(self, watched, event):
            if (
                event.type() == QtCore.QEvent.Show
                and isinstance(watched, QtWidgets.QWidget)
                and watched.isWindow()
            ):
                seen.append(type(watched).__name__)
            return False

    holder = Filter(app)
    app.installEventFilter(holder)
    try:
        for _ in range(3):
            editor._add_axis()
            editor._reconcile_rows("")
            editor._add_axis()
            editor._reconcile_rows("")
        for _ in range(30):
            app.processEvents(QtCore.QEventLoop.AllEvents, 10)
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    finally:
        app.removeEventFilter(holder)
        editor.close()
        editor.deleteLater()
    assert seen == [], seen


def _bound_sequence():
    """A pulse with two API slots: a duration in ns and a DAC level."""

    from zlc_pulse import (
        AnalogStep,
        PulseApiParameter,
        PulseFieldRef,
        PulsePeriod,
        PulsePortSpec,
        PulseSequence,
        PulseTarget,
    )

    target = PulseTarget(
        lanes=("d0", "d1", "a0", "a1"),
        ports=(
            PulsePortSpec("d0", "digital", ("d0",)),
            PulsePortSpec("d1", "digital", ("d1",)),
            PulsePortSpec("dac", "dac", ("a0", "a1"), bus_index=0),
        ),
    )
    return PulseSequence(
        name="bound",
        target=target,
        time_step_ns=20,
        periods=(
            PulsePeriod("p0", 200, "ns", (1, 0, 0, 0), (AnalogStep("dac", "edge", 1),)),
            PulsePeriod("p1", 20, "ns", (0, 1, 0, 0)),
        ),
        api_parameters=(
            PulseApiParameter("hold", PulseFieldRef("duration", "p0"), "ns"),
            PulseApiParameter("level", PulseFieldRef("dac", "p0", "dac"), "value"),
        ),
    )


def _projection(sequence, plan: str = "", api_values: str = "") -> dict:
    from types import SimpleNamespace

    return {
        "workspace_resources": {"pulse_template": SimpleNamespace(value=sequence)},
        "form_values": {"plan": plan, "api_values": api_values},
    }


def test_api_values_are_reconciled_under_the_operators_wheel() -> None:
    """A projection follows every draft; the box under the wheel stays.

    The section was rebuilt on every projection, so a wheel turned one
    notch retired the box it was over and built another -- no focus, no
    selection, and the unit the operator was reading it in reset to the
    pulse's own.  The rows are the shared form now: reconciled by key, the
    focused row is theirs, and a unit re-declared is not a change.
    """

    from PyQt5 import QtCore

    app = ensure_qt_app(["scan-editor-values"])
    sequence = _bound_sequence()
    editor = scan_plan_editor_factory(device_ports=False, hardware_slots=False)
    editor.show()
    editor.update_projection(_projection(sequence))
    app.processEvents()
    form = editor.values_form
    assert form.keys == ("hold", "level")
    hold = form.widget_for("hold")
    level = form.widget_for("level")
    assert level.value() == 1 and type(level.value()) is int, "a DAC level is whole codes"
    assert hold.value() == 200.0 and hold.valueUnit() == "ns"
    assert form.unit_picker_for("hold") is not None, "a duration has a ladder"
    assert form.unit_picker_for("level") is None, "a code has none"
    assert hold.maximum() > 200.0 and hold.minimum() > 0.0, "the board's limit, not a guess"

    # Read the duration in microseconds, then turn the wheel on it.
    form._shown_unit_picked("hold", "µs")
    assert hold.text() == "0.2"
    hold.setFocus(QtCore.Qt.MouseFocusReason)
    app.processEvents()
    drafts: list[dict] = []
    editor.draft_changed.connect(drafts.append)
    hold.stepBy(1)
    assert drafts, "a notch is a draft, now, not after a timer"
    text = drafts[-1]["values"]["api_values"]
    assert text.startswith("hold = ") and "level" not in text, text
    # The host projects the draft straight back, and then again with no
    # draft at all -- a beat -- and the row is the same widget both times,
    # read in the unit the operator chose.
    for _ in range(2):
        editor.update_projection(_projection(sequence, api_values=text))
        app.processEvents()
        assert form.widget_for("hold") is hold
        assert form.unit_picker_for("hold") is not None
        assert hold.shownUnit() == "µs", hold.shownUnit()
    assert "1 of 2 set for this run" in editor.values_note.text()
    editor.close()
    editor.deleteLater()


def test_axis_rows_follow_the_ports_without_being_rebuilt() -> None:
    """The bench re-projects its ports every time a reading moves; the row
    the operator is inside is kept and re-pointed, never replaced."""

    from PyQt5 import QtCore

    app = ensure_qt_app(["scan-editor-rows"])
    editor = _editor()
    editor.show()
    editor._add_axis()
    row = editor._rows[0]
    row.start_spin.setFocus(QtCore.Qt.MouseFocusReason)
    app.processEvents()
    row.start_spin.setValue(-0.25)
    plan = editor._plan_text
    # Wider limits from the bench: the same row, a new range, the value
    # under the cursor untouched.
    wider = ScanPort(BIAS.port, BIAS.label, BIAS.unit, -2.0, 2.0, -0.5, 0.5)
    editor._ports = (wider,)
    editor._reconcile_rows(plan)
    assert editor._rows[0] is row
    assert row.start_spin.minimum() == -2.0
    assert row.start_spin.value() == -0.25
    # A plan with one more axis adds a row and keeps the first.
    editor._add_axis()
    assert editor._rows[0] is row and len(editor._rows) == 2
    # And a shorter plan retires only the row past its end.
    editor._reconcile_rows(plan)
    assert editor._rows == [row]
    editor.close()
    editor.deleteLater()
