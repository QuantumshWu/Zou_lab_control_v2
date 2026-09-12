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

from zlc_atom.nodes.scan import ScanAxis, ScanPlan, manual_axis_name
from zlc_atom.nodes.scan.editor import scan_plan_editor_factory
from zlc_atom.nodes.scan.plan import ScanPort, parse_scan_values, plan_input_rows
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


def test_an_authored_grid_the_spins_cannot_regenerate_is_kept_exactly() -> None:
    """An explicit list of values is the operator's until a spin is edited.

    Uniformity was judged with a relative tolerance, so 1 000 000,
    2 000 000, 3 000 005 -- a grid a notebook authored on purpose -- was
    called uniform, its list dropped, and the row re-authored the middle
    point as 2 000 002.5 the next time anything on the form changed.
    """

    editor = _editor()
    try:
        authored = (1_000_000.0, 2_000_000.0, 3_000_005.0)
        wide = ScanPort(BIAS.port, BIAS.label, BIAS.unit, 0.0, 1e7, 0.0, 1e7)
        editor._ports = (wide,)
        editor._reconcile_rows(
            json.dumps(ScanPlan((ScanAxis(BIAS.port, authored),)).to_tree())
        )
        (row,) = editor._rows
        assert row.custom_label.text() == "custom values"
        assert row.axis().values == authored
        editor._emit_plan()
        assert _plan(editor).axes[0].values == authored, "re-emitted as authored"
        # A grid the spins DO describe is not custom, and an edited spin
        # replaces a custom list with the grid the spins now describe.
        editor._reconcile_rows(
            json.dumps(ScanPlan((ScanAxis(BIAS.port, (0.0, 0.5, 1.0)),)).to_tree())
        )
        assert row.custom_label.text() == ""
        editor._reconcile_rows(
            json.dumps(ScanPlan((ScanAxis(BIAS.port, authored),)).to_tree())
        )
        row.points_spin.setValue(2)
        assert row.axis().values == (1_000_000.0, 3_000_005.0)
    finally:
        editor.deleteLater()


def test_host_only_scan_plans_need_no_dummy_board_axis_in_either_editor() -> None:
    """Both forms describe a device-only plan using its real coordinates."""

    ensure_qt_app()
    device = ScanPort("device:rf:frequency", "rf.frequency", "Hz", 1e5, 5e6)
    stepped = scan_plan_editor_factory(device_ports=True, hardware_slots=False)
    seamless = scan_plan_editor_factory(
        device_ports=True, hardware_slots=True, manual_axes=True
    )
    try:
        plan = json.dumps(ScanPlan((ScanAxis(device.port, (1e6, 2e6)),)).to_tree())
        for editor in (stepped, seamless):
            editor._ports = (device,)
            editor._reconcile_rows(plan)
            editor._refresh_summary()
            assert "2 device settings are applied" in editor.summary.text()
            assert len(editor._current_plan().axes) == 1
    finally:
        stepped.deleteLater()
        seamless.deleteLater()


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
        range_values = row.input_entry()["values"]
        widgets = (row.start_spin, row.stop_spin, row.points_spin, row.values_edit)
        assert row.input_entry()["value_text"] == ""
        row.mode_button.click()
        assert row.input_stack.currentIndex() == 1
        assert row.range_inputs.isHidden() and not row.points_spin.isVisibleTo(row)
        assert row.input_entry()["mode"] == "values"
        assert row.input_entry()["values"] == range_values
        with pytest.raises(ValueError, match="Values is empty"):
            editor._current_plan()
        text = editor._plan_text

        reopened = _editor()
        try:
            reopened._reconcile_rows(text)
            restored = next(row for row in reopened._rows if row.manual)
            assert restored.name_edit.text() == "angle"
            assert int(restored.points_spin.value()) == points
            assert restored.input_entry() == row.input_entry()
            assert restored.values_edit.text() == "" and restored.input_stack.currentIndex() == 1
            restored.values_edit.setText("9, 3, 9")
            restored.values_edit.editingFinished.emit()
            assert reopened._current_plan().axes[0].values == (9.0, 3.0, 9.0)
            assert restored.input_entry()["values"] == range_values
            restored_widgets = (restored.start_spin, restored.stop_spin, restored.points_spin, restored.values_edit)
            restored.mode_button.click()
            assert restored.input_stack.currentIndex() == 0
            assert reopened._current_plan().axes[0].values == tuple(range_values)
            assert restored.values_edit.text() == "9, 3, 9"
            restored.mode_button.click()
            from zlc_runtime import SelectionRange, SelectionState
            from zlc_atom.nodes.seamless_scan import LOGIC_NODE

            selection = SelectionState("curve", "x_range", (SelectionRange("scan.angle", 10., 16., domain="point"),))
            patch = LOGIC_NODE.selection_patch(selection, draft={"plan": reopened._plan_text},
                context={"axis_units": {"scan.angle": "1"}})
            if points > 1:
                assert patch is not None
                untouched = plan_input_rows(reopened._plan_text)[1]
                reopened._reconcile_rows(patch["plan"])
                assert plan_input_rows(patch["plan"])[1] == untouched
                assert restored.start_spin.value() == 10. and restored.stop_spin.value() == 16.
                assert len(restored.input_entry()["values"]) == points
            else:
                assert patch is None
            assert restored.input_entry()["mode"] == "values"
            assert restored.values_edit.text() == "9, 3, 9"
            assert reopened._current_plan().axes[0].values == (9., 3., 9.)
            assert (restored.start_spin, restored.stop_spin, restored.points_spin, restored.values_edit) == restored_widgets
            assert (row.start_spin, row.stop_spin, row.points_spin, row.values_edit) == widgets
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

    ensure_qt_app()

    def port(name: str, lo: float, hi: float) -> ScanPort:
        return ScanPort(name, port_label(name), "", lo, hi)

    ports = (
        ScanPort("pulse:param:mot_duration", "MOT.duration", "", 0.0, 1.0),
        port("device:rf_source:frequency", 1e5, 5e6),
        port("device:rf_source:power", -30.0, 10.0),
        port("device:slm:tilt_x", -1.0, 1.0),
    )
    row = _AxisRow(
        ports,
        plan_input_rows(ScanPlan((ScanAxis("device:rf_source:power", (0.0, 1.0, 2.0)),)))[0],
        device_labels={"rf_source": "Cooling RF", "slm": "Tweezers"},
    )
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
            "pulse": ["MOT.duration"],
            "Cooling RF": ["frequency", "power"],
            "Tweezers": ["tilt_x"],
        }, tree
        # The authored port is still the selection, and its own limits are
        # what the sweep is bounded by.
        assert row.port_combo.currentData() == "device:rf_source:power"
        assert (row.start_spin.minimum(), row.start_spin.maximum()) == (-30.0, 10.0)
        row.set_device_labels({"rf_source": "Probe RF", "slm": "Tweezers"})
        assert "Probe RF" in {
            row.port_combo._model.item(index).text()
            for index in range(row.port_combo._model.rowCount())
        }
        assert row.port_combo.currentData() == "device:rf_source:power"
        assert row.axis().values == (0.0, 1.0, 2.0)
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
    from dataclasses import replace

    app = ensure_qt_app(["scan-editor-values"])
    sequence = _bound_sequence()
    sequence = replace(sequence, periods=(replace(sequence.periods[0], name="MOT"), *sequence.periods[1:]))
    editor = scan_plan_editor_factory(device_ports=False, hardware_slots=False)
    editor.show()
    editor.update_projection(_projection(sequence))
    app.processEvents()
    form = editor.values_form
    assert form.keys == ("hold", "level")
    assert next(field.label for field in form.spec.fields if field.key == "hold") == "MOT.duration"
    hold = form.widget_for("hold")
    level = form.widget_for("level")
    assert form.read_value("level") == 1 and type(form.read_value("level")) is int, "a DAC level is whole codes"
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


def test_axis_rows_follow_the_ports_without_being_rebuilt(caplog) -> None:
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

    from zlc_data.units import DEFAULT_UNITS, Unit, UnitRegistry, VoltageIntoLoad
    from zlc_atom.authoring import AuthoringField, TunableField
    from zlc_workbench.board import attach_qt_worker
    from types import SimpleNamespace
    from threading import Event, get_ident
    import time
    import numpy as np

    power = ScanPort("device:rf:ch1_power", "rf.ch1_power", "dBm",
                     -30.0, 10.0, -20.0, 0.0)
    units = UnitRegistry((DEFAULT_UNITS.resolve("dBm"),
                          Unit("Vpp", "power", VoltageIntoLoad(100.0), prefixable=True)))
    owner_thread = get_ident()
    gate = Event()
    reads = []

    def read(unit="dBm"):
        reads.append(unit)
        assert get_ident() != owner_thread, "device read ran on Qt"
        assert gate.wait(3.0)
        unit = unit or "dBm"
        low, high = units.convert((-30.0, 10.0), "dBm", unit)
        return TunableField(AuthoringField("ch1_power", "float", "Power", None,
                                          minimum=float(low), maximum=float(high), unit=unit),
                            float(units.convert(0.0, "dBm", unit)), True, ("ch1_power",))

    def convert(name, values, source, target):
        assert get_ident() != owner_thread, "device conversion ran on Qt"
        assert gate.wait(3.0)
        assert name == "ch1_power"
        return tuple(units.convert(values, source, target))

    device = SimpleNamespace(tunable_fields=lambda: (read(),),
                             read_tunable_in_unit=lambda name, unit: read(unit),
                             convert_tunable_value=convert)
    run, close_worker = attach_qt_worker("scan-editor-test-read")

    def settled(predicate):
        deadline = time.monotonic() + 3.0
        while not predicate():
            assert time.monotonic() < deadline, "scan editor worker did not settle"
            app.processEvents(QtCore.QEventLoop.AllEvents, 10)
            time.sleep(0.001)

    editor = scan_plan_editor_factory(device_ports=True)
    reopened = scan_plan_editor_factory(device_ports=True)
    initial = ScanAxis(power.port, (-20.0, -15.0, -10.0, -5.0, 0.0), "dBm")
    projection = {"form_values": {"plan": json.dumps(ScanPlan((initial,)).to_tree())},
                  "bench_extras": {"tunable_devices": {"rf": device}}, "run_device_read": run}
    try:
        editor.update_projection(projection)
        # Fresh dicts of the same input must not starve an in-flight read.
        editor.update_projection(dict(projection))
        assert not editor._rows
        gate.set()
        settled(lambda: bool(editor._rows))
        row = editor._rows[0]
        before = len(reads)
        projection["device_labels"] = {"rf": "Cooling RF"}
        editor.update_projection(projection)
        app.processEvents()
        assert len(reads) == before and not editor._port_read_pending
        assert editor._ports[0].label == "Cooling RF.ch1_power"
        root = row.port_combo._model.item(0)
        assert root.text() == "Cooling RF" and root.child(0).text() == "ch1_power"
        assert row.port_combo.currentData() == "device:rf:ch1_power"
        row.values_edit.setText("-17, -13, -17")
        row.values_edit.editingFinished.emit()
        old_values = row.axis().values
        row.unit_picker.unit_picked.emit("mVpp")
        assert row.axis().unit == "dBm", "unit changed before worker accepted it"
        settled(lambda: row._unit_request is None)
        assert tuple(units.convert(row.axis().values, "mVpp", "dBm")) == pytest.approx(old_values)
        assert row.axis().values[-1] == pytest.approx(894.4271909999159), "used the global 50-ohm conversion"
        assert row.start_spin.valueUnit() == "mVpp"
        converted_values = parse_scan_values(row.values_edit.text())
        assert converted_values == tuple(units.convert((-17., -13., -17.), "dBm", "mVpp"))
        assert len(converted_values) == 3 and len(row.input_entry()["values"]) == 5
        row.start_spin.setValue(135.0)
        row.stop_spin.setValue(247.0)
        row.points_spin.setValue(10)
        plan = _plan(editor)
        assert plan.axes[0].unit == "mVpp"
        assert plan.axes[0].values == tuple(np.linspace(135.0, 247.0, 10))
        assert parse_scan_values(row.values_edit.text()) == converted_values
        assert ScanPlan.from_tree(plan.to_tree()) == plan
        reopened.update_projection({**projection, "form_values": {"plan": editor._plan_text}})
        settled(lambda: bool(reopened._rows))
        restored = reopened._rows[0]
        assert restored.start_spin.shownUnit() == "mVpp"
        assert restored.unit_picker.current_choice_key() == "mVpp"
        assert restored.axis().values == plan.axes[0].values
        assert restored.axis().unit == "mVpp"
        assert restored.start_spin.value() == 135.0 and restored.stop_spin.value() == 247.0
        assert parse_scan_values(restored.values_edit.text()) == converted_values
        restored.unit_picker.unit_picked.emit("dBm")
        settled(lambda: restored._unit_request is None)
        assert restored.axis().unit == "dBm"
        assert restored.axis().values == tuple(units.convert(plan.axes[0].values, "mVpp", "dBm"))
        assert restored.custom_label.text() == "custom values"
        assert parse_scan_values(restored.values_edit.text()) == tuple(units.convert(converted_values, "mVpp", "dBm"))
        restored.mode_button.click()
        gate.clear()
        restored.unit_picker.unit_picked.emit("mVpp")
        from PyQt5 import QtTest
        QtTest.QTest.keyClick(restored.values_edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
        QtTest.QTest.keyClicks(restored.values_edit, "-19, -17")
        QtTest.QTest.keyClick(restored.values_edit, QtCore.Qt.Key_Return)
        edited_entry = restored.input_entry()
        gate.set()
        settled(lambda: restored._unit_request is None)
        assert restored.input_entry() == edited_entry, "late conversion overwrote a new draft"
        assert restored.custom_label.text() == "", "discarded conversion still says it is running"
        # The port can change while its unit read is in flight. A code-valued
        # port retires the picker, but not the row's stable unit cell.
        dac = ScanPort("pulse:param:dac", "dac", "code", -100., 100., 0., 1.)
        reopened._ports = (*reopened._ports, dac)
        reopened._reconcile_rows(reopened._plan_text)
        gate.clear()
        restored.unit_picker.unit_picked.emit("mVpp")
        reopened.show()
        restored.port_combo.showPopup()
        tree = restored.port_combo._popup_view
        tree.expandAll()
        app.processEvents()
        matches = restored.port_combo._model.match(restored.port_combo._model.index(0, 0),
            QtCore.Qt.UserRole, dac.port, 1, QtCore.Qt.MatchExactly | QtCore.Qt.MatchRecursive)
        assert len(matches) == 1
        tree.scrollTo(matches[0]); app.processEvents()
        QtTest.QTest.mouseClick(tree.viewport(), QtCore.Qt.LeftButton,
                               pos=tree.visualRect(matches[0]).center())
        assert restored.port_combo.currentData() == dac.port and restored.unit_picker is None
        port_entry = restored.input_entry()
        gate.set()
        settled(lambda: restored._unit_request is None)
        assert restored.input_entry() == port_entry and restored._unit_host.isEnabled()
        reopened._reconcile_rows(json.dumps({"axes": [edited_entry]}))
        gate.clear()
        restored.unit_picker.unit_picked.emit("mVpp")
        reopened._remove_row(restored)
        reopened.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        gate.set()
        settled(close_worker)
    finally:
        gate.set()
        settled(close_worker)
        editor.close()
        editor.deleteLater()
        from PyQt5 import sip
        if not sip.isdeleted(reopened):
            reopened.close()
            reopened.deleteLater()
    assert not [record for record in caplog.records if record.levelno >= 40]


def _plain_sequence():
    """The same pulse with no API parameters at all."""

    from dataclasses import replace

    return replace(_bound_sequence(), api_parameters=())


def _level_only_sequence():
    from dataclasses import replace

    bound = _bound_sequence()
    return replace(bound, api_parameters=bound.api_parameters[1:])


def test_api_values_are_rescoped_to_the_pulse_they_are_written_against() -> None:
    """A value written against one pulse does not follow the node to the next.

    The draft kept the previous pulse's ``hold = 250`` when a pulse with no
    API parameters replaced it; the form had no row to show that line in,
    and Start refused it by name -- a refusal the operator could neither
    see the cause of nor undo.  The editor owns the re-scoping: it writes
    the text back only when the pulse's vocabulary changes it, so one round
    trip settles it; a box mid-word, a missing pulse and a frozen editor
    decide nothing.
    """

    from PyQt5 import QtCore

    app = ensure_qt_app(["scan-editor-rescope"])
    editor = scan_plan_editor_factory(device_ports=False, hardware_slots=False)
    editor.show()
    drafts: list[dict] = []
    editor.draft_changed.connect(drafts.append)
    bound = _bound_sequence()

    # Written against the pulse that declares it: nothing to re-scope.
    editor.update_projection(_projection(bound, api_values="hold = 250"))
    assert drafts == []
    assert editor._values_text == "hold = 250"

    # A pulse with no API parameters: the line is dropped, once.
    editor.update_projection(_projection(_plain_sequence(), api_values="hold = 250"))
    assert drafts == [{"values": {"api_values": ""}}]
    assert editor._values_text == ""
    editor.update_projection(_projection(_plain_sequence(), api_values=""))
    assert len(drafts) == 1, "the re-scoped text projects back and settles"

    # A pulse declaring other names keeps only what it declares.
    drafts.clear()
    editor.update_projection(
        _projection(_level_only_sequence(), api_values="hold = 250\nlevel = 1")
    )
    assert drafts == [{"values": {"api_values": ""}}], drafts
    drafts.clear()
    editor.update_projection(
        _projection(_level_only_sequence(), api_values="hold = 250\nlevel = -1")
    )
    assert drafts == [{"values": {"api_values": "level = -1"}}], drafts

    # No pulse at all decides nothing: the text waits for one.
    drafts.clear()
    editor.update_projection({
        "workspace_resources": {},
        "form_values": {"plan": "", "api_values": "hold = 250"},
    })
    assert drafts == []
    assert editor._values_text == "hold = 250"

    # A box mid-word is the operator's: the text is not re-read over it.
    editor.update_projection(_projection(bound, api_values="hold = 250"))
    hold = editor.values_form.widget_for("hold")
    hold.setFocus(QtCore.Qt.MouseFocusReason)
    app.processEvents()
    hold.lineEdit().setText("2")
    drafts.clear()
    editor.update_projection(_projection(bound, api_values="hold = 250"))
    assert drafts == []
    editor.close()
    editor.deleteLater()


def test_a_value_for_a_parameter_the_pulse_does_not_declare_is_refused_by_name() -> None:
    """The backstop behind the editor: a build never runs a form written
    against another pulse as though it were this one's."""

    from zlc_atom.nodes.scan.plan import apply_api_overrides

    with pytest.raises(ValueError, match="declares no API parameter"):
        apply_api_overrides(_plain_sequence(), {"hold": 240.0})
    applied = apply_api_overrides(_bound_sequence(), {"hold": 240.0})
    assert applied.periods[0].duration == 240
