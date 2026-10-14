from __future__ import annotations

import pytest


#: The preview draws with zlc_plot's compiled kernels; every child loads them
#: from the machine cache instead of compiling its own.
_KERNEL_CACHE = "from zlc_plot._kernel_cache import install\ninstall()\n"


@pytest.fixture
def run_qt(run_qt):
    return lambda code: run_qt(_KERNEL_CACHE + code)


def _schedule_source() -> str:
    return r'''
from zlc_ui import FormChoice
from zlc_ui.pulse import FieldVM, PeriodVM, PortRowVM, ScheduleVM
field = FieldVM("4", validator_kind="int", validator_lo=-8, validator_hi=8)
analog_modes = (
    FormChoice("Step now", "edge"),
    FormChoice("Glide", "ramp"),
    FormChoice("Leave unchanged", "hold"),
)
ports = (PortRowVM("d0", "digital", "Gate", "d0"), PortRowVM("a0", "dac", "Voltage", "a0", lo=-8, hi=8))
periods = (
    PeriodVM("p1", "One", field, "us", ("ns", "us"), digital=(("d0", True),), analog=(("a0", "edge", field),)),
    PeriodVM("p2", "Two", field, "us", ("ns", "us"), digital=(("d0", False),), analog=(("a0", "hold", FieldVM("0", editable=False)),)),
)
vm = ScheduleVM(
    1, 2, "fake", "50 MHz", "10 us", "total", 2, "2/2", "2 periods",
    ports, periods, analog_mode_choices=analog_modes,
)
'''


#: The strip gestures the tests drive, over the snippet's ``app`` and
#: ``strip``.  A drag's payload is the strip's own MIME item key.
_STRIP_GESTURES = r'''
import json

def send(widget, kind, button, buttons, at=QtCore.QPoint(5, 5)):
    """One mouse event at ``at`` in the widget's frame, its global position included."""
    QtWidgets.QApplication.sendEvent(widget, QtGui.QMouseEvent(
        kind, QtCore.QPointF(at), QtCore.QPointF(widget.mapToGlobal(at)),
        button, buttons, QtCore.Qt.NoModifier))
    app.processEvents()

def click(widget):
    send(widget, QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    send(widget, QtCore.QEvent.MouseButtonRelease, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

def item_data(key):
    data = QtCore.QMimeData()
    data.setData(strip.ITEM_MIME, QtCore.QByteArray(json.dumps(key).encode("utf-8")))
    return data

def hover(key, x):
    """One dragMoveEvent, and what the strip decided about it."""
    data = item_data(key)
    event = QtGui.QDragMoveEvent(
        QtCore.QPoint(x, 5), QtCore.Qt.MoveAction, data,
        QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    )
    strip.dragMoveEvent(event)
    return event.isAccepted(), strip._indicator.isVisible()

def drop(key, x):
    """One dropEvent, and whether the strip took it."""
    data = item_data(key)
    event = QtGui.QDropEvent(
        QtCore.QPoint(x, 5), QtCore.Qt.MoveAction, data,
        QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    )
    strip.dropEvent(event)
    return event.isAccepted()
'''


def test_embedded_connection_is_named_and_locked_in_the_real_schedule_view(run_qt) -> None:
    run_qt(
        r'''
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import ConnectionChoiceVM, ConnectionVM, PulseScheduleView

app = ensure_qt_app(["pulse-embedded-connection"])
view = PulseScheduleView()
view.set_connection(ConnectionVM(
    choices=(ConnectionChoiceVM("Experiment session", "given"),),
    selected="given",
    endpoint="",
    status="connected",
    locked=True,
))
assert view.connection_combo.currentText() == "Experiment session"
assert not view.connection_combo.isEnabled()
assert not view.connection_endpoint.isEnabled()
assert not view.connection_button.isEnabled()
try:
    ConnectionVM(
        choices=(ConnectionChoiceVM("Experiment session", "given"),),
        selected="mystery",
        endpoint="",
        status="",
    )
except ValueError:
    pass
else:
    raise AssertionError("an unknown selected connection was accepted")

standalone = PulseScheduleView()
choices = (
    ConnectionChoiceVM("Simulated", "virtual"),
    ConnectionChoiceVM("Bench server", "remote", endpoint_editable=True),
    ConnectionChoiceVM("Edit only", "offline"),
)
standalone.set_connection(ConnectionVM(
    choices=choices,
    selected="offline",
    endpoint="127.0.0.1:18861",
    status="not connected",
))
assert standalone.connection_combo.isEnabled()
assert standalone.connection_button.isEnabled()
assert not standalone.connection_endpoint.isEnabled()
assert standalone.connection_endpoint.text() == "127.0.0.1:18861"
assert "host:port" in standalone.connection_endpoint.toolTip()
standalone.connection_combo.setCurrentIndex(
    standalone.connection_combo.findData("remote")
)
assert standalone.connection_endpoint.isEnabled()
requests = []
standalone.connection_requested.connect(lambda *payload: requests.append(payload))
standalone.connection_button.click()
assert requests == [("remote", "127.0.0.1:18861")]
'''
    )


def test_schedule_stale_rejection_and_widget_reuse(run_qt) -> None:
    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import ScheduleVM, PulseScheduleView
""" + _schedule_source() + r'''
app = ensure_qt_app(["pulse-test"])
view = PulseScheduleView()
assert view.set_schedule(vm)
mode_combo = view._cards["p1"].bus_mode_combos["a0"]
assert [mode_combo.itemText(i) for i in range(mode_combo.count())] == [
    "Step now", "Glide", "Leave unchanged",
]
assert [mode_combo.itemData(i) for i in range(mode_combo.count())] == [
    "edge", "ramp", "hold",
]
view.set_capabilities(False, False, True)
assert not view.sync_button.isEnabled()
first = view._cards["p1"]
assert not view.set_schedule(vm)
try:
    view.set_schedule(ScheduleVM(1, 2, "different", vm.clock_text, vm.total_text, vm.total_tooltip, vm.period_count, vm.visible_text, vm.summary_text, vm.ports, vm.periods, analog_mode_choices=vm.analog_mode_choices))
except ValueError:
    pass
else:
    raise AssertionError("same revision must reject a different projection")
assert not view.set_schedule(ScheduleVM(1, 1, vm.document_name, vm.clock_text, vm.total_text, vm.total_tooltip, vm.period_count, vm.visible_text, vm.summary_text, vm.ports, vm.periods, analog_mode_choices=vm.analog_mode_choices))
assert view._cards["p1"] is first

# A reused card must compare against the newly projected period, not what
# was last typed into this widget before Load/Sync/Clear changed the model.
from dataclasses import replace
from PyQt5 import QtCore, QtTest
committed = []
first.analog_committed.connect(lambda *args: committed.append(args))
def enter(widget, text):
    widget.selectAll()
    QtTest.QTest.keyClicks(widget, text)
    QtTest.QTest.keyClick(widget, QtCore.Qt.Key_Return)
enter(first.bus_value_edits["a0"], "0")
assert committed[-1] == ("p1", "a0", "edge", 0)
view.set_period(periods[0])  # model is 4 again, same live widget
committed.clear()
enter(first.bus_value_edits["a0"], "0")
assert committed == [("p1", "a0", "edge", 0)]
accepted = replace(periods[0], analog=(("a0", "edge", replace(field, text="0")),))
view.set_schedule(replace(vm, document_generation=2, revision=0, periods=(accepted, periods[1])))
assert view._cards["p1"] is first
committed.clear()
enter(first.bus_value_edits["a0"], "0")
assert committed == [], "unchanged accepted value should not be submitted"
first.bus_mode_combos["a0"].setCurrentIndex(first.bus_mode_combos["a0"].findData("ramp"))
assert committed == [("p1", "a0", "ramp", 0)]
# No owner accepted that intent. A retry must not be swallowed either.
enter(first.bus_value_edits["a0"], "0")
assert len(committed) == 2
names, durations = [], []
first.period_name_committed.connect(lambda *args: names.append(args))
first.duration_committed.connect(lambda *args: durations.append(args))
for _ in range(2):
    enter(first.name_edit, "Changed")
    enter(first.duration_edit, "6")
assert names == [("p1", "Changed")] * 2
assert durations == [("p1", 6.0, "us")] * 2

# A value-only signal may not edit Hold, Config or readonly Subpulse rows.
for protected in (
    periods[1],
    replace(periods[1], analog=(("a0", "edge", replace(field, config_active=True)),)),
    replace(periods[1], readonly=True),
):
    view.set_period(protected)
    held = view._cards["p2"]
    rejected = []
    held.analog_committed.connect(lambda *args: rejected.append(args))
    held.bus_value_edits["a0"].editingFinished.emit()
    if not held.bus_mode_combos["a0"].isEnabled():
        held.bus_mode_combos["a0"].setCurrentIndex(held.bus_mode_combos["a0"].findData("ramp"))
    assert rejected == []
    held.analog_committed.disconnect()
'''
    )


def test_stop_is_destructive_and_a_refresh_never_bounces_sync(run_qt) -> None:
    """Stop Pulse ends what the board plays, so it wears the destructive
    face.  Sync's availability has one writer: the presenter's refresh calls
    set_control_state and then set_capabilities, and when the first also
    disabled Sync every refresh switched it off and on again -- which hands
    a focused Sync's focus to the next button."""

    run_qt(
        """
from PyQt5 import QtCore
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseScheduleView
app = ensure_qt_app(["pulse-control-roles"])
view = PulseScheduleView()
assert view.stop_button.role() == "destructive"
view.set_control_state(False, False, False, can_run=False, can_stop=False)
assert view.stop_button.role() == "destructive", "a disabled Stop is still a Stop"

class EnabledChanges(QtCore.QObject):
    seen = []

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.EnabledChange:
            self.seen.append(watched.isEnabled())
        return False

view.set_capabilities(True, False, False)
watch = EnabledChanges()
view.sync_button.installEventFilter(watch)
for can_run in (False, True, False):
    view.set_control_state(False, False, False, can_run=can_run, can_stop=True)
    view.set_capabilities(True, False, False)
assert view.sync_button.isEnabled() and watch.seen == [], watch.seen
view.set_capabilities(False, False, False)
assert watch.seen == [False]
view.close(); view.deleteLater()
"""
    )


def test_schedule_projects_scan_and_api_bindings_into_fields(run_qt) -> None:
    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import FieldVM, PeriodVM, PortRowVM, PulseScheduleView, ScheduleVM, DelayRowVM
app = ensure_qt_app(["pulse-bindings"])
field_scan = FieldVM("1000", scan=True)
field_api = FieldVM("0", source="api")
port = PortRowVM("d0", "digital", "Gate", "d0")
vm = ScheduleVM(
    document_generation=1,
    revision=1,
    document_name="fake",
    clock_text="50 MHz",
    total_text="2 us",
    total_tooltip="total",
    period_count=2,
    visible_text="1/1",
    summary_text="2 periods | scan 1 slot",
    ports=(port,),
    periods=(
        PeriodVM("p1", "One", field_scan, "us", ("ns", "us"), digital=(("d0", True),)),
        PeriodVM("p2", "Two", field_api, "us", ("ns", "us"), digital=(("d0", False),)),
    ),
    delay_rows=(DelayRowVM("d0", field_api, "ns", ("ns",)),),
    scan_summary_text="1 slot · 4 pts",
)
view = PulseScheduleView(); view.set_schedule(vm)
scan_edit = view._cards["p1"].duration_edit
api_edit = view._cards["p2"].duration_edit
delay_edit = view.channel_panel._rows["d0"][0]
assert scan_edit.binding_button.scan and not scan_edit.isReadOnly()
assert api_edit.binding_button.source == "api" and not api_edit.isReadOnly()
assert delay_edit.binding_button.source == "api" and not delay_edit.isReadOnly()
assert view.channel_panel.scan_summary_label.text() == "1 slot · 4 pts"
"""
    )


def test_pulse_binding_popup_emits_independent_scan_source_intents(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from examples.demo_pulse_editor import create_window

app = ensure_qt_app(["pulse-binding-popup"])
# Through the demo's own entry, which is the outside entry: what comes back
# is the handle.  Reaching its view is this package's own business -- the
# clicks below are on real widgets, which is the point of the test.
editor = create_window(window_ratio=0.4)
app.processEvents()
schedule = editor._view.schedule_view
requested = []
schedule.binding_committed.connect(lambda *payload: requested.append(payload))

before = schedule._schedule
duration = schedule._cards["p1"].duration_edit
dac = schedule._cards["p1"].bus_value_edits["da_bias_y"]
delay = schedule.channel_panel._rows["ch01"][0]
for field in (duration, dac, delay):
    QtTest.QTest.mouseClick(field.binding_button, QtCore.Qt.LeftButton)
    assert field._popup.isVisible()
    assert field._popup.width() == field._popup.sizeHint().width()
    QtTest.QTest.mouseClick(field.binding_button, QtCore.Qt.LeftButton)
    assert not field._popup.isVisible()
    QtTest.QTest.mouseClick(field.binding_button, QtCore.Qt.LeftButton)
    assert field._popup.isVisible()
    # Source is one segmented choice; Config is clicked, not cycled to.
    QtTest.QTest.mouseClick(field.source_switch.button("config"), QtCore.Qt.LeftButton)
    assert field._popup.isVisible(), "source choice must not close the binding popup"
    field._popup.hide()
app.processEvents()

assert requested == [
    ("duration", "p1", None, True, "config"),
    ("analog", "p1", "da_bias_y", True, "config"),
    ("delay", None, "ch01", False, "config"),
]
assert schedule._schedule == before
assert duration.binding_button.scan and duration.binding_button.source == "default"
assert dac.binding_button.scan and dac.binding_button.source == "default"
assert delay.binding_button.source == "default"

# This test owns an isolated QApplication subprocess; letting Qt tear down
# with the process avoids an intermittent PyQt5/offscreen native crash in
# the frameless editor's explicit close path after the assertions pass.
"""
    )


def test_dropping_a_period_proposes_a_move_and_moves_nothing_itself(run_qt) -> None:
    """The drag is a real QDrag now, so the DROP is what the test drives.

    It used to be inferred at release from where the button happened to come
    up, with nothing on screen to aim at and no move handling at all -- a
    gesture that clearly went somewhere and did nothing.  The strip still
    reorders nothing by itself: it proposes, and the presenter decides.
    """

    run_qt(
        """
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseScheduleView
""" + _schedule_source() + _STRIP_GESTURES + r'''
app = ensure_qt_app(["pulse-drag"])
view = PulseScheduleView(); view.set_schedule(vm); view.show(); app.processEvents()
events = []
view.reorder_items_requested.connect(events.append)
strip = view.drag_container

cards = strip.pulse_cards()
# Dropped left of the first card: it goes before p1.
drop(("period", "p2"), cards[0].geometry().left())
assert events == [(("period", "p2"), ("period", "p1"))], events

# Dropped past the last card: it goes to the end, which is "before nothing".
events.clear()
drop(("period", "p1"), cards[-1].geometry().right() + 40)
assert events == [(("period", "p2"), ("period", "p1"))], events

# Dropped on itself proposes nothing at all.
events.clear()
drop(("period", "p1"), cards[0].geometry().center().x())
assert events == [], events

# And the strip reorders nothing on its own.
assert tuple(item.period_id for item in strip.pulse_cards()) == ("p1", "p2")

# Close before teardown.  A synthetic QDropEvent built in Python and a widget
# tree torn down by the interpreter's exit race each other, and the process
# dies with an access violation AFTER every assertion has already passed.
view.close()
app.processEvents()
'''
    )


def test_scan_text_is_immediate_state_intent_and_run_has_no_payload(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseEditorView, PulseEditorHandle, PulseTargetView, ScanPageRecord, TargetPortRecord, TargetWidthRule
app = ensure_qt_app(["pulse-pages"])
editor = PulseEditorView(); handle = PulseEditorHandle(None, editor)
scan = editor.scan_view; scan.set_page(ScanPageRecord(source_text="committed", source_dirty=False))
edited = []; runs = []; scan.source_edited.connect(edited.append); scan.run_requested.connect(lambda: runs.append("run"))
scan.scan_code.setPlainText("typing"); scan.scan_run_button.click()
assert edited == ["typing"] and runs == ["run"]
scan.set_page(ScanPageRecord(source_text="typing", table_text="1 2 3"))
table_changes = []
scan.scan_table_view.textChanged.connect(lambda: table_changes.append(True))
scan.set_page(ScanPageRecord(source_text="typing more", table_text="1 2 3", source_dirty=True))
assert not table_changes, 'typing source replaced the unchanged scan-table document'
clears = []
handle.scan_array_clear_requested.connect(lambda: clears.append(True))
scan.scan_clear_array_button.click()
assert clears == [True]
assert scan.scan_code.toPlainText() == "typing more" and scan.scan_table_view.toPlainText() == "1 2 3"
scan.set_workspace_busy(True)
scan.scan_clear_array_button.click()
assert clears == [True] and not scan.scan_clear_array_button.isEnabled()
scan.set_workspace_busy(False)
assert scan.scan_clear_array_button.isEnabled()
scan.set_repeats((1 << 32) - 1)
assert int(scan.scan_repeats_spin.maximum()) == (1 << 32) - 1
assert scan.scan_repeats_spin.text() == str((1 << 32) - 1)
target = PulseTargetView(); target.set_width_rules(TargetWidthRule(1, 1, 1), TargetWidthRule(1, 2, 4))
target.set_ports((TargetPortRecord("d0", "digital", "Gate", ("d0",)),), True, "ready")
events = []; target.apply_requested.connect(events.append)
target.show(); app.processEvents(); QtTest.QTest.mouseClick(target.apply_button, QtCore.Qt.LeftButton)
assert events and isinstance(events[-1][0], TargetPortRecord)
"""
    )


def test_preview_mount_and_demo_smoke(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulsePreviewView
app = ensure_qt_app(["preview"]); view = PulsePreviewView(); first = QtWidgets.QLabel("one"); second = QtWidgets.QLabel("two")
view.mount_content(first, logical_size=(100, 60)); view.mount_content(second, logical_size=(120, 70)); assert view._content_widget is second and first.parentWidget() is None
"""
    )
    completed = run_qt(
        "import runpy, sys; "
        "sys.argv = ['demo_pulse_editor.py', '--once']; "
        "runpy.run_path('examples/demo_pulse_editor.py', run_name='__main__')"
    )
    assert "scan_hold_requested" in completed.stdout
    assert "save_requested" in completed.stdout


def test_linked_panes_show_one_bar_only_while_some_pane_overflows(run_qt) -> None:
    """The group decides "is there more below" ONCE, from the deepest pane.

    Two synchronized bars answer it twice, and the answers diverge the moment
    one column is taller: whichever bar was elected visible then caps how far
    the operator can reach, and rows past that cap are unreachable.
    """

    run_qt(
        """
from PyQt5 import QtCore, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentScrollArea, LinkedScrollPanes
app = ensure_qt_app(["linked-panes"])

panes = LinkedScrollPanes()
short, tall = FluentScrollArea(), FluentScrollArea()
for area, height in ((short, 60), (tall, 900)):
    body = QtWidgets.QWidget(); body.setFixedHeight(height); body.setFixedWidth(120)
    area.setWidgetResizable(False); area.setWidget(body)
    panes.add_pane(area)
panes.resize(400, 300); panes.show(); app.processEvents()

bar = panes.scrollbar
assert bar.isVisible(), "a pane taller than the viewport must be scrollable"
assert bar.maximum() == max(a.verticalScrollBar().maximum() for a in (short, tall)), (
    "the range must come from the DEEPEST pane, not an elected one"
)
bar.setValue(bar.maximum()); app.processEvents()
assert tall.verticalScrollBar().value() + tall.viewport().height() >= 900, (
    "the tall pane's last row must be reachable"
)
tall.verticalScrollBar().setValue(0); app.processEvents()
assert bar.value() == 0, "a pane moving on its own brings the group along"

tall.widget().setFixedHeight(40); app.processEvents()
assert not bar.isVisible(), "nothing overflows any more, so no bar"
"""
    )


def test_schedule_channel_column_is_reachable_without_any_periods(run_qt) -> None:
    """A board's channels scroll even when the timeline beside them is empty."""

    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import DelayRowVM, FieldVM, PortRowVM, PulseScheduleView, ScheduleVM
app = ensure_qt_app(["schedule-scroll"])

ports = tuple(PortRowVM(f"d{n}", "digital", f"Out {n}", f"d{n}") for n in range(24))
view = PulseScheduleView()
view.resize(900, 420); view.show(); app.processEvents()
view.set_schedule(ScheduleVM(
    document_generation=0, revision=0, document_name="",
    clock_text="50 MHz", total_text="0", total_tooltip="", period_count=0,
    visible_text="0/0", summary_text="Add Period to start a pulse on this board",
    ports=ports, periods=(),
    delay_rows=tuple(
        DelayRowVM(port.key, FieldVM("0", editable=False), "ns", ("ns",))
        for port in ports
    ),
))
app.processEvents()

left = view.left_scroll
bar = view.dataset_panes.scrollbar
assert left.widget().height() > left.viewport().height(), "24 ports must overflow"
assert bar.isVisible() and bar.maximum() > 0, "the channel column must be scrollable"
bar.setValue(bar.maximum()); app.processEvents()
assert left.verticalScrollBar().value() + left.viewport().height() >= left.widget().height(), (
    "the last channel must be reachable"
)
"""
    )


def test_clicking_a_period_or_a_gap_decides_where_the_next_one_lands(run_qt) -> None:
    """Selection is what makes a sequence buildable at all.

    A selected gap inserts there, a selected card inserts AFTER it,
    Remove takes the exact selected items and does nothing without a selection.
    Clicking the current selection again clears it; a visible gap inserts at
    that gap rather than silently appending.
    """

    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import FieldVM, PeriodVM, PortRowVM, PulseScheduleView, ScheduleVM
from PyQt5 import QtCore, QtTest
app = ensure_qt_app(["schedule-selection"])

port = PortRowVM("d0", "digital", "Gate", "d0")
periods = tuple(
    PeriodVM(f"p{n}", f"P{n}", FieldVM("1"), "us", ("ns", "us"), digital=(("d0", False),))
    for n in range(8)
)
view = PulseScheduleView()
view.resize(1200, 500); view.show()
view.set_schedule(ScheduleVM(
    document_generation=0, revision=0, document_name="x", clock_text="50 MHz",
    total_text="8 us", total_tooltip="", period_count=8, visible_text="1/1",
    summary_text="x", ports=(port,), periods=periods,
))
app.processEvents()
container = view.drag_container

asked = []
view.insert_period_requested.connect(lambda before: asked.append(("add", before)))
view.remove_items_requested.connect(lambda payload: asked.append(("remove", payload)))

# Nothing selected -> append, but Remove must not guess a deletion target.
view.add_button.click(); view.remove_button.click()
assert asked == [("add", None)], asked

# a selected CARD inserts after it, and Remove takes that one
asked.clear()
QtTest.QTest.mouseClick(view._cards["p0"], QtCore.Qt.LeftButton, pos=QtCore.QPoint(5, 5)); app.processEvents()
assert container.selected_items() == (("period", "p0"),) and container.selected_gap is None
view.add_button.click(); view.remove_button.click()
assert asked == [("add", ("period", "p1")), ("remove", (("period", "p0"),))], asked

# a selected GAP inserts there, and the two selections are exclusive
asked.clear()
QtTest.QTest.mouseClick(container, QtCore.Qt.LeftButton, pos=QtCore.QPoint(1, 15)); app.processEvents()
assert container.selected_items() == () and container.selected_gap == 0
view.add_button.click()
assert asked == [("add", ("period", "p0"))], asked

# clicking the current selection again clears it
QtTest.QTest.mouseClick(container, QtCore.Qt.LeftButton, pos=QtCore.QPoint(1, 15)); app.processEvents()
assert container.selected_items() == () and container.selected_gap is None

# a selection naming a period the strip no longer holds cannot survive a rebuild
QtTest.QTest.mouseClick(view._cards["p1"], QtCore.Qt.LeftButton, pos=QtCore.QPoint(5, 5)); app.processEvents()
before_width = container.width()
before_scroll = view.timeline_scroll.horizontalScrollBar().maximum()
view.set_schedule(ScheduleVM(
    document_generation=0, revision=1, document_name="x", clock_text="50 MHz",
    total_text="1 us", total_tooltip="", period_count=1, visible_text="1/1",
    summary_text="x", ports=(port,), periods=periods[:1],
))
app.processEvents()
assert container.selected_items() == () and container.selected_gap is None
print('removed period geometry', before_width, container.width(), before_scroll,
      view.timeline_scroll.horizontalScrollBar().maximum())
assert container.width() < before_width
assert container.size() == container.sizeHint()
assert view.timeline_scroll.horizontalScrollBar().maximum() < before_scroll
view.close(); view.deleteLater()
app.processEvents()
"""
    )


def test_a_board_locks_its_wiring_and_never_the_names(run_qt) -> None:
    """Two permissions, not one.

    A board owns which lanes exist, how wide a bus is and which pins they
    reach.  What a signal is CALLED is display metadata belonging to whoever
    reads it, and the page says so in its own status line -- "rename freely,
    the topology is the board's".  One flag drove both, so attached the rename
    box was greyed out and Apply with it: the page told the operator to do
    something it had just disabled.
    """

    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseTargetView, TargetPortRecord
app = ensure_qt_app(["target-permissions"])

view = PulseTargetView()
records = (TargetPortRecord("d0", "digital", "cooling", ("F15",)),)
view.set_ports(records, False, "board attached; rename freely")
row = view._rows[0]
assert row.signal.isEnabled() and not row.signal.isReadOnly(), "a name is always the operator's"
assert not row.endpoints.isEnabled(), "wiring belongs to the board"
assert not row.width.isEnabled() and not row.remove_button.isEnabled()
assert not view.add_digital_button.isEnabled() and not view.add_dac_button.isEnabled()
assert view.apply_button.isEnabled(), "Apply commits the rename that is allowed"

view.set_ports(records, True, "offline; author the target")
row = view._rows[0]
assert row.endpoints.isEnabled() and row.width.isEnabled()
assert view.add_digital_button.isEnabled() and view.apply_button.isEnabled()

# Offline the page authors lanes.  Add DAC built its record with one field
# more than the record has, and a TypeError out of a clicked slot ends the
# process -- the standalone editor, or the whole console that opened it.
view.add_dac_button.click()
view.add_digital_button.click()
added = view._current_records()
assert [(record.key, record.kind) for record in added] == [
    ("d0", "digital"), ("digital_1", "digital"), ("dac_1", "dac"),
], added
assert added[-1].endpoints == ("endpoint:dac_1[0]", "endpoint:dac_1[1]"), added[-1]
assert added[-1].clock_endpoint == f"endpoint:{added[-1].clock_key}", added[-1]

# A name the target holds with no row here (a clock no DAC latches with) is
# never minted, across the page's own re-projections: the presenter refuses
# it, and nothing on the page renames what Add minted.
view.set_ports(records, True, "offline; author the target", reserved=("digital_1", "dac_1_clock_1"))
view.add_dac_button.click()
view.add_digital_button.click()
added = view._current_records()
assert [(record.key, record.clock_key) for record in added] == [
    ("d0", None), ("digital_2", None), ("dac_1", "dac_1_clock_2"),
], added
"""
    )


def test_hiding_a_port_takes_its_delay_row_with_it(run_qt) -> None:
    """One visible-row list, three columns.

    The names column and every period card show the ports marked visible,
    and ``_visible_delay_rows`` gives the delay column the same filter, so the
    three read across as one table by construction.  Before that the delay
    rows arrived as "every output the board can delay" and were handed over
    unfiltered, so Hide Off dropped rows from the cards and left their delays
    behind, aligned with whatever happened to be next to them.
    """

    run_qt(
        """
from PyQt5 import QtCore
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import BracketVM, DelayRowVM, FieldVM, PeriodVM, PortRowVM, PulseScheduleView, ScheduleVM
app = ensure_qt_app(["schedule-hide-delay"])

ports = tuple(PortRowVM(f"d{n}", "digital", f"Out {n}", f"d{n}") for n in range(3))

def schedule(ports, revision):
    return ScheduleVM(
        document_generation=0, revision=revision, document_name="pulse",
        clock_text="50 MHz", total_text="1 us", total_tooltip="", period_count=1,
        visible_text=f"{len(ports)}/{len(ports)}", summary_text="",
        ports=ports,
        periods=(PeriodVM("p0", "P0", FieldVM("1"), "us", ("ns", "us"),
                          digital=tuple((port.key, False) for port in ports)),),
        delay_rows=tuple(
            DelayRowVM(port.key, FieldVM("0"), "ns", ("ns",)) for port in ports
        ),
        brackets=(BracketVM('b', 'p0', 'p0', 4),),
    )

def delay_column():
    panel = view.channel_panel
    return sorted(panel._rows, key=lambda key: panel._layout.indexOf(panel._rows[key][0].parentWidget()))

def names_column():
    panel = view.names_panel
    return sorted(panel._row_holders, key=lambda key: panel._layout.indexOf(panel._row_holders[key]))

view = PulseScheduleView()
view.resize(1100, 460); view.show()
view.set_schedule(schedule(ports, 0))
app.processEvents()
assert set(view.channel_panel._rows) == {"d0", "d1", "d2"}
strip = view.drag_container
timeline = strip.items()
posts = strip._posts
strip.show_selection((("bracket", "b:end"),))
layout_moves = []
original_take = strip.layout_main.takeAt
strip.layout_main.takeAt = lambda index: (layout_moves.append(index), original_take(index))[1]

retired_delay = view.channel_panel._rows["d1"][0].parentWidget()
view.set_visible_ports(("d0", "d2"))
assert retired_delay.isHidden(), "retired rows must disappear before deferred deletion"
app.processEvents()
assert strip.items() == timeline and strip._posts == posts
assert not layout_moves, 'hiding a port detached the unchanged timeline'
assert strip.selected_items() == (("bracket", "b:end"),) and strip.selected_gap is None
assert set(view.channel_panel._rows) == {"d0", "d2"}, "a hidden port keeps no delay row"
assert set(view._cards["p0"].port_rows) == {"d0", "d2"}, "and the card agrees"

strip.show_selection(gap=1)
before_height = strip.height()
view.set_visible_ports(("d0",))
for _ in range(4):
    app.processEvents()
print('hidden row geometry', before_height, strip.height(), strip._indicator.height())
assert strip.height() < before_height
assert strip.size() == strip.sizeHint()
assert strip._indicator.height() == strip.pulse_cards()[0].height(), 'selected gap did not follow the real layout'
assert strip.items() == timeline and strip._posts == posts
assert strip.selected_items() == () and strip.selected_gap == 1
view.set_visible_ports(("d0", "d2"))
for _ in range(4):
    app.processEvents()
assert strip._indicator.height() == strip.pulse_cards()[0].height()

# The single-row push is the other way into that column, and it must obey the
# same rule -- otherwise one API binding brings a hidden row back.
view.set_delay_row(DelayRowVM("d1", FieldVM("7"), "ns", ("ns",)))
app.processEvents()
assert set(view.channel_panel._rows) == {"d0", "d2"}

view.set_visible_ports(("d0", "d1", "d2"))
app.processEvents()
assert set(view.channel_panel._rows) == {"d0", "d1", "d2"}, "Show All brings it back"
# The columns are read across: a row shown again goes back to its place,
# not to the bottom of its column, and so does a port added between two.
assert delay_column() == ["d0", "d1", "d2"], delay_column()
wider = ports[:1] + (PortRowVM("d9", "digital", "Out 9", "d9"),) + ports[1:]
view.set_schedule(schedule(wider, 1))
app.processEvents()
order = [port.key for port in wider]
assert names_column() == order, names_column()
assert delay_column() == order, delay_column()
view.close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete); app.processEvents()
"""
    )


def test_the_bracket_posts_are_built_to_frame_the_cards_they_span(run_qt) -> None:
    """The bracket art carries the repeat meaning.

    Two posts the same height as a period card, titled "Bracket N" in the
    pill where a card says "Period k/N", the count on the cards' first
    control line under a "Repeats" header on the end post only, so the pair
    reads as a frame drawn around the span.  It had become a box with a
    glyph in it at whatever height the layout gave -- the thing marking a
    span lined up with nothing inside the span.
    """

    run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import BracketPost, BracketVM, PulseScheduleView
from zlc_ui.pulse._layout import panel_top_height
""" + _schedule_source() + r'''
app = ensure_qt_app(["repeat-art"])

start = BracketPost("b", "start")
end = BracketPost("b", "end", count=4)
for post in (start, end):
    assert post.title() == "Bracket 1", "named in the pill, like the cards it frames"
    assert post.sizePolicy().verticalPolicy() == QtWidgets.QSizePolicy.Expanding
    labels = [w.text() for w in post.findChildren(QtWidgets.QLabel)]
    assert ("Repeats" in labels) == (post.kind == "end"), labels
    assert "Bracket" not in labels, "the pill says it; a second line saying it again said nothing"
    top = post.findChildren(QtWidgets.QWidget)[0]
    assert top.height() == panel_top_height(), (top.height(), panel_top_height())
assert start.width() == end.width(), "the two posts match"

# The count closes the span, so it lives on the end post only -- but both
# posts keep the line, or they would not stay level with each other.
start.show(); end.show(); app.processEvents()
assert not start.count_edit.isVisible()
assert end.count_edit.isVisible() and end.count_edit.text() == "4"
assert isinstance(end.count_spin, QtWidgets.QDoubleSpinBox)
assert end.count_spin.minimum() == 1 and end.count_spin.maximum() == (1 << 32) - 1
assert end.count_spin.decimals() == 0 and end.count_spin.singleStep() == 1
assert end.count_edit._numeric_bounds == (1, (1 << 32) - 1, "int")
counts = []
def accept_count(value):
    counts.append(value)
    end.set_count_field(replace(end._count_field, text=str(value)))
end.count_committed.connect(accept_count)
QtTest.QTest.mouseClick(end.count_spin, QtCore.Qt.LeftButton, pos=QtCore.QPoint(end.count_spin.width()-5, end.count_spin.height()//4))
app.processEvents()
assert end.count_spin.value() == 5 and counts == [5], counts
QtTest.QTest.keyClick(end.count_spin, QtCore.Qt.Key_Down)
assert end.count_spin.value() == 4 and counts[-1] == 4
end.set_count_field(replace(end._count_field, text="1"))
QtTest.QTest.keyClick(end.count_spin, QtCore.Qt.Key_Down)
assert end.count_spin.value() == 1, 'Bracket has no zero/infinite count'
end.set_count_field(replace(end._count_field, text=str((1 << 32)-1)))
QtTest.QTest.keyClick(end.count_spin, QtCore.Qt.Key_Up)
assert end.count_spin.value() == (1 << 32)-1, 'uint32 maximum must not wrap'
end.set_count_field(replace(end._count_field, text="4", source="config", config_active=True))
before = list(counts)
QtTest.QTest.keyClick(end.count_spin, QtCore.Qt.Key_Up)
assert end.count_spin.isReadOnly() and end.count_edit.isReadOnly()
assert end.count_spin.value() == 4 and counts == before
assert end.count_edit.binding_button.isEnabled(), 'Config source remains editable'
end.set_count_field(replace(end._count_field, source="default", config_active=False))
bindings = []
end.binding_committed.connect(lambda scan, source: bindings.append((scan, source)))
end.count_edit.binding_button.click(); app.processEvents()
assert not end.count_edit.scan_toggle.isEnabled()
end.count_edit.source_switch.button("api").click()
assert bindings[-1] == (False, "api")
end.count_edit.source_switch.button("config").click()
assert bindings[-1] == (False, "config")
end.count_edit._popup.close()

# In the strip, the posts stand beside cards of the same build.
view = PulseScheduleView(); view.set_schedule(vm); view.show(); app.processEvents()
view.bracket_committed.emit("b", "p1", "p2", 3)

# One physical period has a non-zero span and may be repeated.  The model and
# compiler accept start == end; the button must not invent a two-card rule.
one = replace(
    vm,
    document_generation=2,
    revision=1,
    period_count=1,
    periods=(vm.periods[0],),
)
assert view.set_schedule(one)
requested = []
feedback = []
view.bracket_add_requested.connect(lambda *payload: requested.append(payload))
view.feedback_requested.connect(feedback.append)
view.bracket_button.click()
assert requested == [("p1", "p1", one.default_bracket_count)]
assert feedback == []
for widget in (view, start, end):
    widget.close()
    widget.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
app.quit()
'''
    )


def test_a_posts_number_and_ink_say_which_bracket_it_frames(run_qt) -> None:
    """Both posts of a bracket wear its number in the title and its colour on
    the title and the edge -- the colour the preview draws that bracket's
    loop in -- and a kept post is renumbered in place when the brackets
    around it change.  A period card stays black on white, a spacer grey and
    dashed: three kinds an eye tells apart."""

    run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import TEXT
from zlc_ui.pulse import BracketPost, BracketVM, PulseScheduleView
""" + _schedule_source() + r'''
app = ensure_qt_app(["bracket-ink"])

outer = BracketVM("b", "p1", "p2", 3, ordinal=1, color="#C96F3D")
inner = BracketVM("c", "p2", "p2", 2, ordinal=2, color="#4F7EA8")
view = PulseScheduleView()
view.set_schedule(replace(vm, brackets=(outer, inner)))
view.show(); app.processEvents()
posts = {post.key: post for post in view.drag_container._posts}
assert set(posts) == {"b:start", "b:end", "c:start", "c:end"}, sorted(posts)
for key in ("b:start", "b:end"):
    assert posts[key].title() == "Bracket 1" and posts[key].color == "#C96F3D", key
    assert posts[key]._zlc_surface == (None, False, "#C96F3D")
    assert posts[key]._zlc_title_color == "#C96F3D"
for key in ("c:start", "c:end"):
    assert posts[key].title() == "Bracket 2" and posts[key].color == "#4F7EA8", key
    assert "Bracket 2" in posts[key].toolTip()
card = view.drag_container.pulse_cards()[0]
assert card._zlc_surface == (None, False, None) and card._zlc_title_color == TEXT

# The outer bracket leaves: the kept posts of the other become "Bracket 1"
# in their new ink, without being rebuilt.
kept = posts["c:end"]
view.set_schedule(replace(vm, document_generation=2, brackets=(replace(inner, ordinal=1, color="#C96F3D"),)))
app.processEvents()
assert view.drag_container._posts[1] is kept
assert kept.title() == "Bracket 1" and kept.color == "#C96F3D"
assert kept._zlc_surface == (None, False, "#C96F3D")

# A bare post is Bracket 1 in the text ink until told otherwise.
plain = BracketPost("z", "end")
assert plain.title() == "Bracket 1" and plain.color == "" and plain._zlc_title_color == TEXT
for widget in (view, plain):
    widget.close()
    widget.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
app.quit()
'''
    )


def test_bracket_posts_drop_on_period_gaps_and_run_repeats_uses_uint32(run_qt) -> None:
    """Both independent repeat layers are editable on the real Edit page."""

    run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import BracketPost, BracketVM, PulseScheduleView
import zlc_ui.pulse.schedule_view as schedule_module
""" + _schedule_source() + _STRIP_GESTURES + r'''
app = ensure_qt_app(["bracket-drag-and-run-repeats"])
view = PulseScheduleView()
bracketed = replace(
    vm,
    revision=3,
    brackets=(BracketVM("b", "p1", "p1", 4),),
    run_repeats=(1 << 32) - 1,
)
assert view.set_schedule(bracketed)
view.show(); app.processEvents()

# The persisted outer count is its own full-range control, including 0=∞.
assert int(view.channel_panel.run_repeats_spin.maximum()) == (1 << 32) - 1
assert int(view.channel_panel.run_repeats_spin.value()) == (1 << 32) - 1
assert view.channel_panel.run_repeats_spin.text() == str((1 << 32) - 1)
run_requests = []
view.run_repeats_committed.connect(run_requests.append)
view.channel_panel.run_repeats_spin.setValue(0)
view.channel_panel.run_repeats_spin.editingFinished.emit()
assert run_requests == [0]
assert view._schedule.run_repeats == (1 << 32) - 1, "the view only proposes"

bracket_requests = []
view.bracket_committed.connect(lambda *payload: bracket_requests.append(payload))
strip = view.drag_container

# A real mouse move on either post starts the strip's QDrag, not a local
# endpoint mutation.  Substitute only the native drag loop so the test can
# inspect what the gesture carries without needing a desktop drag target.
started = []
real_drag = schedule_module.QtGui.QDrag
class InspectDrag:
    def __init__(self, parent): self.data = None; started.append(self)
    def setMimeData(self, data): self.data = data
    def setPixmap(self, _pixmap): pass
    def setHotSpot(self, _point): pass
    def exec_(self, _action): return QtCore.Qt.IgnoreAction
schedule_module.QtGui.QDrag = InspectDrag
start_post = next(post for post in strip.findChildren(BracketPost) if post.kind == "start")
# The gesture is measured in global coordinates, as a real pointer's is;
# a synthetic event carries its global position explicitly.
pressed = QtCore.QPoint(5, 5)
moved = QtCore.QPoint(5 + QtWidgets.QApplication.startDragDistance() + 1, 5)
QtWidgets.QApplication.sendEvent(start_post, QtGui.QMouseEvent(
    QtCore.QEvent.MouseButtonPress, QtCore.QPointF(pressed),
    QtCore.QPointF(start_post.mapToGlobal(pressed)), QtCore.Qt.LeftButton,
    QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
))
QtWidgets.QApplication.sendEvent(start_post, QtGui.QMouseEvent(
    QtCore.QEvent.MouseMove, QtCore.QPointF(moved),
    QtCore.QPointF(start_post.mapToGlobal(moved)),
    QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
))
schedule_module.QtGui.QDrag = real_drag
assert len(started) == 1
assert started[0].data.hasFormat(strip.ITEM_MIME)
assert json.loads(bytes(started[0].data.data(strip.ITEM_MIME))) == ["bracket", "b:start"]

orders = []
view.reorder_items_requested.connect(orders.append)

cards = strip.pulse_cards()
assert drop(("bracket", "b:end"), strip.items()[-1].geometry().right() + 40)
assert orders == [(("bracket", "b:start"), ("period", "p1"), ("period", "p2"), ("bracket", "b:end"))]
assert bracket_requests == [], "drag proposes item order, not a second endpoint update"
assert view._schedule.brackets == bracketed.brackets, "a drop is proposal-only"

# Re-project the accepted span, then move its left post to the first gap.
orders.clear()
right_only = replace(
    bracketed,
    revision=4,
    brackets=(BracketVM("b", "p2", "p2", 4),),
)
assert view.set_schedule(right_only)
app.processEvents()
cards = strip.pulse_cards()
assert drop(("bracket", "b:start"), strip.items()[0].geometry().left())
assert orders == [(("bracket", "b:start"), ("period", "p1"), ("period", "p2"), ("bracket", "b:end"))]
assert view._schedule.brackets == right_only.brackets

# A post cannot cross its partner; an illegal gap commits nothing.
orders.clear()
assert not drop(("bracket", "b:start"), strip.items()[-1].geometry().right() + 40)
assert orders == [] and bracket_requests == []
end_post = next(post for post in strip._posts if post.kind == 'end')
end_post.count_edit.setFocus(); end_post.count_edit.selectAll()
QtTest.QTest.keyClicks(end_post.count_edit, "7")
QtTest.QTest.keyClick(end_post.count_edit, QtCore.Qt.Key_Return)
assert bracket_requests == [('b', 'p2', 'p2', 7)], 'retained post used stale bracket anchors'
end_post.count_edit.selectAll()
QtTest.QTest.keyClicks(end_post.count_edit, str((1 << 32) - 1))
QtTest.QTest.keyClick(end_post.count_edit, QtCore.Qt.Key_Return)
assert bracket_requests[-1] == ('b', 'p2', 'p2', (1 << 32) - 1), 'Qt must not wrap uint32 counts to signed int'
empty = replace(right_only, revision=5, brackets=(BracketVM('b', 'p2', 'p1', 7),))
assert view.set_schedule(empty)
app.processEvents()
assert end_post in strip._posts and start_post in strip._posts
keys = tuple(strip._item_key(item) for item in strip.items())
assert keys.index(('bracket', 'b:end')) == keys.index(('bracket', 'b:start')) + 1
locked = replace(empty, revision=6, brackets=(replace(empty.brackets[0], readonly=True),))
assert view.set_schedule(locked)
before = list(bracket_requests)
QtTest.QTest.keyClick(end_post.count_spin, QtCore.Qt.Key_Up)
assert end_post.count_spin.isReadOnly() and not end_post.count_edit.binding_button.isEnabled()
assert bracket_requests == before, 'file-backed Subpulse counts must remain read-only'
assert view.set_schedule(replace(empty, revision=7))
assert not end_post.count_spin.isReadOnly() and end_post.count_edit.binding_button.isEnabled()

view.close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete); app.processEvents()
app.quit()
'''
    )


def test_a_config_binding_shows_readonly_effective_values_and_keeps_source_editable(run_qt) -> None:
    """Scan is independent of value source and never removes the default."""

    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import API_VIOLET, CONFIG_GREEN, CONFIG_GREEN_DARK, CONFIG_GREEN_TINT, ORANGE
from zlc_ui.pulse.scan_line_edit import FluentScanLineEdit

app = ensure_qt_app(["binding-colours"])

edit = FluentScanLineEdit("12")
edit.set_field_state(editable=True, source="config")
assert edit.binding_button.source == "config"
assert not edit.binding_button.scan
assert not edit.isReadOnly(), "a config number is the operator's to read and type"
assert CONFIG_GREEN in edit.styleSheet()
edit.set_field_state(editable=True, source="config", config_key="bias", effective_text="120 value", config_active=True, config_enabled=True, default_text="12")
assert CONFIG_GREEN_TINT in edit.styleSheet() and CONFIG_GREEN_DARK in edit.styleSheet()
edit.binding_button.click()
assert edit.config_name.currentText() == "bias" and edit.config_name.isVisible()
edit.set_field_state(editable=False, source="config", config_key="bias", effective_text="120 value")
assert not edit.isEnabled() and not edit.binding_button.isEnabled()
assert edit.isReadOnly() and not edit._popup.isVisible()
assert CONFIG_GREEN_TINT not in edit.styleSheet()
edit.binding_button.click()
assert not edit._popup.isVisible(), "a held value cannot open binding authoring"
edit.set_field_state(editable=True, source="config", config_key="bias", effective_text="120 value", config_active=True, config_enabled=True, default_text="12")
assert edit.isEnabled() and edit.binding_button.isEnabled() and edit.isReadOnly()
edit.binding_button.click()
assert edit._popup.isVisible() and edit.config_name.currentText() == "bias"
edit._popup.hide()

edit.set_field_state(editable=True, scan=True, source="config")
assert not edit.isReadOnly() and edit.binding_button.scan
assert edit.binding_button.source == "config"

edit.set_field_state(editable=True)
assert edit.binding_button.source == "default" and not edit.isReadOnly()

try:
    edit.set_field_state(editable=True, source="whatever")
except ValueError as error:
    assert "api" in str(error) and "config" in str(error)
else:
    raise AssertionError("an unknown binding must be refused")

edit.deleteLater()
app.processEvents()
"""
    )



def test_config_page_edits_values_separately_from_shared_field_references(run_qt) -> None:
    run_qt(r'''
from dataclasses import replace
import sys
from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import BindingRecord, ConfigPageRecord, PulseEditorView, PulseEditorHandle, ScanPageRecord
from zlc_ui.fluent import read_editable_combo
app = ensure_qt_app(["pulse-config"])
qt_errors = []
sys.excepthook = lambda kind, error, trace: qt_errors.append(f"{kind.__name__}: {error}")
view = PulseEditorView()
handle = PulseEditorHandle(None, view)
record = ConfigPageRecord(
    file_path="draft.json", active_path="saved.json", dirty=True,
    entries=(("bias", "120", "value"),),
    available_names=("bias", "saved_only"),
    bindings=(("dac:p1:x", "MOT.bias_x", "bias", "0", "100", "Applied"),
              ("dac:p2:x", "PGC.bias_x", "bias", "10", "100", "Applied")),
)
handle.set_config_page(record)
config = view.config_view
view.resize(1100, 800)
view.tabs.setCurrentWidget(config)
view.show()
app.processEvents()
edits, bindings, saves = [], [], []
handle.config_entries_edited.connect(edits.append)
handle.config_entries_edited.connect(lambda entries: handle.set_config_page(replace(record, entries=entries)))
def accept_binding(field, key):
    bindings.append((field, key))
    if key == 'bad name':
        return
    rows = tuple((row[0], row[1], key if row[0] == field else row[2], *row[3:]) for row in record.bindings)
    handle.set_config_page(replace(record, entries=config._entries(), bindings=rows))
handle.config_binding_committed.connect(accept_binding)
handle.config_save_requested.connect(lambda: saves.append(True))
assert config.path_text.text() == "draft.json"
assert "saved.json" in config.file_status.text()
assert not hasattr(view.schedule_view.channel_panel, "config_status_button")
table = config.values_table
index = config.values_model.index(0, 1)
QtTest.QTest.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=table.visualRect(index).center())
editor = app.focusWidget()
assert isinstance(editor, QtWidgets.QLineEdit)
editor.selectAll()
QtTest.QTest.keyClicks(editor, "not yet a number")
QtTest.QTest.keyClick(editor, QtCore.Qt.Key_Return)
app.processEvents()
assert edits[-1] == (("bias", "not yet a number", "value"),)
handle.set_config_page(replace(record, entries=edits[-1]))
assert config.values_model.index(0, 1).data() == "not yet a number"
assert config._binding_rows['dac:p1:x'][3].text() == "100"
assert config._binding_rows['dac:p2:x'][3].text() == "100"
original_items = tuple(config.values_model.item(0, column) for column in range(3))
original_combos = {key: widgets[1] for key, widgets in config._binding_rows.items()}
QtTest.QTest.mouseClick(config.add_button, QtCore.Qt.LeftButton)
app.processEvents()
assert not qt_errors, qt_errors
assert config.values_table.verticalHeader().count() == config.values_model.rowCount()
assert config.values_table.visualRect(config.values_model.index(1, 0)).height() > 0
assert tuple(config.values_model.item(0, column) for column in range(3)) == original_items
editor = app.focusWidget()
assert isinstance(editor, QtWidgets.QLineEdit)
QtTest.QTest.keyClicks(editor, "offset")
QtTest.QTest.keyClick(editor, QtCore.Qt.Key_Return)
app.processEvents()
assert edits[-1][1] == ("offset", "", "")
assert table.verticalScrollBar().maximum() == 0
assert table.geometry().bottom() < config.add_button.geometry().top()
entries_before_names = config._entries()
combo = config._binding_rows["dac:p2:x"][1]
assert tuple(combo.itemData(i) for i in range(combo.count())) == ('bias', 'saved_only')
combo.showPopup()
QtTest.QTest.keyClick(combo.view(), QtCore.Qt.Key_End)
QtTest.QTest.keyClick(combo.view(), QtCore.Qt.Key_Return)
assert bindings == [("dac:p2:x", "saved_only")]
combo.lineEdit().setFocus(); combo.lineEdit().selectAll()
QtTest.QTest.keyClick(combo.lineEdit(), QtCore.Qt.Key_Backspace)
QtTest.QTest.keyClick(combo.lineEdit(), QtCore.Qt.Key_Return)
assert bindings[-1] == ("dac:p2:x", "") and combo.currentText() == ""
combo.lineEdit().setFocus()
combo.lineEdit().selectAll()
QtTest.QTest.keyClicks(combo.lineEdit(), 'Default')
QtTest.QTest.keyClick(combo.lineEdit(), QtCore.Qt.Key_Return)
QtTest.QTest.keyClick(combo.lineEdit(), QtCore.Qt.Key_Tab)
assert bindings[-1] == ('dac:p2:x', 'Default')
assert bindings.count(('dac:p2:x', 'Default')) == 1
assert read_editable_combo(combo) == 'Default'
assert tuple(combo.itemData(i) for i in range(combo.count())) == ('bias', 'saved_only')
combo.lineEdit().setFocus(); combo.lineEdit().selectAll()
QtTest.QTest.keyClicks(combo.lineEdit(), 'bad name')
QtTest.QTest.keyClick(combo.lineEdit(), QtCore.Qt.Key_Return)
assert read_editable_combo(combo) == 'Default', 'rejected text must return to accepted key'
combo.lineEdit().setFocus(); combo.lineEdit().selectAll()
QtTest.QTest.keyClicks(combo.lineEdit(), 'unfinished')
handle.set_config_page(replace(record, entries=(('other', '5', 'value'),), available_names=('other',)))
assert combo.currentText() == 'unfinished', 'a file projection erased an in-progress name'
combo.lineEdit().setModified(False)
unloaded_rows = tuple((row[0], row[1], 'Default' if row[0] == 'dac:p2:x' else row[2], *row[3:]) for row in record.bindings)
handle.set_config_page(replace(record, active_path='', available_names=(), bindings=unloaded_rows))
assert combo.count() == 0 and combo.currentText() == 'Default'
handle.set_config_page(replace(record, entries=entries_before_names))
QtTest.QTest.mouseClick(config.save_button, QtCore.Qt.LeftButton)
assert saves == [True]
assert view.scan_view.scan_load_array_button.text() == "Load Array"
view.tabs.setCurrentWidget(view.schedule_view)
app.processEvents()
channel = view.schedule_view.channel_panel
switch = channel.config_mode
# The row's "Config:" label names the switch; it says nothing more itself.
assert switch.text() == "" and switch.isVisible()
switch_width = switch.width()
current_page = replace(record, entries=config._entries())
handle.set_config_page(replace(current_page, active_path='C:/config/' + 'calibration_' * 30 + '.json'))
app.processEvents()
assert switch.width() == switch_width and switch.text() == ""
handle.set_config_page(current_page)
view.tabs.setCurrentWidget(view.config_view)
assert view.current_page == "Value Config"
kept_items = tuple(config.values_model.item(1, column) for column in range(3))
assert all(item is not None for item in kept_items)
entries = config._entries()
handle.set_config_page(replace(record, entries=entries[1:]))
assert tuple(config.values_model.item(0, column) for column in range(3)) == kept_items
extra = ("dac:new:x", "Added.bias_x", "bias", "0", "100", "Applied")
handle.set_config_page(replace(record, bindings=(extra, *record.bindings)))
assert config._binding_rows['dac:p1:x'][1] is original_combos['dac:p1:x']
assert config._binding_rows['dac:p2:x'][1] is original_combos['dac:p2:x']
handle.set_config_page(replace(record, bindings=(record.bindings[1],)))
assert config._binding_rows['dac:p2:x'][1] is original_combos['dac:p2:x']
handle.set_config_page(replace(
    record, bindings=(*record.bindings, extra),
    binding_groups=(("dac:p1:x", "mot", "MOT & cooling"),
                    ("dac:p2:x", "pgc", "PGC")),
))
assert tuple(config._binding_headers) == ("", "mot", "pgc")
assert config._binding_headers["mot"].text() == "MOT & cooling"
positions = {
    field: config._binding_grid.getItemPosition(config._binding_grid.indexOf(widgets[0]))[0]
    for field, widgets in config._binding_rows.items()
}
assert positions["dac:new:x"] < positions["dac:p1:x"] < positions["dac:p2:x"]
assert config._binding_rows["dac:p2:x"][1] is original_combos["dac:p2:x"]
assert all(widgets[1].currentText() == "bias" for widgets in config._binding_rows.values())
handle.set_config_page(record)
assert not config._binding_headers
assert config._binding_rows["dac:p2:x"][1] is original_combos["dac:p2:x"]
view.scan_view.set_page(ScanPageRecord(bindings=(
    BindingRecord("dac:p1:x", "Load · bias", scan=True, group_id="mot", group_label="MOT & cooling"),
    BindingRecord("dac:p2:x", "Cool · bias", source="api", group_id="pgc", group_label="PGC"),
    BindingRecord("delay:x", "Delay", source="api"),
)))
from zlc_ui.fluent import FluentSectionLabel
listed = view.scan_view._bindings_layout
shown = [listed.itemAt(index).widget() for index in range(listed.count())]
assert [w.text() for w in shown if isinstance(w, FluentSectionLabel)] == ["Pulse", "MOT & cooling", "PGC"]
scan_text = "\n".join(w.text() for w in shown)
assert "Load · bias · Scan" in scan_text and "Cool · bias · API" in scan_text
assert not qt_errors, qt_errors
view.finish_close()
''')


def test_linked_panes_never_starve_a_pane_to_line_the_group_up(run_qt) -> None:
    """Levelling the columns may not cost a column its rows.

    A pane's ``maximum`` is CLAMPED AT ZERO, so a pane that already fits
    cannot report how much it fits by.  Recovering its natural reach by
    subtracting back the padding it was given is therefore only true of a pane
    that overflows -- for one that does not, the clamp hides the padding and
    the next round adds it again.  That compounded (8, 45, 111, 206, ...) until
    the viewport was gone, and the Pulse Editor opened with its period columns
    blank while the file said six periods.

    So the group levels the one quantity every pane can actually give: the
    VIEWPORT, down to the shallowest.  The padding is then bounded by the
    difference it was measured from, and no pane can be shortened past what it
    had.
    """

    run_qt(
        """
from PyQt5 import QtCore, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentScrollArea, LinkedScrollPanes
app = ensure_qt_app(["linked-panes-starve"])

panes = LinkedScrollPanes()
# The shape that made it diverge: one pane whose content is far SHORTER than
# its viewport beside one that overflows.  The short pane reports maximum 0
# however much room it has to spare.
short, tall = FluentScrollArea(), FluentScrollArea()
for area, height in ((short, 8), (tall, 900)):
    body = QtWidgets.QWidget(); body.setFixedHeight(height); body.setFixedWidth(120)
    area.setWidgetResizable(False); area.setWidget(body)
    panes.add_pane(area)
panes.resize(400, 300); panes.show()
for _ in range(40):
    app.processEvents()

for name, area in (("short", short), ("tall", tall)):
    assert area.viewport().height() > 0, (
        f"the {name} pane was left with no viewport, so it shows nothing"
    )
settled = [area.viewportMargins().bottom() for area in (short, tall)]
for _ in range(20):
    app.processEvents()
assert [area.viewportMargins().bottom() for area in (short, tall)] == settled, (
    "the padding is still moving, so it is feeding itself"
)
assert max(settled) <= 300, "no pane may be padded past the group's own height"

# What the levelling is FOR: one value means one row, all the way down.
heights = {area.viewport().height() for area in (short, tall)}
assert len(heights) == 1, f"the viewports were not levelled: {heights}"

# And a pane that stops sooner still stops sooner -- that is its content
# saying so, not something a margin should hide.
assert short.verticalScrollBar().maximum() == 0
assert tall.verticalScrollBar().maximum() > 0

panes.close()
app.processEvents()
"""
    )


def test_a_card_and_a_post_are_one_gesture_with_two_payloads(run_qt) -> None:
    """Dragging offers only the gaps that would do something -- for both.

    The bracket post refused an impossible gap while the cursor was still
    over it: no marker, no drop cursor.  The card accepted every gap, drew
    the marker in all of them, and then threw away a drop onto the two that
    mean "where it already is".  Same gesture, opposite answers to the same
    question, and the half that says yes and does nothing is the worse half.
    """

    run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import BracketVM, PulseScheduleView
""" + _schedule_source() + _STRIP_GESTURES + r'''
app = ensure_qt_app(["drag-symmetry"])
view = PulseScheduleView()
assert view.set_schedule(replace(vm, revision=3, brackets=(BracketVM("b", "p1", "p1", 4),)))
view.show(); app.processEvents()
strip = view.drag_container
cards = strip.pulse_cards()

on_itself = cards[0].geometry().center().x()
elsewhere = cards[-1].geometry().right() + 40

card = ("period", "p1")
post = ("bracket", "b:end")

# A move that would change nothing is refused WHILE dragging, not after.
assert hover(card, on_itself) == (False, False)
end_post = next(item for item in strip.items() if strip._item_key(item) == post)
assert hover(post, end_post.geometry().center().x()) == (False, False)

# A move that would do something is offered, and shows where it lands.
assert hover(card, elsewhere) == (True, True)
assert hover(post, elsewhere) == (True, True)

# And what the marker offered is what the drop commits -- for both.
moves = []
view.reorder_items_requested.connect(moves.append)

assert drop(card, on_itself) is False and moves == []
assert drop(post, end_post.geometry().center().x()) is False and moves == []
assert drop(card, elsewhere) is True
assert moves == [(("bracket", "b:start"), ("bracket", "b:end"), ("period", "p2"), card)]
assert drop(post, elsewhere) is True
assert moves[-1] == (("bracket", "b:start"), card, ("period", "p2"), post)

view.close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
app.quit()
'''
    )


def test_clicking_a_post_marks_it_the_way_clicking_a_card_does(run_qt) -> None:
    """A card and a bracket post are the same kind of thing to pick up.

    Clicking a card drew a border round it; clicking a post drew nothing at
    all, so the two read as different kinds of object when the only real
    difference between them is what they carry.  BracketPost has been a
    FluentGroupBox since it was written -- the outline it needed was already
    on it, and nobody had ever called it.
    """

    run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import BracketPost, BracketVM, PulseScheduleView
""" + _schedule_source() + _STRIP_GESTURES + r'''
app = ensure_qt_app(["click-symmetry"])
view = PulseScheduleView()
assert view.set_schedule(replace(vm, revision=3, brackets=(BracketVM("b", "p1", "p2", 4),)))
view.show(); app.processEvents()
strip = view.drag_container
cards = strip.pulse_cards()
posts = {post.kind: post for post in strip.findChildren(BracketPost)}
assert all(item.cursor().shape() == QtCore.Qt.ArrowCursor for item in (*cards, *posts.values()))

def outlined():
    """Everything currently wearing the selection fill."""
    return {item.period_id for item in cards if item.is_selected()} | {
        f"post:{kind}" for kind, post in posts.items()
        if post.is_selected()
    }

click(cards[0])
assert strip.selected_items() == (("period", "p1"),) and strip.selected_gap is None
assert outlined() == {"p1"}, outlined()

# Clicking a post marks the post -- and takes the mark off the card, because
# at most one thing is what the next edit acts on.
click(posts["end"])
assert strip.selected_items() == (("bracket", "b:end"),) and strip.selected_gap is None
assert outlined() == {"post:end"}, outlined()

# Clicking the marked one again clears it, exactly as a card does.
click(posts["end"])
assert strip.selected_items() == () and strip.selected_gap is None
assert outlined() == set(), outlined()

# And the other way round: a post, then a card.
click(posts["start"])
assert outlined() == {"post:start"}, outlined()
click(cards[1])
assert strip.selected_items() == (("period", "p2"),) and strip.selected_gap is None
assert outlined() == {"p2"}, outlined()

view.close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
app.quit()
'''
    )


def test_collapsing_the_left_panels_hands_the_width_to_the_periods(run_qt) -> None:
    """A QScrollArea does not shrink-wrap.

    ``AdjustToContents`` computes the area's hint once and caches it, and
    hiding the widget's children does not clear that cache -- so Collapse hid
    two panels and the pane stayed its old width, leaving the freed space
    blank between the stub and period cards that never moved.
    """

    run_qt(
        _schedule_source()
        + r'''
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseScheduleView

app = ensure_qt_app(["pulse-collapse-width"])
view = PulseScheduleView()
view.set_schedule(vm)
view.resize(900, 600); view.show()
for _ in range(40):
    app.processEvents()

open_width = view.left_scroll.width()
assert open_width == view.left_body.sizeHint().width(), (
    "the pane is not the width of the panels it holds"
)
open_timeline = view.timeline_scroll.width()

view.collapse_button.click()
for _ in range(40):
    app.processEvents()

collapsed_width = view.left_scroll.width()
assert collapsed_width == view.left_body.sizeHint().width(), (
    f"the pane kept {collapsed_width} px for panels that want "
    f"{view.left_body.sizeHint().width()}"
)
assert collapsed_width < open_width, "Collapse freed no width at all"
assert view.timeline_scroll.width() > open_timeline, (
    "the freed width went nowhere: the periods are no wider than before"
)

view.collapse_button.click()
for _ in range(40):
    app.processEvents()
assert view.left_scroll.width() == open_width, "Show Left did not give it back"

view.close()
app.processEvents()
'''
    )


def test_the_bottom_bar_is_one_table_of_rows_across_three_cards(run_qt) -> None:
    """Row two of Control is read beside row two of Connection and Ports.

    They were three private rhythms -- two row heights and two gaps -- so the
    fourth row of Control sat six pixels below the boxes beside it.
    """

    run_qt(
        r'''
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseScheduleView

app = ensure_qt_app(["pulse-bottom-bar"])
view = PulseScheduleView()
view.resize(1100, 700); view.show()
for _ in range(40):
    app.processEvents()

cards = {}
for box in view.button_frame.findChildren(QtWidgets.QGroupBox):
    cards[box.title()] = box
assert set(cards) == {"Control", "Connection", "Ports"}, sorted(cards)

def rows(card):
    """The top of every row this card lays out, in the window's frame."""

    found = []
    for item in (card.layout().itemAt(i) for i in range(card.layout().count())):
        if item.widget() is not None:
            found.append(item.widget().mapTo(view, item.widget().rect().topLeft()).y())
        elif item.layout() is not None:
            inner = item.layout()
            if isinstance(inner, QtWidgets.QGridLayout):
                for row in range(inner.rowCount()):
                    cell = inner.itemAtPosition(row, 0)
                    if cell is not None and cell.widget() is not None:
                        widget = cell.widget()
                        found.append(widget.mapTo(view, widget.rect().topLeft()).y())
            else:
                first = inner.itemAt(0)
                if first is not None and first.widget() is not None:
                    widget = first.widget()
                    found.append(widget.mapTo(view, widget.rect().topLeft()).y())
    return found

tops = {name: rows(card) for name, card in cards.items()}
shortest = min(len(value) for value in tops.values())
assert shortest >= 3, tops
for index in range(shortest):
    values = {name: value[index] for name, value in tops.items()}
    assert len(set(values.values())) == 1, (
        f"row {index + 1} is at different heights across the bar: {values}"
    )

view.close()
app.processEvents()
'''
    )


def test_a_renamed_port_survives_the_next_rebuild(run_qt) -> None:
    """``set_port_label`` renamed the three visible controls and not the
    model the next Hide/Show rebuilds from, so the old name came back."""

    run_qt(
        """
from PyQt5 import QtCore
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import FieldVM, PeriodVM, PortRowVM, PulseScheduleView, ScheduleVM
app = ensure_qt_app(["schedule-rename"])
ports = (PortRowVM("d0", "digital", "Gate", "pin1"),)
period = PeriodVM("p1", "One", FieldVM("1"), "us", ("ns", "us"), digital=(("d0", True),))
view = PulseScheduleView()
view.set_schedule(ScheduleVM(1, 1, "Pulse", "50 MHz", "1 us", "", 1, "1/1", "", ports, (period,)))
view.set_port_label("d0", "Renamed gate")
assert view.names_panel._rows["d0"].text() == "Renamed gate"
assert view._schedule.ports[0].label == "Renamed gate", "the model the rebuild reads"
view.set_visible_ports(())
view.set_visible_ports(("d0",))
assert view.names_panel._rows["d0"].text() == "Renamed gate"
assert view._cards["p1"].checks["d0"].text() == "Renamed gate"
view.deleteLater(); app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
"""
    )


def test_a_press_on_a_cards_chrome_does_not_start_a_drag_by_itself(run_qt) -> None:
    """The press reaches the strip's filter once through the label and once
    through the card, each in its own local frame.  Compared across frames,
    a pointer that had not moved measured a whole label's offset and a
    click on "Duration" started a drag."""

    run_qt(
        """
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseScheduleView
""" + _schedule_source() + _STRIP_GESTURES + r'''
app = ensure_qt_app(["chrome-press"])
view = PulseScheduleView()
assert view.set_schedule(vm)
view.resize(1200, 700); view.show(); app.processEvents()
strip = view.drag_container
card = view._cards["p1"]
label = next(w for w in card.findChildren(QtWidgets.QLabel) if w.text() == "Duration")
local = label.rect().center()
global_pos = label.mapToGlobal(local)
starts = []
strip._begin_drag = starts.append
assert card.cursor().shape() == QtCore.Qt.ArrowCursor
assert label.cursor().shape() == QtCore.Qt.ArrowCursor

send(label, QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, at=local)
send(label, QtCore.QEvent.MouseMove, QtCore.Qt.NoButton, QtCore.Qt.LeftButton, at=local)
assert starts == [], "no motion, no drag"
send(label, QtCore.QEvent.MouseButtonRelease, QtCore.Qt.LeftButton, QtCore.Qt.NoButton, at=local)
assert strip.selected_items() == (("period", "p1"),), "a press that reaches release without moving is a click"

# Real motion past the threshold, measured in the same global frame, drags.
send(label, QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, at=local)
far = global_pos + QtCore.QPoint(QtWidgets.QApplication.startDragDistance() * 3, 0)
send(label, QtCore.QEvent.MouseMove, QtCore.Qt.NoButton, QtCore.Qt.LeftButton,
     at=label.mapFromGlobal(far))
assert starts == [("period", "p1")]
view.close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
'''
    )


def test_a_port_that_changes_kind_under_its_key_is_rebuilt(run_qt) -> None:
    """A document that turned a digital output into a DAC bus kept the old
    checkbox in the card, because the row reconcile asked only whether the
    key was known -- never what kind of row it was."""

    run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore
from zlc_ui import FormChoice
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import FieldVM, PeriodVM, PortRowVM, PulseScheduleView, ScheduleVM
app = ensure_qt_app(["schedule-kind-change"])
ports = (PortRowVM("d0", "digital", "Gate", "pin1"),)
period = PeriodVM("p1", "One", FieldVM("1"), "us", ("ns", "us"), digital=(("d0", True),))
vm = ScheduleVM(1, 1, "Pulse", "50 MHz", "1 us", "", 1, "1/1", "", ports, (period,))
view = PulseScheduleView()
assert view.set_schedule(vm)
old_row = view._cards["p1"].port_rows["d0"]
dac_port = PortRowVM("d0", "dac", "DAC", "dac pins", width=2, lo=-2, hi=1)
dac_period = replace(period, digital=(), analog=(("d0", "edge", FieldVM("1")),))
incoming = replace(vm, document_generation=2, revision=1, ports=(dac_port,),
                   periods=(dac_period,), analog_mode_choices=(FormChoice("Edge", "edge"),))
assert view.set_schedule(incoming)
card = view._cards["p1"]
assert card.port_rows["d0"] is not old_row
assert "d0" not in card.checks and "d0" in card.bus_value_edits
assert card.bus_mode_combos["d0"].currentData() == "edge"
view.deleteLater(); app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
"""
    )


def test_a_channel_row_offers_on_for_digital_and_off_for_every_port(run_qt) -> None:
    """Every row can turn its output off in all periods; only a digital row
    can turn it on -- an analog output has no level to fill with.  Each
    button names the port it acts on."""

    run_qt(
        _schedule_source()
        + """
from dataclasses import replace
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import DelayRowVM, PulseScheduleView
app = ensure_qt_app(["channel-buttons"])
view = PulseScheduleView()
view.set_schedule(replace(vm, delay_rows=tuple(
    DelayRowVM(port.key, FieldVM("0"), "ns", ("ns",)) for port in ports
)))
view.resize(1200, 700)
view.show()
app.processEvents()
asked = []
view.fill_port_requested.connect(lambda key: asked.append(("on", key)))
view.clear_port_requested.connect(lambda key: asked.append(("off", key)))
_edit, _combo, fill_d, clear_d = view.channel_panel._rows["d0"]
_edit, _combo, fill_a, clear_a = view.channel_panel._rows["a0"]
assert fill_d.isVisibleTo(view) and clear_d.isVisibleTo(view)
assert fill_a.isVisibleTo(view) and clear_a.isVisibleTo(view)
assert fill_d.isEnabled() and not fill_a.isEnabled() and clear_a.isEnabled()
assert [widget.mapTo(view, QtCore.QPoint()).x() for widget in view.channel_panel._rows["d0"]] == [
    widget.mapTo(view, QtCore.QPoint()).x() for widget in view.channel_panel._rows["a0"]
]
assert fill_d.text() == "⬤" and clear_d.text() == "○"
for button in (fill_d, clear_d, fill_a, clear_a):
    QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
assert asked == [("on", "d0"), ("off", "d0"), ("off", "a0")], asked
view.close()
"""
    )


def test_a_press_lights_the_card_and_lifting_it_picks_it(run_qt) -> None:
    """Pressing an unpicked card lights it at once, so what a drag is about
    to lift is the one lit; a press that never moves is still the click
    that picks (or, on the picked one, clears); and the drag's first
    movement makes the pick real, so it is the dragged one that stays lit
    after the drop -- until the operator clears it."""

    run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseScheduleView
""" + _schedule_source() + _STRIP_GESTURES + r'''
app = ensure_qt_app(["press-lights"])
view = PulseScheduleView()
assert view.set_schedule(vm)
view.resize(1200, 700); view.show(); app.processEvents()
strip = view.drag_container
starts = []
strip._begin_drag = starts.append

def lit():
    return {card.period_id for card in strip.pulse_cards() if card.is_selected()}

cards = {card.period_id: card for card in strip.pulse_cards()}
# The press lights it before anything is picked.
send(cards["p2"], QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
assert lit() == {"p2"} and strip.selected_items() == () and strip.selected_gap is None
# ...and the release is the click that picks it.
send(cards["p2"], QtCore.QEvent.MouseButtonRelease, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
assert strip.selected_items() == (("period", "p2"),) and lit() == {"p2"}
# A press on the other card moves the light without moving the pick...
send(cards["p1"], QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
assert lit() == {"p1"} and strip.selected_items() == (("period", "p2"),)
# ...and lifting it (a press that moves) picks it for real.  The drag
# (stubbed here) restores this selection when it ends, dropped or not.
far = QtCore.QPoint(5 + QtWidgets.QApplication.startDragDistance() * 3, 5)
send(cards["p1"], QtCore.QEvent.MouseMove, QtCore.Qt.NoButton, QtCore.Qt.LeftButton, at=far)
assert starts == [("period", "p1")], starts
assert strip.selected_items() == (("period", "p1"),) and lit() == {"p1"}
# The reorder the drop proposes keeps the picked one picked.
assert view.set_schedule(replace(vm, revision=3, periods=(periods[1], periods[0])))
assert tuple(card.period_id for card in strip.pulse_cards()) == ("p2", "p1")
assert strip.selected_items() == (("period", "p1"),) and lit() == {"p1"}
# Clicking the picked one clears it: the press changes nothing it shows.
send(cards["p1"], QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
assert lit() == {"p1"}
send(cards["p1"], QtCore.QEvent.MouseButtonRelease, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
assert strip.selected_items() == () and lit() == set()
view.close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
'''
    )


def test_a_spacer_card_is_narrow_named_by_the_editor_and_holds_every_dac(run_qt) -> None:
    """A spacer takes the period card narrowed: its name shown where every
    card shows its name but not typable, its duration and unit editable,
    its channel circles without labels, and each DAC one disabled Hold."""

    run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseScheduleView
from zlc_ui.pulse._layout import period_card_width, spacer_card_width
""" + _schedule_source() + r'''
app = ensure_qt_app(["spacer-card"])
spacer = PeriodVM(
    "spacer1", "spacer1", FieldVM("1", can_scan=False, can_api=False), "ms", ("ns", "us", "ms"),
    digital=(("d0", True),), analog=(("a0", "hold", FieldVM("0", editable=False)),), kind="spacer",
)
view = PulseScheduleView()
assert view.set_schedule(replace(vm, periods=(periods[0], spacer, periods[1])))
view.resize(1200, 700); view.show(); app.processEvents()
card = view._cards["spacer1"]
period = view._cards["p1"]
assert card.width() == spacer_card_width() < period_card_width(ports=ports, analog_mode_choices=analog_modes) == period.width()
assert card.name_edit.text() == "spacer1" and not card.name_edit.isEnabled()
assert period.name_edit.isEnabled()
assert card.duration_edit.isEnabled() and card.unit_combo.isEnabled()
assert card.checks["d0"].text() == "" and card.checks["d0"].toolTip() == "Gate"
assert card.checks["d0"].isChecked() and card.checks["d0"].isEnabled()
combo = card.bus_mode_combos["a0"]
assert combo.count() == 1 and combo.currentText() == "Leave unchanged" and not combo.isEnabled()
assert "a0" not in card.bus_value_edits
assert card.title() == "Spacer" and period.title() == "Period 1/2"
view.close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
'''
    )


def test_choosing_config_in_the_slot_popup_changes_a_text_not_the_popups_shape(run_qt) -> None:
    """Names are edited at the field; source changes preserve popup geometry."""

    run_qt(
        '''
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse.scan_line_edit import FluentScanLineEdit

app = ensure_qt_app(["popup-shape"])
edit = FluentScanLineEdit("12")
requested = []
def accept_key(key):
    requested.append(key)
    if key != 'bad name':
        edit.set_field_state(editable=True, source='config', config_key=key)
edit.config_key_committed.connect(accept_key)
edit.set_config_names(('bias', 'saved_only'))
edit.show(); app.processEvents()
edit.set_field_state(editable=True)
edit.binding_button.click()
popup = edit._popup
app.processEvents()
assert popup.isVisible()
assert edit.config_name_label.isVisible() and edit.config_name.isVisible()
assert not edit.config_name.isEnabled() and edit.config_name.currentText() == ''
assert edit.default_value.text() == '12'
assert edit.effective_value.text() == '12 · default'
resting = popup.size()
hint = popup.sizeHint()
edit.set_field_state(editable=True, source="config", config_enabled=True, config_key="a_config_name_far_longer_than_the_switch_row")
app.processEvents()
assert popup.isVisible() and popup.size() == resting and popup.sizeHint() == hint, (resting, popup.size())
assert edit.config_name.toolTip() == "a_config_name_far_longer_than_the_switch_row"
assert edit.config_name.isEnabled() and not requested
assert 'not supplied' in edit.source_note.text()
edit.setText("120")
edit.set_field_state(editable=True, source="config", config_key="bias", effective_text="120 value", config_active=True, config_enabled=True, default_text="12")
app.processEvents()
assert edit.config_name.currentText() == "bias" and popup.size() == resting
assert edit.effective_value.text() == '120 value' and edit.isReadOnly()
assert edit.default_value.text() == '12'
combo = edit.config_name
combo.showPopup()
QtTest.QTest.keyClick(combo.view(), QtCore.Qt.Key_End)
QtTest.QTest.keyClick(combo.view(), QtCore.Qt.Key_Return)
assert requested == ['saved_only'] and popup.isVisible()
combo.lineEdit().setFocus(); combo.lineEdit().selectAll()
QtTest.QTest.keyClicks(combo.lineEdit(), 'custom')
QtTest.QTest.keyClick(combo.lineEdit(), QtCore.Qt.Key_Return)
QtTest.QTest.keyClick(combo.lineEdit(), QtCore.Qt.Key_Tab)
assert requested == ['saved_only', 'custom']
combo.lineEdit().setFocus(); combo.lineEdit().selectAll()
QtTest.QTest.keyClicks(combo.lineEdit(), 'bad name')
QtTest.QTest.keyClick(combo.lineEdit(), QtCore.Qt.Key_Return)
assert combo.currentText() == 'custom', 'rejected text returns to the accepted key'
combo.lineEdit().setFocus(); combo.lineEdit().selectAll()
QtTest.QTest.keyClicks(combo.lineEdit(), 'unfinished')
edit.set_config_names(('new_file_name',))
assert combo.currentText() == 'unfinished', 'catalog refresh preserves the in-progress name'
edit.set_field_state(editable=True, source="api")
app.processEvents()
combo.editingFinished.emit()
assert requested == ['saved_only', 'custom', 'bad name'], 'source projection must not commit stale text'
assert combo.currentText() == '' and not combo.isEnabled() and popup.size() == resting
edit.setText("12")
edit.set_field_state(editable=True, source='config', config_enabled=True)
assert 'unassigned' in edit.source_note.text()
assert edit.effective_value.text() == '12 · default'
assert combo.currentText() == '' and tuple(combo.itemText(i) for i in range(combo.count())) == ('new_file_name',)
from zlc_ui.pulse.models import FieldVM
from zlc_ui.pulse.schedule_view import _apply_field
_apply_field(edit, FieldVM('24', source='config'))
assert edit.default_value.text() == '24' and edit.effective_value.text() == '24 · default'
_apply_field(edit, FieldVM('36', source='config'))
assert edit.default_value.text() == '36' and edit.effective_value.text() == '36 · default'
popup.hide()
'''
    )


def test_a_bracket_post_shows_a_count_of_ten_thousand_whole(run_qt) -> None:
    """A post is as wide as five digits of count and no wider.  Cut to a
    fixed 78 px it refused 10000 as too narrow; sized by the box's generic
    hint it was nearly a card."""

    run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse.schedule_view import BracketPost
from zlc_ui.pulse._layout import card_pad

app = ensure_qt_app(["post-width"])
post = BracketPost("b", "end", count=10000)
post.show(); app.processEvents()
assert post.width() == post.count_spin.width() + 2 * card_pad()
assert post.count_edit.contentsRect().width() - post.count_edit.textMargins().right() >= post.count_edit.fontMetrics().horizontalAdvance("10000")
assert post.count_edit.text() == "10000" and not post.count_edit.property("numericError")
post.close(); post.deleteLater()
"""
    )


def test_a_bracket_count_wears_its_binding_without_changing_the_post_width(run_qt) -> None:
    """The repeat box is a bound field like any other: its frame -- the
    spin box's, since the adopted edit draws none -- wears the binding's
    edge (Config green, Scan orange, API violet) and a plain count the
    plain field's.  The badge's room is reserved whatever the binding, so
    binding a count never moves the post; only a longer count widens it."""

    run_qt(
        """
from dataclasses import replace
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import API_VIOLET, CONFIG_GREEN, ORANGE, PLACEHOLDER
from zlc_ui.pulse import FieldVM
from zlc_ui.pulse.schedule_view import BracketPost
from zlc_ui.pulse._layout import card_pad

app = ensure_qt_app(["post-binding"])
plain = FieldVM("4", can_scan=True, validator_kind="int", validator_lo=1,
                validator_hi=(1 << 32) - 1, field_id="b.count")
post = BracketPost("b", "end", count_field=plain)
post.show(); app.processEvents()
width = post.width()
for field, edge in (
    (replace(plain, scan=True), ORANGE),
    (replace(plain, source="config", config_key="n"), CONFIG_GREEN),
    (replace(plain, source="config", config_key="n", config_active=True,
             config_enabled=True, effective_text="7"), CONFIG_GREEN),
    (replace(plain, source="api"), API_VIOLET),
    (plain, None),
):
    post.set_count_field(field); app.processEvents()
    sheet = post.count_spin.styleSheet()
    assert f"solid {edge or PLACEHOLDER}" in sheet, (field, sheet)
    assert "border: none" in post.count_edit.styleSheet(), "the edit draws no second frame"
    assert post.width() == width == post.count_spin.width() + 2 * card_pad(), field
post.set_count_field(replace(plain, text="123456789")); app.processEvents()
assert post.width() > width and post.width() == post.count_spin.width() + 2 * card_pad()
post.close(); post.deleteLater()
"""
    )


def test_components_collapse_expand_and_move_existing_timeline_items(run_qt) -> None:
    run_qt(_schedule_source() + r'''
from dataclasses import replace
from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import BracketVM, ComponentVM, PulseEditorView, PulseEditorHandle
from zlc_ui.pulse.schedule_view import PeriodCard
import json
app = ensure_qt_app(["pulse-components"])
view = PulseEditorView()
handle = PulseEditorHandle(None, view)
third = replace(periods[0], period_id="p3", name="Three")
component = ComponentVM("mot", "MOT", ("p1", "p2"), "16 us", 1, ("b",))
plain = replace(vm, periods=(*periods, third))
handle.set_schedule(plain)
view.resize(1280, 820); view.show(); app.processEvents()
main = view.schedule_view
strip = main.drag_container
actions, deletions, digital = [], [], []
handle.component_action_requested.connect(lambda *args: actions.append(args))
handle.remove_items_requested.connect(deletions.append)
handle.digital_committed.connect(lambda *args: digital.append(args))
for key in ("p1", "p3"):
    QtTest.QTest.mouseClick(main._cards[key], QtCore.Qt.LeftButton,
                           QtCore.Qt.ShiftModifier, QtCore.QPoint(5, 5))
assert strip.selected_items() == (("period", "p1"), ("period", "p3"))
assert not main._cards["p2"].is_selected(), "Shift selection must never fill an unclicked range"
QtTest.QTest.mouseClick(strip, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, QtCore.QPoint(1, 15))
assert strip.selected_items() == (("period", "p1"), ("period", "p3"))
main.group_button.click()
assert actions == [("group", (("period", "p1"), ("period", "p3")))], actions
assert strip.selected_items() == (("period", "p1"), ("period", "p3"))
check = main._cards["p2"].port_rows["d0"]
level = check.isChecked()
for expected in (3, 2):
    QtTest.QTest.mouseClick(check, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier)
    assert len(strip.selected_items()) == expected
    assert check.isChecked() == level and not digital, "Shift selects the item, not its TTL state"
data = QtCore.QMimeData()
data.setData(strip.ITEM_MIME, QtCore.QByteArray(json.dumps(("period", "p1")).encode()))
assert strip._proposal_at(data, len(strip.items())) == (("period", "p2"), ("period", "p1"), ("period", "p3"))
main.remove_button.click()
assert deletions == [(("period", "p1"), ("period", "p3"))]
actions.clear()
vm = replace(plain, revision=3, brackets=(BracketVM("b", "p1", "p2", 2),), components=(component,))
handle.set_schedule(vm)
app.processEvents()
assert set(main._cards) == {"p3"}, "collapsed contents must not construct period widgets"
assert len(main.drag_container.items()) == 2
assert main.drag_container.flat_order() == vm.item_order
card = main._component_cards[("component", "mot")]
assert card.width() == main._cards["p3"].width()
assert not hasattr(card, "_actions_popup") and card._source_popup is None
card.expand_button.click(); app.processEvents()
assert set(main._cards) == {"p1", "p2", "p3"}
assert main.drag_container.flat_order() == vm.item_order
visible = tuple(strip._item_key(item) for item in strip.items())
assert visible[0] == ("component-start", "mot")
assert visible[-2] == ("component-end", "mot")
assert all(item.is_selected() for item in strip.items()[:-1])
assert not main._cards["p3"].is_selected()
assert strip.selected_items() == vm.item_order[:-1]
QtTest.QTest.mouseClick(main._cards["p1"], QtCore.Qt.LeftButton, pos=QtCore.QPoint(5, 5))
main.bracket_button.click()
assert actions[-1] == ("bracket_add", ("mot", ("p1", "p1", vm.default_bracket_count))), actions
outer_brackets = []
handle.bracket_add_requested.connect(lambda *args: outer_brackets.append(args))
strip.show_selection(vm.item_order[:-1])
start_cap = strip.items()[0]
QtTest.QTest.mouseClick(start_cap, QtCore.Qt.LeftButton,
                       QtCore.Qt.ShiftModifier, QtCore.QPoint(5, 5))
assert strip.selection_payload() == (("component", "mot"),)
main.bracket_button.click()
assert outer_brackets == [("p1", "p2", vm.default_bracket_count)], outer_brackets
QtTest.QTest.mouseClick(start_cap.name_edit, QtCore.Qt.LeftButton)
start_cap.name_edit.selectAll()
QtTest.QTest.keyClicks(start_cap.name_edit, "MOT revised")
QtTest.QTest.keyClick(start_cap.name_edit, QtCore.Qt.Key_Return)
assert actions[-1] == ("rename", ("mot", "MOT revised")), actions
vm = replace(vm, revision=4, components=(replace(component, name="MOT revised"),))
handle.set_schedule(vm)
actions.clear()
edits = []
handle.duration_committed.connect(lambda *args: edits.append(args))
main._cards["p1"].duration_edit.setText("5")
main._cards["p1"].duration_edit.editingFinished.emit()
assert edits == [("p1", 5.0, "us")], edits
data = QtCore.QMimeData()
data.setData(main.drag_container.ITEM_MIME, QtCore.QByteArray(json.dumps(("component-start", "mot")).encode()))
assert main.drag_container._proposal_at(data, len(main.drag_container.items())) == (("period", "p3"), *vm.item_order[:-1])
start_cap.expand_button.click(); app.processEvents()
assert set(main._cards) == {"p3"}
data.setData(main.drag_container.ITEM_MIME, QtCore.QByteArray(json.dumps(("component", "mot")).encode()))
assert main.drag_container._proposal_at(data, 2) == (("period", "p3"), *vm.item_order[:-1])
card = main._component_cards[("component", "mot")]
buttons = (card.expand_button, card.edit_button, card.ungroup_button)
assert tuple(button.text() for button in buttons) == ("Expand", "Edit", "Ungroup")
assert len({button.width() for button in buttons}) == 1
centers = [button.mapTo(card, button.rect().center()).x() for button in buttons]
assert len(set(centers)) == 1
tops = [button.mapTo(card, QtCore.QPoint()).y() for button in buttons]
assert tops[0] < tops[1] < tops[2]
card.ungroup_button.click(); app.processEvents()
assert actions == [("ungroup", "mot")], actions
actions.clear()
main.drag_container.show_selection((("component", "mot"),))
main.remove_button.click()
assert deletions[-1] == (("component", "mot"),), deletions
local = replace(vm, document_generation=2, periods=periods, components=())
handle.set_component_document(local, contexts=(("mot", "MOT · current Pulse"),), context_id="mot")
view.tabs.setCurrentWidget(view.component_view); app.processEvents()
assert view.component_view._buttons["save"].text() == "Save Pulse"
assert not view.component_view.schedule_view.run_button.isVisible()
assert not view.component_view.schedule_view.names_panel.name_edit.isVisible()
assert not view.component_view.schedule_view.names_panel._rows["d0"].isEnabled()
assert view.component_view.schedule_view.names_panel._rows["d0"].isReadOnly()
assert main.names_panel._rows["d0"].isEnabled()
assert isinstance(view.component_view.schedule_view._cards["p1"], PeriodCard)
events = []
handle.component_edit_requested.connect(lambda *args: events.append(args))
view.component_view.schedule_view._cards["p1"].name_edit.setText("Load")
view.component_view.schedule_view._cards["p1"].name_edit.editingFinished.emit()
assert events == [("period_name", ("p1", "Load"))], events
view.tabs.setCurrentWidget(main)
assert not hasattr(view, "component_inspector")
assert main.channel_panel_holder.isVisible() and main.names_panel_holder.isVisible()
assert view.tabs.tabText(view.tabs.indexOf(view.component_view)) == "Subpulse Config"
modes, refreshes = [], []
handle.config_mode_committed.connect(modes.append)
handle.refresh_effective_requested.connect(lambda: refreshes.append(True))
handle.set_config_mode(False)
assert main.channel_panel.config_mode.text() == ""
main.channel_panel.config_mode.click()
assert modes == [True]
handle.set_config_mode(True)
assert main.channel_panel.config_mode.text() == ""
main.channel_panel.config_refresh_button.click()
assert refreshes == [True] and modes == [True]
card = main._component_cards[("component", "mot")]
assert card.title() == "Subpulse"
# The duration row holds the total and the source badge, both inside the
# card: the badge has no field frame here and sizes itself.
assert card.source_button.isVisible() and card.total_label.isVisible()
assert 0 < card.source_button.width() < card.width() // 4
assert card.total_label.width() > card.total_label.fontMetrics().horizontalAdvance("0 ms")
assert card.source_button.mapTo(card, card.source_button.rect().bottomRight()).x() < card.width()
configured = replace(component, source="config", config_path="cooling.subpulse.json")
vm = replace(vm, revision=5, components=(configured,))
handle.set_schedule(vm)
card.source_button.click(); app.processEvents()
from zlc_ui.fluent import FluentButton, FluentSegmentedButtons
assert card._source_popup.isVisible() and card.source_switch.current() == "config"
assert isinstance(card.source_switch, FluentSegmentedButtons), "the same Source control as a value slot's"
assert not card._source_popup.findChildren(QtWidgets.QLineEdit), "source popup only selects Default/Config"
assert not card._source_popup.findChildren(FluentButton), "a choice, not commands"
popup_size = card._source_popup.size()
card.source_switch.button("default").click()
assert actions[-1] == ("source", ("mot", "default", "cooling.subpulse.json"))
assert card.source_switch.current() == "config", "rejected source intent must restore its accepted state"
count = len(actions)
card.set_component(replace(configured, source="default"))
assert len(actions) == count and card.source_switch.current() == "default"
assert card._source_popup.size() == popup_size
card.source_switch.button("config").click()
assert actions[-1] == ("source", ("mot", "config", "cooling.subpulse.json"))
assert card.source_switch.current() == "default"
card.source_button.click()
assert not card._source_popup.isVisible()
# A popup holds the mouse while open, so the operator's click outside it --
# on the slot that opened it, say -- is delivered TO the popup, which closes
# itself.  A re-render re-watches the card for selection and drag; that
# watch must leave the card's popups (other windows) alone, or the press is
# taken as a card click and the popup can never be closed again.
from PyQt5 import QtGui
card.source_button.click(); app.processEvents()
assert card._source_popup.isVisible()
main.drag_container.watch_item_chrome(card)
outside = QtCore.QPoint(-4, -4)
QtWidgets.QApplication.sendEvent(card._source_popup, QtGui.QMouseEvent(
    QtCore.QEvent.MouseButtonPress, QtCore.QPointF(outside),
    QtCore.QPointF(card._source_popup.mapToGlobal(outside)),
    QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
app.processEvents()
assert not card._source_popup.isVisible(), "a re-watched card swallowed its popup's outside press"
effective_periods = tuple(replace(period, readonly=True, duration=replace(
    period.duration, config_active=True, config_enabled=True, binding_editable=False,
    default_text=period.duration.text)) for period in periods)
vm = replace(vm, revision=6, periods=(*effective_periods, third),
             brackets=tuple(replace(bracket, readonly=True) for bracket in vm.brackets),
             components=(replace(configured, config_active=True),))
handle.set_schedule(vm)
from zlc_ui.fluent import CONFIG_GREEN_TINT
assert card._zlc_surface[0] == CONFIG_GREEN_TINT
card.edit_button.click()
assert card.edit_button.text() == "Edit" and actions[-1] == ("edit", "mot")
view.tabs.setCurrentWidget(main)
card.expand_button.click(); app.processEvents()
first = main._cards["p1"]
assert first.duration_edit.isReadOnly() and first.duration_edit.isEnabled()
assert not first.duration_edit.binding_button.isEnabled()
assert not first.checks["d0"].isEnabled() and not first.name_edit.isEnabled(), "read-only content is switched off, like its combos"
assert not first.unit_combo.isEnabled()
assert first.bus_value_edits["a0"].isReadOnly() and not first.bus_mode_combos["a0"].isEnabled()
blocked = []
handle.feedback_requested.connect(blocked.append)
strip.show_selection((("period", "p1"),))
before = len(deletions)
main.remove_button.click()
QtTest.QTest.keyClick(strip, QtCore.Qt.Key_Delete)
assert len(deletions) == before and blocked
data.setData(strip.ITEM_MIME, QtCore.QByteArray(json.dumps(("period", "p1")).encode()))
assert strip._proposal_at(data, len(strip.items())) is None
data.setData(strip.ITEM_MIME, QtCore.QByteArray(json.dumps(("component-start", "mot")).encode()))
assert strip._proposal_at(data, len(strip.items())) is not None
view.finish_close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
''')
