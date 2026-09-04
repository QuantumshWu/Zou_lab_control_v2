from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Every snippet starts here.  Without it the subprocess resolves the
#: layers through whatever the editable install points at -- on this
#: machine, sibling checkouts of the same package names -- so the suite
#: silently tested a DIFFERENT zlc_plot than the one beside it.  The
#: product bootstrap is what puts this checkout's layers on the path,
#: and it is the same one every launcher uses.
_BOOTSTRAP = "import zou_lab_control" + chr(10)


def _run_qt(code: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        ""
        if environment.get("ZLC_TEST_INSTALLED") == "1"
        else os.pathsep.join((str(ROOT.parents[1]), str(SRC)))
    )
    environment["QT_QPA_PLATFORM"] = "offscreen"
    # Fixed for the child, not inherited: a matplotlib backend chosen by
    # whatever imported it in the parent is how this harness produced access
    # violations at teardown that had nothing to do with the code under test.
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [sys.executable, "-c", _BOOTSTRAP + code], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


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


def test_embedded_connection_is_named_and_locked_in_the_real_schedule_view() -> None:
    _run_qt(
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


def test_schedule_stale_rejection_and_widget_reuse() -> None:
    _run_qt(
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
'''
    )


def test_schedule_projects_scan_and_api_bindings_into_fields() -> None:
    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import FieldVM, PeriodVM, PortRowVM, PulseScheduleView, ScheduleVM, DelayRowVM
app = ensure_qt_app(["pulse-bindings"])
field_scan = FieldVM("s0", binding_kind="scan", binding_number=1)
field_api = FieldVM("0", binding_kind="api", binding_number=2)
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
    delay_rows=(DelayRowVM("d0", field_api, "ns", (("ns", 1.0),)),),
    scan_summary_text="1 slot · 4 pts",
)
view = PulseScheduleView(); view.set_schedule(vm)
scan_edit = view._cards["p1"].duration_edit
api_edit = view._cards["p2"].duration_edit
delay_edit = view.channel_panel._rows["d0"][0]
assert scan_edit.dot.isChecked() and scan_edit.dot._number == 1
assert not api_edit.dot.isChecked() and api_edit.dot._kind == "api" and api_edit.dot._number == 2
assert not delay_edit.dot.isChecked() and delay_edit.dot._kind == "api" and delay_edit.dot._number == 2
assert view.channel_panel.scan_summary_label.text() == "1 slot · 4 pts"
"""
    )


def test_pulse_binding_dots_emit_intents_without_mutating_view_state() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from examples.demo_pulse_editor import create_window

app = ensure_qt_app(["pulse-binding-cycle"])
# Through the demo's own entry, which is the outside entry: what comes back
# is the handle.  Reaching its view is this package's own business -- the
# clicks below are on real widgets, which is the point of the test.
editor = create_window(window_ratio=0.4)
app.processEvents()
schedule = editor._view.schedule_view
requested = []
schedule.binding_cycle_requested.connect(lambda *payload: requested.append(payload))

before = schedule._schedule
duration = schedule._cards["p1"].duration_edit
dac = schedule._cards["p1"].bus_value_edits["da_bias_y"]
delay = schedule.channel_panel._rows["ch01"][0]
for dot in (duration.dot, dac.dot, delay.dot):
    QtTest.QTest.mouseClick(dot, QtCore.Qt.LeftButton)
app.processEvents()

assert requested == [
    ("duration", "p1", None),
    ("analog", "p1", "da_bias_y"),
    ("delay", None, "ch01"),
]
assert schedule._schedule == before
assert duration.binding_kind == "scan"
assert dac.binding_kind == "scan"
assert delay.binding_kind is None

# This test owns an isolated QApplication subprocess; letting Qt tear down
# with the process avoids an intermittent PyQt5/offscreen native crash in
# the frameless editor's explicit close path after the assertions pass.
"""
    )


def test_dropping_a_period_proposes_a_move_and_moves_nothing_itself() -> None:
    """The drag is a real QDrag now, so the DROP is what the test drives.

    It used to be inferred at release from where the button happened to come
    up, with nothing on screen to aim at and no move handling at all -- a
    gesture that clearly went somewhere and did nothing.  The strip still
    reorders nothing by itself: it proposes, and the presenter decides.
    """

    _run_qt(
        """
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseScheduleView
""" + _schedule_source() + r'''
app = ensure_qt_app(["pulse-drag"])
view = PulseScheduleView(); view.set_schedule(vm); view.show(); app.processEvents()
events = []
view.move_period_requested.connect(lambda period, before: events.append((period, before)))
strip = view.drag_container

def drop(period_id, x):
    data = QtCore.QMimeData()
    data.setData(strip.CARD_MIME, QtCore.QByteArray(period_id.encode("utf-8")))
    event = QtGui.QDropEvent(
        QtCore.QPoint(x, 5), QtCore.Qt.MoveAction, data,
        QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    )
    strip.dropEvent(event)

cards = strip.pulse_cards()
# Dropped left of the first card: it goes before p1.
drop("p2", cards[0].geometry().left())
assert events == [("p2", "p1")], events

# Dropped past the last card: it goes to the end, which is "before nothing".
events.clear()
drop("p1", cards[-1].geometry().right() + 40)
assert events == [("p1", None)], events

# Dropped on itself proposes nothing at all.
events.clear()
drop("p1", cards[0].geometry().center().x())
assert events == [], events

# And the strip reorders nothing on its own.
assert tuple(item.period_id for item in strip.pulse_cards()) == ("p1", "p2")

# A press and release that never moved is a CLICK, which selects.
picked = []
strip.period_clicked.connect(picked.append)
for kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease):
    QtWidgets.QApplication.sendEvent(cards[1], QtGui.QMouseEvent(
        kind, QtCore.QPoint(5, 5), QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
assert picked == ["p2"], picked

# Close before teardown.  A synthetic QDropEvent built in Python and a widget
# tree torn down by the interpreter's exit race each other, and the process
# dies with an access violation AFTER every assertion has already passed.
view.close()
app.processEvents()
'''
    )


def test_scan_text_is_immediate_state_intent_and_run_has_no_payload() -> None:
    _run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseScanView, PulseTargetView, ScanPageRecord, TargetPortRecord, TargetWidthRule
app = ensure_qt_app(["pulse-pages"])
scan = PulseScanView(); scan.set_page(ScanPageRecord(source_text="committed", source_dirty=False))
edited = []; runs = []; scan.source_edited.connect(edited.append); scan.run_requested.connect(lambda: runs.append("run"))
scan.scan_code.setPlainText("typing"); scan.scan_run_button.click()
assert edited == ["typing"] and runs == ["run"]
assert not hasattr(scan, "_code_dirty") and not hasattr(scan, "_source_revision")
scan.set_repeats((1 << 32) - 1)
assert int(scan.scan_repeats_spin.maximum()) == (1 << 32) - 1
assert scan.scan_repeats_spin.text() == str((1 << 32) - 1)
target = PulseTargetView(); target.set_width_rules(TargetWidthRule(1, 1, 1), TargetWidthRule(1, 2, 4))
target.set_ports((TargetPortRecord("d0", "digital", "Gate", ("d0",), lane_order=(0,)),), True, "ready")
events = []; target.apply_requested.connect(events.append)
target.show(); app.processEvents(); QtTest.QTest.mouseClick(target.apply_button, QtCore.Qt.LeftButton)
assert events and isinstance(events[-1][0], TargetPortRecord)
"""
    )


def test_preview_mount_and_demo_smoke() -> None:
    _run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulsePreviewView
app = ensure_qt_app(["preview"]); view = PulsePreviewView(); first = QtWidgets.QLabel("one"); second = QtWidgets.QLabel("two")
view.mount_content(first, logical_size=(100, 60)); view.mount_content(second, logical_size=(120, 70)); assert view._content_widget is second and first.parentWidget() is None
"""
    )
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "examples/demo_pulse_editor.py", "--once"],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "scan_hold_requested" in completed.stdout
    assert "save_requested" in completed.stdout


def test_linked_panes_show_one_bar_only_while_some_pane_overflows() -> None:
    """The group decides "is there more below" ONCE, from the deepest pane.

    Two synchronized bars answer it twice, and the answers diverge the moment
    one column is taller: whichever bar was elected visible then caps how far
    the operator can reach, and rows past that cap are unreachable.
    """

    _run_qt(
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


def test_schedule_channel_column_is_reachable_without_any_periods() -> None:
    """A board's channels scroll even when the timeline beside them is empty."""

    _run_qt(
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
        DelayRowVM(port.key, FieldVM("0", editable=False), "ns", (("ns", 1.0),))
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


def test_clicking_a_period_or_a_gap_decides_where_the_next_one_lands() -> None:
    """Selection is what makes a sequence buildable at all.

    A selected gap inserts there, a selected card inserts AFTER it,
    Remove takes the selected one, and clicking the current selection again
    clears it.  Here ``_selected_before_id`` returned None unconditionally and
    ``gap_clicked`` was declared but never emitted, so a period could only ever
    be appended and Remove could only ever take the last -- an operator had no
    way to say "here".
    """

    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import FieldVM, PeriodVM, PortRowVM, PulseScheduleView, ScheduleVM
app = ensure_qt_app(["schedule-selection"])

port = PortRowVM("d0", "digital", "Gate", "d0")
periods = tuple(
    PeriodVM(f"p{n}", f"P{n}", FieldVM("1"), "us", ("ns", "us"), digital=(("d0", False),))
    for n in range(3)
)
view = PulseScheduleView()
view.resize(1200, 500); view.show()
view.set_schedule(ScheduleVM(
    document_generation=0, revision=0, document_name="x", clock_text="50 MHz",
    total_text="3 us", total_tooltip="", period_count=3, visible_text="1/1",
    summary_text="x", ports=(port,), periods=periods,
))
app.processEvents()
container = view.drag_container

asked = []
view.insert_period_requested.connect(lambda before: asked.append(("add", before)))
view.remove_period_requested.connect(lambda pid: asked.append(("remove", pid)))

# nothing selected -> append, and Remove takes the last
view.add_button.click(); view.remove_button.click()
assert asked == [("add", None), ("remove", "p2")], asked

# a selected CARD inserts after it, and Remove takes that one
asked.clear()
container.period_clicked.emit("p0"); app.processEvents()
assert container.selection() == ("p0", None, None)
view.add_button.click(); view.remove_button.click()
assert asked == [("add", "p1"), ("remove", "p0")], asked

# a selected GAP inserts there, and the two selections are exclusive
asked.clear()
container.gap_clicked.emit(0); app.processEvents()
assert container.selection() == (None, None, 0), container.selection()
view.add_button.click()
assert asked == [("add", "p0")], asked

# clicking the current selection again clears it
container.gap_clicked.emit(0); app.processEvents()
assert container.selection() == (None, None, None)

# a selection naming a period the strip no longer holds cannot survive a rebuild
container.period_clicked.emit("p1"); app.processEvents()
view.set_schedule(ScheduleVM(
    document_generation=0, revision=1, document_name="x", clock_text="50 MHz",
    total_text="1 us", total_tooltip="", period_count=1, visible_text="1/1",
    summary_text="x", ports=(port,), periods=periods[:1],
))
app.processEvents()
assert container.selection() == (None, None, None), container.selection()
"""
    )


def test_a_board_locks_its_wiring_and_never_the_names() -> None:
    """Two permissions, not one.

    A board owns which lanes exist, how wide a bus is and which pins they
    reach.  What a signal is CALLED is display metadata belonging to whoever
    reads it, and the page says so in its own status line -- "rename freely,
    the topology is the board's".  One flag drove both, so attached the rename
    box was greyed out and Apply with it: the page told the operator to do
    something it had just disabled.
    """

    _run_qt(
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
"""
    )


def test_hiding_a_port_takes_its_delay_row_with_it() -> None:
    """One visible-row list, three columns.

    ``_display_rows`` -- already filtered by ``visible_ports`` -- feeds
    the names column, the delay column and every period card, so the three read
    across as one table by construction.  Here the delay rows arrived as "every
    output the board can delay" and were handed over unfiltered, so Hide Off
    dropped rows from the cards and left their delays behind, aligned with
    whatever happened to be next to them.
    """

    _run_qt(
        """
from PyQt5 import QtCore
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import DelayRowVM, FieldVM, PeriodVM, PortRowVM, PulseScheduleView, ScheduleVM
app = ensure_qt_app(["schedule-hide-delay"])

ports = tuple(PortRowVM(f"d{n}", "digital", f"Out {n}", f"d{n}") for n in range(3))
view = PulseScheduleView()
view.resize(1100, 460); view.show()
view.set_schedule(ScheduleVM(
    document_generation=0, revision=0, document_name="pulse",
    clock_text="50 MHz", total_text="1 us", total_tooltip="", period_count=1,
    visible_text="3/3", summary_text="",
    ports=ports,
    periods=(PeriodVM("p0", "P0", FieldVM("1"), "us", ("ns", "us"),
                      digital=tuple((port.key, False) for port in ports)),),
    delay_rows=tuple(
        DelayRowVM(port.key, FieldVM("0"), "ns", (("ns", 1.0),)) for port in ports
    ),
))
app.processEvents()
assert set(view.channel_panel._rows) == {"d0", "d1", "d2"}

view.set_visible_ports(("d0", "d2"))
app.processEvents()
assert set(view.channel_panel._rows) == {"d0", "d2"}, "a hidden port keeps no delay row"
assert set(view._cards["p0"].port_rows) == {"d0", "d2"}, "and the card agrees"

# The single-row push is the other way into that column, and it must obey the
# same rule -- otherwise one API binding brings a hidden row back.
view.set_delay_row(DelayRowVM("d1", FieldVM("7"), "ns", (("ns", 1.0),)))
app.processEvents()
assert set(view.channel_panel._rows) == {"d0", "d2"}

view.set_visible_ports(("d0", "d1", "d2"))
app.processEvents()
assert set(view.channel_panel._rows) == {"d0", "d1", "d2"}, "Show All brings it back"
view.close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete); app.processEvents()
"""
    )


def test_the_bracket_posts_are_built_to_frame_the_cards_they_span() -> None:
    """The bracket art carries the repeat meaning.

    Two untitled posts the same height as a period card, each with "Bracket" on
    the cards' own header line and the count on their first control line, so
    the pair reads as a frame drawn around the span.  It had become a titled
    box with a glyph in it at whatever height the layout gave -- the thing
    marking a span lined up with nothing inside the span.
    """

    _run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import BracketPost, BracketVM, PulseScheduleView
from zlc_ui.pulse._layout import panel_top_height
""" + _schedule_source() + r'''
app = ensure_qt_app(["repeat-art"])

start = BracketPost("start", minimum=2)
end = BracketPost("end", count=4, minimum=2)
for post in (start, end):
    assert post.title() == "", "an untitled column, like the cards it frames"
    assert post.sizePolicy().verticalPolicy() == QtWidgets.QSizePolicy.Expanding
    labels = [w.text() for w in post.findChildren(QtWidgets.QLabel)]
    assert "Bracket" in labels, labels
    top = post.findChildren(QtWidgets.QWidget)[0]
    assert top.height() == panel_top_height(), (top.height(), panel_top_height())
assert start.width() == end.width(), "the two posts match"

# The count closes the span, so it lives on the end post only -- but both
# posts keep the line, or they would not stay level with each other.
start.show(); end.show(); app.processEvents()
assert not start.count_spin.isVisible()
assert end.count_spin.isVisible() and end.count_spin.value() == 4
assert end.count_spin.minimum() == 2, "once is not a repeat"
assert int(end.count_spin.maximum()) == (1 << 32) - 1
assert end.count_spin.text() == "4"

# In the strip, the posts stand beside cards of the same build.
view = PulseScheduleView(); view.set_schedule(vm); view.show(); app.processEvents()
view.bracket_committed.emit("p1", "p2", 3)

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
view.bracket_committed.connect(lambda *payload: requested.append(payload))
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


def test_bracket_posts_drop_on_period_gaps_and_run_repeats_uses_uint32() -> None:
    """Both independent repeat layers are editable on the real Edit page."""

    _run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import BracketPost, BracketVM, PulseScheduleView
import zlc_ui.pulse.schedule_view as schedule_module
""" + _schedule_source() + r'''
app = ensure_qt_app(["bracket-drag-and-run-repeats"])
view = PulseScheduleView()
bracketed = replace(
    vm,
    revision=3,
    bracket=BracketVM("p1", "p1", 4),
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
QtWidgets.QApplication.sendEvent(start_post, QtGui.QMouseEvent(
    QtCore.QEvent.MouseButtonPress, QtCore.QPoint(5, 5), QtCore.Qt.LeftButton,
    QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
))
QtWidgets.QApplication.sendEvent(start_post, QtGui.QMouseEvent(
    QtCore.QEvent.MouseMove,
    QtCore.QPoint(5 + QtWidgets.QApplication.startDragDistance() + 1, 5),
    QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
))
schedule_module.QtGui.QDrag = real_drag
assert len(started) == 1
assert started[0].data.hasFormat(strip.BRACKET_MIME)
assert bytes(started[0].data.data(strip.BRACKET_MIME)).decode("utf-8") == (
    "\0".join(("start", "p1", "p1", "4"))
)

def drop(kind, start, end, count, x):
    data = QtCore.QMimeData()
    payload = "\0".join((kind, start, end, str(count))).encode("utf-8")
    data.setData(strip.BRACKET_MIME, QtCore.QByteArray(payload))
    event = QtGui.QDropEvent(
        QtCore.QPoint(x, 5), QtCore.Qt.MoveAction, data,
        QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    )
    strip.dropEvent(event)
    return event.isAccepted()

cards = strip.pulse_cards()
assert drop("end", "p1", "p1", 4, cards[-1].geometry().right() + 40)
assert bracket_requests == [("p1", "p2", 4)]
assert view._schedule.bracket == bracketed.bracket, "a drop is proposal-only"

# Re-project the accepted span, then move its left post to the first gap.
bracket_requests.clear()
right_only = replace(
    bracketed,
    revision=4,
    bracket=BracketVM("p2", "p2", 4),
)
assert view.set_schedule(right_only)
app.processEvents()
cards = strip.pulse_cards()
assert drop("start", "p2", "p2", 4, cards[0].geometry().left())
assert bracket_requests == [("p1", "p2", 4)]
assert view._schedule.bracket == right_only.bracket

# A post cannot cross its partner; an illegal gap commits nothing.
bracket_requests.clear()
assert not drop("start", "p2", "p2", 4, cards[-1].geometry().right() + 40)
assert bracket_requests == []

view.close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete); app.processEvents()
app.quit()
'''
    )


def test_a_config_binding_wears_its_own_colour_and_stays_editable() -> None:
    """Three owners, three marks, and only the board's one takes the box away.

    A scan column is written per point by the board, so its field is read
    only.  An API parameter and a config parameter both hold a number the
    operator can see and type, so theirs are not -- and the config one is
    slate rather than violet, because nobody supplies it for a run: it is
    already the pulse's own.
    """

    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import API_VIOLET, CONFIG_GREEN, ORANGE
from zlc_ui.pulse.scan_line_edit import FluentScanLineEdit
from zlc_ui.pulse.scan_line_edit import _BINDING_FILL

app = ensure_qt_app(["binding-colours"])
assert _BINDING_FILL == {"scan": ORANGE, "api": API_VIOLET, "config": CONFIG_GREEN}

edit = FluentScanLineEdit("12")
edit.set_field_state(editable=True, binding="config", number=3)
assert edit.dot._kind == "config"
assert edit.dot._number == 3
assert not edit.dot.isChecked(), "only a scan column marks the box as the board's"
assert not edit.isReadOnly(), "a config number is the operator's to read and type"
assert CONFIG_GREEN in edit.styleSheet()

edit.set_field_state(editable=True, binding="scan", number=1)
assert edit.isReadOnly() and edit.dot.isChecked()

edit.set_field_state(editable=True, binding=None)
assert edit.dot._kind is None and not edit.isReadOnly()

try:
    edit.set_field_state(editable=True, binding="whatever")
except ValueError as error:
    assert "scan" in str(error) and "config" in str(error)
else:
    raise AssertionError("an unknown binding must be refused")

edit.deleteLater()
app.processEvents()
"""
    )



def test_linked_panes_never_starve_a_pane_to_line_the_group_up() -> None:
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

    _run_qt(
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


def test_a_card_and_a_post_are_one_gesture_with_two_payloads() -> None:
    """Dragging offers only the gaps that would do something -- for both.

    The bracket post refused an impossible gap while the cursor was still
    over it: no marker, no drop cursor.  The card accepted every gap, drew
    the marker in all of them, and then threw away a drop onto the two that
    mean "where it already is".  Same gesture, opposite answers to the same
    question, and the half that says yes and does nothing is the worse half.
    """

    _run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import BracketVM, PulseScheduleView
""" + _schedule_source() + r'''
app = ensure_qt_app(["drag-symmetry"])
view = PulseScheduleView()
assert view.set_schedule(replace(vm, revision=3, bracket=BracketVM("p1", "p1", 4)))
view.show(); app.processEvents()
strip = view.drag_container
cards = strip.pulse_cards()

def hover(mime, payload, x):
    """One dragMoveEvent, and what the strip decided about it."""
    data = QtCore.QMimeData()
    data.setData(mime, QtCore.QByteArray(payload.encode("utf-8")))
    event = QtGui.QDragMoveEvent(
        QtCore.QPoint(x, 5), QtCore.Qt.MoveAction, data,
        QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    )
    strip.dragMoveEvent(event)
    return event.isAccepted(), strip._indicator.isVisible()

on_itself = cards[0].geometry().center().x()
elsewhere = cards[-1].geometry().right() + 40

card = strip.CARD_MIME
post = strip.BRACKET_MIME

# A move that would change nothing is refused WHILE dragging, not after.
assert hover(card, "p1", on_itself) == (False, False)
assert hover(post, "\0".join(("end", "p1", "p1", "4")), on_itself) == (False, False)

# A move that would do something is offered, and shows where it lands.
assert hover(card, "p1", elsewhere) == (True, True)
assert hover(post, "\0".join(("end", "p1", "p1", "4")), elsewhere) == (True, True)

# And what the marker offered is what the drop commits -- for both.
moves, brackets = [], []
view.move_period_requested.connect(lambda *payload: moves.append(payload))
view.bracket_committed.connect(lambda *payload: brackets.append(payload))

def drop(mime, payload, x):
    data = QtCore.QMimeData()
    data.setData(mime, QtCore.QByteArray(payload.encode("utf-8")))
    event = QtGui.QDropEvent(
        QtCore.QPoint(x, 5), QtCore.Qt.MoveAction, data,
        QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    )
    strip.dropEvent(event)
    return event.isAccepted()

assert drop(card, "p1", on_itself) is False and moves == []
assert drop(post, "\0".join(("end", "p1", "p1", "4")), on_itself) is False
assert brackets == []
assert drop(card, "p1", elsewhere) is True and moves == [("p1", None)]
assert drop(post, "\0".join(("end", "p1", "p1", "4")), elsewhere) is True
assert brackets == [("p1", "p2", 4)]

view.close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
app.quit()
'''
    )


def test_clicking_a_post_marks_it_the_way_clicking_a_card_does() -> None:
    """A card and a bracket post are the same kind of thing to pick up.

    Clicking a card drew a border round it; clicking a post drew nothing at
    all, so the two read as different kinds of object when the only real
    difference between them is what they carry.  BracketPost has been a
    FluentGroupBox since it was written -- the outline it needed was already
    on it, and nobody had ever called it.
    """

    _run_qt(
        """
from dataclasses import replace
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import BracketPost, BracketVM, PulseScheduleView
""" + _schedule_source() + r'''
app = ensure_qt_app(["click-symmetry"])
view = PulseScheduleView()
assert view.set_schedule(replace(vm, revision=3, bracket=BracketVM("p1", "p2", 4)))
view.show(); app.processEvents()
strip = view.drag_container
cards = strip.pulse_cards()
posts = {post.kind: post for post in strip.findChildren(BracketPost)}

def click(widget):
    for kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease):
        QtWidgets.QApplication.sendEvent(widget, QtGui.QMouseEvent(
            kind, QtCore.QPoint(5, 5), QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
    app.processEvents()

def outlined():
    """Everything currently wearing the selection border."""
    return {item.period_id for item in cards if item.outline_colour() is not None} | {
        f"post:{kind}" for kind, post in posts.items()
        if post.outline_colour() is not None
    }

click(cards[0])
assert strip.selection() == ("p1", None, None), strip.selection()
assert outlined() == {"p1"}, outlined()

# Clicking a post marks the post -- and takes the mark off the card, because
# at most one thing is what the next edit acts on.
click(posts["end"])
assert strip.selection() == (None, "end", None), strip.selection()
assert outlined() == {"post:end"}, outlined()

# Clicking the marked one again clears it, exactly as a card does.
click(posts["end"])
assert strip.selection() == (None, None, None), strip.selection()
assert outlined() == set(), outlined()

# And the other way round: a post, then a card.
click(posts["start"])
assert outlined() == {"post:start"}, outlined()
click(cards[1])
assert strip.selection() == ("p2", None, None), strip.selection()
assert outlined() == {"p2"}, outlined()

view.close(); view.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
app.quit()
'''
    )


def test_collapsing_the_left_panels_hands_the_width_to_the_periods() -> None:
    """A QScrollArea does not shrink-wrap.

    ``AdjustToContents`` computes the area's hint once and caches it, and
    hiding the widget's children does not clear that cache -- so Collapse hid
    two panels and the pane stayed its old width, leaving the freed space
    blank between the stub and period cards that never moved.
    """

    _run_qt(
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


def test_the_bottom_bar_is_one_table_of_rows_across_three_cards() -> None:
    """Row two of Control is read beside row two of Connection and Ports.

    They were three private rhythms -- two row heights and two gaps -- so the
    fourth row of Control sat six pixels below the boxes beside it.
    """

    _run_qt(
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
