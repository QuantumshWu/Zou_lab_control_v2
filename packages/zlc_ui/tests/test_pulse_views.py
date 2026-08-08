from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_binding_cycle_contract_is_pure_and_field_specific() -> None:
    from zlc_ui.pulse import cycle_binding_kind

    duration = None
    duration = cycle_binding_kind(duration, field_kind="duration")
    assert duration == "scan"
    duration = cycle_binding_kind(duration, field_kind="duration")
    assert duration == "api"
    assert cycle_binding_kind(duration, field_kind="duration") is None

    assert cycle_binding_kind(None, field_kind="delay") == "api"
    assert cycle_binding_kind("api", field_kind="delay") is None
    try:
        cycle_binding_kind("scan", field_kind="delay")
    except ValueError:
        pass
    else:
        raise AssertionError("output delay must not accept a scan binding")


def _run_qt(code: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    # Fixed for the child, not inherited: a matplotlib backend chosen by
    # whatever imported it in the parent is how this harness produced access
    # violations at teardown that had nothing to do with the code under test.
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def _schedule_source() -> str:
    return r'''
from zlc_ui.pulse import FieldVM, PeriodVM, PortRowVM, ScheduleVM
field = FieldVM("4", validator_kind="int", validator_lo=-8, validator_hi=8)
ports = (PortRowVM("d0", "digital", "Gate", "d0"), PortRowVM("a0", "dac", "Voltage", "a0", lo=-8, hi=8))
periods = (
    PeriodVM("p1", "One", field, "us", ("ns", "us"), digital=(("d0", True),), analog=(("a0", "edge", field),)),
    PeriodVM("p2", "Two", field, "us", ("ns", "us"), digital=(("d0", False),), analog=(("a0", "hold", FieldVM("0", editable=False)),)),
)
vm = ScheduleVM(1, 2, "fake", "50 MHz", "10 us", "total", 2, "2/2", "2 periods", ports, periods)
'''


def test_schedule_stale_rejection_and_widget_reuse() -> None:
    _run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import ScheduleVM, PulseScheduleView
""" + _schedule_source() + r'''
app = ensure_qt_app(["pulse-test"])
view = PulseScheduleView()
assert view.set_schedule(vm)
view.set_capabilities(False, False, True)
assert not view.sync_button.isEnabled()
first = view._cards["p1"]
assert not view.set_schedule(vm)
try:
    view.set_schedule(ScheduleVM(1, 2, "different", vm.clock_text, vm.total_text, vm.total_tooltip, vm.period_count, vm.visible_text, vm.summary_text, vm.ports, vm.periods))
except ValueError:
    pass
else:
    raise AssertionError("same revision must reject a different projection")
assert not view.set_schedule(ScheduleVM(1, 1, vm.document_name, vm.clock_text, vm.total_text, vm.total_tooltip, vm.period_count, vm.visible_text, vm.summary_text, vm.ports, vm.periods))
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
assert not api_edit.dot.isChecked() and api_edit.dot._api and api_edit.dot._number == 2
assert not delay_edit.dot.isChecked() and delay_edit.dot._api and delay_edit.dot._number == 2
assert view.channel_panel.scan_summary_label.text() == "1 slot · 4 pts"
"""
    )


def test_pulse_demo_clicks_cycle_bindings_through_a_fake_presenter() -> None:
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
assert any(port.key == "da_bias_y" and port.kind == "dac" and port.visible for port in schedule._schedule.ports)
assert "da_bias_y" in schedule.channel_panel._rows

duration = schedule._cards["p1"].duration_edit
assert duration.binding_kind == "scan" and duration.text() == "s0"
QtTest.QTest.mouseClick(duration.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert duration.binding_kind == "api" and duration.text() == "1000"
assert schedule._schedule.periods[0].duration.binding_kind == "api"
QtTest.QTest.mouseClick(duration.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert duration.binding_kind is None and duration.text() == "0"
assert schedule._schedule.periods[0].duration.binding_kind == ""
QtTest.QTest.mouseClick(duration.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert duration.binding_kind == "scan" and duration.text() == "s0"

dac = schedule._cards["p1"].bus_value_edits["da_bias_y"]
assert dac.binding_kind == "scan" and dac.text() == "s1"
QtTest.QTest.mouseClick(dac.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert dac.binding_kind == "api" and dac.text() == "500"
assert schedule._cards["p1"].bus_mode_combos["da_bias_y"].currentText() == "Ramp"
QtTest.QTest.mouseClick(dac.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert dac.binding_kind is None and dac.text() == "500"
QtTest.QTest.mouseClick(dac.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert dac.binding_kind == "scan" and dac.text() == "s1"

hold_dac = schedule._cards["p2"].bus_value_edits["da_bias_y"]
assert hold_dac.binding_kind is None
QtTest.QTest.mouseClick(hold_dac.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert hold_dac.binding_kind == "scan"
assert schedule._cards["p2"].bus_mode_combos["da_bias_y"].currentText() == "Edge"
assert schedule._schedule.periods[1].analog[0][1] == "edge"

delay = schedule.channel_panel._rows["ch01"][0]
assert delay.binding_kind is None
QtTest.QTest.mouseClick(delay.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert delay.binding_kind == "api"
assert schedule._schedule.delay_rows[1].value.binding_kind == "api"
QtTest.QTest.mouseClick(delay.dot, QtCore.Qt.LeftButton)
app.processEvents()
assert delay.binding_kind is None
assert schedule._schedule.delay_rows[1].value.binding_kind == ""

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
from PyQt5 import QtWidgets
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

    v1's rule: a selected gap inserts there, a selected card inserts AFTER it,
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
assert container.selection() == ("p0", None)
view.add_button.click(); view.remove_button.click()
assert asked == [("add", "p1"), ("remove", "p0")], asked

# a selected GAP inserts there, and the two selections are exclusive
asked.clear()
container.gap_clicked.emit(0); app.processEvents()
assert container.selection() == (None, 0), container.selection()
view.add_button.click()
assert asked == [("add", "p0")], asked

# clicking the current selection again clears it
container.gap_clicked.emit(0); app.processEvents()
assert container.selection() == (None, None)

# a selection naming a period the strip no longer holds cannot survive a rebuild
container.period_clicked.emit("p1"); app.processEvents()
view.set_schedule(ScheduleVM(
    document_generation=0, revision=1, document_name="x", clock_text="50 MHz",
    total_text="1 us", total_tooltip="", period_count=1, visible_text="1/1",
    summary_text="x", ports=(port,), periods=periods[:1],
))
app.processEvents()
assert container.selection() == (None, None), container.selection()
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

    v1 ran ``_display_rows`` -- already filtered by ``visible_ports`` -- through
    the names column, the delay column and every period card, so the three read
    across as one table by construction.  Here the delay rows arrived as "every
    output the board can delay" and were handed over unfiltered, so Hide Off
    dropped rows from the cards and left their delays behind, aligned with
    whatever happened to be next to them.
    """

    _run_qt(
        """
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
"""
    )


def test_the_repeat_posts_are_built_to_frame_the_cards_they_span() -> None:
    """v1's bracket art, and the art carries the meaning.

    Two untitled posts the same height as a period card, each with "Repeat" on
    the cards' own header line and the count on their first control line, so
    the pair reads as a frame drawn around the span.  It had become a titled
    box with a glyph in it at whatever height the layout gave -- the thing
    marking a span lined up with nothing inside the span.
    """

    _run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse import PulseScheduleView, RepeatBracket
from zlc_ui.pulse._layout import panel_top_height
""" + _schedule_source() + r'''
app = ensure_qt_app(["repeat-art"])

start = RepeatBracket("start", minimum=2)
end = RepeatBracket("end", count=4, minimum=2)
for post in (start, end):
    assert post.title() == "", "an untitled column, like the cards it frames"
    assert post.sizePolicy().verticalPolicy() == QtWidgets.QSizePolicy.Expanding
    labels = [w.text() for w in post.findChildren(QtWidgets.QLabel)]
    assert "Repeat" in labels, labels
    top = post.findChildren(QtWidgets.QWidget)[0]
    assert top.height() == panel_top_height(), (top.height(), panel_top_height())
assert start.width() == end.width(), "the two posts match"

# The count closes the span, so it lives on the end post only -- but both
# posts keep the line, or they would not stay level with each other.
start.show(); end.show(); app.processEvents()
assert not start.count_spin.isVisible()
assert end.count_spin.isVisible() and end.count_spin.value() == 4
assert end.count_spin.minimum() == 2, "once is not a repeat"

# In the strip, the posts stand beside cards of the same build.
view = PulseScheduleView(); view.set_schedule(vm); view.show(); app.processEvents()
view.repeat_committed.emit("p1", "p2", 3)
'''
    )
