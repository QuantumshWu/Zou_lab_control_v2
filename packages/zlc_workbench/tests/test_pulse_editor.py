"""Editing a pulse, with zlc_pulse still deciding what a legal pulse is.

The editor's whole risk is that it becomes a second opinion about what the
hardware can play.  So every test here checks one of two things:

* the projection shows what the sequence actually contains -- lanes, levels,
  durations, delays -- rather than something reassembled from the display;
* an edit that would produce an illegal sequence is refused BY THE MODEL, and
  the editor keeps the last good one instead of holding something that will
  fail at compile time.

Driven against an ordinary three-window JSON pulse, independent of the
Calibration task's template, because a hand-made two-lane sequence would not
exercise a DAC, a clock port, or the 62-lane target the bench actually has.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
from zlc_durable import readable_json_bytes

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_workbench.pulse_editor import (
    PulseEditorPresenter as _PulseEditorPresenter,
    project_schedule,
    replace_sequence,
    timeline_of,
)
from zlc_workbench.pulse_state import PulseEditorState, state_to_tree

from pulse_fixtures import PULSE_NAME, ordinary_imaging_sequence, write_ordinary_pulse


def PulseEditorPresenter(view, sequence=None, **kwargs):
    """Build the real presenter from its one complete authoring record."""

    if kwargs.get("sequencer") is not None and kwargs.get("device_use") is None:
        from zlc_workbench.device_use import DeviceUseCoordinator

        kwargs["device_use"] = DeviceUseCoordinator()

    return _PulseEditorPresenter(
        view,
        PulseEditorState(sequence=sequence),
        **kwargs,
    )


def _run_scan(view, source: str) -> None:
    view.scan_source_edited.emit(str(source))
    view.scan_run_requested.emit()


class _Signal:
    def __init__(self) -> None:
        self._listeners: list = []

    def connect(self, listener) -> None:
        self._listeners.append(listener)

    def emit(self, *args) -> None:
        for listener in list(self._listeners):
            listener(*args)


class _ScheduleView:
    """Every intent the real schedule page raises, with Qt taken out."""

    _INTENTS = (
        "document_name_committed",
        "port_label_committed",
        "period_name_committed",
        "duration_committed",
        "digital_committed",
        "analog_committed",
        "delay_committed",
        "insert_period_requested",
        "move_period_requested",
        "remove_period_requested",
        "bracket_committed",
        "run_repeats_committed",
        "visible_ports_committed",
        "clear_port_requested",
        "binding_cycle_requested",
        "scan_array_load_requested",
        "feedback_requested",
        "run_requested",
        "stop_requested",
        "sync_requested",
        "save_requested",
        "load_requested",
        "connection_requested",
    )

    def __init__(self) -> None:
        for name in self._INTENTS:
            setattr(self, name, _Signal())
        self.schedule = None
        self._version = (-1, -1)
        self.rebuilds = 0
        self.updated_periods: list = []
        self.updated_delays: list = []
        self.updated_labels: list = []
        self.summary: dict = {}
        self.connection = None
        self.capabilities = None
        self.control_state = None
        self.can_run = None
        self.scan_busy = False

    def set_visible_ports(self, ports) -> None:
        """The value path, mirrored: the real view re-flags its own rows."""

        from dataclasses import replace

        if self.schedule is None:
            return
        shown = {str(key) for key in ports}
        self.schedule = replace(
            self.schedule,
            ports=tuple(
                replace(port, visible=port.key in shown) for port in self.schedule.ports
            ),
            visible_text=f"{len(shown)}/{len(self.schedule.ports)}",
        )

    def set_schedule(self, vm) -> bool:
        # The real view refuses two different models under one revision, and a
        # double that accepts anything lets exactly that defect through: this
        # test passed while the window raised.
        version = (int(vm.document_generation), int(vm.revision))
        if version == self._version and self.schedule is not None and vm != self.schedule:
            raise ValueError("one schedule revision cannot identify two view models")
        if version < self._version:
            return False
        self._version = version
        self.schedule = vm
        self.rebuilds += 1
        return True

    def set_period(self, period) -> None:
        # Targeted update: one card, not a rebuild.  Recorded so a test can
        # tell the two paths apart, which is the whole point of the split.
        self.updated_periods.append(period)
        if self.schedule is not None:
            periods = tuple(
                period if item.period_id == period.period_id else item
                for item in self.schedule.periods
            )
            self.schedule = replace(self.schedule, periods=periods)

    def set_delay_row(self, row) -> None:
        self.updated_delays.append(row)

    def set_port_label(self, key: str, label: str) -> None:
        self.updated_labels.append((str(key), str(label)))

    def set_summary(
        self,
        total_text: str,
        total_tooltip: str,
        period_count: int,
        visible_text: str,
        summary_text: str,
        scan_summary_text: str,
    ) -> None:
        self.summary = {
            "total_text": total_text,
            "total_tooltip": total_tooltip,
            "period_count": period_count,
            "visible_text": visible_text,
            "summary_text": summary_text,
            "scan_summary_text": scan_summary_text,
        }

    def set_connection(self, vm) -> None:
        self.connection = vm

    def set_capabilities(self, can_sync: bool, can_hold: bool, can_step: bool) -> None:
        self.capabilities = (bool(can_sync), bool(can_hold), bool(can_step))

    def set_scan_busy(self, busy: bool) -> None:
        self.scan_busy = bool(busy)

    def set_control_state(
        self,
        running: bool,
        synchronized: bool,
        file_dirty: bool,
        *,
        can_run: bool,
        can_stop: bool,
    ) -> None:
        self.control_state = (bool(running), bool(synchronized), bool(file_dirty))
        self.can_run = bool(can_run)
        self.can_stop = bool(can_stop)


class _PreviewView:
    """The preview page's whole contract, with Qt taken out."""

    def __init__(self) -> None:
        self.include_off_toggled = _Signal()
        self.size_committed = _Signal()
        self.selectors_toggled = _Signal()
        self.save_requested = _Signal()
        self.size_names: tuple = ()
        self.size = ""
        self.pinned = None
        self.content = None
        self.logical_size = None
        self.placeholder = ""
        self.status = ""
        self._include_off = False

    @property
    def include_off_rows(self) -> bool:
        # A property, exactly as the real view declares it.  It was a method
        # here once, and the presenter called it: two doubles agreeing with
        # each other while the window raised "'bool' object is not callable".
        return self._include_off

    def mount_content(self, widget, *, logical_size=None, wheel_target=None) -> None:
        self.content = widget
        self.logical_size = logical_size
        self.wheel_target = wheel_target
        self.placeholder = ""

    def show_placeholder(self, text: str) -> None:
        self.placeholder = str(text)
        self.content = None

    def set_status(self, text: str) -> None:
        self.status = str(text)

    def set_size_names(self, names) -> None:
        self.size_names = tuple(names)

    def set_preview_size(self, size: str, *, pinned=None) -> None:
        self.size = str(size)
        if pinned is not None:
            self.pinned = bool(pinned)


class _TargetView:
    """The wiring page's contract, with Qt taken out."""

    def __init__(self) -> None:
        self.apply_requested = _Signal()
        self.feedback_requested = _Signal()
        self.records: tuple = ()
        self.editable = None
        self.status = ""
        self.feedback = ""
        self.width_rules = None

    def set_ports(self, records, editable, status_text) -> None:
        self.records = tuple(records)
        self.editable = bool(editable)
        self.status = str(status_text)

    def set_width_rules(self, digital, dac) -> None:
        self.width_rules = (digital, dac)

    def set_feedback(self, text: str) -> None:
        self.feedback = str(text)


class _ScanView:
    """The Scan page's contract, with Qt taken out."""

    _INTENTS = (
        "repeats_committed",
        "hold_requested",
        "step_requested",
        "load_program_requested",
        "template_requested",
        "source_edited",
        "run_requested",
        "save_array_requested",
        "progress_refresh_requested",
    )

    def __init__(self) -> None:
        for name in self._INTENTS:
            setattr(self, name, _Signal())
        self.page = None
        self.progress = ""

    def set_page(self, record) -> None:
        self.page = record

    def set_progress_text(self, text: str) -> None:
        self.progress = str(text)


class _PreviewHost:
    """What a drawing package hands over: the host, and never its widget.

    A double has to declare what the real collaborator declares, and a real
    host answers for its own size, saves itself, and closes itself.  The fakes
    here used to answer with the timeline data instead, which passed only
    because the window then held the WIDGET and asked the host for nothing.
    """

    def __init__(self, data=None, logical_size=None) -> None:
        self.data = data
        self.logical_size = logical_size
        self.closed = False
        self.saved = None
        #: Whether the operator may drag on it.  A real host gates this, so a
        #: double that could not would let the Selectors switch go back to
        #: doing nothing without a test noticing.
        self.interaction = True

    def qt_widget(self):
        raise AssertionError("a test never mounts a real widget")

    def set_interaction_enabled(self, enabled):
        self.interaction = bool(enabled)

    def update_data(self, data):
        """A standing host takes new data rather than being rebuilt."""

        self.data = data
        return None

    def save(self, path):
        self.saved = path
        return None

    def close(self) -> None:
        self.closed = True


class _EditorView:
    """The double stands in for the HANDLE, so it is flat like the handle.

    A double that mirrors the old widget tree lets the presenter go on reaching
    through pages that no longer exist on the real thing; the sub-doubles below
    survive only as recorders the assertions read.
    """

    _SIGNALS = (
        "clear_all_requested", "close_requested", "page_changed",
        "document_name_committed", "port_label_committed",
        "period_name_committed", "duration_committed", "digital_committed",
        "analog_committed", "delay_committed", "binding_cycle_requested",
        "insert_period_requested", "move_period_requested",
        "remove_period_requested", "bracket_committed",
        "run_repeats_committed",
        "visible_ports_committed", "clear_port_requested",
        "feedback_requested", "connection_requested", "fire_requested",
        "stop_requested", "sync_requested", "save_requested", "load_requested",
        "values_save_requested", "values_load_requested", "binding_renamed",
        "scan_array_load_requested", "scan_source_edited",
        "scan_repeats_committed", "scan_hold_requested", "scan_step_requested",
        "scan_program_load_requested", "scan_template_requested",
        "scan_run_requested", "scan_array_save_requested",
        "scan_progress_refresh_requested",
        "preview_include_off_toggled", "preview_size_committed",
        "preview_selectors_toggled", "preview_save_requested",
        "target_apply_requested",
    )

    def __init__(self) -> None:
        self.schedule_view = _ScheduleView()
        self.scan_view = _ScanView()
        self.preview_view = _PreviewView()
        self.target_view = _TargetView()
        self.title = ""
        self.summary = ""
        self.capabilities = None
        self.status_token = ""
        self.warnings: list[str] = []
        self.done: list[str] = []
        #: What the file dialog would answer; "" is the operator cancelling.
        self.open_answer = ""
        self.save_answer = ""
        for name in self._SIGNALS:
            setattr(self, name, _Signal())

    # -- the document ----------------------------------------------------

    def set_title(self, text: str) -> None:
        self.title = str(text)

    def set_summary(self, text: str) -> None:
        self.summary = str(text)

    def set_capabilities(self, can_sync: bool, can_hold: bool, can_step: bool) -> None:
        self.capabilities = (bool(can_sync), bool(can_hold), bool(can_step))
        self.schedule_view.set_capabilities(can_sync, can_hold, can_step)

    def set_status_color(self, token: str) -> None:
        self.status_token = str(token)

    #: The real window opens on Edit, and the presenter asks rather than
    #: assumes -- so the double has to be able to answer.
    page = "Edit"

    @property
    def current_page(self) -> str:
        return self.page

    def show_warning(self, text: str) -> None:
        self.warnings.append(str(text))

    def show_done(self, text: str) -> None:
        self.done.append(str(text))

    def ask_save_path(self, caption: str, suggested: str, filter: str) -> str:
        self.asked = (str(caption), str(suggested), str(filter))
        return self.save_answer

    def ask_open_path(self, caption: str, start: str, filter: str) -> str:
        self.asked = (str(caption), str(start), str(filter))
        return self.open_answer

    # -- the schedule ----------------------------------------------------

    def set_schedule(self, schedule) -> None:
        self.schedule_view.set_schedule(schedule)

    def set_period(self, period) -> None:
        self.schedule_view.set_period(period)

    def set_delay_row(self, row) -> None:
        self.schedule_view.set_delay_row(row)

    def set_port_label(self, key: str, label: str) -> None:
        self.schedule_view.set_port_label(key, label)

    def set_schedule_summary(
        self,
        total_text: str,
        total_tooltip: str,
        period_count: int,
        visible_text: str,
        summary_text: str,
        scan_summary_text: str,
    ) -> None:
        self.schedule_view.set_summary(
            total_text, total_tooltip, period_count, visible_text,
            summary_text, scan_summary_text,
        )

    def set_visible_ports(self, ports) -> None:
        self.schedule_view.set_visible_ports(ports)

    def set_control_state(
        self,
        running: bool,
        synchronized: bool,
        file_dirty: bool,
        *,
        can_run: bool,
        can_stop: bool,
    ) -> None:
        self.schedule_view.set_control_state(
            running, synchronized, file_dirty, can_run=can_run, can_stop=can_stop
        )

    def set_connection(self, connection) -> None:
        self.schedule_view.set_connection(connection)

    def set_scan_busy(self, busy: bool) -> None:
        self.schedule_view.set_scan_busy(busy)

    # -- the scan --------------------------------------------------------

    def set_scan_page(self, record) -> None:
        self.scan_view.set_page(record)

    def set_scan_progress_text(self, text: str) -> None:
        self.scan_view.set_progress_text(text)

    # -- the preview -----------------------------------------------------

    @property
    def preview_include_off_rows(self) -> bool:
        return bool(self.preview_view.include_off_rows)

    @property
    def preview_size(self) -> str:
        return str(self.preview_view.preview_size)

    @property
    def preview_size_pinned(self) -> bool:
        return bool(self.preview_view.preview_size_pinned)

    def set_preview_size(self, size: str, *, pinned=None) -> None:
        self.preview_view.set_preview_size(size, pinned=pinned)

    def set_preview_size_names(self, names) -> None:
        self.preview_view.set_size_names(names)

    def reset_preview_size_pin(self) -> None:
        self.preview_view.reset_preview_size_pin()

    def set_preview_status(self, text: str) -> None:
        self.preview_view.set_status(text)

    def show_preview_placeholder(self, text: str) -> None:
        self.preview_view.show_placeholder(text)

    def show_preview(self, host) -> None:
        self.preview_view.mount_content(
            host, logical_size=getattr(host, "logical_size", None)
        )

    # -- the target ------------------------------------------------------

    def set_target_ports(self, records, editable: bool, status_text: str) -> None:
        self.target_view.set_ports(records, editable, status_text)

    def set_target_width_rules(self, digital, dac) -> None:
        self.target_view.set_width_rules(digital, dac)

    def set_target_feedback(self, text: str) -> None:
        self.target_view.set_feedback(text)


@pytest.fixture
def sequence():
    """An ordinary product pulse, with Calibration carrying no implicit role."""

    return _ordinary_sequence()


def _ordinary_sequence():
    """The same pulse, reachable without the fixture, for sibling suites."""

    return ordinary_imaging_sequence()


def _run_preview_immediately(work, delivered, failed) -> None:
    """Drive the production async preview state machine without Qt."""

    try:
        result = work()
    except BaseException as error:
        failed(error)
    else:
        delivered(result)


@pytest.fixture
def presenter(sequence):
    view = _EditorView()
    presenter = PulseEditorPresenter(
        view,
        sequence,
        make_preview=lambda data, **_options: _PreviewHost(data),
        run_preview_work=_run_preview_immediately,
    )
    # Turned to Preview, because that is now what asks for a drawing: a window
    # opens on Edit and does not build a render worker for a page behind it.
    presenter.show_page("Preview")
    try:
        yield presenter
    finally:
        presenter.close()


def test_the_projection_shows_what_the_sequence_contains(presenter, sequence) -> None:
    vm = presenter.view.schedule_view.schedule
    assert vm.document_name == sequence.name
    assert vm.period_count == len(sequence.periods)
    # A DAC is one output: its data lanes and the clock that latches them are
    # one bundle, and the clock is not something a pulse drives.
    latched = {
        port.latch_clock
        for port in sequence.target.ports
        if port.kind == "dac" and port.latch_clock
    }
    assert {row.key for row in vm.ports} == {
        port.key for port in sequence.target.ports if port.key not in latched
    }
    assert latched, "this board has no DAC latch clock to fold in"
    assert not any(row.kind == "clock" for row in vm.ports)
    from zlc_pulse.model import ANALOG_MODE_CHOICES

    assert tuple(choice.value for choice in vm.analog_mode_choices) == (
        *ANALOG_MODE_CHOICES,
        "hold",
    )
    assert tuple(choice.label for choice in vm.analog_mode_choices) == (
        "Edge", "Ramp", "Hold",
    )

    first = vm.periods[0]
    original = sequence.periods[0]
    assert first.duration.text == f"{original.duration:g}"
    assert first.unit == original.unit

    lanes = sequence.target.raw_lanes
    for port_key, high in first.digital:
        port = sequence.target.by_key[port_key]
        assert high == bool(original.states[lanes.index(port.lanes[0])])


def test_turning_a_lane_on_changes_that_lane_and_nothing_else(presenter, sequence) -> None:
    period = sequence.periods[0]
    port = next(port for port in sequence.target.ports if port.kind == "digital")
    index = sequence.target.raw_lanes.index(port.lanes[0])
    before = period.states

    presenter.view.digital_committed.emit(
        period.period_id, port.key, not bool(before[index])
    )

    after = presenter.sequence.periods[0].states
    assert after[index] != before[index]
    assert [
        value for position, value in enumerate(after) if position != index
    ] == [value for position, value in enumerate(before) if position != index]


def test_a_duration_off_the_clock_grid_is_rounded_onto_it(presenter) -> None:
    """Typed between ticks is not an argument to have.

    A period that is not a whole number of ticks cannot be played, and the
    editor used to say so in a blocking dialog -- while the operator was still
    typing, when every intermediate value is briefly wrong.  It rounds onto the
    grid instead and shows what it did, so the number on screen is the number
    the board will play.

    Where the grid IS still belongs to zlc_pulse; this only asks.
    """

    period_id = presenter.sequence.periods[0].period_id
    step = presenter.sequence.time_step_ns

    presenter.view.duration_committed.emit(period_id, 3.7, "ns")

    stored = presenter.sequence.periods[0]
    assert stored.unit == "ns"
    assert (stored.duration % step) == 0, f"{stored.duration} is not a whole tick"
    assert stored.duration == step, "3.7 ns should land on the first tick"
    assert not presenter.view.warnings, "rounding is not something to warn about"


def test_a_duration_that_is_not_a_number_still_says_so(presenter) -> None:
    """Rounding answers "which legal value"; it cannot answer "which number"."""

    period_id = presenter.sequence.periods[0].period_id
    kept = presenter.sequence

    presenter.view.duration_committed.emit(period_id, "not a duration", "ns")

    assert presenter.sequence is kept, "a meaningless edit was kept"
    assert presenter.view.warnings, "the operator was told nothing"


def test_an_analog_level_outside_the_dac_range_is_refused(presenter, sequence) -> None:
    dac = next((port for port in sequence.target.ports if port.kind == "dac"), None)
    if dac is None:
        pytest.skip("this target has no DAC port")
    low, high = dac.signed_range
    period_id = presenter.sequence.periods[0].period_id
    kept = presenter.sequence

    presenter.view.analog_committed.emit(
        period_id, dac.key, "edge", high + 1
    )
    assert presenter.sequence is kept
    assert presenter.view.warnings, "the operator was told nothing"

    presenter.view.analog_committed.emit(period_id, dac.key, "edge", high)
    step = next(
        item
        for item in presenter.sequence.periods[0].analog_steps
        if item.port == dac.key
    )
    assert step.value == high


def test_removing_the_last_period_is_refused_with_a_reason(presenter) -> None:
    for period in list(presenter.sequence.periods)[:-1]:
        presenter.view.remove_period_requested.emit(period.period_id)
    remaining = presenter.sequence.periods[0].period_id

    presenter.view.remove_period_requested.emit(remaining)

    assert len(presenter.sequence.periods) == 1
    assert any("at least one period" in text for text in presenter.view.warnings)


def test_inserting_a_period_copies_its_neighbour(presenter) -> None:
    """A new period that zeroes every lane inserts a silent gap mid-sequence."""

    before = presenter.sequence.periods
    presenter.view.insert_period_requested.emit(before[1].period_id)

    after = presenter.sequence.periods
    assert len(after) == len(before) + 1
    assert after[1].states == before[0].states
    assert len({period.period_id for period in after}) == len(after)


def test_clearing_a_port_leaves_the_others_alone(presenter, sequence) -> None:
    ports = [port for port in sequence.target.ports if port.kind == "digital"]
    cleared, kept = ports[0], ports[1]
    lanes = sequence.target.raw_lanes
    cleared_index = lanes.index(cleared.lanes[0])
    kept_index = lanes.index(kept.lanes[0])
    kept_before = [period.states[kept_index] for period in presenter.sequence.periods]

    presenter.view.clear_port_requested.emit(cleared.key)

    assert all(period.states[cleared_index] == 0 for period in presenter.sequence.periods)
    assert [period.states[kept_index] for period in presenter.sequence.periods] == kept_before


def test_clear_all_makes_one_safe_blank_without_moving_the_file_baseline(
    presenter, sequence, tmp_path
) -> None:
    """Clear is destructive authoring, not an alias for reopening the file."""

    presenter.path = str(tmp_path / "mine.json")
    visible = tuple(port.key for port in sequence.target.ports[:2])
    presenter.view.visible_ports_committed.emit(visible)
    period_id = presenter.sequence.periods[0].period_id
    presenter.view.period_name_committed.emit(period_id, "edited")
    saved = presenter._saved_state

    presenter.view.clear_all_requested.emit()

    blank = presenter.sequence
    assert blank.name == sequence.name
    assert blank.target == sequence.target
    assert presenter.path == str(tmp_path / "mine.json")
    assert presenter._state.visible_ports == frozenset(visible)
    assert len(blank.periods) == 1
    assert all(value == 0 for value in blank.periods[0].states)
    assert blank.periods[0].analog_steps == ()
    assert blank.slots == blank.api_parameters == blank.delays == ()
    assert blank.bracket is None
    assert presenter._state.scan_source == ""
    assert presenter._state.scan_rows == ()
    assert presenter._state.scan_source_dirty is False
    assert presenter._saved_state is saved
    assert presenter.view.schedule_view.control_state[2] is True


def test_the_preview_is_built_from_the_periods_that_will_be_played(presenter, sequence) -> None:
    data = presenter.view.preview_view.content.data
    assert data is not None, presenter.view.preview_view.placeholder

    expected = sum(
        float(period.duration) * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[period.unit]
        for period in sequence.periods
    )
    assert data.total_duration == pytest.approx(expected)
    assert data.channels, "no channel was drawn"
    for block in data.blocks:
        assert 0.0 <= block.start < block.stop <= expected + 1e-12


def test_the_preview_follows_an_edit(presenter, sequence) -> None:
    """A preview that does not move is a preview of the last pulse."""

    period = sequence.periods[0]
    lanes = sequence.target.raw_lanes
    # A lane that is already high in this period would add no block, so the
    # test would prove nothing; pick one that is actually low.
    port = next(
        port
        for port in sequence.target.ports
        if port.kind == "digital" and not period.states[lanes.index(port.lanes[0])]
    )
    before = len(presenter.view.preview_view.content.data.blocks)

    presenter.view.digital_committed.emit(period.period_id, port.key, True)

    assert len(presenter.view.preview_view.content.data.blocks) > before


def test_a_timeline_can_be_drawn_for_a_pulse_with_nothing_high(sequence) -> None:
    """Which is how a pulse being written starts."""

    quiet = replace_sequence(
        sequence,
        periods=tuple(
            type(period)(
                period_id=period.period_id,
                duration=period.duration,
                unit=period.unit,
                states=(0,) * len(sequence.target.raw_lanes),
                analog_steps=(),
                name=period.name,
            )
            for period in sequence.periods
        ),
        slots=(),
        bracket=None,
    )
    data = timeline_of(quiet)
    assert data.channels and not data.blocks


def _board_description():
    """A real board description, from the deployed config.

    The double describes itself the way a board does, because "what board is
    this" is part of the streamer contract now; a double that cannot answer
    would let the presenter's board-adoption path go untested while the window
    depends on it entirely.
    """

    from zlc_pulse import load_streamer_config, pulse_target_from_xdc
    from zlc_pulse.device import PulseStreamer
    from zlc_pulse.transport import MemoryRegisterTransport

    config = load_streamer_config()
    geometry = config["params"]
    streamer = PulseStreamer(
        MemoryRegisterTransport(geom=geometry, auto_done=True),
        geometry,
        config["clock_hz"],
        target=pulse_target_from_xdc(config_path=config["source"]),
    )
    streamer.open()
    try:
        return streamer.describe()
    finally:
        streamer.close()


@dataclass(frozen=True)
class _AppliedEcho:
    """What the board is holding, in the shape the real one reports it."""

    program: object
    source: object
    rows: tuple
    run_repeats: int = 1
    scan_repeats: int = 1


class _Sequencer:
    """A board, with the register writes taken out.

    It keeps state rather than only recording calls, because what the editor
    asks a board is not "did I send load" but "what are you holding, and are you
    playing it".  A double that answers those from a list of past calls is a
    double that agrees with whatever the editor already believed -- which is the
    exact bug this stopped being able to hide.
    """

    def __init__(
        self,
        *,
        fail_on_fire: bool = False,
        never_done: bool = False,
        description: object | None = None,
    ) -> None:
        self.events: list[str] = []
        self.fail_on_fire = fail_on_fire
        self.never_done = never_done
        self._digest = ""
        self._firing = False
        self._run_repeats = 1
        self._scan_repeats = 1
        self._applied = None
        self.wait_timeouts: list[object] = []
        self.scan_rows: tuple[tuple[int, ...], ...] = ()
        self.description = description

    def applied(self):
        self.events.append("applied")
        return self._applied

    def describe(self):
        self.events.append("describe")
        return self.description or _board_description()

    def load(self, prog, *, source=None, rows=()) -> None:
        self.events.append("load")
        self._digest = prog.digest
        self._firing = False
        self.scan_rows = tuple(tuple(int(value) for value in row) for row in rows)
        # The real board keeps what it was handed, which is what Sync reads
        # back; a double that forgets it answers "nothing applied" forever.
        self._applied = _AppliedEcho(prog, source, self.scan_rows)

    def fire(self, *, run_repeats: int, scan_repeats: int = 1) -> None:
        self.events.append(
            "fire forever" if run_repeats == 0 or scan_repeats == 0 else "fire"
        )
        if self.fail_on_fire:
            raise RuntimeError("board refused the shot")
        self._firing = True
        self._run_repeats = int(run_repeats)
        self._scan_repeats = int(scan_repeats)
        self._applied = replace(
            self._applied,
            run_repeats=self._run_repeats,
            scan_repeats=self._scan_repeats,
        )

    def wait_done(self, timeout=None) -> object | None:
        self.events.append("wait_done")
        self.wait_timeouts.append(timeout)
        # None means "no shot finished", exactly as the real device reports it.
        if self.never_done:
            return None
        self._firing = False
        return object()

    def safe(self):
        self.events.append("safe")
        self._firing = False
        return None

    def snapshot(self) -> dict:
        return {
            "opened": True,
            "loaded": bool(self._digest),
            "firing": self._firing,
            "run_repeats": self._run_repeats,
            "scan_repeats": self._scan_repeats,
            "applied_digest": self._digest,
        }


def test_on_pulse_runs_until_stop(sequence) -> None:
    """A pulse is a cycle an experiment holds running, not a single shot.

    And a forever run is deliberately not waited on: it never reports done, so
    waiting would hang the window on its own success.
    """

    from zlc_pulse import PulseApiParameter, PulseFieldRef, PulseSlot

    api_period = sequence.periods[1]
    scan_period = sequence.periods[3]
    sequence = replace(
        sequence,
        slots=(
            PulseSlot(
                "duration",
                PulseFieldRef("duration", scan_period.period_id),
                scan_period.unit,
                "scan_probe_duration",
            ),
        ),
        api_parameters=(
            PulseApiParameter(
                "api_probe_duration",
                PulseFieldRef("duration", api_period.period_id),
                api_period.unit,
            ),
        ),
    )
    view = _EditorView()
    board = _Sequencer()
    presenter = PulseEditorPresenter(view, sequence, sequencer=board)
    board.events.clear()
    try:
        view.fire_requested.emit()
        assert board.events == ["load", "fire forever"]
        assert board._applied.source.slots == ()
        assert board._applied.source.api_parameters == ()
        assert board._applied.source.period_by_id[api_period.period_id].duration == (
            api_period.duration
        )
        assert board._applied.source.period_by_id[scan_period.period_id].duration == (
            scan_period.duration
        )
        assert presenter.running is True
        # Stop is the live control now, and Run is not offered twice.
        running, _synchronized, _dirty = view.schedule_view.control_state
        assert running is True

        presenter.stop()
        assert board.events[-1] == "safe"
        assert presenter.running is False
    finally:
        presenter.close()


def test_a_finite_run_is_asked_for_explicitly(sequence) -> None:
    """A finite run is started the same way a forever run is: started."""

    view = _EditorView()
    board = _Sequencer()
    from zlc_workbench.device_use import DeviceClaim, DeviceUseBusy, DeviceUseCoordinator

    device_use = DeviceUseCoordinator()
    presenter = PulseEditorPresenter(
        view,
        replace_sequence(sequence, run_repeats=1),
        sequencer=board,
        device_use=device_use,
    )
    board.events.clear()
    try:
        assert presenter.fire() is True
        assert board.events == ["load", "fire"]
        # Started, not finished: nothing waits for the board any more, so a run
        # that was just asked for is a run that is going.
        assert presenter.running is True
        with pytest.raises(DeviceUseBusy, match="PulseGUI"):
            device_use.acquire_command(
                object(),
                "calibration",
                (
                    DeviceClaim(
                        "sequencer",
                        "sequencer",
                        board,
                    ),
                ),
            )

        presenter.refresh_run_state()
        assert board.events[-1] == "wait_done"
        assert board.wait_timeouts[-1] == 0
        lease = device_use.acquire_command(
            object(),
            "calibration",
            (
                DeviceClaim(
                    "sequencer",
                    "sequencer",
                    board,
                ),
            ),
        )
        lease.release()
    finally:
        presenter.close()


def test_a_shot_that_fails_leaves_the_outputs_safe(sequence) -> None:
    view = _EditorView()
    board = _Sequencer(fail_on_fire=True)
    presenter = PulseEditorPresenter(view, sequence, sequencer=board)
    try:
        assert presenter.fire() is False
        assert board.events[-1] == "safe"
        assert any("firing stopped" in text for text in view.warnings)
    finally:
        presenter.close()


def test_a_finite_run_does_not_block_on_the_board(sequence) -> None:
    """Start it and come back; the beat says what the board is doing.

    The finite path used to wait for done on the calling thread, which for On
    Pulse is the GUI thread -- so a scan long enough to matter froze the window
    for its whole length, and Stop, the one control that would have helped,
    could not be delivered.

    Nothing is lost by not waiting.  Firing over an unfinished shot cannot
    happen: the device requires an idle board for load and fire, and raises
    otherwise.
    """

    view = _EditorView()
    board = _Sequencer(never_done=True)
    presenter = PulseEditorPresenter(
        view,
        replace_sequence(sequence, run_repeats=1),
        sequencer=board,
    )
    try:
        assert presenter.fire() is True
        assert "wait_done" not in board.events, "the GUI thread waited on the board"
        assert board.events.count("fire") == 1
        assert not view.warnings, repr(view.warnings)
    finally:
        presenter.close()


def test_without_a_sequencer_the_editor_says_so_rather_than_pretending(sequence) -> None:
    view = _EditorView()
    presenter = PulseEditorPresenter(view, sequence)
    try:
        assert presenter.fire() is False
        assert any("not connected to a sequencer" in text for text in view.warnings)
    finally:
        presenter.close()


def test_a_pulse_can_be_saved_and_opened_again(sequence, tmp_path) -> None:
    """The JSON round trip owns the whole authoring state, including unrun text."""

    written = tmp_path / "mine.json"
    view = _EditorView()
    view.save_answer = str(written)
    presenter = PulseEditorPresenter(view, sequence)
    try:
        presenter.set_document_name("kept")
        presenter.insert_period(None)
        expected = len(presenter.sequence.periods)
        period_id = presenter.sequence.periods[0].period_id
        presenter.view.binding_cycle_requested.emit("duration", period_id, None)
        presenter.view.scan_source_edited.emit(
            "import numpy as np\nscan_table = np.linspace(0.001, 0.004, 4).reshape(-1, 1)\n"
        )
        presenter.view.scan_run_requested.emit()
        generated_rows = presenter._state.scan_rows
        unrun_source = "import numpy as np\nscan_table = np.linspace(0.002, 0.006, 9).reshape(-1, 1)\n"
        presenter.view.scan_source_edited.emit(unrun_source)
        visible = tuple(port.key for port in presenter.sequence.target.ports[:2])
        presenter.view.visible_ports_committed.emit(visible)
        presenter.view.scan_repeats_committed.emit(3)

        assert presenter.save_pulse() == str(written)
        assert written.exists() and written.stat().st_size > 0
        assert written.read_bytes() == readable_json_bytes(
            state_to_tree(presenter._state)
        )
        assert presenter._state is presenter._saved_state
        assert presenter.view.schedule_view.control_state[2] is False
        tree = json.loads(written.read_text(encoding="utf-8"))
        assert tuple(tree["editor"]) == (
            "visible_ports",
            "scan_source",
            "scan_rows",
            "scan_source_dirty",
            "scan_repeats",
        )
        assert tree["editor"]["scan_source"] == unrun_source
        assert tree["editor"]["scan_source_dirty"] is True
        assert tree["editor"]["scan_rows"] == [list(row) for row in generated_rows]

        from zlc_workbench.session import read_pulse

        decoded = read_pulse(written)
        assert decoded == presenter._state
        assert decoded.visible_ports == frozenset(visible)

        presenter.clear_all()
        view.open_answer = str(written)
        assert presenter.ask_for_pulse() is True
        assert presenter._state == decoded
        assert presenter._saved_state == decoded
        assert presenter.sequence.name == "kept" and len(presenter.sequence.periods) == expected
    finally:
        presenter.close()


def test_save_updates_neither_path_nor_baseline_until_the_write_succeeds(
    sequence, tmp_path, monkeypatch
) -> None:
    """Validation and IO failure both leave the current file identity intact."""

    other = tmp_path / "pulse.txt"
    other.write_text("keep me" + chr(10), encoding="utf-8")
    view = _EditorView()
    view.save_answer = str(other)
    presenter = PulseEditorPresenter(view, sequence)
    try:
        presenter.set_document_name("dirty")
        baseline = presenter._saved_state
        path = presenter.path
        assert presenter.save_pulse() == ""
        assert other.read_text(encoding="utf-8") == "keep me" + chr(10)
        assert any("not a JSON pulse" in text for text in view.warnings)
        assert presenter.path == path and presenter._saved_state is baseline

        import zlc_workbench.pulse_editor as module

        view.save_answer = str(tmp_path / "will-fail.json")

        def refuse_write(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(module, "write_pulse", refuse_write)
        assert presenter.save_pulse() == ""
        assert presenter.path == path and presenter._saved_state is baseline
        assert any("disk full" in text for text in view.warnings)
    finally:
        presenter.close()


def test_connecting_attaches_a_sequencer_and_shows_it(sequence, monkeypatch) -> None:
    """The Connect button, which used to be attached to nothing."""

    view = _EditorView()
    board = _Sequencer(description=_board_description())

    import zlc_pulse

    def _no_local_board_files(*_args, **_kwargs):
        raise AssertionError("an attached editor read this computer's board files")

    monkeypatch.setattr(zlc_pulse, "load_streamer_config", _no_local_board_files)
    monkeypatch.setattr(zlc_pulse, "pulse_target_from_xdc", _no_local_board_files)
    presenter = PulseEditorPresenter(
        view, sequence, dial=lambda _mode, _endpoint: board
    )
    try:
        initial_connection = view.schedule_view.connection
        assert initial_connection.status == "not connected"
        assert initial_connection.locked is False
        assert tuple(choice.value for choice in initial_connection.choices) == (
            "virtual", "remote", "offline",
        )
        assert view.schedule_view.capabilities == (False, False, False)

        with pytest.raises(ValueError, match="unknown connection mode"):
            presenter.connect_to("mystery", "")

        view.connection_requested.emit("remote", "127.0.0.1:18861")

        assert presenter.sequencer is board
        # The status says what was attached, not merely that something was.
        connection = view.schedule_view.connection
        assert (connection.selected, connection.endpoint) == (
            "remote", "127.0.0.1:18861",
        )
        status = connection.status
        assert "127.0.0.1:18861" in status and "ports" in status and "MHz" in status
        # Sync and hold need a board; stepping also needs a table to step
        # through, and offering it without one is a button that cannot work.
        assert view.schedule_view.capabilities == (True, True, False)
        assert view.status_token == "dirty-ready"

        assert presenter.fire() is True
        assert view.status_token == "running-synced"
    finally:
        presenter.close()


def test_a_refused_connection_says_so_and_stays_offline(sequence) -> None:
    """A window that looks connected after a failure is worse than one that does not."""

    view = _EditorView()

    def _refuse(_mode, _endpoint):
        raise ConnectionRefusedError("no server there")

    presenter = PulseEditorPresenter(view, sequence, dial=_refuse)
    try:
        view.connection_requested.emit("remote", "10.0.0.9:18861")

        assert presenter.sequencer is None
        assert "failed" in view.schedule_view.connection.status
        assert view.schedule_view.capabilities == (False, False, False)
        assert any("cannot connect" in text for text in view.warnings)
    finally:
        presenter.close()


def test_reconnecting_releases_the_previous_board(sequence) -> None:
    """Two open connections to one board is not a state anything can reason about."""

    closed: list[str] = []

    class _Closable(_Sequencer):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    boards = [_Closable("first"), _Closable("second")]
    view = _EditorView()
    presenter = PulseEditorPresenter(
        view, sequence, dial=lambda _mode, _endpoint: boards.pop(0)
    )
    try:
        view.connection_requested.emit("virtual", "")
        view.connection_requested.emit("remote", "127.0.0.1:18861")
        assert closed == ["first"]
    finally:
        presenter.close()
    assert closed == ["first", "second"], "closing the editor left a board connected"


def test_an_injected_sequencer_is_not_closed_by_the_editor(sequence) -> None:
    """It belongs to whoever passed it in -- usually a session that is still using it."""

    class _Watched(_Sequencer):
        closed = False
        refuse_safe = False

        def close(self) -> None:
            self.closed = True

        def safe(self):
            if self.refuse_safe:
                raise RuntimeError("board refused safe")
            return super().safe()

    described = _board_description()
    exact = replace(
        described,
        geometry=replace(described.geometry, coeff_frac_bits=3),
    )
    board = _Watched(description=exact)
    from zlc_workbench.device_use import DeviceUseCoordinator

    with pytest.raises(ValueError, match="device-use coordinator"):
        _PulseEditorPresenter(
            _EditorView(),
            PulseEditorState(sequence=sequence),
            sequencer=board,
        )

    device_use = DeviceUseCoordinator()
    presenter = PulseEditorPresenter(
        _EditorView(),
        sequence,
        sequencer=board,
        device_use=device_use,
    )
    assert presenter.board == exact
    connection = presenter.view.schedule_view.connection
    assert connection.selected == "given"
    assert connection.locked is True
    assert tuple((choice.label, choice.value) for choice in connection.choices) == (
        ("Experiment session", "given"),
    )
    with pytest.raises(RuntimeError, match="experiment session owns"):
        presenter.connect_to("given", "")
    presenter.cycle_binding("duration", sequence.periods[0].period_id, None)
    assert presenter.compile().scan_coeff_frac_bits == 3
    assert presenter.fire() is True
    presenter.close()
    assert board.closed is False
    assert board.events[-1] == "safe"
    device_use.assert_idle()

    refusing = _Watched(description=exact)
    refusing.refuse_safe = True
    retained_use = DeviceUseCoordinator()
    retained = PulseEditorPresenter(
        _EditorView(),
        sequence,
        sequencer=refusing,
        device_use=retained_use,
    )
    assert retained.fire() is True
    with pytest.raises(RuntimeError, match="could not release"):
        retained.close()
    with pytest.raises(RuntimeError, match="PulseGUI"):
        retained_use.assert_idle()
    assert refusing.snapshot()["firing"] is True
    refusing.refuse_safe = False
    retained.close()
    retained_use.assert_idle()


def test_the_editor_opens_with_no_pulse_at_all() -> None:
    """An editor opens before it has a subject.

    Requiring a named pulse to start was backwards: the window's job when it
    has nothing open is to say how to get something, not to refuse to appear.
    """

    view = _EditorView()
    presenter = PulseEditorPresenter(view)
    try:
        assert presenter.sequence is None
        vm = view.schedule_view.schedule
        assert vm.periods == () and vm.ports == ()
        assert "Load a pulse" in vm.summary_text
        assert "Load a pulse" in view.summary, "the guidance landed nowhere visible"
        assert "Load a pulse" in view.preview_view.placeholder
        assert view.schedule_view.can_run is False, "Run offered with no pulse"

        # And it refuses the things that need a pulse, by name.
        assert presenter.fire() is False
        assert any("no pulse is open" in text for text in view.warnings)
    finally:
        presenter.close()


def test_load_opens_a_pulse_file_rather_than_refusing(sequence, tmp_path) -> None:
    """Load was wired to the refusal meant for Save.  Loading writes nothing."""

    view = _EditorView()
    view.open_answer = str(write_ordinary_pulse(tmp_path))
    presenter = PulseEditorPresenter(view)
    try:
        view.load_requested.emit()

        assert presenter.sequence is not None
        assert presenter.sequence.name == sequence.name
        assert view.schedule_view.schedule.period_count == len(sequence.periods)
    finally:
        presenter.close()


def test_seeded_calibration_template_is_not_the_editor_current_document(
    tmp_path, monkeypatch
) -> None:
    from zlc_workbench.apps.pulse_editor import resolve
    from zlc_workbench.session import Workspace

    monkeypatch.setenv(Workspace.HOME_VARIABLE, str(tmp_path / "default"))
    default = Workspace.default()
    assert (default.pulses / Workspace.IMAGING_TEMPLATE).is_file()

    space, state, path = resolve(default.root, None)

    assert space.root == default.root
    assert state is None
    assert path == ""


def _process_qt_until(application, predicate, seconds: float = 2.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    assert predicate(), "timed out waiting for the Pulse Editor owner turn"


def _formal_pulse_window(
    tmp_path, monkeypatch, *, sequence, board=None, bound: bool = False, path: str = ""
):
    pytest.importorskip("PyQt5")
    from PyQt5 import QtCore
    from zlc_ui.qt import ensure_qt_app
    from zlc_workbench.apps import pulse_editor as application_module

    application = ensure_qt_app(["formal-pulse-window"])
    if bound:
        from zlc_workbench.device_use import DeviceUseCoordinator

        window = application_module.create_bound_window(
            workspace=tmp_path, sequence=sequence, sequencer=board,
            device_use=DeviceUseCoordinator(), path=path, window_ratio=0.4,
        )
    else:
        monkeypatch.setattr(
            application_module,
            "resolve",
            lambda *_args, **_kwargs: (
                SimpleNamespace(pulses=tmp_path), PulseEditorState(sequence=sequence), path,
            ),
        )
        if board is not None:
            monkeypatch.setattr(application_module, "dial", lambda *_args: board)
        window = application_module.create_window(
            workspace=tmp_path, connect=None if board is None else "virtual", window_ratio=0.4,
        )
    return application, QtCore, window


@pytest.mark.parametrize("bound", (False, True), ids=("standalone", "bound"))
def test_pulse_window_waits_for_asynchronous_retirement(
    tmp_path, monkeypatch, bound: bool
) -> None:
    """A refused SAFE stays visible, and its slow call never parks Qt."""

    from threading import Event

    release = Event()
    sequence = _ordinary_sequence()

    class _RefusingSafe(_Sequencer):
        refusing = True

        def safe(self) -> None:
            if self.refusing:
                release.wait(2.0)
                raise RuntimeError("board refused safe")
            super().safe()

    board = _RefusingSafe(description=_board_description())
    application, QtCore, window = _formal_pulse_window(
        tmp_path, monkeypatch, sequence=sequence, board=board, bound=bound
    )
    summaries: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(window, "set_summary", summaries.append)
    monkeypatch.setattr(window, "show_warning", warnings.append)
    window.fire_requested.emit()
    _process_qt_until(application, lambda: board.snapshot()["firing"] is True)
    summaries.clear()

    try:
        owner_turn: list[bool] = []
        window.close()
        QtCore.QTimer.singleShot(0, lambda: owner_turn.append(True))
        _process_qt_until(application, lambda: bool(owner_turn), 0.2)

        assert owner_turn, "close-triggered SAFE blocked the Qt owner thread"
        assert window.is_visible()
        assert summaries == ["Stopping..."]

        release.set()
        _process_qt_until(application, lambda: bool(warnings))
        assert window.is_visible(), "a failed retirement let the window disappear"
        assert "board refused safe" in warnings[-1]
        assert "board refused safe" in summaries[-1]

        board.refusing = False
        window.close()
        _process_qt_until(application, lambda: not window.is_visible())
    finally:
        board.refusing = False
        release.set()
        if window.is_visible():
            window.close()
            _process_qt_until(application, lambda: not window.is_visible())


def test_pulse_window_stop_projects_stopping_before_background_safe(
    tmp_path, monkeypatch
) -> None:
    """The ordinary Stop click leaves Qt before the board acknowledges SAFE."""

    from threading import Event, Timer

    started = Event()
    release = Event()

    class _SlowSafe(_Sequencer):
        def safe(self) -> None:
            started.set()
            release.wait(1.0)
            super().safe()

    board = _SlowSafe(description=_board_description())
    application, _QtCore, window = _formal_pulse_window(
        tmp_path,
        monkeypatch,
        sequence=_ordinary_sequence(),
        board=board,
    )
    summaries: list[str] = []
    monkeypatch.setattr(window, "set_summary", summaries.append)
    window.fire_requested.emit()
    _process_qt_until(application, lambda: board.snapshot()["firing"] is True)
    summaries.clear()
    try:
        fallback = Timer(0.1, release.set)
        fallback.start()
        before = time.monotonic()
        window.stop_requested.emit()
        elapsed = time.monotonic() - before
        assert elapsed < 0.05, "Stop waited for SAFE on the Qt owner thread"
        assert summaries == ["Stopping..."]
        _process_qt_until(application, started.is_set)
        release.set()
        _process_qt_until(application, lambda: not board.snapshot()["firing"])
    finally:
        fallback.cancel()
        release.set()
        if window.is_visible():
            window.close()
            _process_qt_until(application, lambda: not window.is_visible())


def test_formal_stop_bypasses_blocked_preview_and_device_command(
    tmp_path, monkeypatch
) -> None:
    """Preview, ordinary device work and SAFE are three independent owners."""

    from threading import Event

    from zlc_plot import RasterPlotHost

    preview_started = Event()
    release_preview = Event()
    load_started = Event()
    release_load = Event()
    safe_started = Event()
    release_safe = Event()
    real_wait = RasterPlotHost.wait_for_front

    def blocked_front(self, *args, **kwargs):
        preview_started.set()
        assert release_preview.wait(2.0)
        return real_wait(self, *args, **kwargs)

    class _BlockedLoad(_Sequencer):
        def load(self, *args, **kwargs) -> None:
            load_started.set()
            assert release_load.wait(2.0)
            super().load(*args, **kwargs)

        def safe(self) -> None:
            safe_started.set()
            assert release_safe.wait(2.0)
            super().safe()

    monkeypatch.setattr(RasterPlotHost, "wait_for_front", blocked_front)
    board = _BlockedLoad(description=_board_description())
    application, QtCore, window = _formal_pulse_window(
        tmp_path, monkeypatch, sequence=_ordinary_sequence(), board=board
    )
    summaries: list[str] = []
    monkeypatch.setattr(window, "set_summary", summaries.append)
    try:
        window.page_changed.emit("Preview")
        _process_qt_until(application, preview_started.is_set)

        heartbeat: list[bool] = []
        QtCore.QTimer.singleShot(0, lambda: heartbeat.append(True))
        before = time.monotonic()
        window.fire_requested.emit()
        assert time.monotonic() - before < 0.05
        _process_qt_until(
            application, lambda: load_started.is_set() and bool(heartbeat), 0.2
        )

        before = time.monotonic()
        window.stop_requested.emit()
        assert time.monotonic() - before < 0.05
        assert summaries[-1] == "Stopping..."
        _process_qt_until(application, safe_started.is_set, 0.2)
        assert not release_preview.is_set()
        assert not release_load.is_set(), "SAFE waited behind ordinary device work"

        release_safe.set()
        release_load.set()
        _process_qt_until(
            application,
            lambda: not window.presenter._device_busy
            and not window.presenter._stop_busy,
            3.0,
        )
        assert board.snapshot()["firing"] is False
        assert summaries[-1] == "Stopped", summaries
        assert summaries.count("Started") == 0, "late fire completion overwrote Stop"
    finally:
        release_safe.set()
        release_load.set()
        release_preview.set()
        if window.is_visible():
            window.close()
            _process_qt_until(application, lambda: not window.is_visible(), 4.0)


def test_formal_pulse_preview_build_update_save_and_close_never_wait_on_qt(
    tmp_path, monkeypatch
) -> None:
    """The shipped Preview path keeps every plot Future off the Qt owner."""

    from concurrent.futures import Future
    from threading import Event, Thread, current_thread, main_thread

    from zlc_plot import RasterPlotHost

    sequence = _ordinary_sequence()
    build_started = Event()
    release_build = Event()
    real_wait = RasterPlotHost.wait_for_front

    def slow_first_front(self, *args, **kwargs):
        build_started.set()
        assert release_build.wait(2.0)
        return real_wait(self, *args, **kwargs)

    monkeypatch.setattr(RasterPlotHost, "wait_for_front", slow_first_front)
    application, QtCore, window = _formal_pulse_window(
        tmp_path, monkeypatch, sequence=sequence, path=str(tmp_path / "ordinary.json")
    )
    save_release = Event()
    save_threads: list[Thread] = []

    class GuardedFuture(Future):
        def result(self, timeout=None):
            if current_thread() is main_thread():
                raise AssertionError("Pulse Preview waited for a plot Future on Qt")
            return super().result(timeout)

    def completed(value=None):
        future = GuardedFuture()
        future.set_result(value)
        return future

    try:
        owner_turn: list[bool] = []
        QtCore.QTimer.singleShot(0, lambda: owner_turn.append(True))
        window.page_changed.emit("Preview")
        _process_qt_until(
            application,
            lambda: build_started.is_set() and bool(owner_turn),
            0.2,
        )
        release_build.set()
        _process_qt_until(
            application, lambda: window.presenter._preview_host is not None, 3.0
        )
        host = window.presenter._preview_host

        update_started = Event()

        def planned(_size):
            return completed(
                SimpleNamespace(value=SimpleNamespace(logical_size=(480, 357)))
            )

        def updated(_timeline):
            update_started.set()
            return completed()

        monkeypatch.setattr(host, "set_size", planned)
        monkeypatch.setattr(host, "update_data", updated)
        window.presenter.refresh_preview()
        _process_qt_until(
            application,
            lambda: update_started.is_set() and not window.presenter._preview_busy,
            3.0,
        )

        save_started = Event()

        def save(path):
            future = GuardedFuture()
            save_started.set()

            def complete() -> None:
                assert save_release.wait(2.0)
                Path(path).write_bytes(b"png")
                future.set_result(None)

            thread = Thread(target=complete, name="pulse-preview-save-test")
            save_threads.append(thread)
            thread.start()
            return future

        monkeypatch.setattr(host, "save", save)
        window.preview_save_requested.emit()
        _process_qt_until(application, save_started.is_set, 3.0)

        owner_turn.clear()
        window.close()
        QtCore.QTimer.singleShot(0, lambda: owner_turn.append(True))
        _process_qt_until(application, lambda: bool(owner_turn), 0.2)
        assert window.is_visible(), "the window closed before Preview save retirement"

        save_release.set()
        _process_qt_until(application, lambda: not window.is_visible(), 4.0)
        assert tuple(tmp_path.glob("*.png"))
    finally:
        release_build.set()
        save_release.set()
        for thread in save_threads:
            thread.join(2.0)
        if window.is_visible():
            window.close()
            _process_qt_until(application, lambda: not window.is_visible(), 4.0)


def test_named_pulse_loader_resolves_the_json_product_document(tmp_path) -> None:
    """The app loader carries the complete editor state, not only its sequence."""

    target = write_ordinary_pulse(tmp_path, file_stem="ordinary")
    tree = json.loads(target.read_text(encoding="utf-8"))
    tree["editor"] = {
        "visible_ports": None,
        "scan_source": "half typed",
        "scan_rows": [],
        "scan_source_dirty": True,
        "scan_repeats": 4,
    }
    target.write_text(json.dumps(tree), encoding="utf-8")

    from zlc_workbench.apps.pulse_editor import load_state

    loaded, path = load_state(tmp_path, "ordinary")

    assert loaded.sequence.name == PULSE_NAME
    assert loaded.scan_source == "half typed"
    assert loaded.scan_source_dirty is True
    assert loaded.scan_repeats == 4
    assert Path(path) == target

    named, named_path = load_state(tmp_path, "ordinary.json")
    assert named == loaded
    assert Path(named_path) == target


@pytest.mark.parametrize(
    "name",
    (
        ".",
        "..",
        "./ordinary",
        r".\ordinary",
        "../ordinary",
        r"..\ordinary",
        "folder/ordinary",
        "ordinary.txt",
    ),
)
def test_named_pulse_loader_refuses_paths_and_non_json_suffixes(tmp_path, name) -> None:
    from zlc_workbench.apps.pulse_editor import load_state

    with pytest.raises(ValueError):
        load_state(tmp_path, name)


def test_declining_the_open_dialog_leaves_the_editor_as_it_was(sequence) -> None:
    view = _EditorView()
    view.open_answer = ""
    presenter = PulseEditorPresenter(view, sequence)
    try:
        assert presenter.ask_for_pulse() is False
        assert presenter.sequence is sequence
        assert view.warnings == []
    finally:
        presenter.close()


def test_open_decodes_the_whole_state_before_replacing_any_of_it(sequence, tmp_path) -> None:
    from zlc_pulse import sequence_to_tree

    with pytest.raises(TypeError):
        PulseEditorState(sequence=sequence, visible_ports="d0")

    stray = tmp_path / "bad-editor.json"
    view = _EditorView()
    presenter = PulseEditorPresenter(view, sequence, path=str(tmp_path / "kept.json"))
    try:
        before = presenter._state
        baseline = presenter._saved_state
        path = presenter.path
        for editor in (
            {"scan_use_loaded": True},
            {"scan_source_dirty": "false"},
            {"scan_repeats": 1.5},
            {"scan_repeats": 1 << 32},
            {"scan_rows": ["not a row"]},
            {"visible_ports": "d0"},
        ):
            tree = dict(sequence_to_tree(sequence))
            tree["editor"] = editor
            stray.write_text(json.dumps(tree), encoding="utf-8")
            assert presenter.open_pulse(str(stray)) is False
            assert presenter._state is before
            assert presenter._saved_state is baseline
            assert presenter.path == path
        assert any("bad-editor.json" in text for text in view.warnings)
    finally:
        presenter.close()


def test_add_period_with_nothing_open_starts_a_pulse_on_this_board() -> None:
    """Asking for a period IS asking for a pulse, and a pulse is two periods.

    One period cannot show anything: a pulse is the CHANGES between periods,
    and one period has none.  A new editor therefore opens on the shape the
    established PulseGUI opens on -- one output high, then everything safe --
    which is the smallest thing that draws an edge to edit.
    """

    view = _EditorView()
    presenter = PulseEditorPresenter(view)
    try:
        view.insert_period_requested.emit(None)

        assert presenter.sequence is not None
        first, second = presenter.sequence.periods
        # The board's own pin map, not an invented default.
        from zlc_pulse import pulse_target_from_xdc

        assert presenter.sequence.target == pulse_target_from_xdc()
        assert sum(first.states) == 1, "exactly one output is driven to start"
        assert all(state == 0 for state in second.states), "and then everything is safe"
        # Long enough to see.  At the 20 ns tick both cards are one pixel wide.
        assert first.duration == second.duration > 100
    finally:
        presenter.close()


def test_run_is_offered_only_with_both_a_pulse_and_a_board(sequence, tmp_path) -> None:
    """Either one missing is a button that cannot work, shown as one.

    Only one of the two halves can now go missing on its own: connecting is
    what opens a pulse when none is, so "a board with nothing to fire" is a
    state the editor no longer passes through.
    """

    view = _EditorView()
    board = _Sequencer()
    presenter = PulseEditorPresenter(view, dial=lambda _mode, _endpoint: board)
    try:
        assert view.schedule_view.can_run is False           # neither

        view.connection_requested.emit("virtual", "")
        assert view.schedule_view.can_run is True            # board, and the pulse it opened

        presenter.open_pulse(str(write_ordinary_pulse(tmp_path)))
        assert view.schedule_view.can_run is True            # both

        presenter.connect_to("offline", "")
        assert view.schedule_view.can_run is False           # a pulse, no board
    finally:
        presenter.close()


def test_connecting_with_no_pulse_open_still_shows_the_board() -> None:
    """The complaint in one test: connected, and the editor showed nothing.

    An editor attached to a board knows its ports, its pins and its clock
    before any pulse is open.  Hiding that leaves an operator unable to tell a
    connected editor from a disconnected one -- and with nothing to edit.
    """

    view = _EditorView()
    board = _Sequencer()
    presenter = PulseEditorPresenter(view, dial=lambda _mode, _endpoint: board)
    try:
        assert view.schedule_view.schedule.ports == ()

        view.connection_requested.emit("remote", "127.0.0.1:18861")

        vm = view.schedule_view.schedule
        described = _board_description()
        from zlc_workbench.pulse_editor import programmable_ports

        assert len(vm.ports) == len(programmable_ports(described.target))
        assert len(vm.ports) < len(described.target.ports), "no clock was folded in"
        assert vm.clock_text == f"{described.time_step_ns:g} ns/tick"
        # The pin an operator wires into, not only the compiler's lane name.
        first = programmable_ports(described.target)[0]
        assert vm.ports[0].endpoint_text == described.target.package_pins[first.lanes[0]]
        assert first.lanes[0] in vm.ports[0].endpoint_tooltip
    finally:
        presenter.close()


def test_a_new_pulse_starts_on_the_attached_board(sequence) -> None:
    """Not on this machine's files: a different board makes that pulse a fiction."""

    view = _EditorView()
    board = _Sequencer()
    presenter = PulseEditorPresenter(view, dial=lambda _mode, _endpoint: board)
    try:
        presenter.connect_to("remote", "127.0.0.1:18861")
        view.insert_period_requested.emit(None)

        described = _board_description()
        assert presenter.sequence.target == described.target
        assert presenter.sequence.time_step_ns == described.time_step_ns
    finally:
        presenter.close()


def test_an_open_pulse_is_moved_onto_the_board_by_lane_name(sequence) -> None:
    """Matched by name, not position.

    Two boards exposing the same lane in different slots must not silently swap
    what a period drives; that is a pulse that compiles and fires the wrong
    outputs.
    """

    view = _EditorView()
    board = _Sequencer()
    presenter = PulseEditorPresenter(view, sequence, dial=lambda _mode, _endpoint: board)
    try:
        before = {
            lane: [period.states[index] for period in sequence.periods]
            for index, lane in enumerate(sequence.target.raw_lanes)
        }

        presenter.connect_to("remote", "127.0.0.1:18861")

        described = _board_description()
        assert presenter.sequence.target == described.target
        after = {
            lane: [period.states[index] for period in presenter.sequence.periods]
            for index, lane in enumerate(presenter.sequence.target.raw_lanes)
        }
        for lane, states in before.items():
            if lane in after:
                assert after[lane] == states, lane
        # The operator's work survives the move.
        assert [period.name for period in presenter.sequence.periods] == [
            period.name for period in sequence.periods
        ]
        assert [period.duration for period in presenter.sequence.periods] == [
            period.duration for period in sequence.periods
        ]
    finally:
        presenter.close()


def test_a_board_that_will_not_describe_itself_is_reported(sequence) -> None:
    """Connected is not the same as usable, and the status must not blur them."""

    class _Mute(_Sequencer):
        def describe(self):
            raise RuntimeError("this server is older than describe()")

    view = _EditorView()
    presenter = PulseEditorPresenter(view, sequence, dial=lambda *_args: _Mute())
    try:
        presenter.connect_to("remote", "127.0.0.1:18861")
        assert "cannot read the board" in view.schedule_view.connection.status
        assert any("would not describe itself" in text for text in view.warnings)
    finally:
        presenter.close()


def test_the_preview_offers_its_sizes_and_the_content_picks_one(presenter) -> None:
    """The Size box was empty: nothing ever told the page what the sizes are.

    And the default is not a constant -- a busy pulse needs a bigger surface,
    which is the rule zlc_plot owns so every pulse is drawn comparably.
    """

    from zlc_plot.layout import PANEL_SIZE_NAMES, recommended_pulse_preset

    view = presenter.view.preview_view
    assert view.size_names == PANEL_SIZE_NAMES
    assert view.size == recommended_pulse_preset(
        presenter._preview_rows(), len(presenter.sequence.periods)
    )
    assert view.pinned is False


def test_picking_a_size_pins_it_until_the_content_changes_shape(presenter) -> None:
    """Otherwise the plot snaps back on the next edit and the choice reads as a bug."""

    view = presenter.view.preview_view
    presenter.view.preview_size_committed.emit("8x8")
    assert presenter.preview_size() == "8x8"
    assert view.pinned is True

    # An edit keeps the pin: the pulse is the same shape.
    period_id = presenter.sequence.periods[0].period_id
    presenter.view.period_name_committed.emit(period_id, "edited")
    assert presenter.preview_size() == "8x8"

    # Showing every channel changes how many rows are drawn, so the pin goes.
    view._include_off = True
    presenter.view.preview_include_off_toggled.emit(True)
    assert presenter._pinned_size is None
    assert view.pinned is False


def test_show_all_channels_draws_the_ones_that_are_always_off(presenter, sequence) -> None:
    """"Show off rows" has to actually add rows, or it is a switch that lies."""

    view = presenter.view.preview_view
    lean = presenter._preview_rows()

    view._include_off = True
    presenter.view.preview_include_off_toggled.emit(True)

    from zlc_workbench.pulse_editor import programmable_ports

    assert presenter._preview_rows() > lean
    assert presenter._preview_rows() == len(programmable_ports(sequence.target))
    assert str(lean) not in view.status or "channel" in view.status


def test_a_dac_trace_is_drawable_at_all(sequence) -> None:
    """A step trace is N values over N+1 boundaries.

    Passing equal-length arrays meant no DAC trace could ever be built -- and
    it stayed invisible because the ordinary pulse drives no DAC, so the
    error only appeared the moment someone asked to see every channel.
    """

    from zlc_pulse import AnalogStep

    dac = next(port for port in sequence.target.ports if port.kind == "dac")
    view = _EditorView()
    presenter = PulseEditorPresenter(
        view,
        sequence,
        make_preview=lambda data, **_o: _PreviewHost(data),
        run_preview_work=_run_preview_immediately,
    )
    presenter.show_page("Preview")
    try:
        period_id = sequence.periods[1].period_id
        view.analog_committed.emit(period_id, dac.key, "edge", 250)

        data = view.preview_view.content.data
        assert data is not None, view.preview_view.placeholder
        trace = next(item for item in data.analog_traces if item.name == dac.key)
        assert len(trace.starts) == len(trace.values) + 1
        assert trace.starts[-1] == data.total_duration
        # The level holds from the period it was set in.
        assert trace.values[0] == 0.0 and trace.values[1] == 250.0
    finally:
        presenter.close()


def test_the_target_page_says_which_pins_an_output_reaches(presenter, sequence) -> None:
    """The page had no listener at all, so it showed nothing.

    A pulse names outputs; the Target page is the only place saying which
    physical pins those names reach.  Without it an operator has the pulse and
    the breakout and no way to relate them.
    """

    view = presenter.view.target_view
    assert view.records, "the target page was never filled"

    from zlc_workbench.pulse_editor import programmable_ports

    assert len(view.records) == len(programmable_ports(sequence.target))
    digital = next(r for r in view.records if r.kind == "digital")
    port = sequence.target.by_key[digital.key]
    assert digital.endpoints == (sequence.target.package_pins[port.lanes[0]],)

    dac = next(r for r in view.records if r.kind == "dac")
    spec = sequence.target.by_key[dac.key]
    assert len(dac.endpoints) == spec.width
    # The one wire of the bundle a pulse never drives, and an operator still
    # has to find on the board.
    assert dac.clock_key == spec.latch_clock
    assert dac.clock_endpoint and dac.clock_endpoint not in dac.endpoints


def test_a_board_owns_its_wiring_and_only_names_may_change(presenter, sequence) -> None:
    board = _Sequencer()
    presenter._dial = lambda *_args: board
    presenter.connect_to("remote", "127.0.0.1:18861")
    view = presenter.view.target_view

    assert view.editable is False, "an attached board's topology was offered for editing"
    assert "board's" in view.status
    assert view.width_rules is not None

    from zlc_ui import TargetPortRecord

    edited = tuple(
        TargetPortRecord(
            key=record.key,
            kind=record.kind,
            signal="renamed" if record.kind == "digital" else record.signal,
            endpoints=record.endpoints,
            clock_key=record.clock_key,
            clock_endpoint=record.clock_endpoint,
            lane_order=record.lane_order,
        )
        for record in view.records
    )
    presenter.view.target_apply_requested.emit(edited)

    renamed = [
        port for port in presenter.sequence.target.ports if port.label == "renamed"
    ]
    assert renamed, presenter.view.target_view.feedback
    # A rename is metadata: the wiring and its fingerprint do not move.
    assert presenter.sequence.target.raw_lanes == sequence.target.raw_lanes
    assert presenter.sequence.target.package_pins == sequence.target.package_pins


def test_dropping_a_port_while_a_board_is_attached_is_refused(presenter) -> None:
    """Re-wiring a board from a window would only surface as wrong outputs."""

    board = _Sequencer()
    presenter._dial = lambda *_args: board
    presenter.connect_to("remote", "127.0.0.1:18861")
    view = presenter.view.target_view
    before = presenter.sequence.target

    presenter.view.target_apply_requested.emit(view.records[:-1])

    assert presenter.sequence.target is before
    assert "cannot be added or removed" in view.feedback


def test_offline_the_target_is_the_pulse_file_and_is_editable(presenter) -> None:
    view = presenter.view.target_view
    assert view.editable is True
    assert "Offline" in view.status


def test_toggling_one_lane_updates_one_card_and_rebuilds_nothing(presenter, sequence) -> None:
    """Clicking a checkbox is not a change of shape.

    The card already shows the new state -- that is what the widget IS -- so
    re-projecting the whole board rebuilds every card to arrive back where the
    screen already was, throwing away the scroll position and any partly-typed
    field on the way.
    """

    schedule = presenter.view.schedule_view
    period = sequence.periods[0]
    port = next(port for port in sequence.target.ports if port.kind == "digital")
    before = schedule.rebuilds

    presenter.view.digital_committed.emit(period.period_id, port.key, True)

    assert schedule.rebuilds == before, "one checkbox rebuilt the whole board"
    assert [vm.period_id for vm in schedule.updated_periods] == [period.period_id]
    # And the model really changed.
    index = sequence.target.raw_lanes.index(port.lanes[0])
    assert presenter.sequence.periods[0].states[index] == 1


def test_a_duration_edit_moves_the_totals_without_a_rebuild(presenter, sequence) -> None:
    schedule = presenter.view.schedule_view
    period_id = sequence.periods[0].period_id
    before = schedule.rebuilds

    presenter.view.duration_committed.emit(period_id, 0.004, "s")

    assert schedule.rebuilds == before
    assert schedule.updated_periods[-1].period_id == period_id
    # The header total is a consequence of the edit and must follow it.
    assert schedule.summary["period_count"] == len(sequence.periods)
    assert schedule.summary["total_text"] != ""


def test_a_delay_edit_updates_its_row_only(presenter, sequence) -> None:
    schedule = presenter.view.schedule_view
    port = next(port for port in sequence.target.ports if port.kind == "digital")
    before = schedule.rebuilds

    presenter.view.delay_committed.emit(port.key, 40, "ns")

    assert schedule.rebuilds == before
    assert [row.port_key for row in schedule.updated_delays] == [port.key]
    assert any(delay.port == port.key for delay in presenter.sequence.delays)


def test_renaming_an_output_touches_the_label_and_nothing_else(presenter, sequence) -> None:
    schedule = presenter.view.schedule_view
    port = next(port for port in sequence.target.ports if port.kind == "digital")
    before = schedule.rebuilds

    presenter.view.port_label_committed.emit(port.key, "MOT cooling")

    assert schedule.rebuilds == before
    assert schedule.updated_labels == [(port.key, "MOT cooling")]
    assert presenter.sequence.target.by_key[port.key].label == "MOT cooling"


def test_a_change_of_shape_does_rebuild(presenter, sequence) -> None:
    """The other half of the rule: adding a period IS a change of shape."""

    schedule = presenter.view.schedule_view
    before = schedule.rebuilds

    presenter.view.insert_period_requested.emit(sequence.periods[1].period_id)

    assert schedule.rebuilds == before + 1
    assert len(presenter.sequence.periods) == len(sequence.periods) + 1


def _dac_port(sequence):
    return next(port for port in sequence.target.ports if port.kind == "dac")


def test_a_dot_binds_a_field_into_a_scan_column(presenter, sequence) -> None:
    """The whole Scan page had no listener; this is where it starts.

    Bound, a field stops being a constant in the pulse and becomes a value the
    device writes per point.
    """

    schedule = presenter.view.schedule_view
    period_id = sequence.periods[3].period_id

    presenter.view.binding_cycle_requested.emit("duration", period_id, None)

    assert [slot.field_ref.period_id for slot in presenter.sequence.slots] == [period_id]
    assert presenter.view.scan_view.page is not None
    assert "Bound" in presenter.view.scan_view.page.slots_text


def test_the_dot_cycles_off_scan_api_off(presenter, sequence) -> None:
    schedule = presenter.view.schedule_view
    period_id = sequence.periods[3].period_id

    presenter.view.binding_cycle_requested.emit("duration", period_id, None)
    assert len(presenter.sequence.slots) == 1
    assert presenter.sequence.api_parameters == ()

    presenter.view.binding_cycle_requested.emit("duration", period_id, None)
    assert presenter.sequence.slots == ()
    assert len(presenter.sequence.api_parameters) == 1
    assert presenter.sequence.api_parameters[0].field_ref.period_id == period_id

    presenter.view.binding_cycle_requested.emit("duration", period_id, None)
    assert presenter.sequence.slots == ()
    assert presenter.sequence.api_parameters == ()


def test_the_starter_program_matches_the_bound_fields(presenter, sequence) -> None:
    schedule = presenter.view.schedule_view
    scan = presenter.view.scan_view
    period_id = sequence.periods[3].period_id
    dac = _dac_port(sequence)
    presenter.view.binding_cycle_requested.emit("duration", period_id, None)
    presenter.view.binding_cycle_requested.emit("analog", period_id, dac.key)

    presenter.view.scan_template_requested.emit("column_stack")

    assert "np.column_stack" in scan.page.source_text
    # One column per bound slot, each seeded by its own kind.
    assert scan.page.source_text.count("np.linspace") == 2
    assert "DAC code" in scan.page.source_text


def test_running_the_program_keeps_a_table_of_the_right_width(presenter, sequence) -> None:
    schedule = presenter.view.schedule_view
    scan = presenter.view.scan_view
    presenter.view.binding_cycle_requested.emit("duration", sequence.periods[3].period_id, None)

    _run_scan(
        presenter.view,
        # In the unit the bound period is written in -- this pulse is authored
        # in seconds -- and inside what a 25-bit slot operand can hold.
        "import numpy as np\n"
        "scan_table = np.linspace(0.001, 0.2, 7).reshape(-1, 1)\n"
    )

    assert len(presenter._state.scan_rows) == 7
    assert presenter._state.scan_source_dirty is False
    assert "7 scan point" in scan.page.progress_text
    assert "more point" not in scan.page.table_text


def test_a_table_of_the_wrong_width_is_refused(presenter, sequence) -> None:
    """A column per bound slot: anything else would write the wrong field."""

    schedule = presenter.view.schedule_view
    scan = presenter.view.scan_view
    presenter.view.binding_cycle_requested.emit("duration", sequence.periods[3].period_id, None)

    _run_scan(
        presenter.view,
        "import numpy as np\nscan_table = np.zeros((5, 3))\n"
    )

    assert presenter._state.scan_rows == ()
    assert any("3 column" in text for text in presenter.view.warnings)


def test_a_program_that_raises_says_so_and_keeps_the_last_table(presenter, sequence) -> None:
    schedule = presenter.view.schedule_view
    scan = presenter.view.scan_view
    presenter.view.binding_cycle_requested.emit("duration", sequence.periods[3].period_id, None)
    _run_scan(
        presenter.view,
        "import numpy as np\nscan_table = (np.arange(4) + 1).reshape(-1, 1) * 0.001\n"
    )
    kept = presenter._state.scan_rows

    _run_scan(presenter.view, "raise RuntimeError('bad sweep')")

    assert presenter._state.scan_rows == kept
    assert any("bad sweep" in text for text in presenter.view.warnings)


def test_holding_a_point_stops_the_scan_and_loads_an_ordinary_pulse(presenter, sequence) -> None:
    """A held point is resolved into an ordinary repeating pulse."""

    board = _Sequencer()
    presenter.sequencer = board
    assert presenter.adopt_board() is True
    board.events.clear()
    schedule = presenter.view.schedule_view
    scan = presenter.view.scan_view
    presenter.view.binding_cycle_requested.emit("duration", sequence.periods[3].period_id, None)
    _run_scan(
        presenter.view,
        "import numpy as np\nscan_table = (np.arange(5) + 1).reshape(-1, 1) * 0.001\n"
    )

    presenter.view.scan_hold_requested.emit()
    assert board.events[-3:] == ["safe", "load", "fire forever"]
    assert board._applied.rows == (), "a held point must not become a one-row scan"

    presenter.view.scan_step_requested.emit(1)
    assert board.events[-3:] == ["safe", "load", "fire forever"]
    assert presenter._held_point == 1
    presenter.view.scan_step_requested.emit(-1)
    assert presenter._held_point == 0
    # It cannot step off either end of the table.
    for _ in range(10):
        presenter.view.scan_step_requested.emit(-1)
    assert presenter._held_point == 0
    assert board._applied.rows == ()


def test_the_table_is_uploaded_with_the_pulse(presenter, sequence) -> None:
    uploaded: list = []
    board = _Sequencer()
    original_load = board.load
    board.load = lambda program, *, source=None, rows=(): (
        uploaded.append(tuple(rows)),
        original_load(program, source=source, rows=rows),
    )[-1]
    presenter.sequencer = board
    assert presenter.adopt_board() is True
    board.events.clear()
    period_id = sequence.periods[3].period_id
    presenter.view.duration_committed.emit(period_id, 1.0, "s")
    presenter.view.binding_cycle_requested.emit("duration", period_id, None)
    # The authored duration is long and its span needs a two-tick slot scale.
    # The board receives signed deltas around the one-second base, while Sync
    # must report the exact values those deltas play rather than the
    # unquantized requests.
    _run_scan(
        presenter.view,
        "import numpy as np\n"
        "scan_table = np.array([0.50000002, 1.0, 1.49999998]).reshape(-1, 1)\n"
    )

    assert presenter.load_into_sequencer() is True
    assert board._applied.program.slot_tick_scales == (2,)
    assert uploaded == [((-12_500_000,), (0,), (12_500_000,))]
    assert presenter.sync_from_sequencer() is True
    assert presenter._state.scan_rows == ((0.5,), (1.0,), (1.5,))


def test_the_status_dot_says_what_the_board_is_doing(presenter, sequence) -> None:
    """The same answer as the buttons, at a glance, decided in one place."""

    view = presenter.view
    assert view.status_token == "idle", "no board, yet something looked ready"

    board = _Sequencer()
    presenter._dial = lambda *_args: board
    presenter.connect_to("virtual", "")
    assert view.status_token == "dirty-ready"

    # One host-only API parameter plus one scan field with no table still
    # prepares a fully static source.  If status compiled the authoring
    # document directly, unresolved API would make its digest empty forever.
    view.binding_cycle_requested.emit(
        "duration", sequence.periods[1].period_id, None
    )
    view.binding_cycle_requested.emit(
        "duration", sequence.periods[1].period_id, None
    )
    view.binding_cycle_requested.emit(
        "duration", sequence.periods[3].period_id, None
    )
    presenter.fire()
    assert view.status_token == "running-synced"
    assert board._applied.source.api_parameters == ()
    assert board._applied.source.slots == ()

    # Arming a real table changes only scan state, not PulseSequence revision:
    # it must still invalidate the prepared-program digest immediately.
    _run_scan(
        view,
        "import numpy as np\n"
        "scan_table = np.array([0.004, 0.005]).reshape(-1, 1)\n"
    )
    assert view.status_token == "running-stale"
    presenter.fire()
    assert view.status_token == "running-synced"
    assert len(board._applied.source.slots) == 1
    assert board._applied.source.api_parameters == ()

    # Editing while it runs means the board is playing something older.
    view.duration_committed.emit(sequence.periods[0].period_id, "7", "us")
    assert view.status_token == "running-stale"

    presenter.stop()
    assert view.status_token == "dirty-ready"


def test_renaming_a_period_does_not_make_the_board_stale(presenter, sequence) -> None:
    """Stale means the board is playing something else, not that a name moved.

    The board is asked what it holds, and what it holds is a compiled program.
    A period's name never reaches it.  Lighting the stale lamp for an edit that
    changes nothing the board plays is how an operator learns to ignore it --
    and then misses the edit that did matter.
    """

    board = _Sequencer()
    presenter._dial = lambda *_args: board
    presenter.connect_to("virtual", "")
    presenter.fire()

    presenter.view.period_name_committed.emit(
        sequence.periods[0].period_id, "MOT load"
    )

    assert presenter.view.status_token == "running-synced"


def test_the_window_never_reports_a_board_that_did_not_answer(presenter) -> None:
    """Attached and silent is its own state, and has to look like one.

    Keeping the last good answer on screen is how a window sits lit green over a
    server that died two minutes ago.
    """

    class _Mute(_Sequencer):
        def snapshot(self):
            raise ConnectionResetError("the server went away")

    board = _Mute()
    presenter._dial = lambda *_args: board
    presenter.connect_to("virtual", "")
    presenter.fire()

    assert presenter.view.status_token == "unreachable"
    assert presenter.running is False
    assert presenter.synchronized is False


def test_a_board_someone_else_loaded_reads_as_stale(presenter) -> None:
    """Nobody tells this window when a notebook takes the board.

    Which is why the window does not keep a copy: it asks, and the board says
    it is holding something this editor did not put there.
    """

    board = _Sequencer()
    presenter._dial = lambda *_args: board
    presenter.connect_to("virtual", "")
    presenter.fire()
    assert presenter.synchronized is True

    # A notebook loads its own pulse over the top and runs it.
    board._digest = "0123456789abcdef"
    presenter.refresh_run_state()

    assert presenter.synchronized is False
    assert presenter.view.status_token == "running-stale"


def test_stepping_is_offered_only_once_there_is_a_table(presenter, sequence) -> None:
    board = _Sequencer()
    presenter._dial = lambda *_args: board
    presenter.connect_to("virtual", "")
    assert presenter.view.capabilities[2] is False

    presenter.view.binding_cycle_requested.emit(
        "duration", sequence.periods[3].period_id, None
    )
    _run_scan(
        presenter.view,
        "import numpy as np\nscan_table = (np.arange(4) + 1).reshape(-1, 1) * 0.001\n"
    )
    assert presenter.view.capabilities[2] is True


def test_a_pulse_authored_in_the_units_zlc_pulse_accepts_can_be_opened() -> None:
    """The window used to declare its own four units and leave one out.

    A period authored in ticks then raised KeyError inside the projection --
    from a Qt slot, which ends the process rather than drawing anything.
    """

    from zlc_pulse.model import TIME_UNIT_CHOICES

    from zlc_workbench.pulse_editor import _TIME_UNITS, _nanoseconds

    assert set(_TIME_UNITS) == set(TIME_UNIT_CHOICES)
    for unit in TIME_UNIT_CHOICES:
        assert _nanoseconds(1.0, unit) > 0.0


def test_a_loaded_scan_file_is_checked_the_way_a_generated_one_is(presenter, tmp_path) -> None:
    """The loader used to skip the width check the generated path made.

    So a table with the wrong number of columns reached the board from a file
    and was refused from a program -- two answers to what a legal table is.
    """

    import numpy as np

    from zlc_pulse.scan import scan_columns_for, validate_scan_table

    presenter.view.binding_cycle_requested.emit(
        "duration", presenter.sequence.periods[3].period_id, None
    )
    columns = scan_columns_for(presenter.sequence)
    assert columns, "no slot was bound"

    with pytest.raises(ValueError, match="column"):
        validate_scan_table(np.zeros((4, len(columns) + 1)), columns)
    # One tick is the floor for a duration column, in whatever unit that
    # column is written in -- a zero-length period is not a scan point.
    assert validate_scan_table(np.full((4, len(columns)), 0.001), columns)

    authored = ((0.0041,), (0.0052,), (0.0063,))
    presenter._take_scan_rows(authored)
    presenter.path = str(tmp_path / "pulse.json")
    saved = presenter.save_scan_array()
    np.testing.assert_allclose(np.load(saved), authored)

    presenter._take_scan_rows(((0.01,),))
    presenter.view.open_answer = saved
    assert presenter.load_scan_array() is True
    assert presenter._state.scan_rows == authored


def test_connecting_opens_a_pulse_and_names_which_board_answered() -> None:
    """The two halves of "I connected and cannot tell what happened".

    Attaching used to leave the schedule with no periods, so a successful
    connection and a failed one looked the same; and the status line was built
    from ``endpoint or mode``, which reads the address box -- and the box keeps
    the remote server's address whichever mode is selected.  So the simulated
    board reported the same line a real one gives, next to an empty schedule.
    """

    view = _EditorView()
    board = _Sequencer()
    presenter = PulseEditorPresenter(view, None, dial=lambda _m, _e: board)
    try:
        assert presenter.connect_to("virtual", "127.0.0.1:18861") is True
        schedule = view.schedule_view.schedule
        assert schedule.ports, "an attached board must show its ports"
        assert len(schedule.periods) == 2, "and a pulse to edit on it"
        assert len(schedule.delay_rows) == len(
            [port for port in schedule.ports if port.kind in ("digital", "dac")]
        ), "every delayable output gets a row"

        status = view.schedule_view.connection.status
        assert status.startswith("virtual"), status
        assert "127.0.0.1" not in status, (
            "the address box is not what this editor dialled"
        )
    finally:
        presenter.close()


def test_hanging_up_retires_what_the_board_had_said_about_itself(sequence) -> None:
    """Offline authoring survives having once been connected.

    ``board``/``pins`` are facts of a CONNECTION, not properties of the editor.
    Keeping them past the disconnect left the editor believing it was attached
    forever, so the Target page stayed read-only and Offline -- the one mode
    whose point is authoring a target -- could never author one again.
    """

    view = _EditorView()
    board = _Sequencer()
    presenter = PulseEditorPresenter(view, sequence, dial=lambda _m, _e: board)
    try:
        presenter.connect_to("virtual", "")
        assert presenter.board is not None and presenter.pins is not None
        assert view.target_view.editable is False, "a board's topology is the board's"

        assert presenter.connect_to("offline", "") is True

        assert presenter.board is None, "the description belonged to the connection"
        assert presenter.pins == {}
        assert presenter.sequencer is None
        assert view.target_view.editable is True, "Offline authors the target"
    finally:
        presenter.close()


def test_hide_off_keeps_what_the_pulse_drives_and_show_all_brings_it_back(sequence) -> None:
    """Two defects behind one pair of buttons.

    "Hide Off" read a PortRowVM.active flag that was hardcoded True, so it hid
    nothing, ever.  Whether a lane is driven is edited constantly while the
    port rows are only re-pushed on a structural change, so a flag was always
    going to be stale -- the view computes it from the periods it holds.

    Then Hide Off followed by Show All produced two different models under one
    revision.  The view refuses that, correctly, and the refusal came out of a
    Qt slot: the real window aborted with no traceback at all.
    """

    view = _EditorView()
    presenter = PulseEditorPresenter(view, sequence)
    try:
        schedule = view.schedule_view.schedule
        assert sum(1 for port in schedule.ports if port.visible) == len(schedule.ports)

        # The pulse drives some subset; hiding leaves exactly that.
        driven = {
            key
            for period in schedule.periods
            for key, on in period.digital
            if on
        } | {
            key
            for period in schedule.periods
            for key, _mode, field in period.analog
            if field.text.strip()
        }
        assert driven, "this fixture drives at least one output"

        # Visibility is a VALUE, so it goes the value way: the rows are
        # re-flagged in place and the revision stands, because the SHAPE of
        # what is shown -- which periods, which ports exist -- has not moved.
        rebuilds = view.schedule_view.rebuilds
        presenter.set_visible_ports(tuple(driven))
        hidden = view.schedule_view.schedule
        assert {p.key for p in hidden.ports if p.visible} == driven
        assert view.schedule_view.rebuilds == rebuilds, "a value change is not a rebuild"

        presenter.set_visible_ports(tuple(port.key for port in schedule.ports))
        assert all(port.visible for port in view.schedule_view.schedule.ports)

        # And a real structural change afterwards is still recognised as one.
        presenter.insert_period(None)
        assert view.schedule_view.rebuilds > rebuilds
        assert view.schedule_view.schedule.revision != schedule.revision
    finally:
        presenter.close()


def test_a_bracket_repeats_at_least_twice_or_it_is_not_a_bracket(sequence) -> None:
    """Add Bracket silently undid itself.

    The view model carried default_bracket_count=1 and the presenter reads a
    count below the domain's minimum as "no repeat" -- correctly, because a
    region that plays its periods once IS the sequence.  So the button
    committed a count-1 region and the presenter cleared it, every time.  The
    minimum is the domain's to state, and now does.
    """

    from zlc_pulse import MINIMUM_BRACKET_COUNT

    view = _EditorView()
    presenter = PulseEditorPresenter(view, sequence)
    try:
        vm = view.schedule_view.schedule
        assert vm.default_bracket_count == MINIMUM_BRACKET_COUNT
        assert vm.min_bracket_count == MINIMUM_BRACKET_COUNT

        first, last = vm.periods[0].period_id, vm.periods[-1].period_id
        presenter.set_bracket(first, last, vm.default_bracket_count)
        bracket = presenter.sequence.bracket
        assert bracket is not None and bracket.count == MINIMUM_BRACKET_COUNT

        presenter.set_bracket(None, None, 0)
        assert presenter.sequence.bracket is None
    finally:
        presenter.close()


def test_a_bracket_of_one_is_refused_by_the_model_itself(sequence) -> None:
    """One encoding per pulse: the redundant one cannot be built at all."""

    import pytest as _pytest
    from zlc_pulse import PulseBracket

    with _pytest.raises(ValueError, match="bracket loops at least"):
        PulseBracket(sequence.periods[0].period_id, sequence.periods[-1].period_id, 1)


def test_preview_keeps_run_repeats_and_bracket_as_separate_markers(sequence) -> None:
    """Even a full-span bracket cannot replace the complete-Pulse Run loop."""

    from zlc_workbench.pulse_editor import RUN_FOREVER_LABEL, timeline_of

    view = _EditorView()
    presenter = PulseEditorPresenter(view, sequence)
    try:
        while len(presenter.sequence.periods) < 4:
            presenter.insert_period(None)
        ids = [period.period_id for period in presenter.sequence.periods]
        total = timeline_of(presenter.sequence).total_duration

        # Nothing bracketed: the complete Pulse repeats forever.
        plain = timeline_of(presenter.sequence)
        assert [(m.start, m.stop, m.label) for m in plain.loop_markers] == [
            (0.0, total, RUN_FOREVER_LABEL)
        ]

        # A bracket over everything is still the inner loop; both are shown.
        presenter.set_bracket(ids[0], ids[-1], 3)
        whole = timeline_of(presenter.sequence)
        assert [marker.label for marker in whole.loop_markers] == [
            "Bracket ×3",
            RUN_FOREVER_LABEL,
        ]
        assert all((marker.start, marker.stop) == (0.0, total) for marker in whole.loop_markers)

        # A finite Run value changes only its marker, not the bracket.
        presenter.set_bracket(ids[1], ids[2], 5)
        presenter.set_run_repeats(7)
        part = timeline_of(presenter.sequence)
        assert [marker.label for marker in part.loop_markers] == [
            "Bracket ×5",
            "Run ×7",
        ]
        inner, outer = part.loop_markers
        assert 0.0 < inner.start and inner.stop < total
        assert (outer.start, outer.stop) == (0.0, total)
    finally:
        presenter.close()


def test_every_bindable_field_shows_the_slot_it_is_bound_to(sequence) -> None:
    """The dot has always been able to say which column a field became.

    FluentScanLineEdit takes a binding and a number and paints an orange s1 or
    a violet API mark, and nothing ever told it: the projection built every
    field as a bare value, so a duration bound to a slot looked exactly like
    one that was not bound at all.  The bind took; the screen never said so.
    """

    view = _EditorView()
    presenter = PulseEditorPresenter(view, sequence)
    try:
        period = presenter.sequence.periods[0].period_id
        dac = next(
            port.key
            for port in presenter.sequence.target.ports
            if port.kind == "dac"
        )
        delayable = next(
            port.key
            for port in presenter.sequence.target.ports
            if port.kind == "digital"
        )

        presenter.cycle_binding("duration", period, None)      # -> scan
        presenter.cycle_binding("analog", period, dac)         # -> scan
        presenter.cycle_binding("delay", None, delayable)      # -> api

        schedule = view.schedule_view.schedule
        card = next(item for item in schedule.periods if item.period_id == period)
        assert card.duration.binding_kind == "scan"
        assert card.duration.binding_number >= 1

        analog = {key: field for key, _mode, field in card.analog}
        assert analog[dac].binding_kind == "scan"
        assert analog[dac].binding_number >= 1
        assert not analog[dac].value_is_typed if hasattr(analog[dac], "value_is_typed") else True

        row = next(item for item in schedule.delay_rows if item.port_key == delayable)
        assert row.value.binding_kind == "api"

        # Numbers are the slot positions, so no two bound fields share one.
        numbers = {card.duration.binding_number, analog[dac].binding_number,
                   row.value.binding_number}
        assert len(numbers) == 3, numbers

        # Cycling off takes the mark away again.
        presenter.cycle_binding("duration", period, None)      # scan -> api
        presenter.cycle_binding("duration", period, None)      # api  -> off
        card = next(
            item
            for item in view.schedule_view.schedule.periods
            if item.period_id == period
        )
        assert card.duration.binding_kind == ""
    finally:
        presenter.close()


def test_showing_every_row_grows_the_preview_widget_too(sequence) -> None:
    """The canvas was sized once, at mount, and never again.

    So a pulse that GREW -- two rows becoming twenty-two the moment Show off
    rows is switched on -- was drawn in full into a widget still shaped for the
    old one: the operator saw the top three channels and blank space below.
    The host knows its new size; the update now hands it back and the widget
    is re-sized with it.
    """

    sizes: list[tuple[int, int]] = []
    view = _EditorView()

    def _make(data, *, size):
        sizes.append((len(data.channels), 0))
        return _PreviewHost(data, (100, 50 * max(1, len(data.channels))))

    def _update(host, data, *, size):
        # What the real composition root does: ask the host to re-plan.  The
        # host is what knows the new size, so it is the host that is updated.
        host.logical_size = (100, 50 * max(1, len(data.channels)))
        return host.logical_size

    presenter = PulseEditorPresenter(
        view,
        sequence,
        make_preview=_make,
        update_preview=_update,
        run_preview_work=_run_preview_immediately,
    )
    presenter.show_page("Preview")
    try:
        presenter.view.preview_view._include_off = False
        presenter.refresh_preview()
        narrow = view.preview_view.logical_size

        presenter.view.preview_view._include_off = True
        presenter.refresh_preview()
        wide = view.preview_view.logical_size

        assert narrow is not None and wide is not None
        assert wide[1] > narrow[1], (
            f"more rows must mean a taller widget: {narrow} -> {wide}"
        )
    finally:
        presenter.close()


def test_sync_brings_the_board_s_pulse_back_into_the_editor(sequence) -> None:
    """That is what Sync means, and the direction it has to run.

    A notebook or a raw API call changes the device behind this window's back,
    and nothing else lets the window catch up.  It used to push the other way
    -- editor onto board -- which is what On Pulse does anyway, so the button
    duplicated one action and left the one nobody else performs undone.
    """

    view = _EditorView()
    board = _Sequencer()
    presenter = PulseEditorPresenter(view, sequence, dial=lambda _m, _e: board)
    try:
        presenter.connect_to("virtual", "")

        # Sync needs a BOARD, not a pulse: an editor with nothing open is
        # exactly when reading what the hardware holds is worth doing.
        assert view.schedule_view.capabilities[0] is True
        presenter._accept_state(PulseEditorState())
        presenter.refresh()
        assert view.schedule_view.capabilities[0] is True

        assert presenter.sync_from_sequencer() is False
        assert any("nothing to sync" in text for text in view.warnings)

        presenter._accept_state(PulseEditorState(sequence=sequence))
        presenter.load_into_sequencer()
        held = len(sequence.periods)

        presenter.insert_period(None)
        assert len(presenter.sequence.periods) == held + 1

        assert presenter.sync_from_sequencer() is True
        assert len(presenter.sequence.periods) == held, (
            "the editor shows what the board is holding, not what it had drifted to"
        )
        assert any("synced from the board" in text for text in view.done)
    finally:
        presenter.close()


def test_the_scan_page_says_what_to_do_before_it_says_what_failed(sequence) -> None:
    """Tell the operator their next move instead of naming a symptom.

    A scan table has one column per bound field, so with nothing bound there
    is no table to make or read at all -- and the answer is a click on a dot
    in the Edit tab, not a column count from a validator.  The file dialog is
    also asked for AFTER that check, because picking a file and then being
    told the slots were never bound wastes the choice just made.
    """

    view = _EditorView()
    view.open_answer = "/nowhere/table.npy"
    presenter = PulseEditorPresenter(view, sequence)
    try:
        assert not presenter.sequence.slots, "this fixture starts unbound"

        presenter.edit_scan_source("scan_table = [[1]]")
        assert presenter.run_scan_program() is False
        assert any("click a dot" in text for text in view.warnings)

        view.warnings.clear()
        view.asked = None
        assert presenter.load_scan_array() is False
        assert any("click a dot" in text for text in view.warnings)
        assert view.asked is None, "the dialog must not open before the check"

    finally:
        presenter.close()


def test_new_pulse_replaces_the_whole_editor_state_in_one_candidate(
    sequence, tmp_path
) -> None:
    view = _EditorView()
    presenter = PulseEditorPresenter(
        view,
        sequence,
        path=str(tmp_path / "old.json"),
    )
    try:
        period_id = sequence.periods[3].period_id
        view.binding_cycle_requested.emit("duration", period_id, None)
        _run_scan(
            view,
            "import numpy as np\nscan_table = np.array([[0.001], [0.002]])\n",
        )
        view.scan_source_edited.emit("half typed")
        view.scan_repeats_committed.emit(4)
        view.visible_ports_committed.emit(
            tuple(row.key for row in view.schedule_view.schedule.ports[:2])
        )
        baseline = presenter._saved_state

        assert presenter.start_new_pulse() is True

        assert presenter.path == ""
        assert presenter.sequence.name == "untitled"
        assert presenter.sequence.run_repeats == 0
        assert presenter.sequence.bracket is None
        assert presenter._state.scan_source == ""
        assert presenter._state.scan_rows == ()
        assert presenter._state.scan_source_dirty is False
        assert presenter._state.scan_repeats == 0
        assert presenter._state.visible_ports is None
        assert presenter._saved_state is baseline
    finally:
        presenter.close()


def test_brackets_never_choose_the_outer_execution_count(sequence) -> None:
    """Whole and partial brackets are timeline loops; On Pulse remains continuous."""

    view = _EditorView()
    board = _Sequencer()
    presenter = PulseEditorPresenter(view, sequence, sequencer=board)
    board.events.clear()
    try:
        while len(presenter.sequence.periods) < 3:
            presenter.insert_period(None)
        ids = [period.period_id for period in presenter.sequence.periods]
        view.run_repeats_committed.emit(4)
        assert presenter.sequence.run_repeats == 4
        assert view.schedule_view.schedule.run_repeats == 4

        presenter.set_bracket(ids[0], ids[-1], 3)
        assert presenter.fire() is True
        assert board.events == ["load", "fire"], board.events
        assert (board._run_repeats, board._scan_repeats) == (4, 1)
        assert presenter.compile().loop_count == 3
        assert view.schedule_view.control_state[1] is True
        view.run_repeats_committed.emit(5)
        assert view.schedule_view.control_state[1] is False
        view.run_repeats_committed.emit(4)

        # A bracket over PART has the same outer execution meaning.
        board.events.clear()
        presenter.set_bracket(ids[1], ids[2], 5)
        assert presenter.fire() is True
        assert board.events == ["safe", "load", "fire"], board.events
        assert (board._run_repeats, board._scan_repeats) == (4, 1)
    finally:
        presenter.close()




def test_the_selectors_switch_reaches_the_plot_in_both_directions(presenter) -> None:
    """It set a flag that was read in one place: the arguments a host is BUILT
    with.  After the first draw the switch did nothing, either way."""

    host = presenter._preview_host
    assert host is not None, "the Preview page is open, so there is a plot"

    presenter.view.preview_selectors_toggled.emit(False)
    assert host.interaction is False
    presenter.view.preview_selectors_toggled.emit(True)
    assert host.interaction is True



def test_a_bound_field_is_drawn_where_it_happens(presenter, sequence) -> None:
    """Binding is the most consequential edit on this page, and the picture
    did not show it: zlc_plot has drawn numbered scan regions and coloured DAC
    segments all along, and nothing ever built one, so a bound pulse and an
    unbound one previewed identically."""

    from zlc_workbench.pulse_editor import timeline_of

    view = presenter.view
    dac = next(port for port in sequence.target.ports if port.kind == "dac")
    presenter.set_analog(sequence.periods[1].period_id, dac.key, "edge", 200)

    view.binding_cycle_requested.emit("duration", sequence.periods[3].period_id, None)
    view.binding_cycle_requested.emit("analog", sequence.periods[1].period_id, dac.key)
    view.binding_cycle_requested.emit("analog", sequence.periods[1].period_id, dac.key)

    data = timeline_of(presenter.sequence, include_off=True)
    region = next(iter(data.scan_regions))
    segment = next(iter(data.scan_dac_segments))
    # Over the period whose duration is swept, badged with the column an
    # operator will find it in.
    assert region.number == 1 and region.kind == "scan"
    # And on the trace whose level is written -- by a host, one row at a time,
    # which is what the second press means and what the colour will say.
    assert segment.trace_name == dac.key and segment.kind == "api"
    assert segment.value == 200.0



def test_the_grid_a_field_is_snapped_to_is_in_that_field_s_unit(sequence) -> None:
    """The board's grid is 20 ns.  The box is in milliseconds.

    Passing the raw 20 snapped a 5 ms period to a multiple of 20 MILLISECONDS
    and refused anything under 20 ms -- a device rule applied to a number that
    is not in the device's unit.
    """

    from zlc_pulse import PulsePeriod, PulseSequence
    from zlc_workbench.pulse_editor import project_schedule

    safe = (0,) * len(sequence.target.raw_lanes)
    for unit, expected in (("ns", 20.0), ("us", 0.02), ("ms", 2e-5)):
        one = PulseSequence(
            name="t",
            target=sequence.target,
            time_step_ns=20.0,
            periods=(PulsePeriod("p1", 1000.0 if unit == "ns" else 5.0, unit, safe),),
        )
        field = project_schedule(one).periods[0].duration
        assert field.resolution == pytest.approx(expected), unit
        assert field.validator_lo == pytest.approx(expected), unit


def test_the_edit_page_says_how_many_points_will_be_played(presenter, sequence) -> None:
    """Slots is half the question; the other half is what will actually run."""

    view = presenter.view
    assert view.schedule_view.schedule.scan_summary_text == "no scan slots"

    view.binding_cycle_requested.emit("duration", sequence.periods[3].period_id, None)
    assert view.schedule_view.schedule.scan_summary_text == "1 slot - 0 pts"

    _run_scan(
        view,
        "import numpy as np\n"
        "scan_table = (np.arange(21) + 1).reshape(-1, 1) * 0.001\n"
    )
    assert view.schedule_view.schedule.scan_summary_text == "1 slot - 21 pts"



def test_hold_and_step_play_the_point_they_hold(presenter, sequence) -> None:
    """Hold in the middle of a scan simply turned the outputs off.

    It stopped, wrote the row and left it there -- and writing slot values is
    not playing them.  Holding a point is how a scan is inspected: the pulse
    keeps running with the sweep frozen at one set of values, so a camera sees
    that point and nothing else.
    """

    view = presenter.view
    board = _Sequencer()
    board.cursor = lambda: 0
    presenter.sequencer = board
    assert presenter.adopt_board() is True
    board.events.clear()
    period_id = sequence.periods[3].period_id
    view.duration_committed.emit(period_id, 1.0, "s")
    view.binding_cycle_requested.emit("duration", period_id, None)
    _run_scan(
        view,
        "import numpy as np\n"
        "scan_table = np.array([0.50000002, 1.0, 1.49999998]).reshape(-1, 1)\n"
    )
    assert presenter.load_into_sequencer() is True
    assert board._applied.program.slot_tick_scales == (2,)

    board.events.clear()
    view.scan_hold_requested.emit()
    assert board.events == ["safe", "load", "fire forever"], board.events
    assert presenter._held_point == 0
    held = board._applied.source.period_by_id[period_id].duration
    assert held == pytest.approx(0.5)
    assert board._applied.rows == ()

    board.events.clear()
    view.scan_step_requested.emit(1)
    assert board.events == ["safe", "load", "fire forever"], "a step must play its point too"
    assert presenter._held_point == 1
    assert board._applied.source.period_by_id[period_id].duration == pytest.approx(1.0)

    view.scan_step_requested.emit(-1)
    assert board._applied.source.period_by_id[period_id].duration == pytest.approx(held)
    assert board._applied.rows == ()


def test_scan_repeats_reaches_the_wire(presenter, sequence) -> None:
    """It was stored, shown and saved, and read by nothing at all.

    The board stores one unique table and its independent sweep counter replays
    that table without duplicating rows in host memory or on the wire.
    """

    uploaded: list = []
    board = _Sequencer()
    original_load = board.load
    board.load = lambda program, *, source=None, rows=(): (
        uploaded.append((len(rows), len(rows[0]))),
        original_load(program, source=source, rows=rows),
    )[-1]
    presenter.sequencer = board
    assert presenter.adopt_board() is True
    board.events.clear()
    view = presenter.view
    view.binding_cycle_requested.emit("duration", sequence.periods[3].period_id, None)
    view.binding_cycle_requested.emit("duration", sequence.periods[1].period_id, None)
    view.binding_cycle_requested.emit("duration", sequence.periods[1].period_id, None)
    assert len(presenter.sequence.slots) == 1
    assert len(presenter.sequence.api_parameters) == 1
    _run_scan(
        view,
        "import numpy as np\n"
        "scan_table = np.linspace(0.001, 0.2, 7).reshape(-1, 1)\n"
    )
    assert "stays at the first scan point" in view.scan_view.page.slots_text

    # Both zero counts mean until Stop, without changing either one to 1.
    board.events.clear()
    assert presenter.fire() is True
    assert "fire forever" in board.events, board.events
    assert (board._run_repeats, board._scan_repeats) == (0, 0)

    view.run_repeats_committed.emit(2)
    assert "stays at the first scan point" not in view.scan_view.page.slots_text
    view.scan_repeats_committed.emit(3)
    board.events.clear()
    assert presenter.fire() is True
    # The table is uploaded once; each point runs the Pulse twice and the
    # complete table is swept three times, in independent hardware counters.
    assert uploaded[-1] == (7, 1)
    assert board._run_repeats == 2
    assert board._scan_repeats == 3
    assert len(board._applied.source.slots) == 1
    assert board._applied.source.api_parameters == ()
    # And a counted number of sweeps is a finite run: wrapping it in the outer
    # forever would repeat the whole scan endlessly and the count would mean
    # nothing.
    assert "fire forever" not in board.events, board.events

    view.scan_repeats_committed.emit(0)
    board.events.clear()
    assert presenter.fire() is True
    assert uploaded[-1] == (7, 1)
    assert (board._run_repeats, board._scan_repeats) == (2, 0)
    assert "fire forever" in board.events, board.events



def test_scan_repeats_govern_nothing_when_no_scan_is_left(presenter, sequence) -> None:
    """Scan repeats is a statement about a scan; without one it is silence.

    Bind a field, run a table, set repeats, then UNBIND: yesterday's rows are
    still in memory but no field carries them.  The upload gate already knew
    that and sent no table -- while the run-length decision looked only at the
    rows, so a pulse with no scan left in it fired ONCE and stopped, its own
    repeat-forever overwritten by a ghost table.
    """

    board = _Sequencer()
    presenter.sequencer = board
    assert presenter.adopt_board() is True
    board.events.clear()
    view = presenter.view
    period = sequence.periods[3].period_id

    view.binding_cycle_requested.emit("duration", period, None)
    _run_scan(
        view,
        "import numpy as np\n"
        "scan_table = (np.arange(4) + 1).reshape(-1, 1) * 0.001\n"
    )
    view.scan_repeats_committed.emit(3)
    # scan -> api -> off: the field is unbound again.
    view.binding_cycle_requested.emit("duration", period, None)
    view.binding_cycle_requested.emit("duration", period, None)
    assert not presenter.sequence.slots

    board.events.clear()
    assert presenter.fire() is True
    assert board._applied.rows == ()
    assert (board._run_repeats, board._scan_repeats) == (0, 1)
    assert "fire forever" in board.events, (
        "with no scan, the pulse's own repeat meaning governs: " + str(board.events)
    )


def test_on_pulse_over_a_running_pulse_stops_it_first(sequence) -> None:
    """The complaint in one test: "cannot load this pulse: already firing".

    On Pulse is how an edit reaches an experiment that is already running, so
    it is pressed most often while the board is busy.  The device refuses to
    load over a firing streamer -- rewriting the tables under the engine is
    exactly what it should refuse -- so the gesture that means "play this now"
    is what has to turn it off first.
    """

    view = _EditorView()
    board = _Sequencer()
    presenter = PulseEditorPresenter(view, sequence, dial=lambda _m, _e: board)
    try:
        presenter.connect_to("virtual", "")
        assert presenter.fire() is True
        assert presenter.running is True

        board.events.clear()
        assert presenter.fire() is True, "a second On Pulse must work"
        assert board.events == ["safe", "load", "fire forever"], board.events
    finally:
        presenter.close()


def test_every_dac_in_one_period_can_be_bound(presenter, sequence) -> None:
    """A slot name has to carry all of what identifies the field it names.

    The model says what a field IS: a duration is a period, a delay is a port,
    a DAC is a period AND a port.  The name was built from "the first part that
    happens to be set", so every DAC in one period got the same one and binding
    a second was refused with "slot ids must be unique" -- one scannable field
    per period, and every other dot on that card answering for a rule the
    operator had not broken.
    """

    period_id = presenter.sequence.periods[0].period_id
    dacs = [port for port in sequence.target.ports if port.kind == "dac"]
    assert len(dacs) >= 2, "this board has only one DAC to bind"

    for port in dacs:
        presenter.cycle_binding("analog", period_id, port.key)

    bound = {
        (slot.field_ref.period_id, slot.field_ref.port)
        for slot in presenter.sequence.slots
        if slot.field_ref.kind == "dac"
    }
    assert bound == {(period_id, port.key) for port in dacs}, presenter.view.warnings
    assert not [text for text in presenter.view.warnings if "unique" in text]

    ids = [slot.slot_id for slot in presenter.sequence.slots]
    assert len(ids) == len(set(ids)), ids


def test_choosing_hold_puts_the_dac_back_to_holding(presenter, sequence) -> None:
    """Hold is not a third mode; it is what no step already means.

    The projection reads it that way -- a period with no step for a DAC leaves
    the output where the period before it put it -- and the card offers it in
    the same box as Edge and Ramp.  Coming back, it was built into an
    AnalogStep("hold"), which the model refuses because ANALOG_MODES is edge
    and ramp.  The refusal left a Qt slot, and PyQt5 ends the process on that:
    choosing Hold in the shipped window closed it with no traceback at all.
    """

    from zlc_workbench.pulse_editor import HOLD_MODE, _analog_mode

    period_id = presenter.sequence.periods[0].period_id
    port = next(port for port in sequence.target.ports if port.kind == "dac")

    presenter.view.analog_committed.emit(period_id, port.key, "edge", 120)
    period = presenter.sequence.period_by_id[period_id]
    assert _analog_mode(period, port) == "edge"

    presenter.view.analog_committed.emit(period_id, port.key, HOLD_MODE, 120)

    period = presenter.sequence.period_by_id[period_id]
    assert _analog_mode(period, port) == HOLD_MODE, "Hold has to survive the round trip"
    assert not any(step.port == port.key for step in period.analog_steps), (
        "holding is the absence of a step, not a step that says 'hold'"
    )
    assert presenter.view.warnings == [], presenter.view.warnings


def test_a_dead_server_connection_never_holds_the_window_hostage(sequence) -> None:
    """close(present=False) succeeds when the server is already gone.

    The pulse server's own law drives AUTO-SAFE the moment a client
    disconnects (and on its own shutdown), so a client with no channel has
    no safety obligation left -- and no right to refuse its close.  Only a
    CONNECTED board that refuses SAFE may still block, because then the
    outputs really are live and reachable.
    """

    view = _EditorView()
    board = _Sequencer()
    presenter = PulseEditorPresenter(view, sequence, sequencer=board)
    view.fire_requested.emit()
    assert presenter._drive_lease is not None, "the editor holds the drive"

    def gone(*_args, **_kwargs):
        raise ConnectionError(
            "the connection to the pulse server ended; if another editor "
            "has connected since, that one now holds the board"
        )

    board.safe = gone
    board.snapshot = gone
    presenter.close(present=False)
    assert presenter._drive_lease is None

    # A CONNECTED refusal still blocks -- that guard is load-bearing.
    view2 = _EditorView()
    board2 = _Sequencer()
    presenter2 = PulseEditorPresenter(view2, sequence, sequencer=board2)
    view2.fire_requested.emit()

    def refused(*_args, **_kwargs):
        raise RuntimeError("SAFE readback was not stable")

    board2.safe = refused
    try:
        presenter2.close(present=False)
    except RuntimeError as error:
        assert "did not go safe" in str(error)
    else:
        raise AssertionError("a connected SAFE refusal must still block close")


def test_a_defective_handler_warns_instead_of_killing_the_editor(sequence) -> None:
    """The editor's forty-one view signals cross the same qFatal boundary.

    A dead remote connection raising out of a gesture used to end the
    process; guarded at connect time, it is a warning and the editor keeps
    running -- while direct calls (the rest of this suite) still raise.
    """

    view = _EditorView()
    presenter = PulseEditorPresenter(view, sequence)

    def detonate(*_args):
        raise LookupError("wired to fail")

    presenter.move_period = detonate
    view.move_period_requested.emit("p0", 1)
    assert any(
        "internal error in move_period" in warning and "wired to fail" in warning
        for warning in view.warnings
    ), view.warnings
    import pytest as _pytest

    with _pytest.raises(LookupError):
        presenter.move_period("p0", 1)


# ---------------------------------------------------------------------------
# The device worker: no device conversation the window starts runs on its thread.


class _DeviceWorker:
    """The window's device worker without Qt: work on its own thread, delivery here.

    ``attach_qt_worker`` runs each job on one worker thread and delivers its
    outcome on the owner's next turn.  This does the same with a queue the
    test drains, so a test can look at the presenter BETWEEN the request and
    the delivery -- which is the whole point of the worker: the GUI thread
    is free in between.
    """

    def __init__(self) -> None:
        import queue

        self._outcomes: "queue.Queue" = queue.Queue()

    def __call__(self, work, delivered, failed) -> None:
        from threading import Thread

        def run() -> None:
            try:
                result = work()
            except BaseException as error:
                self._outcomes.put(lambda: failed(error))
            else:
                self._outcomes.put(lambda: delivered(result))

        Thread(target=run, name="pulse-device-worker", daemon=True).start()

    def deliver_until(self, predicate, seconds: float = 3.0) -> None:
        """Deliver outcomes on this thread until ``predicate`` holds."""

        import queue

        deadline = time.monotonic() + seconds
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                outcome = self._outcomes.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            outcome()
        assert predicate(), "the device worker never delivered what was waited for"


class _ThreadWatchingSequencer(_Sequencer):
    """A board that remembers which thread each call came from."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.callers: list[tuple[str, str]] = []
        self.cursor_value = 0

    def _note(self, what: str) -> None:
        from threading import current_thread

        self.callers.append((what, current_thread().name))

    def snapshot(self) -> dict:
        self._note("snapshot")
        return super().snapshot()

    def load(self, prog, *, source=None, rows=()) -> None:
        self._note("load")
        super().load(prog, source=source, rows=rows)

    def fire(self, *, run_repeats: int, scan_repeats: int = 1) -> None:
        self._note("fire")
        super().fire(run_repeats=run_repeats, scan_repeats=scan_repeats)

    def safe(self):
        self._note("safe")
        return super().safe()

    def applied(self):
        self._note("applied")
        return super().applied()

    def describe(self):
        self._note("describe")
        return super().describe()

    def cursor(self) -> int:
        self._note("cursor")
        return self.cursor_value

    def threads_for(self, what: str) -> set[str]:
        return {thread for name, thread in self.callers if name == what}


def test_the_timer_question_goes_to_the_device_worker(sequence) -> None:
    """The 100 ms poll never makes the GUI thread wait on the board's socket.

    On the experiment machine that socket answers only when the pulse
    server is free of its UART transaction, and the main thread spent a
    third of its time waiting there.  The question is asked on the worker;
    five requests while one is pending send one question; the answer is
    shown when it comes.  The public ``refresh_run_state`` still answers
    before returning, on the caller's thread -- that is a notebook's form.
    """

    view = _EditorView()
    board = _ThreadWatchingSequencer(description=_board_description())
    worker = _DeviceWorker()
    presenter = PulseEditorPresenter(
        view, sequence, sequencer=board, run_device_work=worker, run_safe_work=_run_preview_immediately
    )
    board.callers.clear()
    try:
        for _ in range(5):
            assert presenter.ask_run_state() is True
        worker.deliver_until(lambda: not presenter._status_in_flight)
        assert board.callers == [("snapshot", "pulse-device-worker")]
        assert presenter._board_state.attached is True

        presenter.refresh_run_state()
        assert board.callers[-1] == ("snapshot", "MainThread")
    finally:
        presenter.close()


def test_a_status_answer_from_before_a_command_is_dropped(sequence) -> None:
    """After On Pulse the board's state comes from the command, not from a question asked before it."""

    from threading import Event

    gate = Event()

    class _SlowAnswer(_ThreadWatchingSequencer):
        slow = False

        def snapshot(self) -> dict:
            answer = super().snapshot()
            if self.slow and not answer["firing"]:
                # Asked before the shot, answered after it: the answer says
                # "not firing" about a board that by then is.
                gate.wait(3.0)
            return answer

    view = _EditorView()
    board = _SlowAnswer(description=_board_description())
    worker = _DeviceWorker()
    presenter = PulseEditorPresenter(
        view, sequence, sequencer=board, run_device_work=worker, run_safe_work=_run_preview_immediately
    )
    board.slow = True
    board.callers.clear()
    try:
        assert presenter.ask_run_state() is True
        view.fire_requested.emit()
        assert presenter._device_busy is True
        gate.set()
        worker.deliver_until(
            lambda: not presenter._device_busy and not presenter._status_in_flight
        )
        assert board.events[-2:] == ["load", "fire forever"]
        assert presenter.running is True, "a stale answer overwrote the command's state"
        assert view.status_token == "running-synced"
        assert {thread for _name, thread in board.callers} == {"pulse-device-worker"}
    finally:
        board.slow = False
        presenter.close()


def test_connect_hold_step_and_sync_run_on_the_device_worker(sequence) -> None:
    """Every button that talks to the board talks to it off the GUI thread.

    Connect dials and reads the board on the worker and this editor changes
    what it is connected to only when that delivers; Hold and Step ask the
    board where it is, stop it, write the row and play it there; Sync reads
    what is applied there.  Each shows its outcome when the outcome comes.
    """

    view = _EditorView()
    board = _ThreadWatchingSequencer(description=_board_description())
    worker = _DeviceWorker()
    presenter = PulseEditorPresenter(
        view,
        sequence,
        dial=lambda _mode, _endpoint: board,
        run_device_work=worker, run_safe_work=_run_preview_immediately,
    )
    try:
        view.connection_requested.emit("remote", "127.0.0.1:18861")
        assert presenter.sequencer is None, "the connection was made on the GUI thread"
        assert presenter._device_busy is True
        worker.deliver_until(lambda: presenter.sequencer is board)
        assert board.threads_for("describe") == {"pulse-device-worker"}
        status = view.schedule_view.connection.status
        assert "127.0.0.1:18861" in status and "ports" in status
        assert view.schedule_view.capabilities == (True, True, False)

        period_id = sequence.periods[3].period_id
        view.duration_committed.emit(period_id, 1.0, "s")
        view.binding_cycle_requested.emit("duration", period_id, None)
        _run_scan(
            view,
            "import numpy as np\nscan_table = np.array([0.5, 1.0, 1.5]).reshape(-1, 1)\n",
        )
        board.cursor_value = 4  # cumulative row visit: sweep 2, table row 1
        board.events.clear()
        board.callers.clear()
        view.scan_hold_requested.emit()
        assert presenter._device_busy is True
        assert presenter._held_point is None, "the hold was settled on the GUI thread"
        worker.deliver_until(lambda: not presenter._device_busy)
        assert board.events == ["safe", "load", "fire forever"]
        assert {thread for _name, thread in board.callers} == {"pulse-device-worker"}
        assert presenter._held_point == 1
        assert "held at scan point 1" in presenter._scan_progress

        view.scan_step_requested.emit(1)
        worker.deliver_until(lambda: not presenter._device_busy)
        assert presenter._held_point == 2
        assert board.events[-3:] == ["safe", "load", "fire forever"]

        board.callers.clear()
        view.sync_requested.emit()
        assert presenter._device_busy is True
        worker.deliver_until(lambda: not presenter._device_busy)
        assert board.threads_for("applied") == {"pulse-device-worker"}
        assert not [text for text in view.warnings if "cannot sync" in text]
        assert presenter.sequence.period_by_id[period_id].duration == pytest.approx(1.5)
    finally:
        presenter.close()


def test_a_second_command_while_one_runs_is_refused_not_queued(sequence) -> None:
    """Two commands on the board at once is not a state the editor allows."""

    from threading import Event

    gate = Event()

    class _SlowLoad(_ThreadWatchingSequencer):
        def load(self, prog, *, source=None, rows=()) -> None:
            gate.wait(3.0)
            super().load(prog, source=source, rows=rows)

    view = _EditorView()
    board = _SlowLoad(description=_board_description())
    worker = _DeviceWorker()
    presenter = PulseEditorPresenter(
        view, sequence, sequencer=board, run_device_work=worker, run_safe_work=_run_preview_immediately
    )
    try:
        view.fire_requested.emit()
        view.sync_requested.emit()
        assert any("already in progress" in text for text in view.warnings)
        gate.set()
        worker.deliver_until(lambda: not presenter._device_busy)
        assert presenter.running is True
    finally:
        gate.set()
        presenter.close()

