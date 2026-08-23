"""The five behaviours the owner named, pinned so they cannot quietly go.

Each was verified once with a throwaway probe, which proves a moment and
nothing after it -- and this session then moved the scroll keeper, the
visibility path and the panel-kind path underneath all of them.  A behaviour
someone had to ask for twice is a behaviour that earns a standing guard.

Driven through the real presenter over the real virtual board, because every
one of these is a claim about what an operator SEES, and a claim about a view
model is a claim about something else.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_workbench.pulse_editor import programmable_ports, project_schedule


@pytest.fixture
def board():
    """A real virtual board, described by itself."""

    from zlc_pulse import load_streamer_config, pulse_target_from_xdc
    from zlc_pulse.device import PulseStreamer
    from zlc_pulse.transport import MemoryRegisterTransport

    config = load_streamer_config()
    streamer = PulseStreamer(
        MemoryRegisterTransport(geom=config["params"], auto_done=True),
        config["params"],
        config["clock_hz"],
        target=pulse_target_from_xdc(config_path=config["source"]),
    )
    streamer.open()
    try:
        yield streamer.describe()
    finally:
        streamer.close()


def test_a_dac_is_one_row_carrying_its_own_latch_clock(board) -> None:
    """A DAC is ONE output: its data lanes and the clock that latches them are
    one wire bundle to the person editing a pulse, and the clock is not
    something a pulse drives at all -- the compiler emits it from the DAC's own
    edges.  Listing it separately offers an edit that cannot be made and splits
    one output across two rows."""

    vm = project_schedule(None, target=board.target, time_step_ns=board.time_step_ns)
    kinds = {port.kind for port in vm.ports}
    assert "clock" not in kinds, "a latch clock is part of its DAC, not a row"
    assert {"digital", "dac"} <= kinds

    owned = {
        port.latch_clock
        for port in board.target.ports
        if port.kind == "dac" and port.latch_clock
    }
    assert owned, "this board has DACs with latch clocks"
    shown = {port.key for port in programmable_ports(board.target)}
    assert not (owned & shown), "every owned clock is folded into its DAC"
    for row in vm.ports:
        if row.kind == "dac":
            assert "latch clock" in row.endpoint_tooltip, (
                "the folded clock stays visible where it can be read"
            )


def test_every_delayable_output_has_a_delay_row(board) -> None:
    """A column of channel names beside no delays reads as a half-loaded window."""

    vm = project_schedule(None, target=board.target, time_step_ns=board.time_step_ns)
    delayable = [port for port in vm.ports if port.kind in ("digital", "dac")]
    assert len(vm.delay_rows) == len(delayable) > 0
    assert {row.port_key for row in vm.delay_rows} == {port.key for port in delayable}


def test_target_rows_carry_the_package_pin_of_every_lane(board) -> None:
    """A pulse names outputs -- cooling, probe -- and the Target page is the
    only place that says which physical pins those names reach.  Without it an
    operator has the pulse and the breakout and no way to relate them."""

    from zlc_workbench.pulse_editor import project_target

    pins = dict(board.target.package_pins)
    assert pins, "the board publishes its pin map"
    records = project_target(board.target, pins=pins)
    assert records

    for record in records:
        assert record.endpoints, f"{record.key} shows no endpoint at all"
        assert all(text.strip() for text in record.endpoints)
        if record.kind == "dac":
            assert len(record.endpoints) > 1, "a DAC shows every data pin"
            assert record.clock_endpoint, "and the pin its latch clock reaches"

    # Not lane names wearing a pin's clothes: at least one row reaches a pin
    # that is spelled differently from the lane behind it.
    lanes = {lane for port in board.target.ports for lane in port.lanes}
    reached = {text for record in records for text in record.endpoints}
    assert reached - lanes, "endpoints are package pins, not the lane names"


def test_on_pulse_runs_the_whole_pulse_until_stop() -> None:
    """On Pulse is a cycle an experiment holds running, not one shot.

    Asked of a real board, not of the source text.  Reading fire() for the
    explicit outer ``cycles`` passed for as long as that execution was requested
    spelled that way and said nothing about what the board was told -- so it
    broke the moment the value stopped being written out longhand, while the
    behaviour it names was intact.
    """

    from zlc_pulse import load_streamer_config, pulse_target_from_xdc
    from zlc_pulse.device import PulseStreamer
    from zlc_pulse.transport import MemoryRegisterTransport
    from zlc_workbench.device_use import DeviceUseCoordinator
    from zlc_workbench.pulse_editor import PulseEditorPresenter
    from zlc_workbench.pulse_state import PulseEditorState

    from test_pulse_editor import _EditorView, _ordinary_sequence

    config = load_streamer_config()
    streamer = PulseStreamer(
        MemoryRegisterTransport(geom=config["params"], auto_done=True),
        config["params"],
        config["clock_hz"],
        target=pulse_target_from_xdc(config_path=config["source"]),
    )
    streamer.open()
    presenter = PulseEditorPresenter(
        _EditorView(),
        PulseEditorState(sequence=_ordinary_sequence()),
        sequencer=streamer,
        device_use=DeviceUseCoordinator(),
    )
    try:
        assert presenter.fire() is True, presenter.view.warnings
        applied = streamer.applied()
        assert applied is not None, "the board was never told anything"
        assert applied.cycles is None, (
            "On Pulse must ask the board to repeat until Stop"
        )
    finally:
        presenter.stop()
        presenter.close()
        streamer.close()


def test_the_preview_page_offers_sizes_and_a_show_all_switch() -> None:
    """Both were reported missing.  They are controls, so this asks the widget."""

    pytest.importorskip("PyQt5")
    from zlc_ui.pulse.preview_view import PulsePreviewView
    from zlc_ui.qt import ensure_qt_app

    ensure_qt_app(["preview-controls"])
    view = PulsePreviewView()
    names = ("1x2", "2x2", "4x4", "8x8")
    view.set_size_names(names)

    combo = view.preview_size_combo
    assert [combo.itemText(index) for index in range(combo.count())] == list(names)
    assert combo.isEnabled()
    assert view.preview_include_off is not None
    assert view.preview_include_off.isEnabled(), "show off rows must be reachable"


def test_holding_a_scan_point_leaves_no_scan_on_the_board() -> None:
    """Hold plays an ORDINARY pulse, so the board never sees a one-point scan.

    Reported from a scope: with a scan table loaded, the DAC swept correctly --
    and the moment Stop-at-hold was pressed it sat at 0 V and stayed there,
    through every Step, while the TTL channels kept running.

    The two paths differed in exactly one register.  Holding wrote the point as
    a scan table of length one (SCAN_COUNT=1, SCAN_ENABLE=1) and looped it
    forever, a state nothing else ever asks the board for, and its bus segments
    were never re-applied.  Resolve the point into the document and run a plain
    pulse.  This asserts the board is back in ordinary territory, which is the
    part that can be checked without hardware.
    """

    from zlc_pulse import load_streamer_config, pulse_target_from_xdc
    from zlc_pulse.device import PulseStreamer
    from zlc_pulse.transport import MemoryRegisterTransport
    from zlc_workbench.device_use import DeviceUseCoordinator
    from zlc_workbench.pulse_editor import PulseEditorPresenter
    from zlc_workbench.pulse_state import PulseEditorState

    from test_pulse_editor import _EditorView, _ordinary_sequence

    sequence = _ordinary_sequence()
    config = load_streamer_config()
    transport = MemoryRegisterTransport(geom=config["params"], auto_done=True)
    board = PulseStreamer(
        transport,
        config["params"],
        config["clock_hz"],
        target=pulse_target_from_xdc(config_path=config["source"]),
    )
    board.open()
    presenter = PulseEditorPresenter(
        _EditorView(),
        PulseEditorState(sequence=sequence),
        sequencer=board,
        device_use=DeviceUseCoordinator(),
    )
    try:
        presenter.view.binding_cycle_requested.emit(
            "duration", sequence.periods[3].period_id, None
        )
        presenter.view.scan_source_edited.emit(
            "import numpy as np\n"
            "scan_table = (np.arange(5) + 1).reshape(-1, 1) * 0.001\n"
        )
        presenter.view.scan_run_requested.emit()
        presenter.set_scan_repeats(0)
        assert presenter.fire(cycles=5) is True
        scanned = board.applied()
        assert scanned is not None and len(scanned.rows) == 5

        assert presenter._hold(0) is True, presenter.view.warnings
        held = board.applied()
        assert held is not None
        assert held.rows == ()
        assert held.program.slot_count == 0
        assert held.source is not None and held.source.slots == ()
        assert held.cycles is None

        assert presenter.step_scan_point(100) is True, presenter.view.warnings
        stepped = board.applied()
        assert stepped is not None and stepped.rows == () and stepped.cycles is None
        assert not presenter.view.warnings, presenter.view.warnings
    finally:
        presenter.stop()
        presenter.close()
        board.close()
