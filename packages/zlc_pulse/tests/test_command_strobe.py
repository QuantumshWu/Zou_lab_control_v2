"""Every command the host issues must reach the board.

The RTL detects commands on a rising edge and never clears COMMAND itself:

    wire [3:0] cmd_now  = ctrl_reg[C_COMMAND][3:0];
    wire [3:0] cmd_edge = cmd_now & ~cmd_seen;          # top.v
    reg ldr_cmd_clear;   # declared, defaulted to 0, never asserted anywhere

So writing the same command code twice in a row produces no edge and the board
silently ignores the second one -- while the host still starts its observer and
reports a normal DoneReport.  The standalone host shipped without the
return-to-zero that the v1 host always wrote, which killed every fire after the
first one in a write_slots/fire scan loop.

These tests replay the recorded COMMAND writes through the RTL's own edge rule,
so they fail for exactly the reason the board would go quiet.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from zlc_pulse import compile_sequence
from zlc_pulse import device
from zlc_pulse.device import PulseStreamer
from zlc_pulse.transport import MemoryRegisterTransport
from zlc_pulse.wire import CMD_FIRE, CMD_LOAD, CMD_SAFE, CtrlWords, StreamerParams

from test_wire_device import _sequence


RTL = Path(__file__).resolve().parents[1] / "fpga" / "pulse_streamer" / "zlc_pulse_streamer_top.v"


def _dropped(writes: list[int]) -> list[int]:
    """Apply the RTL rule verbatim and return the commands the board ignores.

    A code acts only on a rising edge, and zero is the strobe's return path --
    so every *nonzero* write the host performs must produce an edge.  Anything
    left here is a command the host believes it issued and the board never saw.
    """

    seen = 0
    dropped: list[int] = []
    for value in writes:
        code = value & 0xF
        if code and not (code & ~seen):
            dropped.append(code)
        seen = code
    return dropped


def _acted(writes: list[int]) -> list[int]:
    seen = 0
    acted: list[int] = []
    for value in writes:
        code = value & 0xF
        if code & ~seen:
            acted.append(code)
        seen = code
    return acted


class _Recorder(MemoryRegisterTransport):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.commands: list[int] = []

    def write_words(self, words, **kwargs):  # type: ignore[override]
        for address, value in words:
            if address == CtrlWords.COMMAND:
                self.commands.append(int(value))
        return super().write_words(words, **kwargs)


def _streamer() -> tuple[PulseStreamer, _Recorder, object]:
    geom = StreamerParams()
    transport = _Recorder(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6)
    streamer.open()
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    return streamer, transport, program


def test_rtl_still_detects_commands_on_a_rising_edge_and_never_self_clears() -> None:
    """Anchor the host rule to the frozen RTL it exists for."""

    source = RTL.read_text(encoding="utf-8", errors="replace")
    assert re.search(r"cmd_edge\s*=\s*cmd_now\s*&\s*~cmd_seen", source)
    assignments = re.findall(r"ldr_cmd_clear\s*<=\s*1'b([01])", source)
    assert assignments and set(assignments) == {"0"}, (
        "the RTL now clears COMMAND itself; revisit the host strobe rule"
    )


def test_every_fire_in_a_scan_loop_reaches_the_board() -> None:
    """The scan workflow: one compile, then a slot row and a shot per point."""

    streamer, transport, program = _streamer()
    streamer.load(program)
    for value in (1, 2, 3):
        streamer.write_slots((value,))
        streamer.fire()
        streamer.wait_done(2.0)

    assert _acted(transport.commands).count(CMD_FIRE) == 3, (
        f"host issued 3 fires, board acts on "
        f"{_acted(transport.commands).count(CMD_FIRE)}; "
        f"COMMAND writes were {transport.commands}"
    )


def test_reloading_the_same_program_reaches_the_board() -> None:
    streamer, transport, program = _streamer()
    streamer.load(program)
    streamer.load(program)
    assert _acted(transport.commands).count(CMD_LOAD) == 2


def test_no_command_the_host_issues_is_ever_dropped() -> None:
    """The general invariant, over every command-issuing entry point."""

    streamer, transport, program = _streamer()
    streamer.load(program)
    streamer.fire()
    streamer.wait_done(2.0)
    streamer.write_slots((1,))
    streamer.fire()
    streamer.wait_done(2.0)
    streamer.safe()
    streamer.safe()
    streamer.load(program)
    streamer.fire()
    streamer.wait_done(2.0)

    assert _dropped(transport.commands) == [], (
        f"commands the board never sees: {_dropped(transport.commands)}; "
        f"COMMAND writes were {transport.commands}"
    )


def test_load_reports_a_loader_that_never_acknowledges() -> None:
    """The RTL gates FIRE on STATUS_LOADED.

    A load the board never completed must raise here; otherwise every later
    fire is a silent no-op and the host reports a normal DoneReport.
    """

    geom = StreamerParams()

    class _NeverLoads(MemoryRegisterTransport):
        def read_word(self, word_offset, **kwargs):  # type: ignore[override]
            if int(word_offset) == CtrlWords.STATUS:
                return 0
            return super().read_word(word_offset, **kwargs)

    transport = _NeverLoads(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6)
    streamer.open()
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    device.LOAD_TIMEOUT = 0.05
    try:
        with pytest.raises(RuntimeError, match="LOADED"):
            streamer.load(program)
    finally:
        device.LOAD_TIMEOUT = 5.0


def test_replaying_a_scan_table_re_arms_the_banks_at_point_zero() -> None:
    """A run restarts at scan point 0, so the banks must hold chunks 0/1.

    During a run the observer streams later chunks over the banks; ``_refill``
    below is that same observer path, driven deterministically.  Without
    re-arming, the replay's point 0 is not resident and the engine stalls with
    no output while every command still looks accepted.
    """

    streamer, transport, program = _streamer()
    streamer.load(program)
    rows = tuple((value,) for value in range(1, 3 * StreamerParams().bank_size))
    streamer.write_scan_table(rows)
    streamer.fire()
    streamer.wait_done(2.0)

    streamer._refill(2 * StreamerParams().bank_size)  # the observer's own path
    assert transport.read_word(CtrlWords.BANK0_CHUNK) != 0

    streamer.fire()
    streamer.wait_done(2.0)
    assert transport.read_word(CtrlWords.BANK0_CHUNK) == 0
    assert transport.read_word(CtrlWords.BANK1_CHUNK) == 1
    assert transport.read_word(CtrlWords.BANK_READY) == 0b11


def test_load_drives_v1_physical_safe_before_the_image_and_load_strobe() -> None:
    """The v1 prepare order is SAFE, clocks-off/readback, SAFE, image, LOAD.

    Each strobe is its OWN transaction now, after the data it acts on has been
    acknowledged.  A frame the link loses is sent again, and a command must
    never be: one that was executed and whose acknowledgement went missing
    would be executed twice.  Riding along with the data made that impossible
    to separate, and it also let the board be told to act on an image that was
    still arriving.
    """

    streamer, transport, program = _streamer()
    streamer.load(program)

    strobe = ((CtrlWords.COMMAND, 0), (CtrlWords.COMMAND, CMD_SAFE))
    assert transport.write_batches[:5] == [
        ((CtrlWords.STATUS, 1 << 3),),
        strobe,
        tuple((CtrlWords.CLK_ENABLE + index, 0) for index in range(streamer.geom.clk_enable_words)),
        ((CtrlWords.STATUS, 1 << 3),),
        strobe,
    ]
    image_batch = transport.write_batches[5]
    assert image_batch[-1] == (CtrlWords.BANK_READY, 0b11)
    assert transport.write_batches[6] == (
        (CtrlWords.COMMAND, 0),
        (CtrlWords.COMMAND, CMD_LOAD),
    )


def test_safe_retries_once_and_requires_adjacent_zero_status_reads(monkeypatch) -> None:
    class _DelayedSafe(_Recorder):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self._status_reads = iter((1 << 3, 0, 0, 0, 0))

        def read_word(self, word_offset, **kwargs):  # type: ignore[override]
            if int(word_offset) == CtrlWords.STATUS:
                return next(self._status_reads, 0)
            return super().read_word(word_offset, **kwargs)

    monkeypatch.setattr(device, "SAFE_RETRY_AFTER", 0.001)
    transport = _DelayedSafe(geom=StreamerParams(), auto_done=True)
    streamer = PulseStreamer(transport, StreamerParams(), 50e6)
    streamer.open()

    readback = streamer.safe()

    assert readback.status_reads == (0, 0)
    assert transport.commands.count(CMD_SAFE) == 3


def test_safe_times_out_instead_of_returning_an_unstable_readback(monkeypatch) -> None:
    class _NeverSafe(_Recorder):
        def read_word(self, word_offset, **kwargs):  # type: ignore[override]
            if int(word_offset) == CtrlWords.STATUS:
                return 1 << 3
            return super().read_word(word_offset, **kwargs)

    monkeypatch.setattr(device, "SAFE_TIMEOUT", 0.01)
    monkeypatch.setattr(device, "SAFE_RETRY_AFTER", 0.001)
    transport = _NeverSafe(geom=StreamerParams(), auto_done=True)
    streamer = PulseStreamer(transport, StreamerParams(), 50e6)
    streamer.open()

    with pytest.raises(TimeoutError, match="stable STATUS=0"):
        streamer.safe()

    assert transport.commands.count(CMD_SAFE) == 2


def test_a_board_describes_itself_rather_than_letting_a_client_assume() -> None:
    """The protocol could open, load and fire a board but not ask what it was.

    So a client wanting ports, pins or a clock had to read its own XDC and
    config and hope they were the ones the board was built from -- the exact
    mistake the layout handshake exists to catch, made one layer up.
    """

    from zlc_pulse import load_streamer_config
    from zlc_pulse.device import BoardDescription, PulseStreamer
    from zlc_pulse.transport import MemoryRegisterTransport

    config = load_streamer_config()
    geometry = config["params"]
    streamer = PulseStreamer(
        MemoryRegisterTransport(geom=geometry, auto_done=True),
        geometry,
        config["clock_hz"],
    )

    # Nothing is proven before the handshake, so nothing is claimed.
    with pytest.raises(RuntimeError):
        streamer.describe()

    streamer.open()
    try:
        described = streamer.describe()
        assert isinstance(described, BoardDescription)
        assert len(described.target.raw_lanes) == geometry.channel_count
        assert described.channel_count == geometry.channel_count
        assert described.bus_count == geometry.bus_count
        assert described.clock_hz == float(config["clock_hz"])
        assert described.time_step_ns == 1e9 / float(config["clock_hz"])
        # A pin for every lane: the names an operator wires against.
        # One pin map, on the target that owns it.  It used to be carried
        # twice, and the wire silently emptied the nested copy.
        assert set(described.target.package_pins) == set(described.target.raw_lanes)
        assert all(pin.strip() for pin in described.target.package_pins.values())
    finally:
        streamer.close()


def test_the_description_survives_the_wire() -> None:
    """Ports, pins and the clock have to arrive intact or the client still guesses."""

    from zlc_pulse.remote import REMOTE_METHODS, decode_tree, encode_tree
    from zlc_pulse import load_streamer_config
    from zlc_pulse.device import PulseStreamer
    from zlc_pulse.transport import MemoryRegisterTransport

    assert "describe" in REMOTE_METHODS

    config = load_streamer_config()
    geometry = config["params"]
    streamer = PulseStreamer(
        MemoryRegisterTransport(geom=geometry, auto_done=True),
        geometry,
        config["clock_hz"],
    )
    streamer.open()
    try:
        original = streamer.describe()
        restored = decode_tree(encode_tree(original))

        assert restored.target == original.target
        assert dict(restored.target.package_pins) == dict(
            original.target.package_pins
        ), "the wire dropped the pin map"

        assert restored.clock_hz == original.clock_hz
        assert restored.layout_fingerprint == original.layout_fingerprint
        # The port labels are what an operator reads in an editor.
        assert [port.label for port in restored.target.ports] == [
            port.label for port in original.target.ports
        ]
    finally:
        streamer.close()


def test_a_slot_value_the_multiplier_cannot_hold_is_refused() -> None:
    """The host used to predict a schedule the board would not play.

    The RTL takes the low 25 signed bits of a slot value; only the RTL mirror
    narrowed them, so the compiler computed one tick, the validator approved it,
    and the board played another.  They agree now -- which means they agree on
    something nobody asked for, so a value that does not fit is refused instead.
    """

    from zlc_pulse.compile import (
        evaluate_affine_tick,
        narrow_slot_operand,
        slot_operand_width,
    )
    from zlc_pulse.engine_model import effective_tick

    width = slot_operand_width()
    over = 1 << width

    assert narrow_slot_operand(over) == 0
    assert narrow_slot_operand(7) == 7
    # One arithmetic, so the host cannot predict what the board will not play.
    for value in (7, over, over + 5, -over):
        assert evaluate_affine_tick(0, (1,), (value,), 0) == effective_tick(
            0, (1,), (value,), 0
        )


def test_the_host_refills_one_chunk_ahead_of_the_cursor() -> None:
    """The contract the RTL is built to, and the cycle model implements.

    When the engine crosses from chunk c into c+1 it frees the bank holding c
    and the host refills it with c+2.  The host used to wait for the cursor to
    REACH the chunk it was about to write -- one whole bank late -- so any scan
    longer than the two pre-armed chunks stalled at every chunk boundary while
    the host caught up.
    """

    bank = StreamerParams().bank_size
    streamer, _transport, program = _streamer()
    streamer.load(program)
    streamer.write_scan_table(tuple((value,) for value in range(1, 4 * bank)))

    armed = streamer.snapshot()["scan_next_chunk"]
    # The cursor has only entered chunk 1; chunk 2 must already be on its way.
    streamer._refill(bank)

    assert streamer.snapshot()["scan_next_chunk"] > armed


def test_the_memory_twin_drops_a_repeated_command_the_way_the_board_does() -> None:
    """The board detects a command on a rising edge and never clears the
    register, so writing the same code twice runs it once.  This twin modelled
    the register as level-sensitive, so a host that forgot to zero between
    commands passed here and dropped its second command on hardware -- which is
    the one failure this twin exists to catch."""

    from zlc_pulse.transport import MemoryRegisterTransport
    from zlc_pulse.wire import CMD_FIRE, CtrlWords

    transport = MemoryRegisterTransport(auto_done=True)
    transport.start()
    transport.write_words(((CtrlWords.COMMAND, CMD_FIRE),))
    assert transport.dropped_commands == 0
    transport.write_words(((CtrlWords.COMMAND, CMD_FIRE),))
    assert transport.dropped_commands == 1, "a repeated strobe was accepted"
    transport.write_words(((CtrlWords.COMMAND, 0),))
    transport.write_words(((CtrlWords.COMMAND, CMD_FIRE),))
    assert transport.dropped_commands == 1, "a re-armed strobe was refused"
