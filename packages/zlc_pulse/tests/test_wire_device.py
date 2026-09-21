from __future__ import annotations

from dataclasses import fields, replace
import hashlib
from dataclasses import FrozenInstanceError
import threading
import time

import pytest

from zlc_pulse import (
    AnalogStep,
    PulsePeriod,
    PulseSequence,
    PulseBinding,
    PulseTarget,
    PulseBracket,
    compile_sequence,
    pulse_target_from_xdc,
)
from zlc_pulse.compile import TargetBusDelay
from zlc_pulse.model import OutputDelay, PulseFieldRef
from zlc_pulse.device import DoneReport, PulseStreamer
from zlc_pulse.transport import MemoryRegisterTransport
from zlc_pulse.wire import (
    CMD_FIRE,
    CMD_LOAD,
    CMD_SAFE,
    CtrlWords,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_LINK_ERROR,
    STATUS_LOADED,
    STATUS_RUNNING,
    build_fingerprint,
    check_rtl_assumptions,
    pack_program,
    pack_scan_rows,
    region_bases,
    StreamerParams,
)


_BOARD_TARGET = pulse_target_from_xdc()
_DIGITAL_PORT = next(port for port in _BOARD_TARGET.ports if port.kind == "digital")
_DAC_PORT = next(port for port in _BOARD_TARGET.ports if port.kind == "dac")


def _sequence(*, slotted: bool = False, period_ns: int = 40) -> PulseSequence:
    target = _BOARD_TARGET
    high = [0] * len(target.raw_lanes)
    high[target.raw_lanes.index(_DIGITAL_PORT.lanes[0])] = 1
    low = (0,) * len(target.raw_lanes)
    slots = (PulseBinding(PulseFieldRef('duration', 'p0'), 'ns', scan=True),) if slotted else ()
    return PulseSequence(
        target=target,
        time_step_ns=20,
        periods=(
            PulsePeriod(
                "p0",
                period_ns,
                "ns",
                tuple(high),
                (AnalogStep(_DAC_PORT.key, "edge", 0),),
            ),
            PulsePeriod("p1", period_ns, "ns", low),
        ),
        bindings=slots,
    )


def test_build_fingerprint_covers_each_geometry_field_except_host_cap() -> None:
    params = StreamerParams()
    original = build_fingerprint(params)
    for field in fields(params):
        if field.name == "ttl_delay_max_ticks":
            continue
        value = getattr(params, field.name)
        changed = value + 1 if isinstance(value, int) else value + 1.0
        assert build_fingerprint(replace(params, **{field.name: changed})) != original, field.name


def test_default_geometry_is_pinned_to_deployed_word63() -> None:
    assert build_fingerprint(StreamerParams()) == 0x5AD5A6A0


def test_host_rejects_geometry_the_shipped_rtl_cannot_hold() -> None:
    with pytest.raises(ValueError, match="power of two"):
        check_rtl_assumptions(replace(StreamerParams(), num_slots=3))
    with pytest.raises(ValueError, match="max_rows must be a power of two"):
        check_rtl_assumptions(replace(StreamerParams(), max_rows=500))
    with pytest.raises(ValueError, match="16-bit loop-table row field"):
        check_rtl_assumptions(replace(StreamerParams(), max_rows=1 << 16))
    with pytest.raises(ValueError, match="loop_depth"):
        check_rtl_assumptions(replace(StreamerParams(), max_loops=2, loop_depth=3))


def test_pack_sparse_image_matches_frozen_byte_baseline() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    words = pack_program(program, geom, target=_BOARD_TARGET)
    payload = b"".join(
        int(address).to_bytes(4, "little") + int(value).to_bytes(4, "little")
        for address, value in sorted(words.items())
    )
    assert 0 not in words
    assert words[CtrlWords.PROG_COUNT] == 2
    assert words[CtrlWords.LOOP_TABLE_COUNT] == 0
    assert words[CtrlWords.SCAN_COUNT] == 0
    assert words[CtrlWords.RUN_REPEAT_COUNT] == 1
    assert words[CtrlWords.SCAN_REPEAT_COUNT] == 1
    assert program.clk_enable == sum(1 << bit for bit in (35, 46, 57, 68))
    assert words[CtrlWords.CLK_ENABLE] == 0b1111
    assert CtrlWords.CLK_ENABLE + 1 not in words
    assert geom.num_delay_ch == 25 and geom.row_words == 4 and geom.row_bits == 120
    bases = region_bases(geom)
    assert sorted(address - bases["delay"] for address in words if address >= bases["delay"]) == list(range(29))
    assert sorted(address - bases["rows"] for address in words if bases["rows"] <= address < bases["scan"]) == list(range(8))
    assert not any(bases["loop"] <= address < bases["delay"] for address in words)
    # Row 0: two ticks, literal duration, TTL bit 0 high, bus 0 edge to the
    # mid-scale code; row 1: two ticks, all low, no action.
    assert words[bases["rows"]] == 2
    assert words[bases["rows"] + 1] == (1 << geom.slot_sel_width) | (
        (512 | (1 << (geom.bus_width + geom.slot_sel_width)))
        << (geom.slot_sel_width + geom.num_delay_ch)
    ) & 0xFFFFFFFF
    assert words[bases["rows"] + geom.row_words] == 2
    assert words[bases["rows"] + geom.row_words + 1] == 0
    assert hashlib.sha256(payload).hexdigest() == (
        "38ecf02049ec6d257d377afe7a25c7db34a8f1b64d4e6bfbcb9bc536a7dc877d"
    )
    # Every TTL lane lives in the one mask field of its row; the final
    # physical clock above bit 63 maps to bus 3 rather than a third CTRL word.
    high = (1,) * geom.num_delay_ch + (0,) * (geom.channel_count - geom.num_delay_ch)
    sequence = _sequence()
    all_ttl = compile_sequence(replace(sequence, periods=(
        replace(sequence.periods[0], states=high), sequence.periods[1],
    )), geom, 50e6)
    packed = pack_program(all_ttl, geom, target=_BOARD_TARGET)
    mask_field = (packed[bases["rows"] + 1] >> geom.slot_sel_width) & ((1 << geom.num_delay_ch) - 1)
    assert mask_field == (1 << 25) - 1
    assert pack_program(replace(program, clk_enable=1 << 68), geom, target=_BOARD_TARGET)[CtrlWords.CLK_ENABLE] == 8
    with pytest.raises(ValueError, match="not a DAC latch clock"):
        pack_program(replace(program, clk_enable=1), geom, target=_BOARD_TARGET)


def test_pack_loops_and_slot_rows_into_their_own_regions() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    words = pack_program(program, geom, target=_BOARD_TARGET)
    assert words[CtrlWords.SCAN_COUNT] == 0
    assert words[CtrlWords.SCAN_ENABLE] == 0
    assert words[CtrlWords.RUN_REPEAT_COUNT] == 1
    assert words[CtrlWords.SCAN_REPEAT_COUNT] == 1
    assert words[CtrlWords.SLOT_COUNT] == 1
    bases = region_bases(geom)
    assert not any(bases["scan"] <= address < bases["loop"] for address in words)
    # Row 0 reads its duration from slot 1; its literal is the authored value.
    assert words[bases["rows"]] == 2
    assert words[bases["rows"] + 1] & ((1 << geom.slot_sel_width) - 1) == 1

    looped = compile_sequence(
        replace(_sequence(), brackets=(
            PulseBracket("outer", "p0", "p1", 3), PulseBracket("inner", "p1", "p1", 5),
        )),
        geom, 50e6,
    )
    words = pack_program(looped, geom, target=_BOARD_TARGET)
    assert words[CtrlWords.LOOP_TABLE_COUNT] == 2
    assert [words[bases["loop"] + index] for index in range(4)] == [
        0 | (1 << 16), 3, 1 | (1 << 16), 5,
    ]
    with pytest.raises(ValueError, match="max_loops"):
        pack_program(looped, replace(geom, max_loops=1), target=_BOARD_TARGET)
    with pytest.raises(ValueError, match="loop_depth"):
        pack_program(looped, replace(geom, max_loops=2, loop_depth=1), target=_BOARD_TARGET)
    with pytest.raises(ValueError, match="rows > max_rows"):
        pack_program(looped, replace(geom, max_rows=1), target=_BOARD_TARGET)


def test_fire_applies_the_loaded_rows_without_rewriting_program_regions() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    rows = ((1,), (2,), (1,))
    streamer.load(program, rows=rows)
    before = len(transport.write_batches)
    streamer.fire(run_repeats=3)
    delta = [address for batch in transport.write_batches[before:] for address, _ in batch]
    bases = region_bases(geom)
    assert not any(bases["rows"] <= address < bases["scan"] for address in delta)
    assert not any(bases["loop"] <= address < bases["delay"] for address in delta)
    assert CtrlWords.SCAN_COUNT in delta
    assert any(address >= bases["scan"] for address in delta)
    assert streamer.applied().rows == rows
    assert streamer.applied().run_repeats == 3
    assert streamer.applied().scan_repeats == 1


def test_unslotted_program_uses_run_repeats_without_a_scan_cursor() -> None:

    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    assert program.slot_count == 0
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)

    streamer.fire(run_repeats=5)
    applied = streamer.applied()
    assert applied is not None and applied.rows == () and applied.run_repeats == 5
    assert applied.scan_repeats == 1
    assert streamer.snapshot()["scan_count"] == 0
    assert transport.words[CtrlWords.SCAN_COUNT] == 0
    report = streamer.wait_done(1.0)
    assert report is not None
    assert report.cursor == 0


def test_one_tick_rows_and_brackets_need_no_seam_margin() -> None:
    """A row is one tick at the shortest, at every seam the board plays.

    The period table has no shadow registers to prepare ahead of a seam, so
    a Pulse of one-tick rows repeats, loops and hands off to the next scan
    point exactly as a long one does.
    """

    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    short = _sequence(period_ns=20)
    program = compile_sequence(short, geom, 50e6)
    assert program.durations == (1, 1)

    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    for run_repeats in (1, 2, 0):
        streamer.fire(run_repeats=run_repeats)
        if run_repeats:
            assert streamer.wait_done(1.0) is not None
        else:
            streamer.safe()

    repeated = compile_sequence(
        replace(short, brackets=(
            PulseBracket("whole", "p0", "p1", 2), PulseBracket("one", "p1", "p1", 3),
        )),
        geom,
        50e6,
    )
    assert repeated.frame_ticks() == 2 * (1 + 3)
    other_transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    other = PulseStreamer(other_transport, geom, 50e6, target=_BOARD_TARGET)
    other.open()
    other.load(repeated)
    other.fire(run_repeats=2)
    assert other.wait_done(1.0) is not None

    scanned = compile_sequence(_sequence(slotted=True, period_ns=20), geom, 50e6)
    scan_transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    scan = PulseStreamer(scan_transport, geom, 50e6, target=_BOARD_TARGET)
    scan.open()
    scan.load(scanned, rows=((1,), (2,)))
    scan.fire(run_repeats=1, scan_repeats=2)
    assert scan.wait_done(1.0) is not None


def test_load_requires_one_complete_application_shape() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    with pytest.raises(ValueError, match="requires a non-empty value table"):
        streamer.load(program)
    with pytest.raises(TypeError, match="must be integers"):
        streamer.load(program, rows=((True,),))
    streamer.load(program, rows=((2,), (3,), (4,)))


def test_load_rejects_compiler_identity_before_touching_hardware() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)

    for mismatch, message in (
        (replace(program, target_abi_fingerprint="0" * 32), "target ABI"),
        (replace(program, clock_hz=25e6), "clock"),
        (
            replace(program, geometry_fingerprint=program.geometry_fingerprint ^ 1),
            "geometry",
        ),
    ):
        transport = MemoryRegisterTransport(geom=geom, auto_done=True)
        streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
        streamer.open()
        with pytest.raises(ValueError, match=message):
            streamer.load(mismatch)
        assert transport.write_batches == []

    relabelled_ports = tuple(
        replace(port, label="renamed") if index == 0 else port
        for index, port in enumerate(_BOARD_TARGET.ports)
    )
    relabelled = PulseTarget(
        raw_lanes=_BOARD_TARGET.raw_lanes,
        ports=relabelled_ports,
        package_pins=_BOARD_TARGET.package_pins,
    )
    assert relabelled.abi_fingerprint == _BOARD_TARGET.abi_fingerprint
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=relabelled)
    streamer.open()
    streamer.load(program)


def test_repeat_counts_are_strict_and_zero_is_the_only_infinite_value() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    before = list(transport.write_batches)
    for invalid in (True, 1.5, None, -1, 2**32):
        with pytest.raises((TypeError, ValueError)):
            streamer.fire(run_repeats=invalid)
        assert transport.write_batches == before
    for invalid in (True, 1.5, None, -1, 2**32):
        with pytest.raises((TypeError, ValueError)):
            streamer.fire(run_repeats=1, scan_repeats=invalid)
        assert transport.write_batches == before
    with pytest.raises(ValueError, match="scan_repeats must be 1"):
        streamer.fire(run_repeats=1, scan_repeats=0)
    with pytest.raises(ValueError, match="scan_repeats must be 1"):
        streamer.fire(run_repeats=1, scan_repeats=2)
    with pytest.raises(ValueError, match="loop count"):
        replace(program, loops=((0, 1, 2**32),))

    scan_program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    scan_transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    scan = PulseStreamer(scan_transport, geom, 50e6, target=_BOARD_TARGET)
    scan.open()
    scan.load(scan_program, rows=((1,), (2,)))
    with pytest.raises(ValueError, match="32-bit CURSOR"):
        scan.fire(run_repeats=1, scan_repeats=(1 << 31) + 1)


def test_delay_capacity_covers_execution_repeat_seams_and_terminal_safe() -> None:
    geom = replace(
        StreamerParams(),
        max_rows=256,
        bank_size=2,
    )
    low = (0,) * len(_BOARD_TARGET.raw_lanes)
    bit = _BOARD_TARGET.raw_lanes.index(_DIGITAL_PORT.lanes[0])
    periods = []
    for index in range(130):
        state = list(low)
        state[bit] = index % 2
        periods.append(PulsePeriod(f"p{index}", 20, "ns", tuple(state)))
    ttl_sequence = PulseSequence(
        target=_BOARD_TARGET,
        time_step_ns=20,
        periods=tuple(periods),
        delays=(OutputDelay(_DIGITAL_PORT.key, 4_000, "ns"),),
    )
    ttl_program = compile_sequence(ttl_sequence, geom, 50e6)

    dac_periods = tuple(
        PulsePeriod(
            f"d{index}",
            20,
            "ns",
            low,
            (AnalogStep(_DAC_PORT.key, "edge", index % 16),),
        )
        for index in range(65)
    )
    dac_sequence = PulseSequence(
        target=_BOARD_TARGET,
        time_step_ns=20,
        periods=dac_periods,
        delays=(OutputDelay(_DAC_PORT.key, 4_000, "ns"),),
    )
    dac_program = compile_sequence(dac_sequence, geom, 50e6)

    for program, label in ((ttl_program, "channel"), (dac_program, "DAC bus")):
        transport = MemoryRegisterTransport(geom=geom, auto_done=True)
        streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
        streamer.open()
        streamer.load(program)
        before = list(transport.write_batches)
        with pytest.raises(ValueError, match=label):
            streamer.fire(run_repeats=1)
        assert transport.write_batches == before

    # One delayed DAC descriptor per four-tick Pulse fits a two-entry FIFO for
    # one sweep.  With two finite one-row sweeps, the terminal SAFE descriptor
    # lands at tick 8 beside the descriptors from ticks 0 and 4; omitting final
    # SAFE would incorrectly admit the run.
    terminal_geom = replace(
        StreamerParams(),
        max_rows=8,
        bank_size=2,
        bus_evt_fifo_depth=2,
    )
    terminal_source = replace(
        _sequence(slotted=True),
        delays=(OutputDelay(_DAC_PORT.key, 160, "ns"),),
    )
    terminal_program = compile_sequence(terminal_source, terminal_geom, 50e6)
    terminal_transport = MemoryRegisterTransport(
        geom=terminal_geom,
        auto_done=True,
    )
    terminal_streamer = PulseStreamer(
        terminal_transport,
        terminal_geom,
        50e6,
        target=_BOARD_TARGET,
    )
    terminal_streamer.open()
    terminal_streamer.load(terminal_program, rows=((2,),))
    with pytest.raises(ValueError, match="DAC bus.*needs 3 delayed events"):
        terminal_streamer.fire(run_repeats=1, scan_repeats=2)


def test_a_constant_bracket_body_does_not_crowd_the_runs_after_it() -> None:
    """The capacity walk keeps a Pulse's true length however long its Bracket.

    A body that changes no level adds no queue entry, so the Runs after it
    are spaced by the loop's real length: three tick rise, 3 + 4 x 4 fall,
    22 ticks per Pulse, never more than two edges in any closed 20-tick
    window.  Walking a SHORTENED loop moved the later Runs closer and refused
    the program for three entries in flight that never coexist.
    """

    geometry = replace(
        StreamerParams(),
        max_rows=8,
        bank_size=2,
        evt_fifo_depth=2,
        bus_evt_fifo_depth=2,
    )
    low = (0,) * len(_BOARD_TARGET.raw_lanes)
    high = list(low)
    high[_BOARD_TARGET.raw_lanes.index(_DIGITAL_PORT.lanes[0])] = 1

    def constant_body(count: int) -> PulseSequence:
        return PulseSequence(
            target=_BOARD_TARGET,
            time_step_ns=20,
            periods=(
                PulsePeriod("pre", 60, "ns", low),
                PulsePeriod("body", 80, "ns", tuple(high)),
                PulsePeriod("post", 60, "ns", low),
            ),
            brackets=(PulseBracket("loop", "body", "body", count),),
            delays=(OutputDelay(_DIGITAL_PORT.key, 400, "ns"),),
        )

    for count in (4, 100_000):
        source = constant_body(count)
        program = compile_sequence(source, geometry, 50e6)
        transport = MemoryRegisterTransport(geom=geometry, auto_done=True)
        streamer = PulseStreamer(transport, geometry, 50e6, target=_BOARD_TARGET)
        streamer.open()
        try:
            streamer.load(program, source=source)
            streamer.fire(run_repeats=3)
            assert streamer.wait_done(1.0) is not None
        finally:
            streamer.close()

    # A body whose edges really do crowd the queue is still refused, however
    # deep in the loop the crowding would happen.
    crowded = PulseSequence(
        target=_BOARD_TARGET,
        time_step_ns=20,
        periods=(
            PulsePeriod("pre", 60, "ns", low),
            PulsePeriod("up", 40, "ns", tuple(high)),
            PulsePeriod("down", 40, "ns", low),
            PulsePeriod("post", 60, "ns", low),
        ),
        brackets=(PulseBracket("loop", "up", "down", 100_000),),
        delays=(OutputDelay(_DIGITAL_PORT.key, 400, "ns"),),
    )
    program = compile_sequence(crowded, geometry, 50e6)
    transport = MemoryRegisterTransport(geom=geometry, auto_done=True)
    streamer = PulseStreamer(transport, geometry, 50e6, target=_BOARD_TARGET)
    streamer.open()
    try:
        # LOAD cannot answer this: the walk is over the run the board will
        # actually play, and how many times it plays is FIRE's argument.
        streamer.load(program, source=crowded)
        with pytest.raises(ValueError, match="channel .* needs"):
            streamer.fire(run_repeats=3)
    finally:
        streamer.close()


def test_applied_state_round_trip_and_gui_sync() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    rows = ((1,), (2,), (3,))
    streamer.load(program, source=source, rows=rows)

    loaded = streamer.applied()
    assert loaded is not None
    assert loaded.program == program
    assert loaded.source == source
    assert loaded.rows == rows
    assert loaded.run_repeats == 1
    assert loaded.scan_repeats == 1
    assert loaded.loaded_at > 0

    streamer.fire(run_repeats=0)
    state = streamer.applied()
    assert state is not None
    assert state.rows == rows
    assert state.run_repeats == 0
    assert state.scan_repeats == 1
    with pytest.raises(FrozenInstanceError):
        state.run_repeats = 2

    # A GUI can discard its local objects and rebuild the same static image from
    # the echoed source; the active row is a separate scan-bank write.
    echoed_source = state.source
    echoed_rows = state.rows
    del source, program
    assert echoed_source is not None
    rebuilt = compile_sequence(echoed_source, geom, 50e6)
    assert pack_program(rebuilt, geom, target=echoed_source.target) == pack_program(
        state.program, geom, target=echoed_source.target,
    )
    packed = pack_scan_rows(echoed_rows, geom, 0, 0)
    assert packed == pack_scan_rows(rows, geom, 0, 0)
    remainder = pack_scan_rows(echoed_rows, geom, 1, 1)
    slot_words = (
        [packed[key] for key in sorted(packed)][:: geom.num_slots]
        + [remainder[key] for key in sorted(remainder)][:: geom.num_slots]
    )
    assert slot_words == [1, 2, 3]


def test_applied_state_tracks_scan_table_and_survives_done_and_safe() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    rows = ((1,), (2,), (1,))
    streamer.load(program, source=source, rows=rows)
    streamer.fire(run_repeats=2, scan_repeats=3)
    report = streamer.wait_done(1.0)
    assert report is not None
    assert report.cursor == len(rows) * 3 - 1
    after_done = streamer.applied()
    assert after_done is not None
    assert after_done.rows == rows
    assert after_done.run_repeats == 2
    assert after_done.scan_repeats == 3
    streamer.fire(run_repeats=0, scan_repeats=1)
    safe = streamer.safe()
    assert safe.stable
    after_safe = streamer.applied()
    assert after_safe is not None
    assert after_safe.rows == rows
    assert after_safe.run_repeats == 0
    assert after_safe.scan_repeats == 1
    streamer.close()
    assert streamer.applied() is None


def test_repeated_fire_reuses_resident_program_after_done_and_safe() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    try:
        streamer.load(program, rows=((1,), (2,), (1,)))
        clocks = tuple(transport.read_word(CtrlWords.CLK_ENABLE + i)
                       for i in range(geom.clk_enable_words))
        first = streamer.fire(run_repeats=1)
        assert streamer.wait_done(1.0) is not None
        uploaded = len(transport.write_batches)
        repeated = streamer.fire(run_repeats=1)
        assert repeated is first
        assert streamer.wait_done(1.0) is not None
        infinite = streamer.fire(run_repeats=0)
        assert infinite.rows is first.rows
        assert infinite.program is first.program
        safe = streamer.safe()
        assert safe.stable and safe.status == 0 and safe.command_id > 0
        before = len(transport.write_batches)
        assert streamer.safe() == safe
        assert len(transport.write_batches) == before, "a completed SAFE is already proof"
        streamer.fire(run_repeats=1)
        assert streamer.wait_done(1.0) is not None
        commands = [value for batch in transport.write_batches[uploaded:]
                    for address, value in batch if address == CtrlWords.COMMAND]
        assert commands == [CMD_FIRE, CMD_FIRE, CMD_SAFE, CMD_FIRE]
        assert all(address in (CtrlWords.COMMAND_ID, CtrlWords.COMMAND)
                   for batch in transport.write_batches[uploaded:] for address, _ in batch)
        assert tuple(transport.read_word(CtrlWords.CLK_ENABLE + i)
                     for i in range(geom.clk_enable_words)) == clocks
    finally:
        streamer.close()


def test_runtime_slot_rows_reject_a_duration_the_row_cannot_hold() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    for invalid in (0, -2, 1 << 32):
        with pytest.raises(ValueError, match="tick range"):
            streamer.load(program, rows=((invalid,),))
    assert transport.write_batches == []


def test_open_rejects_mismatched_word63() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    transport = MemoryRegisterTransport(layout_id=build_fingerprint(geom) ^ 1, geom=geom)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    with pytest.raises(RuntimeError, match="geometry/layout mismatch"):
        streamer.open()
    assert transport.closed


def test_wait_done_uses_one_observer_owned_status_cursor_block(monkeypatch) -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    transport.read_log.clear()
    streamer.fire(run_repeats=1)
    assert streamer._done.wait(1.0)
    finished = streamer._fire_finished
    # Reading a completed report later is not additional execution time.
    with monkeypatch.context() as clock:
        clock.setattr("zlc_pulse.device.time.monotonic", lambda: finished + 0.4)
        report = streamer.wait_done(0.0)
    assert report is not None
    assert report.status == STATUS_DONE and report.cursor == 0
    assert report.command_id > 0
    assert report.elapsed_seconds == finished - streamer._fire_started
    assert report.report_delay_seconds == pytest.approx(0.4)
    assert 0 <= report.command_seconds <= report.elapsed_seconds
    assert transport.read_log == list(range(CtrlWords.STATUS, CtrlWords.CURSOR + 1))

    # Stop wakes a waiter tied to the old Event. Even if the next FIRE has
    # completed before that waiter resumes, its report remains unconsumed.
    transport.auto_done = False
    streamer.fire(run_repeats=1)
    old_id = streamer._fire_command_id
    done = streamer._done
    entered = threading.Event()
    awake = threading.Event()
    release = threading.Event()
    event_wait = done.wait
    def paused_wait(timeout=None):
        entered.set()
        answer = event_wait(timeout)
        awake.set()
        assert release.wait(1.0)
        return answer
    monkeypatch.setattr(done, "wait", paused_wait)
    reports = []
    waiter = threading.Thread(target=lambda: reports.append(streamer.wait_done(1.0)))
    waiter.start()
    assert entered.wait(1.0)
    streamer.safe()
    assert awake.wait(1.0)
    transport.auto_done = True
    streamer.fire(run_repeats=1)
    new_id = streamer._fire_command_id
    assert streamer._done.wait(1.0)
    release.set()
    waiter.join(timeout=1.0)
    assert not waiter.is_alive() and reports == [None]
    assert streamer.wait_done(0.0, command_id=old_id) is None
    assert streamer.wait_done(0.0, command_id=new_id).command_id == new_id
    streamer.close()


class _BlockingObserverTransport(MemoryRegisterTransport):
    def __init__(self, *, honor_stop: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.honor_stop = bool(honor_stop)
        self.block_observer = False
        self.observer_entered = threading.Event()
        self.release_observer = threading.Event()
        self.cancelled = threading.Event()
        self.late_operations: list[tuple[str, int]] = []
        self._blocked_once = False
        self._blocked_thread: threading.Thread | None = None

    def read_word(self, word_offset, *, stop=None, deadline=None):
        observer = threading.current_thread().name == "zlc-pulse-observer"
        if (
            observer
            and self.block_observer
            and not self._blocked_once
            and int(word_offset) == CtrlWords.STATUS
        ):
            self._blocked_once = True
            self._blocked_thread = threading.current_thread()
            self.observer_entered.set()
            while not self.release_observer.wait(0.005):
                if self.honor_stop and stop is not None and stop.is_set():
                    self.cancelled.set()
                    raise RuntimeError("blocked observer read cancelled")
        if threading.current_thread() is self._blocked_thread and self.cancelled.is_set():
            self.late_operations.append(("read", int(word_offset)))
        return super().read_word(word_offset, stop=stop, deadline=deadline)

    def write_words(self, rows, **kwargs):
        rows = tuple(rows)
        if (
            threading.current_thread() is self._blocked_thread
            and self.cancelled.is_set()
        ):
            self.late_operations.append(("write", len(rows)))
        return super().write_words(rows, **kwargs)


def test_safe_cancels_blocked_observer_and_leaves_no_late_operation() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = _BlockingObserverTransport(
        honor_stop=True,
        geom=geom,
        auto_done=False,
    )
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    transport.block_observer = True
    streamer.fire(run_repeats=0)
    assert transport.observer_entered.wait(1.0)
    observer = streamer._worker
    assert observer is not None
    try:
        started = time.monotonic()
        safe = streamer.safe()
        assert time.monotonic() - started < 0.5
        assert safe.stable
        assert transport.cancelled.is_set()
        assert not observer.is_alive()
        assert transport.late_operations == []
        assert streamer.snapshot()["firing"] is False

        transport.block_observer = False
        transport.auto_done = True
        streamer.fire(run_repeats=1)
        assert streamer.wait_done(1.0) is not None
        assert transport.late_operations == []
    finally:
        transport.release_observer.set()
        streamer.close()


def test_safe_does_not_claim_observer_exit_when_transport_ignores_stop() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = _BlockingObserverTransport(
        honor_stop=False,
        geom=geom,
        auto_done=False,
    )
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    transport.block_observer = True
    streamer.fire(run_repeats=0)
    assert transport.observer_entered.wait(1.0)
    observer = streamer._worker
    assert observer is not None
    try:
        with pytest.raises(RuntimeError, match="observer did not stop"):
            streamer.safe()
        assert observer.is_alive()
        assert streamer._worker is observer
        assert streamer.snapshot()["firing"] is True
    finally:
        transport.release_observer.set()
        observer.join(1.0)
        streamer.safe()
        streamer.close()


class _AdvancingMemoryTransport(MemoryRegisterTransport):
    def read_word(self, word_offset, **kwargs):
        if word_offset == CtrlWords.CURSOR and self.status == STATUS_RUNNING:
            self.cursor_value += 4 if self.cursor_value == 0 else 1
            if self.cursor_value >= 5:
                self.status = STATUS_DONE
        return super().read_word(word_offset, **kwargs)


class _FailingRefillTransport(_AdvancingMemoryTransport):
    fail_refill = False

    def write_words(self, rows, **kwargs):
        rows = tuple(rows)
        if self.fail_refill and any(
            address in (CtrlWords.BANK0_CHUNK, CtrlWords.BANK1_CHUNK) and value >= 2
            for address, value in rows
        ):
            raise RuntimeError("synthetic scan refill failure")
        return super().write_words(rows, **kwargs)


def test_observer_refills_a_freed_scan_bank() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = _AdvancingMemoryTransport(geom=geom, auto_done=False)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    rows = tuple((value,) for value in (1, 2, 1, 3, 1, 2))
    streamer.load(program, rows=rows)
    streamer.fire(run_repeats=1)
    report = streamer.wait_done(1.0)
    assert report is not None
    assert report.status == STATUS_DONE
    assert report.cursor == 5
    assert report.underflow is False
    assert any(
        (CtrlWords.BANK0_CHUNK, 2) in batch
        for batch in transport.write_batches
    )


def test_observer_refill_failure_becomes_terminal_error() -> None:
    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = _FailingRefillTransport(geom=geom, auto_done=False)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    rows = tuple((value,) for value in (1, 2, 1, 3, 1, 2))
    streamer.load(program, rows=rows)
    transport.fail_refill = True
    streamer.fire(run_repeats=1)
    report = streamer.wait_done(1.0)
    assert report is not None
    assert report.status == STATUS_RUNNING, "do not fabricate a hardware error for a host exception"
    assert report.observer_error == "RuntimeError: synthetic scan refill failure"
    assert report.fault == (
        "pulse observer failed: RuntimeError: synthetic scan refill failure"
    )
    assert streamer.snapshot()["firing"] is False


class _PollFailingTransport(MemoryRegisterTransport):
    """The observer's STATUS polls fail ``failures`` times in a row, the way
    a UART read that exhausted its retries fails: with a TimeoutError."""

    def __init__(self, *, failures: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.failures = failures
        self.failed = 0
        self.resends = 0

    def read_word(self, word_offset, **kwargs):
        if (
            threading.current_thread().name == "zlc-pulse-observer"
            and int(word_offset) == CtrlWords.STATUS
            and self.failed < self.failures
        ):
            self.failed += 1
            self.resends += 60
            raise TimeoutError("UART reply timed out after 60 attempt(s) (simulated)")
        return super().read_word(word_offset, **kwargs)


def test_one_failed_poll_is_a_warning_and_the_shot_still_reports_done() -> None:
    """STATUS and CURSOR are idempotent reads and DONE is a level: a poll
    that fails is asked again, not turned into a fabricated ERROR.

    The archived run: one CURSOR reply one byte short at shot 85 of 200, the
    board played on to 200, the camera collected 200 frames, SAFE
    acknowledged -- and the candidate was thrown away as "pulse observer
    failed".  The failure stays on the report, so a degrading line is
    visible shot by shot.
    """

    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = _PollFailingTransport(failures=1, geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    streamer.fire(run_repeats=1)
    report = streamer.wait_done(1.0)
    assert report is not None
    assert report.fault == ""
    assert report.status == STATUS_DONE
    assert report.observer_error == ""
    assert report.poll_failures == 1
    assert report.resent_frames == 60


def test_two_consecutive_failed_polls_end_the_observation_in_error() -> None:
    """Two whole transaction deadlines without one good answer is a line
    that is down; the report says how the line behaved before it went."""

    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = _PollFailingTransport(failures=2, geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    streamer.fire(run_repeats=1)
    report = streamer.wait_done(1.0)
    assert report is not None
    assert report.status == STATUS_RUNNING, "the FIRE acknowledgement is the last hardware status received"
    assert report.observer_error.startswith("TimeoutError: UART reply timed out")
    assert report.poll_failures == 2
    assert report.resent_frames == 120
    assert report.fault == (
        f"pulse observer failed: {report.observer_error}; "
        "the line failed 2 poll(s) and resent 120 frame(s) during this shot"
    )
    assert streamer.snapshot()["firing"] is False


def test_recovered_link_error_is_visible_but_not_an_engine_fault() -> None:
    report = DoneReport(
        STATUS_DONE | STATUS_LINK_ERROR,
        12,
        False,
        1.5,
        command_id=17,
    )
    assert report.link_error is True
    assert report.fault == ""
    assert report.elapsed_seconds == pytest.approx(1.5)
    failed = replace(report, status=STATUS_ERROR, observer_error="link failed")
    assert failed.fault == "pulse observer failed: link failed; the board reported an error"


def test_pack_scan_rows_only_targets_the_requested_bank_chunk() -> None:
    geom = replace(StreamerParams(), bank_size=2)
    words = pack_scan_rows(
        ((1, 2), (3, 4), (5, 6)), geom, bank=1, chunk=1
    )
    base = region_bases(geom)["scan"] + geom.bank_size * geom.scan_words
    assert set(words) == {base, base + 1, base + 2, base + 3}


def test_a_dac_bus_delay_reaches_the_board_word_it_was_asked_for() -> None:
    """The delay an operator set must be the number the board is given.

    pack read the record's delay through ``getattr(bd, "delay", 0)`` while the
    compiled record calls it ``delay_ticks``, so the default answered every
    time and EVERY DAC bus delay went to hardware as zero -- silently, in the
    one direction nothing checks, on a rig where the only symptom is a waveform
    that is wrong.  Nothing was red, because nothing read this word.
    """

    geom = replace(StreamerParams(), max_rows=8, bank_size=2)
    sequence = replace(_sequence(), delays=(OutputDelay(_DAC_PORT.key, 200, "ns"),))
    program = compile_sequence(sequence, geom, 50e6)

    # 200 ns at 20 ns/tick, carried on the record the compiler actually builds.
    assert program.bus_delays == (TargetBusDelay(bus_index=0, delay_ticks=10),)

    words = pack_program(program, geom, target=_BOARD_TARGET)
    bases = region_bases(geom)
    assert words[bases["delay"] + geom.num_delay_ch] == 10
    assert [words[bases["delay"] + geom.num_delay_ch + bus] for bus in range(4)] == [10, 0, 0, 0]
    invalid_delays = list(program.channel_delays)
    invalid_delays[geom.num_delay_ch] = 1
    with pytest.raises(ValueError, match="NOT delay-eligible"):
        pack_program(replace(program, channel_delays=tuple(invalid_delays)), geom, target=_BOARD_TARGET)
    with pytest.raises(ValueError, match="TTL row mask"):
        pack_program(replace(program, masks=(1 << geom.num_delay_ch, *program.masks[1:])), geom, target=_BOARD_TARGET)
