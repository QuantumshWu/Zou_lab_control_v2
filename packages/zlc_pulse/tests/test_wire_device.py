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
    PulseSlot,
    PulseTarget,
    RepeatRegion,
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
    unpack_program,
)


_BOARD_TARGET = pulse_target_from_xdc()
_DIGITAL_PORT = next(port for port in _BOARD_TARGET.ports if port.kind == "digital")
_DAC_PORT = next(port for port in _BOARD_TARGET.ports if port.kind == "dac")


def _sequence(*, slotted: bool = False, period_ns: int = 40) -> PulseSequence:
    target = _BOARD_TARGET
    high = [0] * len(target.raw_lanes)
    high[target.raw_lanes.index(_DIGITAL_PORT.lanes[0])] = 1
    low = (0,) * len(target.raw_lanes)
    slots = (PulseSlot("duration", PulseFieldRef("duration", "p0"), "ns", "p0_time"),) if slotted else ()
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
        slots=slots,
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
    assert build_fingerprint(StreamerParams()) == 0x5A55DF95


def test_host_rejects_affine_geometry_beyond_the_shipped_four_dsp_lanes() -> None:
    with pytest.raises(ValueError, match="at most 4"):
        check_rtl_assumptions(
            replace(StreamerParams(), num_slots=8, coeff_width=8)
        )
    with pytest.raises(ValueError, match="at most 18"):
        check_rtl_assumptions(
            replace(StreamerParams(), num_slots=2, coeff_width=32)
        )


def test_pack_sparse_image_matches_frozen_byte_baseline() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    words = pack_program(program, geom)
    payload = b"".join(
        int(address).to_bytes(4, "little") + int(value).to_bytes(4, "little")
        for address, value in sorted(words.items())
    )
    assert 0 not in words
    assert words[CtrlWords.PROG_COUNT] == 3
    assert words[CtrlWords.SCAN_COUNT] == 0
    assert words[CtrlWords.RESERVED_19] == 0
    invalid = dict(words)
    invalid[CtrlWords.RESERVED_19] = 1
    with pytest.raises(ValueError, match="reserved control word 19"):
        unpack_program(invalid, geom)
    assert hashlib.sha256(payload).hexdigest() == (
        "9fa95a92e7c93c8b16c329ef8f662e9c749dc122799230262113a6cac2080854"
    )


def test_pack_slot_scan_image_matches_frozen_byte_baseline() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    words = pack_program(program, geom)
    assert words[CtrlWords.SCAN_COUNT] == 0
    assert words[CtrlWords.SCAN_ENABLE] == 0
    assert words[CtrlWords.REPEAT_FOREVER] == 0
    bases = region_bases(geom)
    assert not any(bases["scan"] <= address < bases["bus"] for address in words)


def test_fire_applies_the_loaded_rows_without_rewriting_edge_regions() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    rows = ((1,), (2,), (1,))
    streamer.load(program, rows=rows)
    before = len(transport.write_batches)
    streamer.fire(cycles=3)
    delta = [address for batch in transport.write_batches[before:] for address, _ in batch]
    bases = region_bases(geom)
    assert not any(bases["tick"] <= address < bases["coeff"] for address in delta)
    assert not any(bases["coeff"] <= address < bases["mask"] for address in delta)
    assert not any(bases["mask"] <= address < bases["scan"] for address in delta)
    assert CtrlWords.SCAN_COUNT in delta
    assert any(address >= bases["scan"] for address in delta)
    assert streamer.applied().rows == rows
    assert streamer.applied().cycles == 3


def test_unslotted_program_uses_the_same_finite_cycle_entry() -> None:

    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    assert program.slot_count == 0
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)

    streamer.fire(cycles=5)
    applied = streamer.applied()
    assert applied is not None and applied.rows == () and applied.cycles == 5
    assert streamer.snapshot()["scan_count"] == 5
    assert transport.words[CtrlWords.SCAN_COUNT] == 5
    assert streamer.wait_done(1.0) is not None


def test_short_timeline_is_valid_once_but_rejected_before_a_seam() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    short = _sequence(period_ns=20)
    program = compile_sequence(short, geom, 50e6)
    assert program.ticks == (0, 1, 2)

    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    streamer.fire(cycles=1)
    assert streamer.wait_done(1.0) is not None

    before = list(transport.write_batches)
    with pytest.raises(ValueError, match="at least 3 hardware ticks"):
        streamer.fire(cycles=2)
    with pytest.raises(ValueError, match="at least 3 hardware ticks"):
        streamer.fire(cycles=None)
    assert transport.write_batches == before

    repeated = compile_sequence(
        replace(short, repeat=RepeatRegion("p0", "p1", 2)),
        geom,
        50e6,
    )
    other_transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    other = PulseStreamer(other_transport, geom, 50e6, target=_BOARD_TARGET)
    other.open()
    with pytest.raises(ValueError, match="RepeatRegion boundary.*tick 3"):
        other.load(repeated)
    assert other_transport.write_batches == []

    scanned = compile_sequence(_sequence(slotted=True, period_ns=20), geom, 50e6)
    scan_transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    scan = PulseStreamer(scan_transport, geom, 50e6, target=_BOARD_TARGET)
    scan.open()
    # Point 0 spans 3 ticks and may hand off to a 2-tick final point.  That
    # final point needs no boundary cache in a two-cycle finite run.
    scan.load(scanned, rows=((1,), (0,)))
    scan.fire(cycles=2)
    assert scan.wait_done(1.0) is not None
    before = list(scan_transport.write_batches)
    with pytest.raises(ValueError, match="at least 3 hardware ticks"):
        scan.fire(cycles=3)
    assert scan_transport.write_batches == before

    low = (0,) * len(_BOARD_TARGET.raw_lanes)
    high = short.periods[0].states
    late_short_loop = PulseSequence(
        target=_BOARD_TARGET,
        time_step_ns=20,
        periods=(
            PulsePeriod("pre", 200, "ns", low),
            PulsePeriod("body0", 20, "ns", high),
            PulsePeriod("body1", 20, "ns", low),
        ),
        repeat=RepeatRegion("body0", "body1", 2),
    )
    late_program = compile_sequence(late_short_loop, geom, 50e6)
    late_transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    late = PulseStreamer(late_transport, geom, 50e6, target=_BOARD_TARGET)
    late.open()
    # The only inner rewind can prefetch at absolute tick 10, so one execution
    # is valid even though the loop span is two ticks.  A second outer cycle is
    # not: after the rewind only two ticks remain before the frame boundary.
    late.load(late_program)
    late.fire(cycles=1)
    assert late.wait_done(1.0) is not None
    before = list(late_transport.write_batches)
    with pytest.raises(ValueError, match="after its final restart"):
        late.fire(cycles=2)
    assert late_transport.write_batches == before

    too_many_short_loops = compile_sequence(
        replace(late_short_loop, repeat=RepeatRegion("body0", "body1", 3)),
        geom,
        50e6,
    )
    rejected_transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    rejected = PulseStreamer(rejected_transport, geom, 50e6, target=_BOARD_TARGET)
    rejected.open()
    with pytest.raises(ValueError, match="RepeatRegion span.*3 hardware ticks"):
        rejected.load(too_many_short_loops)
    assert rejected_transport.write_batches == []

    three_tick_loop = compile_sequence(
        replace(
            late_short_loop,
            periods=(
                late_short_loop.periods[0],
                late_short_loop.periods[1],
                replace(late_short_loop.periods[2], duration=40),
            ),
            repeat=RepeatRegion("body0", "body1", 3),
        ),
        geom,
        50e6,
    )
    accepted_transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    accepted = PulseStreamer(accepted_transport, geom, 50e6, target=_BOARD_TARGET)
    accepted.open()
    accepted.load(three_tick_loop)


def test_load_requires_one_complete_application_shape() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
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
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
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


def test_finite_cycle_count_is_strict_and_never_wraps() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    before = list(transport.write_batches)
    for invalid in (True, 1.5, 0, -1, 2**32):
        with pytest.raises((TypeError, ValueError)):
            streamer.fire(cycles=invalid)
        assert transport.write_batches == before
    with pytest.raises(ValueError, match="loop metadata"):
        replace(program, loop_count=2**32)


def test_load_rejects_ttl_and_dac_delay_fifo_overflow() -> None:
    geom = replace(
        StreamerParams(),
        max_edges=256,
        bank_size=2,
        bus_seg_addr_width=7,
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
        with pytest.raises(ValueError, match=label):
            streamer.load(program)
        assert transport.write_batches == []


def test_applied_state_round_trip_and_gui_sync() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    rows = ((-1,), (0,), (1,))
    streamer.load(program, source=source, rows=rows)

    loaded = streamer.applied()
    assert loaded is not None
    assert loaded.program == program
    assert loaded.source == source
    assert loaded.rows == rows
    assert loaded.cycles == 1
    assert loaded.loaded_at > 0

    streamer.fire(cycles=None)
    state = streamer.applied()
    assert state is not None
    assert state.rows == rows
    assert state.cycles is None
    with pytest.raises(FrozenInstanceError):
        state.cycles = 2

    # A GUI can discard its local objects and rebuild the same static image from
    # the echoed source; the active row is a separate scan-bank write.
    echoed_source = state.source
    echoed_rows = state.rows
    del source, program
    assert echoed_source is not None
    rebuilt = compile_sequence(echoed_source, geom, 50e6)
    assert pack_program(rebuilt, geom) == pack_program(state.program, geom)
    packed = pack_scan_rows(echoed_rows, geom, 0, 0, len(rows))
    assert packed == pack_scan_rows(rows, geom, 0, 0, len(rows))
    remainder = pack_scan_rows(echoed_rows, geom, 1, 1, len(rows))
    slot_words = (
        [packed[key] for key in sorted(packed)][:: geom.num_slots]
        + [remainder[key] for key in sorted(remainder)][:: geom.num_slots]
    )
    assert slot_words == [0xFFFFFFFF, 0, 1]


def test_applied_state_tracks_scan_table_and_survives_done_and_safe() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    rows = ((1,), (2,), (1,))
    streamer.load(program, source=source, rows=rows)
    streamer.fire(cycles=3)
    assert streamer.wait_done(1.0) is not None
    after_done = streamer.applied()
    assert after_done is not None
    assert after_done.rows == rows
    assert after_done.cycles == 3
    streamer.fire(cycles=None)
    safe = streamer.safe()
    assert safe.stable
    after_safe = streamer.applied()
    assert after_safe is not None
    assert after_safe.rows == rows
    assert after_safe.cycles is None
    streamer.close()
    assert streamer.applied() is None


class _RtlFireGateTransport(MemoryRegisterTransport):
    """Model the frozen RTL rule that FIRE acts only while STATUS_LOADED is set."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.accepted_loads = 0
        self.accepted_fires = 0

    def write_words(self, rows, **kwargs):
        rows = tuple(rows)
        with self._lock:
            for address, value in rows:
                if address != CtrlWords.COMMAND:
                    continue
                if value & CMD_LOAD:
                    self.accepted_loads += 1
                if value & CMD_FIRE and self.status & STATUS_LOADED:
                    self.accepted_fires += 1
        return super().write_words(rows, **kwargs)


def test_repeated_fire_reloads_the_resident_image_before_the_rtl_gate() -> None:
    """A prior DONE clears LOADED; replay must LOAD without retransmitting edges."""

    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = _RtlFireGateTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    bases = region_bases(geom)
    edge_uploads_before = sum(
        bases["tick"] <= address < bases["scan"]
        for batch in transport.write_batches
        for address, _value in batch
    )

    streamer.fire()
    assert streamer.snapshot()["reloaded_before_fire"] is False
    assert streamer.wait_done(1.0) is not None
    streamer.fire()
    assert streamer.snapshot()["reloaded_before_fire"] is True
    assert streamer.wait_done(1.0) is not None

    edge_uploads_after = sum(
        bases["tick"] <= address < bases["scan"]
        for batch in transport.write_batches
        for address, _value in batch
    )
    assert transport.accepted_loads == 2
    assert transport.accepted_fires == 2
    assert edge_uploads_after == edge_uploads_before


def test_fire_after_safe_reloads_the_resident_image_before_firing() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = _RtlFireGateTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    streamer.safe()

    streamer.fire()
    assert streamer.wait_done(1.0) is not None
    assert transport.accepted_loads == 2
    assert transport.accepted_fires == 1


def test_runtime_slot_rows_reject_colliding_affine_edges() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    with pytest.raises(ValueError, match="edge ticks"):
        streamer.load(program, rows=((-2,),))


def test_safe_readback_uses_stable_status_and_zero_clock_mask() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    streamer.fire()
    assert streamer.wait_done(1.0) is not None
    safe = streamer.safe()
    assert safe.stable
    assert safe.status_reads == (0, 0)
    assert not any(safe.clock_enable_words)
    # Each strobe is its own transaction, sent after the write it acts on has
    # been acknowledged: a lost frame is resent and a command must never be.
    strobe = ((CtrlWords.COMMAND, 0), (CtrlWords.COMMAND, CMD_SAFE))
    assert transport.write_batches[-5:] == [
        ((CtrlWords.STATUS, STATUS_ERROR),),
        strobe,
        tuple((CtrlWords.CLK_ENABLE + index, 0) for index in range(geom.clk_enable_words)),
        ((CtrlWords.STATUS, STATUS_ERROR),),
        strobe,
    ]


def test_open_rejects_mismatched_word63() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    transport = MemoryRegisterTransport(layout_id=build_fingerprint(geom) ^ 1, geom=geom)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    with pytest.raises(RuntimeError, match="geometry/layout mismatch"):
        streamer.open()
    assert transport.closed


def test_layout_check_and_transport_self_test_use_the_frozen_ctrl_contract() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    transport = MemoryRegisterTransport(geom=geom)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()

    streamer.check_register_layout()
    streamer.transport_self_test(count=3)

    assert all(transport.words.get(geom.ctrl_scratch_base + index, 0) == 0 for index in range(3))


def test_wait_done_uses_observer_owned_terminal_double_reads() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    transport.read_log.clear()
    streamer.fire()
    report = streamer.wait_done(1.0)
    assert report is not None
    assert report.status_reads == (STATUS_DONE, STATUS_DONE)
    assert report.cursor_reads == (0, 0)
    assert transport.read_log == [
        CtrlWords.STATUS,
        CtrlWords.CURSOR,
        CtrlWords.STATUS,
        CtrlWords.CURSOR,
    ]


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
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
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
    streamer.fire(cycles=None)
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
        streamer.fire(cycles=1)
        assert streamer.wait_done(1.0) is not None
        assert transport.late_operations == []
    finally:
        transport.release_observer.set()
        streamer.close()


def test_safe_does_not_claim_observer_exit_when_transport_ignores_stop() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
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
    streamer.fire(cycles=None)
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
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = _AdvancingMemoryTransport(geom=geom, auto_done=False)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    rows = tuple((value,) for value in (1, 2, 1, 3, 1, 2))
    streamer.load(program, rows=rows)
    streamer.fire(cycles=len(rows))
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
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = _FailingRefillTransport(geom=geom, auto_done=False)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    rows = tuple((value,) for value in (1, 2, 1, 3, 1, 2))
    streamer.load(program, rows=rows)
    transport.fail_refill = True
    streamer.fire(cycles=len(rows))
    report = streamer.wait_done(1.0)
    assert report is not None
    assert report.status == STATUS_ERROR
    assert report.status_reads == (STATUS_ERROR, STATUS_ERROR)
    assert report.observer_error == "RuntimeError: synthetic scan refill failure"
    assert report.fault == (
        "pulse observer failed: RuntimeError: synthetic scan refill failure"
    )
    assert streamer.snapshot()["firing"] is False


class _PollFailingTransport(MemoryRegisterTransport):
    """The observer's STATUS polls fail ``failures`` times in a row, the way
    a UART read that exhausted its retries fails: with a TimeoutError."""

    lossy_line = True

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

    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = _PollFailingTransport(failures=1, geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    streamer.fire()
    report = streamer.wait_done(1.0)
    assert report is not None
    assert report.fault == ""
    assert report.status == STATUS_DONE
    assert report.status_reads == (STATUS_DONE, STATUS_DONE)
    assert report.observer_error == ""
    assert report.poll_failures == 1
    assert report.resent_frames == 60


def test_two_consecutive_failed_polls_end_the_observation_in_error() -> None:
    """Two whole transaction deadlines without one good answer is a line
    that is down; the report says how the line behaved before it went."""

    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = _PollFailingTransport(failures=2, geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    streamer.fire()
    report = streamer.wait_done(1.0)
    assert report is not None
    assert report.status == STATUS_ERROR
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
        (STATUS_DONE | STATUS_LINK_ERROR,) * 2,
        (12, 12),
    )
    assert report.link_error is True
    assert report.fault == ""
    assert report.elapsed_seconds == pytest.approx(1.5)


def test_pack_scan_rows_only_targets_the_requested_bank_chunk() -> None:
    geom = replace(StreamerParams(), bank_size=2)
    words = pack_scan_rows(
        ((1, 2), (3, 4), (5, 6)), geom, bank=1, chunk=1, cycles=3
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

    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    sequence = replace(_sequence(), delays=(OutputDelay(_DAC_PORT.key, 200, "ns"),))
    program = compile_sequence(sequence, geom, 50e6)

    # 200 ns at 20 ns/tick, carried on the record the compiler actually builds.
    assert program.bus_delays == (TargetBusDelay(bus_index=0, delay_ticks=10),)

    words = pack_program(program, geom)
    bases = region_bases(geom)
    assert words[bases["delay"] + geom.channel_count + 0] == 10


def test_an_unpacked_bus_delay_can_be_packed_again_unchanged() -> None:
    """Unpack and pack must use ONE name for this number, or the trip loses it."""

    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    sequence = replace(_sequence(), delays=(OutputDelay(_DAC_PORT.key, 200, "ns"),))
    program = compile_sequence(sequence, geom, 50e6)
    unpacked = unpack_program(pack_program(program, geom), geom)

    assert unpacked["bus_delays"] == [{"bus_index": 0, "delay_ticks": 10}]

    class _AsUnpacked:
        bus_delays = unpacked["bus_delays"]

    round_tripped = replace(program, bus_delays=tuple(
        TargetBusDelay(**entry) for entry in unpacked["bus_delays"]
    ))
    assert pack_program(round_tripped, geom)[
        region_bases(geom)["delay"] + geom.channel_count
    ] == 10
