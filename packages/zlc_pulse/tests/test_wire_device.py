from __future__ import annotations

from dataclasses import fields, replace
import hashlib
from dataclasses import FrozenInstanceError

import pytest

from zlc_pulse import (
    AnalogStep,
    PulsePeriod,
    PulsePortSpec,
    PulseSequence,
    PulseSlot,
    PulseTarget,
    compile_sequence,
    pulse_target_from_xdc,
)
from zlc_pulse.compile import TargetBusDelay
from zlc_pulse.model import OutputDelay, PulseFieldRef
from zlc_pulse.device import PulseStreamer
from zlc_pulse.transport import MemoryRegisterTransport
from zlc_pulse.wire import (
    CMD_FIRE,
    CMD_LOAD,
    CMD_SAFE,
    CtrlWords,
    IMAGE_MAGIC,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_LOADED,
    STATUS_RUNNING,
    build_fingerprint,
    pack_program,
    pack_scan_rows,
    region_bases,
    StreamerParams,
    unpack_program,
)


_BOARD_TARGET = pulse_target_from_xdc()


def _sequence(*, slotted: bool = False) -> PulseSequence:
    target = PulseTarget(
        lanes=("d0", "a0", "a1"),
        ports=(
            PulsePortSpec("d0", "digital", ("d0",)),
            PulsePortSpec("dac", "dac", ("a0", "a1"), bus_index=0),
        ),
    )
    slots = (PulseSlot("duration", PulseFieldRef("duration", "p0"), "ns", "p0_time"),) if slotted else ()
    return PulseSequence(
        target=target,
        time_step_ns=20,
        periods=(
            PulsePeriod("p0", 20, "ns", (1, 0, 0), (AnalogStep("dac", "edge", 0),)),
            PulsePeriod("p1", 20, "ns", (0, 0, 0)),
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
    assert build_fingerprint(StreamerParams()) == 0x5AFC7CFB


def test_pack_sparse_image_matches_frozen_byte_baseline() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    words = pack_program(program, geom)
    payload = b"".join(
        int(address).to_bytes(4, "little") + int(value).to_bytes(4, "little")
        for address, value in sorted(words.items())
    )
    assert words[CtrlWords.MAGIC] == IMAGE_MAGIC
    assert words[CtrlWords.PROG_COUNT] == 3
    assert words[CtrlWords.SCAN_COUNT] == 0
    assert hashlib.sha256(payload).hexdigest() == (
        "305af1dc067bd245038314b98febea2e2671a7b0bf7d7bcf8fe88a7059bcc380"
    )


def test_pack_slot_scan_image_matches_frozen_byte_baseline() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    program = replace(
        program,
        scan_points=((1,), (2,), (1,)),
        scan_point_durations=(2e-8, 3e-8, 2e-8),
    )
    words = pack_program(program, geom)
    payload = b"".join(
        int(address).to_bytes(4, "little") + int(value).to_bytes(4, "little")
        for address, value in sorted(words.items())
    )
    assert hashlib.sha256(payload).hexdigest() == (
        "16478e3c31fe9a4d7c8eb2f9e2b3631eac1295596a03278819158816dabd9aad"
    )


def test_write_slots_writes_one_scan_row_without_edge_regions() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    before = len(transport.write_batches)
    streamer.write_slots((1,))
    delta = [address for batch in transport.write_batches[before:] for address, _ in batch]
    bases = region_bases(geom)
    assert not any(bases["tick"] <= address < bases["coeff"] for address in delta)
    assert not any(bases["coeff"] <= address < bases["mask"] for address in delta)
    assert not any(bases["mask"] <= address < bases["scan"] for address in delta)
    assert CtrlWords.SCAN_COUNT in delta


def test_write_scan_table_changes_only_scan_regions() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    before = len(transport.write_batches)
    streamer.write_scan_table(((1,), (2,), (1,)))
    delta = [address for batch in transport.write_batches[before:] for address, _ in batch]
    bases = region_bases(geom)
    assert not any(bases["tick"] <= address < bases["coeff"] for address in delta)
    assert not any(bases["coeff"] <= address < bases["mask"] for address in delta)
    assert not any(bases["mask"] <= address < bases["scan"] for address in delta)
    assert any(address >= bases["scan"] for address in delta)


def test_replacing_slots_rearms_new_scan_rows_after_an_existing_table() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    streamer.write_slots((1,))
    before = len(transport.write_batches)

    rows = ((2,), (3,), (4,))
    streamer.write_scan_table(rows)
    delta = [pair for batch in transport.write_batches[before:] for pair in batch]
    expected = pack_scan_rows(rows, geom, bank=0, chunk=0)
    assert (CtrlWords.SCAN_COUNT, len(rows)) in delta
    assert all(pair in delta for pair in expected.items())
    assert (CtrlWords.BANK_READY, 0b11) in delta


def test_applied_state_round_trip_and_gui_sync() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program, source=source)

    loaded = streamer.applied()
    assert loaded is not None
    assert loaded.program == program
    assert loaded.source == source
    assert loaded.slot_values == ()
    assert loaded.scan_rows == ()
    assert loaded.forever is False
    assert loaded.loaded_at > 0

    streamer.write_slots((1,))
    state = streamer.applied()
    assert state is not None
    assert state.slot_values == (1,)
    assert state.scan_rows == ()
    with pytest.raises(FrozenInstanceError):
        state.forever = True

    # A GUI can discard its local objects and rebuild the same static image from
    # the echoed source; the active row is a separate scan-bank write.
    echoed_source = state.source
    echoed_values = state.slot_values
    del source, program
    assert echoed_source is not None
    rebuilt = compile_sequence(echoed_source, geom, 50e6)
    assert pack_program(rebuilt, geom) == pack_program(state.program, geom)
    assert pack_scan_rows((echoed_values,), geom, 0, 0) == pack_scan_rows(((1,),), geom, 0, 0)


def test_applied_state_tracks_scan_table_and_survives_done_and_safe() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program, source=source)
    rows = ((1,), (2,), (1,))
    streamer.write_scan_table(rows)
    streamer.fire(forever=False)
    assert streamer.wait_done(1.0) is not None
    after_done = streamer.applied()
    assert after_done is not None
    assert after_done.scan_rows == rows
    assert after_done.slot_values == ()
    streamer.fire(forever=True)
    safe = streamer.safe()
    assert safe.stable
    after_safe = streamer.applied()
    assert after_safe is not None
    assert after_safe.scan_rows == rows
    assert after_safe.forever is True
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
    streamer.load(program)
    with pytest.raises(ValueError, match="edge ticks"):
        streamer.write_slots((0,))
    with pytest.raises(ValueError, match="edge ticks"):
        streamer.write_scan_table(((0,),))


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
        if self.fail_refill and any(address == CtrlWords.BANK_READY for address, _ in rows):
            raise RuntimeError("synthetic scan refill failure")
        return super().write_words(rows, **kwargs)


def test_observer_refills_a_freed_scan_bank() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    transport = _AdvancingMemoryTransport(geom=geom, auto_done=False)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    streamer.load(program)
    streamer.write_scan_table(tuple((value,) for value in (1, 2, 1, 3, 1, 2)))
    streamer.fire()
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
    streamer.load(program)
    streamer.write_scan_table(tuple((value,) for value in (1, 2, 1, 3, 1, 2)))
    transport.fail_refill = True
    streamer.fire()
    report = streamer.wait_done(1.0)
    assert report is not None
    assert report.status == STATUS_ERROR
    assert report.status_reads == (STATUS_ERROR, STATUS_ERROR)
    assert streamer.snapshot()["firing"] is False


def test_pack_scan_rows_only_targets_the_requested_bank_chunk() -> None:
    geom = replace(StreamerParams(), bank_size=2)
    words = pack_scan_rows(((1, 2), (3, 4), (5, 6)), geom, bank=1, chunk=1)
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
    sequence = replace(_sequence(), delays=(OutputDelay("dac", 200, "ns"),))
    program = compile_sequence(sequence, geom, 50e6)

    # 200 ns at 20 ns/tick, carried on the record the compiler actually builds.
    assert program.bus_delays == (TargetBusDelay(bus_index=0, delay_ticks=10),)

    words = pack_program(program, geom)
    bases = region_bases(geom)
    assert words[bases["delay"] + geom.channel_count + 0] == 10


def test_an_unpacked_bus_delay_can_be_packed_again_unchanged() -> None:
    """Unpack and pack must use ONE name for this number, or the trip loses it."""

    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    sequence = replace(_sequence(), delays=(OutputDelay("dac", 200, "ns"),))
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
