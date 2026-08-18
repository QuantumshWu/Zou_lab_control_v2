from __future__ import annotations

from dataclasses import replace

import numpy as np

from zlc_pulse import (
    AnalogStep,
    OutputDelay,
    PulsePeriod,
    PulsePortSpec,
    PulseSequence,
    PulseSlot,
    PulseTarget,
    compile_sequence,
)
from zlc_pulse.model import PulseFieldRef
from zlc_pulse.schedule import trigger_times, trigger_windows
from zlc_pulse.wire import StreamerParams


class _ScanUnderflow(RuntimeError):
    pass


def _effective_tick(
    base: int,
    coefficients: tuple[int, ...],
    point: tuple[int, ...],
    frac_bits: int,
) -> int:
    """Independent affine oracle for the small, in-range rows in this file."""

    assert len(coefficients) == len(point)
    assert all(-(1 << 24) <= value < (1 << 24) for value in point)
    return int(base) + (
        sum(int(coefficient) * int(value) for coefficient, value in zip(coefficients, point))
        >> int(frac_bits)
    )


def _reference_play(program, n_ticks: int) -> list[int]:
    """Minimal combinatorial edge-engine oracle owned by these tests."""

    assert not any(program.channel_delays), "this oracle does not model output delays"
    ticks = tuple(program.ticks)
    masks = tuple(program.masks)
    coefficients = tuple(program.tick_slot_coeffs)
    points = tuple(program.scan_points)
    zero = (0,) * len(program.slot_kinds)

    def effective(index: int, point: tuple[int, ...]) -> int:
        return _effective_tick(
            ticks[index], coefficients[index], point, program.scan_coeff_frac_bits
        )

    def loop_end(point: tuple[int, ...]) -> int:
        return _effective_tick(
            program.loop_end_tick,
            tuple(program.loop_end_slot_coeffs),
            point,
            program.scan_coeff_frac_bits,
        )

    point_index = 0
    point = points[0] if points else zero
    final_tick = effective(len(ticks) - 1, point)
    active_loop_end = loop_end(point)
    loops_left = program.loop_count
    running = bool(ticks)
    if running and effective(0, point) == 0:
        mask, tick, edge = masks[0], 1, 1
    else:
        mask, tick, edge = 0, 0, 0

    output: list[int] = []
    for _ in range(int(n_ticks)):
        output.append(mask)
        if not running:
            continue
        if program.loop_count > 1 and loops_left > 1 and tick >= active_loop_end:
            mask = masks[program.loop_start_index]
            tick = effective(program.loop_start_index, point) + 1
            edge = program.loop_start_index + 1
            loops_left -= 1
        elif tick >= final_tick:
            if point_index + 1 < len(points):
                point_index += 1
                point = points[point_index]
            elif program.repeat_forever:
                point_index = 0
                point = points[0] if points else zero
            else:
                running = False
                mask = 0
                continue
            final_tick = effective(len(ticks) - 1, point)
            active_loop_end = loop_end(point)
            loops_left = program.loop_count
            if effective(0, point) == 0:
                mask, tick, edge = masks[0], 1, 1
            else:
                mask, tick, edge = 0, 0, 0
        else:
            if edge < len(ticks) and tick == effective(edge, point):
                mask = masks[edge]
                edge += 1
            tick += 1
    return output


def _streaming_scan_play(
    program,
    n_ticks: int,
    *,
    bank_size: int,
    refill_delay: int = 0,
    raise_on_underflow: bool = False,
) -> tuple[list[int], bool, int]:
    """Test-owned two-bank refill oracle for the scan cases below."""

    assert not any(program.channel_delays), "this oracle does not model output delays"
    points = tuple(program.scan_points)
    if not points or bank_size <= 0:
        return _reference_play(program, n_ticks), False, 0

    ticks = tuple(program.ticks)
    masks = tuple(program.masks)
    coefficients = tuple(program.tick_slot_coeffs)

    def effective(index: int, point: tuple[int, ...]) -> int:
        return _effective_tick(
            ticks[index], coefficients[index], point, program.scan_coeff_frac_bits
        )

    def loop_end(point: tuple[int, ...]) -> int:
        return _effective_tick(
            program.loop_end_tick,
            tuple(program.loop_end_slot_coeffs),
            point,
            program.scan_coeff_frac_bits,
        )

    chunk_count = (len(points) + bank_size - 1) // bank_size
    bank_chunk = [-1, -1]
    bank_ready = [False, False]
    pending: list[tuple[int, int, int]] = []
    streaming = chunk_count > 2
    wrap_toggle = (chunk_count & 1) if streaming else 0

    def load(bank: int, chunk: int) -> None:
        bank_chunk[bank] = chunk
        bank_ready[bank] = True

    load(0, 0)
    if chunk_count > 1:
        load(1, 1)

    point_index = 0
    point = points[0]
    final_tick = effective(len(ticks) - 1, point)
    active_loop_end = loop_end(point)
    loops_left = program.loop_count
    bank_base = 0
    running = bool(ticks)
    stalled = False
    cycle = 0
    if running and effective(0, point) == 0:
        mask, tick, edge = masks[0], 1, 1
    else:
        mask, tick, edge = 0, 0, 0

    def bank_for(chunk: int, base: int) -> int:
        return (chunk % 2) ^ base

    def request_refill() -> None:
        if not streaming:
            return
        current_chunk = point_index // bank_size
        if current_chunk + 1 < chunk_count:
            next_chunk, next_base = current_chunk + 1, bank_base
        else:
            next_chunk, next_base = 0, bank_base ^ wrap_toggle
        bank = bank_for(next_chunk, next_base)
        if (bank_ready[bank] and bank_chunk[bank] == next_chunk) or any(
            item[0] == bank and item[1] == next_chunk for item in pending
        ):
            return
        bank_ready[bank] = False
        bank_chunk[bank] = -1
        pending.append((bank, next_chunk, cycle + max(0, int(refill_delay))))

    output: list[int] = []
    for _ in range(int(n_ticks)):
        cycle += 1
        for item in tuple(value for value in pending if value[2] <= cycle):
            pending.remove(item)
            load(item[0], item[1])
        request_refill()
        output.append(mask)
        if not running:
            continue
        if program.loop_count > 1 and loops_left > 1 and tick >= active_loop_end:
            mask = masks[program.loop_start_index]
            tick = effective(program.loop_start_index, point) + 1
            edge = program.loop_start_index + 1
            loops_left -= 1
        elif tick >= final_tick:
            last = point_index + 1 >= len(points)
            if last and not program.repeat_forever:
                running = False
                mask = 0
                continue
            next_index = 0 if last else point_index + 1
            current_chunk = point_index // bank_size
            next_chunk = next_index // bank_size
            next_base = bank_base ^ wrap_toggle if last else bank_base
            crossing = last or next_chunk != current_chunk
            bank = bank_for(next_chunk, next_base)
            if crossing and not (
                bank_ready[bank] and bank_chunk[bank] == next_chunk
            ):
                if raise_on_underflow:
                    raise _ScanUnderflow(
                        f"scan chunk {next_chunk} not ready at tick {tick}"
                    )
                stalled = True
            else:
                if crossing:
                    bank_base = next_base
                point_index = next_index
                point = points[point_index]
                final_tick = effective(len(ticks) - 1, point)
                active_loop_end = loop_end(point)
                loops_left = program.loop_count
                if effective(0, point) == 0:
                    mask, tick, edge = masks[0], 1, 1
                else:
                    mask, tick, edge = 0, 0, 0
        else:
            if edge < len(ticks) and tick == effective(edge, point):
                mask = masks[edge]
                edge += 1
            tick += 1
    return output, stalled, point_index + 1


def _target() -> PulseTarget:
    return PulseTarget(
        lanes=("d0", "d1", "a0", "a1"),
        ports=(
            PulsePortSpec("d0", "digital", ("d0",)),
            PulsePortSpec("d1", "digital", ("d1",)),
            PulsePortSpec("dac", "dac", ("a0", "a1"), bus_index=0),
        ),
    )


def _sequence(*, slots=(), delays=(), first_duration=20) -> PulseSequence:
    return PulseSequence(
        name="test",
        target=_target(),
        time_step_ns=20,
        periods=(
            PulsePeriod("p0", first_duration, "ns", (1, 0, 0, 0), (AnalogStep("dac", "edge", 0),)),
            PulsePeriod("p1", 20, "ns", (0, 1, 0, 0)),
            PulsePeriod("p2", 20, "ns", (0, 0, 0, 0)),
        ),
        slots=slots,
        delays=delays,
    )


def test_static_compile_has_safe_terminal_row_and_pure_trigger_projection() -> None:
    program = compile_sequence(_sequence(), StreamerParams(max_edges=8, bank_size=2), 50e6)
    assert program.masks[-1] == 0
    assert program.ticks[0] == 0
    assert np.array_equal(trigger_times(program, "d0"), np.asarray([0], dtype=np.uint64))
    assert np.array_equal(trigger_times(program, "d1"), np.asarray([1], dtype=np.uint64))


def test_slot_compile_changes_only_affine_data_and_dac_selectors() -> None:
    slot = PulseSlot("duration", PulseFieldRef("duration", "p0"), "ns", "p0_time")
    program = compile_sequence(
        _sequence(slots=(slot,)),
        StreamerParams(max_edges=8, bank_size=2),
        50e6,
    )
    assert program.slot_kinds == ("duration",)
    assert program.slot_count == 1
    assert program.tick_slot_coeffs[1][0] != 0
    assert program.scan_points == ()
    changed = replace(program, scan_points=((2,),), scan_point_durations=(4e-8,))
    assert _reference_play(changed, 12)[:5] == [1, 1, 2, 0, 0]


def test_write_slots_single_row_matches_static_waveform() -> None:
    static = compile_sequence(_sequence(first_duration=20), StreamerParams(max_edges=8, bank_size=2), 50e6)
    slot = PulseSlot("duration", PulseFieldRef("duration", "p0"), "ns", "p0_time")
    slotted = compile_sequence(
        _sequence(slots=(slot,), first_duration=20),
        StreamerParams(max_edges=8, bank_size=2),
        50e6,
    )
    runtime_row = replace(slotted, scan_points=((1,),), scan_point_durations=(4e-8,))
    assert _reference_play(runtime_row, 12) == _reference_play(static, 12)


def test_negative_bus_delay_shifts_every_driven_ttl_lane() -> None:
    program = compile_sequence(
        _sequence(delays=(OutputDelay("dac", -40, "ns"),)),
        StreamerParams(max_edges=8, bank_size=2),
        50e6,
    )
    assert program.channel_delays[:2] == (2, 2)
    assert program.channel_delays[2:] == (0, 0)
    assert program.bus_delays == ()
    declared_zero = compile_sequence(
        _sequence(delays=(OutputDelay("dac", -40, "ns"), OutputDelay("d1", 0, "ns"))),
        StreamerParams(max_edges=8, bank_size=2),
        50e6,
    )
    assert declared_zero.channel_delays[:2] == (2, 2)


def test_scan_table_is_data_and_engine_wrap_is_gapless() -> None:
    slot = PulseSlot("duration", PulseFieldRef("duration", "p0"), "ns", "p0_time")
    program = compile_sequence(
        _sequence(slots=(slot,)),
        StreamerParams(max_edges=8, bank_size=2),
        50e6,
    )
    program = replace(
        program,
        scan_points=((1,), (2,), (1,), (3,), (1,)),
        scan_point_durations=(2e-8, 3e-8, 2e-8, 4e-8, 2e-8),
    )
    expected = _reference_play(program, 80)
    actual, stalled, points_played = _streaming_scan_play(
        program, 80, bank_size=2, refill_delay=0
    )
    assert actual == expected
    assert stalled is False
    assert points_played == len(program.scan_points)


def test_scan_table_wrap_is_gapless_for_forever_program() -> None:
    slot = PulseSlot("duration", PulseFieldRef("duration", "p0"), "ns", "p0_time")
    program = compile_sequence(
        _sequence(slots=(slot,)),
        StreamerParams(max_edges=8, bank_size=2),
        50e6,
    )
    program = replace(
        program,
        repeat_forever=True,
        scan_points=((1,), (2,), (1,), (3,), (1,)),
        scan_point_durations=(2e-8, 3e-8, 2e-8, 4e-8, 2e-8),
    )
    expected = _reference_play(program, 40)
    actual, stalled, _ = _streaming_scan_play(
        program, 40, bank_size=2, refill_delay=0
    )
    assert actual == expected
    assert stalled is False


def test_scan_table_late_refill_raises_underflow() -> None:
    slot = PulseSlot("duration", PulseFieldRef("duration", "p0"), "ns", "p0_time")
    program = compile_sequence(
        _sequence(slots=(slot,)),
        StreamerParams(max_edges=8, bank_size=2),
        50e6,
    )
    program = replace(
        program,
        repeat_forever=True,
        scan_points=((1,), (2,), (1,), (3,), (1,)),
        scan_point_durations=(2e-8, 3e-8, 2e-8, 4e-8, 2e-8),
    )
    with np.testing.assert_raises(_ScanUnderflow):
        _streaming_scan_play(
            program,
            100,
            bank_size=2,
            refill_delay=20,
            raise_on_underflow=True,
        )


def test_model_rejects_non_binary_states_and_non_dac_value_slots() -> None:
    try:
        PulsePeriod("bad", 20, "ns", (2, 0, 0, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("non-binary state was accepted")
    with np.testing.assert_raises(ValueError):
        PulseSequence(
            target=_target(),
            periods=(PulsePeriod("p0", 20, "ns", (1, 0, 0, 0)),),
            slots=(PulseSlot("dac", PulseFieldRef("dac", "p0", "d0"), "value"),),
        )


def test_a_bracket_around_the_whole_pulse_is_the_pulse_saying_how_many_times() -> None:
    """Who decides "forever or N times" -- the document, once.

    A bracket over PART of a pulse repeats that part inside a cycle that still
    runs until stopped.  A bracket END TO END is a statement about the pulse
    itself and overrides the default.  That rule was worked out by the preview
    (to pick which brackets to draw) and by nobody at all on the path that
    fires, so a whole-pulse bracket was drawn faithfully and then run forever
    anyway -- N times over and over reads exactly like over and over.
    """

    from zlc_pulse import RepeatRegion

    plain = _sequence()
    assert plain.whole_pulse_repeat is None, "no bracket means until stopped"

    part = replace(plain, repeat=RepeatRegion("p0", "p1", 3))
    assert part.whole_pulse_repeat is None, "a partial bracket leaves the outer level alone"

    whole = replace(plain, repeat=RepeatRegion("p0", "p2", 3))
    assert whole.whole_pulse_repeat == 3

    # And the count is what the board is handed: one loop region over the whole
    # program, so ONE fire plays the pulse three times.
    program = compile_sequence(whole, StreamerParams(max_edges=8, bank_size=2), 50e6)
    assert program.loop_start_index == 0
    assert program.loop_count == 3
    assert program.loop_end_tick == program.ticks[-1]
    assert not program.repeat_forever
    assert trigger_windows(program, "d0") == ((0, 1), (3, 4), (6, 7))
    assert trigger_windows(program, "d1") == ((1, 2), (4, 5), (7, 8))
