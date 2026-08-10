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
from zlc_pulse.engine_model import ScanUnderflow, reference_play, streaming_scan_play
from zlc_pulse.schedule import trigger_times, trigger_windows
from zlc_pulse.wire import StreamerParams


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
    assert reference_play(changed, 12)[:5] == [1, 1, 2, 0, 0]


def test_write_slots_single_row_matches_static_waveform() -> None:
    static = compile_sequence(_sequence(first_duration=20), StreamerParams(max_edges=8, bank_size=2), 50e6)
    slot = PulseSlot("duration", PulseFieldRef("duration", "p0"), "ns", "p0_time")
    slotted = compile_sequence(
        _sequence(slots=(slot,), first_duration=20),
        StreamerParams(max_edges=8, bank_size=2),
        50e6,
    )
    runtime_row = replace(slotted, scan_points=((1,),), scan_point_durations=(4e-8,))
    assert reference_play(runtime_row, 12) == reference_play(static, 12)


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
    expected = reference_play(program, 80)
    actual, stalled, points_played = streaming_scan_play(program, 80, bank_size=2, refill_delay=0)
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
    expected = reference_play(program, 40)
    actual, stalled, _ = streaming_scan_play(program, 40, bank_size=2, refill_delay=0)
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
    with np.testing.assert_raises(ScanUnderflow):
        streaming_scan_play(
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
