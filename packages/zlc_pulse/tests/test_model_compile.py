from __future__ import annotations

from dataclasses import replace

import numpy as np

from zlc_pulse import (
    AnalogStep,
    OutputDelay,
    PulseBracket,
    PulsePeriod,
    PulsePortSpec,
    PulseSequence,
    PulseSlot,
    PulseTarget,
    compile_sequence,
    sequence_from_tree,
    sequence_to_tree,
)
from zlc_pulse.compile import evaluate_affine_tick
from zlc_pulse.model import PulseFieldRef
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
        slot_tick_scales=(2,),
    )
    assert program.slot_kinds == ("duration",)
    assert program.slot_tick_scales == (2,)
    assert program.slot_count == 1
    assert program.tick_slot_coeffs[1][0] == 2 << program.scan_coeff_frac_bits
    assert evaluate_affine_tick(
        program.ticks[1],
        program.tick_slot_coeffs[1],
        (1,),
        program.scan_coeff_frac_bits,
    ) == 3
    with np.testing.assert_raises_regex(ValueError, "coefficient range"):
        compile_sequence(
            _sequence(slots=(slot,)),
            StreamerParams(max_edges=8, bank_size=2),
            50e6,
            slot_tick_scales=(128,),
        )
    dac_slot = PulseSlot(
        "dac",
        PulseFieldRef("dac", "p0", "dac"),
        "value",
        "dac_value",
    )
    with np.testing.assert_raises_regex(ValueError, "DAC slot tick scale"):
        compile_sequence(
            _sequence(slots=(dac_slot,)),
            StreamerParams(max_edges=8, bank_size=2),
            50e6,
            slot_tick_scales=(2,),
        )


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


def test_a_full_span_bracket_is_only_an_internal_timeline_loop() -> None:
    """A bracket's position never decides finite/forever run policy."""

    whole = replace(_sequence(), bracket=PulseBracket("p0", "p2", 3))
    program = compile_sequence(whole, StreamerParams(max_edges=8, bank_size=2), 50e6)
    assert program.loop_start_index == 0
    assert program.loop_count == 3
    assert program.loop_end_tick == program.ticks[-1]
    assert trigger_windows(program, "d0") == ((0, 1), (3, 4), (6, 7))
    assert trigger_windows(program, "d1") == ((1, 2), (4, 5), (7, 8))


def test_bracket_count_run_repeats_and_scan_slot_domain_are_strict() -> None:
    for invalid in (True, 1.5, 1, 0, -1, 2**32):
        with np.testing.assert_raises((TypeError, ValueError)):
            PulseBracket("p0", "p2", invalid)

    sequence = _sequence()
    for valid in (0, 1, 2**32 - 1):
        assert replace(sequence, run_repeats=valid).run_repeats == valid
    for invalid in (True, 1.5, -1, 2**32):
        with np.testing.assert_raises((TypeError, ValueError)):
            replace(sequence, run_repeats=invalid)

    geometry = StreamerParams(max_edges=8, bank_size=2)
    assert compile_sequence(sequence, geometry, 50e6) == compile_sequence(
        replace(sequence, run_repeats=7), geometry, 50e6
    )

    with np.testing.assert_raises(ValueError):
        PulseSlot(
            "delay",
            PulseFieldRef("delay", port="d0"),
            "ns",
            "d0_delay",
        )


def test_pulse_tree_uses_only_bracket_and_run_repeats() -> None:
    authored = replace(
        _sequence(),
        bracket=PulseBracket("p0", "p2", 3),
        run_repeats=7,
    )
    tree = sequence_to_tree(authored)

    assert tree["bracket"] == {
        "start_period_id": "p0",
        "end_period_id": "p2",
        "count": 3,
    }
    assert tree["run_repeats"] == 7
    assert "repeat" not in tree
    assert not hasattr(authored, "repeat")
    assert sequence_from_tree(tree) == authored

    obsolete = dict(tree)
    obsolete["repeat"] = obsolete.pop("bracket")
    with np.testing.assert_raises_regex(ValueError, "unknown pulse field.*repeat"):
        sequence_from_tree(obsolete)


def test_compile_binds_the_document_clock_and_complete_geometry() -> None:
    geometry = StreamerParams(max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geometry, 50e6)
    assert program.geometry_fingerprint != 0
    with np.testing.assert_raises(ValueError):
        compile_sequence(_sequence(), geometry, 25e6)
