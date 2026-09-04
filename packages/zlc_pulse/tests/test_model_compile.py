from __future__ import annotations

from dataclasses import replace

import numpy as np

from zlc_pulse import (
    AnalogStep,
    OutputDelay,
    PulseBracket,
    PulsePeriod,
    PulsePortSpec,
    PulseApiParameter,
    PulseConfigParameter,
    PulseSequence,
    PulseSlot,
    PulseTarget,
    apply_config_values,
    compile_sequence,
    sequence_from_tree,
    sequence_to_tree,
)
from zlc_pulse.compile import evaluate_affine_tick
from zlc_pulse import pulse_field_value
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


def _configured() -> PulseSequence:
    """A pulse whose probe duration and DAC bias come from its own config."""

    base = _sequence()
    return replace(
        base,
        delays=(OutputDelay("d1", 40, "ns"),),
        config_parameters=(
            PulseConfigParameter(
                "probe_time", PulseFieldRef("duration", "p1"), "ns"
            ),
            PulseConfigParameter(
                "bias_x", PulseFieldRef("dac", "p0", "dac"), "value"
            ),
            PulseConfigParameter(
                "gate_delay", PulseFieldRef("delay", port="d1"), "ns"
            ),
        ),
    )


def test_a_config_parameter_reads_the_number_the_pulse_already_carries() -> None:
    """It is not a hole: the field's own value IS the config value."""

    configured = _configured()
    assert {
        parameter.parameter_id: pulse_field_value(
            configured, parameter.field_ref, parameter.unit
        )
        for parameter in configured.config_parameters
    } == {"probe_time": 20.0, "bias_x": 0.0, "gate_delay": 40.0}


def test_applying_a_config_set_overwrites_the_authored_numbers() -> None:
    """The overwrite is the storage: afterwards the pulse holds what was applied.

    The sequencer fills a config parameter by writing the number into the
    field it names, so the compiled program and the sequence handed back as
    ``source=`` describe the same pulse.
    """

    sequence, applied, unknown = apply_config_values(
        _configured(),
        {
            "probe_time": (80, "ns"),
            "bias_x": (1, "value"),
            "somebody_elses": (1, "ns"),
        },
    )
    assert sorted(applied) == ["bias_x", "probe_time"]
    assert unknown == ("somebody_elses",)
    # Written into the fields themselves, not kept beside them.
    assert sequence.period_by_id["p1"].duration == 80
    assert sequence.period_by_id["p0"].analog_steps[0].value == 1
    # An id the set omitted keeps the number the operator authored.
    assert pulse_field_value(sequence, PulseFieldRef("delay", port="d1"), "ns") == 40.0
    # The declarations survive an apply; only the numbers moved.
    assert len(sequence.config_parameters) == 3


def test_a_declared_config_parameter_needs_no_resolving_to_compile() -> None:
    """Nothing to bake: the number is already the field's.

    An API parameter is a hole, so the compiler refuses one that is still
    open.  A config parameter never is -- its declaration is a name and a
    reason to refresh, not a promise somebody still owes -- so a pulse
    carrying them compiles to exactly the program its numbers describe.
    """

    geometry = StreamerParams(max_edges=8, bank_size=2)
    configured = _configured()
    program = compile_sequence(configured, geometry, 50e6)
    bare = compile_sequence(
        replace(configured, config_parameters=()),
        geometry,
        50e6,
    )
    assert program.ticks == bare.ticks
    assert program.masks == bare.masks


def test_one_field_carries_one_binding_and_one_id_namespace() -> None:
    """Scan, API and config are three answers to one question, so they exclude."""

    base = _sequence()
    duration = PulseFieldRef("duration", "p0")
    with np.testing.assert_raises_regex(ValueError, "at most one binding"):
        replace(
            base,
            api_parameters=(PulseApiParameter("t", duration, "ns"),),
            config_parameters=(PulseConfigParameter("t2", duration, "ns"),),
        )
    with np.testing.assert_raises_regex(ValueError, "unique id namespace"):
        replace(
            base,
            api_parameters=(PulseApiParameter("shared", duration, "ns"),),
            config_parameters=(
                PulseConfigParameter(
                    "shared", PulseFieldRef("duration", "p1"), "ns"
                ),
            ),
        )


