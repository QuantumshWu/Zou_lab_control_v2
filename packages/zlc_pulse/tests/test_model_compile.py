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
    PulseBinding,
    PulseTarget,
    analog_levels,
    apply_config_values,
    compile_sequence,
    resolve_api_parameters,
    sequence_from_tree,
    sequence_to_tree,
)
from zlc_pulse import pulse_field_value
from zlc_pulse.model import PulseFieldRef
from zlc_pulse.schedule import trigger_edge_ticks, trigger_windows_by_channel
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


def _sequence(*, bindings=(), delays=(), first_duration=20) -> PulseSequence:
    return PulseSequence(
        name="test",
        target=_target(),
        time_step_ns=20,
        periods=(
            PulsePeriod("p0", first_duration, "ns", (1, 0, 0, 0), (AnalogStep("dac", "edge", 0),)),
            PulsePeriod("p1", 20, "ns", (0, 1, 0, 0)),
            PulsePeriod("p2", 20, "ns", (0, 0, 0, 0)),
        ),
        bindings=bindings,
        delays=delays,
    )


def test_static_compile_is_one_row_per_period_and_pure_trigger_projection() -> None:
    program = compile_sequence(_sequence(), StreamerParams(max_rows=8, bank_size=2), 50e6)
    assert program.durations == (1, 1, 1)
    assert program.duration_slots == (0, 0, 0)
    assert program.masks == (0b01, 0b10, 0)
    assert program.loops == ()
    assert program.bus_actions[0].row == 0 and program.bus_actions[0].value == 2
    edges = trigger_edge_ticks(program, ("d0", "d1"))
    assert edges["d0"][0::2] == (0,)
    assert edges["d1"][0::2] == (1,)


def test_slot_compile_changes_only_the_row_and_dac_selectors() -> None:
    slot = PulseBinding(PulseFieldRef('duration', 'p0'), 'ns', scan=True)
    program = compile_sequence(
        _sequence(bindings=(slot,)),
        StreamerParams(max_rows=8, bank_size=2),
        50e6,
    )
    assert program.slot_kinds == ("duration",)
    assert program.slot_count == 1
    assert program.duration_slots == (1, 0, 0)
    assert program.resolved_durations((3,)) == (3, 1, 1)
    assert program.frame_visits((3,)) == ((0, 0), (1, 3), (2, 4))
    assert program.frame_ticks((3,)) == 5
    dac_slot = PulseBinding(PulseFieldRef('dac', 'p0', 'dac'), 'value', scan=True)
    scanned = compile_sequence(
        _sequence(bindings=(slot, dac_slot)),
        StreamerParams(max_rows=8, bank_size=2),
        50e6,
    )
    assert scanned.bus_actions[0].value_select == 2
    assert scanned.resolved_bus_value(scanned.bus_actions[0], (3, 1)) == 1
    with np.testing.assert_raises_regex(ValueError, "scan slot"):
        compile_sequence(
            _sequence(bindings=(slot, dac_slot)),
            StreamerParams(max_rows=8, bank_size=2, num_slots=1),
            50e6,
        )


def test_negative_bus_delay_shifts_every_driven_ttl_lane() -> None:
    program = compile_sequence(
        _sequence(delays=(OutputDelay("dac", -40, "ns"),)),
        StreamerParams(max_rows=8, bank_size=2),
        50e6,
    )
    assert program.channel_delays[:2] == (2, 2)
    assert program.channel_delays[2:] == (0, 0)
    assert program.bus_delays == ()
    declared_zero = compile_sequence(
        _sequence(delays=(OutputDelay("dac", -40, "ns"), OutputDelay("d1", 0, "ns"))),
        StreamerParams(max_rows=8, bank_size=2),
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
            bindings=(PulseBinding(PulseFieldRef('dac', 'p0', 'd0'), 'value', scan=True),),
        )


def test_a_full_span_bracket_is_only_an_internal_timeline_loop() -> None:
    """A bracket's position never decides finite/forever run policy."""

    whole = replace(_sequence(), brackets=(PulseBracket("whole", "p0", "p2", 3),))
    program = compile_sequence(whole, StreamerParams(max_rows=8, bank_size=2), 50e6)
    assert program.loops == ((0, 2, 3),)
    assert program.frame_ticks() == 9
    windows = trigger_windows_by_channel(program, ("d0", "d1"))
    assert windows["d0"] == ((0, 1), (3, 4), (6, 7))
    assert windows["d1"] == ((1, 2), (4, 5), (7, 8))


def test_a_sequence_says_how_long_the_board_plays_it() -> None:
    """One pass through the periods is what the strip shows; the board plays
    every bracket, so the model's played duration expands the loop table
    with the compiler's own walk, and an empty bracket plays nothing."""

    from fractions import Fraction

    from zlc_pulse.model import nanoseconds_per

    base = _sequence()
    one_pass = float(sum(
        Fraction(str(float(period.duration))) * nanoseconds_per(period.unit)
        for period in base.periods
    ))
    assert base.loops == ()
    assert base.played_nanoseconds() == one_pass
    whole = replace(base, brackets=(PulseBracket("whole", "p0", "p2", 3),))
    assert whole.loops == ((0, 2, 3),)
    assert whole.played_nanoseconds() == 3 * one_pass
    nested = replace(base, brackets=(
        PulseBracket("inner", "p1", "p1", 3), PulseBracket("outer", "p0", "p1", 2),
    ))
    program = compile_sequence(nested, StreamerParams(max_rows=8, bank_size=2), 50e6)
    assert nested.loops == program.loops
    assert abs(nested.played_nanoseconds() - program.duration_seconds * 1e9) < 1e-6
    empty = replace(base, brackets=(PulseBracket("empty", "p1", "p0", 2),))
    assert empty.loops == ()
    assert empty.played_nanoseconds() == one_pass


def test_a_bracket_of_one_plays_its_range_once_like_no_bracket() -> None:
    """A count of one is legal: the bracket stays in the document and on the
    editor's strip but plays its periods exactly once, which is what an
    operator debugging a loop wants instead of deleting and redrawing it."""

    geometry = StreamerParams(max_rows=8, bank_size=2)
    plain = compile_sequence(_sequence(), geometry, 50e6)
    once = replace(_sequence(), brackets=(PulseBracket("whole", "p0", "p2", 1),))
    program = compile_sequence(once, geometry, 50e6)
    assert program.loops == ((0, 2, 1),)
    assert program.frame_visits() == plain.frame_visits()
    assert program.frame_ticks() == plain.frame_ticks()
    assert trigger_windows_by_channel(program, ("d0", "d1")) == trigger_windows_by_channel(plain, ("d0", "d1"))
    # The wire image carries the count as written: the engine plays 1 once.
    from zlc_pulse.wire import pack_program

    assert pack_program(program, geometry, target=once.target)


def test_brackets_nest_or_stay_apart_and_compile_outermost_first() -> None:
    """Several brackets: disjoint or one inside the other, never crossing.

    The board walks them as nested loops, so the compiled loop table is
    outermost first and its depth is what the geometry must hold.
    """

    base = _sequence()
    inner = PulseBracket("inner", "p1", "p1", 3)
    outer = PulseBracket("outer", "p0", "p1", 2)
    nested = replace(base, brackets=(inner, outer))
    assert [bracket.bracket_id for bracket in nested.brackets] == ["outer", "inner"]
    assert nested.bracket_bounds == ((0, 2), (1, 2))
    assert nested.bracket_depths == (1, 2)
    program = compile_sequence(nested, StreamerParams(max_rows=8, bank_size=2), 50e6)
    assert program.loops == ((0, 1, 2), (1, 1, 3))
    assert program.loop_depth == 2
    assert program.frame_visits() == (
        (0, 0), (1, 1), (1, 2), (1, 3), (0, 4), (1, 5), (1, 6), (1, 7), (2, 8),
    )
    assert program.frame_ticks() == 9
    assert trigger_windows_by_channel(program, ("d0", "d1")) == {
        "d0": ((0, 1), (4, 5)),
        "d1": ((1, 4), (5, 8)),
    }
    apart = replace(base, brackets=(
        PulseBracket("late", "p2", "p2", 2), PulseBracket("early", "p0", "p0", 2),
    ))
    assert [bracket.bracket_id for bracket in apart.brackets] == ["early", "late"]
    assert apart.bracket_depths == (1, 1)
    with np.testing.assert_raises_regex(ValueError, "overlap"):
        replace(base, brackets=(
            PulseBracket("a", "p0", "p1", 2), PulseBracket("b", "p1", "p2", 2),
        ))
    with np.testing.assert_raises_regex(ValueError, "unique"):
        replace(base, brackets=(inner, replace(inner, count=4)))
    with np.testing.assert_raises_regex(ValueError, "nest 2 deep"):
        compile_sequence(nested, StreamerParams(max_rows=8, bank_size=2, loop_depth=1), 50e6)
    with np.testing.assert_raises_regex(ValueError, "2 brackets"):
        compile_sequence(nested, StreamerParams(max_rows=8, bank_size=2, max_loops=1), 50e6)
    # An empty bracket at another bracket's boundary gap sits beside it: an
    # editor draws it from the same rule, so nesting is decided by bounds alone.
    beside = replace(base, brackets=(
        PulseBracket("empty", "p2", "p1", 2), PulseBracket("body", "p0", "p1", 2),
    ))
    assert beside.bracket_depths == (1, 1)
    within = replace(base, brackets=(
        PulseBracket("empty", "p1", "p0", 2), PulseBracket("body", "p0", "p1", 2),
    ))
    assert within.bracket_depths == (1, 2)


def test_bracket_count_run_repeats_and_scan_slot_domain_are_strict() -> None:
    for invalid in (True, 1.5, 0, -1, 2**32):
        with np.testing.assert_raises((TypeError, ValueError)):
            PulseBracket("b", "p0", "p2", invalid)

    sequence = _sequence()
    for valid in (0, 1, 2**32 - 1):
        assert replace(sequence, run_repeats=valid).run_repeats == valid
    for invalid in (True, 1.5, -1, 2**32):
        with np.testing.assert_raises((TypeError, ValueError)):
            replace(sequence, run_repeats=invalid)

    geometry = StreamerParams(max_rows=8, bank_size=2)
    assert compile_sequence(sequence, geometry, 50e6) == compile_sequence(
        replace(sequence, run_repeats=7), geometry, 50e6
    )

    with np.testing.assert_raises(ValueError):
        PulseBinding(PulseFieldRef('delay', port='d0'), 'ns', scan=True)


def test_pulse_tree_uses_only_brackets_and_run_repeats() -> None:
    authored = replace(
        _sequence(),
        brackets=(PulseBracket("loop", "p0", "p2", 3),),
        run_repeats=7,
    )
    tree = sequence_to_tree(authored)

    assert tree["brackets"] == [{
        "bracket_id": "loop",
        "start_period_id": "p0",
        "end_period_id": "p2",
        "count": 3,
    }]
    assert tree["run_repeats"] == 7
    assert "repeat" not in tree and "bracket" not in tree
    assert not hasattr(authored, "repeat") and not hasattr(authored, "bracket")
    assert sequence_from_tree(tree) == authored

    for start, end, gap in (("p0", None, 0), ("p1", "p0", 1), (None, "p2", 3)):
        empty = replace(authored, brackets=(PulseBracket("loop", start, end, 3),))
        assert empty.bracket_bounds == ((gap, gap),)
        raw = {**tree, "brackets": [
            {"bracket_id": "loop", "start_period_id": start, "end_period_id": end, "count": 3},
        ]}
        messages = []
        for operation in (
            empty.require_nonempty_brackets,
            lambda: compile_sequence(empty, StreamerParams(max_rows=8, bank_size=2), 50e6),
            lambda: sequence_to_tree(empty),
            lambda: sequence_from_tree(raw),
        ):
            try:
                operation()
            except ValueError as error:
                messages.append(str(error))
            else:
                raise AssertionError("empty authoring bracket escaped the execution/save boundary")
        assert len(set(messages)) == 1 and "Bracket loop is empty" in messages[0]
    with np.testing.assert_raises_regex(ValueError, "end precedes"):
        replace(authored, brackets=(PulseBracket("loop", "p2", "p0", 3),))

    for obsolete_key in ("repeat", "bracket"):
        obsolete = dict(tree)
        obsolete[obsolete_key] = obsolete.pop("brackets")[0]
        with np.testing.assert_raises_regex(ValueError, f"unknown pulse field.*{obsolete_key}"):
            sequence_from_tree(obsolete)


def test_compile_binds_the_document_clock_and_complete_geometry() -> None:
    geometry = StreamerParams(max_rows=8, bank_size=2)
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
        bindings=(PulseBinding(PulseFieldRef('duration', 'p1'), 'ns', source='config', config_key='probe_time'), PulseBinding(PulseFieldRef('dac', 'p0', 'dac'), 'value', source='config', config_key='bias_x'), PulseBinding(PulseFieldRef('delay', port='d1'), 'ns', source='config', config_key='gate_delay')),
    )


def test_a_config_parameter_reads_the_number_the_pulse_already_carries() -> None:
    """It is not a hole: the field's own value IS the config value."""

    configured = _configured()
    assert {
        parameter.config_key: pulse_field_value(
            configured, parameter.field_ref, parameter.unit
        )
        for parameter in configured.config_bindings
    } == {"probe_time": 20.0, "bias_x": 0.0, "gate_delay": 40.0}


def test_applying_a_config_set_overwrites_the_authored_numbers(monkeypatch) -> None:
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
            "unrelated": (1, "ns"),
        },
    )
    assert sorted(applied) == ["bias_x", "probe_time"]
    assert unknown == ("unrelated",)
    # Written into the fields themselves, not kept beside them.
    assert sequence.period_by_id["p1"].duration == 80
    assert sequence.period_by_id["p0"].analog_steps[0].value == 1
    # An id the set omitted keeps the number the operator authored.
    assert pulse_field_value(sequence, PulseFieldRef("delay", port="d1"), "ns") == 40.0
    # The declarations survive an apply; only the numbers moved.
    assert len(sequence.config_bindings) == 3

    # A shared Config name may feed a different field in another Pulse.
    other = replace(
        _configured(),
        name="other_pulse",
        bindings=(PulseBinding(PulseFieldRef('duration', 'p0'), 'ns', source='config', config_key='probe_time'),),
    )
    changed, applied, unknown = apply_config_values(other, {"probe_time": (100.0, "ns")})
    assert applied == ("probe_time",) and unknown == ()
    assert changed.period_by_id["p0"].duration == 100
    assert changed.period_by_id["p1"].duration == other.period_by_id["p1"].duration

    # The duration and DAC field of p0 must survive one combined update.
    authored = _configured()
    authored = replace(authored, bindings=authored.config_bindings + (PulseBinding(PulseFieldRef('duration', 'p0'), 'ns', source='config', config_key='load_time'),))
    constructions = []
    original = PulseSequence.__init__
    def count_construction(self, *args, **kwargs):
        constructions.append(1)
        original(self, *args, **kwargs)
    monkeypatch.setattr(PulseSequence, "__init__", count_construction)
    entries = {"probe_time": (80, "ns"), "bias_x": (1, "value"), "load_time": (60, "ns")}
    current = apply_config_values(authored, entries)[0]
    assert len(constructions) == 1
    assert current.period_by_id["p0"].duration == 60
    assert current.period_by_id["p0"].analog_steps[0].value == 1
    constructions.clear()
    # Different spelling / sub-grid values still describe the current pulse.
    equivalent = {"probe_time": (0.0801, "us"), "bias_x": (1.1, "value"), "load_time": (60, "ns")}
    assert apply_config_values(authored, equivalent, current=current)[0] is current
    assert constructions == []
    restored = apply_config_values(authored, {}, current=current)[0]
    assert len(constructions) == 1
    assert restored == authored
    constructions.clear()
    assert apply_config_values(authored, {}, current=restored)[0] is restored
    assert constructions == []
    with np.testing.assert_raises(ValueError):
        apply_config_values(authored, {"probe_time": (100, "ns"), "bias_x": (1, "ns")}, current=current)
    assert constructions == []
    assert current.period_by_id["p1"].duration == 80
    assert authored.period_by_id["p1"].duration == 20

    api = replace(authored, bindings=tuple(
        PulseBinding(item.field_ref, item.unit, source="api")
        for item in authored.config_bindings
    ))
    constructions.clear()
    resolved = resolve_api_parameters(api, {
        "p1.duration": 80, "p0.dac": 1, "d1.delay": 40, "p0.duration": 60,
    })
    assert len(constructions) == 1
    assert resolved.api_bindings == ()
    assert resolved.periods == current.periods
    assert resolved.delays == current.delays


def test_a_declared_config_parameter_needs_no_resolving_to_compile() -> None:
    """Nothing to bake: the number is already the field's.

    An API parameter is a hole, so the compiler refuses one that is still
    open.  A config parameter never is -- its declaration is a name and a
    reason to refresh, not a promise somebody still owes -- so a pulse
    carrying them compiles to exactly the program its numbers describe.
    """

    geometry = StreamerParams(max_rows=8, bank_size=2)
    configured = _configured()
    program = compile_sequence(configured, geometry, 50e6)
    bare = compile_sequence(
        replace(configured, bindings=()),
        geometry,
        50e6,
    )
    assert program.durations == bare.durations
    assert program.masks == bare.masks


def test_a_dac_slot_whose_step_is_gone_is_named_by_the_compiler() -> None:
    """The model admits the binding; the compiler says which one has no field.

    Taking a step away is a legal intermediate state of an edit, pruned
    afterwards, so the model and the file reader accept a DAC slot on a
    period with no step on that port.  Compiling such a pulse used to end in
    ``generator raised StopIteration`` -- true, and naming nothing.
    """

    base = _sequence()
    periods = (replace(base.periods[0], analog_steps=()), *base.periods[1:])
    sequence = replace(
        base,
        periods=periods,
        bindings=(PulseBinding(PulseFieldRef('dac', 'p0', 'dac'), 'value', scan=True),),
    )
    assert sequence_from_tree(sequence_to_tree(sequence)) == sequence
    with np.testing.assert_raises_regex(ValueError, "'dac:p0:dac'.*'dac'.*'p0'.*no step"):
        compile_sequence(sequence, StreamerParams(max_rows=8, bank_size=2), 50e6)


def test_a_named_duration_that_is_not_positive_is_refused_not_rounded_up() -> None:
    """Zero and negative durations are not on any grid.

    A value rounds to the nearest legal tick, the way the editor rounds a
    typed number; ``0`` and ``-100`` used to round UP to one tick, so a node
    form that said either played a 20 ns period and called it the request.
    """

    sequence = replace(
        _sequence(first_duration=100),
        bindings=(PulseBinding(PulseFieldRef('duration', 'p0'), 'ns', source='api'),),
    )
    for value in (0, -100):
        with np.testing.assert_raises_regex(
            ValueError, "'duration:p0' must be a positive duration"
        ):
            resolve_api_parameters(sequence, {"p0.duration": value})
    assert resolve_api_parameters(sequence, {"p0.duration": 10}).periods[0].duration == 20
    assert resolve_api_parameters(sequence, {"p0.duration": 100}).periods[0].duration == 100


def test_analog_levels_walk_a_ramp_the_way_the_engine_does() -> None:
    """An edge is one change at the period start; a ramp is the engine's staircase.

    ``start ± floor(k·|delta|/span)`` on the k-th tick, ending on the target
    exactly where the next period begins.  A preview that read only
    ``step.value`` drew both modes as the same edge.
    """

    # A four-lane DAC, so the codes -8..7 leave room for a staircase.
    wide = PulseTarget(
        lanes=("d0", "d1", "a0", "a1", "a2", "a3"),
        ports=(
            PulsePortSpec("d0", "digital", ("d0",)),
            PulsePortSpec("d1", "digital", ("d1",)),
            PulsePortSpec("dac", "dac", ("a0", "a1", "a2", "a3"), bus_index=0),
        ),
    )
    base = PulseSequence(
        name="levels",
        target=wide,
        time_step_ns=20,
        periods=tuple(
            replace(period, states=period.states + (0, 0))
            for period in _sequence().periods
        ),
    )

    def with_step(mode: str, value: int, duration: int = 100) -> PulseSequence:
        return replace(base, periods=(
            base.periods[0],
            replace(
                base.periods[1],
                duration=duration,
                analog_steps=(AnalogStep("dac", mode, value),),
            ),
            base.periods[2],
        ))

    assert analog_levels(with_step("edge", 3))["dac"] == ((0, 0), (1, 3))
    # Three codes over five ticks from tick 1: level j first holds on the
    # tick k = ceil(j * 5 / 3), so ticks 3, 5 and 6 -- the last being where
    # p2 starts.
    assert analog_levels(with_step("ramp", 3))["dac"] == ((0, 0), (3, 1), (5, 2), (6, 3))
    # Seven codes over two ticks move by floor(k * 7 / 2): 3, then 7.
    assert analog_levels(with_step("ramp", 7, duration=40))["dac"] == ((0, 0), (2, 3), (3, 7))
    # A ramp carries the level it starts from, and two changes on one tick
    # are the later one: the ramp reaches 3 on the tick p2's edge takes 5.
    down = replace(
        with_step("ramp", 3),
        periods=(
            *with_step("ramp", 3).periods[:2],
            replace(base.periods[2], analog_steps=(AnalogStep("dac", "edge", 5),)),
        ),
    )
    assert analog_levels(down)["dac"] == ((0, 0), (3, 1), (5, 2), (6, 5))
    falling = replace(
        down,
        periods=(
            *down.periods,
            PulsePeriod("p3", 60, "ns", (0,) * 6, (AnalogStep("dac", "ramp", 2),)),
        ),
    )
    assert analog_levels(falling)["dac"][-3:] == ((8, 4), (9, 3), (10, 2))


def test_scan_and_value_source_share_one_physical_field_declaration() -> None:
    base = _sequence()
    duration = PulseFieldRef("duration", "p0")
    for source in ("default", "api", "config"):
        binding = PulseBinding(duration, "ns", scan=True, source=source,
                               config_key="shared" if source == "config" else "")
        combined = replace(base, bindings=(binding,))
        assert combined.scan_bindings == (binding,)
        assert combined.api_bindings == ((binding,) if source == "api" else ())
        assert sequence_from_tree(sequence_to_tree(combined)) == combined
        with np.testing.assert_raises_regex(ValueError, "one binding"):
            replace(base, bindings=(binding, replace(binding, scan=False)))
    with np.testing.assert_raises(ValueError):
        PulseBinding(duration, "ns", source="api", config_key="shared")
    with np.testing.assert_raises(ValueError):
        PulseBinding(PulseFieldRef("delay", port="d0"), "ns", scan=True)
    for old_member in ("slots", "api_parameters", "config_parameters"):
        tree = sequence_to_tree(base)
        tree[old_member] = []
        with np.testing.assert_raises_regex(ValueError, "unknown pulse field"):
            sequence_from_tree(tree)

    from zlc_pulse import apply_api_values, resolve_scan_point
    shared = replace(base, bindings=(
        PulseBinding(duration, "ns", scan=True, source="config", config_key="timing"),
        PulseBinding(PulseFieldRef("duration", "p1"), "ns", source="config", config_key="timing"),
        PulseBinding(PulseFieldRef("duration", "p2"), "ns", source="config"),
    ))
    entries = {"timing": (0.08, "us")}
    active, applied, unknown = apply_config_values(shared, entries)
    assert applied == ("timing",) and unknown == ()
    assert tuple(period.duration for period in active.periods) == (20, 80, 20)
    inactive = resolve_scan_point(shared)
    supplied, _, _ = apply_config_values(shared, entries, current=inactive)
    assert tuple(period.duration for period in supplied.periods) == (80, 80, 20)
    assert tuple(period.duration for period in apply_config_values(shared, {}, current=supplied)[0].periods) == (20, 20, 20)
    held = resolve_scan_point(shared, (100,))
    # Explicit held scan values win over Config, while other fields still use it.
    effective, _, _ = apply_config_values(shared, entries, current=held)
    assert tuple(period.duration for period in effective.periods) == (100, 80, 20)
    assert held.bindings[0].source == "default" and held.bindings[0].config_key == ""
    api = replace(base, bindings=(PulseBinding(duration, "ns", scan=True, source="api"),))
    updated, applied, _ = apply_api_values(api, {"p0.duration": (80, "ns")})
    assert applied == (duration.key,)
    resolved = resolve_api_parameters(updated)
    assert resolved.scan_bindings and not resolved.api_bindings
    assert resolved.periods[0].duration == 80
    assert resolve_scan_point(resolved, (100,)).periods[0].duration == 100




def test_a_spacer_holds_its_dacs_and_takes_only_a_config_duration() -> None:
    """A spacer is time between two authored periods, given to a slow device.

    It holds every DAC, so it carries no analog step and no DAC field can be
    bound on it; its length is the device's or a Config value, never scanned
    and never set by an API caller.  The kind survives the file.
    """

    from zlc_pulse import PERIOD_KIND_PERIOD, PERIOD_KIND_SPACER

    assert PulsePeriod("p", 20, "ns", (0, 0, 0, 0)).kind == PERIOD_KIND_PERIOD
    spacer = PulsePeriod("spacer1", 20, "ns", (0, 0, 0, 0), kind=PERIOD_KIND_SPACER)
    with np.testing.assert_raises(ValueError):
        PulsePeriod("s", 20, "ns", (0, 0, 0, 0), (AnalogStep("dac", "edge", 0),), kind=PERIOD_KIND_SPACER)
    with np.testing.assert_raises(ValueError):
        PulsePeriod("s", 20, "ns", (0, 0, 0, 0), kind="gap")

    def with_bindings(*bindings) -> PulseSequence:
        return PulseSequence(
            name="test",
            target=_target(),
            time_step_ns=20,
            periods=(
                PulsePeriod("p0", 20, "ns", (1, 0, 0, 0), (AnalogStep("dac", "edge", 0),)),
                spacer,
                PulsePeriod("p1", 20, "ns", (0, 1, 0, 0)),
            ),
            bindings=bindings,
        )

    duration = PulseFieldRef("duration", "spacer1")
    for refused in (
        PulseBinding(duration, "ns", scan=True),
        PulseBinding(duration, "ns", source="api"),
        PulseBinding(PulseFieldRef("dac", "spacer1", "dac"), "value", source="config", config_key="bias"),
    ):
        with np.testing.assert_raises(ValueError):
            with_bindings(refused)

    configured = with_bindings(PulseBinding(duration, "ns", source="config", config_key="shutter_time"))
    compile_sequence(configured, StreamerParams(max_rows=8, bank_size=2), 50e6)
    reread = sequence_from_tree(sequence_to_tree(configured))
    assert [period.kind for period in reread.periods] == [
        PERIOD_KIND_PERIOD, PERIOD_KIND_SPACER, PERIOD_KIND_PERIOD,
    ]
