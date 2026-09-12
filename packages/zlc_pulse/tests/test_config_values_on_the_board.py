"""Config is applied at load and refreshed from its bound file at Fire."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest
import zlc_pulse.device as device_module
from zlc_pulse.codec import config_values_to_tree

from zlc_pulse import (
    PulseConfigParameter,
    PulsePeriod,
    PulsePortSpec,
    PulseSequence,
    PulseTarget,
    RemotePulseStreamer,
    compile_sequence,
)
from zlc_pulse.device import PulseStreamer
from zlc_pulse.model import PulseFieldRef
from zlc_pulse.transport import MemoryRegisterTransport
from zlc_pulse.wire import StreamerParams


def _geometry() -> StreamerParams:
    """A board shaped like the pulse below -- the constructor insists."""

    return replace(
        StreamerParams(),
        channel_count=3,
        bus_count=1,
        bus_width=2,
        max_edges=8,
        bank_size=2,
    )


def _configured() -> PulseSequence:
    target = PulseTarget(
        lanes=("d0", "a0", "a1"),
        ports=(
            PulsePortSpec("d0", "digital", ("d0",)),
            PulsePortSpec("dac", "dac", ("a0", "a1"), bus_index=0),
        ),
    )
    return PulseSequence(
        name="configured",
        target=target,
        time_step_ns=20,
        periods=(
            PulsePeriod("p0", 40, "ns", (1, 0, 0)),
            PulsePeriod("p1", 40, "ns", (0, 0, 0)),
        ),
        config_parameters=(
            PulseConfigParameter("probe_time", PulseFieldRef("duration", "p1"), "ns"),
        ),
    )


@pytest.fixture
def streamer():
    sequence = _configured()
    geom = _geometry()
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    return PulseStreamer(transport, geom, 50e6, target=sequence.target), geom


def test_compilation_is_pure_and_load_applies_the_held_set(streamer):
    device, geom = streamer
    sequence = _configured()

    device.load_config_values({1: (100, "ns")}, source="today.json")
    source, program = device.compile_pulse(sequence, geom, 50e6)
    assert source is sequence
    assert program == compile_sequence(sequence, geom, 50e6)
    assert device.config_values() == {"1": (100.0, "ns")}
    assert device.config_source == "today.json"
    device.open()
    try:
        device.load(program, source=source)
        applied = device.applied()
        assert applied.authored_source is sequence
        assert applied.source.period_by_id["p1"].duration == 100
        assert applied.program != program
        assert applied.program == compile_sequence(applied.source, geom, 50e6)
        assert applied.source.config_parameters == sequence.config_parameters
    finally:
        device.close()


def test_fire_refreshes_file_and_keeps_the_original_defaults(streamer, tmp_path, monkeypatch):
    device, geom = streamer
    path = tmp_path / "current.json"
    def write(values):
        path.write_text(json.dumps(config_values_to_tree(values)), encoding="utf-8")
    write({"1": (80, "ns")})
    device.load_config_file(path)
    write({"1": (100, "ns")})
    source, program = device.compile_pulse(_configured(), geom, 50e6)
    assert device.config_values() == {"1": (80.0, "ns")}
    counts = {"compile": 0, "load": 0, "fire": 0}
    for key, owner, name in (
        ("compile", device_module, "compile_sequence"),
        ("load", device, "_load_program"),
        ("fire", device, "_fire_program"),
    ):
        original = getattr(owner, name)
        def counted(*args, _key=key, _original=original, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(owner, name, counted)

    device.open()
    try:
        device.load(program, source=source)
        assert counts == {"compile": 1, "load": 1, "fire": 0}
        assert device.applied().source.period_by_id["p1"].duration == 100
        for entries, duration, repeats, recompiles in (
            ({"1": (100, "ns")}, 100, 1, 0),
            ({"1": (0.2, "us")}, 200, 3, 1),
            ({}, 40, 2, 1),
            ({"9": (700, "value")}, 40, 1, 0),
        ):
            write(entries)
            before = counts.copy()
            previous_source = device.applied().source
            device.fire(run_repeats=repeats)
            assert device.wait_done(1.0) is not None
            assert counts == {"compile": before["compile"] + recompiles,
                              "load": before["load"] + recompiles,
                              "fire": before["fire"] + 1}
            state = device.applied()
            if not recompiles:
                assert state.source is previous_source
            assert state.authored_source is source
            assert state.source.period_by_id["p1"].duration == duration
            assert state.program == compile_sequence(state.source, geom, 50e6)
            assert state.run_repeats == repeats
            assert device.config_source == str(path), "an empty JSON set still follows its file"
        before = counts.copy()
        write({"1": (100, "Hz")})
        with pytest.raises(ValueError, match="time unit"):
            device.fire(run_repeats=1)
        assert counts == before
        for malformed in (
            "{",
            '{"format":"zlc.pulse.config_values","name":"","source":"hand",'
            '"values":{"1":{"value":1e999,"unit":"ns"}}}',
        ):
            path.write_text(malformed, encoding="utf-8")
            with pytest.raises(ValueError):
                device.fire(run_repeats=1)
            assert counts == before
        # Explicit in-memory data unbinds the file; its label is not a path.
        device.load_config_values({"1": (120, "ns")}, source=str(path))
        device.fire(run_repeats=1)
        assert device.wait_done(1.0) is not None
        assert device.applied().source.period_by_id["p1"].duration == 120
        for cleared_path in (None, ""):
            write({"1": (80, "ns")})
            device.load_config_file(path)
            device.fire(run_repeats=1)
            assert device.wait_done(1.0) is not None
            assert device.applied().source.period_by_id["p1"].duration == 80
            path.write_text("{", encoding="utf-8")
            device.load_config_file(cleared_path)
            assert device.config_values() == {} and device.config_source == ""
            device.fire(run_repeats=1)
            assert device.wait_done(1.0) is not None
            assert device.applied().source.period_by_id["p1"].duration == 40
    finally:
        device.close()


def test_unmatched_config_ids_keep_authored_values(streamer):
    device, geom = streamer
    sequence = _configured()
    sequence = replace(sequence, config_parameters=sequence.config_parameters + (
        PulseConfigParameter("prep_time", PulseFieldRef("duration", "p0"), "ns"),
    ))
    source, program = device.compile_pulse(sequence, geom, 50e6)
    device.open()
    try:
        for entries in ({}, {"9": (1, "value")}):
            device.load_config_values(entries)
            device.load(program, source=source)
            assert device.applied().source is sequence
            assert device.applied().program == program
        device.load_config_values({"1": (0.1, "us"), "9": (1, "value")})
        device.load(program, source=source)
        applied = device.applied()
        assert applied.source.period_by_id["p0"].duration == 40
        assert applied.source.period_by_id["p1"].duration == 100
        assert applied.source.config_parameters == sequence.config_parameters
        assert sequence.period_by_id["p1"].duration == 40
        assert applied.program == compile_sequence(applied.source, geom, 50e6)
        for bad_unit in ("value", "Hz"):
            device.load_config_values({"1": (100, bad_unit)})
            with pytest.raises(ValueError, match="declares|time unit"):
                device.load(program, source=source)
            assert device.applied() is applied
    finally:
        device.close()


def test_a_pulse_declaring_nothing_needs_no_set(streamer):
    """A board with no calibration loaded still plays every ordinary pulse."""

    device, geom = streamer
    bare = _configured()
    bare = bare.__class__(
        name=bare.name,
        target=bare.target,
        time_step_ns=bare.time_step_ns,
        periods=bare.periods,
    )
    filled, program = device.compile_pulse(bare, geom, 50e6)
    assert filled == bare
    assert program.ticks == compile_sequence(bare, geom, 50e6).ticks


def test_the_set_survives_a_close_and_reopen(streamer):
    """A calibration is a fact about the apparatus, not about a connection."""

    device, _geom = streamer
    device.load_config_values({"1": (100, "ns")}, source="today.json")
    device.open()
    device.close()
    device.open()
    try:
        assert device.config_values() == {"1": (100.0, "ns")}
        assert device.snapshot()["config_source"] == "today.json"
    finally:
        device.close()


def test_the_remote_client_holds_its_own_set_and_compiles_without_io() -> None:
    from zlc_pulse.remote import REMOTE_METHODS

    assert REMOTE_METHODS == (
        "open", "describe", "close", "load", "fire",
        "wait_done", "cursor", "safe", "snapshot", "applied",
    )

    # No listener at this port: any call that touched the socket would raise.
    client = RemotePulseStreamer("127.0.0.1", 65280)
    sequence = _configured()
    geom = _geometry()
    filled, program = client.compile_pulse(sequence, geom, 50e6)
    assert filled is sequence
    assert program.ticks == compile_sequence(sequence, geom, 50e6).ticks
    client.load_config_values({"1": (100, "ns")}, source="today.json")
    assert client.config_values() == {"1": (100.0, "ns")}
    assert client.config_source == "today.json"

    filled, program = client.compile_pulse(sequence, geom, 50e6)
    assert filled is sequence
    assert program.ticks == compile_sequence(filled, geom, 50e6).ticks


def test_a_set_that_is_not_a_set_is_refused_at_the_door(streamer) -> None:
    """Whatever reads the file, the device is what says the shape is wrong."""

    device, _geom = streamer
    for bad, message in (
        ({"1": 100}, "must be"),
        ({"1": (float("nan"), "ns")}, "finite"),
        ({"1": (100, "")}, "unit"),
        ({"": (100, "ns")}, "positive"),
        ({"probe_time": (100, "ns")}, "positive"),
        ({1: (100, "ns"), "1": (200, "ns")}, "duplicate"),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            device.load_config_values(bad)
    assert device.config_values() == {}
