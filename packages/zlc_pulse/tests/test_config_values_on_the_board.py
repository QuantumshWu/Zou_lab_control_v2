"""The board holds the calibrated numbers, and is the only door to compiling.

A config parameter's value is a fact about the apparatus -- a channel delay,
a DAC bias -- shared by every pulse fired at that board.  It is baked into
the program at COMPILE (a duration becomes ticks, a step becomes bus
segments), so it cannot be filled at load, and a caller reaching
``compile_sequence`` directly plays the authored placeholder silently.  These
are the guards that make the one door the only door.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

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


def test_compiling_for_a_board_uses_the_held_set_not_the_authored_number(streamer):
    """The whole point: what plays is today's calibration, not the file's."""

    device, geom = streamer
    sequence = _configured()

    device.load_config_values({"probe_time": (100, "ns")}, source="today.json")
    filled, program = device.compile_pulse(sequence, geom, 50e6)

    assert device.config_values() == {"probe_time": (100.0, "ns")}
    assert device.config_source == "today.json"
    # The number moved into the field, and the program is that field's.
    assert filled.period_by_id["p1"].duration == 100
    assert program.ticks != compile_sequence(sequence, geom, 50e6).ticks
    assert program.ticks == compile_sequence(filled, geom, 50e6).ticks
    # The declaration survives, so the next fill still knows what to fill.
    assert filled.config_parameters == sequence.config_parameters


def test_the_filled_sequence_is_what_recompiles_to_what_played(streamer):
    """``load(source=...)`` must be handed the filled one, not the authored one.

    Nothing fails if it is not -- the board plays calibrated numbers while the
    applied state, the archive and the editor's stale-dot all describe the
    authored ones.  The record simply lies, which is why compile_pulse hands
    back both halves rather than only the program.
    """

    device, geom = streamer
    device.load_config_values({"probe_time": (100, "ns")})
    filled, program = device.compile_pulse(_configured(), geom, 50e6)

    device.open()
    try:
        device.load(program, source=filled)
        state = device.applied()
        assert state is not None
        rebuilt = compile_sequence(state.source, geom, 50e6)
        assert rebuilt.ticks == program.ticks
    finally:
        device.close()


def test_a_declaration_the_held_set_cannot_answer_is_refused(streamer):
    """Refusing to run beats running a stale number under a fresh name."""

    device, geom = streamer
    with pytest.raises(ValueError, match="say nothing about"):
        device.compile_pulse(_configured(), geom, 50e6)

    device.load_config_values({"somebody_else": (1, "ns")})
    with pytest.raises(ValueError, match="probe_time"):
        device.compile_pulse(_configured(), geom, 50e6)


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
    device.load_config_values({"probe_time": (100, "ns")}, source="today.json")
    device.open()
    device.close()
    device.open()
    try:
        assert device.config_values() == {"probe_time": (100.0, "ns")}
        assert device.snapshot()["config_source"] == "today.json"
    finally:
        device.close()


def test_the_remote_client_holds_its_own_set_and_sends_nothing() -> None:
    """The wire is unchanged: the values never leave the host that compiles.

    A remote board is driven by a client that compiles locally and ships the
    program, so holding the set client-side is what lets a server built
    before config parameters existed keep serving a host that has them.
    """

    from zlc_pulse.remote import REMOTE_METHODS

    assert REMOTE_METHODS == (
        "open", "describe", "close", "load", "fire",
        "wait_done", "cursor", "safe", "snapshot", "applied",
    )

    # No listener at this port: any call that touched the socket would raise.
    client = RemotePulseStreamer("127.0.0.1", 65280)
    client.load_config_values({"probe_time": (100, "ns")}, source="today.json")
    assert client.config_values() == {"probe_time": (100.0, "ns")}
    assert client.config_source == "today.json"

    geom = _geometry()
    filled, program = client.compile_pulse(_configured(), geom, 50e6)
    assert filled.period_by_id["p1"].duration == 100
    assert program.ticks == compile_sequence(filled, geom, 50e6).ticks


def test_a_set_that_is_not_a_set_is_refused_at_the_door(streamer) -> None:
    """Whatever reads the file, the device is what says the shape is wrong."""

    device, _geom = streamer
    for bad, message in (
        ({"probe_time": 100}, "must be"),
        ({"probe_time": (float("nan"), "ns")}, "finite"),
        ({"probe_time": (100, "")}, "unit"),
        ({"": (100, "ns")}, "non-empty"),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            device.load_config_values(bad)
    assert device.config_values() == {}
