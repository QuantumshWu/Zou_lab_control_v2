"""One set of cases, run against the real sequencer and its virtual twin.

This project has paid twice for the same mistake: a virtual device that does not
model a rule the real one enforces, so the guard for that rule is green forever
and the failure only appears at the bench.  It cost a week on the FPGA command
strobe, and it was about to cost a calibration run -- CalibrationTask fired in a
bare loop with no wait, which the virtual sequencer happily accepted while a
real board drops every fire after the first.

A hand-copied state machine plus a reconciliation table does not prevent this.
Only running the SAME cases against BOTH does, which is what this file is.  The
real device is driven through a memory transport: what is substituted is the
wire, not the logic under test.

When one of these fails for the virtual device, the virtual device is wrong.
The rules come from the hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_atom.devices.sequencer import SequencerDevice
from zlc_atom.devices.simulation import (
    SimulationWorld,
    SimulationWorldConfig,
    VirtualPulseStreamer,
    VirtualSequencer,
)


def test_virtual_sequencer_is_the_canonical_sequencer_device() -> None:
    assert isinstance(VirtualSequencer(world=SimulationWorld()), SequencerDevice)


def _real_streamer():
    """The real host, talking to a memory-backed register file."""

    from zlc_pulse import compile_sequence, load_streamer_config, pulse_target_from_xdc
    from zlc_pulse import PulsePeriod, PulseSequence
    from zlc_pulse.device import PulseStreamer
    from zlc_pulse.transport import MemoryRegisterTransport

    config = load_streamer_config()
    geometry = config["params"]
    target = pulse_target_from_xdc()
    lanes = len(target.raw_lanes)
    sequence = PulseSequence(
        name="contract",
        target=target,
        time_step_ns=20.0,
        periods=(
            PulsePeriod("on", 1e-5, "s", tuple([1] + [0] * (lanes - 1))),
            PulsePeriod("off", 1e-5, "s", tuple([0] * lanes)),
        ),
    )
    program = compile_sequence(sequence, geometry, config["clock_hz"])
    transport = MemoryRegisterTransport(geom=geometry, auto_done=True)
    streamer = PulseStreamer(
        transport,
        geometry,
        config["clock_hz"],
        target=target,
    )
    streamer.open()
    return streamer, program


def _virtual_streamer():
    """The twin, with a program shaped like the one the real host takes."""

    from tests.pulse_fixture import build_calibration_pulse

    streamer = VirtualPulseStreamer(
        world=SimulationWorld(SimulationWorldConfig(seed=0))
    )
    streamer.open()
    program = build_calibration_pulse(streamer.describe())
    return streamer, program


@pytest.fixture(params=["real", "virtual"])
def sequencer(request):
    """The same cases, once per implementation."""

    streamer, program = _real_streamer() if request.param == "real" else _virtual_streamer()
    device = SequencerDevice(streamer)
    device.kind = request.param
    try:
        yield device, program
    finally:
        device.close()


def test_firing_without_a_program_is_refused(sequencer) -> None:
    """There is nothing to fire.  Silently doing nothing is the worse answer."""

    streamer, _program = sequencer
    with pytest.raises(RuntimeError):
        streamer.fire()


def test_a_second_fire_before_the_first_finishes_is_refused(sequencer) -> None:
    """The failure that made a scan take exactly one shot.

    The board detects commands on a rising edge and its COMMAND register is
    never cleared by hardware, so a fire issued while one is running is dropped
    -- silently, with a normal-looking report.  A device that accepts it is
    lying about what the hardware will do.
    """

    streamer, program = sequencer
    streamer.load(program)
    streamer.fire()
    with pytest.raises(RuntimeError):
        streamer.fire()


def test_a_second_fire_after_waiting_is_allowed(sequencer) -> None:
    """The other half: waiting is what makes the next shot legal."""

    streamer, program = sequencer
    streamer.load(program)
    for _shot in range(3):
        streamer.fire()
        assert streamer.wait_done(1.0) is not None


def test_waiting_when_nothing_is_firing_reports_nothing(sequencer) -> None:
    """None means "no shot is running", and it must not be confused with a report.

    A device that always answers with a report lets a caller believe a shot
    completed when none was ever started.
    """

    streamer, program = sequencer
    streamer.load(program)
    assert streamer.wait_done(0.1) is None
    streamer.fire()
    assert streamer.wait_done(1.0) is not None
    assert streamer.wait_done(0.1) is None, "the same shot was reported twice"


def test_loading_while_firing_is_refused(sequencer) -> None:
    """Replacing the program mid-shot would produce data from neither."""

    streamer, program = sequencer
    streamer.load(program)
    streamer.fire()
    with pytest.raises(RuntimeError):
        streamer.load(program)


def test_a_closed_device_refuses_everything(sequencer) -> None:
    streamer, program = sequencer
    streamer.close()
    with pytest.raises(RuntimeError):
        streamer.load(program)
    with pytest.raises(RuntimeError):
        streamer.fire()


def test_wait_done_reports_the_shot_that_just_ran(sequencer) -> None:
    streamer, program = sequencer
    from zlc_pulse import PulseStreamer

    assert isinstance(streamer.streamer, PulseStreamer)
    described = streamer.describe()
    assert described.geometry == streamer.streamer.geom
    assert described.clock_hz == streamer.streamer.clock_hz
    streamer.load(program)
    assert streamer.applied() is streamer.streamer.applied()
    streamer.fire()
    report = streamer.wait_done(1.0)
    assert report is not None
    assert hasattr(report, "status")
    assert hasattr(report, "cursor")


def test_safe_is_answerable_at_any_time(sequencer) -> None:
    """Stopping the experiment must never depend on what state it is in."""

    streamer, program = sequencer
    assert streamer.safe() is not None
    streamer.load(program)
    streamer.fire()
    assert streamer.safe() is not None
