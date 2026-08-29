"""A run is one timeline, and an edge stream says its own levels.

Every lane of a program is played by the same clock over the same point
table, so the shape of a point -- its effective ticks, its loop bounds, its
duration -- has no channel in it.  Deriving it per lane made a fire cost a
multiple of the number of DELAYED lanes, and how many lanes are delayed is
not something an operator chooses on purpose: a delay is authored per port,
and a negative one is expressed by lifting every OTHER driven lane, so a
single ``-100 ns`` turns every driven lane into a delayed channel.  That
walk happens inside ``fire()``, before the board is strobed.

The other half is what the walk carries.  An edge IS a change, so the levels
alternate by construction; storing a bool beside every tick repeated what
the position already says, and the FIFO-capacity check then asked for
exposure windows only to pair every edge with its neighbour, flatten the
pairs, and sort them back into the order they came out in.
"""

from __future__ import annotations

import numpy as np

from zlc_pulse import (
    AnalogStep,
    PulsePeriod,
    PulseSequence,
    PulseSlot,
    RepeatRegion,
    compile_sequence,
    pulse_target_from_xdc,
)
from zlc_pulse.model import PulseFieldRef
from zlc_pulse.wire import StreamerParams
from zlc_pulse import schedule
from zlc_pulse.schedule import (
    trigger_edge_ticks,
    trigger_windows,
    trigger_windows_by_channel,
)

_TARGET = pulse_target_from_xdc()
_DIGITAL = [port for port in _TARGET.ports if port.kind == "digital"][:6]
_DAC = next(port for port in _TARGET.ports if port.kind == "dac")
_LANES = tuple(port.lanes[0] for port in _DIGITAL)


def _program(*, periods: int = 9):
    rows = []
    for index in range(periods):
        states = [0] * len(_TARGET.raw_lanes)
        for offset, lane in enumerate(_LANES):
            if (index + offset) % 3:
                states[_TARGET.raw_lanes.index(lane)] = 1
        steps = (AnalogStep(_DAC.key, "edge", index),) if index == 0 else ()
        rows.append(PulsePeriod(f"p{index}", 40, "ns", tuple(states), steps))
    sequence = PulseSequence(
        target=_TARGET,
        time_step_ns=20,
        periods=tuple(rows),
        slots=(PulseSlot("duration", PulseFieldRef("duration", "p0"), "ns", "p0_time"),),
        repeat=RepeatRegion("p1", f"p{periods - 2}", 4),
    )
    return compile_sequence(sequence, StreamerParams(), 50e6)


def _table(rows: int = 5) -> np.ndarray:
    program = _program()
    return np.asarray(
        [[1000 + 20 * index] * program.slot_count for index in range(rows)],
        dtype=np.int64,
    )


def test_a_point_is_shaped_once_however_many_lanes_ask_about_it(monkeypatch) -> None:
    """The redundancy, stated as a count and not as a stopwatch.

    Five distinct rows played over twenty cycles by six lanes: the point
    shape is a property of the program and the row, so it is derived five
    times.  Per lane it was derived thirty.
    """

    program = _program()
    table = _table(5)
    calls: list[tuple[int, ...]] = []
    original = schedule._point_timing

    def counted(prog, point, point_index):
        calls.append(point)
        return original(prog, point, point_index)

    monkeypatch.setattr(schedule, "_point_timing", counted)
    trigger_edge_ticks(program, _LANES, table, cycles=20)

    assert len(calls) == 5, f"{len(calls)} derivations for 5 distinct rows"


def test_asking_about_every_lane_at_once_answers_what_asking_one_by_one_does(
) -> None:
    program = _program()
    table = _table(4)
    together = trigger_edge_ticks(program, _LANES, table, cycles=9)
    for lane in _LANES:
        alone = trigger_edge_ticks(program, (lane,), table, cycles=9)
        assert together[lane] == alone[lane], lane
    windows = trigger_windows_by_channel(program, _LANES, table, cycles=9)
    for lane in _LANES:
        assert windows[lane] == trigger_windows(program, lane, table, cycles=9)


def test_an_edge_stream_states_its_levels_by_position() -> None:
    """Rise, fall, rise, fall -- so a window is a pair, not a search."""

    program = _program()
    table = _table(3)
    for lane in _LANES:
        edges = trigger_edge_ticks(program, (lane,), table, cycles=7)[lane]
        windows = trigger_windows(program, lane, table, cycles=7)
        assert tuple(zip(edges[0::2], edges[1::2])) == windows
        assert list(edges) == sorted(edges)


def test_the_capacity_question_is_asked_of_edges_not_of_exposures() -> None:
    """A delay FIFO holds an entry per edge, rise and fall alike.

    The check used to ask for exposure windows and then take them apart --
    pairing every edge with its neighbour, flattening the pairs, and
    sorting them back into the order they were already in.  This is that
    stream, delivered in that order, without the round trip.
    """

    program = _program()
    table = _table(4)
    windows = trigger_windows_by_channel(program, _LANES, table, cycles=11)
    edges = trigger_edge_ticks(program, _LANES, table, cycles=11)
    for lane in _LANES:
        flattened = [tick for window in windows[lane] for tick in window]
        assert list(edges[lane]) == flattened == sorted(flattened)
