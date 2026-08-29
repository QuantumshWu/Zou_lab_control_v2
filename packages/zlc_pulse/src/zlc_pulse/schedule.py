"""Pure timing queries derived from a compiled program."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np

from .compile import CompiledProgram, evaluate_affine_tick
from .model import MAXIMUM_REPEAT_COUNT


def trigger_times(
    prog: CompiledProgram,
    channel: str,
    table: np.ndarray | None = None,
    *,
    cycles: int = 1,
) -> np.ndarray:
    """Return rising-edge ticks for one physical digital lane.

    This is an orchestration-side projection.  It never participates in the
    wire image or device session.
    """

    return np.asarray(
        _channel_edges(prog, (channel,), table, cycles)[0][0::2],
        dtype=np.uint64,
    )


def trigger_windows(
    prog: CompiledProgram,
    channel: str,
    table: np.ndarray | None = None,
    *,
    cycles: int = 1,
) -> tuple[tuple[int, int], ...]:
    """Return (rise, fall) tick pairs for one lane, over one finite run.

    The high time of each window is what an exposure IS, so a camera adapter
    that needs exposures asks the program rather than walking the mask table
    itself -- which is how a twin ends up with its own copy of the rising-edge
    rule, drifting from the one the board plays.
    """

    return trigger_windows_by_channel(prog, (channel,), table, cycles=cycles)[channel]


def trigger_windows_by_channel(
    prog: CompiledProgram,
    channels: Sequence[str],
    table: np.ndarray | None = None,
    *,
    cycles: int = 1,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return the exposure windows of several lanes from ONE walk of the run.

    A run is one timeline; which lanes an asker cares about does not change
    it, and the point table it is played over is shared by all of them.  The
    virtual world needs cooling, probe, trap and the camera for every shot.
    """

    names = tuple(channels)
    streams = _channel_edges(prog, names, table, cycles)
    return {name: _windows_of(edges) for name, edges in zip(names, streams)}


def trigger_edge_ticks(
    prog: CompiledProgram,
    channels: Sequence[str],
    table: np.ndarray | None = None,
    *,
    cycles: int = 1,
) -> dict[str, tuple[int, ...]]:
    """Return the tick of every edge each named lane plays.

    An edge is what a delay FIFO holds an entry for -- rise and fall alike --
    so a capacity check asks this.  It used to ask for exposure windows and
    then take them apart again: pairing every edge with its neighbour, then
    flattening the pairs and sorting them back into the order they were
    already in, to recover exactly the stream below.  Asking the exposure
    question also meant refusing a program for an exposure-shaped reason (a
    lane still high at the end) from inside a FIFO-capacity check.
    """

    names = tuple(channels)
    return dict(zip(names, _channel_edges(prog, names, table, cycles)))


def _windows_of(edges: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if len(edges) % 2:
        raise ValueError("program leaves a trigger channel high after a finite run")
    windows = tuple(zip(edges[0::2], edges[1::2]))
    if any(end <= start for start, end in windows):
        raise ValueError("program contains a non-positive trigger window")
    return windows


def run_duration_seconds(
    prog: CompiledProgram,
    table: np.ndarray | None = None,
    *,
    cycles: int = 1,
) -> float:
    """Return the exact finite playback duration for one program/table run."""

    return sum(
        total for _effective, _loop_start, _loop_end, _final, _span, total in (
            _point_timing(prog, point, index)
            for index, point in enumerate(_scan_points(prog, table, cycles))
        )
    ) / float(prog.clock_hz)


def _channel_edges(
    prog: CompiledProgram,
    channels: Sequence[str],
    table: np.ndarray | None,
    cycles: int,
) -> tuple[tuple[int, ...], ...]:
    """Project the tick of every edge each named lane plays, in playback order.

    LEVELS ARE NOT STORED.  An edge is a change, so a lane's levels
    alternate: the first edge raises it, the second lowers it, and the
    position in the stream already says which.  Carrying a bool beside every
    tick allocated a tuple per edge -- 3.9 million of them for one 200-point
    fire -- to repeat what counting says.

    THE SHAPE OF A POINT IS SHARED BY EVERY LANE.  ``_point_timing`` has no
    channel argument, and the table is cycled, so a scan of N rows has N
    shapes however many cycles it plays.  Derived per lane, a program with
    nine delayed lanes derived the same table nine times -- and nine is
    authorable by accident, because a delay is written per port and a
    NEGATIVE one is expressed by lifting every OTHER driven lane, so one
    ``-100 ns`` turns every driven lane into a delayed channel.  This runs
    inside fire(), before the board is strobed, with the operator waiting.
    """

    if not isinstance(prog, CompiledProgram):
        raise TypeError("prog must be CompiledProgram")
    physical_of = dict(prog.logical_digital_outputs)
    bits: list[int] = []
    for channel in channels:
        physical = physical_of.get(channel, channel)
        if physical not in prog.channels:
            raise ValueError(f"unknown channel {channel!r}")
        bit = prog.channels.index(physical)
        if prog.clk_enable & (1 << bit):
            raise ValueError("clock lanes do not have digital trigger results")
        bits.append(bit)

    points = _scan_points(prog, table, cycles)
    loop_count = prog.loop_count
    loop_start_index = prog.loop_start_index
    shapes: dict[tuple[int, ...], tuple] = {}

    streams: list[tuple[int, ...]] = []
    for bit in bits:
        delay = int(prog.channel_delays[bit])
        # One lookup per stop instead of a shift and a mask: the mask table
        # has 41 rows and the walk visits them hundreds of thousands of times.
        levels = tuple(bool((mask >> bit) & 1) for mask in prog.masks)
        terminal = levels[-1]
        ticks: list[int] = []
        previous = False
        run_offset = 0
        for point_index, point in enumerate(points):
            shape = shapes.get(point)
            if shape is None:
                effective, _loop_start, loop_end, final, span, total = _point_timing(
                    prog, point, point_index
                )
                prefix = tuple(range(loop_start_index))
                body = tuple(
                    index
                    for index in range(loop_start_index, len(effective))
                    if effective[index] < loop_end
                )
                extra = tuple(
                    index
                    for index in range(loop_start_index, len(effective))
                    if loop_end <= effective[index] < final
                )
                shape = (effective, prefix, body, extra, final, span, total)
                shapes[point] = shape
            effective, prefix, body, extra, final, span, total = shape

            for index in prefix:
                level = levels[index]
                if level != previous:
                    previous = level
                    ticks.append(run_offset + effective[index] + delay)
            for iteration in range(loop_count):
                offset = run_offset + iteration * span
                for index in body:
                    level = levels[index]
                    if level != previous:
                        previous = level
                        ticks.append(offset + effective[index] + delay)
            offset = run_offset + (loop_count - 1) * span
            for index in extra:
                level = levels[index]
                if level != previous:
                    previous = level
                    ticks.append(offset + effective[index] + delay)

            if terminal != previous:
                previous = terminal
                ticks.append(offset + final + delay)
            run_offset += total
        streams.append(tuple(ticks))

    return tuple(streams)


def _scan_points(
    prog: CompiledProgram,
    table: np.ndarray | None,
    cycles: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(prog, CompiledProgram):
        raise TypeError("prog must be CompiledProgram")
    if isinstance(cycles, bool) or not isinstance(cycles, Integral):
        raise TypeError("cycles must be an integer")
    cycles = int(cycles)
    if not 1 <= cycles <= MAXIMUM_REPEAT_COUNT:
        raise ValueError("cycles must be in the hardware range [1, 2^32-1]")
    if table is None:
        if prog.slot_count:
            raise ValueError("a slotted program requires an explicit value table")
        rows = ((),)
    else:
        array = np.asarray(table)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != prog.slot_count:
            raise ValueError("table width differs from the program slot count")
        raw_rows = tuple(tuple(value for value in row) for row in array.tolist())
        if not raw_rows:
            raise ValueError("table must contain at least one row")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for row in raw_rows
            for value in row
        ):
            raise TypeError("table values must be integers")
        rows = tuple(tuple(int(value) for value in row) for row in raw_rows)
    return tuple(rows[index % len(rows)] for index in range(cycles))


def _point_timing(
    prog: CompiledProgram,
    point: tuple[int, ...],
    point_index: int,
) -> tuple[tuple[int, ...], int, int, int, int, int]:
    effective = tuple(
        evaluate_affine_tick(base, coeffs, point, prog.scan_coeff_frac_bits)
        for base, coeffs in zip(prog.ticks, prog.tick_slot_coeffs)
    )
    loop_start = effective[prog.loop_start_index]
    loop_end = evaluate_affine_tick(
        prog.loop_end_tick,
        prog.loop_end_slot_coeffs,
        point,
        prog.scan_coeff_frac_bits,
    )
    final = effective[-1]
    if loop_end <= loop_start or loop_end > final:
        raise ValueError(f"point {point_index} has invalid loop metadata")
    span = loop_end - loop_start
    total = final + (prog.loop_count - 1) * span
    return effective, loop_start, loop_end, final, span, total


__all__ = [
    "run_duration_seconds",
    "trigger_times",
    "trigger_windows",
    "trigger_windows_by_channel",
    "trigger_edge_ticks",
]
