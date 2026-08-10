"""Pure timing queries derived from a compiled program."""

from __future__ import annotations

import numpy as np

from .compile import CompiledProgram, evaluate_affine_tick


def trigger_times(
    prog: CompiledProgram,
    channel: str,
    table: np.ndarray | None = None,
) -> np.ndarray:
    """Return rising-edge ticks for one physical digital lane.

    This is an orchestration-side projection.  It never participates in the
    wire image or device session.
    """

    return np.asarray(
        [tick for tick, high in _channel_transitions(prog, channel, table) if high],
        dtype=np.uint64,
    )


def trigger_windows(
    prog: CompiledProgram,
    channel: str,
    table: np.ndarray | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return (rise, fall) tick pairs for one lane, over one finite run.

    The high time of each window is what an exposure IS, so a camera adapter
    that needs exposures asks the program rather than walking the mask table
    itself -- which is how a twin ends up with its own copy of the rising-edge
    rule, drifting from the one the board plays.
    """

    windows: list[tuple[int, int]] = []
    rise: int | None = None
    for tick, high in _channel_transitions(prog, channel, table):
        if high and rise is None:
            rise = tick
        elif not high and rise is not None:
            windows.append((rise, tick))
            rise = None
    if rise is not None:
        raise ValueError("program leaves a trigger channel high after a finite run")
    if any(end <= start for start, end in windows):
        raise ValueError("program contains a non-positive trigger window")
    return tuple(windows)


def run_duration_seconds(
    prog: CompiledProgram,
    table: np.ndarray | None = None,
) -> float:
    """Return the exact finite playback duration for one program/table run."""

    return sum(
        total for _effective, _loop_start, _loop_end, _final, _span, total in (
            _point_timing(prog, point, index)
            for index, point in enumerate(_scan_points(prog, table))
        )
    ) / float(prog.clock_hz)


def _channel_transitions(
    prog: CompiledProgram,
    channel: str,
    table: np.ndarray | None,
) -> tuple[tuple[int, bool], ...]:
    """Project the exact finite edge stream played by one digital channel."""

    if not isinstance(prog, CompiledProgram):
        raise TypeError("prog must be CompiledProgram")
    if prog.repeat_forever:
        raise ValueError("a forever program has no finite trigger result")
    physical = dict(prog.logical_digital_outputs).get(channel, channel)
    if physical not in prog.channels:
        raise ValueError(f"unknown channel {channel!r}")
    bit = prog.channels.index(physical)
    if prog.clk_enable & (1 << bit):
        raise ValueError("clock lanes do not have digital trigger results")

    points = _scan_points(prog, table)

    transitions: list[tuple[int, bool]] = []
    previous = False
    run_offset = 0
    delay = int(prog.channel_delays[bit])
    for point_index, point in enumerate(points):
        effective, loop_start, loop_end, final, span, total = _point_timing(
            prog, point, point_index
        )

        def consume(indices: tuple[int, ...], offset: int) -> None:
            nonlocal previous
            for index in indices:
                current = bool((prog.masks[index] >> bit) & 1)
                if current != previous:
                    transitions.append((offset + effective[index] + delay, current))
                    previous = current

        prefix = tuple(range(prog.loop_start_index))
        body = tuple(
            index
            for index in range(prog.loop_start_index, len(effective))
            if effective[index] < loop_end
        )
        tail = tuple(
            index
            for index in range(prog.loop_start_index, len(effective))
            if loop_end <= effective[index] < final
        )
        consume(prefix, run_offset)
        for iteration in range(prog.loop_count):
            consume(body, run_offset + iteration * span)
        consume(tail, run_offset + (prog.loop_count - 1) * span)

        terminal = bool((prog.masks[-1] >> bit) & 1)
        if terminal != previous:
            transitions.append(
                (run_offset + final + (prog.loop_count - 1) * span + delay, terminal)
            )
            previous = terminal
        run_offset += total

    return tuple(transitions)


def _scan_points(
    prog: CompiledProgram,
    table: np.ndarray | None,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(prog, CompiledProgram):
        raise TypeError("prog must be CompiledProgram")
    if prog.repeat_forever:
        raise ValueError("a forever program has no finite trigger result")
    if table is None:
        points = prog.scan_points or ((),)
    else:
        array = np.asarray(table)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != prog.slot_count:
            raise ValueError("table width differs from the program slot count")
        points = tuple(tuple(int(value) for value in row) for row in array)
    if any(len(point) != prog.slot_count for point in points):
        raise ValueError("table width differs from the program slot count")
    return points


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


__all__ = ["run_duration_seconds", "trigger_times", "trigger_windows"]
