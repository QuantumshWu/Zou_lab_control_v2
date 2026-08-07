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

    if not isinstance(prog, CompiledProgram):
        raise TypeError("prog must be CompiledProgram")
    if prog.repeat_forever:
        raise ValueError("a forever program has no finite trigger-time result")
    if channel not in prog.channels:
        logical = dict(prog.logical_digital_outputs)
        channel = logical.get(channel, channel)
    if channel not in prog.channels:
        raise ValueError(f"unknown channel {channel!r}")
    bit = prog.channels.index(channel)
    if prog.clk_enable & (1 << bit):
        raise ValueError("clock lanes do not have digital trigger times")
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

    result: list[int] = []
    # A running total, accumulated once per point.  It used to be re-summed
    # over every earlier point inside the loop -- three affine evaluations per
    # term -- so the cost was quadratic in the number of scan points: 1.9 s at
    # 2 000, on the projection a window calls to say when the camera opens.
    run_offset = 0
    for point_index, point in enumerate(points):
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
        previous = 0

        def consume(indices, offset: int) -> None:
            nonlocal previous
            for index in indices:
                if effective[index] >= final:
                    continue
                current = (prog.masks[index] >> bit) & 1
                if current and not previous:
                    result.append(offset + effective[index] + prog.channel_delays[bit])
                previous = current

        prefix = tuple(index for index in range(prog.loop_start_index))
        body = tuple(
            index
            for index in range(prog.loop_start_index, len(prog.ticks))
            if effective[index] < loop_end
        )
        tail = tuple(
            index
            for index in range(prog.loop_start_index, len(prog.ticks))
            if loop_end <= effective[index] < final
        )
        consume(prefix, run_offset)
        consume(body, run_offset)
        span = loop_end - loop_start
        for iteration in range(1, prog.loop_count):
            consume(body, run_offset + iteration * span)
        consume(tail, run_offset + (prog.loop_count - 1) * span)
        # What this point costs, added once, so the next point starts after it.
        run_offset += final + (prog.loop_count - 1) * span

    return np.asarray(result, dtype=np.uint64)


def trigger_windows(
    prog: CompiledProgram,
    channel: str,
) -> tuple[tuple[int, int], ...]:
    """Return (rise, fall) tick pairs for one lane, over one finite run.

    The high time of each window is what an exposure IS, so a camera adapter
    that needs exposures asks the program rather than walking the mask table
    itself -- which is how a twin ends up with its own copy of the rising-edge
    rule, drifting from the one the board plays.
    """

    if not isinstance(prog, CompiledProgram):
        raise TypeError("prog must be CompiledProgram")
    if prog.repeat_forever:
        raise ValueError("a forever program has no finite trigger-window result")
    if channel not in prog.channels:
        logical = dict(prog.logical_digital_outputs)
        channel = logical.get(channel, channel)
    if channel not in prog.channels:
        raise ValueError(f"unknown channel {channel!r}")
    bit = prog.channels.index(channel)
    if prog.clk_enable & (1 << bit):
        raise ValueError("clock lanes do not have digital trigger windows")

    windows: list[tuple[int, int]] = []
    rise: int | None = None
    for index, mask in enumerate(prog.masks):
        high = bool((int(mask) >> bit) & 1)
        if high and rise is None:
            rise = int(prog.ticks[index])
        elif not high and rise is not None:
            windows.append((rise, int(prog.ticks[index])))
            rise = None
    if rise is not None:
        windows.append((rise, int(prog.ticks[-1])))
    if any(end <= start for start, end in windows):
        raise ValueError("program contains a non-positive trigger window")
    return tuple(windows)


__all__ = ["trigger_times", "trigger_windows"]
