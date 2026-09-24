"""How a loop table plays: the arithmetic the model, the compiler and the schedule share.

A loop table is ``(first_row, last_row, count)`` per bracket, outermost
first.  How long one Pulse lasts and which rows it enters in what order are
facts about that table alone.  The model asks them to say how long a Pulse
plays, the compiler to stamp a program's duration, the schedule to place
every edge -- so they live here, below all three, and there is one walk.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import chain


def bracket_iterations(loop_count: int, bodies: int | None) -> Iterable[int]:
    """Which replays of a Bracket to walk: every one, or a bounded first-and-last.

    A delay-FIFO check needs the TRUE elapsed time of a Pulse -- so the
    Pulses after it land where they really land -- but not every body of a
    Bracket that replays a billion times.  The bodies are identical, so once
    ``bodies`` of them have played a queue has either overflowed or repeats
    its state, and the last ``bodies`` see the loop out exactly the way the
    first saw it in; the ones between are the same again at ticks no window
    can tell apart from those.  Walking a SHORTENED loop instead moved every
    later Pulse earlier, and a body that changed no level at all -- which
    adds no entry to any queue -- was refused for crowding runs together
    that the board plays comfortably apart.  ``None`` walks every replay.
    """

    if bodies is None or loop_count <= 2 * bodies:
        return range(loop_count)
    return chain(range(bodies), range(loop_count - bodies, loop_count))


@dataclass(frozen=True)
class LoopNode:
    """One loop of the table with the loops it encloses, in row order."""

    start: int
    end: int
    count: int
    children: tuple["LoopNode", ...] = ()


def loop_tree(loops: Sequence[tuple[int, int, int]]) -> tuple[LoopNode, ...]:
    """The loop table as a forest, outermost loops at the top level.

    The table is stored outer-first (start ascending, end descending), which
    is exactly the order the board's walker pushes loops, so a stack rebuilds
    the nesting without searching.
    """

    roots: list[LoopNode] = []
    pending: list[tuple[int, int, int, list[LoopNode]]] = []

    def close(node: tuple[int, int, int, list[LoopNode]]) -> None:
        built = LoopNode(node[0], node[1], node[2], tuple(node[3]))
        if pending:
            pending[-1][3].append(built)
        else:
            roots.append(built)

    for start, end, count in loops:
        while pending and not (pending[-1][0] <= start and end <= pending[-1][1]):
            close(pending.pop())
        pending.append((start, end, count, []))
    while pending:
        close(pending.pop())
    return tuple(roots)


def loop_nesting_depth(loops: Sequence[tuple[int, int, int]]) -> int:
    """How many loops the board must hold on its stack at once."""

    return max(
        (
            1 + sum(
                other != index
                and loops[other][0] <= start
                and end <= loops[other][1]
                for other in range(len(loops))
            )
            for index, (start, end, _count) in enumerate(loops)
        ),
        default=0,
    )


def _loop_span(node: LoopNode, durations: Sequence[int]) -> int:
    """How many ticks one replay of a loop body lasts, inner loops fully played."""

    total = 0
    row = node.start
    for child in node.children:
        total += sum(durations[row:child.start]) + child.count * _loop_span(child, durations)
        row = child.end + 1
    return total + sum(durations[row:node.end + 1])


def frame_ticks(durations: Sequence[int], loops: Sequence[tuple[int, int, int]]) -> int:
    """How many ticks one complete Pulse of resolved row durations lasts."""

    return _loop_span(LoopNode(0, len(durations) - 1, 1, loop_tree(loops)), durations)


def frame_visits(
    durations: Sequence[int],
    loops: Sequence[tuple[int, int, int]],
    bracket_bodies: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Every row one Pulse enters, as ``(row, start tick)``, in the board's order."""

    visits: list[tuple[int, int]] = []
    _walk_rows(loop_tree(loops), 0, len(durations) - 1, 0, durations, bracket_bodies, visits)
    return tuple(visits)


def _walk_rows(
    nodes: Sequence[LoopNode],
    first_row: int,
    last_row: int,
    tick: int,
    durations: Sequence[int],
    bodies: int | None,
    visits: list[tuple[int, int]],
) -> int:
    row = first_row
    for node in nodes:
        for index in range(row, node.start):
            visits.append((index, tick))
            tick += durations[index]
        span = _loop_span(node, durations)
        previous = -1
        for iteration in bracket_iterations(node.count, bodies):
            tick += (iteration - previous - 1) * span
            tick = _walk_rows(node.children, node.start, node.end, tick, durations, bodies, visits)
            previous = iteration
        tick += (node.count - 1 - previous) * span
        row = node.end + 1
    for index in range(row, last_row + 1):
        visits.append((index, tick))
        tick += durations[index]
    return tick


__all__ = [
    "LoopNode",
    "bracket_iterations",
    "frame_ticks",
    "frame_visits",
    "loop_nesting_depth",
    "loop_tree",
]
