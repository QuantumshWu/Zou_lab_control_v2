"""Lower PulseSequence values to the frozen period-table program.

The board plays a PERIOD TABLE: one row per authored period holding how long
the row lasts (a tick count, or which scan slot supplies it), the TTL levels
it holds, and what each DAC bus does when the row is entered.  Brackets are a
separate loop table the board's row walker follows, so a bracket body is
stored once however many times it plays.  Nothing here is affine: a scan
point is the plain vector of slot values -- a duration in ticks, a DAC code --
that the rows read at run time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import math

from .canonical import canonical_digest
from .loops import (
    LoopNode,
    bracket_iterations,
    frame_ticks,
    frame_visits,
    loop_nesting_depth,
    loop_tree,
)
from .model import (
    FIELD_DAC,
    FIELD_DURATION,
    MAXIMUM_REPEAT_COUNT,
    MINIMUM_BRACKET_COUNT,
    PORT_CLOCK,
    PORT_DAC,
    PulseFieldRef,
    PulseSequence,
    exact_ticks,
)
from .wire import StreamerParams, build_fingerprint


COMPILER_ID = "zlc-pulse-native"
BUS_MODES = frozenset(("edge", "ramp"))


@dataclass(frozen=True)
class TargetBusDelay:
    bus_index: int
    delay_ticks: int

    def __post_init__(self) -> None:
        if isinstance(self.bus_index, bool) or not isinstance(self.bus_index, int) or self.bus_index < 0:
            raise ValueError("bus_index must be a non-negative integer")
        if isinstance(self.delay_ticks, bool) or not isinstance(self.delay_ticks, int) or self.delay_ticks < 0:
            raise ValueError("delay_ticks must be a non-negative integer")


@dataclass(frozen=True)
class TargetBusAction:
    """What one DAC bus does when one row is entered.

    ``edge`` takes the value at the row's first tick; ``ramp`` walks from the
    level the bus holds when the row is entered to the value over the row's
    whole duration.  ``value`` is the offset-binary code, unless
    ``value_select`` names a scan slot (``k`` = slot ``k-1``) that supplies it.
    """

    row: int
    bus_index: int
    bus_name: str
    mode: str
    value: int
    value_select: int = 0

    def __post_init__(self) -> None:
        for name in ("row", "bus_index", "value", "value_select"):
            item = getattr(self, name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.bus_name, str) or not self.bus_name:
            raise ValueError("bus_name must be non-empty text")
        if self.mode not in BUS_MODES:
            raise ValueError("bus action mode must be 'edge' or 'ramp'")


@dataclass(frozen=True)
class CompiledProgram:
    clock_hz: float
    target_abi_fingerprint: str
    geometry_fingerprint: int
    channels: tuple[str, ...]
    #: Per row: how many ticks it lasts when no slot supplies the duration.
    durations: tuple[int, ...]
    #: Per row: ``0`` plays ``durations[row]``; ``k`` plays scan slot ``k-1``.
    duration_slots: tuple[int, ...]
    #: Per row: the TTL levels held for the whole row (raw-lane bits).
    masks: tuple[int, ...]
    #: ``(first_row, last_row, count)`` per bracket, outermost first.
    loops: tuple[tuple[int, int, int], ...]
    duration_seconds: float
    slot_kinds: tuple[str, ...] = ()
    bus_names: tuple[str, ...] = ()
    bus_actions: tuple[TargetBusAction, ...] = ()
    bus_delays: tuple[TargetBusDelay, ...] = ()
    channel_delays: tuple[int, ...] = ()
    clk_enable: int = 0
    logical_digital_outputs: tuple[tuple[str, str], ...] = ()
    bus_safe_values: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.clock_hz, bool) or not isinstance(self.clock_hz, (int, float)):
            raise TypeError("clock_hz must be numeric")
        if not math.isfinite(float(self.clock_hz)) or self.clock_hz <= 0:
            raise ValueError("clock_hz must be positive and finite")
        if (
            isinstance(self.geometry_fingerprint, bool)
            or not isinstance(self.geometry_fingerprint, int)
            or not 0 <= self.geometry_fingerprint < (1 << 32)
        ):
            raise ValueError("geometry_fingerprint must be an unsigned 32-bit integer")
        channels = tuple(self.channels)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("program channels must be unique and non-empty")
        durations = tuple(int(value) for value in self.durations)
        duration_slots = tuple(int(value) for value in self.duration_slots)
        masks = tuple(int(value) for value in self.masks)
        rows = len(durations)
        if not rows or len(duration_slots) != rows or len(masks) != rows:
            raise ValueError("program rows must be non-empty and of equal lengths")
        if any(not 1 <= value <= MAXIMUM_REPEAT_COUNT for value in durations):
            raise ValueError("every row duration must be from 1 through 2^32-1 ticks")
        slot_kinds = tuple(self.slot_kinds)
        if any(kind not in (FIELD_DURATION, FIELD_DAC) for kind in slot_kinds):
            raise ValueError("program contains an unsupported slot kind")
        for selector in duration_slots:
            if selector < 0 or selector > len(slot_kinds):
                raise ValueError("a row names a duration slot the program does not have")
            if selector and slot_kinds[selector - 1] != FIELD_DURATION:
                raise ValueError("a row duration can only come from a duration slot")
        if any(value < 0 for value in masks):
            raise ValueError("row masks must be non-negative")
        loops = tuple(
            (int(start), int(end), int(count)) for start, end, count in self.loops
        )
        for start, end, count in loops:
            if not 0 <= start <= end < rows:
                raise ValueError("a loop lies outside the row table")
            if not MINIMUM_BRACKET_COUNT <= count <= MAXIMUM_REPEAT_COUNT:
                raise ValueError(
                    f"a loop count must be from {MINIMUM_BRACKET_COUNT} through 2^32-1"
                )
        for index, (start, end, _count) in enumerate(loops):
            for other_start, other_end, _other in loops[index + 1:]:
                nested = (
                    (start <= other_start and other_end <= end)
                    or (other_start <= start and end <= other_end)
                )
                disjoint = end < other_start or other_end < start
                if not (nested or disjoint):
                    raise ValueError("loops overlap without one lying inside the other")
        if loops != tuple(sorted(loops, key=lambda loop: (loop[0], -loop[1]))):
            raise ValueError("loops must be stored outermost first")
        channel_delays = tuple(self.channel_delays) or (0,) * len(channels)
        if len(channel_delays) != len(channels):
            raise ValueError("channel delay vector must match channels")
        bus_names = tuple(self.bus_names)
        actions = tuple(self.bus_actions)
        if any(not isinstance(action, TargetBusAction) for action in actions):
            raise TypeError("bus_actions must contain TargetBusAction values")
        placed: set[tuple[int, int]] = set()
        for action in actions:
            if action.row >= rows:
                raise ValueError("a bus action names a row outside the table")
            if action.bus_index >= len(bus_names) or bus_names[action.bus_index] != action.bus_name:
                raise ValueError("a bus action names a bus the program does not have")
            if action.value_select > len(slot_kinds):
                raise ValueError("a bus action names a slot the program does not have")
            if action.value_select and slot_kinds[action.value_select - 1] != FIELD_DAC:
                raise ValueError("a bus value can only come from a DAC slot")
            key = (action.row, action.bus_index)
            if key in placed:
                raise ValueError("a row has at most one action per DAC bus")
            placed.add(key)
        safe_values = tuple(int(value) for value in self.bus_safe_values)
        if len(safe_values) != len(bus_names):
            raise ValueError("one safe value per DAC bus")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(float(self.duration_seconds))
            or self.duration_seconds < 0
        ):
            raise ValueError("duration_seconds must be a finite non-negative number")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "duration_slots", duration_slots)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "loops", loops)
        object.__setattr__(self, "slot_kinds", slot_kinds)
        object.__setattr__(self, "bus_actions", actions)
        object.__setattr__(self, "bus_delays", tuple(self.bus_delays))
        object.__setattr__(self, "bus_names", bus_names)
        object.__setattr__(self, "logical_digital_outputs", tuple(tuple(item) for item in self.logical_digital_outputs))
        object.__setattr__(self, "bus_safe_values", safe_values)
        object.__setattr__(self, "channel_delays", tuple(int(value) for value in channel_delays))

    @property
    def slot_count(self) -> int:
        return len(self.slot_kinds)

    @property
    def row_count(self) -> int:
        return len(self.durations)

    @property
    def loop_depth(self) -> int:
        """How many loops the board holds on its stack at once for this program."""

        return loop_nesting_depth(self.loops)

    def resolved_durations(self, point: Sequence[int] = ()) -> tuple[int, ...]:
        """How long each row lasts for one scan point, in ticks."""

        values = tuple(int(value) for value in point)
        if len(values) != self.slot_count:
            raise ValueError(
                f"a scan point has one value per slot: {self.slot_count} "
                f"slot(s), {len(values)} value(s)"
            )
        return tuple(
            values[selector - 1] if selector else literal
            for literal, selector in zip(self.durations, self.duration_slots, strict=True)
        )

    def resolved_bus_value(self, action: TargetBusAction, point: Sequence[int] = ()) -> int:
        """The offset-binary code one action drives for one scan point."""

        if action.value_select:
            return int(point[action.value_select - 1])
        return int(action.value)

    def frame_visits(
        self,
        point: Sequence[int] = (),
        *,
        bracket_bodies: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Every row the board enters in one Pulse, as ``(row, start tick)``.

        Loops are expanded in the board's own order.  ``bracket_bodies``
        walks only the first and last that many replays of every loop, at
        their true ticks (see :func:`bracket_iterations`), which is what
        bounds a capacity check over a loop that replays a body a billion
        times.
        """

        return frame_visits(self.resolved_durations(point), self.loops, bracket_bodies)

    def frame_ticks(self, point: Sequence[int] = ()) -> int:
        """How many ticks one complete Pulse lasts for one scan point."""

        return frame_ticks(self.resolved_durations(point), self.loops)

    @property
    def digest(self) -> str:
        """One short name for exactly what this program plays.

        On the program rather than in a function beside it, so anyone holding
        one can name it -- including a package that must not import this one.
        A board answers with the digest of what it applied and a window digests
        what it would compile to now; unequal means the board is playing
        something else, and neither side has to remember anything.

        Of the compiled program, deliberately, not of the document it came
        from: a renamed period changes the document and not one row the board
        will play, and reporting that as stale teaches an operator to ignore
        the light.

        Full length, like every other fingerprint in this project: one format
        for one concept, so a client can validate it without knowing which
        package minted it.
        """

        return canonical_digest(self)


def _slot_index(sequence: PulseSequence) -> dict[PulseFieldRef, int]:
    return {slot.field_ref: index for index, slot in enumerate(sequence.scan_bindings)}


def nominal_slot_values(sequence: PulseSequence) -> tuple[int, ...]:
    """The authored value of every scan slot, as the wire carries it.

    A duration slot holds the period's tick count; a DAC slot holds the
    step's offset-binary code.  This is the point a scan-bound pulse plays
    before any table has been authored, and what previews are drawn from.
    """

    values: list[int] = []
    for slot in sequence.scan_bindings:
        ref = slot.field_ref
        if ref.kind == FIELD_DURATION:
            period = sequence.period_by_id[ref.period_id]
            values.append(exact_ticks(
                period.duration, period.unit, sequence.time_step_ns,
                f"period {period.period_id} duration",
            ))
        elif ref.kind == FIELD_DAC:
            period = sequence.period_by_id[ref.period_id]
            step = next((item for item in period.analog_steps if item.port == ref.port), None)
            if step is None:
                # The model admits a binding whose step is gone -- taking a step
                # away is a legal intermediate state of an edit, pruned afterwards
                # -- so the compiler is where a pulse read as a whole says which
                # binding has nothing to bind.
                raise ValueError(
                    f"scan slot {slot.field_id!r} names the DAC field {ref.port!r} of "
                    f"period {ref.period_id!r}, which has no step on that port"
                )
            port = sequence.target.by_key[ref.port]
            values.append(int(step.value - port.signed_range[0]))
        else:
            raise ValueError(f"unsupported scan slot kind {ref.kind!r}")
    return tuple(values)


def analog_levels(sequence: PulseSequence) -> dict[str, tuple[tuple[int, int], ...]]:
    """The level each DAC port holds from every tick it changes, over one pass.

    ``{port key: ((tick, value), ...)}`` on the pulse's own tick grid, in the
    signed values the model authors, opening at ``(0, 0)`` -- the safe level
    a bus rests at before its first step -- and listing each tick the level
    changes.  An ``edge`` step takes its value at the start of its period.  A
    ``ramp`` walks from the level carried into the period to the step's value
    at the period's end exactly as the engine does, ``start ± floor(k·|delta|
    / span)`` on the k-th tick, so the entries are the staircase of codes the
    pin plays: at most ``min(|delta|, span)`` of them, never one per tick of a
    long gentle ramp.  Two changes on one tick are one entry, the later one.

    Published for previews.  One built from ``step.value`` alone drew the
    target level from the period start in both modes, and two pulses the
    board plays differently were one picture.
    """

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    boundaries = [0]
    for period in sequence.periods:
        boundaries.append(boundaries[-1] + exact_ticks(
            period.duration,
            period.unit,
            sequence.time_step_ns,
            f"period {period.period_id} duration",
        ))
    levels: dict[str, tuple[tuple[int, int], ...]] = {}
    for port in sequence.target.ports:
        if port.kind != PORT_DAC:
            continue
        points: list[tuple[int, int]] = [(0, 0)]

        def change(tick: int, value: int) -> None:
            if points[-1][0] == tick:
                points[-1] = (tick, value)
            elif points[-1][1] != value:
                points.append((tick, value))

        held = 0
        for index, period in enumerate(sequence.periods):
            step = next((item for item in period.analog_steps if item.port == port.key), None)
            if step is None:
                continue
            start, stop = boundaries[index], boundaries[index + 1]
            target = int(step.value)
            if step.mode == "edge":
                change(start, target)
            else:
                span = stop - start
                distance = abs(target - held)
                direction = 1 if target >= held else -1
                if distance <= span:
                    for level in range(1, distance + 1):
                        change(start - (-level * span // distance), held + direction * level)
                else:
                    for k in range(1, span + 1):
                        change(start + k, held + direction * (k * distance // span))
            held = target
        levels[port.key] = tuple(points)
    return levels


def _delay_values(sequence: PulseSequence) -> tuple[tuple[int, ...], tuple[TargetBusDelay, ...]]:
    lane_index = {lane: index for index, lane in enumerate(sequence.target.raw_lanes)}
    raw_lane: dict[int, int] = {}
    raw_bus: dict[int, int] = {}
    driven_lanes = {
        index
        for period in sequence.periods
        for index, state in enumerate(period.states)
        if state
    }
    driven_buses = {
        int(sequence.target.by_key[step.port].bus_index)
        for period in sequence.periods
        for step in period.analog_steps
    }
    for delay in sequence.delays:
        ticks = exact_ticks(delay.value, delay.unit, sequence.time_step_ns, "output delay", minimum=None)
        port = sequence.target.by_key[delay.port]
        if port.kind == PORT_DAC:
            raw_bus[int(port.bus_index)] = ticks
        else:
            for lane in port.lanes:
                raw_lane[lane_index[lane]] = ticks
    for index in driven_lanes:
        raw_lane.setdefault(index, 0)
    for bus_index in driven_buses:
        raw_bus.setdefault(bus_index, 0)
    values = list(raw_lane.values()) + list(raw_bus.values())
    shift = max(0, -min(values)) if values else 0
    channels = [0] * len(sequence.target.raw_lanes)
    for index, value in raw_lane.items():
        channels[index] = value + shift
    buses = tuple(TargetBusDelay(index, value + shift) for index, value in sorted(raw_bus.items()) if value + shift)
    return tuple(channels), buses


def compile_sequence(
    sequence: PulseSequence,
    geom: StreamerParams,
    clock_hz: float,
) -> CompiledProgram:
    """Compile once; slot rows are data written later by the device API."""

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    sequence.require_nonempty_brackets()
    if sequence.api_bindings:
        declared = tuple(
            parameter.field_id for parameter in sequence.api_bindings
        )
        raise ValueError(
            f"pulse API parameters must be resolved before compile: {declared}"
        )
    params = geom
    if not isinstance(params, StreamerParams):
        raise TypeError("geom must be StreamerParams")
    if len(sequence.target.raw_lanes) > params.channel_count:
        raise ValueError("sequence has more lanes than the streamer geometry")
    if sum(port.kind == PORT_DAC for port in sequence.target.ports) > params.bus_count:
        raise ValueError("sequence has more DAC buses than the streamer geometry")
    if any(port.width > params.bus_width for port in sequence.target.ports if port.kind == PORT_DAC):
        raise ValueError("a DAC port is wider than the streamer geometry")
    if isinstance(clock_hz, bool) or not isinstance(clock_hz, (int, float)):
        raise TypeError("clock_hz must be numeric")
    clock_hz = float(clock_hz)
    if not math.isfinite(clock_hz) or clock_hz <= 0:
        raise ValueError("clock_hz must be positive and finite")
    if Fraction(str(sequence.time_step_ns)) * Fraction(str(clock_hz)) != 1_000_000_000:
        raise ValueError("sequence time_step_ns does not match the compiler clock_hz")
    if sequence.slot_count > params.num_slots:
        raise ValueError(
            f"sequence binds {sequence.slot_count} scan slot(s) but the streamer "
            f"geometry holds {params.num_slots}"
        )
    if len(sequence.periods) > params.max_rows:
        raise ValueError(
            f"sequence has {len(sequence.periods)} periods but the streamer geometry "
            f"holds {params.max_rows} rows"
        )
    if len(sequence.brackets) > params.max_loops:
        raise ValueError(
            f"sequence has {len(sequence.brackets)} brackets but the streamer geometry "
            f"holds {params.max_loops} loops"
        )
    depth = max(sequence.bracket_depths, default=0)
    if depth > params.loop_depth:
        raise ValueError(
            f"brackets nest {depth} deep but the streamer geometry holds "
            f"{params.loop_depth} levels"
        )
    binding = _slot_index(sequence)
    nominal = nominal_slot_values(sequence)
    lane_index = {lane: index for index, lane in enumerate(sequence.target.raw_lanes)}
    clk_enable = 0
    for port in sequence.target.ports:
        if port.kind == PORT_CLOCK:
            clk_enable |= 1 << lane_index[port.lanes[0]]
    buses = sorted((port for port in sequence.target.ports if port.kind == PORT_DAC), key=lambda port: port.bus_index)
    bus_names = tuple(port.key for port in buses)
    durations: list[int] = []
    duration_slots: list[int] = []
    masks: list[int] = []
    actions: list[TargetBusAction] = []
    for row, period in enumerate(sequence.periods):
        durations.append(exact_ticks(
            period.duration, period.unit, sequence.time_step_ns,
            f"period {period.period_id} duration",
        ))
        selector = binding.get(PulseFieldRef(FIELD_DURATION, period.period_id))
        duration_slots.append(0 if selector is None else selector + 1)
        mask = 0
        for index, state in enumerate(period.states):
            if state:
                mask |= 1 << index
        masks.append(mask & ~clk_enable)
        for port in buses:
            step = next((item for item in period.analog_steps if item.port == port.key), None)
            if step is None:
                continue
            selector = binding.get(PulseFieldRef(FIELD_DAC, period.period_id, port.key))
            actions.append(TargetBusAction(
                row,
                int(port.bus_index),
                port.key,
                step.mode,
                0 if selector is not None else step.value - port.signed_range[0],
                0 if selector is None else selector + 1,
            ))
    loops = sequence.loops
    channel_delays, bus_delays = _delay_values(sequence)
    logical = tuple(sorted(
        (port.key, port.lanes[0])
        for port in sequence.target.ports
        if port.kind == "digital"
    ))
    nominal_durations = tuple(
        nominal[selector - 1] if selector else literal
        for literal, selector in zip(durations, duration_slots, strict=True)
    )
    return CompiledProgram(
        clock_hz=clock_hz,
        target_abi_fingerprint=sequence.target.abi_fingerprint,
        geometry_fingerprint=build_fingerprint(params),
        channels=sequence.target.raw_lanes,
        durations=tuple(durations),
        duration_slots=tuple(duration_slots),
        masks=tuple(masks),
        loops=loops,
        duration_seconds=frame_ticks(nominal_durations, loops) / clock_hz,
        slot_kinds=tuple(slot.kind for slot in sequence.scan_bindings),
        bus_names=bus_names,
        bus_actions=tuple(actions),
        bus_delays=bus_delays,
        channel_delays=channel_delays,
        clk_enable=clk_enable,
        logical_digital_outputs=logical,
        bus_safe_values=tuple(port.safe_value for port in buses),
    )


__all__ = [
    "COMPILER_ID",
    "CompiledProgram",
    "LoopNode",
    "TargetBusAction",
    "TargetBusDelay",
    "bracket_iterations",
    "compile_sequence",
    "frame_ticks",
    "frame_visits",
    "loop_nesting_depth",
    "loop_tree",
    "nominal_slot_values",
]
