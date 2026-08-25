"""Lower PulseSequence values to the frozen edge-table program."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math

from .canonical import canonical_digest
from .model import (
    FIELD_DAC,
    FIELD_DURATION,
    MAXIMUM_REPEAT_COUNT,
    PORT_CLOCK,
    PORT_DAC,
    PulseFieldRef,
    PulseSequence,
    PulseSlot,
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
class TargetBusSegment:
    bus_index: int
    bus_name: str
    start_tick: int
    stop_tick: int
    start_value: int
    stop_value: int
    mode: str
    value_select: int
    stop_value_select: int
    start_tick_coeffs: tuple[int, ...] = ()
    stop_tick_coeffs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.bus_index, bool) or not isinstance(self.bus_index, int) or self.bus_index < 0:
            raise ValueError("bus_index must be a non-negative integer")
        if not isinstance(self.bus_name, str) or not self.bus_name:
            raise ValueError("bus_name must be non-empty text")
        for name in ("start_tick", "stop_tick", "start_value", "stop_value", "value_select", "stop_value_select"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.start_tick < 0 or self.stop_tick < self.start_tick:
            raise ValueError("bus segment ticks are out of order")
        if self.mode not in BUS_MODES:
            raise ValueError("bus segment mode must be 'edge' or 'ramp'")
        object.__setattr__(self, "start_tick_coeffs", tuple(int(value) for value in self.start_tick_coeffs))
        object.__setattr__(self, "stop_tick_coeffs", tuple(int(value) for value in self.stop_tick_coeffs))


@dataclass(frozen=True)
class CompiledProgram:
    clock_hz: float
    target_abi_fingerprint: str
    geometry_fingerprint: int
    channels: tuple[str, ...]
    ticks: tuple[int, ...]
    masks: tuple[int, ...]
    duration_seconds: float
    loop_start_index: int
    loop_end_tick: int
    loop_count: int
    slot_kinds: tuple[str, ...] = ()
    loop_end_slot_coeffs: tuple[int, ...] = ()
    tick_slot_coeffs: tuple[tuple[int, ...], ...] = ()
    scan_coeff_frac_bits: int = 0
    bus_names: tuple[str, ...] = ()
    bus_segments: tuple[TargetBusSegment, ...] = ()
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
        ticks = tuple(int(value) for value in self.ticks)
        masks = tuple(int(value) for value in self.masks)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("program channels must be unique and non-empty")
        if not ticks or len(ticks) != len(masks) or ticks[0] != 0:
            raise ValueError("program edge rows must start at zero and have equal lengths")
        if masks[-1] != 0:
            raise ValueError("program must finish at the all-low mask")
        slot_kinds = tuple(self.slot_kinds)
        if any(right < left for left, right in zip(ticks, ticks[1:])):
            raise ValueError("program edge tick bases must be non-decreasing")
        if not slot_kinds and any(right <= left for left, right in zip(ticks, ticks[1:])):
            raise ValueError("static program edge ticks must be strictly increasing")
        if any(kind not in (FIELD_DURATION, FIELD_DAC) for kind in slot_kinds):
            raise ValueError("program contains an unsupported slot kind")
        slot_count = len(slot_kinds)
        coeffs = tuple(tuple(int(value) for value in row) for row in self.tick_slot_coeffs)
        if len(coeffs) != len(ticks) or any(len(row) != slot_count for row in coeffs):
            raise ValueError("program coefficient matrix has the wrong shape")
        loop_coeffs = tuple(int(value) for value in self.loop_end_slot_coeffs)
        if len(loop_coeffs) != slot_count:
            raise ValueError("program loop coefficient width differs from slot count")
        channel_delays = tuple(self.channel_delays) or (0,) * len(channels)
        if len(channel_delays) != len(channels):
            raise ValueError("channel delay vector must match channels")
        if (
            isinstance(self.loop_start_index, bool)
            or not isinstance(self.loop_start_index, int)
            or self.loop_start_index < 0
            or self.loop_start_index >= len(ticks)
        ):
            raise ValueError("loop_start_index is outside the edge table")
        if (
            isinstance(self.loop_count, bool)
            or not isinstance(self.loop_count, int)
            or not 1 <= self.loop_count <= MAXIMUM_REPEAT_COUNT
            or self.loop_end_tick <= ticks[self.loop_start_index]
        ):
            raise ValueError("program loop metadata is invalid")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "ticks", ticks)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "slot_kinds", slot_kinds)
        object.__setattr__(self, "tick_slot_coeffs", coeffs)
        object.__setattr__(self, "loop_end_slot_coeffs", loop_coeffs)
        object.__setattr__(self, "bus_segments", tuple(self.bus_segments))
        object.__setattr__(self, "bus_delays", tuple(self.bus_delays))
        object.__setattr__(self, "channel_delays", tuple(int(value) for value in channel_delays))
        # Prove the derived scan-unit metadata now, while malformed programs
        # can still be refused at their construction boundary.
        self._duration_tick_scales()

    @property
    def slot_count(self) -> int:
        return len(self.slot_kinds)

    def _duration_tick_scales(self) -> tuple[int, ...]:
        quantum = 1 << int(self.scan_coeff_frac_bits)
        result: list[int] = []
        for index, kind in enumerate(self.slot_kinds):
            if kind == FIELD_DAC:
                result.append(1)
                continue
            coefficients = {
                row[index]
                for row in self.tick_slot_coeffs
                if row[index]
            }
            if self.loop_end_slot_coeffs[index]:
                coefficients.add(self.loop_end_slot_coeffs[index])
            if (
                not coefficients
                or any(value <= 0 or value % quantum for value in coefficients)
                or len({value // quantum for value in coefficients}) != 1
            ):
                raise ValueError(
                    "duration slot coefficients do not describe one tick scale"
                )
            result.append(next(iter(coefficients)) // quantum)
        return tuple(result)

    @property
    def slot_tick_scales(self) -> tuple[int, ...]:
        """Wire tick quanta, derived from the affine coefficients themselves."""

        return self._duration_tick_scales()

    @property
    def digest(self) -> str:
        """One short name for exactly what this program plays.

        On the program rather than in a function beside it, so anyone holding
        one can name it -- including a package that must not import this one.
        A board answers with the digest of what it applied and a window digests
        what it would compile to now; unequal means the board is playing
        something else, and neither side has to remember anything.

        Of the compiled program, deliberately, not of the document it came
        from: a renamed period changes the document and not one edge the board
        will play, and reporting that as stale teaches an operator to ignore
        the light.

        Full length, like every other fingerprint in this project: one format
        for one concept, so a client can validate it without knowing which
        package minted it.
        """

        return canonical_digest(self)

def _resolve_slot_operand_width() -> int:
    from .wire import load_streamer_config  # noqa: PLC0415 -- config, not a cycle

    return int(load_streamer_config().get("slot_mul_width", 25))


#: How many signed bits the board's affine multiplier takes of a slot value.
#:
#: Resolved once, at import, exactly as the RTL cycle model resolves it -- and
#: for two reasons.  It is read once per slot term of every affine tick, so a
#: disk read there put a stat and a JSON parse inside the compiler's inner
#: loop.  And the config search consults the working directory, so resolving it
#: per call would let a `cd` change the arithmetic mid-session.
SLOT_OPERAND_WIDTH = _resolve_slot_operand_width()


def slot_operand_width() -> int:
    """How many signed bits the board's affine multiplier takes of a slot value."""

    return SLOT_OPERAND_WIDTH


def maximum_duration_tick_scale(params: StreamerParams) -> int:
    """Largest whole-tick duration coefficient the deployed Q format holds."""

    if not isinstance(params, StreamerParams):
        raise TypeError("params must be StreamerParams")
    return ((1 << (params.coeff_width - 1)) - 1) >> params.coeff_frac_bits


def narrow_slot_operand(value: int) -> int:
    """One slot value as the board's multiplier will see it.

    The RTL takes ``$signed(slots[.. +: SLOT_MUL_WIDTH])`` -- the low bits,
    signed.  A value that does not fit wraps, and the board then plays an edge
    at a tick the host never predicted.
    """

    width = SLOT_OPERAND_WIDTH
    mask = (1 << width) - 1
    narrowed = int(value) & mask
    if narrowed & (1 << (width - 1)):
        narrowed -= 1 << width
    return narrowed


def evaluate_affine_tick(base: int, coefficients: Sequence[int], point: Sequence[int], frac_bits: int) -> int:
    """The tick one edge lands on for one scan point, as the BOARD computes it.

    The slot operand is narrowed the way the RTL narrows it.  Duration slots
    carry signed tick deltas around a full-width nominal base; DAC slots carry
    their offset-binary code.  Values outside the multiplier width are refused
    before this function is used for a hardware application.
    """

    return int(base) + (
        sum(
            int(coefficient) * narrow_slot_operand(value)
            for coefficient, value in zip(coefficients, point)
        )
        >> int(frac_bits)
    )


def _slot_index(sequence: PulseSequence) -> dict[PulseFieldRef, int]:
    return {slot.field_ref: index for index, slot in enumerate(sequence.slots)}


def _default_slot_value(sequence: PulseSequence, slot: PulseSlot) -> int:
    ref = slot.field_ref
    if ref.kind == FIELD_DURATION:
        # Duration slots are signed deltas around the period's full-width base.
        return 0
    if ref.kind == FIELD_DAC:
        period = sequence.period_by_id[ref.period_id]
        step = next(item for item in period.analog_steps if item.port == ref.port)
        port = sequence.target.by_key[ref.port]
        return int(step.value - port.signed_range[0])
    raise ValueError(f"unsupported scan slot kind {ref.kind!r}")


def _nominal_slot_row(sequence: PulseSequence) -> tuple[int, ...]:
    """The authored values used only to validate the compiled affine form."""

    return tuple(_default_slot_value(sequence, slot) for slot in sequence.slots)


def _period_starts(
    sequence: PulseSequence,
    binding: Mapping[PulseFieldRef, int],
    frac_bits: int,
    slot_tick_scales: Sequence[int],
) -> list[tuple[int, tuple[int, ...]]]:
    coefficient_scale = 1 << frac_bits
    zeros = tuple(0 for _ in sequence.slots)
    starts: list[tuple[int, tuple[int, ...]]] = [(0, zeros)]
    for period in sequence.periods:
        selector = binding.get(PulseFieldRef(FIELD_DURATION, period.period_id))
        nominal = exact_ticks(
            period.duration,
            period.unit,
            sequence.time_step_ns,
            "period duration",
        )
        if selector is None:
            base = nominal
            coeff = zeros
        else:
            base = nominal
            tick_scale = int(slot_tick_scales[selector])
            coeff = tuple(
                coefficient_scale * tick_scale if index == selector else 0
                for index in range(sequence.slot_count)
            )
        starts.append((
            starts[-1][0] + base,
            tuple(left + right for left, right in zip(starts[-1][1], coeff)),
        ))
    return starts


def _effective_rows(
    sequence: PulseSequence,
    starts: Sequence[tuple[int, tuple[int, ...]]],
    frac_bits: int,
    reference: Sequence[int],
) -> tuple[list[int], list[int], list[tuple[int, ...]]]:
    lane_bits = {lane: index for index, lane in enumerate(sequence.target.raw_lanes)}
    events: dict[tuple[int, tuple[int, ...]], list[tuple[str, int]]] = {}
    for lane_index, lane in enumerate(sequence.target.raw_lanes):
        owner = next(port for port in sequence.target.ports if lane in port.lanes)
        if owner.kind != "digital":
            continue
        active: tuple[int, tuple[int, ...]] | None = None
        for period_index, period in enumerate(sequence.periods):
            if period.states[lane_index] and active is None:
                active = starts[period_index]
            elif not period.states[lane_index] and active is not None:
                events.setdefault(active, []).append((lane, 1))
                events.setdefault(starts[period_index], []).append((lane, 0))
                active = None
        if active is not None:
            events.setdefault(active, []).append((lane, 1))
            events.setdefault(starts[-1], []).append((lane, 0))
    zeros = (0, tuple(0 for _ in sequence.slots))
    events.setdefault(zeros, [])
    events.setdefault(starts[-1], [])
    if sequence.repeat is not None and sequence.repeat.count > 1:
        start_index = next(index for index, period in enumerate(sequence.periods) if period.period_id == sequence.repeat.start_period_id)
        events.setdefault(starts[start_index], [])
    ordered = sorted(
        events,
        key=lambda expression: (
            evaluate_affine_tick(expression[0], expression[1], reference, frac_bits),
            expression,
        ),
    )
    reference_ticks = [evaluate_affine_tick(base, coeff, reference, frac_bits) for base, coeff in ordered]
    if reference_ticks[0] != 0 or any(right <= left for left, right in zip(reference_ticks, reference_ticks[1:])):
        raise ValueError("slot values make edge rows collide or move before time zero")
    masks: list[int] = []
    ticks: list[int] = []
    coeffs: list[tuple[int, ...]] = []
    current = 0
    for expression in ordered:
        for lane, state in events[expression]:
            bit = 1 << lane_bits[lane]
            current = current | bit if state else current & ~bit
        ticks.append(expression[0])
        masks.append(current)
        coeffs.append(expression[1])
    if masks[-1] != 0:
        raise ValueError("sequence must end with all digital outputs low")
    return ticks, masks, coeffs


def _bus_segments(
    sequence: PulseSequence,
    starts: Sequence[tuple[int, tuple[int, ...]]],
    binding: Mapping[PulseFieldRef, int],
) -> tuple[tuple[str, ...], tuple[TargetBusSegment, ...]]:
    buses = sorted((port for port in sequence.target.ports if port.kind == PORT_DAC), key=lambda port: port.bus_index)
    names = tuple(port.key for port in buses)
    segments: list[TargetBusSegment] = []
    for port in buses:
        for period_index, period in enumerate(sequence.periods):
            action = next((item for item in period.analog_steps if item.port == port.key), None)
            if action is None:
                continue
            ref = PulseFieldRef(FIELD_DAC, period.period_id, port.key)
            selector = binding.get(ref)
            if selector is None:
                code = action.value - port.signed_range[0]
                start_value = stop_value = code
                value_select = stop_select = 0
            else:
                start_value = stop_value = 0
                value_select = stop_select = selector + 1
            start_tick, start_coeff = starts[period_index]
            stop_tick, stop_coeff = starts[period_index + 1]
            if action.mode == "edge":
                stop_tick, stop_coeff = start_tick, start_coeff
                stop_value, stop_select = start_value, value_select
            else:
                start_value = 0
                value_select = 0
            segments.append(TargetBusSegment(
                int(port.bus_index),
                port.key,
                start_tick,
                stop_tick,
                start_value,
                stop_value,
                action.mode,
                value_select,
                stop_select,
                tuple(start_coeff),
                tuple(stop_coeff),
            ))
    return names, tuple(segments)


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
    *,
    slot_tick_scales: Sequence[int] | None = None,
) -> CompiledProgram:
    """Compile once; slot rows are data written later by the device API."""

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    if sequence.api_parameters:
        declared = tuple(
            parameter.parameter_id for parameter in sequence.api_parameters
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
    frac_bits = params.coeff_frac_bits if sequence.slots else 0
    selected_slot_scales = (
        (1,) * sequence.slot_count
        if slot_tick_scales is None
        else tuple(slot_tick_scales)
    )
    if (
        len(selected_slot_scales) != sequence.slot_count
        or any(type(value) is not int or value < 1 for value in selected_slot_scales)
    ):
        raise ValueError("slot_tick_scales must contain one positive integer per slot")
    if any(
        slot.kind != FIELD_DURATION and scale != 1
        for slot, scale in zip(sequence.slots, selected_slot_scales, strict=True)
    ):
        raise ValueError("DAC slot tick scale must remain 1")
    maximum_tick_scale = maximum_duration_tick_scale(params) if sequence.slots else 1
    if any(value > maximum_tick_scale for value in selected_slot_scales):
        raise ValueError(
            f"slot tick scale exceeds the {params.coeff_width}-bit Q{frac_bits} "
            f"coefficient range ({maximum_tick_scale})"
        )
    binding = _slot_index(sequence)
    starts = _period_starts(
        sequence,
        binding,
        frac_bits,
        selected_slot_scales,
    )
    reference = _nominal_slot_row(sequence)
    ticks, masks, coeffs = _effective_rows(sequence, starts, frac_bits, reference)
    clk_enable = 0
    lane_index = {lane: index for index, lane in enumerate(sequence.target.raw_lanes)}
    for port in sequence.target.ports:
        if port.kind == PORT_CLOCK:
            clk_enable |= 1 << lane_index[port.lanes[0]]
    masks = [mask & ~clk_enable for mask in masks]
    bus_names, bus_segments = _bus_segments(sequence, starts, binding)
    channel_delays, bus_delays = _delay_values(sequence)
    repeat_start_index = 0
    loop_end_tick, loop_end_coeffs = starts[-1]
    loop_count = 1
    if sequence.repeat is not None:
        start_period = next(index for index, period in enumerate(sequence.periods) if period.period_id == sequence.repeat.start_period_id)
        end_period = next(index for index, period in enumerate(sequence.periods) if period.period_id == sequence.repeat.end_period_id)
        start_expr = starts[start_period]
        repeat_matches = [index for index, row in enumerate(zip(ticks, coeffs)) if row == start_expr]
        if repeat_matches:
            repeat_start_index = repeat_matches[0]
        loop_end_tick, loop_end_coeffs = starts[end_period + 1]
        loop_count = sequence.repeat.count
    final_tick = evaluate_affine_tick(ticks[-1], coeffs[-1], reference, frac_bits)
    loop_start = evaluate_affine_tick(
        ticks[repeat_start_index], coeffs[repeat_start_index], reference, frac_bits
    )
    nominal_loop_end = evaluate_affine_tick(
        loop_end_tick, loop_end_coeffs, reference, frac_bits
    )
    duration = (
        final_tick + (loop_count - 1) * (nominal_loop_end - loop_start)
    ) / clock_hz
    logical = tuple(sorted(
        (port.key, port.lanes[0])
        for port in sequence.target.ports
        if port.kind == "digital"
    ))
    safe_values = tuple(
        port.safe_value
        for port in sorted((port for port in sequence.target.ports if port.kind == PORT_DAC), key=lambda item: item.bus_index)
    )
    return CompiledProgram(
        clock_hz=clock_hz,
        target_abi_fingerprint=sequence.target.abi_fingerprint,
        geometry_fingerprint=build_fingerprint(params),
        channels=sequence.target.raw_lanes,
        ticks=tuple(ticks),
        masks=tuple(masks),
        duration_seconds=duration,
        loop_start_index=repeat_start_index,
        loop_end_tick=loop_end_tick,
        loop_count=loop_count,
        slot_kinds=tuple(slot.kind for slot in sequence.slots),
        loop_end_slot_coeffs=tuple(loop_end_coeffs),
        tick_slot_coeffs=tuple(coeffs),
        scan_coeff_frac_bits=frac_bits,
        bus_names=bus_names,
        bus_segments=bus_segments,
        bus_delays=bus_delays,
        channel_delays=channel_delays,
        clk_enable=clk_enable,
        logical_digital_outputs=logical,
        bus_safe_values=safe_values,
    )


__all__ = [
    "COMPILER_ID",
    "CompiledProgram",
    "TargetBusDelay",
    "TargetBusSegment",
    "compile_sequence",
    "evaluate_affine_tick",
    "maximum_duration_tick_scale",
    "narrow_slot_operand",
    "slot_operand_width",
]
