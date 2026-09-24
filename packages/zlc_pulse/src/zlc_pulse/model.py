"""Device-facing pulse model.

The model contains physical lanes, periods, output actions, and slot bindings.
It deliberately has no editor state, run identity, or acquisition concepts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction

from zlc_data.units import UnitError, resolve_unit
import math
import re
from types import MappingProxyType
from typing import Any

from .canonical import canonical_digest
from .loops import frame_ticks


PORT_DIGITAL = "digital"
PORT_DAC = "dac"
PORT_CLOCK = "clock"
PORT_KINDS = frozenset((PORT_DIGITAL, PORT_DAC, PORT_CLOCK))
DAC_OFFSET_BINARY = "offset_binary"

FIELD_DURATION = "duration"
FIELD_DAC = "dac"
FIELD_DELAY = "delay"
FIELD_KINDS = frozenset((FIELD_DURATION, FIELD_DAC, FIELD_DELAY))
SLOT_KINDS = frozenset((FIELD_DURATION, FIELD_DAC))
BINDING_SCAN = "scan"
BINDING_API = "api"
BINDING_CONFIG = "config"
BINDING_DEFAULT = "default"
BINDING_SOURCES = (BINDING_DEFAULT, BINDING_API, BINDING_CONFIG)
#: A period is authored; a spacer is time given to a slow device between two
#: authored periods.  A spacer's levels are set like any period's, but it
#: holds every DAC, is never scanned and takes no API value: its length is a
#: property of the device it waits for (or a Config value), not of the run.
PERIOD_KIND_PERIOD = "period"
PERIOD_KIND_SPACER = "spacer"
PERIOD_KINDS = (PERIOD_KIND_PERIOD, PERIOD_KIND_SPACER)
#: The coarsest and finest a pulse may be authored in.  A limit of this
#: instrument, stated beside the second's own ladder: the board's clock is
#: tens of nanoseconds, so a finer unit would only ever be refused by the
#: grid check below, and a period is never longer than seconds.  A tick is NOT a unit: it is
#: however long this board's clock says, and listing it as 1 ns once made it
#: a synonym for ns, so on a 20 ns clock the one duration that is on the grid
#: by definition became the one most likely to be refused.
_COARSEST_TIME_DECADE = 0
_FINEST_TIME_DECADE = -9

#: Authorable time units, finest first.  Derived, so the spelling this project
#: shows (``µs``) and the spellings it accepts (``us``) are decided in exactly
#: one place, and a unit cannot exist for the editor and not for the compiler.
TIME_UNIT_CHOICES = tuple(
    resolve_unit(f"{prefix.symbol}s").symbol
    for prefix in sorted(resolve_unit("s").ladder, key=lambda item: item.exponent)
    if _FINEST_TIME_DECADE <= prefix.exponent <= _COARSEST_TIME_DECADE
)


def nanoseconds_per(unit: str) -> Fraction:
    """How long one of ``unit`` is, in nanoseconds -- EXACTLY.

    A Fraction, not a float, because this feeds the tick grid: 1e-6/1e-9 is
    not exactly one thousand in binary, and a duration that misses the grid
    by an ulp is refused with nothing wrong with it.  The exponent comes off
    the unit itself, so the ratio is integer arithmetic all the way down.
    """

    resolved = resolve_unit(canonical_time_unit(unit, "time unit"))
    return Fraction(10) ** (resolved.decade - resolve_unit("ns").decade)


def canonical_time_unit(value: object, field_name: str) -> str:
    """The one spelling this project stores, for any spelling it accepts.

    ``us`` and ``µs`` are the same unit; storing whichever was typed would put
    two names for one thing into every saved document and every comparison.
    """

    text = _text(value, field_name)
    try:
        resolved = resolve_unit(text)
    except UnitError as error:
        raise ValueError(f"{field_name} is unsupported: {text!r}") from error
    if resolved.symbol not in TIME_UNIT_CHOICES:
        raise ValueError(
            f"{field_name} must be one of {TIME_UNIT_CHOICES}, not {text!r}"
        )
    return resolved.symbol
#: Stable domain order for editors and serializers that must offer every
#: model-supported analog step.  Labels remain a presentation concern.
ANALOG_MODE_CHOICES = ("edge", "ramp")
ANALOG_MODES = frozenset(ANALOG_MODE_CHOICES)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _text(value: Any, field_name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise TypeError(f"{field_name} must be non-empty text")
    return value


def _identifier(value: Any, field_name: str) -> str:
    result = _text(value, field_name)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ValueError(f"{field_name} must be an identifier")
    return result


def _number(value: Any, field_name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")
    if isinstance(value, int) or float(value).is_integer():
        return int(value)
    return float(value)


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _unit(value: Any, field_name: str) -> str:
    result = _text(value, field_name)
    if result == "value":
        return result
    return canonical_time_unit(result, field_name)


def _tick_ratio(value: int | float, unit: str, time_step_ns: float,
                field_name: str) -> Fraction:
    """How many device-clock ticks this value is -- exactly, as a fraction.

    Both questions below start here, so "what counts as a tick" is answered in
    one place and they can never disagree about it.
    """

    number = _number(value, field_name)
    if time_step_ns <= 0:
        raise ValueError("time_step_ns must be positive")
    return Fraction(str(number)) * nanoseconds_per(unit) / Fraction(str(time_step_ns))


def exact_ticks(value: int | float, unit: str, time_step_ns: float, field_name: str,
                *, minimum: int | None = 1) -> int:
    """Convert an owner-declared time value to an exact device-clock tick count."""

    ratio = _tick_ratio(value, unit, time_step_ns, field_name)
    if ratio.denominator != 1:
        raise ValueError(f"{field_name} is not on the device clock grid")
    ticks = int(ratio)
    if minimum is not None and ticks < minimum:
        raise ValueError(f"{field_name} must be at least {minimum} tick(s)")
    return ticks


def align_to_grid(value: int | float, unit: str, time_step_ns: float, field_name: str,
                  *, minimum: int | None = 1) -> float:
    """The nearest value ON the clock grid, in the unit it arrived in.

    ``exact_ticks`` asks "is this legal"; this asks "what legal value did they
    mean" -- the question an editor has.  Round with this BEFORE authoring, so
    the model stays strict and a document can never claim 1.003 us while the
    board runs 1.00 us.  Only ns ever felt the rule: on a 20 ns clock a whole
    number of us/ms/s lands on the grid by arithmetic alone.
    """

    ratio = _tick_ratio(value, unit, time_step_ns, field_name)
    half = Fraction(1, 2)
    ticks = int(ratio + half) if ratio >= 0 else -int(-ratio + half)
    if minimum is not None and ticks < minimum:
        ticks = minimum
    return float(Fraction(ticks) * Fraction(str(time_step_ns)) / nanoseconds_per(unit))


@dataclass(frozen=True)
class PulsePortSpec:
    key: str
    kind: str
    lanes: tuple[str, ...]
    label: str = ""
    bus_index: int | None = None
    width: int | None = None
    encoding: str | None = None
    safe_value: int | None = None
    latch_clock: str | None = None

    def __post_init__(self) -> None:
        key = _identifier(self.key, "port key")
        kind = _text(self.kind, "port kind")
        if kind not in PORT_KINDS:
            raise ValueError(f"port kind must be one of {sorted(PORT_KINDS)}")
        lanes = tuple(_identifier(lane, "raw lane") for lane in self.lanes)
        if not lanes or len(set(lanes)) != len(lanes):
            raise ValueError("port lanes must be unique and non-empty")
        width = len(lanes) if self.width is None else _nonnegative_int(self.width, "port width")
        if width != len(lanes):
            raise ValueError("port width differs from lane count")
        bus_index = self.bus_index
        if kind == PORT_DAC:
            if width < 2:
                raise ValueError("DAC ports require at least two lanes")
            if bus_index is None:
                bus_index = 0
            bus_index = _nonnegative_int(bus_index, "DAC bus_index")
            encoding = self.encoding or DAC_OFFSET_BINARY
            safe_value = (1 << (width - 1)) if self.safe_value is None else _nonnegative_int(self.safe_value, "DAC safe_value")
            if encoding != DAC_OFFSET_BINARY:
                raise ValueError("only offset-binary DAC encoding is supported")
        else:
            if width != 1:
                raise ValueError("digital and clock ports require one lane")
            if bus_index is not None:
                raise ValueError("digital and clock ports cannot have a bus index")
            encoding = self.encoding or "binary"
            safe_value = 0 if self.safe_value is None else _nonnegative_int(self.safe_value, "safe_value")
            if encoding != "binary" or safe_value != 0:
                raise ValueError("digital and clock ports require the low safe state")
            if self.latch_clock is not None:
                raise ValueError("digital and clock ports cannot have a latch clock")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lanes", lanes)
        object.__setattr__(self, "label", _text(self.label, "port label", empty=True))
        object.__setattr__(self, "bus_index", bus_index)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "encoding", encoding)
        object.__setattr__(self, "safe_value", safe_value)
        if self.latch_clock is not None:
            object.__setattr__(self, "latch_clock", _identifier(self.latch_clock, "latch clock"))

    @property
    def signed_range(self) -> tuple[int, int] | None:
        if self.kind != PORT_DAC:
            return None
        half = 1 << (self.width - 1)
        return -half, half - 1


@dataclass(frozen=True, init=False)
class PulseTarget:
    raw_lanes: tuple[str, ...]
    ports: tuple[PulsePortSpec, ...]
    _abi_fingerprint: str = field(init=False, repr=False, compare=False)
    _by_key: Mapping[str, PulsePortSpec] = field(init=False, repr=False, compare=False)
    _package_pins: Mapping[str, str] = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        raw_lanes: tuple[str, ...] | None = None,
        ports: tuple[PulsePortSpec, ...] = (),
        *,
        lanes: tuple[str, ...] | None = None,
        package_pins: Mapping[str, str] | None = None,
    ) -> None:
        if raw_lanes is None:
            raw_lanes = lanes
        if raw_lanes is None:
            raise TypeError("PulseTarget requires raw_lanes or lanes")
        raw = tuple(_identifier(lane, "target lane") for lane in raw_lanes)
        if not raw or len(raw) != len(set(raw)):
            raise ValueError("target lanes must be unique and non-empty")
        normalized_ports = tuple(ports)
        if any(not isinstance(port, PulsePortSpec) for port in normalized_ports):
            raise TypeError("target ports must contain PulsePortSpec values")
        keys = tuple(port.key for port in normalized_ports)
        if len(keys) != len(set(keys)):
            raise ValueError("target port keys must be unique")
        owner: dict[str, str] = {}
        for port in normalized_ports:
            for lane in port.lanes:
                if lane not in raw:
                    raise ValueError(f"port {port.key!r} owns unknown lane {lane!r}")
                if lane in owner:
                    raise ValueError(f"lane {lane!r} belongs to two ports")
                owner[lane] = port.key
        missing = tuple(lane for lane in raw if lane not in owner)
        if missing:
            raise ValueError(f"target lanes have no logical owner: {missing}")
        by_key = {port.key: port for port in normalized_ports}
        clocks = {port.key for port in normalized_ports if port.kind == PORT_CLOCK}
        for port in normalized_ports:
            if port.latch_clock is not None:
                if port.latch_clock not in clocks:
                    raise ValueError(f"DAC port {port.key!r} references a missing clock")
        bus_indices = sorted(port.bus_index for port in normalized_ports if port.kind == PORT_DAC)
        if bus_indices != list(range(len(bus_indices))):
            raise ValueError("DAC bus indices must be contiguous from zero")
        pins = {} if package_pins is None else dict(package_pins)
        if pins and set(pins) != set(raw):
            raise ValueError("package_pins must contain exactly one pin for every target lane")
        if any(
            not isinstance(lane, str) or not isinstance(pin, str) or not pin.strip()
            for lane, pin in pins.items()
        ):
            raise TypeError("package_pins must map lane names to non-empty pin names")
        object.__setattr__(self, "raw_lanes", raw)
        object.__setattr__(self, "ports", normalized_ports)
        object.__setattr__(self, "_by_key", MappingProxyType(by_key))
        object.__setattr__(self, "_package_pins", MappingProxyType(pins))
        abi_ports = [
            {
                "key": port.key,
                "kind": port.kind,
                "lanes": list(port.lanes),
                "bus_index": port.bus_index,
                "width": port.width,
                "encoding": port.encoding,
                "safe_value": port.safe_value,
                "latch_clock": port.latch_clock,
            }
            for port in normalized_ports
        ]
        object.__setattr__(
            self,
            "_abi_fingerprint",
            canonical_digest({"schema": "zlc_pulse.PulseTargetABI", "lanes": list(raw), "ports": abi_ports}),
        )

    @property
    def lanes(self) -> tuple[str, ...]:
        return self.raw_lanes

    @property
    def by_key(self) -> Mapping[str, PulsePortSpec]:
        return self._by_key

    @property
    def package_pins(self) -> Mapping[str, str]:
        """Package pin by raw lane; empty for a logical target without XDC metadata."""

        return self._package_pins

    @property
    def abi_fingerprint(self) -> str:
        return self._abi_fingerprint

@dataclass(frozen=True)
class PulseFieldRef:
    kind: str
    period_id: str | None = None
    port: str | None = None

    def __post_init__(self) -> None:
        kind = _text(self.kind, "field kind")
        if kind not in FIELD_KINDS:
            raise ValueError(f"unsupported field kind {kind!r}")
        if kind == FIELD_DURATION:
            if self.period_id is None or self.port is not None:
                raise ValueError("duration fields require period_id only")
            object.__setattr__(self, "period_id", _identifier(self.period_id, "duration period_id"))
        elif kind == FIELD_DAC:
            if self.period_id is None or self.port is None:
                raise ValueError("DAC fields require period_id and port")
            object.__setattr__(self, "period_id", _identifier(self.period_id, "DAC period_id"))
            object.__setattr__(self, "port", _identifier(self.port, "DAC port"))
        else:
            if self.period_id is not None or self.port is None:
                raise ValueError("delay fields require port only")
            object.__setattr__(self, "port", _identifier(self.port, "delay port"))
        object.__setattr__(self, "kind", kind)


    @property
    def key(self) -> str:
        """Stable physical identity, independent of display names and bindings."""
        if self.kind == FIELD_DELAY:
            return f"delay:{self.port}"
        if self.kind == FIELD_DURATION:
            return f"duration:{self.period_id}"
        return f"dac:{self.period_id}:{self.port}"


def config_parameter_key(value: object) -> str:
    """A cross-pulse Config name, never a positional slot number."""
    return _identifier(value, "Config parameter name")


@dataclass(frozen=True)
class PulseBinding:
    """One field's independent scan capability and default/API/Config source."""

    field_ref: PulseFieldRef
    unit: str
    scan: bool = False
    source: str = BINDING_DEFAULT
    config_key: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.field_ref, PulseFieldRef):
            raise TypeError("binding field_ref must be PulseFieldRef")
        if not isinstance(self.scan, bool):
            raise TypeError("binding scan must be boolean")
        if self.scan and self.field_ref.kind not in SLOT_KINDS:
            raise ValueError("this field cannot be scanned")
        if self.source not in BINDING_SOURCES:
            raise ValueError("binding source must be default, api or config")
        unit = _unit(self.unit, "binding unit")
        if (self.field_ref.kind == FIELD_DAC) != (unit == "value"):
            raise ValueError("DAC bindings use 'value'; time bindings use a time unit")
        key = _text(self.config_key, "Config key", empty=True)
        if key:
            config_parameter_key(key)
        if self.source != BINDING_CONFIG and key:
            raise ValueError("only Config bindings may name a Config key")
        object.__setattr__(self, "unit", unit)

    @property
    def field_id(self) -> str:
        return self.field_ref.key

    @property
    def kind(self) -> str:
        return self.field_ref.kind


@dataclass(frozen=True)
class AnalogStep:
    port: str
    mode: str
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", _identifier(self.port, "analog port"))
        mode = _text(self.mode, "analog mode")
        if mode not in ANALOG_MODES:
            raise ValueError(f"unsupported analog mode {mode!r}")
        object.__setattr__(self, "mode", mode)
        number = _number(self.value, "analog value")
        if not isinstance(number, int):
            raise ValueError("analog value must be an integral code")
        object.__setattr__(self, "value", number)


@dataclass(frozen=True)
class PulsePeriod:
    period_id: str
    duration: int | float
    unit: str = "ns"
    states: tuple[int, ...] = ()
    analog_steps: tuple[AnalogStep, ...] = ()
    name: str = ""
    kind: str = PERIOD_KIND_PERIOD

    def __post_init__(self) -> None:
        object.__setattr__(self, "period_id", _identifier(self.period_id, "period_id"))
        kind = _text(self.kind, "period kind")
        if kind not in PERIOD_KINDS:
            raise ValueError(f"period kind must be one of {PERIOD_KINDS}, got {kind!r}")
        object.__setattr__(self, "kind", kind)
        duration = _number(self.duration, "period duration")
        if duration <= 0:
            raise ValueError("period duration must be positive")
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "unit", canonical_time_unit(self.unit, "period unit"))
        states = tuple(self.states)
        if any(not isinstance(value, (int, bool)) or int(value) not in (0, 1) for value in states):
            raise ValueError("period states must contain only 0/1 values")
        object.__setattr__(self, "states", tuple(int(value) for value in states))
        steps = tuple(self.analog_steps)
        if any(not isinstance(step, AnalogStep) for step in steps):
            raise TypeError("analog_steps must contain AnalogStep values")
        if len({step.port for step in steps}) != len(steps):
            raise ValueError("a period has at most one analog step per port")
        if kind == PERIOD_KIND_SPACER and steps:
            raise ValueError("a spacer holds every DAC; it has no analog steps")
        object.__setattr__(self, "analog_steps", steps)
        object.__setattr__(self, "name", _text(self.name, "period name", empty=True))


@dataclass(frozen=True)
class OutputDelay:
    port: str
    value: int | float
    unit: str = "ns"

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", _identifier(self.port, "delay port"))
        object.__setattr__(self, "value", _number(self.value, "delay value"))
        object.__setattr__(self, "unit", canonical_time_unit(self.unit, "delay unit"))


#: The smallest count a timeline bracket may loop.  Once plays the range
#: exactly as it would play unbracketed; the bracket stays in the document
#: and on the editor's strip, which is what debugging a loop needs -- the
#: count goes to 1 and back instead of the bracket being deleted and redrawn.
MINIMUM_BRACKET_COUNT = 1
MAXIMUM_REPEAT_COUNT = (1 << 32) - 1


@dataclass(frozen=True)
class PulseBracket:
    """Loop one continuous range of timeline periods a whole number of times.

    A pulse holds any number of brackets; each is named by ``bracket_id`` the
    way a period is named by ``period_id``, so an editor, the remote API and a
    saved document all point at the same bracket without counting positions.
    Two brackets are either disjoint or one lies inside the other; the board
    plays them as nested loops.
    """

    bracket_id: str
    start_period_id: str | None
    end_period_id: str | None
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bracket_id", _identifier(self.bracket_id, "bracket_id"))
        object.__setattr__(
            self,
            "start_period_id",
            None if self.start_period_id is None else _identifier(self.start_period_id, "bracket start"),
        )
        object.__setattr__(
            self,
            "end_period_id",
            None if self.end_period_id is None else _identifier(self.end_period_id, "bracket end"),
        )
        object.__setattr__(
            self,
            "count",
            _nonnegative_int(self.count, "bracket count"),
        )
        if self.count < MINIMUM_BRACKET_COUNT:
            raise ValueError(
                f"a bracket loops at least {MINIMUM_BRACKET_COUNT} time(s)"
            )
        if self.count > MAXIMUM_REPEAT_COUNT:
            raise ValueError("bracket count does not fit the hardware 32-bit count")


def _bracket_gap_bounds(
    bracket: PulseBracket, period_ids: tuple[str, ...]
) -> tuple[int, int]:
    """Half-open period gaps of one bracket; equal gaps are an empty bracket.

    A missing start anchor means "after the last period" and a missing end
    anchor "before the first", so an empty bracket can sit at either edge of
    the timeline where it has no outer neighbour to anchor to.
    """

    return (
        len(period_ids)
        if bracket.start_period_id is None
        else period_ids.index(bracket.start_period_id),
        0 if bracket.end_period_id is None else period_ids.index(bracket.end_period_id) + 1,
    )


def bracket_contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    """Whether ``inner`` lies inside ``outer`` (gap bounds, both half-open).

    An empty bracket occupies one gap.  It is inside another bracket only when
    that gap is strictly inside it: at a bracket's own boundary gap an empty
    bracket sits beside it, not within it, so a bracket's nesting is decided
    by its bounds alone and an editor can draw it from the same rule.
    """

    start, end = inner
    if not (outer[0] <= start and end <= outer[1]):
        return False
    return start < end or outer[0] < start < outer[1]


def brackets_disjoint(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[1] <= second[0] or second[1] <= first[0]


@dataclass(frozen=True, init=False)
class PulseSequence:
    name: str
    target: PulseTarget
    time_step_ns: float
    periods: tuple[PulsePeriod, ...]
    bindings: tuple[PulseBinding, ...]
    delays: tuple[OutputDelay, ...]
    #: Outer brackets first; a bracket that lies inside another follows it.
    brackets: tuple[PulseBracket, ...]
    run_repeats: int
    _period_by_id: Mapping[str, PulsePeriod] = field(init=False, repr=False, compare=False)
    _bracket_bounds: tuple[tuple[int, int], ...] = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        name: str = "sequence",
        target: PulseTarget | None = None,
        time_step_ns: float = 20.0,
        periods: tuple[PulsePeriod, ...] = (),
        bindings: tuple[PulseBinding, ...] = (),
        delays: tuple[OutputDelay, ...] = (),
        brackets: tuple[PulseBracket, ...] = (),
        run_repeats: int = 0,
    ) -> None:
        if target is None:
            raise TypeError("PulseSequence requires a target")
        if not isinstance(target, PulseTarget):
            raise TypeError("target must be PulseTarget")
        periods = tuple(periods)
        if not periods or any(not isinstance(period, PulsePeriod) for period in periods):
            raise ValueError("periods must contain PulsePeriod values")
        if time_step_ns <= 0 or not math.isfinite(float(time_step_ns)):
            raise ValueError("time_step_ns must be positive and finite")
        ids = tuple(period.period_id for period in periods)
        if len(ids) != len(set(ids)):
            raise ValueError("period ids must be unique")
        display_names = tuple(period.name or period.period_id for period in periods)
        if len(display_names) != len(set(display_names)):
            raise ValueError("period names must be unique (empty Name uses the period ID)")
        lane_owner = {lane: port for port in target.ports for lane in port.lanes}
        for period in periods:
            if len(period.states) != len(target.raw_lanes):
                raise ValueError("every period state vector must match target lanes")
            exact_ticks(period.duration, period.unit, float(time_step_ns), f"period {period.period_id} duration")
            for index, state in enumerate(period.states):
                if state and lane_owner[target.raw_lanes[index]].kind != PORT_DIGITAL:
                    raise ValueError("non-digital lanes cannot carry digital states")
            for step in period.analog_steps:
                port = target.by_key.get(step.port)
                if port is None or port.kind != PORT_DAC:
                    raise ValueError(f"analog step references unknown DAC {step.port!r}")
                low, high = port.signed_range
                if not low <= step.value <= high:
                    raise ValueError(f"analog value for {step.port!r} is outside its range")
        delay_values = tuple(delays)
        if any(not isinstance(delay, OutputDelay) for delay in delay_values):
            raise TypeError("delays must contain OutputDelay values")
        if len({delay.port for delay in delay_values}) != len(delay_values):
            raise ValueError("at most one delay per logical port")
        for delay in delay_values:
            port = target.by_key.get(delay.port)
            if port is None or port.kind not in (PORT_DIGITAL, PORT_DAC):
                raise ValueError(f"delay references unsupported port {delay.port!r}")
            exact_ticks(delay.value, delay.unit, float(time_step_ns), f"delay {delay.port}", minimum=None)
        binding_values = tuple(bindings)
        if any(not isinstance(binding, PulseBinding) for binding in binding_values):
            raise TypeError("bindings must contain PulseBinding values")
        if len({binding.field_ref for binding in binding_values}) != len(binding_values):
            raise ValueError("each physical field has one binding declaration")
        by_period = {period.period_id: period for period in periods}
        for binding in binding_values:
            ref = binding.field_ref
            if ref.kind in (FIELD_DURATION, FIELD_DAC) and ref.period_id not in by_period:
                raise ValueError(f"binding references missing period {ref.period_id!r}")
            if ref.kind in (FIELD_DURATION, FIELD_DAC) and by_period[ref.period_id].kind == PERIOD_KIND_SPACER:
                if ref.kind == FIELD_DAC:
                    raise ValueError(f"spacer {ref.period_id!r} holds its DACs; nothing there can be bound")
                if binding.scan or binding.source == BINDING_API:
                    raise ValueError(
                        f"spacer {ref.period_id!r} duration can take a Config value but is never scanned "
                        "or set by API"
                    )
            if ref.kind == FIELD_DAC:
                port = target.by_key.get(ref.port)
                if port is None or port.kind != PORT_DAC:
                    raise ValueError(f"binding references missing DAC {ref.port!r}")
            if ref.kind == FIELD_DELAY:
                port = target.by_key.get(ref.port)
                if port is None or port.kind not in (PORT_DIGITAL, PORT_DAC):
                    raise ValueError(f"binding references missing delay port {ref.port!r}")
        bracket_values = tuple(brackets)
        if any(not isinstance(bracket, PulseBracket) for bracket in bracket_values):
            raise TypeError("brackets must contain PulseBracket values")
        if len({bracket.bracket_id for bracket in bracket_values}) != len(bracket_values):
            raise ValueError("bracket ids must be unique")
        for bracket in bracket_values:
            if (
                (bracket.start_period_id is not None and bracket.start_period_id not in by_period)
                or (bracket.end_period_id is not None and bracket.end_period_id not in by_period)
            ):
                raise ValueError(f"bracket {bracket.bracket_id!r} references a missing period")
        bounds = {
            bracket.bracket_id: _bracket_gap_bounds(bracket, ids) for bracket in bracket_values
        }
        for bracket in bracket_values:
            start, end = bounds[bracket.bracket_id]
            if start > end:
                raise ValueError(f"bracket {bracket.bracket_id!r} end precedes its start")
        for index, outer in enumerate(bracket_values):
            for inner in bracket_values[index + 1:]:
                first, second = bounds[outer.bracket_id], bounds[inner.bracket_id]
                if not (
                    brackets_disjoint(first, second)
                    or bracket_contains(first, second)
                    or bracket_contains(second, first)
                ):
                    raise ValueError(
                        f"brackets {outer.bracket_id!r} and {inner.bracket_id!r} overlap "
                        "without one lying inside the other"
                    )
        # Outer first: an enclosing bracket starts no later and ends no earlier
        # than what it encloses, and a stable sort keeps the authored order
        # for brackets with identical bounds.
        ordered = tuple(sorted(
            bracket_values,
            key=lambda bracket: (bounds[bracket.bracket_id][0], -bounds[bracket.bracket_id][1]),
        ))
        run_repeats = _nonnegative_int(run_repeats, "run_repeats")
        if run_repeats > MAXIMUM_REPEAT_COUNT:
            raise ValueError("run_repeats does not fit the hardware 32-bit count")
        object.__setattr__(self, "name", _text(name, "sequence name"))
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "time_step_ns", float(time_step_ns))
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "bindings", binding_values)
        object.__setattr__(self, "delays", delay_values)
        object.__setattr__(self, "brackets", ordered)
        object.__setattr__(
            self, "_bracket_bounds", tuple(bounds[bracket.bracket_id] for bracket in ordered)
        )
        object.__setattr__(self, "run_repeats", run_repeats)
        object.__setattr__(self, "_period_by_id", MappingProxyType(by_period))

    @property
    def bracket_bounds(self) -> tuple[tuple[int, int], ...]:
        """Half-open period gaps of each bracket, aligned with ``brackets``.

        Equal gaps preserve an empty authored bracket.
        """

        return self._bracket_bounds

    def bracket_by_id(self, bracket_id: str) -> PulseBracket:
        for bracket in self.brackets:
            if bracket.bracket_id == bracket_id:
                return bracket
        raise ValueError(f"no bracket exists with id {bracket_id!r}")

    @property
    def bracket_depths(self) -> tuple[int, ...]:
        """How many brackets each bracket lies inside, plus one; aligned with ``brackets``.

        The deepest value is the loop nesting the board must hold at once.
        """

        return tuple(
            1 + sum(
                bracket_contains(outer, inner)
                for other, outer in enumerate(self._bracket_bounds)
                if other != index
            )
            for index, inner in enumerate(self._bracket_bounds)
        )

    @property
    def loops(self) -> tuple[tuple[int, int, int], ...]:
        """The row loop table: ``(first_row, last_row, count)`` per bracket, outermost first.

        One period is one row, so a bracket over the period gaps
        ``start..end`` loops rows ``start..end-1``.  An EMPTY bracket loops
        nothing and is not in the table; whether it may stand at all is
        ``require_nonempty_brackets``'s question, asked before any compile,
        save or export.
        """

        return tuple(
            (start, end - 1, bracket.count)
            for bracket, (start, end) in zip(self.brackets, self._bracket_bounds, strict=True)
            if end > start
        )

    def played_nanoseconds(self) -> float:
        """How long the board plays one Pulse: the rows as the loop table expands them.

        The sum of the periods is one PASS.  A bracket plays its rows
        ``count`` times, a nested one that many times per pass of its
        parent, so the time the board takes is the pass with every bracket
        expanded.  This is the same walk the compiler stamps a program's
        duration with, over the authored durations exactly in nanoseconds,
        so an editor's total and the board's clock agree to the digit.
        """

        durations = tuple(
            Fraction(str(float(period.duration))) * nanoseconds_per(period.unit)
            for period in self.periods
        )
        return float(frame_ticks(durations, self.loops))

    def require_nonempty_brackets(self) -> None:
        for bracket, (start, end) in zip(self.brackets, self._bracket_bounds, strict=True):
            if start == end:
                raise ValueError(
                    f"Bracket {bracket.bracket_id} is empty. Put a period inside it or "
                    "remove the bracket before running or saving."
                )

    @property
    def scan_bindings(self) -> tuple[PulseBinding, ...]:
        return tuple(binding for binding in self.bindings if binding.scan)

    @property
    def api_bindings(self) -> tuple[PulseBinding, ...]:
        return tuple(binding for binding in self.bindings if binding.source == BINDING_API)

    @property
    def config_bindings(self) -> tuple[PulseBinding, ...]:
        return tuple(binding for binding in self.bindings if binding.source == BINDING_CONFIG)

    @property
    def slot_count(self) -> int:
        return len(self.scan_bindings)

    @property
    def slot_kinds(self) -> tuple[str, ...]:
        return tuple(slot.kind for slot in self.scan_bindings)

    @property
    def period_by_id(self) -> Mapping[str, PulsePeriod]:
        return self._period_by_id

    def field_unit(self, reference: PulseFieldRef) -> str:
        """The authored unit of one physical field."""

        if not isinstance(reference, PulseFieldRef):
            raise TypeError("reference must be PulseFieldRef")
        if reference.kind == FIELD_DAC:
            return "value"
        if reference.kind == FIELD_DELAY:
            delay = next(
                (item for item in self.delays if item.port == reference.port), None
            )
            return str(delay.unit) if delay is not None else "ns"
        period = self.period_by_id.get(str(reference.period_id))
        if period is None:
            raise ValueError(f"no period exists with id {reference.period_id!r}")
        return str(period.unit)

__all__ = [
    "ANALOG_MODES",
    "PERIOD_KIND_PERIOD",
    "PERIOD_KIND_SPACER",
    "PERIOD_KINDS",
    "DAC_OFFSET_BINARY",
    "BINDING_CONFIG",
    "FIELD_DAC",
    "FIELD_DELAY",
    "FIELD_DURATION",
    "FIELD_KINDS",
    "MAXIMUM_REPEAT_COUNT",
    "MINIMUM_BRACKET_COUNT",
    "OutputDelay",
    "PulseFieldRef",
    "PulseBinding",
    "PulsePeriod",
    "PulseBracket",
    "PulsePortSpec",
    "PulseSequence",
    "PulseTarget",
    "PORT_CLOCK",
    "PORT_DAC",
    "PORT_DIGITAL",
    "TIME_UNIT_CHOICES",
    "bracket_contains",
    "brackets_disjoint",
    "canonical_time_unit",
    "config_parameter_key",
    "BINDING_DEFAULT",
    "BINDING_SOURCES",
    "nanoseconds_per",
    "exact_ticks",
]
