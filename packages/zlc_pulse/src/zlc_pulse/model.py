"""Device-facing pulse model.

The model contains physical lanes, periods, output actions, and slot bindings.
It deliberately has no editor state, run identity, or acquisition concepts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
import math
import re
from types import MappingProxyType
from typing import Any

from .canonical import canonical_digest


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
#: The legal binding transition for each physical pulse field.  Binding
#: availability is a pulse-model fact: a delay has no scan-table column, while
#: durations and DAC values may be supplied by either a scan or the API.
FIELD_BINDING_CYCLES = MappingProxyType(
    {
        FIELD_DURATION: (None, BINDING_SCAN, BINDING_API),
        FIELD_DAC: (None, BINDING_SCAN, BINDING_API),
        FIELD_DELAY: (None, BINDING_API),
    }
)
#: How long one of each unit is, in nanoseconds.  A tick is NOT here: it is
#: however long this board's clock says, and listing it as 1.0 ns made it a
#: synonym for ns -- so on a 20 ns clock the one unit that is on the grid by
#: definition became the one most likely to be refused.
TIME_UNIT_TO_NS = {
    "ns": 1.0,
    "us": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
}
TIME_UNIT_CHOICES = tuple(TIME_UNIT_TO_NS)
#: Stable domain order for editors and serializers that must offer every
#: model-supported analog step.  Labels remain a presentation concern.
ANALOG_MODE_CHOICES = ("edge", "ramp")
ANALOG_MODES = frozenset(ANALOG_MODE_CHOICES)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def cycle_binding_kind(
    binding: str | None,
    *,
    field_kind: str,
) -> str | None:
    """Return the next legal binding for one canonical pulse field kind.

    The caller supplies the ``PulseFieldRef.kind`` value, not a widget alias.
    Refusing unknown field and binding values here keeps the authoring control
    from silently widening the pulse model's finite domain.
    """

    try:
        cycle = FIELD_BINDING_CYCLES[field_kind]
    except KeyError as exc:
        raise ValueError(f"unknown pulse field kind {field_kind!r}") from exc
    if binding not in cycle:
        raise ValueError(
            f"binding {binding!r} is not valid for pulse field kind {field_kind!r}"
        )
    return cycle[(cycle.index(binding) + 1) % len(cycle)]


def _text(value: Any, field_name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise TypeError(f"{field_name} must be non-empty text")
    if not empty and not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
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
    if result not in TIME_UNIT_TO_NS and result != "value":
        raise ValueError(f"{field_name} is unsupported: {result!r}")
    return result


def _tick_ratio(value: int | float, unit: str, time_step_ns: float,
                field_name: str) -> Fraction:
    """How many device-clock ticks this value is -- exactly, as a fraction.

    Both questions below start here, so "what counts as a tick" is answered in
    one place and they can never disagree about it.
    """

    number = _number(value, field_name)
    if time_step_ns <= 0:
        raise ValueError("time_step_ns must be positive")
    if unit not in TIME_UNIT_TO_NS:
        raise ValueError(f"{field_name} has an unsupported time unit")
    return Fraction(str(number)) * Fraction(str(TIME_UNIT_TO_NS[unit])) / Fraction(str(time_step_ns))


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
    return float(Fraction(ticks) * Fraction(str(time_step_ns))
                 / Fraction(str(TIME_UNIT_TO_NS[unit])))


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


@dataclass(frozen=True)
class PulseSlot:
    kind: str
    field_ref: PulseFieldRef
    unit: str
    slot_id: str = ""

    def __post_init__(self) -> None:
        kind = _text(self.kind, "slot kind")
        if kind not in SLOT_KINDS:
            raise ValueError(f"unsupported slot kind {kind!r}")
        if not isinstance(self.field_ref, PulseFieldRef) or self.field_ref.kind != kind:
            raise ValueError("slot kind and field reference differ")
        unit = _unit(self.unit, "slot unit")
        if kind == FIELD_DAC and unit != "value":
            raise ValueError("DAC slots use unit 'value'")
        if kind != FIELD_DAC and unit == "value":
            raise ValueError("time slots use a time unit")
        slot_id = self.slot_id or f"{kind}_{self.field_ref.period_id or self.field_ref.port}"
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "slot_id", _identifier(slot_id, "slot_id"))

    @property
    def field(self) -> PulseFieldRef:
        return self.field_ref


@dataclass(frozen=True)
class PulseApiParameter:
    """One named run-time input owned by the pulse API, never by a scan table."""

    parameter_id: str
    field_ref: PulseFieldRef
    unit: str

    def __post_init__(self) -> None:
        parameter_id = _identifier(self.parameter_id, "parameter_id")
        if not isinstance(self.field_ref, PulseFieldRef):
            raise TypeError("API parameter field_ref must be PulseFieldRef")
        unit = _unit(self.unit, "API parameter unit")
        if self.field_ref.kind == FIELD_DAC and unit != "value":
            raise ValueError("DAC API parameters use unit 'value'")
        if self.field_ref.kind != FIELD_DAC and unit == "value":
            raise ValueError("time API parameters use a time unit")
        object.__setattr__(self, "parameter_id", parameter_id)
        object.__setattr__(self, "unit", unit)

    @property
    def field(self) -> PulseFieldRef:
        return self.field_ref


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "period_id", _identifier(self.period_id, "period_id"))
        duration = _number(self.duration, "period duration")
        if duration <= 0:
            raise ValueError("period duration must be positive")
        object.__setattr__(self, "duration", duration)
        if self.unit not in TIME_UNIT_TO_NS:
            raise ValueError("period unit must be a time unit")
        states = tuple(self.states)
        if any(not isinstance(value, (int, bool)) or int(value) not in (0, 1) for value in states):
            raise ValueError("period states must contain only 0/1 values")
        object.__setattr__(self, "states", tuple(int(value) for value in states))
        steps = tuple(self.analog_steps)
        if any(not isinstance(step, AnalogStep) for step in steps):
            raise TypeError("analog_steps must contain AnalogStep values")
        if len({step.port for step in steps}) != len(steps):
            raise ValueError("a period has at most one analog step per port")
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
        if self.unit not in TIME_UNIT_TO_NS:
            raise ValueError("delay unit must be a time unit")


#: The smallest count that makes a timeline bracket meaningful.
MINIMUM_BRACKET_COUNT = 2
MAXIMUM_REPEAT_COUNT = (1 << 32) - 1


@dataclass(frozen=True)
class PulseBracket:
    """Loop one continuous range of timeline periods, at least twice."""

    start_period_id: str
    end_period_id: str
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_period_id",
            _identifier(self.start_period_id, "bracket start"),
        )
        object.__setattr__(
            self,
            "end_period_id",
            _identifier(self.end_period_id, "bracket end"),
        )
        object.__setattr__(
            self,
            "count",
            _nonnegative_int(self.count, "bracket count"),
        )
        if self.count < MINIMUM_BRACKET_COUNT:
            raise ValueError(
                f"a bracket loops at least {MINIMUM_BRACKET_COUNT} times; "
                "once needs no bracket"
            )
        if self.count > MAXIMUM_REPEAT_COUNT:
            raise ValueError("bracket count does not fit the hardware 32-bit count")


@dataclass(frozen=True, init=False)
class PulseSequence:
    name: str
    target: PulseTarget
    time_step_ns: float
    periods: tuple[PulsePeriod, ...]
    slots: tuple[PulseSlot, ...]
    api_parameters: tuple[PulseApiParameter, ...]
    delays: tuple[OutputDelay, ...]
    bracket: PulseBracket | None
    run_repeats: int
    _period_by_id: Mapping[str, PulsePeriod] = field(init=False, repr=False, compare=False)
    _slot_by_id: Mapping[str, PulseSlot] = field(init=False, repr=False, compare=False)
    _api_parameter_by_id: Mapping[str, PulseApiParameter] = field(
        init=False, repr=False, compare=False
    )

    def __init__(
        self,
        name: str = "sequence",
        target: PulseTarget | None = None,
        time_step_ns: float = 20.0,
        periods: tuple[PulsePeriod, ...] = (),
        slots: tuple[PulseSlot, ...] = (),
        api_parameters: tuple[PulseApiParameter, ...] = (),
        delays: tuple[OutputDelay, ...] = (),
        bracket: PulseBracket | None = None,
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
        slot_values = tuple(slots)
        if any(not isinstance(slot, PulseSlot) for slot in slot_values):
            raise TypeError("slots must contain PulseSlot values")
        if len({slot.slot_id for slot in slot_values}) != len(slot_values):
            raise ValueError("slot ids must be unique")
        if len({slot.field_ref for slot in slot_values}) != len(slot_values):
            raise ValueError("each physical field can have only one slot")
        api_values = tuple(api_parameters)
        if any(not isinstance(parameter, PulseApiParameter) for parameter in api_values):
            raise TypeError("api_parameters must contain PulseApiParameter values")
        parameter_ids = tuple(parameter.parameter_id for parameter in api_values)
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("API parameter ids must be unique")
        if len({parameter.field_ref for parameter in api_values}) != len(api_values):
            raise ValueError("each physical field can have only one API parameter")
        all_binding_ids = tuple(slot.slot_id for slot in slot_values) + parameter_ids
        if len(all_binding_ids) != len(set(all_binding_ids)):
            raise ValueError("scan slots and API parameters share one unique id namespace")
        all_fields = tuple(slot.field_ref for slot in slot_values) + tuple(
            parameter.field_ref for parameter in api_values
        )
        if len(all_fields) != len(set(all_fields)):
            raise ValueError("a physical field cannot be both scan-bound and API-bound")
        by_period = {period.period_id: period for period in periods}
        for binding in (*slot_values, *api_values):
            ref = binding.field_ref
            if ref.kind in (FIELD_DURATION, FIELD_DAC) and ref.period_id not in by_period:
                raise ValueError(f"binding references missing period {ref.period_id!r}")
            if ref.kind == FIELD_DAC:
                port = target.by_key.get(ref.port)
                if port is None or port.kind != PORT_DAC:
                    raise ValueError(f"binding references missing DAC {ref.port!r}")
            if ref.kind == FIELD_DELAY:
                port = target.by_key.get(ref.port)
                if port is None or port.kind not in (PORT_DIGITAL, PORT_DAC):
                    raise ValueError(f"binding references missing delay port {ref.port!r}")
        if bracket is not None:
            if not isinstance(bracket, PulseBracket):
                raise TypeError("bracket must be PulseBracket or None")
            if (
                bracket.start_period_id not in by_period
                or bracket.end_period_id not in by_period
            ):
                raise ValueError("bracket references a missing period")
            if ids.index(bracket.end_period_id) < ids.index(bracket.start_period_id):
                raise ValueError("bracket end precedes bracket start")
        run_repeats = _nonnegative_int(run_repeats, "run_repeats")
        if run_repeats > MAXIMUM_REPEAT_COUNT:
            raise ValueError("run_repeats does not fit the hardware 32-bit count")
        object.__setattr__(self, "name", _text(name, "sequence name"))
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "time_step_ns", float(time_step_ns))
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "slots", slot_values)
        object.__setattr__(self, "api_parameters", api_values)
        object.__setattr__(self, "delays", delay_values)
        object.__setattr__(self, "bracket", bracket)
        object.__setattr__(self, "run_repeats", run_repeats)
        object.__setattr__(self, "_period_by_id", MappingProxyType(by_period))
        object.__setattr__(self, "_slot_by_id", MappingProxyType({slot.slot_id: slot for slot in slot_values}))
        object.__setattr__(
            self,
            "_api_parameter_by_id",
            MappingProxyType({parameter.parameter_id: parameter for parameter in api_values}),
        )

    @property
    def slot_count(self) -> int:
        return len(self.slots)

    @property
    def slot_kinds(self) -> tuple[str, ...]:
        return tuple(slot.kind for slot in self.slots)

    @property
    def period_by_id(self) -> Mapping[str, PulsePeriod]:
        return self._period_by_id

    @property
    def slot_by_id(self) -> Mapping[str, PulseSlot]:
        return self._slot_by_id

    @property
    def api_parameter_by_id(self) -> Mapping[str, PulseApiParameter]:
        return self._api_parameter_by_id

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
    "DAC_OFFSET_BINARY",
    "FIELD_DAC",
    "FIELD_DELAY",
    "FIELD_DURATION",
    "FIELD_KINDS",
    "MAXIMUM_REPEAT_COUNT",
    "MINIMUM_BRACKET_COUNT",
    "OutputDelay",
    "PulseFieldRef",
    "PulseApiParameter",
    "PulsePeriod",
    "PulseBracket",
    "PulsePortSpec",
    "PulseSequence",
    "PulseSlot",
    "PulseTarget",
    "PORT_CLOCK",
    "PORT_DAC",
    "PORT_DIGITAL",
    "TIME_UNIT_CHOICES",
    "TIME_UNIT_TO_NS",
    "exact_ticks",
]
