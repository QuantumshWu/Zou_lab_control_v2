"""Resolve named pulse inputs into the physical fields they own."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from fractions import Fraction
import math

from .model import (
    BINDING_API,
    BINDING_CONFIG,
    BINDING_DEFAULT,
    config_parameter_key,
    FIELD_DAC,
    FIELD_DELAY,
    FIELD_DURATION,
    PORT_DAC,
    PORT_DIGITAL,
    nanoseconds_per,
    OutputDelay,
    PulseBinding,
    PulseFieldRef,
    PulseSequence,
    align_to_grid,
)


def pulse_field_value(
    sequence: PulseSequence,
    reference: PulseFieldRef,
    unit: str,
) -> int | float:
    """Read one referenced field in the unit declared by its binding."""

    _check_inputs(sequence, reference)
    if reference.kind == FIELD_DELAY:
        delay = next(
            (item for item in sequence.delays if item.port == reference.port), None
        )
        if delay is None:
            return 0
        return convert_time(delay.value, delay.unit, unit)

    period = sequence.period_by_id.get(str(reference.period_id))
    if period is None:
        raise ValueError(f"no period exists with id {reference.period_id!r}")
    if reference.kind == FIELD_DURATION:
        return convert_time(period.duration, period.unit, unit)
    step = next(
        (item for item in period.analog_steps if item.port == reference.port), None
    )
    if step is None:
        raise ValueError(
            f"period {period.period_id!r} has no DAC step on port {reference.port!r}"
        )
    if unit != "value":
        raise ValueError("DAC fields use unit 'value'")
    return int(step.value)


def replace_pulse_field(
    sequence: PulseSequence,
    reference: PulseFieldRef,
    value: int | float,
    unit: str,
    *,
    field_name: str,
) -> PulseSequence:
    """Return ``sequence`` with one referenced physical field replaced.

    A duration lands on the clock grid the way the editor rounds a typed
    number, but a zero or negative one is refused rather than rounded up to
    one tick: no rounding of it is a value anybody meant, and a node form
    that said ``-100`` would otherwise play the shortest pulse the board has
    and call it the requested one.
    """

    _check_inputs(sequence, reference)
    return _replace_pulse_fields(sequence, ((reference, value, unit, field_name),))


def _replace_pulse_fields(
    sequence: PulseSequence,
    fields: Iterable[tuple[PulseFieldRef, int | float, str, str]],
    *,
    remove_api_parameters: bool = False,
) -> PulseSequence:
    """Normalize all edits, then validate one final immutable pulse."""

    period_changes: dict[str, dict[str, object]] = {}
    delay_changes: dict[str, OutputDelay] = {}
    for reference, value, unit, field_name in fields:
        if reference.kind == FIELD_DELAY:
            port = sequence.target.by_key.get(reference.port)
            if port is None or port.kind not in (PORT_DIGITAL, PORT_DAC):
                raise ValueError(f"no delay output exists with port {reference.port!r}")
            previous = next((item for item in sequence.delays if item.port == reference.port), None)
            target_unit = previous.unit if previous is not None else unit
            authored = convert_time(value, unit, target_unit)
            if authored == (previous.value if previous is not None else 0):
                continue
            authored = _number_for(align_to_grid(
                authored, target_unit, sequence.time_step_ns, field_name, minimum=None
            ))
            if authored != (previous.value if previous is not None else 0):
                delay_changes[reference.port] = OutputDelay(reference.port, authored, target_unit)
            continue

        period = sequence.period_by_id.get(str(reference.period_id))
        if period is None:
            raise ValueError(f"{field_name!r} names no period on this sequence")
        if reference.kind == FIELD_DURATION:
            authored = convert_time(value, unit, period.unit)
            if authored == period.duration:
                continue
            if authored <= 0:
                raise ValueError(f"{field_name!r} must be a positive duration, not {value} {unit}")
            authored = _number_for(align_to_grid(
                authored, period.unit, sequence.time_step_ns, field_name
            ))
            if authored != period.duration:
                period_changes.setdefault(period.period_id, {})["duration"] = authored
            continue

        if unit != "value":
            raise ValueError("DAC fields use unit 'value'")
        at = next((i for i, step in enumerate(period.analog_steps) if step.port == reference.port), None)
        if at is None:
            raise ValueError(f"{field_name!r} names no DAC step in period {period.period_id!r}")
        authored = int(round(float(value)))
        if authored != period.analog_steps[at].value:
            changes = period_changes.setdefault(period.period_id, {})
            if "analog_steps" not in changes:
                changes["analog_steps"] = list(period.analog_steps)
            changes["analog_steps"][at] = replace(period.analog_steps[at], value=authored)

    changes: dict[str, object] = {}
    if period_changes:
        changes["periods"] = tuple(
            replace(period, **period_changes[period.period_id])
            if period.period_id in period_changes else period
            for period in sequence.periods
        )
    if delay_changes:
        changes["delays"] = tuple(
            delay_changes.pop(delay.port, delay) for delay in sequence.delays
        ) + tuple(delay_changes.values())
    if remove_api_parameters and sequence.api_bindings:
        changes["bindings"] = tuple(
            replace(binding, source=BINDING_DEFAULT)
            if binding.source == BINDING_API else binding
            for binding in sequence.bindings
        )
    return replace(sequence, **changes) if changes else sequence


def prune_orphaned_bindings(
    sequence: PulseSequence,
) -> tuple[PulseSequence, tuple[str, ...]]:
    """Drop every binding whose field the pulse no longer has, and name them.

    A binding is a statement ABOUT a field, so it cannot outlive one.  A DAC
    field exists only while its period carries a step on that port, and the
    gestures that take a step away -- clearing a port, choosing Hold -- used
    to leave the binding behind.  Every reader then raised on a pulse that
    looked fine on screen: reading the value of a field that is not there.

    Durations and delays are never orphaned this way (a period always has a
    duration, and a missing delay reads as zero), and the model already
    refuses a binding whose period is gone, so this is the DAC case.
    """

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")

    def held(reference: PulseFieldRef) -> bool:
        if reference.kind != FIELD_DAC:
            return True
        period = sequence.period_by_id.get(str(reference.period_id))
        return period is not None and any(
            step.port == reference.port for step in period.analog_steps
        )

    retained = tuple(binding for binding in sequence.bindings if held(binding.field_ref))
    dropped = tuple(binding.field_id for binding in sequence.bindings if not held(binding.field_ref))
    return (replace(sequence, bindings=retained), dropped) if dropped else (sequence, ())


def api_bindings_in_period_order(sequence: PulseSequence) -> tuple[PulseBinding, ...]:
    """The API parameters as the pulse plays them, not as somebody declared them.

    By the period each sits on, a duration before that period's DAC fields,
    and an output delay -- which sits on no period -- last.  A caller that
    wants "the first API field" therefore gets the first one in time.
    """

    positions = {period.period_id: index for index, period in enumerate(sequence.periods)}
    kind_rank = {FIELD_DURATION: 0, FIELD_DAC: 1, FIELD_DELAY: 2}

    def played_at(binding: PulseBinding) -> tuple[int, int, str]:
        reference = binding.field_ref
        position = (
            len(positions)
            if reference.kind == FIELD_DELAY
            else positions.get(reference.period_id, len(positions))
        )
        return (position, kind_rank.get(reference.kind, 3), str(reference.port or ""))

    return tuple(sorted(sequence.api_bindings, key=played_at))


def field_label(sequence: PulseSequence, reference: PulseFieldRef) -> str:
    """One physical field, said the way it reads on the pulse.

    A binding is identified by ``(kind, period_id, port)``, none of which an
    operator chose to read: the period may carry a name and the port a label
    from the board manifest, and those are what is on screen.  Every surface
    that lists bindings prints this, so a parameter is the same words in the
    editor, in a node form and in an error.
    """

    _check_inputs(sequence, reference)
    if reference.kind == FIELD_DELAY:
        return f"{_port_text(sequence, reference.port)}.delay"
    period = sequence.period_by_id.get(str(reference.period_id))
    period_text = (
        str(reference.period_id) if period is None else (period.name or period.period_id)
    )
    if reference.kind == FIELD_DURATION:
        return f"{period_text}.duration"
    return f"{period_text}.{_port_text(sequence, reference.port)}"


def _port_text(sequence: PulseSequence, port: str | None) -> str:
    spec = sequence.target.by_key.get(str(port))
    return str(port) if spec is None else (spec.label or spec.key)


def authored_api_entries(sequence: PulseSequence) -> dict[str, tuple[float, str]]:
    """Every API parameter as ``(value, unit)``, both the pulse's own.

    A value alone cannot be saved or handed to another pulse: a duration is
    only a number beside the unit its author chose.
    """

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    entries: dict[str, tuple[float, str]] = {}
    for parameter in sequence.api_bindings:
        try:
            value = pulse_field_value(
                sequence, parameter.field_ref, parameter.unit
            )
        except ValueError as error:
            # Say WHICH binding has no field.  "period 'load' has no DAC step
            # on port 'da_dipole'" is true and unactionable on its own; the
            # operator needs the name they can see on the Scan page.
            raise ValueError(
                f"API parameter {parameter.field_id!r} has no field: {error}"
            ) from None
        entries[parameter.field_id] = (float(value), parameter.unit)
    return entries


def authored_api_values(sequence: PulseSequence) -> dict[str, float]:
    """Every API parameter at the value the pulse itself carries.

    What ``On Pulse`` runs, and the starting point for a caller that owns only
    some of the parameters: it overrides those and leaves the operator's own
    numbers standing for the rest.
    """

    return {
        parameter_id: value
        for parameter_id, (value, _unit) in authored_api_entries(sequence).items()
    }


def normalize_binding_values(
    sequence: PulseSequence,
    entries: Mapping[str, object],
    *,
    source: str = BINDING_API,
) -> dict[str, object]:
    """Resolve stable field IDs or exact displayed paths once at a public boundary."""

    _sequence_of(sequence)
    if not isinstance(entries, Mapping):
        raise TypeError("binding values must be a mapping")
    declared = sequence.scan_bindings if source == "scan" else tuple(
        binding for binding in sequence.bindings if binding.source == source
    )
    ids = {binding.field_id for binding in declared}
    labels: dict[str, list[str]] = {}
    for binding in declared:
        labels.setdefault(field_label(sequence, binding.field_ref), []).append(binding.field_id)
    resolved: dict[str, object] = {}
    for name, value in entries.items():
        if name in ids:
            key = name
        else:
            matches = labels.get(name, ())
            if len(matches) != 1:
                raise ValueError(f"unknown or ambiguous {source} field {name!r}")
            key = matches[0]
        if key in resolved:
            raise ValueError(f"duplicate {source} field {key!r}")
        resolved[key] = value
    return resolved


def apply_config_values(
    sequence: PulseSequence,
    entries: Mapping[str, tuple[int | float, str]],
    *,
    current: PulseSequence | None = None,
) -> tuple[PulseSequence, tuple[str, ...], tuple[str, ...]]:
    """Apply saved names to all matching non-scanned fields, restoring defaults.

    A shared Config key may feed several physical fields. A scan column owns
    its field for this execution; clearing that column restores normal source
    resolution, not another copy of a default value.
    """

    _sequence_of(sequence)
    if not isinstance(entries, Mapping):
        raise TypeError("config values must be a mapping")
    for key in entries:
        config_parameter_key(key)
    destination = sequence if current is None else _sequence_of(current)
    declared = tuple(binding for binding in destination.config_bindings if not binding.scan)
    fields = []
    applied = []
    names = {binding.config_key for binding in declared if binding.config_key}
    for binding in declared:
        key = binding.config_key
        if key and key in entries:
            number, unit = entries[key]
            _check_value_unit(binding.unit, unit, f"Config value {key!r}")
            fields.append((binding.field_ref, number, unit, key))
            if key not in applied:
                applied.append(key)
        elif destination is not sequence:
            fields.append((
                binding.field_ref,
                pulse_field_value(sequence, binding.field_ref, binding.unit),
                binding.unit,
                binding.field_id,
            ))
    return (
        _replace_pulse_fields(destination, fields),
        tuple(applied),
        tuple(key for key in entries if key not in names),
    )


def _sequence_of(sequence: PulseSequence) -> PulseSequence:
    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    return sequence


def _check_value_unit(declared: str, supplied: str, label: str) -> None:
    if (declared == "value") != (supplied == "value"):
        raise ValueError(f"{label} is in {supplied!r} where the pulse declares {declared!r}")


def apply_api_values(
    sequence: PulseSequence,
    entries: Mapping[str, tuple[int | float, str]],
) -> tuple[PulseSequence, tuple[str, ...], tuple[str, ...]]:
    """Apply explicit API field values; omitted inputs retain their defaults."""

    normalized = normalize_binding_values(sequence, entries)
    fields = []
    for binding in sequence.api_bindings:
        if binding.field_id in normalized:
            number, unit = normalized[binding.field_id]
            _check_value_unit(binding.unit, unit, f"API value {binding.field_id!r}")
            fields.append((binding.field_ref, number, unit, binding.field_id))
    return _replace_pulse_fields(sequence, fields), tuple(normalized), ()


def resolve_api_parameters(
    sequence: PulseSequence,
    values: Mapping[str, int | float] | None = None,
) -> PulseSequence:
    """Bake supplied API inputs/defaults and clear only the API source flag."""

    _sequence_of(sequence)
    resolved_values = authored_api_values(sequence)
    if values is not None:
        resolved_values.update(normalize_binding_values(sequence, values))
    return _replace_pulse_fields(
        sequence,
        ((binding.field_ref, resolved_values[binding.field_id], binding.unit,
          binding.field_id) for binding in sequence.api_bindings),
        remove_api_parameters=True,
    )


def _check_inputs(sequence: PulseSequence, reference: PulseFieldRef) -> None:
    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    if not isinstance(reference, PulseFieldRef):
        raise TypeError("reference must be PulseFieldRef")


def convert_time(value: int | float, source_unit: str, target_unit: str) -> float:
    """One duration, said in another unit.

    Published because a caller that owns a duration in SI seconds has to write
    it into a field declared in whatever unit its author chose.  A parameter
    is written in ITS OWN unit, so a task that passed seconds to a slot
    declared in microseconds wrote a number a million times too small and
    nothing said so.
    """

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("time value must be finite")
    source_scale = nanoseconds_per(source_unit)
    if source_unit == target_unit:
        return number
    target_scale = nanoseconds_per(target_unit)
    if source_scale == target_scale:
        return number
    return float(Fraction(str(number)) * source_scale / target_scale)


def _number_for(value: float) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


__all__ = [
    "apply_api_values",
    "apply_config_values",
    "authored_api_entries",
    "authored_api_values",
    "convert_time",
    "field_label",
    "normalize_binding_values",
    "config_parameter_key",
    "prune_orphaned_bindings",
    "pulse_field_value",
    "replace_pulse_field",
    "resolve_api_parameters",
]
