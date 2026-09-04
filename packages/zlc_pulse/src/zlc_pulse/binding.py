"""Resolve named pulse inputs into the physical fields they own."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from .model import (
    FIELD_DAC,
    FIELD_DELAY,
    FIELD_DURATION,
    TIME_UNIT_TO_NS,
    OutputDelay,
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
    """Return ``sequence`` with one referenced physical field replaced."""

    _check_inputs(sequence, reference)
    if reference.kind == FIELD_DELAY:
        index = next(
            (i for i, item in enumerate(sequence.delays) if item.port == reference.port),
            None,
        )
        if index is None:
            authored = align_to_grid(
                value,
                unit,
                float(sequence.time_step_ns),
                field_name,
                minimum=None,
            )
            return replace(
                sequence,
                delays=sequence.delays + (
                    OutputDelay(reference.port, _number_for(authored), unit),
                ),
            )
        delays = list(sequence.delays)
        authored = convert_time(value, unit, delays[index].unit)
        delays[index] = replace(
            delays[index],
            value=_number_for(
                align_to_grid(
                    authored,
                    delays[index].unit,
                    float(sequence.time_step_ns),
                    field_name,
                    minimum=None,
                )
            ),
        )
        return replace(sequence, delays=tuple(delays))

    index = next(
        (i for i, item in enumerate(sequence.periods)
         if item.period_id == reference.period_id),
        None,
    )
    if index is None:
        raise ValueError(f"{field_name!r} names no period on this sequence")
    periods = list(sequence.periods)
    period = periods[index]
    if reference.kind == FIELD_DURATION:
        authored = convert_time(value, unit, period.unit)
        periods[index] = replace(
            period,
            duration=_number_for(
                align_to_grid(
                    authored,
                    period.unit,
                    float(sequence.time_step_ns),
                    field_name,
                )
            ),
        )
        return replace(sequence, periods=tuple(periods))

    if unit != "value":
        raise ValueError("DAC fields use unit 'value'")
    steps = list(period.analog_steps)
    at = next(
        (i for i, step in enumerate(steps) if step.port == reference.port), None
    )
    if at is None:
        raise ValueError(
            f"{field_name!r} names no DAC step in period {period.period_id!r}"
        )
    steps[at] = replace(steps[at], value=int(round(float(value))))
    periods[index] = replace(period, analog_steps=tuple(steps))
    return replace(sequence, periods=tuple(periods))


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

    slots = tuple(slot for slot in sequence.slots if held(slot.field_ref))
    parameters = tuple(
        parameter
        for parameter in sequence.api_parameters
        if held(parameter.field_ref)
    )
    configured = tuple(
        parameter
        for parameter in sequence.config_parameters
        if held(parameter.field_ref)
    )
    dropped = tuple(
        [slot.slot_id for slot in sequence.slots if not held(slot.field_ref)]
        + [
            parameter.parameter_id
            for parameter in sequence.api_parameters
            if not held(parameter.field_ref)
        ]
        + [
            parameter.parameter_id
            for parameter in sequence.config_parameters
            if not held(parameter.field_ref)
        ]
    )
    if not dropped:
        return sequence, ()
    return (
        replace(
            sequence,
            slots=slots,
            api_parameters=parameters,
            config_parameters=configured,
        ),
        dropped,
    )


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
        return f"{_port_text(sequence, reference.port)} delay"
    period = sequence.period_by_id.get(str(reference.period_id))
    period_text = (
        str(reference.period_id) if period is None else (period.name or period.period_id)
    )
    if reference.kind == FIELD_DURATION:
        return f"{period_text} duration"
    return f"{period_text} · {_port_text(sequence, reference.port)}"


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
    for parameter in sequence.api_parameters:
        try:
            value = pulse_field_value(
                sequence, parameter.field_ref, parameter.unit
            )
        except ValueError as error:
            # Say WHICH binding has no field.  "period 'load' has no DAC step
            # on port 'da_dipole'" is true and unactionable on its own; the
            # operator needs the name they can see on the Scan page.
            raise ValueError(
                f"API parameter {parameter.parameter_id!r} has no field: {error}"
            ) from None
        entries[parameter.parameter_id] = (float(value), parameter.unit)
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


def authored_config_entries(sequence: PulseSequence) -> dict[str, tuple[float, str]]:
    """Every config parameter as ``(value, unit)``, both the pulse's own.

    A config parameter is never a hole, so this is simply what the pulse will
    play unless its config file says otherwise -- and after a refresh it IS
    what the file said, because a refresh overwrites the authored number.
    """

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    entries: dict[str, tuple[float, str]] = {}
    for parameter in sequence.config_parameters:
        try:
            value = pulse_field_value(
                sequence, parameter.field_ref, parameter.unit
            )
        except ValueError as error:
            raise ValueError(
                f"config parameter {parameter.parameter_id!r} has no field: {error}"
            ) from None
        entries[parameter.parameter_id] = (float(value), parameter.unit)
    return entries


def apply_config_values(
    sequence: PulseSequence,
    entries: Mapping[str, tuple[int | float, str]],
) -> tuple[PulseSequence, tuple[str, ...], tuple[str, ...]]:
    """Overwrite the authored value of every config parameter the set names.

    THE OVERWRITE IS THE STORAGE.  A config parameter keeps no value of its
    own beside the field: refreshing one writes the number into the period's
    duration, the DAC step or the delay it names, so what the pulse holds
    afterwards is what the file said, and a pulse read back later needs no
    second file to be understood.

    Returns the sequence, the ids applied, and the ids the set named that this
    pulse does not declare -- one calibrated set is meant to serve several
    pulses, most of which declare only part of it.
    """

    return _apply_named_values(
        sequence,
        entries,
        {
            parameter.parameter_id: parameter
            for parameter in _sequence_of(sequence).config_parameters
        },
        "config value",
    )


def _sequence_of(sequence: PulseSequence) -> PulseSequence:
    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    return sequence


def _apply_named_values(
    sequence: PulseSequence,
    entries: Mapping[str, tuple[int | float, str]],
    declared: Mapping[str, object],
    label: str,
) -> tuple[PulseSequence, tuple[str, ...], tuple[str, ...]]:
    """Write one set of named numbers into the fields their names point at."""

    if not isinstance(entries, Mapping):
        raise TypeError(f"{label}s must be a mapping")
    result = sequence
    applied: list[str] = []
    unknown: list[str] = []
    for parameter_id, entry in entries.items():
        parameter = declared.get(str(parameter_id))
        if parameter is None:
            unknown.append(str(parameter_id))
            continue
        number, unit = entry
        if parameter.unit == "value" or unit == "value":
            if parameter.unit != unit:
                raise ValueError(
                    f"{label} {parameter_id!r} is in {unit!r} where the pulse "
                    f"declares {parameter.unit!r}"
                )
            authored = float(number)
        else:
            authored = convert_time(number, unit, parameter.unit)
        result = replace_pulse_field(
            result,
            parameter.field_ref,
            authored,
            parameter.unit,
            field_name=parameter.parameter_id,
        )
        applied.append(parameter.parameter_id)
    return result, tuple(applied), tuple(unknown)


def apply_api_values(
    sequence: PulseSequence,
    entries: Mapping[str, tuple[int | float, str]],
) -> tuple[PulseSequence, tuple[str, ...], tuple[str, ...]]:
    """Overwrite the authored value of every API parameter the set names.

    Returns the sequence, the ids applied and the ids the set named that this
    pulse does not declare.  The intersection is applied rather than the whole
    set demanded: one saved set of bias values is meant to be carried across
    several pulses, most of which declare only some of it, so an id this pulse
    has never heard of is reported and skipped.  An id this pulse declares and
    the set omits keeps the number the operator already authored.

    The declarations themselves survive: this changes what the API slots hold,
    not whether they exist.  Baking them away is :func:`resolve_api_parameters`.
    """

    return _apply_named_values(
        sequence,
        entries,
        {
            parameter.parameter_id: parameter
            for parameter in _sequence_of(sequence).api_parameters
        },
        "API value",
    )


def resolve_api_parameters(
    sequence: PulseSequence,
    values: Mapping[str, int | float] | None = None,
) -> PulseSequence:
    """Bake every API parameter into its field and remove the declarations.

    With no explicit mapping, the currently authored field values are used.
    That is the Pulse Editor's ``On Pulse`` meaning: run exactly what is shown.
    An explicit mapping must name every declared parameter, so misspellings and
    partially configured runs do not silently execute nominal values.  A caller
    that owns only some of them starts from :func:`authored_api_values` and
    overrides its own, which says the same thing at the call site.
    """

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    expected = tuple(parameter.parameter_id for parameter in sequence.api_parameters)
    if values is None:
        resolved_values = authored_api_values(sequence)
    else:
        if not isinstance(values, Mapping):
            raise TypeError("API parameter values must be a mapping")
        resolved_values = dict(values)
        missing = tuple(
            parameter_id
            for parameter_id in expected
            if parameter_id not in resolved_values
        )
        extra = tuple(
            parameter_id
            for parameter_id in resolved_values
            if parameter_id not in expected
        )
        if missing or extra:
            raise ValueError(
                "API parameter values must exactly match the pulse declaration; "
                f"missing={missing}, extra={extra}"
            )

    result = sequence
    for parameter in sequence.api_parameters:
        result = replace_pulse_field(
            result,
            parameter.field_ref,
            resolved_values[parameter.parameter_id],
            parameter.unit,
            field_name=parameter.parameter_id,
        )
    return replace(result, api_parameters=())


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

    if source_unit not in TIME_UNIT_TO_NS or target_unit not in TIME_UNIT_TO_NS:
        raise ValueError("time fields require time units")
    return float(value) * TIME_UNIT_TO_NS[source_unit] / TIME_UNIT_TO_NS[target_unit]


def _number_for(value: float) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


__all__ = [
    "apply_api_values",
    "apply_config_values",
    "authored_api_entries",
    "authored_api_values",
    "authored_config_entries",
    "convert_time",
    "field_label",
    "prune_orphaned_bindings",
    "pulse_field_value",
    "replace_pulse_field",
    "resolve_api_parameters",
]
