"""Frontend-neutral mapping from plot parameter schemas to editor controls.

The plot-kind schema remains authoritative for names, validation and render
effects.  Frontends consume this module to choose a widget without duplicating
those rules; no Qt, Jupyter or Matplotlib object is imported here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from .parameters import ParameterSchema, RenderEffect
from .semantics import SemanticDescription


class ControlKind(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    TEXT = "text"
    CHOICE = "choice"


@dataclass(frozen=True, slots=True)
class ParameterControl:
    """One immutable, toolkit-independent parameter editor description."""

    name: str
    label: str
    kind: ControlKind
    value: object
    allow_none: bool
    choices: tuple[object, ...]
    minimum: float | None
    maximum: float | None
    step: float | None
    effects: RenderEffect
    rebuild: bool = False
    semantic: bool = False
    automatic: bool = False
    unavailable_reason: str = ""


def parameter_controls(
    schema: ParameterSchema,
    values: Mapping[str, object],
    *,
    choice_overrides: Mapping[str, tuple[object, ...]] | None = None,
) -> tuple[ParameterControl, ...]:
    """Project one canonical schema/state pair into ordered UI controls.

    ``choice_overrides`` supplies data- or environment-dependent editor
    domains, such as compatible units and the package-defined colormap
    catalogue. It changes only the editor choices; the core schema still
    validates every submitted value.
    """

    if not isinstance(schema, ParameterSchema):
        raise TypeError("schema must be ParameterSchema")
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
    if choice_overrides is not None and not isinstance(choice_overrides, Mapping):
        raise TypeError("choice_overrides must be a mapping or None")
    overrides = {} if choice_overrides is None else dict(choice_overrides)
    unknown = tuple(name for name in overrides if name not in schema)
    if unknown:
        joined = ", ".join(repr(name) for name in unknown)
        raise KeyError(f"choice override refers to unknown parameter(s): {joined}")
    result = []
    fixed_limits = values.get("relim_mode") == "fixed"
    limit_names = {"color_min", "color_max", "y_min", "y_max"}
    for name, spec in schema.items():
        if name not in values:
            raise KeyError(f"display state is missing parameter {name!r}")
        choices = tuple(overrides.get(name, spec.choices))
        kind = _control_kind(spec.value_type, choices)
        limit_field = name in limit_names and "relim_mode" in values
        result.append(
            ParameterControl(
                name=name,
                label=str(spec.label),
                kind=kind,
                value=values[name],
                allow_none=spec.allow_none,
                choices=choices,
                minimum=spec.minimum,
                maximum=spec.maximum,
                step=spec.step,
                effects=spec.effects,
                automatic=spec.allow_none and not limit_field,
                unavailable_reason=(
                    "Choose Fixed limits to edit."
                    if limit_field and not fixed_limits
                    else ""
                ),
            )
        )
    return tuple(result)


def parameter_controls_for_kind(
    kind: object,
    values: Mapping[str, object] | None = None,
    *,
    facet_cell_kind: object | None = None,
) -> tuple[ParameterControl, ...]:
    """Describe a blank authored kind without constructing data or a host.

    The same schema builder serves a live :class:`PlotSession`; this entry
    point only seeds that schema from its defaults plus authored overrides.
    Semantic axis choices and fit models remain dataset-dependent.
    """

    from .config import DEFAULTS
    from .specs import parameter_schema_for_kind

    schema = parameter_schema_for_kind(
        kind,
        style=DEFAULTS.style,
        facet_cell_kind=facet_cell_kind,
    )
    # ``values`` is a persisted panel appearance, possibly authored under a
    # different kind's vocabulary; describing takes the declared subset.
    state = schema.initial_values(
        None if values is None else schema.declared_subset(values)
    )
    return parameter_controls(schema, state)


def semantic_controls(
    description: SemanticDescription,
) -> tuple[ParameterControl, ...]:
    """Project registry-derived semantic fields into the same UI contract.

    Semantic controls deliberately carry ``rebuild=True`` and a layout effect;
    a frontend can therefore route them to ``replace_spec`` without guessing
    whether a cheap display-parameter update is safe.
    """

    if not isinstance(description, SemanticDescription):
        raise TypeError("description must be SemanticDescription")
    result = []
    for field in description.fields:
        choices = tuple(field.choices)
        allow_none = not field.required or any(
            choice[0] is None for choice in choices
        )
        result.append(
            ParameterControl(
                name=field.name,
                label=field.label,
                kind=ControlKind.CHOICE,
                value=field.value,
                allow_none=allow_none,
                choices=choices,
                minimum=None,
                maximum=None,
                step=None,
                effects=RenderEffect.LAYOUT,
                rebuild=field.rebuild,
                semantic=True,
            )
        )
    return tuple(result)


def _control_kind(value_type: object, choices: tuple[object, ...]) -> ControlKind:
    if choices:
        return ControlKind.CHOICE
    types = (value_type,) if isinstance(value_type, type) else tuple(value_type)
    if bool in types:
        return ControlKind.BOOLEAN
    if types == (int,):
        return ControlKind.INTEGER
    if any(item in (int, float) for item in types):
        return ControlKind.NUMBER
    if str in types:
        return ControlKind.TEXT
    raise TypeError("parameter value type has no standard UI control mapping")


__all__ = [
    "ControlKind",
    "ParameterControl",
    "parameter_controls",
    "parameter_controls_for_kind",
    "semantic_controls",
]
