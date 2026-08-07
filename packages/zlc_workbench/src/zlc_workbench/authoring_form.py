"""Turning a declaration of what something needs told into the form that tells it.

Devices and logic nodes both declare their settings the same way -- one
``AuthoringSchema`` of named, typed, bounded fields -- and both are edited by the
same Fluent form.  The two vocabularies belong to different packages and neither
may import the other, so this is where they meet, once.

It is the only place a declared value type becomes a widget kind.  A type either
has a translation here or is refused with its name in the message; the failure
worth preventing is the quiet one, where an unknown type renders as a text box
and saves a string where the device expected a number.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_ui import FormChoice, FormFieldProps, FormSpec


__all__ = ["display_value", "project_schema", "project_values"]


#: How a declared value type is edited.
_FIELD_KINDS = {
    "int": "int",
    "float": "float",
    "str": "text",
    "text": "text",
    "bool": "bool",
    "choice": "choice",
    #: A pair is two integers, edited as the text "y, x".  It stays one field
    #: because it is one fact -- a frame shape with a half-edited width is not a
    #: state worth being able to reach.
    "pair": "text",
}


def project_schema(schema: AuthoringSchema) -> FormSpec:
    """One declaration, as the form that edits it.

    Bounds come across as bounds rather than as advice: the schema already knows
    that a seed cannot be negative, and a form that lets one be typed only to
    have the save refused has taught the operator nothing.
    """

    if not isinstance(schema, AuthoringSchema):
        raise TypeError("project_schema needs an AuthoringSchema")
    return FormSpec(tuple(_project_field(field) for field in schema.fields))


def _project_field(field: AuthoringField) -> FormFieldProps:
    kind = _FIELD_KINDS.get(str(field.value_type))
    if kind is None:
        raise ValueError(
            f"field {field.name!r} is declared as {field.value_type!r}, "
            "which no widget here knows how to edit"
        )
    if field.choices:
        kind = "choice"
    return FormFieldProps(
        key=field.name,
        kind=kind,
        label=field.label,
        default=display_value(field.default),
        required=bool(field.required),
        minimum=field.minimum,
        maximum=field.maximum,
        choices=tuple(FormChoice(str(item), str(item)) for item in field.choices),
        description=(
            "two integers as y, x" if str(field.value_type) == "pair" else ""
        ),
    )


def display_value(value: object) -> object:
    """A stored value as the form shows it."""

    if isinstance(value, (tuple, list)):
        return ", ".join(str(item) for item in value)
    return value


def project_values(
    schema: AuthoringSchema,
    edited: Mapping[str, Any],
) -> dict[str, Any]:
    """Edited fields back in the types their owner declared, then frozen.

    Frozen by the schema rather than accepted field by field: an exposure that
    is legal alone can be illegal beside a readout speed, and a half-applied
    settings change is not a state worth being able to reach.
    """

    by_name = {field.name: field for field in schema.fields}
    values: dict[str, Any] = {}
    for name, value in dict(edited).items():
        field = by_name.get(str(name))
        if field is not None:
            values[str(name)] = _project_value(field, value)
    return schema.freeze(values)


def _project_value(field: AuthoringField, value: object) -> object:
    declared = str(field.value_type)
    if declared == "pair":
        if isinstance(value, (tuple, list)):
            parts = list(value)
        else:
            parts = [piece for piece in str(value).replace(",", " ").split() if piece]
        if len(parts) != 2:
            raise ValueError(f"{field.label} needs two numbers, as y, x")
        return [int(str(part).strip()) for part in parts]
    if declared == "int":
        return int(value)
    if declared == "float":
        return float(value)
    if declared == "bool":
        return bool(value)
    return value
