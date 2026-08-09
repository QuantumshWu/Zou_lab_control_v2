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

from collections.abc import Mapping, Sequence

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_ui import FormChoice, FormFieldProps, FormSpec


__all__ = [
    "display_value",
    "project_artifact_inputs",
    "project_logic_schema",
    "project_schema",
]


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


def project_logic_schema(
    descriptor: object,
    *,
    resource_choices: Mapping[str, Sequence[str]],
    resource_errors: Mapping[str, str],
) -> FormSpec:
    """Project a logic schema, resolving explicit workspace-resource fields."""

    schema = getattr(descriptor, "authoring_schema", None)
    if not isinstance(schema, AuthoringSchema):
        raise TypeError("project_logic_schema needs a logic descriptor")
    resources = {
        str(spec.field_name): spec
        for spec in getattr(descriptor, "workspace_resources", ())
    }
    fields: list[FormFieldProps] = []
    for field in schema.fields:
        if field.value_type != "resource":
            fields.append(_project_field(field))
            continue
        if field.name not in resources:
            raise ValueError(
                f"resource field {field.name!r} has no WorkspaceResourceSpec"
            )
        choices = tuple(
            FormChoice(str(value), str(value))
            for value in resource_choices.get(field.name, ())
        )
        fields.append(
            _project_resource_field(
                field,
                choices=choices,
                unavailable_reason=str(resource_errors.get(field.name, "")),
            )
        )
    return FormSpec(tuple(fields))


def project_artifact_inputs(
    specs: Sequence[object],
    *,
    base_dir: str,
) -> FormSpec:
    """Declared artifact paths using each domain codec's picker contract."""

    return FormSpec(
        tuple(
            FormFieldProps(
                key=str(spec.name),
                kind="path",
                label=str(spec.label),
                default="",
                required=bool(spec.required),
                description=f"Artifact contract: {spec.contract_id}",
                file_filter=str(spec.codec.file_filter),
                base_dir=str(base_dir),
            )
            for spec in specs
        )
    )


def _project_field(field: AuthoringField) -> FormFieldProps:
    if str(field.value_type) == "resource":
        raise ValueError(
            f"resource field {field.name!r} requires its "
            "WorkspaceResourceSpec projection"
        )
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
        choices=tuple(
            FormChoice(choice.label, choice.value) for choice in field.choices
        ),
        description=(
            "two integers as y, x" if str(field.value_type) == "pair" else ""
        ),
    )


def _project_resource_field(
    field: AuthoringField,
    *,
    choices: tuple[FormChoice, ...],
    unavailable_reason: str,
) -> FormFieldProps:
    if field.choices:
        raise ValueError(
            f"resource field {field.name!r} cannot declare static choices"
        )
    default = (
        field.default
        if any(choice.value == field.default for choice in choices)
        else None
    )
    if not choices and not unavailable_reason.strip():
        unavailable_reason = "no valid workspace resources available"
    return FormFieldProps(
        key=field.name,
        kind="choice",
        label=field.label,
        default=default,
        required=True,
        choices=choices,
        description="Workspace resource",
        unavailable_reason=unavailable_reason if not choices else "",
    )


def display_value(value: object) -> object:
    """A stored value as the form shows it."""

    if isinstance(value, (tuple, list)):
        return ", ".join(str(item) for item in value)
    return value
