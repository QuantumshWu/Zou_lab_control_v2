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
from dataclasses import replace
from pathlib import Path

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
    "numeric_tuple": "text",
    #: A pair is two integers, edited as the text "y, x".  It stays one field
    #: because it is one fact -- a frame shape with a half-edited width is not a
    #: state worth being able to reach.
    "pair": "text",
    #: A directory the operator picks, not a file: saved samples live in one.
    "folder": "path",
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
    workspace_root: str,
    field_availability: Mapping[str, str] | None = None,
) -> FormSpec:
    """Project a logic schema, rooting resource pickers in the workspace.

    ``field_availability`` disables settings this draft's bound devices cannot
    take and puts the reason on the control.  An unavailable boolean is shown
    at its neutral effective value, False, without mutating the authored
    draft; other unavailable authored values still participate in Start
    admission.
    """

    schema = getattr(descriptor, "authoring_schema", None)
    if not isinstance(schema, AuthoringSchema):
        raise TypeError("project_logic_schema needs a logic descriptor")
    unavailable = dict(field_availability or {})
    unknown = set(unavailable) - set(schema.field_names)
    if unknown:
        raise ValueError(f"resolved availability names no such field: {sorted(unknown)!r}")
    resources = {
        str(spec.field_name): spec
        for spec in getattr(descriptor, "workspace_resources", ())
    }
    fields: list[FormFieldProps] = []
    for field in schema.fields:
        reason = str(unavailable.get(field.name, ""))
        if reason:
            fields.append(replace(_project_field(field), unavailable_reason=reason))
            continue
        if field.value_type == "folder":
            # A folder is chosen from the workspace, so it OPENS there and is
            # pre-filled with the declared place rather than with nothing: an
            # empty box makes the operator find a path the workspace already
            # knows.
            fields.append(_project_folder_field(field, workspace_root=workspace_root))
            continue
        if field.value_type != "resource":
            fields.append(_project_field(field))
            continue
        if field.name not in resources:
            raise ValueError(
                f"resource field {field.name!r} has no WorkspaceResourceSpec"
            )
        fields.append(
            _project_resource_field(
                field,
                resources[field.name],
                workspace_root=workspace_root,
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
        # A defaultless numeric STARTS vacant in a draft, required or not:
        # the widget must be able to hold that vacancy, because completeness
        # is Init's law, not the editor's.  (Text kinds hold "" natively.)
        allow_blank=(
            True
            if field.default is None and kind in ("int", "float", "number")
            else None
        ),
        minimum=field.minimum,
        maximum=field.maximum,
        choices=tuple(
            FormChoice(choice.label, choice.value) for choice in field.choices
        ),
        description={
            "pair": "two integers as y, x",
            "numeric_tuple": "comma-separated finite numbers",
        }.get(str(field.value_type), ""),
        path_mode="dir" if str(field.value_type) == "folder" else "file",
        enabled_when=field.enabled_when,
    )


def _project_folder_field(
    field: AuthoringField,
    *,
    workspace_root: str,
) -> FormFieldProps:
    root = Path(workspace_root).expanduser().resolve()
    declared = str(field.default or "").strip()
    default = str((root / declared).resolve()) if declared else str(root)
    return FormFieldProps(
        key=field.name,
        kind="path",
        label=field.label,
        default=default,
        required=bool(field.required),
        description=f"Choose a folder under {root}",
        path_mode="dir",
        base_dir=str(root),
        enabled_when=field.enabled_when,
    )


def _project_resource_field(
    field: AuthoringField,
    spec: object,
    *,
    workspace_root: str,
) -> FormFieldProps:
    if field.choices:
        raise ValueError(
            f"resource field {field.name!r} cannot declare static choices"
        )
    directory = (
        Path(workspace_root).expanduser().resolve()
        / str(getattr(spec, "directory"))
    ).resolve()
    suffixes = tuple(str(value) for value in getattr(spec, "suffixes"))
    patterns = " ".join(f"*{suffix}" for suffix in suffixes)
    default = str((directory / str(field.default)).resolve()) if field.default else ""
    return FormFieldProps(
        key=field.name,
        kind="path",
        label=field.label,
        default=default,
        required=True,
        description=f"Choose a file from {directory}",
        file_filter=f"{field.label} ({patterns})",
        base_dir=str(directory),
    )


def display_value(value: object) -> object:
    """A stored value as the form shows it."""

    if isinstance(value, (tuple, list)):
        return ", ".join(str(item) for item in value)
    return value
