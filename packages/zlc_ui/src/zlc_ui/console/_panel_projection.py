"""UI-only helpers shared by a panel's Setting and Edit projections."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import json
from typing import Any

from zlc_ui.board import DEFAULT_PANEL_SIZE
from zlc_ui.form import FormChoice, FormFieldProps, FormSpec


_STATE_KEYS = (
    "signal",
    "kind",
    "size",
    "interval_ms",
    "title",
    "semantic",
    "display",
    "fit",
    "site_overlay",
)


class _ParameterChoice(Enum):
    NONE = "zlc-ui-panel-parameter-none"


def panel_state_document(state: object) -> dict[str, Any]:
    """Return a plain view projection without importing its Workbench owner."""

    if isinstance(state, Mapping):
        incoming = dict(state)
    else:
        document = getattr(state, "document", None)
        if callable(document):
            value = document()
            if not isinstance(value, Mapping):
                raise TypeError("panel state document() must return a mapping")
            incoming = dict(value)
        else:
            incoming = {
                name: getattr(state, name)
                for name in _STATE_KEYS
                if hasattr(state, name)
            }
    return {
        "signal": str(incoming.get("signal") or ""),
        "kind": str(incoming.get("kind") or ""),
        "size": str(incoming.get("size") or DEFAULT_PANEL_SIZE),
        "interval_ms": int(incoming.get("interval_ms") or 100),
        "title": str(incoming.get("title") or incoming.get("signal") or "Panel"),
        "semantic": dict(incoming.get("semantic") or {}),
        "display": dict(incoming.get("display") or {}),
        "fit": dict(incoming.get("fit") or {}),
        "site_overlay": str(incoming.get("site_overlay") or "off"),
    }


def _pretty_key(key: object) -> str:
    return str(key).replace("_", " ").strip().title()


def _json_text(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    except TypeError:
        return str(value)


def mapping_form_spec(values: Mapping[str, object]) -> FormSpec:
    """Build a UI-only scalar editor for an owner-supplied parameter mapping."""

    fields: list[FormFieldProps] = []
    for raw_key, value in values.items():
        key = str(raw_key)
        if isinstance(value, bool):
            kind, default = "bool", value
        elif isinstance(value, int):
            kind, default = "int", value
        elif isinstance(value, float):
            kind, default = "number", value
        elif isinstance(value, str):
            kind, default = "text", value
        else:
            kind, default = "text", "" if value is None else _json_text(value)
        fields.append(
            FormFieldProps(key=key, kind=kind, label=_pretty_key(key), default=default)
        )
    return FormSpec(tuple(fields))


def mapping_form_values(values: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for raw_key, value in values.items():
        key = str(raw_key)
        if value is None:
            result[key] = ""
        elif isinstance(value, (dict, list, tuple)):
            result[key] = _json_text(value)
        else:
            result[key] = value
    return result


def decode_mapping_value(original: object, edited: object) -> object:
    """Recover the JSON-shaped type of a data-driven text field."""

    if original is None:
        text = str(edited).strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if isinstance(original, (dict, list, tuple)):
        try:
            decoded = json.loads(str(edited))
        except json.JSONDecodeError as error:
            raise ValueError(f"{error.msg} at character {error.pos}") from error
        return tuple(decoded) if isinstance(original, tuple) else decoded
    return edited


def parameter_fields(surface: object, section: str) -> tuple[dict[str, object], ...]:
    """Return one owner-described section without importing its plot owner."""

    if not isinstance(surface, Mapping):
        return ()
    entries = surface.get(str(section), ())
    return tuple(dict(entry) for entry in tuple(entries or ()))


def _parameter_choice(value: object) -> object:
    return _ParameterChoice.NONE if value is None else value


def parameter_form_spec(fields: object) -> FormSpec:
    """Turn Workbench's UI-neutral plot controls into a zlc_ui form."""

    projected: list[FormFieldProps] = []
    for incoming in tuple(fields or ()):
        field = dict(incoming)
        key = str(field["key"])
        owner_kind = str(field.get("kind") or "text")
        kind = {
            "boolean": "bool",
            "integer": "int",
            "number": "number",
            "text": "text",
            "choice": "choice",
        }.get(owner_kind, owner_kind)
        value = field.get("value")
        allow_none = bool(field.get("allow_none"))
        choices: tuple[FormChoice, ...] = ()
        if kind == "choice":
            choice_rows = [
                FormChoice(str(label), _parameter_choice(choice_value))
                for label, choice_value in tuple(field.get("choices") or ())
            ]
            if allow_none and not any(
                choice.value is _ParameterChoice.NONE for choice in choice_rows
            ):
                choice_rows.insert(0, FormChoice("(none)", _ParameterChoice.NONE))
            choices = tuple(choice_rows)
            value = _parameter_choice(value)
        elif kind == "text" and value is None:
            value = ""
        minimum = field.get("minimum")
        maximum = field.get("maximum")
        if kind == "int":
            minimum = None if minimum is None else int(minimum)
            maximum = None if maximum is None else int(maximum)
        projected.append(
            FormFieldProps(
                key=key,
                kind=kind,
                label=str(field.get("label") or _pretty_key(key)),
                default=value,
                required=not allow_none,
                minimum=minimum,
                maximum=maximum,
                choices=choices,
                allow_blank=allow_none if kind in {"int", "number", "float"} else None,
            )
        )
    return FormSpec(tuple(projected))


def parameter_form_values(fields: object) -> dict[str, object]:
    return {
        str(field["key"]): (
            _parameter_choice(field.get("value"))
            if str(field.get("kind")) == "choice"
            else ""
            if field.get("value") is None and str(field.get("kind")) == "text"
            else field.get("value")
        )
        for field in (dict(entry) for entry in tuple(fields or ()))
    }


def decode_parameter_value(field: Mapping[str, object], edited: object) -> object:
    """Recover the exact typed value declared by the plot control surface."""

    if edited is _ParameterChoice.NONE:
        return None
    return edited


def signal_form_choices(groups: object, current: str) -> tuple[FormChoice, ...]:
    """Flatten grouped producer choices while preserving their opaque keys."""

    choices: list[FormChoice] = []
    seen: set[str] = set()
    try:
        incoming = tuple(groups or ())
    except TypeError:
        incoming = ()
    for producer, leaves in incoming:
        for display, key in tuple(leaves):
            value = str(key)
            if value in seen:
                continue
            seen.add(value)
            label = f"{str(producer)} · {str(display)}" if producer else str(display)
            choices.append(FormChoice(label, value))
    if current and current not in seen:
        choices.insert(0, FormChoice(f"Unresolved · {current}", current))
    return tuple(choices)


__all__ = [
    "decode_parameter_value",
    "decode_mapping_value",
    "mapping_form_spec",
    "mapping_form_values",
    "panel_state_document",
    "parameter_fields",
    "parameter_form_spec",
    "parameter_form_values",
    "signal_form_choices",
]
