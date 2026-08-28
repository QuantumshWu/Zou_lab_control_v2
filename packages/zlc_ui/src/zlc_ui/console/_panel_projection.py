"""UI-only helpers shared by a panel's Setting and Edit projections."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from zlc_ui.form import FormChoice, FormFieldProps, FormRuntimeContext, FormSpec


_STATE_KEYS = (
    "signal",
    "kind",
    "cell_kind",
    "size",
    "interval_ms",
    "title",
    "semantic",
    "display",
    "fit",
    "overlay_signal",
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
        "cell_kind": str(incoming.get("cell_kind") or ""),
        "size": str(incoming.get("size") or ""),
        "interval_ms": int(incoming.get("interval_ms") or 0),
        "title": str(incoming.get("title") or incoming.get("signal") or "Panel"),
        "semantic": dict(incoming.get("semantic") or {}),
        "display": dict(incoming.get("display") or {}),
        "fit": dict(incoming.get("fit") or {}),
        "overlay_signal": str(incoming.get("overlay_signal") or ""),
    }


def _pretty_key(key: object) -> str:
    return str(key).replace("_", " ").strip().title()


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
        automatic = bool(field.get("automatic"))
        choices: tuple[FormChoice, ...] = ()
        if kind == "choice":
            choice_rows = [
                FormChoice(str(label), _parameter_choice(choice_value))
                for label, choice_value in tuple(field.get("choices") or ())
            ]
            if allow_none and not automatic and not any(
                choice.value is _ParameterChoice.NONE for choice in choice_rows
            ):
                choice_rows.insert(0, FormChoice("(none)", _ParameterChoice.NONE))
            choices = tuple(choice_rows)
            value = value if automatic else _parameter_choice(value)
        elif kind == "text" and value is None and not automatic:
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
                unit=str(field.get("unit") or ""),
                description=str(field.get("description") or ""),
                minimum=minimum,
                maximum=maximum,
                choices=choices,
                allow_blank=allow_none if kind in {"int", "number", "float"} else None,
                unavailable_reason=str(field.get("unavailable_reason") or ""),
                automatic=automatic,
            )
        )
    return FormSpec(tuple(projected))


def parameter_form_values(fields: object) -> dict[str, object]:
    return {
        str(field["key"]): (
            field.get("value")
            if bool(field.get("automatic"))
            else _parameter_choice(field.get("value"))
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


def parameter_edit_values(fields: object, key: str, read_value) -> dict[str, object]:
    """Read the parameter the operator edited, and anything edited WITH it.

    Which parameters have to move together is declared where the parameters
    are and reaches this layer on the descriptor, as ``co_edited_with``.
    This used to recover the relationship from how the names were SPELLED --
    any ``*_min`` paired with its ``*_max`` -- which is the same fact with a
    second, weaker owner.  It agreed with the declaration by luck, and would
    have paired the first ``*_min`` that was not a limit at all.  zlc_plot
    is a forbidden import root here, so the view cannot ask; the seam that
    can see both packages answers instead.

    A limit pair IS validated as a pair, which is why one end cannot go
    alone: moving (0, 10) to (12, 20) passes through (12, 10), and no owner
    accepts that.  So both ends go, as the operator currently sees them.

    A companion that cannot be read is not a reason to lose the edit they
    actually made.  It used to be: with "1e" left in the colour minimum,
    typing 70 into the colour maximum threw the 70 away and put
    "display__color_min must be a finite decimal number" on the card -- an
    error about a field the operator had not touched.
    """

    if not callable(read_value):
        raise TypeError("read_value must be callable")
    declared = {
        str(field["key"]): dict(field) for field in tuple(fields or ())
    }
    selected = str(key)
    if selected not in declared:
        raise KeyError(selected)
    edited = {
        selected: decode_parameter_value(
            declared[selected], read_value(selected)
        )
    }
    companion = str(declared[selected].get("co_edited_with") or "")
    if companion and companion in declared:
        try:
            edited[companion] = decode_parameter_value(
                declared[companion], read_value(companion)
            )
        except (KeyError, TypeError, ValueError):
            # Theirs stands; the companion joins the next edit that can be
            # read.
            pass
    return edited

def signal_form_runtime(groups_for) -> FormRuntimeContext:
    """Project producer groups into the existing keyed tree-choice form seam."""

    def entries(field: str):
        for producer, leaves in tuple(groups_for(str(field)) or ()):
            for display, key in tuple(leaves):
                yield str(producer), str(display), str(key)

    def names(field: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(key for _producer, _display, key in entries(field)))

    def sources() -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for field in ("signal", "overlay_signal"):
            for producer, _display, key in entries(field):
                values = grouped.setdefault(key, [])
                if producer not in values:
                    values.append(producer)
        return {key: tuple(values) for key, values in grouped.items()}

    def labels() -> dict[str, str]:
        return {
            key: display
            for field in ("signal", "overlay_signal")
            for _producer, display, key in entries(field)
        }

    return FormRuntimeContext(
        choice_names=names,
        choice_sources=sources,
        choice_labels=labels,
    )


def interval_form_field(intervals: object, current: object) -> FormFieldProps:
    """Project the scheduler's closed refresh domain as a choice control."""

    values = tuple(int(value) for value in tuple(intervals or ()))
    if not values:
        raise ValueError("panel refresh intervals were not projected")
    if any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("panel refresh intervals must be unique positive integers")
    selected = int(current)
    if selected not in values:
        raise ValueError(
            f"display interval {selected} is not in {values}"
        )
    return FormFieldProps(
        "interval_ms",
        "choice",
        "Update interval",
        default=selected,
        choices=tuple(FormChoice(f"{value} ms", value) for value in values),
    )


__all__ = [
    "decode_parameter_value",
    "interval_form_field",
    "panel_state_document",
    "parameter_fields",
    "parameter_edit_values",
    "parameter_form_spec",
    "parameter_form_values",
    "signal_form_runtime",
]
