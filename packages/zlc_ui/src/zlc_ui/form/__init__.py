"""Lazy exports for headless form specifications and their Qt projection."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "FormChoice": ("zlc_ui.form.form", "FormChoice"),
    "FormFieldProps": ("zlc_ui.form.form", "FormFieldProps"),
    "FormSpec": ("zlc_ui.form.form", "FormSpec"),
    "FormFieldKind": ("zlc_ui.form.form", "FormFieldKind"),
    "parse_number_text": ("zlc_ui.form.form", "parse_number_text"),
    "choice_value_to_tree": ("zlc_ui.form.form", "choice_value_to_tree"),
    "choice_value_from_tree": ("zlc_ui.form.form", "choice_value_from_tree"),
    "FormRuntimeContext": ("zlc_ui.form.qt_form", "FormRuntimeContext"),
    "FluentParameterForm": ("zlc_ui.form.qt_form", "FluentParameterForm"),
}

__all__ = tuple(sorted(_EXPORTS))


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
