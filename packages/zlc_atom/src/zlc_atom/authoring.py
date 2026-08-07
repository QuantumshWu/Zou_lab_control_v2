"""Small, headless authoring-schema contracts shared by devices and nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class AuthoringField:
    name: str
    value_type: str
    label: str
    default: Any = None
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.value_type or not self.label:
            raise ValueError("authoring fields require name, value_type, and label")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("authoring field minimum exceeds maximum")
        if self.choices and self.default is not None and self.default not in self.choices:
            raise ValueError("authoring field default is not one of its choices")


@dataclass(frozen=True)
class AuthoringSchema:
    fields: tuple[AuthoringField, ...] = ()
    validator: Callable[[Mapping[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        if any(not isinstance(field, AuthoringField) for field in fields):
            raise TypeError("fields must contain AuthoringField values")
        names = tuple(field.name for field in fields)
        if len(set(names)) != len(names):
            raise ValueError("authoring field names must be unique")
        object.__setattr__(self, "fields", fields)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def freeze(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        supplied = dict(values or {})
        unknown = set(supplied) - set(self.field_names)
        if unknown:
            raise ValueError(f"unknown authoring fields: {sorted(unknown)}")
        result: dict[str, Any] = {}
        for field in self.fields:
            value = supplied.get(field.name, field.default)
            if value is None and field.required:
                raise ValueError(f"missing required authoring field {field.name!r}")
            if field.choices and value is not None and value not in field.choices:
                raise ValueError(f"{field.name!r} must be one of {field.choices!r}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if field.minimum is not None and value < field.minimum:
                    raise ValueError(f"{field.name!r} is below its minimum")
                if field.maximum is not None and value > field.maximum:
                    raise ValueError(f"{field.name!r} is above its maximum")
            result[field.name] = value
        if self.validator is not None:
            self.validator(result)
        return result


__all__ = ["AuthoringField", "AuthoringSchema"]
