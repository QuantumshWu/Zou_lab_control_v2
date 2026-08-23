"""Small, headless authoring-schema contracts shared by devices and nodes."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Mapping


def _typed_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and bool(left == right)


@dataclass(frozen=True, slots=True)
class AuthoringChoice:
    """One owner-declared value and the human label that explains it."""

    value: Any
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("authoring choice label must be non-empty")
        try:
            hash(self.value)
        except TypeError as error:
            raise TypeError("authoring choice value must be immutable") from error


@dataclass(frozen=True)
class AuthoringField:
    name: str
    value_type: str
    label: str
    default: Any = None
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[AuthoringChoice, ...] = ()
    #: ``(field name, values)``: this field is editable only while that field
    #: holds one of those values.  It is still SHOWN -- an option that appears
    #: and disappears as another is touched is a moving target, and an
    #: operator cannot see what a setting would offer before choosing it.
    enabled_when: tuple[str, tuple[Any, ...]] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.value_type or not self.label:
            raise ValueError("authoring fields require name, value_type, and label")
        if self.enabled_when is not None:
            controller, values = self.enabled_when
            if not str(controller).strip():
                raise ValueError("enabled_when needs the name of a field")
            if not tuple(values):
                raise ValueError("enabled_when needs at least one enabling value")
            object.__setattr__(
                self,
                "enabled_when",
                (str(controller), tuple(values)),
            )
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("authoring field minimum exceeds maximum")
        choices = tuple(self.choices)
        if any(not isinstance(choice, AuthoringChoice) for choice in choices):
            raise TypeError("authoring field choices must contain AuthoringChoice values")
        values = tuple(choice.value for choice in choices)
        if any(
            any(_typed_equal(value, prior) for prior in values[:index])
            for index, value in enumerate(values)
        ):
            raise ValueError("authoring choice values must be unique")
        if choices and self.default is not None and not any(
            _typed_equal(self.default, value) for value in values
        ):
            raise ValueError("authoring field default is not one of its choices")
        object.__setattr__(self, "choices", choices)


@dataclass(frozen=True)
class TunableField:
    """One runtime setting: stable form metadata beside current device truth."""

    metadata: AuthoringField
    current: Any
    live_write: bool
    dependency_group: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, AuthoringField):
            raise TypeError("tunable metadata must be an AuthoringField")
        if type(self.live_write) is not bool:
            raise TypeError("tunable live_write must be bool")
        group = tuple(str(name).strip() for name in self.dependency_group)
        if (
            not group
            or any(not name for name in group)
            or len(set(group)) != len(group)
            or self.metadata.name not in group
        ):
            raise ValueError(
                "tunable dependency_group must uniquely include its own field"
            )
        current = AuthoringSchema((self.metadata,)).project_values(
            {self.metadata.name: self.current}
        )[self.metadata.name]
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "dependency_group", group)


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

    def project_values(
        self,
        values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert one raw draft and validate it through its sole domain owner."""

        supplied = dict(values or {})
        unknown = set(supplied) - set(self.field_names)
        if unknown:
            raise ValueError(f"unknown authoring fields: {sorted(unknown)}")
        result: dict[str, Any] = {}
        for field in self.fields:
            value = _project_value(field, supplied.get(field.name, field.default))
            if value is None and field.required:
                raise ValueError(f"missing required authoring field {field.name!r}")
            if isinstance(value, str) and field.required and not value.strip():
                raise ValueError(f"missing required authoring field {field.name!r}")
            if field.choices and value is not None and not any(
                _typed_equal(value, choice.value) for choice in field.choices
            ):
                offered = tuple(choice.value for choice in field.choices)
                raise ValueError(f"{field.name!r} must be one of {offered!r}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if field.minimum is not None and value < field.minimum:
                    raise ValueError(f"{field.name!r} is below its minimum")
                if field.maximum is not None and value > field.maximum:
                    raise ValueError(f"{field.name!r} is above its maximum")
            result[field.name] = value
        if self.validator is not None:
            self.validator(result)
        return result


def _project_value(field: AuthoringField, value: object) -> object:
    if value is None:
        return None
    declared = str(field.value_type)
    if declared == "numeric_tuple":
        try:
            parts = tuple(
                piece for piece in value.replace(",", " ").split() if piece
            ) if isinstance(value, str) else tuple(value)
        except TypeError as error:
            raise TypeError(f"{field.label} must be a numeric list") from error
        if not parts:
            raise ValueError(f"{field.label} needs at least one number")
        try:
            if any(isinstance(part, bool) for part in parts):
                raise TypeError
            projected = tuple(float(part) for part in parts)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{field.label} items must be finite numbers") from error
        if not all(math.isfinite(item) for item in projected):
            raise ValueError(f"{field.label} items must be finite")
        return projected
    if declared == "pair":
        if isinstance(value, (tuple, list)):
            parts = tuple(value)
        else:
            parts = tuple(
                piece
                for piece in str(value).replace(",", " ").split()
                if piece
            )
        if len(parts) != 2:
            raise ValueError(f"{field.label} needs two numbers, as y, x")
        return [
            _project_integer(part, label=f"{field.label} item")
            for part in parts
        ]
    if declared == "int":
        return _project_integer(value, label=field.label)
    if declared == "float":
        if isinstance(value, bool):
            raise TypeError(f"{field.label} must be a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{field.label} must be finite")
        return result
    if declared == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        raise TypeError(f"{field.label} must be true or false")
    if declared in {"str", "text", "choice", "resource", "folder"}:
        if not isinstance(value, str):
            if declared not in {"choice"}:
                raise TypeError(f"{field.label} must be text")
            return value
        return value
    raise ValueError(
        f"field {field.name!r} is declared as {field.value_type!r}, "
        "which has no domain value projector"
    )


_DECIMAL_INTEGER = re.compile(r"[+-]?[0-9]+")


def _project_integer(value: object, *, label: str) -> int:
    """Project an authored integer without Python's lossy ``int`` coercions."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if _DECIMAL_INTEGER.fullmatch(normalized):
            return int(normalized, 10)
    raise TypeError(f"{label} must be an integer or decimal integer text")


__all__ = ["AuthoringChoice", "AuthoringField", "AuthoringSchema", "TunableField"]
