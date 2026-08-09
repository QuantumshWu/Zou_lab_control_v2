"""Plain, frozen view models for the pulse-editor widgets.

The records deliberately contain only values that a presenter can serialize:
strings, numbers, booleans and tuples.  No pulse-domain object crosses this
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_ui.form import FormChoice


#: What a field may hold, in the words the widget layer understands.
#:
#: Named here rather than spelled out at each end.  A presenter said "time" and
#: "integer" while this side tested for "float" and "int", so the only
#: client-side guard silently did nothing and every typed value went to the
#: model to be refused -- from inside a Qt slot, where an escaping ValueError
#: ends the process.
VALIDATOR_NONE = "none"
VALIDATOR_INT = "int"
VALIDATOR_FLOAT = "float"
VALIDATOR_KINDS = (VALIDATOR_NONE, VALIDATOR_INT, VALIDATOR_FLOAT)


@dataclass(frozen=True)
class FieldVM:
    text: str
    editable: bool = True
    binding_kind: str = ""
    binding_number: int = 0
    binding_tooltip: str = ""
    validator_kind: str = VALIDATOR_NONE
    validator_lo: float = 0.0
    validator_hi: float = 0.0
    resolution: float = 0.0
    allow_any: bool = True

    def __post_init__(self) -> None:
        if self.validator_kind not in VALIDATOR_KINDS:
            raise ValueError(
                f"validator_kind must be one of {VALIDATOR_KINDS}, "
                f"got {self.validator_kind!r}"
            )


@dataclass(frozen=True)
class PortRowVM:
    key: str
    kind: str
    label: str
    endpoint_text: str = ""
    endpoint_tooltip: str = ""
    width: int = 1
    lo: int = 0
    hi: int = 0
    visible: bool = True


@dataclass(frozen=True)
class PeriodVM:
    period_id: str
    name: str
    duration: FieldVM
    unit: str
    unit_choices: tuple[str, ...] = ()
    unit_locked: bool = False
    digital: tuple[tuple[str, bool], ...] = ()
    analog: tuple[tuple[str, str, FieldVM], ...] = ()


@dataclass(frozen=True)
class RepeatVM:
    start_period_id: str
    end_period_id: str
    count: int


@dataclass(frozen=True)
class DelayRowVM:
    port_key: str
    value: FieldVM
    unit: str
    unit_quantums: tuple[tuple[str, float], ...] = ()


def _string_choice_values(
    choices: tuple[FormChoice, ...],
    *,
    owner: str,
) -> tuple[str, ...]:
    if not isinstance(choices, tuple):
        raise TypeError(f"{owner} choices must be a tuple")
    values: list[str] = []
    for choice in choices:
        if not isinstance(choice, FormChoice):
            raise TypeError(f"{owner} choices must contain FormChoice values")
        if not isinstance(choice.value, str) or not choice.value:
            raise TypeError(f"{owner} choice values must be non-empty strings")
        values.append(choice.value)
    if len(set(values)) != len(values):
        raise ValueError(f"{owner} choice values must be unique")
    return tuple(values)


@dataclass(frozen=True)
class ConnectionChoiceVM:
    """One presenter-owned connection action and its endpoint authority."""

    label: str
    value: str
    endpoint_editable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("connection choice label must be non-empty text")
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("connection choice value must be non-empty text")
        if not isinstance(self.endpoint_editable, bool):
            raise TypeError("connection endpoint_editable must be bool")


@dataclass(frozen=True)
class ConnectionVM:
    """One complete connection control state supplied by its presenter."""

    choices: tuple[ConnectionChoiceVM, ...]
    selected: str
    endpoint: str
    status: str
    locked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.choices, tuple):
            raise TypeError("connection choices must be a tuple")
        if any(not isinstance(choice, ConnectionChoiceVM) for choice in self.choices):
            raise TypeError("connection choices must contain ConnectionChoiceVM values")
        values = tuple(choice.value for choice in self.choices)
        if len(set(values)) != len(values):
            raise ValueError("connection choice values must be unique")
        if not values:
            raise ValueError("a connection control needs at least one choice")
        if self.selected not in values:
            raise ValueError(
                f"selected connection {self.selected!r} is not in {values!r}"
            )
        if not isinstance(self.endpoint, str) or not isinstance(self.status, str):
            raise TypeError("connection endpoint and status must be strings")
        if not isinstance(self.locked, bool):
            raise TypeError("connection locked must be bool")


@dataclass(frozen=True)
class ScheduleVM:
    document_generation: int
    revision: int
    document_name: str
    clock_text: str
    total_text: str
    total_tooltip: str
    period_count: int
    visible_text: str
    summary_text: str
    ports: tuple[PortRowVM, ...]
    periods: tuple[PeriodVM, ...]
    analog_mode_choices: tuple[FormChoice, ...] = ()
    repeat: RepeatVM | None = None
    delay_rows: tuple[DelayRowVM, ...] = ()
    scan_summary_text: str = ""
    min_repeat_count: int = 1
    default_repeat_count: int = 1

    def __post_init__(self) -> None:
        values = _string_choice_values(
            self.analog_mode_choices,
            owner="analog mode",
        )
        offered = set(values)
        for period in self.periods:
            for _port_key, mode, _field in period.analog:
                if mode not in offered:
                    raise ValueError(
                        f"analog mode {mode!r} is not in the supplied choices"
                    )


@dataclass(frozen=True)
class ScanPageRecord:
    slots_text: str = ""
    table_text: str = ""
    source_text: str = ""
    source_dirty: bool = False
    repeats: int = 1
    busy: bool = False
    progress_text: str = ""
    progress_polling: bool = False


@dataclass(frozen=True)
class TargetPortRecord:
    key: str
    kind: str
    signal: str
    endpoints: tuple[str, ...] = ()
    clock_key: str | None = None
    clock_endpoint: str | None = None
    lane_order: tuple[int, ...] = ()


@dataclass(frozen=True)
class TargetWidthRule:
    minimum: int
    default: int
    maximum: int | None = None


__all__ = [
    "ConnectionChoiceVM",
    "ConnectionVM",
    "DelayRowVM",
    "FieldVM",
    "PeriodVM",
    "PortRowVM",
    "RepeatVM",
    "ScanPageRecord",
    "ScheduleVM",
    "TargetPortRecord",
    "TargetWidthRule",
]
