"""Plain, frozen view models for the pulse-editor widgets.

The records deliberately contain only values that a presenter can serialize:
strings, numbers, booleans and tuples.  No pulse-domain object crosses this
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    repeat: RepeatVM | None = None
    delay_rows: tuple[DelayRowVM, ...] = ()
    scan_summary_text: str = ""
    min_repeat_count: int = 1
    default_repeat_count: int = 1


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
