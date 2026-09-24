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

#: A period is authored; a spacer is time between two periods for a slow
#: device to settle.  Spelled here in the widget layer's own words: the card
#: decides its shape from this, and no pulse-domain object crosses.
PERIOD_KIND_PERIOD = "period"
PERIOD_KIND_SPACER = "spacer"
PERIOD_KINDS = (PERIOD_KIND_PERIOD, PERIOD_KIND_SPACER)


def bracket_gap_bounds(
    period_ids: tuple[str, ...], start: str | None, end: str | None,
) -> tuple[int, int]:
    """Half-open period gaps of one bracket's inclusive endpoints.

    A missing start anchor means "after the last period" and a missing end
    anchor "before the first", so an empty bracket can sit at either edge of
    the timeline; equal gaps are an empty bracket.
    """

    return (
        len(period_ids) if start is None else period_ids.index(start),
        0 if end is None else period_ids.index(end) + 1,
    )


def bracket_post_key(bracket_id: str, side: str) -> str:
    """The item key of one bracket post: ``"<bracket id>:start"`` or ``":end"``."""

    return f"{bracket_id}:{side}"


def schedule_item_order(
    period_ids: tuple[str, ...], brackets: tuple["BracketVM", ...] = (),
) -> tuple[tuple[str, str], ...]:
    """Derive visual items from each bracket's inclusive endpoints; store no timeline.

    ``brackets`` is outermost first, the order the model keeps them in.  At a
    gap the posts read inside-out: the ends of brackets closing there
    (innermost first), then any empty bracket sitting in the gap (its start
    then its end), then the starts of brackets opening there (outermost
    first).  An empty bracket at another bracket's boundary gap is therefore
    drawn beside it, never inside it -- the same rule the model decides
    nesting by, so the picture and the loops agree.
    """

    bounds = tuple(
        bracket_gap_bounds(period_ids, bracket.start_period_id, bracket.end_period_id)
        for bracket in brackets
    )
    items = []
    for gap in range(len(period_ids) + 1):
        for bracket, (first, stop) in reversed(tuple(zip(brackets, bounds))):
            if stop == gap and first < stop:
                items.append(("bracket", bracket_post_key(bracket.bracket_id, "end")))
        for bracket, (first, stop) in zip(brackets, bounds):
            if first == stop == gap:
                items.append(("bracket", bracket_post_key(bracket.bracket_id, "start")))
                items.append(("bracket", bracket_post_key(bracket.bracket_id, "end")))
        for bracket, (first, stop) in zip(brackets, bounds):
            if first == gap and first < stop:
                items.append(("bracket", bracket_post_key(bracket.bracket_id, "start")))
        if gap < len(period_ids):
            items.append(("period", period_ids[gap]))
    return tuple(items)


@dataclass(frozen=True)
class FieldVM:
    text: str
    editable: bool = True
    scan: bool = False
    source: str = "default"
    can_scan: bool = True
    can_api: bool = True
    effective_text: str = ""
    source_text: str = ""
    config_key: str = ""
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
    digital: tuple[tuple[str, bool], ...] = ()
    analog: tuple[tuple[str, str, FieldVM], ...] = ()
    kind: str = PERIOD_KIND_PERIOD

    def __post_init__(self) -> None:
        if self.kind not in PERIOD_KINDS:
            raise ValueError(f"period kind must be one of {PERIOD_KINDS}, got {self.kind!r}")


@dataclass(frozen=True)
class BracketVM:
    bracket_id: str
    start_period_id: str | None
    end_period_id: str | None
    count: int
    #: Its number among the pulse's brackets, outermost first, and the ink
    #: its posts wear -- the preview draws its loop in the same.
    ordinal: int = 1
    color: str = ""


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
    #: Outermost first, as the model keeps them.
    brackets: tuple[BracketVM, ...] = ()
    run_repeats: int = 0
    delay_rows: tuple[DelayRowVM, ...] = ()
    scan_summary_text: str = ""
    min_bracket_count: int = 2
    default_bracket_count: int = 2

    @property
    def item_order(self) -> tuple[tuple[str, str], ...]:
        return schedule_item_order(
            tuple(period.period_id for period in self.periods), self.brackets,
        )

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
class BindingRecord:
    """A physical Pulse field; its readable label is not a mutable alias."""

    field_id: str
    label: str
    scan: bool = False
    source: str = "default"


@dataclass(frozen=True)
class ConfigPageRecord:
    """Config draft and saved-file facts projected by the presenter."""

    file_path: str = ""
    dirty: bool = False
    entries: tuple[tuple[str, str, str], ...] = ()
    available_names: tuple[str, ...] = ()
    bindings: tuple[tuple[str, str, str, str, str, str], ...] = ()
    active_path: str = ""
    busy: bool = False


@dataclass(frozen=True)
class ScanPageRecord:
    slots_text: str = ""
    bindings: tuple[BindingRecord, ...] = ()
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
    "BindingRecord",
    "PERIOD_KIND_PERIOD",
    "PERIOD_KIND_SPACER",
    "PERIOD_KINDS",
    "ConnectionChoiceVM",
    "ConnectionVM",
    "ConfigPageRecord",
    "DelayRowVM",
    "FieldVM",
    "PeriodVM",
    "PortRowVM",
    "BracketVM",
    "ScanPageRecord",
    "ScheduleVM",
    "bracket_gap_bounds",
    "bracket_post_key",
    "schedule_item_order",
    "TargetPortRecord",
    "TargetWidthRule",
]
