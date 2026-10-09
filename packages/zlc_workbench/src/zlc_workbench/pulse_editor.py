"""Editing a pulse sequence in a window, and seeing what it will do.

Three packages had everything except the thing between them: zlc_pulse owns an
immutable ``PulseSequence`` and what makes one legal, zlc_ui owns the
editor that speaks in plain view models, and zlc_plot draws a timeline.  Nothing
turned one into the others, so the editor rendered nothing and the notebook had
no way to look at a pulse before firing it.

The division:

* zlc_pulse decides what a legal sequence is.  Every edit here builds a new one
  and lets the model refuse it -- there is no second copy of the rules, and an
  edit that would produce something the hardware cannot play is rejected by the
  same code that would have rejected it at compile time.
* zlc_ui renders view models and raises intents.  It never sees a PulseSequence.
* THIS converts between them, in both directions, and holds the current
  sequence because someone has to.

Edits are whole-sequence replacements rather than mutations.  A pulse is small,
and the alternative -- mutating in place and validating afterwards -- is how an
editor ends up holding something that cannot be compiled, with no way back to
the last good state.
"""

from __future__ import annotations

import logging

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from threading import Event
from typing import Any

from zlc_pulse import (
    analog_levels,
    ANALOG_MODE_CHOICES,
    AnalogStep,
    MINIMUM_BRACKET_COUNT,
    OutputDelay,
    PERIOD_KIND_SPACER,
    PulseBracket,
    PulseBinding,
    PulseFieldRef,
    PulsePeriod,
    PulseSequence,
    PulseTarget,
    Subpulse,
    group_component,
    ungroup_component,
    extract_subpulse,
    insert_subpulse,
    replace_component,
    remove_component,
    read_subpulse,
    write_subpulse,
)
from zlc_pulse import (
    TIME_UNIT_CHOICES,
    nanoseconds_per,
    align_to_grid,
    apply_config_values,
    config_parameter_key,
    field_label,
    prune_orphaned_bindings,
    resolve_api_parameters,
    resolve_scan_point,
    CONFIG_VALUES_DIRECTORY,
    CURRENT_CONFIG_VALUES,
    write_config_values,
    read_config_values,
    pulse_field_value,
)
from zlc_data.units import figure_padded, format_quantity
from zlc_durable import unique_path
from zlc_plot import PANEL_SIZE_NAMES, bracket_color
from zlc_ui import (
    ConnectionChoiceVM,
    ConnectionVM,
    FormChoice,
    VALIDATOR_FLOAT,
    VALIDATOR_INT,
    DelayRowVM,
    BindingRecord,
    ConfigPageRecord,
    ScanPageRecord,
    TargetWidthRule,
    FieldVM,
    PeriodVM,
    PortRowVM,
    BracketVM,
    ScheduleVM,
    ComponentVM,
)

from .board import _guarded_slot
from .device_use import DeviceClaim, DeviceLease, DeviceUseBusy, DeviceUseCoordinator
from .pulse_state import PulseEditorState, read_pulse, write_pulse
from zlc_ui import bracket_post_key, schedule_item_order


_LOG = logging.getLogger(__name__)


__all__ = [
    "PulseEditorPresenter",
    "preview_candidate",
    "programmable_ports",
    "project_ports",
    "project_schedule",
    "project_target",
    "timeline_of",
]


#: What a legal time unit is belongs to zlc_pulse, which enforces it on every
#: period it accepts.  This window used to declare its own four and leave out
#: the one zlc_pulse also takes, so a pulse authored in ticks raised KeyError
#: inside the projection -- from a Qt slot, which ends the process rather than
#: drawing anything.  Offered shortest first, which is presentation and is all
#: this line decides.
_TIME_UNITS = TIME_UNIT_CHOICES


def _bracket_vms(sequence: PulseSequence) -> tuple[BracketVM, ...]:
    """Each bracket numbered and inked as the model orders them: outermost first.

    The number is what its posts say ("Bracket 2") and the ink is what its
    posts and the preview's loop share, so the frame drawn around cards in
    the strip and the loop drawn over the timeline read as one thing.
    """

    return tuple(
        BracketVM(
            b.bracket_id, b.start_period_id, b.end_period_id, b.count,
            ordinal=index + 1, color=bracket_color(index),
        )
        for index, b in enumerate(sequence.brackets)
    )


def _sequence_item_order(sequence: PulseSequence) -> tuple[tuple[str, str], ...]:
    return schedule_item_order(
        tuple(period.period_id for period in sequence.periods), _bracket_vms(sequence),
    )


def _nanoseconds(value: float, unit: str) -> float:
    """A duration in nanoseconds, exact before it is a float.

    ``value * 1e-6`` is not the microsecond count it looks like, and this feeds
    grid checks and totals where the last digit is the difference between a
    duration the board accepts and one it refuses.
    """

    return float(Fraction(str(float(value))) * nanoseconds_per(unit))


def _readable(nanoseconds: float) -> str:
    """One duration, in the scale a person would have written it in.

    This used to pick a unit by walking the table until one fitted and then
    print ``%g``, which is a third number formatter with its own idea of how
    many digits a value has.  There is one now, it knows the whole prefix
    ladder rather than four rows of it, and what it prints can be typed back.
    """

    seconds = float(Fraction(str(float(nanoseconds))) / nanoseconds_per("s"))
    return format_quantity(seconds, "s")


# ------------------------------------------------------------------ projection


#: What an editor holding no sequence shows.  Not an error state: an editor
#: opens before it has a subject, and its job then is to say how to get one.
#: How a preview writes "this scan point plays until Stop".
RUN_FOREVER_LABEL = "×∞"
#: What a period with NO step for a DAC means: the output keeps whatever the
#: period before it left there.  It is a reading of the model, not a mode the
#: model has -- ANALOG_MODES is edge and ramp -- and the one place both the
#: projection and the commit have to agree on the word.
HOLD_MODE = "hold"
CONNECTION_VIRTUAL = "virtual"
CONNECTION_REMOTE = "remote"
CONNECTION_OFFLINE = "offline"
CONNECTION_GIVEN = "given"
#: zlc_pulse owns the legal analog-step values; this presenter adds the label
#: and the authoring-only Hold action (absence of a step) before crossing the
#: view-model boundary.  Qt receives rows and never spells this finite domain.
ANALOG_MODE_ROWS = tuple(
    FormChoice(mode.title(), mode) for mode in ANALOG_MODE_CHOICES
) + (FormChoice("Hold", HOLD_MODE),)
#: Only a standalone editor may replace its sequencer.  An embedded editor gets
#: a separate one-choice, locked authority record in the presenter constructor.
STANDALONE_CONNECTION_CHOICES = (
    ConnectionChoiceVM("Virtual (sim)", CONNECTION_VIRTUAL),
    ConnectionChoiceVM("Remote server", CONNECTION_REMOTE, endpoint_editable=True),
    ConnectionChoiceVM("Offline (edit only)", CONNECTION_OFFLINE),
)
#: How long each period of a brand-new pulse is.  A duration has to be chosen
#: and the board's tick is the wrong choice: at 20 ns the two periods are drawn
#: as one pixel, so a legal pulse looks like no pulse.  1 us is the established
#: PulseGUI's answer and reads on the timeline at its default zoom.
NEW_PULSE_PERIOD_NS = 1000.0
#: The page whose contents cost something to produce.  Drawing a timeline
#: starts a render worker and a drawing session, and doing that for a page
#: nobody has turned to is most of what a window spends before it appears.
PREVIEW_PAGE = "Preview"
SCAN_PAGE = "Scan"


def _connection_name(mode: str, endpoint: str) -> str:
    """What this editor is attached to, named by what was dialled.

    Not by whichever field happened to be non-empty.  This read
    ``endpoint or mode``, and the address box keeps the server's address
    whether or not remote is the selected mode -- so connecting to the
    simulated board reported "127.0.0.1:18861 - 26 ports, 62 lanes, 50 MHz",
    the same line a real board gives, and nothing on screen said which one had
    answered.  Which board is talking is the single most consequential fact in
    this window: it decides whether pressing On Pulse moves atoms.
    """

    if str(mode) == CONNECTION_REMOTE:
        return f"remote {endpoint}" if endpoint else "remote"
    return str(mode)


EMPTY_SCHEDULE = ScheduleVM(
    document_generation=0,
    revision=0,
    document_name="(no pulse)",
    clock_text="",
    total_text="",
    total_tooltip="",
    period_count=0,
    visible_text="",
    summary_text="Load a pulse, or Add Period to start one on this board",
    ports=(),
    periods=(),
    analog_mode_choices=ANALOG_MODE_ROWS,
)


def _pins_of(target: object, pins: Mapping[str, str] | None) -> dict[str, str]:
    """The package pin per lane: the board's answer, else the target's own.

    A pin-aware target already carries its map, so a page that only looked at
    what a board had told it showed lane names offline -- the same wiring
    described two ways depending on whether anything was plugged in.
    """

    return dict(pins or getattr(target, "package_pins", {}) or {})


def programmable_ports(target: object) -> tuple:
    """The ports a pulse can drive, with each DAC owning its latch clock.

    A DAC is ONE output.  Its ten data lanes and the clock that latches them
    are one wire bundle to the person editing a pulse, and the clock is not
    something a pulse drives at all -- the compiler emits it from the DAC's own
    edges.  Listing it as a separate programmable port offered an edit that
    cannot be made and split one output across two rows.

    So: digital ports and DAC ports, in catalog order, and every clock port
    that belongs to a DAC folded into it.  A clock port owned by nothing is
    still shown on the Edit tab, because an unexplained lane is worse than an
    odd row; the Target page gives it no row, carries it unchanged and never
    mints its name (``project_target``, ``refresh_target``).
    """

    owned = {
        port.latch_clock
        for port in target.ports
        if port.kind == "dac" and port.latch_clock
    }
    return tuple(
        port
        for port in target.ports
        if not (port.kind == "clock" and port.key in owned)
    )


def project_ports(
    target: object,
    *,
    pins: Mapping[str, str] | None = None,
    visible: set[str] | None = None,
) -> tuple[PortRowVM, ...]:
    """One target's programmable ports as rows.

    The single place a port becomes a row, shared by the pulse projection and
    the board-only one -- otherwise an editor with a pulse open and the same
    editor without one would describe the same hardware differently.

    A one-lane port shows the package pin, because that is the name written on
    the breakout someone is wiring into.  A DAC cannot: ten pins do not fit a
    column and do not read as one output, so it shows its bus index and width
    and puts every lane, pin and its latch clock in the tooltip.
    """

    pin_by_lane = _pins_of(target, pins)
    clock_lane = {
        port.key: port.lanes[0] for port in target.ports if port.kind == "clock"
    }

    def _wired(lane: str) -> str:
        pin = pin_by_lane.get(lane, "")
        return f"{lane} -> {pin}" if pin else lane

    def _endpoint(port: object) -> str:
        if port.kind == "dac":
            return f"bus{port.bus_index} {port.width}b+clk"
        lane = port.lanes[0]
        return pin_by_lane.get(lane, "") or lane

    def _tooltip(port: object) -> str:
        lines = [f"{port.kind} {port.key}"]
        lines.extend(f"    {_wired(lane)}" for lane in port.lanes)
        if port.kind == "dac" and port.latch_clock:
            latch = clock_lane.get(port.latch_clock, port.latch_clock)
            lines.append(f"    latch clock {port.latch_clock}: {_wired(latch)}")
            low, high = port.signed_range
            lines.append(f"    code range {low}..{high}")
        return "\n".join(lines)

    return tuple(
        PortRowVM(
            key=port.key,
            kind=port.kind,
            label=port.label or port.key,
            endpoint_text=_endpoint(port),
            endpoint_tooltip=_tooltip(port),
            width=port.width or len(port.lanes),
            lo=0 if port.signed_range is None else port.signed_range[0],
            hi=0 if port.signed_range is None else port.signed_range[1],
            visible=visible is None or port.key in visible,
        )
        for port in programmable_ports(target)
    )


def project_target(
    target: object,
    *,
    pins: Mapping[str, str] | None = None,
) -> tuple:
    """The wiring, as the Target page shows it.

    This is where the pin map belongs.  A pulse names outputs -- cooling, probe
    -- and the Target page is the only place that says which physical pins
    those names reach; without it an operator has the pulse and the breakout
    and no way to relate them.

    A DAC carries its latch clock here too, as its own field, because that is
    the one wire of the bundle a pulse never drives and an operator still has
    to find on the board.  A clock no DAC latches with has no row: the page
    has digital and DAC outputs only, and applying it carries such a clock
    over unchanged, until a DAC with no latch clock names its wire as its
    latch endpoint and takes it (``_target_from_records``).
    """

    from zlc_ui import TargetPortRecord

    pin_by_lane = _pins_of(target, pins)
    clock_lane = {
        port.key: port.lanes[0] for port in target.ports if port.kind == "clock"
    }

    def _endpoint(lane: str) -> str:
        return pin_by_lane.get(lane, "") or lane

    records = []
    for port in programmable_ports(target):
        if port.kind == "clock":
            continue
        clock_key = port.latch_clock if port.kind == "dac" else None
        records.append(
            TargetPortRecord(
                key=port.key,
                kind=port.kind,
                signal=port.label or port.key,
                endpoints=tuple(_endpoint(lane) for lane in port.lanes),
                clock_key=clock_key,
                clock_endpoint=(
                    None
                    if clock_key is None
                    else _endpoint(clock_lane.get(clock_key, clock_key))
                ),
            )
        )
    return tuple(records)


def _scan_table_text(rows: Sequence[Sequence[float]], columns: Sequence[object]) -> str:
    """The scan table as a person reads it: a header, then the first rows.

    Truncated on purpose.  A thousand-point scan is not read line by line, and
    a page that pastes all of it hides the shape it was supposed to show.

    Shown in the units its author wrote, with its units in the header: a DAC
    column is a whole signed code, a duration keeps the fraction that made it
    worth sweeping.  Formatting every column as an integer was the wire's rule
    reaching a page that is not about the wire.
    """

    if not rows:
        return "(no scan table yet -- write a program and press Run)"
    specs = list(columns)
    header = "  ".join(
        f"{getattr(column, 'label', '') or getattr(column, 'name', index):>14}"
        for index, column in enumerate(specs)
    )
    units = "  ".join(f"{getattr(column, 'unit', ''):>14}" for column in specs)
    lines = [header, units] if specs else []
    shown = list(rows[:40])
    lines.extend(
        "  ".join(
            f"{int(round(value)):>14d}"
            if getattr(spec, "is_dac", False)
            else f"{float(value):>14g}"
            for value, spec in zip(row, specs, strict=False)
        )
        for row in shown
    )
    if len(rows) > len(shown):
        lines.append(f"... {len(rows) - len(shown)} more point(s)")
    return "\n".join(lines)


def project_period(
    sequence: PulseSequence,
    period: PulsePeriod,
    *,
    visible_ports: Sequence[str] | None = None,
    bindings: Mapping[tuple, PulseBinding] | None = None,
    config_values: Mapping[str, tuple[float, str]] | None = None,
    scan_active: bool = False,
) -> PeriodVM:
    """One period as its card.

    Extracted so a value edit can update the one card it changed instead of
    rebuilding the board: the card the operator just clicked already shows the
    new value, and the whole-schedule path exists for changes of SHAPE.  It is
    also the single place a period becomes a card, so the rebuilt board and a
    targeted update cannot disagree about what a period looks like.
    """

    target = sequence.target
    shown = None if visible_ports is None else {str(key) for key in visible_ports}
    lane_index = {lane: index for index, lane in enumerate(target.raw_lanes)}
    offered = programmable_ports(target)
    if bindings is None:
        bindings = bindings_of(sequence)
    duration_binding = _binding_for(
        bindings, "duration", period.period_id
    )
    spacer = period.kind == PERIOD_KIND_SPACER
    return PeriodVM(
        period_id=period.period_id,
        kind=period.kind,
        name=period.name or period.period_id,
        duration=FieldVM(
            # DISPLAYED, THEN TYPED BACK.  ``_commit_duration`` parses what
            # is in the box, so the way this number is written IS the number
            # the document keeps: %g rounds at six significant digits and
            # goes exponential above them, which silently rewrote an authored
            # duration the first time anyone touched its period.
            text=format_quantity(float(period.duration), "1"),
            **_binding_field_state(duration_binding, config_values, scan_active),
            # A spacer's length is a property of the device it waits for (or
            # a Config value): never scanned, never set by an API caller.
            can_scan=not spacer,
            can_api=not spacer,
            validator_kind=VALIDATOR_FLOAT,
            # One tick, and the shortest legal period, EXPRESSED IN THE UNIT
            # THIS BOX IS IN.  The grid the hardware plays on is 20 ns; the
            # number in the box is milliseconds.  Passing the raw 20 snapped a
            # 5 ms period to a multiple of 20 MILLISECONDS and refused anything
            # under 20 ms -- a device rule applied to a number that is not in
            # the device's unit, which is arithmetic nobody asked for.
            validator_lo=_tick_in(sequence, period.unit),
            validator_hi=0.0,
            resolution=_tick_in(sequence, period.unit),
            allow_any=False,
        ),
        unit=period.unit,
        unit_choices=_TIME_UNITS,
        digital=tuple(
            (port.key, bool(period.states[lane_index[port.lanes[0]]]))
            for port in offered
            if port.kind == "digital" and (shown is None or port.key in shown)
        ),
        analog=tuple(
            (
                port.key,
                _analog_mode(period, port),
                _analog_field(sequence, period, port, bindings, config_values, scan_active),
            )
            for port in offered
            if port.kind == "dac" and (shown is None or port.key in shown)
        ),
    )


def _visible_text(ports: Sequence[PortRowVM]) -> str:
    """How many of the board's ports have rows, worded once for the page."""

    return f"{sum(1 for port in ports if port.visible)}/{len(ports)} ports"


def project_schedule(
    sequence: PulseSequence | None,
    *,
    target: object | None = None,
    time_step_ns: float | None = None,
    path: str = "",
    generation: int = 0,
    revision: int = 0,
    visible_ports: Sequence[str] | None = None,
    pins: Mapping[str, str] | None = None,
    scan_points: int = 0,
    config_values: Mapping[str, tuple[float, str]] | None = None,
    scan_active: bool = False,
) -> ScheduleVM:
    """The schedule page: one pulse, or the bare board it would run on.

    Nothing is invented here.  Every value is read from the sequence or from
    its target, so a field the operator sees is a field the hardware has.

    A pulse supplies all of it -- its target, its clock, its periods and its
    delays.  A board attached with nothing authored yet supplies the first two
    and leaves the rest empty, which is a *state* of this projection and not a
    second projection: writing it separately is exactly what let the board-only
    view drift into a partial copy of this one, showing a column of channel
    names beside a delay column that was never built at all.
    """

    if sequence is not None:
        target = sequence.target
        time_step_ns = sequence.time_step_ns
    if target is None:
        raise ValueError("a schedule needs a pulse or the board it would run on")
    step_ns = float(sequence.time_step_ns if sequence is not None else time_step_ns)

    shown = None if visible_ports is None else {str(key) for key in visible_ports}
    ports = project_ports(target, pins=pins, visible=shown)
    bindings = bindings_of(sequence)
    periods = tuple(
        project_period(
            sequence, period, visible_ports=visible_ports, bindings=bindings,
            config_values=config_values, scan_active=scan_active,
        )
        for period in (() if sequence is None else sequence.periods)
    )

    # One pass through the periods is what the strip shows; what the board
    # PLAYS is that pass with every bracket expanded, and that is the total.
    pass_ns = sum(
        _nanoseconds(period.duration, period.unit)
        for period in (() if sequence is None else sequence.periods)
    )
    total_ns = 0.0 if sequence is None else sequence.played_nanoseconds()
    # Spacers are counted in the time, not among the periods: they are the
    # gaps between the periods someone wrote, and the cards number them so.
    authored = tuple(
        period for period in (() if sequence is None else sequence.periods)
        if period.kind != PERIOD_KIND_SPACER
    )
    slots = () if sequence is None else sequence.scan_bindings
    return ScheduleVM(
        document_generation=int(generation),
        revision=int(revision),
        document_name=sequence.name if sequence is not None else "(no pulse)",
        clock_text=f"{format_quantity(float(step_ns), '1')} ns/tick",
        total_text=_readable(total_ns) if sequence is not None else "",
        total_tooltip=(
            (
                f"{format_quantity(float(total_ns), '1')} ns as the board plays it; "
                f"{format_quantity(float(pass_ns), '1')} ns in one pass through "
                if total_ns != pass_ns
                else f"{format_quantity(float(total_ns), '1')} ns over "
            )
            + f"{len(authored)} period(s)"
            + (f" and {len(periods) - len(authored)} spacer(s)" if len(periods) > len(authored) else "")
            if sequence is not None
            else ""
        ),
        period_count=len(authored),
        visible_text=_visible_text(ports),
        summary_text=(
            (Path(path).name if path else sequence.name)
            if sequence is not None
            else "Add Period to start a pulse on this board"
        ),
        ports=ports,
        periods=periods,
        analog_mode_choices=ANALOG_MODE_ROWS,
        brackets=() if sequence is None else _bracket_vms(sequence),
        components=() if sequence is None else tuple(
            ComponentVM(
                component.component_id, component.name, component.period_ids,
                _readable(fragment.to_sequence().played_nanoseconds()),
                len(fragment.brackets),
                tuple(bracket.bracket_id for bracket in fragment.brackets),
                spacer_count=sum(period.kind == PERIOD_KIND_SPACER for period in fragment.periods),
            )
            for component in sequence.components
            for fragment in (extract_subpulse(sequence, component.component_id),)
        ),
        run_repeats=0 if sequence is None else sequence.run_repeats,
        # Every output the board can delay gets a row whether or not a pulse is
        # open, because the row IS the board telling the operator that output
        # can be delayed.  With no pulse the value is the zero a missing delay
        # means, and it cannot be edited: a delay belongs to the pulse that
        # carries it, and there is not one yet to write into.
        delay_rows=tuple(
            _delay_row(sequence, port.key, bindings, config_values)
            for port in programmable_ports(target)
            if port.kind in ("digital", "dac")
        ),
        # Slots AND points.  How many fields are bound is half the question --
        # the other half is how many points will actually be played, which is
        # the number an operator checks before pressing On Pulse.
        scan_summary_text=(
            f"{len(slots)} slot{'' if len(slots) == 1 else 's'} - "
            f"{scan_points} pt{'' if scan_points == 1 else 's'}"
            if slots
            else "no scan slots"
        ),
        # From the domain, which is what decides that once is not a repeat.
        # The view model carried 1 for both, so Add Bracket committed a
        # count-1 region -- which set_bracket reads as "no bracket" and the
        # button silently undid itself.  zlc_ui may not import zlc_pulse, so
        # the number is carried across by whoever knows both.
        min_bracket_count=MINIMUM_BRACKET_COUNT,
        default_bracket_count=MINIMUM_BRACKET_COUNT,
    )


def bindings_of(sequence: PulseSequence | None) -> dict[tuple, PulseBinding]:
    """Project the one binding per physical field; no second identity list."""

    if sequence is None:
        return {}
    return {
        (b.field_ref.kind, b.field_ref.period_id, b.field_ref.port): b
        for b in sequence.bindings
    }


def _binding_for(
    bindings: Mapping[tuple, PulseBinding],
    kind: str,
    period_id: str | None = None,
    port: str | None = None,
) -> PulseBinding | None:
    return bindings.get((kind, period_id, port))


def _binding_field_state(
    binding: PulseBinding | None,
    config_values: Mapping[str, tuple[float, str]] | None,
    scan_active: bool = False,
) -> dict[str, object]:
    """Display the source of the next execution, distinct from its default."""
    if binding is None:
        return {}
    effective = ""
    if binding.scan and scan_active:
        status = "The Scan table supplies this field; the default remains editable."
    elif binding.source == "config":
        value = (config_values or {}).get(binding.config_key) if binding.config_key else None
        if value is None:
            status = (
                f"Config '{binding.config_key}' is not supplied; using the Pulse default."
                if binding.config_key else "Config name is unassigned; using the Pulse default."
            )
        else:
            effective = f"{format_quantity(value[0], '1')} {value[1]}"
            status = f"Saved Config '{binding.config_key}' supplies {effective}. The input is the Pulse default."
    elif binding.source == "api":
        status = "An API caller may override this field; otherwise its Pulse default is used."
    else:
        status = "Using the Pulse default."
    return dict(scan=binding.scan, source=binding.source,
                effective_text=effective, source_text=status, config_key=binding.config_key)


def _analog_mode(period: PulsePeriod, port: Any) -> str:
    """What this period does to one DAC: step to a value, ramp, or hold.

    No entry means HOLD -- the output keeps whatever the period before it left
    there.  This used to answer "edge", which says "step to a value" beside a
    box with no value in it, so every DAC nobody had touched read as a
    half-filled row rather than as an output sitting where it was put.
    """

    step = next((item for item in period.analog_steps if item.port == port.key), None)
    return HOLD_MODE if step is None else step.mode


def _held_value(sequence: PulseSequence, period: PulsePeriod, port: Any) -> int:
    """What a holding DAC is holding: the last value set before this period.

    Nothing before it means the output is still at rest, and rest for an
    offset-binary DAC is mid-code -- zero volts, which in the signed units an
    operator types is 0.
    """

    held = 0
    for earlier in sequence.periods:
        if earlier.period_id == period.period_id:
            break
        step = next(
            (item for item in earlier.analog_steps if item.port == port.key), None
        )
        if step is not None:
            held = int(step.value)
    return held


def _analog_field(
    sequence: PulseSequence,
    period: PulsePeriod,
    port: Any,
    bindings: Mapping[tuple, PulseBinding],
    config_values: Mapping[str, tuple[float, str]] | None = None,
    scan_active: bool = False,
) -> FieldVM:
    """One DAC's box on one card.

    A holding output shows the level it is holding and cannot be typed into:
    the value is the earlier period's to change, and an editable box over a
    value this period does not own invites an edit that goes nowhere.  It used
    to be blank, which reads as "unknown" for an output whose level is known
    exactly.
    """

    step = next((item for item in period.analog_steps if item.port == port.key), None)
    low, high = port.signed_range or (0, 0)
    value = _held_value(sequence, period, port) if step is None else int(step.value)
    binding = _binding_for(bindings, "dac", period.period_id, port.key)
    return FieldVM(
        text=str(value),
        **_binding_field_state(binding, config_values, scan_active),
        editable=step is not None,
        validator_kind=VALIDATOR_INT,
        validator_lo=float(low),
        validator_hi=float(high),
        allow_any=True,
    )


def _tick_in(sequence: PulseSequence, unit: str) -> float:
    """One device tick, in the unit a field is written in.

    The only conversion between "what the board can resolve" and "what the box
    says", so a period card, a delay row and a scan column cannot disagree
    about how fine an edit may be.
    """

    return float(Fraction(str(float(sequence.time_step_ns))) / nanoseconds_per(str(unit)))


def _delay_of(sequence: PulseSequence, port_key: str) -> tuple[float, str]:
    """One output's delay as the PAIR it is stored as: a number and its unit.

    Flattened to nanoseconds, and the row then hardcoded "ns", so a delay
    stored as 5 us showed as 5000 next to a combo reading ns -- consistent
    until the operator touched the combo, at which point the 5000 was re-sent
    with the new unit and a 5 us delay silently became 5 ms.  A period card
    displays the pair it stores; a delay row is the same widget answering the
    same question, and now does the same.
    """

    delay = next((item for item in sequence.delays if item.port == port_key), None)
    return (0.0, "ns") if delay is None else (float(delay.value), str(delay.unit))


def _delay_row(
    sequence: PulseSequence | None,
    port_key: str,
    bindings: Mapping[tuple, PulseBinding],
    config_values: Mapping[str, tuple[float, str]] | None = None,
) -> DelayRowVM:
    """One output's delay row, wherever it is pushed from.

    The whole-board projection and the single row a typed delay sends back both
    come through here, so a targeted update cannot produce a row that differs
    from the one a rebuild would have shown.
    """

    value, unit = (0.0, "ns") if sequence is None else _delay_of(sequence, port_key)
    binding = _binding_for(bindings, "delay", None, port_key)
    return DelayRowVM(
        port_key=port_key,
        value=FieldVM(
            text="0" if sequence is None else format_quantity(float(value), "1"),
            can_scan=False,
            **_binding_field_state(binding, config_values),
            editable=sequence is not None,
            allow_any=False,
        ),
        unit=unit,
        units=tuple(_TIME_UNITS),
    )


def timeline_of(sequence: PulseSequence, *, include_off: bool = False) -> Any:
    """The sequence as a drawable timeline.

    Built in seconds because that is what the plot's time axis is in, and from
    the same period durations the hardware will play -- a preview computed any
    other way is a drawing of something else.
    """

    sequence.require_nonempty_brackets()
    from zlc_plot import (
        PulseAnalogTrace,
        PulseBlock,
        PulseChannel,
        PulseDacScanSegment,
        PulseLoopMarker,
        PulsePeriodMark,
        PulseScanRegion,
        PulseTimelineData,
    )

    target = sequence.target
    lane_index = {lane: index for index, lane in enumerate(target.raw_lanes)}
    starts: list[float] = []
    elapsed = 0.0
    for period in sequence.periods:
        starts.append(elapsed)
        elapsed += _nanoseconds(period.duration, period.unit) * 1e-9
    total = elapsed
    # The periods themselves, named as the operator named them, for the
    # band the renderer prints above the rows.
    period_marks = tuple(
        PulsePeriodMark(
            start,
            start + _nanoseconds(period.duration, period.unit) * 1e-9,
            sequence.period_label(period.period_id),
            spacer=period.kind == PERIOD_KIND_SPACER,
        )
        for start, period in zip(starts, sequence.periods)
        if _nanoseconds(period.duration, period.unit) > 0
    )

    channels: list[Any] = []
    blocks: list[Any] = []
    for port in target.ports:
        if port.kind != "digital":
            continue
        index = lane_index[port.lanes[0]]
        spans = [
            (starts[position], starts[position] + _nanoseconds(period.duration, period.unit) * 1e-9)
            for position, period in enumerate(sequence.periods)
            if period.states[index]
        ]
        if not spans and not include_off:
            continue
        channels.append(PulseChannel(port.key, port.label or port.key))
        blocks.extend(PulseBlock(port.key, start, stop) for start, stop in spans)

    # The DAC levels the board plays, from the pulse's own compiler: an edge
    # is one change at its period start and a ramp is the engine's staircase
    # from the level carried in to the target at the period end.  A trace read
    # off ``step.value`` alone drew both as the target level from the period
    # start, and two pulses the board plays differently were one picture.
    traces: list[Any] = []
    levels = analog_levels(sequence)
    tick_seconds = sequence.time_step_ns * 1e-9
    for port in target.ports:
        if port.kind != "dac":
            continue
        low, high = port.signed_range
        # A change on the pulse's last tick has nothing to hold: the trace
        # ends where the pulse does.
        points = [
            (tick * tick_seconds, float(value))
            for tick, value in levels[port.key]
            if tick * tick_seconds < total
        ]
        if not any(value for _, value in points) and not include_off:
            continue
        # A step trace is N values over N+1 boundaries: the last start is where
        # the final hold ENDS.  Passing equal-length arrays meant no DAC trace
        # could ever be drawn -- invisible until a pulse actually drove one.
        traces.append(
            PulseAnalogTrace(
                name=port.key,
                label=port.label or port.key,
                minimum=float(low),
                maximum=float(high),
                starts=tuple(at for at, _ in points) + (total,),
                values=tuple(value for _, value in points),
            )
        )

    if not channels:
        # A timeline needs at least one channel, and a sequence with nothing
        # high is a real thing to look at -- it is how a pulse being written
        # starts.  Showing the first port flat says that; refusing to draw says
        # nothing.
        first = next((port for port in target.ports if port.kind == "digital"), None)
        if first is None:
            raise ValueError("this target has no digital port to draw")
        channels.append(PulseChannel(first.key, first.label or first.key))

    # Bracket and Run are separate physical loops and therefore separate
    # markers.  Even identical spans remain two statements: the bracket loops
    # inside the timeline, while Run starts the complete pulse again without
    # advancing the scan point.
    markers: list[Any] = []
    # Innermost first.  The renderer stacks loops by what they contain, so the
    # order only decides between two loops over exactly the same span (a
    # whole-pulse bracket inside the Run loop); the model keeps its brackets
    # outermost first, so this reverses them.
    for series, (bracket, (first, stop_gap)) in reversed(
        tuple(enumerate(zip(sequence.brackets, sequence.bracket_bounds)))
    ):
        last = stop_gap - 1
        stop = starts[last] + _nanoseconds(
            sequence.periods[last].duration, sequence.periods[last].unit
        ) * 1e-9
        # A LABEL, which is what the marker takes.  Passing the count
        # itself raised TypeError inside the primitive, so a bracketed
        # pulse could not be previewed at all.  Only the count: the loop's
        # ink says which bracket it is -- the ink its posts wear -- and the
        # word "Bracket" beside every one of several said nothing the frame
        # did not.  ``series`` is the bracket's number, outermost first.
        markers.append(
            PulseLoopMarker(starts[first], stop, f"×{bracket.count}", series=series)
        )
    run_label = (
        RUN_FOREVER_LABEL
        if sequence.run_repeats == 0
        else f"×{sequence.run_repeats}"
    )
    if total > 0:
        markers.append(PulseLoopMarker(0.0, total, run_label))

    # A single marker describes each field's independent Scan/base source.
    regions: list[Any] = []
    segments: list[Any] = []
    stops = {
        period.period_id: starts[position]
        + _nanoseconds(period.duration, period.unit) * 1e-9
        for position, period in enumerate(sequence.periods)
    }
    positions = {period.period_id: index for index, period in enumerate(sequence.periods)}
    for (kind, period_id, port_key), binding in bindings_of(sequence).items():
        if not binding.scan and binding.source != "api":
            continue
        slot_kind = "scan" if binding.scan else "api"
        label = "+".join(part for part in (
            "S" if binding.scan else "",
            "A" if binding.source == "api" else "C" if binding.source == "config" else "",
        ) if part)
        if period_id is None or period_id not in positions:
            continue
        start, stop = starts[positions[period_id]], stops[period_id]
        if stop <= start:
            continue
        if kind == "duration":
            regions.append(PulseScanRegion(start, stop, label, slot_kind))
        elif kind == "dac" and any(trace.name == port_key for trace in traces):
            held = next(
                (
                    float(step.value)
                    for step in sequence.periods[positions[period_id]].analog_steps
                    if step.port == port_key
                ),
                0.0,
            )
            segments.append(
                PulseDacScanSegment(port_key, start, stop, held, label, slot_kind)
            )

    return PulseTimelineData(
        channels=tuple(channels),
        blocks=tuple(blocks),
        time_unit="s",
        total_duration=total,
        analog_traces=tuple(traces),
        scan_regions=tuple(regions),
        scan_dac_segments=tuple(segments),
        loop_markers=tuple(markers),
        periods=period_marks,
    )


def preview_candidate(
    sequence: PulseSequence,
    include_off: bool,
    pinned_size: str | None,
) -> tuple[object, str, int, int, float]:
    """The one timeline/size/status value every drawing of a pulse uses.

    The editor's Preview page and the Figure viewer's Pulse tab both draw
    through it: the viewer used to repeat the size rule beside its own copy
    of the timeline call, and never said how long the pulse played.
    """

    data = timeline_of(sequence, include_off=include_off)
    rows = len(getattr(data, "channels", ())) + len(
        getattr(data, "analog_traces", ())
    )
    if pinned_size:
        size = pinned_size
    else:
        from zlc_plot import recommended_pulse_preset

        size = recommended_pulse_preset(rows, len(sequence.periods))
    return (
        data,
        size,
        rows,
        len(sequence.periods),
        # The header says how long the board PLAYS the pulse: the pass
        # the axis shows, with every bracket expanded.
        sequence.played_nanoseconds(),
    )


# ------------------------------------------------------------------- presenter


@dataclass(frozen=True)
class BoardState:
    """What the board says it is doing, asked rather than remembered.

    Every one of these is a fact only the board can settle.  Holding a copy
    here -- set at the moment a command went out, cleared when this window
    happens to notice -- is how a window ends up lit green over an idle board:
    a notebook fires, a scan loads its own program, the server restarts, and
    nothing tells the copy.  So there is no copy.  It is read, whole, from the
    one place that knows, and a board that will not answer says exactly that
    instead of leaving the last good answer on screen.
    """

    attached: bool = False
    #: False when the board is attached but did not answer.  Not the same as
    #: idle, and shown differently, because "I cannot tell" is an answer.
    answering: bool = True
    firing: bool = False
    loaded: bool = False
    cursor: int | None = None
    #: What the board is holding, as the board named it.  NOT "is it what the
    #: editor shows" -- that is a comparison, and a comparison between a board
    #: fact and a local one belongs wherever the local one is known.  Keeping
    #: it here meant every answer to "does this still match?" had to be bought
    #: with a round trip, which is how typing in a box came to talk to a
    #: server.
    applied_digest: str = ""
    run_repeats: int = 1
    scan_repeats: int = 1
    fault: str = ""


class PulseEditorPresenter:
    """Connects the pulse editor's pages to one sequence.

    Every intent produces a candidate sequence and hands it to zlc_pulse to
    accept or refuse.  A refusal is shown and the previous sequence is kept, so
    the editor can never be holding something that will not compile.
    """

    def __init__(
        self,
        view: object,
        state: PulseEditorState | None = None,
        *,
        make_preview: Callable[[Any], Any] | None = None,
        update_preview: Callable[..., Any] | None = None,
        sequencer: Any = None,
        device_use: DeviceUseCoordinator | None = None,
        dial: Callable[[str, str], Any] | None = None,
        pulses_directory: str = "",
        path: str = "",
        default_endpoint: str = "",
        connection_label: str = "Experiment session",
        run_preview_work: Callable[..., None] | None = None,
        run_device_work: Callable[..., None] | None = None,
        run_safe_work: Callable[..., None] | None = None,
        run_completion_work: Callable[..., None] | None = None,
        request_preview_close: Callable[[], None] | None = None,
    ) -> None:
        self.view = view
        if state is not None and not isinstance(state, PulseEditorState):
            raise TypeError("state must be PulseEditorState or None")
        self._state = state if state is not None else PulseEditorState()
        self._saved_state = self._state
        self._component_context = ""
        self._subpulse_draft: Subpulse | None = None
        self._subpulse_saved: Subpulse | None = None
        self._subpulse_path = ""
        self._subpulse_visible_ports: frozenset[str] | None = None
        self._component_projection_revision = 0
        self.path = str(path)
        #: Where the Open dialog starts when nothing is open yet.
        self.pulses_directory = str(pulses_directory)
        self.revision = 0
        self._make_preview = make_preview
        # How an existing preview takes new data.  Without one the host is
        # rebuilt, which is correct but slow; the composition root supplies it.
        self._update_preview = update_preview or (
            lambda host, data, *, size: host.update_data(data) and None
        )
        self._run_preview_work = run_preview_work
        if self._make_preview is not None and not callable(self._run_preview_work):
            raise TypeError("a preview host requires a preview worker")
        self._request_preview_close = request_preview_close
        self._run_device_work = run_device_work
        self._run_safe_work = run_safe_work
        self._run_completion_work = run_completion_work
        if (run_device_work is None) != (run_safe_work is None):
            raise ValueError("device command and SAFE workers must be supplied together")
        if run_device_work is not None and (
            not callable(run_device_work) or not callable(run_safe_work)
        ):
            raise TypeError("device command and SAFE workers must be callable")
        self._preview_busy = False
        self._preview_pending: tuple[PulseSequence, bool, str | None] | None = None
        #: A scan program is running on the preview worker.
        self._scan_running = False
        self._preview_close_requested = False
        self._preview_mounted = False
        #: The plotting host behind the preview.  Built once and updated,
        #: and the thing a save actually goes through.
        self._preview_host: Any = None
        #: Whether the page that draws is the page on screen.  Asked of the
        #: window rather than assumed, so a host wired to an already-open
        #: window agrees with what the operator is actually looking at.
        self._preview_on_screen = (
            str(getattr(view, "current_page", "")) == PREVIEW_PAGE
        )
        #: A size the operator picked, which sticks until the content changes
        #: shape; None means the content chooses.
        self._pinned_size: str | None = None
        self._preview_selectors = False
        #: Runtime-only scan feedback.  Authored source, rows and repeats live
        #: only in ``_state`` beside the sequence they belong to.
        self._scan_progress = ""
        self._applied_scan: tuple[
            str,
            tuple[str, ...],
            tuple[tuple[str, ...], ...],
        ] | None = None
        self._held_point: int | None = None
        #: What the last Hold or Step left playing: the revision its row was
        #: resolved from and the digest the board reported for it.
        self._held_program: tuple[int, str] | None = None
        # An editor with a sequencer can fire what it is showing; without one
        # it is a viewer, and says so rather than offering a dead button.
        if sequencer is not None and device_use is None:
            raise ValueError(
                "an injected sequencer requires its session device-use coordinator"
            )
        if sequencer is not None and dial is not None:
            raise ValueError("an injected sequencer cannot also have a dial authority")
        self.sequencer = sequencer
        self._config_loaded_path = "" if sequencer is None else str(sequencer.config_source)
        self._config_loaded_values = {} if sequencer is None else sequencer.config_values()
        self._config_path = self._config_loaded_path
        self._config_rows = self._config_saved_rows = tuple(
            (key, format_quantity(value, "1"), unit)
            for key, (value, unit) in self._config_loaded_values.items()
        )
        self.device_use = device_use if device_use is not None else DeviceUseCoordinator()
        self._device_owner = object()
        self._drive_lease: DeviceLease | None = None
        self._finite_run: int | None = None
        self._device_busy = False
        self._device_operation = 0
        self._device_done: Event | None = None
        self._stop_busy = False
        #: One status question on its way to the board, or none.  A second
        #: request while it is pending sends nothing; the answer serves both.
        self._status_in_flight = False
        #: What to do once the next board answer has been shown.
        self._status_followups: list[Callable[[], None]] = []
        # How to get one.  Dialling belongs to whoever knows what a "remote
        # server" is on this bench, which is the composition root, not here.
        self._dial = dial
        #: The board's own answer, refreshed whenever the window is about to
        #: show it.  Never written from a command site.
        self._board_state = BoardState()
        #: The digest of what the screen would compile to, cached against the
        #: revision that produced it: comparing against the board costs one
        #: compile per edit, not one per refresh.
        self._digest_revision = -1
        self._digest = ""
        #: The board this editor is attached to, as it described itself.
        self.board = None
        self.pins: dict[str, str] = {}
        self._board_target = None
        self._board_step_ns = None
        # Offered, not remembered: the address a board is usually at belongs to
        # whoever knows what a pulse server is, and arrives from there.
        self._connection_locked = sequencer is not None
        self._connection_choices = (
            (ConnectionChoiceVM(str(connection_label), CONNECTION_GIVEN),)
            if self._connection_locked
            else STANDALONE_CONNECTION_CHOICES
        )
        self.connection = (
            (CONNECTION_GIVEN, "")
            if self._connection_locked
            else (CONNECTION_OFFLINE, str(default_endpoint))
        )
        self._owns_sequencer = False
        self._connection_status = "connected" if sequencer is not None else "not connected"
        #: The last model handed to the schedule page, so a new one can be
        #: recognised as new without every caller remembering to say so.
        self._shown: ScheduleVM | None = None
        self._connect()
        if self.sequencer is not None:
            if not self.adopt_board():
                raise RuntimeError("the injected sequencer did not provide its board description")
        else:
            self.refresh()

    @property
    def state(self) -> PulseEditorState:
        """The immutable authoring value currently projected by the editor."""

        return self._state

    @property
    def sequence(self) -> PulseSequence | None:
        """The sequence inside the sole authoring state."""

        return self._state.sequence

    def _accept_state(self, candidate: PulseEditorState) -> None:
        """Replace current authoring atomically and invalidate derived views."""

        if candidate == self._state:
            return
        if candidate.scan_rows != self._state.scan_rows:
            # A held row is an index into the table it was held in.
            self._held_point = None
        self._state = candidate
        self.revision += 1
        self._digest_revision = -1

    def _edit_state(self, **changes: Any) -> None:
        sequence = changes.get("sequence")
        if sequence is not None and "scan_rows" not in changes:
            before_slots = () if self.sequence is None else tuple(b.field_id for b in self.sequence.scan_bindings)
            if tuple(b.field_id for b in sequence.scan_bindings) != before_slots:
                changes.update(scan_rows=(), scan_source_dirty=bool(self._state.scan_source))
        visible = self._state.visible_ports
        if visible is not None and sequence is not None and "visible_ports" not in changes:
            # The shown set names ports of one target.  Across a target
            # change (Target page, Connect, Sync) it keeps the ports the new
            # target still has, and shows the ones it did not have before --
            # nobody chose to hide a port that did not exist.
            before = self._state.sequence.target.by_key if self._state.sequence is not None else {}
            changes["visible_ports"] = frozenset(
                key for key in sequence.target.by_key if key in visible or key not in before
            )
        self._accept_state(replace(self._state, **changes))

    def _edited_candidate(self, action: str, args: tuple) -> PulseSequence | None:
        if self.sequence is None:
            return None
        try:
            candidate, dropped = prune_orphaned_bindings(_sequence_edited(self.sequence, action, args))
            if dropped:
                self._warn("unbound " + ", ".join(dropped) + ": the field is no longer set here")
            return candidate
        except (TypeError, ValueError, KeyError) as error:
            self._warn(str(error))
            return None

    def _refresh_component_document(self) -> None:
        sequence = self.sequence
        contexts = () if sequence is None else tuple((c.component_id, c.name) for c in sequence.components)
        if self._component_context not in dict(contexts):
            self._component_context = ""
        context = self._component_context
        fragment = extract_subpulse(sequence, context) if context else self._subpulse_draft
        self._component_projection_revision += 1
        vm = None
        bindings = ()
        if fragment is not None:
            selected = fragment.to_sequence()
            vm = project_schedule(
                selected, revision=self._component_projection_revision,
                visible_ports=self._state.visible_ports if context else self._subpulse_visible_ports,
                config_values=self._active_config_values(), pins=self.pins if context else None,
            )
            if context:
                # HOLD reads the real preceding Pulse, not an isolated zero-state
                # preview of the fragment. IDs are still the parent's own IDs.
                vm = replace(vm, periods=tuple(project_period(
                    sequence, period, visible_ports=self._state.visible_ports,
                    config_values=self._active_config_values(), scan_active=self._scan_armed(),
                ) for period in selected.periods))
            bindings = tuple((b.field_id, field_label(selected, b.field_ref), b.config_key)
                             for b in selected.config_bindings)
        self.view.set_component_document(
            vm, contexts=contexts, context_id=context,
            path="" if context else self._subpulse_path,
            dirty=(self._state != self._saved_state if context else self._subpulse_draft != self._subpulse_saved),
            bindings=bindings, config_names=tuple(self._active_config_values()),
            busy=False,
        )

    def _discard_subpulse_edits(self) -> bool:
        return self._subpulse_draft == self._subpulse_saved or self.view.confirm_component_discard()

    def component_action(self, action: str, payload: object) -> None:
        """Component file and grouping commands; all scientific edits use zlc_pulse."""
        try:
            if action in {"context", "edit"}:
                context = str(payload or "")
                if context and (self.sequence is None or context not in {c.component_id for c in self.sequence.components}):
                    raise ValueError("that component no longer exists in this Pulse")
                self._component_context = context
            elif action in {"new", "open"}:
                if not self._discard_subpulse_edits():
                    return
                if action == "open":
                    chosen = self.view.ask_open_path(
                        "Open Subpulse", self._subpulse_path or self.pulses_directory,
                        "ZLC Subpulse (*.subpulse.json *.json);;All files (*)",
                    )
                    if not chosen:
                        return
                    self._subpulse_draft = self._subpulse_saved = read_subpulse(chosen)
                    self._subpulse_path = str(chosen)
                else:
                    from zlc_pulse import pulse_target_from_xdc
                    target = self.sequence.target if self.sequence is not None else self._board_target or pulse_target_from_xdc()
                    step = self.sequence.time_step_ns if self.sequence is not None else self._board_step_ns or 1e9 / self._compiler_target()[1]
                    self._subpulse_draft = Subpulse(
                        "component", target, step,
                        (PulsePeriod("period1", NEW_PULSE_PERIOD_NS, "ns", (0,) * len(target.raw_lanes)),),
                    )
                    self._subpulse_saved = None
                    self._subpulse_path = ""
                self._component_context = ""
                self._subpulse_visible_ports = None
            elif action in {"save", "save_as", "export"}:
                if action == "save" and self._component_context:
                    self.save_pulse()
                    self._refresh_component_document()
                    return
                context = str(payload or self._component_context) if action == "export" else self._component_context
                fragment = extract_subpulse(self.sequence, context) if context else self._subpulse_draft
                if fragment is None:
                    raise ValueError("create or open a Subpulse first")
                target = self._subpulse_path if not context and action == "save" else ""
                if not target:
                    target = self.view.ask_save_path(
                        "Save Subpulse", str(Path(self.pulses_directory or ".") / f"{fragment.name}.subpulse.json"),
                        "ZLC Subpulse (*.subpulse.json *.json);;All files (*)",
                    )
                if not target:
                    return
                path = Path(target)
                if not path.suffix:
                    path = path.with_suffix(".subpulse.json")
                if path.suffix.lower() != ".json":
                    raise ValueError("a Subpulse file must have a .json suffix")
                path.parent.mkdir(parents=True, exist_ok=True)
                write_subpulse(path, fragment)
                if not context:
                    self._subpulse_path = str(path)
                    self._subpulse_saved = fragment
                self._done(f"saved Subpulse {path.name}")
            elif action == "insert":
                fragment = (extract_subpulse(self.sequence, self._component_context)
                            if self._component_context else self._subpulse_draft)
                if fragment is None:
                    raise ValueError("create or open a Subpulse first")
                old = set() if self.sequence is None else {c.component_id for c in self.sequence.components}
                candidate = insert_subpulse(self.sequence, fragment)
                self._component_context = next(c.component_id for c in candidate.components if c.component_id not in old)
                self._apply(candidate)
            elif action == "group":
                if self.sequence is None:
                    raise ValueError("open a Pulse before grouping its periods")
                selected = set(tuple(item) for item in payload)
                if any(kind == "component" for kind, _key in selected):
                    raise ValueError("ungroup existing components before making a new group")
                ids = tuple(key for key in self.sequence.period_by_id if ("period", key) in selected)
                if not ids:
                    raise ValueError("select the periods to group; Shift-click adds or removes individual items")
                name = _unique_id(tuple(c.name for c in self.sequence.components), "Component ")
                before = {c.component_id for c in self.sequence.components}
                candidate = group_component(self.sequence, ids, name)
                created = next(c.component_id for c in candidate.components if c.component_id not in before)
                included = {b.bracket_id for b in extract_subpulse(candidate, created).brackets}
                if any(key.rpartition(":")[0] not in included for kind, key in selected if kind == "bracket"):
                    raise ValueError("select the complete contents of each selected bracket; no unselected periods will be added")
                self._apply(candidate)
                self.view.focus_component_name(created)
            elif action in {"ungroup", "remove"}:
                operation = ungroup_component if action == "ungroup" else remove_component
                self._apply(operation(self.sequence, str(payload)))
            elif action == "rename":
                key, name = payload
                if key:
                    self._apply(replace(self.sequence, components=tuple(
                        replace(c, name=str(name)) if c.component_id == key else c for c in self.sequence.components
                    )))
                elif self._subpulse_draft is not None:
                    self._subpulse_draft = replace(self._subpulse_draft, name=str(name))
            elif action in {"insert_period", "insert_spacer"}:
                key, before = payload
                self._edit_component_contents(str(key), action, (before,))
            else:
                raise ValueError(f"unknown Component action {action!r}")
        except (TypeError, ValueError, KeyError, OSError) as error:
            self._warn(f"cannot {action} Component: {error}")
        self._refresh_component_document()

    def edit_component(self, action: str, args: object) -> None:
        self._edit_component_contents(self._component_context, action, tuple(args))

    def _edit_component_contents(self, context: str, action: str, args: tuple) -> None:
        if action == "visible_ports":
            if context:
                self.set_visible_ports(args[0])
            else:
                self._subpulse_visible_ports = frozenset(args[0])
            self._refresh_component_document()
            return
        try:
            if context:
                if action == "document_name":
                    self.component_action("rename", (context, args[0]))
                    return
                if action in {"period_name", "duration", "digital", "analog", "binding", "config_binding"}:
                    # Real parent context supplies incoming DAC carry and stable IDs.
                    self._apply(self._edited_candidate(action, args))
                    self._refresh_component_document()
                    return
                fragment = extract_subpulse(self.sequence, context)
            else:
                fragment = self._subpulse_draft
            if fragment is None:
                raise ValueError("create or open a Subpulse first")
            candidate, dropped = prune_orphaned_bindings(_sequence_edited(fragment.to_sequence(), action, args))
            if dropped:
                self._warn("unbound " + ", ".join(dropped) + ": the field is no longer set here")
            edited = Subpulse(candidate.name, candidate.target, candidate.time_step_ns,
                              candidate.periods, candidate.bindings, candidate.brackets)
            if context:
                self._apply(replace_component(self.sequence, context, edited))
            else:
                self._subpulse_draft = edited
        except (TypeError, ValueError, KeyError) as error:
            self._warn(f"cannot edit Component: {error}")
        self._refresh_component_document()

    # --------------------------------------------------------------- wiring

    def _guarded(self, handler):
        """Wrap one view-signal handler at the one Qt-slot boundary owner.

        Wire bound methods, not lambdas: the boundary holds a lambda strongly,
        and one closing over this presenter would keep it alive with the view.
        """

        return _guarded_slot(handler, handler.__name__, on_error=self._slot_error)

    def _slot_error(self, error: Exception) -> None:
        self._warn(f"internal error: {type(error).__name__}: {error}")

    def _connect(self) -> None:
        view = self.view
        view.document_name_committed.connect(self._guarded(self.set_document_name))
        view.port_label_committed.connect(self._guarded(self.set_port_label))
        view.period_name_committed.connect(self._guarded(self.set_period_name))
        view.duration_committed.connect(self._guarded(self.set_duration))
        view.digital_committed.connect(self._guarded(self.set_digital))
        view.analog_committed.connect(self._guarded(self.set_analog))
        view.delay_committed.connect(self._guarded(self.set_delay))
        view.insert_period_requested.connect(self._guarded(self.insert_period))
        view.insert_spacer_requested.connect(self._guarded(self.insert_spacer))
        view.reorder_items_requested.connect(self._guarded(self.reorder_items))
        view.remove_items_requested.connect(self._guarded(self.remove_items))
        view.bracket_committed.connect(self._guarded(self.set_bracket))
        view.bracket_add_requested.connect(self._guarded(self.add_bracket))
        view.run_repeats_committed.connect(self._guarded(self.set_run_repeats))
        view.visible_ports_committed.connect(self._guarded(self.set_visible_ports))
        view.fill_port_requested.connect(self._guarded(self.fill_port))
        view.clear_port_requested.connect(self._guarded(self.clear_port))
        # From the button that raises it.  The schedule page also declared a
        # clear_all_requested that nothing ever emitted, and this listened to
        # that one -- so the toolbar's Clear All did nothing at all.
        view.clear_all_requested.connect(self._guarded(self.clear_all))
        view.page_changed.connect(self._guarded(self.show_page))
        view.binding_committed.connect(self._guarded(self.set_binding))
        view.scan_array_load_requested.connect(self._guarded(self.load_scan_array))
        view.scan_source_edited.connect(self._guarded(self.edit_scan_source))
        view.feedback_requested.connect(self._guarded(self._warn))
        view.scan_repeats_committed.connect(self._guarded(self.set_scan_repeats))
        view.scan_hold_requested.connect(self._guarded(self._hold_from_view))
        view.scan_step_requested.connect(self._guarded(self._step_from_view))
        view.scan_program_load_requested.connect(self._guarded(self.load_scan_program))
        view.scan_template_requested.connect(self._guarded(self.write_scan_template))
        view.scan_run_requested.connect(self._guarded(self.run_scan_program))
        view.scan_array_save_requested.connect(self._guarded(self.save_scan_array))
        view.scan_progress_refresh_requested.connect(
            self._guarded(self._scan_progress_from_view)
        )
        view.connection_requested.connect(self._guarded(self._connect_from_view))
        view.device_label_changed.connect(self._guarded(self._set_device_label))
        view.fire_requested.connect(self._guarded(self._fire_from_view))
        view.stop_requested.connect(self._guarded(self.stop))
        view.sync_requested.connect(self._guarded(self._sync_from_view))
        view.save_requested.connect(self._guarded(self.save_pulse))
        view.config_new_requested.connect(self._guarded(self.new_config))
        view.config_load_requested.connect(self._guarded(self.load_config_values))
        view.config_refresh_requested.connect(self._guarded(self.refresh_config_values))
        view.config_save_requested.connect(self._guarded(self.save_config_values))
        view.config_save_as_requested.connect(self._guarded(self._save_config_as))
        view.config_unload_requested.connect(self._guarded(self.unload_config))
        view.config_entries_edited.connect(self._guarded(self.edit_config_entries))
        view.config_binding_committed.connect(self._guarded(self.set_config_binding))
        view.load_requested.connect(self._guarded(self.ask_for_pulse))
        view.preview_include_off_toggled.connect(self._guarded(self._on_include_off))
        view.preview_size_committed.connect(self._guarded(self.set_preview_size))
        view.preview_selectors_toggled.connect(self._guarded(self.set_preview_selectors))
        view.preview_save_requested.connect(self._guarded(self.save_preview_image))
        view.target_apply_requested.connect(self._guarded(self.apply_target))
        view.component_action_requested.connect(self._guarded(self.component_action))
        view.component_edit_requested.connect(self._guarded(self.edit_component))

    # ------------------------------------------------------------- the pulse

    def ask_for_pulse(self) -> bool:
        """Let the operator pick a pulse file, and open it.

        The window owns the dialog; this owns what to do with the answer.  The
        same JSON reader serves the session, so the editor cannot show a pulse
        assembled differently from the one that will fire.
        """

        # Asked BEFORE the dialog, as Config's Load asks: choosing a file and
        # then being told the edits on screen would go wastes the choice.
        if not self._discard_pulse_edits():
            return False
        start = str(Path(self.path).parent if self.path else self.pulses_directory or "")
        chosen = self.view.ask_open_path(
            "Open pulse", start,
            "ZLC pulse (*.json);;All files (*)",
        )
        if not chosen:
            return False
        return self.open_pulse(chosen)

    def open_pulse(self, path: str) -> bool:
        """Replace what is being edited with one ``zlc.pulse`` JSON file.

        Attached, a pulse written for other wiring is moved onto the board
        exactly as Connect moves the one already open: the file's own wiring
        was a target the board then refused to load.  One written for this
        board's wiring and clock opens as its file has it.
        """

        try:
            candidate = read_pulse(path)
        except Exception as error:
            self._warn(f"cannot open {Path(path).name}: {error}")
            return False
        self._saved_state = candidate
        self.path = str(path)
        self._accept_state(candidate)
        if self.board is not None and self.sequence is not None:
            self._align_sequence_to(self.board)
        self.refresh()
        return True

    def _set_device_label(self, label: str) -> None:
        if self._connection_locked:
            self._connection_choices = (ConnectionChoiceVM(str(label), CONNECTION_GIVEN),)
            self._show_connection(self._connection_status)

    def _config_values_directory(self) -> Path | None:
        """Where saved sets of the board's calibrated numbers live."""

        base = Path(self.path).parent if self.path else Path(self.pulses_directory or "")
        if not str(base) or str(base) == ".":
            return None
        return base.parent / CONFIG_VALUES_DIRECTORY

    def _save_folder(self) -> Path:
        """Where a file written beside this pulse goes: the pulse's folder,
        or before it has one the workspace's pulses folder.

        Not the process's working directory, which is wherever the launcher
        happened to be started from; only an editor with no workspace at all
        (a notebook) is left with that.
        """

        if self.path:
            return Path(self.path).parent
        if self.pulses_directory:
            return Path(self.pulses_directory)
        return Path.cwd()

    def _binding_records(self) -> tuple[BindingRecord, ...]:
        """Read-only physical fields, never separately authored aliases."""
        if self.sequence is None:
            return ()
        return tuple(
            BindingRecord(
                b.field_id, field_label(self.sequence, b.field_ref), b.scan, b.source,
                "" if component is None else component.component_id,
                "Pulse" if component is None else component.name,
            )
            for b in self.sequence.bindings if b.scan or b.source == "api"
            for component in (self.sequence.component_for_period(b.field_ref.period_id),)
        )

    def _config_dirty(self) -> bool:
        return self._config_rows != self._config_saved_rows

    def _discard_config_edits(self) -> bool:
        return not self._config_dirty() or self.view.confirm_config_discard()

    def _discard_pulse_edits(self) -> bool:
        """Whether the pulse on screen may be replaced, asked only when it
        differs from its file -- the difference the Save button already shows.

        Close, Load and Clear All each threw the draft away and asked nothing,
        while the Config tab beside it, holding far less work, asked first.
        """

        return self._state == self._saved_state or self.view.confirm_pulse_discard()

    def new_config(self) -> None:
        if not self._discard_config_edits():
            return
        self._config_path = ""
        self._config_rows = self._config_saved_rows = ()
        self._refresh_config_page()

    def edit_config_entries(self, rows: object) -> None:
        self._config_rows = tuple(tuple(str(v) for v in row) for row in rows)
        self._refresh_config_page()

    def _active_config_values(self) -> Mapping[str, tuple[float, str]]:
        return (
            self.sequencer.config_values()
            if self.sequencer is not None else self._config_loaded_values
        )

    def _active_config_path(self) -> str:
        return (
            str(self.sequencer.config_source)
            if self.sequencer is not None else self._config_loaded_path
        )

    def _refresh_config_page(self) -> None:
        active = self._active_config_values()
        scan_active = self._scan_armed()
        bindings = []
        binding_groups = []
        if self.sequence is not None:
            for b in self.sequence.config_bindings:
                default = pulse_field_value(self.sequence, b.field_ref, b.unit)
                effective = active.get(b.config_key) if b.config_key else None
                bindings.append((
                    b.field_id, field_label(self.sequence, b.field_ref), b.config_key,
                    f"{format_quantity(default, '1')} {b.unit}",
                    "" if effective is None else f"{format_quantity(effective[0], '1')} {effective[1]}",
                    "Scan table" if b.scan and scan_active else
                    "Using default" if effective is None else "Config override",
                ))
                component = self.sequence.component_for_period(b.field_ref.period_id)
                binding_groups.append((
                    b.field_id, "" if component is None else component.component_id,
                    "Pulse" if component is None else component.name,
                ))
        self.view.set_config_page(ConfigPageRecord(
            file_path=self._config_path, dirty=self._config_dirty(),
            entries=self._config_rows, bindings=tuple(bindings),
            available_names=tuple(active),
            active_path=self._active_config_path(),
            busy=self._device_busy or self._stop_busy,
            binding_groups=tuple(binding_groups),
        ))

    def set_config_binding(self, field_id: str, key: str) -> None:
        self._apply(self._edited_candidate("config_binding", (field_id, key)))

    def _config_file_operation(
        self, path: str, *, entries: Mapping[str, tuple[float, str]] | None = None,
        load_draft: bool = True, activate: bool = True,
    ) -> bool:
        """Use the ordinary worker for one Config read/save/bind operation."""
        if not self._device_available():
            return False
        sequencer = self.sequencer
        def work(_operation: int):
            if entries is not None:
                write_config_values(path, entries)
            if sequencer is not None and activate:
                sequencer.load_config_file(path or None)
                values = sequencer.config_values()
            else:
                values = read_config_values(path) if path else {}
            return values

        # Delivered even after a Stop: the file is written (and maybe
        # activated) by then, and a Config page left on the old table would
        # offer to save the edits that already went.
        def delivered(values: object, error: BaseException | None) -> None:
            if error is not None:
                self._warn(f"cannot update Config: {error}")
                self._refresh_config_page()
                return
            if activate:
                self._config_loaded_path = path
                self._config_loaded_values = dict(values)
            if load_draft:
                self._config_path = path
                self._config_rows = self._config_saved_rows = tuple(
                    (key, format_quantity(value, "1"), unit)
                    for key, (value, unit) in values.items()
                )
            self._digest_revision = -1
            self.refresh()

        if self._run_device_work is not None:
            return self._run_device_command(
                work, delivered, summary="Updating Config...", always_deliver=True
            )
        try:
            values = work(0)
        except Exception as error:
            delivered(None, error)
            return False
        delivered(values, None)
        return True

    def load_config_values(self) -> bool:
        if not self._discard_config_edits():
            return False
        directory = self._config_values_directory()
        chosen = self.view.ask_open_path(
            "Load Config", self._config_path or str(directory or ""), "Config (*.json)",
        )
        if not chosen:
            return False
        return self._config_file_operation(str(Path(chosen).resolve()))

    def refresh_config_values(self) -> bool:
        if not self._config_path or not self._discard_config_edits():
            return False
        return self._config_file_operation(
            self._config_path, activate=self._config_path == self._active_config_path(),
        )

    def unload_config(self) -> bool:
        return self._config_file_operation("", load_draft=False)

    def save_config_values(self, *, save_as: bool = False) -> bool:
        # Only what a mapping cannot say is checked here; the codec's entry
        # rule (a Config name, a finite number, a unit) runs when the file is
        # written, and its refusal reaches the operator as "cannot update".
        entries = {}
        for name, text, unit in self._config_rows:
            key = name.strip()
            if key in entries:
                self._warn(f"duplicate Config name: {key}")
                return False
            try:
                value = float(text)
            except ValueError:
                self._warn(f"Config {key}: value must be a number")
                return False
            entries[key] = (value, unit.strip())
        target = self._config_path
        if save_as or not target:
            directory = self._config_values_directory()
            target = self.view.ask_save_path(
                "Save Config", target or str((directory or Path.cwd()) / CURRENT_CONFIG_VALUES),
                "Config (*.json)",
            )
            if not target:
                return False
        return self._config_file_operation(str(Path(target).with_suffix(".json").resolve()), entries=entries)

    def _save_config_as(self) -> bool:
        return self.save_config_values(save_as=True)

    def _effective_sequence(self, sequence: PulseSequence) -> PulseSequence:
        """Preview uses saved active values, never the Config editor draft."""
        base = sequence if self._scan_armed() else resolve_scan_point(sequence)
        effective = apply_config_values(sequence, self._active_config_values(), current=base)[0]
        # Capability marks remain visible when this preview uses defaults.
        return effective if effective.bindings == sequence.bindings else replace(effective, bindings=sequence.bindings)

    def start_new_pulse(self) -> bool:
        """Begin a pulse on the board this bench actually has.

        A sequence needs a target and at least one period before it is a legal
        sequence at all, so "new" cannot mean empty.  The target comes from the
        deployed board's own pin map rather than from an invented default: a
        pulse authored against an imaginary board compiles and then does
        nothing recognisable.

        The shape is the one the established PulseGUI opens with, and it is a
        starting point rather than a minimum: one period is not something an
        operator can look at, because a pulse is made of the CHANGES between
        periods and one period has none.  A single tick-long period is worse
        still -- 20 ns is below anything the timeline can show, so the window
        was legal and looked empty.
        """

        from zlc_pulse import PulsePeriod, PulseSequence, pulse_target_from_xdc

        try:
            # The attached board first: a pulse authored against this machine's
            # files while a different board is connected compiles and then does
            # nothing recognisable.
            target = self._board_target or pulse_target_from_xdc()
            # The clock the compiler will be held to, not a copy of it.
            step_ns = self._board_step_ns or 1e9 / self._compiler_target()[1]
            safe = (0,) * len(target.raw_lanes)
            # The first digital output high in P1, everything safe in P2: the
            # smallest thing that is actually a pulse, so the preview draws an
            # edge and the operator can see what an edit does to it.
            driven = list(safe)
            first_digital = next(
                (port for port in target.ports if port.kind == "digital"), None
            )
            if first_digital is not None:
                driven[target.raw_lanes.index(first_digital.lanes[0])] = 1
            candidate = PulseSequence(
                name="untitled",
                target=target,
                time_step_ns=step_ns,
                periods=(
                    PulsePeriod(
                        "period1", NEW_PULSE_PERIOD_NS, "ns", tuple(driven)
                    ),
                    PulsePeriod("period2", NEW_PULSE_PERIOD_NS, "ns", safe),
                ),
            )
        except Exception as error:
            self._warn(f"cannot start a pulse for this board: {error}")
            return False
        self._accept_state(PulseEditorState(sequence=candidate))
        self.path = ""
        self.refresh()
        return True

    # ----------------------------------------------------------------- edits

    def set_document_name(self, name: str) -> None:
        """Rename the pulse.  Its shape is untouched, so only the header moves."""

        if self.sequence is None:
            return
        candidate = self._rebuilt(name=str(name))
        if candidate is None:
            return
        self._edit_state(sequence=candidate)
        self.view.set_title(f"PulseGUI - {candidate.name}")
        self._refresh_summary()

    def set_port_label(self, key: str, label: str) -> None:
        """Rename one output.  A label is not a shape, so nothing is rebuilt."""

        if self.sequence is None:
            return
        # Through the same rebuild the Target page uses.  There were two, and
        # this one dropped the package pin map: renaming an output here erased
        # the lane-to-pin map of the open pulse, while renaming the identical
        # output from the Target page kept it.
        renamed = self._retarget_labels(self.sequence.target, {str(key): str(label)})
        if renamed is None:
            return
        candidate = self._rebuilt(target=renamed)
        if candidate is None:
            return
        self._edit_state(sequence=candidate)
        self.view.set_port_label(str(key), str(label))
        self._refresh_summary()
        self.refresh_target()
        self.refresh_preview()

    def set_period_name(self, period_id: str, name: str) -> None:
        self._edit_period(period_id, "period_name", str(name))
        self._refresh_scan_page()

    def set_duration(self, period_id: str, value: object, unit: str) -> None:
        """How long this period lasts, rounded onto the board's clock.

        A duration between ticks is not a mistake to argue with -- it is a
        number the board would round anyway.  Refusing it put a blocking
        dialog in front of an operator typing a number, and typing is when
        every intermediate value is briefly wrong.
        """

        self._edit_period(period_id, "duration", value, unit)

    def set_digital(self, period_id: str, port_key: str, high: bool) -> None:
        self._edit_period(period_id, "digital", port_key, high)

    def set_analog(self, period_id: str, port_key: str, mode: str, value: object) -> None:
        """Set one DAC's level in one period; an empty value removes the step.

        The mode is the model's own vocabulary (an edge changes at the period
        boundary, a ramp sweeps across it), passed through rather than
        translated -- a second set of names for the same thing is how a GUI and
        its hardware end up meaning different pulses.

        HOLD is the exception, and it is not a third mode: it is what a period
        with no step for this port already means, and the projection says so.
        Coming back the other way it was built into an ``AnalogStep("hold")``,
        which the model refuses -- out of a Qt slot, so choosing Hold in the
        shipped window ended the process with no traceback at all.  It removes
        the step instead, which is the same statement read forwards.
        """

        # A level holds until something sets it again, so this changes what
        # every later card displays as well as this one.
        self._edit_period(period_id, "analog", port_key, mode, value, ripples_forward=True)

    def set_delay(self, port_key: str, value: object, unit: str) -> None:
        if self.sequence is None:
            return
        delays = tuple(item for item in self.sequence.delays if item.port != port_key)
        # A delay may legitimately be zero, so it rounds with no floor.
        amount = self._on_grid(value or 0.0, unit, "delay", minimum=None)
        if amount is None:
            return
        if amount:
            delays = delays + (OutputDelay(str(port_key), amount, str(unit)),)
        self._apply_value(
            self._rebuilt(delays=delays), port_key=str(port_key)
        )

    def insert_period(self, before_item: tuple[str, str] | None) -> None:
        """Add one period, copying the state its neighbour was already in.

        With no pulse open this is how one starts: the first period is what
        makes a sequence legal, so asking for a period IS asking for a pulse.

        A new period that resets every lane to zero silently inserts a gap in
        the middle of a sequence; copying the neighbour makes the insertion
        visible as a longer hold, which is what the operator can then edit.
        """

        if before_item is not None and (
            not isinstance(before_item, tuple) or len(before_item) != 2
            or any(not isinstance(value, str) for value in before_item)
        ):
            raise TypeError("insert target must be a schedule item tuple or None")
        if self.sequence is None:
            self.start_new_pulse()
            return
        self._apply(self._edited_candidate("insert_period", (before_item,)))

    def insert_spacer(self, before_item: tuple[str, str] | None) -> None:
        """Add a spacer: time between two periods for a slow device to settle.

        Its lines start as what BOTH neighbours agree on -- a line high on one
        side alone stays low, so a fresh spacer never does more than the
        periods around it -- and the operator raises what the device needs.
        Every DAC holds.  Its length copies the last spacer in the pulse, or
        is one millisecond when it is the first.
        """

        if before_item is not None and (
            not isinstance(before_item, tuple) or len(before_item) != 2
            or any(not isinstance(value, str) for value in before_item)
        ):
            raise TypeError("insert target must be a schedule item tuple or None")
        if self.sequence is None:
            self._warn("add a period first: a spacer is time between periods")
            return
        self._apply(self._edited_candidate("insert_spacer", (before_item,)))

    def reorder_items(self, order: Sequence[tuple[str, str]]) -> None:
        self._apply(self._edited_candidate("reorder_items", (order,)))

    def remove_period(self, period_id: str) -> None:
        self.remove_items((("period", period_id),))

    def remove_items(self, items: object) -> None:
        self._apply(self._edited_candidate("remove_items", (tuple(items),)))

    def set_bracket(self, bracket_id: str, start: object, end: object, count: int) -> None:
        """Move one bracket's inclusive anchors, or recount it.

        One missing neighbour locates an empty bracket at the first or last
        timeline gap.  Only explicit removal takes a bracket away.
        """

        self._apply(self._edited_candidate("bracket", (bracket_id, start, end, count)))

    def add_bracket(self, start: object, end: object, count: int) -> None:
        """A new bracket around these periods; nested or disjoint, the model decides."""

        self._apply(self._edited_candidate("bracket_add", (start, end, count)))

    def remove_bracket(self, bracket_id: str) -> None:
        self.remove_items((("bracket", bracket_post_key(bracket_id, "start")),))

    def set_run_repeats(self, repeats: int) -> None:
        """Persist complete-Pulse runs per scan point; zero means infinite."""

        if repeats == self.sequence.run_repeats:
            return
        self._apply(self._rebuilt(run_repeats=repeats))

    def set_visible_ports(self, ports: object) -> None:
        """Which ports have rows.  A VALUE change, so it goes the value way.

        Which outputs are worth looking at is not the shape of the pulse, and
        the page has a setter that says exactly that -- declared in
        docs/pulse-views.md and never called, so every Hide Off went through a
        whole re-projection instead.  The model this presenter believes is on
        screen is kept in step, or the next full push would not recognise
        itself as different.  The view re-flags its rows; the count it shows
        is the projection's wording, pushed with the summary as a value edit
        pushes it.
        """

        visible = None if ports is None else frozenset(str(key) for key in ports)
        self._edit_state(visible_ports=visible)
        if self._shown is None or self.sequence is None:
            self.refresh()
            return
        keys = tuple(
            port.key
            for port in self._shown.ports
            if visible is None or port.key in visible
        )
        self.view.set_visible_ports(keys)
        shown = set(keys)
        flagged = tuple(
            replace(port, visible=port.key in shown) for port in self._shown.ports
        )
        self._shown = replace(
            self._shown, ports=flagged, visible_text=_visible_text(flagged)
        )
        self._refresh_summary()
        self._render_run_state()

    def fill_port(self, port_key: str) -> None:
        """Turn one digital port on in every period, and nothing else."""

        self._set_port_everywhere(port_key, high=True)

    def clear_port(self, port_key: str) -> None:
        """Turn one port off in every period, and nothing else.

        A digital port goes low; an analog port loses its steps.  The port's
        delay is not a period's state -- it is the output's own timing --
        and stays: turning a channel off is not un-calibrating it.
        """

        self._set_port_everywhere(port_key, high=False)

    def _set_port_everywhere(self, port_key: str, *, high: bool) -> None:
        port = self.sequence.target.by_key.get(str(port_key))
        if port is None:
            self._warn(f"{port_key} is not a port on this target")
            return
        if port.kind == "digital":
            index = self.sequence.target.raw_lanes.index(port.lanes[0])
            level = 1 if high else 0

            def edit(period: PulsePeriod) -> PulsePeriod:
                states = list(period.states)
                states[index] = level
                return replace(period, states=tuple(states))

        elif high:
            self._warn(f"{port_key} is an analog port; there is no level to fill it with")
            return
        else:

            def edit(period: PulsePeriod) -> PulsePeriod:
                return replace(
                    period,
                    analog_steps=tuple(
                        step for step in period.analog_steps if step.port != port_key
                    ),
                )

        self._apply(
            self._rebuilt(periods=tuple(edit(period) for period in self.sequence.periods))
        )

    def clear_all(self) -> None:
        """Clear authored content to one safe period, retaining file context.

        Asked first, as Config's New is: the path is kept, so a Save made by
        reflex afterwards overwrites the file with the blank pulse.
        """

        if self.sequence is None or not self._discard_pulse_edits():
            return
        sequence = self.sequence
        safe = (0,) * len(sequence.target.raw_lanes)
        blank = PulseSequence(
            name=sequence.name,
            target=sequence.target,
            time_step_ns=sequence.time_step_ns,
            periods=(
                PulsePeriod(
                    "period1",
                    sequence.time_step_ns,
                    "ns",
                    safe,
                ),
            ),
        )
        self._accept_state(
            PulseEditorState(sequence=blank, visible_ports=self._state.visible_ports)
        )
        self.refresh()

    # ------------------------------------------------------------- hardware

    def _compiler_target(self) -> tuple[object, float]:
        """The one deployed geometry/clock pair used by compile and scan scaling."""

        if self.sequencer is not None:
            if self.board is None:
                raise RuntimeError(
                    "the attached sequencer has no board description; refusing "
                    "to substitute this computer's board config"
                )
            return self.board.geometry, self.board.clock_hz

        from zlc_pulse import load_streamer_config

        config = load_streamer_config()
        if config["source"] is None:
            raise RuntimeError(
                "no streamer config was found, so the deployed board geometry is "
                "unknown; refusing to compile against built-in defaults"
            )
        return config["params"], config["clock_hz"]

    def compile(self, sequence: Any = None) -> tuple[Any, Any]:
        """The sequence as the board would receive it, and the program.

        Compiled against the DEPLOYED board's geometry, not against defaults:
        a preview that compiles for an imaginary board is exactly the kind of
        confirmation that survives until the bench proves it wrong.

        Both halves retain the authored Config defaults. The device applies
        its loaded Config at load/Fire; this pure compilation never reads a
        file or replaces the draft with an executable resolved sequence.
        """

        from zlc_pulse import compile_sequence

        sequence = sequence if sequence is not None else self.sequence
        if sequence is None:
            raise RuntimeError("no pulse is open, so there is nothing to compile")
        sequence.require_nonempty_brackets()
        geometry, clock_hz = self._compiler_target()
        sequencer = self.sequencer
        if sequencer is not None:
            return sequencer.compile_pulse(sequence, geometry, clock_hz)
        return sequence, compile_sequence(sequence, geometry, clock_hz)

    def connect_to(self, mode: str, endpoint: str) -> bool:
        """Attach this editor to a sequencer, or say why it could not.

        The previous connection is released first.  Two open connections to one
        board is not a state anything downstream can reason about, and the
        second one usually fails in a way that reads as the first one breaking.

        Connected, or failed, before returning: this is what the app calls
        at start-up and what a notebook calls.  The window's connection
        control goes through ``_connect_from_view``, which does the same on
        the device worker -- a socket connect alone may wait its whole
        timeout -- and changes what this editor is connected to only when
        that command has delivered.
        """

        request = self._connection_request(mode, endpoint)
        if request is None:
            return False
        mode, endpoint = request
        if mode == CONNECTION_OFFLINE:
            if not self._release_for_connection_change():
                return False
            self._show_disconnected(mode, endpoint)
            return True
        if not self._release_for_connection_change():
            return False
        try:
            self.sequencer = self._dial(mode, endpoint)
        except Exception as error:
            self._dial_failed(mode, endpoint, error)
            # The previous board was already forgotten; its ports must not
            # stay on screen (the worker path does the same).
            self.refresh()
            return False
        self._owns_sequencer = True
        self.connection = (mode, endpoint)
        had_sequence = self.sequence is not None
        if not self.adopt_board():
            failure_status = self._connection_status
            if not self._release_for_connection_change():
                self.refresh()
                return False
            self.connection = (CONNECTION_OFFLINE, endpoint)
            self._show_connection(failure_status)
            self.refresh()
            return False
        if not had_sequence:
            self.start_new_pulse()
        return True

    def _connection_request(self, mode: str, endpoint: str) -> tuple[str, str] | None:
        """One connection request checked the one way, or None once it was refused."""

        mode, endpoint = str(mode), str(endpoint)
        allowed = {str(choice.value) for choice in self._connection_choices}
        if mode not in allowed:
            raise ValueError(f"unknown connection mode {mode!r}")
        if self._connection_locked:
            raise RuntimeError("the experiment session owns this connection")
        if mode != CONNECTION_OFFLINE and self._dial is None:
            self._warn("this editor was built without a way to connect")
            return None
        return mode, endpoint

    def _connect_from_view(self, mode: str, endpoint: str) -> None:
        if self._run_device_work is None:
            self.connect_to(mode, endpoint)
            return
        request = self._connection_request(mode, endpoint)
        if request is not None:
            self._connect_on_worker(*request)

    def _connect_on_worker(self, mode: str, endpoint: str) -> bool:
        if not self._device_available():
            return False
        previous = self.sequencer if self._owns_sequencer else None
        holding_lease = self._drive_lease is not None
        dial = self._dial
        had_sequence = self.sequence is not None
        #: A previous board this editor was driving that could not be told
        #: to stop.  It does not stop the new connection -- the old board is
        #: unreachable either way -- but the operator has to be told.
        unsafe: list[BaseException] = []

        def work(_operation: int) -> object:
            if previous is not None:
                try:
                    failed_safe = self._hang_up(previous, holding_lease)
                except BaseException as error:  # noqa: BLE001 -- delivered
                    return "release", None, error
                if failed_safe is not None:
                    unsafe.append(failed_safe)
            if mode == CONNECTION_OFFLINE:
                return "offline", None, None
            try:
                sequencer = dial(mode, endpoint)
            except BaseException as error:  # noqa: BLE001 -- delivered
                return "dial", None, error
            describe = getattr(sequencer, "describe", None)
            if not callable(describe):
                return "described", (sequencer, None, self._board_state_for(sequencer)), None
            try:
                board = describe()
                state = self._board_state_for(sequencer)
            except BaseException as error:  # noqa: BLE001 -- delivered
                try:
                    self._hang_up(sequencer, False)
                except BaseException:  # noqa: BLE001 -- the first error is the one to show
                    pass
                return "describe", None, error
            return "described", (sequencer, board, state), None

        def delivered(result: object, failure: BaseException | None) -> None:
            if unsafe:
                self._warn(
                    "the pulse server connection ended before the previous "
                    f"board could be told to stop ({unsafe[0]}). If that "
                    "server is still running it drives the board safe when a "
                    "client disconnects; if it is not, THAT SEQUENCE IS STILL "
                    "PLAYING and the board must be stopped another way."
                )
            if failure is not None:
                self._show_connection(f"failed: {failure}")
                self._warn(f"cannot connect to {_connection_name(mode, endpoint)}: {failure}")
                self._render_run_state()
                return
            stage, payload, error = result
            if stage == "release":
                self._show_connection(f"disconnect failed: {error}")
                self._warn(f"cannot close the current sequencer connection: {error}")
                self._render_run_state()
                return
            if previous is not None:
                self._release_drive()
                self._forget_connection()
            if stage == "offline":
                self._show_disconnected(mode, endpoint)
                return
            if stage == "dial":
                self._dial_failed(mode, endpoint, error)
                self.refresh()
                return
            if stage == "describe":
                self.connection = (CONNECTION_OFFLINE, endpoint)
                self._describe_failed(error)
                self.refresh()
                return
            sequencer, board, state = payload
            self.sequencer = sequencer
            self._owns_sequencer = True
            self.connection = (mode, endpoint)
            if board is None:
                self._adopt_board_state(state)
                self._show_connection("connected (board did not describe itself)")
                self.refresh()
                return
            self._adopt_described(board, state)
            if not had_sequence:
                self.start_new_pulse()

        # Delivered even after a Stop or a close began: by then the worker has
        # already hung up the previous board and dialled the new one, and only
        # this delivery lets go of the one and takes the other.  Skipped, the
        # editor went on naming -- and holding its lease on -- a board it had
        # closed, and every later Stop, Connect and close failed on it.
        return self._run_device_command(
            work, delivered, summary="Connecting...", always_deliver=True
        )

    @staticmethod
    def _hang_up(sequencer: object, holding_lease: bool) -> BaseException | None:
        """End one connection on the thread that owns the device conversation.

        A board this editor was driving is told to go safe first, and the
        connection is closed whatever that told us -- refusing to close
        would hold the window hostage to a board it has no channel to.

        What it does not do is pass the failure off as an orderly hang-up.
        The server's AUTO-SAFE runs in its handler's finally, which needs
        that server still alive and still able to reach the board, and a
        ConnectionError is raised exactly when that is in doubt.  So the
        failure is ANSWERED rather than swallowed, and the caller says so
        on the owner thread where there is somewhere to say it.
        """

        unsafe: BaseException | None = None
        if holding_lease:
            safe = getattr(sequencer, "safe", None)
            if callable(safe):
                try:
                    safe()
                except ConnectionError as error:
                    unsafe = error
        close = getattr(sequencer, "close", None)
        if callable(close):
            close()
        return unsafe

    def _forget_connection(self) -> None:
        """Retire everything the departed board asserted about itself."""

        self.sequencer = None
        self._owns_sequencer = False
        self.board = None
        self.pins = {}
        self._board_target = None
        self._board_step_ns = None
        self._board_state = BoardState()
        self.revision += 1

    def _show_disconnected(self, mode: str, endpoint: str) -> None:
        self.connection = (mode, endpoint)
        self._show_connection("edit only")
        self._done("disconnected - this editor is now edit only")
        # Hanging up changes the whole window, not just its status line:
        # the board's ports are gone, the Target page is authorable again,
        # and the run buttons have nothing to fire on.
        self.refresh()

    def _dial_failed(self, mode: str, endpoint: str, error: BaseException) -> None:
        self.connection = (CONNECTION_OFFLINE, endpoint)
        self._show_connection(f"failed: {error}")
        self._warn(f"cannot connect to {_connection_name(mode, endpoint)}: {error}")

    def _release_for_connection_change(self) -> bool:
        """Keep the current connection truthful when it refuses to close."""

        try:
            self._release()
        except Exception as error:
            self._show_connection(f"disconnect failed: {error}")
            self._warn(f"cannot close the current sequencer connection: {error}")
            return False
        return True

    def adopt_board(self) -> bool:
        """Take the connected board's ports, pins and clock as the truth.

        A client must never supply a hardware fact.  The editor was doing
        exactly that -- reading the local XDC and streamer config -- so a
        window attached to a real board showed whatever this machine's files
        happened to say, and with no pulse open it showed nothing at all.

        A board can now describe itself, so the moment one is attached its
        description replaces every hardware fact here: the port catalog, the
        package pins, and the clock the durations are quantised to.  What is
        kept is the operator's work -- period names, durations and the state of
        every lane the board still has.
        """

        sequencer = self.sequencer
        describe = getattr(sequencer, "describe", None)
        if not callable(describe):
            self._show_connection("connected (board did not describe itself)")
            return False
        try:
            board = describe()
            state = self._board_state_for(sequencer)
        except Exception as error:
            self._describe_failed(error)
            return False
        return self._adopt_described(board, state)

    def _describe_failed(self, error: BaseException) -> None:
        self._show_connection(f"connected, but cannot read the board: {error}")
        self._warn(f"connected, but the board would not describe itself: {error}")

    def _adopt_described(self, board: object, state: BoardState) -> bool:
        """Install one board description and the state it was read with."""

        self.board = board
        self.pins = dict(getattr(board.target, "package_pins", {}) or {})
        self.revision += 1
        if self.sequence is None:
            self._apply_board_only(board)
        else:
            self._align_sequence_to(board)
        self._adopt_board_state(state)
        where = (
            f"{_connection_name(*self.connection)} - {len(board.target.ports)} ports, "
            f"{len(board.target.raw_lanes)} lanes, "
            f"{format_quantity(board.clock_hz / 1e6, '1')} MHz"
        )
        self._show_connection(where)
        self.refresh()
        return True

    def _apply_board_only(self, board) -> None:
        """With no pulse open, the board itself is what there is to show.

        Its catalog is not a pulse, so this does not invent one; it holds the
        target so the port rows, pins and clock are visible and the first
        period lands on the right board.
        """

        self._board_target = board.target
        self._board_step_ns = float(board.time_step_ns)

    def _align_sequence_to(self, board) -> None:
        """Move the open pulse onto this board, keeping what still applies.

        Matched BY NAME, not by position: two boards that expose the same lane
        in different slots must not silently swap what a period drives.  A lane
        the board does not have cannot be driven, and is reported rather than
        dropped quietly.

        A name is not enough on its own.  Kept only where the board has the
        same KIND of thing under it: a level on a lane the board drives as a
        digital output, a step, delay or binding on a port of the kind it was
        written for.  Matching names alone handed the model a level on a DAC
        data lane, or a DAC step on a digital port, and its refusal escaped
        the Connect delivery -- the operator was told nothing and the status
        stayed "Connecting...".  What still does not fit (a DAC value outside
        this board's range, a duration off its clock) is said, and the pulse
        keeps the wiring it was written for.

        A pulse already on this board's wiring and clock is left as it is,
        and where one is moved the names its author gave the outputs go with
        the outputs that keep their kind.  Taking the board's target whole
        put the XDC names back over every renamed output, and made a file
        just opened read as edited -- a Save then wrote the loss.
        """

        current = self.sequence
        if (
            current.target.abi_fingerprint == board.target.abi_fingerprint
            and current.time_step_ns == float(board.time_step_ns)
        ):
            return
        board_lanes = board.target.raw_lanes
        digital_lanes = {
            port.lanes[0] for port in board.target.ports if port.kind == "digital"
        }
        was = {lane: index for index, lane in enumerate(current.target.raw_lanes)}
        lost = sorted(
            lane
            for index, lane in enumerate(current.target.raw_lanes)
            if lane not in digital_lanes
            and any(period.states[index] for period in current.periods)
        )
        board_kinds = {port.key: port.kind for port in board.target.ports}
        written = current.target.by_key

        def fits(key: str) -> bool:
            return board_kinds.get(key) == written[key].kind

        target = self._retarget_labels(
            board.target,
            {key: port.label for key, port in written.items() if fits(key)},
        ) or board.target
        dropped_ports = sorted(
            {
                step.port
                for period in current.periods
                for step in period.analog_steps
                if not fits(step.port)
            }
            | {delay.port for delay in current.delays if not fits(delay.port)}
        )

        # replace(), so every other field of a period -- its kind above all:
        # a spacer must stay a spacer -- rides along without being listed.
        periods = tuple(
            replace(
                period,
                states=tuple(
                    period.states[was[lane]]
                    if lane in was and lane in digital_lanes else 0
                    for lane in board_lanes
                ),
                analog_steps=tuple(
                    step for step in period.analog_steps if fits(step.port)
                ),
            )
            for period in current.periods
        )
        delays = tuple(delay for delay in current.delays if fits(delay.port))
        bindings = tuple(
            binding for binding in current.bindings
            if binding.field_ref.port is None or fits(binding.field_ref.port)
        )
        dropped_bindings = sorted(
            field_label(current, b.field_ref) for b in current.bindings if b not in bindings
        )
        try:
            candidate = PulseSequence(
                name=current.name,
                target=target,
                time_step_ns=float(board.time_step_ns),
                periods=periods,
                bindings=bindings,
                delays=delays,
                brackets=current.brackets,
                components=current.components,
                run_repeats=current.run_repeats,
            )
        except ValueError as error:
            self._warn(
                f"this pulse does not fit the attached board ({error}); it keeps "
                "the wiring it was written for, which this board will not load: "
                "edit it to fit, save it and open it again"
            )
            return
        state_changes: dict[str, Any] = {"sequence": candidate}
        if len(candidate.scan_bindings) != len(current.scan_bindings):
            state_changes.update(
                scan_rows=(),
                scan_source_dirty=bool(self._state.scan_source),
            )
        self._edit_state(**state_changes)
        if lost or dropped_ports or dropped_bindings:
            missing = ", ".join(lost + dropped_ports + dropped_bindings)
            self._warn(
                f"this board has no matching {missing}; those outputs or bindings were dropped from "
                "the pulse rather than driven blind"
            )

    def _release(self, *, present: bool = True) -> None:
        """Close a connection this editor opened, and retire what it asserted.

        Everything in ``board``/``pins``/``_board_state`` is something the
        sequencer said about ITSELF -- a fact of the connection, not a property
        of this editor.  Keeping it after hanging up left the editor believing
        it was still attached forever after its first connection: the Target
        page went on reporting "wiring read from the attached board" and stayed
        read-only, so Offline -- the one mode whose entire point is authoring a
        target -- could never author one again.

        An injected sequencer is left alone, and so is its description: this
        editor did not open that connection and does not get to end it.
        """

        self._retire_drive(present=present)
        if not self._owns_sequencer:
            return
        if self.sequencer is not None:
            close = getattr(self.sequencer, "close", None)
            if callable(close):
                close()
        self._forget_connection()

    def _retire_drive(self, *, present: bool = True) -> None:
        """Safe and release only a command lease held by this editor."""

        if self._drive_lease is None:
            return
        if (
            not self._safe_drive(release=True, present=present)
            or self._drive_lease is not None
        ):
            raise RuntimeError("PulseGUI could not release its sequencer command")

    def _show_connection(self, status: str) -> None:
        """Say what the editor is attached to, and what that makes possible.

        Running needs both a pulse to fire and a board to fire it on; either
        one missing is a button that cannot work, so it is shown as one.
        """

        mode, endpoint = self.connection
        self._connection_status = str(status)
        self.view.set_connection(
            ConnectionVM(
                choices=self._connection_choices,
                selected=mode,
                endpoint=endpoint,
                status=self._connection_status,
                locked=self._connection_locked,
            )
        )
        self._render_run_state()

    def sync_from_sequencer(self) -> bool:
        """Bring what the BOARD is holding back into the editor.

        That is what Sync means, and the direction it has to run: a notebook or
        a raw API call changes the device behind this window's back, and
        nothing else lets the window catch up.  It used to push the other way
        -- editor onto board -- which is what On Pulse does anyway, so the
        button both duplicated one action and left the one nobody else
        performs undone.

        The board keeps the sequence it was handed, so this is a read: what
        comes back is the pulse the hardware will actually play, not this
        window's idea of it.  Read and adopted before returning; the window's
        Sync button reads on the device worker through ``_sync_from_view``.
        """

        question = self._applied_question()
        if question is None:
            return False
        return self._adopt_applied(*question())

    def _sync_from_view(self) -> None:
        if self._run_device_work is None:
            self.sync_from_sequencer()
            return
        question = self._applied_question()
        if question is None or not self._device_available():
            return

        def delivered(result: object, failure: BaseException | None) -> None:
            if failure is not None:
                self._warn(f"cannot sync from the board: {failure}")
                self._render_run_state()
                return
            self._adopt_applied(*result)

        self._run_device_command(
            lambda _operation: question(), delivered, summary="Syncing..."
        )

    def _applied_question(self) -> Callable[[], tuple[object, BoardState]] | None:
        """What the board holds and what it is doing, read together on one thread."""

        sequencer = self.sequencer
        if sequencer is None:
            self._warn("this editor is not connected to a sequencer")
            return None
        applied = getattr(sequencer, "applied", None)

        def question() -> tuple[object, BoardState]:
            state = applied() if callable(applied) else applied
            return state, self._board_state_for(sequencer)

        return question

    def _adopt_applied(self, state: object, board_state: BoardState) -> bool:
        """Make the editor show the program the board reported holding."""

        if state is None:
            self._adopt_board_state(board_state)
            self._warn("the board has no pulse applied yet; there is nothing to sync")
            return False
        source = getattr(state, "source", None)
        if source is None:
            self._adopt_board_state(board_state)
            self._warn(
                "the board is holding a compiled program it was given without "
                "its pulse, so there is nothing an editor can show"
            )
            return False
        wire_rows = tuple(getattr(state, "rows", ()) or ())
        program = getattr(state, "program", None)
        if wire_rows and program is None:
            self._adopt_board_state(board_state)
            self._warn("the board returned scan rows without their compiled program")
            return False
        authored_rows: tuple[tuple[float, ...], ...] = ()
        if wire_rows:
            # Back into the units this window writes tables in.  The board
            # holds offset-binary codes and device ticks; showing those in the
            # table view is showing the operator a different number system for
            # the same fields.
            from zlc_pulse import scan_columns_for, scan_rows_from_wire

            authored_rows = tuple(
                tuple(value for value in row)
                for row in scan_rows_from_wire(
                    wire_rows,
                    scan_columns_for(source, params=self._compiler_target()[0]),
                )
            )
        if wire_rows:
            self._remember_applied_scan(program, source, wire_rows, digest=program.digest)
        else:
            self._applied_scan = None
        # The document as it was handed to the board, before the board's
        # Config set filled it: adopting the filled one baked each Config
        # value into the draft as its authored default.
        adopted = getattr(state, "authored_source", None) or source
        held = None
        if self.sequence is not None:
            execution = self._execution_sequence()
            if adopted == execution:
                # The board is playing this very draft, resolved for execution
                # (API values and an unarmed scan point written in).  Adopting
                # that would erase the marks the draft is authored with.
                adopted = self.sequence
            elif self._held_point is not None and self._scan_armed():
                # The board may hold one row of this draft's scan, as Hold and
                # Step resolved it.  Adopting that would bake the row into the
                # fields as their authored values and empty the table: the
                # draft, its scan binding and its table stay as they are.
                try:
                    rows = self._prepared_scan(execution)[0]
                    point = resolve_scan_point(execution, rows[self._held_point])
                except Exception:  # noqa: BLE001 -- a table that no longer prepares holds nothing
                    point = None
                if adopted == point:
                    held = self._held_point
        # Replacing the draft with another pulse asks first, as Load and
        # Clear All do -- only then: a board playing this draft keeps it.
        if held is None and adopted is not self.sequence and not self._discard_pulse_edits():
            self._adopt_board_state(board_state)
            return False
        if held is None:
            # EVERY execution fact the sync point later compares is adopted
            # here, because this is the one door the board's answer comes
            # through.  The scan repeat count was not: it stayed whatever the
            # editor last held, so a board playing two sweeps against an
            # editor holding one reported "not synchronized" -- and pressing
            # Sync could not fix it, because the comparison read a field the
            # adoption never wrote.
            self._edit_state(
                sequence=adopted,
                scan_rows=authored_rows,
                scan_source_dirty=bool(self._state.scan_source),
                scan_repeats=int(getattr(state, "scan_repeats", 0) or 0),
            )
        self.refresh()
        # What came back IS what the board holds, so the dot must stop saying
        # the board is playing something older the moment it no longer is.
        self._adopt_board_state(board_state)
        self._done(
            f"the board is holding scan point {held + 1} of this pulse; "
            "the draft and its scan table are kept"
            if held is not None
            else f"synced from the board - {len(source.periods)} period(s)"
            + (f", {len(wire_rows)} scan point(s)" if wire_rows else "")
        )
        return True

    def _prepare_execution(self) -> tuple[PulseSequence, Any, tuple[tuple[int, ...], ...], int]:
        """Compile every local fact before attempting to acquire the device."""

        source, rows, sweeps = self._execution_request()
        source, program = self.compile(source)
        return source, program, rows, sweeps

    def _execution_request(
        self,
    ) -> tuple[PulseSequence, tuple[tuple[int, ...], ...], int]:
        source = self._execution_sequence()
        scan_armed = self._scan_armed()
        rows = self._prepared_scan(source)[1] if scan_armed else ()
        return source, rows, self._state.scan_repeats if scan_armed else 1

    def _prepared_scan(
        self,
        source: PulseSequence,
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[int, ...], ...]]:
        """One quantized table shared by Run, Hold, digest and readback."""

        from zlc_pulse import prepare_scan_application

        geometry, _clock_hz = self._compiler_target()
        return prepare_scan_application(
            source,
            self._state.scan_rows,
            params=geometry,
        )

    def _load_prepared(
        self,
        prepared: tuple[PulseSequence, Any, tuple[tuple[int, ...], ...], int],
    ) -> None:
        source, program, rows, _sweeps = prepared
        self.sequencer.load(program, source=source, rows=rows)

    def _remember_applied_scan(
        self,
        program: object,
        source: PulseSequence,
        wire_rows: Sequence[Sequence[int]],
        *,
        digest: str,
    ) -> None:
        """Freeze the display text of the table actually handed to the sequencer:
        each axis in its own decimals, padded with figure spaces to its widest."""

        if not wire_rows:
            self._applied_scan = None
            return
        if not source.scan_bindings:
            self._applied_scan = None
            return
        from zlc_pulse import scan_columns_for, scan_rows_from_wire

        columns = scan_columns_for(source, params=self._compiler_target()[0])
        # Read back from the wire, 0.3 arrives as 0.30000000000000004: each
        # axis is written to twelve digits, every row in that axis's own
        # decimals (or mantissa digits) with a digit-wide minus (U+2012), and
        # padded with figure spaces to its widest, so the progress line's
        # words stay put from point to point.
        texts = []
        for values in zip(*scan_rows_from_wire(wire_rows, columns)):
            shown = [f"{float(value):.12g}" for value in values]
            if any("e" in text for text in shown):
                digits = max(
                    sum(character.isdigit() for character in text.partition("e")[0]) for text in shown
                )
                shown = [f"{float(value):.{digits - 1}e}" for value in values]
            else:
                decimals = max(len(text.partition(".")[2]) for text in shown)
                shown = [f"{float(value):.{decimals}f}" for value in values]
            shown = [text.replace("-", "\u2012") for text in shown]
            widest = max(shown, key=len)
            texts.append([figure_padded(text, widest) for text in shown])
        self._applied_scan = (
            str(digest),
            tuple(column.name for column in columns),
            tuple(zip(*texts)),
        )

    def _acquire_command(self, *, present: bool = True) -> bool:
        if self.sequencer is None:
            self._warn("this editor is not connected to a sequencer")
            return False
        if self._drive_lease is not None:
            return True
        try:
            self._drive_lease = self.device_use.acquire_command(
                self._device_owner,
                "PulseGUI",
                (DeviceClaim("sequencer", "sequencer", self.sequencer),),
            )
        except DeviceUseBusy as error:
            if not present:
                raise RuntimeError(str(error)) from error
            self._warn(str(error))
            return False
        return True

    def _release_drive(self) -> None:
        lease, self._drive_lease = self._drive_lease, None
        self._finite_run = None
        if lease is not None:
            lease.release()

    def _execution_sequence(self) -> PulseSequence:
        """One executable document prepared from the current authoring state.

        API parameters are always replaced by the values currently shown in
        the editor.  Scan bindings remain dynamic only when a real table is
        armed; without a table their authored nominal values become an ordinary
        static pulse.  Both On Pulse and synchronization digest this same
        document, so there is only one definition of what the button runs.
        """

        if self.sequence is None:
            raise RuntimeError("no pulse is open")
        source = resolve_api_parameters(self.sequence)
        if source.scan_bindings and not self._scan_armed():
            source = resolve_scan_point(source)
        return source

    def _fire_from_view(self) -> None:
        if not self._check_bracket():
            return
        if self._run_device_work is None:
            self.fire()
            return
        if not self._device_available():
            return
        if self.sequence is None:
            self._warn("no pulse is open")
            return
        sequencer = self.sequencer
        if sequencer is None:
            self._warn("this editor is not connected to a sequencer")
            return
        authored_revision = self.revision
        try:
            source, rows, sweeps = self._execution_request()
        except Exception as error:
            self._warn(f"cannot load this pulse: {error}")
            return
        if not self._acquire_command():
            return
        finite = bool(source.run_repeats and sweeps)
        previous_run = self._finite_run
        self._finite_run = None

        def work(operation: int) -> object:
            program = None
            loaded = source
            error: BaseException | None = None
            try:
                if operation == self._device_operation:
                    loaded, program = self.compile(source)
                if program is not None:
                    error = self._drive_program(
                        sequencer,
                        program,
                        loaded,
                        rows=rows,
                        run_repeats=source.run_repeats,
                        scan_repeats=sweeps,
                        current=lambda: operation == self._device_operation,
                    )
                    if error is None and operation == self._device_operation:
                        self._watch_completion()
            except BaseException as caught:  # noqa: BLE001 -- delivered, not lost
                error = caught
            return program, self._board_state_for(sequencer), error

        def delivered(result: object, failure: BaseException | None) -> None:
            program, state, error = (
                result if result is not None else (None, self._board_state, failure)
            )
            self._board_state = state
            if error is None:
                # The board plays the pulse (or its scan from row 0) again;
                # Step no longer starts from a row it was holding.
                self._held_point = None
                self._remember_applied_scan(program, source, rows, digest=state.applied_digest)
                self._digest_revision = -1
                if self.revision == authored_revision:
                    self._digest = state.applied_digest
                    self._digest_revision = authored_revision
                self.refresh_preview()
                self.view.set_summary("Started")
            else:
                if not state.firing:
                    self._release_drive()
                elif previous_run is not None:
                    self._finite_run = previous_run
                    self._watch_completion()
                message = f"firing stopped: {error}"
                self.view.set_summary(message)
                self._warn(message)
            self._render_run_state()

        self._run_device_command(work, delivered, summary="Starting...", finite_run=finite)

    def _device_available(self) -> bool:
        if self._preview_close_requested:
            # Closing retires the drive on the SAFE worker; a command taken
            # now would hold its lease past the window that took it.
            return False
        if self._device_busy or self._stop_busy:
            self._warn("a pulse command is already in progress")
            return False
        return True

    def _run_device_command(
        self,
        work: Callable[[int], object],
        delivered: Callable[[object, BaseException | None], None],
        *,
        summary: str,
        finite_run: bool | None = None,
        always_deliver: bool = False,
    ) -> bool:
        """Change the board on the device worker; show the outcome here.

        Every command -- On Pulse, Hold, Step, Sync, Connect -- has the same
        shape: this thread marks the editor busy so the buttons say so, hands
        the device work to the worker with the operation number it was
        started as, and applies what came back only if no later command
        superseded it.  ``work`` takes that operation number so it can stop
        touching the device once it is stale; ``delivered`` takes the result
        and the error, exactly one of them None.  ``_device_done`` is set the
        moment the device work ends, whichever way, for Stop to wait on.
        The ``summary`` it puts up ("Syncing...") comes down when the outcome
        is delivered, back to the editor's resting sentence; a delivery with
        more to say says it after.

        ``always_deliver`` is for work whose delivery is bookkeeping rather
        than a report on the board -- a connection changed, a Config file
        written.  That work has happened whatever superseded it, so its
        delivery runs regardless.
        """

        runner = self._run_device_work
        assert runner is not None
        self._device_operation += 1
        operation = self._device_operation
        if finite_run is not None:
            self._finite_run = operation if finite_run else None
        done = Event()
        self._device_done = done
        self._device_busy = True
        self._refresh_config_page()
        self.view.set_summary(summary)
        self._render_run_state()

        def task() -> object:
            try:
                return work(operation)
            finally:
                done.set()

        def finished(result: object, error: BaseException | None) -> None:
            self._device_done = None
            self._device_busy = False
            try:
                if operation == self._device_operation:
                    # Only a delivery that refreshes the document writes the
                    # line, so a declined Sync, a Hold or a failed Connect
                    # would leave the busy sentence up.  A superseded command
                    # leaves the line to the Stop or command that superseded it.
                    self.view.set_summary(self._document_summary())
                if always_deliver or operation == self._device_operation:
                    delivered(result, error)
            finally:
                # A delivery that raised (its error is reported at the Qt
                # boundary) must still leave the Config page usable, the
                # status line current and a waiting close woken.
                self._refresh_config_page()
                self._run_status_followups()
                self._wake_close_guard()

        return self._submit_work(
            runner,
            task,
            lambda result: finished(result, None),
            lambda error: finished(None, error),
        )

    def _drive_program(
        self,
        sequencer: object,
        program: object,
        source: PulseSequence,
        *,
        rows: Sequence[Sequence[int]] = (),
        run_repeats: int,
        scan_repeats: int = 1,
        current: Callable[[], bool],
        halt_first: bool = False,
    ) -> BaseException | None:
        """Stop what is playing, put ``program`` on the board, play it.

        The three device steps every play gesture is made of -- On Pulse,
        Hold, Step -- in the one order that works, with the one guarantee they
        share: a board that was touched and could not be handed the whole
        program is driven safe, not left with half of it.  On Pulse stops the
        board only when it is playing; Hold and Step (``halt_first``) stop it
        regardless, because freezing the scan IS the gesture.  ``current``
        says whether the gesture that asked is still the latest; a superseded
        one stops touching the device and goes safe.  Runs on whichever
        thread owns the device conversation; returns the error instead of
        raising so the board's state can be read afterwards on the same
        thread.
        """

        touched = False
        try:
            if current() and (halt_first or bool(sequencer.snapshot().get("firing"))):
                touched = True
                sequencer.safe()
            if current():
                touched = True
                sequencer.load(program, source=source, rows=rows)
            if current():
                sequencer.fire(
                    run_repeats=run_repeats,
                    scan_repeats=scan_repeats,
                )
            if not current() and touched:
                sequencer.safe()
        except BaseException as caught:  # noqa: BLE001 -- returned, not lost
            if touched:
                try:
                    sequencer.safe()
                except BaseException as safe_error:  # noqa: BLE001
                    return RuntimeError(f"{caught}; SAFE also failed: {safe_error}")
            return caught
        return None

    @staticmethod
    def _submit_work(runner, work, delivered, failed) -> bool:
        try:
            runner(work, delivered, failed)
        except BaseException as error:
            failed(error)
            return False
        return True

    def fire(self) -> bool:
        """On Pulse: load what is on screen and run it the way the pulse says.

        ``run_repeats`` belongs to the Pulse and repeats the complete timeline
        at one scan point.  ``scan_repeats`` belongs to the armed table and
        repeats complete table sweeps.  A bracket remains an internal timeline
        loop and never chooses either count.

        NOTHING here waits for the board.  A run is started and the display
        beat says what the board is doing, which is how the forever path always
        worked -- the finite path used to block the GUI thread on wait_done()
        instead, and the one control that would have helped, Stop, is the one a
        blocked event loop cannot deliver.

        There is nothing left for that wait to protect.  Firing over an
        unfinished shot cannot happen: ``load`` and ``fire`` both call
        ``_require_idle()`` and
        raise, and On Pulse stops a running board before loading anyway.  The
        invariant lives with the device that owns it.
        """

        if not self._check_bracket():
            return False
        if self.sequence is None:
            self._warn("no pulse is open")
            return False
        if self.sequencer is None:
            self._warn("this editor is not connected to a sequencer")
            return False
        try:
            prepared = self._prepare_execution()
        except Exception as error:
            self._warn(f"cannot load this pulse: {error}")
            return False
        if not self._acquire_command():
            return False
        self._finite_run = None
        self._device_operation += 1
        if not self._board_ready_for_a_program():
            return False
        try:
            self._load_prepared(prepared)
            source, program, rows, sweeps = prepared
            self.sequencer.fire(
                run_repeats=source.run_repeats,
                scan_repeats=sweeps,
            )
            self._finite_run = self._device_operation if source.run_repeats and sweeps else None
            self._watch_completion()
            self._digest_revision = -1
            self._poll_board()
            self._held_point = None
            self._remember_applied_scan(program, source, rows, digest=self._board_state.applied_digest)
            self.refresh_preview()
        except Exception as error:
            self._warn(f"firing stopped: {error}")
            self._safe_drive(release=True)
            return False
        return True

    def _board_ready_for_a_program(self) -> bool:
        """Bring the board to a state that can accept one, which may mean off.

        On Pulse is not "start from idle": it is "play THIS, now", and at the
        bench it is pressed most often while something is already playing --
        that is how an edit reaches a running experiment.  Off-then-on is what
        the gesture means, so the off belongs here.

        The device is right to refuse the alternative.  Loading a program into
        a firing streamer would rewrite the tables under the engine, so
        ``load`` and ``fire`` both demand
        an idle board; a device that silently stopped a running experiment
        because someone called ``load`` would be a far worse thing to own.
        What was missing is that nobody was doing the stopping.

        Asked first, because the answer may be several minutes old -- and only
        stopped if the answer is yes, so an On Pulse on an idle board does not
        drive the outputs safe on the way to driving them.
        """

        if self.sequencer is None:
            return True
        self._poll_board()
        if not self.running:
            return True
        self._safe_drive(release=False)
        self._poll_board()
        if self.running:
            self._warn("the board is still playing and would not stop; not loading over it")
            return False
        return True

    def _safe_drive(self, *, release: bool, present: bool = True) -> bool:
        if self.sequencer is None:
            return True
        if not self._acquire_command(present=present):
            if present:
                self._poll_board()
            else:
                self._board_state = self.board_state()
            return False
        worked = True
        try:
            safe = getattr(self.sequencer, "safe", None)
            if callable(safe):
                safe()
        except ConnectionError as error:
            # There is nothing left HERE to make safe, and refusing to close
            # would hold the window hostage to a board it has no channel to
            # -- so this does not block.  What it must not do is CLAIM the
            # board is safe: the server's AUTO-SAFE runs in its handler's
            # finally, which needs that server still alive and still able to
            # reach the board, and a ConnectionError is raised exactly when
            # that is in doubt.  If the server went with the connection, the
            # sequence is still playing.
            self._release_drive()
            self._warn(
                "the pulse server connection ended before the board could be "
                f"told to stop ({error}). If that server is still running it "
                "drives the board safe when a client disconnects; if it is "
                "not, THE SEQUENCE IS STILL PLAYING and the board must be "
                "stopped another way."
            )
            return True
        except Exception as error:
            if not present:
                raise RuntimeError(f"the board did not go safe: {error}") from error
            worked = False
            self._warn(f"the board did not go safe: {error}")
        finally:
            if present:
                self._poll_board()
            else:
                self._board_state = self.board_state()
        if worked and not self.running:
            self._finite_run = None
            if release:
                self._release_drive()
            return True
        return False

    def stop(self) -> None:
        """Return the outputs to their safe state, whatever state they are in.

        The only way out of a forever run, and it must work from any state --
        an operator pressing Stop is not required to know what the board is
        doing first.
        """

        # A close retires the drive itself, SAFE first; a Stop taken once that
        # has begun would take a lease the closed window then kept.  Until
        # then a close waits on a command or status answer, and Stop is what
        # cuts that short (SAFE goes on the cancel lane); it runs on the SAFE
        # worker ahead of the retire, which releases any lease it left.
        if self._stop_busy or self._retiring():
            return
        if self._run_safe_work is None:
            self._device_operation += 1
            self._finite_run = None
            self._safe_drive(release=True)
            return
        sequencer = self.sequencer
        if sequencer is None or not self._acquire_command():
            return
        # Only a Stop that goes ahead supersedes the command in progress; one
        # refused its lease must not cancel anything.
        self._device_operation += 1
        self._finite_run = None

        command_done = self._device_done
        self._stop_busy = True
        self.view.set_summary("Stopping...")
        self._render_run_state()

        def work() -> object:
            def go_safe() -> BaseException | None:
                try:
                    sequencer.safe()
                    return None
                except BaseException as caught:
                    return caught

            error = go_safe()
            if command_done is not None:
                command_done.wait(5.0)
                error = go_safe()
            return self._board_state_for(sequencer), error

        def delivered(result: object) -> None:
            if self._device_busy:
                # The command this Stop superseded has not delivered: a
                # Connect may yet change which board this editor holds, and
                # says what became of the old one.  Heard after it.
                self._status_followups.append(lambda: delivered(result))
                return
            state, error = result
            self._stop_busy = False
            if sequencer is not self.sequencer:
                # A Connect delivered while this Stop ran: the board it
                # stopped is no longer this editor's, and the Connect has
                # said what became of it.  Its state is not this board's.
                self._render_run_state()
                self._wake_close_guard()
                return
            self._board_state = state
            worked = error is None and state.answering and not state.firing
            if worked:
                self._release_drive()
            self._render_run_state()
            if worked:
                self.view.set_summary("Stopped")
            else:
                message = f"Stop failed: {error}" if error else "the board did not confirm SAFE"
                self.view.set_summary(message)
                self._warn(message)
            self._wake_close_guard()

        def failed(error: BaseException) -> None:
            delivered((self._board_state, error))

        self._submit_work(self._run_safe_work, work, delivered, failed)

    @property
    def running(self) -> bool:
        """Is the board playing?  Its answer, from the last time it was asked."""

        return self._board_state.firing

    @property
    def synchronized(self) -> bool:
        """Is the board holding what this editor shows?

        A comparison of two facts, one from each side: the digest the board
        last reported, and the digest of what is on screen right now.  Local,
        so an edit re-answers it without anyone being asked anything.
        """

        # A board holding nothing (or none attached, the usual authoring
        # case) cannot match, and the digest behind the other side costs a
        # compile -- and offline a read of the streamer config.
        if not self._board_state.applied_digest:
            return False
        shown = self._shown_digest()
        if not shown or self._board_state.applied_digest != shown:
            return False
        if not self._board_state.firing:
            return True
        try:
            source = self._execution_sequence()
        except Exception:
            return False
        # The set compared here must equal the set ``_adopt_applied`` takes
        # in.  A field on one side and not the other is a disagreement that
        # pressing Sync cannot settle, which is worse than not noticing it.
        scan_repeats = self._state.scan_repeats if self._scan_armed() else 1
        return (
            self._board_state.run_repeats == source.run_repeats
            and self._board_state.scan_repeats == scan_repeats
        )

    def board_state(self) -> BoardState:
        """Ask the board what it is doing, in one round trip.

        One call, not two: everything here comes out of ``snapshot``, which the
        board keeps as cheap level state precisely so it can be asked often.
        Pulling the applied program back instead would ship a whole compiled
        pulse across the wire to answer a yes-or-no question.
        """

        if self.sequencer is None:
            return BoardState()
        return self._board_state_for(self.sequencer)

    @staticmethod
    def _board_state_for(sequencer: object) -> BoardState:
        try:
            reported = dict(sequencer.snapshot())
        except Exception as error:
            return BoardState(attached=True, answering=False, fault=str(error))
        cursor = reported.get("cursor")
        return BoardState(
            attached=True,
            answering=True,
            firing=bool(reported.get("firing")),
            loaded=bool(reported.get("loaded")),
            cursor=None if cursor is None else int(cursor),
            applied_digest=str(reported.get("applied_digest") or ""),
            run_repeats=int(reported.get("run_repeats", 1)),
            scan_repeats=int(reported.get("scan_repeats", 1)),
        )

    def _shown_digest(self) -> str:
        """What the screen would compile to, digested once per edit.

        Compiling is cheap and doing it on every status refresh is still waste:
        the answer cannot change while the revision does not.
        """

        if self.sequence is None:
            return ""
        if self._digest_revision != self.revision:
            try:
                # The program alone: the board's digest names its program,
                # so quantizing the scan table here was thrown away.
                source = self._execution_sequence()
                self._digest = self.compile(self._effective_sequence(source))[1].digest
            except Exception:
                # A pulse that does not compile is not one any board can be
                # holding, which is the honest answer to the question asked.
                self._digest = ""
            self._digest_revision = self.revision
        return self._digest

    def show_page(self, page: str) -> None:
        """The operator turned to a page.  Anything deferred for it happens now.

        Preview drawing and Scan cursor polling are both visible-page work.
        Neither is useful while its page is hidden.
        """

        self._preview_on_screen = str(page) == PREVIEW_PAGE
        if self._preview_on_screen:
            self.refresh_preview()
        if self.sequence is not None:
            self._refresh_scan_page()

    def refresh_run_state(self) -> None:
        """Ask the board what it is doing and show the answer before returning.

        The public name for it, because "what is the board doing" is a question
        anything may need re-asked -- a notebook, a test standing in for the
        operator who walked back to the bench, the app attaching a board at
        start-up.  It blocks its caller for the board's round trip, which is
        right for a line in a notebook and wrong for a GUI thread: the window
        asks through ``ask_run_state`` instead.
        """

        sequencer = self.sequencer
        finite = self._finite_run is not None and self._drive_lease is not None
        self._adopt_board_answer(self._board_answer(sequencer, finite))

    def _watch_completion(self) -> None:
        """Await this finite run off both Qt and the device command worker."""

        runner = self._run_completion_work
        sequencer = self.sequencer
        if runner is None or self._finite_run is None or self._drive_lease is None:
            return
        run = self._finite_run

        def current() -> bool:
            return (
                run == self._finite_run and sequencer is self.sequencer
                and not self._preview_close_requested
            )

        def work() -> object:
            if not current():
                return None
            try:
                answer = ("done", sequencer.wait_done(None))
            except Exception as error:
                answer = ("failed", error)
            return (answer, self._board_state_for(sequencer)) if current() else None

        def delivered(answer: object) -> None:
            if answer is not None and current():
                if self._device_busy:
                    self._status_followups.append(lambda: delivered(answer))
                    return
                self._adopt_board_answer(answer)
            self._wake_close_guard()

        def failed(error: BaseException) -> None:
            if current():
                self._warn(f"finite pulse completion failed: {error}")
            self._wake_close_guard()

        self._submit_work(runner, work, delivered, failed)

    def ask_run_state(self, *, then: Callable[[], None] | None = None) -> bool:
        """Ask the board what it is doing; show the answer when it comes.

        The asking is device I/O: a socket round trip to the pulse server,
        which answers only once whatever it is doing on the UART lets it.  The
        window's timer used to make that round trip ON the GUI thread every
        100 ms, and the experiment machine's main thread spent a third of its
        time blocked in that socket -- up to a second at a stretch -- while
        clicks, tabs and paints waited behind it.  With a device worker the
        question goes there and the answer comes back through the owner turn;
        without one it is asked here and answered before this returns.
        ``then`` runs after the answer has been shown.  Returns whether an
        answer is coming.
        """

        sequencer = self.sequencer
        finite = self._finite_run is not None and self._drive_lease is not None
        return self._ask_board(
            lambda: self._board_answer(sequencer, finite),
            self._adopt_board_answer,
            then=then,
        )

    def _poll_board(self) -> None:
        """Ask the board now, on this thread, and show what it said.

        The synchronous form, for the paths that already own the device on
        this thread -- the notebook's fire and load, the safe drive -- where
        the answer is needed before the next line runs.  The window's timer
        and every button ask through ``ask_run_state`` instead, so that
        an answer the board is slow to give never holds the GUI.

        It used to be called from every redraw, and every edit redraws, so
        ticking a checkbox or typing a digit made an RPyC round trip to the
        pulse server.  A local edit cannot change what the board is doing; it
        can only change whether what the board is doing still matches what is
        on screen, and that comparison is local.
        """

        self._adopt_board_state(self.board_state())

    def _ask_board(
        self,
        question: Callable[[], object],
        answered: Callable[[object], None],
        *,
        then: Callable[[], None] | None = None,
    ) -> bool:
        """Put one question to the board without changing it; apply the answer here.

        THE path a status read takes.  ``question`` runs where the I/O
        belongs -- the device worker when this editor has one, this thread
        when it has not -- and ``answered`` runs on the owner with what came
        back.  One question in flight at a time: a second request while one
        is pending only adds ``then`` to the answer's follow-ups.  A command
        in progress makes no request at all -- its own delivery reports the
        board it left behind -- and an answer that arrives after a command
        started is dropped, because it describes the board from before.
        """

        if then is not None:
            self._status_followups.append(then)
        if self._preview_close_requested:
            # The board is being let go of on the SAFE worker; a question
            # sent now would race that, and the close waits for any answer.
            self._run_status_followups()
            return False
        runner = self._run_device_work
        if runner is None or self.sequencer is None:
            try:
                answer = question()
            except Exception as error:
                self._warn(f"board status failed: {error}")
                self._run_status_followups()
                return False
            answered(answer)
            self._run_status_followups()
            return True
        if self._status_in_flight:
            return True
        if self._device_busy or self._stop_busy:
            self._run_status_followups()
            return False
        operation = self._device_operation
        finite_run = self._finite_run
        sequencer = self.sequencer
        self._status_in_flight = True

        def finished(answer: object, error: BaseException | None) -> None:
            self._status_in_flight = False
            if error is not None:
                self._warn(f"board status failed: {error}")
            elif (
                operation == self._device_operation and sequencer is self.sequencer
                and finite_run == self._finite_run
            ):
                answered(answer)
            self._run_status_followups()
            # A close that found this answer outstanding waits for it: the
            # device worker cannot be closed while it is still asking.
            self._wake_close_guard()

        return self._submit_work(
            runner,
            question,
            lambda answer: finished(answer, None),
            lambda error: finished(None, error),
        )

    def _run_status_followups(self) -> None:
        followups, self._status_followups = self._status_followups, []
        for followup in followups:
            try:
                followup()
            except Exception as error:  # noqa: BLE001 -- one follow-up must not eat the rest
                self._warn(f"internal error after a board answer: {error}")

    def _board_answer(
        self, sequencer: object, finite: bool
    ) -> tuple[tuple[str, object] | None, BoardState]:
        """What the board says it is doing, read in one place on whichever thread asks.

        A finite drive is asked whether it has finished first: ``wait_done(0)``
        is a poll, and its report -- done, or a fault -- belongs to the same
        answer as the state, so both arrive together and are shown together.
        """

        if sequencer is None:
            return None, BoardState()
        finite_answer: tuple[str, object] | None = None
        if finite and self._run_completion_work is None:
            try:
                finite_answer = ("done", sequencer.wait_done(0))
            except Exception as error:  # noqa: BLE001 -- reported on the owner
                finite_answer = ("failed", error)
        return finite_answer, self._board_state_for(sequencer)

    def _adopt_board_answer(self, answer: object) -> None:
        finite_answer, state = answer
        faulted = False
        if finite_answer is not None:
            kind, payload = finite_answer
            if kind == "failed":
                self._warn(f"finite pulse status failed: {payload}")
            elif payload is not None:
                fault = str(getattr(payload, "fault", "") or "")
                if fault:
                    # A faulted end is NOT an end the board made safe. A
                    # clean DONE is: the engine drains, parks its outputs
                    # and stops, which is why that case needs no SAFE. An
                    # UNDERFLOW does not stop it at all -- the engine sets
                    # the sticky bit and stalls, still running -- so
                    # releasing the lease here and saying nothing left the
                    # sequence playing behind an idle-looking window.
                    self._warn(f"finite pulse stopped: {fault}")
                    faulted = True
                else:
                    self._release_drive()
            elif state.answering and not state.firing:
                self._release_drive()
        self._adopt_board_state(state)
        if faulted:
            # Through Stop, which is the one implementation of "tell the
            # board to stop": it puts the SAFE on the device worker, where
            # every other board conversation happens.  Sent from here it
            # would be a blocking socket round trip on the OWNER, the one
            # thread this module's whole status path exists to keep off the
            # wire -- and it ran after the state was adopted, so the window
            # would freeze showing the board still firing.
            self.stop()

    def _adopt_board_state(self, state: BoardState) -> None:
        """Show what the board said, wherever and whenever it was asked."""

        was_running = self._board_state.firing
        self._board_state = state
        self._render_run_state()
        if was_running != self._board_state.firing and self.sequence is not None:
            self._refresh_scan_page()

    def _render_run_state(self) -> None:
        """Tell the window what is running, from what the board last said.

        The status dot is the same answer at a glance: green while the board is
        playing what the editor shows, orange while it is playing something
        older, and grey when nothing is attached.  One place decides it, so the
        dot and the buttons cannot disagree.
        """

        # A closing window offers nothing but Stop, and Stop only until the
        # close is retiring the drive, SAFE first.
        attached = self.sequencer is not None and not self._preview_close_requested
        live = attached and self.sequence is not None
        idle = not self._device_busy and not self._stop_busy
        synchronized = False if self._device_busy else bool(self.synchronized)
        self.view.set_control_state(
            running=bool(self.running),
            synchronized=synchronized,
            file_dirty=self._state != self._saved_state,
            can_run=live and idle,
            # Going safe needs a board and nothing else.  Requiring a pulse to
            # be open, or the window to believe the board is busy, makes Stop
            # unavailable in exactly the situations it exists for.
            can_stop=(
                self.sequencer is not None and not self._stop_busy and not self._retiring()
            ),
        )
        # Capabilities go through the shell, which also gates the Scan page's
        # hold and step -- those need a board just as much as Sync does.
        self.view.set_capabilities(
            # Sync READS the board, so it needs a board and not a pulse -- an
            # editor with nothing open is exactly when pulling what the
            # hardware is holding is worth doing.
            can_sync=attached,
            can_hold=live and idle,
            can_step=live and bool(self._state.scan_rows) and idle,
        )
        self.view.set_status_color(self._status_token())

    @property
    def _holding_this_pulse(self) -> bool:
        """Is the board playing the row Hold or Step resolved from this draft?

        A held row is a program of its own -- the row's numbers written into
        the fields, no table, played until Stop -- so it never equals the scan
        On Pulse would load, and ``synchronized`` rightly says no (On Pulse
        keeps its asterisk: pressing it changes what plays).  It is still this
        pulse, which is what Sync says and keeps the draft for, so the dot
        must not call it something older.
        """

        state = self._board_state
        return (
            state.firing
            and self._held_program == (self.revision, state.applied_digest)
            and (state.run_repeats, state.scan_repeats) == (0, 1)
        )

    def _status_token(self) -> str:
        state = self._board_state
        if not state.attached:
            return "idle"
        if not state.answering:
            # Attached and silent is its own state.  Showing the last good
            # answer instead is how a window sits green over a dead server.
            return "unreachable"
        if state.firing:
            return (
                "running-synced"
                if not self._device_busy
                and (self.synchronized or self._holding_this_pulse)
                else "running-stale"
            )
        return "dirty-ready" if self.sequence is not None else "idle"

    def save_pulse(self) -> str:
        """Write what is on screen as a ``zlc.pulse`` JSON document."""

        if not self._check_bracket():
            return ""
        if self.sequence is None:
            self._warn("there is no pulse to save")
            return ""
        start = self.path or str(
            Path(self.pulses_directory or "") / f"{self.sequence.name or 'pulse'}.json"
        )
        chosen = self.view.ask_save_path(
            "Save pulse", str(Path(start).with_suffix(".json")),
            "ZLC pulse (*.json);;All files (*)",
        )
        if not chosen:
            return ""
        target = Path(chosen)
        if target.suffix == "":
            target = target.with_suffix(".json")
        if target.suffix.lower() != ".json":
            self._warn(
                f"{target.name} is not a JSON pulse; save with a .json suffix"
            )
            return ""
        # The file names the pulse.  A document saved as scan.json is "scan"
        # in every record that names it; the name on screen before the first
        # save only proposes the file name.
        if self.sequence.name != target.stem:
            named = self._rebuilt(name=target.stem)
            if named is None:
                return ""
            self._edit_state(sequence=named)
            self.view.set_title(f"PulseGUI - {named.name}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            write_pulse(target, self._state)
        except Exception as error:
            self._warn(f"cannot save {target.name}: {error}")
            return ""
        self.path = str(target)
        self._saved_state = self._state
        self._show_connection(self._connection_status)
        self._done(f"saved {target.name}")
        return str(target)

    # -------------------------------------------------------------- the target

    def refresh_target(self) -> None:
        """Show the wiring, and say who is allowed to change it.

        A board's topology is the board's.  Attached, the page reports it and
        only the display names can be edited -- renaming a signal is a label,
        not a re-wiring, and the ABI fingerprint never moves.  Offline the
        target comes from a pulse file, and authoring one IS the point, so the
        page opens up.  A pulse the board refused to take on keeps its file's
        wiring, and the page says that is what it shows.
        """

        target = self._current_target()
        view = self.view
        if target is None:
            view.set_target_ports((), False, "No pulse and no board: nothing is wired yet.")
            return
        attached = self.board is not None
        if attached:
            width = int(self.board.geometry.bus_width)
            view.set_target_width_rules(
                TargetWidthRule(1, 1, 1), TargetWidthRule(2, width, width)
            )
        records = project_target(target, pins=self.pins)
        # A clock no DAC latches with has no row, and its name is still taken:
        # the page's Add must not mint it, since nothing there renames a port.
        reserved = tuple(
            port.key for port in programmable_ports(target) if port.kind == "clock"
        )
        if not attached:
            caption = "Offline: this target came from the pulse file and can be edited."
        elif target.abi_fingerprint == self.board.target.abi_fingerprint:
            caption = (
                f"Wiring read from the attached board; rename freely, the "
                f"topology is the board's ({len(records)} outputs)."
            )
        else:
            caption = (
                "The pulse file's own wiring, which the attached board will not "
                "load: edit the pulse to fit, save it and open it again."
            )
        view.set_target_ports(records, not attached, caption, reserved=reserved)

    def _current_target(self) -> object | None:
        if self.sequence is not None:
            return self.sequence.target
        return self._board_target

    def apply_target(self, records: object) -> bool:
        """Take the edited target, and refuse anything that would re-wire a board.

        Attached, only display names may change: changing lanes, widths or
        the set of ports would make the editor disagree with the hardware it
        is pointed at, and the disagreement would only surface as a pulse that
        fires the wrong outputs.  Offline the target is the pulse file's and
        authoring it is the point, so the page's records ARE the target --
        ports, widths, wires, latch clocks -- and the pulse is carried onto it
        output by output.  What the new wiring has no place for is refused by
        name, never dropped: an output still driven, stepped, delayed or bound
        stays until the operator clears it.
        """

        target = self._current_target()
        if target is None:
            self.view.set_target_feedback("there is no target to apply to")
            return False
        records = tuple(records)
        if self.board is not None:
            wanted = {str(record.key): str(record.signal).strip() for record in records}
            keys = {
                port.key for port in programmable_ports(target) if port.kind != "clock"
            }
            if set(wanted) != keys:
                self.view.set_target_feedback(
                    "this target is the attached board's; ports cannot be added or "
                    "removed here, only renamed"
                )
                return False
            renamed = self._retarget_labels(target, wanted)
            if renamed is None:
                self.view.set_target_feedback("nothing to rename")
                return False
            changes: dict[str, Any] = {"target": renamed}
            feedback = f"renamed {sum(1 for name in wanted.values() if name)} output(s)"
        else:
            try:
                rewired = _target_from_records(target, records)
                if (
                    rewired == target
                    and dict(rewired.package_pins) == dict(target.package_pins)
                ):
                    self.view.set_target_feedback("nothing to change")
                    return False
                changes = {"target": rewired}
                if self.sequence is not None:
                    changes.update(_carried_onto(self.sequence, rewired))
            except ValueError as error:
                self.view.set_target_feedback(str(error))
                return False
            feedback = f"applied {len(records)} output(s)"
        if self.sequence is not None:
            candidate = self._rebuilt(**changes)
            if candidate is None:
                return False
            self._apply(candidate)
        else:
            self._board_target = changes["target"]
            self.revision += 1
            self.refresh()
        self.view.set_target_feedback(feedback)
        self.refresh_target()
        return True

    def _retarget_labels(self, target: object, labels: Mapping[str, str]) -> object | None:
        """One target with new display labels, and nothing else changed."""

        from zlc_pulse import PulseTarget

        changed = False
        ports = []
        for port in target.ports:
            wanted = labels.get(port.key)
            if wanted and wanted != port.label:
                changed = True
                ports.append(replace(port, label=wanted))
            else:
                ports.append(port)
        if not changed:
            return None
        return PulseTarget(
            target.raw_lanes,
            tuple(ports),
            package_pins=dict(target.package_pins) or None,
        )

    # ------------------------------------------------------------- the scan

    def set_binding(
        self, field_kind: str, period_id: object, port_key: object,
        scan: bool, source: str,
    ) -> None:
        """Change independent Scan capability and the one base value source."""
        if self.sequence is None:
            return
        candidate = self._edited_candidate("binding", (field_kind, period_id, port_key, scan, source))
        if candidate is None:
            return
        state_changes: dict[str, Any] = {"sequence": candidate}
        if tuple(b.field_id for b in candidate.scan_bindings) != tuple(
            b.field_id for b in self.sequence.scan_bindings
        ):
            state_changes.update(scan_rows=(), scan_source_dirty=bool(self._state.scan_source))
        self._edit_state(**state_changes)
        self.refresh()

    def _has_scan_slots(self) -> bool:
        """Whether anything is bound, and say what to do when nothing is.

        A scan table has one column per bound field, so with nothing bound
        there is no table to make or read -- and the operator's next move is a
        click on a dot in the Edit tab.  Failing later, on a column count,
        names the symptom instead.
        """

        if self.sequence is not None and self.sequence.scan_bindings:
            return True
        self._warn(
            "bind at least one field to a scan slot first "
            "(click a dot in the Edit tab)"
        )
        return False

    # ------------------------------------------------------------- scan page

    def _refresh_scan_page(self) -> None:
        """Everything the Scan page shows, from one place.

        Its parts move together -- binding a field changes the columns, which
        makes the generated table stale, which changes what Run would upload --
        so they are projected together rather than nudged one at a time.
        """

        from zlc_pulse import scan_columns_for

        view = self.view
        if self.sequence is None:
            view.set_scan_page(ScanPageRecord(slots_text="No pulse is open."))
            self._render_run_state()
            return
        columns = scan_columns_for(self.sequence)
        rows = self._state.scan_rows
        view.set_scan_page(
            ScanPageRecord(
                slots_text=(
                    (
                        "No bound fields: open a field's binding control and enable Scan or API."
                        if not columns and not self.sequence.api_bindings
                        else "Physical fields available to Scan and API callers:"
                    )
                    + (
                        " Run repeats is 0, so On Pulse stays at the first scan "
                        "point until Stop."
                        if rows and columns and self.sequence.run_repeats == 0
                        else ""
                    )
                ),
                bindings=self._binding_records(),
                table_text=_scan_table_text(rows, columns),
                source_text=self._state.scan_source,
                source_dirty=self._state.scan_source_dirty,
                repeats=self._state.scan_repeats,
                busy=False,
                progress_text=self._scan_progress,
                progress_polling=bool(
                    self._board_state.firing and rows
                    and str(getattr(view, "current_page", "")) == SCAN_PAGE
                ),
            )
        )
        # A table appearing is a change in what the controls can do: stepping
        # through a scan needs one to step through.
        self._render_run_state()

    def _template(self, kind: str) -> str:
        from zlc_pulse import scan_columns_for, scan_table_template

        return scan_table_template(kind, scan_columns_for(self.sequence))

    def write_scan_template(self, kind: str) -> None:
        """Replace the authored source with a starter program for these slots."""

        # The same question the run path asks, asked here too: a starter
        # program for zero bound slots can only be written by inventing one.
        if not self._has_scan_slots():
            return
        self._edit_state(
            scan_source=self._template(str(kind)),
            scan_source_dirty=True,
        )
        self._refresh_scan_page()

    def edit_scan_source(self, source: str) -> None:
        """Accept each text edit immediately into the sole authoring state."""

        self._edit_state(scan_source=str(source), scan_source_dirty=True)
        self._refresh_scan_page()

    def run_scan_program(self) -> bool:
        """Execute the scan program and keep the table it produced.

        The program is experiment code the operator just typed, run in this
        process on purpose: a scan table is a small array and the alternative
        -- a restricted expression language -- is a second language to learn
        for no safety anyone here needs.  What it must produce is checked:
        ``scan_table``, two-dimensional, one column per bound slot.

        In this process, not on its GUI thread.  A window runs it on its
        preview worker, so a program that loops, or builds a table one Python
        point at a time, holds that worker and not the event loop: Stop still
        answers, and in the editor bound to a console every console panel
        does too.  Without a worker (a notebook) it runs here and is answered
        before this returns; with one, True means it was started.
        """

        if self.sequence is None or not self._has_scan_slots():
            return False
        source = self._state.scan_source
        runner = self._run_preview_work
        if runner is None:
            try:
                table = _scan_table_of(source)
            except Exception as error:
                return self._scan_program_refused(f"scan program failed: {error}")
            return self._take_scan_program_table(source, table)
        if self._preview_close_requested:
            return False
        if self._scan_running:
            self._warn("the scan program is still running")
            return False
        # Queued behind any drawing on that worker, and never counted as
        # one: it touches nothing a close retires, so the retire does not
        # wait for it.  The window does -- its preview worker cannot close
        # under a running program -- and says what it is waiting for.
        self._scan_running = True
        self._scan_progress = "running the scan program…"
        self._refresh_scan_page()

        def delivered(table: object) -> None:
            self._scan_running = False
            if self._preview_close_requested:
                self._wake_close_guard()
                return
            self._take_scan_program_table(source, table)

        def failed(error: BaseException) -> None:
            self._scan_running = False
            if self._preview_close_requested:
                self._wake_close_guard()
                return
            self._scan_program_refused(f"scan program failed: {error}")

        self._submit_work(runner, lambda: _scan_table_of(source), delivered, failed)
        return True

    def _scan_program_refused(self, message: str) -> bool:
        """Say why a run left no table, where its progress was being shown."""

        self._scan_progress = message
        self._warn(message)
        self._refresh_scan_page()
        return False

    def _take_scan_program_table(self, source: str, table: object) -> bool:
        """Keep the table a scan program produced, if it is one."""

        from zlc_pulse import scan_columns_for, validate_scan_table

        if table is None:
            return self._scan_program_refused("the scan program did not assign scan_table")
        if self.sequence is None:
            return False
        # Through zlc_pulse, which owns what a legal table is.  This window
        # used to decide it here and again on the file-load path, and the two
        # did not agree: the loader skipped the width check entirely.  Checked
        # against the fields bound NOW, which is what the table will drive.
        try:
            rows = validate_scan_table(
                table,
                scan_columns_for(self.sequence),
            )
        except Exception as error:
            return self._scan_program_refused(str(error))
        self._accept_state(
            replace(
                self._state,
                scan_rows=tuple(tuple(value for value in row) for row in rows),
                # Typed over while it ran: what is in the box now is not what
                # made this table.
                scan_source_dirty=self._state.scan_source != source,
            )
        )
        self._scan_progress = f"{len(self._state.scan_rows)} scan point(s) ready"
        self.refresh()
        return True

    def set_scan_repeats(self, repeats: int) -> None:
        """How many times the table is played; zero means until Stop."""

        self._edit_state(scan_repeats=max(0, int(repeats)))
        self._refresh_scan_page()

    def load_scan_array(self) -> bool:
        """Read a scan table from a file, instead of generating one."""

        import numpy as np

        from zlc_pulse import scan_columns_for, validate_scan_table

        # Asked BEFORE the dialog: picking a file and then being told the
        # slots were never bound wastes the choice that was just made.
        if not self._has_scan_slots():
            return False
        chosen = self.view.ask_open_path(
            "Load scan array", str(Path(self.path).parent if self.path else ""),
            "Scan tables (*.npy *.csv *.txt);;All files (*)",
        )
        if not chosen:
            return False
        try:
            path = Path(chosen)
            columns = scan_columns_for(self.sequence)
            if path.suffix == ".npy":
                data = np.load(path)
                if data.ndim == 1 and len(columns) == 1:
                    # One field, one value per point, however it was saved.
                    data = data.reshape(-1, 1)
            else:
                # A one-column file is one field's points, not one point of
                # many fields: without ndmin the reader squeezed it flat, and
                # the commonest sweep file was refused on its column count.
                data = np.loadtxt(path, delimiter=",", ndmin=2)
            self._take_scan_rows(validate_scan_table(data, columns))
        except Exception as error:
            self._warn(f"cannot read {Path(chosen).name}: {error}")
            return False
        self._done(
            f"loaded {len(self._state.scan_rows)} scan point(s) from {Path(chosen).name}"
        )
        self._refresh_scan_page()
        return True

    def load_scan_program(self) -> bool:
        """Load Python source into the same immediate text state as typing."""

        chosen = self.view.ask_open_path(
            "Load scan program",
            str(Path(self.path).parent if self.path else ""),
            "Python source (*.py);;All files (*)",
        )
        if not chosen:
            return False
        try:
            source = Path(chosen).read_text(encoding="utf-8")
        except Exception as error:
            self._warn(f"cannot read {Path(chosen).name}: {error}")
            return False
        self.edit_scan_source(source)
        self._done(f"loaded scan source from {Path(chosen).name}")
        return True

    def save_scan_array(self) -> str:
        """Write the authored table without discarding fractional durations."""

        import numpy as np

        if not self._state.scan_rows:
            self._warn("there is no scan table to save")
            return ""
        values = np.asarray(self._state.scan_rows, dtype=float)
        try:
            target = unique_path(
                self._save_folder(),
                f"{(self.sequence.name if self.sequence else 'scan')}-scan",
                ".npy",
                writer=lambda temporary: np.save(temporary, values),
            )
        except Exception as error:
            self._warn(f"cannot save the scan table: {error}")
            return ""
        self._scan_progress = f"saved {target.name}"
        self._refresh_scan_page()
        return str(target)

    def hold_scan_point(self, *, worker: object = None) -> bool:
        """Stop advancing and PLAY the point the board is on, over and over.

        Holding is how a scan is inspected: the outputs stay at one row so a
        camera or a scope sees that point and nothing else -- which means the
        pulse keeps PLAYING, with the sweep frozen at one set of values.

        It stopped and wrote the row and left it there, so pressing Hold in the
        middle of a scan simply turned the outputs off: writing slot values is
        not playing them, and nothing fired again.

        A board that reports no cursor is not a reason to give up either.  It
        is idle, or it is not scanning, and "hold a point" still has an answer:
        the first one.
        """

        if self.sequencer is None:
            self._warn("this editor is not connected to a sequencer")
            return False
        if not self._scan_armed():
            self._warn("there is no scan to hold a point of")
            return False
        return self._hold(None, worker=worker)

    def step_scan_point(self, delta: int, *, worker: object = None) -> bool:
        """Move the held point by one, and keep playing the new one."""

        if self.sequencer is None or not self._scan_armed():
            self._warn("nothing is held to step")
            return False
        held = 0 if self._held_point is None else self._held_point
        return self._hold(held + int(delta), worker=worker)

    # A hand asked, so the board command runs on the device worker and its
    # outcome is shown when it is delivered.  That is the ONLY thing that
    # differs from the same action asked for in a script -- these used to be
    # second copies of it, and the copies had quietly lost the guards, so
    # Step with nothing held went straight at the board.  ``_guarded``
    # resolves its handler by NAME, which is what keeps the connection from
    # holding this presenter alive, so the adapter is a method rather than a
    # lambda at the wiring.
    def _hold_from_view(self) -> None:
        self.hold_scan_point(worker=self._run_device_work)

    def _step_from_view(self, delta: int) -> None:
        self.step_scan_point(delta, worker=self._run_device_work)

    def _hold(self, point: int | None, *, worker: object) -> bool:
        """Play one scan row, held, until something else is asked for.

        The whole gesture in one place, because Hold and either Step are the
        same three things in the same order -- stop, write that row, play it --
        and a version of it that skipped the third was how Hold came to mean
        "off".  ``None`` holds the row the board is on, which is a question
        for the board and travels with the command.  With a ``worker`` the
        command runs there and its outcome is shown when it is delivered;
        without one it runs here and is shown before this returns.
        """

        rows = self._state.scan_rows
        count = len(rows)
        # The draft the row is resolved from: an edit made while the command
        # runs on the worker is not what the board ends up holding.
        revision = self.revision
        try:
            # A held point is an ORDINARY pulse whose scanned fields carry that
            # row's numbers -- not a scan of length one.  Saying it the other
            # way round is what broke it: the board was handed a one-point
            # table looping forever, a state nothing else ever asks it for, and
            # its DAC segments were never re-applied while the digital edges
            # kept playing.  So resolve the row into the document and load a
            # plain pulse, which is the state the board is designed to hold.
            source = resolve_api_parameters(self.sequence)
            effective_rows, _wire = self._prepared_scan(source)
        except Exception as error:
            held = self._clamp_scan_point(point, count)
            self._held_point = held
            self._hold_failed(held, error)
            return False

        def prepare(held: int) -> tuple[PulseSequence, object]:
            return self.compile(resolve_scan_point(source, effective_rows[held]))

        # Refused before the lease is taken, as On Pulse is: a lease taken
        # and then refused would hold the sequencer against every other
        # owner with nothing playing.
        if worker is not None and not self._device_available():
            return False
        if not self._acquire_command():
            return False
        sequencer = self.sequencer
        previous_run = self._finite_run
        if worker is None:
            held = self._hold_scan_point(point, count, sequencer)
            self._held_point = held
            try:
                resolved, program = prepare(held)
            except Exception as error:
                # Settled as the worker's delivery settles it: the lease
                # taken above goes back unless the board still plays.
                self._settle_hold(held, count, self.board_state(), error, revision)
                return False
            self._finite_run = None
            error = self._drive_program(
                sequencer,
                program,
                resolved,
                run_repeats=0,
                scan_repeats=1,
                current=lambda: True,
                halt_first=True,
            )
            self._settle_hold(held, count, self.board_state(), error, revision)
            return error is None
        self._finite_run = None

        def work(operation: int) -> object:
            held = self._hold_scan_point(point, count, sequencer)
            try:
                resolved, program = prepare(held)
            except Exception as error:
                return held, self._board_state_for(sequencer), error
            error = self._drive_program(
                sequencer,
                program,
                resolved,
                run_repeats=0,
                scan_repeats=1,
                current=lambda: operation == self._device_operation,
                halt_first=True,
            )
            return held, self._board_state_for(sequencer), error

        def delivered(result: object, failure: BaseException | None) -> None:
            held, state, error = (
                result
                if result is not None
                else (self._clamp_scan_point(point, count), self._board_state, failure)
            )
            self._held_point = held
            self._settle_hold(held, count, state, error, revision)
            if error is not None and state.firing and previous_run is not None:
                self._finite_run = previous_run
                self._watch_completion()

        return self._run_device_command(work, delivered, summary="Holding...")

    @staticmethod
    def _clamp_scan_point(point: int | None, count: int) -> int:
        return max(0, min(count - 1, int(0 if point is None else point)))

    def _hold_scan_point(
        self,
        point: int | None,
        count: int,
        sequencer: object,
    ) -> int:
        """Resolve an explicit point or the board's cumulative row cursor."""

        if count < 1:
            return 0
        if point is not None:
            return self._clamp_scan_point(point, count)
        cursor = self._read_cursor(sequencer)
        return 0 if cursor is None else int(cursor) % count

    @staticmethod
    def _read_cursor(sequencer: object) -> int | None:
        """Current board row for the Hold gesture, on the thread that owns the device."""

        cursor = getattr(sequencer, "cursor", None)
        if not callable(cursor):
            return None
        try:
            return cursor()
        except Exception:  # noqa: BLE001 -- no cursor is an answer: the first row
            return None

    def _hold_failed(self, held: int, error: BaseException) -> None:
        self._scan_progress = f"cannot hold that point: {error}"
        self._warn(f"cannot hold scan point {held + 1}: {error}")
        self._refresh_scan_page()

    def _settle_hold(
        self,
        held: int,
        count: int,
        state: BoardState,
        error: BaseException | None,
        revision: int,
    ) -> None:
        """Show what holding a point left the board doing."""

        if error is None:
            self._finite_run = None
            # What the board reports for it from now on is this draft's row,
            # for ``synchronized`` to recognise until the draft changes.
            self._held_program = (revision, state.applied_digest)
            # Counted from 1, as the running line counts it.
            self._scan_progress = f"held at scan point {figure_padded(held + 1, count)} of {count}"
        else:
            self._scan_progress = f"cannot hold that point: {error}"
            self._warn(f"cannot hold scan point {held + 1}: {error}")
            if not state.firing:
                self._release_drive()
        self._adopt_board_state(state)
        self._refresh_scan_page()

    def _take_scan_rows(self, rows: Sequence[Sequence[float]]) -> None:
        """Hold a new table, and tell both pages that show it.

        The Scan page shows the table; the EDIT page shows how many points it
        has, beside the slots that will carry them -- which is the number an
        operator checks before pressing On Pulse.  One place assigns the rows,
        so the two cannot disagree about what is loaded.
        """

        self._edit_state(
            scan_rows=tuple(tuple(value for value in row) for row in rows)
        )
        self.refresh()

    def _scan_armed(self) -> bool:
        """A scan exists only when there is a table AND fields for it to drive.

        Both halves, because each can outlive the other: unbinding the last
        field leaves yesterday's rows in memory, and a bound field can exist
        before any table is generated.  The upload gate and the run-length
        decision read THIS, together -- they used to disagree, so a stale
        table changed what On Pulse MEANS: a pulse with no scan left in it
        played once and stopped, its own repeat overwritten by a ghost.
        Scan repeats is a statement about a scan; without one it governs
        nothing.
        """

        return bool(self._state.scan_rows) and bool(
            self.sequence is not None and self.sequence.scan_bindings
        )

    def _scan_progress_from_view(self) -> None:
        """The Scan page asking where the board is, answered when the board does."""

        if self.sequencer is None:
            return
        self.ask_run_state(then=self._render_scan_progress)

    def _render_scan_progress(self) -> None:
        cursor = self._board_state.cursor
        applied = self._applied_scan
        if not self.running or cursor is None or applied is None:
            self._scan_progress = ""
            self.view.set_scan_progress_text(self._scan_progress)
            return
        digest, names, rows = applied
        if digest != self._board_state.applied_digest or not rows:
            self._scan_progress = ""
            self.view.set_scan_progress_text(self._scan_progress)
            return
        point = int(cursor) % len(rows)
        values = ", ".join(
            f"{name} = {text}" for name, text in zip(names, rows[point], strict=True)
        )
        self._scan_progress = (
            f"Scan: point {figure_padded(point + 1, len(rows))} / {len(rows)}"
            + (f"  {values}" if values else "")
        )
        self.view.set_scan_progress_text(self._scan_progress)


    def _push_schedule(self, vm: ScheduleVM) -> None:
        """Show this model, advancing the revision when it is a different one.

        The view refuses two different models under one revision, and it is
        right to: a stale card would otherwise be indistinguishable from a
        fresh one.  Leaving each mutator to remember the bump is how one was
        missed -- Hide Off then Show All produced two models at the same
        revision, and the refusal came out of a Qt slot, which aborts the
        process with no traceback.

        So the presenter answers "has what I show changed" by LOOKING, once,
        here.  A ScheduleVM is frozen, so the comparison is an equality test.
        """

        if self._shown is not None and replace(vm, revision=0) != replace(
            self._shown, revision=0
        ):
            self.revision += 1
            vm = replace(vm, revision=self.revision)
        self._shown = vm
        self.view.set_schedule(vm)

    def refresh(self) -> None:
        self._refresh_config_page()
        self._refresh_component_document()
        target = self._current_target()
        if self.sequence is None:
            # No pulse.  If a board is attached its ports, pins and clock are
            # still real and still worth showing -- that is what a first period
            # will land on -- and if none is, there is nothing to show at all.
            self._push_schedule(
                project_schedule(
                    None,
                    target=target,
                    time_step_ns=self._board_step_ns,
                    revision=self.revision,
                    pins=self.pins,
                )
                if target is not None
                else replace(EMPTY_SCHEDULE, revision=self.revision)
            )
            self.view.set_title("PulseGUI - no pulse")
            self.view.set_summary(self._document_summary())
            self.view.show_preview_placeholder("Load a pulse to see its timeline")
            self._show_connection(self._connection_status)
            self.refresh_target()
            return
        self._push_schedule(
            project_schedule(
                self.sequence,
                path=self.path,
                generation=0,
                revision=self.revision,
                visible_ports=self._state.visible_ports,
                pins=self.pins,
                scan_points=len(self._state.scan_rows),
                config_values=self._active_config_values(),
                scan_active=self._scan_armed(),
            )
        )
        self.view.set_title(f"PulseGUI - {self.sequence.name}")
        self.view.set_summary(self._document_summary())
        self._show_connection(self._connection_status)
        self.refresh_target()
        self._refresh_scan_page()
        self.refresh_preview()

    def _document_summary(self) -> str:
        """The status line's resting sentence: what the editor holds."""

        if self.sequence is not None:
            return f"{self.path or self.sequence.name} - {len(self.sequence.periods)} period(s)"
        target = self._current_target()
        if target is None:
            return EMPTY_SCHEDULE.summary_text
        return f"{len(target.ports)} port(s) on this board - Add Period to start a pulse"

    def _on_include_off(self, _included: bool) -> None:
        """Showing every channel changes how many rows are drawn.

        So the size it was drawn at is no longer the right one, and any size
        the operator pinned was pinned for a different picture: drop the pin
        and let the content choose again.
        """

        self._pinned_size = None
        self.refresh_preview()

    def set_preview_size(self, name: str) -> None:
        """Pin the preview to one preset.

        Picking a size is a decision, and it sticks until the content changes
        shape underneath it -- otherwise the plot would snap back the moment
        anything was edited, and the operator's choice would look like a bug.
        """

        self._pinned_size = str(name)
        self.refresh_preview()

    def set_preview_selectors(self, enabled: bool) -> None:
        """Whether the operator may drag on the preview.

        The same question the console's switch asks, answered the same way:
        the host gates interaction.  This only set a flag and redrew, and the
        flag was read in ONE place -- the arguments the host is BUILT with --
        so after the first draw the switch did nothing at all, in either
        direction.
        """

        self._preview_selectors = bool(enabled)
        if self._preview_host is not None:
            self._preview_host.set_interaction_enabled(self._preview_selectors)

    def refresh_preview(self) -> None:
        """Redraw the preview from what is on screen, IF it is on screen.

        The host is built once and updated after that.  Rebuilding it per edit
        spawned a render worker and a matplotlib session for every keystroke,
        blocked the GUI thread waiting for the new one's first frame, and never
        closed the old -- an afternoon of editing left hundreds of live threads
        behind a window that felt slower with every change.

        And it is not built at all until the Preview page is looked at.  A
        window opens on Edit; building the drawing stack to fill a tab behind
        it was over a third of the time between double-click and a window, and
        the operator who never opens Preview paid it every time.  Turning to
        the page is what asks for the drawing -- and it stays built after that,
        because turning away does not mean the answer stopped being wanted.
        """

        if self._make_preview is None or self.sequence is None:
            return
        if self._preview_host is None and not self._preview_on_screen:
            return
        self._preview_pending = (
            self._effective_sequence(self.sequence),
            bool(self.view.preview_include_off_rows),
            self._pinned_size,
        )
        if not self._preview_busy and not self._preview_close_requested:
            self._start_preview_refresh()

    def _start_preview_refresh(self) -> None:
        """Submit only the newest pending preview to the window's one worker."""

        request = self._preview_pending
        if request is None or self._preview_close_requested:
            return
        self._preview_pending = None
        self._preview_busy = True
        sequence, include_off, pinned_size = request
        previous = self._preview_host
        self.view.set_preview_status("drawing preview…")

        def work() -> object:
            data, size, rows, periods, total_ns = preview_candidate(
                sequence,
                include_off,
                pinned_size,
            )
            if previous is None:
                host = self._make_preview(data, size=size)
                grown = True
            else:
                host = previous
                grown = bool(self._update_preview(host, data, size=size))
            return (
                host,
                size,
                rows,
                periods,
                total_ns,
                grown,
            )

        def delivered(result: object) -> None:
            host, size, rows, periods, total_ns, grown = result
            if previous is None:
                self._preview_host = host
            if not self._preview_close_requested and self._preview_pending is None:
                host.set_interaction_enabled(self._preview_selectors)
                if not self._preview_mounted or grown:
                    self.view.show_preview(host)
                    self._preview_mounted = True
                self._show_preview_state(size, rows, periods, total_ns)
            self._finish_preview_operation()

        def failed(error: BaseException) -> None:
            message = f"cannot draw this pulse: {error}"
            if self._preview_host is None:
                self.view.show_preview_placeholder(message)
            else:
                self.view.set_preview_status(message)
            self._finish_preview_operation()

        try:
            assert self._run_preview_work is not None
            self._run_preview_work(work, delivered, failed)
        except BaseException as error:
            failed(error)

    def _finish_preview_operation(self) -> None:
        self._preview_busy = False
        if self._preview_close_requested:
            self._preview_pending = None
            if self._request_preview_close is not None:
                self._request_preview_close()
            return
        if self._preview_pending is not None:
            self._start_preview_refresh()

    def _show_preview_state(
        self,
        size: str,
        rows: int,
        periods: int,
        total_ns: float,
    ) -> None:
        """What the preview is showing, in the words above it."""

        view = self.view
        view.set_preview_size_names(PANEL_SIZE_NAMES)
        view.set_preview_size(size)
        view.set_preview_status(
            f"{periods} period(s), "
            f"{rows} channel(s), "
            f"{_readable(total_ns)}"
            f"  -  {size}"
        )

    def save_preview_image(self) -> None:
        """Write the preview exactly as it is drawn.

        The plotting host renders its own file, so what lands on disk is the
        preview rather than a second drawing of the same pulse.
        """

        if not self._check_bracket() or self._preview_close_requested:
            return
        if self._preview_host is None:
            self._warn("there is no preview to save")
            return
        name = (self.sequence.name if self.sequence is not None else "pulse") or "pulse"
        folder = self._save_folder()
        # The HOST writes the file.  The widget is a Qt view onto it and has
        # never had a save(), so this always answered "cannot save itself" in
        # the shipped window while the test substituted a fake that could.
        save = getattr(self._preview_host, "save", None)
        if not callable(save):
            self._warn("this preview cannot save itself")
            return

        if self._preview_busy:
            self.view.set_preview_status(
                "preview is busy; save again when the drawing is ready"
            )
            return

        self._preview_busy = True
        self.view.set_preview_status("saving preview…")

        def work() -> object:
            def write(temporary: Path) -> None:
                result = save(temporary)
                # Waited out, not given five seconds: the render child writes
                # this temporary, and a writer that gave up on it left the
                # file it then wrote behind in the pulses folder.  The render
                # child answers, or fails every request when it dies, and the
                # wait is on the preview worker, not Qt.
                if hasattr(result, "result"):
                    result.result()

            return unique_path(folder, name, ".png", writer=write)

        def delivered(target: object) -> None:
            self.view.set_preview_status(f"saved {target}")
            self._finish_preview_operation()

        def failed(error: BaseException) -> None:
            self._warn(f"cannot save the preview: {error}")
            self._finish_preview_operation()

        try:
            assert self._run_preview_work is not None
            self._run_preview_work(work, delivered, failed)
        except BaseException as error:
            failed(error)

    # -------------------------------------------------------------- private

    def _edit_period(
        self,
        period_id: str,
        action: str,
        *args: object,
        ripples_forward: bool = False,
    ) -> None:
        """Change one period's values.  The card it lives in already shows them.

        Some changes are not confined to their own card.  A DAC level holds
        until something else sets it, so every LATER period displays the level
        this one left -- change it, and their cards are showing a number that
        is no longer true.  Those cards are pushed too.
        """

        if self.sequence is None:
            return
        if str(period_id) not in self.sequence.period_by_id:
            self._warn(f"{period_id} is not a period in this sequence")
            return
        following = ()
        if ripples_forward:
            order = list(self.sequence.period_by_id)
            following = tuple(order[order.index(str(period_id)) + 1 :])
        candidate = self._edited_candidate(action, (period_id, *args))
        if candidate is None:
            self.view.set_period(project_period(
                self.sequence, self.sequence.period_by_id[str(period_id)],
                visible_ports=self._state.visible_ports,
                config_values=self._active_config_values(),
                scan_active=self._scan_armed(),
            ))
            return
        self._apply_value(
            candidate,
            period_id=str(period_id),
            also=following,
        )

    def _rebuilt(self, **changes: Any) -> PulseSequence | None:
        """The sequence this edit would produce, or None with the reason shown.

        The model decides what is legal; this only makes its refusal something
        an operator reads instead of something that ends the process.
        """

        if self.sequence is None:
            return None
        try:
            candidate = replace_sequence(self.sequence, **changes)
            # A binding cannot outlive the field it is about.  Clearing a port
            # and choosing Hold both take a DAC step away, and the binding on
            # it used to stay: every later read raised on a pulse that looked
            # fine.  Pruning here covers every gesture that changes shape,
            # including ones written after this one.
            pruned, dropped = prune_orphaned_bindings(candidate)
            if dropped:
                self._warn(
                    "unbound "
                    + ", ".join(dropped)
                    + ": the field it named is no longer set here"
                )
            return pruned
        except Exception as error:
            self._warn(str(error))
            return None

    def _apply(self, candidate: PulseSequence | None) -> None:
        """A change of SHAPE: periods, ports, target, or a different pulse.

        The whole board is re-projected, because what changed is what the board
        is made of.  Any accepted edit also means the board no longer holds
        what is on screen -- that is what the On Pulse asterisk says.
        """

        if candidate is None:
            return
        self._edit_state(sequence=candidate)
        self.refresh()

    def _apply_value(
        self,
        candidate: PulseSequence | None,
        *,
        period_id: str | None = None,
        port_key: str | None = None,
        also: Sequence[str] = (),
    ) -> None:
        """A change of VALUE inside a shape that did not move.

        The widget that raised it already shows the new value -- a checkbox is
        checked because the operator clicked it -- so re-projecting the whole
        board would rebuild every card to arrive back where the screen already
        was, throwing away the scroll position and any partly-typed field on
        the way.  Only what changed is pushed back.
        """

        if candidate is None:
            return
        self._edit_state(sequence=candidate)
        schedule = self.view
        bindings = bindings_of(candidate)
        for identifier in ((period_id,) if period_id is not None else ()) + tuple(also):
            period = next(
                (item for item in candidate.periods if item.period_id == identifier),
                None,
            )
            if period is not None:
                schedule.set_period(
                    project_period(
                        candidate,
                        period,
                        visible_ports=self._state.visible_ports,
                        bindings=bindings,
                        config_values=self._active_config_values(),
                        scan_active=self._scan_armed(),
                    )
                )
        if port_key is not None:
            schedule.set_delay_row(_delay_row(candidate, port_key, bindings, self._active_config_values()))
        self._refresh_summary()
        self._refresh_config_page()
        self._refresh_component_document()
        self._render_run_state()
        self.refresh_preview()

    def _refresh_summary(self) -> None:
        """The totals a value edit can move, without touching the cards.

        Read off the same projection that draws the page rather than worked out
        a second time here.  The second derivation disagreed with the first: a
        value edit rewrote the name label into "<path> - 3 period(s)" where the
        page had been showing the file name, and the scan label into "1 scan
        slot(s)" where the page says "1 slot - 21 pts".  Every label the page
        shows is the projection's to word.
        """

        if self.sequence is None:
            return
        shown = project_schedule(
            self.sequence,
            path=self.path,
            revision=self.revision,
            visible_ports=self._state.visible_ports,
            pins=self.pins,
            scan_points=len(self._state.scan_rows),
            config_values=self._active_config_values(),
            scan_active=self._scan_armed(),
        )
        if shown.components:
            self._push_schedule(shown)
        self.view.set_schedule_summary(
            total_text=shown.total_text,
            total_tooltip=shown.total_tooltip,
            period_count=shown.period_count,
            visible_text=shown.visible_text,
            summary_text=shown.summary_text,
            scan_summary_text=shown.scan_summary_text,
        )

    def _on_grid(self, value: object, unit: str, field: str,
                 *, minimum: int | None = 1) -> float | None:
        """One rounding rule for every time an operator types, or None if the
        value is not a number at all -- which IS worth saying out loud.

        The unit decides nothing here.  On a 20 ns clock a whole number of
        us/ms/s lands on the grid by arithmetic, so only ns ever felt the
        rule; routing every unit through the same call is what stops that
        difference from being written into the code as a special case.
        """

        if self.sequence is None:
            return None
        try:
            # A field sends what was typed, which is text.  Numbers arrive as
            # numbers from a notebook.  Both are the same question.
            return align_to_grid(float(value), str(unit),
                                 float(self.sequence.time_step_ns),
                                 field, minimum=minimum)
        except Exception as error:
            self._warn(str(error))
            return None

    def _check_bracket(self) -> bool:
        try:
            if self.sequence is not None:
                self.sequence.require_nonempty_brackets()
        except ValueError as error:
            self._warn(str(error))
            return False
        return True

    def _warn(self, text: str) -> None:
        warn = getattr(self.view, "show_warning", None)
        if warn is not None:
            warn(text)

    def _done(self, text: str) -> None:
        """Say that something the operator asked for has happened.

        Only ever spoke up to refuse, so a Save that worked and a Save that
        did nothing looked identical.

        For the OCCASIONAL actions only.  This modal blocks, and On Pulse,
        Stop and load-onto-board are pressed over and over at the bench: a
        dialog in that path is not confirmation, it is something to dismiss
        before the next press.  Connecting, saving a file and running a scan
        program are done once and are worth a sentence; running a pulse says
        what it did through the status dot and the On Pulse label, which are
        already there and do not have to be clicked away.
        """

        done = getattr(self.view, "show_done", None)
        if done is not None:
            done(text)

    def _wake_close_guard(self) -> None:
        if self._preview_close_requested and self._request_preview_close is not None:
            self._request_preview_close()

    def may_close(self) -> bool:
        """Whether unsaved pulse and Config edits may go; asked once per close."""

        return self._preview_close_requested or (
            self._discard_pulse_edits() and self._discard_config_edits() and self._discard_subpulse_edits()
        )

    def prepare_preview_close(self) -> bool:
        """Stop accepting work and report when every owned delivery is idle.

        A status question on its way counts: the device worker cannot be
        closed while it is asking, and its answer wakes the close.  A scan
        program does not: it touches nothing the retire closes, so only the
        preview worker it runs on waits for it, and the window says so.
        """

        if not self._preview_close_requested:
            self._device_operation += 1
            self._preview_close_requested = True
        self._preview_pending = None
        # Rendered on every wake: Stop stays offered until the retire starts.
        self._render_run_state()
        retiring = self._retiring()
        if retiring and self._scan_running:
            self.view.set_summary("Stopping... waiting for the scan program to finish")
        return retiring

    def _retiring(self) -> bool:
        """Closing with every owned delivery idle: the retire has the drive."""

        return self._preview_close_requested and not (
            self._preview_busy
            or self._device_busy
            or self._stop_busy
            or self._status_in_flight
        )

    def cancel_preview_close(self) -> None:
        """Resume preview requests after owned retirement was refused."""

        self._preview_close_requested = False
        self._render_run_state()

    def close(self, *, present: bool = True) -> None:
        """Let go of the board and the preview -- both, whichever fails.

        The preview HOST owns a render worker and a matplotlib session, and
        close only cleared the widget: an editor opened and shut left the
        thread running for the life of the process.  Its two sibling presenters
        both close their hosts.
        """

        error: BaseException | None = None
        try:
            self._release(present=present)
        except BaseException as caught:
            error = caught
        try:
            self._close_preview()
        except BaseException as caught:
            if error is None:
                error = caught
        if error is not None:
            raise error

    def _close_preview(self) -> None:
        host = self._preview_host
        if host is not None:
            if host.close() is False:
                raise RuntimeError("PulseGUI preview worker did not close")
            self._preview_host = None
            self._preview_mounted = False


def replace_sequence(sequence: PulseSequence, **changes: Any) -> PulseSequence:
    """Rebuild one sequence with some parts replaced.

    ``PulseSequence`` validates in its constructor, which is exactly what makes
    this safe: an edit that would produce something the hardware cannot play
    raises here rather than surviving to compile time.

    It raises, and the presenter catches it in one place -- see ``_rebuilt``.
    Every call site used to test the result for None, which this never returned,
    so a refused value escaped through a Qt slot instead: PyQt5 ends the process
    on an exception out of a slot, so typing a duration the model rejects closed
    the window with no message at all.

    ``dataclasses.replace`` carries every field the model has, so the model is
    the one list of what a pulse is made of.  A list written here forgets the
    field added after it was written -- the config parameters were -- and a
    rename or a typed duration then rebuilt the pulse without them: a board
    calibration silently unbound by an edit that never mentioned it.
    """

    return replace(sequence, **changes)


def _reordered_sequence(
    sequence: PulseSequence, order: Sequence[tuple[str, str]],
    periods: Mapping[str, PulsePeriod] | None = None,
) -> PulseSequence:
    """One edit of the flat timeline, for both a Pulse and a Subpulse draft."""
    by_id = dict(sequence.period_by_id) if periods is None else dict(periods)
    if not by_id:
        raise ValueError("a sequence needs at least one period")
    items = tuple(tuple(item) for item in order)
    expected = {("period", key) for key in by_id}
    for bracket in sequence.brackets:
        expected.update((
            ("bracket", bracket_post_key(bracket.bracket_id, "start")),
            ("bracket", bracket_post_key(bracket.bracket_id, "end")),
        ))
    if len(items) != len(expected) or set(items) != expected:
        raise ValueError("schedule order must contain each current item exactly once")
    brackets = []
    for bracket in sequence.brackets:
        start = items.index(("bracket", bracket_post_key(bracket.bracket_id, "start")))
        end = items.index(("bracket", bracket_post_key(bracket.bracket_id, "end")))
        if end < start:
            raise ValueError(f"bracket {bracket.bracket_id} end precedes its start")
        first = next((key for kind, key in items[start + 1:] if kind == "period"), None)
        last = next((key for kind, key in reversed(items[:end]) if kind == "period"), None)
        brackets.append(replace(bracket, start_period_id=first, end_period_id=last))
    ids = tuple(key for kind, key in items if kind == "period")
    members = {component.component_id: set(component.period_ids) & set(ids)
               for component in sequence.components}
    # An inserted card strictly between two members belongs to that group.
    # Boundary insertions stay outside; the Component editor owns its edges.
    for index, key in enumerate(ids):
        if key in sequence.period_by_id or not 0 < index < len(ids) - 1:
            continue
        left = sequence.component_for_period(ids[index - 1])
        right = sequence.component_for_period(ids[index + 1])
        if left is not None and left == right:
            members[left.component_id].add(key)
    return replace(
        sequence, periods=tuple(by_id[key] for key in ids), brackets=tuple(brackets),
        bindings=tuple(binding for binding in sequence.bindings
                       if binding.field_ref.period_id is None or binding.field_ref.period_id in by_id),
        components=tuple(replace(component, period_ids=tuple(
            key for key in ids if key in members[component.component_id]
        )) for component in sequence.components if members[component.component_id]),
    )


def _sequence_edited(sequence: PulseSequence, action: str, args: tuple) -> PulseSequence:
    """The existing authoring operations, independent of which view emitted them.

    There is no second Subpulse editor algorithm: a candidate is built here,
    then its owning document accepts it. No device or view is touched.
    """
    if action == "document_name":
        return replace(sequence, name=str(args[0]))
    if action in {"period_name", "duration", "digital", "analog"}:
        key = str(args[0])
        period = sequence.period_by_id[key]
        if action == "period_name":
            edited = replace(period, name=str(args[1]))
        elif action == "duration":
            value, unit = args[1:]
            edited = replace(period, duration=align_to_grid(
                float(value), str(unit), sequence.time_step_ns, "duration"
            ), unit=str(unit))
        elif action == "digital":
            port_key, high = args[1:]
            port = sequence.target.by_key[str(port_key)]
            states = list(period.states)
            states[sequence.target.raw_lanes.index(port.lanes[0])] = int(bool(high))
            edited = replace(period, states=tuple(states))
        else:
            port_key, mode, value = args[1:]
            steps = tuple(step for step in period.analog_steps if step.port != port_key)
            text = "" if value is None else str(value).strip()
            if text and str(mode) != HOLD_MODE:
                steps += (AnalogStep(str(port_key), str(mode), int(float(text))),)
            edited = replace(period, analog_steps=steps)
        return replace(sequence, periods=tuple(
            edited if item.period_id == key else item for item in sequence.periods
        ))
    if action == "binding":
        kind, period_id, port_key, scan, source = args
        reference = PulseFieldRef(
            "dac" if kind == "analog" else str(kind),
            period_id=None if kind == "delay" else str(period_id),
            port=None if kind == "duration" else str(port_key),
        )
        current = next((b for b in sequence.bindings if b.field_ref == reference), None)
        binding = PulseBinding(
            reference, current.unit if current else sequence.field_unit(reference),
            scan=scan, source=source,
            config_key=current.config_key if current and source == "config" else "",
        )
        bindings = tuple(binding if b.field_ref == reference else b
                         for b in sequence.bindings
                         if b.field_ref != reference or scan or source != "default")
        if current is None and (scan or source != "default"):
            bindings += (binding,)
        periods = sequence.periods
        period = sequence.period_by_id.get(str(reference.period_id))
        if (scan or source != "default") and reference.kind == "dac" and period is not None:
            if not any(step.port == reference.port for step in period.analog_steps):
                port = sequence.target.by_key[reference.port]
                owned = replace(period, analog_steps=period.analog_steps + (
                    AnalogStep(reference.port, ANALOG_MODE_CHOICES[0], _held_value(sequence, period, port)),
                ))
                periods = tuple(owned if p.period_id == period.period_id else p for p in periods)
        return replace(sequence, periods=periods, bindings=bindings)
    if action == "config_binding":
        field_id, key = args
        key = str(key).strip()
        if key:
            key = config_parameter_key(key)
        if not any(b.field_id == field_id for b in sequence.config_bindings):
            raise ValueError("the field is not a Config parameter")
        return replace(sequence, bindings=tuple(
            replace(b, config_key=key) if b.field_id == field_id else b for b in sequence.bindings
        ))
    if action in {"insert_period", "insert_spacer"}:
        before, = args
        periods = sequence.periods
        order = list(_sequence_item_order(sequence))
        position = len(order) if before is None else order.index(tuple(before))
        index = sum(kind == "period" for kind, _key in order[:position])
        spacer = action == "insert_spacer"
        model = (next((p for p in reversed(periods) if p.kind == PERIOD_KIND_SPACER), None)
                 if spacer else periods[max(0, index - 1)])
        neighbours = [periods[i] for i in (index - 1, index) if 0 <= i < len(periods)]
        states = (tuple(int(all(p.states[lane] for p in neighbours))
                        for lane in range(len(sequence.target.raw_lanes))) if spacer else model.states)
        key = _unique_id((*sequence.period_by_id, *(p.name for p in periods)), "spacer" if spacer else "period")
        added = PulsePeriod(
            key, model.duration if model else 1.0, model.unit if model else "ms", states,
            kind=PERIOD_KIND_SPACER if spacer else "period",
        )
        order.insert(position, ("period", key))
        return _reordered_sequence(sequence, order, {p.period_id: p for p in (*periods, added)})
    if action == "reorder_items":
        return _reordered_sequence(sequence, args[0])
    if action == "remove_items":
        selected = tuple(args[0])
        if not selected:
            raise ValueError("select the timeline items to remove")
        periods = {key for kind, key in selected if kind == "period"}
        brackets = {key.rpartition(":")[0] for kind, key in selected if kind == "bracket"}
        for kind, key in selected:
            if kind == "component":
                fragment = extract_subpulse(sequence, key)
                periods.update(p.period_id for p in fragment.periods)
                brackets.update(b.bracket_id for b in fragment.brackets)
        retained = replace(sequence, brackets=tuple(b for b in sequence.brackets if b.bracket_id not in brackets))
        return _reordered_sequence(
            retained, tuple(item for item in _sequence_item_order(retained) if item[0] != "period" or item[1] not in periods),
            {p.period_id: p for p in retained.periods if p.period_id not in periods},
        )
    if action in {"bracket", "bracket_add"}:
        if action == "bracket_add":
            start, end, count = args
            key = _unique_id(tuple(b.bracket_id for b in sequence.brackets), "bracket")
        else:
            key, start, end, count = args
        changed = PulseBracket(str(key), None if start is None else str(start),
                               None if end is None else str(end), int(count))
        brackets = (sequence.brackets + (changed,) if action == "bracket_add" else tuple(
            changed if b.bracket_id == key else b for b in sequence.brackets
        ))
        return replace(sequence, brackets=brackets)
    raise ValueError(f"unknown Pulse edit {action!r}")


def _target_from_records(target: object, records: Sequence[object]) -> PulseTarget:
    """The target the Offline page's records describe.

    An endpoint is what the page shows for one wire: its package pin when the
    target carries a pin map, otherwise its lane name.  A wire keeps the lane
    it has -- found by its pin, or by the output and bit it sits at -- so an
    output re-pinned or reordered carries its levels with it; a wire the
    target never had is a new lane, named by its output and bit when only its
    pin is known.  Without a pin map an endpoint has to be a lane name, so the
    page's ``endpoint:...`` placeholder is refused in the model's words rather
    than turned into a wire nobody named.

    Lanes and ports keep the order the target had, new ones after them, and
    DAC buses keep their numbers, so applying the page unchanged yields the
    same target -- the ABI fingerprint is made of exactly those orders.
    """

    from zlc_pulse import PulsePortSpec

    pins = dict(target.package_pins)
    lane_of_pin = {pin: lane for lane, pin in pins.items()}
    old = {port.key: port for port in target.ports}
    # A clock no DAC latches with has no row on the page: it keeps its name
    # and its wire, both booked before the page's outputs, so an output
    # written onto either is refused in the clock's words -- the page lists
    # no such port to look for.  The page's one way to it is a DAC with no
    # latch clock naming its wire as the latch endpoint: that DAC takes it.
    carried = [port for port in programmable_ports(target) if port.kind == "clock"]
    clock_on = {port.lanes[0]: port.key for port in carried}
    # A clock a DAC takes leaves ``clock_on`` but keeps its name booked.
    wire_of_clock = {key: lane for lane, key in clock_on.items()}
    lanes: list[str] = []
    new_pins: dict[str, str] = {}

    def wire(endpoint: object, key: str, bit: int, fallback: str) -> str:
        text = str(endpoint).strip()
        if not text:
            raise ValueError(f"{key}: every wire needs an endpoint")
        if pins:
            lane = lane_of_pin.get(text)
            if lane is None:
                previous = old.get(key)
                lane = (
                    previous.lanes[bit]
                    if previous is not None and bit < len(previous.lanes)
                    else fallback
                )
            new_pins[lane] = text
        else:
            lane = text
        if lane in lanes:
            raise ValueError(
                f"{key}: wire {text!r} is held by clock {clock_on[lane]}, "
                "which no DAC latches with"
                if lane in clock_on
                else f"{key}: wire {text!r} is already used by another output"
            )
        lanes.append(lane)
        return lane

    def unclaimed(key: str) -> str:
        lane = wire_of_clock.get(key)
        if lane is not None:
            raise ValueError(
                f"{key}: that name is taken by the clock on wire "
                f"{pins.get(lane, lane)!r}, "
                + (
                    "which no DAC latches with; a DAC without a latch clock "
                    "takes it by naming that wire as its latch endpoint"
                    if lane in clock_on
                    else "which another DAC latches with in this apply"
                )
            )
        return key

    def port_spec(key: str, *parts: object, **fields: object) -> PulsePortSpec:
        try:
            return PulsePortSpec(key, *parts, **fields)
        except ValueError as error:
            raise ValueError(f"{key}: {error}") from error

    dac_records = [record for record in records if str(record.kind) == "dac"]
    dac_order = sorted(
        range(len(dac_records)),
        key=lambda index: (
            old[str(dac_records[index].key)].bus_index
            if str(dac_records[index].key) in old
            and old[str(dac_records[index].key)].kind == "dac"
            else len(old),
            index,
        ),
    )
    bus_of = {
        str(dac_records[index].key): bus for bus, index in enumerate(dac_order)
    }
    for port in carried:
        lane = port.lanes[0]
        wire(pins.get(lane, lane), port.key, 0, lane)
    ports: list[PulsePortSpec] = list(carried)
    for record in records:
        key = unclaimed(str(record.key))
        kind = str(record.kind)
        label = str(record.signal).strip()
        endpoints = tuple(record.endpoints)
        previous = old.get(key)
        if kind == "digital":
            if len(endpoints) != 1:
                raise ValueError(f"{key}: a digital output has one wire, not {len(endpoints)}")
            ports.append(port_spec(key, "digital", (wire(endpoints[0], key, 0, key),), label=label))
        elif kind == "dac":
            data = tuple(
                wire(endpoint, key, bit, f"{key}_{bit}")
                for bit, endpoint in enumerate(endpoints)
            )
            # A DAC may have no latch clock; it gets one only when a clock
            # wire is written for it, never a clock wired to "None".  Written
            # onto the wire of a clock no DAC latches with, it latches with
            # that clock, already booked, which from here on is its own.
            clock_text = str(record.clock_endpoint or "").strip()
            taken = (
                None if record.clock_key or not clock_text
                else clock_on.pop(lane_of_pin.get(clock_text) if pins else clock_text, None)
            )
            clock_key = (
                taken if taken is not None
                else unclaimed(str(record.clock_key)) if record.clock_key
                else unclaimed(f"{key}_clock") if clock_text else None
            )
            clock_lane = (
                None if clock_key is None or taken is not None
                else wire(record.clock_endpoint, clock_key, 0, clock_key)
            )
            same_width = (
                previous is not None
                and previous.kind == "dac"
                and len(previous.lanes) == len(data)
            )
            ports.append(port_spec(
                key,
                "dac",
                data,
                label=label,
                bus_index=bus_of[key],
                encoding=previous.encoding if same_width else None,
                safe_value=previous.safe_value if same_width else None,
                latch_clock=clock_key,
            ))
            if clock_key is not None and taken is None:
                clock = old.get(clock_key)
                ports.append(port_spec(
                    clock_key,
                    "clock",
                    (clock_lane,),
                    label=clock.label if clock is not None else "",
                ))
        else:
            raise ValueError(f"{key}: unknown output kind {kind!r}")
    position = {port.key: index for index, port in enumerate(target.ports)}
    ports.sort(key=lambda port: position.get(port.key, len(position)))
    raw_lanes = tuple(lane for lane in target.raw_lanes if lane in lanes) + tuple(
        lane for lane in lanes if lane not in target.raw_lanes
    )
    return PulseTarget(raw_lanes, tuple(ports), package_pins=new_pins or None)


def _carried_onto(sequence: PulseSequence, target: PulseTarget) -> dict[str, Any]:
    """The pulse's periods and delays on a re-wired target, or why it cannot go.

    Levels belong to outputs, not to lanes: each period's states follow the
    output and bit they were authored on, so a re-pinned or reordered output
    plays what it played.  An output the new target has no place for -- gone,
    or of another kind -- is refused while anything is authored on it, a
    level, a step, a delay or a binding, and nothing is dropped on the way.
    """

    old = sequence.target
    old_index = {lane: index for index, lane in enumerate(old.raw_lanes)}
    new_index = {lane: index for index, lane in enumerate(target.raw_lanes)}
    kept = {
        key: port
        for key, port in target.by_key.items()
        if key in old.by_key and old.by_key[key].kind == port.kind
    }
    bindings = sequence.bindings
    blocked: list[str] = []
    for port in old.ports:
        if port.key in kept or port.kind == "clock":
            continue
        uses: list[str] = []
        if port.kind == "digital":
            high = [
                period.period_id
                for period in sequence.periods
                if period.states[old_index[port.lanes[0]]]
            ]
            if high:
                uses.append("high in " + ", ".join(high))
        else:
            stepped = [
                period.period_id
                for period in sequence.periods
                if any(step.port == port.key for step in period.analog_steps)
            ]
            if stepped:
                uses.append("stepped in " + ", ".join(stepped))
        if any(delay.port == port.key for delay in sequence.delays):
            uses.append("delayed")
        bound = [
            field_label(sequence, binding.field_ref)
            for binding in bindings
            if binding.field_ref.port == port.key
        ]
        if bound:
            uses.append("bound as " + ", ".join(bound))
        if uses:
            blocked.append(f"{port.label or port.key} is {'; '.join(uses)}")
    if blocked:
        raise ValueError(
            "clear these outputs before removing them: " + " | ".join(blocked)
        )
    periods = []
    for period in sequence.periods:
        states = [0] * len(target.raw_lanes)
        for key, port in kept.items():
            if port.kind == "digital":
                states[new_index[port.lanes[0]]] = period.states[
                    old_index[old.by_key[key].lanes[0]]
                ]
        periods.append(replace(
            period,
            states=tuple(states),
            analog_steps=tuple(
                step for step in period.analog_steps if step.port in kept
            ),
        ))
    return {
        "periods": tuple(periods),
        "delays": tuple(delay for delay in sequence.delays if delay.port in kept),
    }


def _scan_table_of(source: str) -> object:
    """What one scan program leaves in ``scan_table``, or None; any thread."""

    namespace: dict = {}
    exec(compile(source, "<scan program>", "exec"), namespace)  # noqa: S102
    return namespace.get("scan_table")


def _unique_id(existing: Sequence[str], stem: str) -> str:
    """The first free ``<stem><n>``, counting from one.

    A spacer's name is read off its card (it cannot be typed), so the first
    spacer is "spacer1" -- not "spacer8" because seven periods came first.
    """

    taken = set(existing)
    ordinal = 1
    while f"{stem}{ordinal}" in taken:
        ordinal += 1
    return f"{stem}{ordinal}"
