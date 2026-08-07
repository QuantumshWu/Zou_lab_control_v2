"""Editing a pulse sequence in a window, and seeing what it will do.

Three packages had everything except the thing between them: zlc_pulse owns an
immutable ``PulseSequence`` and what makes one legal, zlc_ui owns a four-page
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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from zlc_pulse import (
    MINIMUM_REPEAT_COUNT,
    sequence_from_tree,
    sequence_to_tree,
    AnalogStep,
    OutputDelay,
    PulsePeriod,
    PulseSequence,
    RepeatRegion,
)
from zlc_pulse import TIME_UNIT_CHOICES, TIME_UNIT_TO_NS
from zlc_durable import unique_path
from zlc_plot import PANEL_SIZE_NAMES
from zlc_ui import (
    VALIDATOR_FLOAT,
    VALIDATOR_INT,
    DelayRowVM,
    ScanPageRecord,
    TargetWidthRule,
    FieldVM,
    PeriodVM,
    PortRowVM,
    RepeatVM,
    ScheduleVM,
)


__all__ = [
    "PulseEditorPresenter",
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
_TIME_UNITS = tuple(
    sorted(TIME_UNIT_CHOICES, key=lambda unit: TIME_UNIT_TO_NS[unit])
)


def _nanoseconds(value: float, unit: str) -> float:
    return float(value) * TIME_UNIT_TO_NS[unit]


def _readable(nanoseconds: float) -> str:
    """One duration in whichever unit reads without exponents."""

    for unit in reversed(_TIME_UNITS):
        scale = TIME_UNIT_TO_NS[unit]
        if nanoseconds >= scale:
            return f"{nanoseconds / scale:g} {unit}"
    return f"{nanoseconds:g} ns"


# ------------------------------------------------------------------ projection


#: What an editor holding no sequence shows.  Not an error state: an editor
#: opens before it has a subject, and its job then is to say how to get one.
#: How a preview writes "this plays until Stop".  One spelling, because the
#: outer marker and the summary line must not disagree about it.
FOREVER_NOTATION = "x∞"
#: What a period with NO step for a DAC means: the output keeps whatever the
#: period before it left there.  It is a reading of the model, not a mode the
#: model has -- ANALOG_MODES is edge and ramp -- and the one place both the
#: projection and the commit have to agree on the word.
HOLD_MODE = "hold"


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
    still shown, because an unexplained lane is worse than an odd row.
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
    to find on the board.
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
                lane_order=tuple(range(len(port.lanes))),
            )
        )
    return tuple(records)


def _scan_table_text(rows: Sequence[Sequence[int]], columns: Sequence[object]) -> str:
    """The scan table as a person reads it: a header, then the first rows.

    Truncated on purpose.  A thousand-point scan is not read line by line, and
    a page that pastes all of it hides the shape it was supposed to show.
    """

    if not rows:
        return "(no scan table yet -- write a program and press Run)"
    header = "  ".join(f"{getattr(column, 'name', index):>12}" for index, column in enumerate(columns))
    lines = [header] if columns else []
    shown = list(rows[:40])
    lines.extend("  ".join(f"{value:>12d}" for value in row) for row in shown)
    if len(rows) > len(shown):
        lines.append(f"... {len(rows) - len(shown)} more point(s)")
    return "\n".join(lines)


def project_period(
    sequence: PulseSequence,
    period: PulsePeriod,
    *,
    visible_ports: Sequence[str] | None = None,
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
    bindings = bindings_of(sequence)
    duration_binding, duration_number = _binding_for(
        bindings, "duration", period.period_id
    )
    return PeriodVM(
        period_id=period.period_id,
        name=period.name or period.period_id,
        duration=FieldVM(
            text=f"{period.duration:g}",
            binding_kind=duration_binding or "",
            binding_number=duration_number or 0,
            validator_kind=VALIDATOR_FLOAT,
            validator_lo=sequence.time_step_ns,
            validator_hi=0.0,
            resolution=sequence.time_step_ns,
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
            (port.key, _analog_mode(period, port), _analog_field(sequence, period, port))
            for port in offered
            if port.kind == "dac" and (shown is None or port.key in shown)
        ),
    )


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
    periods = tuple(
        project_period(sequence, period, visible_ports=visible_ports)
        for period in (() if sequence is None else sequence.periods)
    )

    total_ns = sum(
        _nanoseconds(period.duration, period.unit)
        for period in (() if sequence is None else sequence.periods)
    )
    repeat = None if sequence is None else sequence.repeat
    slots = () if sequence is None else sequence.slots
    return ScheduleVM(
        document_generation=int(generation),
        revision=int(revision),
        document_name=sequence.name if sequence is not None else "(no pulse)",
        clock_text=f"{step_ns:g} ns/tick",
        total_text=_readable(total_ns) if sequence is not None else "",
        total_tooltip=(
            f"{total_ns:g} ns over {len(periods)} period(s)"
            if sequence is not None
            else ""
        ),
        period_count=len(periods),
        visible_text=f"{sum(1 for port in ports if port.visible)}/{len(ports)} ports",
        summary_text=(
            (Path(path).name if path else sequence.name)
            if sequence is not None
            else "Add Period to start a pulse on this board"
        ),
        ports=ports,
        periods=periods,
        repeat=(
            None
            if repeat is None
            else RepeatVM(repeat.start_period_id, repeat.end_period_id, repeat.count)
        ),
        # Every output the board can delay gets a row whether or not a pulse is
        # open, because the row IS the board telling the operator that output
        # can be delayed.  With no pulse the value is the zero a missing delay
        # means, and it cannot be edited: a delay belongs to the pulse that
        # carries it, and there is not one yet to write into.
        delay_rows=tuple(
            DelayRowVM(
                port_key=port.key,
                value=FieldVM(
                    text=(
                        f"{_delay_of(sequence, port.key):g}"
                        if sequence is not None
                        else "0"
                    ),
                    binding_kind=_binding_for(
                        bindings_of(sequence), "delay", None, port.key
                    )[0] or "",
                    binding_number=_binding_for(
                        bindings_of(sequence), "delay", None, port.key
                    )[1] or 0,
                    editable=sequence is not None,
                    allow_any=False,
                ),
                unit="ns",
                unit_quantums=tuple((unit, TIME_UNIT_TO_NS[unit]) for unit in _TIME_UNITS),
            )
            for port in programmable_ports(target)
            if port.kind in ("digital", "dac")
        ),
        scan_summary_text=(
            f"{len(slots)} scan slot(s)" if slots else "no scan slots"
        ),
        scan_source_loaded=bool(slots),
        scan_file_path=str(path),
        # From the domain, which is what decides that once is not a repeat.
        # The view model carried 1 for both, so Add Bracket committed a
        # count-1 region -- which set_repeat reads as "no bracket" and the
        # button silently undid itself.  zlc_ui may not import zlc_pulse, so
        # the number is carried across by whoever knows both.
        min_repeat_count=MINIMUM_REPEAT_COUNT,
        default_repeat_count=MINIMUM_REPEAT_COUNT,
    )


def bindings_of(sequence: PulseSequence | None) -> dict[tuple, tuple[str, int]]:
    """Every bound field, as (kind, number) keyed by what it refers to.

    A dot is how a field becomes a scan column, and the widget that draws it
    has always been able to show which: FluentScanLineEdit takes a binding and
    a number and paints an orange s0 or a violet API mark.  Nothing ever told
    it.  The projection built every field as a bare value, so a duration bound
    to slot 0 looked exactly like one that was not bound at all -- the bind
    took, and the screen never said so.

    The number is the slot's position on the sequence, which is the column an
    operator will find it in; api slots share the counting because they share
    the mechanism -- an api slot IS a scan slot the host writes one row into.
    """

    if sequence is None:
        return {}
    found: dict[tuple, tuple[str, int]] = {}
    for index, slot in enumerate(sequence.slots):
        reference = slot.field_ref
        kind = "api" if str(slot.slot_id).startswith("api_") else "scan"
        found[(reference.kind, reference.period_id, reference.port)] = (kind, index + 1)
    return found


def _binding_for(
    bindings: Mapping[tuple, tuple[str, int]],
    kind: str,
    period_id: str | None = None,
    port: str | None = None,
) -> tuple[str | None, int | None]:
    found = bindings.get((kind, period_id, port))
    return found if found is not None else (None, None)


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


def _analog_field(sequence: PulseSequence, period: PulsePeriod, port: Any) -> FieldVM:
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
    binding, number = _binding_for(
        bindings_of(sequence), "dac", period.period_id, port.key
    )
    return FieldVM(
        text=str(value),
        binding_kind=binding or "",
        binding_number=number or 0,
        # A bound field's value comes from the scan table, not from typing.
        editable=step is not None and binding is None,
        validator_kind=VALIDATOR_INT,
        validator_lo=float(low),
        validator_hi=float(high),
        allow_any=True,
    )


def _delay_of(sequence: PulseSequence, port_key: str) -> float:
    delay = next((item for item in sequence.delays if item.port == port_key), None)
    return 0.0 if delay is None else _nanoseconds(delay.value, delay.unit)


def timeline_of(sequence: PulseSequence, *, include_off: bool = False) -> Any:
    """The sequence as a drawable timeline.

    Built in seconds because that is what the plot's time axis is in, and from
    the same period durations the hardware will play -- a preview computed any
    other way is a drawing of something else.
    """

    from zlc_plot import (
        PulseAnalogTrace,
        PulseBlock,
        PulseChannel,
        PulseRepeatMarker,
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

    traces: list[Any] = []
    for port in target.ports:
        if port.kind != "dac":
            continue
        low, high = port.signed_range
        values: list[float] = []
        held = 0.0
        for period in sequence.periods:
            step = next((item for item in period.analog_steps if item.port == port.key), None)
            if step is not None:
                held = float(step.value)
            values.append(held)
        if not any(values) and not include_off:
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
                starts=tuple(starts) + (total,),
                values=tuple(values),
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

    # What the pulse actually plays, bracket or no bracket.
    #
    # The OUTER level of a pulse repeats forever -- On Pulse is a cycle an
    # experiment holds running -- unless a bracket spans the whole thing and
    # says how many times instead.  A bracket over PART of the pulse repeats
    # that part, and the outer level is still forever around it.  Neither was
    # drawn: a pulse with no bracket showed nothing, so the one thing that is
    # always true of a running pulse was the one thing the preview never said.
    markers: list[Any] = []
    ids = [period.period_id for period in sequence.periods]
    # Asked, not re-derived: the document decides what its own bracket means.
    spans_everything = sequence.whole_pulse_repeat is not None
    if sequence.repeat is not None:
        try:
            first = ids.index(sequence.repeat.start_period_id)
            last = ids.index(sequence.repeat.end_period_id)
        except ValueError:
            first = last = None
        if first is not None and last is not None:
            stop = starts[last] + _nanoseconds(
                sequence.periods[last].duration, sequence.periods[last].unit
            ) * 1e-9
            # A LABEL, which is what the marker takes.  Passing the count
            # itself raised TypeError inside the primitive, so a bracketed
            # pulse could not be previewed at all.
            markers.append(
                PulseRepeatMarker(starts[first], stop, f"x{sequence.repeat.count}")
            )
    if not spans_everything and total > 0:
        markers.append(PulseRepeatMarker(0.0, total, FOREVER_NOTATION))

    return PulseTimelineData(
        channels=tuple(channels),
        blocks=tuple(blocks),
        time_unit="s",
        total_duration=total,
        analog_traces=tuple(traces),
        repeat_markers=tuple(markers),
        repeat_notation=(
            f"x{sequence.repeat.count}"
            if spans_everything
            else FOREVER_NOTATION
        ),
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
    forever: bool = False
    loaded: bool = False
    #: The board is holding exactly the program the editor is showing.
    holds_shown: bool = False
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
        sequence: PulseSequence | None = None,
        *,
        make_preview: Callable[[Any], Any] | None = None,
        update_preview: Callable[..., Any] | None = None,
        sequencer: Any = None,
        dial: Callable[[str, str], Any] | None = None,
        pulses_directory: str = "",
        path: str = "",
        default_endpoint: str = "",
    ) -> None:
        self.view = view
        self.sequence = sequence
        self.original = sequence
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
        #: The plotting host behind the preview.  Built once and updated,
        #: and the thing a save actually goes through.
        self._preview_host: Any = None
        #: A size the operator picked, which sticks until the content changes
        #: shape; None means the content chooses.
        self._pinned_size: str | None = None
        self._preview_selectors = False
        #: The scan program, the table it produced, and where the board is in it.
        self._scan_source = ""
        self._scan_rows: tuple[tuple[int, ...], ...] = ()
        self._scan_shape = None
        self._scan_dirty = True
        self._scan_repeats = 1
        self._scan_use_loaded = False
        self._scan_path = ""
        self._scan_progress = ""
        self._held_point: int | None = None
        # An editor with a sequencer can fire what it is showing; without one
        # it is a viewer, and says so rather than offering a dead button.
        self.sequencer = sequencer
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
        self.connection = (
            ("offline", str(default_endpoint))
            if sequencer is None
            else ("given", str(default_endpoint))
        )
        self._owns_sequencer = False
        self._connection_status = "connected" if sequencer is not None else "not connected"
        self._visible: set[str] | None = None
        #: The last model handed to the schedule page, so a new one can be
        #: recognised as new without every caller remembering to say so.
        self._shown: ScheduleVM | None = None
        self._connect()
        self.refresh()

    # --------------------------------------------------------------- wiring

    def _connect(self) -> None:
        view = self.view
        view.document_name_committed.connect(self.set_document_name)
        view.port_label_committed.connect(self.set_port_label)
        view.period_name_committed.connect(self.set_period_name)
        view.duration_committed.connect(self.set_duration)
        view.digital_committed.connect(self.set_digital)
        view.analog_committed.connect(self.set_analog)
        view.delay_committed.connect(self.set_delay)
        view.insert_period_requested.connect(self.insert_period)
        view.move_period_requested.connect(self.move_period)
        view.remove_period_requested.connect(self.remove_period)
        view.repeat_committed.connect(self.set_repeat)
        view.visible_ports_committed.connect(self.set_visible_ports)
        view.clear_port_requested.connect(self.clear_port)
        # From the button that raises it.  The schedule page also declared a
        # clear_all_requested that nothing ever emitted, and this listened to
        # that one -- so the toolbar's Clear All did nothing at all.
        view.clear_all_requested.connect(self.clear_all)
        view.binding_cycle_requested.connect(self.cycle_binding)
        view.scan_array_load_requested.connect(self.load_scan_array)
        view.scan_source_committed.connect(self.set_scan_source)
        view.feedback_requested.connect(self._warn)
        view.scan_repeats_committed.connect(self.set_scan_repeats)
        view.scan_hold_requested.connect(self.hold_scan_point)
        view.scan_step_requested.connect(self.step_scan_point)
        view.scan_program_load_requested.connect(self.load_scan_program)
        view.scan_template_requested.connect(self.write_scan_template)
        view.scan_run_requested.connect(self.run_scan_program)
        view.scan_array_save_requested.connect(self.save_scan_array)
        view.scan_progress_refresh_requested.connect(self.refresh_scan_progress)
        view.connection_requested.connect(self.connect_to)
        view.fire_requested.connect(self.fire)
        view.stop_requested.connect(self.stop)
        view.sync_requested.connect(self.sync_from_sequencer)
        view.save_requested.connect(self.save_pulse)
        view.load_requested.connect(self.ask_for_pulse)
        view.preview_include_off_toggled.connect(self._on_include_off)
        view.preview_size_committed.connect(self.set_preview_size)
        view.preview_selectors_toggled.connect(self.set_preview_selectors)
        view.preview_save_requested.connect(self.save_preview_image)
        view.target_apply_requested.connect(self.apply_target)

    # ------------------------------------------------------------- the pulse

    def ask_for_pulse(self) -> bool:
        """Let the operator pick a pulse file, and open it.

        The window owns the dialog; this owns what to do with the answer.  A
        pulse is a Python module that builds a sequence, so opening one means
        running its build() -- the same reader the session uses, so the editor
        cannot show a pulse assembled differently from the one that will fire.
        """

        start = str(Path(self.path).parent if self.path else self.pulses_directory or "")
        chosen = self.view.ask_open_path(
            "Open pulse", start,
            "Pulses (*.json *.py);;ZLC pulse (*.json);;Pulse modules (*.py);;All files (*)",
        )
        if not chosen:
            return False
        return self.open_pulse(chosen)

    def open_pulse(self, path: str) -> bool:
        """Replace what is being edited with the pulse in one file.

        Two kinds of file, because a pulse has two homes.  A ``.py`` is an
        author's module and is opened by RUNNING its build(), through the same
        reader the session uses, so the editor can never show a pulse assembled
        differently from the one that will fire.  A ``.json`` is what this
        editor itself writes, and is read straight back into the model.
        """

        from .session import read_pulse

        session: dict[str, Any] = {}
        try:
            if Path(path).suffix.lower() == ".json":
                tree = json.loads(Path(path).read_text(encoding="utf-8"))
                sequence = sequence_from_tree(tree)
                session = dict(tree.get("editor") or {})
            else:
                _program, metadata = read_pulse(path)
                sequence = metadata.get("sequence")
            if sequence is None:
                raise ValueError(
                    "this pulse builds a program but does not publish its "
                    "'sequence', so there is nothing to edit"
                )
        except Exception as error:
            self._warn(f"cannot open {Path(path).name}: {error}")
            return False
        self.sequence = self.original = sequence
        self.path = str(path)
        self.revision += 1
        self._restore_session(session)
        self.refresh()
        self._done(
            f"opened {Path(path).name} - {len(sequence.periods)} period(s), "
            f"{len(sequence.slots)} scan slot(s)"
        )
        return True

    def _restore_session(self, session: Mapping[str, Any]) -> None:
        """Put the editing session back: which rows, which scan, which source.

        A pulse file that reopens showing every port and an empty Scan page has
        kept the pulse and lost the work around it, which is what the operator
        was actually doing.
        """

        visible = session.get("visible_ports")
        self._visible = None if visible is None else {str(key) for key in visible}
        self._scan_source = str(session.get("scan_source", "") or "")
        self._scan_rows = tuple(
            tuple(int(value) for value in row) for row in session.get("scan_rows", ())
        )
        self._scan_use_loaded = bool(session.get("scan_use_loaded", False))
        self._scan_repeats = int(session.get("scan_repeats", 1) or 1)
        self._scan_dirty = not self._scan_source

    def start_new_pulse(self) -> bool:
        """Begin a pulse on the board this bench actually has.

        A sequence needs a target and at least one period before it is a legal
        sequence at all, so "new" cannot mean empty.  The target comes from the
        deployed board's own pin map rather than from an invented default: a
        pulse authored against an imaginary board compiles and then does
        nothing recognisable.
        """

        from zlc_pulse import PulsePeriod, PulseSequence, pulse_target_from_xdc

        try:
            # The attached board first: a pulse authored against this machine's
            # files while a different board is connected compiles and then does
            # nothing recognisable.
            target = self._board_target or pulse_target_from_xdc()
            step_ns = self._board_step_ns or 20.0
            self.sequence = self.original = PulseSequence(
                name="untitled",
                target=target,
                time_step_ns=step_ns,
                periods=(
                    PulsePeriod("period1", step_ns, "ns", (0,) * len(target.raw_lanes)),
                ),
            )
        except Exception as error:
            self._warn(f"cannot start a pulse for this board: {error}")
            return False
        self.path = ""
        self.revision += 1
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
        self.sequence = candidate
        self.revision += 1
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
        self.sequence = candidate
        self.revision += 1
        self.view.set_port_label(str(key), str(label))
        self._refresh_summary()
        self.refresh_target()
        self.refresh_preview()

    def set_period_name(self, period_id: str, name: str) -> None:
        self._edit_period(period_id, lambda period: replace(period, name=str(name)))

    def set_duration(self, period_id: str, value: object, unit: str) -> None:
        self._edit_period(
            period_id,
            lambda period: replace(period, duration=float(value), unit=str(unit)),
        )

    def set_digital(self, period_id: str, port_key: str, high: bool) -> None:
        port = self.sequence.target.by_key.get(str(port_key))
        if port is None:
            self._warn(f"{port_key} is not a port on this target")
            return
        index = self.sequence.target.raw_lanes.index(port.lanes[0])

        def _edit(period: PulsePeriod) -> PulsePeriod:
            states = list(period.states)
            states[index] = 1 if high else 0
            return replace(period, states=tuple(states))

        self._edit_period(period_id, _edit)

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

        def _edit(period: PulsePeriod) -> PulsePeriod:
            steps = tuple(step for step in period.analog_steps if step.port != port_key)
            text = "" if value is None else str(value).strip()
            if text and str(mode or "edge") != HOLD_MODE:
                steps = steps + (
                    AnalogStep(str(port_key), str(mode or "edge"), int(float(text))),
                )
            return replace(period, analog_steps=steps)

        # A level holds until something sets it again, so this changes what
        # every later card displays as well as this one.
        self._edit_period(period_id, _edit, ripples_forward=True)

    def set_delay(self, port_key: str, value: object, unit: str) -> None:
        if self.sequence is None:
            return
        delays = tuple(item for item in self.sequence.delays if item.port != port_key)
        amount = float(value or 0.0)
        if amount:
            delays = delays + (OutputDelay(str(port_key), amount, str(unit)),)
        self._apply_value(
            self._rebuilt(delays=delays), port_key=str(port_key)
        )

    def insert_period(self, before_period_id: object) -> None:
        """Add one period, copying the state its neighbour was already in.

        With no pulse open this is how one starts: the first period is what
        makes a sequence legal, so asking for a period IS asking for a pulse.

        A new period that resets every lane to zero silently inserts a gap in
        the middle of a sequence; copying the neighbour makes the insertion
        visible as a longer hold, which is what the operator can then edit.
        """

        if self.sequence is None:
            self.start_new_pulse()
            return
        periods = list(self.sequence.periods)
        ids = [period.period_id for period in periods]
        position = ids.index(str(before_period_id)) if before_period_id in ids else len(periods)
        model = periods[max(0, position - 1)] if periods else None
        new_id = _unique_id(ids, "period")
        periods.insert(
            position,
            PulsePeriod(
                period_id=new_id,
                duration=model.duration if model else self.sequence.time_step_ns,
                unit=model.unit if model else "ns",
                states=model.states if model else (0,) * len(self.sequence.target.raw_lanes),
                analog_steps=(),
                name="",
            ),
        )
        self._apply(self._rebuilt(periods=tuple(periods)))

    def move_period(self, period_id: str, before_period_id: object) -> None:
        periods = list(self.sequence.periods)
        ids = [period.period_id for period in periods]
        if period_id not in ids:
            return
        moving = periods.pop(ids.index(period_id))
        ids = [period.period_id for period in periods]
        position = ids.index(str(before_period_id)) if before_period_id in ids else len(periods)
        periods.insert(position, moving)
        self._apply(self._rebuilt(periods=tuple(periods)))

    def remove_period(self, period_id: str) -> None:
        periods = tuple(
            period for period in self.sequence.periods if period.period_id != period_id
        )
        if not periods:
            self._warn("a sequence needs at least one period")
            return
        repeat = self.sequence.repeat
        if repeat is not None and period_id in (repeat.start_period_id, repeat.end_period_id):
            repeat = None
        self._apply(self._rebuilt(periods=periods, repeat=repeat))

    def set_repeat(self, start: object, end: object, count: int) -> None:
        """Bracket these periods, or clear the bracket.

        No start, no end, or a count below the domain's minimum all mean the
        same thing: there is no repeat region.
        """

        if start is None or end is None or int(count) < MINIMUM_REPEAT_COUNT:
            self._apply(self._rebuilt(repeat=None))
            return
        self._apply(
            self._rebuilt(repeat=RepeatRegion(str(start), str(end), int(count)))
        )

    def set_visible_ports(self, ports: object) -> None:
        """Which ports have rows.  A VALUE change, so it goes the value way.

        Which outputs are worth looking at is not the shape of the pulse, and
        the page has a setter that says exactly that -- declared in
        docs/pulse-views.md and never called, so every Hide Off went through a
        whole re-projection instead.  The model this presenter believes is on
        screen is kept in step, or the next full push would not recognise
        itself as different.
        """

        self._visible = None if ports is None else {str(key) for key in ports}
        if self._shown is None or self.sequence is None:
            self.refresh()
            return
        keys = tuple(sorted(self._visible)) if self._visible is not None else tuple(
            port.key for port in self._shown.ports
        )
        self.view.set_visible_ports(keys)
        shown = set(keys)
        self._shown = replace(
            self._shown,
            ports=tuple(
                replace(port, visible=port.key in shown) for port in self._shown.ports
            ),
            visible_text=f"{len(shown)}/{len(self._shown.ports)}",
        )

    def clear_port(self, port_key: str) -> None:
        """Return one port to rest everywhere, without touching the others."""

        port = self.sequence.target.by_key.get(str(port_key))
        if port is None:
            self._warn(f"{port_key} is not a port on this target")
            return
        index = self.sequence.target.raw_lanes.index(port.lanes[0])
        periods = []
        for period in self.sequence.periods:
            states = list(period.states)
            if port.kind == "digital":
                states[index] = 0
            periods.append(
                replace(
                    period,
                    states=tuple(states),
                    analog_steps=tuple(
                        step for step in period.analog_steps if step.port != port_key
                    ),
                )
            )
        delays = tuple(item for item in self.sequence.delays if item.port != port_key)
        self._apply(
            self._rebuilt(periods=tuple(periods), delays=delays)
        )

    def clear_all(self) -> None:
        """Back to the pulse as it was opened, not to an empty one.

        "Clear" in an editor over a file means undo my edits; an empty sequence
        would discard the target and the file with it.
        """

        self._apply(self.original)

    # ------------------------------------------------------------- hardware

    def compile(self) -> Any:
        """The sequence as the board would receive it.


        Compiled against the DEPLOYED board's geometry, not against defaults:
        a preview that compiles for an imaginary board is exactly the kind of
        confirmation that survives until the bench proves it wrong.
        """

        from zlc_pulse import compile_sequence, load_streamer_config

        if self.sequence is None:
            raise RuntimeError("no pulse is open, so there is nothing to compile")
        config = load_streamer_config()
        if config["source"] is None:
            raise RuntimeError(
                "no streamer config was found, so the deployed board geometry is "
                "unknown; refusing to compile against built-in defaults"
            )
        return compile_sequence(self.sequence, config["params"], config["clock_hz"])

    def connect_to(self, mode: str, endpoint: str) -> bool:
        """Attach this editor to a sequencer, or say why it could not.

        The previous connection is released first.  Two open connections to one
        board is not a state anything downstream can reason about, and the
        second one usually fails in a way that reads as the first one breaking.
        """

        if self._dial is None:
            self._warn("this editor was built without a way to connect")
            return False
        self._release()
        mode, endpoint = str(mode), str(endpoint)
        if mode == "offline":
            self.connection = (mode, endpoint)
            self._show_connection("edit only")
            # Hanging up changes the whole window, not just its status line:
            # the board's ports are gone, the Target page is authorable again,
            # and the run buttons have nothing to fire on.
            self.refresh()
            return True
        try:
            self.sequencer = self._dial(mode, endpoint)
        except Exception as error:
            self.connection = ("offline", endpoint)
            self._show_connection(f"failed: {error}")
            self._warn(f"cannot connect to {endpoint or mode}: {error}")
            return False
        self._owns_sequencer = True
        self.connection = (mode, endpoint)
        self.adopt_board()
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

        describe = getattr(self.sequencer, "describe", None)
        if not callable(describe):
            self._show_connection("connected (board did not describe itself)")
            return False
        try:
            board = describe()
        except Exception as error:
            self._show_connection(f"connected, but cannot read the board: {error}")
            self._warn(f"connected, but the board would not describe itself: {error}")
            return False

        self.board = board
        # From the target that owns them.  A description carried the pin map
        # twice and the wire emptied the nested copy, so reading the top-level
        # one was reading whichever copy happened to survive.
        self.pins = dict(getattr(board.target, "package_pins", {}) or {})
        # Adopting a board changes what the editor is showing, so it is a new
        # revision.  The view refuses two different models under one revision,
        # and it is right to: a stale card would otherwise be indistinguishable
        # from a fresh one.
        self.revision += 1
        if self.sequence is None:
            self._apply_board_only(board)
        else:
            self._align_sequence_to(board)
        mode, endpoint = self.connection
        where = endpoint or mode
        self._show_connection(
            f"{where} - {len(board.target.ports)} ports, "
            f"{len(board.target.raw_lanes)} lanes, "
            f"{board.clock_hz / 1e6:g} MHz"
        )
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
        """

        from zlc_pulse import PulsePeriod

        current = self.sequence
        board_lanes = board.target.raw_lanes
        was = {lane: index for index, lane in enumerate(current.target.raw_lanes)}
        lost = sorted(
            lane
            for index, lane in enumerate(current.target.raw_lanes)
            if lane not in set(board_lanes)
            and any(period.states[index] for period in current.periods)
        )
        ports = {port.key for port in board.target.ports}
        dropped_steps = sorted(
            {
                step.port
                for period in current.periods
                for step in period.analog_steps
                if step.port not in ports
            }
        )

        periods = tuple(
            PulsePeriod(
                period_id=period.period_id,
                duration=period.duration,
                unit=period.unit,
                states=tuple(
                    period.states[was[lane]] if lane in was else 0
                    for lane in board_lanes
                ),
                analog_steps=tuple(
                    step for step in period.analog_steps if step.port in ports
                ),
                name=period.name,
            )
            for period in current.periods
        )
        delays = tuple(delay for delay in current.delays if delay.port in ports)
        candidate = PulseSequence(
            name=current.name,
            target=board.target,
            time_step_ns=float(board.time_step_ns),
            periods=periods,
            slots=(),
            delays=delays,
            repeat=current.repeat,
        )
        self.sequence = candidate
        self.original = candidate
        if lost or dropped_steps:
            missing = ", ".join(lost + dropped_steps)
            self._warn(
                f"this board has no {missing}; those outputs were dropped from "
                "the pulse rather than driven blind"
            )

    def _release(self) -> None:
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

        if not self._owns_sequencer:
            return
        if self.sequencer is not None:
            close = getattr(self.sequencer, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # a board that will not hang up is not fatal
                    pass
        self.sequencer = None
        self._owns_sequencer = False
        self.board = None
        self.pins = {}
        self._board_target = None
        self._board_step_ns = None
        self._board_state = BoardState()
        # Losing a board changes what the editor is showing, so it is a new
        # revision -- the same rule adopting one obeys.  The view refuses two
        # different models under one revision, and it is right to.
        self.revision += 1

    def _show_connection(self, status: str) -> None:
        """Say what the editor is attached to, and what that makes possible.

        Running needs both a pulse to fire and a board to fire it on; either
        one missing is a button that cannot work, so it is shown as one.
        """

        mode, endpoint = self.connection
        self._connection_status = str(status)
        self.view.set_connection(mode, endpoint, status)
        self._show_run_state()

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
        window's idea of it.
        """

        if self.sequencer is None:
            self._warn("this editor is not connected to a sequencer")
            return False
        applied = getattr(self.sequencer, "applied", None)
        state = applied() if callable(applied) else applied
        if state is None:
            self._warn("the board has no pulse applied yet; there is nothing to sync")
            return False
        source = getattr(state, "source", None)
        if source is None:
            self._warn(
                "the board is holding a compiled program it was given without "
                "its pulse, so there is nothing an editor can show"
            )
            return False
        self.sequence = self.original = source
        self.revision += 1
        rows = tuple(getattr(state, "scan_rows", ()) or ())
        if rows:
            self._scan_rows = rows
        self.refresh()
        self._done(
            f"synced from the board - {len(source.periods)} period(s)"
            + (f", {len(rows)} scan point(s)" if rows else "")
        )
        return True

    def load_into_sequencer(self) -> bool:
        """Put what is on screen onto the board, without firing it."""

        if self.sequence is None:
            self._warn("no pulse is open")
            return False
        if self.sequencer is None:
            self._warn("this editor is not connected to a sequencer")
            return False
        try:
            # The source goes with it.  A board that was handed a program and
            # not the document it came from can say what it holds but not what
            # it means, and reading a saved run then depends on whoever was at
            # the keyboard.
            self.sequencer.load(self.compile(), source=self.sequence)
            if self._scan_rows and self.sequence.slots:
                self.sequencer.write_scan_table(self._scan_rows)
        except Exception as error:
            self._warn(f"cannot load this pulse: {error}")
            self._show_run_state()
            return False
        self._show_run_state()
        return True

    def fire(self, *, forever: bool | None = None, shots: int = 1, timeout: float = 5.0) -> bool:
        """On Pulse: load what is on screen and run it the way the pulse says.

        A pulse is a cycle, and an experiment holds it running -- the MOT loads
        for as long as it is on.  Firing once and stopping is a diagnostic, not
        the normal thing an operator means by On Pulse, so forever is the
        default.

        Unless the pulse says otherwise, and it can: a bracket spanning the
        whole pulse means "play it this many times", and the count reaches the
        board inside the compiled program as its one loop region.  Wrapping
        that in a forever run is what made the outermost bracket look like it
        did nothing -- N times, over and over, is indistinguishable from over
        and over.  So the document is asked, here, on the path that fires.

        A forever run is deliberately NOT waited on.  It never reports done, so
        waiting would hang the window on its own success.
        """

        if not self.load_into_sequencer():
            return False
        if forever is None:
            forever = self.sequence.whole_pulse_repeat is None
        try:
            if forever:
                self.sequencer.fire(forever=True)
                self._show_run_state()
                return True
            for shot in range(max(1, int(shots))):
                self.sequencer.fire()
                # A shot that does not report done is not a shot that
                # succeeded.  Firing again over an unfinished one is the
                # failure this project has already paid for twice: the board
                # takes the second command while the first is still playing.
                if self.sequencer.wait_done(timeout) is None:
                    self._warn(
                        f"shot {shot + 1} did not report done within {timeout:g}s; "
                        "stopping rather than firing over it"
                    )
                    self.stop()
                    return False
        except Exception as error:
            self._warn(f"firing stopped: {error}")
            self.stop()
            return False
        return True

    def stop(self) -> None:
        """Return the outputs to their safe state, whatever state they are in.

        The only way out of a forever run, and it must work from any state --
        an operator pressing Stop is not required to know what the board is
        doing first.
        """

        # Safe first, then look.  Recording "stopped" before the board has
        # stopped is how a window comes to claim idle over driving outputs: if
        # safe() refuses, the board is still playing and must still read as
        # playing, with the refusal on screen next to it.
        try:
            if self.sequencer is not None:
                safe = getattr(self.sequencer, "safe", None)
                if callable(safe):
                    safe()
        except Exception as error:
            self._warn(f"the board did not go safe: {error}")
        finally:
            self._show_run_state()

    @property
    def running(self) -> bool:
        """Is the board playing?  Its answer, from the last time it was asked."""

        return self._board_state.firing

    @property
    def synchronized(self) -> bool:
        """Is the board holding what this editor shows?  Also its answer."""

        return self._board_state.holds_shown

    def board_state(self) -> BoardState:
        """Ask the board what it is doing, in one round trip.

        One call, not two: everything here comes out of ``snapshot``, which the
        board keeps as cheap level state precisely so it can be asked often.
        Pulling the applied program back instead would ship a whole compiled
        pulse across the wire to answer a yes-or-no question.
        """

        if self.sequencer is None:
            return BoardState()
        try:
            reported = dict(self.sequencer.snapshot())
        except Exception as error:
            return BoardState(attached=True, answering=False, fault=str(error))
        return BoardState(
            attached=True,
            answering=True,
            firing=bool(reported.get("firing")),
            forever=bool(reported.get("forever")),
            loaded=bool(reported.get("loaded")),
            holds_shown=bool(
                self._shown_digest()
                and reported.get("applied_digest") == self._shown_digest()
            ),
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
                self._digest = self.compile().digest
            except Exception:
                # A pulse that does not compile is not one any board can be
                # holding, which is the honest answer to the question asked.
                self._digest = ""
            self._digest_revision = self.revision
        return self._digest

    def refresh_run_state(self) -> None:
        """Ask the board again and show the answer.

        The public name for it, because "what is the board doing" is a question
        anything may need re-asked -- a window on a timer, a notebook, a test
        standing in for the operator who walked back to the bench.
        """

        self._show_run_state()

    def _show_run_state(self) -> None:
        """Tell the window what is running, so Stop is the live control.

        The status dot is the same answer at a glance: green while the board is
        playing what the editor shows, orange while it is playing something
        older, and grey when nothing is attached.  One place decides it, so the
        dot and the buttons cannot disagree.
        """

        self._board_state = self.board_state()
        live = self.sequencer is not None and self.sequence is not None
        self.view.set_control_state(
            running=bool(self.running),
            synchronized=bool(self.synchronized),
            file_dirty=False,
            can_run=live,
        )
        # Capabilities go through the shell, which also gates the Scan page's
        # hold and step -- those need a board just as much as Sync does.
        self.view.set_capabilities(
            # Sync READS the board, so it needs a board and not a pulse -- an
            # editor with nothing open is exactly when pulling what the
            # hardware is holding is worth doing.
            can_sync=self.sequencer is not None,
            can_hold=live,
            can_step=live and bool(self._scan_rows),
        )
        self.view.set_status_color(self._status_token())

    def _status_token(self) -> str:
        state = self._board_state
        if not state.attached:
            return "idle"
        if not state.answering:
            # Attached and silent is its own state.  Showing the last good
            # answer instead is how a window sits green over a dead server.
            return "unreachable"
        if state.firing:
            return "running-synced" if state.holds_shown else "running-stale"
        return "dirty-ready" if self.sequence is not None else "idle"

    def save_pulse(self) -> str:
        """Write what is on screen, as a pulse file this editor can reopen.

        An editor that cannot keep an edit is not an editor.  This used to
        refuse outright, and the reason it gave -- a pulse is a Python module,
        do not overwrite the author's file with a generated one -- was true and
        answered a different question.  v1 saved a pulse as JSON beside the
        module, and that protection is kept: a .py is never written over,
        because the reasoning inside one cannot be regenerated from a table.
        """

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
        if target.suffix.lower() == ".py":
            self._warn(
                f"{target.name} is a pulse MODULE; save as .json rather than "
                "overwriting the reasoning inside it"
            )
            return ""
        # The pulse, plus the editing session around it.  v1 called its JSON
        # "the sole persisted source of the scan table", and a scan slot
        # without the table it indexes is half a scan; which ports are worth
        # looking at is the operator's, and reopening to all twenty-two rows
        # throws that away.  The pulse half is zlc_pulse's own tree; this half
        # is the editor's and is named so, so neither can be mistaken for the
        # other by whoever reads the file next.
        tree = dict(sequence_to_tree(self.sequence))
        tree["editor"] = {
            "visible_ports": (
                None if self._visible is None else sorted(self._visible)
            ),
            "scan_source": self._scan_source,
            "scan_rows": [list(row) for row in self._scan_rows],
            "scan_use_loaded": bool(self._scan_use_loaded),
            "scan_repeats": int(self._scan_repeats),
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(tree, indent=2) + chr(10), encoding="utf-8")
        except Exception as error:
            self._warn(f"cannot save {target.name}: {error}")
            return ""
        self.path = str(target)
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
        page opens up.
        """

        target = self._current_target()
        view = self.view
        if target is None:
            view.set_target_ports((), False, "No pulse and no board: nothing is wired yet.")
            return
        attached = self.board is not None
        if attached:
            width = int(self.board.bus_width)
            view.set_target_width_rules(
                TargetWidthRule(1, 1, 1), TargetWidthRule(2, width, width)
            )
        view.set_target_ports(
            project_target(target, pins=self.pins),
            not attached,
            (
                f"Wiring read from the attached board; rename freely, the "
                f"topology is the board's ({len(programmable_ports(target))} outputs)."
                if attached
                else "Offline: this target came from the pulse file and can be edited."
            ),
        )

    def _current_target(self) -> object | None:
        if self.sequence is not None:
            return self.sequence.target
        return self._board_target

    def apply_target(self, records: object) -> bool:
        """Take the edited names, and refuse anything that would re-wire a board.

        Renaming is display metadata and safe.  Changing lanes, widths or the
        set of ports while a board is attached would make the editor disagree
        with the hardware it is pointed at, and the disagreement would only
        surface as a pulse that fires the wrong outputs.
        """

        target = self._current_target()
        if target is None:
            self.view.set_target_feedback("there is no target to apply to")
            return False
        wanted = {str(record.key): str(record.signal).strip() for record in records}
        if self.board is not None:
            keys = {port.key for port in programmable_ports(target)}
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
        if self.sequence is not None:
            self._apply(self._rebuilt(target=renamed))
        else:
            self._board_target = renamed
            self.revision += 1
            self.refresh()
        self.view.set_target_feedback(
            f"renamed {sum(1 for key, name in wanted.items() if name)} output(s)"
        )
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

    def cycle_binding(self, field_kind: str, period_id: object, port_key: object) -> None:
        """Bind or unbind one field, cycling off -> scan -> api -> off.

        A dot is how a field becomes a scan column: bound, it stops being a
        constant in the pulse and becomes a value the device writes per point.
        The cycle order and which fields may be scanned belong to zlc_ui, which
        owns the control; what a binding MEANS is a slot on the sequence, which
        is this.
        """

        from zlc_pulse import PulseFieldRef, PulseSlot
        from zlc_ui import cycle_binding_kind

        if self.sequence is None:
            return
        kind = "delay" if str(field_kind) == "delay" else str(field_kind)
        reference = self._field_reference(kind, period_id, port_key)
        if reference is None:
            return
        current = next(
            (
                slot
                for slot in self.sequence.slots
                if slot.field_ref == reference
            ),
            None,
        )
        binding = None if current is None else self._binding_of(current)
        wanted = cycle_binding_kind(binding, field_kind=kind)
        slots = tuple(
            slot for slot in self.sequence.slots if slot.field_ref != reference
        )
        if wanted is not None:
            slots = slots + (
                PulseSlot(
                    reference.kind,
                    reference,
                    "value" if reference.kind == "dac" else "ns",
                    slot_id=self._slot_id(reference, wanted),
                ),
            )
        self._apply(self._rebuilt(slots=slots))
        self._refresh_scan_page()

    @staticmethod
    def _binding_of(slot: object) -> str:
        """Which cycle position a slot represents.

        An api slot is a scan slot the host writes one row into instead of
        streaming a table, so the two differ by name, not by mechanism.
        """

        return "api" if str(slot.slot_id).startswith("api_") else "scan"

    def _has_scan_slots(self) -> bool:
        """Whether anything is bound, and say what to do when nothing is.

        A scan table has one column per bound field, so with nothing bound
        there is no table to make or read -- and the operator's next move is a
        click on a dot in the Edit tab, which is what v1 told them.  Failing
        later, on a column count, names the symptom instead.
        """

        if self.sequence is not None and self.sequence.slots:
            return True
        self._warn(
            "bind at least one field to a scan slot first "
            "(click a dot in the Edit tab)"
        )
        return False

    @staticmethod
    def _slot_id(reference: object, binding: str) -> str:
        """A name for one bound field, built from ALL of what identifies it.

        The model names the parts: a duration is a period, a delay is a port,
        and a DAC is a period AND a port.  This took ``period_id or port`` --
        the first part that happened to be set -- so every DAC in one period
        got the same name, and binding a second one was refused with "slot ids
        must be unique".  One field per period could be scanned; every other
        dot on that card answered a rule the operator had not broken.
        """

        parts = [str(reference.kind)]
        parts += [
            str(part)
            for part in (reference.period_id, reference.port)
            if part is not None
        ]
        prefix = "api_" if binding == "api" else ""
        return prefix + "_".join(parts)

    def _field_reference(self, kind: str, period_id: object, port_key: object):
        from zlc_pulse import PulseFieldRef

        try:
            if kind == "duration":
                return PulseFieldRef("duration", period_id=str(period_id))
            if kind in ("analog", "dac"):
                return PulseFieldRef("dac", period_id=str(period_id), port=str(port_key))
            return PulseFieldRef("delay", port=str(port_key))
        except (TypeError, ValueError) as error:
            self._warn(f"cannot bind that field: {error}")
            return None

    # ------------------------------------------------------------- scan page

    def _refresh_scan_page(self) -> None:
        """Everything the Scan page shows, from one place.

        Its parts move together -- binding a field changes the columns, which
        makes the generated table stale, which changes what Run would upload --
        so they are projected together rather than nudged one at a time.
        """

        from zlc_pulse import scan_columns_for, validate_scan_table

        view = self.view
        if self.sequence is None:
            view.set_scan_page(ScanPageRecord(slots_text="No pulse is open."))
            self._show_run_state()
            return
        columns = scan_columns_for(self.sequence)
        if not self._scan_source:
            self._scan_source = self._template("column_stack")
        rows = self._scan_rows
        view.set_scan_page(
            ScanPageRecord(
                slots_text=(
                    "No bound fields: click a dot on a duration or DAC value to "
                    "make it a scan column."
                    if not columns
                    else "Bound: " + ", ".join(column.name for column in columns)
                ),
                table_text=_scan_table_text(rows, columns),
                source_text=self._scan_source,
                source_revision=self.revision,
                source_dirty=self._scan_dirty,
                repeats=self._scan_repeats,
                busy=False,
                progress_text=self._scan_progress,
            )
        )
        # A table appearing is a change in what the controls can do: stepping
        # through a scan needs one to step through.
        self._show_run_state()

    def _template(self, kind: str) -> str:
        from zlc_pulse import scan_columns_for, scan_table_template

        return scan_table_template(kind, scan_columns_for(self.sequence))

    def write_scan_template(self, kind: str) -> None:
        """Replace the editor's draft with a starter program for these slots."""

        if self.sequence is None:
            return
        self._scan_source = self._template(str(kind))
        self._scan_dirty = True
        self.view.replace_scan_draft(self._scan_source)

    def run_scan_program(self, source: str) -> bool:
        """Execute the scan program and keep the table it produced.

        The program is experiment code the operator just typed, run in this
        process on purpose: a scan table is a small array and the alternative
        -- a restricted expression language -- is a second language to learn
        for no safety anyone here needs.  What it must produce is checked:
        ``scan_table``, two-dimensional, one column per bound slot.
        """

        import numpy as np

        from zlc_pulse import scan_columns_for, validate_scan_table

        if self.sequence is None or not self._has_scan_slots():
            return False
        self._scan_source = str(source)
        namespace: dict = {}
        try:
            exec(compile(self._scan_source, "<scan program>", "exec"), namespace)  # noqa: S102
        except Exception as error:
            self._scan_progress = f"scan program failed: {error}"
            self._warn(f"scan program failed: {error}")
            self._refresh_scan_page()
            return False
        table = namespace.get("scan_table")
        if table is None:
            self._warn("the scan program did not assign scan_table")
            return False
        # Through zlc_pulse, which owns what a legal table is.  This window
        # used to decide it here and again on the file-load path, and the two
        # did not agree: the loader skipped the width check entirely.
        try:
            self._scan_rows = validate_scan_table(
                table, scan_columns_for(self.sequence)
            )
        except Exception as error:
            self._warn(str(error))
            return False
        self._scan_shape = namespace.get("scan_shape")
        self._scan_dirty = False
        self._scan_progress = f"{len(self._scan_rows)} scan point(s) ready"
        self._done(f"scan program produced {len(self._scan_rows)} point(s)")
        self.view.acknowledge_scan_draft(dirty=False, source_revision=self.revision)
        self._refresh_scan_page()
        return True

    def set_scan_repeats(self, repeats: int) -> None:
        """How many times the table is played; zero means until Stop."""

        self._scan_repeats = max(0, int(repeats))
        self._refresh_scan_page()

    def set_scan_source(self, use_loaded: bool) -> None:
        """Whether the uploaded table is the loaded file or the generated one."""

        if use_loaded and not self._scan_path:
            self._warn(
                "no file has been loaded yet -- use Load Array first; "
                "the generated table is what will be used"
            )
            self.view.set_scan_source(False, self._scan_path)
            self._scan_use_loaded = False
            self._refresh_scan_page()
            return
        self._scan_use_loaded = bool(use_loaded)
        self.view.set_scan_source(self._scan_use_loaded, self._scan_path)
        self._refresh_scan_page()

    def load_scan_array(self) -> bool:
        """Read a scan table from a file, instead of generating one."""

        import numpy as np

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
        # A scan table off a network share takes long enough for a second click
        # to land, and the second read would race the first into _scan_rows.
        # The button says so while it is busy, which is also why the setter
        # exists; nobody was calling it.
        self.view.set_scan_busy(True)
        try:
            path = Path(chosen)
            data = np.load(path) if path.suffix == ".npy" else np.loadtxt(path, delimiter=",")
            self._scan_rows = validate_scan_table(
                data, scan_columns_for(self.sequence)
            )
        except Exception as error:
            self._warn(f"cannot read {Path(chosen).name}: {error}")
            return False
        finally:
            self.view.set_scan_busy(False)
        self._scan_path = str(chosen)
        self._scan_use_loaded = True
        self._done(
            f"loaded {len(self._scan_rows)} scan point(s) from {Path(chosen).name}"
        )
        self.view.set_scan_source(True, self._scan_path)
        self._refresh_scan_page()
        return True

    load_scan_program = load_scan_array

    def save_scan_array(self) -> str:
        """Write the table exactly as it will be uploaded."""

        import numpy as np

        if not self._scan_rows:
            self._warn("there is no scan table to save")
            return ""
        folder = Path(self.path).parent if self.path else Path.cwd()
        target = unique_path(folder, f"{(self.sequence.name if self.sequence else 'scan')}-scan", ".npy")
        np.save(target, np.asarray(self._scan_rows, dtype=np.int64))
        self._scan_progress = f"saved {target.name}"
        self._refresh_scan_page()
        return str(target)

    def hold_scan_point(self) -> bool:
        """Stop advancing and hold the point the board is on.

        Holding is how a scan is inspected: the outputs stay at one row so a
        camera or a scope sees that point and nothing else.
        """

        if self.sequencer is None:
            self._warn("this editor is not connected to a sequencer")
            return False
        cursor = self._scan_cursor()
        self.stop()
        if cursor is None:
            self._scan_progress = "held (the board reported no scan point)"
        else:
            self._held_point = int(cursor)
            self._scan_progress = f"held at scan point {self._held_point}"
            self._write_held_point()
        self._refresh_scan_page()
        return True

    def step_scan_point(self, delta: int) -> bool:
        """Move the held point by one, staying held."""

        if self.sequencer is None or not self._scan_rows:
            self._warn("nothing is held to step")
            return False
        held = 0 if self._held_point is None else self._held_point
        self._held_point = max(0, min(len(self._scan_rows) - 1, held + int(delta)))
        self._write_held_point()
        self._scan_progress = f"held at scan point {self._held_point}"
        self._refresh_scan_page()
        return True

    def _write_held_point(self) -> None:
        """Put one scan row on the board, as the values the slots take."""

        if self.sequencer is None or self._held_point is None:
            return
        if not self._scan_rows:
            return
        row = self._scan_rows[min(self._held_point, len(self._scan_rows) - 1)]
        try:
            self.sequencer.write_slots(tuple(int(value) for value in row))
        except Exception as error:
            self._warn(f"cannot hold that point: {error}")

    def _scan_cursor(self) -> int | None:
        cursor = getattr(self.sequencer, "cursor", None)
        if not callable(cursor):
            return None
        try:
            return cursor()
        except Exception:
            return None

    def refresh_scan_progress(self) -> None:
        """Where the board is in the table, asked only while the page is visible."""

        if self.sequencer is None:
            return
        cursor = self._scan_cursor()
        total = len(self._scan_rows)
        if cursor is None:
            self._scan_progress = "running" if self.running else ""
        elif total:
            self._scan_progress = f"point {int(cursor) % total + 1}/{total}"
        else:
            self._scan_progress = f"point {int(cursor)}"
        self.view.set_scan_progress_text(self._scan_progress)

    # -------------------------------------------------------------- refresh

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
            self.view.set_summary(
                f"{len(target.ports)} port(s) on this board - Add Period to start a pulse"
                if target is not None
                else EMPTY_SCHEDULE.summary_text
            )
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
                visible_ports=self._visible,
                pins=self.pins,
            )
        )
        self.view.set_title(f"PulseGUI - {self.sequence.name}")
        self.view.set_summary(
            f"{self.path or self.sequence.name} - "
            f"{len(self.sequence.periods)} period(s)"
        )
        self._show_connection(self._connection_status)
        self.refresh_target()
        self._refresh_scan_page()
        self.refresh_preview()

    def _on_include_off(self, _included: bool) -> None:
        """Showing every channel changes how many rows are drawn.

        So the size it was drawn at is no longer the right one, and any size
        the operator pinned was pinned for a different picture: drop the pin
        and let the content choose again, exactly as v1 did.
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
        """Whether the preview accepts measuring gestures."""

        self._preview_selectors = bool(enabled)
        self.refresh_preview()

    def preview_size(self) -> str:
        """The preset this pulse should be drawn at.

        A pinned choice wins; otherwise the content decides, through the one
        rule zlc_plot owns for pulses -- the same rule the console's loaded
        pulse panels use, so the same pulse is drawn the same size everywhere.
        """

        if self._pinned_size:
            return self._pinned_size
        from zlc_plot import recommended_pulse_preset

        # Counted off the drawing itself.  There used to be a second rule here
        # deciding which rows the preview "will" draw, and it could disagree
        # with timeline_of -- so a pulse was sized for a different number of
        # rows than it then drew.
        drawn = self._preview_rows()
        periods = 0 if self.sequence is None else len(self.sequence.periods)
        return recommended_pulse_preset(drawn, periods)

    def _preview_rows(self) -> int:
        """How many rows the preview draws, from the projection that draws them."""

        if self.sequence is None:
            return 0
        try:
            data = timeline_of(
                self.sequence,
                include_off=bool(self.view.preview_include_off_rows),
            )
        except Exception:
            return 0
        return len(getattr(data, "channels", ())) + len(
            getattr(data, "analog_traces", ())
        )

    def refresh_preview(self) -> None:
        """Redraw the preview from what is on screen.

        The host is built once and updated after that.  Rebuilding it per edit
        spawned a render worker and a matplotlib session for every keystroke,
        blocked the GUI thread waiting for the new one's first frame, and never
        closed the old -- an afternoon of editing left hundreds of live threads
        behind a window that felt slower with every change.
        """

        if self._make_preview is None or self.sequence is None:
            return
        view = self.view
        size = self.preview_size()
        try:
            data = timeline_of(
                self.sequence, include_off=bool(view.preview_include_off_rows)
            )
        except Exception as error:
            view.show_preview_placeholder(f"cannot draw this pulse: {error}")
            return
        if self._preview_host is not None:
            try:
                grown = self._update_preview(self._preview_host, data, size=size)
            except Exception as error:
                view.show_preview_placeholder(f"cannot draw this pulse: {error}")
                return
            if grown:
                # The content changed shape, so the widget showing it has to as
                # well.  It was sized once at mount and never again.  The host
                # is handed back rather than a widget: it knows its own new
                # size, and this side of the wall does not hold widgets.
                view.show_preview(self._preview_host)
            self._show_preview_state(size)
            return
        try:
            host = self._make_preview(
                data, size=size, selectors=self._preview_selectors
            )
        except Exception as error:
            view.show_preview_placeholder(f"cannot draw this pulse: {error}")
            return
        # A builder answers with the HOST that draws, and nothing else.  It used
        # to answer with the host, its widget and the size the widget had not
        # painted yet -- so this layer held a QWidget, sized it, and mounted it,
        # which is a composition root assembling a UI.  The host knows all
        # three; the window is the only place that unwraps it.
        self._preview_host = host
        view.show_preview(host)
        self._show_preview_state(size)

    def _show_preview_state(self, size: str) -> None:
        """What the preview is showing, in the words above it."""

        view = self.view
        view.set_preview_size_names(PANEL_SIZE_NAMES)
        view.set_preview_size(size, pinned=bool(self._pinned_size))
        if self.sequence is None:
            return
        view.set_preview_status(
            f"{len(self.sequence.periods)} period(s), "
            f"{self._preview_rows()} channel(s), "
            f"{_readable(sum(_nanoseconds(p.duration, p.unit) for p in self.sequence.periods))}"
            f"  -  {size}"
        )

    def save_preview_image(self) -> str:
        """Write the preview exactly as it is drawn.

        The plotting host renders its own file, so what lands on disk is the
        preview rather than a second drawing of the same pulse.
        """

        if self._preview_host is None:
            self._warn("there is no preview to save")
            return ""
        name = (self.sequence.name if self.sequence is not None else "pulse") or "pulse"
        folder = Path(self.path).parent if self.path else Path.cwd()
        target = unique_path(folder, name, ".png")
        # The HOST writes the file.  The widget is a Qt view onto it and has
        # never had a save(), so this always answered "cannot save itself" in
        # the shipped window while the test substituted a fake that could.
        save = getattr(self._preview_host, "save", None)
        if not callable(save):
            self._warn("this preview cannot save itself")
            return ""
        try:
            result = save(target)
            if hasattr(result, "result"):
                result.result()
        except Exception as error:
            self._warn(f"cannot save the preview: {error}")
            return ""
        self.view.set_preview_status(f"saved {target}")
        return str(target)

    # -------------------------------------------------------------- private

    def _edit_period(
        self,
        period_id: str,
        edit: Callable[[PulsePeriod], PulsePeriod],
        *,
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
        periods = []
        found = False
        for period in self.sequence.periods:
            if period.period_id == str(period_id):
                found = True
                periods.append(edit(period))
            else:
                periods.append(period)
        if not found:
            self._warn(f"{period_id} is not a period in this sequence")
            return
        following = ()
        if ripples_forward:
            order = [item.period_id for item in periods]
            following = tuple(order[order.index(str(period_id)) + 1 :])
        self._apply_value(
            self._rebuilt(periods=tuple(periods)),
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
            return replace_sequence(self.sequence, **changes)
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
        self.sequence = candidate
        self.revision += 1
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
        self.sequence = candidate
        self.revision += 1
        schedule = self.view
        for identifier in ((period_id,) if period_id is not None else ()) + tuple(also):
            period = next(
                (item for item in candidate.periods if item.period_id == identifier),
                None,
            )
            if period is not None:
                schedule.set_period(
                    project_period(candidate, period, visible_ports=self._visible)
                )
        if port_key is not None:
            row = next(
                (
                    row
                    for row in project_schedule(
                        candidate, visible_ports=self._visible, pins=self.pins
                    ).delay_rows
                    if row.port_key == port_key
                ),
                None,
            )
            if row is not None:
                schedule.set_delay_row(row)
        self._refresh_summary()
        self._show_run_state()
        self.refresh_preview()

    def _refresh_summary(self) -> None:
        """The totals a value edit can move, without touching the cards."""

        if self.sequence is None:
            return
        total_ns = sum(
            _nanoseconds(period.duration, period.unit) for period in self.sequence.periods
        )
        ports = project_ports(self.sequence.target, pins=self.pins, visible=self._visible)
        visible = sum(1 for port in ports if port.visible)
        self.view.set_schedule_summary(
            total_text=_readable(total_ns),
            total_tooltip=f"{total_ns:g} ns over {len(self.sequence.periods)} period(s)",
            period_count=len(self.sequence.periods),
            visible_text=f"{visible}/{len(ports)} ports",
            summary_text=(
                f"{self.path or self.sequence.name} - "
                f"{len(self.sequence.periods)} period(s)"
            ),
            scan_summary_text=(
                f"{len(self.sequence.slots)} scan slot(s)"
                if self.sequence.slots
                else "no scan slots"
            ),
        )

    def _warn(self, text: str) -> None:
        warn = getattr(self.view, "show_warning", None)
        if warn is not None:
            warn(text)

    def _done(self, text: str) -> None:
        """Say that something the operator asked for has happened.

        v1 confirmed its actions and this only ever spoke up to refuse, so a
        Save that worked and a Save that did nothing looked identical.
        """

        done = getattr(self.view, "show_done", None)
        if done is not None:
            done(text)

    def close(self) -> None:
        """Let go of the board and the preview -- both, whichever fails.

        The preview HOST owns a render worker and a matplotlib session, and
        close only cleared the widget: an editor opened and shut left the
        thread running for the life of the process.  Its two sibling presenters
        both close their hosts.
        """

        failures: list[BaseException] = []
        for step in (self._release, self._close_preview):
            try:
                step()
            except BaseException as error:  # noqa: BLE001 - both steps still run
                failures.append(error)
        if failures:
            raise failures[0]

    def _close_preview(self) -> None:
        host, self._preview_host = self._preview_host, None
        if host is not None:
            host.close()


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
    """

    fields = {
        "name": sequence.name,
        "target": sequence.target,
        "time_step_ns": sequence.time_step_ns,
        "periods": sequence.periods,
        "slots": sequence.slots,
        "delays": sequence.delays,
        "repeat": sequence.repeat,
    }
    fields.update(changes)
    return PulseSequence(**fields)


def _unique_id(existing: Sequence[str], stem: str) -> str:
    taken = set(existing)
    ordinal = len(taken) + 1
    while f"{stem}{ordinal}" in taken:
        ordinal += 1
    return f"{stem}{ordinal}"
