"""What a scan IS: ordered axes over parameter ports, as pure data, and the
template that offers them.

A scan is not a kind of measurement logic.  It is a declaration -- which
knobs, which values, in which nesting order -- and everything else derives
from it: the execution steps, the dataset axes, the editor's form.  The
vocabulary for "which knob" is a PORT, projected from whatever owns the knob;
nothing here invents one, so a plan can never name a parameter its pulse does
not declare.

The port family says WHO can advance the knob, and that is what decides which
node runs the plan:

* ``pulse:param:<parameter_id>`` -- a pulse API parameter.  Either node takes
  it: the board can advance it from its own scan table (``seamless_scan``),
  and the host can resolve and reload the template per point
  (``stepped_scan``).
* ``device:<key>:<field>`` -- a runtime knob on an installed device.  Only the
  host can move it, with a ``tune(field, value)`` call before the point fires,
  so only ``stepped_scan`` accepts it.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path

import numpy as np

from zlc_data.units import DEFAULT_UNITS, format_quantity
from zlc_pulse import (
    apply_api_values,
    PulseSequence,
    api_parameter_columns_for,
    scan_columns_for,
)
from zlc_pulse.codec import read_pulse_document

from zlc_atom.authoring import TunableField, read_tunable_in_unit
from zlc_atom.nodes._framework.descriptor import (
    SelectionMapping,
    WorkspaceResourceSpec,
)


PULSE_PARAM_FAMILY = "pulse:param:"

#: ``device:<key>:<field>`` -- a runtime knob on an installed device.  Its
#: advance is a ``tune(field, value)`` call before the point fires; the board
#: cannot advance it itself, so a board-advanced plan refuses it.
DEVICE_PARAM_FAMILY = "device:"

#: ``manual:<name>`` -- a knob no machine here can reach.  Nothing advances
#: it: the run stops, the OPERATOR moves it, and the run continues.  That is
#: what makes power, polarization and anything else that lives behind a
#: thumbscrew scannable at all, and it is why such an axis always stands
#: outside every axis a machine advances.
MANUAL_PARAM_FAMILY = "manual:"

#: What a file must be to be a scan's pulse template.
SCAN_PULSE_CONTRACT = "zlc.pulse/scan-template"


@dataclass(frozen=True)
class ScanPort:
    """One knob the bench offers a scan, projected from its owner."""

    port: str
    label: str
    unit: str
    lo: float
    hi: float
    seed_lo: float | None = None
    seed_hi: float | None = None

    def __post_init__(self) -> None:
        if not str(self.port):
            raise ValueError("a scan port needs a name")
        if not math.isfinite(self.lo) or not math.isfinite(self.hi) or self.lo > self.hi:
            raise ValueError(f"port {self.port!r} has no usable range")
        seed_lo = self.lo if self.seed_lo is None else float(self.seed_lo)
        seed_hi = self.hi if self.seed_hi is None else float(self.seed_hi)
        if (
            not math.isfinite(seed_lo)
            or not math.isfinite(seed_hi)
            or seed_lo < self.lo
            or seed_hi > self.hi
            or seed_lo >= seed_hi
        ):
            raise ValueError(f"port {self.port!r} has no usable initial sweep")
        object.__setattr__(self, "seed_lo", seed_lo)
        object.__setattr__(self, "seed_hi", seed_hi)


def port_label(port: str) -> str:
    """The human name of a port, derived from the port itself.

    THE definition, so a label and its port cannot drift: a pulse parameter
    is named by its parameter id, and a device knob by the device and the
    field it belongs to.
    """

    text = str(port)
    if text.startswith(PULSE_PARAM_FAMILY):
        return text[len(PULSE_PARAM_FAMILY):]
    if text.startswith(DEVICE_PARAM_FAMILY):
        return text[len(DEVICE_PARAM_FAMILY):].replace(":", ".")
    if text.startswith(MANUAL_PARAM_FAMILY):
        # The one definition, here with the others: labels for manual axes
        # used to be produced at a call site instead, so axis naming had two
        # owners that could drift.
        return manual_axis_name(text)
    raise ValueError(f"{port!r} belongs to no known port family")


def port_group(port: str) -> str:
    """Which branch a port hangs under, derived from the port itself.

    Beside :func:`port_label` for the same reason: what a knob is called and
    where it is found must not be able to drift apart.  A device's knobs
    gather under THE DEVICE, because that is how an operator looks for one --
    "what can the RF source sweep" -- and not in one flat list where a pulse
    parameter and a laser current sit next to each other by accident.
    """

    text = str(port)
    if text.startswith(PULSE_PARAM_FAMILY):
        return "pulse"
    if text.startswith(DEVICE_PARAM_FAMILY):
        return text[len(DEVICE_PARAM_FAMILY):].split(":", 1)[0]
    if text.startswith(MANUAL_PARAM_FAMILY):
        return "manual"
    raise ValueError(f"{port!r} belongs to no known port family")


def port_leaf(port: str) -> str:
    """The port's name WITHIN its group -- the label, less the branch."""

    text = str(port)
    if text.startswith(DEVICE_PARAM_FAMILY):
        _device, separator, field = text[len(DEVICE_PARAM_FAMILY):].partition(":")
        if separator and field:
            return field
    return port_label(port)


def host_advanced_port(port: str) -> bool:
    """Whether the HOST advances this port between fires of the board.

    Two families qualify, for one structural reason: the board plays its
    whole table from a single load, and neither a hand on a thumbscrew nor
    a ``tune()`` call on an installed device can reach inside that.  Both
    therefore stand outside every board-advanced axis, walked between
    segments -- the run pauses, the knob moves (by hand or by call), the
    next segment fires.
    """

    text = str(port)
    return text.startswith(MANUAL_PARAM_FAMILY) or text.startswith(
        DEVICE_PARAM_FAMILY
    )


def scan_axis_ids(labels: Sequence[str]) -> tuple[str, ...]:
    """What the dataset calls each axis a plan sweeps, from the plan alone.

    One spelling, shared by the writer that names the axes and by the
    selection that reads a name back off a published dataset.  Two ports
    may carry one human name -- a manual knob and a pulse parameter both
    called ``bias`` -- so the second and later namesakes take their rank
    among the namesakes as a suffix.  The ids are a pure function of the
    plan's labels in order, which is what lets a range drawn on
    ``scan.bias.2`` land on the axis the picture drew and not on the first
    axis that happened to share the word.
    """

    seen: dict[str, int] = {}
    ids = []
    for label in labels:
        base = f"scan.{label}"
        rank = seen.get(base, 0) + 1
        seen[base] = rank
        ids.append(base if rank == 1 else f"{base}.{rank}")
    return tuple(ids)


def _ports_from_columns(columns) -> tuple[ScanPort, ...]:
    ports = []
    for column in columns:
        lo = float(column.limit_lo if column.limit_lo is not None else column.lo)
        hi = float(column.limit_hi if column.limit_hi is not None else column.hi)
        # The port's unit becomes the dataset axis's unit, and an axis unit is
        # something the plot's registry RESOLVES -- "s", "ms", "" -- not a
        # phrase.  A DAC code is a dimensionless count; the scan column's
        # "DAC code (0 = 0 V)" is the editor's LABEL for this, and carrying a
        # label as a unit broke the first plot ever drawn over a scan
        # ("raster plot host failed to start: unknown unit ...").  Blanking it
        # stopped the crash and told the next reader the axis is
        # dimensionless, which it is not: it is a count of codes, and the
        # registry has one now.
        port = PULSE_PARAM_FAMILY + str(column.name)
        ports.append(
            ScanPort(
                port,
                port_label(port),
                "code" if column.is_dac else str(column.unit),
                lo,
                hi,
                float(column.lo),
                float(column.hi),
            )
        )
    return tuple(ports)


def scan_ports_for(sequence: PulseSequence) -> tuple[ScanPort, ...]:
    """Every API-parameter port this pulse offers -- the STEPPED vocabulary.

    A stepped scan re-resolves the template per point through its API
    surface, so what it can vary is what the pulse exports as an API
    parameter.  The hard limits come from the same projection the pulse
    editor's scan page uses, so a plan cannot promise a value the board
    would refuse.
    """

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    return _ports_from_columns(api_parameter_columns_for(sequence))


def hardware_scan_ports_for(sequence: PulseSequence) -> tuple[ScanPort, ...]:
    """Every hardware-slot port this pulse offers -- the SEAMLESS vocabulary.

    A seamless scan is the BOARD advancing its own slot table, so what it
    can vary is exactly the scan slots the template's author placed, in
    slot order.
    """

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    return _ports_from_columns(scan_columns_for(sequence))


def scan_ports_for_devices(
    tunables: Mapping | None, *, units: Mapping[str, str] | None = None,
) -> tuple[ScanPort, ...]:
    """Every port the bench's tunable devices offer, from their own words.

    A device volunteers through ``tunable_fields()``.  A scan exposes only a
    bounded, live-writable field whose dependency group is that field alone:
    this executor advances one scalar port at a time and cannot pretend a
    coupled hardware transaction is atomic.  A field whose bounds leave no
    interval -- a knob pinned to one value by policy -- is a control, not
    an axis, and is passed over rather than failing the whole projection.
    The port's unit is the field's own declared unit -- an RF frequency
    axis publishes hertz -- and its label names the device and the knob.
    """

    ports: list[ScanPort] = []
    for key in sorted(dict(tunables or {})):
        device = tunables[key]
        fields = getattr(device, "tunable_fields", None)
        if not callable(fields):
            continue
        for tunable in fields():
            if not isinstance(tunable, TunableField):
                raise TypeError("device tunable_fields must contain TunableField values")
            field = tunable.metadata
            port = f"{DEVICE_PARAM_FAMILY}{key}:{field.name}"
            selected_unit = (units or {}).get(port)
            if selected_unit:
                tunable = read_tunable_in_unit(device, field.name, selected_unit)
                field = tunable.metadata
            if field.minimum is None or field.maximum is None:
                continue
            if float(field.minimum) >= float(field.maximum):
                continue
            if not tunable.live_write or tunable.dependency_group != (field.name,):
                continue
            ports.append(
                ScanPort(
                    port,
                    port_label(port),
                    field.unit or "",
                    float(field.minimum),
                    float(field.maximum),
                )
            )
    return tuple(ports)


@dataclass(frozen=True)
class ScanAxis:
    """One axis: a port and the values it takes, in play order."""

    port: str
    values: tuple[float, ...]
    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", str(self.port))
        values = tuple(float(value) for value in self.values)
        if not values:
            raise ValueError(f"axis {self.port!r} has no values to play")
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"axis {self.port!r} contains a non-finite value")
        object.__setattr__(self, "values", values)
        if not isinstance(self.unit, str):
            raise TypeError("scan axis unit must be text")
        object.__setattr__(self, "unit", self.unit.strip())

    def native_value(self, port: ScanPort, value: float) -> float:
        """Convert an authored coordinate only at the port boundary."""
        unit = self.unit or port.unit
        native = float(value) if unit == port.unit else float(
            DEFAULT_UNITS.convert(value, unit, port.unit)
        )
        if not math.isfinite(native):
            raise ValueError(f"axis {self.port!r} converts to a non-finite port value")
        return native


@dataclass(frozen=True)
class ScanPlan:
    """Ordered axes, outermost first.  The whole scan, as one document."""

    axes: tuple[ScanAxis, ...]

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not axes:
            raise ValueError("a scan plan needs at least one axis")
        if any(not isinstance(axis, ScanAxis) for axis in axes):
            raise TypeError("plan axes must be ScanAxis values")
        names = tuple(axis.port for axis in axes)
        if len(set(names)) != len(names):
            raise ValueError("a port may appear on one axis only")
        object.__setattr__(self, "axes", tuple(sorted(
            axes, key=lambda axis: host_advanced_port(axis.port), reverse=True,
        )))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(axis.values) for axis in self.axes)

    @property
    def point_count(self) -> int:
        return math.prod(self.shape)

    def rows(self) -> tuple[tuple[float, ...], ...]:
        """Every point, one value per axis, the LAST axis advancing fastest.

        That makes the declared order the nesting order: axes[0] is the
        outermost loop, exactly as an operator reads the plan.
        """

        return tuple(itertools.product(*(axis.values for axis in self.axes)))

    def to_tree(self) -> dict:
        return {
            "axes": [
                {"port": axis.port, "values": list(axis.values), "unit": axis.unit}
                for axis in self.axes
            ]
        }

    @classmethod
    def from_tree(cls, tree: object) -> "ScanPlan":
        axes = []
        for index, entry in enumerate(plan_input_rows(tree), 1):
            try:
                if entry["port"].startswith(MANUAL_PARAM_FAMILY):
                    manual_axis_name(entry["port"])
                values = (
                    parse_scan_values(entry["value_text"])
                    if entry["mode"] == "values" else entry["values"]
                )
                axes.append(ScanAxis(entry["port"], tuple(values), entry["unit"]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"scan axis {index} ({entry['port']!r}): {error}") from None
        return cls(tuple(axes))


def manual_axis_name(port: str) -> str:
    """The operator-facing name behind a ``manual:`` port."""

    text = str(port)
    if not text.startswith(MANUAL_PARAM_FAMILY):
        raise ValueError(f"{port!r} is not a manual axis")
    name = text[len(MANUAL_PARAM_FAMILY):].strip()
    if not name:
        raise ValueError("a manual axis carries a name")
    return name


def manual_axis(name: str, values: Sequence[float], unit: str = "") -> ScanAxis:
    """One manual axis: a name, and the values a HAND will set.

    Authored exactly like every other axis, values and all.  A coordinate
    is known before its data whichever knob carries it -- the dataset's
    schema is fixed the moment the first point lands, and a number typed
    later can no longer become an axis.  What makes this axis manual is
    only WHO advances it: the run stops and asks, where a board axis
    advances a slot.
    """

    label = str(name).strip()
    if not label:
        raise ValueError("a manual axis carries a name")
    return ScanAxis(MANUAL_PARAM_FAMILY + label, tuple(values), unit)


def split_outer_axes(plan: ScanPlan) -> tuple[tuple[ScanAxis, ...], tuple[ScanAxis, ...]]:
    """The host-advanced axes, and the plan the board plays underneath.

    A manual axis is walked by hand and a device axis by a ``tune()`` call,
    both BETWEEN plays of the inner plan -- not a preference, a fact about
    who moves what: the inner plan plays from one load, and neither a hand
    nor a host call can reach inside it. ScanPlan already stably places
    these axes outside board axes; the editor displays that same order.
    An empty board part means a fixed Pulse at each host point, not a fake
    scan axis or an empty public ScanPlan.
    """

    axes = plan.axes
    outer = tuple(axis for axis in axes if host_advanced_port(axis.port))
    return outer, axes[len(outer):]


def bind_plan(
    plan: ScanPlan,
    ports: Sequence[ScanPort],
) -> tuple[ScanPort, ...]:
    """The port behind each axis, or the refusal that names what is wrong.

    Binding is where a plan meets a bench: an axis naming a port this pulse
    does not offer, or promising values outside the port's hard limits, is
    refused HERE, before anything touches a device.
    """

    by_name = {port.port: port for port in ports}
    bound = []
    for axis in plan.axes:
        port = by_name.get(axis.port)
        if port is None:
            offered = ", ".join(sorted(by_name)) or "nothing"
            raise ValueError(
                f"this pulse offers no scan port named {axis.port!r}; "
                f"it offers {offered}"
            )
        for value in axis.values:
            native = axis.native_value(port, value)
            if native < port.lo or native > port.hi:
                raise ValueError(
                    f"axis {axis.port!r} plays {value!r} {axis.unit or port.unit}, outside the port's "
                    f"range [{port.lo:g}, {port.hi:g}] {port.unit}"
                )
        bound.append(port)
    return tuple(bound)


def _template_sequence(path: str | Path) -> PulseSequence:
    sequence, _editor = read_pulse_document(path)
    return sequence


def slots_from_plan(
    sequence: PulseSequence,
    ports: Sequence[ScanPort],
) -> PulseSequence:
    """Compile an API-driven template's planned parameters into slots.

    An API-surface caller (temperature's release scan) authors WHAT varies
    through API parameters; the board still needs slots.  This is that one
    compilation step: each planned parameter becomes a slot, every other
    API parameter stays for the caller to resolve.  A seamless TEMPLATE
    never takes this path -- its author placed the slots directly.
    """

    from zlc_pulse import PulseSlot

    slots = []
    scanned = []
    for port in ports:
        parameter_id = port.port[len(PULSE_PARAM_FAMILY):]
        parameter = next(
            value
            for value in sequence.api_parameters
            if value.parameter_id == parameter_id
        )
        slots.append(
            PulseSlot(
                parameter.field_ref.kind,
                parameter.field_ref,
                sequence.field_unit(parameter.field_ref),
                slot_id=parameter_id,
            )
        )
        scanned.append(parameter_id)
    return replace(
        sequence,
        slots=tuple(slots),
        api_parameters=tuple(
            value
            for value in sequence.api_parameters
            if value.parameter_id not in set(scanned)
        ),
    )


def load_stepped_template(path: str | Path) -> PulseSequence:
    """A stepped/API-driven template: API parameters vary, no slots.

    The plan is the only thing that says what varies; a template carrying
    hardware slots would be a second voice.
    """

    sequence = _template_sequence(path)
    if not sequence.api_parameters:
        raise ValueError(
            "a stepped scan template declares API parameters; this pulse "
            "declares none, so it offers nothing to scan"
        )
    if sequence.slots:
        raise ValueError(
            "a stepped scan template cannot carry hardware scan slots; the "
            "plan is the only thing that says what varies"
        )
    return sequence


#: The stepped/API-driven template, selected from the workspace's ``pulses``.
STEPPED_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    SCAN_PULSE_CONTRACT,
    "pulses",
    (".json",),
    load_stepped_template,
    argument_name="pulse_resource",
)

#: A seamless plan may carry only host axes and play an ordinary fixed Pulse;
#: any declared board slots still bind to the plan in the scan owner.
SEAMLESS_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    SCAN_PULSE_CONTRACT,
    "pulses",
    (".json",),
    _template_sequence,
    argument_name="pulse_resource",
)


def api_overrides_from_authored(payload: object) -> dict[str, float]:
    """One run's API parameter values, from the authored ``api_values`` text.

    ``name = number`` a line, in the unit the pulse declares for that name --
    the same one the form prints beside the box.  Only what this run sets
    differently is written down: a parameter left alone keeps whatever the
    pulse carries, which is where the workspace's current set of values has
    already landed.
    """

    text = "" if payload is None else str(payload)
    overrides: dict[str, float] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name:
            raise ValueError(
                f"API value line {number} must read 'name = number': {raw.strip()!r}"
            )
        try:
            overrides[name] = float(value.strip())
        except ValueError:
            raise ValueError(
                f"API value {name!r} is not a number: {value.strip()!r}"
            ) from None
    return overrides


def apply_api_overrides(
    sequence: PulseSequence, overrides: Mapping[str, float]
) -> PulseSequence:
    """The pulse with this run's API values written into its fields.

    Each number is read in the unit the pulse declares for that name, which
    is the unit the form printed beside the box.  The declarations survive,
    so a plan that scans one of these still overrides it per point.  A name
    the pulse does not declare is refused: it is a form written against a
    different pulse, and running the nominal value instead is the silence
    this whole path exists to end.
    """

    declared = {
        parameter.parameter_id: parameter.unit
        for parameter in sequence.api_parameters
    }
    unknown = tuple(sorted(name for name in overrides if name not in declared))
    if unknown:
        raise ValueError(
            f"this pulse declares no API parameter(s) {', '.join(unknown)}"
        )
    if not overrides:
        return sequence
    applied, _ids, _absent = apply_api_values(
        sequence,
        {name: (float(value), declared[name]) for name, value in overrides.items()},
    )
    return applied


def api_overrides_to_authored(overrides: Mapping[str, float]) -> str:
    """The text behind one mapping of API parameter values."""

    lines = [
        f"{name} = {format_quantity(value, '1')}" for name, value in overrides.items()
    ]
    return "\n".join(lines)


def plan_input_rows(payload: object) -> tuple[dict, ...]:
    """Read the two independent input banks without compiling Values text."""
    if isinstance(payload, ScanPlan):
        payload = payload.to_tree()
    elif isinstance(payload, str):
        if not payload.strip():
            raise ValueError("the scan plan is empty; add an axis")
        payload = json.loads(payload)
    if not isinstance(payload, Mapping) or set(payload) != {"axes"}:
        raise ValueError("a scan plan document carries only its axes")
    entries = payload["axes"]
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("scan plan axes must be a list")
    rows = []
    for index, entry in enumerate(entries, 1):
        allowed = {"port", "values", "unit", "mode", "value_text"}
        if not isinstance(entry, Mapping) or not {"port", "values"} <= set(entry) or set(entry) - allowed:
            raise ValueError("scan axis fields must be port, values, and optional unit, mode, value_text")
        row = {"port": entry["port"], "values": entry["values"], "unit": entry.get("unit", ""),
               "mode": entry.get("mode", "range"), "value_text": entry.get("value_text", "")}
        if any(not isinstance(row[key], str) for key in ("port", "unit", "mode", "value_text")):
            raise TypeError(f"scan axis {index}: port, unit, mode and value_text must be text")
        if row["mode"] not in ("range", "values"):
            raise ValueError(f"scan axis {index} ({row['port']!r}): mode must be range or values")
        values = row["values"]
        if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
                or any(isinstance(value, bool) or not isinstance(value, Real) for value in values)):
            raise TypeError(f"scan axis {index} ({row['port']!r}): Range values must be a numeric list")
        row["values"] = list(values)
        rows.append(row)
    return tuple(sorted(rows, key=lambda row: host_advanced_port(row["port"]), reverse=True))


def parse_scan_values(text: str) -> tuple[float, ...]:
    """Comma-separated finite numbers, retaining the authored order/repeats."""
    if not isinstance(text, str):
        raise TypeError("Values must be comma-separated text")
    if not text.strip():
        raise ValueError("Values is empty; enter comma-separated numbers")
    values = []
    for index, item in enumerate(text.split(","), 1):
        try:
            value = float(item.strip())
        except ValueError:
            raise ValueError(f"Values item {index} is not a number: {item!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"Values item {index} must be finite")
        values.append(value)
    return tuple(values)


def plan_from_authored(payload: object) -> ScanPlan:
    """Compile the active bank to the existing immutable execution plan."""
    return payload if isinstance(payload, ScanPlan) else ScanPlan.from_tree(payload)


def _selected_plan(
    selection: object,
    draft: Mapping[str, object],
    context: Mapping[str, object],
) -> dict[str, object] | None:
    """The authored plan, narrowed to the region the operator drew.

    A box or an x range on a scan's own plot names SCANNED axes -- that is
    what the dataset's axes are -- so it says "sweep this part next".  Each
    named axis keeps its point COUNT, because that is what the plan's form
    offers (from, to, points): the same effort spent over a smaller range.
    An axis the region does not name is untouched, and a region that names
    none of them -- a camera ROI drawn on the frames a scan captured --
    leaves the plan exactly as it was.  The frames belong to the camera.
    """

    rows = plan_input_rows(draft.get("plan"))
    wanted = {
        str(getattr(item, "axis", "")): item
        for item in getattr(selection, "ranges", ())
    }
    axis_ids = scan_axis_ids([port_label(row["port"]) for row in rows])
    changed = False
    for row, axis_id in zip(rows, axis_ids, strict=True):
        chosen = wanted.get(axis_id)
        if chosen is None or len(row["values"]) < 2:
            continue
        source_unit = context.get("axis_units", {}).get(axis_id)
        if source_unit is None:
            raise ValueError(f"selected scan axis {axis_id!r} has no recorded unit")
        unit = row["unit"] or source_unit
        bounds = (float(chosen.lower), float(chosen.upper))
        if source_unit != unit:
            bounds = DEFAULT_UNITS.convert(bounds, source_unit or "1", unit or "1")
        row["values"] = [float(value) for value in np.linspace(
            float(bounds[0]), float(bounds[1]), len(row["values"]))]
        changed = True
    return {"plan": json.dumps({"axes": rows})} if changed else None


#: What a region drawn on a scan's plot does to that scan's plan.  A cell of a
#: facet grid reports its OWN kind here -- the grid is a layout, the cell is
#: the picture -- so a box inside one cell narrows the axes that cell draws.
SCAN_PLAN_SELECTIONS = (
    SelectionMapping(
        plot_kind="image",
        selector_kind="area",
        draft_fields=("plan",),
        map_patch=_selected_plan,
    ),
    SelectionMapping(
        plot_kind="curve",
        selector_kind="x_range",
        draft_fields=("plan",),
        map_patch=_selected_plan,
    ),
    SelectionMapping(
        plot_kind="image",
        selector_kind="x_range",
        draft_fields=("plan",),
        map_patch=_selected_plan,
    ),
)


__all__ = [
    "DEVICE_PARAM_FAMILY",
    "SCAN_PLAN_SELECTIONS",
    "PULSE_PARAM_FAMILY",
    "SCAN_PULSE_CONTRACT",
    "SEAMLESS_PULSE_RESOURCE",
    "STEPPED_PULSE_RESOURCE",
    "ScanAxis",
    "ScanPlan",
    "ScanPort",
    "bind_plan",
    "hardware_scan_ports_for",
    "load_stepped_template",
    "slots_from_plan",
    "plan_from_authored",
    "plan_input_rows",
    "parse_scan_values",
    "scan_ports_for",
    "scan_ports_for_devices",
]
