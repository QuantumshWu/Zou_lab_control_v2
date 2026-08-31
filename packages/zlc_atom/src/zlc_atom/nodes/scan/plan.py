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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dataclasses import replace

from zlc_pulse import (
    PulseSequence,
    api_parameter_columns_for,
    scan_columns_for,
)
from zlc_pulse.codec import parse_pulse_tree_json, sequence_from_document_tree

from zlc_atom.authoring import TunableField
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
    raise ValueError(f"{port!r} belongs to no known port family")


def scan_axis_id(label: str) -> str:
    """What the dataset calls the axis a scan sweeps under ``label``.

    One spelling, shared by the writer that names the axis and by anyone --
    a selection, say -- reading a name back off a published dataset.
    """

    return f"scan.{label}"


def _ports_from_columns(columns) -> tuple[ScanPort, ...]:
    ports = []
    for column in columns:
        lo = float(column.limit_lo if column.limit_lo is not None else column.lo)
        hi = float(column.limit_hi if column.limit_hi is not None else column.hi)
        # The port's unit becomes the dataset axis's unit, and an axis unit is
        # something the plot's registry RESOLVES -- "s", "ms", "" -- not a
        # phrase.  A DAC code is a dimensionless count; the scan column's
        # "DAC code (0 = 0 V)" is the editor's label for the same fact, and
        # carrying it as the unit broke the first plot ever drawn over a scan
        # ("raster plot host failed to start: unknown unit ...").
        port = PULSE_PARAM_FAMILY + str(column.name)
        ports.append(
            ScanPort(
                port,
                port_label(port),
                "" if column.is_dac else str(column.unit),
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


def scan_ports_for_devices(tunables: Mapping | None) -> tuple[ScanPort, ...]:
    """Every port the bench's tunable devices offer, from their own words.

    A device volunteers through ``tunable_fields()``.  A scan exposes only a
    bounded, live-writable field whose dependency group is that field alone:
    this executor advances one scalar port at a time and cannot pretend a
    coupled hardware transaction is atomic.  The port's unit is the field's
    own declared unit -- an RF frequency axis publishes hertz -- and its
    label names the device and the knob.
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
            if field.minimum is None or field.maximum is None:
                continue
            if not tunable.live_write or tunable.dependency_group != (field.name,):
                continue
            port = f"{DEVICE_PARAM_FAMILY}{key}:{field.name}"
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", str(self.port))
        values = tuple(float(value) for value in self.values)
        if not values:
            raise ValueError(f"axis {self.port!r} has no values to play")
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"axis {self.port!r} contains a non-finite value")
        object.__setattr__(self, "values", values)


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
        object.__setattr__(self, "axes", axes)

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
                {"port": axis.port, "values": list(axis.values)}
                for axis in self.axes
            ]
        }

    @classmethod
    def from_tree(cls, tree: Mapping) -> "ScanPlan":
        if not isinstance(tree, Mapping) or "axes" not in tree:
            raise ValueError("a scan plan document carries its axes")
        axes = tuple(
            ScanAxis(str(entry["port"]), tuple(entry["values"]))
            for entry in tree["axes"]
        )
        return cls(axes)


def manual_axis_name(port: str) -> str:
    """The operator-facing name behind a ``manual:`` port."""

    text = str(port)
    if not text.startswith(MANUAL_PARAM_FAMILY):
        raise ValueError(f"{port!r} is not a manual axis")
    name = text[len(MANUAL_PARAM_FAMILY):].strip()
    if not name:
        raise ValueError("a manual axis carries a name")
    return name


def manual_axis(name: str, values: Sequence[float]) -> ScanAxis:
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
    return ScanAxis(MANUAL_PARAM_FAMILY + label, tuple(values))


def split_manual_axes(plan: ScanPlan) -> tuple[tuple[ScanAxis, ...], ScanPlan]:
    """The operator's axes, and the plan a machine can play underneath.

    A manual axis is walked by hand BETWEEN runs of the inner plan, so it
    is outside everything a machine advances -- not a preference, a fact
    about who moves what: the inner plan plays from one load, and a hand
    cannot reach into it.  A plan that nests one the other way round is
    refused here, by name, rather than silently reordered into something
    the operator did not author.
    """

    axes = plan.axes
    manual = tuple(
        axis for axis in axes if axis.port.startswith(MANUAL_PARAM_FAMILY)
    )
    if not manual:
        return (), plan
    if axes[: len(manual)] != manual:
        inside = tuple(
            manual_axis_name(axis.port)
            for axis in axes[len(manual):]
            if axis.port.startswith(MANUAL_PARAM_FAMILY)
        )
        raise ValueError(
            "an operator walks a manual axis between plays of the inner "
            "plan, so it stands outside every axis a machine advances; "
            f"move {', '.join(repr(name) for name in inside)} above the "
            "machine axes"
        )
    board = axes[len(manual):]
    if not board:
        raise ValueError(
            "a seamless scan plays a table the board advances; a plan of "
            "manual axes alone has no table to play"
        )
    return manual, ScanPlan(board)


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
            if value < port.lo or value > port.hi:
                raise ValueError(
                    f"axis {axis.port!r} plays {value!r}, outside the port's "
                    f"range [{port.lo:g}, {port.hi:g}] {port.unit}"
                )
        bound.append(port)
    return tuple(bound)


def _template_sequence(path: str | Path) -> PulseSequence:
    source = Path(path).expanduser().resolve()
    return sequence_from_document_tree(
        parse_pulse_tree_json(source.read_text(encoding="utf-8"))
    )


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


def load_seamless_template(path: str | Path) -> PulseSequence:
    """A seamless/board-driven template: its hardware scan slots ARE the axes.

    The template's author placed the slots in the pulse editor; the plan
    supplies the values every slot plays.  API parameters may also be
    declared -- they bake to their authored values before the table plays.
    """

    sequence = _template_sequence(path)
    if not sequence.slots:
        raise ValueError(
            "a seamless scan template declares hardware scan slots; this "
            "pulse declares none.  Place the slots in the pulse editor -- "
            "the board plays exactly what the template scans"
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

#: The seamless/board-driven template, selected from the same folder.
SEAMLESS_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    SCAN_PULSE_CONTRACT,
    "pulses",
    (".json",),
    load_seamless_template,
    argument_name="pulse_resource",
)


def plan_from_authored(payload: object) -> ScanPlan:
    """The plan behind one authored value: a document, a tree, or its JSON."""

    if isinstance(payload, ScanPlan):
        return payload
    if isinstance(payload, Mapping):
        return ScanPlan.from_tree(payload)
    text = str(payload or "").strip()
    if not text:
        raise ValueError(
            'the scan plan is empty; it reads like '
            '{"axes": [{"port": "pulse:param:da_bias_x", "values": [-256, 0, 256]}]}'
        )
    return ScanPlan.from_tree(json.loads(text))


def _selected_plan(
    selection: object,
    draft: Mapping[str, object],
    context: Mapping[str, object],
) -> dict[str, object]:
    """The authored plan, narrowed to the region the operator drew.

    A box or an x range on a scan's own plot names SCANNED axes -- that is
    what the dataset's axes are -- so it says "sweep this part next".  Each
    named axis keeps its point COUNT, because that is what the plan's form
    offers (from, to, points): the same effort spent over a smaller range.
    An axis the region does not name is untouched, and a region that names
    none of them -- a camera ROI drawn on the frames a scan captured --
    leaves the plan exactly as it was.  The frames belong to the camera.
    """

    del context
    plan = plan_from_authored(draft.get("plan"))
    wanted = {
        str(getattr(item, "axis", "")): item
        for item in getattr(selection, "ranges", ())
    }
    axes: list[ScanAxis] = []
    for axis in plan.axes:
        chosen = wanted.get(scan_axis_id(port_label(axis.port)))
        if chosen is None or len(axis.values) < 2:
            axes.append(axis)
            continue
        axes.append(
            ScanAxis(
                axis.port,
                tuple(
                    float(value)
                    for value in np.linspace(
                        float(chosen.lower), float(chosen.upper), len(axis.values)
                    )
                ),
            )
        )
    return {"plan": json.dumps(ScanPlan(tuple(axes)).to_tree())}


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
    "load_seamless_template",
    "load_stepped_template",
    "slots_from_plan",
    "plan_from_authored",
    "scan_ports_for",
    "scan_ports_for_devices",
]
