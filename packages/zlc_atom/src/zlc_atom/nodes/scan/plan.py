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

from zlc_pulse import PulseSequence, api_parameter_columns_for, sequence_from_tree

from zlc_atom.nodes._framework.descriptor import WorkspaceResourceSpec


PULSE_PARAM_FAMILY = "pulse:param:"

#: ``device:<key>:<field>`` -- a runtime knob on an installed device.  Its
#: advance is a ``tune(field, value)`` call before the point fires; the board
#: cannot advance it itself, so a board-advanced plan refuses it.
DEVICE_PARAM_FAMILY = "device:"

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

    def __post_init__(self) -> None:
        if not str(self.port):
            raise ValueError("a scan port needs a name")
        if not math.isfinite(self.lo) or not math.isfinite(self.hi) or self.lo > self.hi:
            raise ValueError(f"port {self.port!r} has no usable range")


def scan_ports_for(sequence: PulseSequence) -> tuple[ScanPort, ...]:
    """Every port this pulse offers, from its own declarations.

    The hard limits come from the same projection the pulse editor's scan page
    uses, so a plan cannot promise a value the board would refuse.
    """

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    ports = []
    for column in api_parameter_columns_for(sequence):
        lo = float(column.limit_lo if column.limit_lo is not None else column.lo)
        hi = float(column.limit_hi if column.limit_hi is not None else column.hi)
        # The port's unit becomes the dataset axis's unit, and an axis unit is
        # something the plot's registry RESOLVES -- "s", "ms", "" -- not a
        # phrase.  A DAC code is a dimensionless count; the scan column's
        # "DAC code (0 = 0 V)" is the editor's label for the same fact, and
        # carrying it as the unit broke the first plot ever drawn over a scan
        # ("raster plot host failed to start: unknown unit ...").
        ports.append(
            ScanPort(
                PULSE_PARAM_FAMILY + str(column.name),
                str(column.name),
                "" if column.is_dac else str(column.unit),
                lo,
                hi,
            )
        )
    return tuple(ports)


def scan_ports_for_devices(tunables: Mapping | None) -> tuple[ScanPort, ...]:
    """Every port the bench's tunable devices offer, from their own words.

    A device volunteers through ``tunable_fields()``; only fields with BOTH
    bounds declared become ports, because a plan must be refusable against a
    finite range before anything touches hardware.  Authoring fields carry
    no unit vocabulary, so the port's unit is empty and its label names the
    device and the knob.
    """

    ports: list[ScanPort] = []
    for key in sorted(dict(tunables or {})):
        device = tunables[key]
        fields = getattr(device, "tunable_fields", None)
        if not callable(fields):
            continue
        for field in fields():
            if field.minimum is None or field.maximum is None:
                continue
            ports.append(
                ScanPort(
                    f"{DEVICE_PARAM_FAMILY}{key}:{field.name}",
                    f"{key}.{field.name}",
                    "",
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


def load_scan_template(path: str | Path) -> PulseSequence:
    """One workspace pulse file, admitted only if it offers something to scan."""

    source = Path(path).expanduser().resolve()
    sequence = sequence_from_tree(json.loads(source.read_text(encoding="utf-8")))
    if not sequence.api_parameters:
        raise ValueError(
            "a scan template declares API parameters; this pulse declares none, "
            "so it offers nothing to scan"
        )
    if sequence.slots:
        raise ValueError(
            "a scan template cannot carry hardware scan slots; the plan is the "
            "only thing that says what varies"
        )
    return sequence


#: The pulse template both scan nodes select from the workspace's ``pulses``.
SCAN_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    SCAN_PULSE_CONTRACT,
    "pulses",
    (".json",),
    load_scan_template,
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


__all__ = [
    "DEVICE_PARAM_FAMILY",
    "PULSE_PARAM_FAMILY",
    "SCAN_PULSE_CONTRACT",
    "SCAN_PULSE_RESOURCE",
    "ScanAxis",
    "ScanPlan",
    "ScanPort",
    "bind_plan",
    "load_scan_template",
    "plan_from_authored",
    "scan_ports_for",
    "scan_ports_for_devices",
]
