"""What a scan IS: ordered axes over parameter ports, as pure data.

A scan is not a kind of measurement logic.  It is a declaration -- which
knobs, which values, in which nesting order -- and everything else derives
from it: the execution steps, the dataset axes, the editor's form.  The
vocabulary for "which knob" is a PORT, projected from whatever owns the knob;
nothing here invents one, so a plan can never name a parameter its pulse does
not declare.

Port families are open.  Today the bench projects one:

* ``pulse:param:<parameter_id>`` -- a pulse API parameter.  Advancing it is
  stepped: resolve the template with this point's values and load the result,
  so the board only ever plays an ordinary pulse.  This is how v1 held and
  scanned points, and it is the path the board is good at.

A knob the board can advance BY ITSELF between cycles (a hardware slot) would
join as its own family with a streamed advance; the plan's shape does not
change for it, which is the point of declaring the family in the port name.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from zlc_pulse import PulseSequence, api_parameter_columns_for


PULSE_PARAM_FAMILY = "pulse:param:"


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
        ports.append(
            ScanPort(
                PULSE_PARAM_FAMILY + str(column.name),
                str(column.name),
                str(column.unit),
                lo,
                hi,
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


__all__ = [
    "PULSE_PARAM_FAMILY",
    "ScanAxis",
    "ScanPlan",
    "ScanPort",
    "bind_plan",
    "scan_ports_for",
]
