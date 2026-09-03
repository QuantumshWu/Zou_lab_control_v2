"""Public plot-kind and named-axis view vocabulary.

The data producer owns three axis domains.  This module only states how one
producer-declared axis participates in a view.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PlotKind(str, Enum):
    CURVE = "curve"
    IMAGE = "image"
    HISTOGRAM = "histogram"
    ROLLING = "rolling"
    FACET_GRID = "facet_grid"
    PULSE_TIMELINE = "pulse_timeline"


class AxisDomain(str, Enum):
    REPEAT = "repeat"
    POINT = "point"
    CELL_DATA = "cell_data"


@dataclass(frozen=True, slots=True)
class AxisRef:
    """A plot-side reference to one producer-declared data source."""

    domain: AxisDomain
    axis_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.domain, AxisDomain):
            raise TypeError("domain must be AxisDomain")
        if not isinstance(self.axis_id, str) or not self.axis_id.strip():
            raise ValueError(f"{self.domain.value} requires a non-empty axis_id")
        object.__setattr__(self, "axis_id", self.axis_id.strip())

    @classmethod
    def repeat(cls, axis_id: str) -> "AxisRef":
        return cls(AxisDomain.REPEAT, axis_id)

    @classmethod
    def point(cls, axis_id: str) -> "AxisRef":
        return cls(AxisDomain.POINT, axis_id)

    @classmethod
    def cell_data(cls, axis_id: str) -> "AxisRef":
        return cls(AxisDomain.CELL_DATA, axis_id)


__all__ = [
    "AxisDomain",
    "AxisRef",
    "PlotKind",
]
