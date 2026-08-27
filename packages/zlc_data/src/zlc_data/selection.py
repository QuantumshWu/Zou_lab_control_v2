"""Serializable, axis-named selection semantics with no presentation state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, floor
from numbers import Real

import numpy as np

from ._diagnostic import exact_integer_text
from .validation import (
    integer,
    nonnegative_integer,
)
from .axis import (
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    CoordinateScalar,
    canonical_coordinate_scalar,
)


class SelectionChange(str, Enum):
    """What happened to a selection.

    Here because it is the vocabulary of an event that crosses packages: a
    plotting surface raises it, a runtime bridge consumes it, and neither of
    those two depends on the other.  Both used to declare their own copy, which
    agreed only because a str enum compares equal to its own value -- so the
    day one of them gained a fifth member, the other would have gone on
    silently matching four.
    """

    ADDED = "added"
    UPDATED = "updated"
    COMMITTED = "committed"
    REMOVED = "removed"


@dataclass(frozen=True)
class IndexSelection:
    """Select one logical index and remove its named axis."""

    axis_id: AxisId
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        object.__setattr__(
            self,
            "index",
            nonnegative_integer(self.index, "selection index"),
        )


@dataclass(frozen=True)
class IndexRangeSelection:
    """Retain the half-open logical index interval ``[start, stop)``."""

    axis_id: AxisId
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        for name, value in (("start", self.start), ("stop", self.stop)):
            normalized = integer(value, f"selection {name}")
            assert normalized is not None
            object.__setattr__(self, name, normalized)
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("index range must be a non-empty half-open interval")


@dataclass(frozen=True)
class CoordinateRangeSelection:
    """Retain a numeric closed interval or one exact text/null coordinate."""

    axis_id: AxisId
    lower: CoordinateScalar
    upper: CoordinateScalar
    coordinate_frame: CoordinateFrameId | None

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        for name, value in (("lower", self.lower), ("upper", self.upper)):
            normalized = canonical_coordinate_scalar(value, f"coordinate {name}")
            object.__setattr__(
                self,
                name,
                normalized,
            )
        numeric = isinstance(self.lower, (int, float)) and isinstance(
            self.upper, (int, float)
        )
        if numeric and self.lower > self.upper:
            raise ValueError("coordinate range lower bound cannot exceed upper bound")
        if not numeric and self.lower != self.upper:
            raise ValueError(
                "text or null coordinate selection must name one exact value"
            )
        if self.coordinate_frame is not None and not isinstance(
            self.coordinate_frame, CoordinateFrameId
        ):
            raise TypeError("coordinate_frame must be CoordinateFrameId or None")


SelectionTerm = IndexSelection | IndexRangeSelection | CoordinateRangeSelection


@dataclass(frozen=True)
class Selection:
    """One immutable selection snapshot over one or more named axes.

    A rectangle is exactly two coordinate-range terms.  Facet scope, widget
    identity and processor bindings deliberately live in their respective
    adapters rather than in this value.
    """

    terms: tuple[SelectionTerm, ...]

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        if not terms:
            raise ValueError("Selection requires at least one term")
        if any(
            not isinstance(term, (IndexSelection, IndexRangeSelection, CoordinateRangeSelection))
            for term in terms
        ):
            raise TypeError("Selection contains an unsupported term")
        axis_ids = tuple(term.axis_id for term in terms)
        if len(set(axis_ids)) != len(axis_ids):
            raise ValueError("Selection may name each AxisId only once")
        object.__setattr__(self, "terms", tuple(sorted(terms, key=lambda term: term.axis_id.value)))

    @classmethod
    def index(cls, axis_id: AxisId, index: int) -> "Selection":
        return cls((IndexSelection(axis_id, index),))

    @classmethod
    def index_range(
        cls,
        axis_id: AxisId,
        start: int,
        stop: int,
    ) -> "Selection":
        return cls((IndexRangeSelection(axis_id, start, stop),))

    @classmethod
    def coordinate_range(
        cls,
        axis_id: AxisId,
        lower: CoordinateScalar,
        upper: CoordinateScalar,
        *,
        coordinate_frame: CoordinateFrameId | None,
    ) -> "Selection":
        return cls(
            (CoordinateRangeSelection(axis_id, lower, upper, coordinate_frame),),
        )

    @classmethod
    def rectangle(
        cls,
        x_axis_id: AxisId,
        y_axis_id: AxisId,
        x_lower: CoordinateScalar,
        x_upper: CoordinateScalar,
        y_lower: CoordinateScalar,
        y_upper: CoordinateScalar,
        *,
        coordinate_frame: CoordinateFrameId | None,
    ) -> "Selection":
        return cls(
            (
                CoordinateRangeSelection(
                    x_axis_id, x_lower, x_upper, coordinate_frame
                ),
                CoordinateRangeSelection(
                    y_axis_id, y_lower, y_upper, coordinate_frame
                ),
            ),
        )


def take_indices(
    array: np.ndarray,
    indices: range | tuple[int, ...],
    *,
    axis: int,
    drop: bool = False,
) -> np.ndarray:
    """Index one axis while retaining a view for a contiguous selection."""

    if isinstance(indices, range) and indices.step != 1:
        raise ValueError("selection range step must be 1")
    if drop:
        return np.take(array, indices[0], axis=axis)
    if isinstance(indices, range):
        selection = [slice(None)] * array.ndim
        selection[axis] = slice(indices.start, indices.stop)
        return array[tuple(selection)]
    return np.take(array, indices, axis=axis)


class EmptySelection(ValueError):
    """A selection that names no coordinate of the axis it speaks about.

    It is a fact about where the region was drawn, not a fault: a box in
    the letterbox band beside a picture selects nothing, and a caller that
    can say so to the operator needs to tell this apart from a genuine
    programming error.  Unnamed, it escaped a worker thread as a bare
    ValueError and the panel silently kept the region it had.
    """


def resolve_selection_indices(
    axis: AxisSpec,
    term: SelectionTerm,
) -> tuple[range | tuple[int, ...], bool]:
    """Resolve one named term without expanding a contiguous logical range."""

    if not isinstance(axis, AxisSpec):
        raise TypeError("axis must be AxisSpec")
    if not isinstance(term, (IndexSelection, IndexRangeSelection, CoordinateRangeSelection)):
        raise TypeError(f"unsupported selection term {type(term).__name__}")
    if term.axis_id != axis.axis_id:
        raise ValueError("selection term axis does not match AxisSpec")
    if isinstance(term, IndexSelection):
        if term.index >= axis.size:
            raise IndexError(
                "selection index "
                f"{exact_integer_text(term.index)} is outside axis "
                f"{axis.axis_id}"
            )
        return range(term.index, term.index + 1), True
    if isinstance(term, IndexRangeSelection):
        if term.stop > axis.size:
            raise IndexError(
                "selection range stop "
                f"{exact_integer_text(term.stop)} is outside axis "
                f"{axis.axis_id}"
            )
        return range(term.start, term.stop), False
    if axis.coordinate_frame != term.coordinate_frame:
        raise ValueError(f"coordinate frame mismatch for axis {axis.axis_id}")
    if axis.coordinates is None:
        # An implicit axis HAS coordinates -- index_origin + i -- and says so
        # through AxisSpec.coordinate().  Refusing a coordinate selection on it
        # meant a box drawn on a camera frame only worked because the producer
        # had written out the very tuple this axis exists to avoid.  They are
        # increasing, so the answer is arithmetic rather than a scan.
        if not isinstance(term.lower, (int, float)) or not isinstance(
            term.upper, (int, float)
        ):
            raise TypeError(
                f"implicit axis {axis.axis_id} accepts only numeric coordinates"
            )
        lower = max(0, ceil(term.lower - axis.index_origin))
        upper = min(axis.size - 1, floor(term.upper - axis.index_origin))
        if upper < lower:
            raise EmptySelection(
                f"coordinate selection is empty on axis {axis.axis_id}"
            )
        return range(lower, upper + 1), False
    if isinstance(term.lower, (int, float)) and isinstance(
        term.upper, (int, float)
    ):
        if any(
            value is not None
            and (isinstance(value, (bool, str)) or not isinstance(value, Real))
            for value in axis.coordinates
        ):
            raise TypeError(f"axis {axis.axis_id} coordinates are not entirely numeric")
        indices = tuple(
            index
            for index, value in enumerate(axis.coordinates)
            if value is not None and term.lower <= value <= term.upper
        )
    else:
        indices = tuple(
            index
            for index, value in enumerate(axis.coordinates)
            if value == term.lower
        )
    if not indices:
        raise EmptySelection(
            f"coordinate selection is empty on axis {axis.axis_id}"
        )
    if indices[-1] - indices[0] + 1 == len(indices):
        return range(indices[0], indices[-1] + 1), False
    return indices, False


__all__ = [
    "CoordinateRangeSelection",
    "EmptySelection",
    "SelectionChange",
    "IndexRangeSelection",
    "IndexSelection",
    "Selection",
    "SelectionTerm",
    "resolve_selection_indices",
    "take_indices",
]
