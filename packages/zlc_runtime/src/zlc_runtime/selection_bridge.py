"""Numeric selection-to-signal bridging over an existing signal publication.

The bridge accepts only canonical selector coordinates and fit values.  Plot
adapters own the conversion from their interaction objects to these values;
this module never imports a frontend, GUI, or fit engine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import math
from numbers import Real
from threading import RLock
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    CoordinateFrameId,
    CellValidity,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    EmptySelection,
    OwnedSnapshot,
    PointColumn,
    point_domain_admits,
    PointTable,
    REPEAT,
    SCAN_POINT,
    Selection,
    ValidityContract,
    ValueSchema,
    compact_dataset_validity,
    expand_dataset_validity,
    materialize_derived_dataset,
    point_ordinal_axis,
)
from zlc_data import SelectionChange
from zlc_data.snapshot_projection import (
    axis_catalog,
    restricted_schema,
    restricted_values,
    selection_indices,
)
from zlc_data.selection import IndexRangeSelection
from zlc_data import canonical_text

from .dataset import MonitorCoverage
from .dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from .plane import GenerationSchemaAdvanced
from .plane import SignalDataPlane, SignalPublication, SignalValue

__all__ = [
    "FacetCondition",
    "FitEventValue",
    "SelectionBridge",
    "SelectionChange",
    "SelectionEventSource",
    "SelectionRange",
    "SelectionState",
    "selection_output_catalog",
]


def _tail_average(tail: np.ndarray) -> float:
    """The mean of a handful of samples, in an order nothing can change.

    ``partition`` settles WHICH samples are in the tail, never their order
    among themselves, and floating point addition is not associative -- so
    the answer moved in its last bit when the partition was asked for both
    ends at once instead of one at a time.  It had always depended on
    numpy's internals; it just had no occasion to show it.  Ten values
    sorted cost nothing and the answer stops depending on how they were
    found.
    """

    return float(np.mean(np.sort(tail), dtype=np.float64))


#: How many of the extreme samples the tail statistics average.
_TAIL_SAMPLES = 10


class _Sample:
    """One region's values, and the summary every scalar reads off it.

    The catalogue asks five questions of the same numbers, and two of them
    -- the mean of the ten smallest and of the ten largest -- were each
    answered by partitioning the WHOLE region: two full reorderings of two
    hundred thousand pixels, or of two million when the box is the frame, to
    find ten values.  Measured on a 346x345 camera window that was 95 per
    cent of the reduction; on the whole frame it was twenty of its
    twenty-two milliseconds.

    A camera frame is small unsigned integers, so it has at most sixty-five
    thousand distinct values and one ``bincount`` pass answers all five at
    once.  The array path stays for everything else AND is the
    specification: ``test_roi_statistics_agree`` asserts the two give
    identical numbers.
    """

    __slots__ = ("_values", "_counts", "_size")

    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self._size = int(values.size)
        self._counts: np.ndarray | None = None
        if values.dtype.kind == "u" and values.dtype.itemsize <= 2 and self._size:
            self._counts = np.bincount(np.ravel(values))

    @property
    def size(self) -> int:
        return self._size

    def _levels(self) -> np.ndarray:
        assert self._counts is not None
        return np.arange(self._counts.size, dtype=np.float64)

    def mean(self) -> float:
        if self._counts is None:
            return float(np.mean(self._values, dtype=np.float64))
        total = float(np.dot(self._counts, self._levels()))
        return total / float(self._size)

    def minimum(self) -> float:
        if self._counts is None:
            return float(np.min(self._values))
        return float(np.argmax(self._counts > 0))

    def maximum(self) -> float:
        if self._counts is None:
            return float(np.max(self._values))
        present = np.nonzero(self._counts)[0]
        return float(present[-1])

    def _tail_mean(self, count: int, *, from_top: bool) -> float:
        counts = self._counts
        assert counts is not None
        levels = self._levels()
        if from_top:
            counts = counts[::-1]
            levels = levels[::-1]
        taken = np.minimum(counts, np.maximum(
            count - np.concatenate(([0], np.cumsum(counts)[:-1])), 0
        ))
        return float(np.dot(taken, levels)) / float(count)

    def bottom_mean(self) -> float:
        count = min(_TAIL_SAMPLES, self._size)
        if self._counts is None:
            # One position at a time.  Asking ``partition`` for both ends in
            # a single call reorders the region once instead of twice and
            # sounds obviously better; measured, it is 5 to 25 per cent
            # SLOWER on the float regions that reach this path at all (uint8
            # is faster that way and never gets here -- it is counted).
            ordered = np.partition(self._values, count - 1)[:count]
            return _tail_average(ordered)
        return self._tail_mean(count, from_top=False)

    def top_mean(self) -> float:
        count = min(_TAIL_SAMPLES, self._size)
        if self._counts is None:
            ordered = np.partition(self._values, self._size - count)[-count:]
            return _tail_average(ordered)
        return self._tail_mean(count, from_top=True)


def _mean(sample: _Sample) -> float:
    return sample.mean()


def _minimum(sample: _Sample) -> float:
    return sample.minimum()


def _maximum(sample: _Sample) -> float:
    return sample.maximum()


def _bottom_10_mean(sample: _Sample) -> float:
    return sample.bottom_mean()


def _top_10_mean(sample: _Sample) -> float:
    return sample.top_mean()


#: The region itself, cut out of the source.  Every geometry has one --
#: an area on an image, an x range on a curve, a rolling window, a band of
#: a histogram: the region is a restriction of the signal, and the
#: restricted signal is what an operator wants downstream.  Only the
#: scalar REDUCTIONS depend on the geometry, because only an area consumes
#: axes there is anything to reduce over.
_SELECTED_FRAME = ("roi_frame", "ROI frame", None)
_AREA_SELECTION_OUTPUTS = (
    _SELECTED_FRAME,
    ("roi_mean", "Mean", _mean),
    ("roi_min", "Min", _minimum),
    ("roi_max", "Max", _maximum),
    ("roi_min_10_mean", "Min 10 mean", _bottom_10_mean),
    ("roi_max_10_mean", "Max 10 mean", _top_10_mean),
)
_RANGE_SELECTION_OUTPUTS = (_SELECTED_FRAME, ("roi_mean", "Mean", _mean))
#: Every name a selection route could ever claim, whatever the geometry.
#: Enabling and disabling outputs is bookkeeping over the whole vocabulary,
#: not over the one catalog the current region happens to offer.
_EVERY_SELECTION_OUTPUT = frozenset(
    name for name, _label, _reducer in _AREA_SELECTION_OUTPUTS
)

#: A bound that names no Dataset axis and restricts VALIDITY instead: the
#: cells outside it stop counting and the schema does not change.  It cannot
#: be a Selection term, because a Selection cuts axes.
#:
#: The shot ordinal is NOT one of these.  It names which PUBLICATION answers,
#: and a derivation only ever sees the newest one, so a rolling region
#: derives nothing at all; its bounds are carried for the panel alone.
_FILTER_DOMAINS = frozenset({"value"})


def selection_output_catalog(
    selector_kind: str,
    plot_kind: str,
) -> tuple[tuple[str, str], ...]:
    """Stable derived-output names for one region.

    Almost always a question about geometry alone -- Runtime reduces a
    selected area or a selected x range, and which Plot kind drew it does
    not matter.  A ROLLING region is the exception, and it is a fact about
    the DATA, not the drawing: its x is how many shots back a point is,
    counted from the newest, and the derivation only ever sees the newest
    publication.  Nothing it could publish would be the shots the operator
    boxed, so it publishes nothing and the region stays what it also is on
    every other kind -- the panel's own mark, which Fit and restore read.
    """

    if str(plot_kind) == "rolling":
        return ()
    kind = str(selector_kind)
    if kind == "area":
        catalog = _AREA_SELECTION_OUTPUTS
    elif kind == "x_range":
        catalog = _RANGE_SELECTION_OUTPUTS
    else:
        raise ValueError(f"unsupported derived selector kind {kind!r}")
    return tuple((name, label) for name, label, _reducer in catalog)


def _selection_filters(
    state: "SelectionState",
) -> tuple[float, float] | None:
    """The region's value band, if it drew one."""

    for item in state.ranges:
        if item.domain in _FILTER_DOMAINS:
            return float(item.lower), float(item.upper)
    return None


def _countable(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Which samples count: valid, and a number.

    ``isfinite`` is a question about floating point.  Asked of an integer
    frame -- which is what a camera delivers -- it walks every pixel to
    answer "yes" for all of them, allocates a boolean image to say so, and
    then the ``and`` walks the pair to allocate another.  Two passes and two
    megabytes, per shot, for a region whose validity the caller already
    knows.
    """

    if values.dtype.kind in "biu":
        return valid
    return valid & np.isfinite(values)


def _roi_statistics(
    values: np.ndarray,
    finite: np.ndarray,
    reducers: Mapping[str, Callable[[np.ndarray], float]],
) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
    """Evaluate the enabled scalar catalog over one finite-pixel set."""

    shape = values.shape[:2]
    flat_values = values.reshape(*shape, -1)
    flat_finite = finite.reshape(*shape, -1)
    result = {name: np.zeros(shape, dtype=np.float64) for name in reducers}
    valid = np.zeros(shape, dtype=np.bool_)
    # Asked once for the whole set rather than per cell: where nothing is
    # excluded the sample IS the row, and compacting it through a boolean
    # mask copies every pixel of the region to arrive at the same numbers.
    everything_counts = bool(flat_finite.all())
    for index in np.ndindex(shape):
        sample = flat_values[index]
        if not everything_counts:
            sample = sample[flat_finite[index]]
        elif not sample.flags.c_contiguous:
            # A region cut out of a frame is a strided window, and five
            # reductions plus two partitions each walk it the hard way.  The
            # boolean compaction this branch replaces happened to leave a
            # packed copy behind; say so on purpose rather than paying for
            # it by accident.
            sample = np.ascontiguousarray(sample)
        if not sample.size:
            continue
        valid[index] = True
        summary = _Sample(sample)
        for name, reducer in reducers.items():
            result[name][index] = reducer(summary)
    return MappingProxyType(
        {name: (answer, valid) for name, answer in result.items()}
    )


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite real number")
    return normalized


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class SelectionRange:
    """One canonical closed coordinate range over an upstream axis.

    Named producer axes carry ``axis``.  Repeat and point-row axes are
    structural, so ``domain`` tells the bridge which source-schema axis the
    same bounds describe without inventing a fake name.
    """

    axis: str
    lower: float
    upper: float
    domain: str
    coordinate_frame: str | None = None

    def __post_init__(self) -> None:
        domain = canonical_text(self.domain, "selection domain")
        if domain not in {
            "repeat",
            "point_row",
            "point_coordinate",
            "point_dimension",
            "data",
            # Two bounds name no Dataset axis: the measured VALUE, and
            # the session's own SHOT ordinal -- the rolling history's x,
            # which counts publications rather than rows of any one of
            # them.  Both still cut the signal, by restricting validity
            # rather than by selecting axis rows; see _FILTER_DOMAINS.
            "value",
            "shot",
        }:
            raise ValueError(
                "selection domain must be an exact Dataset axis domain, "
                "value, or shot"
            )
        if not isinstance(self.axis, str):
            raise TypeError("selection axis must be text")
        axis = self.axis.strip()
        if domain in {"point_coordinate", "point_dimension", "data"}:
            axis = canonical_text(axis, "selection axis")
        elif axis:
            raise ValueError("a structural selection range cannot carry an axis name")
        lower = _finite(self.lower, "selection lower")
        upper = _finite(self.upper, "selection upper")
        if lower > upper:
            lower, upper = upper, lower
        frame = self.coordinate_frame
        if frame is not None:
            frame = canonical_text(frame, "selection coordinate_frame")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "coordinate_frame", frame)
        object.__setattr__(self, "domain", domain)


@dataclass(frozen=True, slots=True)
class FacetCondition:
    """One numeric facet-axis condition carried across the bridge."""

    axis: str
    value: int | float | str
    domain: str

    def __post_init__(self) -> None:
        domain = canonical_text(self.domain, "facet domain")
        if domain not in {
            "point_row",
            "point_coordinate",
            "point_dimension",
            "data",
        }:
            raise ValueError("facet domain must be an exact Dataset axis domain")
        if not isinstance(self.axis, str):
            raise TypeError("facet axis must be text")
        axis = self.axis.strip()
        if domain == "point_row":
            if axis:
                raise ValueError("a point-row facet cannot carry an AxisId")
        else:
            axis = canonical_text(axis, "facet axis")
        value = self.value
        if isinstance(value, bool):
            raise TypeError("facet value must not be bool")
        if isinstance(value, str):
            value = canonical_text(value, "facet value")
        elif isinstance(value, Real):
            value = _finite(value, "facet value")
        else:
            raise TypeError("facet value must be text or a finite real number")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "domain", domain)


#: What a plot adapter may REPORT a region on.  Not the same question as
#: what a region can be DERIVED from -- a rolling trace's x is the shot
#: history, so an operator can mark a stretch of it and both of the panel's
#: surfaces must show the same mark, while nothing upstream is cut by it.
#: Conflating the two is why a rolling region was dropped on the floor: it
#: could not derive, so it was not remembered either.
SELECTION_PLOT_KINDS = frozenset({"image", "curve", "histogram", "rolling"})


@dataclass(frozen=True, slots=True)
class DrawnRegion:
    """The shape the hand put on the picture.

    Not always the shape the derivation reads.  A histogram's drag draws a
    rectangle whose y is a bin COUNT: it restricts nothing upstream, so the
    derivation reads an x range.  Saying that by REWRITING
    ``selector_kind`` made one word answer two questions -- which geometry
    to derive, and what a surface should show and remove -- and the two
    surfaces of one panel then disagreed: the card kept the box the hand
    drew while the Setting editor was handed a full-height band, and
    removing "the region" asked a surface to drop a kind it never had.

    ``lower``/``upper`` carry only the bound ``ranges`` cannot: the one
    that names nothing upstream.  When the drawn shape and the derived
    geometry agree, there is nothing extra to carry and this is ``None``.
    """

    kind: str
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        kind = canonical_text(self.kind, "drawn region kind")
        if kind not in {"area", "x_range"}:
            raise ValueError("drawn region kind must be area or x_range")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("drawn region bounds must be given together")
        if self.lower is not None:
            lower = _finite(self.lower, "drawn region lower")
            upper = _finite(self.upper, "drawn region upper")
            # Ordered the way SelectionRange orders its own: a drag that
            # ends above where it began is the same rectangle.
            if lower > upper:
                lower, upper = upper, lower
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class SelectionState:
    """Pure numeric selector state supplied by a plot adapter.

    ``repeat_index`` restricts the repeat axis STRUCTURALLY: the repeat axis
    is never name-addressed anywhere -- it is the first tensor dimension,
    identified by its role, deliberately anonymous on the plot side -- so a
    focused repeat facet crosses the bridge as a plain row index rather than
    as a named-axis condition.
    """

    plot_kind: str
    selector_kind: str
    ranges: tuple[SelectionRange, ...]
    facets: tuple[FacetCondition, ...] = ()
    repeat_index: int | None = None
    revision: int = 0
    #: What the hand drew, when that differs from what is derived.  The
    #: runtime never reads it; the panel's two surfaces do.
    drawn: "DrawnRegion | None" = None

    def __post_init__(self) -> None:
        plot_kind = canonical_text(self.plot_kind, "selection plot_kind")
        if plot_kind not in SELECTION_PLOT_KINDS:
            raise ValueError(
                "selection plot_kind must be one of "
                + ", ".join(sorted(SELECTION_PLOT_KINDS))
            )
        selector_kind = canonical_text(self.selector_kind, "selection selector_kind")
        if selector_kind not in {"area", "x_range"}:
            raise ValueError("selection selector_kind must be area or x_range")
        ranges = tuple(self.ranges)
        if not ranges or any(not isinstance(item, SelectionRange) for item in ranges):
            raise ValueError("selection ranges must contain SelectionRange values")
        facets = tuple(self.facets)
        if any(not isinstance(item, FacetCondition) for item in facets):
            raise TypeError("selection facets must contain FacetCondition values")
        if len({
            (
                "point"
                if item.domain in {"point_coordinate", "point_dimension"}
                else item.domain,
                item.axis,
            )
            for item in facets
        }) != len(facets):
            raise ValueError("selection facet axes must be unique")
        repeat_index = self.repeat_index
        if repeat_index is not None:
            repeat_index = _nonnegative_integer(
                repeat_index, "selection repeat_index"
            )
        object.__setattr__(self, "plot_kind", plot_kind)
        object.__setattr__(self, "selector_kind", selector_kind)
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "facets", facets)
        object.__setattr__(self, "repeat_index", repeat_index)
        if self.drawn is not None and not isinstance(self.drawn, DrawnRegion):
            raise TypeError("selection drawn must be a DrawnRegion or None")
        object.__setattr__(
            self,
            "revision",
            _nonnegative_integer(self.revision, "selection revision"),
        )


def _immutable_float_vector(value: object, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be a float64 array") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{field} must be a non-empty one-dimensional array")
    if np.any(np.isinf(array)):
        raise ValueError(f"{field} must not contain infinity")
    # A bytes-backed array cannot be made writable by changing its flags.  Fit
    # results cross an asynchronous callback boundary, so retaining a mutable
    # producer buffer here would make the event non-deterministic.
    payload = np.array(array, dtype=np.float64, copy=True).tobytes(order="C")
    return np.frombuffer(payload, dtype=np.float64).reshape(array.shape)


def _immutable_bool_vector(value: object, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.bool_):
        raise TypeError(f"{field} must be a bool array")
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{field} must be a non-empty one-dimensional array")
    payload = np.array(array, dtype=np.bool_, copy=True).tobytes(order="C")
    return np.frombuffer(payload, dtype=np.bool_).reshape(array.shape)


@dataclass(frozen=True, slots=True, eq=False)
class FitEventValue:
    """One fit parameter table with an optional sample axis.

    A scalar fit is represented by the same table with one sample.  Arrays
    and mappings are copied into immutable values at this boundary so a
    solver or plot callback cannot mutate a result after publication.
    """

    parameter_names: tuple[str, ...]
    parameter_units: Mapping[str, str]
    parameter_values: Mapping[str, np.ndarray]
    parameter_errors: Mapping[str, np.ndarray]
    success: np.ndarray
    sample_axis_domain: str
    sample_axis_id: str
    sample_axis_name: str
    sample_coordinates: np.ndarray
    sample_unit: str
    sample_labels: tuple[str, ...] | None
    source_generation: str
    source_revision: int
    batch_revision: int

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        if not names:
            raise ValueError("fit parameter_names must be non-empty")
        normalized_names: list[str] = []
        for name in names:
            normalized = canonical_text(name, "fit parameter name")
            if "/" in normalized or normalized.startswith("@"):
                raise ValueError("fit parameter names must be bare")
            normalized_names.append(normalized)
        names = tuple(normalized_names)
        if len(set(names)) != len(names):
            raise ValueError("fit parameter names must be unique")

        if not isinstance(self.parameter_units, Mapping):
            raise TypeError("fit parameter_units must be a mapping")
        units = dict(self.parameter_units)
        if set(units) != set(names):
            raise ValueError("fit parameter_units keys must match parameter_names")
        units = {
            name: canonical_text(units[name], f"fit parameter unit {name!r}", empty=True)
            for name in names
        }

        if not isinstance(self.parameter_values, Mapping):
            raise TypeError("fit parameter_values must be a mapping")
        if not isinstance(self.parameter_errors, Mapping):
            raise TypeError("fit parameter_errors must be a mapping")
        raw_values = dict(self.parameter_values)
        raw_errors = dict(self.parameter_errors)
        if set(raw_values) != set(names):
            raise ValueError("fit parameter_values keys must match parameter_names")
        if set(raw_errors) != set(names):
            raise ValueError("fit parameter_errors keys must match parameter_names")

        values: dict[str, np.ndarray] = {}
        errors: dict[str, np.ndarray] = {}
        sample_count: int | None = None
        for name in names:
            parameter_values = _immutable_float_vector(
                raw_values[name],
                f"fit parameter_values[{name!r}]",
            )
            parameter_errors = _immutable_float_vector(
                raw_errors[name],
                f"fit parameter_errors[{name!r}]",
            )
            if sample_count is None:
                sample_count = int(parameter_values.size)
            if parameter_values.size != sample_count:
                raise ValueError("all fit parameter value arrays must have one length")
            if parameter_errors.size != sample_count:
                raise ValueError("all fit parameter error arrays must match value length")
            finite_errors = np.isfinite(parameter_errors)
            if np.any(parameter_errors[finite_errors] < 0.0):
                raise ValueError("fit parameter errors must be non-negative or NaN")
            values[name] = parameter_values
            errors[name] = parameter_errors
        assert sample_count is not None

        success = _immutable_bool_vector(self.success, "fit success")
        if success.size != sample_count:
            raise ValueError("fit success must match the parameter table length")

        for name in names:
            values_for_name = values[name]
            if np.any(success & ~np.isfinite(values_for_name)):
                raise ValueError(
                    f"successful fit values for {name!r} must be finite"
                )
            if np.any(~success & np.isfinite(values_for_name)):
                raise ValueError(
                    f"failed fit values for {name!r} must be NaN"
                )

        sample_axis_name = canonical_text(
            self.sample_axis_name,
            "fit sample_axis_name",
            empty=True,
        )
        if sample_count > 1 and not sample_axis_name:
            raise ValueError("a batch fit must have a sample_axis_name")
        sample_axis_domain = canonical_text(
            self.sample_axis_domain,
            "fit sample_axis_domain",
            empty=True,
        )
        sample_axis_id = canonical_text(
            self.sample_axis_id,
            "fit sample_axis_id",
            empty=True,
        )
        domains = {
            "repeat",
            "point_row",
            "point_coordinate",
            "point_dimension",
            "data",
        }
        is_batch = bool(sample_axis_name)
        if is_batch and sample_axis_domain not in domains:
            raise ValueError("a batch fit must declare its exact sample axis domain")
        if not is_batch and (sample_axis_domain or sample_axis_id):
            raise ValueError("a scalar fit must not declare a sample axis identity")
        if sample_axis_domain in {"repeat", "point_row"}:
            if sample_axis_id:
                raise ValueError("a structural sample axis cannot carry an AxisId")
        elif sample_axis_domain and not sample_axis_id:
            raise ValueError("this sample axis domain requires an exact AxisId")
        sample_coordinates = _immutable_float_vector(
            self.sample_coordinates,
            "fit sample_coordinates",
        )
        if sample_coordinates.size != sample_count:
            raise ValueError("fit sample_coordinates must match the parameter table length")
        if not np.all(np.isfinite(sample_coordinates)):
            raise ValueError("fit sample_coordinates must be finite")
        sample_unit = canonical_text(self.sample_unit, "fit sample_unit", empty=True)

        labels = self.sample_labels
        if labels is not None:
            labels = tuple(labels)
            if len(labels) != sample_count:
                raise ValueError("fit sample_labels must match the parameter table length")
            if sample_count == 1 and not sample_axis_name:
                raise ValueError("a scalar fit must not have sample_labels")
            labels = tuple(
                canonical_text(label, "fit sample label")
                for label in labels
            )
            if sample_unit:
                raise ValueError("text sample coordinates cannot declare a unit")
            expected = np.arange(sample_count, dtype=np.float64)
            if not np.array_equal(sample_coordinates, expected):
                raise ValueError(
                    "text sample coordinates must use numeric indices 0..N-1"
                )

        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_units", MappingProxyType(units))
        object.__setattr__(self, "parameter_values", MappingProxyType(values))
        object.__setattr__(self, "parameter_errors", MappingProxyType(errors))
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "sample_axis_domain", sample_axis_domain)
        object.__setattr__(self, "sample_axis_id", sample_axis_id)
        object.__setattr__(self, "sample_axis_name", sample_axis_name)
        object.__setattr__(self, "sample_coordinates", sample_coordinates)
        object.__setattr__(self, "sample_unit", sample_unit)
        object.__setattr__(self, "sample_labels", labels)
        object.__setattr__(
            self,
            "source_generation",
            canonical_text(self.source_generation, "fit source_generation"),
        )
        object.__setattr__(
            self,
            "source_revision",
            _nonnegative_integer(self.source_revision, "fit source_revision"),
        )
        object.__setattr__(
            self,
            "batch_revision",
            _nonnegative_integer(self.batch_revision, "fit batch_revision"),
        )


@runtime_checkable
class SelectionEventSource(Protocol):
    def subscribe_fit(
        self,
        callback: Callable[[FitEventValue | None], object],
    ) -> Callable[[], None]: ...


class _TriggeredOutputs(dict[str, LiveDatasetOutput]):
    def __init__(
        self,
        values: Mapping[str, LiveDatasetOutput],
        trigger: tuple[str, int],
    ) -> None:
        super().__init__(values)
        self.trigger = trigger


class _StaleFit(RuntimeError):
    pass


class _BridgeProcessor:
    def __init__(
        self,
        bridge: "SelectionBridge",
        role: str,
        instance_id: str,
        output_names: tuple[str, ...],
    ) -> None:
        self._bridge = bridge
        self._role = role
        self.instance_id = instance_id
        self.dataset_output_declarations = tuple(
            DatasetOutputDeclaration(
                name,
                bridge._contract_id(role, name),
                index_by_source=True,
            )
            for name in output_names
        )

    def signal_key(self, output_name: str) -> str:
        return self._bridge._signal_key(output_name)

    def validate_processor_source(self, source: SignalValue) -> None:
        if not isinstance(source, SignalValue):
            raise TypeError("SelectionBridge processor source must be SignalValue")
        if not isinstance(source.snapshot, OwnedSnapshot):
            raise TypeError("SelectionBridge processor source must own a snapshot")

    def evaluate_processor(
        self,
        source: SignalValue,
        source_publication: SignalPublication,
    ) -> Mapping[str, LiveDatasetOutput]:
        return self._bridge._evaluate_processor(
            self,
            source,
            source_publication,
        )

    def accept_processor_result(
        self,
        source: SignalValue,
        source_publication: SignalPublication,
        result: Mapping[str, LiveDatasetOutput],
    ) -> None:
        self._bridge._accept_processor_result(
            self,
            source,
            source_publication,
            result,
        )

    def accept_processor_failure(self, error: Exception) -> None:
        self._bridge._accept_processor_failure(self, error)

    def accept_processor_cancelled(self) -> None:
        self._bridge._accept_processor_cancelled(self)

    def request_processor_owner_wake(self) -> None:
        self._bridge._processor_wake()


class SelectionBridge:
    """Turn committed numeric selector/fit events into derived signals."""

    def __init__(
        self,
        plane: SignalDataPlane,
        source_signal: str,
        selection_source: SelectionEventSource,
        *,
        bridge_id: str,
        source_publication_for: (
            Callable[[str, int], object | None] | None
        ) = None,
        request_owner_wake: Callable[[], object] | None = None,
    ) -> None:
        if not isinstance(plane, SignalDataPlane):
            raise TypeError("plane must be SignalDataPlane")
        if not isinstance(selection_source, SelectionEventSource):
            raise TypeError("selection_source must implement SelectionEventSource")
        if source_publication_for is not None and not callable(
            source_publication_for
        ):
            raise TypeError("source_publication_for must be callable or None")
        if request_owner_wake is not None and not callable(request_owner_wake):
            raise TypeError("request_owner_wake must be callable or None")
        source_signal = canonical_text(source_signal, "SelectionBridge source_signal")
        bridge_id = canonical_text(bridge_id, "SelectionBridge bridge_id")
        if "/" in bridge_id or bridge_id.startswith("@"):
            raise ValueError("SelectionBridge bridge_id must be bare")
        self._plane = plane
        self._source_signal = source_signal
        self._selection_source = selection_source
        #: Resolves a fit's exact parent publication by data generation and
        #: revision.  The
        #: renderer's panel port is the deterministic causal holder (the fit
        #: accepted inside that revision's own commit), so provenance comes
        #: from the presentation side, never from a plane retention window.
        self._source_publication_for = source_publication_for
        self._request_owner_wake = request_owner_wake
        self.bridge_id = bridge_id
        self._lock = RLock()
        #: One route per role, so publishing one is not a concurrent
        #: operation.  A publication CLAIMS its output names in the plane
        #: (attach/reserve) before it can decide whether the claim is still
        #: wanted, so two in-flight events for the same role both held the
        #: same names for a moment and the plane refused the second with
        #: "signal '@logic/<panel>/amplitude' is already owned by
        #: '<panel>:fit:1'" -- an operator arming a fit while a shot landed
        #: saw their fit break and stay broken.  These serialize the
        #: decide-claim-install sequence; ``_lock`` still guards state and is
        #: never held across a plane call.
        self._selection_publish_lock = RLock()
        self._fit_publish_lock = RLock()
        self._started = False
        self._closed = False
        self._selection: SelectionState | None = None
        self._selection_publication: SignalPublication | None = None
        self._selection_epoch = 0
        self._selection_processor: _BridgeProcessor | None = None
        self._fit_processor: _BridgeProcessor | None = None
        self._fit_event: FitEventValue | None = None
        self._fit_publication: SignalPublication | None = None
        self._fit_trigger_revision = 0
        self._last_fit_batch_revision: int | None = None
        self._processor_revision = 0
        self._block_revision = 0
        self._subscriptions: list[Callable[[], None]] = []
        self._last_error: Exception | None = None
        #: What this bridge cannot currently answer, as opposed to what
        #: went wrong in it.  A box that names no sample is the standing
        #: example: the instrument is fine, the question has no answer
        #: where it was asked, and widening the box answers it.  A level,
        #: so it clears itself the moment it stops being true.
        self._last_condition: str = ""
        self._output_enabled: dict[str, bool] = {}

    @property
    def last_error(self) -> Exception | None:
        with self._lock:
            return self._last_error

    @property
    def last_condition(self) -> str:
        """What this bridge cannot answer right now, or "".

        Separate from :attr:`last_error` because they are separate facts and
        the console treats them separately: an error says the instrument
        failed, a condition says the question has no answer where it was
        asked.  Reading the two through one channel is how an ROI a shade
        too small became a broken panel.
        """

        with self._lock:
            return self._last_condition

    def configure_outputs(self, enabled: Mapping[str, bool]) -> None:
        """Replace output switches and immediately replay held answers."""

        normalized = {str(name): bool(value) for name, value in enabled.items()}
        with self._lock:
            if self._closed:
                raise RuntimeError("SelectionBridge is closed")
            if normalized == self._output_enabled:
                return
            previous = self._output_enabled
            self._output_enabled = normalized
            selection = self._selection
            selection_publication = self._selection_publication
            fit_event = self._fit_event
            fit_publication = self._fit_publication
            selection_names = set(_EVERY_SELECTION_OUTPUT)
            fit_names = (
                set()
                if fit_event is None
                else set(self._unfiltered_fit_output_names(fit_event))
            )
            selection_changed = any(
                previous.get(name, True) != normalized.get(name, True)
                for name in selection_names
            )
            fit_changed = any(
                previous.get(name, True) != normalized.get(name, True)
                for name in fit_names
            )
            if selection_changed:
                self._selection = None
                self._selection_publication = None
                self._selection_epoch += 1
            started = self._started
        if selection_changed:
            self._release_route("selection")
        if fit_changed:
            self._release_route("fit")
        if not started:
            return
        if selection_changed and selection is not None:
            assert selection_publication is not None
            self._commit_selection(
                selection,
                source_publication=selection_publication,
            )
        if fit_changed and fit_event is not None and fit_publication is not None:
            self._publish_fit_event(fit_event, fit_publication, accept_revision=False)

    def start(
        self,
        *,
        initial_selection: SelectionState | None = None,
        initial_publication: SignalPublication | None = None,
    ) -> None:
        if initial_selection is not None and not isinstance(
            initial_selection, SelectionState
        ):
            raise TypeError("initial_selection must be SelectionState or None")
        if initial_publication is not None and not isinstance(
            initial_publication, SignalPublication
        ):
            raise TypeError("initial_publication must be SignalPublication or None")
        if (initial_selection is None) != (initial_publication is None):
            raise ValueError(
                "initial selection and its exact publication must be supplied together"
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("SelectionBridge is closed")
            if self._started:
                return
            self._started = True
        subscriptions: list[Callable[[], None]] = []
        try:
            subscriptions.append(
                self._selection_source.subscribe_fit(self._on_fit)
            )
        except BaseException:
            for unsubscribe in reversed(subscriptions):
                unsubscribe()
            with self._lock:
                self._started = False
            raise
        with self._lock:
            self._subscriptions.extend(subscriptions)
        if initial_selection is not None:
            assert initial_publication is not None
            self._commit_selection(
                initial_selection,
                source_publication=initial_publication,
            )

    def commit_selection(
        self,
        state: SelectionState,
        *,
        source_publication: SignalPublication,
    ) -> None:
        """Commit one owner-routed region against the exact displayed parent."""

        if not isinstance(state, SelectionState):
            raise TypeError("state must be SelectionState")
        if not isinstance(source_publication, SignalPublication):
            raise TypeError("source_publication must be SignalPublication")
        self._commit_selection(
            state,
            source_publication=source_publication,
        )

    def clear_selection(self) -> None:
        """Withdraw the current region and every derived output it owns."""

        with self._lock:
            self._selection = None
            self._selection_publication = None
            self._selection_epoch += 1
        self._release_route("selection")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            subscriptions, self._subscriptions = self._subscriptions, []
            self._selection = None
            self._selection_publication = None
            self._selection_epoch += 1
            self._fit_event = None
            self._fit_publication = None
        for unsubscribe in subscriptions:
            unsubscribe()
        for role in ("selection", "fit"):
            self._release_route(role)

    def _on_fit(self, event: FitEventValue | None) -> None:
        if event is None:
            # The fit was withdrawn.  Its outputs described an answer that is
            # no longer being made, so they retire with it and the names they
            # held are free again -- the same thing a removed box does.
            with self._lock:
                self._fit_event = None
                self._fit_publication = None
                self._last_fit_batch_revision = None
            self._release_route("fit")
            return
        self._publish_fit_event(event, None, accept_revision=True)

    def _publish_fit_event(
        self,
        event: FitEventValue,
        publication: SignalPublication | None,
        *,
        accept_revision: bool,
    ) -> None:
        if not isinstance(event, FitEventValue):
            raise TypeError("fit callback event must be FitEventValue")
        with self._lock:
            if self._closed or not self._started:
                return
        # The EXACT parent the fit derived from comes from the panel port
        # that rendered it: the fit accepted inside that revision's own
        # commit, so the port still holds the publication (pending or
        # presented).  A bridge without a port (headless benches, tests)
        # resolves against the current publication when the revision still
        # matches exactly.
        resolver = self._source_publication_for
        if publication is None and resolver is not None:
            candidate = resolver(
                event.source_generation,
                event.source_revision,
            )
            if candidate is not None and not isinstance(candidate, SignalPublication):
                raise TypeError("source_publication_for must return SignalPublication or None")
            publication = candidate
        if publication is None:
            current = self._current_source_publication()
            current_generation = None
            current_revision = None
            if current is not None:
                value = current.value(self._source_signal)
                if value is not None:
                    current_generation = str(
                        value.snapshot.ref.stream_generation.value
                    )
                    current_revision = value.snapshot.ref.revision.value
                    if (
                        current_generation == event.source_generation
                        and current_revision == event.source_revision
                    ):
                        publication = current
            if publication is None:
                if (
                    current_generation is not None
                    and current_generation != event.source_generation
                ) or (
                    current_generation == event.source_generation
                    and current_revision is not None
                    and current_revision > event.source_revision
                ):
                    # The panel already moved past this fit's shot: the fit
                    # is superseded flow control, not a failure — the next
                    # pair publishes a fresher one.
                    return
                self._record_error(
                    RuntimeError(
                        "fit event source publication is no longer resolvable"
                    )
                )
                return
        source = publication.value(self._source_signal)
        if source is None:
            raise RuntimeError("fit parent publication lacks its source signal")
        if (
            str(source.snapshot.ref.stream_generation.value)
            != event.source_generation
            or source.snapshot.ref.revision.value != event.source_revision
        ):
            raise RuntimeError("fit resolver returned another source identity")
        with self._lock:
            if self._closed or not self._started:
                return
            previous_event = self._fit_event
            source_changed = bool(
                previous_event is not None
                and previous_event.source_generation != event.source_generation
            )
            if accept_revision and (
                not source_changed
                and self._last_fit_batch_revision is not None
                and event.batch_revision <= self._last_fit_batch_revision
            ):
                self._record_error(
                    ValueError("fit batch_revision must increase for every accepted batch")
                )
                return
            schema_changed = previous_event is not None and self._fit_schema_key(
                previous_event
            ) != self._fit_schema_key(event)
            self._fit_event = event
            self._fit_publication = publication
            if accept_revision:
                self._last_fit_batch_revision = event.batch_revision
            self._fit_trigger_revision += 1
            trigger_revision = self._fit_trigger_revision
            output_names = self._fit_output_names(event)
            replaced = source_changed or schema_changed
        # Retiring the route this event replaces is the OTHER HALF of the
        # claim below -- both answer "who owns this role's names" -- so it
        # runs under the same lock.  Above it, the slot read empty while the
        # plane still held the names: a second event in flight attached over
        # them and was refused, "signal '@logic/<panel>/amplitude' is already
        # owned by '<panel>:fit:1'".  ``_release_route`` was taught this; this
        # second release site, inline here, was not -- and a fit's sample
        # coordinates change on ordinary shots, so it is the one the bench
        # actually walked through.
        with self._fit_publish_lock:
            with self._lock:
                stale = self._fit_processor
                if stale is not None and not (
                    replaced
                    or tuple(
                        item.name for item in stale.dataset_output_declarations
                    )
                    != output_names
                ):
                    stale = None
                if stale is not None:
                    self._fit_processor = None
            if stale is not None:
                self._withdraw_processor(stale)
        if not output_names:
            return
        if not self._source_retained():
            self._record_error(
                RuntimeError(
                    "this run is no longer held, so its fit derives nothing"
                )
            )
            return
        outputs = self._materialize_fit_outputs(
            self._source_snapshot(publication),
            event,
        )
        # From here the plane's output NAMES are claimed -- by the attach, or
        # by the terminal reserve -- and only afterwards can this event learn
        # whether its claim is still wanted.  The bridge owns ONE fit route,
        # so that sequence is not a concurrent operation: two events in
        # flight both held the same names for a moment and the plane refused
        # the second ("signal ... is already owned by ...").  The lock covers
        # the CLAIM only, never the materialization above it: a superseded
        # event must still bail at its own trigger check instead of queueing
        # behind the work that superseded it.
        with self._fit_publish_lock:
            with self._lock:
                if (
                    self._closed
                    or not self._started
                    or self._fit_event is not event
                    or self._fit_trigger_revision != trigger_revision
                ):
                    return
            if not self._plane.is_generation_live(self._source_signal):
                if publication is not self._current_source_publication():
                    # The generation finished on a NEWER shot than this
                    # fit's; a terminal answer must describe the final
                    # snapshot.
                    self._record_error(
                        RuntimeError(
                            "fit event trails a finished source generation"
                        )
                    )
                    return
                # Whatever holds the route now is withdrawn by the terminal
                # publish itself: the owner it must replace is the one in
                # the slot AT CLAIM TIME, not whichever was read before this
                # event materialized.
                self._publish_terminal("fit", outputs, publication)
                return
            with self._lock:
                processor = self._fit_processor
                attach = processor is None
                if attach:
                    processor = self._new_processor("fit", output_names)
            assert processor is not None
            if attach:
                # A fit signal is a presentation-paced follower: it advances
                # only after its source presents, so it retains lineage but
                # does not hold the source's coherent front waiting for it.
                current = self._current_source_publication()
                if current is None:
                    self._record_error(
                        RuntimeError("fit route lost its source while attaching")
                    )
                    return
                self._plane.attach_latest_only_processor(
                    processor,
                    source_name=self._source_signal,
                    initial_publication=current,
                    coherent=False,
                )
                with self._lock:
                    install = (
                        not self._closed
                        and self._started
                        and self._fit_event is event
                        and self._fit_trigger_revision == trigger_revision
                        and self._fit_processor is None
                    )
                    if install:
                        self._fit_processor = processor
                if not install:
                    self._withdraw_processor(processor)
                    return
            try:
                self._commit_processor(
                    processor,
                    outputs,
                    publication,
                    trigger=("fit", trigger_revision),
                )
            except RuntimeError as error:
                if "obsolete parent" in str(error):
                    return
                with self._lock:
                    stale = (
                        self._closed
                        or self._fit_event is not event
                        or self._fit_trigger_revision != trigger_revision
                    )
                    if self._fit_processor is processor:
                        self._fit_processor = None
                if stale:
                    self._withdraw_processor(processor)
                    return
                raise

    def _retire_selection_outputs(self) -> None:
        """Take down what the PREVIOUS region derived.

        A newly committed region that names no data replaces one that may
        have named plenty.  Left published, the old signal stands as the
        answer to the new box -- the same stale-slot mistake a withdrawn
        fit once made -- so the route is taken down and its output names
        freed.
        """

        processor = self._take_processor("selection")
        if processor is not None:
            self._withdraw_processor(processor)

    def _commit_selection(
        self,
        state: SelectionState,
        *,
        source_publication: SignalPublication,
        rearm: bool = False,
    ) -> None:
        """Claim this region's outputs in a fresh generation.

        ``rearm`` re-runs it for the region already committed, which is how
        a derivation whose SHAPE changed under it gets a new generation
        without the operator touching anything.
        """

        if not isinstance(source_publication, SignalPublication):
            raise TypeError("source_publication must be SignalPublication")
        with self._lock:
            if self._closed or not self._started:
                return
            previous = self._selection
            if (
                not rearm
                and previous is not None
                and state.revision <= previous.revision
            ):
                raise ValueError("selection revisions must increase")
            output_names = self._selection_output_names(state)
            self._selection_epoch += 1
            selection_epoch = self._selection_epoch
        publication = source_publication
        outputs = None
        if output_names:
            if not self._source_retained():
                # A real answer, not a failure: the picture is still on the
                # panel, the run behind it is not.  Nothing can be cut from
                # data the plane has let go.
                self._record_error(
                    RuntimeError(
                        "this run is no longer held, so a selection drawn "
                        "on it derives nothing"
                    )
                )
                self._retire_selection_outputs()
                return
            source = publication.value(self._source_signal)
            if source is None:
                raise RuntimeError(
                    "exact source publication has no selected signal"
                )
            try:
                outputs = self._materialize_selection_outputs(
                    self._source_snapshot(publication),
                    state,
                )
            except EmptySelection as error:
                # A real answer, not a failure: a box drawn in the band
                # beside the picture -- or past the edge of the sensor --
                # names no data, and "no data there" is what it means.
                # Raised out of a worker it escaped uncaught and the panel
                # silently kept the region it already had, so the operator
                # saw a mark move and nothing follow it.
                #
                # Said as a CONDITION, which is what that paragraph
                # describes.  It used to be handed to the failure channel on
                # the very next line, and that channel has no clear: a box
                # one pixel too small ended the panel's Setting form for the
                # rest of the session, widening it again included.
                self._record_condition(str(error))
                self._retire_selection_outputs()
                return
        self._selection_succeeded()
        # From here the plane's output NAMES are claimed, and only afterwards
        # can this commit learn whether its claim is still wanted -- the same
        # sequence the fit route runs, and the same reason it is serialized:
        # one selection route per bridge.  The materialization above stays
        # outside, so a superseded box still bails at its epoch check instead
        # of queueing behind the box that superseded it.
        with self._selection_publish_lock:
            with self._lock:
                if (
                    self._closed
                    or not self._started
                    or self._selection_epoch != selection_epoch
                ):
                    return
                # A committed image selection may change the derived sub-box
                # schema.  The plane deliberately freezes schemas per
                # generation, so every committed selection starts a fresh
                # latest-only generation while upstream publications still
                # re-cut it in place.
                old_processor = self._selection_processor
                self._selection_processor = None
                self._selection = state
                self._selection_publication = publication

            if old_processor is not None:
                self._withdraw_processor(old_processor)
            if not output_names:
                return
            assert publication is not None and outputs is not None

            # A box drawn on a run that has already finished is not a live
            # derivation and cannot be one: no further parent publication will
            # ever arrive to re-cut it.  It is still a real question with a
            # real answer, so it is answered once, terminally, instead of
            # being refused because the only machinery on offer was the live
            # kind.
            if not self._plane.is_generation_live(self._source_signal):
                self._publish_terminal("selection", outputs, publication)
                return

            with self._lock:
                if (
                    self._closed
                    or not self._started
                    or self._selection is not state
                ):
                    return
                self._selection_publication = publication
                processor = self._new_processor("selection", output_names)
            self._plane.attach_latest_only_processor(
                processor,
                source_name=self._source_signal,
                initial_publication=publication,
                paused=True,
            )
            with self._lock:
                install = (
                    not self._closed
                    and self._started
                    and self._selection is state
                    and self._selection_processor is None
                )
                if install:
                    self._selection_processor = processor
            if not install:
                self._withdraw_processor(processor)
                return

            try:
                self._commit_processor(
                    processor,
                    outputs,
                    publication,
                    trigger=("selection", state.revision),
                )
                # The first answer belongs to the exact publication already
                # on screen.  Only after that answer exists may this same
                # derived generation join the source's current latest
                # publication; doing it in the opposite order either rejects
                # a lagged restored panel or lets its first ROI silently
                # describe a newer shot.
                self._plane.catch_up_latest_only_processor(processor)
                self._processor_wake()
            except RuntimeError as error:
                if "obsolete parent" in str(error):
                    return
                with self._lock:
                    stale = self._closed or self._selection is not state
                    if self._selection_processor is processor:
                        self._selection_processor = None
                self._withdraw_processor(processor)
                if stale:
                    return
                raise

    def _publish_terminal(
        self,
        role: str,
        outputs: Mapping[str, LiveDatasetOutput],
        source_publication: SignalPublication,
    ) -> None:
        """Answer one selection or fit over a finished parent, once.

        Materialization is identical to the live path; only its lifetime is
        terminal because there is no source stream left to follow.
        """

        previous = self._take_processor(role)
        if previous is not None:
            # ONE owner per role.  The live path clears the slot before it
            # claims new names; this path minted its owner and left the old
            # one holding them, so the plane refused the terminal answer as
            # a name conflict with the bridge's own live route.
            self._withdraw_processor(previous)
        with self._lock:
            if self._closed or not self._started:
                return
            owner = self._new_processor(role, tuple(outputs))
            expected = self._selection if role == "selection" else self._fit_event
        try:
            self._plane.reserve_frozen_processor(
                owner,
                source_name=self._source_signal,
                source_publication=source_publication,
            )
            with self._lock:
                current = self._selection if role == "selection" else self._fit_event
                install = (
                    not self._closed
                    and self._started
                    and current is expected
                    and (
                        self._selection_processor
                        if role == "selection"
                        else self._fit_processor
                    )
                    is None
                )
                if install:
                    if role == "selection":
                        self._selection_processor = owner
                    else:
                        self._fit_processor = owner
            if not install:
                self._withdraw_processor(owner)
                return
            self._plane.commit_processor(
                owner,
                outputs,
                source_publication=source_publication,
                retain=True,
            )
            self._plane.seal_processor(owner)
        except BaseException:
            with self._lock:
                current = self._selection if role == "selection" else self._fit_event
                stale = self._closed or current is not expected
                if self._selection_processor is owner:
                    self._selection_processor = None
                if self._fit_processor is owner:
                    self._fit_processor = None
            self._withdraw_processor(owner)
            if stale:
                return
            raise
        with self._lock:
            current = self._selection if role == "selection" else self._fit_event
            wake = not self._closed and current is expected
        if wake:
            self._processor_wake()

    def _evaluate_processor(
        self,
        processor: _BridgeProcessor,
        source: SignalValue,
        source_publication: SignalPublication,
    ) -> Mapping[str, LiveDatasetOutput]:
        snapshot = self._source_snapshot(source_publication)
        with self._lock:
            if processor._role == "selection":
                state = self._selection
                if state is None:
                    raise RuntimeError("SelectionBridge has no committed selection")
                trigger = ("selection", state.revision)
                event = None
            else:
                state = None
                event = self._fit_event
                trigger_revision = self._fit_trigger_revision
                if event is None:
                    raise RuntimeError("SelectionBridge has no accepted fit event")
                if (
                    str(snapshot.ref.stream_generation.value)
                    != event.source_generation
                    or snapshot.ref.revision.value != event.source_revision
                ):
                    raise _StaleFit(
                        "fit result is stale for the current source publication"
                    )
                trigger = ("fit", trigger_revision)
        outputs = (
            self._materialize_selection_outputs(snapshot, state)
            if state is not None
            else self._materialize_fit_outputs(snapshot, event)
        )
        return _TriggeredOutputs(outputs, trigger)

    def _source_retained(self) -> bool:
        """Is the run this bridge derives from still HERE?

        A panel keeps its last picture after its run is retired, and its
        selectors keep working: a box drawn there asks for a dataset the
        plane no longer holds.  Asked as an exception, that answer left the
        bridge as ``LookupError`` -- a class the console's interaction
        drain did not name -- and killed the process from inside a Qt slot.
        The plane knows; ask it before materializing anything.
        """

        return bool(self._plane.retains(self._source_signal))

    def _source_snapshot(
        self,
        publication: SignalPublication,
    ) -> OwnedSnapshot:
        """The exact dataset prefix the panel and this derivation both mean."""

        return self._plane.current_dataset(
            self._source_signal,
            publication,
        )

    def _accept_processor_result(
        self,
        processor: _BridgeProcessor,
        _source: SignalValue,
        source_publication: SignalPublication,
        result: Mapping[str, LiveDatasetOutput],
    ) -> None:
        trigger = getattr(result, "trigger", None)
        if not isinstance(trigger, tuple) or len(trigger) != 2:
            self._record_error(RuntimeError("SelectionBridge processor lost its trigger"))
            return
        with self._lock:
            if processor._role == "selection":
                current = self._selection
                active = self._selection_processor
                expected = (
                    None
                    if current is None
                    else ("selection", current.revision)
                )
            else:
                active = self._fit_processor
                expected = (
                    None
                    if self._fit_event is None
                    else ("fit", self._fit_trigger_revision)
                )
        # A worker may finish after a newer control event has already issued
        # the same source publication.  Do not let that older result overwrite
        # the newer trigger merely because both parents are otherwise exact.
        if active is not processor or expected is None or trigger != expected:
            return
        try:
            self._commit_processor(
                processor,
                result,
                source_publication,
                trigger=trigger,
            )
            if processor._role == "selection":
                with self._lock:
                    if self._selection_processor is processor:
                        self._selection_publication = source_publication
        except RuntimeError as error:
            if "obsolete parent" in str(error) or "generation is no longer active" in str(error):
                return
            self._accept_processor_failure(processor, error)

    def _accept_processor_failure(
        self,
        processor: _BridgeProcessor,
        error: Exception,
    ) -> None:
        if isinstance(error, _StaleFit):
            # The source can advance while the plot is still fitting its last
            # accepted frame.  That is normal latest-only timing, not a failed
            # signal generation: retain the last accepted Fit publication
            # until the plot emits the next batch for this same panel.
            return
        if (
            isinstance(error, GenerationSchemaAdvanced)
            and processor._role == "selection"
        ):
            # Not a failure: what this region derives changed shape, which
            # a new generation is exactly the answer to.  A panel pooling a
            # window hands its own ROI a source that grows one shot per
            # publication while the window fills, so this fires once per
            # shot until it is full and then stops.  Releasing instead --
            # which is what an unnamed ValueError got -- retired the
            # region's outputs for good, and the operator saw the ROI they
            # had drawn stop publishing the moment they set a window.
            #
            # Not recorded either: the operator has nothing to do about a
            # window filling, and a panel that reports an error on every
            # shot of it looks broken while it is working.
            self._release_processor(processor)
            with self._lock:
                state = self._selection
            publication = self._current_source_publication()
            if state is not None and publication is not None:
                try:
                    self._commit_selection(
                        state,
                        source_publication=publication,
                        rearm=True,
                    )
                except Exception as retry:
                    self._record_error(retry)
            return
        # An EmptySelection reaches here too, and RELEASING is right for it:
        # a region that names no data names none on every publication -- it
        # is a fact about where the box was drawn, not about this shot -- so
        # what the PREVIOUS region derived has to be retired rather than left
        # standing as though it were still the answer.  The reason is
        # recorded either way, which is what the operator reads.
        self._record_error(error)
        self._release_processor(processor)

    def _accept_processor_cancelled(self, processor: _BridgeProcessor) -> None:
        return None

    def _commit_processor(
        self,
        processor: _BridgeProcessor,
        outputs: Mapping[str, LiveDatasetOutput],
        publication: SignalPublication,
        *,
        trigger: tuple[str, int],
    ) -> None:
        self._plane.commit_processor(
            processor,
            outputs,
            source_publication=publication,
            trigger=trigger,
        )

    def _role_publish_lock(self, role: str) -> RLock:
        return (
            self._selection_publish_lock
            if role == "selection"
            else self._fit_publish_lock
        )

    def _release_route(self, role: str) -> None:
        """Take one role's route out and free the names it holds.

        Claiming and releasing are the two halves of the same thing -- who
        owns this role's output names in the plane -- so they run under the
        same lock.  Releasing outside it left a window between "the slot is
        empty" and "the names are free" in which a claim arriving on another
        thread attached and the plane refused it: "signal ... is already
        owned by ...".  That window is what an operator saw as a fit that
        broke the moment they touched its publish switches while shots were
        landing.
        """

        with self._role_publish_lock(role):
            processor = self._take_processor(role)
            if processor is not None:
                self._withdraw_processor(processor)

    def _release_processor(self, processor: "_BridgeProcessor") -> None:
        """Release one EXACT processor, under its role's lock."""

        with self._role_publish_lock(processor._role):
            with self._lock:
                if self._selection_processor is processor:
                    self._selection_processor = None
                if self._fit_processor is processor:
                    self._fit_processor = None
            self._withdraw_processor(processor)

    def _take_processor(self, role: str) -> "_BridgeProcessor | None":
        """Remove one role's processor from the bridge, to be withdrawn.

        The bridge owns exactly one route per role; taking it out and
        withdrawing it is what frees its output names.  Written once here
        because every call site that open-coded it had to remember to do
        both, and the terminal path remembered neither.
        """

        with self._lock:
            if role == "selection":
                processor = self._selection_processor
                self._selection_processor = None
            else:
                processor = self._fit_processor
                self._fit_processor = None
        return processor

    def _withdraw_processor(self, processor: _BridgeProcessor) -> None:
        self._plane.cancel_latest_only_processor(processor)

    def _new_processor(
        self,
        role: str,
        output_names: tuple[str, ...],
    ) -> _BridgeProcessor:
        self._processor_revision += 1
        return _BridgeProcessor(
            self,
            role,
            f"{self.bridge_id}:{role}:{self._processor_revision}",
            output_names,
        )

    def _current_source_publication(self) -> SignalPublication | None:
        return self._plane.latest_publication(self._source_signal)

    def _record_error(self, error: Exception) -> None:
        with self._lock:
            self._last_error = error

    def _record_condition(self, condition: str) -> None:
        """Say what cannot be answered right now.  Empty means nothing."""

        with self._lock:
            self._last_condition = str(condition)

    def _selection_succeeded(self) -> None:
        """A commit that worked ends whatever the last one could not do.

        THE CONDITION ONLY.  ``_last_error`` is shared with the fit route,
        which reports its own refusals through it, so a selection has no
        business clearing one -- that is why the condition is a separate
        level in the first place.  What made an ROI a shade too small end
        the panel for the session was routing it into a channel that has no
        clear at all; it now has its own, and this is it.
        """

        with self._lock:
            self._last_condition = ""

    def _processor_wake(self) -> None:
        with self._lock:
            callback = None if self._closed else self._request_owner_wake
        if callback is not None:
            callback()

    def _signal_key(self, name: str) -> str:
        return f"@logic/{self.bridge_id}/{name}"

    @staticmethod
    def _contract_id(role: str, name: str) -> str:
        if role == "selection":
            return f"zlc.selection.{name}"
        if name.endswith("_err"):
            return "zlc.selection.fit.error"
        return "zlc.selection.fit.parameter"

    def _selection_output_names(self, state: SelectionState) -> tuple[str, ...]:
        return tuple(
            name
            for name, _label in selection_output_catalog(
                state.selector_kind,
                state.plot_kind,
            )
            if self._output_enabled.get(name, True)
        )

    def _fit_output_names(self, event: FitEventValue) -> tuple[str, ...]:
        return tuple(
            name
            for name in self._unfiltered_fit_output_names(event)
            if self._output_enabled.get(name, True)
        )

    @staticmethod
    def _unfiltered_fit_output_names(event: FitEventValue) -> tuple[str, ...]:
        names: list[str] = []
        for parameter in event.parameter_names:
            names.append(str(parameter))
            names.append(f"{parameter}_err")
        if len(set(names)) != len(names):
            raise ValueError("fit output names must be unique")
        return tuple(names)

    @staticmethod
    def _fit_schema_key(
        event: FitEventValue,
    ) -> tuple[object, ...]:
        return (
            event.parameter_names,
            tuple(
                (name, event.parameter_units[name])
                for name in event.parameter_names
            ),
            event.sample_axis_domain,
            event.sample_axis_id,
            event.sample_axis_name,
            tuple(float(value) for value in event.sample_coordinates),
            event.sample_unit,
            event.sample_labels,
        )

    def _resolve_axis(
        self,
        schema: DatasetSchema,
        domain: str,
        axis_id: str,
    ) -> tuple[AxisId, AxisSpec, str]:
        """Resolve one exact Dataset axis domain without consulting a label."""

        if domain == "repeat":
            if axis_id:
                raise ValueError("a repeat axis cannot carry an AxisId")
            axis = schema.repeat_axis
            return axis.axis_id, axis, "repeat"
        if domain == "point_row":
            if axis_id:
                raise ValueError("a point-row axis cannot carry an AxisId")
            axis = point_ordinal_axis(schema.point_table.row_count)
            return axis.axis_id, axis, "point"

        wanted = AxisId(canonical_text(axis_id, "selection axis id"))
        topology = schema.grid_topology
        present = {
            "point_coordinate": any(
                column.coordinate_id == wanted
                for column in schema.point_table.columns
            ),
            "point_dimension": bool(
                topology is not None and wanted in topology.dimension_ids
            ),
            "data": any(
                axis.axis_id == wanted for axis in schema.cell_schema.data_axes
            ),
        }
        if domain not in present:
            raise ValueError(f"unsupported selection axis domain {domain!r}")
        if not present[domain]:
            raise ValueError(
                f"{domain} AxisId {axis_id!r} is not present in the source snapshot"
            )
        expected_kind = "data" if domain == "data" else "point"
        matches = {
            (catalog_id, axis, kind)
            for _label, catalog_id, axis, kind in axis_catalog(schema)
            if catalog_id == wanted and kind == expected_kind
        }
        if len(matches) != 1:
            raise ValueError(
                f"{domain} AxisId {axis_id!r} is not uniquely present in the source snapshot"
            )
        return next(iter(matches))

    def _faceted_axis(
        self,
        schema: DatasetSchema,
        sample_axis_domain: str,
        sample_axis_id: str,
    ) -> tuple[AxisSpec, str] | None:
        """The parent axis a fit was faceted over, when the parent declares it.

        The fit carries the Plot facet's exact domain and AxisId; the display
        label never participates in resolution.  A scalar fit names no axis,
        and a facet over the bare point ordinal names no parent axis either.
        """

        if not sample_axis_domain:
            return None
        if sample_axis_domain == "repeat":
            return schema.repeat_axis, "repeat"
        if sample_axis_domain == "point_row":
            return None
        _axis_id, axis, kind = self._resolve_axis(
            schema,
            sample_axis_domain,
            sample_axis_id,
        )
        return axis, kind

    def _build_selection(
        self,
        schema: DatasetSchema,
        state: SelectionState,
    ) -> Selection:
        def range_term(value: SelectionRange):
            _axis_id, axis, _kind = self._resolve_axis(
                schema,
                value.domain,
                value.axis,
            )
            frame = (
                axis.coordinate_frame
                if value.coordinate_frame is None
                else CoordinateFrameId(value.coordinate_frame)
            )
            return Selection.coordinate_range(
                axis.axis_id,
                value.lower,
                value.upper,
                coordinate_frame=frame,
            ).terms[0]

        # A region restricts what its bounds NAME.  Bounds on Dataset axes
        # become Selection terms; bounds on the measured value or on the
        # shot ordinal restrict validity instead and are applied where the
        # values are (see ``_selection_filters``), so a histogram band and a
        # rolling window cut a signal exactly as an image box does.
        axis_ranges = tuple(
            item for item in state.ranges if item.domain not in _FILTER_DOMAINS
        )
        if state.plot_kind == "image":
            if state.selector_kind != "area" or len(axis_ranges) != 2:
                raise ValueError("image SelectionBridge requires a two-axis area")
            first, second = axis_ranges
            def axis_kind(value: SelectionRange) -> str:
                _axis_id, _axis, kind = self._resolve_axis(
                    schema,
                    value.domain,
                    value.axis,
                )
                return kind

            # An image has two axes in one domain: either two point quantities
            # (a scan heatmap) or two tensor quantities (repeat/data).  Mixing
            # point rows with a tensor axis cannot describe a rectangular
            # source surface.
            first_kind, second_kind = axis_kind(first), axis_kind(second)
            if (first_kind == "point") != (second_kind == "point"):
                raise ValueError(
                    "image area axes must both be point axes or both be tensor axes"
                )
            terms = [range_term(first), range_term(second)]
        else:
            terms = [range_term(item) for item in axis_ranges]
        for facet in state.facets:
            axis_id, axis, kind = self._resolve_axis(
                schema,
                facet.domain,
                facet.axis,
            )
            if isinstance(facet.value, str):
                if kind != "point" or axis.coordinates is None:
                    raise ValueError(
                        f"text facet {facet.axis!r} is not an indexed point axis"
                    )
                indices = tuple(
                    index
                    for index, value in enumerate(axis.coordinates)
                    if value == facet.value
                )
                if len(indices) != 1:
                    raise ValueError(
                        f"text facet {facet.axis!r} must identify one source point"
                    )
                from zlc_data import IndexSelection

                terms.append(IndexSelection(axis_id, indices[0]))
            else:
                frame = axis.coordinate_frame
                terms.append(
                    Selection.coordinate_range(
                        axis_id,
                        float(facet.value),
                        float(facet.value),
                        coordinate_frame=frame,
                    ).terms[0]
                )
        if not terms:
            # A region made only of value or shot bounds restricts no axis at
            # all -- it restricts what COUNTS.  Saying so as a full-range term
            # keeps one path through the projection instead of a second way to
            # mean "every row", and a contiguous range indexes as a slice.
            terms.append(
                IndexRangeSelection(
                    schema.repeat_axis.axis_id, 0, schema.repeat_axis.size
                )
            )
        if state.repeat_index is not None:
            # Structural, not named: the repeat axis is identified by its
            # role and position in the schema, so the restriction is a plain
            # logical row interval on that axis.
            terms.append(
                IndexRangeSelection(
                    schema.repeat_axis.axis_id,
                    state.repeat_index,
                    state.repeat_index + 1,
                )
            )
        return Selection(tuple(terms))

    def _next_reference(
        self,
        source: OwnedSnapshot,
        label: str,
        schema: DatasetSchema,
        *,
        revision: DatasetRevision | None = None,
    ) -> DatasetRevisionRef:
        self._block_revision += 1
        output_revision = source.ref.revision if revision is None else revision
        return DatasetRevisionRef(
            BlockId(
                f"{self.bridge_id}:{label}:{output_revision.value}:{self._block_revision}"
            ),
            source.ref.stream_generation,
            schema.fingerprint,
            output_revision,
        )

    def _materialize_selection_outputs(
        self,
        source: OwnedSnapshot,
        state: SelectionState,
    ) -> Mapping[str, LiveDatasetOutput]:
        """Cut one committed selection into signals that keep the parent's axes.

        A derivation SUBSETS its parent's axes: the repeat axis and the point
        axis of the derived signal are the parent's, and only the axes the
        reduction actually CONSUMES disappear.  A box on an image consumes the
        two image DATA axes and nothing else, so ``roi_mean`` is one value per
        (repeat, point) on the same derived schema ``roi_frame`` carries, the
        parent's point columns intact.  Pooling the point axis into a single
        scalar instead would silently average a cycle's physically distinct
        frames, which are different POINTS -- different moments of the pulse.
        """

        source_schema = source.block.schema
        selection = self._build_selection(source_schema, state)
        repeat_indices, point_indices, data_indices = selection_indices(
            source_schema,
            selection,
        )
        valid = expand_dataset_validity(source.block.validity, source_schema)
        values = restricted_values(
            source.block.values,
            source_schema,
            repeat_indices,
            point_indices,
            data_indices,
        )
        valid_values = restricted_values(
            valid,
            source_schema,
            repeat_indices,
            point_indices,
            data_indices,
        )
        value_band = _selection_filters(state)
        if value_band is not None:
            # A band on the measured value cuts no axis: the cells outside
            # it simply stop counting, which is what the region means on a
            # histogram (and on the value half of a rolling region).
            low, high = value_band
            with np.errstate(invalid="ignore"):
                valid_values = valid_values & (values >= low) & (values <= high)
        derived_schema = restricted_schema(
            source_schema,
            repeat_indices,
            point_indices,
            data_indices,
        )
        if value_band is not None:
            cell = derived_schema.cell_schema
            # The canonical scalar carrier has no components to vary over --
            # its value-level validity IS per cell -- and declaring one is
            # refused by ValueSchema itself.
            if cell != ValueSchema.scalar(cell.dtype, cell.value_unit):
                # A band decides cell by cell, so the derived signal's
                # validity varies along every axis its values do.  Carrying
                # the source's coarser contract forward would be a claim
                # the data no longer supports.
                derived_schema = DatasetSchema(
                    derived_schema.repeat_axis,
                    derived_schema.point_table,
                    derived_schema.grid_topology,
                    ValueSchema(
                        cell.data_axes,
                        ValidityContract.components(
                            *(axis.axis_id for axis in cell.data_axes)
                        ),
                        cell.dtype,
                        cell.value_unit,
                    ),
                )
        catalog = (
            _AREA_SELECTION_OUTPUTS
            if state.selector_kind == "area"
            else _RANGE_SELECTION_OUTPUTS
        )
        output: dict[str, LiveDatasetOutput] = {}
        scalar_outputs: dict[str, Callable[[np.ndarray], float]] = {}
        for name, _label, reducer in catalog:
            if not self._output_enabled.get(name, True):
                continue
            if reducer is not None:
                scalar_outputs[name] = reducer
                continue
            derived = materialize_derived_dataset(
                source.ref,
                values,
                schema=derived_schema,
                validity=compact_dataset_validity(valid_values, derived_schema),
                reference_for=lambda schema, output_name=name: self._next_reference(
                    source,
                    output_name,
                    schema,
                ),
            )
            total = derived_schema.repeat_axis.size * derived_schema.point_table.row_count
            output[name] = LiveDatasetOutput(
                DatasetOutputDeclaration(
                    name,
                    self._contract_id("selection", name),
                    index_by_source=True,
                ),
                derived,
                MonitorCoverage(total, total),
            )
        if not scalar_outputs:
            return output
        statistics = _roi_statistics(
            values,
            _countable(values, valid_values),
            scalar_outputs,
        )
        scalar_schema = DatasetSchema(
            derived_schema.repeat_axis,
            derived_schema.point_table,
            derived_schema.grid_topology,
            ValueSchema.scalar(
                np.dtype("float64"),
                source_schema.cell_schema.value_unit,
            ),
        )
        total = scalar_schema.repeat_axis.size * scalar_schema.point_table.row_count
        for name in scalar_outputs:
            answer, validity = statistics[name]
            derived = materialize_derived_dataset(
                source.ref,
                answer.reshape(scalar_schema.physical_shape),
                schema=scalar_schema,
                validity=compact_dataset_validity(
                    validity.reshape(scalar_schema.physical_shape),
                    scalar_schema,
                ),
                reference_for=lambda schema, output_name=name: self._next_reference(
                    source,
                    output_name,
                    schema,
                ),
            )
            output[name] = LiveDatasetOutput(
                DatasetOutputDeclaration(
                    name,
                    self._contract_id("selection", name),
                    index_by_source=True,
                ),
                derived,
                MonitorCoverage(total, total),
            )
        return output

    def _materialize_fit_outputs(
        self,
        source: OwnedSnapshot,
        event: FitEventValue,
    ) -> Mapping[str, LiveDatasetOutput]:
        output: dict[str, LiveDatasetOutput] = {}
        source_schema = source.block.schema
        sample_count = int(event.success.size)
        faceted = self._faceted_axis(
            source_schema,
            event.sample_axis_domain,
            event.sample_axis_id,
        )
        if faceted is not None and faceted[1] == "repeat":
            # The fit was faceted over the REPEAT axis, so its samples ARE
            # repeats: same conditions measured again.  They keep the parent's
            # repeat identity instead of being restated as point rows -- a
            # point row is an independent variable, which a repeat is not.
            schema_repeat = replace(
                faceted[0],
                size=sample_count,
                coordinates=tuple(
                    float(value) for value in event.sample_coordinates
                ),
                unit=event.sample_unit or None,
                index_origin=0,
                coordinate_labels=event.sample_labels,
            )
            schema_point_table = PointTable(1)
        else:
            # Every other faceted axis is a point axis on the fit's own table,
            # and it carries the ROLE of the axis it was faceted over: a
            # frame-faceted fit is a READOUT_EVENT column, a scan-faceted fit
            # a SCAN_POINT one.  An axis the parent does not declare (the
            # scalar fit, a point-row ordinal facet) has no role to inherit
            # and takes the point ordinal's own role -- and so does an axis
            # whose role the point domain does not admit.  A grid may be
            # faceted over ANY axis, including a component or the implicit
            # scalar, and re-stating one of those as a point column raised
            # "point column role is outside the point-domain role set" from
            # inside the fit, where an operator reads it as the fit being
            # broken.  Inheriting a role means inheriting one this domain
            # can carry; where it cannot, there is no role to inherit.
            sample_name = event.sample_axis_name or "sample"
            sample_axis_id = event.sample_axis_id or sample_name
            sample_role = (
                faceted[0].role
                if faceted is not None and point_domain_admits(faceted[0].role)
                else SCAN_POINT
            )
            point_columns = [
                PointColumn(
                    AxisId(sample_axis_id),
                    sample_name,
                    sample_role,
                    PointColumn.NUMERIC,
                    tuple(float(value) for value in event.sample_coordinates),
                    event.sample_unit or None,
                )
            ]
            if event.sample_labels is not None:
                label_name = f"{sample_name}_label"
                point_columns.append(
                    PointColumn(
                        AxisId(label_name),
                        label_name,
                        sample_role,
                        PointColumn.TEXT,
                        tuple(event.sample_labels),
                    )
                )
            schema_repeat = AxisSpec(
                AxisId("fit.repeat"),
                "repeat",
                REPEAT,
                1,
                (0,),
            )
            schema_point_table = PointTable(sample_count, tuple(point_columns))
        cell_shape = (schema_repeat.size, schema_point_table.row_count)
        value_validity = CellValidity(event.success.reshape(cell_shape))
        error_validity = {
            parameter: CellValidity(
                (event.success & np.isfinite(event.parameter_errors[parameter])).reshape(
                    cell_shape
                )
            )
            for parameter in event.parameter_names
        }
        fit_source_ref = DatasetRevisionRef(
            source.ref.block_id,
            source.ref.stream_generation,
            source.ref.schema_fingerprint,
            DatasetRevision(event.batch_revision),
        )
        coverage = MonitorCoverage(sample_count, sample_count)

        enabled = set(self._fit_output_names(event))
        for parameter in event.parameter_names:
            unit = event.parameter_units[parameter] or None
            schema = DatasetSchema(
                schema_repeat,
                schema_point_table,
                None,
                ValueSchema.scalar(np.dtype("float64"), unit),
            )
            for suffix, values, contract_id, validity in (
                (
                    "",
                    event.parameter_values[parameter],
                    "zlc.selection.fit.parameter",
                    value_validity,
                ),
                (
                    "_err",
                    event.parameter_errors[parameter],
                    "zlc.selection.fit.error",
                    error_validity[parameter],
                ),
            ):
                name = f"{parameter}{suffix}"
                if name not in enabled:
                    continue
                output[name] = self._materialize_fit_vector(
                    source,
                    fit_source_ref,
                    name,
                    values,
                    schema,
                    validity,
                    contract_id,
                    coverage,
                )
        return output

    def _materialize_fit_vector(
        self,
        source: OwnedSnapshot,
        fit_source_ref: DatasetRevisionRef,
        name: str,
        values: np.ndarray,
        schema: DatasetSchema,
        validity: CellValidity,
        contract_id: str,
        coverage: MonitorCoverage,
    ) -> LiveDatasetOutput:
        derived = materialize_derived_dataset(
            fit_source_ref,
            values.reshape(schema.physical_shape),
            schema=schema,
            validity=validity,
            reference_for=lambda derived_schema: self._next_reference(
                source,
                name,
                derived_schema,
                revision=fit_source_ref.revision,
            ),
        )
        return LiveDatasetOutput(
            DatasetOutputDeclaration(
                name,
                contract_id,
                index_by_source=True,
            ),
            derived,
            coverage,
        )
