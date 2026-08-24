"""Plot-neutral projections of immutable ``(R, P, *D)`` datasets.

The data producer is the only authority for point topology.  In particular,
``AxisRef.point_dimension(...)`` resolves only through an explicit
``GridTopology``; repeated values in a ``PointTable`` are never promoted to a
tensor dimension here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import math
from numbers import Integral
from typing import Any, TypeAlias
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from zlc_data import OwnedSnapshot
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID

from .data_contract import (
    DEFAULT_UNITS,
    Unit,
    UnitRegistry,
    descriptor_from_axis,
    descriptor_from_point_column,
    descriptor_from_topology,
    point_column,
    resolve_unit,
    schema_data_axes,
    schema_point_count,
    schema_repeat_count,
    schema_shape,
    schema_value_unit,
    snapshot_generation,
    snapshot_revision,
    snapshot_schema,
    snapshot_validity,
    snapshot_values,
    topology_position,
)

from .kinds import AxisDomain, AxisRef
from .specs import CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot, Reduction


class DataViewError(ValueError):
    """Base error for an invalid plot projection request."""


class AxisResolutionError(DataViewError):
    """A requested :class:`AxisRef` is not declared by the dataset."""


class TopologyRequiredError(AxisResolutionError):
    """A point-dimension request has no producer-declared topology."""


def _readonly(values: ArrayLike, *, dtype: Any | None = None) -> NDArray[Any]:
    array = np.asarray(values, dtype=dtype)
    if array.flags.writeable:
        array = np.array(array, copy=True)
    else:
        array = array.view()
    array.setflags(write=False)
    return array


def _require_same_shape(left: NDArray[Any], right: NDArray[Any], what: str) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{what} arrays must have identical shapes")


def _source_revisions(values: Iterable[int], revision: int) -> tuple[int, ...]:
    revisions = tuple(values) or (revision,)
    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 0
        for value in revisions
    ):
        raise TypeError("source_revisions must contain non-negative integers")
    return tuple(int(value) for value in revisions)


@dataclass(frozen=True, slots=True)
class QuantityArray:
    """Canonical and display representations of the same physical values."""

    canonical: NDArray[Any] | ArrayLike
    display: NDArray[Any] | ArrayLike
    canonical_unit: Unit
    display_unit: Unit
    label: str

    def __post_init__(self) -> None:
        canonical = _readonly(self.canonical)
        display = _readonly(self.display)
        _require_same_shape(canonical, display, "quantity")
        if not isinstance(self.canonical_unit, Unit) or not isinstance(
            self.display_unit, Unit
        ):
            raise TypeError("quantity units must be Unit objects")
        if not self.canonical_unit.compatible_with(self.display_unit):
            raise ValueError("canonical and display units must be compatible")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("quantity label must be a non-empty string")
        object.__setattr__(self, "canonical", canonical)
        object.__setattr__(self, "display", display)


@dataclass(frozen=True, slots=True)
class CoordinateArray:
    """A resolved coordinate broadcast over every physical dataset sample."""

    ref: AxisRef
    canonical: NDArray[Any] | ArrayLike
    display: NDArray[Any] | ArrayLike
    indices: NDArray[np.int64] | ArrayLike
    canonical_unit: Unit
    display_unit: Unit
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.ref, AxisRef):
            raise TypeError("ref must be AxisRef")
        canonical = _readonly(self.canonical)
        display = _readonly(self.display)
        indices = _readonly(self.indices, dtype=np.int64)
        _require_same_shape(canonical, display, "coordinate")
        _require_same_shape(canonical, indices, "coordinate/index")
        if not isinstance(self.canonical_unit, Unit) or not isinstance(
            self.display_unit, Unit
        ):
            raise TypeError("coordinate units must be Unit objects")
        if not self.canonical_unit.compatible_with(self.display_unit):
            raise ValueError("coordinate units must be compatible")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("coordinate label must be a non-empty string")
        object.__setattr__(self, "canonical", canonical)
        object.__setattr__(self, "display", display)
        object.__setattr__(self, "indices", indices)


@dataclass(frozen=True, slots=True)
class SampleProjection:
    revision: int
    generation: str
    shape: tuple[int, ...]
    value: QuantityArray
    valid_mask: NDArray[np.bool_] | ArrayLike

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        valid = _readonly(self.valid_mask, dtype=np.bool_)
        if self.value.canonical.shape != shape or valid.shape != shape:
            raise ValueError("sample value and validity must match projection shape")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "valid_mask", valid)


@dataclass(frozen=True, slots=True)
class AxisValue:
    ref: AxisRef
    index: int | None
    canonical: Any
    display: Any
    label: str


@dataclass(frozen=True, slots=True)
class RollingSample:
    """One scalar-per-group projection owned by a rolling plot."""

    revision: int
    generation: str
    values: NDArray[Any] | ArrayLike
    valid: NDArray[np.bool_] | ArrayLike
    group_keys: tuple[tuple[AxisValue, ...], ...] = ()
    #: Standard error of each MEAN entry over what this shot pooled, or
    #: None when uncertainty was not requested.  Canonical-only, like the
    #: curve companion.
    sem: NDArray[np.float64] | ArrayLike | None = None

    def __post_init__(self) -> None:
        values = _readonly(self.values)
        valid = _readonly(self.valid, dtype=np.bool_)
        if values.ndim != 1 or valid.shape != values.shape:
            raise ValueError("rolling sample values and validity must be one-dimensional")
        keys = tuple(tuple(item) for item in self.group_keys)
        if len(keys) != values.size:
            raise ValueError("rolling sample group keys must match value count")
        if self.sem is not None:
            sem = _readonly(self.sem, dtype=np.float64)
            if sem.shape != values.shape:
                raise ValueError("rolling sample sem must match value count")
            object.__setattr__(self, "sem", sem)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "group_keys", keys)


@dataclass(frozen=True, slots=True)
class CurveSeries:
    x: QuantityArray
    y: QuantityArray
    valid: NDArray[np.bool_] | ArrayLike
    counts: NDArray[np.int64] | ArrayLike
    #: Standard error of each MEAN-reduced point, in the y CANONICAL unit,
    #: or None when the projection was not asked for uncertainty.  Stored
    #: canonical-only on purpose: an affine display conversion (an offset
    #: unit) is wrong for a difference-like quantity, so consumers convert
    #: the y±sem BOUNDS, never sem itself.
    sem: NDArray[np.float64] | ArrayLike | None = None
    #: One display name per plotted x position when the axis declares
    #: coordinate labels (a pair axis, a model axis), or None for numeric
    #: axes.  The renderer puts these on the ticks so the axis reads the
    #: same names the legend, hover and scope already use.
    x_labels: tuple[str, ...] | None = None
    group_key: tuple[AxisValue, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        valid = _readonly(self.valid, dtype=np.bool_)
        counts = _readonly(self.counts, dtype=np.int64)
        if self.sem is not None:
            sem = _readonly(self.sem, dtype=np.float64)
            if sem.shape != valid.shape:
                raise ValueError("curve sem must match the point shape")
            object.__setattr__(self, "sem", sem)
        if self.x_labels is not None:
            labels = tuple(str(item) for item in self.x_labels)
            if len(labels) != int(valid.shape[0]):
                raise ValueError("curve x labels must match the point count")
            object.__setattr__(self, "x_labels", labels)
        if self.x.canonical.ndim != 1 or self.y.canonical.ndim != 1:
            raise ValueError("curve x and y must be one-dimensional")
        if not (
            self.x.canonical.shape
            == self.y.canonical.shape
            == valid.shape
            == counts.shape
        ):
            raise ValueError("curve arrays must have identical shapes")
        key = tuple(self.group_key)
        if any(not isinstance(value, AxisValue) for value in key):
            raise TypeError("group_key must contain AxisValue objects")
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "group_key", key)


@dataclass(frozen=True, slots=True)
class CurveData:
    revision: int
    generation: str
    x_ref: AxisRef
    group_by: tuple[AxisRef, ...]
    series: tuple[CurveSeries, ...]
    source_revisions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        group_by = tuple(self.group_by)
        series = tuple(self.series)
        if any(not isinstance(ref, AxisRef) for ref in group_by):
            raise TypeError("group_by must contain AxisRef objects")
        if any(not isinstance(item, CurveSeries) for item in series):
            raise TypeError("series must contain CurveSeries objects")
        revisions = _source_revisions(self.source_revisions, self.revision)
        object.__setattr__(self, "group_by", group_by)
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "source_revisions", revisions)


@dataclass(frozen=True, slots=True)
class ImageData:
    revision: int
    generation: str
    x_ref: AxisRef
    y_ref: AxisRef
    x: QuantityArray
    y: QuantityArray
    z: QuantityArray
    valid: NDArray[np.bool_] | ArrayLike
    source_revisions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        valid = _readonly(self.valid, dtype=np.bool_)
        expected = (self.y.canonical.size, self.x.canonical.size)
        if self.x.canonical.ndim != 1 or self.y.canonical.ndim != 1:
            raise ValueError("image x and y coordinates must be one-dimensional")
        if self.z.canonical.shape != expected or valid.shape != expected:
            raise ValueError("image z and validity do not match its coordinate grid")
        object.__setattr__(self, "valid", valid)
        object.__setattr__(
            self,
            "source_revisions",
            _source_revisions(self.source_revisions, self.revision),
        )


@dataclass(frozen=True, slots=True)
class HistogramData:
    revision: int
    generation: str
    edges: QuantityArray
    centers: QuantityArray
    counts: NDArray[np.int64] | ArrayLike
    source_revisions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        counts = _readonly(self.counts, dtype=np.int64)
        if self.edges.canonical.ndim != 1 or self.centers.canonical.ndim != 1:
            raise ValueError("histogram edges and centers must be one-dimensional")
        if self.edges.canonical.size != counts.size + 1:
            raise ValueError("histogram requires one more edge than count")
        if self.centers.canonical.size != counts.size:
            raise ValueError("histogram centers and counts must have equal length")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(
            self,
            "source_revisions",
            _source_revisions(self.source_revisions, self.revision),
        )


FacetPayload: TypeAlias = CurveData | ImageData | HistogramData


@dataclass(frozen=True, slots=True)
class FacetCell:
    facet_index: int | None
    facet_value_canonical: Any
    facet_value_display: Any
    label: str
    payload: FacetPayload

    def __post_init__(self) -> None:
        if not isinstance(self.payload, (CurveData, ImageData, HistogramData)):
            raise TypeError("facet payload must be homogeneous plot data")


@dataclass(frozen=True, slots=True)
class FacetData:
    revision: int
    generation: str
    spec: FacetGridPlot
    cells: tuple[FacetCell, ...]
    source_revisions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        if not isinstance(self.spec, FacetGridPlot):
            raise TypeError("spec must be FacetGridPlot")
        if any(not isinstance(cell, FacetCell) for cell in cells):
            raise TypeError("cells must contain FacetCell objects")
        expected = {
            CurvePlot: CurveData,
            ImagePlot: ImageData,
            HistogramPlot: HistogramData,
        }[type(self.spec.cell)]
        if any(not isinstance(cell.payload, expected) for cell in cells):
            raise ValueError("all facet cells must use the declared homogeneous kind")
        if any(
            cell.payload.revision != self.revision
            or cell.payload.generation != self.generation
            for cell in cells
        ):
            raise ValueError("facet cell revisions must match their FacetData revision")
        object.__setattr__(self, "cells", cells)
        revisions = _source_revisions(self.source_revisions, self.revision)
        if any(
            tuple(getattr(cell.payload, "source_revisions", (self.revision,)))
            != revisions
            for cell in cells
        ):
            raise ValueError("facet cells must share FacetData source revisions")
        object.__setattr__(self, "source_revisions", revisions)


@dataclass(frozen=True)
class _FactoredPlanes:
    """The folded lattice moments a curve or a facet grid assembles from."""

    x_quantity: "QuantityArray"
    x_labels: tuple[str, ...] | None
    row_domains: tuple
    row_sizes: tuple[int, ...]
    row_presence: NDArray[np.bool_]
    group_domains: tuple
    group_sizes: tuple[int, ...]
    y_plane: NDArray[np.float64]
    counts_plane: NDArray[np.int64]
    sem_plane: NDArray[np.float64] | None


@dataclass(frozen=True, slots=True)
class _ResolvedAxis:
    coordinate: CoordinateArray
    domain_canonical: NDArray[Any]
    domain_display: NDArray[Any]
    coordinate_labels: tuple[str, ...] | None
    declared_domain: bool
    #: Which tensor dimension this axis IS, as the resolver worked it out.
    #: Recomputing it elsewhere by a different rule is what put a supported way
    #: of naming an axis onto the slow path.
    dimension: int


class _Domain:
    """Grouping codes plus the domain's canonical/display value planes.

    ``values`` -- the labelled AxisValue objects -- materializes on first
    access.  Group and facet consumers read a handful of them; a continuous
    curve axis has one distinct value PER POINT, and building a Python
    scalar pair plus a formatted label for each of 100k points every frame
    was the hottest loop of the whole curve payload path.  The arrays carry
    everything the hot consumers actually use.
    """

    __slots__ = ("canonical", "display", "codes", "_build_values", "_values")

    def __init__(
        self,
        canonical: NDArray[Any],
        display: NDArray[Any],
        codes: NDArray[np.int64],
        build_values: Callable[[], tuple[AxisValue, ...]],
    ) -> None:
        self.canonical = canonical
        self.display = display
        self.codes = codes
        self._build_values = build_values
        self._values: tuple[AxisValue, ...] | None = None

    @property
    def size(self) -> int:
        return int(self.canonical.size)

    @property
    def values(self) -> tuple[AxisValue, ...]:
        cached = self._values
        if cached is None:
            cached = self._build_values()
            self._values = cached
        return cached


class DataView:
    """Resolve and aggregate one immutable dataset revision for renderers."""

    __slots__ = (
        "_snapshot",
        "_schema",
        "_axis_display_units",
        "_unit_registry",
        "_samples",
        "_axis_cache",
        "_flat_cache",
        "_pooled_cache",
        "_positions_cache",
        "_domain_carry",
        "_unit_registry_revision",
    )

    def __init__(
        self,
        snapshot: OwnedSnapshot,
        *,
        axis_display_units: Mapping[AxisRef, str | Unit] | None = None,
        value_display_unit: str | Unit | None = None,
        unit_registry: UnitRegistry | None = None,
        inherit_domains_from: "DataView | None" = None,
    ) -> None:
        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
        values = snapshot_values(snapshot)
        schema = snapshot_schema(snapshot)
        if values.dtype.kind == "c":
            raise DataViewError(
                "complex dataset values require an explicit real-valued transform "
                "before plotting"
            )
        if axis_display_units is not None and not isinstance(
            axis_display_units, Mapping
        ):
            raise TypeError("axis_display_units must be a mapping or None")
        overrides = {} if axis_display_units is None else dict(axis_display_units)
        if any(not isinstance(ref, AxisRef) for ref in overrides):
            raise TypeError("axis_display_units keys must be AxisRef objects")
        if unit_registry is not None and not isinstance(unit_registry, UnitRegistry):
            raise TypeError("unit_registry must be UnitRegistry or None")
        registry = unit_registry or DEFAULT_UNITS
        value_canonical_unit = schema_value_unit(schema, registry)
        value_display = (
            value_canonical_unit
            if value_display_unit is None
            else resolve_unit(value_display_unit, registry)
        )
        if not value_canonical_unit.compatible_with(value_display):
            raise DataViewError("value display unit is incompatible with dataset values")
        value_canonical = values
        value_displayed = value_canonical_unit.convert_value_to(
            value_canonical, value_display
        )
        # Integer and boolean samples are finite by construction.  Reuse the
        # snapshot's immutable validity plane instead of allocating two more
        # megapixel boolean arrays for the ordinary camera path.
        if value_canonical.dtype.kind in "biu":
            valid = snapshot_validity(snapshot)
        else:
            validity = snapshot_validity(snapshot)
            finite = np.isfinite(value_canonical)
            if bool(finite.all()):
                # All-finite floats keep the snapshot's validity plane --
                # usually the stride-0 all-true broadcast, which the
                # reductions recognise in O(1) instead of scanning a
                # 20-megabyte merged mask.
                valid = validity
            elif _stride_zero_all_true(validity):
                valid = finite
            else:
                valid = validity & finite
        self._snapshot = snapshot
        self._schema = schema
        self._axis_display_units = overrides
        self._unit_registry = registry
        self._unit_registry_revision = registry.revision
        self._axis_cache: dict[AxisRef, _ResolvedAxis] = {}
        self._flat_cache: dict[
            AxisRef,
            tuple[NDArray[Any], NDArray[np.int64]],
        ] = {}
        self._pooled_cache: NDArray[Any] | None = None
        self._positions_cache: NDArray[np.int64] | None = None
        #: Whole-dataset domains carried from the PREVIOUS revision's view.
        #: A domain is a fact about the coordinate plane alone -- np.unique
        #: over a million-point x column costs ~60 ms and its input rarely
        #: changes between live revisions -- so each entry keeps the exact
        #: coordinate array it was derived from and is reused only after an
        #: equality check against the current one (~1 ms for the same
        #: million points).  Display-unit context must match exactly.
        self._domain_carry: dict[AxisRef, tuple[NDArray[Any], _Domain]] = {}
        if (
            inherit_domains_from is not None
            and isinstance(inherit_domains_from, DataView)
            and inherit_domains_from._axis_display_units == overrides
            and inherit_domains_from._unit_registry is registry
            and inherit_domains_from._unit_registry_revision
            == self._unit_registry_revision
        ):
            self._domain_carry = inherit_domains_from._domain_carry
        self._samples = SampleProjection(
            revision=snapshot_revision(snapshot),
            generation=snapshot_generation(snapshot),
            shape=schema_shape(schema),
            value=QuantityArray(
                canonical=value_canonical,
                display=value_displayed,
                canonical_unit=value_canonical_unit,
                display_unit=value_display,
                label="value",
            ),
            valid_mask=valid,
        )
        # Fail early for misspelled or undeclared override keys.
        for ref in overrides:
            self._resolve(ref)

    @property
    def samples(self) -> SampleProjection:
        return self._samples

    @property
    def has_primary_index(self) -> bool:
        return any(
            column.coordinate_id == PRIMARY_INDEX_AXIS_ID
            for column in self._schema.point_table.columns
        )

    @staticmethod
    def _primary_index_ref() -> AxisRef:
        return AxisRef.point(PRIMARY_INDEX_AXIS_ID.value)

    def coordinate(self, ref: AxisRef) -> CoordinateArray:
        return self._resolve(ref).coordinate

    def validate_curve(
        self,
        x: AxisRef,
        *,
        group_by: Iterable[AxisRef] = (),
    ) -> None:
        """Check a curve projection without computing it.

        This is the single validation authority shared by :meth:`curve` and
        the semantic feasibility probe: whatever passes here projects, and
        whatever projects passed here.

        An axis this projection does not name -- repeat, an unplotted scan
        dimension, a dense data axis -- is not a defect: it pools under the
        spec's authored ``reduction``, which is the one rule R, P and D all
        obey.  "You are averaging your whole scan into one number" is a hint
        for the editor to give, not a construction-time refusal.
        """

        self._validate_curve_shape(x, tuple(group_by))

    def _validate_curve_shape(
        self,
        x: AxisRef,
        groups: tuple[AxisRef, ...],
    ) -> tuple[AxisRef, ...]:
        if not isinstance(x, AxisRef):
            raise TypeError("x must be AxisRef")
        _validate_refs(groups, "group_by")
        if len(set(groups)) != len(groups):
            raise DataViewError("group_by axes must be unique")
        if x in groups:
            raise DataViewError("curve x axis cannot also be a group axis")
        # Both projection kernels plot x as a number, so a text coordinate is
        # a REFUSAL, not a build-time surprise: this used to live at the two
        # build sites only, which made ``validate_curve`` accept specs that
        # then raised on their first draw.
        _require_real_numeric(self._resolve(x).coordinate.canonical, x)
        for ref in groups:
            self._resolve(ref)
        return groups

    def curve(
        self,
        x: AxisRef,
        *,
        group_by: Iterable[AxisRef] = (),
        aggregation: Reduction = Reduction.MEAN,
        uncertainty: bool = False,
    ) -> CurveData:
        groups = tuple(group_by)
        self.validate_curve(x, group_by=groups)
        aggregation = _validate_aggregation(aggregation)
        if uncertainty:
            # The standard error IS the spread of the samples the MEAN pooled:
            # for any other reduction the quantity is undefined, and pretending
            # otherwise would attach a number with no meaning to the plot.
            if aggregation is not Reduction.MEAN:
                raise ValueError(
                    "uncertainty is defined for Reduction.MEAN only, "
                    f"not {aggregation.value!r}"
                )
            if self._samples.value.canonical.dtype.kind == "c":
                raise ValueError("uncertainty is undefined for complex values")
        dense = self._dense_data_curve(x, groups, aggregation, uncertainty)
        if dense is not None:
            return dense
        factored = self._factored_curve(x, groups, aggregation, uncertainty)
        if factored is not None:
            return factored
        positions = self._all_positions()
        return self._curve_from_positions(
            x, positions, groups, aggregation, uncertainty
        )

    def _dense_data_curve(
        self,
        x: AxisRef,
        groups: tuple[AxisRef, ...],
        aggregation: Reduction,
        uncertainty: bool = False,
    ) -> CurveData | None:
        """Project one declared dense data axis without materializing samples.

        The narrow twin of ``_dense_data_image``: an ungrouped curve over a
        declared DATA axis with finite, strictly increasing coordinates is a
        straight tensor reduction along every other dimension -- the same
        operation the generic path performs by flattening every sample into a
        (position, value) pair and aggregating per unique coordinate code,
        which on a camera frame is millions of pairs, a sort, and a Python
        loop of per-bucket reductions.  Groups, point domains, facet subsets
        and unordered coordinates keep the generic algorithm.
        """

        if groups or x.domain is not AxisDomain.DATA:
            return None
        try:
            x_resolved = self._resolve(x)
        except AxisResolutionError:
            # Let the normal resolver produce the public missing-axis error.
            return None
        x_canonical = np.asarray(x_resolved.domain_canonical)
        if not np.all(_finite_coordinate(x_canonical)):
            return None
        if x_canonical.size > 1 and not np.all(np.diff(x_canonical) > 0):
            # The generic path aggregates over SORTED unique coordinates;
            # only a strictly increasing declared domain is bit-identical.
            return None

        return self._dense_curve_data(
            x,
            x_resolved,
            self._samples.value.canonical,
            self._samples.valid_mask,
            aggregation,
            uncertainty,
        )

    def _dense_curve_data(
        self,
        x: AxisRef,
        x_resolved: "_ResolvedAxis",
        values: NDArray[Any],
        usable: NDArray[np.bool_],
        aggregation: Reduction,
        uncertainty: bool = False,
    ) -> CurveData:
        """One dense curve out of one (possibly row-sliced) value tensor."""

        x_canonical = np.asarray(x_resolved.domain_canonical)
        nx = int(x_canonical.size)
        moved = np.reshape(
            np.moveaxis(values, x_resolved.dimension, -1), (-1, nx), order="C"
        )
        moved_usable = np.reshape(
            np.moveaxis(usable, x_resolved.dimension, -1), (-1, nx), order="C"
        )
        y, counts = _masked_leading_reduce(moved, moved_usable, aggregation)
        y = np.asarray(y, dtype=np.float64)
        sem = None
        if uncertainty:
            # The SEM is the SAME reduction run over the squares: no second
            # kernel, no binomial special case -- for a boolean column the
            # sample spread sqrt(p(1-p)) IS the binomial spread.
            squares = np.square(moved.astype(np.float64, copy=False))
            mean_sq, _sq_counts = _masked_leading_reduce(
                squares, moved_usable, Reduction.MEAN
            )
            sem = _sem_from_moments(y, np.asarray(mean_sq, np.float64), counts)
        valid = (counts > 0) & np.isfinite(y)
        y_display = self._samples.value.canonical_unit.convert_value_to(
            y, self._samples.value.display_unit
        )
        series = CurveSeries(
            x=QuantityArray(
                x_canonical,
                np.asarray(x_resolved.domain_display),
                x_resolved.coordinate.canonical_unit,
                x_resolved.coordinate.display_unit,
                x_resolved.coordinate.label,
            ),
            x_labels=_axis_coordinate_labels(x_resolved, x_canonical),
            y=QuantityArray(
                y,
                y_display,
                self._samples.value.canonical_unit,
                self._samples.value.display_unit,
                self._samples.value.label,
            ),
            valid=valid,
            counts=counts,
            sem=sem,
            group_key=(),
            label=self._samples.value.label,
        )
        return CurveData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            x_ref=x,
            group_by=(),
            series=(series,),
        )

    def _factored_curve(
        self,
        x: AxisRef,
        groups: tuple[AxisRef, ...],
        aggregation: Reduction,
        uncertainty: bool = False,
    ) -> CurveData | None:
        """The lattice fast path: moments by tensor axes, then a tiny fold.

        Most bench signals are LARGE OVERALL but factored into many modest
        axes -- (repeat) x (scan rows) x (frame) x (site) -- and a curve
        over such a block never needed per-sample bucket codes: every
        pooled dimension is a tensor axis, so the moments (sums, counts,
        squares) reduce through plain masked axis-reductions in one pass
        over the data, and only a (rows x series)-sized residue is left to
        fold by x-coordinate -- thousands of entries where the generic
        path built codes, gathers and buckets for millions.

        Coverage: x determined by the point ROW (a scan column or topology
        dimension); groups over DATA axes or the repeat axis.  Everything
        else -- FIRST (whose result depends on the exact sample order the
        pre-reduction destroys), complex values (np.sum(where=) rejects
        them), point-domain groups, undeclared shapes -- keeps the generic
        path, whose outputs this path must match series for series (the
        oracle tests hold both to that).
        """

        planes = self._factored_planes((x,), groups, aggregation, uncertainty)
        if planes is None:
            return None
        series = self._series_from_planes(
            planes,
            planes.group_domains,
            planes.group_sizes,
            planes.y_plane,
            planes.counts_plane,
            planes.sem_plane,
        )
        return CurveData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            x_ref=x,
            group_by=groups,
            series=series,
        )

    def _factored_facet(
        self,
        spec: FacetGridPlot,
        uncertainty: bool,
    ) -> FacetData | None:
        """Every curve cell of a lattice facet from ONE pass over the data.

        A facet over a DATA axis (or the repeat axis) is just one more kept
        dimension: prepending it to the cell's own groups makes the whole
        grid one factored-curve computation, and each cell is a column
        slice of the result -- where the generic path re-ran the full
        per-sample aggregation once PER CELL.  A facet over a point-domain
        axis or a non-curve cell keeps its existing paths.  Cell for cell
        the output must match the generic facet (the oracle tests hold
        both to that); for a tensor facet every cell shares the global
        row set, so the per-cell x domains agree by construction.
        """

        cell = spec.cell
        row_domains = (AxisDomain.POINT_COORDINATE, AxisDomain.POINT_DIMENSION)
        if (
            isinstance(cell, ImagePlot)
            and cell.x.domain in row_domains
            and cell.y.domain in row_domains
        ):
            return self._factored_facet_images(spec, cell)
        if not isinstance(cell, CurvePlot):
            return None
        cell_groups = () if cell.group is None else (cell.group,)
        row_facet = spec.facet.domain in (
            AxisDomain.POINT_COORDINATE,
            AxisDomain.POINT_DIMENSION,
        )
        if row_facet:
            planes = self._factored_planes(
                (spec.facet, cell.x),
                cell_groups,
                cell.reduction,
                uncertainty,
            )
        elif spec.facet.domain in (AxisDomain.DATA, AxisDomain.REPEAT):
            planes = self._factored_planes(
                (cell.x,),
                (spec.facet, *cell_groups),
                cell.reduction,
                uncertainty,
            )
        else:
            return None
        if planes is None:
            return None
        if row_facet:
            facet_domain = planes.row_domains[0]
            facet_size, nx = planes.row_sizes
            cell_domains = planes.group_domains
            cell_sizes = planes.group_sizes
        else:
            facet_domain = planes.group_domains[0]
            facet_size = planes.group_sizes[0]
            cell_domains = planes.group_domains[1:]
            cell_sizes = planes.group_sizes[1:]
        cell_combos = 1
        for size in cell_sizes:
            cell_combos *= size
        cells: list[FacetCell] = []
        for facet_index in range(facet_size):
            if row_facet:
                # A cell over a scan dimension owns only the x coordinates
                # that co-occur with its facet value among the rows -- the
                # generic path's per-cell used-set, read off the presence
                # plane instead of re-deriving codes per cell.
                rows_window = slice(facet_index * nx, (facet_index + 1) * nx)
                used = planes.row_presence[rows_window]
                if not bool(used.any()):
                    continue
                x_quantity = QuantityArray(
                    np.asarray(planes.x_quantity.canonical)[used],
                    np.asarray(planes.x_quantity.display)[used],
                    planes.x_quantity.canonical_unit,
                    planes.x_quantity.display_unit,
                    planes.x_quantity.label,
                )
                x_labels = (
                    None
                    if planes.x_labels is None
                    else tuple(
                        label
                        for label, keep in zip(planes.x_labels, used)
                        if keep
                    )
                )
                series = self._series_from_planes(
                    planes,
                    cell_domains,
                    cell_sizes,
                    planes.y_plane[rows_window][used],
                    planes.counts_plane[rows_window][used],
                    (
                        None
                        if planes.sem_plane is None
                        else planes.sem_plane[rows_window][used]
                    ),
                    x_quantity=x_quantity,
                    x_labels=x_labels,
                )
            else:
                window = slice(
                    facet_index * cell_combos, (facet_index + 1) * cell_combos
                )
                series = self._series_from_planes(
                    planes,
                    cell_domains,
                    cell_sizes,
                    planes.y_plane[:, window],
                    planes.counts_plane[:, window],
                    (
                        None
                        if planes.sem_plane is None
                        else planes.sem_plane[:, window]
                    ),
                )
            payload = CurveData(
                revision=self._samples.revision,
                generation=self._samples.generation,
                x_ref=cell.x,
                group_by=cell_groups,
                series=series,
            )
            facet_value = facet_domain.values[facet_index]
            cells.append(
                FacetCell(
                    facet_index=len(cells),
                    facet_value_canonical=facet_value.canonical,
                    facet_value_display=facet_value.display,
                    label=facet_value.label,
                    payload=payload,
                )
            )
        return FacetData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            spec=spec,
            cells=tuple(cells),
        )

    def _factored_facet_images(
        self,
        spec: FacetGridPlot,
        cell: ImagePlot,
    ) -> FacetData | None:
        """Every heatmap cell of a lattice facet from ONE pass over the data.

        A facet of scan heatmaps is the heatmap computation with one more
        key: over a DATA/repeat axis the facet is a kept tensor dimension
        and each cell is a column of the folded plane; over a scan
        dimension the facet joins the combined row key and each cell is a
        row window, compressed to its own used axis sets exactly as the
        generic per-cell domains are.  The oracle tests hold every cell
        pixel for pixel to the generic facet.
        """

        row_facet = spec.facet.domain in (
            AxisDomain.POINT_COORDINATE,
            AxisDomain.POINT_DIMENSION,
        )
        if row_facet:
            planes = self._factored_planes(
                (spec.facet, cell.y, cell.x), (), cell.reduction, False
            )
        elif spec.facet.domain in (AxisDomain.DATA, AxisDomain.REPEAT):
            planes = self._factored_planes(
                (cell.y, cell.x), (spec.facet,), cell.reduction, False
            )
        else:
            return None
        if planes is None:
            return None
        if row_facet:
            facet_domain, y_domain, x_domain = planes.row_domains
            facet_size, ny, nx = planes.row_sizes
        else:
            facet_domain = planes.group_domains[0]
            facet_size = planes.group_sizes[0]
            y_domain, x_domain = planes.row_domains
            ny, nx = planes.row_sizes
        cells: list[FacetCell] = []
        for facet_index in range(facet_size):
            if row_facet:
                window = slice(
                    facet_index * ny * nx, (facet_index + 1) * ny * nx
                )
                presence = planes.row_presence[window].reshape(ny, nx)
                if not bool(presence.any()):
                    continue
                payload = self._image_from_planes(
                    cell.x,
                    cell.y,
                    x_domain,
                    y_domain,
                    planes.y_plane[window].reshape(ny, nx),
                    planes.counts_plane[window].reshape(ny, nx),
                    used_y=presence.any(axis=1),
                    used_x=presence.any(axis=0),
                )
            else:
                payload = self._image_from_planes(
                    cell.x,
                    cell.y,
                    x_domain,
                    y_domain,
                    planes.y_plane[:, facet_index].reshape(ny, nx),
                    planes.counts_plane[:, facet_index].reshape(ny, nx),
                )
            facet_value = facet_domain.values[facet_index]
            cells.append(
                FacetCell(
                    facet_index=len(cells),
                    facet_value_canonical=facet_value.canonical,
                    facet_value_display=facet_value.display,
                    label=facet_value.label,
                    payload=payload,
                )
            )
        return FacetData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            spec=spec,
            cells=tuple(cells),
        )

    def _factored_planes(
        self,
        row_refs: tuple[AxisRef, ...],
        groups: tuple[AxisRef, ...],
        aggregation: Reduction,
        uncertainty: bool,
    ) -> "_FactoredPlanes | None":
        """The ONE lattice computation behind curves, heatmaps and facets.

        ``row_refs`` are the row-determined axes the fold keys on, outer
        first: ``(x,)`` for a curve, ``(y, x)`` for a scan heatmap,
        ``(facet, x)`` for a facet over a scan dimension.  Their combined
        code is one fold key, so every consumer of a row-shaped bucket
        rides this single kernel instead of growing its own.
        """

        if aggregation not in (
            Reduction.MEAN,
            Reduction.SUM,
            Reduction.MIN,
            Reduction.MAX,
        ):
            return None
        values = self._samples.value.canonical
        if values.dtype.kind == "c":
            return None
        if not row_refs or any(
            ref.domain
            not in (
                AxisDomain.POINT_COORDINATE,
                AxisDomain.POINT_DIMENSION,
            )
            for ref in row_refs
        ):
            return None
        x = row_refs[-1]
        try:
            x_resolved = self._resolve(x)
            group_resolved = tuple(self._resolve(ref) for ref in groups)
        except AxisResolutionError:
            return None
        shape = values.shape
        data_size = 1
        for size in shape[2:]:
            data_size *= int(size)
        kept_dims: list[int] = []
        for ref, resolved in zip(groups, group_resolved):
            if ref.domain is AxisDomain.REPEAT:
                dimension = 0
            elif ref.domain is AxisDomain.DATA:
                dimension = int(resolved.dimension)
            else:
                return None
            if dimension in kept_dims:
                return None
            kept_dims.append(dimension)

        # One representative element per row / per group coordinate puts
        # the existing domain machinery (used-set compression, labels,
        # units) to work on arrays the size of the AXIS, not the dataset.
        rows = int(shape[1])
        row_representatives = np.arange(rows, dtype=np.int64) * data_size
        row_domains = tuple(
            self._domain(ref, row_representatives) for ref in row_refs
        )
        row_sizes = tuple(
            int(domain.canonical.size) for domain in row_domains
        )
        if any(size == 0 for size in row_sizes):
            return None
        combined_row_codes = np.zeros(rows, dtype=np.int64)
        row_ok = np.ones(rows, dtype=np.bool_)
        for domain, size in zip(row_domains, row_sizes):
            codes = np.asarray(domain.codes)
            row_ok &= codes >= 0
            combined_row_codes = combined_row_codes * size + np.where(
                codes >= 0, codes, 0
            )
        row_buckets = 1
        for size in row_sizes:
            row_buckets *= size
        x_domain = row_domains[-1]
        # Which combined row keys EXIST among the rows, validity aside --
        # the generic path's per-cell x domains are used-sets over
        # positions, and this is that fact at row scale.
        row_presence = (
            np.bincount(
                combined_row_codes[row_ok], minlength=row_buckets
            )
            > 0
        )
        strides = []
        acc = 1
        for size in reversed(shape):
            strides.insert(0, acc)
            acc *= int(size)
        group_domains = []
        group_orders = []
        for ref, resolved, dimension in zip(groups, group_resolved, kept_dims):
            representatives = np.arange(shape[dimension], dtype=np.int64) * (
                strides[dimension]
            )
            domain = self._domain(ref, representatives)
            codes = np.asarray(domain.codes)
            if (
                domain.canonical.size != shape[dimension]
                or bool((codes < 0).any())
            ):
                # Duplicate or unusable group coordinates would need
                # per-sample codes again: the generic path's business.
                return None
            # The generic path walks series in CODE order (value-sorted for
            # value-derived domains); the tensor dimension is in INDEX
            # order.  This tiny permutation is the bridge.
            order = np.empty(codes.size, dtype=np.int64)
            order[codes] = np.arange(codes.size, dtype=np.int64)
            group_domains.append(domain)
            group_orders.append(order)

        usable = self._samples.valid_mask
        reduce_axes = tuple(
            axis
            for axis in range(values.ndim)
            if axis != 1 and axis not in kept_dims
        )
        # Sums accumulate in float64 exactly as the generic kernel's
        # bincount does, so a uint8 camera frame cannot wrap either way.
        # A hole-free mask (the common live case) takes the plain kernels:
        # masked reductions cost half again as much, and the masked
        # square-sum's 160 MB temporary costs 7x the einsum that replaces
        # it -- einsum reduces v*v in one fused pass with no temporary.
        as_double = values.astype(np.float64, copy=False)
        all_valid = _stride_zero_all_true(usable) or bool(usable.all())
        if all_valid:
            reduced = 1
            for axis in reduce_axes:
                reduced *= int(shape[axis])
            kept_shape = tuple(
                int(shape[axis])
                for axis in range(values.ndim)
                if axis == 1 or axis in kept_dims
            )
            counts_pg = np.full(kept_shape, reduced, dtype=np.int64)
        else:
            counts_pg = np.sum(usable, axis=reduce_axes, dtype=np.int64)
        if aggregation in (Reduction.MEAN, Reduction.SUM):
            if all_valid:
                moments_pg = np.sum(
                    as_double, axis=reduce_axes, dtype=np.float64
                )
            else:
                moments_pg = np.sum(
                    as_double,
                    axis=reduce_axes,
                    where=usable,
                    dtype=np.float64,
                )
        else:
            ufunc = np.min if aggregation is Reduction.MIN else np.max
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if all_valid:
                    moments_pg = ufunc(as_double, axis=reduce_axes)
                else:
                    moments_pg = ufunc(
                        as_double,
                        axis=reduce_axes,
                        where=usable,
                        initial=(
                            np.inf
                            if aggregation is Reduction.MIN
                            else -np.inf
                        ),
                    )
        squares_pg = None
        if uncertainty:
            if all_valid:
                letters = "abcdefghijklmnopqrstuvwxyz"[: values.ndim]
                output = "".join(
                    letters[axis]
                    for axis in range(values.ndim)
                    if axis == 1 or axis in kept_dims
                )
                squares_pg = np.einsum(
                    f"{letters},{letters}->{output}", as_double, as_double
                )
            else:
                squares_pg = np.sum(
                    np.square(as_double),
                    axis=reduce_axes,
                    where=usable,
                    dtype=np.float64,
                )

        # The reductions keep the surviving dims in ORIGINAL tensor order
        # (the repeat dim precedes the rows dim when it is grouped); the
        # fold and the series walk both speak (rows, *groups-as-given).
        remaining = sorted([1, *kept_dims])
        permutation = [remaining.index(1)] + [
            remaining.index(dimension) for dimension in kept_dims
        ]
        def to_groups_order(plane: NDArray[Any]) -> NDArray[Any]:
            return np.transpose(plane, permutation)

        group_sizes = tuple(int(shape[d]) for d in kept_dims)
        combos = 1
        for size in group_sizes:
            combos *= size

        def code_ordered(plane: NDArray[Any]) -> NDArray[Any]:
            plane = to_groups_order(plane)
            for position, order in enumerate(group_orders):
                plane = np.take(plane, order, axis=1 + position)
            return plane.reshape(rows, combos)

        counts_pg = code_ordered(counts_pg)
        moments_pg = code_ordered(moments_pg)
        if squares_pg is not None:
            squares_pg = code_ordered(squares_pg)

        # Fold the residue by the combined row key with the SAME grouped
        # kernel the generic path uses, at (rows x series) scale instead
        # of samples.
        fold_codes = np.where(
            row_ok[:, None],
            combined_row_codes[:, None] * combos + np.arange(combos)[None, :],
            -1,
        ).reshape(-1)
        buckets = row_buckets * combos
        counts_fold, _ = _aggregate_by_codes(
            counts_pg.reshape(-1).astype(np.float64),
            np.ones(fold_codes.shape, dtype=np.bool_),
            fold_codes,
            buckets,
            Reduction.SUM,
        )
        counts = np.nan_to_num(counts_fold, nan=0.0).astype(np.int64)
        if aggregation in (Reduction.MEAN, Reduction.SUM):
            sums_fold, _ = _aggregate_by_codes(
                moments_pg.reshape(-1),
                np.ones(fold_codes.shape, dtype=np.bool_),
                fold_codes,
                buckets,
                Reduction.SUM,
            )
            sums_fold = np.nan_to_num(sums_fold, nan=0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                y_flat = (
                    sums_fold / counts
                    if aggregation is Reduction.MEAN
                    else sums_fold
                )
            y_flat = np.where(counts > 0, y_flat, np.nan)
        else:
            y_flat, _ = _aggregate_by_codes(
                moments_pg.reshape(-1),
                (counts_pg > 0).reshape(-1),
                fold_codes,
                buckets,
                aggregation,
            )
            y_flat = np.where(counts > 0, y_flat, np.nan)
        sem_flat = None
        if squares_pg is not None:
            sq_fold, _ = _aggregate_by_codes(
                squares_pg.reshape(-1),
                np.ones(fold_codes.shape, dtype=np.bool_),
                fold_codes,
                buckets,
                Reduction.SUM,
            )
            sq_fold = np.nan_to_num(sq_fold, nan=0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_sq = np.where(counts > 0, sq_fold / counts, np.nan)
            sem_flat = _sem_from_moments(
                np.asarray(y_flat, np.float64), mean_sq, counts
            )

        x_canonical = np.asarray(x_domain.canonical)
        return _FactoredPlanes(
            x_quantity=QuantityArray(
                x_canonical,
                np.asarray(x_domain.display),
                x_resolved.coordinate.canonical_unit,
                x_resolved.coordinate.display_unit,
                x_resolved.coordinate.label,
            ),
            x_labels=_axis_coordinate_labels(x_resolved, x_canonical),
            row_domains=row_domains,
            row_sizes=row_sizes,
            row_presence=row_presence,
            group_domains=tuple(group_domains),
            group_sizes=group_sizes,
            y_plane=np.asarray(y_flat, np.float64).reshape(row_buckets, combos),
            counts_plane=counts.reshape(row_buckets, combos),
            sem_plane=(
                None
                if sem_flat is None
                else sem_flat.reshape(row_buckets, combos)
            ),
        )

    def _series_from_planes(
        self,
        planes: "_FactoredPlanes",
        group_domains: tuple,
        group_sizes: tuple[int, ...],
        y_plane: NDArray[np.float64],
        counts_plane: NDArray[np.int64],
        sem_plane: NDArray[np.float64] | None,
        x_quantity: "QuantityArray | None" = None,
        x_labels: tuple[str, ...] | None = None,
    ) -> tuple[CurveSeries, ...]:
        """Column slices of the folded planes, one CurveSeries each."""

        if x_quantity is None:
            x_quantity = planes.x_quantity
            x_labels = planes.x_labels
        combos = 1
        for size in group_sizes:
            combos *= size
        value = self._samples.value
        series: list[CurveSeries] = []
        for flat_index in range(combos):
            key_indices = np.unravel_index(flat_index, group_sizes or (1,))
            key = tuple(
                domain.values[int(index)]
                for domain, index in zip(group_domains, key_indices)
            )
            y_column = y_plane[:, flat_index]
            counts_column = counts_plane[:, flat_index]
            sem_column = (
                None if sem_plane is None else sem_plane[:, flat_index]
            )
            label = value.label if not key else ", ".join(
                item.label for item in key
            )
            series.append(
                CurveSeries(
                    x=x_quantity,
                    x_labels=x_labels,
                    y=QuantityArray(
                        y_column,
                        value.canonical_unit.convert_value_to(
                            y_column, value.display_unit
                        ),
                        value.canonical_unit,
                        value.display_unit,
                        value.label,
                    ),
                    valid=(counts_column > 0) & np.isfinite(y_column),
                    counts=counts_column,
                    sem=sem_column,
                    group_key=key,
                    label=label,
                )
            )
        return tuple(series)

    def _curve_from_positions(
        self,
        x: AxisRef,
        positions: NDArray[np.int64],
        groups: tuple[AxisRef, ...],
        aggregation: Reduction,
        uncertainty: bool = False,
    ) -> CurveData:
        series: list[CurveSeries] = []
        flat_values = self._samples.value.canonical.reshape(-1)
        flat_valid = self._samples.valid_mask.reshape(-1)
        x_resolved = self._resolve(x)
        for key, group_positions in self._groups(groups, positions):
            x_domain = self._domain(x, group_positions)
            usable = flat_valid[group_positions] & (x_domain.codes >= 0)
            group_values = flat_values[group_positions]
            y, counts = _aggregate_by_codes(
                group_values,
                usable,
                x_domain.codes,
                x_domain.size,
                aggregation,
            )
            sem = None
            if uncertainty:
                # Same kernel over the squares (see _dense_curve_data).
                mean_sq, _sq_counts = _aggregate_by_codes(
                    np.square(group_values.astype(np.float64, copy=False)),
                    usable,
                    x_domain.codes,
                    x_domain.size,
                    Reduction.MEAN,
                )
                sem = _sem_from_moments(
                    np.asarray(y, np.float64),
                    np.asarray(mean_sq, np.float64),
                    counts,
                )
            y_display = self._samples.value.canonical_unit.convert_value_to(
                y, self._samples.value.display_unit
            )
            x_canonical = x_domain.canonical
            x_display = x_domain.display
            valid = (counts > 0) & np.isfinite(y)
            label = self._samples.value.label if not key else ", ".join(
                item.label for item in key
            )
            series.append(
                CurveSeries(
                    x=QuantityArray(
                        x_canonical,
                        x_display,
                        x_resolved.coordinate.canonical_unit,
                        x_resolved.coordinate.display_unit,
                        x_resolved.coordinate.label,
                    ),
                    x_labels=_axis_coordinate_labels(x_resolved, x_canonical),
                    y=QuantityArray(
                        y,
                        y_display,
                        self._samples.value.canonical_unit,
                        self._samples.value.display_unit,
                        self._samples.value.label,
                    ),
                    valid=valid,
                    counts=counts,
                    sem=sem,
                    group_key=key,
                    label=label,
                )
            )
        return CurveData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            x_ref=x,
            group_by=groups,
            series=tuple(series),
        )

    def validate_image(self, x: AxisRef, y: AxisRef) -> None:
        """Check an image projection without computing it (see validate_curve).

        Point rows beyond the two image axes pool under the declared
        reduction, the same fate every unassigned axis has (both image
        kernels already reduce every non-image dimension).
        ``image.default_spec`` admits a camera cycle's ``(repeat,
        frame-points, y, x)`` box on exactly that promise, so the build
        honours it -- admits and buildable are one decision, owned here.
        """

        self._validate_image_shape(x, y)

    def _validate_image_shape(self, x: AxisRef, y: AxisRef) -> None:
        if not isinstance(x, AxisRef) or not isinstance(y, AxisRef):
            raise TypeError("image x and y must be AxisRef objects")
        if x == y:
            raise DataViewError("image x and y axes must differ")
        # Same reason as the curve x check: both image kernels need numeric
        # coordinates, so that requirement belongs to the admission decision.
        _require_real_numeric(self._resolve(x).coordinate.canonical, x)
        _require_real_numeric(self._resolve(y).coordinate.canonical, y)
        point_domains = {x.domain, y.domain}
        if point_domains <= {
            AxisDomain.POINT_ROW,
            AxisDomain.POINT_COORDINATE,
        }:
            raise DataViewError(
                "Image requires two declared GridTopology dimensions when both "
                "axes come from ordinary point rows/coordinates"
            )

    def image(
        self,
        x: AxisRef,
        y: AxisRef,
        *,
        aggregation: Reduction = Reduction.MEAN,
    ) -> ImageData:
        self.validate_image(x, y)
        aggregation = _validate_aggregation(aggregation)
        dense = self._dense_data_image(x, y, aggregation)
        if dense is not None:
            return dense
        factored = self._factored_image(x, y, aggregation)
        if factored is not None:
            return factored
        positions = self._all_positions()
        return self._image_from_positions(x, y, positions, aggregation)

    def _dense_data_image(
        self,
        x: AxisRef,
        y: AxisRef,
        aggregation: Reduction,
    ) -> ImageData | None:
        """Project two declared dense data axes without cell-wise grouping.

        This path is deliberately narrow.  Point coordinates/topology and
        facet subsets use the generic grouping algorithm.  For two data axes,
        however, the immutable dataset shape already proves
        a regular dense tensor.  Moving those axes to ``(y, x)`` and reducing
        every remaining dimension is exactly the same operation as grouping
        every flattened sample by its two declared data-axis indices.
        """

        if x.domain is not AxisDomain.DATA or y.domain is not AxisDomain.DATA:
            return None
        assert x.axis_id is not None and y.axis_id is not None
        # From the resolver, which is the one thing that decides what an axis
        # reference means.  This used to build its own map keyed on the axis
        # NAME, while the resolver accepts a name OR a full axis id -- so
        # naming an axis the precise way dropped silently off this dense path
        # into the generic per-pixel aggregator: measured 830 ms against 30 ms
        # on one 640x480 frame, and it grows with the pixel count.
        try:
            x_resolved = self._resolve(x)
            y_resolved = self._resolve(y)
        except AxisResolutionError:
            # Let the normal resolver produce the public missing-axis error.
            return None
        x_dimension = x_resolved.dimension
        y_dimension = y_resolved.dimension
        x_canonical = np.asarray(x_resolved.domain_canonical)
        y_canonical = np.asarray(y_resolved.domain_canonical)
        if not (
            np.all(_finite_coordinate(x_canonical))
            and np.all(_finite_coordinate(y_canonical))
        ):
            # The generic path omits non-finite declared coordinates from its
            # domains.  Keep that uncommon policy in one implementation.
            return None

        return self._dense_image_data(
            x,
            y,
            x_resolved,
            y_resolved,
            self._samples.value.canonical,
            self._samples.valid_mask,
            aggregation,
        )

    def _dense_image_data(
        self,
        x: AxisRef,
        y: AxisRef,
        x_resolved: "_ResolvedAxis",
        y_resolved: "_ResolvedAxis",
        values: NDArray[Any],
        usable: NDArray[np.bool_],
        aggregation: Reduction,
    ) -> ImageData:
        """One dense image out of one (possibly row-sliced) value tensor."""

        x_canonical = np.asarray(x_resolved.domain_canonical)
        y_canonical = np.asarray(y_resolved.domain_canonical)
        ny, nx = int(y_canonical.size), int(x_canonical.size)
        moved = np.reshape(
            np.moveaxis(
                values,
                (y_resolved.dimension, x_resolved.dimension),
                (-2, -1),
            ),
            (-1, ny, nx),
            order="C",
        )
        moved_usable = np.reshape(
            np.moveaxis(
                usable,
                (y_resolved.dimension, x_resolved.dimension),
                (-2, -1),
            ),
            (-1, ny, nx),
            order="C",
        )
        if moved.shape[0] == 1:
            z = np.asarray(moved[0])
            valid = np.asarray(moved_usable[0], dtype=np.bool_)
            if z.dtype.kind in "fc":
                valid = valid & np.isfinite(z)
        else:
            z, counts = _masked_leading_reduce(moved, moved_usable, aggregation)
            valid = (counts > 0) & np.isfinite(z)
        # Reductions may promote their result; the singleton camera path keeps
        # the producer's native dtype and immutable storage.
        z = np.asarray(z)
        z_display = self._samples.value.canonical_unit.convert_value_to(
            z, self._samples.value.display_unit
        )
        z.setflags(write=False)
        return ImageData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            x_ref=x,
            y_ref=y,
            x=QuantityArray(
                x_canonical,
                np.asarray(x_resolved.domain_display),
                x_resolved.coordinate.canonical_unit,
                x_resolved.coordinate.display_unit,
                x_resolved.coordinate.label,
            ),
            y=QuantityArray(
                y_canonical,
                np.asarray(y_resolved.domain_display),
                y_resolved.coordinate.canonical_unit,
                y_resolved.coordinate.display_unit,
                y_resolved.coordinate.label,
            ),
            z=QuantityArray(
                z,
                z_display,
                self._samples.value.canonical_unit,
                self._samples.value.display_unit,
                self._samples.value.label,
            ),
            valid=valid,
        )

    def _factored_image(
        self,
        x: AxisRef,
        y: AxisRef,
        aggregation: Reduction,
    ) -> ImageData | None:
        """The scan-heatmap assembly over the one lattice core.

        A heatmap's buckets are the combined (y, x) row key, so the whole
        computation IS ``_factored_planes((y, x), ())``; this method only
        reshapes the folded plane into the mesh and speaks ImageData.
        The oracle tests hold it pixel for pixel to the generic path.
        """

        planes = self._factored_planes((y, x), (), aggregation, False)
        if planes is None:
            return None
        y_domain, x_domain = planes.row_domains
        return self._image_from_planes(
            x,
            y,
            x_domain,
            y_domain,
            planes.y_plane.reshape(planes.row_sizes),
            planes.counts_plane.reshape(planes.row_sizes),
        )

    def _image_from_planes(
        self,
        x: AxisRef,
        y: AxisRef,
        x_domain: "_Domain",
        y_domain: "_Domain",
        z: NDArray[np.float64],
        counts: NDArray[np.int64],
        *,
        used_y: NDArray[np.bool_] | None = None,
        used_x: NDArray[np.bool_] | None = None,
    ) -> ImageData:
        """One (ny, nx) folded plane spoken as ImageData.

        ``used_y``/``used_x`` compress the mesh to a cell's own used set,
        the way the generic path's per-cell domains do: the domain values
        are value-sorted, so restricting them to the present subset keeps
        the generic order exactly.
        """

        x_canonical = np.asarray(x_domain.canonical)
        x_display = np.asarray(x_domain.display)
        y_canonical = np.asarray(y_domain.canonical)
        y_display = np.asarray(y_domain.display)
        if used_x is not None and not bool(used_x.all()):
            x_canonical = x_canonical[used_x]
            x_display = x_display[used_x]
            z = z[:, used_x]
            counts = counts[:, used_x]
        if used_y is not None and not bool(used_y.all()):
            y_canonical = y_canonical[used_y]
            y_display = y_display[used_y]
            z = z[used_y]
            counts = counts[used_y]
        z = np.ascontiguousarray(z)
        counts = np.ascontiguousarray(counts)
        z_display = self._samples.value.canonical_unit.convert_value_to(
            z, self._samples.value.display_unit
        )
        z.setflags(write=False)
        x_coordinate = self._resolve(x).coordinate
        y_coordinate = self._resolve(y).coordinate
        return ImageData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            x_ref=x,
            y_ref=y,
            x=QuantityArray(
                x_canonical,
                x_display,
                x_coordinate.canonical_unit,
                x_coordinate.display_unit,
                x_coordinate.label,
            ),
            y=QuantityArray(
                y_canonical,
                y_display,
                y_coordinate.canonical_unit,
                y_coordinate.display_unit,
                y_coordinate.label,
            ),
            z=QuantityArray(
                z,
                z_display,
                self._samples.value.canonical_unit,
                self._samples.value.display_unit,
                self._samples.value.label,
            ),
            valid=(counts > 0) & np.isfinite(z),
        )

    def _image_from_positions(
        self,
        x: AxisRef,
        y: AxisRef,
        positions: NDArray[np.int64],
        aggregation: Reduction,
    ) -> ImageData:
        x_domain = self._domain(x, positions)
        y_domain = self._domain(y, positions)
        x_canonical = x_domain.canonical
        y_canonical = y_domain.canonical
        nx = x_domain.size
        ny = y_domain.size
        usable = (
            self._samples.valid_mask.reshape(-1)[positions]
            & (x_domain.codes >= 0)
            & (y_domain.codes >= 0)
        )
        combined_codes = np.full(x_domain.codes.shape, -1, dtype=np.int64)
        combined_codes[usable] = y_domain.codes[usable] * nx + x_domain.codes[usable]
        z_flat, counts_flat = _aggregate_by_codes(
            self._samples.value.canonical.reshape(-1)[positions],
            usable,
            combined_codes,
            nx * ny,
            aggregation,
        )
        z = z_flat.reshape(ny, nx)
        counts = counts_flat.reshape(ny, nx)
        z_display = self._samples.value.canonical_unit.convert_value_to(
            z, self._samples.value.display_unit
        )
        x_resolved = self._resolve(x).coordinate
        y_resolved = self._resolve(y).coordinate
        return ImageData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            x_ref=x,
            y_ref=y,
            x=QuantityArray(
                x_canonical,
                x_domain.display,
                x_resolved.canonical_unit,
                x_resolved.display_unit,
                x_resolved.label,
            ),
            y=QuantityArray(
                y_canonical,
                y_domain.display,
                y_resolved.canonical_unit,
                y_resolved.display_unit,
                y_resolved.label,
            ),
            z=QuantityArray(
                z,
                z_display,
                self._samples.value.canonical_unit,
                self._samples.value.display_unit,
                self._samples.value.label,
            ),
            valid=(counts > 0) & np.isfinite(z),
        )

    def histogram(
        self,
        *,
        bins: int | Sequence[float],
    ) -> HistogramData:
        """Distribution of every acquired value: the whole box is the pool."""

        return self._histogram_from_values(
            bins,
            self._samples.value.canonical,
            valid=self._samples.valid_mask,
        )

    def pooled_values(self) -> NDArray[Any]:
        """Every value this revision would pool, in canonical units.

        What a histogram of this revision alone is built from -- and therefore
        what a window over several revisions accumulates.  Canonical, because
        two revisions are only comparable in the unit the data is stored in;
        the display conversion happens once, where the edges are made.
        """

        cached = self._pooled_cache
        if cached is not None:
            return cached
        valid = self._samples.valid_mask
        values = self._samples.value.canonical
        if (
            valid.size
            and all(stride == 0 for stride in valid.strides)
            and bool(valid.flat[0])
        ) or bool(np.all(valid)):
            pooled = values.reshape(-1).view()
        else:
            pooled = values[valid].reshape(-1)
        pooled.setflags(write=False)
        self._pooled_cache = pooled
        return pooled

    def pooled_values_by_repeat(self) -> tuple[NDArray[Any], ...]:
        """The values each repeat of this revision would pool, oldest first.

        A publication that carries R repeats carries R shots, and history
        counts shots -- so a snapshot seeding a window contributes one entry
        per repeat, exactly as :meth:`rolling_history_samples` reduces one per
        repeat.  A snapshot without repeats degenerates to the single pool.
        """

        flat_valid = self._samples.valid_mask.reshape(-1)
        flat_values = self._samples.value.canonical.reshape(-1)
        repeats = schema_repeat_count(self._schema)
        if not self.has_primary_index:
            if repeats <= 1:
                return (self.pooled_values(),)
            block = flat_values.size // repeats
            return tuple(
                flat_values[start : start + block][
                    flat_valid[start : start + block]
                ]
                for start in range(0, flat_values.size, block)
            )
        positions = self._all_positions()
        primary = self._domain(self._primary_index_ref(), positions)
        return tuple(
            flat_values[
                positions[
                    flat_valid[positions] & (primary.codes == index)
                ]
            ]
            for index in range(primary.size)
        )

    def histogram_of(
        self,
        values: NDArray[Any],
        *,
        bins: int | Sequence[float],
        source_revisions: Sequence[int] = (),
    ) -> HistogramData:
        """Distribution of values pooled elsewhere -- a window over revisions.

        The pool is the caller's; the binning, the units and the labels are
        this revision's, so a windowed distribution is the same picture as an
        unwindowed one with more in it.  ``source_revisions`` records which
        revisions actually contributed, so a window that reached fewer shots
        than it asked for says so rather than looking full.
        """

        data = self._histogram_from_values(bins, np.asarray(values))
        if not len(source_revisions):
            return data
        return replace(
            data, source_revisions=tuple(int(item) for item in source_revisions)
        )

    def validate_rolling(self, group: AxisRef | None) -> None:
        """Check a rolling projection without computing it (see validate_curve)."""

        if group is not None:
            if not isinstance(group, AxisRef):
                raise TypeError("rolling group must be an AxisRef or None")
            self._resolve(group)

    def rolling_sample(
        self,
        *,
        group: AxisRef | None = None,
        aggregation: Reduction = Reduction.MEAN,
    ) -> RollingSample:
        """Reduce one source revision to scalar values for rolling history.

        The MEAN's standard error is ALWAYS computed alongside: a rolling
        sample is cached as history, and whether the operator is showing
        the band is a display choice that must never decide what the cache
        contains -- a band flipped on later must reach every retained shot.
        """

        self.validate_rolling(group)
        aggregation = _validate_aggregation(aggregation)
        uncertainty = aggregation is Reduction.MEAN
        if group is None:
            pooled = self.pooled_values()
            value = _reduce_scalar(pooled, aggregation)
            sem = None
            if uncertainty:
                # Sum first: a finite total proves a hole-free pool, whose
                # square sum is one BLAS dot -- no isfinite scan, no
                # ``square`` temporary.  Masked sums otherwise, never a
                # gather of the finite subset: the copy cost more than
                # the moment.
                count = None
                if pooled.size and math.isfinite(
                    float(np.sum(pooled, dtype=np.float64))
                ):
                    squares = float(np.dot(pooled.reshape(-1), pooled.reshape(-1)))
                    if math.isfinite(squares):
                        count = int(pooled.size)
                        mean_square = squares / count
                if count is None:
                    finite = np.isfinite(pooled)
                    count = int(np.count_nonzero(finite))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mean_square = (
                            np.sum(
                                np.square(pooled),
                                where=finite,
                                dtype=np.float64,
                            )
                            / count
                            if count
                            else np.nan
                        )
                sem = _sem_from_moments(
                    np.asarray([value], dtype=np.float64),
                    np.asarray([mean_square], dtype=np.float64),
                    np.asarray([count], dtype=np.int64),
                )
            return RollingSample(
                revision=self._samples.revision,
                generation=self._samples.generation,
                values=np.asarray([value], dtype=np.float64),
                valid=np.asarray([pooled.size > 0 and np.isfinite(value)]),
                group_keys=((),),
                sem=sem,
            )
        positions = self._all_positions()
        flat_values = self._samples.value.canonical.reshape(-1)
        flat_valid = self._samples.valid_mask.reshape(-1)
        domain = self._domain(group, positions)
        codes = domain.codes
        domain_size = len(domain.values)
        keys = tuple((value,) for value in domain.values)
        usable = flat_valid[positions] & (codes >= 0)
        group_values = flat_values[positions]
        values, counts = _aggregate_by_codes(
            group_values,
            usable,
            codes,
            domain_size,
            aggregation,
        )
        sem = None
        if uncertainty:
            mean_sq, _sq_counts = _aggregate_by_codes(
                np.square(group_values.astype(np.float64, copy=False)),
                usable,
                codes,
                domain_size,
                Reduction.MEAN,
            )
            sem = _sem_from_moments(
                np.asarray(values, np.float64),
                np.asarray(mean_sq, np.float64),
                counts,
            )
        valid = (counts > 0) & np.isfinite(values)
        return RollingSample(
            revision=self._samples.revision,
            generation=self._samples.generation,
            values=values,
            valid=valid,
            group_keys=keys,
            sem=sem,
        )

    def rolling_history_samples(
        self,
        *,
        group: AxisRef | None = None,
        aggregation: Reduction = Reduction.MEAN,
    ) -> tuple[RollingSample, ...]:
        """Expand the repeat axis into per-shot rolling samples, oldest first.

        A static snapshot carries its shot history on the repeat axis; each
        repeat reduces to one rolling sample exactly as :meth:`rolling_sample`
        reduces one whole revision.  A snapshot without repeats degenerates to
        the single whole-revision sample.
        """

        if self.has_primary_index:
            return self._history_samples_by_primary_index(
                group=group,
                aggregation=aggregation,
            )
        repeats = schema_repeat_count(self._schema)
        if repeats <= 1:
            return (
                self.rolling_sample(group=group, aggregation=aggregation),
            )
        self.validate_rolling(group)
        aggregation = _validate_aggregation(aggregation)
        positions = self._all_positions()
        flat_values = self._samples.value.canonical.reshape(-1)
        flat_valid = self._samples.valid_mask.reshape(-1)
        if group is None:
            codes = np.zeros(positions.size, dtype=np.int64)
            domain_size = 1
            keys: tuple[tuple[AxisValue, ...], ...] = ((),)
        else:
            domain = self._domain(group, positions)
            codes = domain.codes
            domain_size = len(domain.values)
            keys = tuple((value,) for value in domain.values)
        block = flat_values.size // repeats
        repeat_of_position = positions // block
        usable = flat_valid[positions] & (codes >= 0)
        position_values = flat_values[positions]
        squared = (
            np.square(position_values.astype(np.float64, copy=False))
            if aggregation is Reduction.MEAN
            else None
        )
        samples: list[RollingSample] = []
        for repeat in range(repeats):
            in_repeat = repeat_of_position == repeat
            values, counts = _aggregate_by_codes(
                position_values,
                usable & in_repeat,
                codes,
                domain_size,
                aggregation,
            )
            sem = None
            if squared is not None:
                mean_sq, _sq_counts = _aggregate_by_codes(
                    squared,
                    usable & in_repeat,
                    codes,
                    domain_size,
                    Reduction.MEAN,
                )
                sem = _sem_from_moments(
                    np.asarray(values, np.float64),
                    np.asarray(mean_sq, np.float64),
                    counts,
                )
            valid = (counts > 0) & np.isfinite(values)
            samples.append(
                RollingSample(
                    revision=self._samples.revision,
                    generation=self._samples.generation,
                    values=values,
                    valid=valid,
                    group_keys=keys,
                    sem=sem,
                )
            )
        return tuple(samples)

    def _history_samples_by_primary_index(
        self,
        *,
        group: AxisRef | None,
        aggregation: Reduction,
    ) -> tuple[RollingSample, ...]:
        """Reduce every authored primary-index cell without arrival history."""

        self.validate_rolling(group)
        aggregation = _validate_aggregation(aggregation)
        positions = self._all_positions()
        flat_values = self._samples.value.canonical.reshape(-1)
        flat_valid = self._samples.valid_mask.reshape(-1)
        primary = self._domain(self._primary_index_ref(), positions)
        if group is None:
            codes = np.zeros(positions.size, dtype=np.int64)
            domain_size = 1
            keys: tuple[tuple[AxisValue, ...], ...] = ((),)
        else:
            grouped = self._domain(group, positions)
            codes = grouped.codes
            domain_size = len(grouped.values)
            keys = tuple((value,) for value in grouped.values)
        usable = flat_valid[positions] & (codes >= 0)
        position_values = flat_values[positions]
        squared = (
            np.square(position_values.astype(np.float64, copy=False))
            if aggregation is Reduction.MEAN
            else None
        )
        samples: list[RollingSample] = []
        for index, source in enumerate(primary.values):
            selected = usable & (primary.codes == index)
            values, counts = _aggregate_by_codes(
                position_values,
                selected,
                codes,
                domain_size,
                aggregation,
            )
            sem = None
            if squared is not None:
                mean_sq, _sq_counts = _aggregate_by_codes(
                    squared,
                    selected,
                    codes,
                    domain_size,
                    Reduction.MEAN,
                )
                sem = _sem_from_moments(
                    np.asarray(values, np.float64),
                    np.asarray(mean_sq, np.float64),
                    counts,
                )
            valid = (counts > 0) & np.isfinite(values)
            samples.append(
                RollingSample(
                    revision=int(source.canonical),
                    generation=self._samples.generation,
                    values=values,
                    valid=valid,
                    group_keys=keys,
                    sem=sem,
                )
            )
        return tuple(samples)

    def _histogram_from_positions(
        self,
        bins: int | Sequence[float],
        positions: NDArray[np.int64],
    ) -> HistogramData:
        if positions is self._positions_cache:
            # The whole revision: bin values + mask directly, no gather.
            return self._histogram_from_values(
                bins,
                self._samples.value.canonical,
                valid=self._samples.valid_mask,
            )
        flat_valid = self._samples.valid_mask.reshape(-1)
        flat_values = self._samples.value.canonical.reshape(-1)
        return self._histogram_from_values(
            bins, flat_values[positions[flat_valid[positions]]]
        )

    def _histogram_from_values(
        self,
        bins: int | Sequence[float],
        values: NDArray[Any],
        *,
        valid: NDArray[np.bool_] | None = None,
    ) -> HistogramData:
        _require_real_numeric(values, None)
        canonical_bins: int | NDArray[Any]
        if isinstance(bins, bool):
            raise TypeError("histogram bin count must be an integer")
        if isinstance(bins, (int, np.integer)):
            if int(bins) <= 0:
                raise ValueError("histogram bin count must be positive")
            canonical_bins = int(bins)
        else:
            edges = np.asarray(tuple(bins))
            _require_real_numeric(edges, None)
            if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
                raise ValueError("histogram edges must be a finite one-dimensional sequence")
            edges = self._samples.value.display_unit.convert_value_to(
                edges, self._samples.value.canonical_unit
            )
            if np.any(np.diff(edges) <= 0):
                raise ValueError("histogram edges must be strictly increasing")
            canonical_bins = edges
        counts = (
            None
            if isinstance(canonical_bins, int)
            else _uniform_integer_counts(values, valid, canonical_bins)
        )
        if counts is None:
            source = np.asarray(values)
            if valid is None or bool(np.all(valid)):
                selected = source.reshape(-1)
            else:
                selected = source[np.asarray(valid, dtype=np.bool_)].reshape(-1)
            counts, edges = np.histogram(selected, bins=canonical_bins)
        else:
            edges = canonical_bins
        centers = (edges[:-1] + edges[1:]) / 2.0
        display_edges = self._samples.value.canonical_unit.convert_value_to(
            edges, self._samples.value.display_unit
        )
        display_centers = self._samples.value.canonical_unit.convert_value_to(
            centers, self._samples.value.display_unit
        )
        return HistogramData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            edges=QuantityArray(
                edges,
                display_edges,
                self._samples.value.canonical_unit,
                self._samples.value.display_unit,
                self._samples.value.label,
            ),
            centers=QuantityArray(
                centers,
                display_centers,
                self._samples.value.canonical_unit,
                self._samples.value.display_unit,
                self._samples.value.label,
            ),
            counts=counts,
        )

    def validate_facet(self, spec: FacetGridPlot) -> None:
        """Check a facet projection without building any cell (see validate_curve).

        The cell used to bypass projection validation entirely because the
        facet path builds cells positionally; the shared checks here close
        that hole for probe and build alike.  A cell is validated by exactly
        the rules its standalone kind is validated by -- a faceted curve and
        a curve are the same projection, one slice at a time.
        """

        if not isinstance(spec, FacetGridPlot):
            raise TypeError("spec must be FacetGridPlot")
        self._resolve(spec.facet)
        cell = spec.cell
        if isinstance(cell, CurvePlot):
            self._validate_curve_shape(
                cell.x, () if cell.group is None else (cell.group,)
            )
        elif isinstance(cell, ImagePlot):
            self._validate_image_shape(cell.x, cell.y)
        elif not isinstance(cell, HistogramPlot):
            raise TypeError(
                "facet cell must be CurvePlot, ImagePlot, or HistogramPlot"
            )

    def facet_cell_count(self, spec: FacetGridPlot) -> int:
        """Return the facet domain size without building any cell.

        A DECLARED all-finite facet domain has a known size from axis-sized
        arrays alone: repeat, point-row and data domains use every declared
        index by construction, and a grid dimension's used indices come off
        the topology's row-to-cell map.  Counting used to materialize
        ``np.arange`` over every ELEMENT (~20M on a camera facet) plus full
        flat coordinate copies just to size a declared domain; the element
        pass remains only for the undeclared/non-finite fallback (point
        coordinates, NaN declared coordinates).
        """

        resolved = self._resolve(spec.facet)
        if (
            self._samples.value.canonical.size
            and resolved.declared_domain
            and bool(_finite_coordinate(resolved.domain_canonical).all())
        ):
            if spec.facet.domain is AxisDomain.POINT_DIMENSION:
                topology = self._schema.grid_topology
                assert topology is not None  # _resolve proved the topology
                assert spec.facet.axis_id is not None
                position = topology_position(topology, spec.facet.axis_id)
                return len({cell[position] for cell in topology.row_to_cell})
            return int(resolved.domain_canonical.size)
        positions = self._all_positions()
        return self._domain(spec.facet, positions).size

    def facet(
        self,
        spec: FacetGridPlot,
        *,
        bins: int | Sequence[float] | None = None,
        uncertainty: bool = False,
    ) -> FacetData:
        self.validate_facet(spec)
        cell = spec.cell
        if not isinstance(cell, HistogramPlot) and bins is not None:
            raise ValueError("bins are accepted only for Histogram facet cells")
        if uncertainty and not isinstance(cell, CurvePlot):
            raise ValueError("uncertainty is accepted only for Curve facet cells")
        shared_bins = bins
        if isinstance(cell, HistogramPlot) and isinstance(bins, bool):
            raise TypeError("histogram bin count must be an integer")
        if isinstance(cell, HistogramPlot) and isinstance(bins, (int, np.integer)):
            if int(bins) <= 0:
                raise ValueError("histogram bin count must be positive")
            # Shared edges over every value keep all cell histograms
            # comparable on one axis.
            flat_valid = self._samples.valid_mask.reshape(-1)
            display_values = np.asarray(self._samples.value.display).reshape(-1)
            values = display_values[flat_valid]
            _require_real_numeric(values, None)
            values = values[np.isfinite(values)]
            shared_bins = aligned_histogram_edges(values, int(bins))
        if isinstance(cell, HistogramPlot) and bins is None:
            raise DataViewError("histogram facet cells require explicit bins")
        dense = self._dense_facet(spec, shared_bins, uncertainty)
        if dense is not None:
            return dense
        factored = self._factored_facet(spec, uncertainty)
        if factored is not None:
            return factored
        return self._facet_from_positions(
            spec,
            shared_bins,
            self._all_positions(),
            uncertainty,
        )

    def _facet_from_positions(
        self,
        spec: FacetGridPlot,
        shared_bins: int | Sequence[float] | None,
        base_positions: NDArray[np.int64],
        uncertainty: bool = False,
    ) -> FacetData:
        cell = spec.cell
        cells: list[FacetCell] = []
        for facet_index, (key, cell_positions) in enumerate(
            self._groups((spec.facet,), base_positions)
        ):
            facet_value = key[0]
            if isinstance(cell, CurvePlot):
                payload: FacetPayload = self._curve_from_positions(
                    cell.x,
                    cell_positions,
                    () if cell.group is None else (cell.group,),
                    cell.reduction,
                    uncertainty,
                )
            elif isinstance(cell, ImagePlot):
                payload = self._image_from_positions(
                    cell.x,
                    cell.y,
                    cell_positions,
                    cell.reduction,
                )
            else:
                assert shared_bins is not None
                payload = self._histogram_from_positions(
                    shared_bins,
                    cell_positions,
                )
            cells.append(
                FacetCell(
                    facet_index=facet_index,
                    facet_value_canonical=facet_value.canonical,
                    facet_value_display=facet_value.display,
                    label=facet_value.label,
                    payload=payload,
                )
            )
        return FacetData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            spec=spec,
            cells=tuple(cells),
        )

    def _dense_facet(
        self,
        spec: FacetGridPlot,
        shared_bins: int | Sequence[float] | None,
        uncertainty: bool = False,
    ) -> FacetData | None:
        """Row-sliced twin of the dense projections, one cell at a time.

        A facet over the repeat axis or any point-domain axis selects whole
        repeats or whole point rows, and slicing those preserves the
        regularity the dense paths rely on: each cell is the same dense
        tensor, shorter.  Cells over DATA axes therefore reduce through the
        same kernel as their single-kind projections -- no (position, value)
        pairs, no code sort, no per-pixel grouping.  Facets over DATA axes
        and grouped curve cells keep the generic algorithm.
        """

        facet = spec.facet
        cell = spec.cell
        values = self._samples.value.canonical
        valid_mask = self._samples.valid_mask
        if facet.domain is AxisDomain.REPEAT:
            slice_axis = 0
        elif facet.domain in (
            AxisDomain.POINT_ROW,
            AxisDomain.POINT_COORDINATE,
            AxisDomain.POINT_DIMENSION,
        ):
            slice_axis = 1
        else:
            return None

        x_resolved = y_resolved = None
        if isinstance(cell, CurvePlot):
            # The same qualifying guards as _dense_data_curve, so a cell
            # this path draws and the one the single kind draws agree.
            if cell.group is not None or cell.x.domain is not AxisDomain.DATA:
                return None
            try:
                x_resolved = self._resolve(cell.x)
            except AxisResolutionError:
                return None
            x_domain = np.asarray(x_resolved.domain_canonical)
            if not np.all(_finite_coordinate(x_domain)):
                return None
            if x_domain.size > 1 and not np.all(np.diff(x_domain) > 0):
                return None
        elif isinstance(cell, ImagePlot):
            if (
                cell.x.domain is not AxisDomain.DATA
                or cell.y.domain is not AxisDomain.DATA
            ):
                return None
            try:
                x_resolved = self._resolve(cell.x)
                y_resolved = self._resolve(cell.y)
            except AxisResolutionError:
                return None
            if not (
                np.all(_finite_coordinate(np.asarray(x_resolved.domain_canonical)))
                and np.all(_finite_coordinate(np.asarray(y_resolved.domain_canonical)))
            ):
                return None
        elif not isinstance(cell, HistogramPlot):
            return None

        # One representative element per candidate slice puts the existing
        # domain machinery (labels, declared indices, units) to work on an
        # array the size of the SLICE COUNT, not of the dataset.
        data_size = 1
        for size in values.shape[2:]:
            data_size *= int(size)
        if slice_axis == 0:
            representatives = np.arange(values.shape[0], dtype=np.int64) * (
                values.shape[1] * data_size
            )
        else:
            representatives = np.arange(values.shape[1], dtype=np.int64) * data_size
        domain = self._domain(facet, representatives)
        if not domain.values:
            return None

        cells: list[FacetCell] = []
        for facet_index, facet_value in enumerate(domain.values):
            selector = np.flatnonzero(domain.codes == facet_index)
            contiguous = bool(selector.size) and (
                selector.size == 1 or bool(np.all(np.diff(selector) == 1))
            )
            selected: slice | NDArray[np.int64] = (
                slice(int(selector[0]), int(selector[-1]) + 1)
                if contiguous
                else selector
            )
            if slice_axis == 0:
                cell_values = values[selected]
                cell_valid = valid_mask[selected]
            else:
                cell_values = values[:, selected]
                cell_valid = valid_mask[:, selected]
            if isinstance(cell, CurvePlot):
                payload: FacetPayload = self._dense_curve_data(
                    cell.x,
                    x_resolved,
                    cell_values,
                    cell_valid,
                    cell.reduction,
                    uncertainty,
                )
            elif isinstance(cell, ImagePlot):
                payload = self._dense_image_data(
                    cell.x,
                    cell.y,
                    x_resolved,
                    y_resolved,
                    cell_values,
                    cell_valid,
                    cell.reduction,
                )
            else:
                assert shared_bins is not None
                payload = self._histogram_from_values(
                    shared_bins,
                    cell_values,
                    valid=cell_valid,
                )
            cells.append(
                FacetCell(
                    facet_index=facet_index,
                    facet_value_canonical=facet_value.canonical,
                    facet_value_display=facet_value.display,
                    label=facet_value.label,
                    payload=payload,
                )
            )
        return FacetData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            spec=spec,
            cells=tuple(cells),
        )

    def _all_positions(self) -> NDArray[np.int64]:
        cached = self._positions_cache
        if cached is None:
            cached = np.arange(self._samples.value.canonical.size, dtype=np.int64)
            self._positions_cache = cached
        return cached

    def _groups(
        self,
        refs: tuple[AxisRef, ...],
        positions: NDArray[np.int64],
    ) -> Iterator[tuple[tuple[AxisValue, ...], NDArray[np.int64]]]:
        if not refs:
            yield (), positions
            return
        domains = tuple(self._domain(ref, positions) for ref in refs)
        usable = np.ones(positions.shape, dtype=np.bool_)
        for domain in domains:
            usable &= domain.codes >= 0
        usable_local = np.flatnonzero(usable)
        if usable_local.size == 0:
            return
        if len(domains) == 1:
            # The common one-axis grouping case does not need the generic
            # two-dimensional ``unique(..., axis=0, return_inverse=True)``
            # path.  A stable code sort plus one bincount gives the same
            # ordering while avoiding a second full-size inverse array.
            domain = domains[0]
            selected_codes = domain.codes[usable_local]
            order = np.argsort(selected_codes, kind="stable")
            sorted_codes = selected_codes[order]
            counts = np.bincount(
                sorted_codes,
                minlength=domain.size,
            )
            start = 0
            for code in np.flatnonzero(counts):
                stop = start + int(counts[code])
                local = usable_local[order[start:stop]]
                yield (domain.values[int(code)],), positions[local]
                start = stop
            return
        code_rows = np.stack(
            [domain.codes[usable_local] for domain in domains],
            axis=1,
        )
        combinations, inverse = np.unique(code_rows, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse, minlength=len(combinations))
        stops = np.cumsum(counts)
        start = 0
        for combination, stop in zip(combinations, stops, strict=True):
            key = tuple(
                domain.values[int(code)] for domain, code in zip(domains, combination)
            )
            local = usable_local[order[start:int(stop)]]
            yield key, positions[local]
            start = int(stop)

    def _domain(
        self,
        ref: AxisRef,
        positions: NDArray[np.int64],
    ) -> _Domain:
        whole = positions is self._positions_cache
        if whole:
            carried = self._domain_carry.get(ref)
            if carried is not None:
                coords, domain = carried
                current = np.asarray(self._resolve(ref).coordinate.canonical).reshape(-1)
                if coords.shape == current.shape and np.array_equal(
                    coords, current
                ):
                    return domain
        resolved = self._resolve(ref)
        # ``CoordinateArray`` keeps broadcast tensor views for renderers.
        # Grouping is flat and hot, so the one materialization lives in this
        # cache and every domain call reuses the immutable planes.
        cached_flat = self._flat_cache.get(ref)
        sparse = (
            cached_flat is None
            and positions.size < resolved.coordinate.canonical.size
        )
        if cached_flat is None and not sparse:
            cached_flat = (
                _readonly(np.array(resolved.coordinate.canonical, copy=True).reshape(-1)),
                _readonly(
                    np.array(
                        resolved.coordinate.indices,
                        dtype=np.int64,
                        copy=True,
                    ).reshape(-1)
                ),
            )
            self._flat_cache[ref] = cached_flat
        if sparse:
            selected = np.asarray(resolved.coordinate.canonical).flat[positions]
            selected_indices = np.asarray(resolved.coordinate.indices).flat[
                positions
            ]
        else:
            assert cached_flat is not None
            canonical, indices_flat = cached_flat
            selected = canonical[positions]
            selected_indices = indices_flat[positions]
        # A declared, all-finite domain (checked once, at domain size) makes
        # every element's coordinate valid by construction, so the
        # per-element canonical gather and isfinite pass -- two full-size
        # temporaries per axis, millions of elements on a camera facet --
        # carry no information.  Codes then come straight off the index plane.
        valid_local: NDArray[np.int64] | None = None
        if resolved.declared_domain and bool(
            _finite_coordinate(resolved.domain_canonical).all()
        ):
            declared = selected_indices
        else:
            coordinate_valid = _finite_coordinate(selected)
            valid_local = np.flatnonzero(coordinate_valid)
            if valid_local.size == 0:
                codes = np.full(positions.shape, -1, dtype=np.int64)
                codes.setflags(write=False)
                empty = _readonly(np.empty(0))
                return _Domain(empty, empty, codes, tuple)
            declared = (
                selected_indices[valid_local]
                if resolved.declared_domain
                else None
            )
        if declared is not None:
            # The domain is DECLARED, so its size is known and small; a
            # bincount + remap finds the used indices in one O(N) pass where
            # np.unique paid a full sort of one value per element.
            used_indices = np.flatnonzero(
                np.bincount(declared, minlength=resolved.domain_canonical.size)
            )
            remap = np.full(resolved.domain_canonical.size, -1, dtype=np.int64)
            remap[used_indices] = np.arange(used_indices.size, dtype=np.int64)
            inverse = remap[declared]
            canonical_values = resolved.domain_canonical[used_indices]
            display_values = resolved.domain_display[used_indices]
        else:
            used_indices = None
            canonical_values, inverse = np.unique(selected[valid_local], return_inverse=True)
            display_values = resolved.coordinate.canonical_unit.convert_value_to(
                canonical_values, resolved.coordinate.display_unit
            )
        if valid_local is None:
            codes = inverse
        else:
            codes = np.full(positions.shape, -1, dtype=np.int64)
            codes[valid_local] = inverse
        codes.setflags(write=False)

        def build_values() -> tuple[AxisValue, ...]:
            if used_indices is not None:
                indices: tuple[int | None, ...] = tuple(
                    int(index) for index in used_indices
                )
                coordinate_labels = (
                    (None,) * len(indices)
                    if resolved.coordinate_labels is None
                    else tuple(
                        resolved.coordinate_labels[int(index)]
                        for index in used_indices
                    )
                )
            else:
                indices = (None,) * int(canonical_values.size)
                if resolved.coordinate_labels is None:
                    coordinate_labels = (None,) * int(canonical_values.size)
                else:
                    label_by_coordinate = dict(
                        zip(
                            map(_python_scalar, resolved.domain_canonical),
                            resolved.coordinate_labels,
                            strict=True,
                        )
                    )
                    coordinate_labels = tuple(
                        label_by_coordinate[_python_scalar(value)]
                        for value in canonical_values
                    )
            return tuple(
                AxisValue(
                    ref=ref,
                    index=index,
                    canonical=_python_scalar(canonical_value),
                    display=_python_scalar(display_value),
                    label=_axis_value_label(
                        resolved.coordinate.label,
                        display_value,
                        resolved.coordinate.display_unit,
                        coordinate_label,
                    ),
                )
                for index, canonical_value, display_value, coordinate_label in zip(
                    indices,
                    canonical_values,
                    display_values,
                    coordinate_labels,
                    strict=True,
                )
            )

        domain = _Domain(
            _readonly(_scalar_kind_array(canonical_values)),
            _readonly(_scalar_kind_array(display_values)),
            codes,
            build_values,
        )
        if whole:
            self._domain_carry[ref] = (
                np.asarray(resolved.coordinate.canonical).reshape(-1),
                domain,
            )
        return domain

    def _resolve(self, ref: AxisRef) -> _ResolvedAxis:
        if not isinstance(ref, AxisRef):
            raise TypeError("axis reference must be AxisRef")
        cached = self._axis_cache.get(ref)
        if cached is not None:
            return cached
        schema = self._schema
        dimension: int
        declared_domain = True
        if ref.domain is AxisDomain.REPEAT:
            descriptor = descriptor_from_axis(schema.repeat_axis)
            source_coordinates = np.asarray(descriptor.coordinates)
            source_indices = np.arange(descriptor.size, dtype=np.int64)
            domain_canonical = source_coordinates
            label = descriptor.label
            dimension = 0
        elif ref.domain is AxisDomain.POINT_ROW:
            descriptor = None
            source_coordinates = np.arange(schema_point_count(schema), dtype=np.int64)
            source_indices = source_coordinates
            domain_canonical = source_coordinates
            label = "Point row"
            dimension = 1
        elif ref.domain is AxisDomain.POINT_COORDINATE:
            assert ref.axis_id is not None
            try:
                descriptor = descriptor_from_point_column(
                    point_column(schema.point_table, ref.axis_id)
                )
            except KeyError as exc:
                raise AxisResolutionError(
                    f"PointTable has no coordinate {ref.axis_id!r}"
                ) from exc
            source_coordinates = np.asarray(descriptor.coordinates)
            source_indices = np.arange(schema_point_count(schema), dtype=np.int64)
            domain_canonical = source_coordinates
            declared_domain = False
            label = descriptor.label
            dimension = 1
        elif ref.domain is AxisDomain.POINT_DIMENSION:
            assert ref.axis_id is not None
            topology = schema.grid_topology
            if topology is None:
                raise TopologyRequiredError(
                    f"point dimension {ref.axis_id!r} requires producer-declared GridTopology"
                )
            try:
                topology_index = topology_position(topology, ref.axis_id)
            except KeyError as exc:
                raise AxisResolutionError(
                    f"GridTopology has no dimension {ref.axis_id!r}"
                ) from exc
            descriptor = descriptor_from_topology(
                topology,
                topology_index,
                point_table=schema.point_table,
            )
            source_indices = np.asarray(
                [cell[topology_index] for cell in topology.row_to_cell],
                dtype=np.int64,
            )
            domain_canonical = np.asarray(descriptor.coordinates)
            source_coordinates = domain_canonical[source_indices]
            label = descriptor.label
            dimension = 1
        elif ref.domain is AxisDomain.DATA:
            assert ref.axis_id is not None
            try:
                data_index, axis = next(
                    (index, axis)
                    for index, axis in enumerate(schema_data_axes(schema))
                    if str(axis.axis_id) == ref.axis_id or axis.name == ref.axis_id
                )
            except StopIteration as exc:
                raise AxisResolutionError(
                    f"dataset has no data axis {ref.axis_id!r}"
                ) from exc
            descriptor = descriptor_from_axis(axis)
            source_coordinates = np.asarray(descriptor.coordinates)
            source_indices = np.arange(descriptor.size, dtype=np.int64)
            domain_canonical = source_coordinates
            label = descriptor.label
            dimension = 2 + data_index
        else:  # defensive if AxisDomain grows without a resolver
            raise AxisResolutionError(f"unsupported axis domain: {ref.domain!r}")

        canonical_unit = (
            DEFAULT_UNITS.resolve("1")
            if descriptor is None
            else descriptor.canonical_unit(self._unit_registry)
        )
        default_display = canonical_unit
        requested_display = self._axis_display_units.get(ref)
        display_unit = (
            default_display
            if requested_display is None
            else resolve_unit(requested_display, self._unit_registry)
        )
        if not canonical_unit.compatible_with(display_unit):
            raise DataViewError(f"display unit for {ref!r} is incompatible with its axis")
        display_source = canonical_unit.convert_value_to(source_coordinates, display_unit)
        display_domain = canonical_unit.convert_value_to(domain_canonical, display_unit)
        shape = schema_shape(schema)
        canonical_full = _broadcast_1d(source_coordinates, dimension, shape)
        display_full = _broadcast_1d(display_source, dimension, shape)
        index_full = _broadcast_1d(source_indices, dimension, shape)
        coordinate = CoordinateArray(
            ref=ref,
            canonical=canonical_full,
            display=display_full,
            indices=index_full,
            canonical_unit=canonical_unit,
            display_unit=display_unit,
            label=label,
        )
        resolved = _ResolvedAxis(
            coordinate=coordinate,
            domain_canonical=_readonly(domain_canonical),
            domain_display=_readonly(display_domain),
            coordinate_labels=(
                None if descriptor is None else descriptor.coordinate_labels
            ),
            declared_domain=declared_domain,
            dimension=dimension,
        )
        self._axis_cache[ref] = resolved
        return resolved


def aligned_histogram_edges(
    values: ArrayLike,
    bins: int,
    *,
    limits: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Bin edges for one histogram; integer-valued samples get integer bins.

    Equal-width float bins over integer-valued samples alias: a non-integer
    bin width leaves some bins containing no representable value, which shows
    up as structural zero-count holes in the middle of the distribution.
    Integer-valued data therefore bins with an integer width on half-open
    ``k - 0.5`` boundaries (the bin count may shrink below the request when
    the value range is narrower); everything else keeps NumPy's equal-width
    edges over the same range.
    """

    bins = max(1, int(bins))
    flat = np.asarray(values).reshape(-1)
    if flat.dtype.kind == "f":
        flat = flat[np.isfinite(flat)]
    if limits is not None:
        low, high = (float(value) for value in limits)
    elif flat.size:
        low, high = float(np.min(flat)), float(np.max(flat))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    integral = flat.dtype.kind in "iub"
    if not integral and flat.size:
        probe = flat[:65536]
        integral = bool(np.all(probe == np.floor(probe)))
    if not integral:
        return np.linspace(low, high, bins + 1, dtype=float)
    first = math.floor(low + 0.5)
    last = math.floor(high + 0.5)
    covered = max(1, last - first + 1)
    width = max(1, math.ceil(covered / bins))
    count = math.ceil(covered / width)
    return (first - 0.5) + width * np.arange(count + 1, dtype=float)


def _uniform_integer_counts(
    values: NDArray[Any],
    valid: NDArray[np.bool_] | None,
    edges: NDArray[Any],
) -> NDArray[np.int64] | None:
    """Count an aligned integer histogram without sorting every sample."""

    source = np.asarray(values)
    flat = source.reshape(-1)
    if source.dtype.kind not in "biu" or edges.size < 2:
        return None
    widths = np.diff(edges)
    int64 = np.iinfo(np.int64)
    if (
        not np.all(np.isfinite(widths))
        or float(edges[0]) < int64.min
        or float(edges[-1]) > int64.max
    ):
        return None
    width = int(round(float(widths[0])))
    first = int(round(float(edges[0]) + 0.5))
    expected = (first - 0.5) + width * np.arange(edges.size, dtype=float)
    if width <= 0 or not np.array_equal(edges, expected):
        return None

    usable = None if valid is None else np.asarray(valid, dtype=np.bool_)
    if usable is None or (
        usable.size
        and all(stride == 0 for stride in usable.strides)
        and bool(usable.flat[0])
    ) or bool(np.all(usable)):
        selected = flat
    else:
        selected = source[usable].reshape(-1)
    counts = np.zeros(edges.size - 1, dtype=np.int64)
    if not selected.size:
        return counts

    low = int(np.min(selected))
    high = int(np.max(selected))
    if low < int64.min or high > int64.max:
        return None
    span = high - low + 1
    if span > selected.size + 1:
        return None
    shifted = selected if low == 0 else np.subtract(selected, low, dtype=np.int64)
    frequency = np.bincount(shifted, minlength=span)
    for index in range(counts.size):
        start = max(first + index * width, low) - low
        stop = min(first + (index + 1) * width, high + 1) - low
        if stop > start:
            counts[index] = np.sum(frequency[start:stop], dtype=np.int64)
    return counts


def _stride_zero_all_true(mask: NDArray[np.bool_]) -> bool:
    """True for a stride-0 broadcast plane that is constant True."""

    if mask.size == 0:
        return False
    if any(stride != 0 for stride in mask.strides):
        return False
    return bool(mask.flat[0])


def _finite_probe(
    flat: NDArray[Any],
    finite: NDArray[np.bool_],
    limit: int = 65536,
) -> NDArray[np.float64]:
    """The first ``limit`` finite values, without gathering the whole pool.

    Exactly ``flat[finite][:limit]`` -- block-scanned so a pool whose head
    is finite (every real pool) stops after one block.
    """

    collected: list[NDArray[np.float64]] = []
    total = 0
    for start in range(0, int(flat.size), limit):
        block = flat[start : start + limit][finite[start : start + limit]]
        if block.size:
            take = block[: limit - total]
            collected.append(np.asarray(take, dtype=np.float64))
            total += int(take.size)
        if total >= limit:
            break
    if not collected:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(collected)


def _reduce_scalar(values: NDArray[Any], aggregation: Reduction) -> float:
    """Reduce one already-valid flat pool with the canonical rolling rules."""

    if not values.size:
        return math.nan
    if aggregation is Reduction.MEAN:
        return float(np.mean(values, dtype=np.float64))
    if aggregation is Reduction.SUM:
        return float(np.sum(values, dtype=np.float64))
    if aggregation is Reduction.MIN:
        return float(np.min(values))
    if aggregation is Reduction.MAX:
        return float(np.max(values))
    if aggregation is Reduction.FIRST:
        return float(values[0])
    raise AssertionError(f"unsupported reduction: {aggregation!r}")


def _broadcast_1d(
    values: ArrayLike,
    dimension: int,
    target_shape: tuple[int, ...],
) -> NDArray[Any]:
    array = np.asarray(values)
    reshape = [1] * len(target_shape)
    reshape[dimension] = array.size
    return np.broadcast_to(array.reshape(reshape), target_shape)


def _validate_refs(refs: tuple[AxisRef, ...], what: str) -> None:
    if any(not isinstance(ref, AxisRef) for ref in refs):
        raise TypeError(f"{what} must contain AxisRef objects")


def _validate_aggregation(value: Reduction) -> Reduction:
    if not isinstance(value, Reduction):
        raise TypeError("aggregation must be Reduction")
    return value


def _finite_coordinate(values: NDArray[Any]) -> NDArray[np.bool_]:
    if values.dtype.kind in "biufc":
        return np.isfinite(values)
    return np.ones(values.shape, dtype=np.bool_)


def _require_real_numeric(values: NDArray[Any], ref: AxisRef | None) -> None:
    if np.asarray(values).dtype.kind not in "biuf":
        target = "dataset values" if ref is None else repr(ref)
        raise DataViewError(f"{target} must be real numeric for this projection")


def _masked_leading_reduce(
    values: NDArray[Any],
    usable: NDArray[np.bool_],
    aggregation: Reduction,
) -> tuple[NDArray[Any], NDArray[np.int64]]:
    """Reduce the leading axis under a validity mask: THE dense reduction.

    ``values``/``usable`` are ``(samples, *cell_shape)``.  Dense curves and
    multi-sample dense images/facet cells reduce through this one kernel.  A
    single leading curve sample still needs its count plane, so it
    short-circuits here; a singleton image bypasses the helper one level up
    to retain native values and boolean validity without allocating that
    int64 plane.  Values where ``counts`` is zero are unspecified; validity
    is the contract.
    """

    if values.shape[0] == 1:
        return np.asarray(values[0]), np.asarray(usable[0], dtype=np.int64)
    counts = np.sum(usable, axis=0, dtype=np.int64)
    with warnings.catch_warnings():
        # Empty positions are intentionally NaN and marked invalid by the
        # caller through ``counts``.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if aggregation is Reduction.MEAN:
            result = np.mean(values, axis=0, where=usable)
        elif aggregation is Reduction.SUM:
            result = np.sum(values, axis=0, where=usable, initial=0)
            result = np.where(counts > 0, result, np.nan)
        elif aggregation is Reduction.MIN:
            converted = np.asarray(values, dtype=np.float64)
            result = np.min(converted, axis=0, where=usable, initial=np.inf)
            result = np.where(counts > 0, result, np.nan)
        elif aggregation is Reduction.MAX:
            converted = np.asarray(values, dtype=np.float64)
            result = np.max(converted, axis=0, where=usable, initial=-np.inf)
            result = np.where(counts > 0, result, np.nan)
        elif aggregation is Reduction.FIRST:
            first = np.argmax(usable, axis=0)
            result = np.take_along_axis(values, np.expand_dims(first, 0), axis=0)[0]
            result = np.where(counts > 0, result, np.nan)
        else:
            raise AssertionError(f"unsupported reduction: {aggregation!r}")
    return np.asarray(result), counts


def _axis_coordinate_labels(
    resolved: "_ResolvedAxis",
    canonical: NDArray[Any],
) -> tuple[str, ...] | None:
    """The declared display name of every plotted coordinate, in plot order.

    None when the axis has no labels; positions whose coordinate is not in
    the declared domain (never expected, but honest) keep their number.
    """

    declared = resolved.coordinate_labels
    if declared is None:
        return None
    by_coordinate = dict(
        zip(
            map(_python_scalar, np.asarray(resolved.domain_canonical)),
            declared,
            strict=True,
        )
    )
    return tuple(
        by_coordinate.get(_python_scalar(value), f"{value:g}")
        for value in np.asarray(canonical)
    )


def _sem_from_moments(
    mean: NDArray[np.float64],
    mean_of_squares: NDArray[np.float64],
    counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Standard error of the mean from (mean, mean-of-squares, n).

    sem^2 = s^2/n with the unbiased sample variance s^2, which collapses to
    (E[x^2] - mean^2) / (n - 1).  A single-sample bucket has no defined
    spread and reports NaN, never zero: zero would claim certainty.
    """

    n = counts.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.clip(mean_of_squares - np.square(mean), 0.0, None)
        sem = np.sqrt(spread / (n - 1.0))
    sem[n < 2.0] = np.nan
    return sem


def _bucket_sums(
    group: NDArray[Any],
    codes: NDArray[np.int64],
    bucket_count: int,
    output_dtype: np.dtype,
) -> NDArray[Any]:
    """Per-bucket sums in one O(N) counting pass -- never a sort.

    ``bincount`` accumulates in float64 whatever the input dtype, so a
    uint8 camera frame cannot wrap at 256 the way its own arithmetic
    would; a complex plane is two real passes.
    """

    if output_dtype == np.complex128:
        real = np.bincount(codes, weights=group.real, minlength=bucket_count)
        imag = np.bincount(codes, weights=group.imag, minlength=bucket_count)
        return real + 1j * imag
    return np.bincount(
        codes,
        weights=group.astype(np.float64, copy=False),
        minlength=bucket_count,
    )


def _aggregate_by_codes(
    values: NDArray[Any],
    usable: NDArray[np.bool_],
    codes: NDArray[np.int64],
    bucket_count: int,
    aggregation: Reduction,
) -> tuple[NDArray[Any], NDArray[np.int64]]:
    """Reduce every bucket in O(N) passes; only the extremes sort, by radix.

    The buckets arrive as one small-int code per sample.  SUM/MEAN
    accumulate straight into their bucket (``bincount``), FIRST is a
    reversed scatter (the first occurrence is the last write in reversed
    order) -- sequential passes of ~3-5 ms at 2M samples.  The
    comparison-based stable argsort that used to stand in front of EVERY
    reduction was the projection's single largest cost and grew
    superlinearly once the codes outran the cache: 45-456 ms at 2M before
    touching a single value.  MIN/MAX genuinely need per-bucket order, so
    they sort -- but by RADIX: codes narrowed to uint16 take NumPy's O(N)
    radix path, measured flat at 14-18 ms for any bucket count.  (The
    ufunc's indexed ``at`` loop was measured with an unexplained 20x
    buffer-dependent cliff on this platform -- same dtype, flags and
    content, 3 ms or 60 ms by allocation lineage -- and a projection
    cannot ride a primitive with moods.)  Per bucket, members are visited
    in the same original order the sorted path visited them, so the sums
    are bit-identical.
    """

    output_dtype = np.complex128 if values.dtype.kind == "c" else np.float64
    output = np.full(bucket_count, np.nan, dtype=output_dtype)
    counts = np.zeros(bucket_count, dtype=np.int64)
    positions = np.flatnonzero(usable & (codes >= 0))
    if positions.size:
        selected_codes = codes[positions]
        group = values[positions]
        counts = np.bincount(selected_codes, minlength=bucket_count)
        filled = counts > 0
        if aggregation in (Reduction.SUM, Reduction.MEAN):
            sums = _bucket_sums(
                group, selected_codes, bucket_count, output_dtype
            )
            if aggregation is Reduction.MEAN:
                output[filled] = sums[filled] / counts[filled]
            else:
                output[filled] = sums[filled]
        elif aggregation is Reduction.FIRST:
            output[selected_codes[::-1]] = group[::-1]
        elif aggregation in (Reduction.MIN, Reduction.MAX):
            ufunc = np.minimum if aggregation is Reduction.MIN else np.maximum
            narrow = (
                selected_codes.astype(np.uint16)
                if bucket_count <= (1 << 16)
                else selected_codes
            )
            order = np.argsort(narrow, kind="stable")
            ordered_codes = selected_codes[order]
            ordered = group[order]
            boundaries = np.flatnonzero(np.diff(ordered_codes)) + 1
            starts = np.concatenate(([0], boundaries))
            output[ordered_codes[starts]] = ufunc.reduceat(ordered, starts)
        else:
            raise AssertionError(f"unsupported reduction: {aggregation!r}")
    output.setflags(write=False)
    counts.setflags(write=False)
    return output, counts


def _scalar_kind_array(values: NDArray[Any]) -> NDArray[Any]:
    """The dtype the per-value ``_python_scalar`` materialization produced.

    Building these planes from Python scalars promoted narrow floats and
    integers to float64/int64; the direct array path keeps that contract so
    downstream consumers see identical dtypes either way.
    """

    array = np.asarray(values)
    if array.dtype.kind == "f" and array.dtype != np.float64:
        return array.astype(np.float64)
    if array.dtype.kind == "i" and array.dtype != np.int64:
        return array.astype(np.int64)
    if array.dtype.kind == "u":
        # The former AxisValue path first converted NumPy scalars to Python
        # integers.  NumPy then chose int64 only while every value fit;
        # uint64 values above INT64_MAX remained uint64.  Reproduce that
        # lossless promotion without rebuilding one Python object per point.
        if (
            array.dtype.itemsize < 8
            or not array.size
            or int(np.max(array)) <= np.iinfo(np.int64).max
        ):
            return array.astype(np.int64)
        return array.astype(np.uint64, copy=False)
    return array


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _axis_value_label(
    label: str,
    value: Any,
    unit: Unit,
    coordinate_label: str | None = None,
) -> str:
    if coordinate_label is not None:
        return f"{label}={coordinate_label}"
    scalar = _python_scalar(value)
    suffix = "" if unit.symbol == "1" else f" {unit.symbol}"
    return f"{label}={scalar}{suffix}"


__all__ = [
    "AxisResolutionError",
    "AxisValue",
    "CoordinateArray",
    "CurveData",
    "CurveSeries",
    "DataView",
    "DataViewError",
    "FacetCell",
    "FacetData",
    "FacetPayload",
    "HistogramData",
    "ImageData",
    "RollingSample",
    "QuantityArray",
    "SampleProjection",
    "TopologyRequiredError",
]
