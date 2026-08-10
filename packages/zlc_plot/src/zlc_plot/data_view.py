"""Plot-neutral projections of immutable ``(R, P, *D)`` datasets.

The data producer is the only authority for point topology.  In particular,
``AxisRef.point_dimension(...)`` resolves only through an explicit
``GridTopology``; repeated values in a ``PointTable`` are never promoted to a
tensor dimension here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, TypeAlias
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from zlc_data import OwnedSnapshot

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

    def __post_init__(self) -> None:
        values = _readonly(self.values)
        valid = _readonly(self.valid, dtype=np.bool_)
        if values.ndim != 1 or valid.shape != values.shape:
            raise ValueError("rolling sample values and validity must be one-dimensional")
        keys = tuple(tuple(item) for item in self.group_keys)
        if len(keys) != values.size:
            raise ValueError("rolling sample group keys must match value count")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "group_keys", keys)


@dataclass(frozen=True, slots=True)
class CurveSeries:
    x: QuantityArray
    y: QuantityArray
    valid: NDArray[np.bool_] | ArrayLike
    counts: NDArray[np.int64] | ArrayLike
    group_key: tuple[AxisValue, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        valid = _readonly(self.valid, dtype=np.bool_)
        counts = _readonly(self.counts, dtype=np.int64)
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


@dataclass(frozen=True, slots=True)
class _Domain:
    values: tuple[AxisValue, ...]
    codes: NDArray[np.int64]


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
    )

    def __init__(
        self,
        snapshot: OwnedSnapshot,
        *,
        axis_display_units: Mapping[AxisRef, str | Unit] | None = None,
        value_display_unit: str | Unit | None = None,
        unit_registry: UnitRegistry | None = None,
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
            valid = snapshot_validity(snapshot) & np.isfinite(
                value_canonical
            )
        self._snapshot = snapshot
        self._schema = schema
        self._axis_display_units = overrides
        self._unit_registry = registry
        self._axis_cache: dict[AxisRef, _ResolvedAxis] = {}
        self._flat_cache: dict[
            AxisRef,
            tuple[NDArray[Any], NDArray[np.int64]],
        ] = {}
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
        """

        groups = self._validate_curve_shape(x, tuple(group_by))
        self._validate_axis_coverage((x, *groups), kind="curve")

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
        self._resolve(x)
        for ref in groups:
            self._resolve(ref)
        return groups

    def curve(
        self,
        x: AxisRef,
        *,
        group_by: Iterable[AxisRef] = (),
        aggregation: Reduction = Reduction.MEAN,
    ) -> CurveData:
        groups = tuple(group_by)
        self.validate_curve(x, group_by=groups)
        aggregation = _validate_aggregation(aggregation)
        positions = self._all_positions()
        return self._curve_from_positions(x, positions, groups, aggregation)

    def _curve_from_positions(
        self,
        x: AxisRef,
        positions: NDArray[np.int64],
        groups: tuple[AxisRef, ...],
        aggregation: Reduction,
    ) -> CurveData:
        series: list[CurveSeries] = []
        flat_values = self._samples.value.canonical.reshape(-1)
        flat_valid = self._samples.valid_mask.reshape(-1)
        x_resolved = self._resolve(x)
        _require_real_numeric(x_resolved.coordinate.canonical, x)
        for key, group_positions in self._groups(groups, positions):
            x_domain = self._domain(x, group_positions)
            usable = flat_valid[group_positions] & (x_domain.codes >= 0)
            y, counts = _aggregate_by_codes(
                flat_values[group_positions],
                usable,
                x_domain.codes,
                len(x_domain.values),
                aggregation,
            )
            y_display = self._samples.value.canonical_unit.convert_value_to(
                y, self._samples.value.display_unit
            )
            x_canonical = _domain_canonical(x_domain)
            x_display = _domain_display(x_domain)
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
                    y=QuantityArray(
                        y,
                        y_display,
                        self._samples.value.canonical_unit,
                        self._samples.value.display_unit,
                        self._samples.value.label,
                    ),
                    valid=valid,
                    counts=counts,
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
        """Check an image projection without computing it (see validate_curve)."""

        self._validate_image_shape(x, y)
        self._validate_axis_coverage((x, y), kind="image")

    def _validate_image_shape(self, x: AxisRef, y: AxisRef) -> None:
        if not isinstance(x, AxisRef) or not isinstance(y, AxisRef):
            raise TypeError("image x and y must be AxisRef objects")
        if x == y:
            raise DataViewError("image x and y axes must differ")
        self._resolve(x)
        self._resolve(y)
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
        _require_real_numeric(x_resolved.coordinate.canonical, x)
        _require_real_numeric(y_resolved.coordinate.canonical, y)
        x_canonical = np.asarray(x_resolved.domain_canonical)
        y_canonical = np.asarray(y_resolved.domain_canonical)
        if not (
            np.all(_finite_coordinate(x_canonical))
            and np.all(_finite_coordinate(y_canonical))
        ):
            # The generic path omits non-finite declared coordinates from its
            # domains.  Keep that uncommon policy in one implementation.
            return None

        values = np.moveaxis(
            self._samples.value.canonical,
            (y_dimension, x_dimension),
            (-2, -1),
        )
        usable = np.moveaxis(
            self._samples.valid_mask,
            (y_dimension, x_dimension),
            (-2, -1),
        )
        ny, nx = int(y_canonical.size), int(x_canonical.size)
        values = np.reshape(values, (-1, ny, nx), order="C")
        usable = np.reshape(usable, (-1, ny, nx), order="C")
        if values.shape[0] == 1:
            # A monitor-camera frame normally has R=P=1 and no additional
            # reduction dimensions.  Every supported reduction of one sample
            # is that sample.  Preserve the immutable source view and its dtype;
            # validity stays separate so a camera frame does not acquire full
            # float64 and int64 companions before it reaches the renderer.
            z = np.asarray(values[0])
            valid = np.asarray(usable[0], dtype=np.bool_)
        else:
            counts = np.sum(usable, axis=0, dtype=np.int64)
            with warnings.catch_warnings():
                # Empty cells are intentionally NaN and marked invalid below.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if aggregation is Reduction.MEAN:
                    z = np.mean(values, axis=0, where=usable)
                elif aggregation is Reduction.MEDIAN:
                    converted = np.asarray(values, dtype=np.float64)
                    z = np.nanmedian(
                        np.where(usable, converted, np.nan),
                        axis=0,
                    )
                elif aggregation is Reduction.SUM:
                    z = np.sum(values, axis=0, where=usable, initial=0)
                    z = np.where(counts > 0, z, np.nan)
                elif aggregation is Reduction.MIN:
                    converted = np.asarray(values, dtype=np.float64)
                    z = np.min(
                        converted,
                        axis=0,
                        where=usable,
                        initial=np.inf,
                    )
                    z = np.where(counts > 0, z, np.nan)
                elif aggregation is Reduction.MAX:
                    converted = np.asarray(values, dtype=np.float64)
                    z = np.max(
                        converted,
                        axis=0,
                        where=usable,
                        initial=-np.inf,
                    )
                    z = np.where(counts > 0, z, np.nan)
                elif aggregation is Reduction.FIRST:
                    first = np.argmax(usable, axis=0)
                    z = np.take_along_axis(
                        values,
                        first[np.newaxis, :, :],
                        axis=0,
                    )[0]
                    z = np.where(counts > 0, z, np.nan)
                else:
                    raise AssertionError(
                        f"unsupported reduction: {aggregation!r}"
                    )
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

    def _image_from_positions(
        self,
        x: AxisRef,
        y: AxisRef,
        positions: NDArray[np.int64],
        aggregation: Reduction,
    ) -> ImageData:
        x_domain = self._domain(x, positions)
        y_domain = self._domain(y, positions)
        x_canonical = _domain_canonical(x_domain)
        y_canonical = _domain_canonical(y_domain)
        _require_real_numeric(self._resolve(x).coordinate.canonical, x)
        _require_real_numeric(self._resolve(y).coordinate.canonical, y)
        nx = len(x_domain.values)
        ny = len(y_domain.values)
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
                _domain_display(x_domain),
                x_resolved.canonical_unit,
                x_resolved.display_unit,
                x_resolved.label,
            ),
            y=QuantityArray(
                y_canonical,
                _domain_display(y_domain),
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

        return self._histogram_from_positions(bins, self._all_positions())

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
        """Reduce one source revision to scalar values for rolling history."""

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
        usable = flat_valid[positions] & (codes >= 0)
        values, counts = _aggregate_by_codes(
            flat_values[positions],
            usable,
            codes,
            domain_size,
            aggregation,
        )
        valid = (counts > 0) & np.isfinite(values)
        return RollingSample(
            revision=self._samples.revision,
            generation=self._samples.generation,
            values=values,
            valid=valid,
            group_keys=keys,
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

        repeats = schema_repeat_count(self._schema)
        if repeats <= 1:
            return (self.rolling_sample(group=group, aggregation=aggregation),)
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
        samples: list[RollingSample] = []
        for repeat in range(repeats):
            in_repeat = repeat_of_position == repeat
            values, counts = _aggregate_by_codes(
                flat_values[positions],
                usable & in_repeat,
                codes,
                domain_size,
                aggregation,
            )
            valid = (counts > 0) & np.isfinite(values)
            samples.append(
                RollingSample(
                    revision=self._samples.revision,
                    generation=self._samples.generation,
                    values=values,
                    valid=valid,
                    group_keys=keys,
                )
            )
        return tuple(samples)

    def _histogram_from_positions(
        self,
        bins: int | Sequence[float],
        positions: NDArray[np.int64],
    ) -> HistogramData:
        flat_valid = self._samples.valid_mask.reshape(-1)
        flat_values = self._samples.value.canonical.reshape(-1)
        values = flat_values[positions[flat_valid[positions]]]
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
        counts, edges = np.histogram(values, bins=canonical_bins)
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
        that hole for probe and build alike.  Axis coverage is judged over
        the union of cell and facet references — the facet axis is a
        consumed axis.
        """

        if not isinstance(spec, FacetGridPlot):
            raise TypeError("spec must be FacetGridPlot")
        self._resolve(spec.facet)
        cell = spec.cell
        if isinstance(cell, CurvePlot):
            groups = () if cell.group is None else (cell.group,)
            self._validate_curve_shape(cell.x, groups)
            self._validate_axis_coverage(
                (cell.x, *groups, spec.facet), kind="facet curve cell"
            )
        elif isinstance(cell, ImagePlot):
            self._validate_image_shape(cell.x, cell.y)
            self._validate_axis_coverage(
                (cell.x, cell.y, spec.facet), kind="facet image cell"
            )
        elif not isinstance(cell, HistogramPlot):
            raise TypeError(
                "facet cell must be CurvePlot, ImagePlot, or HistogramPlot"
            )

    def facet_cell_count(self, spec: FacetGridPlot) -> int:
        """Return the facet domain size without building any cell."""

        positions = self._all_positions()
        return len(self._domain(spec.facet, positions).values)

    def facet(
        self,
        spec: FacetGridPlot,
        *,
        bins: int | Sequence[float] | None = None,
    ) -> FacetData:
        self.validate_facet(spec)
        base_positions = self._all_positions()
        cell = spec.cell
        if not isinstance(cell, HistogramPlot) and bins is not None:
            raise ValueError("bins are accepted only for Histogram facet cells")
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
                )
            elif isinstance(cell, ImagePlot):
                payload = self._image_from_positions(
                    cell.x,
                    cell.y,
                    cell_positions,
                    cell.reduction,
                )
            else:
                if bins is None:
                    raise DataViewError("histogram facet cells require explicit bins")
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

    def _all_positions(self) -> NDArray[np.int64]:
        return np.arange(self._samples.value.canonical.size, dtype=np.int64)

    def _validate_axis_coverage(
        self,
        refs: tuple[AxisRef, ...],
        *,
        kind: str,
    ) -> None:
        """Validate that a projection's axis assignments are well posed.

        Every axis has exactly one fate: plotted (x/y), grouped, faceted,
        sampled — or collapsed by the projection's declared reduction.  An
        axis absent from ``refs`` (repeat, an unplotted scan dimension, a
        dense data axis) is therefore pooled into the aggregation, exactly
        as the spec's explicit ``reduction`` field authorizes.  This is not
        v1's silent coercion: the reduction is an authored, visible part of
        the specification.

        The point-row domain is the one exception: it is the scan itself,
        and a projection that references no point coordinate at all is a
        kind mismatch (rolling and histogram are the kinds that collapse
        the whole scan), so it stays an error rather than a pool.
        """

        schema = self._schema
        references = tuple(refs)
        for ref in references:
            self._resolve(ref)
        point_covered = any(
            ref.domain
            in {
                AxisDomain.POINT_ROW,
                AxisDomain.POINT_COORDINATE,
                AxisDomain.POINT_DIMENSION,
            }
            for ref in references
        )
        if schema_point_count(schema) > 1 and not point_covered:
            raise DataViewError(
                f"{kind} leaves the authored point-row domain unreferenced; "
                "select a point row/coordinate or group it explicitly"
            )

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
                minlength=len(domain.values),
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
        resolved = self._resolve(ref)
        # ``CoordinateArray`` keeps broadcast tensor views for renderers.
        # Grouping is flat and hot, so the one materialization lives in this
        # cache and every domain call reuses the immutable planes.
        cached_flat = self._flat_cache.get(ref)
        if cached_flat is None:
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
        canonical, indices_flat = cached_flat
        selected = canonical[positions]
        coordinate_valid = _finite_coordinate(selected)
        valid_local = np.flatnonzero(coordinate_valid)
        codes = np.full(positions.shape, -1, dtype=np.int64)
        if valid_local.size == 0:
            codes.setflags(write=False)
            return _Domain((), codes)
        if resolved.declared_domain:
            declared = indices_flat[positions[valid_local]]
            used_indices, inverse = np.unique(declared, return_inverse=True)
            canonical_values = resolved.domain_canonical[used_indices]
            display_values = resolved.domain_display[used_indices]
            indices: tuple[int | None, ...] = tuple(int(index) for index in used_indices)
            coordinate_labels = (
                (None,) * len(indices)
                if resolved.coordinate_labels is None
                else tuple(
                    resolved.coordinate_labels[int(index)] for index in used_indices
                )
            )
        else:
            canonical_values, inverse = np.unique(selected[valid_local], return_inverse=True)
            display_values = resolved.coordinate.canonical_unit.convert_value_to(
                canonical_values, resolved.coordinate.display_unit
            )
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
        codes[valid_local] = inverse
        codes.setflags(write=False)
        values = tuple(
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
        return _Domain(values, codes)

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


def _aggregate_by_codes(
    values: NDArray[Any],
    usable: NDArray[np.bool_],
    codes: NDArray[np.int64],
    bucket_count: int,
    aggregation: Reduction,
) -> tuple[NDArray[Any], NDArray[np.int64]]:
    output_dtype = np.complex128 if values.dtype.kind == "c" else np.float64
    output = np.full(bucket_count, np.nan, dtype=output_dtype)
    counts = np.zeros(bucket_count, dtype=np.int64)
    positions = np.flatnonzero(usable & (codes >= 0))
    if positions.size:
        selected_codes = codes[positions]
        order = np.argsort(selected_codes, kind="stable")
        ordered_codes = selected_codes[order]
        ordered_values = values[positions[order]]
        boundaries = np.flatnonzero(np.diff(ordered_codes)) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [ordered_codes.size]))
        for start, stop in zip(starts, stops):
            code = int(ordered_codes[start])
            group = ordered_values[start:stop]
            counts[code] = group.size
            if aggregation is Reduction.MEAN:
                output[code] = np.mean(group)
            elif aggregation is Reduction.MEDIAN:
                output[code] = np.median(group)
            elif aggregation is Reduction.SUM:
                output[code] = np.sum(group)
            elif aggregation is Reduction.MIN:
                output[code] = np.min(group)
            elif aggregation is Reduction.MAX:
                output[code] = np.max(group)
            elif aggregation is Reduction.FIRST:
                output[code] = group[0]
            else:
                raise AssertionError(f"unsupported reduction: {aggregation!r}")
    output.setflags(write=False)
    counts.setflags(write=False)
    return output, counts


def _domain_canonical(domain: _Domain) -> NDArray[Any]:
    return _readonly(np.asarray([value.canonical for value in domain.values]))


def _domain_display(domain: _Domain) -> NDArray[Any]:
    return _readonly(np.asarray([value.display for value in domain.values]))


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
