"""Plot-neutral projections of immutable ``(R, P, *D)`` datasets.

The data producer is the only authority for point topology.  In particular,
``AxisRef.point_dimension(...)`` resolves only through an explicit
``GridTopology``; repeated values in a ``PointTable`` are never promoted to a
tensor dimension here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, TypeAlias
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from zlc_data import (
    CoordinateScalar,
    DatasetSchema,
    LATEST_COORDINATE,
    OwnedSnapshot,
    canonical_coordinate_scalar,
)
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID

from .data_contract import (
    DEFAULT_UNITS,
    Unit,
    UnitRegistry,
    ResolvedAxis,
    resolve_axis,
    resolve_unit,
    schema_repeat_count,
    schema_shape,
    schema_value_unit,
    snapshot_generation,
    snapshot_sigma,
    snapshot_revision,
    snapshot_schema,
    snapshot_validity,
    snapshot_values,
)

from .kinds import AxisDomain, AxisRef, PlotKind
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotSpec,
    Reduction,
    RollingPlot,
    semantic_spec,
)


class DataViewError(ValueError):
    """Base error for an invalid plot projection request."""


class AxisResolutionError(DataViewError):
    """A requested :class:`AxisRef` is not declared by the dataset."""


class TopologyRequiredError(AxisResolutionError):
    """A point-dimension request has no producer-declared topology."""


@dataclass(frozen=True, slots=True)
class SelectionSubject:
    """Exact upstream quantities cut by one accepted plot projection."""

    plot_kind: PlotKind
    x: AxisRef | None
    y: AxisRef | None
    x_coordinate_frame: str | None = None
    y_coordinate_frame: str | None = None
    scope: tuple[tuple[AxisRef, CoordinateScalar], ...] = ()
    repeat_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plot_kind, PlotKind):
            raise TypeError("selection subject plot_kind must be PlotKind")
        for name in ("x", "y"):
            ref = getattr(self, name)
            if ref is not None and not isinstance(ref, AxisRef):
                raise TypeError(
                    f"selection subject {name} must be AxisRef or None"
                )
            frame = getattr(self, f"{name}_coordinate_frame")
            if frame is not None and (
                not isinstance(frame, str) or not frame.strip()
            ):
                raise TypeError(
                    f"selection subject {name} coordinate frame must be "
                    "non-empty text or None"
                )
            if ref is None and frame is not None:
                raise ValueError(
                    f"selection subject {name} coordinate frame requires an axis"
                )
        if not isinstance(self.scope, tuple):
            raise TypeError("selection subject scope must be a tuple")
        normalized_scope: list[tuple[AxisRef, CoordinateScalar]] = []
        for term in self.scope:
            if not isinstance(term, tuple) or len(term) != 2:
                raise TypeError(
                    "selection subject scope entries must be "
                    "(AxisRef, coordinate) pairs"
                )
            ref, coordinate = term
            if not isinstance(ref, AxisRef):
                raise TypeError("selection subject scope axis must be AxisRef")
            if ref.domain is AxisDomain.REPEAT:
                raise ValueError(
                    "selection subject repeat scope belongs in repeat_index"
                )
            normalized_scope.append(
                (
                    ref,
                    canonical_coordinate_scalar(
                        coordinate,
                        "selection subject scope coordinate",
                    ),
                )
            )
        repeat_index = self.repeat_index
        if repeat_index is not None:
            if isinstance(repeat_index, bool) or not isinstance(
                repeat_index, Integral
            ):
                raise TypeError(
                    "selection subject repeat_index must be an integer or None"
                )
            repeat_index = int(repeat_index)
            if repeat_index < 0:
                raise ValueError(
                    "selection subject repeat_index cannot be negative"
                )
        object.__setattr__(self, "scope", tuple(normalized_scope))
        object.__setattr__(self, "repeat_index", repeat_index)


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
    #: The uncertainty of each SAMPLE, canonical-unit, or None when the
    #: producer states none.  It rides beside the values through every
    #: reduction so a mean of samples that know their own error can say so
    #: even where there is only one of them to scatter.
    sigma: NDArray[np.float64] | ArrayLike | None = None

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        valid = _readonly(self.valid_mask, dtype=np.bool_)
        if self.value.canonical.shape != shape or valid.shape != shape:
            raise ValueError("sample value and validity must match projection shape")
        if self.sigma is not None:
            sigma = _readonly(self.sigma, dtype=np.float64)
            if sigma.shape != shape:
                raise ValueError("sample sigma must match projection shape")
            object.__setattr__(self, "sigma", sigma)
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
    counts: NDArray[np.int64] | ArrayLike
    source_index: int | None = None
    group_keys: tuple[tuple[AxisValue, ...], ...] = ()
    #: Standard error of each MEAN entry over what this shot pooled, or
    #: None when uncertainty was not requested.  Canonical-only, like the
    #: curve companion.
    sem: NDArray[np.float64] | ArrayLike | None = None

    def __post_init__(self) -> None:
        values = _readonly(self.values)
        valid = _readonly(self.valid, dtype=np.bool_)
        counts = _readonly(self.counts, dtype=np.int64)
        source_index = self.source_index
        if source_index is not None and (
            isinstance(source_index, bool)
            or not isinstance(source_index, Integral)
        ):
            raise TypeError("rolling source_index must be an integer or None")
        if (
            values.ndim != 1
            or valid.shape != values.shape
            or counts.shape != values.shape
        ):
            raise ValueError(
                "rolling sample values, validity, and counts must be one-dimensional"
            )
        if np.any(counts < 0):
            raise ValueError("rolling sample counts cannot be negative")
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
        object.__setattr__(self, "counts", counts)
        object.__setattr__(
            self,
            "source_index",
            None if source_index is None else int(source_index),
        )
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

    def __post_init__(self) -> None:
        group_by = tuple(self.group_by)
        series = tuple(self.series)
        if any(not isinstance(ref, AxisRef) for ref in group_by):
            raise TypeError("group_by must contain AxisRef objects")
        if any(not isinstance(item, CurveSeries) for item in series):
            raise TypeError("series must contain CurveSeries objects")
        object.__setattr__(self, "group_by", group_by)
        object.__setattr__(self, "series", series)


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

    def __post_init__(self) -> None:
        valid = _readonly(self.valid, dtype=np.bool_)
        expected = (self.y.canonical.size, self.x.canonical.size)
        if self.x.canonical.ndim != 1 or self.y.canonical.ndim != 1:
            raise ValueError("image x and y coordinates must be one-dimensional")
        if self.z.canonical.shape != expected or valid.shape != expected:
            raise ValueError("image z and validity do not match its coordinate grid")
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True, slots=True)
class HistogramData:
    revision: int
    generation: str
    edges: QuantityArray
    centers: QuantityArray
    counts: NDArray[np.int64] | ArrayLike

    def __post_init__(self) -> None:
        counts = _readonly(self.counts, dtype=np.int64)
        if self.edges.canonical.ndim != 1 or self.centers.canonical.ndim != 1:
            raise ValueError("histogram edges and centers must be one-dimensional")
        if self.edges.canonical.size != counts.size + 1:
            raise ValueError("histogram requires one more edge than count")
        if self.centers.canonical.size != counts.size:
            raise ValueError("histogram centers and counts must have equal length")
        object.__setattr__(self, "counts", counts)


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
class _ProjectedAxis:
    contract: ResolvedAxis
    coordinate: CoordinateArray
    domain_canonical: NDArray[Any]
    domain_display: NDArray[Any]
    coordinate_labels: tuple[str, ...] | None

    @property
    def declared_domain(self) -> bool:
        return self.contract.declared_domain

    @property
    def dimension(self) -> int:
        return self.contract.dimension


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


@dataclass(frozen=True)
class _ReductionBuckets:
    """The identity of what survives a reduction, and how it is laid out.

    The layout travels with the codes because a consumer that knows which
    kept axis it cares about can read that axis's index straight out of a
    bucket number -- which is how a facet learns each bucket's cell without
    asking every sample.
    """

    codes: NDArray[np.int64]
    count: int
    shape: tuple[int, ...]
    axes: tuple[int, ...]
    strides: tuple[int, ...]
    extents: tuple[int, ...]
    point_groups: NDArray[np.int64] | None

    def axis_index(self, axis: int) -> NDArray[np.int64]:
        """Which index along ``axis`` each bucket number stands for."""

        position = self.axes.index(int(axis))
        numbers = np.arange(self.count, dtype=np.int64)
        return (numbers // self.strides[position]) % self.extents[position]


@dataclass(frozen=True)
class _FacetHistogramPlan:
    """What each histogram cell of one grid will bin, decided once.

    The shared bin edges must cover what is ACTUALLY binned, and with a
    reduced cell that is the per-group statistic rather than the raw
    samples -- so the projection asks for these pools before it chooses the
    edges, and the cells then bin the very same arrays.  Asked twice per
    frame, computed once.
    """

    pools: tuple[NDArray[Any], ...]
    facet_values: tuple[AxisValue, ...]
    pool: NDArray[Any]


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
        "_facet_histogram_cache",
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
        self._axis_cache: dict[AxisRef, _ProjectedAxis] = {}
        self._flat_cache: dict[
            AxisRef,
            tuple[NDArray[Any], NDArray[np.int64]],
        ] = {}
        self._pooled_cache: NDArray[Any] | None = None
        self._positions_cache: NDArray[np.int64] | None = None
        self._facet_histogram_cache: tuple[object, "_FacetHistogramPlan"] | None = None
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
            and inherit_domains_from._schema.fingerprint == schema.fingerprint
            and inherit_domains_from._axis_display_units == overrides
            and inherit_domains_from._unit_registry is registry
            and inherit_domains_from._unit_registry_revision
            == self._unit_registry_revision
        ):
            # Resolved declared axes are immutable schema/unit facts, not
            # revision data.  Rebuilding a two-million-coordinate repeat axis
            # converted its stored tuple, allocated an identity index, gathered
            # the same coordinates and froze two copies on every live frame.
            # Carry the small cache under the same exact context gate as the
            # domains; copy the dict so either view may still resolve another
            # axis without mutating its sibling.
            self._axis_cache = dict(inherit_domains_from._axis_cache)
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
            sigma=snapshot_sigma(snapshot),
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

    @staticmethod
    def _coordinate_index(
        schema: DatasetSchema,
        ref: AxisRef,
        coordinate: CoordinateScalar,
    ) -> int:
        """Resolve one unique source ordinal without materializing a sample plane."""

        found: int | None = None
        for index, candidate in enumerate(resolve_axis(schema, ref).coordinates):
            if candidate != coordinate:
                continue
            if found is not None:
                found = -1
                break
            found = index
        if found is None or found < 0:
            raise ValueError(
                f"selection subject {ref!r} coordinate {coordinate!r} "
                "is not uniquely present"
            )
        return found

    def selection_subject(
        self,
        spec: PlotSpec,
        payload: CurveData | ImageData | HistogramData | FacetData,
        *,
        facet_index: int | None = None,
        source_schema: DatasetSchema | None = None,
    ) -> SelectionSubject:
        """Interaction identity of this already accepted view and payload.

        The payload proves which axes were actually projected; this DataView's
        exact axis contract supplies frames and resolved scope coordinates.
        No schema-only caller can manufacture a subject for a projection that
        was never accepted.
        """

        if not isinstance(
            spec,
            (CurvePlot, ImagePlot, HistogramPlot, RollingPlot, FacetGridPlot),
        ):
            raise TypeError("selection subject requires a dataset PlotSpec")
        if not isinstance(
            payload,
            (CurveData, ImageData, HistogramData, FacetData),
        ):
            raise TypeError("selection subject requires an accepted plot payload")
        # The session hands over its current accepted view/payload pair.
        selected_payload: CurveData | ImageData | HistogramData
        if isinstance(spec, FacetGridPlot):
            if not isinstance(payload, FacetData) or payload.spec != spec:
                raise ValueError("selection subject payload differs from FacetGrid spec")
            if not payload.cells:
                raise ValueError("selection subject FacetGrid has no accepted cells")
            selected_payload = payload.cells[0].payload
        else:
            if isinstance(payload, FacetData):
                raise ValueError("standalone selection subject received FacetData")
            selected_payload = payload
        if (
            selected_payload.revision != self._samples.revision
            or selected_payload.generation != self._samples.generation
        ):
            raise ValueError(
                "selection subject payload differs from its accepted DataView"
            )

        semantic = semantic_spec(spec)
        if isinstance(semantic, CurvePlot):
            if not isinstance(selected_payload, CurveData):
                raise ValueError("accepted curve spec and payload differ")
            x_ref, y_ref = selected_payload.x_ref, None
            if x_ref != semantic.x:
                raise ValueError("accepted curve payload has the wrong x axis")
        elif isinstance(semantic, ImagePlot):
            if not isinstance(selected_payload, ImageData):
                raise ValueError("accepted image spec and payload differ")
            x_ref, y_ref = selected_payload.x_ref, selected_payload.y_ref
            if (x_ref, y_ref) != (semantic.x, semantic.y):
                raise ValueError("accepted image payload has the wrong axes")
        else:
            if isinstance(semantic, HistogramPlot) and not isinstance(
                selected_payload, HistogramData
            ):
                raise ValueError("accepted histogram spec and payload differ")
            # Histogram x is the measured value and Rolling x is a plot-owned
            # history ordinal.  Neither is an upstream axis.
            x_ref = y_ref = None

        x_frame = (
            None
            if x_ref is None
            else self._resolve(x_ref).contract.coordinate_frame
        )
        y_frame = (
            None
            if y_ref is None
            else self._resolve(y_ref).contract.coordinate_frame
        )
        original = self._schema if source_schema is None else source_schema
        if not isinstance(original, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema or None")
        scope: list[tuple[AxisRef, CoordinateScalar]] = []
        repeat_index: int | None = None
        for ref, authored in getattr(spec, "scope", ()):
            coordinate = (
                canonical_coordinate_scalar(
                    self._resolve(ref).contract.coordinates[-1],
                    "selection subject scope coordinate",
                )
                if authored is LATEST_COORDINATE
                else canonical_coordinate_scalar(
                    authored,
                    "selection subject scope coordinate",
                )
            )
            if ref.domain is AxisDomain.REPEAT:
                repeat_index = self._coordinate_index(original, ref, coordinate)
            else:
                scope.append((ref, coordinate))

        if facet_index is not None:
            if isinstance(facet_index, bool) or not isinstance(
                facet_index, Integral
            ):
                raise TypeError("facet_index must be an integer or None")
            if not isinstance(payload, FacetData):
                raise ValueError("facet_index requires an accepted FacetData payload")
            selected_index = int(facet_index)
            if not 0 <= selected_index < len(payload.cells):
                raise ValueError("facet_index is outside the accepted facet payload")
            coordinate = canonical_coordinate_scalar(
                payload.cells[selected_index].facet_value_canonical,
                "selection subject facet coordinate",
            )
            if spec.facet.domain is AxisDomain.REPEAT:
                repeat_index = self._coordinate_index(
                    original, spec.facet, coordinate
                )
            else:
                scope.append((spec.facet, coordinate))

        return SelectionSubject(
            semantic.kind,
            x_ref,
            y_ref,
            x_frame,
            y_frame,
            tuple(scope),
            repeat_index,
        )

    def validate_curve(
        self,
        x: AxisRef,
        *,
        group_by: Iterable[AxisRef] = (),
    ) -> None:
        """Check a curve projection without computing it.

        This is the single validation authority shared by the explicit
        validation API and :meth:`curve`: whatever passes here projects,
        and whatever projects passed here.

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
        exact = self._curve_from_axes(
            x, groups, aggregation, uncertainty=uncertainty
        )
        if exact is not None:
            return exact
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

        A curve whose x/group axes are real tensor dimensions is one masked
        reduction along every other dimension. Flattening it into millions
        of (position, value) pairs and rediscovering those dimensions with
        codes is never legitimate. Point-domain groups and unordered x
        coordinates keep the generic algorithm.
        """

        tensor_domains = (AxisDomain.REPEAT, AxisDomain.DATA)
        if x.domain not in tensor_domains or any(
            group.domain not in tensor_domains for group in groups
        ):
            return None
        try:
            x_resolved = self._resolve(x)
            group_resolved = tuple(self._resolve(group) for group in groups)
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

        dimensions = [int(x_resolved.dimension)]
        group_domains = []
        group_orders = []
        shape = self._samples.value.canonical.shape
        for group, resolved in zip(groups, group_resolved):
            dimension = int(resolved.dimension)
            if dimension in dimensions:
                return None
            dimensions.append(dimension)
            stride = 1
            for size in shape[dimension + 1:]:
                stride *= int(size)
            representatives = np.arange(shape[dimension], dtype=np.int64) * stride
            domain = self._domain(group, representatives)
            codes = np.asarray(domain.codes, dtype=np.int64)
            if (
                domain.canonical.size != shape[dimension]
                or codes.shape != (shape[dimension],)
                or bool((codes < 0).any())
                or np.unique(codes).size != codes.size
            ):
                return None
            order = _inverse_code_order(codes)
            group_domains.append(domain)
            group_orders.append(order)

        return self._dense_curve_data(
            x,
            x_resolved,
            self._samples.value.canonical,
            self._samples.valid_mask,
            aggregation,
            uncertainty,
            self._samples.sigma,
            groups=groups,
            group_domains=tuple(group_domains),
            group_dimensions=tuple(dimensions[1:]),
            group_orders=tuple(group_orders),
        )

    def _dense_curve_data(
        self,
        x: AxisRef,
        x_resolved: "_ProjectedAxis",
        values: NDArray[Any],
        usable: NDArray[np.bool_],
        aggregation: Reduction,
        uncertainty: bool = False,
        sigma: NDArray[Any] | None = None,
        *,
        groups: tuple[AxisRef, ...] = (),
        group_domains: tuple = (),
        group_dimensions: tuple[int, ...] = (),
        group_orders: tuple[NDArray[np.int64] | None, ...] = (),
    ) -> CurveData:
        """One dense curve out of one (possibly row-sliced) value tensor."""

        x_canonical = np.asarray(x_resolved.domain_canonical)
        nx = int(x_canonical.size)

        group_sizes = tuple(int(domain.size) for domain in group_domains)
        combinations = math.prod(group_sizes) if group_sizes else 1

        def laid_out(array: NDArray[Any]) -> NDArray[Any]:
            kept = (int(x_resolved.dimension), *group_dimensions)
            destinations = tuple(range(np.ndim(array) - len(kept), np.ndim(array)))
            moved = np.moveaxis(array, kept, destinations)
            return np.reshape(moved, (-1, nx, combinations), order="C")

        def code_ordered(array: NDArray[Any]) -> NDArray[Any]:
            result = np.asarray(array).reshape((nx, *group_sizes))
            for position, order in enumerate(group_orders):
                if order is not None:
                    result = np.take(result, order, axis=1 + position)
            return result.reshape(nx, combinations)

        moved = laid_out(values)
        moved_usable = laid_out(usable)
        identity = _leading_identity(moved, moved_usable)
        if identity is not None:
            # Nothing is pooled: every kept (x, group...) bucket owns exactly
            # one physical sample.  All five Reduction choices are therefore
            # the identity.  The general reducer still scanned validity and
            # materialised an int64 count plane, then the identity group order
            # copied both full planes once more.
            y, valid_plane = identity
            counts = (
                np.broadcast_to(np.asarray(1, dtype=np.int64), y.shape)
                if _stride_zero_all_true(valid_plane)
                else valid_plane.astype(np.int64)
            )
        else:
            y, counts = _masked_leading_reduce(moved, moved_usable, aggregation)
            valid_plane = None
        y = code_ordered(np.asarray(y, dtype=np.float64))
        counts = code_ordered(counts)
        if valid_plane is not None:
            valid_plane = code_ordered(valid_plane)
        sem = None
        if uncertainty:
            if identity is not None and sigma is None:
                # One sample cannot state a scatter.  With no sample-owned
                # sigma the exact answer is NaN in every identity bucket;
                # reducing values and their squares over two million
                # singleton buckets only rediscovers that invariant.
                sem = np.broadcast_to(
                    np.asarray(np.nan, dtype=np.float64), y.shape
                )
            else:
                # The SEM is the SAME reduction run over the squares: no second
                # kernel, no binomial special case -- for a boolean column the
                # sample spread sqrt(p(1-p)) IS the binomial spread.
                def mean_of_squares(plane: Any, offset: float) -> Any:
                    array = np.asarray(plane)
                    marks = (
                        None
                        if _stride_zero_all_true(moved_usable)
                        else moved_usable
                    )
                    if array.ndim == 3 and array.flags.c_contiguous:
                        from . import _raster_kernels as kernels

                        flat = array.reshape(array.shape[0], -1, 1)
                        flat_marks = (
                            None
                            if marks is None
                            else np.asarray(marks).reshape(flat.shape)
                        )
                        square_sums = kernels.masked_centred_square_sums(
                            flat,
                            offset,
                            flat_marks,
                        )
                        if square_sums is not None:
                            ordered = code_ordered(
                                square_sums.reshape(nx, combinations)
                            )
                            with np.errstate(invalid="ignore", divide="ignore"):
                                return np.where(
                                    counts > 0,
                                    ordered / counts,
                                    np.nan,
                                )
                    reduced, _ = _masked_leading_reduce(
                        np.square(
                            np.asarray(array, dtype=np.float64) - offset
                        ),
                        moved_usable,
                        Reduction.MEAN,
                    )
                    return code_ordered(reduced)

                sem = _sem_of_mean(
                    y,
                    counts,
                    moved,
                    None if sigma is None else laid_out(sigma),
                    mean_of_squares,
                )
        x_quantity = QuantityArray(
            x_canonical,
            np.asarray(x_resolved.domain_display),
            x_resolved.coordinate.canonical_unit,
            x_resolved.coordinate.display_unit,
            x_resolved.coordinate.label,
        )
        series = self._series_from_columns(
            x_quantity,
            _axis_coordinate_labels(x_resolved, x_canonical),
            tuple(group_domains),
            group_sizes,
            y,
            counts,
            sem,
            valid_plane=valid_plane,
        )
        return CurveData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            x_ref=x,
            group_by=groups,
            series=series,
        )

    def _curve_from_axes(
        self,
        x: AxisRef,
        groups: tuple[AxisRef, ...],
        aggregation: Reduction,
        *,
        uncertainty: bool = False,
    ) -> CurveData | None:
        """Exact full-Dataset Curve aggregation without position planes."""

        projection = self._axis_projection((*groups, x))
        domains, axis_codes, dimensions = projection
        domains, values, counts, presence = self._aggregate_axes(
            (*groups, x), aggregation, projection=projection
        )
        group_domains = domains[:-1]
        x_domain = domains[-1]
        nx = int(x_domain.size)

        def columns(array: NDArray[Any]) -> NDArray[Any]:
            return np.moveaxis(np.asarray(array), -1, 0).reshape(nx, -1)

        resolved = self._resolve(x)
        x_quantity = QuantityArray(
            np.asarray(x_domain.canonical),
            np.asarray(x_domain.display),
            resolved.coordinate.canonical_unit,
            resolved.coordinate.display_unit,
            resolved.coordinate.label,
        )
        group_sizes = tuple(int(domain.size) for domain in group_domains)
        sem = None
        if uncertainty:
            shape = (*group_sizes, nx)
            means = np.asarray(values, dtype=np.float64).reshape(shape)
            flat_means = means.reshape((-1, nx))
            references = np.asarray(
                [_sem_reference(row) for row in flat_means],
                dtype=np.float64,
            )
            offsets = np.broadcast_to(
                references[:, None], flat_means.shape
            ).reshape(-1)
            domain_sizes = tuple(int(domain.size) for domain in domains)
            squared = _axis_kernel_aggregate(
                self._samples.value.canonical,
                self._samples.valid_mask,
                axis_codes,
                dimensions,
                domain_sizes,
                Reduction.SUM,
                offsets=offsets,
            )
            if squared is None:
                return None
            square_sums = squared[0].reshape(shape)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_squares = square_sums / counts
            sigma_squares = None
            if self._samples.sigma is not None:
                propagated = _axis_kernel_aggregate(
                    self._samples.sigma,
                    self._samples.valid_mask,
                    axis_codes,
                    dimensions,
                    domain_sizes,
                    Reduction.SUM,
                    offsets=np.zeros(offsets.shape, dtype=np.float64),
                )
                if propagated is None:
                    return None
                with np.errstate(invalid="ignore", divide="ignore"):
                    sigma_squares = propagated[0].reshape(shape) / counts
            sem = _sem_from_moments(
                means - offsets.reshape(shape),
                mean_squares,
                counts,
                sigma_squares,
            )
        return CurveData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            x_ref=x,
            group_by=groups,
            series=self._series_from_columns(
                x_quantity,
                _axis_coordinate_labels(resolved, np.asarray(x_domain.canonical)),
                group_domains,
                group_sizes,
                columns(values),
                columns(counts),
                None if sem is None else columns(sem),
                used_plane=columns(presence),
            ),
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

        point_domains = (
            AxisDomain.POINT_COORDINATE,
            AxisDomain.POINT_DIMENSION,
        )
        point_groups = tuple(group for group in groups if group.domain in point_domains)
        tensor_groups = tuple(group for group in groups if group not in point_groups)
        row_refs = (*point_groups, x)
        planes = self._factored_planes(
            row_refs, tensor_groups, aggregation, uncertainty
        )
        if planes is None:
            return None
        if point_groups:
            point_sizes = planes.row_sizes[:-1]
            nx = planes.row_sizes[-1]
            tensor_sizes = planes.group_sizes
            shape = (*point_sizes, nx, *tensor_sizes)

            def columns(array: NDArray[Any]) -> NDArray[Any]:
                moved = np.moveaxis(np.asarray(array).reshape(shape), len(point_sizes), 0)
                internal = (*point_groups, *tensor_groups)
                permutation = [0] + [
                    1 + internal.index(group) for group in groups
                ]
                return np.transpose(moved, permutation).reshape(nx, -1)

            presence = np.moveaxis(
                planes.row_presence.reshape((*point_sizes, nx)),
                len(point_sizes),
                0,
            )
            presence = np.broadcast_to(
                presence.reshape((nx, *point_sizes, *([1] * len(tensor_sizes)))),
                (nx, *point_sizes, *tensor_sizes),
            )
            internal = (*point_groups, *tensor_groups)
            permutation = [0] + [1 + internal.index(group) for group in groups]
            used = np.transpose(presence, permutation).reshape(nx, -1)
            domain_by_group = {
                group: domain
                for group, domain in zip(point_groups, planes.row_domains[:-1])
            }
            domain_by_group.update(zip(tensor_groups, planes.group_domains))
            group_domains = tuple(domain_by_group[group] for group in groups)
            group_sizes = tuple(int(domain.size) for domain in group_domains)
            series = self._series_from_columns(
                planes.x_quantity,
                planes.x_labels,
                group_domains,
                group_sizes,
                columns(planes.y_plane),
                columns(planes.counts_plane),
                None if planes.sem_plane is None else columns(planes.sem_plane),
                used_plane=used,
            )
            return CurveData(
                revision=self._samples.revision,
                generation=self._samples.generation,
                x_ref=x,
                group_by=groups,
                series=series,
            )
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
        if isinstance(cell, ImagePlot):
            return self._factored_facet_images(spec, cell)
        if not isinstance(cell, CurvePlot):
            return None
        cell_groups = () if cell.group is None else (cell.group,)
        if (
            spec.facet.domain in (AxisDomain.REPEAT, AxisDomain.DATA)
            and cell.x.domain in (AxisDomain.REPEAT, AxisDomain.DATA)
        ):
            combined = self._dense_data_curve(
                cell.x,
                (spec.facet, *cell_groups),
                cell.reduction,
                uncertainty,
            )
            if combined is not None:
                return self._curve_groups_to_facets(
                    spec, cell, cell_groups, combined
                )
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
            combined = self._curve_from_axes(
                cell.x,
                (spec.facet, *cell_groups),
                cell.reduction,
                uncertainty=uncertainty,
            )
            if combined is None:
                return None
            return self._curve_groups_to_facets(
                spec, cell, cell_groups, combined
            )
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

    def _curve_groups_to_facets(
        self,
        spec: FacetGridPlot,
        cell: CurvePlot,
        cell_groups: tuple[AxisRef, ...],
        combined: CurveData,
    ) -> FacetData:
        """Split one `(facet, *groups)` Curve projection into cell views."""

        grouped: list[tuple[AxisValue, list[CurveSeries]]] = []
        for series in combined.series:
            facet_value = series.group_key[0]
            if not grouped or grouped[-1][0] != facet_value:
                grouped.append((facet_value, []))
            key = series.group_key[1:]
            label = self._samples.value.label if not key else ", ".join(
                item.label for item in key
            )
            grouped[-1][1].append(CurveSeries(
                x=series.x,
                x_labels=series.x_labels,
                y=series.y,
                valid=series.valid,
                counts=series.counts,
                sem=series.sem,
                group_key=key,
                label=label,
            ))
        return FacetData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            spec=spec,
            cells=tuple(
                FacetCell(
                    facet_index=index,
                    facet_value_canonical=value.canonical,
                    facet_value_display=value.display,
                    label=value.label,
                    payload=CurveData(
                        revision=self._samples.revision,
                        generation=self._samples.generation,
                        x_ref=cell.x,
                        group_by=cell_groups,
                        series=tuple(series),
                    ),
                )
                for index, (value, series) in enumerate(grouped)
            ),
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
            domains, z, counts, presence = self._aggregate_axes(
                (spec.facet, cell.y, cell.x), cell.reduction
            )
            facet_domain, y_domain, x_domain = domains
            cells = []
            for facet_index, facet_value in enumerate(facet_domain.values):
                geometry = presence[facet_index]
                if not bool(geometry.any()):
                    continue
                cells.append(FacetCell(
                    facet_index=len(cells),
                    facet_value_canonical=facet_value.canonical,
                    facet_value_display=facet_value.display,
                    label=facet_value.label,
                    payload=self._image_from_planes(
                        cell.x,
                        cell.y,
                        x_domain,
                        y_domain,
                        z[facet_index],
                        counts[facet_index],
                        used_y=geometry.any(axis=1),
                        used_x=geometry.any(axis=0),
                    ),
                ))
            return FacetData(
                revision=self._samples.revision,
                generation=self._samples.generation,
                spec=spec,
                cells=tuple(cells),
            )
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
            order = _inverse_code_order(codes)
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
                if order is not None:
                    plane = np.take(plane, order, axis=1 + position)
            return plane.reshape(rows, combos)

        counts_pg = code_ordered(counts_pg)
        moments_pg = code_ordered(moments_pg)

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
        if uncertainty:
            # Per (row, group) first, then folded by the same combined row
            # key as the means: one kernel, two stages, and einsum fuses the
            # square into the sum so no copy of the tensor is materialised.
            letters = "abcdefghijklmnopqrstuvwxyz"[: values.ndim]
            output = "".join(
                letters[axis]
                for axis in range(values.ndim)
                if axis == 1 or axis in kept_dims
            )

            def mean_of_squares(plane: Any, offset: float) -> Any:
                plane = np.asarray(plane, dtype=np.float64)
                per_group = _centred_square_sums(
                    plane,
                    offset,
                    None if all_valid else usable,
                    remaining,
                    shape,
                )
                if per_group is None and all_valid:
                    centred = plane - offset
                    per_group = np.einsum(
                        f"{letters},{letters}->{output}", centred, centred
                    )
                elif per_group is None:
                    per_group = np.sum(
                        np.square(plane - offset),
                        axis=reduce_axes,
                        where=usable,
                        dtype=np.float64,
                    )
                folded, _ = _aggregate_by_codes(
                    code_ordered(per_group).reshape(-1),
                    np.ones(fold_codes.shape, dtype=np.bool_),
                    fold_codes,
                    buckets,
                    Reduction.SUM,
                )
                folded = np.nan_to_num(folded, nan=0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.where(counts > 0, folded / counts, np.nan)

            sem_flat = _sem_of_mean(
                np.asarray(y_flat, np.float64),
                counts,
                as_double,
                self._samples.sigma,
                mean_of_squares,
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
        return self._series_from_columns(
            x_quantity,
            x_labels,
            group_domains,
            group_sizes,
            y_plane,
            counts_plane,
            sem_plane,
        )

    def _series_from_columns(
        self,
        x_quantity: QuantityArray,
        x_labels: tuple[str, ...] | None,
        group_domains: tuple,
        group_sizes: tuple[int, ...],
        y_plane: NDArray[np.float64],
        counts_plane: NDArray[np.int64],
        sem_plane: NDArray[np.float64] | None,
        *,
        used_plane: NDArray[np.bool_] | None = None,
        valid_plane: NDArray[np.bool_] | None = None,
    ) -> tuple[CurveSeries, ...]:
        """One shared x column and its tensor-ordered group columns."""

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
            shown_x = x_quantity
            shown_labels = x_labels
            if used_plane is not None:
                used = np.asarray(used_plane[:, flat_index], dtype=np.bool_)
            else:
                used = None
            if used is not None and not bool(used.all()):
                shown_x = QuantityArray(
                    np.asarray(x_quantity.canonical)[used],
                    np.asarray(x_quantity.display)[used],
                    x_quantity.canonical_unit,
                    x_quantity.display_unit,
                    x_quantity.label,
                )
                shown_labels = (
                    None
                    if x_labels is None
                    else tuple(label for label, keep in zip(x_labels, used) if keep)
                )
                y_column = y_column[used]
                counts_column = counts_column[used]
                if sem_column is not None:
                    sem_column = sem_column[used]
            valid_column = (
                np.asarray(valid_plane[:, flat_index], dtype=np.bool_)
                if valid_plane is not None
                else (counts_column > 0) & np.isfinite(y_column)
            )
            # These arrays are either immutable source views or fresh results
            # owned by this projection.  Seal the latter before the public
            # immutable wrappers consume them, otherwise their validators make
            # a second full-size safety copy of storage no caller can mutate.
            for array in (y_column, counts_column, valid_column):
                if array.flags.writeable:
                    array.setflags(write=False)
            y_display = value.canonical_unit.convert_value_to(
                y_column, value.display_unit
            )
            if y_display.flags.writeable:
                y_display.setflags(write=False)
            label = value.label if not key else ", ".join(
                item.label for item in key
            )
            series.append(
                CurveSeries(
                    x=shown_x,
                    x_labels=shown_labels,
                    y=QuantityArray(
                        y_column,
                        y_display,
                        value.canonical_unit,
                        value.display_unit,
                        value.label,
                    ),
                    valid=valid_column,
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
                def mean_of_squares(plane: Any, offset: float) -> Any:
                    reduced, _ = _aggregate_by_codes(
                        np.square(
                            np.asarray(plane, dtype=np.float64) - offset
                        ),
                        usable,
                        x_domain.codes,
                        x_domain.size,
                        Reduction.MEAN,
                    )
                    return reduced

                sem = _sem_of_mean(
                    np.asarray(y, np.float64),
                    counts,
                    group_values,
                    self._flat_sigma_at(group_positions),
                    mean_of_squares,
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
        return self._image_from_axes(x, y, aggregation)

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

        tensor_domains = (AxisDomain.REPEAT, AxisDomain.DATA)
        if x.domain not in tensor_domains or y.domain not in tensor_domains:
            return None
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
        x_resolved: "_ProjectedAxis",
        y_resolved: "_ProjectedAxis",
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
        identity = _leading_identity(moved, moved_usable)
        if identity is not None:
            z, valid = identity
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

    def _axis_projection(
        self, refs: tuple[AxisRef, ...]
    ) -> tuple[tuple[_Domain, ...], tuple[NDArray[np.int64], ...], tuple[int, ...]]:
        """Resolve each kept axis to its small code vector and tensor dimension."""

        shape = self._samples.value.canonical.shape
        domains = []
        axis_codes = []
        dimensions = []
        for ref in refs:
            resolved = self._resolve(ref)
            dimension = int(resolved.dimension)
            stride = 1
            for size in shape[dimension + 1:]:
                stride *= int(size)
            representatives = np.arange(shape[dimension], dtype=np.int64) * stride
            domain = self._domain(ref, representatives)
            codes = np.asarray(domain.codes, dtype=np.int64)
            axis_codes.append(codes)
            dimensions.append(dimension)
            domains.append(domain)
        return tuple(domains), tuple(axis_codes), tuple(dimensions)

    def _aggregate_axes(
        self,
        refs: tuple[AxisRef, ...],
        aggregation: Reduction,
        *,
        projection: tuple[
            tuple[_Domain, ...],
            tuple[NDArray[np.int64], ...],
            tuple[int, ...],
        ] | None = None,
    ) -> tuple[tuple[_Domain, ...], NDArray[Any], NDArray[np.int64], NDArray[np.bool_]]:
        """Aggregate the full Dataset once by small per-axis code vectors.

        Each coordinate is fixed by one physical tensor dimension. Combining
        those axis-sized codes by broadcasting preserves the generic path's
        row-major bucket order without materialising positions or one full
        coordinate plane per axis. The returned presence ignores value
        validity and therefore describes geometry, not measurement success.
        """

        shape = self._samples.value.canonical.shape
        domains, axis_codes, dimensions = (
            self._axis_projection(refs) if projection is None else projection
        )
        domain_sizes = tuple(int(domain.size) for domain in domains)
        compiled = _axis_kernel_aggregate(
            self._samples.value.canonical,
            self._samples.valid_mask,
            tuple(axis_codes),
            tuple(dimensions),
            domain_sizes,
            aggregation,
        )
        if compiled is not None:
            reduced, counts, present = compiled
            return (
                tuple(domains),
                reduced.reshape(domain_sizes),
                counts.reshape(domain_sizes),
                present.reshape(domain_sizes),
            )
        combined: Any = np.int64(0)
        admitted: Any = np.bool_(True)
        for dimension, domain, codes in zip(
            dimensions, domains, axis_codes, strict=True
        ):
            reshape = [1] * len(shape)
            reshape[dimension] = codes.size
            placed = codes.reshape(reshape)
            combined = combined * int(domain.size) + np.where(
                placed >= 0, placed, 0
            )
            admitted = admitted & (placed >= 0)
        full_codes = np.broadcast_to(combined, shape).reshape(-1)
        full_admitted = np.broadcast_to(admitted, shape).reshape(-1)
        full_codes = np.where(full_admitted, full_codes, -1)
        usable = (
            np.asarray(
                np.broadcast_to(self._samples.valid_mask, shape),
                dtype=np.bool_,
            ).reshape(-1)
            & full_admitted
        )
        bucket_count = math.prod(domain_sizes)
        reduced, counts = _aggregate_by_codes(
            self._samples.value.canonical.reshape(-1),
            usable,
            full_codes,
            bucket_count,
            aggregation,
        )
        present = np.bincount(
            full_codes[full_admitted], minlength=bucket_count
        ) > 0
        out_shape = domain_sizes
        return (
            tuple(domains),
            np.asarray(reduced).reshape(out_shape),
            np.asarray(counts).reshape(out_shape),
            np.asarray(present).reshape(out_shape),
        )

    def _image_from_axes(
        self,
        x: AxisRef,
        y: AxisRef,
        aggregation: Reduction,
    ) -> ImageData:
        domains, z, counts, _presence = self._aggregate_axes(
            (y, x), aggregation
        )
        y_domain, x_domain = domains
        return self._image_from_planes(
            x, y, x_domain, y_domain, z, counts
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
        values: NDArray[Any] | None = None,
        valid: NDArray[np.bool_] | None = None,
        reduce_axes: Sequence[AxisRef] = (),
        aggregation: Reduction = Reduction.MEAN,
    ) -> HistogramData:
        """Distribution of the acquired values.

        Every axis pools into the one distribution unless it is named in
        ``reduce_axes``, which collapses it under ``aggregation`` first --
        the difference between the distribution of every shot and the
        distribution of each site's mean over shots.
        """

        selected, usable = self.histogram_pool(
            values=values,
            valid=valid,
            reduce_axes=reduce_axes,
            aggregation=aggregation,
        )
        return self._histogram_from_values(bins, selected, valid=usable)

    def _reduction_plan(
        self, refs: Sequence[AxisRef]
    ) -> tuple[tuple[int, ...], tuple[AxisRef, ...]]:
        """What a reduction names: whole tensor axes, and point coordinates.

        A POINT COORDINATE IS NOT A TENSOR AXIS.  Every point column and
        every topology dimension resolves to dimension 1 -- the shared point
        axis -- so mapping a ref straight to a numpy axis collapsed the WHOLE
        point table for any of them, and a set() of dimensions made two
        different columns literally the same reduction: on a detuning x power
        scan, reducing detuning and reducing power were two fate rows
        producing one byte-identical answer, neither of them the one the row
        named.

        Said once here because it is asked from two directions: a whole
        tensor (a standalone histogram) and a set of sample positions (one
        facet cell), which must agree on what survives.
        """

        coordinates: list[AxisRef] = []
        dimensions: set[int] = set()
        for ref in refs:
            if ref.domain in (
                AxisDomain.POINT_COORDINATE,
                AxisDomain.POINT_DIMENSION,
            ):
                coordinates.append(ref)
            else:
                dimensions.add(int(self._resolve(ref).dimension))
        if 1 in dimensions:
            # The point axis goes whole, so naming one of its coordinates
            # adds nothing to what is already being collapsed.
            coordinates = []
        return tuple(sorted(dimensions)), tuple(coordinates)

    def _point_group_codes(
        self, coordinates: Sequence[AxisRef]
    ) -> tuple[NDArray[np.int64], int]:
        """One group code per POINT ROW, from the coordinates NOT named."""

        named = {self._resolve(ref).contract.axis_id for ref in coordinates}
        table = self._schema.point_table
        kept = tuple(
            column
            for column in table.columns
            if column.coordinate_id not in named
        )
        return _point_row_codes(kept, table.row_count)

    def _reduction_buckets(
        self,
        dimensions: Sequence[int],
        coordinates: Sequence[AxisRef],
        *,
        shape: tuple[int, ...] | None = None,
    ) -> "_ReductionBuckets":
        """One bucket code per sample, naming what survives the reduction.

        The bucket IS the identity of what is left when the named axes are
        gone: the kept tensor indices, with the point axis standing for the
        group its row falls in when a point coordinate is named.  Two
        different readers want it -- a standalone histogram, which bins the
        buckets, and a faceted one, which also asks which cell each bucket
        fell in -- and they must agree on what survives, so it is built once
        here.

        Returns the codes together with the layout they were built on, so a
        consumer can read one kept axis's index out of a bucket number.
        """

        shape = tuple(self._samples.value.canonical.shape) if shape is None else shape
        collapse = set(int(axis) for axis in dimensions)
        point_codes: NDArray[np.int64] | None = None
        point_count = 0
        if coordinates:
            point_codes, point_count = self._point_group_codes(coordinates)

        def extent(axis: int) -> int:
            if axis == 1 and point_codes is not None:
                return point_count
            return int(shape[axis])

        keep_axes = [axis for axis in range(len(shape)) if axis not in collapse]
        out_shape = tuple(extent(axis) for axis in keep_axes)
        stride, strides = 1, {}
        for axis in reversed(keep_axes):
            strides[axis] = stride
            stride *= extent(axis)

        def described(codes: NDArray[np.int64]) -> "_ReductionBuckets":
            return _ReductionBuckets(
                codes=codes,
                count=max(1, stride),
                shape=out_shape,
                axes=tuple(keep_axes),
                strides=tuple(strides[axis] for axis in keep_axes),
                extents=out_shape,
                point_groups=point_codes,
            )

        index = np.indices(shape, sparse=True)
        bucket = np.zeros(shape, dtype=np.int64)
        for axis in keep_axes:
            place = (
                point_codes[index[1]]
                if axis == 1 and point_codes is not None
                else index[axis]
            )
            # In place: each kept axis otherwise allocated a whole
            # sample-sized plane to add one term to.
            bucket += place * strides[axis]
        return described(bucket)

    def _collapse_axes(
        self,
        values: NDArray[Any],
        valid: NDArray[np.bool_],
        refs: Sequence[AxisRef],
        aggregation: Reduction,
    ) -> tuple[NDArray[Any], NDArray[np.bool_]]:
        """Collapse whole box axes, keeping validity honest.

        A collapsed cell is valid when it had anything to collapse; the
        aggregation reads only the usable entries, so a partly invalid row
        still reports the statistic of what was measured.
        """

        if not isinstance(aggregation, Reduction):
            raise TypeError("aggregation must be Reduction")
        dimensions, coordinates = self._reduction_plan(refs)
        if not dimensions and not coordinates:
            return values, valid

        usable = np.asarray(np.broadcast_to(valid, values.shape), dtype=bool)
        if coordinates:
            return self._collapse_by_coordinates(
                values, usable, dimensions, coordinates, aggregation
            )

        axes = dimensions
        counts = np.count_nonzero(usable, axis=axes)
        present = counts > 0
        as_double = values.astype(np.float64, copy=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if aggregation in (Reduction.MEAN, Reduction.SUM):
                totals = np.sum(as_double, axis=axes, where=usable)
                collapsed = (
                    np.divide(
                        totals,
                        counts,
                        out=np.zeros_like(totals),
                        where=present,
                    )
                    if aggregation is Reduction.MEAN
                    else totals
                )
            elif aggregation in (Reduction.MIN, Reduction.MAX):
                ufunc = np.min if aggregation is Reduction.MIN else np.max
                collapsed = ufunc(
                    as_double,
                    axis=axes,
                    where=usable,
                    initial=(
                        np.inf if aggregation is Reduction.MIN else -np.inf
                    ),
                )
            elif aggregation is Reduction.FIRST:
                # FIRST used to fall into the else branch above and come
                # back as MAX -- no error, no warning, just a different
                # statistic drawn under the name the operator chose.  Every
                # other reducer in this file dispatches it explicitly.
                collapsed, present = _leading_along_axes(
                    as_double, usable, axes
                )
            else:
                raise AssertionError(f"unsupported reduction: {aggregation!r}")
        return collapsed, present

    def _collapse_by_coordinates(
        self,
        values: NDArray[Any],
        usable: NDArray[np.bool_],
        dimensions: Sequence[int],
        coordinates: Sequence[AxisRef],
        aggregation: Reduction,
    ) -> tuple[NDArray[Any], NDArray[np.bool_]]:
        """Collapse named point coordinates, keeping the rest apart.

        Reducing "detuning" over a detuning x power scan means one value
        per power, not one value for the whole scan: the point rows are
        grouped by the coordinates NOT named, and each group is reduced.
        The whole tensor axes named alongside are reduced in the same pass,
        so a joint reduction stays joint -- a mean over repeats and
        detuning together is one mean, not a mean of means.
        """

        buckets = self._reduction_buckets(
            dimensions, coordinates, shape=values.shape
        )
        out_shape = buckets.shape
        collapsed, counts = _aggregate_by_codes(
            values.astype(np.float64, copy=False).reshape(-1),
            usable.reshape(-1),
            np.ascontiguousarray(buckets.codes).reshape(-1),
            buckets.count,
            aggregation,
        )
        present = (counts > 0).reshape(out_shape)
        collapsed = np.where(present, collapsed.reshape(out_shape), 0.0)
        return collapsed, present

    def history_validity(self, window: int) -> NDArray[np.bool_]:
        """Which samples the last ``window`` shots contribute, over the WHOLE shape.

        The selection rule, said once and in the sample space every other
        projection speaks.  ``history_values`` narrows this to the repeats
        that can carry it, which is a saving and not a second rule; a facet
        cannot take that narrowing -- its cells are indexed in the original
        space -- so it takes this plane instead.
        """

        window = _history_window(window)
        values = self._samples.value.canonical
        validity = self._samples.valid_mask
        if self.has_primary_index:
            point_mask = self._history_point_mask(window)
            return (
                point_mask
                if _stride_zero_all_true(validity)
                else np.asarray(validity, dtype=np.bool_) & point_mask
            )
        count = min(window, max(1, schema_repeat_count(self._schema)))
        keep = np.zeros(values.shape[0], dtype=np.bool_)
        keep[values.shape[0] - count:] = True
        repeat_mask = np.broadcast_to(
            keep.reshape(-1, *([1] * (values.ndim - 1))), values.shape
        )
        return (
            repeat_mask
            if _stride_zero_all_true(validity)
            else np.asarray(validity, dtype=np.bool_) & repeat_mask
        )

    def _history_point_mask(self, window: int) -> NDArray[np.bool_]:
        """The indexed-history selection, as a plane over the sample shape."""

        values = self._samples.value.canonical
        column = self._schema.point_table.column(PRIMARY_INDEX_AXIS_ID)
        ordered: list[int] = []
        seen: set[int] = set()
        for value in column.values:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(
                    "primary-index coordinates must be integer relative offsets"
                )
            source_index = int(value)
            if not ordered or ordered[-1] != source_index:
                if source_index in seen or (
                    ordered and source_index < ordered[-1]
                ):
                    raise ValueError(
                        "primary-index offsets must form ordered cells"
                    )
                ordered.append(source_index)
                seen.add(source_index)
        if ordered and ordered[-1] != 0:
            raise ValueError(
                "relative primary-index coordinates must end at latest offset 0"
            )
        source_indices = tuple(ordered[-window:])
        selected = np.isin(
            np.asarray(column.values, dtype=object),
            np.asarray(source_indices, dtype=object),
        )
        return np.broadcast_to(
            np.reshape(selected, (1, selected.size, *([1] * (values.ndim - 2)))),
            values.shape,
        )

    def history_values(
        self, window: int
    ) -> tuple[NDArray[Any], NDArray[np.bool_]]:
        """Return the last accepted history cells without binning policy.

        The same selection as ``history_validity``, narrowed where it can be:
        an unindexed dataset carries its shots on the repeat axis, so the
        repeats outside the window are dropped rather than masked, and a
        window of two over a thousand repeats reads two.
        """

        window = _history_window(window)
        values = self._samples.value.canonical
        validity = self._samples.valid_mask
        if self.has_primary_index:
            return values, self.history_validity(window)
        count = min(window, max(1, schema_repeat_count(self._schema)))
        return values[-count:], validity[-count:]

    def pooled_values(self) -> NDArray[Any]:
        """Every value this revision would pool, in canonical units.

        Canonical values for one whole-revision rolling reduction.  The
        display conversion happens only after the reduction.
        """

        cached = self._pooled_cache
        if cached is not None:
            return cached
        pooled = self._pool(self._samples.value.canonical)
        pooled.setflags(write=False)
        self._pooled_cache = pooled
        return pooled

    def _pool(self, plane: NDArray[Any]) -> NDArray[Any]:
        """One plane of this revision, flattened to what a pool contains.

        Which samples a whole-revision reduction pools is a fact about the
        VALIDITY, not about the plane being pooled, so values and the
        samples' own sigma go through the same door and come back the same
        length.  Two doors is how the sigma of a 900-sample pool ended up
        beside the values of an 870-sample one.
        """

        valid = self._samples.valid_mask
        plane = np.asarray(plane)
        if _stride_zero_all_true(valid) or bool(np.all(valid)):
            return plane.reshape(-1).view()
        return plane[valid].reshape(-1)

    def _pooled_sigma(self) -> NDArray[np.float64] | None:
        """The samples' own sigma, pooled exactly as the values are."""

        sigma = self._samples.sigma
        return None if sigma is None else self._pool(sigma)

    def _flat_sigma_at(
        self, positions: NDArray[np.int64]
    ) -> NDArray[np.float64] | None:
        """The samples' own sigma gathered at the SAME positions as values.

        Every generic bucket reduction gathers ``flat_values[positions]``;
        this is that gather for the other plane, so the two can never come
        back indexed differently.
        """

        sigma = self._samples.sigma
        if sigma is None:
            return None
        return np.asarray(sigma, dtype=np.float64).reshape(-1)[positions]

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

        The MEAN's standard error and count are always computed alongside;
        whether the operator shows the band is only a display choice.
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
                # square sum is one BLAS dot.  Masked sums otherwise, never
                # a gather of the finite subset: the copy cost more than
                # the moment.
                hole_free = bool(pooled.size) and math.isfinite(
                    float(np.sum(pooled, dtype=np.float64))
                )
                finite = None if hole_free else np.isfinite(pooled).reshape(-1)
                count = (
                    int(pooled.size)
                    if hole_free
                    else int(np.count_nonzero(finite))
                )

                def mean_of_squares(plane: Any, offset: float) -> Any:
                    if not count:
                        return np.asarray([np.nan])
                    total = _centred_square_sum(
                        np.asarray(plane).reshape(-1),
                        offset,
                        None if hole_free else finite,
                    )
                    return np.asarray([total / count])

                sem = _sem_of_mean(
                    np.asarray([value], dtype=np.float64),
                    np.asarray([count], dtype=np.int64),
                    pooled,
                    self._pooled_sigma(),
                    mean_of_squares,
                )
            return RollingSample(
                revision=self._samples.revision,
                generation=self._samples.generation,
                values=np.asarray([value], dtype=np.float64),
                valid=np.asarray([pooled.size > 0 and np.isfinite(value)]),
                counts=np.asarray([pooled.size], dtype=np.int64),
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
            def mean_of_squares(plane: Any, offset: float) -> Any:
                reduced, _ = _aggregate_by_codes(
                    np.square(np.asarray(plane, dtype=np.float64) - offset),
                    usable,
                    codes,
                    domain_size,
                    Reduction.MEAN,
                )
                return reduced

            sem = _sem_of_mean(
                np.asarray(values, np.float64),
                counts,
                group_values,
                self._flat_sigma_at(positions),
                mean_of_squares,
            )
        valid = (counts > 0) & np.isfinite(values)
        return RollingSample(
            revision=self._samples.revision,
            generation=self._samples.generation,
            values=values,
            valid=valid,
            counts=counts,
            group_keys=keys,
            sem=sem,
        )

    def rolling_history_samples(
        self,
        *,
        group: AxisRef | None = None,
        aggregation: Reduction = Reduction.MEAN,
        uncertainty: bool = True,
    ) -> tuple[RollingSample, ...]:
        """Expand the repeat axis into per-shot rolling samples, oldest first.

        A static snapshot carries its shot history on the repeat axis; each
        repeat reduces to one rolling sample exactly as :meth:`rolling_sample`
        reduces one whole revision.  A snapshot without repeats degenerates to
        the single whole-revision sample.

        ``uncertainty`` is whether the caller will DRAW the band.  Its
        standard error needs a second pass over every value -- squared,
        masked and reduced again -- which the rolling panel paid on every
        revision whether or not the band was switched on.
        """

        if self.has_primary_index:
            return self._history_samples_by_primary_index(
                group=group,
                aggregation=aggregation,
                uncertainty=uncertainty,
            )
        repeats = schema_repeat_count(self._schema)
        if repeats <= 1:
            return (
                self.rolling_sample(group=group, aggregation=aggregation),
            )
        self.validate_rolling(group)
        aggregation = _validate_aggregation(aggregation)
        tensor = self._repeat_history_tensor(
            group=group,
            aggregation=aggregation,
            repeats=repeats,
            uncertainty=uncertainty,
        )
        if tensor is not None:
            return tensor
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
        combined = repeat_of_position * domain_size + codes
        bucket_count = repeats * domain_size
        values, counts = _aggregate_by_codes(
            position_values,
            usable,
            combined,
            bucket_count,
            aggregation,
        )
        values = np.asarray(values).reshape(repeats, domain_size)
        counts = np.asarray(counts).reshape(repeats, domain_size)
        sem = None
        if uncertainty and aggregation is Reduction.MEAN:
            # The band's standard error is a SECOND full pass -- a float64
            # copy of every value, squared, then reduced again.  The panel
            # paid it on every revision whether or not the band was drawn.
            def mean_of_squares(plane: Any, offset: float) -> Any:
                reduced, _ = _aggregate_by_codes(
                    np.square(np.asarray(plane, dtype=np.float64) - offset),
                    usable,
                    combined,
                    bucket_count,
                    Reduction.MEAN,
                )
                return np.asarray(reduced, dtype=np.float64).reshape(
                    repeats, domain_size
                )

            sem = _sem_of_mean(
                np.asarray(values, dtype=np.float64),
                counts,
                position_values,
                self._flat_sigma_at(positions),
                mean_of_squares,
            )
        valid = (counts > 0) & np.isfinite(values)
        return tuple(
            RollingSample(
                revision=self._samples.revision,
                generation=self._samples.generation,
                values=values[index],
                valid=valid[index],
                counts=counts[index],
                group_keys=keys,
                sem=None if sem is None else sem[index],
            )
            for index in range(repeats)
        )

    def _repeat_history_tensor(
        self,
        *,
        group: AxisRef | None,
        aggregation: Reduction,
        repeats: int,
        uncertainty: bool = True,
    ) -> tuple[RollingSample, ...] | None:
        """Reduce a regular repeat history once, not once per repeat.

        Runtime remains the only history owner; this is only a projection of
        its immutable Dataset.  A DATA group is one retained tensor axis.
        Anything whose group/domain cannot be proven one-to-one returns None
        and keeps the generic position path above.
        """

        values = self._samples.value.canonical
        usable = self._samples.valid_mask
        if values.shape[0] != repeats:
            return None
        if group is None:
            group_count = 1
            keys: tuple[tuple[AxisValue, ...], ...] = ((),)

            def cube(plane: Any) -> NDArray[Any]:
                return np.asarray(plane).reshape(repeats, 1, -1)

        else:
            if group.domain is not AxisDomain.DATA:
                return None
            resolved = self._resolve(group)
            dimension = int(resolved.dimension)
            if dimension <= 0 or dimension >= values.ndim:
                return None
            group_count = int(values.shape[dimension])
            stride = int(np.prod(values.shape[dimension + 1 :], dtype=np.int64))
            representatives = np.arange(group_count, dtype=np.int64) * stride
            domain = self._domain(group, representatives)
            if domain.size != group_count or not np.array_equal(
                domain.codes,
                np.arange(group_count, dtype=np.int64),
            ):
                return None
            keys = tuple((value,) for value in domain.values)

            def cube(plane: Any) -> NDArray[Any]:
                return np.moveaxis(np.asarray(plane), dimension, 1).reshape(
                    repeats, group_count, -1
                )

        # One layout, applied to every plane: values, validity and the
        # samples' own sigma cannot end up shaped differently.
        value_cube = cube(values)
        usable_cube = cube(usable)

        # The generic bucket reducer always accumulates numerics in float64;
        # matching that here also prevents integer SUM/square overflow.
        working = value_cube.astype(np.float64, copy=False)
        reduced, counts = _masked_leading_reduce(
            np.moveaxis(working, -1, 0),
            np.moveaxis(usable_cube, -1, 0),
            aggregation,
        )
        reduced = np.asarray(reduced, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.int64)
        reduced = np.where(counts > 0, reduced, np.nan)
        sem = None
        if uncertainty and aggregation is Reduction.MEAN:
            # ``np.square`` on the whole cube materialises a second copy of
            # every value in the history -- sixteen megabytes on a
            # two-million-sample pool -- before reducing it.  einsum sums the
            # products in one pass, and over the SAME masked values, so the
            # moment it feeds is identical.
            leading_usable = np.moveaxis(usable_cube, -1, 0)

            def mean_of_squares(plane: Any, offset: float) -> Any:
                leading = np.moveaxis(
                    cube(np.asarray(plane, dtype=np.float64)), -1, 0
                )
                # Invalid entries become the offset so the shift below
                # leaves them at zero -- one temporary, not two.
                masked = np.where(leading_usable, leading, offset)
                masked -= offset
                squared_sum = np.einsum("i...,i...->...", masked, masked)
                return np.divide(
                    squared_sum,
                    counts,
                    out=np.zeros_like(squared_sum, dtype=np.float64),
                    where=counts > 0,
                )

            sem = _sem_of_mean(
                reduced,
                counts,
                values,
                self._samples.sigma,
                mean_of_squares,
            )
        valid = (counts > 0) & np.isfinite(reduced)
        return tuple(
            RollingSample(
                revision=self._samples.revision,
                generation=self._samples.generation,
                values=reduced[index],
                valid=valid[index],
                counts=counts[index],
                group_keys=keys,
                sem=None if sem is None else sem[index],
            )
            for index in range(repeats)
        )

    def _history_samples_by_primary_index(
        self,
        *,
        group: AxisRef | None,
        aggregation: Reduction,
        uncertainty: bool = True,
    ) -> tuple[RollingSample, ...]:
        """Reduce every authored primary-index cell without arrival history."""

        self.validate_rolling(group)
        aggregation = _validate_aggregation(aggregation)
        positions = self._all_positions()
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
        flat_values = self._samples.value.canonical.reshape(-1)
        flat_valid = self._samples.valid_mask.reshape(-1)
        usable = flat_valid[positions] & (codes >= 0) & (primary.codes >= 0)
        position_values = flat_values[positions]
        history_count = len(primary.values)
        combined = primary.codes * domain_size + codes
        bucket_count = history_count * domain_size
        values, counts = _aggregate_by_codes(
            position_values,
            usable,
            combined,
            bucket_count,
            aggregation,
        )
        values = np.asarray(values).reshape(history_count, domain_size)
        counts = np.asarray(counts).reshape(history_count, domain_size)
        sem = None
        if uncertainty and aggregation is Reduction.MEAN:
            def mean_of_squares(plane: Any, offset: float) -> Any:
                reduced, _ = _aggregate_by_codes(
                    np.square(np.asarray(plane, dtype=np.float64) - offset),
                    usable,
                    combined,
                    bucket_count,
                    Reduction.MEAN,
                )
                return np.asarray(reduced, dtype=np.float64).reshape(
                    history_count, domain_size
                )

            sem = _sem_of_mean(
                np.asarray(values, dtype=np.float64),
                counts,
                position_values,
                self._flat_sigma_at(positions),
                mean_of_squares,
            )
        valid = (counts > 0) & np.isfinite(values)
        return tuple(
            RollingSample(
                revision=self._samples.revision,
                generation=self._samples.generation,
                values=values[index],
                valid=valid[index],
                counts=counts[index],
                source_index=int(source.canonical),
                group_keys=keys,
                sem=None if sem is None else sem[index],
            )
            for index, source in enumerate(primary.values)
        )

    def histogram_pool(
        self,
        *,
        values: NDArray[Any] | None = None,
        valid: NDArray[np.bool_] | None = None,
        reduce_axes: Sequence[AxisRef] = (),
        aggregation: Reduction = Reduction.MEAN,
    ) -> tuple[NDArray[Any], NDArray[np.bool_]]:
        """The values a histogram will ACTUALLY bin, and their validity.

        Raw samples, or the history window, or the per-group statistic when
        axes are reduced -- whichever this spec means.  It is a separate
        question from binning them because the bin domain has to cover what
        is binned: taken from the raw pool instead, a reduced histogram --
        whose values are means, and therefore narrower by construction --
        landed in two bins out of twelve.
        """

        if (values is None) != (valid is None):
            raise ValueError("histogram pool values and validity must appear together")
        selected = self._samples.value.canonical if values is None else values
        usable = self._samples.valid_mask if valid is None else valid
        if reduce_axes:
            selected, usable = self._collapse_axes(
                selected, usable, reduce_axes, aggregation
            )
        return selected, usable

    def facet_histogram_pool(
        self,
        spec: FacetGridPlot,
        *,
        window: int = 1,
    ) -> tuple[NDArray[Any], NDArray[np.bool_]]:
        """Every value this grid's histogram cells will bin, and its validity.

        The cells partition the samples, so without a reduction the union is
        simply every sample the window admits -- and asking that costs
        nothing, where building the partition would cost a walk of the whole
        dataset that the dense paths exist to avoid.
        """

        cell = spec.cell
        if not isinstance(cell, HistogramPlot):
            raise TypeError("facet histogram pools require a Histogram cell")
        if not cell.reduced:
            validity = (
                self.history_validity(window)
                if self.has_primary_index or _history_window(window) > 1
                else self._samples.valid_mask
            )
            return self._samples.value.canonical, validity
        plan = self._facet_histogram_plan(spec, window)
        return plan.pool, np.ones(plan.pool.shape, dtype=np.bool_)

    def _facet_histogram_plan(
        self, spec: FacetGridPlot, window: int
    ) -> "_FacetHistogramPlan":
        window = _history_window(window)
        key = (spec, window)
        remembered = self._facet_histogram_cache
        if remembered is not None and remembered[0] == key:
            return remembered[1]
        plan = self._build_facet_histogram_plan(spec, window)
        self._facet_histogram_cache = (key, plan)
        return plan

    def _build_facet_histogram_plan(
        self, spec: FacetGridPlot, window: int
    ) -> "_FacetHistogramPlan":
        cell = spec.cell
        if not isinstance(cell, HistogramPlot):
            raise TypeError("facet histogram pools require a Histogram cell")
        if not cell.reduced:
            raise ValueError("a facet histogram plan is only needed for a reduction")
        shape = self._samples.value.canonical.shape
        validity = (
            self.history_validity(window)
            if self.has_primary_index or window > 1
            else np.broadcast_to(self._samples.valid_mask, shape)
        )
        dimensions, coordinates = self._reduction_plan(tuple(cell.reduced))

        # ASK THE FACET AXIS, NOT EVERY SAMPLE.  The facet axis survives the
        # reduction -- validate_facet refuses a grid that reduces the axis it
        # facets by -- so every reduced value lies in exactly one cell, and
        # which one is fixed by its index along that axis.  Reading it per
        # sample instead grouped two million positions to learn four answers:
        # 0.40 s of a 0.58 s build, nearly all of it one argsort and one
        # unique.  One representative element per index along the axis puts
        # the same domain machinery on an array the size of the AXIS.
        facet_axis = int(self._resolve(spec.facet).dimension)
        strides_flat = [1] * len(shape)
        for axis in range(len(shape) - 2, -1, -1):
            strides_flat[axis] = strides_flat[axis + 1] * int(shape[axis + 1])
        representatives = (
            np.arange(int(shape[facet_axis]), dtype=np.int64)
            * strides_flat[facet_axis]
        )
        domain = self._domain(spec.facet, representatives)
        axis_codes = np.asarray(domain.codes, dtype=np.int64)
        cell_count = len(domain.values)

        if not coordinates:
            # WHOLE AXES ARE A UFUNC.  Naming only tensor axes -- reduce over
            # repeat, the ordinary case -- is exactly what _collapse_axes
            # already does with np.sum/np.min over an axis, and the answer
            # keeps the array's own shape minus those axes.  So the facet
            # index is an INDEX, read straight off the surviving axis, and
            # none of the per-sample machinery below is needed.  Measured on
            # 2M samples: the sum itself is 0.7 ms where scattering the same
            # reduction into buckets costs 20.7 ms plus 7.4 ms to build the
            # codes.
            reduced, present = self._collapse_axes(
                self._samples.value.canonical,
                validity,
                tuple(cell.reduced),
                cell.reduction,
            )
            kept = [axis for axis in range(len(shape)) if axis not in dimensions]
            spread = [1] * len(kept)
            spread[kept.index(facet_axis)] = -1
            cells = axis_codes.reshape(spread)
            pools = tuple(
                np.asarray(reduced[present & (cells == index)], dtype=float)
                for index in range(cell_count)
            )
        else:
            # A point coordinate regroups the point ROWS, so the surviving
            # point axis is no longer the array's: the identity has to be
            # built per sample.
            flat_values = np.asarray(self._samples.value.canonical).reshape(-1)
            usable = np.asarray(validity, dtype=bool).reshape(-1)
            buckets = self._reduction_buckets(dimensions, coordinates)
            codes = np.ascontiguousarray(buckets.codes).reshape(-1)
            reduced, counts = _aggregate_by_codes(
                flat_values, usable, codes, buckets.count, cell.reduction
            )
            present = np.asarray(counts) > 0
            if facet_axis == 1 and buckets.point_groups is not None:
                # The kept point axis stands for GROUPS of rows, and every row
                # in a group shares the facet coordinate -- it is not the one
                # being reduced -- so any row of the group names its cell.
                grouped = np.full(
                    int(buckets.extents[buckets.axes.index(1)]), -1, dtype=np.int64
                )
                grouped[buckets.point_groups] = axis_codes
                axis_codes = grouped
            bucket_facet = axis_codes[buckets.axis_index(facet_axis)]
            pools = tuple(
                np.asarray(reduced[present & (bucket_facet == index)], dtype=float)
                for index in range(cell_count)
            )

        joined = np.concatenate(pools) if pools else np.empty(0, dtype=float)
        return _FacetHistogramPlan(pools, tuple(domain.values), joined)

    def _reduced_histogram_facet(
        self,
        spec: FacetGridPlot,
        shared_bins: int | Sequence[float],
        window: int,
    ) -> FacetData:
        """Every cell of a reducing histogram grid, from one pass of the data.

        One walk builds the buckets, one aggregation fills them, and the
        cells are slices of that result -- rather than a reduction per cell,
        which would re-derive the same identities once per facet value.
        """

        plan = self._facet_histogram_plan(spec, window)
        cells = tuple(
            FacetCell(
                facet_index=index,
                facet_value_canonical=facet_value.canonical,
                facet_value_display=facet_value.display,
                label=facet_value.label,
                payload=self._histogram_from_values(shared_bins, plan.pools[index]),
            )
            for index, facet_value in enumerate(plan.facet_values)
        )
        return FacetData(
            revision=self._samples.revision,
            generation=self._samples.generation,
            spec=spec,
            cells=cells,
        )

    def _histogram_from_positions(
        self,
        bins: int | Sequence[float],
        positions: NDArray[np.int64],
        *,
        validity: NDArray[np.bool_] | None = None,
    ) -> HistogramData:
        mask = self._samples.valid_mask if validity is None else validity
        if positions is self._positions_cache:
            # The whole revision: bin values + mask directly, no gather.
            return self._histogram_from_values(
                bins,
                self._samples.value.canonical,
                valid=mask,
            )
        flat_valid = np.asarray(np.broadcast_to(mask, self._samples.value.canonical.shape)).reshape(-1)
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
        canonical_bins = self._canonical_histogram_bins(bins)
        if isinstance(canonical_bins, int):
            source = np.asarray(values)
            if valid is None or bool(np.all(valid)):
                selected = source.reshape(-1)
            else:
                selected = source[np.asarray(valid, dtype=np.bool_)].reshape(-1)
            counts, edges = np.histogram(selected, bins=canonical_bins)
        else:
            counts = histogram_counts(values, canonical_bins, valid)
            edges = canonical_bins
        return self._histogram_from_counts(edges, counts)

    def _canonical_histogram_bins(
        self, bins: int | Sequence[float]
    ) -> int | NDArray[Any]:
        """Validate bins once and express explicit edges canonically."""

        if isinstance(bins, bool):
            raise TypeError("histogram bin count must be an integer")
        if isinstance(bins, (int, np.integer)):
            if int(bins) <= 0:
                raise ValueError("histogram bin count must be positive")
            return int(bins)
        edges = np.asarray(tuple(bins))
        _require_real_numeric(edges, None)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
            raise ValueError(
                "histogram edges must be a finite one-dimensional sequence"
            )
        edges = self._samples.value.display_unit.convert_value_to(
            edges, self._samples.value.canonical_unit
        )
        if np.any(np.diff(edges) <= 0):
            raise ValueError("histogram edges must be strictly increasing")
        return edges

    def _histogram_from_counts(
        self,
        edges: NDArray[Any],
        counts: NDArray[np.int64],
    ) -> HistogramData:
        """Speak already-counted canonical bins as one Histogram payload."""

        edges = np.asarray(edges)
        counts = np.asarray(counts, dtype=np.int64)
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
        elif isinstance(cell, HistogramPlot):
            for ref in cell.reduced:
                self._resolve(ref)
            if any(
                ref.physical_identity == spec.facet.physical_identity
                for ref in cell.reduced
            ):
                # An axis cannot both name the cells and be averaged away
                # inside them: the cells would have nothing to be told apart
                # by.  The fate table gives an axis ONE fate, so this cannot
                # arrive from the editor -- it can only be authored.
                raise DataViewError(
                    "a facet axis cannot also be reduced: the cells it names "
                    "would be collapsed into one"
                )
        else:
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
                position = resolved.contract.topology_position
                assert position is not None
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
        window: int = 1,
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
        if isinstance(cell, HistogramPlot) and cell.reduced:
            # A reducing cell needs a per-sample bucket identity, which the
            # slab paths below have no shape for; and one pass over the whole
            # grid answers for every cell at once.
            assert shared_bins is not None
            return self._reduced_histogram_facet(spec, shared_bins, window)
        # WHICH SAMPLES COUNT.  Window is part of the Histogram cell's own
        # vocabulary; Image and Curve cells have no such control and consume
        # every valid facet already retained by Runtime.  Treating their
        # internal default ``1`` as a window left all history facet titles in
        # place while masking every cell except the latest one.
        validity = (
            self.history_validity(window)
            if isinstance(cell, HistogramPlot)
            and (self.has_primary_index or int(window) > 1)
            else None
        )
        dense = self._dense_facet(spec, shared_bins, uncertainty, validity=validity)
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
            validity=validity,
        )

    def _facet_from_positions(
        self,
        spec: FacetGridPlot,
        shared_bins: int | Sequence[float] | None,
        base_positions: NDArray[np.int64],
        uncertainty: bool = False,
        *,
        validity: NDArray[np.bool_] | None = None,
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
                    validity=validity,
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
        *,
        validity: NDArray[np.bool_] | None = None,
    ) -> FacetData | None:
        """Row-sliced twin of the dense projections, one cell at a time.

        A facet over a tensor dimension selects slices of the same dense
        block, and slicing preserves the regularity the cell projection
        relies on. Repeat/point facets shorten those axes; a Histogram over
        a DATA facet bins a view of each data-axis slice. No case needs
        (position, value) pairs, code sorting, or per-sample grouping.
        Curve/Image DATA facets keep their one-pass factored paths.
        """

        facet = spec.facet
        cell = spec.cell
        values = self._samples.value.canonical
        valid_mask = self._samples.valid_mask if validity is None else validity
        if facet.domain is AxisDomain.REPEAT:
            slice_axis = 0
        elif facet.domain in (
            AxisDomain.POINT_ROW,
            AxisDomain.POINT_COORDINATE,
            AxisDomain.POINT_DIMENSION,
        ):
            slice_axis = 1
        elif (
            facet.domain is AxisDomain.DATA
            and isinstance(cell, HistogramPlot)
        ):
            try:
                slice_axis = int(self._resolve(facet).dimension)
            except AxisResolutionError:
                return None
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
        stride = 1
        for size in values.shape[slice_axis + 1:]:
            stride *= int(size)
        representatives = (
            np.arange(values.shape[slice_axis], dtype=np.int64)
            * stride
        )
        domain = self._domain(facet, representatives)
        if not domain.values:
            return None
        if isinstance(cell, CurvePlot):
            assert x_resolved is not None
            batch_curve = self._dense_curve_facet_payloads(
                cell,
                x_resolved,
                values,
                valid_mask,
                slice_axis,
                np.asarray(domain.codes, dtype=np.int64),
                uncertainty,
            )
            if batch_curve is not None:
                return FacetData(
                    revision=self._samples.revision,
                    generation=self._samples.generation,
                    spec=spec,
                    cells=tuple(
                        FacetCell(
                            facet_index=index,
                            facet_value_canonical=value.canonical,
                            facet_value_display=value.display,
                            label=value.label,
                            payload=batch_curve[index],
                        )
                        for index, value in enumerate(domain.values)
                    ),
                )
        batch_edges = None
        batch_counts = None
        if isinstance(cell, HistogramPlot):
            assert shared_bins is not None
            canonical_bins = self._canonical_histogram_bins(shared_bins)
            if not isinstance(canonical_bins, int):
                batch_edges = canonical_bins
                batch_counts = _facet_kernel_counts(
                    values,
                    valid_mask,
                    np.asarray(domain.codes, dtype=np.int64),
                    slice_axis,
                    len(domain.values),
                    batch_edges,
                )

        cells: list[FacetCell] = []
        for facet_index, facet_value in enumerate(domain.values):
            selector = np.flatnonzero(domain.codes == facet_index)
            # A SLICE IS A VIEW AT ANY STEP, not only at step one.  A facet
            # over a point DIMENSION takes every tenth row, and asking for
            # step one only meant those cells fell to fancy indexing and
            # copied their whole share of the tensor -- twice, once for the
            # values and once for the validity.  Measured on a ten-cell
            # image facet that was 4.67 ms of a 25.9 ms revision; the
            # equivalent strided view is 0.0001 ms and bit-identical.
            steps = np.diff(selector)
            regular = bool(selector.size) and (
                selector.size == 1 or bool(np.all(steps == steps[0]))
            )
            selected: slice | NDArray[np.int64] = (
                slice(
                    int(selector[0]),
                    int(selector[-1]) + 1,
                    1 if selector.size == 1 else int(steps[0]),
                )
                if regular
                else selector
            )
            def sliced(plane: Any) -> Any:
                if plane is None:
                    return None
                slices = [slice(None)] * np.ndim(plane)
                slices[slice_axis] = selected
                return plane[tuple(slices)]

            cell_values = sliced(values)
            cell_valid = sliced(valid_mask)
            if isinstance(cell, CurvePlot):
                payload: FacetPayload = self._dense_curve_data(
                    cell.x,
                    x_resolved,
                    cell_values,
                    cell_valid,
                    cell.reduction,
                    uncertainty,
                    sliced(self._samples.sigma),
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
                payload = (
                    self._histogram_from_counts(
                        batch_edges, batch_counts[facet_index]
                    )
                    if batch_counts is not None and batch_edges is not None
                    else self._histogram_from_values(
                        shared_bins,
                        cell_values,
                        valid=cell_valid,
                    )
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

    def _dense_curve_facet_payloads(
        self,
        cell: CurvePlot,
        x_resolved: "_ProjectedAxis",
        values: NDArray[Any],
        usable: NDArray[np.bool_],
        facet_dimension: int,
        facet_codes: NDArray[np.int64],
        uncertainty: bool,
    ) -> tuple[CurveData, ...] | None:
        """Reduce every one-to-one tensor Facet Curve in one numeric pass."""

        facet_size = int(values.shape[facet_dimension])
        x_dimension = int(x_resolved.dimension)
        x_canonical = np.asarray(x_resolved.domain_canonical)
        nx = int(x_canonical.size)
        if (
            x_dimension == facet_dimension
            or facet_codes.shape != (facet_size,)
            or bool(np.any(facet_codes < 0))
            or np.unique(facet_codes).size != facet_size
            or int(np.max(facet_codes)) + 1 != facet_size
        ):
            return None

        moved = np.moveaxis(
            values,
            (facet_dimension, x_dimension),
            (-2, -1),
        ).reshape(-1, facet_size, nx)
        moved_usable = np.moveaxis(
            usable,
            (facet_dimension, x_dimension),
            (-2, -1),
        ).reshape(moved.shape)
        if not moved.flags.c_contiguous:
            moved = np.ascontiguousarray(moved)
        if not moved_usable.flags.c_contiguous:
            moved_usable = np.ascontiguousarray(moved_usable)
        y, counts = _masked_leading_reduce(
            moved, moved_usable, cell.reduction
        )
        y = np.asarray(y, dtype=np.float64)
        counts = np.broadcast_to(np.asarray(counts, dtype=np.int64), y.shape)
        sem = None
        if uncertainty:
            moved_sigma = None
            if self._samples.sigma is not None:
                moved_sigma = np.moveaxis(
                    self._samples.sigma,
                    (facet_dimension, x_dimension),
                    (-2, -1),
                ).reshape(moved.shape)
                if not moved_sigma.flags.c_contiguous:
                    moved_sigma = np.ascontiguousarray(moved_sigma)

            def mean_of_squares(plane: Any, offset: float) -> Any:
                array = np.asarray(plane)
                marks = (
                    None
                    if _stride_zero_all_true(moved_usable)
                    else moved_usable
                )
                from . import _raster_kernels as kernels

                sums = kernels.masked_centred_square_sums(
                    array.reshape(array.shape[0], -1, 1),
                    offset,
                    None if marks is None else marks.reshape(array.shape[0], -1, 1),
                )
                if sums is None:
                    reduced, _ = _masked_leading_reduce(
                        np.square(np.asarray(array, dtype=np.float64) - offset),
                        moved_usable,
                        Reduction.MEAN,
                    )
                    return reduced
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.where(
                        counts > 0,
                        sums.reshape(facet_size, nx) / counts,
                        np.nan,
                    )

            sem = _sem_of_mean(
                y, counts, moved, moved_sigma, mean_of_squares
            )

        order = _inverse_code_order(facet_codes)
        if order is not None:
            y = np.take(y, order, axis=0)
            counts = np.take(counts, order, axis=0)
            if sem is not None:
                sem = np.take(sem, order, axis=0)
        valid = (counts > 0) & np.isfinite(y)
        value = self._samples.value
        y_display = value.canonical_unit.convert_value_to(
            y, value.display_unit
        )
        x_quantity = QuantityArray(
            x_canonical,
            np.asarray(x_resolved.domain_display),
            x_resolved.coordinate.canonical_unit,
            x_resolved.coordinate.display_unit,
            x_resolved.coordinate.label,
        )
        x_labels = _axis_coordinate_labels(x_resolved, x_canonical)
        for array in (y, counts, valid, y_display):
            if array.flags.writeable:
                array.setflags(write=False)
        if sem is not None and sem.flags.writeable:
            sem.setflags(write=False)
        return tuple(
            CurveData(
                revision=self._samples.revision,
                generation=self._samples.generation,
                x_ref=cell.x,
                group_by=(),
                series=(
                    CurveSeries(
                        x=x_quantity,
                        x_labels=x_labels,
                        y=QuantityArray(
                            y[index],
                            y_display[index],
                            value.canonical_unit,
                            value.display_unit,
                            value.label,
                        ),
                        valid=valid[index],
                        counts=counts[index],
                        sem=None if sem is None else sem[index],
                        label=value.label,
                    ),
                ),
            )
            for index in range(facet_size)
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

    def _resolve(self, ref: AxisRef) -> _ProjectedAxis:
        if not isinstance(ref, AxisRef):
            raise TypeError("axis reference must be AxisRef")
        cached = self._axis_cache.get(ref)
        if cached is not None:
            return cached
        schema = self._schema
        try:
            contract = resolve_axis(schema, ref)
        except KeyError as exc:
            if (
                ref.domain is AxisDomain.POINT_DIMENSION
                and schema.grid_topology is None
            ):
                raise TopologyRequiredError(
                    f"point dimension {ref.axis_id!r} requires "
                    "producer-declared GridTopology"
                ) from exc
            raise AxisResolutionError(
                f"dataset has no exact {ref.domain.value} axis {ref.axis_id!r}"
            ) from exc
        domain_canonical = np.asarray(contract.coordinates)
        source_indices = contract.source_indices(schema)
        source_coordinates = domain_canonical[source_indices]
        canonical_unit = contract.canonical_unit(self._unit_registry)
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
        canonical_full = _broadcast_1d(
            source_coordinates, contract.dimension, shape
        )
        display_full = _broadcast_1d(display_source, contract.dimension, shape)
        index_full = _broadcast_1d(source_indices, contract.dimension, shape)
        coordinate = CoordinateArray(
            ref=ref,
            canonical=canonical_full,
            display=display_full,
            indices=index_full,
            canonical_unit=canonical_unit,
            display_unit=display_unit,
            label=contract.label,
        )
        resolved = _ProjectedAxis(
            contract=contract,
            coordinate=coordinate,
            domain_canonical=_readonly(domain_canonical),
            domain_display=_readonly(display_domain),
            coordinate_labels=contract.coordinate_labels,
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


def _uniform_counts(
    values: NDArray[Any],
    valid: NDArray[np.bool_] | None,
    edges: NDArray[Any],
) -> NDArray[np.int64] | None:
    """Count a UNIFORMLY binned histogram without sorting every sample.

    ``np.histogram`` given an edge ARRAY must assume the bins are irregular,
    so it sorts the whole pool: twelve milliseconds a revision on two
    million values, more than half of what a live histogram panel costs.
    Our edges are a linspace, and numpy already has the linear path for
    exactly that shape -- ``bins=count, range=(first, last)`` bincounts the
    scaled indices and builds the very same edges.  Handing it the count
    instead of the edges is the same question asked in the form numpy can
    answer cheaply; the edges it returns are compared before its answer is
    accepted, so nothing here decides what a bin IS.
    """

    if edges.ndim != 1 or edges.size < 2:
        return None
    first = float(edges[0])
    last = float(edges[-1])
    if not (math.isfinite(first) and math.isfinite(last)) or last <= first:
        return None
    count = int(edges.size) - 1
    if not np.array_equal(edges, np.linspace(first, last, edges.size)):
        return None
    source = np.asarray(values)
    if valid is None or bool(np.all(valid)):
        selected = source.reshape(-1)
    else:
        selected = source[np.asarray(valid, dtype=np.bool_)].reshape(-1)
    kernelled = _kernel_counts(selected, edges, first, last, count)
    if kernelled is not None:
        return kernelled
    counts, produced = np.histogram(selected, bins=count, range=(first, last))
    if not np.array_equal(produced, edges):
        return None
    return counts


def _kernel_counts(
    selected: NDArray[Any],
    edges: NDArray[Any],
    first: float,
    last: float,
    count: int,
) -> NDArray[np.int64] | None:
    """The same equal-bin count, in one pass, or ``None`` to defer.

    numpy walks the pool in blocks and pays five vector passes per block --
    two range comparisons, a cast, an index plane, two corrections and a
    bincount.  The kernel does each sample once.  It answers only where its
    arithmetic is numpy's own: the edge dtype numpy would have picked must
    be float64, and the edges it would have built must be the edges we were
    handed, which is the same guard the reference applies afterwards.
    """

    from . import _raster_kernels as kernels

    if not kernels.engaged():
        return None
    bin_type = np.result_type(first, last, selected)
    if np.issubdtype(bin_type, np.integer):
        bin_type = np.result_type(bin_type, float)
    if bin_type != np.float64:
        return None
    produced = np.linspace(first, last, count + 1, dtype=bin_type)
    if not np.array_equal(produced, edges):
        return None
    if bool(np.any(produced[:-1] >= produced[1:])):
        return None
    flat = kernels.readable(selected)
    if flat.ndim != 1:
        return None
    threads = kernels.histogram_threads()
    partials = np.empty((threads, count), dtype=np.int64)
    counted = np.empty(count, dtype=np.int64)
    kernels.uniform_histogram(
        flat, kernels.readable(produced), count, partials, counted
    )
    return counted


def _facet_kernel_counts(
    values: NDArray[Any],
    valid: NDArray[np.bool_],
    facet_codes: NDArray[np.int64],
    facet_dimension: int,
    facet_count: int,
    edges: NDArray[Any],
) -> NDArray[np.int64] | None:
    """Count all regular tensor Facet cells in one compiled pass."""

    from . import _raster_kernels as kernels

    if not kernels.engaged() or edges.ndim != 1 or edges.size < 2:
        return None
    first = float(edges[0])
    last = float(edges[-1])
    count = int(edges.size) - 1
    bin_type = np.result_type(first, last, values)
    if np.issubdtype(bin_type, np.integer):
        bin_type = np.result_type(bin_type, float)
    if bin_type != np.float64:
        return None
    produced = np.linspace(first, last, count + 1, dtype=bin_type)
    if not np.array_equal(produced, edges) or bool(
        np.any(produced[:-1] >= produced[1:])
    ):
        return None
    source = np.asarray(values)
    flat = kernels.readable(source).reshape(-1)
    use_valid = not (
        _stride_zero_all_true(valid) or bool(np.asarray(valid).all())
    )
    marks = (
        kernels.readable(
            np.asarray(np.broadcast_to(valid, source.shape), dtype=np.bool_)
        ).reshape(-1)
        if use_valid
        else np.zeros(1, dtype=np.bool_)
    )
    stride = 1
    for size in source.shape[facet_dimension + 1:]:
        stride *= int(size)
    threads = kernels.histogram_threads()
    partials = np.empty((threads, facet_count, count), dtype=np.int64)
    counted = np.empty((facet_count, count), dtype=np.int64)
    kernels.uniform_facet_histograms(
        flat,
        marks,
        use_valid,
        kernels.readable(np.asarray(facet_codes, dtype=np.int64)),
        stride,
        kernels.readable(produced),
        count,
        partials,
        counted,
    )
    return counted


def _axis_kernel_aggregate(
    values: NDArray[Any],
    valid: NDArray[np.bool_],
    codes: tuple[NDArray[np.int64], ...],
    dimensions: tuple[int, ...],
    domain_sizes: tuple[int, ...],
    aggregation: Reduction,
    *,
    offsets: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.bool_]] | None:
    """Compiled exact-order aggregation from axis-sized code vectors."""

    from . import _raster_kernels as kernels

    operations = {
        Reduction.MEAN: 0,
        Reduction.SUM: 1,
        Reduction.MIN: 2,
        Reduction.MAX: 3,
        Reduction.FIRST: 4,
    }
    operation = 5 if offsets is not None else operations.get(aggregation)
    source = np.asarray(values)
    if operation is None or not kernels.engaged() or source.dtype.kind == "c":
        return None
    maximum = max((code.size for code in codes), default=0)
    table = np.full((len(codes), maximum), -1, dtype=np.int64)
    for index, code in enumerate(codes):
        table[index, : code.size] = code
    shape = source.shape
    strides = []
    for dimension in dimensions:
        stride = 1
        for size in shape[dimension + 1:]:
            stride *= int(size)
        strides.append(stride)
    use_valid = not (
        _stride_zero_all_true(valid) or bool(np.asarray(valid).all())
    )
    marks = (
        kernels.readable(
            np.asarray(np.broadcast_to(valid, shape), dtype=np.bool_)
        ).reshape(-1)
        if use_valid
        else np.zeros(1, dtype=np.bool_)
    )
    bucket_count = math.prod(domain_sizes)
    out = np.empty(bucket_count, dtype=np.float64)
    counts = np.empty(bucket_count, dtype=np.int64)
    presence = np.empty(bucket_count, dtype=np.bool_)
    kernels.aggregate_axis_codes(
        kernels.readable(source).reshape(-1),
        kernels.readable(marks),
        use_valid,
        kernels.readable(table),
        kernels.readable(
            np.asarray([code.size for code in codes], dtype=np.int64)
        ),
        kernels.readable(np.asarray(domain_sizes, dtype=np.int64)),
        kernels.readable(np.asarray(strides, dtype=np.int64)),
        bucket_count,
        operation,
        (
            kernels.readable(np.zeros(1, dtype=np.float64))
            if offsets is None
            else kernels.readable(np.asarray(offsets, dtype=np.float64))
        ),
        out,
        counts,
        presence,
    )
    return out, counts, presence


def histogram_counts(
    values: NDArray[Any],
    edges: NDArray[Any],
    valid: NDArray[np.bool_] | None = None,
) -> NDArray[Any]:
    """Counts for one explicit edge array, the cheapest way that is exact.

    THE one place that turns values plus edges into counts.  Every caller
    used to hand ``np.histogram`` the edge array, which sorts the whole
    pool because it must assume irregular bins; every set of edges this
    library produces is uniform, integer-aligned, or both.
    """

    edge_array = np.asarray(edges)
    counts = _uniform_integer_counts(values, valid, edge_array)
    if counts is None:
        counts = _uniform_counts(values, valid, edge_array)
    if counts is not None:
        return counts
    source = np.asarray(values)
    if valid is None or bool(np.all(valid)):
        selected = source.reshape(-1)
    else:
        selected = source[np.asarray(valid, dtype=np.bool_)].reshape(-1)
    counted, _produced = np.histogram(selected, bins=edge_array)
    return counted


def _stride_zero_all_true(mask: NDArray[np.bool_]) -> bool:
    """True for a stride-0 broadcast plane that is constant True."""

    if mask.size == 0:
        return False
    if any(stride != 0 for stride in mask.strides):
        return False
    return bool(mask.flat[0])


def finite_probe(
    flat: NDArray[Any],
    valid: NDArray[np.bool_] | None = None,
    limit: int = 65536,
) -> NDArray[np.float64]:
    """The first ``limit`` finite values, without a full-pool mask plane.

    Same values, same order as ``_finite_probe`` over a materialised mask --
    the mask is simply built one block at a time, because a pool whose head
    is finite (every real pool) stops after the first.
    """

    collected: list[NDArray[Any]] = []
    total = 0
    for start in range(0, int(flat.size), limit):
        block = flat[start : start + limit]
        mask = np.isfinite(block)
        if valid is not None:
            mask &= valid[start : start + limit]
        chosen = block[mask]
        if chosen.size:
            take = chosen[: limit - total]
            collected.append(np.asarray(take, dtype=np.float64))
            total += int(take.size)
        if total >= limit:
            break
    if not collected:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(collected)


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


def _inverse_code_order(codes: NDArray[np.int64]) -> NDArray[np.int64] | None:
    """Tensor-index order to value-domain order, or no work when identical."""

    selected = np.asarray(codes, dtype=np.int64)
    natural = np.arange(selected.size, dtype=np.int64)
    if np.array_equal(selected, natural):
        return None
    order = np.empty(selected.size, dtype=np.int64)
    order[selected] = natural
    return order


def _finite_coordinate(values: NDArray[Any]) -> NDArray[np.bool_]:
    if values.dtype.kind in "biufc":
        return np.isfinite(values)
    return np.ones(values.shape, dtype=np.bool_)


def _require_real_numeric(values: NDArray[Any], ref: AxisRef | None) -> None:
    if np.asarray(values).dtype.kind not in "biuf":
        target = "dataset values" if ref is None else repr(ref)
        raise DataViewError(f"{target} must be real numeric for this projection")


def _leading_identity(
    values: NDArray[Any],
    usable: NDArray[np.bool_],
) -> tuple[NDArray[Any], NDArray[np.bool_]] | None:
    """The value and validity planes when no tensor dimension is pooled.

    ``DataView`` has already merged numeric finiteness into ``usable`` at its
    immutable snapshot boundary.  Rechecking the value plane here would scan
    every identity bucket a second time -- the whole two-million-point curve
    and the whole singleton camera image -- to rediscover the same mask.
    """

    if values.shape[0] != 1:
        return None
    value = np.asarray(values[0])
    valid = np.asarray(usable[0], dtype=np.bool_)
    return value, valid


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

    identity = _leading_identity(values, usable)
    if identity is not None:
        value, valid = identity
        return value, np.asarray(valid, dtype=np.int64)
    if _stride_zero_all_true(usable) or bool(np.all(usable)):
        # With no holes, ``where=usable`` and a separately summed count plane
        # are pure overhead.  Plain leading-axis reductions are NumPy's
        # contiguous fast path and produce the same dtype/order; counts are
        # the constant pool size and need no allocation.
        count = int(values.shape[0])
        counts = np.broadcast_to(
            np.asarray(count, dtype=np.int64), values.shape[1:]
        )
        if aggregation is Reduction.MEAN:
            result = np.mean(values, axis=0)
        elif aggregation is Reduction.SUM:
            result = np.sum(values, axis=0)
        elif aggregation is Reduction.MIN:
            result = np.min(np.asarray(values, dtype=np.float64), axis=0)
        elif aggregation is Reduction.MAX:
            result = np.max(np.asarray(values, dtype=np.float64), axis=0)
        elif aggregation is Reduction.FIRST:
            result = np.asarray(values[0])
        else:
            raise AssertionError(f"unsupported reduction: {aggregation!r}")
        return np.asarray(result), counts
    from . import _raster_kernels as kernels

    reduction_code = {
        Reduction.MEAN: kernels.REDUCE_MEAN,
        Reduction.SUM: kernels.REDUCE_SUM,
        Reduction.MIN: kernels.REDUCE_MIN,
        Reduction.MAX: kernels.REDUCE_MAX,
        Reduction.FIRST: kernels.REDUCE_FIRST,
    }[aggregation]
    fused = kernels.fused_masked_leading_float64(
        values, usable, reduction_code
    )
    if fused is not None:
        return fused
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
    resolved: "_ProjectedAxis",
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


#: How many samples are enough to find a value near the data.
_SEM_REFERENCE_SAMPLE = 4096


def _sem_reference(mean: NDArray[np.float64]) -> float:
    """A value near the samples, to square about instead of zero.

    The standard error does not depend on where the origin is, but the way
    it is computed does: ``E[x^2] - mean^2`` subtracts two numbers that are
    equal to as many digits as the offset exceeds the spread.  A fitted
    resonance centre at 6.834 GHz with a kilohertz scatter loses six of
    sixteen digits that way -- 2.2 per cent of the variance over eight
    samples, measured -- and the clip at zero then hides what is left.

    Every caller has already reduced the mean by the time it reduces the
    squares, so the reference costs one pass over the bucket means.
    """

    array = np.asarray(mean, dtype=np.float64).reshape(-1)
    if array.size > _SEM_REFERENCE_SAMPLE:
        # A reference only has to be NEAR the data; reading all of a camera
        # tensor to find one would cost more than the sem it protects.
        array = array[:: max(1, array.size // _SEM_REFERENCE_SAMPLE)]
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else 0.0


#: How many samples one centring pass shifts at a time.  Small enough that
#: the scratch buffer is a rounding error against a megapixel pool, large
#: enough that the per-block overhead disappears into the arithmetic.
_CENTRING_BLOCK = 1 << 17


def _centred_square_sum(
    flat: NDArray[Any],
    offset: float,
    where: NDArray[np.bool_] | None = None,
) -> float:
    """Sum of ``(x - offset)**2`` without materialising ``x - offset``.

    The whole-revision pool is the one place where that copy is a real
    cost: it is every sample of the revision, and the rolling path has a
    memory budget precisely so a per-shot reduction cannot double the
    footprint of the data it reduces.  Blocking through one small buffer
    keeps both the conditioning and the budget.
    """

    total = 0.0
    size = int(flat.size)
    if size == 0:
        return total
    scratch = np.empty(min(_CENTRING_BLOCK, size), dtype=np.float64)
    for start in range(0, size, _CENTRING_BLOCK):
        stop = min(start + _CENTRING_BLOCK, size)
        piece = scratch[: stop - start]
        np.subtract(flat[start:stop], offset, out=piece)
        if where is None:
            total += float(np.dot(piece, piece))
        else:
            np.square(piece, out=piece)
            total += float(
                np.sum(piece, where=where[start:stop], dtype=np.float64)
            )
    return total


def _centred_square_sums(
    plane: Any,
    offset: float,
    usable: Any | None,
    kept: list[int],
    shape: tuple[int, ...],
) -> Any:
    """Per kept position, the sum of squares about ``offset``, or None.

    THE KEPT AXES ARE ONE BLOCK OR THEY ARE NOTHING.  A reduction that
    keeps axis 1 and a group axis keeps a run of adjacent axes whenever a
    three-dimensional view of the tensor exists at all, and where it does
    the view costs nothing -- the array is C-contiguous, so the reshape is
    a reshape and not a copy.  Where it does not, the caller's einsum is
    still the answer.
    """

    if not kept or kept != list(range(kept[0], kept[-1] + 1)):
        return None
    array = np.asarray(plane)
    if array.shape != tuple(shape) or not array.flags.c_contiguous:
        return None
    outer = 1
    for axis in range(kept[0]):
        outer *= int(shape[axis])
    keep = 1
    for axis in kept:
        keep *= int(shape[axis])
    inner = 1
    for axis in range(kept[-1] + 1, len(shape)):
        inner *= int(shape[axis])
    marks = None
    if usable is not None:
        candidate = np.asarray(usable)
        if (
            candidate.dtype != np.bool_
            or candidate.shape != tuple(shape)
            or not candidate.flags.c_contiguous
        ):
            return None
        marks = candidate.reshape(outer, keep, inner)
    from . import _raster_kernels as kernels

    summed = kernels.masked_centred_square_sums(
        array.reshape(outer, keep, inner), float(offset), marks
    )
    if summed is None:
        return None
    return summed.reshape(tuple(int(shape[axis]) for axis in kept))


def _sem_of_mean(
    means: NDArray[np.float64],
    counts: NDArray[np.int64],
    samples: NDArray[Any],
    sigma: NDArray[Any] | None,
    mean_of_squares: Callable[[Any, float], Any],
) -> NDArray[np.float64]:
    """The standard error of a mean, formed the ONE way this repo forms it.

    Every plot kind that draws a band arrives here.  What differs between
    them is only how a bucket is summed -- a strided tensor reduction, a
    bincount over codes, an einsum, a dot -- so that is all a caller brings:
    ``mean_of_squares(plane, offset)`` returns, per bucket, the mean of
    ``(plane - offset)**2`` over exactly the samples that formed the mean.

    What does NOT differ, and therefore lives here:

      * the squares are taken about a reference near the data, because
        ``E[x^2] - mean^2`` about zero subtracts two nearly equal numbers
        and a resonance centre at 6.834 GHz loses six of sixteen digits;

      * the samples' own sigma is offered to the estimator, which uses it
        only where the scatter cannot speak.  A sigma is already a
        difference about zero, so it is squared about zero -- passing the
        value reference there would be a category error.

    Written out at each call site instead, this rule had already drifted:
    two of the eight sites still squared about zero, and none of the eight
    passed the sigma at all.
    """

    reference = _sem_reference(means)
    mean_square = np.asarray(mean_of_squares(samples, reference), dtype=np.float64)
    mean_sigma_square = (
        None
        if sigma is None
        else np.asarray(mean_of_squares(sigma, 0.0), dtype=np.float64)
    )
    return _sem_from_moments(
        means - reference, mean_square, counts, mean_sigma_square
    )


def _sem_from_moments(
    mean: NDArray[np.float64],
    mean_of_squares: NDArray[np.float64],
    counts: NDArray[np.int64],
    mean_sigma_square: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Standard error of the mean from (mean, mean-of-squares, n).

    sem^2 = s^2/n with the unbiased sample variance s^2, which collapses to
    (E[x^2] - mean^2) / (n - 1).  A single-sample bucket has no defined
    spread and reports NaN, never zero: zero would claim certainty.

    ``mean_sigma_square`` is <sigma_i^2>, the mean over the bucket of the
    samples' OWN uncertainties, for samples that carry one -- a fitted
    parameter does, a camera pixel does not.

    THE COMBINATION IS A SUM.  Writing x_i = mu + eps_i + delta_i, with
    eps_i the measurement error (variance sigma_i^2) and delta_i the
    genuine variation between samples (variance sigma_pop^2), both
    independent,

        Var(m) = (1/n^2) sum_i (sigma_i^2 + sigma_pop^2)
               = (<sigma_i^2> + sigma_pop^2) / n

    AND THE SCATTER ALREADY CONTAINS THE ERRORS.  E[s^2] = <sigma_i^2> +
    sigma_pop^2 -- for unequal sigma_i too, since E[s^2] is the mean of the
    per-sample variances -- so s^2/n is an UNBIASED estimate of the whole
    of Var(m), and adding <sigma_i^2> to it would count the measurement
    error twice.

    That is worth stating because the obvious-looking alternative is wrong.
    Estimating sigma_pop^2 by max(0, s^2 - <sigma_i^2>) and substituting
    collapses the sum to max(<sigma_i^2>, s^2)/n -- the larger of the
    propagated error and the observed scatter, which is a rule physics uses
    -- and it is BIASED HIGH, because clipping a noisy difference at zero
    keeps only its positive excursions.  Measured over 400k Monte Carlo
    buckets with sigma_pop = 0: at n = 8 it returns 1.51e5 where the truth
    is 1.25e5, and at n = 2 it returns 7.4e5 where the truth is 5.0e5 --
    22 per cent high in the error bar.  The plain scatter returns 1.250e5
    and 5.002e5.

    So the per-sample sigma is used exactly where the scatter cannot speak:
    a bucket of ONE, which has no spread and used to report NaN even though
    the sample knew its own error.  That is the common shape of one fit per
    shot.  Everywhere else the scatter is both unbiased and better informed,
    and a camera or a survival panel does not move a digit.

    The mean itself stays the ARITHMETIC mean.  Inverse-variance weighting
    is the better estimator when every sample measures the same value, but
    it changes the number that is plotted, and MEAN means mean.
    """

    n = counts.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.clip(mean_of_squares - np.square(mean), 0.0, None)
        # The unbiased sample variance; NaN where one sample cannot show a
        # spread, so that fmax below takes the sigma instead of a zero.
        variance = np.where(n > 1.0, spread * n / (n - 1.0), np.nan)
        if mean_sigma_square is not None:
            # Only where there is no scatter to measure.  fmax here would
            # double-count the measurement error and bias the bar high.
            variance = np.where(
                np.isnan(variance),
                np.asarray(mean_sigma_square, dtype=np.float64),
                variance,
            )
        sem = np.sqrt(variance / n)
    sem[n < 1.0] = np.nan
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


def _leading_along_axes(
    values: NDArray[Any],
    usable: NDArray[np.bool_],
    axes: tuple[int, ...],
) -> tuple[NDArray[Any], NDArray[np.bool_]]:
    """The first usable entry along ``axes``, in the array's own order."""

    moved_values = np.moveaxis(values, axes, range(-len(axes), 0))
    moved_usable = np.moveaxis(usable, axes, range(-len(axes), 0))
    head = moved_values.shape[: moved_values.ndim - len(axes)]
    flat_values = moved_values.reshape(head + (-1,))
    flat_usable = moved_usable.reshape(head + (-1,))
    present = flat_usable.any(axis=-1)
    first = np.argmax(flat_usable, axis=-1)
    taken = np.take_along_axis(flat_values, first[..., None], axis=-1)[..., 0]
    return np.where(present, taken, 0.0), present


def _point_row_codes(
    columns: Sequence[Any],
    row_count: int,
) -> tuple[NDArray[np.int64], int]:
    """One small-int group code per point row, from the columns given.

    With no columns left every row is one group: naming every coordinate
    is naming the point axis.
    """

    if not columns:
        return np.zeros(int(row_count), dtype=np.int64), 1
    combined = np.zeros(int(row_count), dtype=np.int64)
    span = 1
    for column in columns:
        # Codes per column rather than one np.unique over the stacked rows:
        # a column may be TEXT, and object arrays have no row-wise unique.
        _distinct, local = np.unique(
            np.asarray(column.values), return_inverse=True
        )
        combined = combined + np.asarray(local, dtype=np.int64) * span
        span *= int(_distinct.size)
    distinct, codes = np.unique(combined, return_inverse=True)
    return np.asarray(codes, dtype=np.int64), int(distinct.size)


def _history_window(window: object) -> int:
    if isinstance(window, bool) or not isinstance(window, Integral):
        raise TypeError("history window must be an integer")
    window = int(window)
    if window <= 0:
        raise ValueError("history window must be positive")
    return window


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
    "SelectionSubject",
    "QuantityArray",
    "SampleProjection",
    "TopologyRequiredError",
]
