"""Plot-neutral projections of immutable ``(R, P, *D)`` datasets.

Repeat and Point axes arrive as declared logical domains plus row codes.  The
projection layer consumes that one authority directly and never reconstructs
topology from values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
import math
from operator import is_
from numbers import Integral
from typing import Any, TypeAlias
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from zlc_data import (
    CoordinateScalar,
    BlockId,
    DatasetRevisionRef,
    DatasetSchema,
    LATEST_COORDINATE,
    OwnedSnapshot,
    canonical_coordinate_scalar,
)
from zlc_data.snapshot_projection import (
    PRIMARY_INDEX_AXIS_ID,
    SHOT_TIME_AXIS_ID,
    IndexedHistoryLayout,
    indexed_history_layout,
    restrict_snapshot,
    restricted_values,
    selection_indices,
    value_selection,
)

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

from .kinds import AxisRef, PlotKind
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


@dataclass(frozen=True, slots=True)
class SelectionSubject:
    """Exact upstream quantities cut by one accepted plot projection."""

    plot_kind: PlotKind
    x: AxisRef | None
    y: AxisRef | None
    x_coordinate_frame: str | None = None
    y_coordinate_frame: str | None = None
    scope: tuple[tuple[AxisRef, CoordinateScalar], ...] = ()
    source_window: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plot_kind, PlotKind):
            raise TypeError("selection subject plot_kind must be PlotKind")
        if self.source_window is not None and (type(self.source_window) is not int or self.source_window < 1):
            raise ValueError("selection source_window must be a positive integer or None")
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
            normalized_scope.append(
                (
                    ref,
                    canonical_coordinate_scalar(
                        coordinate,
                        "selection subject scope coordinate",
                    ),
                )
            )
        object.__setattr__(self, "scope", tuple(normalized_scope))


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
    _display: NDArray[Any] | ArrayLike | None
    canonical_unit: Unit
    display_unit: Unit
    label: str

    def __post_init__(self) -> None:
        canonical = _readonly(self.canonical)
        display = None if self._display is None else _readonly(self._display)
        if display is not None:
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
        object.__setattr__(self, "_display", display)

    @property
    def display(self) -> NDArray[Any]:
        if self._display is None:
            displayed = self.canonical_unit.convert_value_to(self.canonical, self.display_unit)
            if displayed.flags.writeable:
                displayed.setflags(write=False)
            object.__setattr__(self, "_display", displayed)
        return self._display


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
class RollingHistory:
    """Every shot of one rolling projection, as whole planes.

    One projection batch is one truth: its shots share a revision, a
    generation and one group-keys tuple, and their values live in
    ``(shots, groups)`` planes validated ONCE.  The per-shot object this
    replaced was constructed five thousand times per drawn frame -- most
    of that re-proving, one shot at a time, exactly what the batch
    establishes here once -- and the payload's first act was to stack the
    shots straight back into these planes.  Row access (``history[i]``)
    hands out a lightweight :class:`RollingShot` view for the callers
    that want one shot.
    """

    revision: int
    generation: str
    values: NDArray[Any] | ArrayLike
    valid: NDArray[np.bool_] | ArrayLike
    counts: NDArray[np.int64] | ArrayLike
    group_keys: tuple[tuple[AxisValue, ...], ...] = ()
    #: The authored primary index of each shot, oldest first, or None when
    #: the shots are arrival-ordered repeats.
    source_indices: NDArray[np.int64] | ArrayLike | None = None
    #: When each shot was taken, seconds from the run's first shot, oldest
    #: first, or None when the history is not stamped.
    source_times: NDArray[np.float64] | ArrayLike | None = None
    #: Standard error of each MEAN entry over what its shot pooled, or
    #: None when uncertainty was not requested.  Canonical-only, like the
    #: curve companion.
    sem: NDArray[np.float64] | ArrayLike | None = None

    def __post_init__(self) -> None:
        values = _readonly(self.values)
        valid = _readonly(self.valid, dtype=np.bool_)
        counts = _readonly(self.counts, dtype=np.int64)
        if (
            values.ndim != 2
            or valid.shape != values.shape
            or counts.shape != values.shape
        ):
            raise ValueError(
                "rolling history planes must share one (shots, groups) shape"
            )
        if np.any(counts < 0):
            raise ValueError("rolling history counts cannot be negative")
        source_keys = self.group_keys
        if type(source_keys) is tuple and all(
            type(item) is tuple for item in source_keys
        ):
            keys = source_keys
        else:
            keys = tuple(tuple(item) for item in source_keys)
        if len(keys) != values.shape[1]:
            raise ValueError(
                "rolling history group keys must match the group count"
            )
        if self.source_times is not None:
            source_times = _readonly(self.source_times, dtype=np.float64)
            if source_times.shape != (values.shape[0],):
                raise ValueError("rolling history source_times must be one per shot")
            object.__setattr__(self, "source_times", source_times)
        if self.source_indices is not None:
            source_indices = _readonly(self.source_indices, dtype=np.int64)
            if source_indices.shape != (values.shape[0],):
                raise ValueError(
                    "rolling history source indices must be one per shot"
                )
            object.__setattr__(self, "source_indices", source_indices)
        if self.sem is not None:
            sem = _readonly(self.sem, dtype=np.float64)
            if sem.shape != values.shape:
                raise ValueError(
                    "rolling history sem must match the value planes"
                )
            object.__setattr__(self, "sem", sem)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "group_keys", keys)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int) -> RollingShot:
        shots = len(self)
        if not -shots <= index < shots:
            raise IndexError(index)
        return RollingShot(self, index % shots)

    def __iter__(self) -> Iterator[RollingShot]:
        for index in range(len(self)):
            yield RollingShot(self, index)


@dataclass(frozen=True, slots=True)
class RollingShot:
    """One shot of a :class:`RollingHistory`, as row views of its planes."""

    history: RollingHistory
    index: int

    @property
    def revision(self) -> int:
        return self.history.revision

    @property
    def generation(self) -> str:
        return self.history.generation

    @property
    def values(self) -> NDArray[Any]:
        return self.history.values[self.index]

    @property
    def valid(self) -> NDArray[np.bool_]:
        return self.history.valid[self.index]

    @property
    def counts(self) -> NDArray[np.int64]:
        return self.history.counts[self.index]

    @property
    def group_keys(self) -> tuple[tuple[AxisValue, ...], ...]:
        return self.history.group_keys

    @property
    def source_index(self) -> int | None:
        indices = self.history.source_indices
        return None if indices is None else int(indices[self.index])

    @property
    def sem(self) -> NDArray[np.float64] | None:
        sem = self.history.sem
        return None if sem is None else sem[self.index]


@dataclass(frozen=True, slots=True)
class CurveSeries:
    x: QuantityArray
    y: QuantityArray
    valid: NDArray[np.bool_] | ArrayLike
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
        ):
            raise ValueError("curve arrays must have identical shapes")
        key = tuple(self.group_key)
        if any(not isinstance(value, AxisValue) for value in key):
            raise TypeError("group_key must contain AxisValue objects")
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "group_key", key)


@dataclass(frozen=True, slots=True)
class CurveData:
    revision: int
    generation: str
    x_ref: AxisRef | None
    group_by: tuple[AxisRef, ...]
    series: tuple[CurveSeries, ...]

    def __post_init__(self) -> None:
        group_by = tuple(self.group_by)
        series = tuple(self.series)
        if self.x_ref is not None and not isinstance(self.x_ref, AxisRef):
            raise TypeError("x_ref must be an AxisRef or None")
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
    group_keys: tuple[tuple[AxisValue, ...], ...] = ((),)

    def __post_init__(self) -> None:
        counts = _readonly(self.counts, dtype=np.int64)
        if self.edges.canonical.ndim != 1 or self.centers.canonical.ndim != 1:
            raise ValueError("histogram edges and centers must be one-dimensional")
        keys = tuple(tuple(key) for key in self.group_keys)
        if counts.ndim != 2 or counts.shape[0] != len(keys):
            raise ValueError("histogram counts must have shape (groups, bins)")
        if any(not isinstance(value, AxisValue) for key in keys for value in key):
            raise TypeError("histogram group keys must contain AxisValue objects")
        if self.edges.canonical.size != counts.shape[1] + 1:
            raise ValueError("histogram requires one more edge than count")
        if self.centers.canonical.size != counts.shape[1]:
            raise ValueError("histogram centers and counts must have equal length")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "group_keys", keys)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(
            ", ".join(value.label for value in key) if key else self.edges.label
            for key in self.group_keys
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
    retained_domain: _Domain | None = field(default=None, compare=False, repr=False)

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

    codes: tuple[NDArray[np.int64], ...]
    count: int
    shape: tuple[int, ...]
    axes: tuple[int, ...]
    strides: tuple[int, ...]
    extents: tuple[int, ...]
    carrier_groups: tuple[tuple[int, NDArray[np.int64]], ...]

    def axis_index(self, axis: int) -> NDArray[np.int64]:
        """Which index along ``axis`` each bucket number stands for."""

        position = self.axes.index(int(axis))
        numbers = np.arange(self.count, dtype=np.int64)
        return (numbers // self.strides[position]) % self.extents[position]

    def groups_for_axis(self, axis: int) -> NDArray[np.int64] | None:
        for dimension, codes in self.carrier_groups:
            if dimension == int(axis):
                return codes
        return None


@dataclass(frozen=True)
class _HistogramPlan:
    """Reduced samples and coordinate-owned distribution codes, prepared once."""

    values: NDArray[Any]
    valid: NDArray[np.bool_]
    group_codes: NDArray[np.int64]
    code_axis: int
    group_keys: tuple[tuple[AxisValue, ...], ...]


#: The most integer levels a window frequency table is kept for.  A narrow
#: dtype's whole range is always kept (65 536 levels for 16-bit pixels);
#: a wider one is kept over its observed span while that stays small.
_FREQUENCY_LEVEL_LIMIT = 1 << 18


@dataclass(slots=True)
class _WindowFrequency:
    """One view's window frequency table and what it was counted from.

    ``counts[v - offset]`` is how many valid samples of the last ``window``
    shots equal ``v``.  ``ids`` are those shots' absolute numbers in
    ``positions`` order; ``snapshot`` and ``valid`` are what the shots that
    later LEAVE will be subtracted from, which is why the previous revision
    is kept alive here one revision longer than it otherwise would be.
    """

    snapshot: OwnedSnapshot
    valid: NDArray[np.bool_]
    revision: int
    window: int
    inner_count: int
    dtype: np.dtype
    ids: NDArray[np.int64]
    positions: NDArray[np.int64]
    offset: int
    counts: NDArray[np.int64]


class DataView:
    """Resolve and aggregate one immutable dataset revision for renderers."""

    __slots__ = (
        "_snapshot",
        "_schema",
        "_axis_display_units",
        "_value_display_unit",
        "_unit_registry",
        "_samples",
        "_axis_cache",
        "_flat_cache",
        "_pooled_cache",
        "_positions_cache",
        "_histogram_cache",
        "_domain_carry",
        "_unit_registry_revision",
        "_history_layout",
        "_history_mask_cache",
        "_frequency_carry",
        "_rolling_carry",
        "_packed_segments",
        "_packed_carry",
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
        schema = snapshot_schema(snapshot)
        if schema.value_schema.dtype.kind == "c":
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
        self._snapshot = snapshot
        self._schema = schema
        self._axis_display_units = overrides
        self._value_display_unit = value_display
        self._samples: SampleProjection | None = None
        self._packed_segments: tuple | None = None
        self._packed_carry = (
            (inherit_domains_from._snapshot, inherit_domains_from._packed_segments)
            if snapshot.block.values is None and inherit_domains_from is not None
            and inherit_domains_from._packed_segments is not None
            else None
        )
        self._unit_registry = registry
        self._unit_registry_revision = registry.revision
        self._axis_cache: dict[AxisRef, _ProjectedAxis] = {}
        self._flat_cache: dict[AxisRef, NDArray[np.int64]] = {}
        self._pooled_cache: NDArray[Any] | None = None
        self._positions_cache: NDArray[np.int64] | None = None
        self._histogram_cache: tuple[object, "_HistogramPlan"] | None = None
        #: The Runtime history's shot structure, read off the schema once
        #: (it is cached there) and the per-window sample mask derived from
        #: it, built once per view however many projections ask.
        self._history_layout: IndexedHistoryLayout | None = indexed_history_layout(
            schema
        )
        self._history_mask_cache: dict[int, NDArray[np.bool_]] = {}
        #: The previous view's window frequency table, to be moved by the
        #: shots that entered and left rather than rebuilt: ``window_frequency``.
        self._frequency_carry: _WindowFrequency | None = None
        self._rolling_carry: tuple | None = None
        if isinstance(inherit_domains_from, DataView):
            self._frequency_carry = inherit_domains_from._frequency_carry
            if (
                self._history_layout is not None and snapshot.block.values is None
                and inherit_domains_from._axis_display_units == overrides
                and inherit_domains_from._unit_registry is registry
                and inherit_domains_from._unit_registry_revision == registry.revision
            ):
                self._rolling_carry = inherit_domains_from._rolling_carry
        #: Whole-dataset domains carried from the PREVIOUS revision's view.
        #: A schema fingerprint includes axis domains and codes, so an exact
        #: fingerprint/unit match proves this small derived domain remains
        #: valid without comparing a full coordinate plane.
        self._domain_carry: dict[AxisRef, _Domain] = {}
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
            # revision data.
            # Carry the small cache under the same exact context gate as the
            # domains; copy the dict so either view may still resolve another
            # axis without mutating its sibling.
            self._axis_cache = dict(inherit_domains_from._axis_cache)
            self._domain_carry = inherit_domains_from._domain_carry
        # Fail early for misspelled or undeclared override keys.
        for ref in overrides:
            self._resolve(ref)

    @property
    def samples(self) -> SampleProjection:
        if self._samples is None:
            snapshot = self._snapshot
            if snapshot.block.values is None:
                values, valid, sigma, rows = self._segment_arrays(sigma=True)
                if rows is not None:
                    # The public samples API promises the full schema shape.
                    # Scatter physical cells once, not one Python copy per block.
                    shape = schema_shape(self._schema)
                    whole_values = np.zeros(shape, dtype=values.dtype)
                    whole_valid = np.zeros(shape, dtype=np.bool_)
                    whole_values[rows], whole_valid[rows] = values, valid
                    whole_sigma = None
                    if sigma is not None:
                        whole_sigma = np.full(shape, np.nan)
                        whole_sigma[rows] = sigma
                        whole_sigma.setflags(write=False)
                    values, valid, sigma = whole_values, whole_valid, whole_sigma
                    values.setflags(write=False)
                    valid.setflags(write=False)
            else:
                values = snapshot_values(snapshot)
                valid = snapshot_validity(snapshot)
                sigma = snapshot_sigma(snapshot)
                if values.dtype.kind not in "biu":
                    finite = np.isfinite(values)
                    if not bool(finite.all()):
                        valid = finite if _stride_zero_all_true(valid) else valid & finite
                        valid.setflags(write=False)
            self._samples = SampleProjection(
                revision=snapshot_revision(snapshot),
                generation=snapshot_generation(snapshot),
                shape=schema_shape(self._schema),
                value=QuantityArray(
                    canonical=values, _display=None,
                    canonical_unit=schema_value_unit(self._schema, self._unit_registry),
                    display_unit=self._value_display_unit,
                    label=self._schema.value_schema.name or "value",
                ),
                valid_mask=valid,
                sigma=sigma,
            )
        return self._samples

    @property
    def has_primary_index(self) -> bool:
        return self._history_layout is not None

    def coordinate(self, ref: AxisRef) -> CoordinateArray:
        return self._resolve(ref).coordinate

    def _last_view(
        self, *, keep: Sequence[AxisRef] = (), reduced: Sequence[AxisRef] | None = None
    ) -> "DataView":
        """Reuse the Dataset Scope cutter, including sparse empty selections."""

        from .semantics import axis_choices_for_schema

        families = {
            self._resolve(ref).contract.domain.coordinate_axis(self._resolve(ref).contract.axis_id).axis_id
            for ref in keep
        }
        refs = tuple(reduced) if reduced is not None else tuple(
            ref for ref in axis_choices_for_schema(self._schema)
            if self._resolve(ref).contract.domain.axis(self._resolve(ref).contract.axis_id).coordinate_of is None
            and self._resolve(ref).contract.axis_id not in families
        )
        if all(self._resolve(ref).contract.size == 1 for ref in refs):
            return self
        terms = {self._resolve(ref).contract.axis_id: LATEST_COORDINATE for ref in refs}
        source = self._snapshot
        identity = ",".join(sorted(str(axis) for axis in terms))
        snapshot = restrict_snapshot(
            source, value_selection(self._schema, terms),
            reference_for=lambda schema: DatasetRevisionRef(
                BlockId(f"{source.ref.block_id.value}|last:{identity}"),
                source.ref.stream_generation, schema.fingerprint, source.ref.revision,
            ),
        )
        return DataView(
            snapshot, axis_display_units=self._axis_display_units,
            value_display_unit=self._value_display_unit,
            unit_registry=self._unit_registry,
        )

    def selection_subject(
        self,
        spec: PlotSpec,
        payload: CurveData | ImageData | HistogramData | FacetData,
        *,
        source_window: int | None = None,
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
            selected_payload.revision != snapshot_revision(self._snapshot)
            or selected_payload.generation != snapshot_generation(self._snapshot)
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
        elif isinstance(semantic, RollingPlot) and self.has_primary_index:
            x_ref = semantic.x or AxisRef.point(str(PRIMARY_INDEX_AXIS_ID))
            y_ref = None
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
        scope: list[tuple[AxisRef, CoordinateScalar]] = []
        from .semantics import projection_scope

        for ref, authored in projection_scope(self._schema, spec):
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
            scope.append((ref, coordinate))

        # A focused cell only supplies the coordinate transform of the drag.
        # The numeric region applies to every facet; only authored scope/Last
        # limits which rows belong to this selection.
        return SelectionSubject(
            semantic.kind,
            x_ref,
            y_ref,
            x_frame,
            y_frame,
            tuple(scope),
            source_window if self.has_primary_index else None,
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
            if aggregation.statistic is not Reduction.MEAN:
                raise ValueError(
                    "uncertainty is defined for Reduction.MEAN only, "
                    f"not {aggregation.value!r}"
                )
            if self._schema.value_schema.dtype.kind == "c":
                raise ValueError("uncertainty is undefined for complex values")
        if aggregation is Reduction.LAST:
            return self._last_view(keep=(x, *groups)).curve(
                x, group_by=groups, aggregation=Reduction.MEAN, uncertainty=uncertainty,
            )
        dense = self._dense_data_curve(x, groups, aggregation, uncertainty)
        if dense is not None:
            return dense
        factored = self._factored_curve(x, groups, aggregation, uncertainty)
        if factored is not None:
            return factored
        return self._curve_from_axes(
            x, groups, aggregation, uncertainty=uncertainty
        )

    def _dense_tensor_projection(
        self,
        refs: tuple[AxisRef, ...],
        aggregation: Reduction,
        *,
        uncertainty: bool = False,
    ) -> tuple[
        tuple["_Domain", ...],
        NDArray[Any],
        NDArray[np.int64],
        NDArray[np.float64] | None,
        NDArray[np.bool_],
    ] | None:
        """Reduce one regular tensor once while retaining ``refs`` in order.

        This is the dense numeric owner shared by standalone and Facet
        Curve/Image projections.  A retained axis must map one-to-one to a
        physical tensor dimension; scan coordinates that share the point-row
        dimension stay with ``_factored_planes``.  The caller only repackages
        the returned planes into its public plot payload.
        """

        if not refs:
            return None
        try:
            domains, codes, dimensions = self._axis_projection(refs)
        except AxisResolutionError:
            return None
        if self._snapshot.block.values is None:
            source, source_usable, source_sigma, rows = self._segment_arrays(sigma=uncertainty)
            if rows is not None:
                return None
        else:
            source = self.samples.value.canonical
            source_usable = self.samples.valid_mask
            source_sigma = self.samples.sigma if uncertainty else None
        shape = schema_shape(self._schema)
        if len(set(dimensions)) != len(dimensions):
            return None
        orders: list[NDArray[np.int64] | None] = []
        kept_sizes: list[int] = []
        for domain, axis_codes, dimension in zip(domains, codes, dimensions):
            size = int(shape[dimension])
            selected = np.asarray(axis_codes, dtype=np.int64)
            if (
                int(domain.size) != size
                or selected.shape != (size,)
                or bool(np.any(selected < 0))
                or bool(np.any(selected >= size))
                or not np.all(_finite_coordinate(np.asarray(domain.canonical)))
            ):
                return None
            kept_sizes.append(size)
            natural = np.arange(size, dtype=np.int64)
            if np.array_equal(selected, natural):
                orders.append(None)
            else:
                if not bool(np.all(np.bincount(selected, minlength=size) == 1)):
                    return None
                orders.append(_inverse_code_order(selected))

        destinations = tuple(
            range(len(shape) - len(dimensions), len(shape))
        )

        def laid_out(array: NDArray[Any]) -> NDArray[Any]:
            moved = np.moveaxis(array, dimensions, destinations).reshape(
                (-1, *kept_sizes), order="C"
            )
            return moved if moved.flags.c_contiguous else np.ascontiguousarray(moved)

        moved = laid_out(source)
        moved_usable = (
            np.broadcast_to(np.asarray(True, dtype=np.bool_), moved.shape)
            if _stride_zero_all_true(source_usable)
            else laid_out(source_usable)
        )
        identity = _leading_identity(moved, moved_usable)
        if identity is None:
            values, counts = _masked_leading_reduce(
                moved, moved_usable, aggregation
            )
            valid_plane = (counts > 0) & np.isfinite(values)
        else:
            values, valid = identity
            valid_plane = np.asarray(valid, dtype=np.bool_)
            counts = (
                np.broadcast_to(np.asarray(1, dtype=np.int64), values.shape)
                if _stride_zero_all_true(valid)
                else np.asarray(valid, dtype=np.int64)
            )
        values = np.asarray(values)
        counts = np.asarray(counts, dtype=np.int64)

        sem = None
        if uncertainty:
            if aggregation is not Reduction.MEAN:
                return None
            moved_sigma = (
                None
                if source_sigma is None
                else laid_out(source_sigma)
            )
            if identity is not None and moved_sigma is None:
                sem = np.broadcast_to(
                    np.asarray(np.nan, dtype=np.float64), values.shape
                )
            else:
                marks = (
                    None
                    if _stride_zero_all_true(moved_usable)
                    else moved_usable
                )

                def centred_moments(
                    plane: Any, offsets: NDArray[np.float64]
                ) -> tuple[Any, Any]:
                    array = np.asarray(plane)
                    from . import _raster_kernels as kernels

                    sums = kernels.masked_centred_moment_sums(
                        array.reshape(array.shape[0], -1, 1),
                        np.asarray(offsets, dtype=np.float64).reshape(-1),
                        (
                            None
                            if marks is None
                            else marks.reshape(array.shape[0], -1, 1)
                        ),
                    )
                    if sums is None:
                        delta = (
                            np.asarray(array, dtype=np.float64)
                            - np.asarray(offsets, dtype=np.float64)
                        )
                        first, _ = _masked_leading_reduce(
                            delta,
                            moved_usable,
                            Reduction.MEAN,
                        )
                        np.square(delta, out=delta)
                        second, _ = _masked_leading_reduce(
                            delta, moved_usable, Reduction.MEAN
                        )
                        return first, second
                    with np.errstate(invalid="ignore", divide="ignore"):
                        return tuple(
                            np.where(
                                counts > 0,
                                moment.reshape(values.shape) / counts,
                                np.nan,
                            )
                            for moment in sums
                        )

                sem = _sem_of_mean(
                    np.asarray(values, dtype=np.float64),
                    counts,
                    moved,
                    moved_sigma,
                    centred_moments,
                )

        for axis, order in enumerate(orders):
            if order is None:
                continue
            values = np.take(values, order, axis=axis)
            counts = np.take(counts, order, axis=axis)
            valid_plane = np.take(valid_plane, order, axis=axis)
            if sem is not None:
                sem = np.take(sem, order, axis=axis)
        return domains, values, counts, sem, valid_plane

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

        projected = self._dense_tensor_projection(
            (x, *groups), aggregation, uncertainty=uncertainty
        )
        if projected is None:
            return None
        domains, values, counts, sem, valid = projected
        x_domain, group_domains = domains[0], domains[1:]
        x_resolved = self._resolve(x)
        nx = int(x_domain.size)
        group_sizes = tuple(int(domain.size) for domain in group_domains)
        combinations = math.prod(group_sizes) if group_sizes else 1
        x_canonical = np.asarray(x_domain.canonical)
        return CurveData(
            revision=snapshot_revision(self._snapshot),
            generation=snapshot_generation(self._snapshot),
            x_ref=x,
            group_by=groups,
            series=self._series_from_columns(
                QuantityArray(
                    x_canonical,
                    np.asarray(x_domain.display),
                    x_resolved.coordinate.canonical_unit,
                    x_resolved.coordinate.display_unit,
                    x_resolved.coordinate.label,
                ),
                _axis_coordinate_labels(x_resolved, x_canonical),
                group_domains,
                group_sizes,
                np.asarray(values, dtype=np.float64).reshape(nx, combinations),
                np.asarray(counts, dtype=np.int64).reshape(nx, combinations),
                (
                    None
                    if sem is None
                    else np.asarray(sem, dtype=np.float64).reshape(nx, combinations)
                ),
                valid_plane=np.asarray(valid, dtype=np.bool_).reshape(
                    nx, combinations
                ),
            ),
        )

    def _curve_from_axes(
        self,
        x: AxisRef,
        groups: tuple[AxisRef, ...],
        aggregation: Reduction,
        *,
        uncertainty: bool = False,
    ) -> CurveData:
        """Exact full-Dataset Curve aggregation without position planes."""

        projection = self._axis_projection((*groups, x))
        domains, axis_codes, dimensions = projection
        if self._snapshot.block.values is None:
            values, counts, presence, sem = self._segmented_axes(
                axis_codes, dimensions, tuple(domain.size for domain in domains),
                aggregation, uncertainty=uncertainty,
            )
        else:
            domains, values, counts, presence = self._aggregate_axes(
                (*groups, x), aggregation, projection=projection,
            )
            sem = None
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
        if uncertainty and self._snapshot.block.values is not None:
            shape = (*group_sizes, nx)
            means = np.asarray(values, dtype=np.float64).reshape(shape)
            domain_sizes = tuple(int(domain.size) for domain in domains)

            def centred_moments(
                plane: Any, offsets: NDArray[np.float64]
            ) -> tuple[Any, Any] | None:
                reduced = _axis_aggregate(
                    np.asarray(plane),
                    self.samples.valid_mask,
                    axis_codes,
                    dimensions,
                    domain_sizes,
                    Reduction.SUM,
                    offsets=np.asarray(offsets, dtype=np.float64).reshape(-1),
                )
                first, second, _moment_counts, _presence = reduced
                return first.reshape(shape), second.reshape(shape)

            sem = _sem_of_mean(
                means,
                counts,
                self.samples.value.canonical,
                self.samples.sigma,
                centred_moments,
            )
            assert sem is not None
        return CurveData(
            revision=snapshot_revision(self._snapshot),
            generation=snapshot_generation(self._snapshot),
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

        Coverage: x determined by one mapped Repeat/Point carrier; groups over
        other one-to-one carrier/data axes.  Everything else -- FIRST (whose
        result depends on exact sample order), complex values, or irregular
        group mappings -- keeps the shared axis-code path.
        """

        x_dimension = int(self._resolve(x).dimension)
        carrier_groups = tuple(
            group
            for group in groups
            if int(self._resolve(group).dimension) == x_dimension
        )
        tensor_groups = tuple(group for group in groups if group not in carrier_groups)
        row_refs = (*carrier_groups, x)
        planes = self._factored_planes(
            row_refs, tensor_groups, aggregation, uncertainty
        )
        if planes is None:
            return None
        if carrier_groups:
            carrier_sizes = planes.row_sizes[:-1]
            nx = planes.row_sizes[-1]
            tensor_sizes = planes.group_sizes
            shape = (*carrier_sizes, nx, *tensor_sizes)

            def columns(array: NDArray[Any]) -> NDArray[Any]:
                moved = np.moveaxis(
                    np.asarray(array).reshape(shape), len(carrier_sizes), 0
                )
                internal = (*carrier_groups, *tensor_groups)
                permutation = [0] + [
                    1 + internal.index(group) for group in groups
                ]
                return np.transpose(moved, permutation).reshape(nx, -1)

            presence = np.moveaxis(
                planes.row_presence.reshape((*carrier_sizes, nx)),
                len(carrier_sizes),
                0,
            )
            presence = np.broadcast_to(
                presence.reshape((nx, *carrier_sizes, *([1] * len(tensor_sizes)))),
                (nx, *carrier_sizes, *tensor_sizes),
            )
            internal = (*carrier_groups, *tensor_groups)
            permutation = [0] + [1 + internal.index(group) for group in groups]
            used = np.transpose(presence, permutation).reshape(nx, -1)
            domain_by_group = {
                group: domain
                for group, domain in zip(carrier_groups, planes.row_domains[:-1])
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
                revision=snapshot_revision(self._snapshot),
                generation=snapshot_generation(self._snapshot),
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
            revision=snapshot_revision(self._snapshot),
            generation=snapshot_generation(self._snapshot),
            x_ref=x,
            group_by=groups,
            series=series,
        )

    def _dense_tensor_facet(
        self,
        spec: FacetGridPlot,
        uncertainty: bool,
    ) -> FacetData | None:
        """Package one retained-axis tensor reduction as every Facet cell."""

        cell = spec.cell
        if isinstance(cell, CurvePlot):
            groups = () if cell.group is None else (cell.group,)
            projected = self._dense_tensor_projection(
                (spec.facet, cell.x, *groups),
                cell.reduction,
                uncertainty=uncertainty,
            )
            if projected is None:
                return None
            domains, values, counts, sem, valid = projected
            facet_domain, x_domain, group_domains = (
                domains[0],
                domains[1],
                domains[2:],
            )
            facet_size = int(facet_domain.size)
            nx = int(x_domain.size)
            group_sizes = tuple(int(domain.size) for domain in group_domains)
            combinations = math.prod(group_sizes) if group_sizes else 1
            values = np.asarray(values, dtype=np.float64).reshape(
                facet_size, nx, combinations
            )
            counts = np.asarray(counts, dtype=np.int64).reshape(
                facet_size, nx, combinations
            )
            if sem is not None:
                sem = np.asarray(sem, dtype=np.float64).reshape(
                    facet_size, nx, combinations
                )
            valid = np.asarray(valid, dtype=np.bool_).reshape(
                facet_size, nx, combinations
            )
            x_resolved = self._resolve(cell.x)
            x_canonical = np.asarray(x_domain.canonical)
            x_quantity = QuantityArray(
                x_canonical,
                np.asarray(x_domain.display),
                x_resolved.coordinate.canonical_unit,
                x_resolved.coordinate.display_unit,
                x_resolved.coordinate.label,
            )
            x_labels = _axis_coordinate_labels(x_resolved, x_canonical)
            payloads = tuple(
                CurveData(
                    revision=snapshot_revision(self._snapshot),
                    generation=snapshot_generation(self._snapshot),
                    x_ref=cell.x,
                    group_by=groups,
                    series=self._series_from_columns(
                        x_quantity,
                        x_labels,
                        group_domains,
                        group_sizes,
                        values[index],
                        counts[index],
                        None if sem is None else sem[index],
                        valid_plane=valid[index],
                    ),
                )
                for index in range(facet_size)
            )
        elif isinstance(cell, ImagePlot):
            projected = self._dense_tensor_projection(
                (spec.facet, cell.y, cell.x), cell.reduction
            )
            if projected is None:
                return None
            domains, values, counts, _sem, valid = projected
            facet_domain, y_domain, x_domain = domains
            facet_size = int(facet_domain.size)
            ny, nx = int(y_domain.size), int(x_domain.size)
            values = np.asarray(values).reshape(facet_size, ny, nx)
            counts = np.asarray(counts, dtype=np.int64).reshape(
                facet_size, ny, nx
            )
            valid = np.asarray(valid, dtype=np.bool_).reshape(
                facet_size, ny, nx
            )
            payloads = tuple(
                self._image_from_planes(
                    cell.x,
                    cell.y,
                    x_domain,
                    y_domain,
                    values[index],
                    counts[index],
                    valid=valid[index],
                )
                for index in range(facet_size)
            )
        else:
            return None

        return FacetData(
            revision=snapshot_revision(self._snapshot),
            generation=snapshot_generation(self._snapshot),
            spec=spec,
            cells=tuple(
                FacetCell(
                    facet_index=index,
                    facet_value_canonical=value.canonical,
                    facet_value_display=value.display,
                    label=value.label,
                    payload=payloads[index],
                )
                for index, value in enumerate(facet_domain.values)
            ),
        )

    def _factored_facet(
        self,
        spec: FacetGridPlot,
        uncertainty: bool,
    ) -> FacetData:
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
        dense = self._dense_tensor_facet(spec, uncertainty)
        if dense is not None:
            return dense
        if isinstance(cell, ImagePlot):
            return self._factored_facet_images(spec, cell)
        cell_groups = () if cell.group is None else (cell.group,)
        row_facet = int(self._resolve(spec.facet).dimension) == int(
            self._resolve(cell.x).dimension
        )
        if row_facet:
            planes = self._factored_planes(
                (spec.facet, cell.x),
                cell_groups,
                cell.reduction,
                uncertainty,
            )
        else:
            planes = self._factored_planes(
                (cell.x,),
                (spec.facet, *cell_groups),
                cell.reduction,
                uncertainty,
            )
        if planes is None:
            combined = self._curve_from_axes(
                cell.x,
                (spec.facet, *cell_groups),
                cell.reduction,
                uncertainty=uncertainty,
            )
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
                revision=snapshot_revision(self._snapshot),
                generation=snapshot_generation(self._snapshot),
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
            revision=snapshot_revision(self._snapshot),
            generation=snapshot_generation(self._snapshot),
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
            label = (self._schema.value_schema.name or "value") if not key else ", ".join(
                item.label for item in key
            )
            grouped[-1][1].append(CurveSeries(
                x=series.x,
                x_labels=series.x_labels,
                y=series.y,
                valid=series.valid,
                sem=series.sem,
                group_key=key,
                label=label,
            ))
        return FacetData(
            revision=snapshot_revision(self._snapshot),
            generation=snapshot_generation(self._snapshot),
            spec=spec,
            cells=tuple(
                FacetCell(
                    facet_index=index,
                    facet_value_canonical=value.canonical,
                    facet_value_display=value.display,
                    label=value.label,
                    payload=CurveData(
                        revision=snapshot_revision(self._snapshot),
                        generation=snapshot_generation(self._snapshot),
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
    ) -> FacetData:
        """Every heatmap cell of a lattice facet from ONE pass over the data.

        A facet of scan heatmaps is the heatmap computation with one more
        key: over a DATA/repeat axis the facet is a kept tensor dimension
        and each cell is a column of the folded plane; over a scan
        dimension the facet joins the combined row key and each cell is a
        row window, compressed to its own used axis sets exactly as the
        generic per-cell domains are.  The oracle tests hold every cell
        pixel for pixel to the generic facet.
        """

        facet_dimension = int(self._resolve(spec.facet).dimension)
        row_facet = facet_dimension == int(
            self._resolve(cell.x).dimension
        ) == int(self._resolve(cell.y).dimension)
        if row_facet:
            planes = self._factored_planes(
                (spec.facet, cell.y, cell.x), (), cell.reduction, False
            )
        else:
            planes = self._factored_planes(
                (cell.y, cell.x), (spec.facet,), cell.reduction, False
            )
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
                revision=snapshot_revision(self._snapshot),
                generation=snapshot_generation(self._snapshot),
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
            revision=snapshot_revision(self._snapshot),
            generation=snapshot_generation(self._snapshot),
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
        if self._snapshot.block.values is None:
            values, usable, source_sigma, rows = self._segment_arrays(sigma=uncertainty)
            if rows is not None:
                return None
        else:
            values = self.samples.value.canonical
            usable = self.samples.valid_mask
            source_sigma = self.samples.sigma if uncertainty else None
        if values.dtype.kind == "c":
            return None
        if not row_refs:
            return None
        x = row_refs[-1]
        try:
            x_resolved = self._resolve(x)
            row_resolved = tuple(self._resolve(ref) for ref in row_refs)
            group_resolved = tuple(self._resolve(ref) for ref in groups)
        except AxisResolutionError:
            return None
        shape = values.shape
        row_dimension = int(x_resolved.dimension)
        if any(int(resolved.dimension) != row_dimension for resolved in row_resolved):
            return None
        strides = []
        acc = 1
        for size in reversed(shape):
            strides.insert(0, acc)
            acc *= int(size)
        kept_dims: list[int] = []
        for ref, resolved in zip(groups, group_resolved):
            dimension = int(resolved.dimension)
            if dimension == row_dimension or dimension in kept_dims:
                return None
            kept_dims.append(dimension)

        # One representative element per row / per group coordinate puts
        # the existing domain machinery (used-set compression, labels,
        # units) to work on arrays the size of the AXIS, not the dataset.
        rows = int(shape[row_dimension])
        row_representatives = (
            np.arange(rows, dtype=np.int64) * strides[row_dimension]
        )
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

        reduce_axes = tuple(
            axis
            for axis in range(values.ndim)
            if axis != row_dimension and axis not in kept_dims
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
                if axis == row_dimension or axis in kept_dims
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
        remaining = sorted([row_dimension, *kept_dims])
        permutation = [remaining.index(row_dimension)] + [
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

        def from_code_order(plane: NDArray[Any]) -> NDArray[Any]:
            """Undo ``code_ordered`` for bucket centres used by the tensor."""

            restored = np.asarray(plane).reshape((rows, *group_sizes))
            for position, order in enumerate(group_orders):
                if order is not None:
                    restored = np.take(
                        restored, np.argsort(order), axis=1 + position
                    )
            return np.transpose(restored, np.argsort(permutation))

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
            # An absent bucket has a zero count and is masked below; a
            # present one keeps whatever its sum IS, an overflowed
            # infinity included, so that the finiteness rule of validity
            # can refuse it exactly as the generic path does.
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
            # key as the means.  The compiled pass reads every sample once
            # and accumulates both centred moments about its FINAL bucket's
            # own mean; the small residue is then folded as before.
            safe_fold_codes = np.maximum(fold_codes, 0)

            def centred_moments(
                plane: Any, offsets: NDArray[np.float64]
            ) -> tuple[Any, Any]:
                plane = np.asarray(plane, dtype=np.float64)
                bucket_offsets = np.asarray(offsets, dtype=np.float64).reshape(-1)
                per_group_offsets = from_code_order(
                    bucket_offsets[safe_fold_codes].reshape(rows, combos)
                )
                per_group = _centred_moment_sums(
                    plane,
                    per_group_offsets,
                    None if all_valid else usable,
                    remaining,
                    shape,
                )
                if per_group is None:
                    centre_shape = tuple(
                        int(shape[axis]) if axis in remaining else 1
                        for axis in range(values.ndim)
                    )
                    delta = plane - per_group_offsets.reshape(centre_shape)
                    if all_valid:
                        first = np.sum(
                            delta, axis=reduce_axes, dtype=np.float64
                        )
                    else:
                        first = np.sum(
                            delta,
                            axis=reduce_axes,
                            where=usable,
                            dtype=np.float64,
                        )
                    np.square(delta, out=delta)
                    if all_valid:
                        second = np.sum(
                            delta, axis=reduce_axes, dtype=np.float64
                        )
                    else:
                        second = np.sum(
                            delta,
                            axis=reduce_axes,
                            where=usable,
                            dtype=np.float64,
                        )
                    per_group = first, second
                folded = []
                for moment in per_group:
                    total, _ = _aggregate_by_codes(
                        code_ordered(moment).reshape(-1),
                        np.ones(fold_codes.shape, dtype=np.bool_),
                        fold_codes,
                        buckets,
                        Reduction.SUM,
                    )
                    folded.append(total)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return tuple(
                        np.where(counts > 0, moment / counts, np.nan)
                        for moment in folded
                    )

            sem_flat = _sem_of_mean(
                np.asarray(y_flat, np.float64),
                counts,
                as_double,
                source_sigma,
                centred_moments,
            )
            assert sem_flat is not None

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
            for array in (y_column, valid_column, sem_column):
                if array is not None and array.flags.writeable:
                    array.setflags(write=False)
            y_display = schema_value_unit(self._schema, self._unit_registry).convert_value_to(
                y_column, self._value_display_unit
            )
            if y_display.flags.writeable:
                y_display.setflags(write=False)
            label = (self._schema.value_schema.name or "value") if not key else ", ".join(
                item.label for item in key
            )
            series.append(
                CurveSeries(
                    x=shown_x,
                    x_labels=shown_labels,
                    y=QuantityArray(
                        y_column,
                        y_display,
                        schema_value_unit(self._schema, self._unit_registry),
                        self._value_display_unit,
                        (self._schema.value_schema.name or "value"),
                    ),
                    valid=valid_column,
                    sem=sem_column,
                    group_key=key,
                    label=label,
                )
            )
        return tuple(series)


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

    def image(
        self,
        x: AxisRef,
        y: AxisRef,
        *,
        aggregation: Reduction = Reduction.MEAN,
    ) -> ImageData:
        self.validate_image(x, y)
        aggregation = _validate_aggregation(aggregation)
        if aggregation is Reduction.LAST:
            return self._last_view(keep=(x, y)).image(x, y, aggregation=Reduction.MEAN)
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

        This path is deliberately narrow.  Any pair of axes that each maps
        one-to-one onto a distinct physical tensor dimension can use it;
        mapped Repeat/Point grids otherwise continue through the shared
        factored or axis-code kernels.
        """

        projected = self._dense_tensor_projection((y, x), aggregation)
        if projected is None:
            return None
        domains, values, counts, _sem, valid = projected
        y_domain, x_domain = domains
        return self._image_from_planes(
            x,
            y,
            x_domain,
            y_domain,
            np.asarray(values),
            np.asarray(counts, dtype=np.int64),
            valid=np.asarray(valid, dtype=np.bool_),
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
        valid: NDArray[np.bool_] | None = None,
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
            if valid is not None:
                valid = valid[:, used_x]
        if used_y is not None and not bool(used_y.all()):
            y_canonical = y_canonical[used_y]
            y_display = y_display[used_y]
            z = z[used_y]
            counts = counts[used_y]
            if valid is not None:
                valid = valid[used_y]
        z = np.ascontiguousarray(z)
        valid = (
            (counts > 0) & np.isfinite(z)
            if valid is None
            else np.asarray(valid, dtype=np.bool_)
        )
        if valid.flags.writeable:
            valid.setflags(write=False)
        z_display = schema_value_unit(self._schema, self._unit_registry).convert_value_to(
            z, self._value_display_unit
        )
        z.setflags(write=False)
        z_display.setflags(write=False)
        x_coordinate = self._resolve(x).coordinate
        y_coordinate = self._resolve(y).coordinate
        return ImageData(
            revision=snapshot_revision(self._snapshot),
            generation=snapshot_generation(self._snapshot),
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
                schema_value_unit(self._schema, self._unit_registry),
                self._value_display_unit,
                (self._schema.value_schema.name or "value"),
            ),
            valid=valid,
        )

    def _axis_projection(
        self, refs: tuple[AxisRef, ...]
    ) -> tuple[tuple[_Domain, ...], tuple[NDArray[np.int64], ...], tuple[int, ...]]:
        """Resolve each kept axis to its small code vector and tensor dimension."""

        shape = schema_shape(self._schema)
        domains = []
        axis_codes = []
        dimensions = []
        for ref in refs:
            resolved = self._resolve(ref)
            dimension = int(resolved.dimension)
            domain = resolved.retained_domain
            if domain is None:
                stride = 1
                for size in shape[dimension + 1:]:
                    stride *= int(size)
                representatives = np.arange(shape[dimension], dtype=np.int64) * stride
                domain = self._domain(ref, representatives)
                # These codes depend only on the resolved schema/unit context,
                # which already gates _axis_cache inheritance across revisions.
                object.__setattr__(resolved, "retained_domain", domain)
            codes = np.asarray(domain.codes, dtype=np.int64)
            axis_codes.append(codes)
            dimensions.append(dimension)
            domains.append(domain)
        return tuple(domains), tuple(axis_codes), tuple(dimensions)

    def _segment_arrays(self, *, sigma: bool = False, selection: Any = None) -> tuple:
        """One numeric scratch for this view, independent of storage block count."""
        cached = self._packed_segments if selection is None else None
        source_block = self._snapshot.block
        segments = source_block.segments
        carry, self._packed_carry = self._packed_carry, None
        if cached is None and selection is None and carry is not None:
            old_snapshot, prepared = carry
            old_block = old_snapshot.block
            retained = len(old_block.segments)
            shape = schema_shape(self._schema)
            cell_shape = shape[2:]
            old_rows = prepared[0].size // math.prod(cell_shape)
            last = (old_rows - 1 if prepared[3] is None else
                    int(prepared[3][0][-1] * shape[1] + prepared[3][1][-1]) if old_rows else -1)
            origins = source_block.segment_origins[retained:]
            extents = source_block.segment_shapes[retained:]
            starts = origins[:, 0] * shape[1] + origins[:, 1]
            ends = (origins[:, 0] + extents[:, 0] - 1) * shape[1] + origins[:, 1] + extents[:, 1]
            reusable = (
                retained <= len(segments) and prepared[4] is None
                and old_block.schema.physical_shape[1:] == shape[1:]
                and old_block.schema.cell_domain == self._schema.cell_domain
                and old_block.schema.value_schema == self._schema.value_schema
                and all(map(is_, old_block.segments, segments))
                and np.array_equal(old_block.segment_origins, source_block.segment_origins[:retained])
                and np.array_equal(old_block.segment_shapes, source_block.segment_shapes[:retained])
                and (not starts.size or (starts[0] > last and bool(np.all(starts[1:] >= ends[:-1]))))
            )
            if reusable:
                tail_values, tail_valid, tail_sigma, tail_rows = self._segment_arrays(
                    sigma=sigma and prepared[5], selection=np.arange(retained, len(segments)),
                )
                old_values = prepared[0].reshape((-1, *cell_shape))
                old_valid = prepared[1].reshape(old_values.shape)
                tail_values = tail_values.reshape((-1, *cell_shape))
                tail_valid = tail_valid.reshape(tail_values.shape)
                values = (np.concatenate((old_values, tail_values), axis=0)
                          if tail_values.shape[0] else old_values)
                valid = (old_valid if not tail_values.shape[0] else
                         np.broadcast_to(np.asarray(True), values.shape)
                         if _stride_zero_all_true(old_valid) and _stride_zero_all_true(tail_valid) else
                         np.concatenate((old_valid, tail_valid), axis=0))
                errors = None
                sigma_read = sigma and prepared[5]
                if sigma_read and (prepared[2] is not None or tail_sigma is not None):
                    previous_sigma = (np.broadcast_to(np.asarray(np.nan), old_values.shape)
                                      if prepared[2] is None else prepared[2].reshape(old_values.shape))
                    arriving_sigma = (np.broadcast_to(np.asarray(np.nan), tail_values.shape)
                                      if tail_sigma is None else tail_sigma.reshape(tail_values.shape))
                    errors = (np.concatenate((previous_sigma, arriving_sigma), axis=0)
                              if tail_values.shape[0] else previous_sigma)
                origins, extents = source_block.segment_origins, source_block.segment_shapes
                counts = extents[:, 0] * extents[:, 1]
                complete = (values.shape[0] == shape[0] * shape[1]
                            and np.array_equal(origins[:, 0] * shape[1] + origins[:, 1],
                                               np.cumsum(counts) - counts)
                            and bool(np.all((extents[:, 0] - 1) * shape[1] + extents[:, 1] == counts)))
                if complete:
                    values, valid = values.reshape(shape), valid.reshape(shape)
                    if errors is not None:
                        errors = errors.reshape(shape)
                    row_indices = None
                else:
                    previous_rows = (np.divmod(np.arange(old_rows), shape[1])
                                     if prepared[3] is None else prepared[3])
                    arriving_rows = (np.divmod(np.arange(tail_values.shape[0]), shape[1])
                                     if tail_rows is None else tail_rows)
                    row_indices = tuple(np.concatenate((old, new)) for old, new in
                                        zip(previous_rows, arriving_rows))
                for array in (values, valid, errors):
                    if array is not None:
                        array.setflags(write=False)
                cached = values, valid, errors, row_indices, None, sigma_read
        if cached is None:
            values, mask, errors, row_indices, order = source_block.packed_planes(
                selection=selection, sigma=sigma,
            )
            if isinstance(mask, bool):
                valid = np.broadcast_to(np.asarray(mask), values.shape)
            else:
                components = self._schema.value_schema.validity_contract.component_axis_ids
                leading = values.shape[:2] if row_indices is None else values.shape[:1]
                spread = (*leading, *(axis.size if axis.axis_id in components else 1
                                     for axis in self._schema.cell_domain.axes))
                valid = np.broadcast_to(mask.reshape(spread), values.shape)
            if values.dtype.kind not in "biu":
                finite = np.isfinite(values)
                if not bool(finite.all()):
                    valid = finite if _stride_zero_all_true(valid) else valid & finite
            valid.setflags(write=False)
            cached = values, valid, errors, row_indices, order, sigma
        values, valid, errors, row_indices, order, sigma_read = cached
        if sigma and not sigma_read:
            errors = source_block._pack_sigma(values.shape, selection=selection, order=order)
            cached = values, valid, errors, row_indices, order, True
        if selection is None:
            self._packed_segments = cached
        return values, valid, errors, row_indices

    def _segmented_axes(
        self, axis_codes: tuple[NDArray[np.int64], ...], dimensions: tuple[int, ...],
        sizes: tuple[int, ...], aggregation: Reduction, *, uncertainty: bool,
    ) -> tuple:
        """Apply the ordinary axis reduction once to the calculation scratch."""
        source, valid, sigma, row_indices = self._segment_arrays(sigma=uncertainty)
        if row_indices is None:
            codes, kept = axis_codes, dimensions
        else:
            codes = tuple(axis[row_indices[dimension]] if dimension < 2 else axis
                          for axis, dimension in zip(axis_codes, dimensions))
            kept = tuple(0 if dimension < 2 else dimension - 1 for dimension in dimensions)
        values, counts, _present = _axis_aggregate(source, valid, codes, kept, sizes, aggregation)
        values, counts = values.reshape(sizes), counts.reshape(sizes)

        # Existence is the parent schema's geometry, not acquisition success.
        shape = schema_shape(self._schema)
        reachable = np.zeros(1, dtype=np.int64)
        for dimension in sorted(set(dimensions)):
            contribution = np.zeros(shape[dimension], dtype=np.int64)
            admitted = np.ones(contribution.shape, dtype=np.bool_)
            for index, (axis, carrier) in enumerate(zip(axis_codes, dimensions)):
                if carrier == dimension:
                    admitted &= axis >= 0
                    contribution += axis * math.prod(sizes[index + 1:])
            used = np.unique(contribution[admitted])
            reachable = (reachable[:, None] + used[None, :]).reshape(-1)
        presence = np.zeros(math.prod(sizes), dtype=np.bool_)
        presence[reachable] = True
        sem = None
        if uncertainty:
            def centred_moments(plane: Any, offsets: NDArray[np.float64]) -> tuple[Any, Any]:
                first, second, _counts, _present = _axis_aggregate(
                    np.asarray(plane), valid, codes, kept, sizes, Reduction.SUM,
                    offsets=np.asarray(offsets, dtype=np.float64).reshape(-1),
                )
                return first.reshape(sizes), second.reshape(sizes)

            sem = _sem_of_mean(values, counts, source, sigma, centred_moments)
            assert sem is not None
        return values, counts, presence.reshape(sizes), sem

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

        projection = self._axis_projection(refs) if projection is None else projection
        domains, axis_codes, dimensions = projection
        if self._snapshot.block.values is None:
            reduced, counts, present, _sem = self._segmented_axes(
                axis_codes, dimensions, tuple(domain.size for domain in domains),
                aggregation, uncertainty=False,
            )
            return domains, reduced, counts, present
        domain_sizes = tuple(domain.size for domain in domains)
        reduced, counts, present = _axis_aggregate(
            self.samples.value.canonical, self.samples.valid_mask,
            axis_codes, dimensions, domain_sizes, aggregation,
        )
        return (
            domains, reduced.reshape(domain_sizes), counts.reshape(domain_sizes),
            present.reshape(domain_sizes),
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


    def histogram(
        self,
        *,
        bins: int | Sequence[float],
        values: NDArray[Any] | None = None,
        valid: NDArray[np.bool_] | None = None,
        reduce_axes: Sequence[AxisRef] = (),
        aggregation: Reduction = Reduction.MEAN,
        group_by: tuple[AxisRef, ...] = (),
        window: int = 1,
    ) -> HistogramData:
        """Distribution of the acquired values.

        Every axis pools into the one distribution unless it is named in
        ``reduce_axes``, which collapses it under ``aggregation`` first --
        the difference between the distribution of every shot and the
        distribution of each site's mean over shots.
        """

        if values is None and valid is None:
            plan = self._histogram_plan(tuple(group_by), tuple(reduce_axes), aggregation, window)
        else:
            if group_by:
                raise ValueError("grouped histogram uses its coordinate-owned source values")
            selected, usable = self.histogram_pool(
                values=values, valid=valid, reduce_axes=reduce_axes, aggregation=aggregation,
            )
            plan = _HistogramPlan(selected, usable, np.zeros(1, dtype=np.int64), 0, ((),))
        return self._histogram_from_plan(bins, plan)

    def _reduction_plan(
        self, refs: Sequence[AxisRef]
    ) -> tuple[tuple[int, ...], tuple[AxisRef, ...]]:
        """What a reduction names: dense axes and mapped-domain axes.

        Any domain may carry several logical axes over one physical dimension.
        Reducing one such axis preserves the combinations of its siblings;
        treating its physical dimension as a whole would silently reduce every
        sibling too.  A sole axis owns its physical dimension and takes the
        direct NumPy reduction path.

        Said once here because it is asked from two directions: a whole
        tensor (a standalone histogram) and a set of sample positions (one
        facet cell), which must agree on what survives.
        """

        coordinates: list[AxisRef] = []
        dimensions: set[int] = set()
        for ref in refs:
            resolved = self._resolve(ref).contract
            local_dimension = resolved.domain.physical_dimension(resolved.axis_id)
            siblings = tuple(
                axis
                for axis in resolved.domain.axes
                if axis.coordinate_of is None and resolved.domain.physical_dimension(axis.axis_id)
                == local_dimension
            )
            if len(siblings) == 1:
                dimensions.add(int(resolved.dimension))
            else:
                coordinates.append(ref)
        return tuple(sorted(dimensions)), tuple(coordinates)

    def _carrier_group_codes(
        self, dimension: int, coordinates: Sequence[AxisRef]
    ) -> tuple[NDArray[np.int64], int]:
        """One compact group code per carrier row after named axes collapse."""

        resolved_coordinates = tuple(
            self._resolve(ref).contract
            for ref in coordinates
            if int(self._resolve(ref).dimension) == dimension
        )
        if not resolved_coordinates:
            raise ValueError("carrier group requires a mapped axis")
        domain = resolved_coordinates[0].domain
        if any(resolved.domain is not domain for resolved in resolved_coordinates):
            raise ValueError("one physical dimension cannot span two domains")
        local_dimension = domain.physical_dimension(
            resolved_coordinates[0].axis_id
        )
        named = {
            domain.coordinate_axis(resolved.axis_id).axis_id for resolved in resolved_coordinates
        }
        kept = tuple(
            axis
            for axis in domain.axes
            if axis.coordinate_of is None and axis.axis_id not in named
            and domain.physical_dimension(axis.axis_id) == local_dimension
        )
        if not kept:
            return np.zeros(domain.size, dtype=np.int64), 1
        combined = np.zeros(domain.size, dtype=np.int64)
        span = 1
        for axis in kept:
            combined = combined * int(axis.size) + domain.codes(axis.axis_id)
            span *= int(axis.size)
        used = np.flatnonzero(np.bincount(combined, minlength=span))
        remap = np.full(span, -1, dtype=np.int64)
        remap[used] = np.arange(used.size, dtype=np.int64)
        return remap[combined], int(used.size)

    def _reduction_buckets(
        self,
        dimensions: Sequence[int],
        coordinates: Sequence[AxisRef],
        *,
        shape: tuple[int, ...] | None = None,
    ) -> "_ReductionBuckets":
        """Axis-sized codes naming what survives the reduction.

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

        shape = tuple(schema_shape(self._schema)) if shape is None else shape
        collapse = set(int(axis) for axis in dimensions)
        grouped: dict[int, tuple[NDArray[np.int64], int]] = {}
        for dimension in sorted(
            {int(self._resolve(ref).dimension) for ref in coordinates}
        ):
            grouped[dimension] = self._carrier_group_codes(
                dimension, coordinates
            )

        def extent(axis: int) -> int:
            if axis in grouped:
                return grouped[axis][1]
            return int(shape[axis])

        keep_axes = [axis for axis in range(len(shape)) if axis not in collapse]
        out_shape = tuple(extent(axis) for axis in keep_axes)
        stride, strides = 1, {}
        for axis in reversed(keep_axes):
            strides[axis] = stride
            stride *= extent(axis)

        return _ReductionBuckets(
            codes=tuple(grouped[axis][0] if axis in grouped else
                        np.arange(shape[axis], dtype=np.int64) for axis in keep_axes),
            count=max(1, stride),
            shape=out_shape,
            axes=tuple(keep_axes),
            strides=tuple(strides[axis] for axis in keep_axes),
            extents=out_shape,
            carrier_groups=tuple(
                (axis, codes) for axis, (codes, _count) in grouped.items()
            ),
        )

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
        if aggregation is Reduction.LAST:
            terms = {self._resolve(ref).contract.axis_id: LATEST_COORDINATE for ref in refs}
            indices = selection_indices(self._schema, value_selection(self._schema, terms))
            scoped = self._last_view(reduced=refs)
            return scoped._collapse_axes(
                restricted_values(values, self._schema, *indices),
                restricted_values(np.broadcast_to(valid, values.shape), self._schema, *indices),
                refs, Reduction.MEAN,
            )
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
        collapsed, counts, _presence = _axis_aggregate(
            values, usable, buckets.codes, buckets.axes, buckets.shape, aggregation,
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
        values = self.samples.value.canonical
        validity = self.samples.valid_mask
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

        layout = self._history_layout
        assert layout is not None
        window = _history_window(window)
        cached = self._history_mask_cache.get(window)
        if cached is None:
            cached = self._spread_rows(layout.row_mask(window))
            self._history_mask_cache[window] = cached
        return cached

    def _spread_rows(self, plane: NDArray[Any]) -> NDArray[Any]:
        """One value per point row, broadcast over the whole sample tensor."""

        values = self.samples.value.canonical
        return np.broadcast_to(
            np.reshape(plane, (1, plane.size, *([1] * (values.ndim - 2)))),
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
        values = self.samples.value.canonical
        validity = self.samples.valid_mask
        if self.has_primary_index:
            return values, self.history_validity(window)
        count = min(window, max(1, schema_repeat_count(self._schema)))
        return values[-count:], validity[-count:]

    def window_frequency(
        self, window: int
    ) -> tuple[int, NDArray[np.int64]] | None:
        """How many samples of each integer value the last ``window`` shots hold.

        A pixel-pool histogram at a deep window recounted every value in
        the window on every shot -- seventeen million values, seventy
        milliseconds -- for a picture that one shot's worth of pixels had
        changed by a thousandth.  The count of each value is a sum over
        shots, so it moves by adding the shot that arrived and subtracting
        the one that left.  The block says which those are -- its
        ``IndexedWindow`` names the shots by absolute number and says since
        when the retained ones have been untouched -- and the previous
        revision's view hands its table across, so the steady state costs
        one shot's bincount, not the window's.

        Returns ``(offset, counts)`` with ``counts[v - offset]`` the number
        of valid samples equal to ``v``, or None when this dataset is not an
        integer indexed history or spans more levels than a table is worth;
        the callers then count from the values as before.  Exactness is the
        contract: the table equals a fresh count of the same window, always.
        """

        layout = self._history_layout
        provenance = self._snapshot.block.window
        values = self.samples.value.canonical
        if layout is None or provenance is None or values.dtype.kind not in "iu":
            return None
        window = _history_window(window)
        if layout.inner_count is None:
            self._frequency_carry = None
            return self._count_frequency(window)
        carry = self._frequency_carry
        if (
            carry is not None
            and carry.snapshot is self._snapshot
            and carry.window == window
        ):
            return carry.offset, carry.counts
        keep = min(window, layout.shot_count)
        positions = np.arange(
            layout.shot_count - keep, layout.shot_count, dtype=np.int64
        )
        ids = np.asarray(provenance.latest + layout.cells[positions], dtype=np.int64)
        revision = snapshot_revision(self._snapshot)
        counted = None
        if (
            carry is not None
            and carry.window == window
            and carry.inner_count == int(layout.inner_count)
            and carry.dtype == values.dtype
            and carry.snapshot.ref.block_id == self._snapshot.ref.block_id
            and carry.snapshot.ref.stream_generation
            == self._snapshot.ref.stream_generation
            and provenance.stable_since <= carry.revision <= revision
        ):
            counted = self._advance_frequency(carry, ids, positions)
        if counted is None:
            counted = self._count_frequency(window)
        if counted is None:
            self._frequency_carry = None
            return None
        offset, counts = counted
        counts.setflags(write=False)
        self._frequency_carry = _WindowFrequency(
            self._snapshot,
            self.samples.valid_mask,
            revision,
            window,
            int(layout.inner_count),
            values.dtype,
            ids,
            positions,
            offset,
            counts,
        )
        return offset, counts

    def _count_frequency(self, window: int) -> tuple[int, NDArray[np.int64]] | None:
        """The window's frequency table from scratch: every valid sample once."""

        values = self.samples.value.canonical
        usable = self.history_validity(window)
        selected = (
            values.reshape(-1) if _stride_zero_all_true(usable) else values[usable]
        )
        span = _frequency_span(values.dtype, selected)
        if span is None:
            return None
        offset, size = span
        if not selected.size:
            return offset, np.zeros(size, dtype=np.int64)
        shifted = np.subtract(selected, offset, dtype=np.int64)
        return offset, np.bincount(shifted, minlength=size)

    def _advance_frequency(
        self,
        carry: _WindowFrequency,
        ids: NDArray[np.int64],
        positions: NDArray[np.int64],
    ) -> tuple[int, NDArray[np.int64]] | None:
        """The carried table moved to this window, or None when it is not worth it."""

        layout = self._history_layout
        assert layout is not None
        leaving = np.isin(carry.ids, ids, invert=True)
        arriving = np.isin(ids, carry.ids, invert=True)
        changed = int(np.count_nonzero(leaving)) + int(np.count_nonzero(arriving))
        if changed == 0:
            return carry.offset, carry.counts
        if changed > max(1, ids.size // 4):
            return None
        inner = int(layout.inner_count)
        offset = carry.offset
        table = carry.counts.copy()
        previous_values = snapshot_values(carry.snapshot)
        for position in carry.positions[leaving]:
            moved = _apply_shot(
                table, offset, previous_values, carry.valid, inner, int(position), -1
            )
            if moved is None:
                return None
            table, offset = moved
        values = self.samples.value.canonical
        valid = self.samples.valid_mask
        for position in positions[arriving]:
            moved = _apply_shot(table, offset, values, valid, inner, int(position), 1)
            if moved is None:
                return None
            table, offset = moved
        return offset, table

    def histogram_from_frequency(
        self,
        *,
        bins: int | Sequence[float],
        frequency: tuple[int, NDArray[np.int64]],
    ) -> HistogramData | None:
        """The histogram of a window from its frequency table.

        None when the bins are not integer-aligned -- then the values must
        be counted against the edges themselves.
        """

        canonical_bins = self._canonical_histogram_bins(bins)
        if isinstance(canonical_bins, int):
            return None
        offset, table = frequency
        counts = _counts_from_frequency(offset, table, np.asarray(canonical_bins))
        if counts is None:
            return None
        return self._histogram_from_counts(canonical_bins, counts)

    def pooled_values(self) -> NDArray[Any]:
        """Every value this revision would pool, in canonical units.

        Canonical values for one whole-revision rolling reduction.  The
        display conversion happens only after the reduction.
        """

        cached = self._pooled_cache
        if cached is not None:
            return cached
        pooled = self._pool(self.samples.value.canonical)
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

        valid = self.samples.valid_mask
        plane = np.asarray(plane)
        if _stride_zero_all_true(valid) or bool(np.all(valid)):
            return plane.reshape(-1).view()
        return plane[valid].reshape(-1)

    def _pooled_sigma(self) -> NDArray[np.float64] | None:
        """The samples' own sigma, pooled exactly as the values are."""

        sigma = self.samples.sigma
        return None if sigma is None else self._pool(sigma)


    def validate_rolling(
        self, group: AxisRef | None, *, x: AxisRef | None = None,
    ) -> None:
        """Validate the fixed record carrier before allocating history buckets.

        ``x`` chooses only the carrier's coordinate, not another data axis.
        Without indexed history each physical Repeat row is one record.
        """

        if x is not None:
            if not isinstance(x, AxisRef):
                raise TypeError("rolling coordinate must be an AxisRef or None")
            contract = resolve_axis(self._schema, x)
            if (
                not self.has_primary_index
                or x.domain.value != "point"
                or contract.domain.coordinate_axis(contract.axis_id).axis_id
                != PRIMARY_INDEX_AXIS_ID
                or contract.axis_id not in (PRIMARY_INDEX_AXIS_ID, SHOT_TIME_AXIS_ID)
            ):
                raise DataViewError(
                    "Rolling has a fixed record axis; choose its source index "
                    "or shot time coordinate"
                )
        if group is not None:
            if not isinstance(group, AxisRef):
                raise TypeError("rolling group must be an AxisRef or None")
            contract = resolve_axis(self._schema, group)
            record_group = (
                group.domain.value == "point"
                and contract.domain.coordinate_axis(contract.axis_id).axis_id
                == PRIMARY_INDEX_AXIS_ID
            ) if self.has_primary_index else group.domain.value == "repeat"
            if record_group:
                raise DataViewError("Rolling's fixed record axis cannot also be Group")
            self._resolve(group)

    def _single_revision_history(
        self,
        *,
        group: AxisRef | None = None,
        aggregation: Reduction = Reduction.MEAN,
        uncertainty: bool = True,
    ) -> RollingHistory:
        """Reduce one source revision to a one-shot rolling history.

        ``uncertainty`` is whether the caller will DRAW the band, exactly
        as for :meth:`rolling_history`: the standard error is a second
        pass over every value, and only a MEAN has one.
        """

        aggregation = _validate_aggregation(aggregation)
        uncertainty = bool(uncertainty) and aggregation is Reduction.MEAN
        if group is None:
            if self._snapshot.block.values is None:
                return self._history_from_axes(
                    (np.zeros(1, dtype=np.int64),), (0,), (1,), ((),),
                    aggregation=aggregation, uncertainty=uncertainty,
                )
            pooled = self.pooled_values()
            value = _reduce_scalar(pooled, aggregation)
            sem = None
            if uncertainty:
                # This pool already contains only the snapshot's finite,
                # valid samples. Its length is the complete count.
                count = int(pooled.size)

                def centred_moments(
                    plane: Any, offsets: NDArray[np.float64]
                ) -> tuple[Any, Any]:
                    if not count:
                        empty = np.asarray([np.nan])
                        return empty, empty
                    first, second = _centred_moment_totals(
                        np.asarray(plane).reshape(-1),
                        float(np.asarray(offsets).reshape(-1)[0]),
                        None,
                    )
                    return (
                        np.asarray([first / count]),
                        np.asarray([second / count]),
                    )

                sem = _sem_of_mean(
                    np.asarray([value], dtype=np.float64),
                    np.asarray([count], dtype=np.int64),
                    pooled,
                    self._pooled_sigma(),
                    centred_moments,
                )
                assert sem is not None
            return RollingHistory(
                revision=snapshot_revision(self._snapshot),
                generation=snapshot_generation(self._snapshot),
                values=np.asarray([[value]], dtype=np.float64),
                valid=np.asarray([[pooled.size > 0 and np.isfinite(value)]]),
                counts=np.asarray([[pooled.size]], dtype=np.int64),
                group_keys=((),),
                sem=None if sem is None else np.asarray(sem).reshape(1, 1),
            )
        domains, codes, dimensions = self._axis_projection((group,))
        keys = tuple((value,) for value in domains[0].values)
        return self._history_from_axes(
            (np.zeros(1, dtype=np.int64), *codes), (0, *dimensions),
            (1, domains[0].size), keys,
            aggregation=aggregation, uncertainty=uncertainty,
        )

    def rolling_history(
        self,
        *,
        x: AxisRef | None = None,
        group: AxisRef | None = None,
        aggregation: Reduction = Reduction.MEAN,
        uncertainty: bool = True,
    ) -> RollingHistory:
        """The shot history of this snapshot as one batch of planes.

        A static snapshot carries its shot history on the repeat axis; each
        repeat reduces to one row exactly as a whole revision reduces to
        one.  A snapshot without repeats degenerates to a one-shot history.

        ``uncertainty`` is whether the caller will DRAW the band.  Its
        standard error needs a second pass over every value -- squared,
        masked and reduced again -- which the rolling panel paid on every
        revision whether or not the band was switched on.
        """

        self.validate_rolling(group, x=x)
        if aggregation is Reduction.LAST:
            from .semantics import axis_choices_for_schema

            retained = (
                (AxisRef.point(PRIMARY_INDEX_AXIS_ID.value),)
                if self.has_primary_index
                else tuple(ref for ref in axis_choices_for_schema(self._schema)
                           if ref.domain.value == "repeat")
            )
            return self._last_view(keep=retained + (() if group is None else (group,))).rolling_history(
                x=x, group=group, aggregation=Reduction.MEAN, uncertainty=uncertainty,
            )
        if self.has_primary_index:
            return self._history_by_primary_index(
                group=group,
                aggregation=aggregation,
                uncertainty=uncertainty,
            )
        repeats = schema_repeat_count(self._schema)
        if repeats <= 1:
            return self._single_revision_history(
                group=group, aggregation=aggregation, uncertainty=uncertainty
            )
        aggregation = _validate_aggregation(aggregation)
        tensor = self._repeat_history_tensor(
            group=group,
            aggregation=aggregation,
            repeats=repeats,
            uncertainty=uncertainty,
        )
        if tensor is not None:
            return tensor
        axis_codes = (np.arange(repeats, dtype=np.int64),)
        dimensions, sizes = (0,), (repeats,)
        keys = ((),)
        if group is not None:
            domains, codes, group_dimensions = self._axis_projection((group,))
            axis_codes += codes
            dimensions += group_dimensions
            sizes += (domains[0].size,)
            keys = tuple((value,) for value in domains[0].values)
        return self._history_from_axes(
            axis_codes, dimensions, sizes, keys,
            aggregation=aggregation, uncertainty=uncertainty,
        )

    def _repeat_history_tensor(
        self,
        *,
        group: AxisRef | None,
        aggregation: Reduction,
        repeats: int,
        uncertainty: bool = True,
    ) -> RollingHistory | None:
        """Reduce a regular repeat history once, not once per repeat.

        Runtime remains the only history owner; this is only a projection of
        its immutable Dataset.  A one-to-one group is one retained tensor axis.
        Anything whose group/domain cannot be proven one-to-one returns None
        and keeps the generic position path above.
        """

        if self._snapshot.block.values is None:
            return None
        values = self.samples.value.canonical
        usable = self.samples.valid_mask
        if values.shape[0] != repeats:
            return None
        if group is None:
            group_count = 1
            keys: tuple[tuple[AxisValue, ...], ...] = ((),)

            def cube(plane: Any) -> NDArray[Any]:
                return np.asarray(plane).reshape(repeats, 1, -1)

        else:
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
            pool = int(value_cube.shape[-1])
            marks = (
                None
                if _stride_zero_all_true(usable_cube)
                else usable_cube.reshape(1, repeats * group_count, pool)
            )

            def centred_moments(
                plane: Any, offsets: NDArray[np.float64]
            ) -> tuple[Any, Any]:
                from . import _raster_kernels as kernels

                shaped = cube(np.asarray(plane, dtype=np.float64)).reshape(
                    1, repeats * group_count, pool
                )
                sums = kernels.masked_centred_moment_sums(
                    shaped,
                    np.asarray(offsets, dtype=np.float64).reshape(-1),
                    marks,
                )
                if sums is None:
                    leading = np.moveaxis(
                        cube(np.asarray(plane, dtype=np.float64)), -1, 0
                    )
                    leading_usable = np.moveaxis(usable_cube, -1, 0)
                    delta = np.where(
                        leading_usable,
                        leading,
                        np.asarray(offsets, dtype=np.float64),
                    )
                    delta -= np.asarray(offsets, dtype=np.float64)
                    first_sum = np.sum(delta, axis=0, dtype=np.float64)
                    np.square(delta, out=delta)
                    second_sum = np.sum(delta, axis=0, dtype=np.float64)
                    sums = first_sum, second_sum
                return tuple(
                    np.divide(
                        moment.reshape(counts.shape),
                        counts,
                        out=np.full(counts.shape, np.nan, dtype=np.float64),
                        where=counts > 0,
                    )
                    for moment in sums
                )

            sem = _sem_of_mean(
                reduced,
                counts,
                values,
                self.samples.sigma,
                centred_moments,
            )
            assert sem is not None
        valid = (counts > 0) & np.isfinite(reduced)
        return RollingHistory(
            revision=snapshot_revision(self._snapshot),
            generation=snapshot_generation(self._snapshot),
            values=reduced,
            valid=valid,
            counts=counts,
            group_keys=keys,
            sem=sem,
        )

    def _history_by_primary_index(
        self,
        *,
        group: AxisRef | None,
        aggregation: Reduction,
        uncertainty: bool = True,
    ) -> RollingHistory:
        """Reduce every authored primary-index cell without arrival history."""

        layout = self._history_layout
        assert layout is not None
        aggregation = _validate_aggregation(aggregation)
        if self._snapshot.block.values is None:
            segmented = self._indexed_segment_history(group, aggregation, uncertainty)
            if segmented is not None:
                return segmented
        axis_codes = (layout.codes(),)
        dimensions = (1,)
        domain_sizes = (layout.shot_count,)
        if group is None:
            keys: tuple[tuple[AxisValue, ...], ...] = ((),)
        else:
            domains, group_codes, group_dimensions = self._axis_projection((group,))
            grouped = domains[0]
            domain_size = grouped.size
            keys = tuple((value,) for value in grouped.values)
            axis_codes += group_codes
            dimensions += group_dimensions
            domain_sizes += (domain_size,)
        return self._history_from_axes(
            axis_codes, dimensions, domain_sizes, keys,
            aggregation=aggregation, uncertainty=uncertainty,
            source_indices=layout.cells, source_times=layout.times,
        )

    def _indexed_segment_history(
        self, group: AxisRef | None, aggregation: Reduction, uncertainty: bool,
    ) -> RollingHistory | None:
        """Carry rows by immutable plane identity; compute changes as one batch."""
        layout = self._history_layout
        assert layout is not None
        block = self._snapshot.block
        origins, extents, segments = block.segment_origins, block.segment_shapes, block.segments
        row_codes = layout.codes()
        if (bool(np.any(origins[:, 0] != 0)) or bool(np.any(extents[:, 0] != 1))
                or bool(np.any(extents[:, 1] < 1))):
            self._rolling_carry = None
            return None
        shots = row_codes[origins[:, 1]]
        if (not np.array_equal(shots, row_codes[origins[:, 1] + extents[:, 1] - 1])
                or np.unique(shots).size != len(segments)):
            self._rolling_carry = None
            return None
        group_codes, group_dimension = None, None
        if group is None:
            keys = ((),)
        else:
            domains, codes, dimensions = self._axis_projection((group,))
            keys = tuple((value,) for value in domains[0].values)
            group_codes, group_dimension = codes[0], dimensions[0]
        query = (group, aggregation, uncertainty)
        carried = self._rolling_carry
        reusable = (carried is not None and carried[0] == query
                    and carried[2].group_keys == keys and carried[3] == group_dimension)
        if reusable and group_dimension != 1:
            reusable = np.array_equal(carried[4], group_codes)
        previous = carried[1] if reusable else {}
        old_history = carried[2] if reusable else None
        old_rows, old_points, matched, pending = [], [], [], []
        retained = {}
        for index, segment in enumerate(segments):
            # The carried snapshot holds these tuples alive until matching is
            # complete. IDs cannot be recycled; there is no child ref/schema.
            identity = id(segment)
            retained[identity] = (int(shots[index]), int(origins[index, 1]))
            old = previous.get(identity)
            if old is None:
                pending.append(index)
            else:
                old_rows.append(old[0])
                old_points.append(old[1])
                matched.append(index)
        matched = np.asarray(matched, dtype=np.intp)
        old_rows = np.asarray(old_rows, dtype=np.intp)
        if matched.size and group_dimension == 1:
            # A Point group also depends on its placement in the parent. Check
            # the actual selected row codes in one batch, not just group labels.
            lengths = extents[matched, 1]
            starts = np.cumsum(lengths) - lengths
            owners = np.repeat(np.arange(matched.size), lengths)
            local = np.arange(int(lengths.sum())) - starts[owners]
            current_points = origins[matched, 1][owners] + local
            previous_points = np.asarray(old_points)[owners] + local
            same = group_codes[current_points] == carried[4][previous_points]
            unchanged = np.logical_and.reduceat(same, starts)
            pending.extend(matched[~unchanged].tolist())
            matched, old_rows = matched[unchanged], old_rows[unchanged]
        values = np.full((layout.shot_count, len(keys)), np.nan)
        counts = np.zeros(values.shape, dtype=np.int64)
        valid = np.zeros(values.shape, dtype=np.bool_)
        want_sem = uncertainty and aggregation is Reduction.MEAN
        sem = np.full(values.shape, np.nan) if want_sem else None
        if matched.size:
            destination = shots[matched]
            source = old_rows
            if bool(np.all(np.diff(source) == 1)):
                source = slice(int(source[0]), int(source[-1]) + 1)
            if bool(np.all(np.diff(destination) == 1)):
                destination = slice(int(destination[0]), int(destination[-1]) + 1)
            values[destination] = old_history.values[source]
            counts[destination] = old_history.counts[source]
            valid[destination] = old_history.valid[source]
            if sem is not None:
                sem[destination] = old_history.sem[source]
        if pending:
            pending = np.asarray(pending, dtype=np.intp)
            new_shots = np.sort(shots[pending])
            source, marks, sigma, rows = self._segment_arrays(sigma=want_sem, selection=pending)
            selected_codes = row_codes if rows is None else row_codes[rows[1]]
            codes = (np.searchsorted(new_shots, selected_codes),)
            dimensions = (1,) if rows is None else (0,)
            sizes = (len(new_shots),)
            if group is not None:
                codes += (group_codes if rows is None or group_dimension > 1 else
                          group_codes[rows[group_dimension]],)
                dimensions += (group_dimension if rows is None else
                               0 if group_dimension < 2 else group_dimension - 1,)
                sizes += (len(keys),)
            reduced, counted, _presence = _axis_aggregate(
                source, marks, codes, dimensions, sizes, aggregation,
            )
            new_shape = (len(new_shots), len(keys))
            values[new_shots] = reduced.reshape(new_shape)
            counts[new_shots] = counted.reshape(new_shape)
            valid[new_shots] = ((counted > 0) & np.isfinite(reduced)).reshape(new_shape)
            if sem is not None:
                def centred_moments(plane: Any, offsets: NDArray[np.float64]) -> tuple[Any, Any]:
                    first, second, _counts, _presence = _axis_aggregate(
                        np.asarray(plane), marks, codes, dimensions, sizes, Reduction.SUM,
                        offsets=np.asarray(offsets, dtype=np.float64).reshape(-1),
                    )
                    return first, second

                errors = _sem_of_mean(reduced, counted, source, sigma, centred_moments)
                assert errors is not None
                sem[new_shots] = errors.reshape(new_shape)
        for plane in (values, counts, valid, sem):
            if plane is not None:
                plane.setflags(write=False)
        result = RollingHistory(
            snapshot_revision(self._snapshot), snapshot_generation(self._snapshot),
            values, valid, counts, keys, layout.cells, layout.times, sem,
        )
        # Reused statistics may need no raw packing at all. Do not leave an
        # unconsumed previous raw window pinned after accepting this result.
        self._packed_carry = None
        self._rolling_carry = query, retained, result, group_dimension, group_codes, self._snapshot
        return result

    def _history_from_axes(
        self, axis_codes: tuple[NDArray[np.int64], ...], dimensions: tuple[int, ...],
        domain_sizes: tuple[int, ...], keys: tuple[tuple[AxisValue, ...], ...], *,
        aggregation: Reduction, uncertainty: bool,
        source_indices: NDArray[np.int64] | None = None,
        source_times: NDArray[np.float64] | None = None,
    ) -> RollingHistory:
        shape = (domain_sizes[0], math.prod(domain_sizes[1:]))
        if self._snapshot.block.values is None:
            values, counts, _presence, sem = self._segmented_axes(
                axis_codes, dimensions, domain_sizes, aggregation,
                uncertainty=uncertainty and aggregation is Reduction.MEAN,
            )
            values, counts = values.reshape(shape), counts.reshape(shape)
            if sem is not None:
                sem = sem.reshape(shape)
        else:
            source = self.samples.value.canonical
            valid_source = self.samples.valid_mask
            values, counts, _presence = _axis_aggregate(
                source, valid_source, axis_codes, dimensions, domain_sizes, aggregation,
            )
            values, counts = values.reshape(shape), counts.reshape(shape)
            sem = None
            if uncertainty and aggregation is Reduction.MEAN:
                def centred_moments(plane: Any, offsets: NDArray[np.float64]) -> tuple[Any, Any]:
                    first, second, _counts, _presence = _axis_aggregate(
                        np.asarray(plane), valid_source, axis_codes, dimensions, domain_sizes,
                        Reduction.SUM, offsets=np.asarray(offsets, dtype=np.float64).reshape(-1),
                    )
                    return first.reshape(shape), second.reshape(shape)

                sem = _sem_of_mean(values, counts, source, self.samples.sigma, centred_moments)
                assert sem is not None
        valid = (counts > 0) & np.isfinite(values)
        for plane in (values, counts, valid, sem):
            if plane is not None:
                plane.setflags(write=False)
        return RollingHistory(
            revision=snapshot_revision(self._snapshot),
            generation=snapshot_generation(self._snapshot),
            values=values,
            valid=valid,
            counts=counts,
            group_keys=keys,
            source_indices=source_indices,
            source_times=source_times,
            sem=sem,
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
        selected = self.samples.value.canonical if values is None else values
        usable = self.samples.valid_mask if valid is None else valid
        if reduce_axes:
            selected, usable = self._collapse_axes(
                selected, usable, reduce_axes, aggregation
            )
        return selected, usable

    def facet_histogram_pool(
        self, spec: FacetGridPlot, *, window: int = 1,
    ) -> tuple[NDArray[Any], NDArray[np.bool_]]:
        plan = self._histogram_plan(
            tuple(ref for ref in (spec.facet, spec.cell.group) if ref is not None),
            spec.cell.reduced, spec.cell.reduction, window,
        )
        return plan.values, plan.valid

    def _histogram_plan(
        self, groups: tuple[AxisRef, ...], reduced: tuple[AxisRef, ...],
        aggregation: Reduction, window: int,
    ) -> "_HistogramPlan":
        """Keep grouping identities through the same named-axis reduction."""
        window = _history_window(window)
        if aggregation is Reduction.LAST:
            return self._last_view(keep=groups, reduced=reduced)._histogram_plan(
                groups, (), Reduction.MEAN, window,
            )
        key = (groups, reduced, aggregation, window)
        remembered = self._histogram_cache
        if remembered is not None and remembered[0] == key:
            return remembered[1]
        values = self.samples.value.canonical
        shape = values.shape
        valid = (self.history_validity(window)
                 if self.has_primary_index or window > 1 else self.samples.valid_mask)
        dimensions, coordinates = self._reduction_plan(reduced)
        domains = []
        for ref in groups:
            dimension = int(self._resolve(ref).dimension)
            stride = math.prod(shape[dimension + 1:])
            domain = self._domain(ref, np.arange(shape[dimension], dtype=np.int64) * stride)
            domains.append((dimension, domain))
        keys = tuple(product(*(domain.values for _, domain in domains))) if groups else ((),)
        if coordinates:
            buckets = self._reduction_buckets(dimensions, coordinates)
            values, counts, _presence = _axis_aggregate(
                values, valid, buckets.codes, buckets.axes, buckets.shape, aggregation,
            )
            valid = np.asarray(counts) > 0
            combined = np.zeros(buckets.count, dtype=np.int64)
            for dimension, domain in domains:
                codes = np.asarray(domain.codes, dtype=np.int64)
                carrier = buckets.groups_for_axis(dimension)
                if carrier is not None:
                    grouped = np.full(
                        int(buckets.extents[buckets.axes.index(dimension)]), -1, dtype=np.int64,
                    )
                    grouped[carrier] = codes
                    codes = grouped
                combined = combined * len(domain.values) + codes[buckets.axis_index(dimension)]
            code_axis = 0
        else:
            if reduced:
                values, valid = self._collapse_axes(values, valid, reduced, aggregation)
            kept = tuple(axis for axis in range(len(shape)) if axis not in dimensions)
            if groups:
                first = min(kept.index(dimension) for dimension, _domain in domains)
                code_axis = max(kept.index(dimension) for dimension, _domain in domains)
                code_shape = values.shape[first:code_axis + 1]
                combined = np.zeros(code_shape, dtype=np.int64)
                for dimension, domain in domains:
                    spread = [1] * len(code_shape)
                    spread[kept.index(dimension) - first] = -1
                    combined = combined * len(domain.values) + np.asarray(domain.codes).reshape(spread)
                combined = combined.reshape(-1)
            else:
                combined, code_axis = np.zeros(1, dtype=np.int64), 0
        plan = _HistogramPlan(values, valid, combined, code_axis, keys)
        self._histogram_cache = (key, plan)
        return plan

    def _histogram_from_plan(
        self, bins: int | Sequence[float], plan: "_HistogramPlan",
    ) -> HistogramData:
        _require_real_numeric(plan.values, None)
        edges = self._canonical_histogram_bins(bins)
        if isinstance(edges, int):
            values = np.asarray(plan.values)
            usable = np.broadcast_to(plan.valid, values.shape)
            edges = np.histogram_bin_edges(values[usable], bins=edges)
        counts = _histogram_kernel_counts(
            plan.values, plan.valid, plan.group_codes, plan.code_axis,
            len(plan.group_keys), edges,
        )
        if counts is None:
            values = np.asarray(plan.values)
            stride = math.prod(values.shape[plan.code_axis + 1:])
            codes = plan.group_codes[(np.arange(values.size) // stride) % plan.group_codes.size]
            usable = np.broadcast_to(plan.valid, values.shape).reshape(-1)
            counts = np.asarray([
                histogram_counts(values.reshape(-1), edges, usable & (codes == index))
                for index in range(len(plan.group_keys))
            ], dtype=np.int64)
        return self._histogram_from_counts(edges, counts, plan.group_keys)

    def _histogram_facet(
        self, spec: FacetGridPlot, bins: int | Sequence[float], window: int,
    ) -> FacetData:
        groups = tuple(ref for ref in (spec.facet, spec.cell.group) if ref is not None)
        plan = self._histogram_plan(groups, spec.cell.reduced, spec.cell.reduction, window)
        histogram = self._histogram_from_plan(bins, plan)
        if spec.facet is None:
            cells = (FacetCell(0, 1, 1, "Facet 1", histogram),)
        else:
            cells = []
            start = 0
            while start < len(plan.group_keys):
                value = plan.group_keys[start][0]
                stop = start + 1
                while stop < len(plan.group_keys) and plan.group_keys[stop][0] == value:
                    stop += 1
                cells.append(FacetCell(
                    len(cells), value.canonical, value.display, value.label,
                    replace(histogram, counts=histogram.counts[start:stop],
                            group_keys=tuple(key[1:] for key in plan.group_keys[start:stop])),
                ))
                start = stop
            cells = tuple(cells)
        return FacetData(snapshot_revision(self._snapshot), snapshot_generation(self._snapshot), spec, cells)


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
        edges = self._value_display_unit.convert_value_to(
            edges, schema_value_unit(self._schema, self._unit_registry)
        )
        if np.any(np.diff(edges) <= 0):
            raise ValueError("histogram edges must be strictly increasing")
        return edges

    def _histogram_from_counts(
        self,
        edges: NDArray[Any],
        counts: NDArray[np.int64],
        group_keys: tuple[tuple[AxisValue, ...], ...] = ((),),
    ) -> HistogramData:
        """Speak already-counted canonical bins as one Histogram payload."""

        edges = np.asarray(edges)
        counts = np.asarray(counts, dtype=np.int64).reshape(len(group_keys), edges.size - 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        display_edges = schema_value_unit(self._schema, self._unit_registry).convert_value_to(
            edges, self._value_display_unit
        )
        display_centers = schema_value_unit(self._schema, self._unit_registry).convert_value_to(
            centers, self._value_display_unit
        )
        return HistogramData(
            revision=snapshot_revision(self._snapshot),
            generation=snapshot_generation(self._snapshot),
            edges=QuantityArray(
                edges,
                display_edges,
                schema_value_unit(self._schema, self._unit_registry),
                self._value_display_unit,
                (self._schema.value_schema.name or "value"),
            ),
            centers=QuantityArray(
                centers,
                display_centers,
                schema_value_unit(self._schema, self._unit_registry),
                self._value_display_unit,
                (self._schema.value_schema.name or "value"),
            ),
            counts=counts,
            group_keys=group_keys,
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
        if spec.facet is not None:
            self._resolve(spec.facet)
        cell = spec.cell
        if isinstance(cell, CurvePlot):
            self._validate_curve_shape(
                cell.x, () if cell.group is None else (cell.group,)
            )
        elif isinstance(cell, ImagePlot):
            self._validate_image_shape(cell.x, cell.y)
        elif isinstance(cell, HistogramPlot):
            if cell.group is not None:
                self._resolve(cell.group)
            for ref in cell.reduced:
                self._resolve(ref)
            if spec.facet is not None and spec.facet in cell.reduced:
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

        The used set comes from one code per physical carrier row, never one
        coordinate per dataset element.  This remains O(R), O(P), or O(Di)
        regardless of the cell payload size.
        """

        if spec.facet is None:
            return 1
        resolved = self._resolve(spec.facet)
        dimension = int(resolved.contract.dimension)
        size = int(schema_shape(self._schema)[dimension])
        stride = math.prod(schema_shape(self._schema)[dimension + 1 :])
        representatives = np.arange(size, dtype=np.int64) * stride
        return self._domain(spec.facet, representatives).size

    def facet(
        self, spec: FacetGridPlot, *, bins: int | Sequence[float] | None = None,
        uncertainty: bool = False, window: int = 1,
    ) -> FacetData:
        self.validate_facet(spec)
        cell = spec.cell
        if isinstance(cell, HistogramPlot):
            if uncertainty:
                raise ValueError("uncertainty is accepted only for Curve facet cells")
            if bins is None:
                raise DataViewError("histogram facet cells require explicit bins")
            return self._histogram_facet(spec, bins, window)
        if cell.reduction is Reduction.LAST:
            kept = tuple(ref for ref in (
                spec.facet, getattr(cell, "x", None), getattr(cell, "y", None),
                getattr(cell, "group", None), *(ref for ref, _value in spec.scope),
            ) if ref is not None)
            payload = self._last_view(keep=kept).facet(
                replace(spec, cell=replace(cell, reduction=Reduction.MEAN)),
                bins=bins, uncertainty=uncertainty, window=window,
            )
            return replace(payload, spec=spec)
        if bins is not None:
            raise ValueError("bins are accepted only for Histogram facet cells")
        if uncertainty and not isinstance(cell, CurvePlot):
            raise ValueError("uncertainty is accepted only for Curve facet cells")
        if spec.facet is None:
            payload = (
                self.curve(cell.x, group_by=(() if cell.group is None else (cell.group,)),
                           aggregation=cell.reduction, uncertainty=uncertainty)
                if isinstance(cell, CurvePlot)
                else self.image(cell.x, cell.y, aggregation=cell.reduction)
            )
            return FacetData(
                snapshot_revision(self._snapshot), snapshot_generation(self._snapshot), spec,
                (FacetCell(0, 1, 1, "Facet 1", payload),),
            )
        return self._factored_facet(spec, uncertainty)



    def _all_positions(self) -> NDArray[np.int64]:
        cached = self._positions_cache
        if cached is None:
            cached = np.arange(self.samples.value.canonical.size, dtype=np.int64)
            self._positions_cache = cached
        return cached


    def _domain(
        self,
        ref: AxisRef,
        positions: NDArray[np.int64],
    ) -> _Domain:
        whole = positions is self._positions_cache
        if whole:
            carried = self._domain_carry.get(ref)
            if carried is not None:
                return carried
        resolved = self._resolve(ref)
        # ``CoordinateArray`` keeps broadcast tensor views for renderers.
        # Grouping only needs the producer's small integer axis codes, not a
        # second full copy of the coordinate plane.
        cached_flat = self._flat_cache.get(ref)
        sparse = (
            cached_flat is None
            and positions.size < resolved.coordinate.canonical.size
        )
        if cached_flat is None and not sparse:
            # One copy, sealed in place: the plane is this owner's own,
            # made this instant, so there is nothing to isolate it from.
            cached_flat = np.asarray(
                resolved.coordinate.indices, dtype=np.int64
            ).flatten()
            cached_flat.setflags(write=False)
            self._flat_cache[ref] = cached_flat
        if sparse:
            selected_indices = np.asarray(resolved.coordinate.indices).flat[
                positions
            ]
        else:
            assert cached_flat is not None
            selected_indices = cached_flat[positions]
        # An all-finite declared domain (checked once, at domain size) makes
        # every element's coordinate valid by construction, so the
        # per-element canonical gather and isfinite pass -- two full-size
        # temporaries per axis, millions of elements on a camera facet --
        # carry no information.  Codes then come straight off the index plane.
        valid_local: NDArray[np.int64] | None = None
        domain_valid = _finite_coordinate(resolved.domain_canonical)
        if bool(domain_valid.all()):
            declared = selected_indices
        else:
            coordinate_valid = domain_valid[selected_indices]
            valid_local = np.flatnonzero(coordinate_valid)
            if valid_local.size == 0:
                codes = np.full(positions.shape, -1, dtype=np.int64)
                codes.setflags(write=False)
                empty = _readonly(np.empty(0))
                return _Domain(empty, empty, codes, tuple)
            declared = selected_indices[valid_local]
        # The domain is declared, so its size is axis-sized; a bincount +
        # remap finds the used indices in O(carrier rows), with no coordinate
        # sorting or value-derived identity.
        used_indices = np.flatnonzero(
            np.bincount(declared, minlength=resolved.domain_canonical.size)
        )
        remap = np.full(resolved.domain_canonical.size, -1, dtype=np.int64)
        remap[used_indices] = np.arange(used_indices.size, dtype=np.int64)
        inverse = remap[declared]
        canonical_values = resolved.domain_canonical[used_indices]
        display_values = resolved.domain_display[used_indices]
        # Both gathers are ours. Seal them before the immutable domain wrappers
        # so its lazy labels do not retain another pair of writable copies.
        canonical_values.setflags(write=False)
        display_values.setflags(write=False)
        if valid_local is None:
            codes = inverse
        else:
            codes = np.full(positions.shape, -1, dtype=np.int64)
            codes[valid_local] = inverse
        codes.setflags(write=False)

        # A retained domain belongs to resolved: do not capture that owner in
        # its lazy callback and create a cycle holding the coordinate arrays.
        declared_labels = resolved.coordinate_labels
        axis_label = resolved.coordinate.label
        display_unit = resolved.coordinate.display_unit

        def build_values() -> tuple[AxisValue, ...]:
            indices: tuple[int | None, ...] = tuple(
                int(index) for index in used_indices
            )
            coordinate_labels = (
                (None,) * len(indices)
                if declared_labels is None
                else tuple(
                    declared_labels[int(index)]
                    for index in used_indices
                )
            )
            return tuple(
                AxisValue(
                    ref=ref,
                    index=index,
                    canonical=_python_scalar(canonical_value),
                    display=_python_scalar(display_value),
                    label=_axis_value_label(
                        axis_label,
                        display_value,
                        display_unit,
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
            self._domain_carry[ref] = domain
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
    """Bin edges for one histogram pool; integer-valued samples get integer bins.

    ``values`` is EVERY sample the histogram will bin.  Whether they are all
    whole numbers is a fact about all of them and no prefix can prove it:
    the same multiset with its two fractions stored last once binned as a
    single integer bin, and stored first as the ten bins asked for.  A
    caller that already made its own pass over the pool hands the facts to
    :func:`histogram_edges`, which is where the edges are decided.
    """

    flat = np.asarray(values).reshape(-1)
    if flat.dtype.kind == "f":
        flat = flat[np.isfinite(flat)]
    if limits is not None:
        low, high = (float(value) for value in limits)
    elif flat.size:
        low, high = float(np.min(flat)), float(np.max(flat))
    else:
        low, high = 0.0, 1.0
    integral = flat.dtype.kind in "iub" or (
        bool(flat.size) and bool(np.all(flat == np.floor(flat)))
    )
    return histogram_edges(low, high, bins, integral=integral)


def histogram_edges(
    low: float, high: float, bins: int, *, integral: bool
) -> NDArray[np.float64]:
    """The edges of ``bins`` bins over ``[low, high]``.

    Equal-width float bins over integer-valued samples alias: a non-integer
    bin width leaves some bins containing no representable value, which shows
    up as structural zero-count holes in the middle of the distribution.
    Integer-valued data therefore bins with an integer width on half-open
    ``k - 0.5`` boundaries (the bin count may shrink below the request when
    the value range is narrower); everything else keeps NumPy's equal-width
    edges over the same range.  ``integral`` is the caller's proof that
    every binned sample is a whole number, and only a pass over all of them
    can give it: an integer dtype proves it by itself, a float pool by
    being checked to its end.
    """

    bins = max(1, int(bins))
    low, high = float(low), float(high)
    if high <= low:
        high = low + 1.0
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
    if source.dtype.kind not in "biu" or _aligned_integer_bins(edges) is None:
        return None
    int64 = np.iinfo(np.int64)

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
    return _counts_from_frequency(low, frequency, edges)


def _aligned_integer_bins(edges: NDArray[Any]) -> tuple[int, int] | None:
    """``(first, width)`` of integer-aligned uniform edges, or None.

    The edges this library makes for integer samples sit on half-open
    ``k - 0.5`` boundaries with an integer width; these are the only edges
    a frequency table can be summed into without asking which bin a value
    on a boundary belongs to.
    """

    if edges.size < 2:
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
    return first, width


def _counts_from_frequency(
    offset: int,
    frequency: NDArray[np.int64],
    edges: NDArray[Any],
) -> NDArray[np.int64] | None:
    """Sum a frequency table (``frequency[v - offset]`` samples equal ``v``) into aligned integer bins."""

    aligned = _aligned_integer_bins(edges)
    if aligned is None:
        return None
    first, width = aligned
    counts = np.zeros(edges.size - 1, dtype=np.int64)
    if not frequency.size:
        return counts
    low = int(offset)
    high = low + int(frequency.size) - 1
    for index in range(counts.size):
        start = max(first + index * width, low) - low
        stop = min(first + (index + 1) * width, high + 1) - low
        if stop > start:
            counts[index] = np.sum(frequency[start:stop], dtype=np.int64)
    return counts


def _frequency_span(dtype: np.dtype, selected: NDArray[Any]) -> tuple[int, int] | None:
    """``(offset, size)`` of the table one integer pool is counted into.

    A dtype of two bytes or fewer gets its whole range, so the table never
    has to grow whatever a later shot brings; a wider one gets the span the
    pool actually uses, while that stays within the limit.
    """

    if dtype.kind not in "iu":
        return None
    info = np.iinfo(dtype)
    if dtype.itemsize <= 2:
        return int(info.min), int(info.max) - int(info.min) + 1
    if not selected.size:
        return 0, 1
    low = int(np.min(selected))
    high = int(np.max(selected))
    if not _int64_holds(low, high) or high - low + 1 > _FREQUENCY_LEVEL_LIMIT:
        return None
    return low, high - low + 1


def _int64_holds(low: int, high: int) -> bool:
    """Whether both levels are int64 values.

    The table is addressed by an int64 difference from its offset, so a
    level past int64 -- the upper half of uint64 -- has no place in it
    and the histogram counts the pool the ordinary way instead.
    """

    int64 = np.iinfo(np.int64)
    return int64.min <= low and high <= int64.max


def _apply_shot(
    table: NDArray[np.int64],
    offset: int,
    values: NDArray[Any],
    valid: NDArray[np.bool_],
    inner: int,
    position: int,
    sign: int,
) -> tuple[NDArray[np.int64], int] | None:
    """Add (``sign`` 1) or remove (-1) one shot's valid samples from a table.

    The table is grown when a shot brings a value outside it -- only a wide
    dtype can -- and given up when growing it would pass the limit.
    """

    rows = slice(position * inner, (position + 1) * inner)
    chunk = values[:, rows]
    if _stride_zero_all_true(valid):
        selected = chunk.reshape(-1)
    else:
        selected = chunk[valid[:, rows]]
    if not selected.size:
        return table, offset
    low = int(np.min(selected))
    high = int(np.max(selected))
    if not _int64_holds(low, high):
        return None
    if low < offset or high >= offset + table.size:
        new_offset = min(offset, low)
        new_end = max(offset + table.size, high + 1)
        if new_end - new_offset > _FREQUENCY_LEVEL_LIMIT:
            return None
        grown = np.zeros(new_end - new_offset, dtype=np.int64)
        grown[offset - new_offset : offset - new_offset + table.size] = table
        table, offset = grown, new_offset
    delta = np.bincount(np.subtract(selected, offset, dtype=np.int64), minlength=table.size)
    if sign > 0:
        table += delta
    else:
        table -= delta
    return table, offset


def _uniform_edges(edges: NDArray[Any]) -> NDArray[np.float64] | None:
    """Use the actual bin boundaries, widening without moving any boundary."""
    edges = np.asarray(edges)
    if edges.ndim != 1 or edges.size < 2:
        return None
    first, last = float(edges[0]), float(edges[-1])
    if not (math.isfinite(first) and math.isfinite(last) and math.isfinite(last - first)) or last <= first:
        return None
    dtype = np.float32 if edges.dtype == np.dtype(np.float32) else np.float64
    produced = np.linspace(first, last, edges.size, dtype=dtype)
    if not np.array_equal(edges, produced) or bool(np.any(edges[:-1] >= edges[1:])):
        return None
    # Float32 subtraction can overflow even when both finite endpoints are
    # valid. Every float32 boundary is represented EXACTLY in float64; the
    # samples are tested against these same edges, not a regenerated grid.
    return np.asarray(edges, dtype=np.float64)


def _uniform_counts(
    values: NDArray[Any],
    valid: NDArray[np.bool_] | None,
    edges: NDArray[Any],
) -> NDArray[np.int64] | None:
    """A single distribution uses the same binning owner as grouped data."""
    source = np.asarray(values)
    usable = np.broadcast_to(True, source.shape) if valid is None else valid
    compiled = _histogram_kernel_counts(
        source, usable, np.empty(0, dtype=np.int64), 0, 1, edges,
    )
    if compiled is not None:
        return compiled[0]
    if _uniform_edges(edges) is None:
        return None
    selected = source.reshape(-1) if valid is None or bool(np.all(valid)) else source[valid]
    counts, produced = np.histogram(
        selected, bins=edges.size - 1, range=(float(edges[0]), float(edges[-1])),
    )
    return counts if np.array_equal(produced, edges) else None


def _histogram_kernel_counts(
    values: NDArray[Any],
    valid: NDArray[np.bool_],
    facet_codes: NDArray[np.int64],
    facet_dimension: int,
    facet_count: int,
    edges: NDArray[Any],
) -> NDArray[np.int64] | None:
    """Count coordinate-owned distributions in one shared compiled pass."""

    from . import _raster_kernels as kernels

    if not kernels.engaged():
        return None
    source = np.asarray(values)
    if source.dtype.kind not in "biuf" or source.dtype == np.dtype(np.float16):
        return None
    produced = _uniform_edges(edges)
    if produced is None:
        return None
    count = int(produced.size) - 1
    flat = kernels.readable(source).reshape(-1)
    use_valid = not (
        _stride_zero_all_true(valid) or bool(np.asarray(valid).all())
    )
    # Sealed either way: the unused placeholder must carry the same
    # mutability as a real mask, or the kernel compiles twice.
    marks = kernels.readable(
        np.asarray(np.broadcast_to(valid, source.shape), dtype=np.bool_).reshape(-1)
        if use_valid
        else np.zeros(1, dtype=np.bool_)
    )
    stride = 1
    for size in source.shape[facet_dimension + 1:]:
        stride *= int(size)
    threads = kernels.histogram_threads()
    partials = np.empty((threads, facet_count, count), dtype=np.int64)
    counted = np.empty((facet_count, count), dtype=np.int64)
    kernels.uniform_histogram(
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


def _axis_aggregate(
    values: NDArray[Any], valid: NDArray[np.bool_],
    codes: tuple[NDArray[np.int64], ...], dimensions: tuple[int, ...],
    domain_sizes: tuple[int, ...], aggregation: Reduction, *,
    offsets: NDArray[np.float64] | None = None,
) -> tuple:
    """One axis-code reduction, with the same NumPy reference for every caller."""
    compiled = _axis_kernel_aggregate(
        values, valid, codes, dimensions, domain_sizes, aggregation, offsets=offsets,
    )
    if compiled is not None:
        return compiled
    shape = values.shape
    combined: Any = np.int64(0)
    admitted: Any = np.bool_(True)
    for axis_codes, dimension, size in zip(codes, dimensions, domain_sizes):
        spread = [1] * len(shape)
        spread[dimension] = axis_codes.size
        placed = axis_codes.reshape(spread)
        combined = combined * size + np.where(placed >= 0, placed, 0)
        admitted = admitted & (placed >= 0)
    full_codes = np.broadcast_to(combined, shape).reshape(-1)
    full_admitted = np.broadcast_to(admitted, shape).reshape(-1)
    usable = np.broadcast_to(valid, shape).reshape(-1) & full_admitted
    bucket_count = math.prod(domain_sizes)
    selected = np.asarray(values).reshape(-1)
    present = np.bincount(full_codes[full_admitted], minlength=bucket_count) > 0
    if offsets is not None:
        first, second = _centred_moments_by_codes(selected, offsets, usable, full_codes, bucket_count)
        counts = np.bincount(full_codes[usable], minlength=bucket_count)
        return first, second, counts, present
    reduced, counts = _aggregate_by_codes(
        selected, usable, full_codes, bucket_count, aggregation,
    )
    return reduced, counts, present


def _axis_kernel_aggregate(
    values: NDArray[Any],
    valid: NDArray[np.bool_],
    codes: tuple[NDArray[np.int64], ...],
    dimensions: tuple[int, ...],
    domain_sizes: tuple[int, ...],
    aggregation: Reduction,
    *,
    offsets: NDArray[np.float64] | None = None,
) -> (
    tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.bool_]]
    | tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.bool_],
    ]
    | None
):
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
    second = np.empty(
        bucket_count if offsets is not None else 1, dtype=np.float64
    )
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
        second,
        counts,
        presence,
    )
    if offsets is not None:
        return out, second, counts, presence
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

    A floating stack's MEAN and SUM accumulate in float64 whatever its
    dtype, as every other reduction in this module does: three float32
    samples 1e8, 1, -1e8 sum to 1 in float64 and to 0 in float32, and the
    same view answered both depending on which layout the data took.
    Integers keep NumPy's own exact accumulators.
    """

    identity = _leading_identity(values, usable)
    if identity is not None:
        value, valid = identity
        return value, np.asarray(valid, dtype=np.int64)
    wide = np.float64 if values.dtype.kind == "f" else None
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
            result = np.mean(values, axis=0, dtype=wide)
        elif aggregation is Reduction.SUM:
            result = np.sum(values, axis=0, dtype=wide)
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
            result = np.mean(values, axis=0, where=usable, dtype=wide)
        elif aggregation is Reduction.SUM:
            result = np.sum(values, axis=0, where=usable, initial=0, dtype=wide)
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


#: How many samples one centring pass shifts at a time.  Small enough that
#: the scratch buffer is a rounding error against a megapixel pool, large
#: enough that the per-block overhead disappears into the arithmetic.
_CENTRING_BLOCK = 1 << 17


def _centred_moment_totals(
    flat: NDArray[Any],
    offset: float,
    where: NDArray[np.bool_] | None = None,
) -> tuple[float, float]:
    """Sums of ``d`` and ``d**2`` without materialising all of ``d``.

    The whole-revision pool is the one place where that copy is a real
    cost: it is every sample of the revision, and the rolling path has a
    memory budget precisely so a per-shot reduction cannot double the
    footprint of the data it reduces.  Blocking through one small buffer
    keeps both the conditioning and the budget.
    """

    first = 0.0
    second = 0.0
    size = int(flat.size)
    if size == 0:
        return first, second
    scratch = np.empty(min(_CENTRING_BLOCK, size), dtype=np.float64)
    for start in range(0, size, _CENTRING_BLOCK):
        stop = min(start + _CENTRING_BLOCK, size)
        piece = scratch[: stop - start]
        np.subtract(flat[start:stop], offset, out=piece)
        if where is None:
            first += float(np.sum(piece, dtype=np.float64))
            second += float(np.dot(piece, piece))
        else:
            first += float(
                np.sum(piece, where=where[start:stop], dtype=np.float64)
            )
            np.square(piece, out=piece)
            second += float(
                np.sum(piece, where=where[start:stop], dtype=np.float64)
            )
    return first, second


def _centred_moment_sums(
    plane: Any,
    offsets: Any,
    usable: Any | None,
    kept: list[int],
    shape: tuple[int, ...],
) -> Any:
    """Per kept position, sums of ``d`` and ``d**2``, or ``None``.

    THE KEPT AXES ARE ONE BLOCK OR THEY ARE NOTHING.  A reduction that
    keeps axis 1 and a group axis keeps a run of adjacent axes whenever a
    three-dimensional view of the tensor exists at all, and where it does
    the view costs nothing -- the array is C-contiguous, so the reshape is
    a reshape and not a copy.  A size-1 axis inside the run is part of the
    block by layout: it contributes no positions and no strides, and
    reading the span literally sent every facet curve with such an axis
    down the einsum fallback -- a full centred copy of the tensor per
    revision.  Where no block exists, the caller's einsum is still the
    answer.
    """

    if not kept:
        return None
    span = list(range(kept[0], kept[-1] + 1))
    if any(
        axis not in kept and int(shape[axis]) != 1 for axis in span
    ):
        return None
    kept_shape = tuple(int(shape[axis]) for axis in kept)
    kept = span
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

    centres = np.asarray(offsets, dtype=np.float64)
    if centres.shape != kept_shape:
        return None
    summed = kernels.masked_centred_moment_sums(
        array.reshape(outer, keep, inner), centres.reshape(-1), marks
    )
    if summed is None:
        return None
    # Back to the CALLER'S kept axes: the span may carry size-1 padding
    # axes that exist in the layout but not in the caller's vocabulary.
    return tuple(moment.reshape(kept_shape) for moment in summed)


def _centred_moments_by_codes(
    plane: Any,
    offsets: NDArray[np.float64],
    usable: NDArray[np.bool_],
    codes: NDArray[np.int64],
    bucket_count: int,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Centred first and second bucket moments for an irregular mapping."""

    selected = np.asarray(plane, dtype=np.float64).reshape(-1)
    bucket_codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    centres = np.asarray(offsets, dtype=np.float64).reshape(-1)
    delta = selected - centres[np.maximum(bucket_codes, 0)]
    first, _ = _aggregate_by_codes(
        delta, usable, bucket_codes, bucket_count, Reduction.MEAN
    )
    np.square(delta, out=delta)
    second, _ = _aggregate_by_codes(
        delta, usable, bucket_codes, bucket_count, Reduction.MEAN
    )
    return first, second


def _sem_of_mean(
    means: NDArray[np.float64],
    counts: NDArray[np.int64],
    samples: NDArray[Any],
    sigma: NDArray[Any] | None,
    centred_moments: Callable[
        [Any, NDArray[np.float64]], tuple[Any, Any] | None
    ],
) -> NDArray[np.float64] | None:
    """The standard error of a mean, formed the ONE way this repo forms it.

    Every plot kind that draws a band arrives here.  What differs between
    them is only how a bucket is summed -- a strided tensor reduction, a
    bincount over codes, an einsum, a dot -- so that is all a caller brings.
    ``centred_moments(plane, offsets)`` returns, per bucket, both ``E[d]``
    and ``E[d**2]`` over exactly the samples that formed the mean, where
    every bucket uses its own first-pass mean as ``offset``.

    What does NOT differ, and therefore lives here:

      * every bucket is centred on its own mean.  One scalar reference for
        a whole series still loses a small bucket's spread when another
        bucket is many orders of magnitude away;

      * the centred first moment is measured, not assumed zero.  The
        first-pass mean is rounded, so the stable identity is
        ``E[d**2] - E[d]**2``;

      * the samples' own sigma is offered to the estimator, which uses it
        only where the scatter cannot speak.  A sigma is already a
        difference about zero, so it is squared about zero -- passing the
        value reference there would be a category error.

    One public owner keeps all projection paths on that same two-pass rule.
    """

    if sigma is None and not np.any(counts > 1):
        # Nothing to estimate FROM: no bucket has a second member, so the
        # scatter is undefined everywhere (n - 1 = 0), and no sample
        # states its own error.  The moments below would only spell the
        # same all-NaN answer at two full passes over every value --
        # which an indexed rolling history, one member per bucket by
        # construction, paid on every drawn frame.
        sem = np.full(np.shape(means), np.nan, dtype=np.float64)
        return sem
    centres = np.asarray(means, dtype=np.float64)
    moments = centred_moments(samples, centres)
    if moments is None:
        return None
    first, second = moments
    mean_delta = np.asarray(first, dtype=np.float64)
    mean_delta_square = np.asarray(second, dtype=np.float64)
    mean_sigma_square = None
    if sigma is not None:
        sigma_moments = centred_moments(
            sigma, np.zeros(np.shape(centres), dtype=np.float64)
        )
        if sigma_moments is None:
            return None
        mean_sigma_square = np.asarray(sigma_moments[1], dtype=np.float64)
    return _sem_from_moments(
        mean_delta, mean_delta_square, counts, mean_sigma_square
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
        squared_mean = np.square(mean)
        raw_spread = mean_of_squares - squared_mean
        # The moment reduction sums ``n`` squares, divides once, squares the
        # mean and subtracts.  A constant bucket can therefore leave a small
        # POSITIVE residual as well as a negative one when the two equal
        # moments round in opposite directions.  Clipping only below zero
        # turns that last bit into a fake SEM and hence an enormous fit
        # weight.  Values inside the first-order forward-error bound are
        # numerically indistinguishable from zero; this scales with the
        # arithmetic that formed the moment, not with the observed data.
        roundoff = np.abs(mean_of_squares)
        roundoff += squared_mean
        roundoff *= np.finfo(np.float64).eps
        # ``squared_mean`` is no longer needed; reuse it for the operation
        # count so this common large-tensor path retains the original three
        # temporary planes instead of allocating a fourth and fifth.
        np.maximum(n, 1.0, out=squared_mean)
        squared_mean += 3.0
        roundoff *= squared_mean
        np.copyto(raw_spread, 0.0, where=raw_spread <= roundoff)
        np.maximum(raw_spread, 0.0, out=raw_spread)
        spread = raw_spread
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
    if bucket_count == codes.size and np.array_equal(
        codes, np.arange(bucket_count, dtype=codes.dtype)
    ):
        # Identity layout: every sample IS its own bucket, so every
        # reduction of one member is the member.  The indexed rolling
        # projection reduces (shots x groups) cells laid out exactly this
        # way on every drawn frame; the scatter below only re-derived a
        # masked copy of the input.
        output = np.where(
            usable, values.astype(output_dtype, copy=False), np.nan
        )
        counts = usable.astype(np.int64)
        output.setflags(write=False)
        counts.setflags(write=False)
        return output, counts
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
    "RollingHistory",
    "RollingShot",
    "SelectionSubject",
    "QuantityArray",
    "SampleProjection",
]
