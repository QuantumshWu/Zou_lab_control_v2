"""Value and materialized dataset schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from math import prod
from typing import Any

import numpy as np
from .validation import canonical_text, integer, positive_integer

from ._arrays import canonical_dtype, immutable_array
from .axis import (
    AxisId,
    AxisSpec,
    REPEAT,
    SCALAR,
    SCALAR_AXIS,
)
from .validity import ValidityContract, ValidityMode


def _unique_axis_ids(axes: tuple[AxisSpec, ...], *, context: str) -> None:
    ids = tuple(axis.axis_id for axis in axes)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{context} axis ids must be unique")


def _ordered_subset(candidate: tuple[AxisId, ...], available: tuple[AxisId, ...]) -> bool:
    positions = []
    for axis_id in candidate:
        try:
            positions.append(available.index(axis_id))
        except ValueError:
            return False
    return positions == sorted(positions)


@dataclass(frozen=True, eq=False)
class DomainSpec:
    """One physical domain and its logical named axes.

    Repeat and Point use explicit, axis-major ``axis_codes`` over their flat
    carrier. Cell-data leaves ``axis_codes`` as ``None``: each axis then maps
    by position to one dense physical dimension. Dense codes are only the
    corresponding one-dimensional identity, never a broadcast pixel plane.
    A range and optional inner/outer repeats retain a declared regular mapping
    without a carrier-sized code array. ``codes`` expands it only on request.
    """

    shape: tuple[int, ...]
    axes: tuple[AxisSpec, ...] = ()
    axis_codes: tuple[np.ndarray | range, ...] | None = None
    #: Each base code is repeated inner times, then that vector outer times.
    #: None is the ordinary, already complete mapping: (1, 1) for every axis.
    axis_code_repeats: tuple[tuple[int, int], ...] | None = None
    _codes: Any = field(init=False, repr=False, compare=False, default=None)
    #: Filled on first request, per row asked: the live commit asks the
    #: same domain the same question for every event it publishes.
    _coordinate_counts: Any = field(init=False, repr=False, compare=False, default=None)
    _row_groups: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        shape = tuple(
            positive_integer(size, "domain dimension size") for size in self.shape
        )
        if not shape:
            raise ValueError("domain shape must contain at least one dimension")
        axes = tuple(self.axes)
        if any(not isinstance(axis, AxisSpec) for axis in axes):
            raise TypeError("domain axes must contain AxisSpec values")
        _unique_axis_ids(axes, context="domain")
        authored_codes = self.axis_codes
        for axis in axes:
            if axis.coordinate_of is None:
                continue
            primary = next((item for item in axes if item.axis_id == axis.coordinate_of), None)
            if primary is None or primary.coordinate_of is not None or axis.size != primary.size:
                raise ValueError("alternative coordinates need one same-sized primary axis in their domain")
            if authored_codes is None:
                raise ValueError("alternative coordinates require an explicitly mapped domain")
        normalized_codes: tuple[np.ndarray | range, ...] | None
        repeats = None
        if authored_codes is None:
            if self.axis_code_repeats is not None:
                raise ValueError("code repeats require an explicitly mapped domain")
            if len(axes) != len(shape) or any(
                axis.size != size for axis, size in zip(axes, shape, strict=True)
            ):
                raise ValueError(
                    "a dense domain needs one same-sized axis per physical dimension"
                )
            normalized_codes = None
        else:
            authored_codes = tuple(authored_codes)
            if len(shape) != 1:
                raise ValueError("an explicitly mapped domain must have one physical dimension")
            if len(authored_codes) != len(axes):
                raise ValueError("axis_codes must contain one vector per axis")
            carrier_size = prod(shape)
            authored_repeats = ((1, 1),) * len(axes) if self.axis_code_repeats is None else tuple(self.axis_code_repeats)
            if len(authored_repeats) != len(axes):
                raise ValueError("axis_code_repeats must contain one pair per axis")
            factors = []
            for pair in authored_repeats:
                if len(pair) != 2:
                    raise ValueError("each axis code repeat needs inner and outer counts")
                factors.append(tuple(positive_integer(value, "axis code repeat") for value in pair))
            normalized: list[np.ndarray | range] = []
            for axis, entries, (inner, outer) in zip(axes, authored_codes, factors, strict=True):
                if isinstance(entries, range):
                    count = len(entries)
                    if count * inner * outer != carrier_size:
                        raise ValueError("each axis code vector must match domain size")
                    low, high = min(entries[0], entries[-1]), max(entries[0], entries[-1])
                    if low < 0 or high >= axis.size or high > np.iinfo(np.int64).max:
                        raise ValueError("axis code is outside its coordinate domain")
                    step = entries.step if count > 1 else 1
                    normalized.append(range(entries.start, entries.start + step * count, step))
                else:
                    array = np.asarray(entries)
                    if array.ndim != 1 or array.size * inner * outer != carrier_size:
                        raise ValueError("each axis code vector must match domain size")
                    if not issubclass(array.dtype.type, np.integer):
                        raise TypeError("axis codes must be integers")
                    if bool(np.any(array < 0)) or bool(np.any(array >= axis.size)) or bool(np.any(array > np.iinfo(np.int64).max)):
                        raise ValueError("axis code is outside its coordinate domain")
                    normalized.append(immutable_array(
                        np.asarray(array, dtype="<i8"), dtype=np.dtype("<i8"), shape=array.shape,
                    ))
            for index, axis in enumerate(axes):
                if axis.coordinate_of is not None:
                    primary_index = next(i for i, item in enumerate(axes) if item.axis_id == axis.coordinate_of)
                    left, right = normalized[index], normalized[primary_index]
                    same_base = left == right if isinstance(left, range) and isinstance(right, range) else np.array_equal(left, right)
                    if factors[index] != factors[primary_index] or not same_base:
                        mappings = [
                            np.tile(np.repeat(normalized[position], factors[position][0]), factors[position][1])
                            for position in (index, primary_index)
                        ]
                        if not np.array_equal(*mappings):
                            raise ValueError("alternative coordinates must name the same physical rows")
                    normalized[index] = normalized[primary_index]
                    factors[index] = factors[primary_index]
            normalized_codes = tuple(normalized)
            repeats = tuple(factors) if any(pair != (1, 1) for pair in factors) else None
            if not axes and carrier_size != 1:
                raise ValueError("a mapped domain with multiple rows needs a named axis")

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "axis_codes", normalized_codes)
        object.__setattr__(self, "axis_code_repeats", repeats)
        object.__setattr__(self, "_codes", [None] * len(axes))
        object.__setattr__(self, "_coordinate_counts", {})
        object.__setattr__(self, "_row_groups", None)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, DomainSpec):
            return NotImplemented
        if self.shape != other.shape or self.axes != other.axes or self.axis_code_repeats != other.axis_code_repeats:
            return False
        if self.axis_codes is None or other.axis_codes is None:
            return self.axis_codes is other.axis_codes
        return all(
            left == right if isinstance(left, range) and isinstance(right, range)
            else False if isinstance(left, range) or isinstance(right, range)
            else np.array_equal(left, right)
            for left, right in zip(self.axis_codes, other.axis_codes, strict=True)
        )

    def __hash__(self) -> int:
        return hash((self.shape, self.axes, None if self.axis_codes is None else tuple(
            codes if isinstance(codes, range) else tuple(codes) for codes in self.axis_codes
        ), self.axis_code_repeats))

    @cached_property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes if axis.coordinate_of is None)

    def axis(self, axis_id: AxisId) -> AxisSpec:
        if not isinstance(axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        for axis in self.axes:
            if axis.axis_id == axis_id:
                return axis
        raise KeyError(axis_id)

    def code_mapping(self, axis_id: AxisId) -> tuple[np.ndarray | range, int, int]:
        """The stored codes and their declared inner/outer repetition."""

        axis = self.axis(axis_id)
        if self.axis_codes is None:
            return range(axis.size), 1, 1
        index = self.axes.index(axis)
        inner, outer = (1, 1) if self.axis_code_repeats is None else self.axis_code_repeats[index]
        return self.axis_codes[index], inner, outer

    def codes(self, axis_id: AxisId, rows: np.ndarray | None = None) -> np.ndarray:
        """Read requested rows, or expand and cache the complete vector once."""

        axis = self.axis(axis_id)
        if axis.coordinate_of is not None:
            return self.codes(axis.coordinate_of, rows)
        index = self.axes.index(axis)
        if rows is None:
            cached = self._codes[index]
            if cached is not None:
                return cached
        base, inner, outer = self.code_mapping(axis_id)
        if rows is not None:
            rows = np.asarray(rows)
            if not issubclass(rows.dtype.type, np.integer):
                raise TypeError("domain rows must be integers")
            if bool(np.any(rows < 0)) or bool(np.any(rows >= len(base) * inner * outer)):
                raise IndexError("domain row is outside its physical dimension")
            positions = (rows.astype(np.int64, copy=False) // inner) % len(base)
            array = (base.start + positions * base.step if isinstance(base, range)
                     else base[positions])
            return immutable_array(array, dtype=np.dtype("<i8"), shape=array.shape)
        array = np.arange(base.start, base.stop, base.step, dtype="<i8") if isinstance(base, range) else base
        if inner != 1:
            array = np.repeat(array, inner)
        if outer != 1:
            array = np.tile(array, outer)
        cached = immutable_array(array, dtype=np.dtype("<i8"), shape=array.shape)
        self._codes[index] = cached
        return cached

    def code_at(self, axis_id: AxisId, row: int) -> int:
        """Read one physical row's code directly from its declared mapping."""

        base, inner, outer = self.code_mapping(axis_id)
        row = integer(row, "domain row", minimum=0)
        if row >= len(base) * inner * outer:
            raise IndexError("domain row is outside its physical dimension")
        return int(base[(row // inner) % len(base)])

    def coordinate_axis(self, axis_id: AxisId) -> AxisSpec:
        """The physical axis whose positions this coordinate names."""
        axis = self.axis(axis_id)
        return axis if axis.coordinate_of is None else self.axis(axis.coordinate_of)

    def coordinate_axes(self, axis_id: AxisId) -> tuple[AxisSpec, ...]:
        primary = self.coordinate_axis(axis_id)
        return (primary,) + tuple(axis for axis in self.axes if axis.coordinate_of == primary.axis_id)

    def coordinate_counts(self, current_row: int = -1) -> tuple[int, ...]:
        """Distinct coordinates along each axis, the other axes held at ``current_row``.

        With every carrier row present this is a property of the domain and
        the row, so it is computed once per row and kept: a monitor asks it
        of one domain at its last row for every event it ever publishes.
        """

        if not self.axes:
            return ()
        row = int(current_row) % self.size
        counts = self._coordinate_counts.get(row)
        if counts is None:
            result: list[int] = []
            primary_ids = tuple(axis.coordinate_of or axis.axis_id for axis in self.axes)
            codes = tuple(self.codes(axis.axis_id) for axis in self.axes)
            for target, target_codes in enumerate(codes):
                rows = np.ones(self.size, dtype=bool)
                for index, other_codes in enumerate(codes):
                    if primary_ids[index] != primary_ids[target]:
                        rows &= other_codes == other_codes[row]
                result.append(int(np.unique(target_codes[rows]).size))
            counts = tuple(result)
            self._coordinate_counts[row] = counts
        return counts

    def coordinate_rows(self, current_row: int) -> slice | np.ndarray:
        """Physical rows at this exact logical coordinate; topology only."""
        if self.axis_codes is None:
            raise ValueError("coordinate rows require a mapped domain")
        row = int(current_row) % self.size
        groups = self._row_groups
        if groups is None:
            codes = tuple(self.codes(axis.axis_id) for axis in self.axes if axis.coordinate_of is None)
            groups = ()
            if codes and self.size > 1:
                order = np.lexsort(codes[::-1])
                changed = np.zeros(self.size, dtype=bool)
                changed[0] = True
                for code in codes:
                    ordered = code[order]
                    changed[1:] |= ordered[1:] != ordered[:-1]
                if not bool(np.all(changed)):
                    starts = np.append(np.flatnonzero(changed), self.size)
                    lookup = np.empty(self.size, dtype=np.intp)
                    lookup[order] = np.cumsum(changed) - 1
                    for array in (order, starts, lookup):
                        array.flags.writeable = False
                    groups = order, starts, lookup
            object.__setattr__(self, "_row_groups", groups)
        if not groups:
            return slice(row, row + 1)
        order, starts, lookup = groups
        group = lookup[row]
        return order[starts[group]:starts[group + 1]]

    def __reduce__(self):
        return DomainSpec, (self.shape, self.axes, self.axis_codes, self.axis_code_repeats)

    def physical_dimension(self, axis_id: AxisId) -> int:
        """The domain-local physical dimension carrying one logical axis."""

        axis = self.axis(axis_id)
        return self.axes.index(axis) if self.axis_codes is None else 0


SCALAR_DOMAIN = DomainSpec((1,), (SCALAR_AXIS,))


#: Why a schema has a digest at all, so this is not re-litigated.
#:
#: ``DatasetRevisionRef`` carries a schema fingerprint across runtime and file
#: boundaries.  Comparing it with the decoded schema is what prevents a
#: revision identity from being paired with a different shape or axis contract.
#: Python's ``hash()`` cannot do that: it is randomised per process, so it would
#: agree today and disagree after a restart -- silently.
#:
#: Everything else compares two schemas that are both in hand, and for that
#: ``==`` is exact and fifteen times cheaper.  The digest is for the crossing,
#: not for the comparison.
#:
#: BLAKE2b-128 is the existing persisted format.  Changing it would make
#: revision references and archives disagree with the schemas they name.


@dataclass(frozen=True)
class ValueSchema:
    validity_contract: ValidityContract
    dtype: np.dtype
    value_unit: str | None = None
    name: str | None = None
    #: Cached on first request.  Computed eagerly it cost 23 us per schema,
    #: paid by every intermediate schema construction that never names it.
    _fingerprint: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.validity_contract, ValidityContract):
            raise TypeError("validity_contract must be ValidityContract")
        object.__setattr__(self, "dtype", canonical_dtype(self.dtype))
        if self.value_unit is not None:
            canonical_text(self.value_unit, "value_unit")
        if self.name is not None:
            canonical_text(self.name, "value name")
        object.__setattr__(self, "_fingerprint", None)

    @classmethod
    def scalar(
        cls,
        dtype: np.dtype,
        value_unit: str | None = None,
        *,
        name: str | None = None,
    ) -> "ValueSchema":
        return cls(
            ValidityContract.value(),
            dtype,
            value_unit,
            name,
        )

    @property
    def fingerprint(self) -> str:
        """This schema's canonical name, computed once, on request."""

        if self._fingerprint is None:
            from .codec import value_schema_fingerprint

            object.__setattr__(self, "_fingerprint", value_schema_fingerprint(self))
        return self._fingerprint


@dataclass(frozen=True)
class DatasetSchema:
    repeat_domain: DomainSpec
    point_domain: DomainSpec
    cell_domain: DomainSpec
    value_schema: ValueSchema
    #: Cached on first request.  Computed eagerly it cost 23 us per schema,
    #: paid by every intermediate schema construction that never names it.
    _fingerprint: str | None = field(init=False, repr=False, compare=False, default=None)
    _structure_fingerprint: str | None = field(
        init=False, repr=False, compare=False, default=None
    )
    #: Cached on first request, like the fingerprint beside it.  Building
    #: it walks stable axis metadata that cannot change with a publication.
    _axis_catalog: tuple | None = field(
        init=False, repr=False, compare=False, default=None
    )
    #: The indexed-history layout, read once per schema by
    #: ``zlc_data.snapshot_projection.indexed_history_layout`` and cached
    #: here like the catalog: every consumer of a Runtime history -- the
    #: window mask, the shot codes, the compatibility gate, the title's
    #: shot count -- reads this one object instead of walking the rows.
    _indexed_layout: Any = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.repeat_domain, DomainSpec):
            raise TypeError("repeat_domain must be DomainSpec")
        if self.repeat_domain.axis_codes is None or len(self.repeat_domain.shape) != 1:
            raise ValueError("repeat_domain must be an explicitly mapped flat domain")
        if any(axis.role != REPEAT for axis in self.repeat_domain.axes):
            raise ValueError("repeat-domain axes must carry the repeat role")
        if not isinstance(self.point_domain, DomainSpec):
            raise TypeError("point_domain must be DomainSpec")
        if self.point_domain.axis_codes is None or len(self.point_domain.shape) != 1:
            raise ValueError("point_domain must be an explicitly mapped flat domain")
        if any(axis.role == REPEAT for axis in self.point_domain.axes):
            raise ValueError("repeat-role axes belong only to the Repeat domain")
        if not isinstance(self.cell_domain, DomainSpec):
            raise TypeError("cell_domain must be DomainSpec")
        if self.cell_domain.axis_codes is not None:
            raise ValueError("cell_domain must use dense implicit axis mapping")
        if any(axis.role == REPEAT for axis in self.cell_domain.axes):
            raise ValueError("REPEAT role belongs only to the Repeat domain")
        scalar_axes = tuple(axis for axis in self.cell_domain.axes if axis.role == SCALAR)
        if scalar_axes and self.cell_domain != SCALAR_DOMAIN:
            raise ValueError("the canonical scalar domain must be the whole Cell domain")
        if not isinstance(self.value_schema, ValueSchema):
            raise TypeError("value_schema must be ValueSchema")
        if scalar_axes and self.value_schema.validity_contract.mode is not ValidityMode.VALUE:
            raise ValueError("the scalar carrier uses value-level validity")
        available = tuple(axis.axis_id for axis in self.cell_domain.axes)
        declared = self.value_schema.validity_contract.component_axis_ids
        if self.value_schema.validity_contract.mode is ValidityMode.COMPONENTS and not _ordered_subset(
            declared, available
        ):
            raise ValueError(
                "validity component axes must be an ordered subset of Cell axes"
            )
        all_ids = (
            *(axis.axis_id for axis in self.repeat_domain.axes),
            *(axis.axis_id for axis in self.point_domain.axes),
            *(axis.axis_id for axis in self.cell_domain.axes),
        )
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("dataset axis ids must be unique across all domains")
        object.__setattr__(self, "_fingerprint", None)
        object.__setattr__(self, "_structure_fingerprint", None)
        object.__setattr__(self, "_axis_catalog", None)
        object.__setattr__(self, "_indexed_layout", None)

    @cached_property
    def physical_shape(self) -> tuple[int, ...]:
        return (
            *self.repeat_domain.shape,
            *self.point_domain.shape,
            *self.cell_domain.shape,
        )

    def __reduce__(self):
        return DatasetSchema, (self.repeat_domain, self.point_domain, self.cell_domain, self.value_schema)

    @property
    def fingerprint(self) -> str:
        """This schema's canonical name, computed once, on request."""

        if self._fingerprint is None:
            from .codec import dataset_schema_fingerprint

            object.__setattr__(self, "_fingerprint", dataset_schema_fingerprint(self))
        return self._fingerprint

    @property
    def structure_fingerprint(self) -> str:
        """What this schema IS, without what its coordinates currently read.

        Two schemas share this name when they declare the same axes -- the
        same identities, roles, units, frames and shape -- however their
        coordinates happen to be numbered right now.  A bounded shot history
        slides its coordinates forward every shot by design; that renames the
        dataset, and it does not change the world an interaction was started
        in.
        """

        if self._structure_fingerprint is None:
            from .codec import dataset_schema_structure_fingerprint

            object.__setattr__(
                self,
                "_structure_fingerprint",
                dataset_schema_structure_fingerprint(self),
            )
        return self._structure_fingerprint
