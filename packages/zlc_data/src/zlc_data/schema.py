"""Value and materialized dataset schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

import numpy as np
from .validation import canonical_text, positive_integer

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


@dataclass(frozen=True)
class DomainSpec:
    """One physical domain and its logical named axes.

    Repeat and Point use explicit, axis-major ``axis_codes`` over their flat
    carrier. Cell-data leaves ``axis_codes`` as ``None``: each axis then maps
    by position to one dense physical dimension. Dense codes are only the
    corresponding one-dimensional identity, never a broadcast pixel plane.
    """

    shape: tuple[int, ...]
    axes: tuple[AxisSpec, ...] = ()
    axis_codes: tuple[tuple[int, ...], ...] | None = None
    _codes: Any = field(init=False, repr=False, compare=False, default=None)

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
        for axis in axes:
            if axis.coordinates is not None:
                if any(value is None for value in axis.coordinates):
                    raise ValueError("domain axis coordinates cannot be missing")
                if len(set(axis.coordinates)) != axis.size:
                    raise ValueError("domain axis coordinates must be unique")

        authored_codes = self.axis_codes
        normalized_codes: tuple[tuple[int, ...], ...] | None
        cached_codes: list[np.ndarray] = []
        if authored_codes is None:
            if len(axes) != len(shape) or any(
                axis.size != size for axis, size in zip(axes, shape, strict=True)
            ):
                raise ValueError(
                    "a dense domain needs one same-sized axis per physical dimension"
                )
            normalized_codes = None
            for axis in axes:
                codes = immutable_array(
                    np.arange(axis.size, dtype=np.int64),
                    dtype=np.dtype("<i8"),
                    shape=(axis.size,),
                )
                cached_codes.append(codes)
        else:
            authored_codes = tuple(authored_codes)
            if len(shape) != 1:
                raise ValueError("an explicitly mapped domain must have one physical dimension")
            if len(authored_codes) != len(axes):
                raise ValueError("axis_codes must contain one vector per axis")
            carrier_size = prod(shape)
            normalized: list[tuple[int, ...]] = []
            for axis, entries in zip(axes, authored_codes, strict=True):
                array = np.asarray(entries)
                if array.ndim != 1 or array.size != carrier_size:
                    raise ValueError("each axis code vector must match domain size")
                if not issubclass(array.dtype.type, np.integer):
                    raise TypeError("axis codes must be integers")
                codes = np.asarray(array, dtype=np.int64)
                if bool(np.any(codes < 0)) or bool(np.any(codes >= axis.size)):
                    raise ValueError("axis code is outside its coordinate domain")
                canonical = tuple(codes.tolist())
                cached_codes.append(
                    immutable_array(
                        codes,
                        dtype=np.dtype("<i8"),
                        shape=(carrier_size,),
                    )
                )
                normalized.append(canonical)
            normalized_codes = tuple(normalized)
            if not axes and carrier_size != 1:
                raise ValueError("a mapped domain with multiple rows needs a named axis")

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "axis_codes", normalized_codes)
        object.__setattr__(self, "_codes", tuple(cached_codes))

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)

    def axis(self, axis_id: AxisId) -> AxisSpec:
        if not isinstance(axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        for axis in self.axes:
            if axis.axis_id == axis_id:
                return axis
        raise KeyError(axis_id)

    def codes(self, axis_id: AxisId) -> np.ndarray:
        """Readonly one-dimensional codes along the axis's physical dimension."""

        axis = self.axis(axis_id)
        return self._codes[self.axes.index(axis)]

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
    #: Cached on first request.  Computed eagerly it cost 23 us per schema,
    #: paid by every intermediate schema construction that never names it.
    _fingerprint: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.validity_contract, ValidityContract):
            raise TypeError("validity_contract must be ValidityContract")
        object.__setattr__(self, "dtype", canonical_dtype(self.dtype))
        if self.value_unit is not None:
            canonical_text(self.value_unit, "value_unit")
        object.__setattr__(self, "_fingerprint", None)

    @classmethod
    def scalar(
        cls,
        dtype: np.dtype,
        value_unit: str | None = None,
    ) -> "ValueSchema":
        return cls(
            ValidityContract.value(),
            dtype,
            value_unit,
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

    @property
    def physical_shape(self) -> tuple[int, ...]:
        return (
            *self.repeat_domain.shape,
            *self.point_domain.shape,
            *self.cell_domain.shape,
        )

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
