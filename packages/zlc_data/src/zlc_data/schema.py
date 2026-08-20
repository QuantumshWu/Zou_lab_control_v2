"""Value and materialized dataset schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from .validation import canonical_text, nonnegative_integer, positive_integer

from ._arrays import canonical_dtype
from .axis import (
    _POINT_ORDINAL_AXIS_ID,
    HISTOGRAM_BIN,
    READOUT_EVENT,
    PRIMARY_INDEX,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    SPECTRAL,
    AxisId,
    AxisRoleId,
    AxisSpec,
    CoordinateFrameId,
    CoordinateScalar,
    REPEAT,
    SCALAR,
    SCALAR_AXIS,
    canonical_coordinate_scalar,
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


_POINT_ROLES = frozenset(
    {
        PRIMARY_INDEX,
        SCAN_POINT,
        READOUT_EVENT,
        SPATIAL_X,
        SPATIAL_Y,
        SPECTRAL,
        HISTOGRAM_BIN,
        SITE,
    }
)


@dataclass(frozen=True)
class PointColumn:
    """One correlated metadata column over the shared point-row domain."""

    coordinate_id: AxisId
    name: str
    role: AxisRoleId
    value_kind: str
    values: tuple[CoordinateScalar, ...]
    unit: str | None = None
    coordinate_frame: CoordinateFrameId | None = None
    coordinate_labels: tuple[str, ...] | None = None

    NUMERIC: ClassVar[str] = "NUMERIC"
    TEXT: ClassVar[str] = "TEXT"

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_id, AxisId):
            raise TypeError("coordinate_id must be AxisId")
        canonical_text(self.name, "point column name")
        if not isinstance(self.role, AxisRoleId) or self.role not in _POINT_ROLES:
            raise ValueError("point column role is outside the point-domain role set")
        if self.value_kind not in {self.NUMERIC, self.TEXT}:
            raise ValueError("point column value_kind must be NUMERIC or TEXT")
        values = tuple(
            canonical_coordinate_scalar(value, "point coordinate")
            for value in self.values
        )
        if not values:
            raise ValueError("point column values must be non-empty")
        if self.value_kind == self.NUMERIC:
            if any(
                value is not None and not isinstance(value, (int, float))
                for value in values
            ):
                raise TypeError("NUMERIC point columns accept only numbers or None")
        else:
            if any(value is not None and not isinstance(value, str) for value in values):
                raise TypeError("TEXT point columns accept only text or None")
            if self.unit is not None:
                raise ValueError("TEXT point columns cannot declare a unit")
        object.__setattr__(self, "values", values)
        if self.unit is not None:
            canonical_text(self.unit, "point column unit")
        if self.coordinate_frame is not None and not isinstance(
            self.coordinate_frame, CoordinateFrameId
        ):
            raise TypeError("coordinate_frame must be CoordinateFrameId or None")
        if self.coordinate_labels is not None:
            labels = tuple(
                canonical_text(label, "point coordinate label")
                for label in self.coordinate_labels
            )
            if len(labels) != len(values):
                raise ValueError(
                    "point coordinate_labels length must match point values"
                )
            labels_by_value: dict[CoordinateScalar, str] = {}
            for value, label in zip(values, labels, strict=True):
                previous = labels_by_value.setdefault(value, label)
                if previous != label:
                    raise ValueError(
                        "equal point coordinates must share one display label"
                    )
            object.__setattr__(self, "coordinate_labels", labels)


@dataclass(frozen=True)
class PointTable:
    """The authored ordered point rows shared by every repeat of a Dataset."""

    row_count: int
    columns: tuple[PointColumn, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "row_count",
            positive_integer(self.row_count, "point row_count"),
        )
        columns = tuple(self.columns)
        if any(not isinstance(column, PointColumn) for column in columns):
            raise TypeError("columns must contain PointColumn values")
        ids = tuple(column.coordinate_id for column in columns)
        if len(set(ids)) != len(ids):
            raise ValueError("point coordinate ids must be unique")
        if any(len(column.values) != self.row_count for column in columns):
            raise ValueError("every point column must contain exactly row_count values")
        object.__setattr__(self, "columns", columns)

    def column(self, coordinate_id: AxisId) -> PointColumn:
        if not isinstance(coordinate_id, AxisId):
            raise TypeError("coordinate_id must be AxisId")
        for column in self.columns:
            if column.coordinate_id == coordinate_id:
                return column
        raise KeyError(coordinate_id)


@dataclass(frozen=True)
class GridTopology:
    """Explicit producer-owned mapping from point rows to logical grid cells.

    A topology dimension may be declared only here; a matching PointColumn is
    optional metadata, not part of the topology's completeness requirement.
    """

    dimension_ids: tuple[AxisId, ...]
    coordinate_domains: tuple[tuple[CoordinateScalar, ...], ...]
    row_to_cell: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        dimensions = tuple(self.dimension_ids)
        if not dimensions or any(not isinstance(item, AxisId) for item in dimensions):
            raise TypeError("dimension_ids must contain at least one AxisId")
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("grid dimension ids must be unique")
        domains = tuple(
            tuple(
                canonical_coordinate_scalar(value, "grid coordinate")
                for value in domain
            )
            for domain in self.coordinate_domains
        )
        if len(domains) != len(dimensions):
            raise ValueError("grid domains must match dimension_ids")
        if any(not domain for domain in domains):
            raise ValueError("grid coordinate domains must be non-empty")
        if any(any(value is None for value in domain) for domain in domains):
            raise ValueError("grid coordinate domains cannot contain missing values")
        if any(len(set(domain)) != len(domain) for domain in domains):
            raise ValueError("grid coordinate domains must contain unique values")
        mapping: list[tuple[int, ...]] = []
        for cell in self.row_to_cell:
            normalized = tuple(
                nonnegative_integer(index, "grid cell index") for index in cell
            )
            if len(normalized) != len(dimensions):
                raise ValueError("grid cell rank must match dimension_ids")
            if any(index >= len(domain) for index, domain in zip(normalized, domains)):
                raise ValueError("grid cell index is outside its coordinate domain")
            mapping.append(normalized)
        if not mapping:
            raise ValueError("row_to_cell must be non-empty")
        if len(set(mapping)) != len(mapping):
            raise ValueError("row_to_cell must be injective")
        object.__setattr__(self, "dimension_ids", dimensions)
        object.__setattr__(self, "coordinate_domains", domains)
        object.__setattr__(self, "row_to_cell", tuple(mapping))

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return tuple(len(domain) for domain in self.coordinate_domains)


def _validate_grid_topology(point_table: PointTable, topology: GridTopology) -> None:
    if len(topology.row_to_cell) != point_table.row_count:
        raise ValueError("GridTopology must map every PointTable row")
    columns = {column.coordinate_id: column for column in point_table.columns}
    for position, dimension_id in enumerate(topology.dimension_ids):
        column = columns.get(dimension_id)
        if column is None:
            continue
        domain = topology.coordinate_domains[position]
        if any(
            column.values[ordinal] != domain[cell[position]]
            for ordinal, cell in enumerate(topology.row_to_cell)
        ):
            raise ValueError(
                "GridTopology cell values must match their PointTable columns"
            )


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
    data_axes: tuple[AxisSpec, ...]
    validity_contract: ValidityContract
    dtype: np.dtype
    value_unit: str | None = None
    #: Cached on first request.  Computed eagerly it cost 23 us per schema,
    #: paid by every intermediate schema construction that never names it.
    _fingerprint: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        axes = tuple(self.data_axes)
        if any(not isinstance(axis, AxisSpec) for axis in axes):
            raise TypeError("data_axes must contain AxisSpec values")
        if not axes:
            raise ValueError(
                "data_axes must be non-empty; use ValueSchema.scalar() for a scalar value"
            )
        scalar_axes = tuple(axis for axis in axes if axis.role == SCALAR)
        if scalar_axes and axes != (SCALAR_AXIS,):
            raise ValueError(
                "the canonical scalar carrier must be the sole data axis"
            )
        _unique_axis_ids(axes, context="data")
        object.__setattr__(self, "data_axes", axes)
        if not isinstance(self.validity_contract, ValidityContract):
            raise TypeError("validity_contract must be ValidityContract")
        if scalar_axes and self.validity_contract.mode is not ValidityMode.VALUE:
            raise ValueError("the scalar carrier uses value-level validity")
        object.__setattr__(self, "dtype", canonical_dtype(self.dtype))
        if self.value_unit is not None:
            canonical_text(self.value_unit, "value_unit")
        available = tuple(axis.axis_id for axis in axes)
        declared = self.validity_contract.component_axis_ids
        if self.validity_contract.mode is ValidityMode.COMPONENTS and not _ordered_subset(
            declared, available
        ):
            raise ValueError("validity component axes must be an ordered subset of data axes")
        object.__setattr__(self, "_fingerprint", None)

    @property
    def data_shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.data_axes)

    @property
    def is_scalar(self) -> bool:
        return self.data_axes == (SCALAR_AXIS,)

    @classmethod
    def scalar(
        cls,
        dtype: np.dtype,
        value_unit: str | None = None,
    ) -> "ValueSchema":
        return cls(
            (SCALAR_AXIS,),
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

    def axis(self, axis_id: AxisId) -> AxisSpec:
        for axis in self.data_axes:
            if axis.axis_id == axis_id:
                return axis
        raise KeyError(axis_id)


@dataclass(frozen=True)
class DatasetSchema:
    repeat_axis: AxisSpec
    point_table: PointTable
    grid_topology: GridTopology | None
    cell_schema: ValueSchema
    #: Cached on first request.  Computed eagerly it cost 23 us per schema,
    #: paid by every intermediate schema construction that never names it.
    _fingerprint: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.repeat_axis, AxisSpec) or self.repeat_axis.role != REPEAT:
            raise ValueError("repeat_axis must be an AxisSpec with role 'repeat'")
        if not isinstance(self.point_table, PointTable):
            raise TypeError("point_table must be PointTable")
        if self.grid_topology is not None and not isinstance(
            self.grid_topology, GridTopology
        ):
            raise TypeError("grid_topology must be GridTopology or None")
        if not isinstance(self.cell_schema, ValueSchema):
            raise TypeError("cell_schema must be ValueSchema")
        if any(axis.role == REPEAT for axis in self.cell_schema.data_axes):
            raise ValueError("REPEAT role belongs only to DatasetSchema.repeat_axis")
        all_ids = (
            self.repeat_axis.axis_id,
            *(column.coordinate_id for column in self.point_table.columns),
            *(axis.axis_id for axis in self.cell_schema.data_axes),
        )
        topology_ids = () if self.grid_topology is None else self.grid_topology.dimension_ids
        if _POINT_ORDINAL_AXIS_ID in (*all_ids, *topology_ids):
            raise ValueError(
                "dataset axis ids cannot use the reserved point-ordinal identity"
            )
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("dataset tensor and point coordinate ids must be unique")
        if set(topology_ids) & {
            self.repeat_axis.axis_id,
            *(axis.axis_id for axis in self.cell_schema.data_axes),
        }:
            raise ValueError("grid topology dimension ids must be distinct from dataset tensor ids")
        if self.grid_topology is not None:
            _validate_grid_topology(self.point_table, self.grid_topology)
        object.__setattr__(self, "_fingerprint", None)

    @property
    def physical_shape(self) -> tuple[int, ...]:
        return (
            self.repeat_axis.size,
            self.point_table.row_count,
            *self.cell_schema.data_shape,
        )

    @property
    def fingerprint(self) -> str:
        """This schema's canonical name, computed once, on request."""

        if self._fingerprint is None:
            from .codec import dataset_schema_fingerprint

            object.__setattr__(self, "_fingerprint", dataset_schema_fingerprint(self))
        return self._fingerprint
