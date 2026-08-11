"""Small role-axis fixtures used by the offscreen plotting suite.

The production package never imports this module.  Keeping construction here
lets each test exercise the real ``zlc_data`` objects without recreating a
second data model in ``zlc_plot``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from zlc_data import (
    AxisId,
    AxisSpec,
    COMPONENT,
    DatasetSchema as RoleDatasetSchema,
    GridTopology,
    OwnedSnapshot,
    PointColumn,
    PointTable as RolePointTable,
    REPEAT,
    SCAN_POINT,
    SITE,
    ValueSchema,
    ValidityContract,
    owned_snapshot_from_arrays,
)


class Axis:
    """Test-only ergonomic constructor that returns a real ``AxisSpec``."""

    @staticmethod
    def create(
        name: str,
        *,
        size: int | None = None,
        values: Any | None = None,
        canonical_unit: str | None = None,
        label: str | None = None,
        role: object | None = None,
        **_: object,
    ) -> AxisSpec:
        del label
        if values is not None:
            coordinates = tuple(np.asarray(values).tolist())
            selected_size = len(coordinates)
        else:
            selected_size = 1 if size is None else int(size)
            coordinates = None
        # A declared role is honoured.  It used to fall into **_ and every
        # axis came out COMPONENT, so no test could express what a real
        # producer publishes -- which is how a role-blind image projection
        # stayed green.
        if role is None:
            role = REPEAT if name == "repeat" else COMPONENT
        return AxisSpec(
            AxisId(name),
            name,
            role,
            selected_size,
            coordinates,
            canonical_unit,
        )


class PointTable:
    @staticmethod
    def from_columns(
        columns: Mapping[str, Any],
        *,
        units: Mapping[str, str] | None = None,
        ids: Mapping[str, str] | None = None,
        roles: Mapping[str, Any] | None = None,
        **_: object,
    ) -> RolePointTable:
        """Build a point table; ``ids`` gives a column an id unlike its name.

        A real producer names a column for a reader and identifies it for a
        resolver, and the two need not match.  This factory used to force them
        equal, which made every "looked it up by the wrong one" bug untestable
        here -- and one shipped: a kind's default spec named its x axis by
        display name while the resolver looks up by coordinate id.

        A column of strings becomes a real TEXT column, because
        ``PointColumn`` has always had that kind and a projection asked to
        plot one must refuse -- which is untestable while the factory can
        only build numbers.

        ``roles`` declares what a column IS: a camera cycle's frame index is
        a ``READOUT_EVENT`` point, not a scan point, and a test that cannot
        say so cannot stand in for the producer it claims to model.
        """
        values = {
            str(name): _column_values(column) for name, column in columns.items()
        }
        if not values:
            raise ValueError("tests require at least one point column")
        row_count = len(next(iter(values.values())))
        unit_map = {} if units is None else dict(units)
        role_map = {} if roles is None else dict(roles)
        return RolePointTable(
            row_count,
            tuple(
                PointColumn(
                    AxisId((ids or {}).get(name, name)),
                    name,
                    role_map.get(
                        name,
                        SITE
                        if (ids or {}).get(name, name) == "site"
                        else SCAN_POINT,
                    ),
                    PointColumn.TEXT
                    if all(isinstance(item, str) for item in column)
                    else PointColumn.NUMERIC,
                    tuple(column),
                    None
                    if all(isinstance(item, str) for item in column)
                    else unit_map.get(name),
                )
                for name, column in values.items()
            ),
        )


def _column_values(column: Any) -> tuple[Any, ...]:
    items = tuple(column)
    if items and all(isinstance(item, str) for item in items):
        return items
    return tuple(np.asarray(column).tolist())


class PointTopology(GridTopology):
    @property
    def shape(self) -> tuple[int, ...]:
        return self.logical_shape

    @staticmethod
    def from_cartesian(
        dimensions: tuple[AxisSpec, ...] | list[AxisSpec],
        *,
        point_table: RolePointTable | None = None,
    ) -> GridTopology:
        dims = tuple(dimensions)
        domains = tuple(tuple(axis.coordinates or range(axis.size)) for axis in dims)
        mapping = tuple(np.ndindex(*(len(domain) for domain in domains)))
        if point_table is not None and len(mapping) != point_table.row_count:
            raise ValueError("cartesian topology does not match point row count")
        return PointTopology(
            tuple(axis.axis_id for axis in dims),
            domains,
            mapping,
        )


class DatasetSchema(RoleDatasetSchema):
    @property
    def shape(self) -> tuple[int, ...]:
        return self.physical_shape

    @property
    def ndim(self) -> int:
        return len(self.physical_shape)

    @property
    def R(self) -> int:
        return self.repeat_axis.size

    @property
    def P(self) -> int:
        return self.point_table.row_count

    @property
    def dtype(self) -> np.dtype:
        return self.cell_schema.dtype

    @staticmethod
    def create(
        repeat_axis: AxisSpec,
        point_table: RolePointTable,
        *,
        data_axes: tuple[AxisSpec, ...] = (),
        point_topology: GridTopology | None = None,
        dtype: Any = np.float64,
        canonical_unit: str | None = None,
        **_: object,
    ) -> RoleDatasetSchema:
        if not data_axes:
            value_schema = ValueSchema.scalar(np.dtype(dtype), canonical_unit)
        else:
            value_schema = ValueSchema(
                tuple(data_axes),
                ValidityContract.value(),
                np.dtype(dtype),
                canonical_unit,
            )
        return DatasetSchema(
            repeat_axis,
            point_table,
            point_topology,
            value_schema,
        )


def DatasetSnapshot(
    schema: RoleDatasetSchema,
    values: Any,
    revision: int,
    *,
    validity: Any | None = None,
    metadata: object | None = None,
) -> OwnedSnapshot:
    del metadata
    array = np.asarray(values)
    physical_shape = schema.physical_shape
    if schema.cell_schema.is_scalar and array.shape == physical_shape[:-1]:
        array = array[..., None]
        if validity is not None:
            valid_array = np.asarray(validity)
            if valid_array.shape == physical_shape[:-1]:
                validity = valid_array[..., None]
    return owned_snapshot_from_arrays(
        schema=schema,
        values=array,
        revision=revision,
        validity=validity,
    )


def snapshot_values(snapshot: OwnedSnapshot) -> np.ndarray:
    return snapshot.block.values


def snapshot_validity(snapshot: OwnedSnapshot) -> np.ndarray:
    return snapshot.expanded_validity()


__all__ = [
    "Axis",
    "DatasetSchema",
    "DatasetSnapshot",
    "PointTable",
    "PointTopology",
    "snapshot_validity",
    "snapshot_values",
]
