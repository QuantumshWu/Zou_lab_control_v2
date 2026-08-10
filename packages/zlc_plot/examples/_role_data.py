"""Ergonomic constructors for the public role-axis examples."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

# This checkout must win over any installed zlc_* distribution.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import zou_lab_control_v2  # noqa: F401

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
    @staticmethod
    def create(
        name: str,
        *,
        size: int | None = None,
        values: Any | None = None,
        canonical_unit: str | None = None,
        label: str | None = None,
        **_: object,
    ) -> AxisSpec:
        del label
        coordinates = None if values is None else tuple(np.asarray(values).tolist())
        return AxisSpec(
            AxisId(name),
            name,
            REPEAT if name == "repeat" else COMPONENT,
            len(coordinates) if coordinates is not None else (1 if size is None else int(size)),
            coordinates,
            canonical_unit,
        )


class PointTable:
    @staticmethod
    def from_columns(
        columns: Mapping[str, Any],
        *,
        units: Mapping[str, str] | None = None,
        **_: object,
    ) -> RolePointTable:
        values = {str(name): tuple(np.asarray(column).tolist()) for name, column in columns.items()}
        count = len(next(iter(values.values())))
        unit_map = {} if units is None else dict(units)
        return RolePointTable(
            count,
            tuple(
                PointColumn(
                    AxisId(name), name,
                    SITE if name == "site" else SCAN_POINT,
                    PointColumn.NUMERIC, tuple(column), unit_map.get(name),
                )
                for name, column in values.items()
            ),
        )


class PointTopology(GridTopology):
    @staticmethod
    def from_cartesian(dimensions: tuple[AxisSpec, ...] | list[AxisSpec], *, point_table=None) -> "PointTopology":
        dims = tuple(dimensions)
        domains = tuple(tuple(axis.coordinates or range(axis.size)) for axis in dims)
        mapping = tuple(np.ndindex(*(len(domain) for domain in domains)))
        if point_table is not None and len(mapping) != point_table.row_count:
            raise ValueError("cartesian topology does not match point rows")
        return PointTopology(tuple(axis.axis_id for axis in dims), domains, mapping)


class DatasetSchema(RoleDatasetSchema):
    @property
    def shape(self) -> tuple[int, ...]:
        return self.physical_shape

    @property
    def R(self) -> int:
        return self.repeat_axis.size

    @property
    def P(self) -> int:
        return self.point_table.row_count

    @property
    def data_axes(self) -> tuple[AxisSpec, ...]:
        return self.cell_schema.data_axes

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
    ) -> "DatasetSchema":
        cell = (
            ValueSchema.scalar(np.dtype(dtype), canonical_unit)
            if not data_axes
            else ValueSchema(tuple(data_axes), ValidityContract.value(), np.dtype(dtype), canonical_unit)
        )
        return DatasetSchema(repeat_axis, point_table, point_topology, cell)


def DatasetSnapshot(schema: RoleDatasetSchema, values: Any, revision: int, *, validity=None, metadata=None) -> OwnedSnapshot:
    del metadata
    array = np.asarray(values)
    if schema.cell_schema.is_scalar and array.shape == schema.physical_shape[:-1]:
        array = array[..., None]
        if validity is not None and np.asarray(validity).shape == schema.physical_shape[:-1]:
            validity = np.asarray(validity)[..., None]
    return owned_snapshot_from_arrays(schema=schema, values=array, revision=revision, validity=validity)


__all__ = ["Axis", "DatasetSchema", "DatasetSnapshot", "OwnedSnapshot", "PointTable", "PointTopology"]
