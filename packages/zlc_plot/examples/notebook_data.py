"""Small public-zlc_data constructors used by the tutorial notebook."""

from __future__ import annotations

from pathlib import Path
import sys

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
    PointColumn,
    PointTable as RolePointTable,
    REPEAT,
    SCAN_POINT,
    SITE,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)


class Axis:
    @staticmethod
    def create(
        name: str,
        *,
        size: int | None = None,
        values: object | None = None,
        canonical_unit: str | None = None,
        label: str | None = None,
    ) -> AxisSpec:
        coordinates = None if values is None else tuple(np.asarray(values).tolist())
        count = len(coordinates) if coordinates is not None else (1 if size is None else int(size))
        return AxisSpec(
            AxisId(name),
            name if label is None else label,
            REPEAT if name == "repeat" else COMPONENT,
            count,
            coordinates,
            canonical_unit,
        )


class PointTable:
    @staticmethod
    def from_columns(
        columns: dict[str, object],
        *,
        units: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
    ) -> RolePointTable:
        del labels
        arrays = {str(name): tuple(np.asarray(values).tolist()) for name, values in columns.items()}
        count = len(next(iter(arrays.values())))
        unit_map = {} if units is None else dict(units)
        return RolePointTable(
            count,
            tuple(
                PointColumn(
                    AxisId(name),
                    name,
                    SITE if name == "site" else SCAN_POINT,
                    PointColumn.NUMERIC,
                    values,
                    unit_map.get(name),
                )
                for name, values in arrays.items()
            ),
        )


class PointTopology(GridTopology):
    @staticmethod
    def from_cartesian(dimensions: tuple[AxisSpec, ...], *, point_table: RolePointTable) -> GridTopology:
        domains = tuple(tuple(axis.coordinates or range(axis.size)) for axis in dimensions)
        mapping = tuple(np.ndindex(*(len(domain) for domain in domains)))
        if len(mapping) != point_table.row_count:
            raise ValueError("cartesian topology does not match point rows")
        return PointTopology(
            tuple(axis.axis_id for axis in dimensions),
            domains,
            mapping,
        )


class DatasetSchema(RoleDatasetSchema):
    @staticmethod
    def create(
        repeat_axis: AxisSpec,
        point_table: RolePointTable,
        *,
        data_axes: tuple[AxisSpec, ...] = (),
        point_topology: GridTopology | None = None,
        dtype: object = np.float64,
        canonical_unit: str | None = None,
    ) -> "DatasetSchema":
        cell = (
            ValueSchema.scalar(np.dtype(dtype), canonical_unit)
            if not data_axes
            else ValueSchema(
                tuple(data_axes),
                ValidityContract.value(),
                np.dtype(dtype),
                canonical_unit,
            )
        )
        return DatasetSchema(repeat_axis, point_table, point_topology, cell)


def DatasetSnapshot(
    schema: DatasetSchema,
    values: object,
    revision: int,
    *,
    validity: object | None = None,
) -> object:
    array = np.asarray(values)
    if schema.cell_schema.is_scalar and array.shape == schema.physical_shape[:-1]:
        array = array[..., None]
    return owned_snapshot_from_arrays(
        schema=schema,
        values=array,
        revision=revision,
        validity=validity,
    )


__all__ = ["Axis", "DatasetSchema", "DatasetSnapshot", "PointTable", "PointTopology"]
