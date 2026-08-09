"""The plotting boundary for the role-axis :mod:`zlc_data` contract.

This module deliberately contains no data model of its own.  It only names
the small amount of presentation plumbing needed to consume an immutable
``zlc_data.OwnedSnapshot``: revision/validity accessors and descriptors for
axes whose labels and display units belong to the plot layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from zlc_data import (
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    DatasetSchema,
    GridTopology,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    expand_snapshot_validity,
)

from .units import DEFAULT_UNITS, Unit, UnitRegistry, resolve_unit


def snapshot_schema(snapshot: OwnedSnapshot) -> DatasetSchema:
    """Return the immutable schema owned by a real ``OwnedSnapshot``."""

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    return snapshot.block.schema


def snapshot_values(snapshot: OwnedSnapshot) -> NDArray[Any]:
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    return snapshot.block.values


def snapshot_validity(snapshot: OwnedSnapshot) -> NDArray[np.bool_]:
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    return np.asarray(expand_snapshot_validity(snapshot), dtype=np.bool_)


def snapshot_revision(snapshot: OwnedSnapshot) -> int:
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    return int(snapshot.ref.revision.value)


def snapshot_generation(snapshot: OwnedSnapshot) -> str:
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    return str(snapshot.ref.stream_generation.value)


def schema_shape(schema: DatasetSchema) -> tuple[int, ...]:
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be zlc_data.DatasetSchema")
    return schema.physical_shape


def schema_repeat_count(schema: DatasetSchema) -> int:
    return int(schema.repeat_axis.size)


def schema_point_count(schema: DatasetSchema) -> int:
    return int(schema.point_table.row_count)


def schema_data_axes(schema: DatasetSchema) -> tuple[AxisSpec, ...]:
    return tuple(schema.cell_schema.data_axes)


def schema_dtype(schema: DatasetSchema) -> np.dtype:
    return np.dtype(schema.cell_schema.dtype)


def schema_value_unit(schema: DatasetSchema, registry: UnitRegistry) -> Unit:
    return resolve_unit(schema.cell_schema.value_unit or "1", registry)


def image_axes(schema: DatasetSchema) -> tuple[Any, Any] | None:
    """The two cell axes that ARE the image, or None if this is not one.

    By role first.  A dataset says which of its axes are spatial -- that is
    what AxisRoleId is for, and what lets a reader that has never seen this
    dataset tell a camera frame from a stack of them.  Choosing by size and
    position instead worked only while the other axes happened to be length
    one: a two-window bracket has three cell axes and was refused outright, and
    a one-pixel-tall ROI strip was refused from the other side.

    Falls back to the trailing pair when no roles are declared, which is where
    an image lives when nothing says otherwise.  Slowest-first C order, so the
    last axis is x and the one above it is y.
    """

    if not isinstance(schema, DatasetSchema):
        return None
    axes = tuple(schema.cell_schema.data_axes)
    by_role = {axis.role: axis for axis in axes}
    x_axis, y_axis = by_role.get(SPATIAL_X), by_role.get(SPATIAL_Y)
    if x_axis is not None and y_axis is not None:
        return x_axis, y_axis
    if x_axis is not None or y_axis is not None:
        # One spatial axis is a profile, not an image.
        return None
    significant = tuple(axis for axis in axes if axis.size > 1)
    if len(significant) != 2:
        return None
    return significant[-1], significant[-2]


def schema_equal(left: DatasetSchema, right: DatasetSchema) -> bool:
    """Are these two schemas the same schema?

    Directly, because both are in hand.  A digest is how a schema is recognised
    across a process boundary, where only its name travelled; asking for two
    digests to compare two objects is slower and answers the same question --
    the dtype is canonicalised when a schema is built, so nothing is normalised
    by the encoding that equality would miss.
    """

    if not isinstance(left, DatasetSchema) or not isinstance(right, DatasetSchema):
        return False
    return left == right


def _resolve_annotation(annotation: str | None, registry: UnitRegistry) -> Unit:
    # ``None`` is the data-layer spelling for an unlabelled/dimensionless
    # coordinate; ``arb`` remains a public plotting alias for ``1``.
    return resolve_unit(annotation or "1", registry)


def implicit_coordinates(axis: AxisSpec) -> tuple[Any, ...]:
    if axis.coordinates is not None:
        return tuple(axis.coordinates)
    return tuple(axis.index_origin + index for index in range(axis.size))


@dataclass(frozen=True, slots=True)
class AxisDescriptor:
    """Presentation descriptor derived from one role-axis declaration."""

    axis_id: AxisId
    name: str
    size: int
    coordinates: tuple[Any, ...]
    unit_annotation: str | None = None
    coordinate_labels: tuple[str, ...] | None = None

    @property
    def label(self) -> str:
        return self.name

    def canonical_unit(self, registry: UnitRegistry) -> Unit:
        return _resolve_annotation(self.unit_annotation, registry)


def descriptor_from_axis(axis: AxisSpec) -> AxisDescriptor:
    if not isinstance(axis, AxisSpec):
        raise TypeError("axis must be zlc_data.AxisSpec")
    return AxisDescriptor(
        axis.axis_id,
        axis.name,
        int(axis.size),
        implicit_coordinates(axis),
        axis.unit,
        axis.coordinate_labels,
    )


def descriptor_from_point_column(column: PointColumn) -> AxisDescriptor:
    if not isinstance(column, PointColumn):
        raise TypeError("column must be zlc_data.PointColumn")
    return AxisDescriptor(
        column.coordinate_id,
        column.name,
        len(column.values),
        tuple(column.values),
        column.unit,
        column.coordinate_labels,
    )


def descriptor_from_topology(
    topology: GridTopology,
    position: int,
    *,
    point_table: PointTable | None = None,
) -> AxisDescriptor:
    if not isinstance(topology, GridTopology):
        raise TypeError("topology must be zlc_data.GridTopology")
    axis_id = topology.dimension_ids[position]
    unit: str | None = None
    name = str(axis_id)
    coordinate_labels: tuple[str, ...] | None = None
    # W-round GridTopology intentionally stores geometry, not presentation
    # annotations.  When the producer also carries the dimension in its
    # PointTable, reuse that column's unit and label; otherwise the plotting
    # layer correctly treats the topology coordinate as dimensionless.
    if point_table is not None:
        try:
            column = point_column(point_table, axis_id)
        except (KeyError, TypeError):
            pass
        else:
            name = column.name
            unit = column.unit
            if column.coordinate_labels is not None:
                label_by_coordinate = dict(
                    zip(column.values, column.coordinate_labels, strict=True)
                )
                coordinate_labels = tuple(
                    label_by_coordinate[value]
                    for value in topology.coordinate_domains[position]
                )
    return AxisDescriptor(
        axis_id,
        name,
        len(topology.coordinate_domains[position]),
        tuple(topology.coordinate_domains[position]),
        unit,
        coordinate_labels,
    )


def point_column(table: PointTable, axis_id: AxisId) -> PointColumn:
    if not isinstance(table, PointTable):
        raise TypeError("table must be zlc_data.PointTable")
    if isinstance(axis_id, str):
        axis_id = AxisId(axis_id)
    return table.column(axis_id)


def topology_position(topology: GridTopology, axis_id: AxisId) -> int:
    if not isinstance(topology, GridTopology):
        raise TypeError("topology must be zlc_data.GridTopology")
    if isinstance(axis_id, str):
        axis_id = AxisId(axis_id)
    try:
        return topology.dimension_ids.index(axis_id)
    except ValueError as exc:
        raise KeyError(axis_id) from exc


__all__ = [
    "AxisDescriptor",
    "AxisId",
    "AxisSpec",
    "DEFAULT_UNITS",
    "DatasetSchema",
    "GridTopology",
    "OwnedSnapshot",
    "PointColumn",
    "PointTable",
    "Unit",
    "UnitRegistry",
    "descriptor_from_axis",
    "descriptor_from_point_column",
    "descriptor_from_topology",
    "implicit_coordinates",
    "point_column",
    "resolve_unit",
    "schema_data_axes",
    "schema_dtype",
    "schema_equal",
    "schema_point_count",
    "schema_repeat_count",
    "schema_shape",
    "schema_value_unit",
    "snapshot_generation",
    "snapshot_revision",
    "snapshot_schema",
    "snapshot_validity",
    "snapshot_values",
    "topology_position",
]
