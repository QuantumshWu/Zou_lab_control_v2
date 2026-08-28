"""The plotting boundary for the role-axis :mod:`zlc_data` contract.

This module deliberately contains no data model of its own.  It only names
the small amount of presentation plumbing needed to consume an immutable
``zlc_data.OwnedSnapshot``: revision/validity accessors and descriptors for
axes whose labels and display units belong to the plot layer.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    point_ordinal_axis,
)

from .kinds import AxisDomain, AxisRef
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


def snapshot_sigma(snapshot: OwnedSnapshot) -> NDArray[np.float64] | None:
    """The uncertainty of the samples themselves, or None if none is stated.

    Not the uncertainty of a reduction over them: that one is derived where
    the reduction happens, from the samples and their validity, and is
    never transported.  This is the other kind -- a property of one sample,
    which a fitted parameter has and a camera pixel does not -- and it
    cannot be recovered downstream, so the producer sends it along.
    """

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    sigma = snapshot.block.sigma
    return None if sigma is None else np.asarray(sigma, dtype=np.float64)


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


def live_grid_dimensions(schema: DatasetSchema) -> tuple[str, ...]:
    """Scan dimensions with more than one value, slowest-first.

    A degenerate dimension (one value) is real provenance but not structure:
    nothing can facet or image over it, so kind inference treats it as
    invisible.  This is the one place that says so -- every default_spec
    that reads the grid reads it through here.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be zlc_data.DatasetSchema")
    topology = schema.grid_topology
    if topology is None:
        return ()
    return tuple(
        str(dimension)
        for dimension, size in zip(topology.dimension_ids, topology.logical_shape)
        if size > 1
    )


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


@dataclass(frozen=True, slots=True)
class ResolvedAxis:
    """One exact schema-axis contract, before any full-sample broadcast.

    Data projection, semantic authoring and interaction identity all resolve
    through this object.  It contains only producer metadata and one tensor
    dimension; the potentially large ``(R, P, *D)`` coordinate planes remain
    a :class:`DataView` concern.
    """

    axis_id: AxisId
    name: str
    size: int
    coordinates: Sequence[Any]
    dimension: int
    unit_annotation: str | None = None
    coordinate_frame: str | None = None
    coordinate_labels: tuple[str, ...] | None = None
    declared_domain: bool = True
    topology_position: int | None = None

    @property
    def label(self) -> str:
        return self.name

    def canonical_unit(self, registry: UnitRegistry) -> Unit:
        return _resolve_annotation(self.unit_annotation, registry)

    def source_indices(self, schema: DatasetSchema) -> NDArray[np.int64]:
        """Indices selecting this axis coordinate at each source row."""

        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be zlc_data.DatasetSchema")
        if self.topology_position is None:
            return np.arange(self.size, dtype=np.int64)
        topology = schema.grid_topology
        if topology is None:
            raise KeyError(self.axis_id)
        # One column of the topology's own cached index array: the walk
        # over every row's tuple was the single largest cost of a dense
        # scan's revision.
        return np.ascontiguousarray(
            topology.cell_indices[:, self.topology_position]
        )


def resolve_axis(schema: DatasetSchema, ref: AxisRef) -> ResolvedAxis:
    """Resolve an exact :class:`AxisRef` against one schema.

    Human labels never participate.  Point coordinates and topology
    dimensions may share an ``AxisId`` as two views of the same physical point
    domain, but callers must choose the exact domain they intend.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be zlc_data.DatasetSchema")
    if not isinstance(ref, AxisRef):
        raise TypeError("ref must be AxisRef")

    def dense_axis(axis: AxisSpec, dimension: int) -> ResolvedAxis:
        coordinates: Sequence[Any] = (
            axis.coordinates
            if axis.coordinates is not None
            else range(axis.index_origin, axis.index_origin + axis.size)
        )
        return ResolvedAxis(
            axis.axis_id,
            axis.name,
            int(axis.size),
            coordinates,
            dimension,
            axis.unit,
            None
            if axis.coordinate_frame is None
            else str(axis.coordinate_frame),
            axis.coordinate_labels,
        )

    if ref.domain is AxisDomain.REPEAT:
        return dense_axis(schema.repeat_axis, 0)
    if ref.domain is AxisDomain.POINT_ROW:
        return dense_axis(point_ordinal_axis(schema.point_table.row_count), 1)
    assert ref.axis_id is not None
    axis_id = AxisId(ref.axis_id)
    if ref.domain is AxisDomain.POINT_COORDINATE:
        column = schema.point_table.column(axis_id)
        return ResolvedAxis(
            column.coordinate_id,
            column.name,
            len(column.values),
            column.values,
            1,
            column.unit,
            None
            if column.coordinate_frame is None
            else str(column.coordinate_frame),
            column.coordinate_labels,
            declared_domain=False,
        )
    if ref.domain is AxisDomain.POINT_DIMENSION:
        topology = schema.grid_topology
        if topology is None:
            raise KeyError(axis_id)
        try:
            position = topology.dimension_ids.index(axis_id)
        except ValueError as error:
            raise KeyError(axis_id) from error
        name = str(axis_id)
        unit: str | None = None
        frame: str | None = None
        labels: tuple[str, ...] | None = None
        try:
            column = schema.point_table.column(axis_id)
        except KeyError:
            pass
        else:
            name = column.name
            unit = column.unit
            frame = (
                None
                if column.coordinate_frame is None
                else str(column.coordinate_frame)
            )
            if column.coordinate_labels is not None:
                label_by_coordinate = dict(
                    zip(column.values, column.coordinate_labels, strict=True)
                )
                labels = tuple(
                    label_by_coordinate[value]
                    for value in topology.coordinate_domains[position]
                )
        return ResolvedAxis(
            axis_id,
            name,
            len(topology.coordinate_domains[position]),
            topology.coordinate_domains[position],
            1,
            unit,
            frame,
            labels,
            topology_position=position,
        )
    if ref.domain is AxisDomain.DATA:
        for position, axis in enumerate(schema.cell_schema.data_axes):
            if axis.axis_id == axis_id:
                return dense_axis(axis, 2 + position)
        raise KeyError(axis_id)
    raise KeyError(ref)


__all__ = [
    "ResolvedAxis",
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
    "live_grid_dimensions",
    "resolve_unit",
    "resolve_axis",
    "schema_equal",
    "schema_repeat_count",
    "schema_shape",
    "schema_value_unit",
    "snapshot_generation",
    "snapshot_revision",
    "snapshot_schema",
    "snapshot_validity",
    "snapshot_values",
]
