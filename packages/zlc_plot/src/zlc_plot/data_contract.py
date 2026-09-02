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
    PRIMARY_INDEX,
    READOUT_EVENT,
    SCAN_POINT,
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


def rows_are_named(schema: DatasetSchema) -> bool:
    """Whether something the producer declared identifies each point row.

    A declared topology does (the rows are its grid, in order), and so does
    any point column whose values are distinct -- a camera cycle's frame
    number, a scan's swept parameter.  When one of those exists, the rows ARE
    it, and offering a generic ordinal beside it puts the same axis in the
    table twice: an operator saw "point row (3)" and "frame (3)" and had to
    guess which of the two was the frames.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be zlc_data.DatasetSchema")
    if schema.grid_topology is not None:
        return True
    for column in schema.point_table.columns:
        values = tuple(column.values)
        if values and len(set(values)) == len(values):
            return True
    return False


#: One axis reference with the number of distinct positions it spans.
AxisEntry = tuple[AxisRef, int]


@dataclass(frozen=True)
class AxisFamilies:
    """One dataset's axes grouped by what they ARE, each with its size.

    The grouping is by declared role and place, never by name:

    ``repeat``   the repeat axis (R).
    ``history``  the Runtime's shot index (a PRIMARY_INDEX point axis).
    ``scan``     authored scan dimensions (SCAN_POINT), slowest first.
    ``events``   event sequences inside one cycle (READOUT_EVENT: frames,
                 frame pairs), and any topology dimension no column names.
    ``picture``  the two cell axes that ARE an image, as (x, y), when the
                 dataset declares one (:func:`image_axes`).
    ``content``  every other content axis -- sites, components, a point
                 column of sites -- slowest first, point axes first.
    ``data``     ``content`` and the picture together, in declaration order.
    ``rows``     the bare point-row ordinal, when nothing names the rows.

    ``topology`` says whether the points are a declared grid; a plot's
    defaults treat an unscanned cycle's point column as its authored cell
    identity even at one value, and a scan's degenerate dimension as
    invisible, which is why the two are told apart here rather than by
    every reader.
    """

    repeat: AxisEntry
    history: AxisEntry | None
    scan: tuple[AxisEntry, ...]
    events: tuple[AxisEntry, ...]
    picture: tuple[AxisEntry, AxisEntry] | None
    content: tuple[AxisEntry, ...]
    data: tuple[AxisEntry, ...]
    rows: AxisEntry | None
    topology: bool
    has_point_columns: bool

    def live_scan(self) -> tuple[AxisEntry, ...]:
        return tuple(entry for entry in self.scan if entry[1] > 1)

    def live_events(self) -> tuple[AxisEntry, ...]:
        return tuple(entry for entry in self.events if entry[1] > 1)

    def live_content(self) -> tuple[AxisEntry, ...]:
        return tuple(entry for entry in self.content if entry[1] > 1)

    def live_data(self) -> tuple[AxisEntry, ...]:
        return tuple(entry for entry in self.data if entry[1] > 1)

    def live_rows(self) -> AxisRef | None:
        return None if self.rows is None or self.rows[1] <= 1 else self.rows[0]

    def first_data_axis(self) -> AxisRef | None:
        return self.data[0][0] if self.data else None

    def rows_or_none(self) -> AxisRef | None:
        return None if self.rows is None else self.rows[0]


def _value_changes(column: PointColumn) -> int:
    """How many times a column's value changes from one row to the next."""

    values = tuple(column.values)
    return sum(1 for before, after in zip(values, values[1:]) if before != after)


def classify_axes(schema: DatasetSchema) -> AxisFamilies:
    """Group a dataset's axes into :class:`AxisFamilies`.

    A topology dimension takes the role of the point column that shares its
    id -- a scan axis is a SCAN_POINT column, a cycle's frames a
    READOUT_EVENT column -- and a dimension no column names (the anonymous
    source-point fallback) is an event sequence.  A point column that is
    not a dimension is classified the same way by its own role.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be zlc_data.DatasetSchema")
    columns = tuple(schema.point_table.columns)
    role_of = {str(column.coordinate_id): column.role for column in columns}
    entries: list[tuple[AxisRef, int, object]] = []
    topology = schema.grid_topology
    dimension_ids: set[str] = set()
    if topology is not None:
        for dimension, size in zip(topology.dimension_ids, topology.logical_shape):
            dimension_ids.add(str(dimension))
            entries.append(
                (
                    AxisRef.point_dimension(str(dimension)),
                    int(size),
                    role_of.get(str(dimension)),
                )
            )
    bare = [
        column
        for column in columns
        if str(column.coordinate_id) not in dimension_ids
    ]
    # A topology says which dimension is outermost.  Bare columns say so
    # with their rows: the column whose value changes least often down the
    # table is the slowest loop, so it comes first.  At a tie the LATER
    # declared column is taken as the slower, which leaves the first
    # declared as the sweep a curve walks.
    for _index, column in sorted(
        enumerate(bare), key=lambda item: (_value_changes(item[1]), -item[0])
    ):
        entries.append(
            (
                AxisRef.point(str(column.coordinate_id)),
                len(set(column.values)),
                column.role,
            )
        )
    history: AxisEntry | None = None
    scan: list[AxisEntry] = []
    events: list[AxisEntry] = []
    point_content: list[AxisEntry] = []
    for ref, size, role in entries:
        if role == PRIMARY_INDEX:
            if history is None:
                history = (ref, size)
        elif role == SCAN_POINT:
            scan.append((ref, size))
        elif role == READOUT_EVENT or role is None:
            events.append((ref, size))
        else:
            point_content.append((ref, size))
    pair = image_axes(schema)
    picture: tuple[AxisEntry, AxisEntry] | None = None
    if pair is not None:
        x_axis, y_axis = pair
        picture = (
            (AxisRef.data(str(x_axis.axis_id)), int(x_axis.size)),
            (AxisRef.data(str(y_axis.axis_id)), int(y_axis.size)),
        )
    picture_ids = (
        set() if pair is None else {str(axis.axis_id) for axis in pair}
    )
    cell_axes = tuple(
        (AxisRef.data(str(axis.axis_id)), int(axis.size))
        for axis in schema.cell_schema.data_axes
    )
    content = tuple(point_content) + tuple(
        entry for entry in cell_axes if entry[0].axis_id not in picture_ids
    )
    rows = (
        None
        if rows_are_named(schema)
        else (AxisRef.point_rows(), int(schema.point_table.row_count))
    )
    return AxisFamilies(
        repeat=(AxisRef.repeat(), int(schema.repeat_axis.size)),
        history=history,
        scan=tuple(scan),
        events=tuple(events),
        picture=picture,
        content=content,
        data=tuple(point_content) + cell_axes,
        rows=rows,
        topology=topology is not None,
        has_point_columns=bool(columns),
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
        # The labels are the domain's own: a cropped view keeps every domain
        # coordinate but only the surviving rows, so joining them through
        # the matching column's rows lost the labels of every row cropped.
        labels = (
            None
            if topology.coordinate_labels is None
            else topology.coordinate_labels[position]
        )
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
    "AxisEntry",
    "AxisFamilies",
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
    "classify_axes",
    "image_axes",
    "resolve_unit",
    "rows_are_named",
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
