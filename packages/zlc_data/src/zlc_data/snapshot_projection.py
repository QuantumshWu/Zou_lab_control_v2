"""Materialize derived Dataset snapshots without losing physical axes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from .axis import (
    PRIMARY_INDEX,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    LATEST_COORDINATE,
    canonical_coordinate_scalar,
    point_ordinal_axis,
)
from .schema import DatasetSchema, GridTopology, PointColumn, PointTable, ValueSchema
from .selection import (
    IndexSelection,
    Selection,
    resolve_selection_indices,
    take_indices,
)
from .validity import (
    CellValidity,
    DatasetComponentValidity,
    Invalid,
    Valid,
)
from .value import (
    DataBlock,
    DatasetRevisionRef,
    OwnedSnapshot,
    compact_dataset_validity,
    expand_dataset_validity,
)

__all__ = [
    "PRIMARY_INDEX_AXIS_ID",
    "axis_catalog",
    "indexed_schemas_compatible",
    "materialize_derived_dataset",
    "restrict_snapshot",
    "restricted_schema",
    "restricted_values",
    "selection_indices",
    "value_selection",
]


#: The coordinate carried by every cell of a Runtime indexed-derived event.
#: It is a point coordinate because an event may retain its own repeat and
#: point geometry; putting the outer stream index on either of those physical
#: dimensions would erase producer-authored meaning.
PRIMARY_INDEX_AXIS_ID = AxisId("zlc_data.primary-index")


def _indexed_layout(schema: DatasetSchema) -> tuple[object, ...] | None:
    """Return the invariant event layout below a growing primary-index axis."""

    columns = tuple(schema.point_table.columns)
    primary = next(
        (column for column in columns if column.coordinate_id == PRIMARY_INDEX_AXIS_ID),
        None,
    )
    if primary is None:
        return None
    if primary.role != PRIMARY_INDEX:
        return None
    coordinates = tuple(primary.values)
    if any(type(value) is not int or value < 0 for value in coordinates):
        return None
    unique: list[int] = []
    counts: list[int] = []
    for coordinate in coordinates:
        if not unique or coordinate != unique[-1]:
            if unique and coordinate <= unique[-1]:
                return None
            unique.append(coordinate)
            counts.append(1)
        else:
            counts[-1] += 1
    if not counts or len(set(counts)) != 1:
        return None
    inner_count = counts[0]

    column_layout: list[object] = []
    for column in columns:
        if column.coordinate_id == PRIMARY_INDEX_AXIS_ID:
            continue
        values = tuple(column.values[:inner_count])
        labels = (
            None
            if column.coordinate_labels is None
            else tuple(column.coordinate_labels[:inner_count])
        )
        if any(
            tuple(column.values[offset : offset + inner_count]) != values
            for offset in range(inner_count, len(coordinates), inner_count)
        ):
            return None
        if column.coordinate_labels is not None and any(
            tuple(column.coordinate_labels[offset : offset + inner_count]) != labels
            for offset in range(inner_count, len(coordinates), inner_count)
        ):
            return None
        column_layout.append(
            (
                column.coordinate_id,
                column.name,
                column.role,
                column.value_kind,
                values,
                column.unit,
                column.coordinate_frame,
                labels,
            )
        )

    topology = schema.grid_topology
    topology_layout: object = None
    if topology is not None:
        if (
            not topology.dimension_ids
            or topology.dimension_ids[0] != PRIMARY_INDEX_AXIS_ID
            or tuple(topology.coordinate_domains[0]) != tuple(unique)
        ):
            return None
        base = tuple(cell[1:] for cell in topology.row_to_cell[:inner_count])
        for group, offset in enumerate(
            range(0, len(topology.row_to_cell), inner_count)
        ):
            block = topology.row_to_cell[offset : offset + inner_count]
            if any(cell[0] != group for cell in block):
                return None
            if tuple(cell[1:] for cell in block) != base:
                return None
        topology_layout = (
            topology.dimension_ids[1:],
            topology.coordinate_domains[1:],
            base,
        )
    return (
        schema.repeat_axis,
        schema.cell_schema,
        inner_count,
        tuple(column_layout),
        topology_layout,
    )


def indexed_schemas_compatible(
    left: DatasetSchema,
    right: DatasetSchema,
) -> bool:
    """Whether only an indexed Dataset's retained coordinate window changed."""

    if not isinstance(left, DatasetSchema) or not isinstance(right, DatasetSchema):
        raise TypeError("indexed schema compatibility requires DatasetSchema values")
    left_layout = _indexed_layout(left)
    return left_layout is not None and left_layout == _indexed_layout(right)


def _derived_reference(
    source_ref: DatasetRevisionRef,
    schema: DatasetSchema,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> DatasetRevisionRef:
    if not isinstance(source_ref, DatasetRevisionRef):
        raise TypeError("derived dataset source_ref must be DatasetRevisionRef")
    if not callable(reference_for):
        raise TypeError("derived dataset reference_for must be callable")
    ref = reference_for(schema)
    if not isinstance(ref, DatasetRevisionRef):
        raise TypeError("reference_for must return DatasetRevisionRef")
    if ref.block_id == source_ref.block_id:
        raise ValueError("a derived dataset cannot reuse its source BlockId")
    if ref.revision != source_ref.revision:
        raise ValueError("a derived dataset must retain its source revision")
    if ref.schema_fingerprint != schema.fingerprint:
        raise ValueError("derived reference schema differs from derived data")
    return ref


def materialize_derived_dataset(
    source_ref: DatasetRevisionRef,
    values: object,
    *,
    schema: DatasetSchema,
    validity: Valid | Invalid | CellValidity | DatasetComponentValidity,
    sigma: object | None = None,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Materialize one typed derived Dataset without interpreting its domain.

    ``sigma`` rides with the values because it belongs to them: a sample's
    own uncertainty survives every scope the operator can choose, which is
    exactly why it is not recomputed downstream the way a reduction's is.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("derived dataset schema must be DatasetSchema")
    ref = _derived_reference(source_ref, schema, reference_for)
    return OwnedSnapshot(
        ref,
        DataBlock(
            ref.block_id,
            ref.revision,
            np.asarray(values),
            validity,
            schema,
            None if sigma is None else np.asarray(sigma),
        ),
    )

def axis_catalog(
    schema: DatasetSchema,
) -> tuple[tuple[str, AxisId, AxisSpec, str], ...]:
    """Every axis a selection may name, under every label it answers to.

    Four columns: the label, the axis id, the axis, and which of the three
    physical dimensions the label restricts -- ``"repeat"``, ``"point"`` or
    ``"data"``.  Both the human name and the id text are listed, because a
    selection is authored against whichever the operator had in front of them.

    A grid-topology dimension is a point-row quantity whether or not a matching
    PointColumn spells its per-row values out: the topology itself is the
    declaration.  A scan-heatmap cell plots exactly these dimensions, so a box
    drawn on one names them -- and before they were catalogued here, every such
    box died as "not uniquely present".
    """

    catalog: list[tuple[str, AxisId, AxisSpec, str]] = []
    repeat_axis = schema.repeat_axis
    catalog.append((repeat_axis.name, repeat_axis.axis_id, repeat_axis, "repeat"))
    catalog.append(
        (repeat_axis.axis_id.value, repeat_axis.axis_id, repeat_axis, "repeat")
    )
    for column in schema.point_table.columns:
        axis = AxisSpec(
            column.coordinate_id,
            column.name,
            column.role,
            schema.point_table.row_count,
            column.values,
            column.unit,
            column.coordinate_frame,
            coordinate_labels=column.coordinate_labels,
        )
        catalog.extend(
            (
                (column.name, column.coordinate_id, axis, "point"),
                (column.coordinate_id.value, column.coordinate_id, axis, "point"),
            )
        )
    topology = schema.grid_topology
    if topology is not None:
        column_ids = {column.coordinate_id for column in schema.point_table.columns}
        for position, dimension_id in enumerate(topology.dimension_ids):
            if dimension_id in column_ids:
                continue  # the matching column above already catalogs it
            domain = topology.coordinate_domains[position]
            axis = AxisSpec(
                dimension_id,
                dimension_id.value,
                SCAN_POINT,
                schema.point_table.row_count,
                tuple(domain[cell[position]] for cell in topology.row_to_cell),
            )
            catalog.append((dimension_id.value, dimension_id, axis, "point"))
    for axis in schema.cell_schema.data_axes:
        catalog.extend(
            (
                (axis.name, axis.axis_id, axis, "data"),
                (axis.axis_id.value, axis.axis_id, axis, "data"),
            )
        )
    return tuple(catalog)


def selection_indices(
    schema: DatasetSchema,
    selection: Selection,
) -> tuple[
    range | tuple[int, ...],
    range | tuple[int, ...],
    dict[AxisId, range | tuple[int, ...]],
]:
    """Which repeats, point rows and data indices one selection keeps.

    Ranges wherever the surviving run is contiguous, which is every axis
    nobody selected on: a range indexes as a slice, and a tuple would make the
    axis a gather -- and a gather of one axis copies the whole frame behind it.
    A 512x512 box on a 1200x1920 frame costs 7.3 ms of pure copying that way
    where the slice costs 0.09.
    """

    catalog = {
        axis_id: (axis, kind) for _label, axis_id, axis, kind in axis_catalog(schema)
    }
    point_ordinal = point_ordinal_axis(schema.point_table.row_count)
    catalog[point_ordinal.axis_id] = (point_ordinal, "point")
    terms = {term.axis_id: term for term in selection.terms}

    def indices_for(axis_id: AxisId, default_size: int) -> range | tuple[int, ...]:
        selected = terms.get(axis_id)
        if selected is None:
            return range(default_size)
        axis, _kind = catalog[axis_id]
        resolved, _removes_axis = resolve_selection_indices(axis, selected)
        if not len(resolved):
            raise ValueError(f"selection for axis {axis_id} is empty")
        return resolved

    repeat = indices_for(schema.repeat_axis.axis_id, schema.repeat_axis.size)
    points_set = set(range(schema.point_table.row_count))
    # Every "point"-kind catalog axis masks point ROWS: the table's own columns
    # and the grid-topology dimensions catalogued for them.  Restricting this
    # to declared columns silently ignored a term on a column-less dimension.
    point_ids = {
        axis_id for axis_id, (_axis, kind) in catalog.items() if kind == "point"
    }
    for axis_id in point_ids:
        if axis_id in terms:
            axis, _kind = catalog[axis_id]
            resolved, _removes_axis = resolve_selection_indices(axis, terms[axis_id])
            points_set.intersection_update(resolved)
    ordered = sorted(points_set)
    if not ordered:
        raise ValueError("selection removed every source point row")
    points = (
        range(ordered[0], ordered[-1] + 1)
        if ordered[-1] - ordered[0] + 1 == len(ordered)
        else tuple(ordered)
    )
    data: dict[AxisId, range | tuple[int, ...]] = {}
    for axis in schema.cell_schema.data_axes:
        data[axis.axis_id] = indices_for(axis.axis_id, axis.size)
    for axis_id in terms:
        if axis_id not in catalog:
            raise ValueError(f"selection axis {axis_id} is absent from source schema")
    return repeat, points, data


def _subset_axis(axis: AxisSpec, indices: range | tuple[int, ...]) -> AxisSpec:
    labels = (
        None
        if axis.coordinate_labels is None
        else tuple(axis.coordinate_labels[index] for index in indices)
    )
    if isinstance(indices, range) and axis.coordinates is None:
        # An implicit axis cropped to a contiguous run is still implicit: it
        # starts later.  Writing the coordinates out here undid, one layer
        # down, the very thing the producer stopped doing -- build a tuple,
        # validate every element, and digest all of it per frame.
        return replace(
            axis,
            size=len(indices),
            index_origin=axis.index_origin + indices.start,
            coordinate_labels=labels,
        )
    coordinates = tuple(axis.coordinate_at(index) for index in indices)
    return AxisSpec(
        axis.axis_id,
        axis.name,
        axis.role,
        len(indices),
        coordinates,
        axis.unit,
        axis.coordinate_frame,
        coordinate_labels=labels,
    )


def _subset_point_table(
    point_table: PointTable,
    indices: range | tuple[int, ...],
) -> PointTable:
    return PointTable(
        len(indices),
        tuple(
            PointColumn(
                column.coordinate_id,
                column.name,
                column.role,
                column.value_kind,
                tuple(column.values[index] for index in indices),
                column.unit,
                column.coordinate_frame,
                None
                if column.coordinate_labels is None
                else tuple(column.coordinate_labels[index] for index in indices),
            )
            for column in point_table.columns
        ),
    )


def _keeps_everything(indices: range | tuple[int, ...], size: int) -> bool:
    return len(indices) == size and all(
        position == index for position, index in enumerate(indices)
    )


def restricted_schema(
    schema: DatasetSchema,
    repeat_indices: range | tuple[int, ...],
    point_indices: range | tuple[int, ...],
    data_indices: Mapping[AxisId, range | tuple[int, ...]],
) -> DatasetSchema:
    """The schema of what survives -- the source's axes, cropped, not dropped."""

    repeat_axis = _subset_axis(schema.repeat_axis, repeat_indices)
    point_table = _subset_point_table(schema.point_table, point_indices)
    cell_axes = tuple(
        _subset_axis(axis, data_indices[axis.axis_id])
        for axis in schema.cell_schema.data_axes
    )
    cell_schema = ValueSchema(
        cell_axes,
        schema.cell_schema.validity_contract,
        schema.cell_schema.dtype,
        schema.cell_schema.value_unit,
    )
    grid_topology = schema.grid_topology
    if grid_topology is not None and not _keeps_everything(
        point_indices, schema.point_table.row_count
    ):
        grid_topology = GridTopology(
            grid_topology.dimension_ids,
            grid_topology.coordinate_domains,
            tuple(grid_topology.row_to_cell[index] for index in point_indices),
        )
    return DatasetSchema(repeat_axis, point_table, grid_topology, cell_schema)


def restricted_values(
    values: np.ndarray,
    schema: DatasetSchema,
    repeat_indices: range | tuple[int, ...],
    point_indices: range | tuple[int, ...],
    data_indices: Mapping[AxisId, range | tuple[int, ...]],
) -> np.ndarray:
    """The values that survive, as a view wherever the selection is contiguous."""

    result = take_indices(values, repeat_indices, axis=0)
    result = take_indices(result, point_indices, axis=1)
    for position, axis in enumerate(schema.cell_schema.data_axes):
        result = take_indices(result, data_indices[axis.axis_id], axis=2 + position)
    return result


def value_selection(
    schema: DatasetSchema,
    terms: Mapping[str | AxisId, object],
) -> Selection:
    """Keep one named coordinate on each named axis.

    The one translation from "this axis, that value" -- a facet cell, a scope
    row, a site number typed into a panel -- into the Selection vocabulary.
    An axis that spells its coordinates out is selected BY coordinate; an
    implicit one is selected by index, because its coordinate is its index and
    a range over floats would only be a longer way to say so.
    """

    catalog = list(axis_catalog(schema))
    ordinal = point_ordinal_axis(schema.point_table.row_count)
    catalog.extend(
        (
            (ordinal.axis_id.value, ordinal.axis_id, ordinal, "point"),
            (ordinal.name, ordinal.axis_id, ordinal, "point"),
        )
    )
    resolved: list[object] = []
    for label, value in terms.items():
        if not isinstance(label, (str, AxisId)):
            raise TypeError("selection axis reference must be text or AxisId")
        matches = (
            [entry for entry in catalog if entry[1] == label]
            if isinstance(label, AxisId)
            else [entry for entry in catalog if entry[0] == label]
        )
        axis_ids = {entry[1] for entry in matches}
        if not matches:
            raise ValueError(f"selection axis {label!r} is absent from source schema")
        if len(axis_ids) != 1:
            raise ValueError(
                f"selection axis {label!r} is not uniquely present in the source schema"
            )
        axis = matches[0][2]
        coordinate = (
            axis.coordinate_at(axis.size - 1)
            if value is LATEST_COORDINATE
            else canonical_coordinate_scalar(value, f"axis {axis.axis_id} coordinate")
        )
        if axis.coordinates is None:
            if not isinstance(coordinate, int):
                raise TypeError(
                    f"implicit axis {axis.axis_id} requires an integer coordinate"
                )
            index = coordinate - axis.index_origin
            if not 0 <= index < axis.size:
                raise ValueError(
                    f"axis {label!r} has no index {value!r}: it runs "
                    f"{axis.index_origin}..{axis.index_origin + axis.size - 1}"
                )
            resolved.append(IndexSelection(axis.axis_id, index))
        else:
            resolved.append(
                Selection.coordinate_range(
                    axis.axis_id,
                    coordinate,
                    coordinate,
                    coordinate_frame=axis.coordinate_frame,
                ).terms[0]
            )
    return Selection(tuple(resolved))


def restrict_snapshot(
    snapshot: OwnedSnapshot,
    selection: Selection,
    *,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """One selection applied to one snapshot: the same axes, over less of them.

    The single cutter.  Restriction is arithmetic on a schema, its values and
    its validity, so it lives with the data, and everyone who restricts -- the
    selection bridge deriving a signal, a panel scoped to one site -- asks
    here.  Two cutters is how a box drawn on a scoped panel came to cut from
    the whole signal instead: they agreed about the intent and disagreed about
    the domain it applied to.
    """

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("restrict_snapshot requires an OwnedSnapshot")
    schema = snapshot.block.schema
    repeat_indices, point_indices, data_indices = selection_indices(schema, selection)
    derived = restricted_schema(schema, repeat_indices, point_indices, data_indices)
    values = restricted_values(
        snapshot.block.values, schema, repeat_indices, point_indices, data_indices
    )
    mask = restricted_values(
        expand_dataset_validity(snapshot.block.validity, schema),
        schema,
        repeat_indices,
        point_indices,
        data_indices,
    )
    return materialize_derived_dataset(
        snapshot.ref,
        values,
        schema=derived,
        validity=compact_dataset_validity(mask, derived),
        reference_for=reference_for,
    )
