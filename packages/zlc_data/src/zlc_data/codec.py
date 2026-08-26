"""Canonical codecs owned by the zlc_data bounded context."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._tree import digest as _tree_digest, encode as _encode
from .validation import canonical_text as _text, exact_mapping as _exact_map

from .axis import AxisId, AxisRoleId, AxisSpec, CoordinateFrameId
from .schema import (
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    ValueSchema,
)
from .value import BlockId, DatasetRevision, DatasetRevisionRef, StreamGenerationId
from .validity import (
    ValidityContract,
    ValidityMode,
)


AXIS_SCHEMA = "zlc_data.AxisSpec"
POINT_COLUMN_SCHEMA = "zlc_data.PointColumn"
POINT_TABLE_SCHEMA = "zlc_data.PointTable"
GRID_TOPOLOGY_SCHEMA = "zlc_data.GridTopology"
VALUE_SCHEMA = "zlc_data.ValueSchema"
DATASET_SCHEMA = "zlc_data.DatasetSchema"
DATASET_REVISION_REF_SCHEMA = "zlc_data.DatasetRevisionRef"


def dataset_revision_ref_to_tree(value: DatasetRevisionRef) -> dict[str, Any]:
    """Project one revision identity through its data-owner field model."""

    if not isinstance(value, DatasetRevisionRef):
        raise TypeError("value must be DatasetRevisionRef")
    return {
        "schema": DATASET_REVISION_REF_SCHEMA,
        "block_id": value.block_id.value,
        "stream_generation": value.stream_generation.value,
        "schema_fingerprint": value.schema_fingerprint,
        "revision": value.revision.value,
    }


def dataset_revision_ref_from_tree(tree: Any) -> DatasetRevisionRef:
    """Decode only the current exact DatasetRevisionRef representation."""

    data = _exact_map(
        tree,
        {
            "schema",
            "block_id",
            "stream_generation",
            "schema_fingerprint",
            "revision",
        },
        DATASET_REVISION_REF_SCHEMA,
    )
    value = DatasetRevisionRef(
        block_id=BlockId(data["block_id"]),
        stream_generation=StreamGenerationId(data["stream_generation"]),
        schema_fingerprint=data["schema_fingerprint"],
        revision=DatasetRevision(data["revision"]),
    )
    if _encode(dataset_revision_ref_to_tree(value)) != _encode(tree):
        raise ValueError("DatasetRevisionRef tree is typed but non-canonical")
    return value


def axis_to_tree(axis: AxisSpec) -> dict[str, Any]:
    return {
        "schema": AXIS_SCHEMA,
        "axis_id": axis.axis_id.value,
        "name": axis.name,
        "role": axis.role.value,
        "size": axis.size,
        "coordinates": None if axis.coordinates is None else list(axis.coordinates),
        "unit": axis.unit,
        "coordinate_frame": None
        if axis.coordinate_frame is None
        else axis.coordinate_frame.value,
        "index_origin": axis.index_origin,
        "coordinate_labels": None
        if axis.coordinate_labels is None
        else list(axis.coordinate_labels),
    }


def axis_from_tree(tree: Any) -> AxisSpec:
    data = _exact_map(
        tree,
        {
            "schema",
            "axis_id",
            "name",
            "role",
            "size",
            "coordinates",
            "unit",
            "coordinate_frame",
            "index_origin",
            "coordinate_labels",
        },
        AXIS_SCHEMA,
    )
    coordinates = data["coordinates"]
    if coordinates is not None and not isinstance(coordinates, list):
        raise ValueError("AxisSpec coordinates must be a list or null")
    frame = data["coordinate_frame"]
    coordinate_labels = data["coordinate_labels"]
    if coordinate_labels is not None and not isinstance(coordinate_labels, list):
        raise ValueError("AxisSpec coordinate_labels must be a list or null")
    axis = AxisSpec(
        axis_id=AxisId(data["axis_id"]),
        name=data["name"],
        role=AxisRoleId(data["role"]),
        size=data["size"],
        coordinates=None if coordinates is None else tuple(coordinates),
        unit=data["unit"],
        coordinate_frame=None if frame is None else CoordinateFrameId(frame),
        index_origin=data["index_origin"],
        coordinate_labels=None
        if coordinate_labels is None
        else tuple(coordinate_labels),
    )
    if _encode(axis_to_tree(axis)) != _encode(tree):
        raise ValueError("AxisSpec tree is typed but non-canonical")
    return axis


def point_column_to_tree(column: PointColumn) -> dict[str, Any]:
    if not isinstance(column, PointColumn):
        raise TypeError("column must be PointColumn")
    return {
        "schema": POINT_COLUMN_SCHEMA,
        "coordinate_id": column.coordinate_id.value,
        "name": column.name,
        "role": column.role.value,
        "value_kind": column.value_kind,
        "values": list(column.values),
        "unit": column.unit,
        "coordinate_frame": None
        if column.coordinate_frame is None
        else column.coordinate_frame.value,
        "coordinate_labels": None
        if column.coordinate_labels is None
        else list(column.coordinate_labels),
    }


def point_column_from_tree(tree: Any) -> PointColumn:
    data = _exact_map(
        tree,
        {
            "schema",
            "coordinate_id",
            "name",
            "role",
            "value_kind",
            "values",
            "unit",
            "coordinate_frame",
            "coordinate_labels",
        },
        POINT_COLUMN_SCHEMA,
    )
    values = data["values"]
    if not isinstance(values, list):
        raise ValueError("PointColumn values must be a list")
    frame = data["coordinate_frame"]
    coordinate_labels = data["coordinate_labels"]
    if coordinate_labels is not None and not isinstance(coordinate_labels, list):
        raise ValueError("PointColumn coordinate_labels must be a list or null")
    column = PointColumn(
        coordinate_id=AxisId(data["coordinate_id"]),
        name=data["name"],
        role=AxisRoleId(data["role"]),
        value_kind=data["value_kind"],
        values=tuple(values),
        unit=data["unit"],
        coordinate_frame=None if frame is None else CoordinateFrameId(frame),
        coordinate_labels=None
        if coordinate_labels is None
        else tuple(coordinate_labels),
    )
    if _encode(point_column_to_tree(column)) != _encode(tree):
        raise ValueError("PointColumn tree is typed but non-canonical")
    return column


def point_table_to_tree(table: PointTable) -> dict[str, Any]:
    if not isinstance(table, PointTable):
        raise TypeError("table must be PointTable")
    return {
        "schema": POINT_TABLE_SCHEMA,
        "row_count": table.row_count,
        "columns": [point_column_to_tree(column) for column in table.columns],
    }


def point_table_from_tree(tree: Any) -> PointTable:
    data = _exact_map(
        tree,
        {"schema", "row_count", "columns"},
        POINT_TABLE_SCHEMA,
    )
    columns = data["columns"]
    if not isinstance(columns, list):
        raise ValueError("PointTable columns must be a list")
    table = PointTable(
        row_count=data["row_count"],
        columns=tuple(point_column_from_tree(column) for column in columns),
    )
    if _encode(point_table_to_tree(table)) != _encode(tree):
        raise ValueError("PointTable tree is typed but non-canonical")
    return table


def grid_topology_to_tree(topology: GridTopology) -> dict[str, Any]:
    if not isinstance(topology, GridTopology):
        raise TypeError("topology must be GridTopology")
    return {
        "schema": GRID_TOPOLOGY_SCHEMA,
        "dimension_ids": [axis_id.value for axis_id in topology.dimension_ids],
        "coordinate_domains": [list(domain) for domain in topology.coordinate_domains],
        "row_to_cell": [list(cell) for cell in topology.row_to_cell],
    }


def grid_topology_from_tree(tree: Any) -> GridTopology:
    data = _exact_map(
        tree,
        {"schema", "dimension_ids", "coordinate_domains", "row_to_cell"},
        GRID_TOPOLOGY_SCHEMA,
    )
    dimensions = data["dimension_ids"]
    domains = data["coordinate_domains"]
    mapping = data["row_to_cell"]
    if not isinstance(dimensions, list):
        raise ValueError("GridTopology dimension_ids must be a list")
    if not isinstance(domains, list) or any(not isinstance(item, list) for item in domains):
        raise ValueError("GridTopology coordinate_domains must be lists")
    if not isinstance(mapping, list) or any(not isinstance(item, list) for item in mapping):
        raise ValueError("GridTopology row_to_cell must be a list of lists")
    topology = GridTopology(
        dimension_ids=tuple(AxisId(item) for item in dimensions),
        coordinate_domains=tuple(tuple(item) for item in domains),
        row_to_cell=tuple(tuple(item) for item in mapping),
    )
    if _encode(grid_topology_to_tree(topology)) != _encode(tree):
        raise ValueError("GridTopology tree is typed but non-canonical")
    return topology


def value_schema_to_tree(schema: ValueSchema) -> dict[str, Any]:
    return {
        "schema": VALUE_SCHEMA,
        "data_axes": [axis_to_tree(axis) for axis in schema.data_axes],
        "validity_contract": {
            "mode": schema.validity_contract.mode.value,
            "component_axis_ids": [
                axis_id.value for axis_id in schema.validity_contract.component_axis_ids
            ],
        },
        "dtype": schema.dtype.str,
        "value_unit": schema.value_unit,
    }


def value_schema_from_tree(tree: Any) -> ValueSchema:
    data = _exact_map(
        tree,
        {"schema", "data_axes", "validity_contract", "dtype", "value_unit"},
        VALUE_SCHEMA,
    )
    axes = data["data_axes"]
    if not isinstance(axes, list):
        raise ValueError("ValueSchema data_axes must be a list")
    validity = data["validity_contract"]
    if not isinstance(validity, dict) or set(validity) != {"mode", "component_axis_ids"}:
        raise ValueError("invalid ValueSchema validity_contract")
    mode = ValidityMode(validity["mode"])
    component_ids = validity["component_axis_ids"]
    if not isinstance(component_ids, list):
        raise ValueError("component_axis_ids must be a list")
    contract = ValidityContract(mode, tuple(AxisId(item) for item in component_ids))
    unit = data["value_unit"]
    schema = ValueSchema(
        data_axes=tuple(axis_from_tree(axis) for axis in axes),
        validity_contract=contract,
        dtype=np.dtype(_text(data["dtype"], "dtype")),
        value_unit=unit,
    )
    if _encode(value_schema_to_tree(schema)) != _encode(tree):
        raise ValueError("ValueSchema tree is typed but non-canonical")
    return schema


def dataset_schema_to_tree(schema: DatasetSchema) -> dict[str, Any]:
    return {
        "schema": DATASET_SCHEMA,
        "repeat_axis": axis_to_tree(schema.repeat_axis),
        "point_table": point_table_to_tree(schema.point_table),
        "grid_topology": None
        if schema.grid_topology is None
        else grid_topology_to_tree(schema.grid_topology),
        "cell_schema": value_schema_to_tree(schema.cell_schema),
    }


def dataset_schema_from_tree(tree: Any) -> DatasetSchema:
    data = _exact_map(
        tree,
        {"schema", "repeat_axis", "point_table", "grid_topology", "cell_schema"},
        DATASET_SCHEMA,
    )
    topology = data["grid_topology"]
    schema = DatasetSchema(
        repeat_axis=axis_from_tree(data["repeat_axis"]),
        point_table=point_table_from_tree(data["point_table"]),
        grid_topology=None if topology is None else grid_topology_from_tree(topology),
        cell_schema=value_schema_from_tree(data["cell_schema"]),
    )
    if _encode(dataset_schema_to_tree(schema)) != _encode(tree):
        raise ValueError("DatasetSchema tree is typed but non-canonical")
    return schema


def value_schema_fingerprint(schema: ValueSchema) -> str:
    return _tree_digest(value_schema_to_tree(schema))


def dataset_schema_fingerprint(schema: DatasetSchema) -> str:
    return _tree_digest(dataset_schema_to_tree(schema))


#: Tree keys holding one entry per coordinate rather than a structural fact.
_COORDINATE_KEYS = frozenset({"values", "coordinates", "coordinate_labels"})


def _structure_only(node: object) -> object:
    """The same tree with every coordinate LIST replaced by its length."""

    if isinstance(node, dict):
        return {
            key: (
                None
                if item is None
                else len(item)
                if key in _COORDINATE_KEYS and isinstance(item, list)
                else _structure_only(item)
            )
            for key, item in node.items()
        }
    if isinstance(node, list):
        return [_structure_only(item) for item in node]
    return node


def dataset_schema_structure_fingerprint(schema: DatasetSchema) -> str:
    """What this schema IS, without what its coordinates currently read.

    The full fingerprint includes every coordinate value, which is right for
    "is this the same dataset".  It is the wrong question for "is this the
    same world an interaction was started in": a bounded shot history slides
    its own coordinates forward by design -- every shot renames them -- while
    the axes, their roles, units and shape stand still.  Judging that by the
    full fingerprint made every shot look like a new geometry.
    """

    return _tree_digest(_structure_only(dataset_schema_to_tree(schema)))
