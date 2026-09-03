"""Canonical codecs owned by the zlc_data bounded context."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._tree import digest as _tree_digest, encode as _encode
from .validation import canonical_text as _text, exact_mapping as _exact_map

from .axis import AxisId, AxisRoleId, AxisSpec, CoordinateFrameId
from .schema import (
    DatasetSchema,
    DomainSpec,
    ValueSchema,
)
from .value import BlockId, DatasetRevision, DatasetRevisionRef, StreamGenerationId
from .validity import (
    ValidityContract,
    ValidityMode,
)


AXIS_SCHEMA = "zlc_data.AxisSpec"
DOMAIN_SCHEMA = "zlc_data.DomainSpec"
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


def domain_to_tree(domain: DomainSpec) -> dict[str, Any]:
    if not isinstance(domain, DomainSpec):
        raise TypeError("domain must be DomainSpec")
    return {
        "schema": DOMAIN_SCHEMA,
        "shape": list(domain.shape),
        "axes": [axis_to_tree(axis) for axis in domain.axes],
        "axis_codes": None
        if domain.axis_codes is None
        else [list(codes) for codes in domain.axis_codes],
    }


def domain_from_tree(tree: Any) -> DomainSpec:
    data = _exact_map(
        tree,
        {"schema", "shape", "axes", "axis_codes"},
        DOMAIN_SCHEMA,
    )
    shape = data["shape"]
    axes = data["axes"]
    codes = data["axis_codes"]
    if not isinstance(shape, list):
        raise ValueError("DomainSpec shape must be a list")
    if not isinstance(axes, list):
        raise ValueError("DomainSpec axes must be a list")
    if codes is not None and (
        not isinstance(codes, list)
        or any(not isinstance(item, list) for item in codes)
    ):
        raise ValueError("DomainSpec axis_codes must be a list of lists or null")
    domain = DomainSpec(
        shape=tuple(shape),
        axes=tuple(axis_from_tree(axis) for axis in axes),
        axis_codes=None if codes is None else tuple(tuple(item) for item in codes),
    )
    if _encode(domain_to_tree(domain)) != _encode(tree):
        raise ValueError("DomainSpec tree is typed but non-canonical")
    return domain


def value_schema_to_tree(schema: ValueSchema) -> dict[str, Any]:
    return {
        "schema": VALUE_SCHEMA,
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
        {"schema", "validity_contract", "dtype", "value_unit"},
        VALUE_SCHEMA,
    )
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
        "repeat_domain": domain_to_tree(schema.repeat_domain),
        "point_domain": domain_to_tree(schema.point_domain),
        "cell_domain": domain_to_tree(schema.cell_domain),
        "value_schema": value_schema_to_tree(schema.value_schema),
    }


def dataset_schema_from_tree(tree: Any) -> DatasetSchema:
    data = _exact_map(
        tree,
        {"schema", "repeat_domain", "point_domain", "cell_domain", "value_schema"},
        DATASET_SCHEMA,
    )
    schema = DatasetSchema(
        repeat_domain=domain_from_tree(data["repeat_domain"]),
        point_domain=domain_from_tree(data["point_domain"]),
        cell_domain=domain_from_tree(data["cell_domain"]),
        value_schema=value_schema_from_tree(data["value_schema"]),
    )
    if _encode(dataset_schema_to_tree(schema)) != _encode(tree):
        raise ValueError("DatasetSchema tree is typed but non-canonical")
    return schema


def value_schema_fingerprint(schema: ValueSchema) -> str:
    return _tree_digest(value_schema_to_tree(schema))


def dataset_schema_fingerprint(schema: DatasetSchema) -> str:
    return _tree_digest(dataset_schema_to_tree(schema))


#: Tree keys holding one entry per coordinate rather than a structural fact.
_COORDINATE_KEYS = frozenset({"coordinates", "coordinate_labels"})


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
