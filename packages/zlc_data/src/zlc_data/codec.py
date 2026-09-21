"""Canonical codecs owned by the zlc_data bounded context."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._tree import digest as _tree_digest
from .validation import canonical_text as _text, exact_mapping as _exact_map, integer as _integer

from .axis import PRIMARY_INDEX, AxisId, AxisRoleId, AxisSpec, CoordinateFrameId, canonical_coordinate_scalar
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
    return value


def _coordinate_tree(values: Any) -> list[Any]:
    # AxisSpec already validated this immutable vector. JSON's canonical
    # numeric spelling only needs integral floats changed to Python ints.
    if isinstance(values, np.ndarray):
        source = values.tolist()
        return [int(value) if value.is_integer() else value for value in source] if values.dtype.kind == "f" else source
    return list(values)


def _read_coordinates(values: Any, field: str) -> list[Any] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError(f"AxisSpec {field} must be a list or null")
    for value in values:
        scalar = value.item() if isinstance(value, np.generic) else value
        normalized = canonical_coordinate_scalar(scalar)
        if type(scalar) is not type(normalized) or scalar != normalized:
            raise ValueError("AxisSpec tree is typed but non-canonical")
    return values


def axis_to_tree(axis: AxisSpec, *, structure: bool = False) -> dict[str, Any]:
    return {
        "schema": AXIS_SCHEMA,
        "axis_id": axis.axis_id.value,
        "name": axis.name,
        "role": axis.role.value,
        "size": axis.size,
        "coordinates": None if axis.coordinates is None else len(axis.coordinates) if structure else _coordinate_tree(axis.coordinates),
        "unit": axis.unit,
        "coordinate_frame": None
        if axis.coordinate_frame is None
        else axis.coordinate_frame.value,
        "index_origin": axis.index_origin,
        "coordinate_labels": None
        if axis.coordinate_labels is None
        else len(axis.coordinate_labels) if structure else list(axis.coordinate_labels),
        **({"coordinate_of": axis.coordinate_of.value} if axis.coordinate_of is not None else {}),
        **({"coordinate_origins": len(axis.coordinate_origins) if structure else _coordinate_tree(axis.coordinate_origins)}
           if axis.coordinate_origins is not None else {}),
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
        } | ({"coordinate_of"} if isinstance(tree, dict) and "coordinate_of" in tree else set())
        | ({"coordinate_origins"} if isinstance(tree, dict) and "coordinate_origins" in tree else set()),
        AXIS_SCHEMA,
    )
    coordinates = _read_coordinates(data["coordinates"], "coordinates")
    origins = _read_coordinates(data.get("coordinate_origins"), "coordinate_origins")
    if ("coordinate_of" in data and data["coordinate_of"] is None) or ("coordinate_origins" in data and origins is None):
        raise ValueError("AxisSpec tree is typed but non-canonical")
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
        coordinate_of=None if data.get("coordinate_of") is None else AxisId(data["coordinate_of"]),
        coordinate_origins=origins,
    )
    return axis


def domain_to_tree(domain: DomainSpec, *, structure: bool = False) -> dict[str, Any]:
    if not isinstance(domain, DomainSpec):
        raise TypeError("domain must be DomainSpec")
    sliding = {axis.axis_id for axis in domain.axes if axis.role == PRIMARY_INDEX} if structure else set()
    axes = []
    for axis in domain.axes:
        tree = axis_to_tree(axis, structure=structure)
        if sliding and (axis.axis_id in sliding or axis.coordinate_of in sliding or axis.coordinate_origins is not None):
            tree["size"] = tree["index_origin"] = "sliding"
            if axis.coordinate_origins is None and tree["coordinates"] is not None:
                tree["coordinates"] = "sliding"
            if tree["coordinate_labels"] is not None:
                tree["coordinate_labels"] = "sliding"
            if axis.coordinate_origins is not None:
                tree["coordinate_origins"] = "sliding"
        axes.append(tree)
    return {
        "schema": DOMAIN_SCHEMA,
        "shape": ["sliding" if sliding and (domain.axis_codes is not None or domain.axes[index].axis_id in sliding) else size
                  for index, size in enumerate(domain.shape)],
        "axes": axes,
        "axis_codes": None
        if domain.axis_codes is None
        else "sliding" if sliding else [
            {"range": [codes.start, codes.stop, codes.step]} if isinstance(codes, range) else codes.tolist()
            for codes in domain.axis_codes
        ],
        **({"axis_code_repeats": [list(pair) for pair in domain.axis_code_repeats]}
           if domain.axis_code_repeats is not None and not sliding else {}),
    }


def domain_from_tree(tree: Any) -> DomainSpec:
    data = _exact_map(
        tree,
        {"schema", "shape", "axes", "axis_codes"}
        | ({"axis_code_repeats"} if isinstance(tree, dict) and "axis_code_repeats" in tree else set()),
        DOMAIN_SCHEMA,
    )
    shape = data["shape"]
    axes = data["axes"]
    codes = data["axis_codes"]
    if not isinstance(shape, list):
        raise ValueError("DomainSpec shape must be a list")
    if not isinstance(axes, list):
        raise ValueError("DomainSpec axes must be a list")
    if codes is not None and not isinstance(codes, list):
        raise ValueError("DomainSpec axis_codes must be a list or null")
    mappings = None
    if codes is not None:
        mappings = []
        for item in codes:
            if isinstance(item, list):
                mappings.append(tuple(item))
            else:
                entry = _exact_map(item, {"range"}, "DomainSpec axis range", discriminator=None)
                parts = entry["range"]
                if not isinstance(parts, list) or len(parts) != 3:
                    raise ValueError("axis code range needs start, stop and step")
                mappings.append(range(*(_integer(value, "axis code range") for value in parts)))
    repeats = data.get("axis_code_repeats")
    if "axis_code_repeats" in data and (
        not isinstance(repeats, list) or any(not isinstance(pair, list) for pair in repeats)
    ):
        raise ValueError("axis_code_repeats must be a list of pairs")
    domain = DomainSpec(
        shape=tuple(shape),
        axes=tuple(axis_from_tree(axis) for axis in axes),
        axis_codes=None if mappings is None else tuple(mappings),
        axis_code_repeats=None if repeats is None else tuple(tuple(pair) for pair in repeats),
    )
    if "axis_code_repeats" in data and domain.axis_code_repeats is None:
        raise ValueError("DomainSpec tree is typed but non-canonical")
    if repeats is not None and repeats != [list(pair) for pair in domain.axis_code_repeats]:
        raise ValueError("DomainSpec tree is typed but non-canonical")
    if codes is not None:
        for raw, normalized in zip(codes, domain.axis_codes, strict=True):
            if isinstance(raw, dict):
                if not isinstance(normalized, range) or raw["range"] != [normalized.start, normalized.stop, normalized.step]:
                    raise ValueError("DomainSpec tree is typed but non-canonical")
            elif isinstance(normalized, range):
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
        **({"name": schema.name} if schema.name is not None else {}),
    }


def value_schema_from_tree(tree: Any) -> ValueSchema:
    fields = {"schema", "validity_contract", "dtype", "value_unit"}
    if isinstance(tree, dict) and "name" in tree:
        fields.add("name")
    data = _exact_map(
        tree,
        fields,
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
    if "name" in data and data["name"] is None:
        raise ValueError("ValueSchema tree is typed but non-canonical")
    schema = ValueSchema(
        validity_contract=contract,
        dtype=np.dtype(_text(data["dtype"], "dtype")),
        value_unit=unit,
        name=data.get("name"),
    )
    if data["dtype"] != schema.dtype.str:
        raise ValueError("ValueSchema tree is typed but non-canonical")
    return schema


def dataset_schema_to_tree(schema: DatasetSchema, *, structure: bool = False) -> dict[str, Any]:
    return {
        "schema": DATASET_SCHEMA,
        "repeat_domain": domain_to_tree(schema.repeat_domain, structure=structure),
        "point_domain": domain_to_tree(schema.point_domain, structure=structure),
        "cell_domain": domain_to_tree(schema.cell_domain, structure=structure),
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
    return schema


def value_schema_fingerprint(schema: ValueSchema) -> str:
    return _tree_digest(value_schema_to_tree(schema))


def dataset_schema_fingerprint(schema: DatasetSchema) -> str:
    return _tree_digest(dataset_schema_to_tree(schema))


def dataset_schema_structure_fingerprint(schema: DatasetSchema) -> str:
    """What this schema IS, without what its coordinates currently read.

    The full fingerprint includes every coordinate value, which is right for
    "is this the same dataset".  It is the wrong question for "is this the
    same world an interaction was started in": a bounded shot history slides
    its own coordinates forward by design -- every shot renames them -- while
    the axes, their roles and units stand still.  Judging that by the full
    fingerprint made every shot look like a new geometry.

    The history's DEPTH goes with its coordinates, for the same reason: a
    window that is still filling grows by one on every shot, and a panel
    holding a viewport or an open drag would have lost it on each one.
    """

    return _tree_digest(dataset_schema_to_tree(schema, structure=True))
