"""Materialize derived Dataset snapshots without losing physical axes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._arrays import canonical_dtype
from .axis import REPEAT, AxisId, AxisSpec
from .schema import DatasetSchema, PointTable, ValueSchema
from .validity import (
    INVALID,
    VALID,
    CellValidity,
    ComponentValidity,
    DatasetComponentValidity,
    Invalid,
    Valid,
)
from .value import (
    DataBlock,
    DatasetRevisionRef,
    OwnedSnapshot,
    Value,
)

__all__ = [
    "materialize_derived_dataset",
    "materialize_scalar_dataset",
    "materialize_value_dataset",
]


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


def _single_cell_schema(cell_schema: ValueSchema) -> DatasetSchema:
    return DatasetSchema(
        AxisSpec(AxisId("derived.repeat"), "repeat", REPEAT, 1, (0,)),
        PointTable(1),
        None,
        cell_schema,
    )


def materialize_derived_dataset(
    source_ref: DatasetRevisionRef,
    values: object,
    *,
    schema: DatasetSchema,
    validity: Valid | Invalid | CellValidity | DatasetComponentValidity,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Materialize one typed derived Dataset without interpreting its domain."""

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
        ),
    )


def materialize_scalar_dataset(
    source_ref: DatasetRevisionRef,
    value: object,
    *,
    valid: bool,
    unit: str | None,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Materialize one typed scalar in the canonical (1,1,1) carrier."""

    if type(valid) is not bool:
        raise TypeError("scalar dataset valid must be bool")
    raw = np.asarray(value)
    if raw.shape not in {(), (1,)}:
        raise ValueError("scalar dataset value must contain exactly one item")
    dtype = canonical_dtype(raw.dtype)
    if dtype.kind not in "biuf":
        raise TypeError("scalar dataset value must be real numeric or boolean")
    array = np.asarray(raw, dtype=dtype).reshape(1)
    if valid and dtype.kind == "f" and not bool(np.isfinite(array[0])):
        raise ValueError("valid scalar dataset value must be finite")
    schema = _single_cell_schema(ValueSchema.scalar(dtype, unit))
    ref = _derived_reference(source_ref, schema, reference_for)
    return OwnedSnapshot(
        ref,
        DataBlock(
            ref.block_id,
            ref.revision,
            array.reshape(schema.physical_shape),
            CellValidity(np.asarray([[valid]], dtype=np.bool_)),
            schema,
        ),
    )


def materialize_value_dataset(
    source_ref: DatasetRevisionRef,
    value: Value,
    *,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Place one typed Value in the canonical single-cell carrier."""

    if not isinstance(value, Value):
        raise TypeError("single-cell dataset materialization requires Value")
    schema = _single_cell_schema(value.schema)
    ref = _derived_reference(source_ref, schema, reference_for)
    if isinstance(value.validity, Valid):
        validity = VALID
    elif isinstance(value.validity, Invalid):
        validity = INVALID
    elif isinstance(value.validity, ComponentValidity):
        validity = DatasetComponentValidity(
            value.validity.axis_ids,
            value.validity.mask.reshape(1, 1, *value.validity.mask.shape),
        )
    else:  # the closed Value validity vocabulary is enforced by Value itself
        raise TypeError("Value contains another validity representation")
    return OwnedSnapshot(
        ref,
        DataBlock(
            ref.block_id,
            ref.revision,
            value.values.reshape(schema.physical_shape),
            validity,
            schema,
        ),
    )
