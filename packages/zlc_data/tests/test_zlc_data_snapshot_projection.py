"""Public Dataset snapshot projection contracts."""

from __future__ import annotations

from zlc_data.validation import DIGEST_HEX

import numpy as np
import pytest

from zlc_data.axis import AxisId, AxisSpec, REPEAT, SPATIAL_X
from zlc_data.schema import DatasetSchema, PointTable, ValueSchema
from zlc_data.snapshot_projection import (
    materialize_derived_dataset,
    materialize_scalar_dataset,
    materialize_value_dataset,
)
from zlc_data.validity import (
    CellValidity,
    ComponentValidity,
    DatasetComponentValidity,
    VALID,
    ValidityContract,
)
from zlc_data.value import (
    BlockId,
    DatasetRevision,
    DatasetRevisionRef,
    StreamGenerationId,
    Value,
)


def _source_ref() -> DatasetRevisionRef:
    return DatasetRevisionRef(
        BlockId("source-block"),
        StreamGenerationId("source-generation"),
        "a" * DIGEST_HEX,
        DatasetRevision(7),
    )


def _reference_for(block_id: str = "derived-block"):
    def reference(schema: DatasetSchema) -> DatasetRevisionRef:
        return DatasetRevisionRef(
            BlockId(block_id),
            StreamGenerationId("source-generation"),
            schema.fingerprint,
            DatasetRevision(7),
        )

    return reference


def _scalar_schema() -> DatasetSchema:
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    return DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema.scalar(np.dtype("<f4"), "count"),
    )


def _signal_schema() -> ValueSchema:
    signal = AxisSpec(AxisId("signal"), "signal", SPATIAL_X, 2, (0, 1))
    return ValueSchema(
        (signal,),
        ValidityContract.components(signal.axis_id),
        np.dtype("<f4"),
        "count",
    )


def test_materialize_derived_dataset_preserves_schema_and_source_revision():
    schema = _scalar_schema()
    snapshot = materialize_derived_dataset(
        _source_ref(),
        np.asarray([[[2.5]]], dtype="<f4"),
        schema=schema,
        validity=VALID,
        reference_for=_reference_for(),
    )

    assert snapshot.block.schema == schema
    assert snapshot.ref.revision == DatasetRevision(7)
    assert snapshot.ref.block_id == BlockId("derived-block")
    np.testing.assert_array_equal(snapshot.block.values, [[[2.5]]])


def test_materialize_derived_dataset_rejects_bad_schema_and_reused_reference():
    with pytest.raises(TypeError, match="schema"):
        materialize_derived_dataset(
            _source_ref(),
            [[[1.0]]],
            schema=object(),  # type: ignore[arg-type]
            validity=VALID,
            reference_for=_reference_for(),
        )
    with pytest.raises(ValueError, match="reuse"):
        materialize_derived_dataset(
            _source_ref(),
            [[[1.0]]],
            schema=_scalar_schema(),
            validity=VALID,
            reference_for=_reference_for("source-block"),
        )
    with pytest.raises(ValueError, match="revision"):
        materialize_derived_dataset(
            _source_ref(),
            [[[1.0]]],
            schema=_scalar_schema(),
            validity=VALID,
            reference_for=lambda schema: DatasetRevisionRef(
                BlockId("other-block"),
                StreamGenerationId("source-generation"),
                schema.fingerprint,
                DatasetRevision(8),
            ),
        )


def test_materialize_scalar_dataset_uses_single_cell_carrier():
    snapshot = materialize_scalar_dataset(
        _source_ref(),
        np.float32(3.25),
        valid=True,
        unit="count",
        reference_for=_reference_for(),
    )

    assert snapshot.block.schema.physical_shape == (1, 1, 1)
    assert snapshot.block.schema.cell_schema.is_scalar
    assert isinstance(snapshot.block.validity, CellValidity)
    np.testing.assert_array_equal(snapshot.block.values, [[[3.25]]])
    np.testing.assert_array_equal(snapshot.block.validity.mask, [[True]])


def test_materialize_scalar_dataset_rejects_invalid_scalar_inputs():
    with pytest.raises(TypeError, match="must be bool"):
        materialize_scalar_dataset(
            _source_ref(),
            1.0,
            valid=1,  # type: ignore[arg-type]
            unit=None,
            reference_for=_reference_for(),
        )
    with pytest.raises(ValueError, match="exactly one"):
        materialize_scalar_dataset(
            _source_ref(),
            np.zeros(2),
            valid=True,
            unit=None,
            reference_for=_reference_for(),
        )
    with pytest.raises(ValueError, match="finite"):
        materialize_scalar_dataset(
            _source_ref(),
            np.nan,
            valid=True,
            unit=None,
            reference_for=_reference_for(),
        )


def test_materialize_value_dataset_preserves_named_component_validity():
    value_schema = _signal_schema()
    value = Value(
        np.asarray([10.0, 20.0], dtype="<f4"),
        ComponentValidity(
            (value_schema.data_axes[0].axis_id,),
            np.array([True, False]),
        ),
        value_schema,
    )
    snapshot = materialize_value_dataset(
        _source_ref(),
        value,
        reference_for=_reference_for(),
    )

    assert snapshot.block.schema.cell_schema == value_schema
    assert isinstance(snapshot.block.validity, DatasetComponentValidity)
    assert snapshot.block.validity.axis_ids == (value_schema.data_axes[0].axis_id,)
    np.testing.assert_array_equal(snapshot.block.values, [[[10.0, 20.0]]])
    np.testing.assert_array_equal(
        snapshot.block.validity.mask,
        [[[True, False]]],
    )


def test_materialize_value_dataset_rejects_non_value_and_bad_reference():
    with pytest.raises(TypeError, match="requires Value"):
        materialize_value_dataset(
            _source_ref(),
            object(),  # type: ignore[arg-type]
            reference_for=_reference_for(),
        )
    with pytest.raises(ValueError, match="schema"):
        materialize_value_dataset(
            _source_ref(),
            Value(
                np.asarray([1.0, 2.0], dtype="<f4"),
                VALID,
                _signal_schema(),
            ),
            reference_for=lambda schema: DatasetRevisionRef(
                BlockId("derived-block"),
                StreamGenerationId("source-generation"),
                "b" * DIGEST_HEX,
                DatasetRevision(7),
            ),
        )
