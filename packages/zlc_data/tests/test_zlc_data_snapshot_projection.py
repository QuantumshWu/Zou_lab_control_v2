"""Public Dataset snapshot projection contracts."""

from __future__ import annotations

from zlc_data.validation import DIGEST_HEX

import numpy as np
import pytest

from zlc_data.axis import AxisId, AxisSpec, REPEAT
from zlc_data.schema import DatasetSchema, PointTable, ValueSchema
from zlc_data.snapshot_projection import materialize_derived_dataset
from zlc_data.validity import VALID
from zlc_data.value import (
    BlockId,
    DatasetRevision,
    DatasetRevisionRef,
    StreamGenerationId,
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
