"""Direct immutable Dataset snapshot construction contracts."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointTable,
    REPEAT,
    SPATIAL_X,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    compact_dataset_validity,
    expand_snapshot_validity,
    owned_snapshot_from_arrays,
)


def _schema(*, component_validity: bool) -> DatasetSchema:
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2, (0, 1))
    component = AxisSpec(
        AxisId("component"),
        "component",
        SPATIAL_X,
        3,
        (0, 1, 2),
    )
    contract = (
        ValidityContract.components(component.axis_id)
        if component_validity
        else ValidityContract.value()
    )
    return DatasetSchema(
        repeat,
        PointTable(2),
        None,
        ValueSchema((component,), contract, np.dtype("<f4")),
    )


def test_direct_constructor_matches_explicit_runtime_construction():
    schema = _schema(component_validity=True)
    values = np.arange(12, dtype="<f4").reshape(schema.physical_shape)
    dense_validity = np.asarray(
        [
            [[True, False, True], [False, True, True]],
            [[True, True, False], [False, False, True]],
        ],
        dtype=np.bool_,
    )
    revision = DatasetRevision(7)
    block_id = BlockId("direct-block")
    generation = StreamGenerationId("direct-generation")

    direct = owned_snapshot_from_arrays(
        schema,
        values,
        revision,
        validity=dense_validity,
        block_id=block_id,
        stream_generation=generation,
    )
    runtime_block = DataBlock(
        block_id,
        revision,
        values,
        compact_dataset_validity(dense_validity, schema),
        schema,
    )
    runtime = OwnedSnapshot(runtime_block.ref(generation), runtime_block)

    assert direct.exactly_equals(runtime)
    changed = owned_snapshot_from_arrays(
        schema,
        values + 1,
        revision,
        validity=dense_validity,
        block_id=block_id,
        stream_generation=generation,
    )
    assert not direct.exactly_equals(changed)
    assert not direct.exactly_equals(object())
    np.testing.assert_array_equal(
        expand_snapshot_validity(direct),
        expand_snapshot_validity(runtime),
    )
    assert isinstance(direct.block.validity, type(runtime.block.validity))


def test_cell_schema_and_axes_form_constructs_a_single_carrier():
    schema = _schema(component_validity=False)
    snapshot = owned_snapshot_from_arrays(
        schema.cell_schema,
        np.ones(schema.physical_shape, dtype="<f4"),
        3,
        repeat_axis=schema.repeat_axis,
        point_table=schema.point_table,
        block_id="cell-schema-block",
        stream_generation="cell-schema-generation",
    )

    assert snapshot.block.schema == schema
    assert snapshot.ref.revision == DatasetRevision(3)
    assert snapshot.ref.block_id == BlockId("cell-schema-block")
    assert snapshot.ref.stream_generation == StreamGenerationId(
        "cell-schema-generation"
    )
    assert snapshot.block.validity is not None
    np.testing.assert_array_equal(
        snapshot.expanded_validity(),
        np.ones(schema.physical_shape, dtype=np.bool_),
    )


def test_direct_constructor_defaults_identity_and_all_validity():
    schema = _schema(component_validity=False)
    snapshot = owned_snapshot_from_arrays(
        schema=schema,
        values=np.zeros(schema.physical_shape, dtype="<f4"),
        revision=0,
    )

    assert isinstance(snapshot.ref.block_id, BlockId)
    assert snapshot.ref.stream_generation == StreamGenerationId("direct")
    assert snapshot.ref.schema_fingerprint == schema.fingerprint
    assert snapshot.expanded_validity().flags.writeable is False
    np.testing.assert_array_equal(
        snapshot.expanded_validity(),
        np.ones(schema.physical_shape, dtype=np.bool_),
    )


def test_direct_constructor_rejects_uncompactable_dense_validity():
    schema = _schema(component_validity=False)
    mask = np.asarray(
        [
            [[True, False, True], [True, True, True]],
            [[True, True, True], [True, True, True]],
        ],
        dtype=np.bool_,
    )

    with pytest.raises(RuntimeError, match="varies along an axis"):
        owned_snapshot_from_arrays(
            schema,
            np.zeros(schema.physical_shape, dtype="<f4"),
            0,
            validity=mask,
        )


def test_direct_constructor_rejects_wrong_dense_validity_shape():
    schema = _schema(component_validity=True)

    with pytest.raises(ValueError, match="disagrees with schema"):
        owned_snapshot_from_arrays(
            schema,
            np.zeros(schema.physical_shape, dtype="<f4"),
            0,
            validity=np.ones((1, 1), dtype=np.bool_),
        )


@pytest.mark.parametrize(
    "validity",
    (
        np.ones((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2, 3), dtype=np.float32),
    ),
)
def test_direct_constructor_rejects_numeric_truthiness_validity(validity):
    schema = _schema(component_validity=True)

    with pytest.raises(TypeError, match="validity mask dtype must be bool"):
        owned_snapshot_from_arrays(
            schema,
            np.zeros(schema.physical_shape, dtype="<f4"),
            0,
            validity=validity,
        )


def test_snapshot_validity_expander_rejects_other_objects():
    with pytest.raises(TypeError, match="OwnedSnapshot"):
        expand_snapshot_validity(object())  # type: ignore[arg-type]
