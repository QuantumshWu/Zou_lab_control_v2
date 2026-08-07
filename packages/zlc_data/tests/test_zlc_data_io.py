"""Strict NPZ round-trips and rejection of incomplete archives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zlc_data.axis import AxisId, AxisSpec, REPEAT, SPATIAL_X
from zlc_data.io import NPZFormatError, load_npz, save_npz
from zlc_data.schema import DatasetSchema, GridTopology, PointTable, ValueSchema
from zlc_data.validity import (
    CellValidity,
    DatasetComponentValidity,
    VALID,
    ValidityContract,
)
from zlc_data.value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    OwnedSnapshot,
    StreamGenerationId,
)


def _snapshot(validity=VALID) -> OwnedSnapshot:
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2, (0, 1))
    component = AxisSpec(AxisId("component"), "component", SPATIAL_X, 3, (0, 1, 2))
    schema = DatasetSchema(
        repeat,
        PointTable(2),
        None,
        ValueSchema((component,), ValidityContract.value(), np.dtype("<f4")),
    )
    block = DataBlock(
        BlockId("io-block"),
        DatasetRevision(4),
        np.arange(12, dtype="<f4").reshape(schema.physical_shape),
        validity,
        schema,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("io-generation")), block)


def _archive_members(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def test_npz_round_trip_preserves_owned_snapshot_and_masks(tmp_path: Path):
    source = _snapshot(CellValidity(np.array([[True, False], [False, True]])))
    path = tmp_path / "snapshot.npz"
    save_npz(path, source)

    restored = load_npz(path)
    assert restored.ref == source.ref
    assert restored.block.schema == source.block.schema
    np.testing.assert_array_equal(restored.block.values, source.block.values)
    np.testing.assert_array_equal(restored.block.validity.mask, source.block.validity.mask)


def test_npz_round_trip_preserves_dataset_component_masks(tmp_path: Path):
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2, (0, 1))
    component = AxisSpec(AxisId("component"), "component", SPATIAL_X, 3, (0, 1, 2))
    schema = DatasetSchema(
        repeat,
        PointTable(2),
        None,
        ValueSchema(
            (component,),
            ValidityContract.components(component.axis_id),
            np.dtype("<f4"),
        ),
    )
    validity = DatasetComponentValidity(
        (component.axis_id,),
        np.array(
            [
                [[True, False, True], [False, True, True]],
                [[True, True, False], [False, False, True]],
            ]
        ),
    )
    block = DataBlock(
        BlockId("component-io-block"),
        DatasetRevision(0),
        np.arange(12, dtype="<f4").reshape(schema.physical_shape),
        validity,
        schema,
    )
    source = OwnedSnapshot(block.ref(StreamGenerationId("component-generation")), block)
    path = tmp_path / "component.npz"
    save_npz(path, source)

    restored = load_npz(path)
    np.testing.assert_array_equal(restored.block.validity.mask, validity.mask)


def test_npz_round_trip_preserves_topology_dimensions_without_point_columns(
    tmp_path: Path,
):
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    component = AxisSpec(AxisId("component"), "component", SPATIAL_X, 3, (0, 1, 2))
    topology = GridTopology(
        (AxisId("b.x"),),
        ((0, 1),),
        ((0,), (1,)),
    )
    schema = DatasetSchema(
        repeat,
        PointTable(2),
        topology,
        ValueSchema((component,), ValidityContract.value(), np.dtype("<f4")),
    )
    block = DataBlock(
        BlockId("topology-only-io-block"),
        DatasetRevision(0),
        np.arange(6, dtype="<f4").reshape(schema.physical_shape),
        CellValidity(np.array([[True, False]])),
        schema,
    )
    source = OwnedSnapshot(
        block.ref(StreamGenerationId("topology-only-generation")),
        block,
    )
    path = tmp_path / "topology-only.npz"

    save_npz(path, source)
    restored = load_npz(path)

    assert restored.block.schema == schema
    assert restored.block.schema.point_table.columns == ()
    assert restored.block.schema.grid_topology == topology
    np.testing.assert_array_equal(restored.block.values, source.block.values)
    np.testing.assert_array_equal(restored.block.validity.mask, source.block.validity.mask)


def test_npz_missing_manifest_is_rejected(tmp_path: Path):
    path = tmp_path / "missing-manifest.npz"
    np.savez_compressed(path, values=np.zeros((1, 1, 1), dtype=np.float32))
    with pytest.raises(NPZFormatError, match="manifest"):
        load_npz(path)


def test_npz_missing_value_member_is_rejected(tmp_path: Path):
    source = _snapshot()
    original = tmp_path / "original.npz"
    malformed = tmp_path / "missing-values.npz"
    save_npz(original, source)
    arrays = _archive_members(original)
    arrays.pop("values")
    np.savez_compressed(malformed, **arrays)
    with pytest.raises(NPZFormatError, match="missing array|members mismatch"):
        load_npz(malformed)


def test_npz_extra_member_is_rejected(tmp_path: Path):
    source = _snapshot()
    original = tmp_path / "original.npz"
    malformed = tmp_path / "extra-member.npz"
    save_npz(original, source)
    arrays = _archive_members(original)
    arrays["unexpected"] = np.asarray(3)
    np.savez_compressed(malformed, **arrays)
    with pytest.raises(NPZFormatError, match="members mismatch"):
        load_npz(malformed)


def test_npz_missing_cell_validity_member_is_rejected(tmp_path: Path):
    source = _snapshot(CellValidity(np.ones((2, 2), dtype=bool)))
    original = tmp_path / "original.npz"
    malformed = tmp_path / "missing-validity.npz"
    save_npz(original, source)
    arrays = _archive_members(original)
    arrays.pop("validity")
    np.savez_compressed(malformed, **arrays)
    with pytest.raises(NPZFormatError, match="missing array|members mismatch"):
        load_npz(malformed)


@pytest.mark.parametrize("field, value", [("format", "other-format"), ("version", 99)])
def test_npz_rejects_manifest_format_and_version_changes(
    tmp_path: Path,
    field: str,
    value: object,
):
    original = tmp_path / "original.npz"
    malformed = tmp_path / f"bad-{field}.npz"
    save_npz(original, _snapshot())
    arrays = _archive_members(original)
    manifest = json.loads(str(arrays["manifest"].item()))
    manifest[field] = value
    arrays["manifest"] = np.asarray(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )
    np.savez_compressed(malformed, **arrays)

    with pytest.raises(NPZFormatError, match="unsupported data format"):
        load_npz(malformed)


def test_npz_rejects_duplicate_manifest_json_keys(tmp_path: Path):
    original = tmp_path / "original.npz"
    malformed = tmp_path / "duplicate-key.npz"
    save_npz(original, _snapshot())
    arrays = _archive_members(original)
    manifest_text = str(arrays["manifest"].item())
    duplicate = manifest_text.replace('"version":1', '"version":1,"version":1', 1)
    assert duplicate != manifest_text
    arrays["manifest"] = np.asarray(duplicate)
    np.savez_compressed(malformed, **arrays)

    with pytest.raises(NPZFormatError, match="duplicate key"):
        load_npz(malformed)


def test_npz_rejects_unknown_validity_kind(tmp_path: Path):
    original = tmp_path / "original.npz"
    malformed = tmp_path / "bad-validity-kind.npz"
    save_npz(original, _snapshot())
    arrays = _archive_members(original)
    manifest = json.loads(str(arrays["manifest"].item()))
    manifest["validity"] = {"kind": "not-a-validity"}
    arrays["manifest"] = np.asarray(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )
    np.savez_compressed(malformed, **arrays)

    with pytest.raises(NPZFormatError, match="invalid validity kind"):
        load_npz(malformed)
