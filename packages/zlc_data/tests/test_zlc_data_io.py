"""Strict NPZ round-trips and rejection of incomplete archives."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import warnings
import zipfile

import numpy as np
import pytest

from zlc_data.axis import AxisId, AxisSpec, REPEAT, SITE, SPATIAL_X
from zlc_data.figure_archive import (
    FIGURE_SCHEMA,
    read_archive,
    read_dataset,
    write_figure_archive,
)
from zlc_data.io import NPZFormatError, load_npz, save_npz, snapshot_manifest
from zlc_data.schema import (
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    ValueSchema,
)
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


def _write_npz(path: Path, snapshot: OwnedSnapshot) -> None:
    with path.open("wb") as stream:
        save_npz(stream, snapshot)


def test_npz_writer_requires_caller_owned_binary_io(tmp_path: Path):
    path = tmp_path / "snapshot.npz"
    path.write_bytes(b"previous complete file")

    with pytest.raises(TypeError, match="writable binary IO"):
        save_npz(path, _snapshot())  # type: ignore[arg-type]

    assert path.read_bytes() == b"previous complete file"


def test_npz_round_trip_preserves_owned_snapshot_and_masks(tmp_path: Path):
    source = _snapshot(CellValidity(np.array([[True, False], [False, True]])))
    path = tmp_path / "snapshot.npz"
    _write_npz(path, source)

    restored = load_npz(path)
    assert restored.ref == source.ref
    assert restored.block.schema == source.block.schema
    np.testing.assert_array_equal(restored.block.values, source.block.values)
    np.testing.assert_array_equal(restored.block.validity.mask, source.block.validity.mask)


def test_npz_round_trip_preserves_canonical_coordinates_and_display_labels(
    tmp_path: Path,
):
    repeat = AxisSpec(
        AxisId("capture.repeat"),
        "Shot",
        REPEAT,
        2,
        ("shot_dark", "shot_bright"),
        coordinate_labels=("Dark", "Bright"),
    )
    sites = PointColumn(
        AxisId("calibration.site"),
        "Site",
        SITE,
        PointColumn.TEXT,
        ("site_0001", "site_0002"),
        coordinate_labels=("1", "2"),
    )
    schema = DatasetSchema(
        repeat,
        PointTable(2, (sites,)),
        None,
        ValueSchema.scalar(np.dtype("<f4"), "count"),
    )
    block = DataBlock(
        BlockId("coordinate-label-io-block"),
        DatasetRevision(0),
        np.arange(4, dtype="<f4").reshape(schema.physical_shape),
        VALID,
        schema,
    )
    source = OwnedSnapshot(
        block.ref(StreamGenerationId("coordinate-label-generation")),
        block,
    )
    path = tmp_path / "coordinate-labels.npz"

    _write_npz(path, source)
    restored = load_npz(path)

    assert restored.block.schema.repeat_axis.coordinates == ("shot_dark", "shot_bright")
    assert restored.block.schema.repeat_axis.coordinate_labels == ("Dark", "Bright")
    restored_sites = restored.block.schema.point_table.column(AxisId("calibration.site"))
    assert restored_sites.values == ("site_0001", "site_0002")
    assert restored_sites.coordinate_labels == ("1", "2")


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
    _write_npz(path, source)

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

    _write_npz(path, source)
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
    _write_npz(original, source)
    arrays = _archive_members(original)
    arrays.pop("values")
    np.savez_compressed(malformed, **arrays)
    with pytest.raises(NPZFormatError, match="missing array|members mismatch"):
        load_npz(malformed)

    missing_ref = tmp_path / "missing-ref-fingerprint.npz"
    arrays = _archive_members(original)
    manifest = json.loads(str(arrays["manifest"].item()))
    manifest["ref"].pop("schema_fingerprint")
    arrays["manifest"] = np.asarray(json.dumps(manifest, sort_keys=True))
    np.savez_compressed(missing_ref, **arrays)
    with pytest.raises(NPZFormatError, match="missing schema_fingerprint"):
        load_npz(missing_ref)


def test_npz_extra_member_is_rejected(tmp_path: Path):
    source = _snapshot()
    original = tmp_path / "original.npz"
    malformed = tmp_path / "extra-member.npz"
    _write_npz(original, source)
    arrays = _archive_members(original)
    arrays["unexpected"] = np.asarray(3)
    np.savez_compressed(malformed, **arrays)
    with pytest.raises(NPZFormatError, match="members mismatch"):
        load_npz(malformed)


def test_npz_missing_cell_validity_member_is_rejected(tmp_path: Path):
    source = _snapshot(CellValidity(np.ones((2, 2), dtype=bool)))
    original = tmp_path / "original.npz"
    malformed = tmp_path / "missing-validity.npz"
    _write_npz(original, source)
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
    _write_npz(original, _snapshot())
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
    _write_npz(original, _snapshot())
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
    _write_npz(original, _snapshot())
    arrays = _archive_members(original)
    manifest = json.loads(str(arrays["manifest"].item()))
    manifest["validity"] = {"kind": "not-a-validity"}
    arrays["manifest"] = np.asarray(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )
    np.savez_compressed(malformed, **arrays)

    with pytest.raises(NPZFormatError, match="invalid validity kind"):
        load_npz(malformed)


def _figure_stream(name: str, *, arrays, sections) -> BytesIO:
    stream = BytesIO()
    write_figure_archive(stream, name, arrays=arrays, sections=sections)
    stream.seek(0)
    return stream


def _figure_members(stream: BytesIO) -> dict[str, np.ndarray]:
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _figure_payload(members: dict[str, np.ndarray]) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **members)
    return stream.getvalue()


def test_figure_archive_round_trip_validates_version_members_and_dataset_shape():
    snapshot = _snapshot(CellValidity(np.array([[True, False], [False, True]])))
    stream = _figure_stream(
        "strict figure",
        arrays={"data": snapshot, "trace": np.arange(3, dtype="<f4")},
        sections={"panel": {"kind": "image"}},
    )

    info, arrays = read_archive(stream)

    assert info["schema"] == FIGURE_SCHEMA
    assert info["version"] == 2
    assert set(info["members"]) == {"data", "data.validity", "trace"}
    assert set(arrays) == {"data", "data.validity", "trace"}
    assert read_dataset(info, arrays, "data").exactly_equals(snapshot)


def test_figure_v1_migrates_only_exact_legacy_dataset_fields():
    repeat = AxisSpec(AxisId("legacy.repeat"), "repeat", REPEAT, 2, (0, 1))
    sites = PointColumn(
        AxisId("legacy.site"), "site", SITE, PointColumn.TEXT, ("a", "b")
    )
    schema = DatasetSchema(
        repeat,
        PointTable(2, (sites,)),
        None,
        ValueSchema.scalar(np.dtype("<f4")),
    )
    block = DataBlock(
        BlockId("legacy-block"),
        DatasetRevision(2),
        np.arange(4, dtype="<f4").reshape(schema.physical_shape),
        VALID,
        schema,
    )
    source = OwnedSnapshot(block.ref(StreamGenerationId("legacy-generation")), block)
    stored: dict[str, np.ndarray] = {}
    manifest = snapshot_manifest(source, stored, values_key="data")
    manifest["ref"].pop("schema_fingerprint")

    def without_labels(value):
        if isinstance(value, dict):
            result = {key: without_labels(item) for key, item in value.items()}
            if result.get("schema") in ("zlc_data.AxisSpec", "zlc_data.PointColumn"):
                result.pop("coordinate_labels")
            return result
        if isinstance(value, list):
            return [without_labels(item) for item in value]
        return value

    legacy_info = {
        "schema": "zlc.figure/v1",
        "name": "legacy figure",
        "sections": {"dataset": {"data": without_labels(manifest)}},
    }
    stream = BytesIO()
    np.savez_compressed(
        stream,
        info=np.asarray(json.dumps(legacy_info, separators=(",", ":"))),
        **stored,
    )
    stream.seek(0)

    info, arrays = read_archive(stream)

    assert info["schema"] == FIGURE_SCHEMA
    assert info["version"] == 2
    assert info["members"] == {
        "data": {"dtype": "<f4", "shape": list(source.block.values.shape)}
    }
    assert read_dataset(info, arrays, "data").exactly_equals(source)

    empty_member = BytesIO()
    np.savez_compressed(
        empty_member,
        info=np.asarray(json.dumps(legacy_info)),
        **{"": np.zeros(1, dtype="|i1")},
    )
    empty_member.seek(0)
    with pytest.raises(ValueError, match="member names must be non-empty"):
        read_archive(empty_member)

    legacy_info["unexpected"] = True
    malformed = BytesIO()
    np.savez_compressed(
        malformed,
        info=np.asarray(json.dumps(legacy_info)),
        **stored,
    )
    malformed.seek(0)
    with pytest.raises(ValueError, match="legacy figure metadata keys mismatch"):
        read_archive(malformed)

    object_member = BytesIO()
    np.savez_compressed(
        object_member,
        info=np.asarray(json.dumps(legacy_info)),
        data=np.asarray([object()], dtype=object),
    )
    object_member.seek(0)
    with pytest.raises(ValueError, match="legacy figure metadata keys mismatch"):
        read_archive(object_member)


def test_figure_writer_preplans_snapshot_member_namespace():
    snapshot = _snapshot(CellValidity(np.array([[True, False], [False, True]])))
    stream = BytesIO()

    with pytest.raises(ValueError, match="member name collision.*data.validity"):
        write_figure_archive(
            stream,
            "collision",
            arrays={"data": snapshot, "data.validity": np.ones((1,), dtype=bool)},
            sections={},
        )
    assert stream.getvalue() == b"", "validation failure wrote a partial archive"


def test_large_figure_stream_does_not_allocate_archive_sized_python_bytes(
    tmp_path,
) -> None:
    import tracemalloc

    def measured(size_mib: int) -> tuple[int, int]:
        values = np.random.default_rng(size_mib).integers(
            0,
            256,
            size=size_mib * 1024 * 1024,
            dtype=np.uint8,
        )
        target = tmp_path / f"large-{size_mib}.npz"
        with target.open("wb") as stream:
            tracemalloc.start()
            write_figure_archive(
                stream,
                f"large-{size_mib}",
                arrays={"frame": values},
                sections={},
            )
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        return peak, target.stat().st_size

    small_peak, small_archive = measured(16)
    large_peak, large_archive = measured(64)
    assert large_archive > small_archive * 3.5
    assert large_peak < small_peak * 1.25, (small_peak, large_peak)


@pytest.mark.parametrize(
    "bad",
    [
        object(),
        {1: "numeric key"},
        {"items": {1, 2}},
        ("tuple", "is not JSON"),
        np.int64(1),
    ],
)
def test_figure_writer_rejects_unknown_or_lossy_metadata(bad):
    with pytest.raises(TypeError, match="metadata"):
        write_figure_archive(
            BytesIO(),
            "bad metadata",
            arrays={"trace": np.arange(2)},
            sections={"bad": bad},
        )


@pytest.mark.parametrize(
    "field,value",
    (("schema", "other-format"), ("version", 99)),
)
def test_figure_reader_rejects_wrong_format_or_version(field, value):
    members = _figure_members(
        _figure_stream("format", arrays={"trace": np.arange(2)}, sections={})
    )
    info = json.loads(str(members["info"].item()))
    info[field] = value
    members["info"] = np.asarray(json.dumps(info, sort_keys=True))

    with pytest.raises(ValueError, match="unsupported figure format"):
        read_archive(BytesIO(_figure_payload(members)))


def test_figure_reader_rejects_extra_and_shape_changed_members():
    original = _figure_members(
        _figure_stream("members", arrays={"trace": np.arange(2)}, sections={})
    )
    members = dict(original)
    members["extra"] = np.arange(1)
    with pytest.raises(ValueError, match="members mismatch"):
        read_archive(BytesIO(_figure_payload(members)))

    members = dict(original)
    members.pop("trace")
    with pytest.raises(ValueError, match="members mismatch"):
        read_archive(BytesIO(_figure_payload(members)))

    members = dict(original)
    members["trace"] = np.arange(3)
    with pytest.raises(ValueError, match="shape"):
        read_archive(BytesIO(_figure_payload(members)))

    members = dict(original)
    members["trace"] = np.arange(2, dtype=np.float64)
    with pytest.raises(ValueError, match="dtype"):
        read_archive(BytesIO(_figure_payload(members)))


def test_figure_reader_rejects_duplicate_zip_members():
    stream = _figure_stream(
        "duplicates", arrays={"trace": np.arange(2)}, sections={}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(stream, "a") as archive:
            encoded = archive.read("trace.npy")
            archive.writestr("trace.npy", encoded)

    with pytest.raises(ValueError, match="duplicate.*trace"):
        read_archive(BytesIO(stream.getvalue()))


def test_figure_reader_rejects_duplicate_keys_and_nonfinite_metadata():
    members = _figure_members(
        _figure_stream("metadata", arrays={"trace": np.arange(2)}, sections={})
    )
    text = str(members["info"].item())
    duplicate = text.replace('"version":2', '"version":2,"version":2', 1)
    assert duplicate != text
    members["info"] = np.asarray(duplicate)
    with pytest.raises(ValueError, match="duplicate metadata key"):
        read_archive(BytesIO(_figure_payload(members)))

    nonfinite = text.replace('"sections":{}', '"sections":{"bad":NaN}', 1)
    assert nonfinite != text
    members["info"] = np.asarray(nonfinite)
    with pytest.raises(ValueError, match="non-finite metadata"):
        read_archive(BytesIO(_figure_payload(members)))


def test_figure_reader_validates_embedded_dataset_before_returning():
    members = _figure_members(
        _figure_stream("dataset", arrays={"data": _snapshot()}, sections={})
    )
    members["data"] = np.zeros((1,), dtype="<f4")

    with pytest.raises(ValueError, match="shape"):
        read_archive(BytesIO(_figure_payload(members)))

    members = _figure_members(
        _figure_stream("dataset", arrays={"data": _snapshot()}, sections={})
    )
    info = json.loads(str(members["info"].item()))
    manifest = info["sections"]["dataset"]["data"]
    manifest["ref"]["schema_fingerprint"] = "unexpected"
    members["info"] = np.asarray(json.dumps(info, sort_keys=True))
    with pytest.raises(ValueError, match="must not repeat schema_fingerprint"):
        read_archive(BytesIO(_figure_payload(members)))
