"""Strict NPZ round-trips and rejection of incomplete archives."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import warnings
import zipfile
import zlib

import numpy as np
import pytest

from zlc_data.axis import AxisId, AxisSpec, REPEAT, SCAN_POINT, SITE, SPATIAL_X
from zlc_data.figure_archive import (
    FIGURE_SCHEMA,
    read_archive,
    write_figure_archive,
)
from zlc_data.io import NPZFormatError, load_npz, save_npz, snapshot_manifest
from zlc_data.schema import (
    DatasetSchema,
    DomainSpec,
    SCALAR_DOMAIN,
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
    point = AxisSpec(AxisId("point"), "point", SCAN_POINT, 2, (0, 1))
    component = AxisSpec(AxisId("component"), "component", SPATIAL_X, 3, (0, 1, 2))
    schema = DatasetSchema(
        DomainSpec((2,), (repeat,), ((0, 1),)),
        DomainSpec((2,), (point,), ((0, 1),)),
        DomainSpec((3,), (component,)),
        ValueSchema(ValidityContract.value(), np.dtype("<f4")),
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
    """Every member by its physical name -- the oracle must not guess either."""

    with np.load(path, allow_pickle=False) as archive:
        return {
            name[:-4]: np.asarray(archive[name]) for name in archive.zip.namelist()
        }


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
    sites = AxisSpec(
        AxisId("calibration.site"),
        "Site",
        SITE,
        2,
        ("site_0001", "site_0002"),
        coordinate_labels=("1", "2"),
    )
    schema = DatasetSchema(
        DomainSpec((2,), (repeat,), ((0, 1),)),
        DomainSpec((2,), (sites,), ((0, 1),)),
        SCALAR_DOMAIN,
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

    restored_repeat = restored.block.schema.repeat_domain.axis(AxisId("capture.repeat"))
    assert restored_repeat.coordinates == ("shot_dark", "shot_bright")
    assert restored_repeat.coordinate_labels == ("Dark", "Bright")
    restored_sites = restored.block.schema.point_domain.axis(AxisId("calibration.site"))
    assert restored_sites.coordinates == ("site_0001", "site_0002")
    assert restored_sites.coordinate_labels == ("1", "2")


def test_npz_round_trip_preserves_dataset_component_masks(tmp_path: Path):
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2, (0, 1))
    point = AxisSpec(AxisId("point"), "point", SCAN_POINT, 2, (0, 1))
    component = AxisSpec(AxisId("component"), "component", SPATIAL_X, 3, (0, 1, 2))
    schema = DatasetSchema(
        DomainSpec((2,), (repeat,), ((0, 1),)),
        DomainSpec((2,), (point,), ((0, 1),)),
        DomainSpec((3,), (component,)),
        ValueSchema(
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


def test_npz_round_trip_preserves_point_domain_axes_and_codes(
    tmp_path: Path,
):
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    component = AxisSpec(AxisId("component"), "component", SPATIAL_X, 3, (0, 1, 2))
    b_x = AxisSpec(AxisId("b.x"), "b.x", SCAN_POINT, 2, (0, 1))
    point_domain = DomainSpec((2,), (b_x,), ((0, 1),))
    schema = DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        point_domain,
        DomainSpec((3,), (component,)),
        ValueSchema(ValidityContract.value(), np.dtype("<f4")),
    )
    block = DataBlock(
        BlockId("point-domain-io-block"),
        DatasetRevision(0),
        np.arange(6, dtype="<f4").reshape(schema.physical_shape),
        CellValidity(np.array([[True, False]])),
        schema,
    )
    source = OwnedSnapshot(
        block.ref(StreamGenerationId("point-domain-generation")),
        block,
    )
    path = tmp_path / "point-domain.npz"

    _write_npz(path, source)
    restored = load_npz(path)

    assert restored.block.schema == schema
    assert restored.block.schema.point_domain == point_domain
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


def test_npz_rejects_non_current_manifest_format(tmp_path: Path):
    original = tmp_path / "original.npz"
    malformed = tmp_path / "bad-format.npz"
    _write_npz(original, _snapshot())
    arrays = _archive_members(original)
    manifest = json.loads(str(arrays["manifest"].item()))
    manifest["format"] = "other-format"
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
    duplicate = manifest_text.replace(
        '"format":"zlc.dataset"',
        '"format":"zlc.dataset","format":"zlc.dataset"',
        1,
    )
    assert duplicate != manifest_text
    arrays["manifest"] = np.asarray(duplicate)
    np.savez_compressed(malformed, **arrays)

    with pytest.raises(NPZFormatError, match="duplicate key"):
        load_npz(malformed)


def test_npz_rejects_a_duplicate_member_instead_of_choosing_one(tmp_path: Path):
    """Two ZIP entries of one name are an ambiguous archive, not a choice.

    NpzFile answers a logical lookup with whichever entry it finds first;
    a reader that calls itself strict must refuse rather than pick.
    """

    original = tmp_path / "original.npz"
    malformed = tmp_path / "duplicate-member.npz"
    _write_npz(original, _snapshot())
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(malformed, "w") as target:
        for info in source.infolist():
            target.writestr(info, source.read(info))
        stream = BytesIO()
        np.save(stream, np.full(_snapshot().block.values.shape, 22.0, dtype="<f4"))
        with warnings.catch_warnings():
            # The duplicate name is the point; zipfile warns about it.
            warnings.simplefilter("ignore", UserWarning)
            target.writestr("values.npy", stream.getvalue())

    with pytest.raises(NPZFormatError, match="duplicate members.*values"):
        load_npz(malformed)


def test_npz_rejects_a_manifest_shape_the_writer_never_produces(tmp_path: Path):
    """``ref`` must be an object, and a present ``sigma_key`` a member name.

    A list of pairs happened to survive ``dict(...)``, and an explicit null
    read as "absent"; both are shapes the writer never emits and the strict
    reader claimed to reject.
    """

    original = tmp_path / "original.npz"
    _write_npz(original, _snapshot())
    arrays = _archive_members(original)
    manifest = json.loads(str(arrays["manifest"].item()))

    listed = dict(manifest)
    listed["ref"] = list(manifest["ref"].items())
    arrays["manifest"] = np.asarray(json.dumps(listed))
    malformed = tmp_path / "ref-as-list.npz"
    np.savez_compressed(malformed, **arrays)
    with pytest.raises(NPZFormatError, match="manifest.ref must be an object"):
        load_npz(malformed)

    nulled = dict(manifest)
    nulled["sigma_key"] = None
    arrays["manifest"] = np.asarray(json.dumps(nulled))
    malformed = tmp_path / "sigma-null.npz"
    np.savez_compressed(malformed, **arrays)
    with pytest.raises(NPZFormatError, match="sigma_key must be a non-empty string"):
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
        return {
            name[:-4]: np.asarray(archive[name]) for name in archive.zip.namelist()
        }


def _figure_payload(members: dict[str, np.ndarray]) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **members)
    return stream.getvalue()


def test_figure_archive_round_trip_validates_members_and_dataset_shape(monkeypatch):
    import zlc_data.figure_archive as figures

    decoded = []
    original = figures.snapshot_from_manifest

    def decode(*args, **kwargs):
        result = original(*args, **kwargs)
        decoded.append(result)
        return result

    monkeypatch.setattr(figures, "snapshot_from_manifest", decode)
    snapshot = _snapshot(CellValidity(np.array([[True, False], [False, True]])))
    stream = _figure_stream(
        "strict figure",
        arrays={"data": snapshot, "trace": np.arange(3, dtype="<f4")},
        sections={"panel": {"kind": "image"}},
    )

    info, arrays, datasets = read_archive(stream)

    assert info["schema"] == FIGURE_SCHEMA
    assert set(info["members"]) == {"data", "data.validity", "trace"}
    assert set(arrays) == {"data", "data.validity", "trace"}
    assert datasets["data"].exactly_equals(snapshot)
    assert decoded == [datasets["data"]]
    assert datasets["data"].block.values is arrays["data"]
    assert datasets["data"].block.validity.mask is arrays["data.validity"]
    with pytest.raises(ValueError):
        arrays["data"].setflags(write=True)


def test_figure_archive_compresses_only_members_that_shrink_materially(monkeypatch):
    rng = np.random.default_rng(7)
    compressible = np.zeros(2 << 20, dtype=np.uint8)
    camera_noise = rng.integers(0, 256, size=2 << 20, dtype=np.uint8)
    # The archive does not record its deflate level, so the compressor
    # construction is the witness.  Level 1: zlib's default took ten times
    # as long on a thousand-shot camera history for 8 % of the size, and a
    # Save is waited for.
    levels: list[int] = []
    real_compressobj = zlib.compressobj

    def spying_compressobj(level=-1, *args, **kwargs):
        levels.append(level)
        return real_compressobj(level, *args, **kwargs)

    monkeypatch.setattr(zlib, "compressobj", spying_compressobj)
    stream = _figure_stream(
        "adaptive compression",
        arrays={"compressible": compressible, "camera_noise": camera_noise},
        sections={},
    )
    assert levels and set(levels) == {1}, levels

    with zipfile.ZipFile(stream) as archive:
        assert archive.getinfo("compressible.npy").compress_type == zipfile.ZIP_DEFLATED
        assert archive.getinfo("camera_noise.npy").compress_type == zipfile.ZIP_STORED

    _info, arrays, datasets = read_archive(stream)
    np.testing.assert_array_equal(arrays["compressible"], compressible)
    np.testing.assert_array_equal(arrays["camera_noise"], camera_noise)


@pytest.mark.parametrize("extra", ({"unexpected": 2}, {"schema": "not-zlc.figure"}))
def test_figure_reader_rejects_non_current_roots(extra):
    members = _figure_members(
        _figure_stream("current", arrays={"trace": np.arange(2)}, sections={})
    )
    info = json.loads(str(members["info"].item()))
    info.update(extra)
    members["info"] = np.asarray(json.dumps(info, sort_keys=True))
    with pytest.raises(ValueError, match="metadata keys mismatch|unsupported figure format"):
        read_archive(BytesIO(_figure_payload(members)))


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


def test_figure_members_are_read_by_their_own_physical_names():
    """``signal`` and ``signal.npy`` are two legal array keys; both come back.

    The writer stores them as ``signal.npy`` and ``signal.npy.npy``.  Asked
    for the logical key ``signal.npy``, NpzFile first matched it as a
    physical name and answered with the OTHER array -- values swapped
    under an untouched ref, and every shape and dtype check passing.
    """

    stream = BytesIO()
    write_figure_archive(
        stream,
        "two members",
        arrays={"signal": np.array([11]), "signal.npy": np.array([22])},
        sections={},
    )
    stream.seek(0)
    _info, arrays, datasets = read_archive(stream)
    assert arrays["signal"].tolist() == [11]
    assert arrays["signal.npy"].tolist() == [22]

    first = _snapshot()
    second = OwnedSnapshot(
        first.ref,
        DataBlock(
            first.block.block_id,
            first.block.revision,
            np.full(first.block.values.shape, 22.0, dtype="<f4"),
            VALID,
            first.block.schema,
        ),
    )
    typed = BytesIO()
    write_figure_archive(
        typed, "two datasets", arrays={"signal": first, "signal.npy": second}, sections={}
    )
    typed.seek(0)
    info, arrays, datasets = read_archive(typed)
    assert datasets["signal"].block.values.tolist() == (
        first.block.values.tolist()
    )
    assert datasets["signal.npy"].block.values.tolist() == (
        second.block.values.tolist()
    )


@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_large_figure_stream_does_not_allocate_archive_sized_python_bytes(
    tmp_path, layout: str
) -> None:
    """Writing a large member costs a bounded working set, whatever its strides.

    The compressibility probe samples one mebibyte.  Gathered with
    ``np.take`` it first flattened a strided view into a contiguous copy of
    the whole plane, so saving a bytes-backed slice -- which the Data
    contract keeps as a view on purpose -- cost the size of the dataset.
    """

    import tracemalloc

    def measured(size_mib: int) -> tuple[int, int]:
        rng = np.random.default_rng(size_mib)
        count = size_mib * 1024 * 1024
        if layout == "strided":
            values = rng.integers(0, 256, size=2 * count, dtype=np.uint8)[::2]
            assert not values.flags.c_contiguous and not values.flags.f_contiguous
        else:
            values = rng.integers(0, 256, size=count, dtype=np.uint8)
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


def test_figure_reader_rejects_wrong_format():
    members = _figure_members(
        _figure_stream("format", arrays={"trace": np.arange(2)}, sections={})
    )
    info = json.loads(str(members["info"].item()))
    info["schema"] = "other-format"
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
    duplicate = text.replace('"schema":"zlc.figure"', '"schema":"zlc.figure","schema":"zlc.figure"', 1)
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
