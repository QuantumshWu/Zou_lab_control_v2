"""Strict figure-archive round-trips and rejection of incomplete archives."""

from __future__ import annotations

import json
from io import BytesIO
import warnings
import zipfile
import zlib

import numpy as np
import pytest

from zlc_data.axis import AxisId, AxisSpec, REPEAT, SCAN_POINT, SPATIAL_X
from zlc_data.figure_archive import (
    FIGURE_SCHEMA,
    read_archive,
    write_figure_archive,
)
from zlc_data.schema import (
    DatasetSchema,
    DomainSpec,
    ValueSchema,
)
from zlc_data.validity import (
    CellValidity,
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
