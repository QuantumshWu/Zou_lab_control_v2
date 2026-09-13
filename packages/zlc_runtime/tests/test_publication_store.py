"""What a run leaves on disk, and what an interruption costs.

A run's data lived in memory until somebody saved a figure: a day of it was
hundreds of megabytes nobody could open afterwards, and an interruption lost
all of it.  These are the properties the store exists to have -- every
complete chunk survives a crash, a bool plane costs a bit and not a byte,
and what comes back is what was published.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SITE,
    AxisId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetComponentValidity,
    DatasetRevision,
    DatasetSchema,
    DomainSpec,
    OwnedSnapshot,
    SCALAR_DOMAIN,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_data.store import PublicationStoreError
from zlc_runtime.publication_store import (
    STORE_MANIFEST,
    PublicationReader,
    PublicationWriter,
)


SITES = 4


def _schema(dtype: str, unit: str) -> DatasetSchema:
    site_axis = AxisSpec(AxisId("t.site"), "site", SITE, SITES)
    return DatasetSchema(
        DomainSpec(
            (1,),
            (AxisSpec(AxisId("t.repeat"), "repeat", REPEAT, 1, (0,)),),
            ((0,),),
        ),
        DomainSpec(
            (1,),
            (AxisSpec(AxisId("t.point"), "point", SCAN_POINT, 1),),
            ((0,),),
        ),
        DomainSpec((SITES,), (site_axis,)),
        ValueSchema(
            ValidityContract.components(site_axis.axis_id),
            np.dtype(dtype),
            unit,
            name="signal",
        ),
    )


def _shot(index: int, *, dtype: str = "<f4", unit: str = "count",
          sigma: bool = False) -> OwnedSnapshot:
    schema = _schema(dtype, unit)
    if dtype == "?":
        values = (np.arange(SITES) + index) % 2 == 0
        values = values.reshape((1, 1, SITES))
    else:
        values = (np.arange(SITES, dtype=np.float32) + index).reshape((1, 1, SITES))
    block = DataBlock(
        BlockId("signal"),
        DatasetRevision(index + 1),
        values.astype(np.dtype(dtype)),
        DatasetComponentValidity(
            (AxisId("t.site"),),
            np.asarray(
                [site != index % SITES for site in range(SITES)]
            ).reshape((1, 1, SITES)),
        ),
        schema,
        np.full((1, 1, SITES), 0.25) if sigma else None,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("run")), block)


def _write(path, shots, **options) -> PublicationWriter:
    writer = PublicationWriter(path, **options)
    for shot in shots:
        writer.append(shot)
    writer.close()
    return writer


def test_every_shot_comes_back_exactly_as_it_was_published(tmp_path) -> None:
    shots = [_shot(index, sigma=True) for index in range(7)]
    _write(tmp_path / "run", shots, events_per_chunk=3)

    reader = PublicationReader(tmp_path / "run")
    assert reader.events == 7
    for index, original in enumerate(shots):
        assert reader.snapshot(index).exactly_equals(original)


def test_an_interruption_costs_only_what_was_still_buffered(tmp_path) -> None:
    """The manifest is the store; a chunk it does not name never happened.

    Written the other way round -- name it, then write it -- a crash between
    the two leaves a store that claims events it cannot produce, which is
    worse than losing them, because nothing says so.
    """

    root = tmp_path / "run"
    writer = PublicationWriter(root, events_per_chunk=4)
    for index in range(10):
        writer.append(_shot(index))
    # Two chunks are complete; two shots are buffered.  This is the crash.
    assert writer.durable_events == 8
    assert writer.events == 10
    del writer

    reader = PublicationReader(root)
    assert reader.events == 8
    assert reader.snapshot(7).exactly_equals(_shot(7))


def test_a_chunk_the_manifest_never_named_is_not_read(tmp_path) -> None:
    root = tmp_path / "run"
    _write(root, [_shot(index) for index in range(6)], events_per_chunk=3)
    manifest = json.loads((root / STORE_MANIFEST).read_text(encoding="utf-8"))
    # Exactly the state a crash between the chunk write and the manifest
    # replacement leaves behind: the files are there, the store is not.
    manifest["chunks"] = manifest["chunks"][:1]
    manifest["events"] = manifest["chunks"][0]["events"]
    (root / STORE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    reader = PublicationReader(root)
    assert reader.events == 3
    assert reader.snapshot(2).exactly_equals(_shot(2))


def test_occupancy_costs_a_bit_on_disk_and_not_a_byte(tmp_path) -> None:
    """A hundred sites for a day is a hundred megabytes of bool, or twelve.

    The values come back as the bool they were published as; only what is
    written down is packed.
    """

    shots = [_shot(index, dtype="?", unit="1") for index in range(2048)]
    writer = _write(tmp_path / "occupancy", shots, events_per_chunk=2048)
    plain = 2048 * SITES  # one byte per site per shot, as bool arrays are
    values = [
        path for path in (tmp_path / "occupancy" / "chunks").glob("*-values.npy")
    ]
    assert len(values) == 1
    # The .npy header is a fixed 128 bytes whatever the payload, so the
    # property is about the payload: eight sites to the byte.
    assert values[0].stat().st_size - 128 == plain // 8

    reader = PublicationReader(tmp_path / "occupancy")
    assert reader.values().dtype == np.dtype(bool)
    assert reader.snapshot(2000).exactly_equals(shots[2000])
    assert writer.nbytes == reader.nbytes


def test_a_window_of_a_long_run_reads_as_the_slice_of_it(tmp_path) -> None:
    shots = [_shot(index) for index in range(20)]
    _write(tmp_path / "run", shots, events_per_chunk=6)

    reader = PublicationReader(tmp_path / "run")
    every = reader.values()
    assert every.shape == (20, 1, 1, SITES)
    assert np.array_equal(reader.values(7, 15), every[7:15])
    assert np.array_equal(reader.values(6, 12), every[6:12])


def test_a_signal_that_changes_shape_starts_its_own_chunk(tmp_path) -> None:
    """A chunk is a run of events that agree; disagreement is not an error.

    A store has to hold a whole session, and a session legitimately contains
    a schema change -- a derived output renamed, a site map edited.  Refusing
    one would make the store useless exactly when the run gets interesting.
    """

    root = tmp_path / "run"
    writer = PublicationWriter(root, events_per_chunk=100)
    writer.append(_shot(0))
    writer.append(_shot(1, dtype="<f8"))
    writer.append(_shot(2, dtype="<f8"))
    writer.close()

    reader = PublicationReader(root)
    assert [chunk.events for chunk in reader.chunks] == [1, 2]
    assert reader.snapshot(0).block.values.dtype == np.dtype("<f4")
    assert reader.snapshot(2).block.values.dtype == np.dtype("<f8")


def test_a_window_across_a_schema_change_is_refused_not_guessed_at(tmp_path) -> None:
    """A chunk boundary exists because the events stopped agreeing.

    Stacked anyway, two different quantities either raise deep inside NumPy
    or silently upcast into a shape nobody asked for.  Said out loud, the
    caller reads one chunk's range at a time.
    """

    root = tmp_path / "run"
    writer = PublicationWriter(root, events_per_chunk=100)
    writer.append(_shot(0))
    writer.append(_shot(1, dtype="<f8"))
    writer.close()

    reader = PublicationReader(root)
    assert reader.values(0, 1).dtype == np.dtype("<f4")
    assert reader.values(1, 2).dtype == np.dtype("<f8")
    with pytest.raises(PublicationStoreError):
        reader.values()


def test_what_comes_back_is_shaped_the_same_whatever_the_window(tmp_path) -> None:
    """Rank, dtype and mutability must not depend on where a window lands.

    Inside one chunk the answer IS the file's mapping; across two it is a
    fresh array; empty it used to be a bare float64 vector.  Three different
    answers to one question is a trap for the caller who slices with numbers
    that happen to move.
    """

    shots = [_shot(index) for index in range(10)]
    _write(tmp_path / "run", shots, events_per_chunk=4)
    reader = PublicationReader(tmp_path / "run")

    inside = reader.values(0, 3)
    across = reader.values(2, 9)
    empty = reader.values(5, 5)
    for window in (inside, across, empty):
        assert window.ndim == inside.ndim
        assert window.dtype == inside.dtype
        assert not window.flags.writeable
    assert empty.shape == (0, *inside.shape[1:])


def test_a_flush_that_cannot_be_written_keeps_its_events(tmp_path, monkeypatch) -> None:
    """The buffer is what a retry has left to try with.

    Cleared before the write, a full disk took the events out of memory
    without putting them anywhere -- and the writer went on counting them as
    accepted.
    """

    from zlc_runtime import publication_store

    root = tmp_path / "run"
    writer = PublicationWriter(root, events_per_chunk=2)
    real = publication_store._write_array
    monkeypatch.setattr(
        publication_store, "_write_array",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("no space left on device")),
    )
    with pytest.raises(OSError):
        writer.append(_shot(0))
        writer.append(_shot(1))
    assert writer.durable_events == 0

    monkeypatch.setattr(publication_store, "_write_array", real)
    writer.close()
    assert PublicationReader(root).events == 2


def test_a_second_writer_refuses_the_store_it_would_have_emptied(tmp_path) -> None:
    """Writing a store is a once.

    The manifest IS the store, so a fresh writer's empty one would turn every
    chunk already there into a file nothing references -- a day of a run lost
    with nothing said.  Refusing costs a caller one message; the alternative
    costs the data.
    """

    root = tmp_path / "run"
    _write(root, [_shot(index) for index in range(4)], events_per_chunk=2)
    with pytest.raises(PublicationStoreError):
        PublicationWriter(root)
    assert PublicationReader(root).events == 4


def test_a_directory_that_is_not_a_store_says_so(tmp_path) -> None:
    (tmp_path / STORE_MANIFEST).write_text('{"format": "something else"}',
                                           encoding="utf-8")
    with pytest.raises(PublicationStoreError):
        PublicationReader(tmp_path)
