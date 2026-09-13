"""An append-only directory of everything a run published.

:mod:`zlc_data.store` owns what a chunk MEANS; this owns where its bytes
land and how durably -- the same split ``save_npz`` already makes, and the
reason zlc_data may not import the durable layer at all.

WHAT SURVIVES A CRASH.  Chunk files are written and fsynced first; only then
is the manifest atomically replaced to name them.  A chunk the manifest does
not name never happened -- it is ignored on open -- so the worst an
interruption costs is the events still buffered for the chunk in progress.
Written the other way round, a crash between the two would leave a store
claiming events it cannot produce, which is worse than losing them because
nothing says so.

WHY .npy AND NOT A CHUNK LIBRARY.  Every plane is a plain NumPy file: it
memory-maps, so reading the part of a long run that is on screen does not
read the run; ``np.load`` opens it with nothing of ours installed; and the
distribution's pinned dependency list does not grow.  Compression or a
remote object store would be a different writer over the same documents.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
import json
import os
from typing import Any, BinaryIO

import numpy as np

from zlc_data import DatasetSchema, OwnedSnapshot
from zlc_data.store import (
    ChunkRecord,
    PublicationStoreError,
    chunk_event,
    chunk_plane,
    chunk_plane_keys,
    chunk_schema,
    encode_chunk,
    events_agree,
    store_chunks,
    store_document,
)
from zlc_durable import atomic_write_file, durable_makedirs


#: The manifest's name inside a store, and the folder its chunks live in.
STORE_MANIFEST = "manifest.json"
CHUNK_DIRECTORY = "chunks"

#: How many events one chunk holds unless the caller says otherwise.  This
#: is the crash window: what is still buffered when the power goes is what is
#: lost, and a thousand shots is a minute or two of a live run.
DEFAULT_EVENTS_PER_CHUNK = 1000


__all__ = [
    "CHUNK_DIRECTORY",
    "DEFAULT_EVENTS_PER_CHUNK",
    "STORE_MANIFEST",
    "PublicationReader",
    "PublicationWriter",
]


def _write_json(path: Path, tree: Mapping[str, Any]) -> None:
    try:
        payload = json.dumps(
            tree,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PublicationStoreError(f"store metadata is not serializable: {error}")
    atomic_write_file(path, lambda stream: stream.write(payload))


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        tree = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationStoreError(f"{path.name} is not readable JSON: {error}")
    if not isinstance(tree, Mapping):
        raise PublicationStoreError(f"{path.name} must hold an object")
    return tree


def _write_array(path: Path, array: np.ndarray) -> None:
    """One plain ``.npy``, fsynced and atomically published."""

    def writer(stream: BinaryIO) -> None:
        np.lib.format.write_array(
            stream, np.ascontiguousarray(array), allow_pickle=False
        )

    atomic_write_file(path, writer)


class PublicationWriter:
    """Append publications to a store directory, one chunk at a time.

    Buffering is by EVENT COUNT, not by bytes, because the number that
    matters to an operator is how many shots an interruption costs, and that
    is the same number whatever a shot contains.

    A run of events is stacked into one file per plane, so a chunk is three
    files at most rather than three per event: a day at ten shots a second is
    a million events, and a million-file directory is its own disaster.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        events_per_chunk: int = DEFAULT_EVENTS_PER_CHUNK,
        note: str = "",
    ) -> None:
        count = int(events_per_chunk)
        if count < 1:
            raise ValueError("a chunk holds at least one event")
        self._root = durable_makedirs(Path(directory))
        self._chunk_root = durable_makedirs(self._root / CHUNK_DIRECTORY)
        self._events_per_chunk = count
        self._note = str(note)
        self._chunks: list[ChunkRecord] = []
        self._buffer: list[OwnedSnapshot] = []
        self._events = 0
        self._closed = False
        _write_json(self._root / STORE_MANIFEST, store_document((), note=self._note))

    @property
    def root(self) -> Path:
        return self._root

    @property
    def events(self) -> int:
        """Events accepted, including those still buffered."""

        return self._events

    @property
    def durable_events(self) -> int:
        """Events a crash right now would leave behind."""

        return sum(chunk.events for chunk in self._chunks)

    @property
    def nbytes(self) -> int:
        return sum(chunk.nbytes for chunk in self._chunks)

    def append(self, snapshot: OwnedSnapshot) -> None:
        if self._closed:
            raise PublicationStoreError("this store is closed")
        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("a publication store holds OwnedSnapshot values")
        if self._buffer and not events_agree(self._buffer[0], snapshot):
            self.flush()
        self._buffer.append(snapshot)
        self._events += 1
        if len(self._buffer) >= self._events_per_chunk:
            self.flush()

    def flush(self) -> None:
        """Make every accepted event durable, then name it in the manifest."""

        if self._closed or not self._buffer:
            return
        events = tuple(self._buffer)
        self._buffer.clear()
        name = f"{len(self._chunks):06d}"
        first_event = self.durable_events
        document, arrays = encode_chunk(events)
        written = 0
        for key, array in arrays.items():
            path = self._chunk_root / f"{name}-{key}.npy"
            _write_array(path, array)
            written += int(path.stat().st_size)
            document["planes"][key]["file"] = path.name
        _write_json(self._chunk_root / f"{name}.json", document)
        self._chunks.append(
            ChunkRecord(
                name=name,
                first_event=first_event,
                events=len(events),
                schema_fingerprint=str(events[0].ref.schema_fingerprint),
                nbytes=written,
            )
        )
        # The chunk's bytes are on disk and fsynced; naming it is what makes
        # it part of the store, and that replacement is atomic.  Between the
        # two, a crash leaves files nobody references -- which is exactly
        # what "the last chunk did not happen" should look like.
        _write_json(
            self._root / STORE_MANIFEST,
            store_document(self._chunks, note=self._note),
        )

    def close(self) -> None:
        self.flush()
        self._closed = True

    def __enter__(self) -> "PublicationWriter":
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()


class PublicationReader:
    """Read a store without loading it.

    One chunk's plane is one memory-mapped ``.npy``, so asking for the
    thousand shots on screen reads a thousand shots' worth of pages, not a
    day's.  A packed bool plane is the exception: unpacking is a copy, and
    that is the price of it being eight times smaller.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._root = Path(directory)
        self._records = store_chunks(_read_json(self._root / STORE_MANIFEST))
        self._documents: dict[str, Mapping[str, Any]] = {}
        self._events = sum(record.events for record in self._records)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def events(self) -> int:
        return self._events

    @property
    def chunks(self) -> tuple[ChunkRecord, ...]:
        return self._records

    @property
    def nbytes(self) -> int:
        return sum(record.nbytes for record in self._records)

    def __len__(self) -> int:
        return self._events

    def schema(self, event: int = 0) -> DatasetSchema:
        record, _offset = self._locate(event)
        return chunk_schema(self._document(record))

    def _locate(self, event: int) -> tuple[ChunkRecord, int]:
        index = int(event)
        if index < 0:
            index += self._events
        if not 0 <= index < self._events:
            raise IndexError(f"event {event} is outside a store of {self._events}")
        for record in self._records:
            if index < record.first_event + record.events:
                return record, index - record.first_event
        raise PublicationStoreError("store chunk table does not cover its events")

    def _document(self, record: ChunkRecord) -> Mapping[str, Any]:
        found = self._documents.get(record.name)
        if found is None:
            found = _read_json(self._root / CHUNK_DIRECTORY / f"{record.name}.json")
            self._documents[record.name] = found
        return found

    def _plane(self, record: ChunkRecord, key: str) -> np.ndarray:
        document = self._document(record)
        entry = document["planes"][key]
        # mmap, because "open yesterday's run" must not mean "read all of it".
        stored = np.load(
            self._root / CHUNK_DIRECTORY / str(entry["file"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        return chunk_plane(document, key, stored)

    def values(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Stacked values for ``[start, stop)``, read chunk by chunk."""

        first = int(start)
        last = self._events if stop is None else int(stop)
        if first < 0 or last > self._events or first > last:
            raise IndexError(f"[{start}, {stop}) is outside {self._events} events")
        if first == last:
            return np.empty((0,), dtype=np.float64)
        pieces = []
        for record in self._records:
            low = max(first, record.first_event)
            high = min(last, record.first_event + record.events)
            if low >= high:
                continue
            key = chunk_plane_keys(self._document(record))[0]
            plane = self._plane(record, key)
            pieces.append(
                np.asarray(plane[low - record.first_event: high - record.first_event])
            )
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

    def snapshot(self, event: int) -> OwnedSnapshot:
        """One event, rebuilt exactly as it was published."""

        record, offset = self._locate(event)
        return chunk_event(
            self._document(record),
            lambda key, record=record: self._plane(record, key),
            offset,
        )

    def __iter__(self) -> Iterator[OwnedSnapshot]:
        for index in range(self._events):
            yield self.snapshot(index)
