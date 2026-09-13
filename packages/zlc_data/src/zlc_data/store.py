"""What a chunk of published events IS, as a document and some arrays.

A run's data lives in memory until somebody saves a figure: a day of it is
hundreds of megabytes nobody can open afterwards, and an interruption loses
all of it.  The other half of that is an append-only log of publications,
and this module owns what one chunk of such a log means.  Where the bytes
land is the caller's -- the same split :func:`save_npz` already makes, and
for the same reason: this layer may not know about paths.

WHAT A CHUNK IS.  A run of consecutive events that agree on their schema,
their validity shape and whether they carry sigma, stacked into one array
per plane.  Disagreement is not an error, it starts the next chunk: a
session legitimately contains a schema change, and refusing one would make
the log useless exactly when the run got interesting.

WHY A CHUNK IS SMALL.  Two things that would otherwise dominate:

* An event's identity is three hundred bytes of JSON -- block id, stream
  generation, schema fingerprint -- and an occupancy shot's values are
  twenty-five.  Everything equal across the chunk is said once
  (:func:`_factored`), so only the revision repeats.
* A bool plane is packed to one bit per value, and a plane every event
  repeats -- the validity mask of a run where nothing is missing -- is
  written once.

Measured at a hundred sites: a million-shot day of ``occupied`` is 40 MB on
disk against 100 MB of bool in memory, and of float32 ``counts`` 410 MB
against 400 MB.  The identity is no longer the file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .codec import dataset_schema_from_tree, dataset_schema_to_tree
from .io import (
    NPZFormatError,
    manifest_array_keys,
    snapshot_from_manifest,
    snapshot_manifest,
)
from .schema import DatasetSchema
from .value import OwnedSnapshot


#: What a store directory declares itself to be.
PUBLICATION_STORE_CONTRACT = "zlc.publication-store"

#: What one chunk document declares itself to be.
PUBLICATION_CHUNK_CONTRACT = "zlc.publication-chunk"

#: How a plane is stored.  ``bits`` is ``np.packbits``; ``raw`` is the array
#: as it stands.  Named in the chunk so a reader never has to guess.
RAW = "raw"
BITS = "bits"


class PublicationStoreError(ValueError):
    """A store or chunk does not say what it claims to say."""


def _encode_plane(array: np.ndarray) -> tuple[np.ndarray, str]:
    if array.dtype == np.dtype(bool):
        return np.packbits(np.ascontiguousarray(array).reshape(-1)), BITS
    return array, RAW


def _decode_plane(
    array: np.ndarray, encoding: str, shape: tuple[int, ...]
) -> np.ndarray:
    if encoding == RAW:
        return np.asarray(array).reshape(shape)
    if encoding != BITS:
        raise PublicationStoreError(f"unknown plane encoding {encoding!r}")
    count = 1
    for size in shape:
        count *= size
    return np.unpackbits(np.asarray(array), count=count).astype(bool).reshape(shape)


def _factored(trees: Sequence[Mapping[str, Any]]) -> tuple[dict, list[dict]]:
    """What every event says identically, lifted out of all of them.

    Nothing is named here.  A key whose value is equal across the chunk
    belongs to the chunk, recursing into nested objects, so a field added to
    an event's identity tomorrow factors itself.
    """

    common: dict[str, Any] = {}
    rest: list[dict[str, Any]] = [{} for _ in trees]
    for key, value in dict(trees[0]).items():
        others = [tree.get(key) for tree in trees[1:]]
        if all(other == value for other in others):
            common[key] = value
            continue
        if isinstance(value, Mapping) and all(
            isinstance(other, Mapping) for other in others
        ):
            nested_common, nested_rest = _factored(
                [dict(tree[key]) for tree in trees]
            )
            if nested_common:
                common[key] = nested_common
            for target, piece in zip(rest, nested_rest, strict=True):
                if piece:
                    target[key] = piece
            continue
        for target, tree in zip(rest, trees, strict=True):
            target[key] = tree[key]
    return common, rest


def _unfactored(common: Mapping[str, Any], rest: Mapping[str, Any]) -> dict:
    merged = dict(common)
    for key, value in rest.items():
        shared = merged.get(key)
        if isinstance(shared, Mapping) and isinstance(value, Mapping):
            merged[key] = _unfactored(shared, value)
        else:
            merged[key] = value
    return merged


def _event_tree(snapshot: OwnedSnapshot) -> dict[str, Any]:
    """One event's own manifest, minus the schema its chunk states once.

    Built by :func:`snapshot_manifest` rather than assembled here, so the
    format name and the array keys have exactly one author.  The arrays that
    manifest wanted are discarded with the dictionary it put them in -- the
    chunk stacks them itself, under the very keys it just named.
    """

    arrays: dict[str, np.ndarray] = {}
    manifest = snapshot_manifest(snapshot, arrays)
    return {key: value for key, value in manifest.items() if key != "schema"}


def events_agree(first: OwnedSnapshot, other: OwnedSnapshot) -> bool:
    """Whether two events can be stacked into one chunk."""

    return bool(
        other.ref.schema_fingerprint == first.ref.schema_fingerprint
        and type(other.block.validity) is type(first.block.validity)
        and (other.block.sigma is None) == (first.block.sigma is None)
    )


def encode_chunk(
    events: Sequence[OwnedSnapshot],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """One chunk's document and the arrays it names, ready to be written.

    The caller owns where those arrays land and how durably; this owns what
    they mean.  Plane files are named in the document rather than spelled by
    a rule the reader would have to know by heart.
    """

    if not events:
        raise ValueError("a chunk holds at least one event")
    first = events[0]
    for event in events[1:]:
        if not events_agree(first, event):
            raise ValueError("a chunk holds events that agree; start another")
    trees = [_event_tree(event) for event in events]
    common, varying = _factored(trees)
    document: dict[str, Any] = {
        "format": PUBLICATION_CHUNK_CONTRACT,
        "schema": dataset_schema_to_tree(first.block.schema),
        "event": common,
        "events": varying,
        "planes": {},
    }
    stacks: list[tuple[str, np.ndarray]] = [
        (trees[0]["values_key"], np.stack([event.block.values for event in events]))
    ]
    mask_key = trees[0]["validity"].get("mask_key")
    if isinstance(mask_key, str):
        stacks.append(
            (mask_key, np.stack([event.block.validity.mask for event in events]))
        )
    sigma_key = trees[0].get("sigma_key")
    if isinstance(sigma_key, str):
        stacks.append(
            (sigma_key, np.stack([event.block.sigma for event in events]))
        )
    arrays: dict[str, np.ndarray] = {}
    for key, stacked in stacks:
        # A plane every event of a chunk repeats is written once.  The
        # ordinary case is the validity mask of a run where nothing is
        # missing: stacked, it is the size of the data it qualifies, and it
        # says one thing.
        repeated = bool(
            stacked.shape[0] > 1
            and np.array_equal(
                stacked, np.broadcast_to(stacked[:1], stacked.shape)
            )
        )
        encoded, encoding = _encode_plane(stacked[:1] if repeated else stacked)
        arrays[key] = encoded
        document["planes"][key] = {
            "encoding": encoding,
            "shape": [int(size) for size in stacked.shape],
            "dtype": stacked.dtype.str,
            **({"repeated": True} if repeated else {}),
        }
    return document, arrays


def chunk_schema(document: Mapping[str, Any]) -> DatasetSchema:
    """The schema every event of this chunk was published under."""

    return dataset_schema_from_tree(_require_chunk(document)["schema"])


def chunk_plane_keys(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Which planes this chunk stacked, in the order its events name them."""

    tree = _require_chunk(document)
    return manifest_array_keys({**tree["event"], "schema": tree["schema"]})


def chunk_plane(
    document: Mapping[str, Any], key: str, stored: np.ndarray
) -> np.ndarray:
    """A stored plane as the stack its events named.

    A plane written once because every event repeated it is broadcast back
    to the stack rather than materialised into one.
    """

    planes = _require_chunk(document).get("planes")
    if not isinstance(planes, Mapping) or key not in planes:
        raise PublicationStoreError(f"this chunk has no {key!r} plane")
    entry = planes[key]
    shape = tuple(int(size) for size in entry["shape"])
    if entry.get("repeated"):
        one = _decode_plane(stored, str(entry["encoding"]), (1, *shape[1:]))
        return np.broadcast_to(one, shape)
    return _decode_plane(stored, str(entry["encoding"]), shape)


def chunk_event(
    document: Mapping[str, Any],
    plane: Callable[[str], np.ndarray],
    offset: int,
) -> OwnedSnapshot:
    """One event of a chunk, given a way to read its stacked planes.

    ``plane`` is called with a plane key and returns that plane already
    decoded -- which is how a caller that memory-maps its files keeps the
    mapping out of this layer.
    """

    tree = _require_chunk(document)
    events = tree["events"]
    if not 0 <= int(offset) < len(events):
        raise IndexError(f"offset {offset} is outside a chunk of {len(events)}")
    manifest: dict[str, Any] = {
        **_unfactored(tree["event"], events[int(offset)]),
        "schema": tree["schema"],
    }
    arrays = {
        key: np.asarray(plane(key)[int(offset)])
        for key in manifest_array_keys(manifest)
    }
    try:
        return snapshot_from_manifest(manifest, arrays)
    except NPZFormatError as error:
        raise PublicationStoreError(f"event {offset} does not rebuild: {error}")


def _require_chunk(document: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(document, Mapping):
        raise PublicationStoreError("a chunk document is a mapping")
    declared = str(document.get("format", ""))
    if declared != PUBLICATION_CHUNK_CONTRACT:
        raise PublicationStoreError(
            f"a chunk declares {PUBLICATION_CHUNK_CONTRACT!r}, not {declared!r}"
        )
    return document


@dataclass(frozen=True)
class ChunkRecord:
    """One complete chunk, as a store's manifest names it."""

    name: str
    first_event: int
    events: int
    schema_fingerprint: str
    nbytes: int

    def to_tree(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "first_event": self.first_event,
            "events": self.events,
            "schema_fingerprint": self.schema_fingerprint,
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_tree(cls, tree: object) -> "ChunkRecord":
        if not isinstance(tree, Mapping):
            raise PublicationStoreError("a chunk record must be an object")
        missing = sorted(
            {"name", "first_event", "events", "schema_fingerprint", "nbytes"}
            - set(tree)
        )
        if missing:
            raise PublicationStoreError(f"a chunk record needs its {missing[0]!r}")
        return cls(
            name=str(tree["name"]),
            first_event=int(tree["first_event"]),
            events=int(tree["events"]),
            schema_fingerprint=str(tree["schema_fingerprint"]),
            nbytes=int(tree["nbytes"]),
        )


def store_document(
    chunks: Sequence[ChunkRecord], *, note: str = ""
) -> dict[str, Any]:
    """A store's manifest: the chunks that exist, and nothing that does not."""

    return {
        "format": PUBLICATION_STORE_CONTRACT,
        "events": sum(chunk.events for chunk in chunks),
        "nbytes": sum(chunk.nbytes for chunk in chunks),
        "chunks": [chunk.to_tree() for chunk in chunks],
        **({"note": note} if note else {}),
    }


def store_chunks(document: Mapping[str, Any]) -> tuple[ChunkRecord, ...]:
    """The chunks a store manifest names, checked to cover its events once."""

    if not isinstance(document, Mapping):
        raise PublicationStoreError("a store manifest is a mapping")
    declared = str(document.get("format", ""))
    if declared != PUBLICATION_STORE_CONTRACT:
        raise PublicationStoreError(
            f"a publication store declares {PUBLICATION_STORE_CONTRACT!r}, "
            f"not {declared!r}"
        )
    records = document.get("chunks")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise PublicationStoreError("a store's chunks are a list")
    chunks = tuple(ChunkRecord.from_tree(item) for item in records)
    expected = 0
    for chunk in chunks:
        if chunk.first_event != expected:
            raise PublicationStoreError(
                f"chunk {chunk.name!r} starts at event {chunk.first_event}, "
                f"not {expected}"
            )
        expected += chunk.events
    return chunks


__all__ = [
    "BITS",
    "PUBLICATION_CHUNK_CONTRACT",
    "PUBLICATION_STORE_CONTRACT",
    "RAW",
    "ChunkRecord",
    "PublicationStoreError",
    "chunk_event",
    "chunk_plane",
    "chunk_plane_keys",
    "chunk_schema",
    "encode_chunk",
    "events_agree",
    "store_chunks",
    "store_document",
]
