"""Strict, pickle-free figure archives containing arrays and typed datasets.

This module owns what a figure file means. The caller owns where the bytes
land. Arrays stay NPZ members; explanatory metadata is one strict JSON tree.
"""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
import json
import math
from typing import Any

import numpy as np

from .io import manifest_array_keys, snapshot_from_manifest, snapshot_manifest
from .validity import CellValidity, DatasetComponentValidity
from .value import OwnedSnapshot


__all__ = [
    "FIGURE_SCHEMA",
    "figure_bytes",
    "read_archive",
    "read_dataset",
]


FIGURE_SCHEMA = "zlc.figure"
_FIGURE_VERSION = 2
_INFO_KEY = "info"
_ROOT_KEYS = {"schema", "version", "name", "members", "sections"}


def _jsonable(value: Any, path: str = "metadata") -> Any:
    """Return the strict JSON tree admitted by the figure format."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} contains a non-finite metadata number")
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} metadata keys must be text")
            result[key] = _jsonable(item, f"{path}.{key}")
        return result
    if isinstance(value, list):
        return [
            _jsonable(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        raise TypeError(
            f"{path}: arrays belong beside info as their own entries, "
            "not inside metadata"
        )
    raise TypeError(f"{path} contains unsupported metadata type {type(value).__name__}")


def _claimed_members(key: str, value: np.ndarray | OwnedSnapshot) -> tuple[str, ...]:
    if isinstance(value, OwnedSnapshot) and isinstance(
        value.block.validity, (CellValidity, DatasetComponentValidity)
    ):
        return key, f"{key}.validity"
    return (key,)


def _member_descriptor(array: np.ndarray) -> dict[str, Any]:
    if array.dtype.hasobject or array.dtype.fields is not None:
        raise TypeError("figure arrays cannot use object or structured dtype")
    return {"dtype": array.dtype.str, "shape": list(array.shape)}


def figure_bytes(
    name: str,
    *,
    arrays: Mapping[str, np.ndarray | OwnedSnapshot],
    sections: Mapping[str, Any],
) -> bytes:
    """Encode one complete figure after planning its NPZ member namespace."""

    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError("figure name must be canonical non-empty text")
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("a figure archive must carry at least one array")
    if not isinstance(sections, dict):
        raise TypeError("figure sections must be a metadata dict")

    owners: dict[str, str] = {}
    for key, value in arrays.items():
        if not isinstance(key, str) or not key:
            raise TypeError("figure array names must be non-empty text")
        if key == _INFO_KEY:
            raise ValueError(f"{_INFO_KEY!r} is reserved for the metadata document")
        for member in _claimed_members(key, value):
            previous = owners.setdefault(member, key)
            if previous != key:
                raise ValueError(
                    f"figure member name collision for {member!r}: "
                    f"claimed by {previous!r} and {key!r}"
                )

    stored: dict[str, np.ndarray] = {}
    datasets: dict[str, Any] = {}
    for key, value in arrays.items():
        if isinstance(value, OwnedSnapshot):
            manifest = snapshot_manifest(
                value,
                stored,
                values_key=key,
                validity_key=f"{key}.validity",
            )
            ref = dict(manifest["ref"])
            ref.pop("schema_fingerprint", None)
            manifest["ref"] = ref
            datasets[key] = manifest
        else:
            stored[key] = np.asarray(value)

    plain_sections = _jsonable(sections)
    if datasets:
        if "dataset" in plain_sections:
            raise ValueError(
                "the dataset section is written from the snapshots themselves"
            )
        plain_sections["dataset"] = datasets
    members = {
        key: _member_descriptor(array) for key, array in sorted(stored.items())
    }
    info = json.dumps(
        {
            "schema": FIGURE_SCHEMA,
            "version": _FIGURE_VERSION,
            "name": name,
            "members": members,
            "sections": plain_sections,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    buffer = BytesIO()
    np.savez_compressed(buffer, **{_INFO_KEY: np.asarray(info), **stored})
    return buffer.getvalue()


def _metadata_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"figure contains duplicate metadata key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"figure contains non-finite metadata number {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        _reject_nonfinite(value)
    return result


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{path} keys mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _decode_info(array: np.ndarray) -> dict[str, Any]:
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError("figure info must be one scalar Unicode JSON document")
    try:
        info = json.loads(
            str(array.item()),
            object_pairs_hook=_metadata_pairs,
            parse_constant=_reject_nonfinite,
            parse_float=_finite_float,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid figure metadata JSON: {exc}") from exc
    if not isinstance(info, dict):
        raise ValueError("figure metadata root must be an object")
    _exact_keys(info, _ROOT_KEYS, "figure metadata")
    if (
        info["schema"] != FIGURE_SCHEMA
        or isinstance(info["version"], bool)
        or not isinstance(info["version"], int)
        or info["version"] != _FIGURE_VERSION
    ):
        raise ValueError(
            f"unsupported figure format {info['schema']!r} "
            f"version {info['version']!r}"
        )
    if (
        not isinstance(info["name"], str)
        or not info["name"]
        or info["name"].strip() != info["name"]
    ):
        raise ValueError("figure metadata name must be non-empty text")
    if not isinstance(info["sections"], dict):
        raise ValueError("figure metadata sections must be an object")
    members = info["members"]
    if not isinstance(members, dict) or not members:
        raise ValueError("figure metadata members must be a non-empty object")
    for key, descriptor in members.items():
        if not isinstance(key, str) or not key or key == _INFO_KEY:
            raise ValueError(
                "figure member names must be non-empty, non-reserved text"
            )
        if not isinstance(descriptor, dict):
            raise ValueError(f"figure member {key!r} descriptor must be an object")
        _exact_keys(descriptor, {"dtype", "shape"}, f"figure member {key!r}")
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        if not isinstance(dtype, str):
            raise ValueError(f"figure member {key!r} dtype must be text")
        try:
            parsed_dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"figure member {key!r} dtype is invalid") from exc
        if (
            parsed_dtype.hasobject
            or parsed_dtype.fields is not None
            or parsed_dtype.str != dtype
        ):
            raise ValueError(f"figure member {key!r} dtype is not canonical")
        if not isinstance(shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in shape
        ):
            raise ValueError(
                f"figure member {key!r} shape must be nonnegative integers"
            )
    return info


def _validate_datasets(
    info: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    raw = info["sections"].get("dataset")
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ValueError("figure dataset section must be an object")
    referenced: set[str] = set()
    for name, manifest in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("figure dataset names must be non-empty text")
        if not isinstance(manifest, Mapping) or manifest.get("values_key") != name:
            raise ValueError(f"figure dataset {name!r} must own its same-named array")
        snapshot_from_manifest(manifest, arrays, embedded=True)
        keys = manifest_array_keys(manifest)
        if keys not in ((name,), (name, f"{name}.validity")):
            raise ValueError(f"figure dataset {name!r} uses non-canonical member names")
        duplicate = referenced.intersection(keys)
        if duplicate:
            raise ValueError(
                "figure datasets contain duplicate array reference "
                f"{sorted(duplicate)!r}"
            )
        referenced.update(keys)


def read_archive(path: object) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read and fully validate one figure before returning any member."""

    with np.load(path, allow_pickle=False) as archive:
        zip_names = archive.zip.namelist()
        if any(not name.endswith(".npy") for name in zip_names):
            raise ValueError("figure NPZ may contain only NPY members")
        names = [name[:-4] for name in zip_names]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"figure contains duplicate NPZ members {duplicates!r}")
        if _INFO_KEY not in names:
            raise ValueError(f"figure carries no {_INFO_KEY} document")
        info = _decode_info(np.asarray(archive[_INFO_KEY]))
        expected = {_INFO_KEY, *info["members"]}
        actual = set(names)
        if actual != expected:
            raise ValueError(
                "figure NPZ members mismatch; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        arrays: dict[str, np.ndarray] = {}
        for name, descriptor in info["members"].items():
            array = np.asarray(archive[name])
            expected_shape = tuple(descriptor["shape"])
            if array.shape != expected_shape:
                raise ValueError(
                    f"figure member {name!r} shape {array.shape} "
                    f"does not match metadata {expected_shape}"
                )
            if array.dtype.str != descriptor["dtype"]:
                raise ValueError(
                    f"figure member {name!r} dtype {array.dtype.str!r} "
                    f"does not match metadata {descriptor['dtype']!r}"
                )
            arrays[name] = array
        _validate_datasets(info, arrays)
    return info, arrays


def read_dataset(
    info: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    name: str,
) -> OwnedSnapshot:
    """Rebuild one typed dataset from an already validated figure."""

    sections = info.get("sections")
    if not isinstance(sections, Mapping):
        raise ValueError("figure metadata has no valid sections object")
    manifests = sections.get("dataset")
    if not isinstance(manifests, Mapping):
        raise KeyError(f"{name!r} was saved as a bare array")
    manifest = manifests.get(name)
    if manifest is None:
        raise KeyError(f"{name!r} was saved as a bare array")
    return snapshot_from_manifest(manifest, arrays, embedded=True)
