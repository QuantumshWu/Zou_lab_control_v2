"""Strict, pickle-free NPZ persistence for role-axis data snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO
import json
import os
import zipfile

import numpy as np

from .codec import (
    dataset_revision_ref_from_tree,
    dataset_revision_ref_to_tree,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
)
from .value import (
    DataBlock,
    OwnedSnapshot,
)
from .validity import (
    INVALID,
    VALID,
    CellValidity,
    DatasetComponentValidity,
    Invalid,
    Valid,
)


_FORMAT = "zlc-data-role-axis-npz"
_VERSION = 1
_MANIFEST = "manifest"
_VALUES = "values"
_VALIDITY = "validity"


class NPZFormatError(ValueError):
    """Raised when an NPZ payload is missing or violates the data contract."""


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NPZFormatError(f"manifest JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise NPZFormatError(f"manifest JSON contains non-finite number {value}")


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise NPZFormatError(
            f"{path} keys mismatch; missing={missing}, extra={extra}"
        )


def _manifest_validity(
    validity: object,
    arrays: dict[str, np.ndarray],
    validity_key: str,
) -> dict[str, Any]:
    if isinstance(validity, Valid):
        return {"kind": "valid"}
    if isinstance(validity, Invalid):
        return {"kind": "invalid"}
    if isinstance(validity, CellValidity):
        arrays[validity_key] = validity.mask
        return {"kind": "cell", "mask_key": validity_key}
    if isinstance(validity, DatasetComponentValidity):
        arrays[validity_key] = validity.mask
        return {
            "kind": "dataset_component",
            "axis_ids": [axis_id.value for axis_id in validity.axis_ids],
            "mask_key": validity_key,
        }
    raise TypeError(f"unsupported DataBlock validity {type(validity).__name__}")


def _array(
    arrays: Mapping[str, Any],
    key: Any,
    expected_keys: set[str],
    path: str,
) -> np.ndarray:
    if not isinstance(key, str) or not key:
        raise NPZFormatError(f"{path} must be a non-empty array key")
    if key in expected_keys:
        raise NPZFormatError(f"{path} contains duplicate array reference {key!r}")
    expected_keys.add(key)
    if key not in arrays:
        raise NPZFormatError(f"missing array {key!r}")
    try:
        return np.asarray(arrays[key])
    except (TypeError, ValueError) as exc:
        raise NPZFormatError(f"cannot read array {key!r}: {exc}") from exc


def snapshot_manifest(
    snapshot: OwnedSnapshot,
    arrays: dict[str, np.ndarray],
    *,
    values_key: str = _VALUES,
    validity_key: str = _VALIDITY,
) -> dict[str, Any]:
    """One snapshot's identity as plain data, putting its arrays in ``arrays``.

    Separated from :func:`save_npz` so a file holding SEVERAL datasets can
    carry each one's identity without re-deriving what identity means.  A saved
    figure needs exactly that: without it the arrays survive but their axes,
    units and revision do not, and what comes back is numbers nobody can plot
    the way they were plotted.

    The array keys are parameters for the same reason -- one file, many
    datasets, no collisions.
    """

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    if values_key == validity_key:
        raise ValueError("values and validity need distinct array keys")
    arrays[values_key] = snapshot.block.values
    return {
        "format": _FORMAT,
        "version": _VERSION,
        "schema": dataset_schema_to_tree(snapshot.block.schema),
        "ref": dataset_revision_ref_to_tree(snapshot.ref),
        "values_key": values_key,
        "validity": _manifest_validity(
            snapshot.block.validity, arrays, validity_key
        ),
    }


def manifest_array_keys(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Which arrays one manifest refers to, in the order it names them."""

    keys = [manifest["values_key"]]
    validity = manifest.get("validity")
    if isinstance(validity, Mapping) and isinstance(validity.get("mask_key"), str):
        keys.append(validity["mask_key"])
    return tuple(keys)


def snapshot_from_manifest(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    embedded: bool = False,
) -> OwnedSnapshot:
    """Rebuild one snapshot from its manifest and the arrays it names."""

    if type(embedded) is not bool:
        raise TypeError("embedded must be bool")
    if not isinstance(manifest, Mapping):
        raise NPZFormatError("manifest root must be an object")
    _exact_keys(
        manifest,
        {"format", "version", "schema", "ref", "values_key", "validity"},
        "manifest",
    )
    if (
        manifest["format"] != _FORMAT
        or isinstance(manifest["version"], bool)
        or not isinstance(manifest["version"], int)
        or manifest["version"] != _VERSION
    ):
        raise NPZFormatError(
            f"unsupported data format {manifest['format']!r} "
            f"version {manifest['version']!r}"
        )
    referenced: set[str] = set()
    schema = dataset_schema_from_tree(manifest["schema"])
    ref_tree = dict(manifest["ref"])
    has_fingerprint = "schema_fingerprint" in ref_tree
    if embedded:
        if has_fingerprint:
            raise NPZFormatError(
                "embedded manifest ref must not repeat schema_fingerprint"
            )
        ref_tree["schema_fingerprint"] = schema.fingerprint
    elif not has_fingerprint:
        raise NPZFormatError("manifest.ref is missing schema_fingerprint")
    ref = dataset_revision_ref_from_tree(ref_tree)
    values = _array(arrays, manifest["values_key"], referenced, "manifest.values_key")
    validity = _validity_from_manifest(manifest["validity"], arrays, referenced)
    block = DataBlock(ref.block_id, ref.revision, values, validity, schema)
    return OwnedSnapshot(ref, block)


def save_npz(
    path: str | os.PathLike[str] | BinaryIO,
    snapshot: OwnedSnapshot,
) -> None:
    """Save one owned snapshot without pickle or plotting/device state."""

    arrays: dict[str, np.ndarray] = {}
    manifest = snapshot_manifest(snapshot, arrays)
    try:
        encoded = json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise NPZFormatError(f"metadata is not serializable: {exc}") from exc
    arrays[_MANIFEST] = np.asarray(encoded)
    if hasattr(path, "write"):
        np.savez_compressed(path, **arrays)
        return
    destination = Path(path)
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def _validity_from_manifest(
    raw: Any,
    arrays: Mapping[str, Any],
    expected_keys: set[str],
) -> Valid | Invalid | CellValidity | DatasetComponentValidity:
    if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
        raise NPZFormatError("manifest.validity must be a tagged object")
    kind = raw["kind"]
    if kind == "valid":
        _exact_keys(raw, {"kind"}, "manifest.validity")
        return VALID
    if kind == "invalid":
        _exact_keys(raw, {"kind"}, "manifest.validity")
        return INVALID
    if kind == "cell":
        _exact_keys(raw, {"kind", "mask_key"}, "manifest.validity")
        return CellValidity(_array(arrays, raw["mask_key"], expected_keys, "validity.mask_key"))
    if kind == "dataset_component":
        _exact_keys(
            raw,
            {"kind", "axis_ids", "mask_key"},
            "manifest.validity",
        )
        axis_ids = raw["axis_ids"]
        if not isinstance(axis_ids, list) or any(not isinstance(item, str) for item in axis_ids):
            raise NPZFormatError("manifest.validity.axis_ids must be a string list")
        from .axis import AxisId

        return DatasetComponentValidity(
            tuple(AxisId(item) for item in axis_ids),
            _array(arrays, raw["mask_key"], expected_keys, "validity.mask_key"),
        )
    raise NPZFormatError(f"invalid validity kind {kind!r}")


def load_npz(path: str | os.PathLike[str] | BinaryIO) -> OwnedSnapshot:
    """Load and exhaustively validate a :func:`save_npz` file."""

    try:
        with np.load(path, allow_pickle=False) as npz:
            if "manifest" not in npz.files:
                raise NPZFormatError("missing manifest array")
            manifest_array = np.asarray(npz[_MANIFEST])
            if manifest_array.shape != () or manifest_array.dtype.kind != "U":
                raise NPZFormatError("manifest must be a scalar Unicode array")
            try:
                manifest = json.loads(
                    str(manifest_array.item()),
                    object_pairs_hook=_strict_json_object,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, TypeError) as exc:
                raise NPZFormatError(f"invalid manifest JSON: {exc}") from exc
            members = {name: npz[name] for name in npz.files}
            snapshot = snapshot_from_manifest(manifest, members)
            expected_keys = {_MANIFEST, *manifest_array_keys(manifest)}
            if set(npz.files) != expected_keys:
                raise NPZFormatError(
                    f"NPZ members mismatch; missing={sorted(expected_keys - set(npz.files))}, "
                    f"extra={sorted(set(npz.files) - expected_keys)}"
                )
            return snapshot
    except NPZFormatError:
        raise
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        raise NPZFormatError(f"invalid NPZ dataset: {exc}") from exc


__all__ = [
    "NPZFormatError",
    "load_npz",
    "manifest_array_keys",
    "save_npz",
    "snapshot_from_manifest",
    "snapshot_manifest",
]
