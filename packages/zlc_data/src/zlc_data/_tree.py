"""Private deterministic encoding for zlc_data's in-memory primitive trees."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .validation import DIGEST_BITS


#: Types _normalize provably leaves untouched: exactly the leaves json
#: serializes as-is.  Exact-type membership on purpose -- an int subclass
#: (an IntEnum, a numpy scalar) falls through to the per-item walk.
_SCALAR_TYPES = frozenset((str, int, float, bool, type(None)))


def _normalize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("tree mapping keys must be strings")
        return {key: _normalize(value[key]) for key in value}
    if isinstance(value, (list, tuple)):
        # A coordinate table is one list of hundreds of thousands of plain
        # scalars; walking it item by item in Python was ~100 ms per
        # materialized indexed schema.  One C-level type sweep proves the
        # whole list is already normal.
        if _SCALAR_TYPES.issuperset(map(type, value)):
            return list(value)
        return [_normalize(item) for item in value]
    return value


def encode(value: Any) -> bytes:
    """Encode a validated primitive tree deterministically for equality checks."""

    return json.dumps(
        _normalize(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def digest(value: Any) -> str:
    """One content name for a validated primitive tree."""

    return hashlib.blake2b(encode(value), digest_size=DIGEST_BITS // 8).hexdigest()

