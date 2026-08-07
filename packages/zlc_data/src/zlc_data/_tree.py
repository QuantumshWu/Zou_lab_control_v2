"""Private deterministic encoding for zlc_data's in-memory primitive trees."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .validation import DIGEST_BITS


def _normalize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("tree mapping keys must be strings")
        return {key: _normalize(value[key]) for key in value}
    if isinstance(value, (list, tuple)):
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

