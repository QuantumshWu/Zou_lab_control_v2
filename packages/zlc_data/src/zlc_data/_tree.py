"""Private deterministic encoding for zlc_data's in-memory primitive trees."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .validation import DIGEST_BITS


def encode(value: Any) -> bytes:
    """Encode a validated primitive tree deterministically for equality checks."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def digest(value: Any) -> str:
    """One content name for a validated primitive tree."""

    return hashlib.blake2b(encode(value), digest_size=DIGEST_BITS // 8).hexdigest()

