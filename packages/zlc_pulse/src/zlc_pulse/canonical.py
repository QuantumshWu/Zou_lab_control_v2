"""Small deterministic encoding helpers used only at the ABI boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from enum import Enum
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a package dependency
    np = None


def _normal(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical values must be finite")
        return value
    if isinstance(value, Enum):
        return _normal(value.value)
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if np is not None and isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "values": array.tolist(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _normal(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normal(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _normal(getattr(value, field.name))
            for field in fields(value)
            if field.init
        }
    if hasattr(value, "__dict__"):
        return _normal(vars(value))
    raise TypeError(f"unsupported canonical value {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Encode a value with stable map ordering and compact UTF-8 JSON."""

    return json.dumps(
        _normal(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


#: The same width zlc_data uses for a content name.  Stated again rather than
#: imported: this package depends on numpy and pyserial and nothing else, so a
#: board can be driven with no data layer present.  zlc_workbench, which sees
#: both, checks that the two agree.
DIGEST_BITS = 128


def canonical_digest(value: Any) -> str:
    """One content name for a canonically encoded value.

    128 bits, not 256.  These names are compared for equality by code that
    wrote both sides -- nobody forges a compiled program -- so cryptographic
    strength buys nothing and the extra thirty-two characters are noise.
    """

    return hashlib.blake2b(
        canonical_bytes(value), digest_size=DIGEST_BITS // 8
    ).hexdigest()


__all__ = ["DIGEST_BITS", "canonical_bytes", "canonical_digest"]
