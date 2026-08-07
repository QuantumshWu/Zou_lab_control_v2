"""Small scalar and mapping validators owned by the zlc_data layer."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any


#: How wide a content name is, everywhere in this project.
#:
#: 128 bits.  These names are compared for equality by code that wrote both
#: sides; nobody forges a schema, so cryptographic strength buys nothing, and
#: 256 bits would be claiming a guarantee this does not need while putting
#: sixty-four characters of noise in every manifest a human reads.  At 128 bits
#: an accidental collision across every artifact this experiment will ever
#: write is not a thing that happens.
#:
#: BLAKE2b because it is in the standard library, takes a digest size directly,
#: and is faster than SHA-256 at the sizes involved.
DIGEST_BITS = 128
DIGEST_HEX = DIGEST_BITS // 4

def canonical_text(value: object, field: str, *, empty: bool = False) -> str:
    """Validate one already-canonical text value."""

    if not isinstance(value, str) or value.strip() != value or (not empty and not value):
        qualifier = "canonical text" if empty else "canonical non-empty text"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def digest_text(value: object, field: str, *, optional: bool = False) -> str | None:
    """Validate one lowercase content name without recomputing it."""

    if optional and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != DIGEST_HEX
        or any(character not in "0123456789abcdef" for character in value)
    ):
        suffix = " or None" if optional else ""
        raise ValueError(
            f"{field} must be a lowercase {DIGEST_BITS}-bit content name{suffix}"
        )
    return value


def integer(
    value: object,
    field: str = "value",
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return normalized


def nonnegative_integer(value: object, field: str) -> int:
    return integer(value, field, minimum=0)


def positive_integer(value: object, field: str) -> int:
    return integer(value, field, minimum=1)


def finite_real(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    positive: bool = False,
    normalize_zero: bool = True,
) -> float:
    """Validate one finite real value at the data boundary."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return 0.0 if normalize_zero and result == 0.0 else result


def exact_mapping(
    value: object,
    fields: set[str] | frozenset[str],
    format_name: str,
    *,
    discriminator: str | None = "schema",
) -> dict[str, Any]:
    """Admit exactly one typed mapping shape."""

    expected = set(fields)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{format_name} must contain exactly {sorted(expected)}")
    if discriminator is not None and value.get(discriminator) != format_name:
        raise ValueError(
            f"expected {discriminator} {format_name!r}, "
            f"got {value.get(discriminator)!r}"
        )
    return value


__all__ = [
    "DIGEST_BITS",
    "DIGEST_HEX",
    "canonical_text",
    "exact_mapping",
    "finite_real",
    "integer",
    "nonnegative_integer",
    "positive_integer",
    "digest_text",
]
