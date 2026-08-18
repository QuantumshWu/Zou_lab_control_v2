"""Exact diagnostic text for integer authority values."""

from __future__ import annotations

from numbers import Integral


def exact_integer_text(value: object) -> str:
    """Return a complete, reversible integer representation.

    Python may reject very large decimal conversions through its interpreter
    safety limit.  Hexadecimal has no such limit and still preserves every bit,
    so it is the exact fallback rather than an application-level truncation.
    """

    if isinstance(value, bool) or not isinstance(value, Integral):
        return repr(value)
    integer = int(value)
    try:
        return str(integer)
    except ValueError:
        return hex(integer)


__all__ = ["exact_integer_text"]
