"""Small, domain-free helpers for visible shape labels."""

from __future__ import annotations

import re

__all__ = ["describe_shape_text", "indexed_unique_name"]


def describe_shape_text(value: str) -> str:
    """Return an already-computed shape label for display.

    Shape validation and formatting belong to the data owner.  The UI layer
    accepts only the resulting string and never imports a schema or array
    implementation.
    """

    if not isinstance(value, str):
        raise TypeError("shape label must be str")
    if not value:
        raise ValueError("shape label must be non-empty")
    return value


_INDEX_SUFFIX_RE = re.compile(r"^(.*?)\s*#\d+$")


def indexed_unique_name(base: str, taken) -> str:
    """Return ``"<root> #N"`` using the smallest free positive index."""

    text = str(base or "panel").strip() or "panel"
    match = _INDEX_SUFFIX_RE.match(text)
    root = (match.group(1).strip() if match else text) or "panel"
    occupied = set(taken)
    index = 1
    while f"{root} #{index}" in occupied:
        index += 1
    return f"{root} #{index}"
