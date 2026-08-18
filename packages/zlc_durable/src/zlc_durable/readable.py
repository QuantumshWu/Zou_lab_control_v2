"""JSON a person is expected to read, written the way this project writes files.

``json.dumps(indent=2)`` gives every element of every list its own line, so a
board's 62 channel names arrive as 62 lines of ``"ch07",`` and the structure
around them -- which is the part being read -- scrolls off the screen.  The
files this produces are read by people: a saved pulse, a calibration, an
apparatus configuration.  Reading them is what they are for.

The rule is one line: a list whose items are all scalars stays on one line,
wrapped at a sensible width; anything with structure inside it expands.  That
is the whole difference between a file you can see the shape of and a file you
have to scroll.

Not for anything hashed or sent.  A digest wants the compact, sorted,
separator-pinned form and must not move when this changes its mind about where
to wrap -- those callers stay on ``json.dumps`` with their own separators, and
should.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .durability import atomic_write_bytes


__all__ = ["readable_json", "readable_json_bytes", "write_readable_json"]

#: Where an inline list is wrapped.  Wide enough that a row of numbers or a
#: handful of names is one line, narrow enough to read without scrolling
#: sideways -- the two things this exists to balance.
WIDTH = 96


def readable_json(tree: Any, *, indent: int = 2) -> str:
    """One JSON document, laid out for a reader.  Ends with a newline."""

    if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
        raise TypeError("indent must be a non-negative integer")
    _validate(tree, "$")
    return _render(tree, indent, 0) + "\n"


def readable_json_bytes(tree: Any, *, indent: int = 2) -> bytes:
    """The UTF-8 bytes of :func:`readable_json`, with no second layout path."""

    return readable_json(tree, indent=indent).encode("utf-8")


def write_readable_json(path: str | Path, tree: Any, *, indent: int = 2) -> Path:
    """Write one readable JSON document, atomically.

    Atomically because that is what this package is for: a file half-written
    by a crash is worse than no file, and a saved pulse is something an
    experiment is resumed from.
    """

    return atomic_write_bytes(path, readable_json_bytes(tree, indent=indent))


def _scalar(value: Any) -> bool:
    return value is None or type(value) in (bool, int, float, str)


def _validate(value: Any, path: str) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} JSON object keys must be str")
            _validate(item, f"{path}.{key}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate(item, f"{path}[{index}]")
        return
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return
    raise TypeError(f"{path} is not a plain JSON value")


def _render(value: Any, indent: int, depth: int) -> str:
    pad = " " * (indent * depth)
    inner = " " * (indent * (depth + 1))
    if type(value) is dict:
        if not value:
            return "{}"
        items = [
            f"{inner}{json.dumps(key, ensure_ascii=False)}: "
            f"{_render(item, indent, depth + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if type(value) is list:
        items = value
        if not items:
            return "[]"
        if all(_scalar(item) for item in items):
            return _inline(items, indent, depth)
        rendered = [f"{inner}{_render(item, indent, depth + 1)}" for item in items]
        return "[\n" + ",\n".join(rendered) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _inline(items: list[Any], indent: int, depth: int) -> str:
    """A list of scalars on one line, or wrapped into as few lines as fit."""

    pad = " " * (indent * depth)
    inner = " " * (indent * (depth + 1))
    parts = [json.dumps(item, ensure_ascii=False, allow_nan=False) for item in items]
    one_line = "[" + ", ".join(parts) + "]"
    if len(one_line) + len(pad) <= WIDTH:
        return one_line
    lines: list[str] = []
    current = inner
    for index, part in enumerate(parts):
        piece = part + ("," if index < len(parts) - 1 else "")
        if current != inner and len(current) + 1 + len(piece) > WIDTH:
            lines.append(current)
            current = inner
        current += (" " if current != inner else "") + piece
    lines.append(current)
    return "[\n" + "\n".join(lines) + "\n" + pad + "]"
