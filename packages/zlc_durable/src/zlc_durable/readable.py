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
from pathlib import Path
from typing import Any

from .durability import atomic_write_text


__all__ = ["readable_json", "write_readable_json"]

#: Where an inline list is wrapped.  Wide enough that a row of numbers or a
#: handful of names is one line, narrow enough to read without scrolling
#: sideways -- the two things this exists to balance.
WIDTH = 96


def readable_json(tree: Any, *, indent: int = 2) -> str:
    """One JSON document, laid out for a reader.  Ends with a newline."""

    return _render(tree, int(indent), 0) + "\n"


def write_readable_json(path: str | Path, tree: Any, *, indent: int = 2) -> Path:
    """Write one readable JSON document, atomically.

    Atomically because that is what this package is for: a file half-written
    by a crash is worse than no file, and a saved pulse is something an
    experiment is resumed from.
    """

    target = Path(path)
    atomic_write_text(target, readable_json(tree, indent=indent))
    return target


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _render(value: Any, indent: int, depth: int) -> str:
    pad = " " * (indent * depth)
    inner = " " * (indent * (depth + 1))
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f"{inner}{json.dumps(str(key))}: {_render(item, indent, depth + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, (list, tuple)):
        items = list(value)
        if not items:
            return "[]"
        if all(_scalar(item) for item in items):
            return _inline(items, indent, depth)
        rendered = [f"{inner}{_render(item, indent, depth + 1)}" for item in items]
        return "[\n" + ",\n".join(rendered) + "\n" + pad + "]"
    return json.dumps(value, allow_nan=False)


def _inline(items: list[Any], indent: int, depth: int) -> str:
    """A list of scalars on one line, or wrapped into as few lines as fit."""

    pad = " " * (indent * depth)
    inner = " " * (indent * (depth + 1))
    parts = [json.dumps(item, allow_nan=False) for item in items]
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
