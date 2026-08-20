"""This package stays at the standard-library-only bottom of the stack."""

from __future__ import annotations

import ast
from pathlib import Path

import zlc_durable

SRC = Path(zlc_durable.__file__).resolve().parent

_STDLIB_ALLOWED = {
    "__future__",
    "ctypes",
    "datetime",
    "json",
    "math",
    "os",
    "pathlib",
    "re",
    "tempfile",
    "typing",
}


def test_source_imports_only_the_standard_library() -> None:
    """No numpy, no zlc_*, nothing installable.

    This package is the bottom of the stack; a dependency here is a dependency
    everywhere.
    """

    offenders: list[tuple[str, str]] = []
    files = tuple(SRC.glob("*.py"))
    assert files, "scan found no source files"

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [] if node.level else [(node.module or "").split(".")[0]]
            else:
                continue
            offenders.extend((path.name, root) for root in roots if root not in _STDLIB_ALLOWED)

    assert offenders == [], f"unexpected imports: {offenders}"
