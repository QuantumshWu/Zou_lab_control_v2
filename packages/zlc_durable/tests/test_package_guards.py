"""This package stays at the standard-library-only bottom of the stack."""

from __future__ import annotations

import __future__
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


def _public_namespace() -> set[str]:
    """Every name a caller can reach through the package, not just __all__.

    An allow-list over __all__ alone misses names that leak in from elsewhere,
    which is how a sibling package froze 70 exports while believing it had a
    guard.  Two kinds of name are reachable without being API and are excluded
    by TYPE rather than by spelling, so the exclusion cannot rot: submodules,
    and the ``annotations`` object that ``from __future__ import annotations``
    binds in every module of every package in this workspace.
    """

    submodules = {path.stem for path in SRC.glob("*.py")}
    namespace = set()
    for name in dir(zlc_durable):
        if name.startswith("_") or name in submodules:
            continue
        if isinstance(getattr(zlc_durable, name), __future__._Feature):
            continue
        namespace.add(name)
    return namespace | {"__version__"}


def test_public_surface_is_the_declared_list() -> None:
    assert set(zlc_durable.__all__) == _public_namespace()
    assert all(hasattr(zlc_durable, name) for name in zlc_durable.__all__)


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
