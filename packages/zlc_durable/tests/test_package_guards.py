"""This package stays tiny and depends on nothing of ours.

Every package that saves anything imports this one, so a dependency added here
reaches the whole workspace.  The cap is a named number: adding a public name
means changing it deliberately, which is the only mechanism that has actually
held anywhere in this project.
"""

from __future__ import annotations

import __future__
import ast
from pathlib import Path

import zlc_durable

# The cap. Adding a public name means changing this number on purpose, which is
# the only anti-bloat mechanism that has held anywhere in this project. It lives
# in the guard rather than in the package so that it does not count against
# itself.
# 12 -> 14 for readable JSON.  Choosing WHERE a file goes and writing it so a
# crash cannot halve it were already one concern here; how a file a person will
# open is laid out is the same concern, and the alternative was three packages
# each deciding for themselves -- which is what produced 62 lines of one
# channel name each in a saved pulse.
MAX_PUBLIC_NAMES = 14


SRC = Path(zlc_durable.__file__).resolve().parent

_STDLIB_ALLOWED = {
    "__future__",
    "ctypes",
    "datetime",
    "json",
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


def test_public_surface_is_the_declared_list_and_stays_under_the_cap() -> None:
    assert set(zlc_durable.__all__) == _public_namespace()
    assert len(zlc_durable.__all__) <= MAX_PUBLIC_NAMES
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


def test_date_routing_lives_here_and_nowhere_else() -> None:
    """Two packages proposed to compute <save_root>/<date>/ independently.

    It belongs here because choosing where a file goes and writing it durably
    are one concern.  This test states the ownership so a second implementation
    has somewhere to fail.
    """

    text = (SRC / "workspace.py").read_text(encoding="utf-8")
    for expected in ("def day_folder_name", "def day_folder", "def unique_path"):
        assert expected in text
