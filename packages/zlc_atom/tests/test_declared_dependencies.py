"""What the package declares it needs must be what it actually imports.

Every logic node imported zlc_runtime at module scope while the package declared
only numpy, scipy and zlc-data.  A clean `pip install zlc-atom` therefore
produced a package whose three nodes raised ImportError the moment they were
used -- and nothing noticed, because the development checkout had everything
installed anyway.

Derived from the source rather than listed by hand: a list maintained beside the
imports drifts from them, which is the failure it is supposed to catch.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import zlc_atom


SRC = Path(zlc_atom.__file__).resolve().parent
ROOT = SRC.parents[1]

#: Distribution name for each of our top-level packages.
_DISTRIBUTION = {
    "zlc_data": "zlc-data",
    "zlc_runtime": "zlc-runtime",
    "zlc_durable": "zlc-durable",
    "zlc_pulse": "zlc-pulse",
    "zlc_plot": "zlc-plot",
    "zlc_ui": "zlc-ui",
}


def _imported_distributions() -> set[str]:
    found: set[str] = set()
    files = tuple(SRC.rglob("*.py"))
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
            for root in roots:
                if root in _DISTRIBUTION and root != "zlc_atom":
                    found.add(_DISTRIBUTION[root])
    return found


def _declared_distributions() -> set[str]:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = metadata["project"]["dependencies"]
    return {
        name.split("==")[0].split(">=")[0].strip()
        for name in declared
        if name.split("==")[0].split(">=")[0].strip().startswith("zlc-")
    }


def test_every_package_the_source_imports_is_declared() -> None:
    imported = _imported_distributions()
    assert imported, "scan found no sibling imports"
    missing = imported - _declared_distributions()
    assert missing == set(), f"imported but not declared: {sorted(missing)}"


def test_nothing_is_declared_that_the_source_never_imports() -> None:
    """An unused dependency is a claim about the boundary that is not true."""

    unused = _declared_distributions() - _imported_distributions()
    assert unused == set(), f"declared but never imported: {sorted(unused)}"


def test_the_pulse_client_is_not_imported_by_this_package() -> None:
    """The boundary: this package says WHERE a board is, never HOW to reach it.

    The endpoint lives in a device configuration and the composition root
    supplies the dialler, so the domain stays independent of the device packages.
    """

    assert "zlc-pulse" not in _imported_distributions()
