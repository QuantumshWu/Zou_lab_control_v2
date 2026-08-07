"""Assert that every package name resolves to the repo that owns it.

This project has been bitten repeatedly by shadow imports: an old monolith
installed under the same top-level names, a package that was never installed
resolving to an empty namespace package because the current directory happened
to sit beside it, and an editable install pointing at a copy that had been
deleted.  In every case ``import`` succeeded and the wrong code ran.

Run this before anything else, from a directory that is NOT the workspace root,
so a namespace package cannot mask a missing install.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[4]

OWNED = {
    "zlc_data": "zlc_data",
    "zlc_runtime": "zlc_runtime",
    "zlc_plot": "zlc_plot",
    "zlc_pulse": "zlc_pulse",
    "zlc_ui": "zlc_ui",
    "zlc_atom": "zlc_atom",
    "zlc_workbench": "zlc_workbench",
    "zlc_durable": "zlc_durable",
}

# Names the pre-split monolith used.  Nothing may resolve them: the reference
# tree is a read-only source to migrate FROM, never an importable dependency.
RETIRED = ("zlc_storage", "zlc_frontend", "zlc_neutral_atom")


def _origin(name: str) -> tuple[str, Path | None]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return "missing", None
    if spec is None:
        return "missing", None
    if spec.origin in (None, "namespace"):
        locations = list(spec.submodule_search_locations or ())
        return "namespace", Path(locations[0]) if locations else None
    return "module", Path(spec.origin)


def check() -> list[str]:
    problems: list[str] = []

    for name, repo in OWNED.items():
        kind, where = _origin(name)
        expected = WORKSPACE / repo / "src" / name
        if kind == "missing":
            problems.append(f"{name}: not installed (pip install -e {WORKSPACE / repo})")
        elif kind == "namespace":
            problems.append(
                f"{name}: resolves to an EMPTY namespace package at {where} — it is not "
                f"installed, and only the current directory made the import appear to work"
            )
        elif expected not in where.parents and where.parent != expected:
            problems.append(f"{name}: resolves to {where}, expected under {expected}")

    for name in RETIRED:
        kind, where = _origin(name)
        if kind != "missing":
            problems.append(
                f"{name}: still importable from {where} — the pre-split monolith is a "
                f"read-only reference, not an installed package (pip uninstall zou-lab-control)"
            )

    return problems


def main() -> int:
    if Path.cwd() == WORKSPACE:
        print(f"run this from outside {WORKSPACE}, or a namespace package can mask a missing install")
        return 2

    problems = check()
    for name in OWNED:
        kind, where = _origin(name)
        marker = "  " if kind == "module" else "!!"
        print(f"{marker} {name:16s} {where}")
    print()
    if problems:
        for problem in problems:
            print(f"FAIL  {problem}")
        return 1
    print(f"all {len(OWNED)} packages resolve to their own repo; {len(RETIRED)} retired names are gone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
