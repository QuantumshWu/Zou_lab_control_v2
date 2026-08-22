"""Verify that every layer comes from this one source or installed product."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, distribution, packages_distributions
from pathlib import Path

from zou_lab_control_v2 import DISTRIBUTION_NAME, ROOT, entry_specs


_LAYER_GROUP = "zou_lab_control.layers"
OWNED = {name: name for name in entry_specs(_LAYER_GROUP)}
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
        return "namespace", Path(locations[0]).resolve() if locations else None
    return "module", Path(spec.origin).resolve()


def _normalized(name: str) -> str:
    return name.lower().replace("_", "-")


def check() -> list[str]:
    problems: list[str] = []
    source_manifest = ROOT / "pyproject.toml"
    installed_files: set[Path] = set()
    if not source_manifest.is_file():
        try:
            product = distribution(DISTRIBUTION_NAME)
        except PackageNotFoundError:
            return [f"{DISTRIBUTION_NAME}: product distribution is not installed"]
        installed_files = {
            Path(product.locate_file(item)).resolve()
            for item in (product.files or ())
        }
    ownership = packages_distributions()

    for name in OWNED:
        kind, where = _origin(name)
        if kind != "module" or where is None:
            problems.append(f"{name}: expected an installed module, got {kind} at {where}")
            continue
        if source_manifest.is_file():
            expected = (ROOT / "packages" / name / "src" / name).resolve()
            if where.parent != expected:
                problems.append(f"{name}: resolves to {where}, expected under {expected}")
        else:
            owners = {_normalized(item) for item in ownership.get(name, ())}
            expected_owners = {_normalized(DISTRIBUTION_NAME)}
            if owners != expected_owners or where not in installed_files:
                problems.append(
                    f"{name}: {where} owners={sorted(owners)}; expected only "
                    f"{DISTRIBUTION_NAME}"
                )

    for name in RETIRED:
        kind, where = _origin(name)
        if kind != "missing":
            problems.append(f"{name}: retired package is still importable from {where}")
    return problems


def main() -> int:
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
    mode = "source manifest" if (ROOT / "pyproject.toml").is_file() else DISTRIBUTION_NAME
    print(f"all {len(OWNED)} layers belong to one {mode}; {len(RETIRED)} retired names are gone")
    return 0
