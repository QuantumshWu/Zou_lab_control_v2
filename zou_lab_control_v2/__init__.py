"""Bootstrap and installed metadata for the single Zou Lab Control product."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Mapping
import tomllib


__all__ = ["DISTRIBUTION_NAME", "ROOT", "entry_specs"]

DISTRIBUTION_NAME = "zou-lab-control"
# In a checkout this is the repository root; in a wheel it is site-packages.
ROOT = Path(__file__).resolve().parent.parent


def entry_specs(group: str) -> Mapping[str, str]:
    """Return one entry-point group from source manifest or installed metadata."""

    manifest = ROOT / "pyproject.toml"
    if manifest.is_file():
        document = tomllib.loads(manifest.read_text(encoding="utf-8"))
        groups = document.get("project", {}).get("entry-points", {})
        entries = groups.get(group)
        if not isinstance(entries, dict) or not entries:
            raise RuntimeError(f"product manifest has no {group!r} entry-point group")
        if any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in entries.items()
        ):
            raise RuntimeError(f"product manifest {group!r} entries must be text")
        return dict(sorted(entries.items()))

    try:
        installed = distribution(DISTRIBUTION_NAME)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"{DISTRIBUTION_NAME} is not installed") from exc
    entries = {
        item.name: item.value
        for item in installed.entry_points
        if item.group == group
    }
    if not entries:
        raise RuntimeError(f"installed product has no {group!r} entry-point group")
    return dict(sorted(entries.items()))
