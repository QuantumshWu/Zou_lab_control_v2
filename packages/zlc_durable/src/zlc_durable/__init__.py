"""Durable filesystem primitives: write atomically, land where you meant to.

Every package that saves anything depends on this and on nothing else of ours.
It deliberately does NOT carry a canonical encoder or content digests -- an
archive in this project is plain JSON and plain arrays, readable without
importing any of our packages.
"""

from __future__ import annotations

from pathlib import Path as _Path

_PACKAGE_DIR = _Path(__file__).resolve().parent
if _PACKAGE_DIR.name != "zlc_durable" or __package__ != "zlc_durable":
    raise ImportError(f"unexpected zlc_durable installation path: {_PACKAGE_DIR}")

from .readable import readable_json_bytes, write_readable_json
from .durability import (  # noqa: E402
    DirectoryDurabilityError,
    atomic_write_bytes,
    atomic_write_file,
    atomic_write_text,
    durable_makedirs,
)
from .workspace import day_folder, day_folder_path, unique_path  # noqa: E402

# The whole public surface. A caller writes bytes, creates a directory tree, or
# asks where durable work should land -- implementation primitives stay owned
# by their submodules.
__all__ = [
    "DirectoryDurabilityError",
    "atomic_write_bytes",
    "atomic_write_file",
    "atomic_write_text",
    "readable_json_bytes",
    "write_readable_json",
    "day_folder",
    "day_folder_path",
    "durable_makedirs",
    "unique_path",
]
