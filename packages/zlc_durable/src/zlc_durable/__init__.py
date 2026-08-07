"""Durable filesystem primitives: write atomically, land where you meant to.

Every package that saves anything depends on this and on nothing else of ours.
It deliberately does NOT carry a canonical encoder or content digests -- an
archive in this project is plain JSON and plain arrays, readable without
importing any of our packages.

Migrated verbatim from the pre-split tree's ``zlc_storage`` (durability.py 224
lines, paths.py 33 lines, byte-identical), minus its 701-line canonical module.
``workspace.py`` is new: it owns the save-root/date-folder layout, because
choosing where a file goes and writing it safely are one concern.
"""

from __future__ import annotations

from pathlib import Path as _Path

__version__ = "0.1.0"
_PACKAGE_DIR = _Path(__file__).resolve().parent
if _PACKAGE_DIR.name != "zlc_durable" or __package__ != "zlc_durable":
    raise ImportError(f"unexpected zlc_durable installation path: {_PACKAGE_DIR}")

from .readable import readable_json, write_readable_json
from .durability import (  # noqa: E402
    DirectoryDurabilityError,
    atomic_write_bytes,
    atomic_write_file,
    atomic_write_text,
    durable_makedirs,
    durable_mkdir,
    flush_directory,
)
from .paths import resolve_under  # noqa: E402
from .workspace import day_folder, day_folder_name, unique_path  # noqa: E402

# The whole public surface.  A caller writes bytes, makes a directory, resolves
# a path safely, or asks where today's work goes -- there is nothing else here.
__all__ = [
    "DirectoryDurabilityError",
    "atomic_write_bytes",
    "atomic_write_file",
    "atomic_write_text",
    "readable_json",
    "write_readable_json",
    "day_folder",
    "day_folder_name",
    "durable_makedirs",
    "durable_mkdir",
    "flush_directory",
    "resolve_under",
    "unique_path",
    "__version__",
]
