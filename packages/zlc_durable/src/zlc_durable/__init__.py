"""Durable filesystem primitives: write atomically, land where you meant to.

Every package that saves anything depends on this and on nothing else of ours.
It deliberately does NOT carry a canonical encoder or content digests -- an
archive in this project is plain JSON and plain arrays, readable without
importing any of our packages.
"""

from __future__ import annotations

from .readable import readable_json_bytes, strict_json_loads, write_readable_json
from .durability import (
    DirectoryDurabilityError,
    atomic_write_bytes,
    atomic_write_file,
    atomic_write_text,
    durable_makedirs,
)
from .workspace import day_folder, day_folder_path, unique_path

# The whole public surface. A caller writes bytes, creates a directory tree,
# reads JSON back strictly, or asks where durable work should land --
# implementation primitives stay owned by their submodules.
__all__ = [
    "DirectoryDurabilityError",
    "atomic_write_bytes",
    "atomic_write_file",
    "atomic_write_text",
    "readable_json_bytes",
    "strict_json_loads",
    "write_readable_json",
    "day_folder",
    "day_folder_path",
    "durable_makedirs",
    "unique_path",
]
