"""Vendor files live WITH the device family that needs them.

Bench-wide rule (user decree): a driver that needs a vendor artifact -- a
DLL, an SDK -- looks in the ``vendor/`` folder beside its own module, and
nowhere magic.  The operator either copies the file into that folder or
writes its absolute path into ``vendor/vendor.json``; a missing artifact is
reported as exactly that instruction, never as a bare "not found" against
a path nobody can see or change.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_NAME = "vendor.json"


def vendor_directory(anchor_file: str) -> Path:
    """The family's own vendor folder, beside the module that asks."""

    return Path(anchor_file).resolve().parent / "vendor"


def resolve_vendor_file(anchor_file: str, filename: str, *, what: str) -> str:
    """One vendor artifact, or an instruction naming the folder and file.

    Resolution order: an explicit path in ``vendor/vendor.json`` (keyed by
    the file name), then the file itself inside ``vendor/``.  ``what`` names
    the vendor package in the operator's terms ("the Vaunix LMS SDK").
    """

    directory = vendor_directory(anchor_file)
    manifest = directory / MANIFEST_NAME
    if manifest.is_file():
        try:
            mapping = json.loads(manifest.read_text(encoding="utf-8"))
        except ValueError as error:
            raise FileNotFoundError(
                f"{manifest} is not valid JSON: {error}"
            ) from error
        entry = mapping.get(filename) if isinstance(mapping, dict) else None
        if entry:
            path = Path(str(entry)).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f"{manifest} points {filename!r} at {path}, which does "
                    "not exist"
                )
            return str(path)
    candidate = directory / filename
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        f"{what} is not installed: copy {filename} into {directory}, or "
        f'write {{"{filename}": "<absolute path to {filename}>"}} into '
        f"{manifest}"
    )


__all__ = ["MANIFEST_NAME", "resolve_vendor_file", "vendor_directory"]
