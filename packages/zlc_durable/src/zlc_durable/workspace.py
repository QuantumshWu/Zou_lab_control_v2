"""Where saved work lands: a save root, a folder per calendar day, unique names.

The owner's rule is that everything saved on a given day sits together under a
folder named for that day -- ``<save_root>/2026_08_05/`` -- with no per-run
subdirectory.  A physicist looks for today's data by today's date, so the date
IS the organising key.

Date routing lives here, beside atomic writing and escape-safe path resolution,
because they are one concern: deciding where a file goes and putting it there
without leaving a half-written or stray file behind.  Nothing else in the
workspace may compute a save path.
"""

from __future__ import annotations

from datetime import date as _date
import os
from pathlib import Path
import re

from .durability import durable_makedirs
from .paths import resolve_under


__all__ = [
    "DAY_FOLDER_PATTERN",
    "day_folder",
    "day_folder_name",
    "unique_path",
]


DAY_FOLDER_PATTERN = re.compile(r"^\d{4}_\d{2}_\d{2}$")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def day_folder_name(when: _date) -> str:
    """The folder name for a calendar day, e.g. ``2026_08_05``."""

    if not isinstance(when, _date):
        raise TypeError("when must be a datetime.date")
    return f"{when.year:04d}_{when.month:02d}_{when.day:02d}"


def day_folder(save_root: str | os.PathLike[str], when: _date) -> Path:
    """Return ``<save_root>/<YYYY_MM_DD>``, creating it durably.

    The caller owns ``save_root``; this creates only the day folder beneath it,
    and refuses a root that does not exist so a typo cannot silently scatter
    data into a new tree.
    """

    root = Path(save_root)
    if not root.is_dir():
        raise NotADirectoryError(f"save root does not exist: {root}")
    return durable_makedirs(resolve_under(root, day_folder_name(when)))


def unique_path(folder: str | os.PathLike[str], stem: str, suffix: str) -> Path:
    """A path in ``folder`` that no file occupies yet.

    Returns ``<stem><suffix>`` when free, otherwise ``<stem>-2<suffix>``,
    ``-3`` and so on.  Saving twice in one day must never overwrite the morning's
    data, and asking the caller to invent unique names invites exactly that.
    """

    directory = Path(folder)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    safe = _UNSAFE.sub("-", str(stem)).strip("-") or "untitled"
    if not suffix.startswith("."):
        raise ValueError("suffix must start with a dot")

    candidate = resolve_under(directory, f"{safe}{suffix}")
    if not candidate.exists():
        return candidate
    ordinal = 2
    while True:
        candidate = resolve_under(directory, f"{safe}-{ordinal}{suffix}")
        if not candidate.exists():
            return candidate
        ordinal += 1
