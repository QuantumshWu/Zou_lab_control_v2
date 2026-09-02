"""Where saved work lands: a save root, calendar days, and unique names.

Everything saved on a given day starts under a folder named for that day --
``<save_root>/2026_08_05/``.  An ordinary Save can allocate a unique file there;
a long-running Task can allocate a unique directory there and own everything
inside it.  A physicist still finds the work by date, while one Task run remains
one copyable and failure-retaining unit.

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
from typing import Callable, Iterator

from .durability import _atomic_write_unique_path, durable_makedirs, flush_directory
from .paths import resolve_under


__all__ = [
    "DAY_FOLDER_PATTERN",
    "day_folder",
    "day_folder_path",
    "day_folder_name",
    "unique_path",
]


DAY_FOLDER_PATTERN = re.compile(r"^\d{4}_\d{2}_\d{2}$")

_UNSAFE = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*\s]+')
_SUFFIX = re.compile(r"\.[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_WINDOWS_RESERVED = re.compile(
    r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z",
    re.IGNORECASE,
)


def day_folder_name(when: _date) -> str:
    """The folder name for a calendar day, e.g. ``2026_08_05``."""

    if not isinstance(when, _date):
        raise TypeError("when must be a datetime.date")
    return f"{when.year:04d}_{when.month:02d}_{when.day:02d}"


def day_folder_path(save_root: str | os.PathLike[str], when: _date) -> Path:
    """Where ``<save_root>/<YYYY_MM_DD>`` is, whether or not it exists yet.

    Pure: no directory is created, stat'ed or resolved.  A form that shows
    today's folder, a dialog that opens there, a title that names it, ask
    WHERE -- asking used to create the folder and flush its directory entry
    to disk, then merely to resolve it three times over, once per question,
    on the GUI thread, for every state the panel published.  The day name
    is a fixed token, so nothing can escape the root the caller owns.
    """

    root = Path(save_root).expanduser()
    if not root.is_absolute():
        raise ValueError(f"save root must be absolute: {root}")
    return root / day_folder_name(when)


def day_folder(save_root: str | os.PathLike[str], when: _date) -> Path:
    """Return ``<save_root>/<YYYY_MM_DD>``, creating it durably.

    The write-side twin of :func:`day_folder_path`: the same answer, made to
    exist with the durability evidence a run root needs.  A root that does
    not exist is refused so a typo cannot silently scatter data into a new
    tree.
    """

    folder = day_folder_path(save_root, when)
    if not folder.parent.is_dir():
        raise NotADirectoryError(f"save root does not exist: {folder.parent}")
    return durable_makedirs(folder)


def unique_path(
    folder: str | os.PathLike[str],
    stem: str,
    suffix: str,
    *,
    writer: Callable[[Path], object] | None = None,
) -> Path:
    """Create one uniquely named file or directory and return its final path.

    A file writer receives a same-directory temporary path bearing the requested
    suffix.  Only after the writer returns and that file is flushed is its full
    content published, without replacement, at ``<stem><suffix>`` or the first
    free numbered variant.  Concurrent processes therefore cannot select or
    overwrite the same artifact.

    An empty suffix creates a directory by the same numbered-name rule.  It has
    no writer because the successful ``mkdir`` is itself the atomic allocation.
    """

    directory = Path(folder).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    if not isinstance(stem, str):
        raise TypeError("stem must be str")
    if not isinstance(suffix, str):
        raise TypeError("suffix must be str")
    safe = _UNSAFE.sub("-", stem).strip(" .-") or "untitled"
    if _WINDOWS_RESERVED.fullmatch(safe):
        safe = f"_{safe}"

    def candidates() -> Iterator[Path]:
        ordinal = 1
        while True:
            numbered = safe if ordinal == 1 else f"{safe}-{ordinal}"
            yield resolve_under(directory, f"{numbered}{suffix}")
            ordinal += 1

    if not suffix:
        if writer is not None:
            raise TypeError("a directory allocation does not accept a writer")
        for candidate in candidates():
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            flush_directory(candidate)
            flush_directory(directory)
            return candidate

    if _SUFFIX.fullmatch(suffix) is None:
        raise ValueError("suffix must be one dotted file extension, not a path")
    if writer is None:
        raise TypeError("a file allocation requires writer")
    return _atomic_write_unique_path(
        candidates(),
        writer,
        temporary_prefix=f".{safe}.",
        temporary_suffix=suffix,
    )
