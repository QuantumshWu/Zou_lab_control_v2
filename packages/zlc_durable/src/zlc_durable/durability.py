"""Filesystem directory durability primitives shared by storage owners."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import BinaryIO, Callable, Iterable


class DirectoryDurabilityError(RuntimeError):
    """The filesystem could not durably flush a directory entry.

    ``published`` names the artifact that is already complete and visible
    when the flush that would have made its directory entry crash-durable
    is what failed: the file replaced or linked, the directory made.  It
    stays on disk.  A save window that shows the error can then say where
    the work landed instead of "cannot save", and a caller that retries
    knows the name is taken by its own complete work rather than allocating
    a numbered copy beside it.  ``None`` when nothing was published by the
    call that failed.
    """

    def __init__(self, message: str, *, published: Path | None = None) -> None:
        super().__init__(message)
        self.published = published


def _flush_windows_directory(directory: Path) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    flush_file_buffers = kernel32.FlushFileBuffers
    flush_file_buffers.argtypes = (wintypes.HANDLE,)
    flush_file_buffers.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        str(directory),
        0x40000000,  # GENERIC_WRITE, required by FlushFileBuffers.
        0x00000001 | 0x00000002 | 0x00000004,  # share read/write/delete
        None,
        3,  # OPEN_EXISTING
        0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS opens a directory handle.
        None,
    )
    invalid = wintypes.HANDLE(-1).value
    if handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    flush_error: OSError | None = None
    try:
        if not flush_file_buffers(handle):
            flush_error = ctypes.WinError(ctypes.get_last_error())
    finally:
        if not close_handle(handle) and flush_error is None:
            flush_error = ctypes.WinError(ctypes.get_last_error())
    if flush_error is not None:
        raise flush_error


def _flush_posix_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def flush_directory(directory: str | os.PathLike[str]) -> None:
    """Durably flush one existing directory or fail explicitly.

    Windows uses a real directory handle plus ``FlushFileBuffers``; POSIX uses
    ``fsync`` on an open directory descriptor.  There is deliberately no
    best-effort/no-op backend because callers use this as commit evidence.
    """

    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    try:
        if os.name == "nt":
            _flush_windows_directory(path)
        else:
            _flush_posix_directory(path)
    except DirectoryDurabilityError:
        raise
    except OSError as exc:
        raise DirectoryDurabilityError(
            f"directory durability flush failed for {path}"
        ) from exc


def _flush_published(directory: Path, published: Path) -> None:
    """Acknowledge the directory entry of an artifact that is already visible.

    Publication -- the replace, the link, the mkdir -- is the irreversible
    step; the flush after it is only the acknowledgement.  When just the
    acknowledgement fails, the error says so and names the artifact.  An
    error that named only the directory read as "nothing was saved": the
    operator retried, and a second numbered copy landed beside a file that
    had been complete on disk all along.
    """

    try:
        flush_directory(directory)
    except DirectoryDurabilityError as error:
        raise DirectoryDurabilityError(
            f"{published} is published and visible, but its directory entry "
            f"could not be made crash-durable: {error}",
            published=published,
        ) from error


def atomic_write_file(
    target: str | os.PathLike[str],
    writer: Callable[[BinaryIO], None],
) -> Path:
    """Write and durably replace one file through a same-directory temporary."""

    if not callable(writer):
        raise TypeError("writer must be callable")
    destination = Path(target).expanduser().resolve()
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"target parent directory does not exist: {parent}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _flush_published(parent, destination)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def _atomic_write_unique_path(
    candidates: Iterable[Path],
    writer: Callable[[Path], object],
    *,
    temporary_prefix: str,
    temporary_suffix: str,
) -> Path:
    """Write once, then atomically publish at the first unoccupied candidate."""

    if not callable(writer):
        raise TypeError("writer must be callable")
    choices = iter(candidates)
    try:
        first = next(choices)
    except StopIteration as exc:
        raise ValueError("candidates must not be empty") from exc
    parent = first.parent
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=temporary_prefix,
        suffix=temporary_suffix,
        dir=parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        destination = first
        while True:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                destination = next(choices)
                continue
            temporary.unlink()
            _flush_published(parent, destination)
            return destination
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_bytes(
    target: str | os.PathLike[str],
    payload: bytes | bytearray | memoryview,
) -> Path:
    """Atomically replace one file with bytes."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    return atomic_write_file(target, lambda stream: stream.write(payload))


def atomic_write_text(
    target: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically replace one file with encoded text."""

    if not isinstance(text, str):
        raise TypeError("text must be str")
    return atomic_write_bytes(target, text.encode(encoding))


def durable_mkdir(directory: str | os.PathLike[str]) -> Path:
    """Create and flush a missing directory; an existing directory is a no-op."""

    target = Path(directory).expanduser().resolve()
    parent = target.parent
    if target.exists():
        if not target.is_dir():
            raise NotADirectoryError(target)
        return target
    if not parent.exists():
        raise FileNotFoundError(
            f"parent directory does not exist for durable mkdir: {parent}"
        )
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    try:
        target.mkdir()
    except FileExistsError:
        if not target.is_dir():
            raise NotADirectoryError(target)
        return target
    _flush_published(target, target)
    _flush_published(parent, target)
    return target


def durable_makedirs(directory: str | os.PathLike[str]) -> Path:
    """Create only missing levels, flushing each new child and its parent.

    Naming an existing hierarchy changes no directory entry.  A failed
    creation flush reports that its directory is already visible through
    DirectoryDurabilityError.published; it is not retried on unrelated saves.
    """

    target = Path(directory).expanduser().resolve()
    missing: list[Path] = []
    anchor = target
    while not anchor.exists():
        missing.append(anchor)
        parent = anchor.parent
        if parent == anchor:
            break
        anchor = parent
    if not anchor.is_dir():
        raise NotADirectoryError(anchor)
    for level in reversed(missing):
        durable_mkdir(level)
    return target


__all__ = [
    "DirectoryDurabilityError",
    "atomic_write_bytes",
    "atomic_write_file",
    "atomic_write_text",
    "durable_makedirs",
    "durable_mkdir",
    "flush_directory",
]
