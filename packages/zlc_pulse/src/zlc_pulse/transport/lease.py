"""A process-local lease abstraction with a file-shaped public API."""

from __future__ import annotations

from pathlib import Path
import threading


_OWNERS: dict[str, object] = {}
_LOCK = threading.RLock()


class DeviceLease:
    def acquire(self) -> None: ...

    def release(self) -> None: ...


class InterprocessDeviceLease:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(Path(path or "zlc_pulse.device.lock").resolve())
        self._owner = object()
        self.acquired = False

    def acquire(self) -> None:
        with _LOCK:
            current = _OWNERS.get(self.path)
            if current is not None and current is not self._owner:
                raise RuntimeError(f"device lease is already held: {self.path}")
            _OWNERS[self.path] = self._owner
            self.acquired = True

    def release(self) -> None:
        with _LOCK:
            if _OWNERS.get(self.path) is self._owner:
                del _OWNERS[self.path]
            self.acquired = False

    def __enter__(self) -> "InterprocessDeviceLease":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


__all__ = ["DeviceLease", "InterprocessDeviceLease"]
