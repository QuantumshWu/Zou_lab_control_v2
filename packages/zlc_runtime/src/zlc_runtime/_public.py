"""Small cross-module protocol kept independent of the public facade."""

from __future__ import annotations

from typing import Any, Protocol


class RunHandleLike(Protocol):
    """Opaque domain-side execution handle viewed by the runtime."""

    def snapshot(self) -> Any: ...

    def cancel(self, reason: str | None = None) -> None: ...

    def result(self) -> Any: ...


__all__ = ("RunHandleLike",)
