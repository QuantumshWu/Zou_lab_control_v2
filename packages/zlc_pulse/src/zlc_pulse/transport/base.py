"""Register-level transport protocol shared by device implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
import threading


DEFAULT_OBSERVER_INTERVAL = 0.001
UART_OBSERVER_INTERVAL = 0.001
JTAG_AXI_OBSERVER_INTERVAL = 0.05


class TransportAborted(RuntimeError):
    """A pending register action was cancelled by the owning session."""


class RegisterTransport(Protocol):
    transport_id: str
    observer_interval: float

    def start(self) -> None: ...

    def close(self) -> None: ...

    def write_words(
        self,
        rows: Sequence[tuple[int, int]],
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> None: ...

    def read_word(
        self,
        word_offset: int,
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> int: ...

    def record_diagnostic(self, name: str, text: str) -> None: ...


__all__ = [
    "DEFAULT_OBSERVER_INTERVAL",
    "JTAG_AXI_OBSERVER_INTERVAL",
    "RegisterTransport",
    "TransportAborted",
    "UART_OBSERVER_INTERVAL",
]
