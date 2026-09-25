"""Register-level transport protocol shared by device implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
import threading


#: How long the pulse observer sleeps between two STATUS/CURSOR reads while
#: a FIRE plays.  5 ms: DONE is seen at most 5 ms (plus one read) after the
#: board reaches it, a third of the ~15 ms Windows timer tick a timed wait
#: used to cost, and a shot is at most ~200 reads a second (fewer over a
#: UART, whose read adds its round trip) instead of a back-to-back loop in
#: the bench process for the whole of a forever pulse.  A streamed scan's
#: bank refill rides the same read and is still three times sooner than
#: when the tick paced it.
DEFAULT_OBSERVER_INTERVAL = 0.005
UART_OBSERVER_INTERVAL = DEFAULT_OBSERVER_INTERVAL
JTAG_AXI_OBSERVER_INTERVAL = 0.05


class TransportAborted(RuntimeError):
    """A pending register action was cancelled by the owning session."""


class RegisterTransport(Protocol):
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

    def read_words(
        self,
        word_offset: int,
        count: int,
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> tuple[int, ...]: ...

    def command(
        self,
        code: int,
        command_id: int,
        *,
        run_repeats: int = 1,
        scan_repeats: int = 1,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> tuple[int, int]: ...

    def read_word(
        self,
        word_offset: int,
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> int: ...

__all__ = [
    "DEFAULT_OBSERVER_INTERVAL",
    "JTAG_AXI_OBSERVER_INTERVAL",
    "RegisterTransport",
    "TransportAborted",
    "UART_OBSERVER_INTERVAL",
]
