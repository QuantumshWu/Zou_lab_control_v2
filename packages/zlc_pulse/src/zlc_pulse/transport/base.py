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
    #: Whether this line can LOSE a request or its acknowledgement outright.
    #: True only for the UART: a frame either executes within microseconds of
    #: arriving or is gone forever, so a timeout implies nothing is still in
    #: flight -- which is the precondition for verify-and-retry on a command
    #: strobe.  A Vivado TCL that timed out may still execute later, so on
    #: that transport the same retry would risk firing twice.
    lossy_line: bool

    def start(self) -> None: ...

    def close(self) -> None: ...

    def write_words(
        self,
        rows: Sequence[tuple[int, int]],
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
        #: Whether a frame the link never answered may be sent again.  False
        #: for a command strobe: one that WAS executed and whose acknowledgement
        #: was lost would be executed twice.  A transport with no frames to lose
        #: takes the argument and ignores it, because the caller's meaning is
        #: the same either way and a caller should not have to ask which
        #: transport it has.
        resend: bool = True,
    ) -> None: ...

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
