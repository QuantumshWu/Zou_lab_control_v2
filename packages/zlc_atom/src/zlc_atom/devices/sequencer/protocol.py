"""What this package needs of a pulse board, and nothing more.

Only what this package needs, not everything a board offers: the protocol is a
requirement, and a requirement that lists a method nobody here calls rejects
every twin that has not implemented it yet.

Written out rather than imported: a device adapter depends on no parallel
package, so a camera or a sequencer can be driven with no runtime and no
zlc_pulse present.  The price is that these are copies, and copies drift -- this
one had, to the point where ``SafeReadback`` named three fields the real board
has never had.  Nothing read them, so nothing failed, and it would have gone on
being wrong until the first piece of code that did.

So the copy is checked against the original where both are visible at once, in
zlc_workbench, which is the one place that depends on both.  Changing anything
here without changing zlc_pulse fails there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

#: Status word bits this package needs to read.  Mirrored, like the records
#: below, because a device adapter imports no parallel package.
STATUS_ERROR = 1 << 3
STATUS_UNDERFLOW = 1 << 4


@dataclass(frozen=True)
class DoneReport:
    """One finished shot, as the board reports it."""

    status: int
    cursor: int | None
    underflow: bool
    tail_elapsed: float
    status_reads: tuple[int, int] = ()
    cursor_reads: tuple[int, int] = ()

    @property
    def fault(self) -> str:
        """Why this shot was not a good shot, or "" if it was.

        The same rule the real report states, in the same words: a shot that
        reported an error or underran must not look like a clean one.  The bit
        positions come from the board's status word and are checked against
        zlc_pulse where both are visible.
        """

        reasons = []
        if self.status & STATUS_ERROR:
            reasons.append("the board reported an error")
        if self.status & STATUS_UNDERFLOW or self.underflow:
            reasons.append("the scan bank underran")
        return "; ".join(reasons)


@dataclass(frozen=True)
class SafeReadback:
    """The outputs after being driven safe, read back to prove it.

    Two status reads and the clock-enable words, because "it says zero" is not
    proof on a bus that can be read mid-transition; ``stable`` is the board's
    verdict on whether the two agreed.
    """

    status_reads: tuple[int, int]
    clock_enable_words: tuple[int, ...]
    stable: bool

    @property
    def status(self) -> int:
        return self.status_reads[-1]


@runtime_checkable
class RegisterTransport(Protocol):
    def read(self, address: int) -> int: ...
    def write(self, address: int, value: int) -> None: ...


@runtime_checkable
class PulseStreamer(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def load(self, prog: object, *, source: object | None = None) -> None: ...
    def write_slots(self, values: Sequence[int]) -> None: ...
    def write_scan_table(self, rows: np.ndarray, *, sweeps: int = 1) -> None: ...
    def fire(self, *, forever: bool = False) -> None: ...
    def wait_done(self, timeout: float | None = None) -> DoneReport | None: ...
    def cursor(self) -> int | None: ...
    def safe(self) -> SafeReadback: ...
    def snapshot(self) -> Mapping[str, object]: ...


__all__ = ["DoneReport", "PulseStreamer", "RegisterTransport", "SafeReadback"]
