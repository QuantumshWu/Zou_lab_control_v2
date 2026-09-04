"""Small register-dictionary transport for offline tests and notebooks."""

from __future__ import annotations

from collections.abc import Sequence
import threading

from ..wire import (
    CMD_FIRE,
    CMD_LOAD,
    CMD_RESET,
    CMD_SAFE,
    CtrlWords,
    REGISTER_LAYOUT_ID,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_LOADED,
    STATUS_RUNNING,
    StreamerParams,
    build_fingerprint,
)
from .base import DEFAULT_OBSERVER_INTERVAL


class MemoryRegisterTransport:
    transport_id = "memory"
    #: Nothing on this transport is ever lost -- it may be slow, and a
    #: timed-out action may still complete later, which is exactly why the
    #: strobe verify-and-retry machinery must NOT run here.
    lossy_line = False
    observer_interval = DEFAULT_OBSERVER_INTERVAL

    def __init__(
        self,
        *,
        layout_id: int | None = None,
        geom: StreamerParams | None = None,
        auto_done: bool = False,
        record_history: bool = True,
    ) -> None:
        if layout_id is None:
            layout_id = build_fingerprint(geom or StreamerParams())
        self.words: dict[int, int] = {CtrlWords.LAYOUT_ID: int(layout_id) & 0xFFFFFFFF}
        # Full list history is the established diagnostic surface.  Product
        # owners that never inspect it can opt out instead of retaining every
        # register transaction for their process lifetime.
        self.write_batches: list[tuple[tuple[int, int], ...]] = []
        self.read_log: list[int] = []
        self._record_history = bool(record_history)
        self.status = 0
        self.cursor_value = 0
        self.auto_done = bool(auto_done)
        self.started = False
        self.closed = False
        #: The command bits currently held high.  The board detects a command
        #: on a rising edge and never clears this register itself.
        self._command_seen = 0
        #: How many command writes this twin ignored because the bits were
        #: already high.  A real board ignores them the same way, silently.
        self.dropped_commands = 0
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            self.started = True
            self.closed = False

    def close(self) -> None:
        with self._lock:
            self.closed = True
            self.started = False

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
    ) -> None:
        del deadline
        if stop is not None and stop.is_set():
            raise RuntimeError("memory transport write cancelled")
        batch = tuple((int(address), int(value) & 0xFFFFFFFF) for address, value in rows)
        with self._lock:
            if self._record_history:
                self.write_batches.append(batch)
            for address, value in batch:
                if address == CtrlWords.STATUS:
                    self.status = value
                elif address == CtrlWords.CURSOR:
                    self.cursor_value = value
                elif address == CtrlWords.COMMAND:
                    # The board detects commands on a RISING edge and never
                    # clears the register itself, so writing the same code
                    # twice runs it once.  This modelled the register as
                    # level-sensitive, so a host that forgot to zero between
                    # commands passed here and dropped the second command on
                    # hardware -- the one failure this twin exists to catch.
                    written = value
                    risen = written & ~self._command_seen
                    self._command_seen = written
                    if written and not risen:
                        self.dropped_commands += 1
                    value = risen
                    if value & CMD_RESET:
                        self.status = 0
                    if value & CMD_SAFE:
                        self.status = 0
                        self.words[CtrlWords.CLK_ENABLE] = 0
                    if value & CMD_LOAD:
                        self.status = STATUS_LOADED
                    if value & CMD_FIRE:
                        infinite = (
                            self.words.get(CtrlWords.RUN_REPEAT_COUNT, 1) == 0
                            or self.words.get(CtrlWords.SCAN_REPEAT_COUNT, 1) == 0
                        )
                        if infinite or not self.auto_done:
                            self.cursor_value = 0
                            self.status = STATUS_RUNNING
                        else:
                            # The instant-completion twin publishes the same
                            # terminal row-visit ordinal as RTL, not an initial
                            # cursor that contradicts its DONE status.
                            scan_count = self.words.get(CtrlWords.SCAN_COUNT, 0)
                            scan_enabled = bool(
                                self.words.get(CtrlWords.SCAN_ENABLE, 0)
                            )
                            scan_repeats = self.words.get(
                                CtrlWords.SCAN_REPEAT_COUNT, 1
                            )
                            self.cursor_value = (
                                (scan_count * scan_repeats - 1) & 0xFFFFFFFF
                                if scan_enabled and scan_count
                                else 0
                            )
                            self.status = STATUS_DONE
                    value = written
                self.words[address] = value

    def read_word(
        self,
        word_offset: int,
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> int:
        del deadline
        if stop is not None and stop.is_set():
            raise RuntimeError("memory transport read cancelled")
        with self._lock:
            if self._record_history:
                self.read_log.append(int(word_offset))
            if int(word_offset) == CtrlWords.STATUS:
                return int(self.status) & 0xFFFFFFFF
            if int(word_offset) == CtrlWords.CURSOR:
                return int(self.cursor_value) & 0xFFFFFFFF
            return int(self.words.get(int(word_offset), 0)) & 0xFFFFFFFF

    def publish_execution_readback(self, *, status: int, cursor: int) -> bool:
        """Publish one board-owned runtime state transition.

        ``write_words`` is the host side of this in-memory register file.  A
        virtual FPGA still needs a board side: its physical-world worker calls
        this method at the same row-visit and terminal seams at which RTL would
        update STATUS/CURSOR.  Keeping those registers here makes the memory
        transport the sole hardware twin; the device observer continues to
        learn runtime state through ordinary register reads.

        A transition arriving after SAFE is deliberately ignored.  That is
        the important race rule: once the host has driven the twin safe, a
        late virtual-world callback must not resurrect RUNNING or overwrite
        its final cursor.
        """

        status = int(status) & 0xFFFFFFFF
        cursor = int(cursor) & 0xFFFFFFFF
        if status not in (STATUS_RUNNING, STATUS_DONE, STATUS_ERROR):
            raise ValueError(
                "memory execution readback must be RUNNING, DONE, or ERROR"
            )
        with self._lock:
            if not self.status & STATUS_RUNNING:
                return False
            self.cursor_value = cursor
            self.status = status
            return True

__all__ = ["MemoryRegisterTransport"]
