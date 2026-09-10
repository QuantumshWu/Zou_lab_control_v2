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
        self._resident = False
        self._last_command_id = 0
        self._last_command_reply = (0, 0)
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
                    if value:
                        self._complete_command(
                            value, self.words.get(CtrlWords.COMMAND_ID, 0),
                            self.words.get(CtrlWords.RUN_REPEAT_COUNT, 1),
                            self.words.get(CtrlWords.SCAN_REPEAT_COUNT, 1),
                        )
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

    def read_words(self, word_offset: int, count: int, *, stop=None, deadline=None) -> tuple[int, ...]:
        with self._lock:
            return tuple(self.read_word(word_offset + index, stop=stop, deadline=deadline)
                         for index in range(count))

    def command(self, code: int, command_id: int, *, run_repeats: int = 1,
                scan_repeats: int = 1, stop=None, deadline=None) -> tuple[int, int]:
        if stop is not None and stop.is_set():
            raise RuntimeError("memory command cancelled")
        with self._lock:
            if self._record_history:
                self.write_batches.append(((CtrlWords.COMMAND_ID, command_id),
                                           (CtrlWords.COMMAND, code)))
            return self._complete_command(code, command_id, run_repeats, scan_repeats)

    def _complete_command(self, code: int, command_id: int,
                          run_repeats: int, scan_repeats: int) -> tuple[int, int]:
        if command_id and command_id == self._last_command_id:
            return self._last_command_reply
        self.words[CtrlWords.COMMAND_ID] = command_id
        if code == CMD_RESET:
            self._resident = False
            self.status = 0
            self.cursor_value = 0
        elif code == CMD_SAFE:
            self.status = 0
            self.cursor_value = 0
        elif code == CMD_LOAD:
            self._resident = True
            self.status = STATUS_LOADED
        elif code == CMD_FIRE and self._resident:
            self.words[CtrlWords.RUN_REPEAT_COUNT] = run_repeats
            self.words[CtrlWords.SCAN_REPEAT_COUNT] = scan_repeats
            self.cursor_value = 0
            self.status = STATUS_RUNNING
        else:
            self.status = STATUS_ERROR
        reply = (self.status, self.cursor_value)
        self._last_command_id = command_id
        self._last_command_reply = reply
        self.words[CtrlWords.ACK_ID] = command_id
        self.words[CtrlWords.ACK_STATUS], self.words[CtrlWords.ACK_CURSOR] = reply
        if code == CMD_FIRE and self.status == STATUS_RUNNING and self.auto_done:
            if run_repeats and scan_repeats:
                count = self.words.get(CtrlWords.SCAN_COUNT, 0)
                self.cursor_value = ((count * scan_repeats - 1) & 0xFFFFFFFF
                                     if count and self.words.get(CtrlWords.SCAN_ENABLE, 0) else 0)
                self.status = STATUS_DONE
        return reply

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
