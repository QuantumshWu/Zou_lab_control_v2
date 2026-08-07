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
    ) -> None:
        if layout_id is None:
            layout_id = build_fingerprint(geom or StreamerParams())
        self.words: dict[int, int] = {CtrlWords.LAYOUT_ID: int(layout_id) & 0xFFFFFFFF}
        self.write_batches: list[tuple[tuple[int, int], ...]] = []
        self.read_log: list[int] = []
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
    ) -> None:
        del deadline
        if stop is not None and stop.is_set():
            raise RuntimeError("memory transport write cancelled")
        batch = tuple((int(address), int(value) & 0xFFFFFFFF) for address, value in rows)
        with self._lock:
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
                        self.status = STATUS_DONE if self.auto_done else STATUS_RUNNING
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
            self.read_log.append(int(word_offset))
            if int(word_offset) == CtrlWords.STATUS:
                return int(self.status) & 0xFFFFFFFF
            if int(word_offset) == CtrlWords.CURSOR:
                return int(self.cursor_value) & 0xFFFFFFFF
            return int(self.words.get(int(word_offset), 0)) & 0xFFFFFFFF

    def record_diagnostic(self, name: str, text: str) -> None:
        del name, text


__all__ = ["MemoryRegisterTransport"]
