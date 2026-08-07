"""3 Mbaud-style CRC-framed UART register transport."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
import threading
import time
from typing import Protocol

from ..wire import CtrlWords
from . import uart_frame as framing
from .base import UART_OBSERVER_INTERVAL, TransportAborted


class UartError(RuntimeError):
    pass


class UartLink(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def exchange(self, request: bytes, *, deadline: float, stop: threading.Event | None = None) -> bytes: ...
    def write_batch(self, requests: Sequence[bytes], *, deadline: float, stop: threading.Event | None = None) -> list[bytes]: ...


class PySerialLink:
    def __init__(self, port: str, baud: int = 3_000_000) -> None:
        if not port:
            raise ValueError("UART port is required")
        self.port = str(port)
        self.baud = int(baud)
        self._serial = None

    def open(self) -> None:
        import serial
        serial_port = serial.Serial(
            self.port,
            self.baud,
            timeout=0.05,
            write_timeout=1.0,
            dsrdtr=False,
            rtscts=False,
            xonxoff=False,
        )
        # Never let opening the pulse-streamer reset or drive another instrument on the bus.
        serial_port.dtr = False
        serial_port.rts = False
        self._serial = serial_port

    def close(self) -> None:
        serial_port, self._serial = self._serial, None
        if serial_port is not None:
            serial_port.close()

    def exchange(self, request: bytes, *, deadline: float, stop: threading.Event | None = None) -> bytes:
        if stop is not None and stop.is_set():
            raise TransportAborted("UART request cancelled")
        serial_port = self._require_open()
        serial_port.reset_input_buffer()
        serial_port.write_timeout = _remaining(deadline, "UART write")
        serial_port.write(request)
        serial_port.flush()
        return self._read_replies(1, deadline=deadline, stop=stop)[0]

    def write_batch(self, requests: Sequence[bytes], *, deadline: float, stop: threading.Event | None = None) -> list[bytes]:
        if stop is not None and stop.is_set():
            raise TransportAborted("UART request cancelled")
        if not requests:
            return []
        serial_port = self._require_open()
        serial_port.reset_input_buffer()
        serial_port.write_timeout = _remaining(deadline, "UART write")
        serial_port.write((b"\xff" * 8).join(requests))
        serial_port.flush()
        return self._read_replies(len(requests), deadline=deadline, stop=stop)

    def _read_replies(self, count: int, *, deadline: float, stop: threading.Event | None) -> list[bytes]:
        serial_port = self._require_open()
        buffer = bytearray()
        replies: list[bytes] = []
        while len(replies) < count and time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                raise TransportAborted("UART read cancelled")
            available = serial_port.in_waiting
            if available:
                buffer.extend(serial_port.read(available))
                while True:
                    frame = _extract_reply(buffer)
                    if frame is None:
                        break
                    replies.append(frame)
                    if len(replies) == count:
                        break
            else:
                time.sleep(min(0.0005, max(0.0, deadline - time.monotonic())))
        if len(replies) != count:
            raise TimeoutError("UART reply timed out")
        return replies

    def _require_open(self):
        if self._serial is None:
            raise UartError("serial link is not open")
        return self._serial


class UartRegisterTransport:
    transport_id = "uart"
    observer_interval = UART_OBSERVER_INTERVAL

    def __init__(
        self,
        *,
        state_dir: str | Path,
        link: UartLink | None = None,
        port: str | None = None,
        baud: int = 3_000_000,
        action_timeout: float = 5.0,
        max_frame_words: int = framing.MAX_FRAME_WORDS,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(action_timeout, bool) or not isinstance(action_timeout, (int, float)) or not math.isfinite(float(action_timeout)) or action_timeout <= 0:
            raise ValueError("action_timeout must be positive and finite")
        self.action_timeout = float(action_timeout)
        self.max_frame_words = max(1, min(int(max_frame_words), framing.MAX_FRAME_WORDS))
        self._link = link or PySerialLink(str(port or ""), baud)
        self._lock = threading.RLock()
        self._sequence = 0
        self._closed = True

    def start(self) -> None:
        with self._lock:
            self._link.open()
            self._closed = False

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._link.close()
            self._closed = True

    def write_words(self, rows: Sequence[tuple[int, int]], *, stop: threading.Event | None = None, deadline: float | None = None) -> None:
        absolute = self._deadline(deadline)
        with self._lock:
            self._require_open()
            pending = tuple((int(address), int(value) & 0xFFFFFFFF) for address, value in rows)
            frames = [
                framing.encode_write(base, values, seq=self._next_sequence())
                for base, values in framing.coalesce_runs(pending, max_words=self.max_frame_words)
            ]
            replies = self._link.write_batch(frames, deadline=absolute, stop=stop)
            if len(replies) != len(frames):
                raise UartError("UART reply count differs from frame count")
            for request, reply in zip(frames, replies):
                sequence, status, words = framing.decode_reply(reply)
                if sequence != request[3]:
                    raise UartError("UART write acknowledgement sequence mismatch")
                if status == framing.ST_CRC_FAIL:
                    raise UartError("UART CRC error status in write acknowledgement")
                if status != framing.ST_OK or words:
                    raise UartError(f"UART write acknowledgement was invalid (status=0x{status:02X})")

    def read_word(self, word_offset: int, *, stop: threading.Event | None = None, deadline: float | None = None) -> int:
        absolute = self._deadline(deadline)
        with self._lock:
            self._require_open()
            request = framing.encode_read(int(word_offset), 1, seq=self._next_sequence())
            reply = self._link.exchange(request, deadline=absolute, stop=stop)
            sequence, status, words = framing.decode_reply(reply)
            if sequence != request[3]:
                raise UartError("UART read reply sequence mismatch")
            if status == framing.ST_CRC_FAIL:
                raise UartError("UART CRC error status in read reply")
            if status != framing.ST_OK or len(words) != 1:
                raise UartError(f"UART read reply was invalid (status=0x{status:02X})")
            return int(words[0]) & 0xFFFFFFFF

    def rewrite_scan_bank(
        self,
        *,
        unarmed_bank_ready: int,
        bank_words: Sequence[tuple[int, int]],
        chunk_word: int,
        chunk_index: int,
        rearmed_bank_ready: int,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> None:
        if chunk_word not in (CtrlWords.BANK0_CHUNK, CtrlWords.BANK1_CHUNK):
            raise ValueError("invalid scan chunk register")
        absolute = self._deadline(deadline)
        self.write_words(((CtrlWords.BANK_READY, unarmed_bank_ready),), stop=stop, deadline=absolute)
        self.write_words(tuple(bank_words), stop=stop, deadline=absolute)
        self.write_words(((chunk_word, chunk_index),), stop=stop, deadline=absolute)
        self.write_words(((CtrlWords.BANK_READY, rearmed_bank_ready),), stop=stop, deadline=absolute)

    def record_diagnostic(self, name: str, text: str) -> None:
        try:
            (self.state_dir / f"{name}.log").write_text(text, encoding="utf-8", errors="replace")
        except OSError:
            pass

    def _deadline(self, value: float | None) -> float:
        result = time.monotonic() + self.action_timeout if value is None else float(value)
        if not math.isfinite(result) or result <= time.monotonic():
            raise TimeoutError("UART transaction deadline expired")
        return result

    def _next_sequence(self) -> int:
        self._sequence = (self._sequence + 1) & 0xFF
        return self._sequence

    def _require_open(self) -> None:
        if self._closed:
            raise UartError("UART transport is closed")


def _extract_reply(buffer: bytearray) -> bytes | None:
    while len(buffer) >= 2 and buffer[:2] != bytes((framing.SYNC0, framing.SYNC1)):
        del buffer[0]
    if len(buffer) < 7:
        return None
    count = int.from_bytes(buffer[5:7], "little")
    length = framing.reply_frame_len(count)
    if len(buffer) < length:
        return None
    frame = bytes(buffer[:length])
    del buffer[:length]
    return frame


def _remaining(deadline: float, action: str) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError(f"{action} exceeded its deadline")
    return value


__all__ = ["PySerialLink", "UartError", "UartLink", "UartRegisterTransport"]
