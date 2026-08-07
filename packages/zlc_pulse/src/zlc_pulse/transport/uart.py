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
        #: What the last short read looked like, for whoever decides it is fatal.
        self.last_shortfall = ""

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
        self.last_shortfall = ""
        replies = self._read_replies(1, deadline=deadline, stop=stop)
        if not replies:
            # _read_replies reports rather than judges, so the judging is here:
            # one request, no reply, and the caller may only retry if the
            # request is idempotent -- which is its decision, not the line's.
            raise TimeoutError(
                f"UART reply timed out on {self.port} at {self.baud} baud: "
                f"{self.last_shortfall or 'no reply'}"
            )
        return replies[0]

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
        self.last_shortfall = ""
        return self._read_replies(len(requests), deadline=deadline, stop=stop)

    def _read_replies(self, count: int, *, deadline: float, stop: threading.Event | None) -> list[bytes]:
        serial_port = self._require_open()
        buffer = bytearray()
        replies: list[bytes] = []
        started = time.monotonic()
        read_bytes = 0
        while len(replies) < count and time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                raise TransportAborted("UART read cancelled")
            available = serial_port.in_waiting
            if available:
                chunk = serial_port.read(available)
                read_bytes += len(chunk)
                buffer.extend(chunk)
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
            # Carried, not raised.  Whether a short answer is fatal depends on
            # whether the frames that went unanswered may be sent again, and
            # only the transport above knows that.  What it needs to say so is
            # WHAT was in flight -- nothing answered, some answered, or bytes
            # that never formed a frame are three faults with three causes.
            self.last_shortfall = (
                f"{len(replies)} of {count} replies in "
                f"{time.monotonic() - started:.2f}s ({read_bytes} byte(s) read, "
                f"{len(buffer)} unparsed: {bytes(buffer[:16]).hex(' ') or 'none'})"
            )
        return replies

    def _require_open(self):
        if self._serial is None:
            raise UartError("serial link is not open")
        return self._serial


class UartRegisterTransport:
    transport_id = "uart"
    observer_interval = UART_OBSERVER_INTERVAL
    #: A frame on this line either executes within microseconds of arriving or
    #: is gone forever -- the property that makes verify-and-retry sound.
    lossy_line = True

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
        #: What one request/reply round trip may cost beyond the bytes: a USB
        #: serial adapter's latency timer is 16 ms by default and the board
        #: answers within microseconds, so this is the host's cost, not the
        #: board's, and it is per FRAME because every frame is acknowledged.
        self.round_trip_allowance = 0.05
        #: How many times a frame may be sent before the link is called broken.
        #: More than one because losing a frame is normal on a line with no
        #: flow control; few, because a link that needs many is not working.
        self.resend_attempts = 3
        #: Host-side slack per retry attempt, beyond the bytes' own wire time:
        #: the USB adapter's latency timer (~16 ms) plus scheduler jitter.
        #: Waiting too LITTLE here is benign -- a write is idempotent, and a
        #: reply that was merely late arrives as a duplicate the classifier
        #: drops by SEQ -- while waiting too much is a stall the operator
        #: feels on every lost frame.  It started at half a second and a lossy
        #: cycle cost visible over-a-second hangs; the physics needs ~20 ms.
        self.retry_slack = 0.08
        #: How many frames have had to be sent again, so a link that is quietly
        #: degrading can be seen before it fails.
        self.resends = 0
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

    def write_words(
        self,
        rows: Sequence[tuple[int, int]],
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
        resend: bool = True,
    ) -> None:
        """Write register words, resending any frame the board did not answer.

        A serial line with no flow control loses a frame now and then, and the
        board cannot ask for one back: its bridge is a single-frame state
        machine, and one mis-sampled stop bit makes it abandon the frame it was
        reading and go back to hunting for a sync pair.  That frame is never
        acknowledged, and nothing downstream ever hears about it.

        Which frame is not a mystery: every frame carries a SEQ and every
        acknowledgement carries it back.  That field existed and was used only
        to assert equality, so one lost frame in a load of ten failed the whole
        load -- on one machine and not another, because what differs is the
        USB-serial adapter's own clock at 3 Mbaud, not the board.

        ``resend=False`` for frames that must not be repeated: a command strobe
        that WAS executed and whose acknowledgement was lost would be executed
        twice.  A register write is an absolute value and repeating it is the
        same write.
        """

        with self._lock:
            self._require_open()
            pending = tuple((int(address), int(value) & 0xFFFFFFFF) for address, value in rows)
            frames = [
                framing.encode_write(base, values, seq=self._next_sequence())
                for base, values in framing.coalesce_runs(pending, max_words=self.max_frame_words)
            ]
            # Budgeted AFTER the frames exist, because how long this may take is
            # a fact about how much there is to send.
            absolute = self._deadline(deadline, frames=len(frames), words=len(pending))
            attempts = self.resend_attempts if resend else 1
            # In groups that cannot exhaust the 8-bit SEQ space: replies are
            # matched by SEQ, and 256 frames in flight would give two of them
            # the same one -- an acknowledgement that answers both answers
            # neither.
            for start in range(0, len(frames), 255):
                self._deliver(frames[start:start + 255], attempts, absolute, stop)

    def _deliver(
        self,
        frames: list[bytes],
        attempts: int,
        absolute: float,
        stop: threading.Event | None,
    ) -> None:
        """Send one SEQ-distinct group until every frame is acknowledged.

        Every attempt but the last waits only as long as these bytes need plus
        host slack -- NOT the whole deadline.  One shared deadline was what
        made the first resend implementation theatre: a genuinely lost frame
        kept attempt one waiting until the deadline itself, so the retry began
        with no time left and died in the write path without retransmitting.
        The last attempt inherits whatever remains, which preserves the old
        patience for a link that is merely slow.

        A frame the board REJECTED (ST_CRC_FAIL: the request arrived damaged)
        is provably unexecuted, so retrying it is safe even when ``resend`` was
        refused -- that refusal exists for the ambiguous case, a command whose
        acknowledgement went missing after it may have run.
        """

        outstanding = list(frames)
        attempt = 0
        limit = attempts
        rejected: set[int] = set()
        while attempt < limit:
            attempt += 1
            final = attempt >= limit
            now = time.monotonic()
            attempt_deadline = (
                absolute
                if final
                else min(absolute, now + self._attempt_budget(outstanding))
            )
            if attempt_deadline <= now:
                break
            try:
                replies = self._link.write_batch(
                    outstanding, deadline=attempt_deadline, stop=stop
                )
            except TimeoutError:
                if final:
                    raise
                replies = []
            answered, rejected = self._classify(outstanding, replies)
            outstanding = [
                frame for frame in outstanding if frame[3] not in answered
            ]
            if not outstanding:
                return
            if all(frame[3] in rejected for frame in outstanding):
                # Everything still missing was explicitly refused, none of it
                # ran: even a strobe may go again.
                limit = max(limit, self.resend_attempts)
            if attempt < limit:
                # Counted when they are actually SENT again, not when they are
                # found missing: a link that gives up is not a link that
                # retried.
                self.resends += len(outstanding)
        if outstanding and all(frame[3] in rejected for frame in outstanding):
            # The board answered every time and refused every time: our
            # requests keep arriving damaged.  That is a CRC verdict about the
            # host-to-board direction, and naming it is what separates a bad
            # cable from a dead one.
            raise UartError(
                f"UART write rejected as damaged (request CRC) on {self.port} at "
                f"{self.baud} baud after {attempt} attempt(s): "
                f"{len(outstanding)} of {len(frames)} frame(s)"
            )
        raise TimeoutError(
            f"UART reply timed out on {self.port} at {self.baud} baud after "
            f"{attempt} attempt(s): {len(outstanding)} of {len(frames)} frame(s) "
            f"unanswered ({getattr(self._link, 'last_shortfall', '') or 'no detail'})"
        )

    def _attempt_budget(self, frames: "Sequence[bytes]") -> float:
        """How long one attempt may reasonably take for THESE frames.

        The bytes' own time on the wire, both directions, plus flat host
        slack.  Small on purpose: this is what a retry costs, and three of
        them are still well under one old five-second wait.
        """

        payload = sum(len(frame) + 8 for frame in frames)
        payload += framing.reply_frame_len(0) * len(frames)
        return payload * 10.0 / max(1.0, float(self.baud)) + self.retry_slack

    def _classify(
        self,
        frames: Sequence[bytes],
        replies: Sequence[bytes],
    ) -> tuple[set[int], set[int]]:
        """Sort acknowledgements into answered and refused, dropping noise.

        Three verdicts per reply.  Clean ST_OK: answered.  ST_CRC_FAIL: the
        REQUEST arrived damaged and committed nothing -- refused, and provably
        safe to send again.  A reply that itself arrived damaged, or answers a
        SEQ nobody in this group sent (a late duplicate of an earlier
        exchange): noise on a noisy line, dropped, leaving its frame
        unanswered so the caller resends it.  Only a well-formed reply with a
        WRONG answer -- bad opcode, address range, unexpected payload -- still
        raises, because that is a protocol bug and no amount of retrying fixes
        a disagreement about the protocol.
        """

        expected = {frame[3] for frame in frames}
        answered: set[int] = set()
        rejected: set[int] = set()
        for reply in replies:
            try:
                sequence, status, words = framing.decode_reply(reply)
            except framing.FrameError:
                continue
            if sequence not in expected:
                continue
            if status == framing.ST_CRC_FAIL:
                rejected.add(sequence)
                continue
            if status != framing.ST_OK or words:
                raise UartError(
                    f"UART write acknowledgement was invalid (status=0x{status:02X})"
                )
            answered.add(sequence)
        return answered, rejected

    def read_word(self, word_offset: int, *, stop: threading.Event | None = None, deadline: float | None = None) -> int:
        absolute = self._deadline(deadline)
        with self._lock:
            self._require_open()
            request = framing.encode_read(int(word_offset), 1, seq=self._next_sequence())
            return self._read_with_retry(request, absolute, stop)

    def _read_with_retry(
        self,
        request: bytes,
        absolute: float,
        stop: threading.Event | None,
    ) -> int:
        """Ask once, and ask again if the answer is missing, stale or damaged.

        A read is idempotent, so the SAME frame goes again -- if the first
        answer was merely late, its duplicate says the same thing.  This is
        what kept Stop from stalling: the safe path reads back status and
        clock words, and every one of those reads used to wait out the whole
        deadline over a single lost byte.
        """

        last = ""
        for attempt in range(self.resend_attempts):
            final = attempt + 1 == self.resend_attempts
            now = time.monotonic()
            attempt_deadline = (
                absolute
                if final
                else min(absolute, now + self._attempt_budget((request,)))
            )
            if attempt_deadline <= now:
                attempt_deadline, final = absolute, True
            try:
                reply = self._link.exchange(request, deadline=attempt_deadline, stop=stop)
            except TimeoutError as error:
                last = str(error)
                if final:
                    raise
                self.resends += 1
                continue
            sequence, status, words = framing.decode_reply(reply)
            if sequence != request[3] or status == framing.ST_CRC_FAIL:
                # A stale answer to an earlier question, or our question
                # arrived damaged.  Either way, ask again -- and if the board
                # keeps refusing, say CRC out loud: that verdict is what
                # separates a corrupting cable from a dead one.
                last = (
                    "the board rejected the request as damaged (CRC error)"
                    if status == framing.ST_CRC_FAIL
                    else f"stale reply seq={sequence}, expected {request[3]}"
                )
                if final:
                    raise UartError(f"UART read failed after retries: {last}")
                self.resends += 1
                continue
            if status != framing.ST_OK or len(words) != 1:
                raise UartError(f"UART read reply was invalid (status=0x{status:02X})")
            return int(words[0]) & 0xFFFFFFFF
        raise UartError(f"UART read failed ({last or 'no attempt ran'})")

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

    def _deadline(self, value: float | None, *, frames: int = 1, words: int = 1) -> float:
        """When this transaction has waited long enough.

        Scaled by what is being sent.  ``action_timeout`` was one constant for
        every transaction -- a single register read and a whole register image
        alike -- so it was either generous enough to make a dead link take five
        seconds to say so, or tight enough that a slow host, a slow USB latency
        timer or a longer program turned a working link into a timeout.  Both
        halves of that were reported the same way.

        The budget is the time the bytes physically take at this baud, plus one
        round trip per frame, plus ``action_timeout`` of slack for the host and
        its driver.  Small transactions therefore still fail fast.
        """

        if value is not None:
            result = float(value)
        else:
            bytes_out = max(1, int(words)) * framing.BYTES_PER_WORD_ESTIMATE
            on_the_wire = bytes_out * 10.0 / max(1.0, float(self.baud))
            result = (
                time.monotonic()
                + self.action_timeout
                + on_the_wire
                + max(1, int(frames)) * self.round_trip_allowance
            )
        if not math.isfinite(result) or result <= time.monotonic():
            raise TimeoutError("UART transaction deadline expired")
        return result

    @property
    def port(self) -> str:
        """Which port this transport is on, for anything that reports a fault."""

        return str(getattr(self._link, "port", "?"))

    @property
    def baud(self) -> int:
        """The link's baud, for anything that budgets time by how long bytes take."""

        return int(getattr(self._link, "baud", 3_000_000))

    def _next_sequence(self) -> int:
        self._sequence = (self._sequence + 1) & 0xFF
        return self._sequence

    def _require_open(self) -> None:
        if self._closed:
            raise UartError("UART transport is closed")


def _extract_reply(buffer: bytearray) -> bytes | None:
    """The next VERIFIED reply in the buffer, resynchronising past damage.

    The CRC is checked here, not later, because a damaged frame does not only
    lie about its contents -- a corrupted COUNT slices the wrong number of
    bytes and misaligns every frame behind it.  On any damage the stream is
    re-hunted from one byte later, which is what the board's own bridge does
    with damage in the other direction.
    """

    sync = bytes((framing.SYNC0, framing.SYNC1))
    while True:
        while len(buffer) >= 2 and buffer[:2] != sync:
            del buffer[0]
        if len(buffer) < 7:
            return None
        count = int.from_bytes(buffer[5:7], "little")
        if count > framing.MAX_FRAME_WORDS:
            # An impossible length: this sync pair was noise, or the frame is
            # damaged beyond trusting anything it says.
            del buffer[0]
            continue
        length = framing.reply_frame_len(count)
        if len(buffer) < length:
            return None
        frame = bytes(buffer[:length])
        try:
            framing.decode_reply(frame)
        except framing.FrameError:
            del buffer[0]
            continue
        del buffer[:length]
        return frame


def _remaining(deadline: float, action: str) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError(f"{action} exceeded its deadline")
    return value


__all__ = ["PySerialLink", "UartError", "UartLink", "UartRegisterTransport"]
