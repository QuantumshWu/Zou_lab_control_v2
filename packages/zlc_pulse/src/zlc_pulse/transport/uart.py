"""3 Mbaud-style CRC-framed UART register transport."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import threading
import time
from typing import Protocol

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
        if not isinstance(port, str) or not port.strip():
            raise ValueError("UART port is required")
        if isinstance(baud, bool) or not isinstance(baud, int) or baud <= 0:
            raise ValueError("UART baud must be a positive integer")
        self.port = port.strip()
        self.baud = baud
        self._serial = None
        #: What the last short read looked like, for whoever decides it is fatal.
        self.last_shortfall = ""

    def open(self) -> None:
        import serial
        if self._serial is not None:
            return
        serial_port = None
        try:
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
        except BaseException:
            if serial_port is not None:
                serial_port.close()
            raise
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
        # Cleared BEFORE the write: a write that stalls raises past the read,
        # and whoever reports this attempt must not find the previous one's
        # shortfall still lying here.
        self.last_shortfall = ""
        self._write(serial_port, request)
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
        self.last_shortfall = ""
        self._write(serial_port, (b"\xff" * 8).join(requests))
        return self._read_replies(len(requests), deadline=deadline, stop=stop)

    def _write(self, serial_port, payload: bytes) -> None:
        """Write and drain, speaking THIS layer's timeout vocabulary.

        pyserial reports a write that missed its timeout as
        ``SerialTimeoutException`` -- an OSError, not a TimeoutError -- so
        every ``except TimeoutError`` above this line (the retry loop in
        ``_deliver``, the reconnect guard in the device) looked straight
        past it: one slow WriteFile was a hard failure on the first
        attempt, with pyserial's own words and no resend, on a line whose
        whole retry machinery exists for exactly that moment.  The
        translation lives here because the link is where pyserial's
        vocabulary stops being spoken.
        """

        import serial

        try:
            serial_port.write(payload)
            serial_port.flush()
        except serial.SerialTimeoutException as error:
            raise TimeoutError(
                f"UART write timed out on {self.port} at {self.baud} baud: "
                f"{len(payload)} byte(s) were not accepted in time"
            ) from error

    def _read_replies(self, count: int, *, deadline: float, stop: threading.Event | None) -> list[bytes]:
        """Collect replies until there are ``count`` or the deadline passes.

        The deadline is the ONLY judgement of a reply that has not arrived,
        and it is the attempt budget the transport above derives from the
        bytes in flight (``_attempt_budget``): a complete reply crosses the
        wire in tens of microseconds and the adapter hands it over within
        milliseconds, so a reply still incomplete when that budget ends --
        12 of 13 bytes, every field valid -- is a lost byte, not a slow one.
        There is deliberately no second, earlier judgement of a frame that
        began and stopped: the transport charges every attempt its budget
        before the next one goes out, so ending an attempt sooner would save
        nothing, and a stray byte that reached the buffer before a merely
        late reply would look like a frame that stopped, cutting the window
        for the real reply short.  What was in the buffer when the budget
        ended is reported (``_describe_shortfall``); it is not judged twice.
        """

        serial_port = self._require_open()
        buffer = bytearray()
        replies: list[bytes] = []
        read_bytes = 0
        while len(replies) < count:
            now = time.monotonic()
            if now >= deadline:
                break
            if stop is not None and stop.is_set():
                raise TransportAborted("UART read cancelled")
            available = serial_port.in_waiting
            if available:
                chunk = serial_port.read(available)
                read_bytes += len(chunk)
                buffer.extend(chunk)
                while len(replies) < count:
                    frame = _extract_reply(buffer)
                    if frame is None:
                        break
                    replies.append(frame)
            else:
                time.sleep(min(0.0005, max(0.0, deadline - now)))
        if len(replies) != count:
            # Carried, not raised.  Whether a short answer is fatal depends on
            # whether the frames that went unanswered may be sent again, and
            # only the transport above knows that.
            self.last_shortfall = _describe_shortfall(replies, count, read_bytes, buffer)
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
        link: UartLink | None = None,
        port: str | None = None,
        baud: int = 3_000_000,
        action_timeout: float = 5.0,
        max_frame_words: int = framing.MAX_FRAME_WORDS,
    ) -> None:
        if isinstance(action_timeout, bool) or not isinstance(action_timeout, (int, float)) or not math.isfinite(float(action_timeout)) or action_timeout <= 0:
            raise ValueError("action_timeout must be positive and finite")
        self.action_timeout = float(action_timeout)
        #: What one request/reply round trip may cost beyond the bytes: the
        #: host's USB delivery, not the board's answer (which takes
        #: microseconds).  Sized for the slowest adapter this code has met, an
        #: FTDI with its 16 ms latency timer; the CH340C on the board has no
        #: such timer and hands over what it holds at every millisecond USB
        #: poll, so for it this is generous, which is the benign direction
        #: (a successful attempt returns the moment its replies land).  Per
        #: FRAME because every frame is acknowledged.
        self.round_trip_allowance = 0.05
        #: Host-side slack per attempt, beyond the bytes' own wire time: the
        #: same USB delivery bound plus scheduler jitter.  Waiting too LITTLE
        #: here is benign -- a write is idempotent, and a reply that was
        #: merely late arrives as a duplicate the classifier drops by SEQ --
        #: while waiting too much is a stall the operator feels on every lost
        #: frame.  It started at half a second and a lossy cycle cost visible
        #: over-a-second hangs; the physics needs ~20 ms.
        self.retry_slack = 0.08
        #: How many frames have had to be sent again, so a link that is quietly
        #: degrading can be seen before it fails.
        self.resends = 0
        if (
            isinstance(max_frame_words, bool)
            or not isinstance(max_frame_words, int)
            or not 1 <= max_frame_words <= framing.MAX_FRAME_WORDS
        ):
            raise ValueError("max_frame_words is outside the UART frame range")
        self.max_frame_words = max_frame_words
        self._link = link or PySerialLink(port, baud)
        self._lock = threading.RLock()
        self._sequence = 0
        self._closed = True

    def start(self) -> None:
        with self._lock:
            if self._closed:
                self._link.open()
                self._closed = False

    def close(self) -> None:
        with self._lock:
            try:
                if not self._closed:
                    self._link.close()
            finally:
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
            pending = tuple(rows)
            frames = [
                framing.encode_write(base, values, seq=self._next_sequence())
                for base, values in framing.coalesce_runs(pending, max_words=self.max_frame_words)
            ]
            # Budgeted AFTER the frames exist, because how long this may take is
            # a fact about how much there is to send.
            absolute = self._deadline(deadline, frames=len(frames), words=len(pending))
            # In groups that cannot exhaust the 8-bit SEQ space: replies are
            # matched by SEQ, and 256 frames in flight would give two of them
            # the same one -- an acknowledgement that answers both answers
            # neither.
            for start in range(0, len(frames), 255):
                self._deliver(frames[start:start + 255], absolute, stop, resend=resend)

    def _deliver(
        self,
        frames: list[bytes],
        absolute: float,
        stop: threading.Event | None,
        *,
        resend: bool,
    ) -> None:
        """Send one SEQ-distinct group until every frame is acknowledged.

        A frame the board REJECTED (ST_CRC_FAIL: the request arrived damaged)
        is provably unexecuted, so sending it again is safe even when
        ``resend`` was refused -- that refusal exists for the ambiguous case,
        a command whose acknowledgement went missing after it may have run.
        The board answering "damaged" every time is a CRC verdict about the
        host-to-board direction, and the attempt record says so in those
        words: that is what separates a corrupting cable from a dead one.
        """

        outstanding = list(frames)
        rejected: set[int] = set()

        def attempt(attempt_deadline: float) -> str | None:
            nonlocal outstanding, rejected
            # A verdict belongs to the attempt that earned it.  The refusal
            # that lifts the no-resend ban is "the board answered THIS and
            # rejected it"; an attempt that timed out in the write or got no
            # reply answered nothing, and the previous attempt's refusal must
            # not speak for it -- a strobe refused once and then unanswered
            # would otherwise be sent a third time into exactly the ambiguity
            # the ban exists for.  The read path clears its verdict the same
            # way (``_read_with_retry``).
            rejected = set()
            try:
                replies = self._link.write_batch(
                    outstanding, deadline=attempt_deadline, stop=stop
                )
            except TimeoutError as error:
                return str(error)
            answered, rejected = self._classify(outstanding, replies)
            outstanding = [
                frame for frame in outstanding if frame[3] not in answered
            ]
            if not outstanding:
                return None
            if all(frame[3] in rejected for frame in outstanding):
                return (
                    f"{len(outstanding)} of {len(frames)} frame(s) rejected as "
                    "damaged (request CRC)"
                )
            return (
                f"{len(outstanding)} of {len(frames)} frame(s) unanswered: "
                f"{getattr(self._link, 'last_shortfall', '') or 'no detail'}"
            )

        self._until_answered(
            absolute,
            attempt,
            outstanding=lambda: outstanding,
            resend=resend,
            refused=lambda: all(frame[3] in rejected for frame in outstanding),
            what=f"write of {len(frames)} frame(s)",
            stop=stop,
        )

    def _until_answered(
        self,
        absolute: float,
        attempt: "Callable[[float], str | None]",
        *,
        outstanding: "Callable[[], Sequence[bytes]]",
        resend: bool,
        refused: "Callable[[], bool]",
        what: str,
        stop: threading.Event | None,
    ) -> None:
        """THE retry law -- reads and writes alike run their attempts here.

        The line is lossy (``lossy_line``): a frame either executes within
        microseconds of arriving or is gone forever, and so is its reply.
        Therefore waiting is never the answer to a missing frame -- sending
        again is -- and an attempt is worth exactly what its bytes need plus
        host slack (``_attempt_budget`` of what is still ``outstanding``),
        after which the next attempt goes out.  How many attempts there are
        is not a rule of its own: it is whatever the transaction deadline
        divides into.  The former rule -- three attempts, the last inheriting
        the remaining deadline "to keep patience for a slow link" -- answered
        a LOST byte with the patience owed to a SLOW one, and a read whose
        reply arrived one byte short three times spent 4.88 s waiting for
        that byte instead of asking sixty more times.

        EVERY attempt is charged its budget, however it ended.  A refusal
        (the board answered within a round trip: "that arrived damaged") or
        a reply that decoded as noise ends the attempt early, and asking
        again in the same millisecond is not a retry, it is a storm: a
        board refusing everything was asked 324,338 times inside one
        half-second deadline, ``resends`` -- the operator's one view of a
        quietly degrading line -- said 324,337, and a corrupting cable got
        thousands of chances at a frame in the time a lost one gets sixty.
        The budget is also the length of the burst the next attempt is kept
        out of: corruption on a serial line comes in bursts (a USB hiccup, a
        moment of interference), and sending the same bytes back into one
        earns the same refusal.  So the next attempt waits for the budget to
        elapse -- ``stop``-aware, because that wait is where a cancelled
        transaction now sits -- and the deadline divides into the same
        number of attempts whether the line is silent or refusing.

        ``attempt(deadline)`` returns None when the transaction is complete,
        otherwise how that try failed.  EVERY failure is kept and reported:
        the previous law raised the last attempt's words only, and what the
        first two attempts had seen -- the one fact that tells a run of lost
        bytes from a line that went dead -- was nowhere.

        ``resend=False`` is the command-strobe contract: a strobe whose
        acknowledgement went missing may have run, so it is never sent again
        blindly.  ``refused()`` -- the board answered THIS attempt and
        rejected what is outstanding as damaged (ST_CRC_FAIL), so nothing
        ran -- lifts that ban for the next attempt only, and when the
        deadline ends on refusals the error says CRC, not timeout: that
        verdict about the host-to-board direction is what separates a
        corrupting cable from a dead one.

        Resends are counted when a frame is actually SENT again, not when it
        is found missing: a link that gives up is not a link that retried.
        """

        started = time.monotonic()
        # (first attempt, last attempt, how it failed): runs, so the record
        # of a whole deadline of the same failure is one entry.
        failures: list[tuple[int, int, str]] = []
        attempts = 0
        next_at = started
        while True:
            now = time.monotonic()
            if now >= absolute or (failures and not (resend or refused())):
                break
            if now < next_at:
                pause = min(next_at, absolute) - now
                if stop is None:
                    time.sleep(pause)
                elif stop.wait(pause):
                    raise TransportAborted("UART request cancelled")
                continue
            if failures:
                self.resends += len(outstanding())
            attempts += 1
            next_at = now + self._attempt_budget(outstanding())
            failure = attempt(min(absolute, next_at))
            if failure is None:
                return
            if failures and failures[-1][2] == failure:
                failures[-1] = (failures[-1][0], attempts, failure)
            else:
                failures.append((attempts, attempts, failure))
        record = (
            f"on {self.port} at {self.baud} baud after {attempts} attempt(s) in "
            f"{time.monotonic() - started:.2f}s ({what}): {_attempt_record(failures)}"
        )
        if failures and refused():
            raise UartError(f"UART request rejected as damaged (request CRC) {record}")
        raise TimeoutError(f"UART reply timed out {record}")

    def _attempt_budget(self, frames: "Sequence[bytes]") -> float:
        """How long one attempt may reasonably take for THESE frames.

        SCALED BY WHAT IS BEING SENT, like ``_deadline`` above -- they are
        the same physical claim, and this one was written flat.  A scan
        start streams its point table as sixty-four full frames: 223 ms of
        wire, over which the flat 80 ms of host slack left the first two
        attempts an 8 per cent margin for the USB latency timer, the
        driver's flush and the scheduler together.  A pulse On/Off strobe
        is one frame and twenty microseconds of wire, over which the same
        80 ms was a 4000 per cent margin.  So the same bench saw strobes
        that never failed and scan starts that timed out at random -- the
        operator's exact report.

        The budget is the wire time with a host factor (an attempt whose
        host runs half again slower than the line is late, not lost), the
        flat scheduling slack, and a small per-frame term for reply
        handling (replies arrive batched by the ~16 ms USB latency timer).
        Overshooting here is benign -- a successful attempt returns the
        moment its replies land -- while undershooting burns a resend of
        everything still outstanding.
        """

        payload = sum(len(frame) + 8 for frame in frames)
        payload += framing.reply_frame_len(0) * len(frames)
        on_the_wire = payload * 10.0 / max(1.0, float(self.baud))
        return on_the_wire * 1.5 + self.retry_slack + 0.002 * len(frames)

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
            request = framing.encode_read(word_offset, 1, seq=self._next_sequence())
            return self._read_with_retry(request, absolute, stop)

    def _read_with_retry(
        self,
        request: bytes,
        absolute: float,
        stop: threading.Event | None,
    ) -> int:
        """Ask, and ask again while the answer is missing, stale or damaged.

        A read is idempotent, so the SAME frame goes again -- if the first
        answer was merely late, its duplicate says the same thing.  Under the
        one retry law (``_until_answered``), which is what kept Stop from
        stalling: the safe path reads back status and clock words, and every
        one of those reads used to wait out the whole deadline over a single
        lost byte.
        """

        value: int | None = None
        refused = False

        def attempt(attempt_deadline: float) -> str | None:
            nonlocal value, refused
            refused = False
            try:
                reply = self._link.exchange(request, deadline=attempt_deadline, stop=stop)
            except TimeoutError as error:
                return getattr(self._link, "last_shortfall", "") or str(error)
            sequence, status, words = framing.decode_reply(reply)
            if sequence != request[3]:
                # A stale answer to an earlier question, dropped like the
                # write path's classifier drops it.
                return f"stale reply seq={sequence}, expected {request[3]}"
            if status == framing.ST_CRC_FAIL:
                refused = True
                return "the board rejected the request as damaged (CRC error)"
            if status != framing.ST_OK or len(words) != 1:
                raise UartError(f"UART read reply was invalid (status=0x{status:02X})")
            value = int(words[0]) & 0xFFFFFFFF
            return None

        self._until_answered(
            absolute,
            attempt,
            outstanding=lambda: (request,),
            resend=True,
            refused=lambda: refused,
            what=f"read of word {int.from_bytes(request[4:8], 'little')}",
            stop=stop,
        )
        assert value is not None
        return value

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


def _describe_shortfall(
    replies: Sequence[bytes],
    count: int,
    read_bytes: int,
    buffer: bytearray,
) -> str:
    """What a short read looked like, in words that name the fault.

    Three shapes with three causes, kept apart: nothing arrived at all; a
    frame BEGAN and stopped ("incomplete frame: 12 of 13 bytes" -- every
    field valid, one byte short, the board-to-host direction dropped it);
    bytes arrived that formed no frame (noise, or a frame damaged past
    resynchronisation).  One word, "unparsed", used to cover the last two,
    and a reply missing only its final CRC byte was read as garbage.
    """

    if read_bytes == 0:
        return "no bytes"
    parts = [f"{len(replies)} of {count} replies"]
    # The extractor leaves behind only a sync-led frame that has not
    # finished, or a single stray byte it could not yet judge.
    partial = buffer if len(buffer) >= 2 or buffer[:1] == bytes((framing.SYNC0,)) else b""
    if len(partial) >= 7:
        words = int.from_bytes(partial[5:7], "little")
        parts.append(
            f"incomplete frame: {len(partial)} of "
            f"{framing.reply_frame_len(words)} bytes (count={words})"
        )
    elif partial:
        parts.append(
            f"incomplete frame: {len(partial)} byte(s) of header "
            f"({bytes(partial).hex(' ')})"
        )
    discarded = read_bytes - sum(len(reply) for reply in replies) - len(partial)
    if discarded:
        parts.append(f"{discarded} byte(s) that formed no frame")
    return ", ".join(parts)


def _attempt_record(failures: Sequence[tuple[int, int, str]]) -> str:
    """Every attempt's failure, in order, runs of the same shape folded.

    "#1-3: incomplete frame: 12 of 13 bytes (count=1); #4: no bytes" says
    what a line did across a whole transaction; the last attempt alone said
    nothing about the first two.
    """

    if not failures:
        return "no attempt ran"
    return "; ".join(
        f"#{first}: {failure}" if first == last else f"#{first}-{last}: {failure}"
        for first, last, failure in failures
    )


def _remaining(deadline: float, action: str) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError(f"{action} exceeded its deadline")
    return value


__all__ = ["PySerialLink", "UartError", "UartLink", "UartRegisterTransport"]
