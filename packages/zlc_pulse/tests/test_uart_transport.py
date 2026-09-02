"""What the host does when the serial line loses a frame.

It will.  The line has no flow control, the bridge on the board is a
single-frame state machine with no receive queue, and one mis-sampled stop bit
at 3 Mbaud makes it abandon the frame it was reading and go back to hunting for
a sync pair.  That frame is never acknowledged.
"""

from __future__ import annotations


def test_a_frame_the_board_never_answered_is_sent_again() -> None:
    """One mis-sampled stop bit and the bridge abandons that frame.

    It is a single-frame state machine with no receive queue: on a framing
    error it stops reading and goes back to hunting for a sync pair, so the
    frame is never acknowledged and nothing downstream hears about it.  On a
    3 Mbaud line whose other end is a USB-serial adapter with its own clock,
    that happens occasionally on one machine and never on another -- which is
    why the same board loaded fine here and timed out there, "9 of 10 replies".

    Every frame already carried a SEQ and every acknowledgement carried it
    back.  Using it is the difference between losing a frame and losing a load.
    """

    from zlc_pulse.transport import uart as uart_module
    from zlc_pulse.transport.uart import UartRegisterTransport
    from zlc_pulse.transport import uart_frame as framing

    class _LossyLink:
        """Answers every frame but the first of each four."""

        port = "COM-TEST"
        baud = 3_000_000

        def __init__(self) -> None:
            self.seen = 0
            self.sent: list[int] = []
            self.last_shortfall = ""

        def open(self) -> None: ...

        def close(self) -> None: ...

        def write_batch(self, requests, *, deadline, stop=None):
            replies = []
            for request in requests:
                self.seen += 1
                self.sent.append(request[3])
                if self.seen % 4 == 1:
                    continue
                replies.append(framing.encode_reply(request[3], framing.ST_OK, ()))
            if len(replies) != len(requests):
                self.last_shortfall = f"{len(replies)} of {len(requests)} replies"
            return replies

        def exchange(self, request, *, deadline, stop=None):
            return framing.encode_reply(request[3], framing.ST_OK, (0,))

    link = _LossyLink()
    transport = UartRegisterTransport(link=link)
    transport.start()

    rows = tuple((100 + index, index) for index in range(0, 40, 2))
    transport.write_words(rows)

    assert transport.resends > 0, "the dropped frames must have been noticed"
    # Every address ends up written, which is the point: a lost frame is not a
    # lost load.
    assert len(set(link.sent)) < len(link.sent), "some SEQ was sent twice"


def test_a_command_strobe_is_never_sent_twice() -> None:
    """A command that WAS executed and whose acknowledgement was lost would be
    executed twice, and "fire twice" is not a recoverable arithmetic error.

    Its one attempt waits the attempt budget, not the transaction deadline:
    the acknowledgement is a courtesy and the status read that follows is
    the authority, so a lost FIRE acknowledgement costs milliseconds before
    verification.  The device used to hand the line a window of its own for
    this (a 0.3 s constant) -- a second owner of the same claim, and a dead
    one, because the line's budget was already shorter.
    """

    import time

    import pytest

    from zlc_pulse.transport import uart_frame as framing
    from zlc_pulse.transport.uart import UartRegisterTransport

    class _SilentLink:
        port = "COM-TEST"
        baud = 3_000_000
        last_shortfall = "0 of 1 replies"

        def __init__(self) -> None:
            self.windows: list[float] = []

        def open(self) -> None: ...

        def close(self) -> None: ...

        def write_batch(self, requests, *, deadline, stop=None):
            # A real link waits out its deadline before returning short.
            self.windows.append(deadline - time.monotonic())
            time.sleep(max(0.0, deadline - time.monotonic()))
            return []

    link = _SilentLink()
    transport = UartRegisterTransport(link=link)
    transport.start()
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="1 attempt"):
        transport.write_words(((1, 0), (1, 8)), resend=False)
    elapsed = time.monotonic() - started
    assert transport.resends == 0
    budget = transport._attempt_budget([framing.encode_write(1, (0, 8), seq=1)])
    assert len(link.windows) == 1
    assert link.windows[0] <= budget + 0.005, f"{link.windows[0]:.3f}s window for one strobe"
    assert elapsed < budget + 0.1, f"{elapsed:.3f}s for one unanswered strobe"


def test_resending_happens_while_there_is_still_time_to_resend() -> None:
    """The first implementation was theatre, and this is the test it lacked.

    A genuinely lost frame keeps the reply read waiting -- a REAL link waits
    out its deadline before returning short.  With one deadline shared across
    attempts, attempt two therefore began with nothing left and died in the
    write path without ever retransmitting.  The earlier lossy-link test
    missed it because its fake returned instantly, spending no clock.
    """

    import time

    from zlc_pulse.transport import uart_frame as framing
    from zlc_pulse.transport.uart import UartRegisterTransport

    class _RealisticallyLossyLink:
        port = "COM-TEST"
        baud = 3_000_000
        last_shortfall = "9 of 10 replies"

        def __init__(self) -> None:
            self.margins: list[float] = []

        def open(self) -> None: ...

        def close(self) -> None: ...

        def write_batch(self, requests, *, deadline, stop=None):
            self.margins.append(deadline - time.monotonic())
            if len(self.margins) == 1:
                # A lost frame: wait for a reply that never comes, all the way
                # to the deadline, exactly as _read_replies does.
                time.sleep(max(0.0, deadline - time.monotonic()))
                return [
                    framing.encode_reply(request[3], framing.ST_OK, ())
                    for request in requests[:-1]
                ]
            return [
                framing.encode_reply(request[3], framing.ST_OK, ())
                for request in requests
            ]

    link = _RealisticallyLossyLink()
    transport = UartRegisterTransport(link=link)
    transport.start()
    transport.write_words(tuple((100 + index, index) for index in range(10)))

    assert len(link.margins) == 2, "the lost frame must actually be retransmitted"
    assert link.margins[1] > 0.05, (
        "the retry must begin with time of its own, not the exhausted deadline: "
        f"{link.margins[1]:.3f}s left"
    )
    assert transport.resends == 1


def test_a_damaged_acknowledgement_means_send_that_frame_again() -> None:
    """A corrupted reply is the same physical fault as a lost one.

    decode_reply raising FrameError out of the write path was the rig's
    "method=fire error=FrameError: UART reply CRC mismatch" -- fatal, for a
    frame whose write is idempotent and whose retry is free.
    """

    from zlc_pulse.transport import uart_frame as framing
    from zlc_pulse.transport.uart import UartRegisterTransport

    class _CorruptingLink:
        port = "COM-TEST"
        baud = 3_000_000
        last_shortfall = ""

        def __init__(self) -> None:
            self.calls = 0

        def open(self) -> None: ...

        def close(self) -> None: ...

        def write_batch(self, requests, *, deadline, stop=None):
            self.calls += 1
            replies = [
                framing.encode_reply(request[3], framing.ST_OK, ())
                for request in requests
            ]
            if self.calls == 1:
                # One acknowledgement arrives with a flipped byte.
                damaged = bytearray(replies[0])
                damaged[4] ^= 0x40
                replies[0] = bytes(damaged)
            return replies

    transport = UartRegisterTransport(link=_CorruptingLink())
    transport.start()
    transport.write_words(((7, 1), (9, 2)))
    assert transport.resends == 1


def test_a_rejected_strobe_is_provably_unexecuted_and_may_go_again() -> None:
    """ST_CRC_FAIL is the board saying "that arrived damaged, I did nothing".

    Nothing ran, so even a command strobe -- which must never be resent into
    AMBIGUITY -- may safely be sent again.  The refusal to resend exists for
    the lost-acknowledgement case, where the command may have executed.

    Two laws ride on that verdict.  It belongs to the attempt that earned
    it: a strobe refused once and then UNANSWERED (a write that timed out,
    a reply that never came) is back in the ambiguous case, and the stale
    refusal must not send it a third time.  And a refused attempt is still
    charged its budget: a board that refuses within a round trip and is
    asked again within a round trip is a storm, not a retry (324,338
    attempts in one half-second deadline, ``resends`` meaningless).
    """

    import time

    import pytest

    from zlc_pulse.transport import uart_frame as framing
    from zlc_pulse.transport.uart import UartRegisterTransport

    class _RejectingOnceLink:
        port = "COM-TEST"
        baud = 3_000_000
        last_shortfall = ""

        def __init__(self, then_stall: bool = False) -> None:
            self.calls = 0
            self.then_stall = then_stall

        def open(self) -> None: ...

        def close(self) -> None: ...

        def write_batch(self, requests, *, deadline, stop=None):
            self.calls += 1
            if self.calls == 2 and self.then_stall:
                raise TimeoutError("UART write timed out on COM-TEST")
            status = framing.ST_CRC_FAIL if self.calls == 1 else framing.ST_OK
            return [
                framing.encode_reply(request[3], status, ())
                for request in requests
            ]

    link = _RejectingOnceLink()
    transport = UartRegisterTransport(link=link)
    transport.start()
    budget = transport._attempt_budget([framing.encode_write(1, (0, 8), seq=1)])
    started = time.monotonic()
    # resend=False is the strobe contract, and a rejected strobe still goes again.
    transport.write_words(((1, 0), (1, 8)), resend=False)
    assert link.calls == 2
    assert transport.resends == 2, "both refused frames went again, once each"
    assert time.monotonic() - started >= budget - 0.005, "the refused attempt was not charged its budget"

    link = _RejectingOnceLink(then_stall=True)
    transport = UartRegisterTransport(link=link)
    transport.start()
    with pytest.raises(TimeoutError, match="after 2 attempt"):
        transport.write_words(((1, 0), (1, 8)), resend=False)
    assert link.calls == 2, "a refusal must not speak for the unanswered attempt after it"


def test_extraction_walks_past_a_damaged_frame_to_the_good_one_behind_it() -> None:
    """A damaged frame must not be believed, and must not misalign the stream.

    A corrupted COUNT field slices the wrong number of bytes, which eats the
    start of the NEXT frame -- so the CRC is checked at extraction and the
    stream re-hunted one byte later, the same recovery the board's own bridge
    performs in the other direction.
    """

    from zlc_pulse.transport import uart_frame as framing
    from zlc_pulse.transport.uart import _extract_reply

    good = framing.encode_reply(7, framing.ST_OK, ())
    damaged = bytearray(framing.encode_reply(6, framing.ST_OK, ()))
    damaged[4] ^= 0x01  # status byte flipped after the CRC was computed

    buffer = bytearray(bytes(damaged) + good)
    assert _extract_reply(buffer) == good
    assert _extract_reply(buffer) is None


def test_a_retry_costs_milliseconds_not_seconds() -> None:
    """The stall the operator feels IS the attempt budget.

    Every lost frame charges one budget before its resend, so an On Pulse over
    a lossy line hangs by exactly this number times the losses.  It began life
    at half a second and a normal cycle stalled for over a second; the physics
    -- wire time plus a ~16 ms USB latency timer -- needs tens of
    milliseconds.  Waiting too little is benign (writes are idempotent and a
    late duplicate is dropped by SEQ), so this pins the ceiling.
    """

    from zlc_pulse.transport import uart_frame as framing
    from zlc_pulse.transport.uart import UartRegisterTransport

    class _Idle:
        port = "COM-TEST"
        baud = 3_000_000

        def open(self) -> None: ...

        def close(self) -> None: ...

    transport = UartRegisterTransport(link=_Idle())
    ten_frames = [framing.encode_write(index * 4, (0,), seq=index) for index in range(10)]
    assert transport._attempt_budget(ten_frames) < 0.2


def test_the_attempt_budget_scales_with_the_transfer() -> None:
    """A scan start is sixty-four full frames, not a strobe.

    The budget and the deadline are the same physical claim about the same
    bytes, and the budget was written flat: 80 ms of host slack over 223 ms
    of wire left the first two attempts an 8 per cent margin, so the same
    bench saw one-frame strobes that never failed and scan starts that
    timed out at random -- the operator's exact report.  The budget must
    exceed the wire time by a real host factor, not a constant.
    """

    from zlc_pulse.transport import uart_frame as framing
    from zlc_pulse.transport.uart import UartRegisterTransport

    class _Idle:
        port = "COM-TEST"
        baud = 3_000_000

        def open(self) -> None: ...

        def close(self) -> None: ...

    transport = UartRegisterTransport(link=_Idle())
    table = [
        framing.encode_write(index * 4, tuple(range(256)), seq=index)
        for index in range(64)
    ]
    wire = (
        sum(len(frame) + 8 for frame in table)
        + framing.reply_frame_len(0) * len(table)
    ) * 10.0 / 3_000_000.0
    budget = transport._attempt_budget(table)
    # The host factor is the contract: an attempt whose host runs half
    # again slower than the line is late, not lost.  The old flat form gave
    # this transfer 0.08 s of absolute headroom -- 1.36x wire -- and failed
    # in the field, so the bar sits above what the defect provided.
    assert budget >= wire * 1.5, (
        f"a {wire * 1e3:.0f} ms transfer got only {(budget - wire) * 1e3:.0f} ms "
        "of host headroom"
    )


class _TailDroppingPort:
    """A serial port whose board answers every READ, dropping the reply's
    last byte on the first ``drops`` exchanges.

    The archived slm_feedback failure, byte for byte: a 13-byte READ reply
    (seq 0x71, ST_OK, CURSOR=85) arriving as its first 12 bytes -- every
    field valid, only the final CRC byte missing.
    """

    write_timeout = None

    def __init__(self, drops: int) -> None:
        self.drops = drops
        self.exchanges = 0
        self.sent_at: list[float] = []
        self._pending = b""

    def reset_input_buffer(self) -> None:
        self._pending = b""

    def write(self, payload: bytes) -> None:
        import time

        from zlc_pulse.transport import uart_frame as framing

        self.exchanges += 1
        self.sent_at.append(time.monotonic())
        reply = framing.encode_reply(payload[3], framing.ST_OK, (85,))
        self._pending = reply[:-1] if self.exchanges <= self.drops else reply

    def flush(self) -> None: ...

    @property
    def in_waiting(self) -> int:
        return len(self._pending)

    def read(self, size: int) -> bytes:
        chunk, self._pending = self._pending[:size], self._pending[size:]
        return chunk


def test_a_reply_one_byte_short_is_asked_again_within_milliseconds() -> None:
    """The lost byte is never coming; the next request is.

    Three replies in a row arrived one byte short and the read spent 4.88 s
    waiting for the third one's last byte -- the whole transaction deadline,
    because the frame parser saw "not finished yet" and the retry law gave
    its final attempt the patience owed to a slow link.  The line loses
    bytes, it does not delay them: EVERY attempt, the last included, is
    worth the budget its bytes need and then the request goes again, for as
    many attempts as the deadline holds.  The law is pinned as the spacing
    between requests on the wire: each one budget, none the deadline.
    """

    import time

    from zlc_pulse.transport import uart_frame as framing
    from zlc_pulse.transport.uart import PySerialLink, UartRegisterTransport

    link = PySerialLink("COM-TEST")
    link._serial = _TailDroppingPort(drops=3)
    transport = UartRegisterTransport(link=link)
    transport.start()
    budget = transport._attempt_budget([framing.encode_read(15, 1, seq=1)])

    started = time.monotonic()
    assert transport.read_word(15) == 85
    elapsed = time.monotonic() - started

    assert link._serial.exchanges == 4
    assert transport.resends == 3
    gaps = [b - a for a, b in zip(link._serial.sent_at, link._serial.sent_at[1:])]
    assert all(budget - 0.005 <= gap <= budget + 0.05 for gap in gaps), (
        f"requests spaced {[f'{gap:.3f}' for gap in gaps]}s; the budget is {budget:.3f}s"
    )
    assert elapsed < 4 * budget + 0.1, f"{elapsed:.3f}s for three lost bytes"


def test_a_read_that_never_completes_reports_every_attempt_by_shape() -> None:
    """What each attempt saw is the diagnosis; the last one alone is not.

    "incomplete frame: 12 of 13 bytes" is a byte lost on the board-to-host
    direction; "no bytes" is a request lost on the way out or a board that
    did not answer; bytes that formed no frame are noise.  A record that
    keeps only the final attempt cannot tell a run of dropped bytes from a
    line that died -- and the archived failure kept only the final attempt.
    """

    import pytest

    from zlc_pulse.transport.uart import PySerialLink, UartRegisterTransport

    link = PySerialLink("COM-TEST")
    link._serial = _TailDroppingPort(drops=10_000)
    transport = UartRegisterTransport(link=link, action_timeout=0.3)
    transport.start()

    with pytest.raises(TimeoutError) as caught:
        transport.read_word(15)
    message = str(caught.value)
    attempts = link._serial.exchanges
    assert attempts >= 4, message
    assert f"after {attempts} attempt(s)" in message
    assert f"#1-{attempts}: 0 of 1 replies, incomplete frame: 12 of 13 bytes (count=1)" in message
    assert "unparsed" not in message
    assert transport.resends == attempts - 1


def test_a_slow_write_is_a_timeout_this_layer_can_retry() -> None:
    """pyserial's write timeout is an OSError; the link translates it.

    ``SerialTimeoutException`` is not a ``TimeoutError``, so every retry
    handler above the link looked straight past it: one slow WriteFile was
    a hard first-attempt failure with pyserial's own words and no resend,
    on a line whose retry machinery exists for exactly that moment.
    """

    import time

    import pytest
    import serial

    from zlc_pulse.transport.uart import PySerialLink

    class _StalledPort:
        write_timeout = None

        def reset_input_buffer(self) -> None: ...

        def write(self, payload: bytes) -> None:
            raise serial.SerialTimeoutException("write timeout")

        def flush(self) -> None: ...

    link = PySerialLink("COM-TEST")
    link._serial = _StalledPort()
    with pytest.raises(TimeoutError, match="UART write timed out on COM-TEST"):
        link.exchange(b"\x01\x02\x03\x04", deadline=time.monotonic() + 1.0)
    with pytest.raises(TimeoutError, match="UART write timed out on COM-TEST"):
        link.write_batch([b"\x01\x02"], deadline=time.monotonic() + 1.0)


def test_a_write_timeout_on_an_early_attempt_is_retried() -> None:
    """The first attempt stalling in the WRITE path must not end the call."""

    from zlc_pulse.transport import uart_frame as framing
    from zlc_pulse.transport.uart import UartRegisterTransport

    class _FirstWriteStalls:
        port = "COM-TEST"
        baud = 3_000_000
        last_shortfall = ""

        def __init__(self) -> None:
            self.calls = 0

        def open(self) -> None: ...

        def close(self) -> None: ...

        def write_batch(self, requests, *, deadline, stop=None):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("UART write timed out on COM-TEST")
            return [
                framing.encode_reply(request[3], framing.ST_OK, ())
                for request in requests
            ]

    link = _FirstWriteStalls()
    transport = UartRegisterTransport(link=link)
    transport.start()
    transport.write_words([(0, 1), (4, 2), (8, 3)])
    assert link.calls == 2
