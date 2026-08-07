"""What the host does when the serial line loses a frame.

It will.  The line has no flow control, the bridge on the board is a
single-frame state machine with no receive queue, and one mis-sampled stop bit
at 3 Mbaud makes it abandon the frame it was reading and go back to hunting for
a sync pair.  That frame is never acknowledged.
"""

from __future__ import annotations

import tempfile


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
    transport = UartRegisterTransport(state_dir=tempfile.mkdtemp(), link=link)
    transport.start()

    rows = tuple((100 + index, index) for index in range(0, 40, 2))
    transport.write_words(rows)

    assert transport.resends > 0, "the dropped frames must have been noticed"
    # Every address ends up written, which is the point: a lost frame is not a
    # lost load.
    assert len(set(link.sent)) < len(link.sent), "some SEQ was sent twice"


def test_a_command_strobe_is_never_sent_twice() -> None:
    """A command that WAS executed and whose acknowledgement was lost would be
    executed twice, and "fire twice" is not a recoverable arithmetic error."""

    import pytest

    from zlc_pulse.transport.uart import UartRegisterTransport

    class _SilentLink:
        port = "COM-TEST"
        baud = 3_000_000
        last_shortfall = "0 of 1 replies"

        def open(self) -> None: ...

        def close(self) -> None: ...

        def write_batch(self, requests, *, deadline, stop=None):
            return []

    transport = UartRegisterTransport(state_dir=tempfile.mkdtemp(), link=_SilentLink())
    transport.start()
    with pytest.raises(TimeoutError, match="1 attempt"):
        transport.write_words(((1, 0), (1, 8)), resend=False)
    assert transport.resends == 0


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
    transport = UartRegisterTransport(state_dir=tempfile.mkdtemp(), link=link)
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

    transport = UartRegisterTransport(state_dir=tempfile.mkdtemp(), link=_CorruptingLink())
    transport.start()
    transport.write_words(((7, 1), (9, 2)))
    assert transport.resends == 1


def test_a_rejected_strobe_is_provably_unexecuted_and_may_go_again() -> None:
    """ST_CRC_FAIL is the board saying "that arrived damaged, I did nothing".

    Nothing ran, so even a command strobe -- which must never be resent into
    AMBIGUITY -- may safely be sent again.  The refusal to resend exists for
    the lost-acknowledgement case, where the command may have executed.
    """

    from zlc_pulse.transport import uart_frame as framing
    from zlc_pulse.transport.uart import UartRegisterTransport

    class _RejectingOnceLink:
        port = "COM-TEST"
        baud = 3_000_000
        last_shortfall = ""

        def __init__(self) -> None:
            self.calls = 0

        def open(self) -> None: ...

        def close(self) -> None: ...

        def write_batch(self, requests, *, deadline, stop=None):
            self.calls += 1
            status = framing.ST_CRC_FAIL if self.calls == 1 else framing.ST_OK
            return [
                framing.encode_reply(request[3], status, ())
                for request in requests
            ]

    link = _RejectingOnceLink()
    transport = UartRegisterTransport(state_dir=tempfile.mkdtemp(), link=link)
    transport.start()
    # resend=False is the strobe contract, and a rejected strobe still goes again.
    transport.write_words(((1, 0), (1, 8)), resend=False)
    assert link.calls == 2


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

    transport = UartRegisterTransport(state_dir=tempfile.mkdtemp(), link=_Idle())
    ten_frames = [framing.encode_write(index * 4, (0,), seq=index) for index in range(10)]
    assert transport._attempt_budget(ten_frames) < 0.2
