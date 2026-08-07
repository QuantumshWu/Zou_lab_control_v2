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
