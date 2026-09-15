"""The module's own configuration console, over the same serial line.

An N100 normally streams binary FDILink frames and listens to nothing.  It
also carries an ASCII command console, documented in chapter 5 of FDI's
《通信协议》: send ``#fconfig``, the module STOPS navigating and stops
emitting frames, every ``#f`` command is answered in plain text, and
``#fdeconfig`` (or a confirmed ``#freboot``) puts it back on the air.  That
is the whole of this module.

What it IS, this bench already has a name for: a ``ScpiLink`` -- write a
text command, query one and read the answer, close when done.  A Rigol and
a Tektronix speak that over VISA; an N100 speaks it over the bare serial
line it streams on, and the difference is confined to how a reply ends.
So the only things here that are the N100's own are entering and leaving
the console, and what its two configuration commands print.

Two facts shape the rest.  The console is the ONLY documented way to move a
setting -- the vendor's own ground station uses a MAVLink parameter path
whose wire format is nowhere in the shipped material, and the binary config
packets (0x7C/0x7D) have no units, no stated direction and two contradictory
payload lengths, so neither can be written from documentation alone.  And
entering the console silences the stream, so the reader that owns this port
has to be parked first: a capture cannot be running while a knob moves.

What this module answers, recorded off its wire and not read out of a
manual::

    > #fconfig                      *#OK
    > #faxis                        imu_algn_roll = 0.000000
                                    imu_algn_pitch = 0.000000
                                    imu_algn_yaw = 0.000000
                                    *#OK
    > #fparam get MSG_IMU           MSG_IMU=4
    > #fparam                       *#ERROR
    > #fmsg                         MSG_IMU[40]   10.0Hz
                                    MSG_AHRS[41]    0.0Hz
                                    ... one line per packet, about 1900 bytes
    > #fparam get FILT_LPF_ENABLED  FILT_LPF_ENABLED=0.000000
    > #fdeconfig                    (the binary stream resumes)

Read off that.  ``#fmsg`` with no argument is how the module enumerates
itself, and it gives each packet's rate in HERTZ -- so the packet list is
the module's and nothing here decides what packets exist.  ``#fparam get``
prints ``NAME=value`` with no spaces and no ``*#OK`` after it, while
``#faxis`` prints ``name = value`` WITH spaces: no one reply layout covers
this console, which is why nothing here is judged by one.  A bare
``#fparam`` is an error, so parameters cannot be enumerated and have to be
asked for by name.  And ``MSG_IMU=4`` standing beside ``MSG_IMU[40]
10.0Hz`` is what says a rate is stored as a LADDER INDEX: rung 4 is 10 Hz.
That is the whole reason ``#fmsg 40 100`` did nothing -- the rate never
arrives as a number of hertz.

And nothing here judges the module by a BANNER.  The manual prints
``Config Mode`` as the reply to ``#fconfig``; a real one answers ``*#OK``.
The manual also states the judgement that actually holds -- "if the data
stops being sent, config mode was entered" -- and that is the one used
here: entering is the stream STOPPING, leaving is the stream COMING BACK.
A string a document printed once is not a contract; what the module does
with its serial line is.
"""

from __future__ import annotations

import re
import time

from typing import Callable

from zlc_atom.authoring import TuneRefused

#: The first byte of every FDILink frame -- what the module's stream looks
#: like, and therefore what "it is navigating again" looks like.
FRAME_HEAD = 0xFC

#: A frame header followed by a NAVIGATION packet type.  0xF0 is the
#: module's 1 Hz heartbeat, which it sends while it is NOT navigating --
#: this driver's own stream check says exactly that, and counting it here
#: as "still streaming" made the two halves read the same two bytes in
#: opposite directions, so a module that entered config mode on the wrong
#: side of a heartbeat tick was refused.
_STREAM_MARKS = (b"\xfc\x40", b"\xfc\x41", b"\xfc\x42")


#: What the console appends to every command, and what it answers with.
#: These are the literal bytes the vendor's own FDILinkTool puts on the wire
#: (``#fconfig\r\n`` / ``#fdeconfig\r\n``), not a guess at the line ending.
LINE_END = "\r\n"

#: Stands for "the module says it has not got this one", so that a real
#: absence can be told from "not heard yet" without either being None.
_ABSENT = object()

#: The module's own refusal token, seen in the ground truth: a bare
#: ``#fparam`` answers ``*#ERROR``.  It is not a banner being judged -- it
#: is the one word this console uses to say NO, and a command whose reply
#: is that word did not happen.
ERROR = "*#ERROR"

#: What the module says when a command went through.  Both of these have
#: been seen: the manual documents ``*#OK`` for most commands and
#: ``Config Mode`` for ``#fconfig``, while a real module answers ``*#OK``
#: to ``#fconfig`` as well.  They are logged and passed on, never used to
#: decide whether a command worked -- see the module docstring.
OK = "*#OK"
CONFIG_BANNER = "Config Mode"

#: Commands the module refuses to run until it is answered ``y``.
CONFIRM_PROMPT = "(y/n)"

#: How long one command may take to answer.  Generous on purpose:
#: configuring is something an operator does now and then, never a hot
#: path, and the cost of being wrong in the two directions is not
#: symmetric -- waiting too long makes a settings page slow, cutting a
#: reply short makes the driver believe the module said something it did
#: not finish saying.
REPLY_TIMEOUT_SECONDS = 4.0

#: The module keeps sending for a moment after ``#fconfig`` -- frames
#: already in flight -- so entering waits this long for the line to go
#: quiet.  This one is a judgement about the STREAM, which is dense, so it
#: does not need the margin a printed reply does.
ENTER_QUIET_SECONDS = 0.25

#: A reply has no end marker, so it ends when the module has said nothing
#: for this long.  It was 0.08 s, tuned down to make the tests quick, and
#: that is the wrong thing to trade: a module that prints an
#: acknowledgement and then takes a breath before the rest would have been
#: read as having answered with only the acknowledgement.  Tests that need
#: to be fast pass their own timeout to the console instead.
REPLY_QUIET_SECONDS = 0.35

#: And for the one reply that is thirty lines long.  Measured on a real
#: module: ``#fmsg`` does not arrive in one piece, it comes in batches with
#: gaps between them, and a gap wider than the ordinary quiet window cut it
#: off mid-line -- five packets read as the module's whole enumeration,
#: with the remaining twenty-five spilling into the next command.
LISTING_QUIET_SECONDS = 1.5

#: And the whole listing may take this long to finish arriving.  On the
#: bench all 1904 bytes were in hand within a second; this is that with
#: room for a module having a slower day.
LISTING_TIMEOUT_SECONDS = 12.0

#: ``MSG_IMU=4`` from ``#fparam get``, and ``imu_algn_yaw = 0.000000``
#: from ``#faxis``: the same shape with and without spaces, which is why
#: the spaces are optional here rather than assumed to be there.
_PARAM_LINE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*) *= *(?P<value>[-+]?[0-9]+(?:\.[0-9]+)?)"
)

#: ``MSG_IMU[40]   10.0Hz`` from ``#fmsg``: the module enumerating itself,
#: one line per packet, with the rate in hertz.
_PACKET_LINE = re.compile(
    r"(?P<name>MSG_[A-Z0-9_]+)\[(?P<id>[0-9A-Fa-f]{1,2})\] *"
    r"(?P<hz>[0-9]+(?:\.[0-9]+)?)Hz",
    re.IGNORECASE,
)

#: The packet every N100 sends, and the one this bench reads.  Discovery
#: recognises the module by these frames, so a module that is here at all
#: lists this packet -- which makes its absence from a ``#fmsg`` answer a
#: statement about the ANSWER, not about the module.
IMU_PACKET_NAME = "MSG_IMU"

#: Anything that is neither printable ASCII nor a line ending: a frame's
#: bytes, and nothing the console prints.  A run between two of these is a
#: candidate for something the module SAID.
_NOT_PRINTED = re.compile(rb"[^\x09\x0a\x0d\x20-\x7e]+")

#: A noise floor, not a reply shape.  A binary payload can land on a short
#: run of printable bytes followed by CRLF by chance -- measured at 30 in
#: 200 000 IMU frames at one character, 6 at three -- and the shortest
#: thing this module is recorded printing is ``*#OK``, four characters.
#: So three is below everything it says and above most of what a frame
#: can fake.  It judges that the module printed SOMETHING, never
#: what.
_SHORTEST_PRINTED = 3


def printed_lines(data: bytes) -> tuple[str, ...]:
    """The console's own words in ``data``, without the stream's bytes.

    A reply is printed text ending in CRLF; a frame is binary, starts 0xFC
    and ends 0xFD.  So the stream's bytes are cut out as separators and
    what survives, terminated, is what the module said.  It has to be done
    this way round rather than by splitting on the line ending first: the
    frames that drain out after ``#fconfig`` carry no CRLF of their own, so
    they and the acknowledgement behind them arrive as ONE piece, and a
    test that asked whether that piece was printable would throw the
    acknowledgement away with them.

    This is what tells the module's ANSWER from the module still draining
    out, which a quiet rule cannot do: bytes arriving and then stopping
    look the same either way.
    """

    lines: list[str] = []
    for run in _NOT_PRINTED.split(data):
        for piece in run.split(LINE_END.encode("ascii"))[:-1]:
            if len(piece.strip()) >= _SHORTEST_PRINTED:
                lines.append(piece.decode("ascii").strip("\r\n"))
    return tuple(lines)


class FdiConfigConsole:
    """The module's text-command link, open while its stream is stopped.

    Satisfies ``devices.visa.ScpiLink`` -- ``write``, ``query``, ``close`` --
    so it is the same shape as the link a Rigol or a Tektronix is driven
    through, over a different transport.  ``enter`` is the one thing that
    has no counterpart there: a VISA instrument is always listening, and
    this one has to be taken off the air first.

    The caller owns the port and is responsible for having parked whatever
    reads it; this class only talks.  Use it as a context manager so the
    module always gets its ``#fdeconfig`` even when a command raises.
    """

    def __init__(
        self,
        port,
        *,
        reply_timeout: float = REPLY_TIMEOUT_SECONDS,
        reply_quiet: float = REPLY_QUIET_SECONDS,
    ) -> None:
        self._port = port
        self._reply_timeout = float(reply_timeout)
        self._reply_quiet = float(reply_quiet)
        self._entered = False
        #: Whatever the module printed on the way in, for the record.  Not
        #: a judgement: see the module docstring.
        self.greeting = ""
        #: The last command sent and what came back, verbatim.  Kept because
        #: every wrong turn in this driver so far has been an assumption
        #: about what the module would say, and the fastest way to settle
        #: the next one is to have its actual words to hand.
        self.last_exchange: tuple[str, str] = ("", "")

    # ------------------------------------------------------------ session
    def __enter__(self) -> "FdiConfigConsole":
        self.enter()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def enter(self) -> None:
        """Stop the stream and take the module into config mode.

        Two things have to be true when this returns, and both are about
        what the module DID rather than what it printed.

        First, it must have finished saying whatever it says.  Returning
        while its acknowledgement is still on the way is not a harmless
        early exit: the console has no sequence numbers, so that reply
        arrives during the NEXT command and every command afterwards reads
        the previous one's answer.  A whole settings page then comes back
        as "this firmware has none of these parameters", which is exactly
        the symptom that sent me looking here.

        Second, it must have stopped navigating -- and that is read off the
        line, by looking for frame headers rather than for a banner.  A
        module still streaming never entered, whatever it printed.

        The first of those is why this read is told the reply will be
        PRINTED.  The frames already in flight when ``#fconfig`` landed
        arrive before the acknowledgement does, and to a plain quiet rule
        they are indistinguishable from it: bytes came, then the line went
        quiet, so the reply must be over.  It is not -- the module has not
        started speaking yet -- and returning there puts its ``*#OK`` in
        ``#fmsg``'s window, which is how Device Control came up with no
        packet rates on it.  What separates them is not timing but kind:
        the answer is text, the stream is not.
        """

        if self._entered:
            return
        self._port.reset_input_buffer()
        self._write("#fconfig")
        answer, _quiet = self._read(ENTER_QUIET_SECONDS, printed_reply=True)
        self._entered = True
        self.greeting = LINE_END.join(printed_lines(answer)).strip()
        # Whatever it said is said; now look at what it is doing.
        self._port.reset_input_buffer()
        listening, _ = self._read(ENTER_QUIET_SECONDS, silence_ends_it=True)
        if any(mark in listening for mark in _STREAM_MARKS):
            self.close()
            raise RuntimeError(
                "the module on this port kept streaming through #fconfig, so "
                f"it never entered config mode (it said {self.greeting[:120]!r})"
            )

    def close(self) -> None:
        """Put the module back on the air, whatever happened in between."""

        if not self._entered:
            return
        try:
            self._write("#fdeconfig")
            # Waiting for quiet would wait forever, because the module
            # answers this one by going back ON the air.  So the judgement
            # is again what it does: the first frame header off the stream
            # says it is navigating, whatever it printed first.
            self._read(ENTER_QUIET_SECONDS, until=bytes((FRAME_HEAD,)))
        finally:
            self._entered = False

    # --------------------------------------------------------------- link
    def write(self, command: str) -> None:
        """Send one command and do not wait for what it says back."""

        self._require_console(command)
        self._write(command)

    def query(self, command: str, *, quiet: float | None = None) -> str:
        """Send one command and answer with everything the module said back.

        ``quiet`` lengthens the window for a reply that arrives in batches
        rather than in one go.

        Use this only for commands whose reply carries nothing to identify
        it by.  Where the reply names itself -- and the two that matter
        both do -- ``read_until_named`` is the one to use, because clearing
        the line cannot help a reply that has not been sent yet.
        """

        self._require_console(command)
        # Whatever is still on the line belongs to the command before this
        # one.  This console has no sequence numbers, so an unread tail
        # would be read as THIS command's answer and every command after it
        # would read the one before -- which is exactly how a settings page
        # came back as "this firmware has none of these parameters".  The
        # previous command has already had whatever it could get from those
        # bytes; nothing here wants them.
        self._port.reset_input_buffer()
        self._write(command)
        answer = self._read_until_quiet(self._reply_quiet if quiet is None else quiet)
        self.last_exchange = (command, answer)
        if not answer.strip():
            # Nothing came back within the timeout.  It may still be on its
            # way, and if it is it will arrive during the next command and
            # be read as that command's answer -- so this session cannot be
            # trusted with another question.
            raise RuntimeError(
                f"the module did not answer {command!r} within "
                f"{self._reply_timeout:g} s; the console cannot stay in step "
                "after that, so this settings session is abandoned"
            )
        return answer

    def read_until_named(
        self,
        command: str,
        identifies: "Callable[[str], object]",
        *,
        quiet: float | None = None,
        timeout: float | None = None,
    ) -> object:
        """Send one command and read until its OWN reply arrives.

        This console has no sequence numbers, and clearing the line before a
        command only discards what has already been delivered -- a reply the
        module has not sent yet cannot be cleared, and lands in the next
        command's window.  That is one step of desynchronisation, and from
        there every command reads the one before it.

        So the reply is not taken on timing at all.  ``identifies`` is
        handed everything heard so far and returns the answer once it can
        see it -- the parameter that was asked for, the packet list with the
        packet every module has in it -- and until then the reading goes on.
        A reply belonging to an earlier command is read, found not to name
        this one, and simply kept waiting past.  Nothing is inferred from
        how long anything took.
        """

        self._require_console(command)
        self._port.reset_input_buffer()
        self._write(command)
        window = self._reply_quiet if quiet is None else quiet
        deadline = time.monotonic() + (
            self._reply_timeout if timeout is None else timeout
        )
        heard = ""
        while time.monotonic() < deadline:
            chunk, quiet_now = self._read(window)
            if chunk:
                heard += chunk.decode("ascii", "replace")
                self.last_exchange = (command, heard)
            if not quiet_now:
                # Still mid-batch.  Asking now would take the first line of
                # a listing for the whole of it -- the packet every module
                # has is the FIRST one printed, so a check for its presence
                # is satisfied before the other twenty-nine arrive, and they
                # then spill into the next command.
                continue
            answer = identifies(heard)
            if answer is not None:
                return answer
        self.last_exchange = (command, heard)
        raise RuntimeError(
            f"the module never answered {command!r} in time; it said "
            f"{heard.strip()[:160]!r}"
        )

    def _require_console(self, command: str) -> None:
        if not self._entered:
            raise RuntimeError(
                f"{command!r} is a config-mode command and this console is not "
                "in config mode"
            )

    def get_parameter(self, name: str) -> str | None:
        """One named parameter's value, or None when this firmware lacks it.

        Recorded: ``#fparam get MSG_IMU`` answers ``MSG_IMU=4`` -- the name
        echoed, no spaces, no ``*#OK`` -- and a name it has not got draws
        ``*#ERROR``.  Both of those identify themselves, which is what lets
        this wait for its own reply rather than read whichever one turns up.
        """

        wanted = str(name).upper()

        def identifies(heard: str) -> object:
            for found in _PARAM_LINE.finditer(heard):
                if found["name"].upper() == wanted:
                    return found["value"]
            # The module's own word for "not present" -- but only once
            # nothing else is still owed, since an earlier command's reply
            # could carry it.
            if ERROR in heard and not _PARAM_LINE.search(heard):
                return _ABSENT
            return None

        answer = self.read_until_named(f"#fparam get {name}", identifies)
        return None if answer is _ABSENT else str(answer)

    def packet_rates(self) -> tuple[tuple[str, int, float], ...]:
        """Every packet this module has, as ``(name, id, hertz)``.

        The module enumerating itself.  It is also the only readback there
        is for a rate, since the parameter holding one reads back as the
        ladder index that was written to it.

        The listing arrives in batches on a real module, so it is read until
        the packet EVERY N100 has is in it -- which both identifies the
        reply as this command's and proves the listing is not a fragment.
        """

        def identifies(heard: str) -> object:
            found = tuple(
                (match["name"], int(match["id"], 16), float(match["hz"]))
                for match in _PACKET_LINE.finditer(heard)
            )
            if any(name.upper() == IMU_PACKET_NAME for name, _id, _hz in found):
                return found
            return None

        return self.read_until_named(
            "#fmsg",
            identifies,
            quiet=LISTING_QUIET_SECONDS,
            timeout=LISTING_TIMEOUT_SECONDS,
        )

    def set_parameter(self, name: str, value: str) -> str:
        """Write one named parameter and answer with what it reads back as.

        The reply to ``#fparam set`` is not documented, so the write is not
        judged by what it printed: the value is read back, and the readback
        IS the answer.  A parameter this firmware does not have reads back
        as nothing, which is a refusal rather than a silent no-op.
        """

        self.query(f"#fparam set {name} {value}")
        reading = self.get_parameter(name)
        if reading is None:
            raise TuneRefused(
                f"this module has no parameter {name!r}: it would not read "
                "the name back after the write"
            )
        return reading

    def reboot(self) -> None:
        """Warm-restart the module, discarding anything not written to flash.

        The manual is explicit: "on restart all unsaved settings will not be
        saved and will not take effect."  That makes this the UNDO for a
        session that never called ``save`` -- whatever it changed goes away
        and the module comes back on the configuration it booted with.  It
        is the only undo that does not depend on knowing how the module
        spells the value it was given, which is exactly the thing this
        driver has been wrong about.

        The command needs confirming with ``y``.  The module is restarting
        when this returns, so the console is over.
        """

        self._write("#freboot")
        answer, _finished = self._read(
            self._reply_quiet, until=CONFIRM_PROMPT.encode("ascii")
        )
        if CONFIRM_PROMPT.encode("ascii") not in answer:
            # No prompt means the module is not waiting for a yes, and
            # sending one anyway puts a bare "y" on the wire for it to read
            # as a command.  It is still in the console, so say so and let
            # the caller's exit take it out properly.
            raise RuntimeError(
                "the module did not ask to confirm the restart; it answered "
                f"{answer.decode('ascii', 'replace').strip()[:120]!r}"
            )
        self._write("y")
        self._entered = False

    def save(self) -> str:
        """Commit to flash, and refuse only on the module's own refusal word.

        Nothing reports what is in flash, so a save cannot be CONFIRMED the
        way a written setting can, and this does not try to: a reply that
        is not a refusal is passed back unjudged.  But ``*#ERROR`` is not a
        banner -- it is this console's word for no, recorded in its own
        answers -- and a save that was refused must not be reported as
        done, because everything after it is built on the value having
        reached flash.
        """

        answer = self.query("#fsave")
        if ERROR in answer:
            raise TuneRefused(
                f"the module refused to save: {answer.strip()[:120]!r}"
            )
        return answer

    # -------------------------------------------------------------- lines
    def _write(self, text: str) -> None:
        self._port.write((text + LINE_END).encode("ascii"))
        flush = getattr(self._port, "flush", None)
        if callable(flush):
            flush()

    def _read_until_quiet(self, quiet: float) -> str:
        """What the module said, once it had finished saying it.

        Running out of time with bytes still arriving is NOT the module
        finishing: the rest of that reply is still on its way and will land
        inside the next command's window.  The console has no sequence
        numbers, so from there on every command reads the one before it --
        which is how a whole settings page came back as "this firmware has
        none of these parameters".  A truncated reply therefore ends the
        session instead of being handed back as if it were whole.
        """

        answer, finished = self._read(quiet)
        if not finished:
            raise RuntimeError(
                "the module was still talking when the reply timed out after "
                f"{self._reply_timeout:g} s; the rest of it would be read as "
                "the next command's answer, so this settings session is "
                f"abandoned (it had said {answer.decode('ascii', 'replace')[:80]!r})"
            )
        return answer.decode("ascii", "replace")

    def _read(
        self,
        quiet: float,
        *,
        until: bytes | None = None,
        silence_ends_it: bool = False,
        printed_reply: bool = False,
    ) -> tuple[bytes, bool]:
        """What the module said, and whether the line then went quiet.

        The console has no general end-of-reply marker: ``#fmsg`` answers
        with one line per packet, ``#faxis`` with three, ``#fsave`` with
        one.  So a reply is normally "what arrived before the line went
        quiet", bounded by the command timeout.

        The second half of the answer is the important one.  Quiet is how
        this driver knows the module stopped navigating; running out of
        time with bytes still arriving is how it knows the module never
        did.  ``until`` short-circuits the wait for the one command whose
        reply is followed by the stream starting again, where quiet never
        comes.

        ``silence_ends_it`` is for entering config mode, and ONLY for it:
        there, hearing nothing IS the answer.  Everywhere else a reply that
        has not begun yet is not a reply that will not come -- ``#fmsg``
        prints some 1900 bytes and takes its time about starting -- and
        treating the pause before it as "the module said nothing" is what
        emptied Device Control.

        ``printed_reply`` says the same thing about bytes that are not
        text.  Quiet only ends a reply that has BEGUN, and what begins it
        is a printed line; frames draining out of a module that has just
        been told to stop are the previous state of affairs ending, not
        this command's answer starting.  Waiting out the full timeout for a
        module that prints nothing is the right price: by then nothing is
        left in flight, and the session stays in step.
        """

        deadline = time.monotonic() + self._reply_timeout
        chunks: list[bytes] = []
        last = time.monotonic()
        while time.monotonic() < deadline:
            waiting = getattr(self._port, "in_waiting", 0)
            chunk = self._port.read(waiting if waiting else 1)
            now = time.monotonic()
            if chunk:
                chunks.append(chunk)
                last = now
                if until is not None and until in b"".join(chunks):
                    return b"".join(chunks), False
            elif printed_reply and now - last >= quiet:
                if printed_lines(b"".join(chunks)):
                    return b"".join(chunks), True
                # Bytes arrived and stopped, but none of them were words:
                # that was the stream draining, and the answer has not
                # started.  Re-arm the quiet window and keep listening.
                last = now
            elif chunks and now - last >= quiet:
                return b"".join(chunks), True
            elif silence_ends_it and now - last >= quiet:
                # Nothing at all, and nothing is what was being asked about.
                return b"", True
        return b"".join(chunks), False


__all__ = [
    "CONFIG_BANNER",
    "ERROR",
    "CONFIRM_PROMPT",
    "ENTER_QUIET_SECONDS",
    "FdiConfigConsole",
    "IMU_PACKET_NAME",
    "LISTING_QUIET_SECONDS",
    "LISTING_TIMEOUT_SECONDS",
    "LINE_END",
    "OK",
    "printed_lines",
    "REPLY_QUIET_SECONDS",
]
