"""The module's own configuration console, over the same serial line.

An N100 normally streams binary FDILink frames and listens to nothing.  It
also carries an ASCII command console, documented in chapter 5 of FDI's
《通信协议》: send ``#fconfig``, the module STOPS navigating and stops
emitting frames, every ``#f`` command is answered in plain text, and
``#fdeconfig`` puts it back on the air.

What it IS, this bench already has a name for: a ``ScpiLink`` -- write a
text command, query one and read the answer, close when done.  A Rigol and
a Tektronix speak that over VISA; an N100 speaks it over the bare serial
line it streams on.

ONE RULE, and everything else here follows from it:

    ONE COMMAND AT A TIME, AND LEAVE A GAP.  Write it, read until the
    module has answered IT, wait, and only then write the next.  A command
    that goes unanswered ends the session; nothing is sent after it.

The gap is not politeness.  Measured on the bench, one command at a time,
each with its reply read in full::

    #fparam get MSG_IMU                  MSG_IMU=7
    #fmsg                                MSG_IMU[40]  100.0Hz ... 1904 bytes
    #fmsg            (again)             the same 1904 bytes
    #fparam get MSG_IMU  then 0.05 s
    #fmsg                                *#ERROR

Same command, and the only difference is that the last one was sent 0.05 s
behind the one before it.  ``#fmsg`` takes no argument and cannot be
refused for any other reason, so this is the console refusing a command
that arrived too soon -- and it is why a driver that reads a reply and
fires the next command straight away gets nonsense out of a module that
answers a human typing perfectly.

That is the whole cure for the fault this file was rewritten over.  The
console has no sequence numbers, so a reply that arrives after its command
was given up on lands inside the NEXT command's window, and from there
every command reads the previous one's answer -- which is what opened
Device Control empty.  But that fault needs a next command to land in.
Never having two in flight, and abandoning the session the moment one goes
unanswered, makes it unreachable, with no machinery at all.

It is also what the probe that works on this bench does, and the reason it
works: it waits after every command before sending another.

Two cleverer designs were tried first and both were wrong, so they are
written down rather than repeated.  Batching a question behind a second
question the module "must" answer, to put an edge on silence: the bench
answered ``*#ERROR\r\n*#ERROR`` to a pair sent back to back, so this
firmware does not take two lines the way that design assumed.  And asking
that question repeatedly while entering: the unread answers of earlier
attempts stayed in the module and satisfied a LATER command's terminator,
so ``#fmsg`` came back as the four bytes ``MSG_IMU=4`` with not one packet
line in it -- the desynchronisation, walked back in through the door built
to keep it out.

What this module answers, recorded off its wire and not read out of a
manual::

    > #fconfig                      *#OK        (the manual says "Config Mode")
    > #fparam get MSG_IMU           MSG_IMU=4
    > #fparam get FILT_LPF_ENABLED  FILT_LPF_ENABLED=0.000000
    > #fparam                       *#ERROR
    > #fmsg                         MSG_IMU[40]   10.0Hz
                                    MSG_AHRS[41]    0.0Hz
                                    ... one line per packet, about 1900 bytes,
                                    in batches with gaps between them
    > #fparam set MSG_IMU 7         *#OK        (written, NOT yet live)
    > #fsave                        *#OK
    > #freboot                      (y/n)  then y -> back in 2.5 s at 100 Hz
    > #fdeconfig                    (the binary stream resumes)

Every one of those has a reply, and every reply says which KIND of thing it
is: an echo ``NAME=value``, a listing line ``MSG_x[id] nHz``, the module's
yes ``*#OK``, its no ``*#ERROR``, or the restart's ``(y/n)``.  So each
command here names what it is waiting for and waits for that.  ``*#OK`` and
``*#ERROR`` identify no particular command on their own -- but with one
command in flight and the line cleared before it, nothing else could have
printed them.

``#fmsg`` is the one reply with no last line to recognise, so it is read
until the listing has begun and the module has then been quiet for a
moment.  That is a timing judgement, and it is sound HERE for the same
reason: nothing else is in flight to be mistaken for it.

Entering is judged by the module CEASING TO NAVIGATE -- the state change
itself, not the acknowledgement, which the handler prints before it, and
not a banner, which the manual and the module disagree about.  Then one
command proves the console is listening: an echo says so, and so does
``*#ERROR``, which only this console emits.

``MSG_IMU=4`` standing beside ``MSG_IMU[40] 10.0Hz`` is what says a rate is
stored as a LADDER INDEX -- rung 4 is 10 Hz -- which is why the manual's
``#fmsg 40 100`` is answered ``*#OK`` and changes nothing.  And a bare
``#fparam`` is an error, so parameters cannot be enumerated: the rates come
from ``#fmsg``, which is the module listing itself, and named parameters
have to be asked for one at a time.
"""

from __future__ import annotations

import re
import time

from zlc_atom.authoring import TuneRefused

#: The first byte of every FDILink frame -- what the module's stream looks
#: like, and therefore what "it is navigating again" looks like.
FRAME_HEAD = 0xFC

#: What the console appends to every command.  These are the literal bytes
#: the vendor's own FDILinkTool puts on the wire (``#fconfig\r\n``), not a
#: guess at the line ending.
LINE_END = "\r\n"

#: The module's yes and its no.  Neither says WHICH command it is about, so
#: neither is ever read from a window carrying more than one -- which,
#: under the rule above, is no window at all.
OK = "*#OK"
ERROR = "*#ERROR"

#: What ``#freboot`` waits to be answered ``y``.
CONFIRM_PROMPT = "(y/n)"

#: The packet every N100 sends, and the one this bench reads.
IMU_PACKET_NAME = "MSG_IMU"

#: How long to leave the module alone between commands.  See the module
#: docstring: 0.05 s is measurably too short and the bench answered every
#: command correctly at two seconds, so this sits inside that bracket with
#: the margin on the short side.  It is only ever the time SINCE THE LAST
#: REPLY, so a command that follows a long read -- ``#fmsg``'s 1904 bytes,
#: or a restart -- pays nothing for it.
SETTLE_BETWEEN_COMMANDS = 0.5

#: How long one command may take to answer.  Generous on purpose:
#: configuring is something an operator does now and then, never a hot
#: path, and this is only a deadline -- a command ends when its answer
#: arrives.  Running past it does not shorten a reply; it ends the session.
REPLY_TIMEOUT_SECONDS = 4.0

#: ``#fmsg`` prints some 1900 bytes with gaps between the batches and has
#: no last line to recognise, so it ends when the module has said nothing
#: for this long.  Measured on a real module: the whole listing was in hand
#: within a second, and a window narrower than the gaps cut it off mid-line.
LISTING_QUIET_SECONDS = 1.0
LISTING_TIMEOUT_SECONDS = 12.0

#: A frame header followed by a NAVIGATION packet type -- what "it is still
#: navigating" looks like on the line.  0xF0 is the module's 1 Hz
#: heartbeat, which it sends while it is NOT navigating, and counting that
#: as navigation had this driver reading the same two bytes in the opposite
#: direction from its own stream check.
_STREAM_MARKS = (b"\xfc\x40", b"\xfc\x41", b"\xfc\x42")

#: The FLOOR on how long the line must carry no navigation frame before the
#: module is taken to have stopped navigating.  Only a floor: a gap this
#: long is silence at 100 Hz and an ordinary pause between frames at 2 Hz,
#: so the window is scaled to the period actually being watched and this is
#: what it may not go below -- the drain time of the frames already in
#: flight when ``#fconfig`` landed.
ENTER_QUIET_SECONDS = 0.25

#: How many packet periods of nothing mean the stream has stopped rather
#: than hiccupped.  One period would be the gap itself; this is that with
#: room for the host delivering in clumps, which a USB serial adapter's
#: latency timer does as a matter of course.
ENTER_QUIET_PERIODS = 3.0

#: What to assume when nobody has measured the period.  The slowest rung on
#: this module's ladder is 1 Hz, so anything shorter would read that
#: module's ordinary gaps as silence.  The caller that has measured the
#: stream passes the real figure and pays nothing for this.
SLOWEST_RUNG_SECONDS = 1.0

#: ``MSG_IMU=4`` from ``#fparam get``, and ``imu_algn_yaw = 0.000000`` from
#: ``#faxis``: the same shape with and without spaces, which is why the
#: spaces are optional here rather than assumed to be there.
_PARAM_ECHO = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*) *= *(?P<value>[-+]?[0-9]+(?:\.[0-9]+)?)"
)

#: ``MSG_IMU[40]   10.0Hz`` from ``#fmsg``: the module enumerating itself,
#: one line per packet, with the rate in hertz.  No ``=`` in it, so a
#: listing and a parameter echo cannot be mistaken for one another.
_PACKET_LINE = re.compile(
    r"(?P<name>MSG_[A-Z0-9_]+)\[(?P<id>[0-9A-Fa-f]{1,2})\] *"
    r"(?P<hz>[0-9]+(?:\.[0-9]+)?)Hz",
    re.IGNORECASE,
)


def _as_text(data: bytes) -> str:
    """The transcript so far.  A frame's bytes are not ASCII and stay noise."""

    return data.decode("ascii", "replace")


def parameters_in(transcript: str) -> dict[str, str]:
    """Every ``NAME=value`` the module printed, by name."""

    return {
        match["name"].upper(): match["value"]
        for match in _PARAM_ECHO.finditer(transcript)
    }


def packets_in(transcript: str) -> tuple[tuple[str, int, float], ...]:
    """Every ``MSG_x[id] nHz`` the module printed, as ``(name, id, hertz)``."""

    return tuple(
        (match["name"], int(match["id"], 16), float(match["hz"]))
        for match in _PACKET_LINE.finditer(transcript)
    )


def _said_yes_or_no(transcript: str) -> bool:
    """The module answered the one command it was given."""

    return OK in transcript or ERROR in transcript


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

    A command that goes unanswered raises, and the console is finished
    after that: the caller may close it and must not ask it anything else.
    See the module docstring -- that is what makes losing step impossible
    rather than merely unlikely.
    """

    def __init__(
        self,
        port,
        *,
        reply_timeout: float = REPLY_TIMEOUT_SECONDS,
        packet_interval: float | None = None,
    ) -> None:
        self._port = port
        self._reply_timeout = float(reply_timeout)
        #: How long the line must be free of navigation frames to count as
        #: not navigating.  Scaled to the stream being watched, because the
        #: same 250 ms is silence at 100 Hz and the ordinary gap between
        #: frames at 2 Hz -- and a module on the ladder's slow rungs would
        #: otherwise be declared stopped while it was still sending.
        period = (
            float(packet_interval)
            if packet_interval and packet_interval > 0.0
            else SLOWEST_RUNG_SECONDS
        )
        self._enter_quiet = max(ENTER_QUIET_SECONDS, ENTER_QUIET_PERIODS * period)
        self._entered = False
        #: When this console last finished listening.  The next command
        #: waits out ``SETTLE_BETWEEN_COMMANDS`` from here, which is what
        #: keeps it from arriving while the module is still busy with the
        #: one before -- the state the bench answers ``*#ERROR`` from.
        self._listened_until = 0.0
        #: The last command and its transcript, verbatim.  Kept because
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

        Two things have to be true, and both are things the module DID.

        It has to have stopped navigating.  That is the state change
        itself, so it is what is waited for -- not the acknowledgement,
        which the handler prints before the change completes, and not a
        banner, which the manual and the module disagree about.

        And it has to be listening, which one command proves.  EITHER
        answer proves it: the echo says the parameter was read, and
        ``*#ERROR`` says the console parsed a line and refused it.  Only
        this console prints either, so either one is config mode.
        """

        if self._entered:
            return
        self._port.reset_input_buffer()
        self._write("#fconfig")
        if not self._navigation_stopped():
            raise RuntimeError(
                "the module on this port kept streaming through #fconfig, so "
                "it never entered config mode"
            )
        self._entered = True
        opening = f"#fparam get {IMU_PACKET_NAME}"
        transcript = ""
        for _attempt in (1, 2):
            transcript, answered = self._exchange(
                opening,
                # The echo, or the module's word for no -- NOT its word for
                # yes.  A get does not draw *#OK from this module, and the
                # one *#OK nobody here has claimed is #fconfig's own: this
                # command's window is exactly where a slow one lands, and
                # accepting it would leave the real answer in flight for
                # #fmsg to trip over.
                lambda heard: IMU_PACKET_NAME in parameters_in(heard)
                or ERROR in heard,
            )
            if answered:
                return
            # The one place a second go is allowed, and only because the
            # line is emptied first: whatever the module still owed has
            # either arrived and been thrown away, or is not coming.  This
            # is also the one place where a command CAN go missing rather
            # than unanswered -- nobody knows how long after #fconfig the
            # module starts reading again, and the probe that works on this
            # bench never found out, because it waited 1.2 s after every
            # command and never asked the question.
            self._drain()
        self.close()
        raise RuntimeError(
            "the module on this port stopped streaming for #fconfig but then "
            f"would not answer {opening!r}, so it never entered config mode; "
            f"it said {transcript.strip()[:160]!r}"
        )

    def _drain(self) -> None:
        """Read until the module has stopped saying anything at all.

        Only entering uses this, and only between its two attempts.  What
        it throws away is whatever an attempt that timed out may still have
        been owed, which is what makes asking again clean rather than the
        move that loses step.
        """

        deadline = time.monotonic() + self._reply_timeout
        last = time.monotonic()
        while time.monotonic() < deadline:
            waiting = getattr(self._port, "in_waiting", 0)
            chunk = self._port.read(waiting if waiting else 1)
            now = time.monotonic()
            if chunk:
                last = now
            elif now - last >= ENTER_QUIET_SECONDS:
                break
        self._listened_until = time.monotonic()

    def close(self) -> None:
        """Put the module back on the air, whatever happened in between."""

        if not self._entered:
            return
        try:
            self._port.reset_input_buffer()
            self._write("#fdeconfig")
            # The one command the module answers by going back ON the air,
            # so what identifies its reply is a frame header -- a byte the
            # console never prints.
            self._read_until(
                lambda data: bytes((FRAME_HEAD,)) in data,
                deadline=time.monotonic() + self._reply_timeout,
            )
        finally:
            self._entered = False

    # --------------------------------------------------------------- link
    def ask(self, command: str, is_the_answer, *, timeout: float | None = None) -> str:
        """Send ONE command and read until the module has answered IT.

        ``is_the_answer`` is handed the transcript so far and says whether
        the reply this command was waiting for is in it.  Nothing else is
        sent until it is, and if the deadline passes first this raises and
        the console is done: a reply that is merely late would otherwise
        arrive inside the next command's window, and from there every
        command reads the one before it.
        """

        self._require_console(command)
        transcript, answered = self._exchange(command, is_the_answer, timeout=timeout)
        if not answered:
            raise RuntimeError(
                f"the module did not answer {command!r} within "
                f"{timeout or self._reply_timeout:g} s; a reply arriving after "
                "this would be read as the next command's, so this settings "
                f"session is over. It said {transcript.strip()[:160]!r}"
            )
        return transcript

    def write(self, command: str) -> None:
        """Send one command and do not wait for what it says back."""

        self._require_console(command)
        self._write(command)

    def query(self, command: str) -> str:
        """Send one command and answer with the transcript of its reply.

        The ``ScpiLink`` shape.  An arbitrary command's reply cannot be
        recognised by kind, so this waits for the module's yes or no, which
        is what every command here that is not a question answers with.
        """

        return self.ask(command, _said_yes_or_no)

    def _require_console(self, command: str) -> None:
        if not self._entered:
            raise RuntimeError(
                f"{command!r} is a config-mode command and this console is not "
                "in config mode"
            )

    # ---------------------------------------------------------- the module
    def get_parameter(self, name: str) -> str | None:
        """One named parameter's value, or None when this firmware lacks it.

        A module that has it echoes it, ``MSG_IMU=4``; a module that has not
        got it says ``*#ERROR``, which is its word for no.  Both are
        answers, so neither is inferred from how long anything took.

        Those two, and nothing else.  A ``get`` does not draw ``*#OK`` from
        this module, so reading one as "not present" would score a knob off
        the panel on the strength of a reply that answers a different
        question -- which is how the settings page emptied the first time.
        Anything else is no answer, and no answer ends the session.
        """

        wanted = str(name).upper()
        transcript = self.ask(
            f"#fparam get {name}",
            lambda heard: wanted in parameters_in(heard) or ERROR in heard,
        )
        return parameters_in(transcript).get(wanted)

    def packet_rates(self) -> tuple[tuple[str, int, float], ...]:
        """Every packet this module has, as ``(name, id, hertz)``.

        The module enumerating itself, and the only readback there is for a
        rate -- the parameter holding one reads back as the ladder index
        that was written to it, not as hertz.

        This is the one reply with no last line to recognise: it arrives in
        batches with gaps between them, so it is read until the listing has
        begun and the module has then been quiet for a moment.  That is a
        timing judgement, and it is sound here only because nothing else is
        in flight to be mistaken for it.
        """

        listed: tuple[tuple[str, int, float], ...] = ()
        silent_for = [0.0]

        def whole_listing(heard: str) -> bool:
            nonlocal listed
            found = packets_in(heard)
            if found:
                listed = found
            return bool(found) and silent_for[0] >= LISTING_QUIET_SECONDS

        transcript, complete = self._exchange(
            "#fmsg",
            whole_listing,
            timeout=LISTING_TIMEOUT_SECONDS,
            silent_for=silent_for,
        )
        if not complete:
            raise RuntimeError(
                "the module never finished listing its packets; it said "
                f"{transcript.strip()[:160]!r}"
            )
        if not any(name.upper() == IMU_PACKET_NAME for name, _id, _hz in listed):
            raise RuntimeError(
                f"the module listed its packets but not {IMU_PACKET_NAME}: it "
                f"said {transcript.strip()[:160]!r}"
            )
        return listed

    def set_parameter(self, name: str, value: str) -> str:
        """Write one named parameter and answer with what it reads back as.

        Two commands, one after the other, because that is the only way
        anything is sent here.  The write is refused on the module's own
        refusal word; the readback is what this returns.

        What comes back is the PARAMETER TABLE's value -- not what the
        module is running.  Nothing takes effect until ``save`` and
        ``reboot``.
        """

        written = self.ask(f"#fparam set {name} {value}", _said_yes_or_no)
        if ERROR in written:
            raise TuneRefused(
                f"the module refused {name}={value}: {written.strip()[:120]!r}"
            )
        reading = self.get_parameter(name)
        if reading is None:
            raise TuneRefused(
                f"this module has no parameter {name!r}: it would not read "
                "the name back after the write"
            )
        return reading

    def save(self) -> str:
        """Commit to flash, and refuse only on the module's own refusal word.

        Nothing reports what is in flash, so a save cannot be CONFIRMED the
        way a written setting can, and this does not try to.  But
        ``*#ERROR`` is this console's word for no, and a save that was
        refused must not be reported as done: everything after it is built
        on the value having reached flash.
        """

        transcript = self.ask("#fsave", _said_yes_or_no)
        if ERROR in transcript:
            raise TuneRefused(
                f"the module refused to save: {transcript.strip()[:120]!r}"
            )
        return transcript

    def reboot(self) -> None:
        """Restart the module, which is what makes a saved setting live.

        The command needs confirming with ``y``, and the prompt is what
        identifies its reply.  No prompt means the module is not waiting
        for a yes, and sending one anyway puts a bare ``y`` on the wire for
        it to read as a command -- so that is an error, and the caller's
        exit takes the module out of the console properly.

        The module is restarting when this returns, so the console is over.
        """

        self.ask("#freboot", lambda heard: CONFIRM_PROMPT in heard)
        self._write("y")
        self._entered = False

    # -------------------------------------------------------------- lines
    def _write(self, text: str) -> None:
        """Put one line on the wire, no sooner than the module will take it."""

        waited = time.monotonic() - self._listened_until
        if waited < SETTLE_BETWEEN_COMMANDS:
            time.sleep(SETTLE_BETWEEN_COMMANDS - waited)
        self._port.write((text + LINE_END).encode("ascii"))
        flush = getattr(self._port, "flush", None)
        if callable(flush):
            flush()

    def _exchange(
        self,
        command: str,
        is_the_answer,
        *,
        timeout: float | None = None,
        silent_for: list[float] | None = None,
    ) -> tuple[str, bool]:
        """Clear the line, send ONE command, and read until it is answered.

        Clearing first matters only for the frames draining out of a module
        being taken off the air: under this file's rule nothing is ever
        owed from an earlier command, because an unanswered one ends the
        session.
        """

        self._port.reset_input_buffer()
        self._write(command)
        transcript, answered = self._read_until(
            lambda data: is_the_answer(_as_text(data)),
            deadline=time.monotonic()
            + (self._reply_timeout if timeout is None else timeout),
            silent_for=silent_for,
        )
        self._listened_until = time.monotonic()
        self.last_exchange = (command, transcript)
        return transcript, answered

    def _read_until(
        self, is_the_answer, *, deadline: float, silent_for: list[float] | None = None
    ) -> tuple[str, bool]:
        """Read until the answer is recognisable, and say whether it was.

        ``silent_for``, when given, is a one-slot box kept filled with how
        long the module has been quiet -- for the one reply that ends in
        nothing rather than in something.
        """

        heard = bytearray()
        last = time.monotonic()
        while time.monotonic() < deadline:
            waiting = getattr(self._port, "in_waiting", 0)
            chunk = self._port.read(waiting if waiting else 1)
            now = time.monotonic()
            if chunk:
                heard += chunk
                last = now
            if silent_for is not None:
                silent_for[0] = now - last
            if (chunk or silent_for is not None) and is_the_answer(bytes(heard)):
                return _as_text(heard), True
        return _as_text(heard), False

    def _navigation_stopped(self) -> bool:
        """Wait until no navigation frame has arrived for a moment.

        Two things end this: the frames stopping, which is the state
        change, and the acknowledgement arriving, which is the reply to
        ``#fconfig`` and has to be READ rather than left for the next
        command's window to find.  Leaving it there is how the door itself
        put the session one command out of step.

        The frames stopping is the one that decides.  A module that stops
        navigating and never prints goes on, because entering is about what
        the module did, not about what it said -- the next command is what
        proves the console is listening.
        """

        now = time.monotonic()
        deadline = now + self._reply_timeout
        quiet_since = now
        carry = b""
        heard = bytearray()
        while time.monotonic() < deadline:
            waiting = getattr(self._port, "in_waiting", 0)
            chunk = self._port.read(waiting if waiting else 1)
            now = time.monotonic()
            if chunk:
                heard += chunk
                window = carry + chunk
                if any(mark in window for mark in _STREAM_MARKS):
                    quiet_since = now
                # A mark split across two reads is still a mark.
                carry = window[-1:]
            if now - quiet_since < self._enter_quiet:
                continue
            # The stream has stopped.  Wait for the acknowledgement too, so
            # that it is CONSUMED here: #fconfig is a command like any
            # other, and the one reply this file used to leave unclaimed is
            # the one that goes on to end somebody else's.
            if _said_yes_or_no(_as_text(heard)):
                self._listened_until = now
                return True
        # It stopped navigating and never said so.  Entering is about what
        # the module DID, and it did stop -- so this goes on, and the
        # command that follows is what proves the console is listening.
        self._listened_until = now
        return now - quiet_since >= self._enter_quiet


__all__ = [
    "CONFIRM_PROMPT",
    "ENTER_QUIET_PERIODS",
    "ENTER_QUIET_SECONDS",
    "ERROR",
    "FRAME_HEAD",
    "FdiConfigConsole",
    "IMU_PACKET_NAME",
    "LINE_END",
    "LISTING_QUIET_SECONDS",
    "LISTING_TIMEOUT_SECONDS",
    "OK",
    "REPLY_TIMEOUT_SECONDS",
    "SETTLE_BETWEEN_COMMANDS",
    "SLOWEST_RUNG_SECONDS",
    "packets_in",
    "parameters_in",
]
