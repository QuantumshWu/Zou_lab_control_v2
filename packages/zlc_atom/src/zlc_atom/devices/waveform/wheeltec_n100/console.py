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

TWO RULES, and they are the whole file.

**A question is bounded by a question the module must answer.**  This
firmware says NOTHING for a parameter it has not got -- not ``*#ERROR``,
nothing -- and the console carries no sequence numbers, so silence and
"not yet" are the same bytes.  A console that reads by timing has to guess
between them, and one wrong guess desynchronises everything after it: the
late reply lands in the next command's window, and from there every
command reads the previous one's answer.  That is what put an empty page
in Device Control.  So every question ends with ``#fparam get MSG_IMU``:
``MSG_IMU`` is the packet every N100 has -- discovery finds the module BY
those frames -- so the module always answers it, and its answer arriving
means everything asked in front of it has been answered or was never going
to be.  One read, one transcript, one parse; absence is then a fact about
the transcript rather than a guess about the clock.

**A command that changes the module's state stands alone.**  A question is
a pure read: the firmware parses a line and prints, so a second question
may follow it down the wire immediately.  ``#fconfig``, ``#fsave`` and
``#freboot`` are not that -- they switch modes and write flash -- and each
is sent by itself and waited for on its own reply.

``#fconfig`` is the sharpest case of the second rule.  It is answered
``*#OK``, but that is a string the handler prints, not the state change --
the manual prints a different one for the same command.  What IS the state
change is the module CEASING TO NAVIGATE, so that is what entering waits
for: no FDILink navigation frame on the line for a moment.

Entering must also not ASK anything, and that is not a matter of taste.  A
question whose answer nobody reads stays in the module, arrives during
some later exchange, and satisfies that exchange's terminator before its
own commands have said a word -- the desynchronisation this file exists to
prevent, walked back in through the entry door.  Reading the line until
the frames stop is the one way in that leaves nothing owed.  So the
invariant is exact:

    at most one question is ever outstanding, and every exchange consumes
    its own answer.

``ask`` reads until the certain answer, ``alone`` until the
acknowledgement, ``reboot`` until the prompt.  None of them returns
leaving a reply owed, and entering never creates one.

What this module answers, recorded off its wire and not read out of a
manual::

    > #fconfig                      *#OK        (the manual says "Config Mode")
    > #fparam get MSG_IMU           MSG_IMU=4
    > #fparam get FILT_LPF_ENABLED  FILT_LPF_ENABLED=0.000000
    > #fparam                       *#ERROR
    > #fmsg                         MSG_IMU[40]   10.0Hz
                                    MSG_AHRS[41]    0.0Hz
                                    ... one line per packet, about 1900 bytes
    > #fparam set MSG_IMU 7         *#OK        (written, NOT yet live)
    > #fsave                        *#OK
    > #freboot                      (y/n)  then y -> back in 2.5 s at 100 Hz
    > #fdeconfig                    (the binary stream resumes)

Read off that.  ``#fmsg`` with no argument is how the module enumerates
itself, and it gives each packet's rate in HERTZ -- so the packet list is
the module's own and nothing here decides what packets exist.  A bare
``#fparam`` is an error, so parameters cannot be enumerated and must be
asked for by name, which is the whole reason the certain question is
needed.  And ``MSG_IMU=4`` standing beside ``MSG_IMU[40] 10.0Hz`` is what
says a rate is stored as a LADDER INDEX -- rung 4 is 10 Hz -- which is why
the manual's ``#fmsg 40 100`` is answered ``*#OK`` and changes nothing.

Nothing here is judged by a banner: the manual prints ``Config Mode`` as
the reply to ``#fconfig`` and a real module answers ``*#OK``.
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

#: The two words the module says about a command it has just run.  They are
#: only ever read from an exchange carrying ONE command, where nothing else
#: could have printed them -- which is what keeps them from being the kind
#: of banner this driver has been wrong about before.
OK = "*#OK"
ERROR = "*#ERROR"

#: What ``#freboot`` waits to be answered ``y``.
CONFIRM_PROMPT = "(y/n)"

#: The packet every N100 sends, and the one this bench reads.
IMU_PACKET_NAME = "MSG_IMU"

#: The question this module is certain to answer, and therefore the end of
#: every exchange that asks anything.  See the module docstring: this one
#: line is why nothing here reasons about how long a reply took.
CERTAIN_QUESTION = f"#fparam get {IMU_PACKET_NAME}"

#: How long one exchange may take.  Generous on purpose: configuring is
#: something an operator does now and then, never a hot path, and this is
#: only a deadline -- an exchange ends when its answer arrives, so a large
#: number costs nothing in the ordinary case.  The longest reply this
#: module has, ``#fmsg``'s 1904 bytes, was in hand in under a second.
REPLY_TIMEOUT_SECONDS = 4.0

#: A frame header followed by a NAVIGATION packet type -- what "it is
#: still navigating" looks like on the line.  0xF0 is the module's 1 Hz
#: heartbeat, which it sends while it is NOT navigating, and counting that
#: as navigation had this driver reading the same two bytes in the opposite
#: direction from its own stream check.
_STREAM_MARKS = (b"\xfc\x40", b"\xfc\x41", b"\xfc\x42")

#: The FLOOR on how long the line must carry no navigation frame before
#: the module is taken to have stopped navigating.  It is only a floor: a
#: gap this long is silence at 100 Hz and an ordinary pause between frames
#: at 2 Hz, so the window is scaled to the period actually being watched
#: and this is what it may not go below -- the drain time of the frames
#: already in flight when ``#fconfig`` landed.
ENTER_QUIET_SECONDS = 0.25

#: How many packet periods of nothing mean the stream has stopped rather
#: than hiccupped.  One period would be the gap itself; this is that with
#: room for the host delivering in clumps, which a USB serial adapter's
#: latency timer does as a matter of course.
ENTER_QUIET_PERIODS = 3.0

#: What to assume when nobody has measured the period.  The slowest rung
#: on this module's ladder is 1 Hz, so anything shorter than its period
#: would read that module's ordinary gaps as silence.  The caller that has
#: measured the stream passes the real figure and pays nothing for this.
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


def _certainly_answered(data: bytes) -> bool:
    """The certain question has been answered, so the exchange is complete."""

    return IMU_PACKET_NAME in parameters_in(_as_text(data))


def _acknowledged(data: bytes) -> bool:
    """The module said something about the one command it was given."""

    text = _as_text(data)
    return OK in text or ERROR in text


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
        #: The last exchange and its transcript, verbatim.  Kept because
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

        It has to have stopped navigating -- the state change itself, not
        the acknowledgement, which is only a string the handler prints, and
        not a banner, which the manual and the module disagree about.
        Waiting for it is also the only way in that leaves nothing behind:
        a question asked here and not read would arrive during a later
        exchange and end it before its own commands had spoken.

        And it has to be answering, which the first exchange proves, and
        consumes its own answer proving it.

        That exchange gets a second go, and the condition on it is exact.
        Only two things in this console are ever OWED to a question: a
        ``NAME=value`` echo and a ``MSG_x[id] nHz`` listing.  If neither is
        in what came back, no answer is in flight, the question was simply
        never taken, and asking again cannot collide with anything.  If one
        of them IS there, a reply is in flight and a second question is
        exactly how this console loses step -- so that case fails instead.

        ``#fconfig``'s own ``*#OK`` is why this matters.  It is the one
        acknowledgement nothing waits for -- entering is judged by the
        stream, not by it -- so on a module that prints it slowly it lands
        in this exchange.  It is owed to no question, so it does not stop
        the second go.
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
        transcript = ""
        for _attempt in (1, 2):
            transcript, answered = self._exchange(
                (CERTAIN_QUESTION,),
                is_the_answer=_certainly_answered,
                deadline=time.monotonic() + self._reply_timeout,
            )
            self.last_exchange = (f"#fconfig ; {CERTAIN_QUESTION}", transcript)
            if answered:
                return
            if parameters_in(transcript) or packets_in(transcript):
                break
        self.close()
        raise RuntimeError(
            "the module on this port stopped streaming for #fconfig but would "
            f"not answer {CERTAIN_QUESTION!r}, so it never entered config "
            f"mode; it said {transcript.strip()[:160]!r}"
        )

    def _navigation_stopped(self) -> bool:
        """Wait until no navigation frame has arrived for a moment.

        Everything read here is thrown away: the acknowledgement, the
        frames still draining out, whatever a module left in an earlier
        session had to say.  None of it is a judgement and none of it is
        owed to anybody, which is what makes this a safe way in.
        """

        deadline = time.monotonic() + self._reply_timeout
        quiet_since = time.monotonic()
        carry = b""
        while time.monotonic() < deadline:
            waiting = getattr(self._port, "in_waiting", 0)
            chunk = self._port.read(waiting if waiting else 1)
            now = time.monotonic()
            if chunk:
                window = carry + chunk
                if any(mark in window for mark in _STREAM_MARKS):
                    quiet_since = now
                # A mark split across two reads is still a mark.
                carry = window[-1:]
            if now - quiet_since >= self._enter_quiet:
                return True
        return False

    def close(self) -> None:
        """Put the module back on the air, whatever happened in between."""

        if not self._entered:
            return
        try:
            self._port.reset_input_buffer()
            self._write("#fdeconfig")
            # The certain question is no use here: the module answers this
            # one by going back ON the air, so what identifies the answer is
            # a frame header -- a byte the console never prints.
            self._read_until(
                lambda data: bytes((FRAME_HEAD,)) in data,
                deadline=time.monotonic() + self._reply_timeout,
            )
        finally:
            self._entered = False

    # --------------------------------------------------------------- link
    def ask(self, *commands: str) -> str:
        """Ask the module some questions and read ONE transcript of them all.

        The certain question goes last, and the reading stops when its
        answer arrives -- by which time every question in front of it has
        either been answered or was never going to be.  So a name missing
        from the transcript is a name this firmware has not got, which is a
        fact about what the module said rather than a guess about how long
        it took to not say it.

        QUESTIONS ONLY.  These are pure reads -- the firmware parses a line
        and prints -- so they may follow one another down the wire without
        waiting.  A command that changes the module's state may not; see
        ``alone``.
        """

        asked = (*commands, CERTAIN_QUESTION)
        for command in asked:
            self._require_console(command)
        transcript, answered = self._exchange(
            asked,
            is_the_answer=_certainly_answered,
            deadline=time.monotonic() + self._reply_timeout,
        )
        self.last_exchange = (" ; ".join(asked), transcript)
        if not answered:
            raise RuntimeError(
                f"the module did not answer {CERTAIN_QUESTION!r} within "
                f"{self._reply_timeout:g} s, so nothing it did say can be "
                f"matched to what was asked; it said {transcript.strip()[:160]!r}"
            )
        return transcript

    def alone(self, command: str) -> str:
        """Send ONE state-changing command and read its own acknowledgement.

        ``#fsave`` writes flash and ``#fparam set`` moves a value: neither
        is a question, and neither may share an exchange with one.  Their
        reply is ``*#OK`` or ``*#ERROR``, which identifies nothing by
        itself -- but this exchange carries one command and starts from a
        cleared line, so nothing else could have printed it.  That is the
        whole reason state-changing commands are sent alone.
        """

        self._require_console(command)
        transcript, answered = self._exchange(
            (command,),
            is_the_answer=_acknowledged,
            deadline=time.monotonic() + self._reply_timeout,
        )
        self.last_exchange = (command, transcript)
        if not answered:
            raise RuntimeError(
                f"the module did not acknowledge {command!r} within "
                f"{self._reply_timeout:g} s; it said {transcript.strip()[:160]!r}"
            )
        return transcript

    def write(self, command: str) -> None:
        """Send one command and do not wait for what it says back."""

        self._require_console(command)
        self._write(command)

    def query(self, command: str) -> str:
        """Send one command and answer with the transcript of its reply."""

        return self.ask(command)

    def _require_console(self, command: str) -> None:
        if not self._entered:
            raise RuntimeError(
                f"{command!r} is a config-mode command and this console is not "
                "in config mode"
            )

    # ---------------------------------------------------------- the module
    def get_parameter(self, name: str) -> str | None:
        """One named parameter's value, or None when this firmware lacks it.

        A module that has the parameter echoes it; a module that has not got
        it says nothing.  The certain question bounds that silence, so None
        here means the module was given its chance to answer and did not.
        """

        return parameters_in(self.ask(f"#fparam get {name}")).get(str(name).upper())

    def packet_rates(self) -> tuple[tuple[str, int, float], ...]:
        """Every packet this module has, as ``(name, id, hertz)``.

        The module enumerating itself, and the only readback there is for a
        rate -- the parameter holding one reads back as the ladder index
        that was written to it, not as hertz.

        The listing arrives in batches on a real module and there is nothing
        at the end of it to say it is over.  The certain question is that
        end: its answer is printed after the last packet line, so a
        transcript containing it contains the whole listing.
        """

        listed = packets_in(self.ask("#fmsg"))
        if not any(name.upper() == IMU_PACKET_NAME for name, _id, _hz in listed):
            raise RuntimeError(
                "the module answered but listed no packets, not even "
                f"{IMU_PACKET_NAME}: it said {self.last_exchange[1].strip()[:160]!r}"
            )
        return listed

    def set_parameter(self, name: str, value: str) -> str:
        """Write one named parameter and answer with what it reads back as.

        Two exchanges, because a write is not a question: the write goes
        alone and is refused on the module's own refusal word, and the
        readback is asked for afterwards.  A parameter this firmware does
        not have reads back as nothing, which is a refusal rather than a
        silent no-op.

        What comes back is the PARAMETER TABLE's value -- not what the
        module is running.  Nothing takes effect until ``save`` and
        ``reboot``.
        """

        written = self.alone(f"#fparam set {name} {value}")
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

        It goes alone.  Writing flash is the longest thing this firmware
        does, and a question sent behind it would be arriving while the
        module is busy doing it.
        """

        transcript = self.alone("#fsave")
        if ERROR in transcript:
            raise TuneRefused(f"the module refused to save: {transcript.strip()[:120]!r}")
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

        self._require_console("#freboot")
        transcript, prompted = self._exchange(
            ("#freboot",),
            is_the_answer=lambda data: CONFIRM_PROMPT in _as_text(data),
            deadline=time.monotonic() + self._reply_timeout,
        )
        self.last_exchange = ("#freboot", transcript)
        if not prompted:
            raise RuntimeError(
                "the module did not ask to confirm the restart; it answered "
                f"{transcript.strip()[:120]!r}"
            )
        self._write("y")
        self._entered = False

    # -------------------------------------------------------------- lines
    def _write(self, text: str) -> None:
        self._port.write((text + LINE_END).encode("ascii"))
        flush = getattr(self._port, "flush", None)
        if callable(flush):
            flush()

    def _exchange(
        self, commands: tuple[str, ...], *, is_the_answer, deadline: float
    ) -> tuple[str, bool]:
        """Clear the line, send these commands, and read until they answer.

        Clearing first is not load-bearing -- every exchange ends on its own
        answer, so nothing is left behind for the next one to trip over --
        it just keeps a transcript from carrying the previous exchange's
        tail, and the frames that drain out of a module being taken off the
        air.
        """

        self._port.reset_input_buffer()
        for command in commands:
            self._write(command)
        return self._read_until(is_the_answer, deadline=deadline)

    def _read_until(self, is_the_answer, *, deadline: float) -> tuple[str, bool]:
        """Read until the answer is recognisable, and say whether it was.

        There is no quiet window and no end marker: what ends a read is
        seeing the thing that was waited for.  Running out of time is a
        failure with the transcript attached, never a reply treated as
        complete.
        """

        heard = bytearray()
        while time.monotonic() < deadline:
            waiting = getattr(self._port, "in_waiting", 0)
            chunk = self._port.read(waiting if waiting else 1)
            if not chunk:
                continue
            heard += chunk
            if is_the_answer(bytes(heard)):
                return _as_text(heard), True
        return _as_text(heard), False


__all__ = [
    "CERTAIN_QUESTION",
    "CONFIRM_PROMPT",
    "ENTER_QUIET_PERIODS",
    "ENTER_QUIET_SECONDS",
    "ERROR",
    "FRAME_HEAD",
    "FdiConfigConsole",
    "IMU_PACKET_NAME",
    "LINE_END",
    "OK",
    "REPLY_TIMEOUT_SECONDS",
    "SLOWEST_RUNG_SECONDS",
    "packets_in",
    "parameters_in",
]
