"""The module's own configuration console, over the same serial line.

An N100 normally streams binary FDILink frames and listens to nothing.  It
also carries an ASCII command console, documented in chapter 5 of FDI's
《通信协议》: send ``#fconfig``, the module STOPS navigating and stops
emitting frames, every ``#f`` command is answered in plain text, and
``#fdeconfig`` puts it back on the air.

THREE RULES, and the rest of this file follows from them.

ONE COMMAND AT A TIME, AND READ ITS WHOLE REPLY.  There are no sequence
numbers here and no end-of-reply marker, so a reply belongs to a command
only because nothing else was in flight.  Stopping a read at the first
thing that looks like an answer leaves the other half of it to turn up in
the next command's window, and from there every command reads the one
before it.

THE GAP BELONGS TO THE NEXT COMMAND.  The module refuses a command that
arrives too soon after the last one -- see ``SPACING_SECONDS``, which
carries the measurement.  Waiting is therefore charged before the next line
goes out, not after a reply is in hand: the last command of a session pays
nothing, and a caller that asks one question is not held for a gap nobody
will use.

A REPLY IS COLLECTED, NOT RECOGNISED.  The window is read to its end and
then parsed in one go.  ``NAME=value`` is a parameter echo; ``*#OK`` and
``*#ERROR`` are the module's yes and no, and with one command per window
nothing else can have printed them.

WHAT THIS MODULE ANSWERS, recorded off its own wire and not read out of a
manual::

    > #fconfig                      (frames drain, then) *#OK
    > #fparam get MSG_IMU           MSG_IMU=7
    > #fparam get FILT_LPF_ENABLED  FILT_LPF_ENABLED=0.000000
    > #fparam                       *#ERROR       (the bare form; no arguments)
    > #fmsg                         MSG_IMU[40]  100.0Hz ... one line per packet
    > #fparam set MSG_IMU 7         *#OK          (written, NOT yet live)
    > #fsave                        *#OK
    > #freboot                      (y/n)  then y -> back in about 2.5 s
    > #fdeconfig                    (the binary stream resumes)

``MSG_IMU=7`` standing beside ``#fmsg``'s ``MSG_IMU[40] 100.0Hz`` is what
says a rate is stored as a LADDER INDEX -- rung 7 is 100 Hz -- which is why
the manual's ``#fmsg 40 100`` is answered ``*#OK`` and changes nothing.  A
bare ``#fparam`` is an error, so parameters cannot be enumerated: every one
is asked for by name.  ``#fmsg`` is listed above because it is part of what
the module was seen to answer; this driver turns one packet and reads that
one by name, and does not send it.
"""

from __future__ import annotations

import re
import time

from zlc_atom.authoring import TuneRefused

#: What the console appends to every command.  These are the literal bytes
#: the vendor's own FDILinkTool puts on the wire (``#fconfig\r\n``).
LINE_END = "\r\n"

#: The module's yes and its no.
OK = "*#OK"
ERROR = "*#ERROR"

#: What ``#freboot`` waits to be answered ``y``.
CONFIRM_PROMPT = "(y/n)"

#: The packet every N100 sends, the one this bench reads, and the name of
#: the parameter that holds its rate -- one string for one thing, which is
#: why ``source.py`` imports it from here rather than spelling it again.
IMU_PACKET_NAME = "MSG_IMU"

#: How far apart two commands must go out, measured from the previous
#: WRITE -- which is the reference the bench confirmed, not a guess.
#:
#: Swept on the real module, write to write, five trials a step::
#:
#:     0.70 s  0/5      1.00 s  5/5
#:     0.80 s  0/5      1.10 s  5/5
#:     0.90 s  0/5      1.20 s  5/5
#:                      1.30 s  5/5
#:
#: and 20/20 at a measured 1.01 s against 0/15 below 0.90 s: a hard
#: boundary in (0.90, 1.00], not a probability.  1.2 s is a fifth above the
#: top of that bracket, which is the right shape of margin for a sharp edge.
#:
#: Two things that sweep also settled.  Reading the line during the gap
#: makes no difference (0.30 s failed both ways, 1.00 s worked both ways) --
#: the module wants TIME, not attention.  And a heavy first command changes
#: nothing: ``#fmsg`` takes about a second to answer and the next command
#: still went through at 1.00 s write-to-write, three for three.  So the
#: module counts from being SPOKEN TO, not from finishing, which is why
#: ``_write`` measures from its own last write and why ``#fsave`` -- the
#: slowest thing here -- needs no allowance of its own.
SPACING_SECONDS = 1.2

#: A reply is over when the module has said nothing for this long.
REPLY_QUIET_SECONDS = 0.25

#: And a module that says nothing at all within this is not answering.
REPLY_TIMEOUT_SECONDS = 3.0

#: A frame header followed by a NAVIGATION packet type -- what "it is still
#: navigating" looks like.  0xF0 is the 1 Hz heartbeat, which the module
#: sends while it is NOT navigating, so counting it here would have this
#: driver reading the same two bytes in the opposite direction from the
#: stream check in ``source.py``.
_STREAM_MARKS = (b"\xfc\x40", b"\xfc\x41", b"\xfc\x42")

#: How long to watch for frames after the reply to ``#fconfig``.  By then
#: the frames in flight have long drained, so anything still arriving is a
#: module that never left the air.
STILL_NAVIGATING_SECONDS = 0.5

#: ``MSG_IMU=7`` from ``#fparam get``, and ``imu_algn_yaw = 0.000000`` from
#: ``#faxis``: the same shape with and without spaces, which is why the
#: spaces are optional here rather than assumed to be there.
_PARAM_ECHO = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*) *= *(?P<value>[-+]?[0-9]+(?:\.[0-9]+)?)"
)


def _words(data: bytes) -> str:
    """The window as words.  A frame's bytes are not ASCII and stay noise."""

    return data.decode("ascii", "replace")


def _parameters_in(transcript: str) -> dict[str, str]:
    """Every ``NAME=value`` the module printed, by name."""

    return {
        match["name"].upper(): match["value"]
        for match in _PARAM_ECHO.finditer(transcript)
    }


class FdiConfigConsole:
    """The module's text-command link, open while its stream is stopped.

    The caller owns the port and is responsible for having parked whatever
    reads it; this class only talks.  Use it as a context manager so the
    module always gets its ``#fdeconfig`` even when a command raises.

    Every command goes through ``say``, which is what keeps the three rules
    in the module docstring true of every line this file puts on the wire.
    Commands are spaced ``SPACING_SECONDS`` apart, which is the module's
    requirement rather than a preference; tests that do not need the real
    pacing pass their own.
    """

    def __init__(self, port, *, spacing: float | None = None) -> None:
        self._port = port
        self._spacing = float(SPACING_SECONDS if spacing is None else spacing)
        #: When the last command went out.  The next one is held back until
        #: the spacing has passed from here.
        self._wrote_at = 0.0
        self._entered = False
        #: The last command and the whole window that followed it, verbatim.
        #: Kept because every wrong turn in this driver has been an
        #: assumption about what the module would say, and the fastest way
        #: to settle the next one is to have its actual words to hand.
        self.last_exchange: tuple[str, str] = ("", "")

    # ------------------------------------------------------------ session
    def __enter__(self) -> "FdiConfigConsole":
        self.enter()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def enter(self) -> None:
        """Stop the stream and take the module into config mode.

        TWO things are checked, and the second one costs a command.

        It must have STOPPED NAVIGATING.  That is the state change itself,
        so it is what is watched for -- not the acknowledgement, which is a
        string the handler prints, and not a banner, which the manual and
        the module disagree about.

        And it must be LISTENING, which needs positive evidence.  Silence is
        not evidence: an unplugged module, a dead port and a module that
        entered perfectly all send no frames, so "no frames" alone let a
        port with nothing on it enter, and the failure surfaced one command
        later as a timeout on ``#fparam set`` -- a clear fault dragged into
        a later one, pointing the operator at the wrong thing.

        So one question is asked, and EITHER answer proves the console is
        there: the echo says it read a parameter, ``*#ERROR`` says it parsed
        a line and refused it, and only this console prints either.
        ``*#ERROR`` is deliberately not read as a failure to enter -- a
        module already sitting in its console from a session that never
        closed may well answer ``#fconfig`` that way, and refusing it would
        turn the one state this is meant to recover from into a dead end.
        """

        if self._entered:
            return
        self._port.reset_input_buffer()
        self._write("#fconfig")
        self.last_exchange = ("#fconfig", _words(self._listen_until_quiet()))
        if any(
            mark in self._listen(STILL_NAVIGATING_SECONDS) for mark in _STREAM_MARKS
        ):
            raise RuntimeError(
                "the module on this port kept streaming through #fconfig, so "
                "it never entered config mode"
            )
        self._entered = True
        asked = f"#fparam get {IMU_PACKET_NAME}"
        answered = self.say(asked)
        if IMU_PACKET_NAME not in _parameters_in(answered) and ERROR not in answered:
            self._entered = False
            raise RuntimeError(
                "the module on this port stopped streaming for #fconfig but "
                f"then would not answer {asked!r}, so there is no console "
                f"here; it said {answered.strip()[:160]!r}"
            )

    def close(self) -> None:
        """Put the module back on the air, whatever happened in between.

        Nothing is read afterwards: what the module does next is frames,
        and the caller -- which owns the reader -- is the one that waits for
        them and says so if they do not come.
        """

        if not self._entered:
            return
        try:
            self._port.reset_input_buffer()
            self._write("#fdeconfig")
        finally:
            self._entered = False

    # --------------------------------------------------------------- link
    def say(self, command: str) -> str:
        """Send ONE command and return its whole reply.

        The reply is read until the module goes quiet, so what comes back is
        all of it.  What is NOT waited for here is the gap before the next
        command: that is charged to the next command, which is where it
        belongs and where it costs nobody anything if none comes.
        """

        if not self._entered:
            raise RuntimeError(
                f"{command!r} is a config-mode command and this console is not "
                "in config mode"
            )
        self._port.reset_input_buffer()
        self._write(command)
        transcript = _words(self._listen_until_quiet())
        self.last_exchange = (command, transcript)
        return transcript

    # ---------------------------------------------------------- the module
    def get_parameter(self, name: str) -> str | None:
        """One named parameter's value, or None when this firmware lacks it.

        A module that has it echoes it, ``MSG_IMU=7``; a module that has not
        got it says ``*#ERROR``, which is its own word for no.  Both come
        out of the same window, so absence is what the module SAID rather
        than something inferred from a clock.

        Anything else is neither, and it is reported rather than scored as
        absence: reading "no such parameter" off a reply that answers some
        other question is how a knob the panel was showing a moment ago
        vanishes with no reason recorded.
        """

        wanted = str(name).upper()
        transcript = self.say(f"#fparam get {name}")
        found = _parameters_in(transcript)
        if wanted in found:
            return found[wanted]
        if ERROR in transcript:
            return None
        raise RuntimeError(
            f"asked this module for {name} and it answered "
            f"{transcript.strip()[:120]!r}, which is neither the value nor "
            "its refusal"
        )

    def set_parameter(self, name: str, value: str) -> str:
        """Write one named parameter and answer with what it reads back as.

        Two commands, two windows.  The write is refused on the module's own
        refusal word; the readback is what this returns, and it is the
        PARAMETER TABLE's value -- not what the module is running.  Nothing
        takes effect until ``save`` and ``reboot``.
        """

        written = self.say(f"#fparam set {name} {value}")
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
        """Commit the parameter table to flash, or refuse.

        Nothing reports what is IN flash, so a save cannot be confirmed the
        way a written setting can, and this does not try to.  What it will
        not do is report a save nobody acknowledged: the restart behind it
        makes live whatever flash holds, so "not known" is a refusal here
        and the caller discards the uncommitted table rather than commit on
        a guess.
        """

        transcript = self.say("#fsave")
        if ERROR in transcript:
            raise TuneRefused(
                f"the module refused to save: {transcript.strip()[:120]!r}"
            )
        if OK not in transcript:
            raise TuneRefused(
                "the module did not acknowledge #fsave, so nothing is known "
                f"to have reached its flash; it said {transcript.strip()[:120]!r}"
            )
        return transcript

    def reboot(self) -> None:
        """Restart the module, which is what makes a saved setting live.

        The command needs confirming with ``y``.  No prompt in the window
        means the module is not waiting for one, and sending it anyway puts
        a bare ``y`` on the wire for it to read as a command -- so that is an
        error, and the caller's exit takes the module out of the console
        properly.

        The module is restarting when this returns, so the console is over.
        """

        transcript = self.say("#freboot")
        if CONFIRM_PROMPT not in transcript:
            raise RuntimeError(
                "the module did not ask to confirm the restart; it answered "
                f"{transcript.strip()[:120]!r}"
            )
        self._write("y")
        self._entered = False

    # -------------------------------------------------------------- lines
    def _write(self, text: str) -> None:
        """Put one line on the wire, no sooner than the module will take it.

        The wait is spent reading, and it is measured from the last line to
        go OUT rather than from the last byte to come back: the module is
        busy from the moment it is spoken to.

        The line is flushed before the clock is read, so ``_wrote_at`` marks
        when the bytes LEFT rather than when they were handed over.  That is
        the same quantity the sweep behind ``SPACING_SECONDS`` measured, and
        a buffered write timed from the handover would make every real gap
        shorter than the one asked for -- in the direction of the cliff.
        """

        owed = self._spacing - (time.monotonic() - self._wrote_at)
        if owed > 0.0:
            self._listen(owed)
        self._port.write((text + LINE_END).encode("ascii"))
        self._port.flush()
        self._wrote_at = time.monotonic()

    def _listen_until_quiet(self) -> bytes:
        """Read until the module has finished saying whatever it is saying.

        There is no end marker on this console, so the end of a reply is the
        module going quiet.  Nothing is judged here -- the caller parses what
        comes back -- and nothing stops at the first thing that looks like an
        answer.
        """

        got = bytearray()
        deadline = time.monotonic() + REPLY_TIMEOUT_SECONDS
        last = time.monotonic()
        while time.monotonic() < deadline:
            chunk = self._read_some()
            now = time.monotonic()
            if chunk:
                got += chunk
                last = now
            elif got and now - last >= REPLY_QUIET_SECONDS:
                break
        return bytes(got)

    def _listen(self, seconds: float) -> bytes:
        """Read the line for a fixed time, and read all of it.

        For the two waits whose length is not the module's to decide: the
        gap before a command, and the moment after ``#fconfig`` in which a
        module that never left the air would still be sending frames.
        """

        got = bytearray()
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            got += self._read_some()
        return bytes(got)

    def _read_some(self) -> bytes:
        """Whatever is on the port now, without waiting for a full buffer.

        ``in_waiting`` bytes when there are any, and otherwise a read of one
        that the port's own timeout ends -- which is what makes a listen
        both responsive and cheap.
        """

        waiting = self._port.in_waiting
        return self._port.read(waiting if waiting else 1)


__all__ = [
    "CONFIRM_PROMPT",
    "ERROR",
    "FdiConfigConsole",
    "IMU_PACKET_NAME",
    "LINE_END",
    "OK",
    "REPLY_QUIET_SECONDS",
    "REPLY_TIMEOUT_SECONDS",
    "SPACING_SECONDS",
    "STILL_NAVIGATING_SECONDS",
]
