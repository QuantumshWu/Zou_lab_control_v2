"""The module's own configuration console, over the same serial line.

An N100 normally streams binary FDILink frames and listens to nothing.  It
also carries an ASCII command console, documented in chapter 5 of FDI's
《通信协议》: send ``#fconfig``, the module STOPS navigating and stops
emitting frames, every ``#f`` command is answered in plain text, and
``#fdeconfig`` puts it back on the air.

THE WIRE HERE LOOKS EXACTLY LIKE THE PROBE THAT WORKS ON THE BENCH.

    write one command, read the line continuously, and do not write the
    next one until SPACING_SECONDS after the last write.

The spacing is not waiting for a reply.  A reply is eleven bytes and
arrives in a fraction of a millisecond; what takes the time is the module,
which refuses a command that arrives while it is still busy with the one
before.  The bench measured exactly that: the same command answers
perfectly when it is given room, and ``*#ERROR`` when it is sent 0.05 s
behind another -- and a later sweep refused it at every gap up to 0.8 s.

So the gap is charged to the NEXT command rather than to this one.  A reply
is read until the module goes quiet and handed straight back; the waiting
happens before the next line goes out, and it is spent reading, which is
what the probe does too.  The last command of a session then pays nothing,
and a caller that asks one question is not held for a gap nobody will use.

What is NOT done here is stopping a read at the first thing that looks like
an answer.  Half a reply taken for a whole one is how this console used to
lose step: the other half turned up in the next command's window, and from
there every command read the one before it.

The bench settled it.  Sent one at a time, each followed by a two-second
listen, every command answers::

    > #fconfig                      (frames drain, then) *#OK
    > #fparam get MSG_IMU           MSG_IMU=7
    > #fparam get FILT_LPF_ENABLED  FILT_LPF_ENABLED=0.000000
    > #fparam                       *#ERROR       (the bare form; no arguments)
    > #fparam set MSG_IMU 7         *#OK          (written, NOT yet live)
    > #fsave                        *#OK
    > #freboot                      (y/n)  then y -> back in 2.5 s
    > #fdeconfig                    (the binary stream resumes)

And the same command, sent 0.05 s behind the one before it with the host
not reading in between, is refused::

    > #fmsg                         *#ERROR

A later sweep refused it at every gap up to 0.8 s as well.  (``#fmsg``
lists every packet the module has; this driver turns one of them, and reads
that one by name, but the pacing it proved applies to every command here.)  Whatever the
module wants -- time, or its reply taken off the line, or both -- the
listen gives it, because the listen IS the gap and it is spent reading.

So a reply is not recognised, it is COLLECTED: the window is read to its
end and then parsed in one go.  ``NAME=value`` is a parameter echo,
``MSG_x[id] nHz`` a listing line, ``*#OK`` and ``*#ERROR`` the module's yes
and no.  With one command per window, nothing else can have printed them.

``MSG_IMU=4`` standing beside ``#fmsg``'s ``MSG_IMU[40] 10.0Hz`` is what
says a rate is stored as a LADDER INDEX -- rung 4 is 10 Hz -- which is why
the manual's ``#fmsg 40 100`` is answered ``*#OK`` and changes nothing.  A
bare ``#fparam`` is an error, so parameters cannot be enumerated: every one
is asked for by name, one command at a time.

Entering is judged by the module CEASING TO NAVIGATE, which is the state
change itself -- not by a banner, which the manual and the module disagree
about, and not by the acknowledgement, which arrives after the frames have
drained and is simply read with them.
"""

from __future__ import annotations

import re
import time

from zlc_atom.authoring import TuneRefused

#: The first byte of every FDILink frame -- what the module's stream looks
#: like, and therefore what "it is navigating again" looks like.
FRAME_HEAD = 0xFC

#: What the console appends to every command.  These are the literal bytes
#: the vendor's own FDILinkTool puts on the wire (``#fconfig\r\n``).
LINE_END = "\r\n"

#: The module's yes and its no.  One command per window, so neither can
#: have come from anything but the command that window belongs to.
OK = "*#OK"
ERROR = "*#ERROR"

#: What ``#freboot`` waits to be answered ``y``.
CONFIRM_PROMPT = "(y/n)"

#: The packet every N100 sends, and the one this bench reads.
IMU_PACKET_NAME = "MSG_IMU"

#: How far apart two commands must go out.  Measured on the bench, not
#: chosen: 0.05 s is answered ``*#ERROR`` and so is every gap up to 0.8 s,
#: while the probe that works leaves two seconds.  It is a spacing and
#: never a timeout -- it is spent reading the line, and it is charged to
#: the command that comes next rather than to the one that just answered.
SPACING_SECONDS = 2.0

#: A reply is over when the module has said nothing for this long.  There
#: is no end marker on this console, so this is what says the whole of it
#: is in hand before anybody parses it.
REPLY_QUIET_SECONDS = 0.25

#: And a module that says nothing at all within this is not answering.
REPLY_TIMEOUT_SECONDS = 3.0

#: A frame header followed by a NAVIGATION packet type -- what "it is still
#: navigating" looks like.  0xF0 is the 1 Hz heartbeat, which the module
#: sends while it is NOT navigating, so counting it here would have this
#: driver reading the same two bytes in the opposite direction from its own
#: stream check.
_STREAM_MARKS = (b"\xfc\x40", b"\xfc\x41", b"\xfc\x42")

#: How long to watch for frames after the listen that follows ``#fconfig``.
#: By then the frames in flight have long drained, so anything still
#: arriving is a module that never left the air.
STILL_NAVIGATING_SECONDS = 0.5

#: ``MSG_IMU=4`` from ``#fparam get``, and ``imu_algn_yaw = 0.000000`` from
#: ``#faxis``: the same shape with and without spaces, which is why the
#: spaces are optional here rather than assumed to be there.
_PARAM_ECHO = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*) *= *(?P<value>[-+]?[0-9]+(?:\.[0-9]+)?)"
)

def _as_text(data: bytes) -> str:
    """The window as words.  A frame's bytes are not ASCII and stay noise."""

    return data.decode("ascii", "replace")


def parameters_in(transcript: str) -> dict[str, str]:
    """Every ``NAME=value`` the module printed, by name."""

    return {
        match["name"].upper(): match["value"]
        for match in _PARAM_ECHO.finditer(transcript)
    }


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

    Commands are spaced ``SPACING_SECONDS`` apart, which is the module's
    requirement rather than a preference -- see the module docstring.
    Tests that do not need the real pacing pass their own.
    """

    def __init__(
        self,
        port,
        *,
        spacing: float | None = None,
    ) -> None:
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

        One command and one window, like everything else.  What the window
        holds -- the frames still draining, then ``*#OK`` -- is read and
        kept for the record; what decides is whether the module is STILL
        NAVIGATING afterwards, which is the state change itself rather than
        anything it printed.
        """

        if self._entered:
            return
        self._port.reset_input_buffer()
        self._write("#fconfig")
        opening = self._listen_until_quiet()
        self.last_exchange = ("#fconfig", _as_text(opening))
        still = self._listen(STILL_NAVIGATING_SECONDS)
        if any(mark in still for mark in _STREAM_MARKS):
            raise RuntimeError(
                "the module on this port kept streaming through #fconfig, so "
                "it never entered config mode"
            )
        self._entered = True

    def close(self) -> None:
        """Put the module back on the air, whatever happened in between."""

        if not self._entered:
            return
        try:
            self._port.reset_input_buffer()
            self._write("#fdeconfig")
            self._listen(STILL_NAVIGATING_SECONDS)
        finally:
            self._entered = False

    # --------------------------------------------------------------- link
    def say(self, command: str) -> str:
        """Send ONE command and return its whole reply.

        The reply is read until the module goes quiet, so what comes back
        is all of it.  What is NOT waited for here is the gap the module
        wants before the next command: that is charged to the next command,
        which is where it belongs and where it costs nobody anything if no
        next command comes.
        """

        self._require_console(command)
        self._port.reset_input_buffer()
        self._write(command)
        transcript = _as_text(self._listen_until_quiet())
        self.last_exchange = (command, transcript)
        return transcript

    def write(self, command: str) -> None:
        """Send one command and do not wait for what it says back."""

        self._require_console(command)
        self._write(command)

    def query(self, command: str) -> str:
        """Send one command and answer with its window.  The ScpiLink shape."""

        return self.say(command)

    def _require_console(self, command: str) -> None:
        if not self._entered:
            raise RuntimeError(
                f"{command!r} is a config-mode command and this console is not "
                "in config mode"
            )

    # ---------------------------------------------------------- the module
    def get_parameter(self, name: str) -> str | None:
        """One named parameter's value, or None when this firmware lacks it.

        A module that has it echoes it, ``MSG_IMU=7``; a module that has not
        got it says ``*#ERROR``, which is its own word for no.  Both come
        out of the same window, so absence is what the module said rather
        than something inferred from a clock.

        Anything else is neither, and it is reported rather than scored as
        absence: reading "no such parameter" off a reply that answers some
        other question is how a knob the panel was showing a moment ago
        vanishes with no reason recorded.
        """

        wanted = str(name).upper()
        transcript = self.say(f"#fparam get {name}")
        found = parameters_in(transcript)
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

        Two commands, two windows.  The write is refused on the module's
        own refusal word; the readback is what this returns, and it is the
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
        """Commit to flash, and refuse only on the module's own refusal word.

        Nothing reports what is in flash, so a save cannot be CONFIRMED the
        way a written setting can, and this does not try to.  But a save
        that was refused must not be reported as done: everything after it
        is built on the value having reached flash.
        """

        transcript = self.say("#fsave")
        if ERROR in transcript:
            raise TuneRefused(
                f"the module refused to save: {transcript.strip()[:120]!r}"
            )
        return transcript

    def reboot(self) -> None:
        """Restart the module, which is what makes a saved setting live.

        The command needs confirming with ``y``.  No prompt in the window
        means the module is not waiting for one, and sending it anyway puts
        a bare ``y`` on the wire for it to read as a command -- so that is
        an error, and the caller's exit takes the module out of the console
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

        The wait is spent READING, which is what the probe that works does
        between its commands, and it is measured from the last line to go
        OUT rather than from the last byte to come back: the module is busy
        from the moment it is spoken to.
        """

        owed = self._spacing - (time.monotonic() - self._wrote_at)
        if owed > 0.0:
            self._listen(owed)
        self._port.write((text + LINE_END).encode("ascii"))
        self._wrote_at = time.monotonic()
        flush = getattr(self._port, "flush", None)
        if callable(flush):
            flush()

    def _listen_until_quiet(self) -> bytes:
        """Read until the module has finished saying whatever it is saying.

        There is no end marker on this console, so the end of a reply is
        the module going quiet.  Nothing is judged here -- the caller parses
        what comes back -- and nothing stops at the first thing that looks
        like an answer.
        """

        got = bytearray()
        deadline = time.monotonic() + REPLY_TIMEOUT_SECONDS
        last = time.monotonic()
        while time.monotonic() < deadline:
            waiting = getattr(self._port, "in_waiting", 0)
            chunk = self._port.read(waiting if waiting else 1)
            now = time.monotonic()
            if chunk:
                got += chunk
                last = now
            elif got and now - last >= REPLY_QUIET_SECONDS:
                break
        return bytes(got)

    def _listen(self, seconds: float) -> bytes:
        """Read the line for this long, and read all of it."""

        got = bytearray()
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            waiting = getattr(self._port, "in_waiting", 0)
            chunk = self._port.read(waiting if waiting else 1)
            if chunk:
                got += chunk
        return bytes(got)


__all__ = [
    "CONFIRM_PROMPT",
    "ERROR",
    "FRAME_HEAD",
    "FdiConfigConsole",
    "IMU_PACKET_NAME",
    "LINE_END",
    "OK",
    "REPLY_QUIET_SECONDS",
    "REPLY_TIMEOUT_SECONDS",
    "SPACING_SECONDS",
    "STILL_NAVIGATING_SECONDS",
    "parameters_in",
]
