"""The module's own configuration console, over the same serial line.

An N100 normally streams binary FDILink frames and listens to nothing.  It
also carries an ASCII command console, documented in chapter 5 of FDI's
《通信协议》: send ``#fconfig``, the module STOPS navigating and stops
emitting frames, every ``#f`` command is answered in plain text, and
``#fdeconfig`` puts it back on the air.

THIS FILE DOES EXACTLY WHAT THE PROBE THAT WORKS ON THE BENCH DOES.

    write one command, listen for a fixed window, then read the whole
    window and move on.

Nothing here stops reading early because it thinks it has seen enough, and
nothing sends a command before the window of the one before it has run out.
That is the entire protocol, and it is not a style: every version of this
file that tried to be quicker than that broke on the real module, and this
is the shape that has never broken on it.

The bench settled it.  Sent one at a time, each followed by a two-second
listen, every command answers::

    > #fconfig                      (frames drain, then) *#OK
    > #fparam get MSG_IMU           MSG_IMU=7
    > #fmsg                         MSG_IMU[40]  100.0Hz
                                    MSG_AHRS[41]   0.0Hz
                                    ... 1904 bytes, forty packets
    > #fparam get FILT_LPF_ENABLED  FILT_LPF_ENABLED=0.000000
    > #fparam                       *#ERROR       (the bare form; no arguments)
    > #fparam set MSG_IMU 7         *#OK          (written, NOT yet live)
    > #fsave                        *#OK
    > #freboot                      (y/n)  then y -> back in 2.5 s
    > #fdeconfig                    (the binary stream resumes)

And the same ``#fmsg``, sent 0.05 s behind the command before it, with the
host not reading in between::

    > #fmsg                         *#ERROR

A later sweep refused it at every gap up to 0.8 s as well.  Whatever the
module wants -- time, or its reply taken off the line, or both -- the
listen gives it, because the listen IS the gap and it is spent reading.

So a reply is not recognised, it is COLLECTED: the window is read to its
end and then parsed in one go.  ``NAME=value`` is a parameter echo,
``MSG_x[id] nHz`` a listing line, ``*#OK`` and ``*#ERROR`` the module's yes
and no.  With one command per window, nothing else can have printed them.

``MSG_IMU=4`` standing beside ``MSG_IMU[40] 10.0Hz`` is what says a rate is
stored as a LADDER INDEX -- rung 4 is 10 Hz -- which is why the manual's
``#fmsg 40 100`` is answered ``*#OK`` and changes nothing.  A bare
``#fparam`` is an error, so parameters cannot be enumerated: the rates come
from ``#fmsg``, which is the module listing itself, and named parameters
have to be asked for one at a time.

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

#: How long to listen after a command before sending another.  This is the
#: number the working probe uses, and the thing this driver kept trying to
#: beat: 0.05 s is answered ``*#ERROR``, and so is every gap up to 0.8 s.
#: It is not a timeout -- the window always runs out, and the whole of it
#: is read.
LISTEN_SECONDS = 2.0

#: And for ``#fmsg``, which prints some 1904 bytes in batches.
LISTING_LISTEN_SECONDS = 5.0

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

#: ``MSG_IMU[40]   10.0Hz`` from ``#fmsg``: the module enumerating itself,
#: one line per packet, with the rate in hertz.  No ``=`` in it, so a
#: listing and a parameter echo cannot be mistaken for one another.
_PACKET_LINE = re.compile(
    r"(?P<name>MSG_[A-Z0-9_]+)\[(?P<id>[0-9A-Fa-f]{1,2})\] *"
    r"(?P<hz>[0-9]+(?:\.[0-9]+)?)Hz",
    re.IGNORECASE,
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


def packets_in(transcript: str) -> tuple[tuple[str, int, float], ...]:
    """Every ``MSG_x[id] nHz`` the module printed, as ``(name, id, hertz)``."""

    return tuple(
        (match["name"], int(match["id"], 16), float(match["hz"]))
        for match in _PACKET_LINE.finditer(transcript)
    )


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

    Every command costs ``listen_seconds``, and that is deliberate -- see
    the module docstring.  Tests that do not need the real pacing pass
    their own.
    """

    def __init__(
        self,
        port,
        *,
        listen_seconds: float | None = None,
        listing_listen_seconds: float | None = None,
    ) -> None:
        self._port = port
        self._listen_seconds = float(
            LISTEN_SECONDS if listen_seconds is None else listen_seconds
        )
        self._listing_listen = (
            float(listing_listen_seconds)
            if listing_listen_seconds is not None
            else max(LISTING_LISTEN_SECONDS, self._listen_seconds)
        )
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
        opening = self._listen(self._listen_seconds)
        self.last_exchange = ("#fconfig", _as_text(opening))
        still = self._listen(STILL_NAVIGATING_SECONDS)  # noqa: E501 -- read live
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
            self._listen(self._listen_seconds)
        finally:
            self._entered = False

    # --------------------------------------------------------------- link
    def say(self, command: str, *, listen: float | None = None) -> str:
        """Send ONE command and return the whole window that follows it.

        The window always runs to its end.  Stopping it early on something
        that looks like the answer is what every broken version of this
        file did: the rest of the reply then arrives inside the next
        command's window, and the next command arrives before this module
        will take it.
        """

        self._require_console(command)
        self._port.reset_input_buffer()
        self._write(command)
        transcript = _as_text(
            self._listen(self._listen_seconds if listen is None else listen)
        )
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

    def packet_rates(self) -> tuple[tuple[str, int, float], ...]:
        """Every packet this module has, as ``(name, id, hertz)``.

        The module enumerating itself, and the only readback there is for a
        rate -- the parameter holding one reads back as the ladder index
        that was written to it, not as hertz.
        """

        transcript = self.say("#fmsg", listen=self._listing_listen)
        listed = packets_in(transcript)
        if not any(name.upper() == IMU_PACKET_NAME for name, _id, _hz in listed):
            raise RuntimeError(
                "the module did not list its packets; it said "
                f"{transcript.strip()[:160]!r}"
            )
        return listed

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
        self._port.write((text + LINE_END).encode("ascii"))
        flush = getattr(self._port, "flush", None)
        if callable(flush):
            flush()

    def _listen(self, seconds: float) -> bytes:
        """Read the line for this long, and read all of it.

        No early exit.  The window is the reply AND the gap before the next
        command, which is what the module wants and what this driver spent
        three rewrites trying to shorten.
        """

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
    "LISTEN_SECONDS",
    "LISTING_LISTEN_SECONDS",
    "OK",
    "STILL_NAVIGATING_SECONDS",
    "packets_in",
    "parameters_in",
]
