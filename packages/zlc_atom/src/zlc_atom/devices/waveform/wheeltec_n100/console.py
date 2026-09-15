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

ONE RULE HOLDS THE WHOLE THING TOGETHER, and it is the only one:

    every exchange ends with a question this module is CERTAIN to answer,
    and the reading stops when that certain answer arrives.

The rule exists because of one property of this console: it has no sequence
numbers, and this firmware says NOTHING for a parameter it has not got --
not ``*#ERROR``, nothing.  Silence and "not yet" are then the same thing on
the wire, so a console that reads by timing cannot tell them apart.  It
guesses, and one wrong guess desynchronises everything after it: the late
reply lands in the next command's window, and from there every command
reads the previous one's answer.  That is what put an empty page in Device
Control, and no amount of clearing buffers or widening windows fixes it,
because the bytes in question have not been sent yet.

``#fparam get MSG_IMU`` is the certain question.  ``MSG_IMU`` is the packet
every N100 has -- discovery finds the module BY those frames -- so the
module always answers, and ``MSG_IMU=4`` arriving means everything asked
before it has already been answered or was never going to be.  One read,
one transcript, one parse.  Absence is then a fact about the transcript
rather than a guess about the clock.

That single rule is what replaces the quiet windows, the per-command buffer
clearing, the read-until-it-names-itself loop and the truncation checks
this file used to carry.  It also makes entering config mode honest:
config mode MEANS the module answers ``#f`` commands, so entering is
confirmed by asking the certain question and getting it back, not by a
banner and not by counting frames.

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

#: Two words the module says.  ``*#ERROR`` is its own way of saying no, and
#: is read as a refusal; ``*#OK`` carries no information beyond "a command
#: was received", so nothing is judged by it -- see the module docstring.
OK = "*#OK"
ERROR = "*#ERROR"

#: What ``#freboot`` waits to be answered ``y``.
CONFIRM_PROMPT = "(y/n)"

#: The packet every N100 sends, and the one this bench reads.
IMU_PACKET_NAME = "MSG_IMU"

#: The question this module is certain to answer, and therefore the end of
#: every exchange.  See the module docstring: this one line is the reason
#: nothing here has to reason about how long a reply took.
CERTAIN_QUESTION = f"#fparam get {IMU_PACKET_NAME}"

#: How long one exchange may take.  Generous on purpose: configuring is
#: something an operator does now and then, never a hot path, and this is
#: only a deadline -- an exchange ends when the certain answer arrives, so
#: a large number here costs nothing in the ordinary case.  The longest
#: reply this module has, ``#fmsg``'s 1904 bytes, was in hand in under a
#: second.
REPLY_TIMEOUT_SECONDS = 4.0

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
    ) -> None:
        self._port = port
        self._reply_timeout = float(reply_timeout)
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

        Config mode MEANS the module answers ``#f`` commands, so that is
        what is checked: the certain question is asked, and getting its
        answer back is the proof.  A banner is not proof -- the manual
        prints ``Config Mode`` here and a real module answers ``*#OK`` --
        and neither is the stream going quiet, which is also what a module
        that has simply been unplugged looks like.

        Reading the acknowledgement first is not a second judgement; it is
        how the frames already in flight get drained and how the module
        gets the moment it needs to change state, before anything is asked
        of it.
        """

        if self._entered:
            return
        self._port.reset_input_buffer()
        self._write("#fconfig")
        # Unambiguous here and nowhere else: the port was just cleared and
        # the module was streaming binary, so this is the only command that
        # could have printed anything.
        self._read_until(lambda data: OK in _as_text(data) or ERROR in _as_text(data))
        self._entered = True
        try:
            self.ask()
        except RuntimeError as deaf:
            self.close()
            raise RuntimeError(
                f"the module on this port did not answer {CERTAIN_QUESTION!r} "
                "after #fconfig, so it is not in config mode"
            ) from deaf

    def close(self) -> None:
        """Put the module back on the air, whatever happened in between."""

        if not self._entered:
            return
        try:
            self._write("#fdeconfig")
            # The certain question is no use here: the module answers this
            # one by going back ON the air, so what identifies the answer is
            # a frame header -- a byte the console never prints.
            self._read_until(lambda data: bytes((FRAME_HEAD,)) in data)
        finally:
            self._entered = False

    # --------------------------------------------------------------- link
    def ask(self, *commands: str) -> str:
        """Send commands in order and read ONE transcript of their replies.

        The certain question goes last, and the reading stops when its
        answer arrives -- by which time every command in front of it has
        either been answered or was never going to be.  So a name missing
        from the transcript is a name this firmware has not got, which is
        a fact about what the module said rather than a guess about how
        long it took to not say it.

        This is the only reading method the console has.  ``close`` and
        ``reboot`` are the two exceptions, and only because their replies
        are not printed text at all.
        """

        asked = (*commands, CERTAIN_QUESTION)
        for command in asked:
            self._require_console(command)
        self._port.reset_input_buffer()
        for command in asked:
            self._write(command)
        transcript, answered = self._read_until(
            lambda data: IMU_PACKET_NAME in parameters_in(_as_text(data))
        )
        self.last_exchange = (" ; ".join(asked), transcript)
        if not answered:
            raise RuntimeError(
                f"the module did not answer {CERTAIN_QUESTION!r} within "
                f"{self._reply_timeout:g} s, so nothing it did say can be "
                f"matched to what was asked; it said {transcript.strip()[:160]!r}"
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

        The reply to ``#fparam set`` is not documented, so the write is not
        judged by what it printed: the readback IS the answer, and it is
        asked for in the same exchange rather than in a second one.  A
        parameter this firmware does not have reads back as nothing, which
        is a refusal rather than a silent no-op.

        What comes back is the PARAMETER TABLE's value -- not what the
        module is running.  Nothing takes effect until ``save`` and
        ``reboot``.
        """

        transcript = self.ask(f"#fparam set {name} {value}", f"#fparam get {name}")
        reading = parameters_in(transcript).get(str(name).upper())
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

        transcript = self.ask("#fsave")
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
        self._port.reset_input_buffer()
        self._write("#freboot")
        transcript, prompted = self._read_until(
            lambda data: CONFIRM_PROMPT in _as_text(data)
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

    def _read_until(self, is_the_answer) -> tuple[str, bool]:
        """Read until the answer is recognisable, and say whether it was.

        There is no quiet window and no end marker: what ends a read is
        seeing the thing that was waited for.  Running out of time is a
        failure with the transcript attached, never a reply treated as
        complete.
        """

        deadline = time.monotonic() + self._reply_timeout
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
    "ERROR",
    "FRAME_HEAD",
    "FdiConfigConsole",
    "IMU_PACKET_NAME",
    "LINE_END",
    "OK",
    "REPLY_TIMEOUT_SECONDS",
    "packets_in",
    "parameters_in",
]
