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

Nothing here decides what the module's legal values are.  ``#fmsg`` with no
argument makes the module list its own packets and their current rates, and
every write is read back, so the rate ladder, the packet set and the
refusals are all the module's answers rather than a table in this file that
a firmware revision would quietly falsify.

And nothing here judges the module by a BANNER.  The manual prints
``Config Mode`` as the reply to ``#fconfig``; a real one answers ``*#OK``.
The manual also states the judgement that actually holds -- "if the data
stops being sent, config mode was entered" -- and that is the one used
here: entering is the stream STOPPING, leaving is the stream COMING BACK.
A string a document printed once is not a contract; what the module does
with its serial line is.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time

from zlc_atom.authoring import TuneRefused

#: The first byte of every FDILink frame -- what the module's stream looks
#: like, and therefore what "it is navigating again" looks like.
FRAME_HEAD = 0xFC


#: What the console appends to every command, and what it answers with.
#: These are the literal bytes the vendor's own FDILinkTool puts on the wire
#: (``#fconfig\r\n`` / ``#fdeconfig\r\n``), not a guess at the line ending.
LINE_END = "\r\n"

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

#: ``IMU        [40]  100.0Hz`` -- one packet the module can emit, with its
#: id and its current rate.  Searched for across the whole reply rather than
#: matched per line: the manual's own transcript of this reply reaches us
#: through a PDF table, so whether the module puts the three parts on one
#: line or three is not something the archive can settle.  The three parts
#: in that order are the reply either way.
_RATE_ENTRY = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*\[\s*(?P<id>[0-9A-Fa-f]{1,2})\s*\]\s*"
    r"(?P<hz>[0-9]+(?:\.[0-9]+)?)\s*Hz",
    re.IGNORECASE,
)

#: ``imu_algn_yaw = 30.000000`` -- how the console prints one named value.
_PARAM_LINE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>\S+)")


@dataclass(frozen=True)
class PacketRate:
    """One packet the module can emit, and how often it is emitting it."""

    name: str
    packet_id: int
    rate_hz: float


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

        The judgement is the LINE GOING QUIET, which is the manual's own
        test and the only one that holds across firmwares: a module that
        entered stops navigating and stops emitting, so whatever it printed
        on the way in -- ``Config Mode``, ``*#OK``, nothing at all -- the
        silence that follows is the answer.  A module still streaming after
        the command never entered.
        """

        if self._entered:
            return
        self._port.reset_input_buffer()
        self._write("#fconfig")
        answer, went_quiet = self._read(ENTER_QUIET_SECONDS)
        self._entered = True
        self.greeting = answer.decode("ascii", "replace").strip()
        if not went_quiet:
            self.close()
            raise RuntimeError(
                f"the module on this port kept streaming through #fconfig, so "
                f"it never entered config mode (it said "
                f"{self.greeting[:120]!r})"
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

    def query(self, command: str) -> str:
        """Send one command and answer with everything the module said back."""

        self._require_console(command)
        self._write(command)
        answer = self._read_until_quiet(self._reply_quiet)
        self.last_exchange = (command, answer)
        return answer

    def _require_console(self, command: str) -> None:
        if not self._entered:
            raise RuntimeError(
                f"{command!r} is a config-mode command and this console is not "
                "in config mode"
            )

    def packet_rates(self) -> tuple[PacketRate, ...]:
        """Every packet this module says it emits, which may be none.

        The manual says the bare ``#fmsg`` prints every supported packet
        with its id and rate.  A real module answers ``*#OK`` and lists
        nothing.  An empty list is therefore a fact about this firmware and
        not a fault: the caller falls back to what it can MEASURE, which
        for the one packet this driver reads is the stream itself.  What
        the module actually said is on ``last_exchange`` either way.
        """

        answer = self.query("#fmsg")
        return tuple(
            PacketRate(found["name"], int(found["id"], 16), float(found["hz"]))
            for found in _RATE_ENTRY.finditer(answer)
        )

    def set_packet_rate(self, packet_id: int, rate_hz: float) -> float | None:
        """Ask for a rate; answer with the rate the module ended up at.

        The module may echo the new setting, or it may simply acknowledge --
        this firmware answers a bare ``#fmsg`` with nothing but ``*#OK``, so
        an echo cannot be required.  A write whose success was judged by a
        reply string would report a rate that WAS set as a refusal.  So the
        echo is read if it comes, the module is asked again if it does not,
        and ``None`` means neither told us -- which the caller answers by
        measuring the stream, the only reading that was never in doubt.
        """

        wanted = float(rate_hz)
        answer = self.query(f"#fmsg {packet_id:02x} {wanted:g}")
        for found in _RATE_ENTRY.finditer(answer):
            if int(found["id"], 16) == int(packet_id):
                return float(found["hz"])
        for packet in self.packet_rates():
            if packet.packet_id == int(packet_id):
                return packet.rate_hz
        return None

    def get_parameter(self, name: str) -> str | None:
        """One named parameter's value, or None when this firmware lacks it.

        Parameter names differ between firmwares -- the manual's own example
        names one that the shipped parameter tables do not have -- so a
        missing name is a fact about this module, not an error.
        """

        answer = self.query(f"#fparam get {name}")
        for line in answer.splitlines():
            found = _PARAM_LINE.match(line)
            if found and found["name"].upper() == name.upper():
                return found["value"]
        return None

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

    def save(self) -> str:
        """Commit to flash, and answer with whatever the module said.

        There is nothing to read back: no command reports what is in flash,
        so a save cannot be verified the way a written setting can.  Judging
        it by the reply text would put this back on a banner -- which is
        exactly what got entering config mode wrong -- so the reply is
        returned for the operator to see rather than turned into a verdict
        this code is not entitled to reach.
        """

        return self.query("#fsave")

    # -------------------------------------------------------------- lines
    def _write(self, text: str) -> None:
        self._port.write((text + LINE_END).encode("ascii"))
        flush = getattr(self._port, "flush", None)
        if callable(flush):
            flush()

    def _read_until_quiet(self, quiet: float) -> str:
        """What the module said, as text."""

        return self._read(quiet)[0].decode("ascii", "replace")

    def _read(self, quiet: float, *, until: bytes | None = None) -> tuple[bytes, bool]:
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
            elif chunks and now - last >= quiet:
                return b"".join(chunks), True
            elif not chunks and now - last >= quiet:
                # It never said anything at all, which for a module that
                # was not streaming is itself quiet.
                return b"", True
        return b"".join(chunks), False


__all__ = [
    "CONFIG_BANNER",
    "ENTER_QUIET_SECONDS",
    "FdiConfigConsole",
    "LINE_END",
    "OK",
    "PacketRate",
    "REPLY_QUIET_SECONDS",
]
