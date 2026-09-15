"""The module's own configuration console, over the same serial line.

An N100 normally streams binary FDILink frames and listens to nothing.  It
also carries an ASCII command console, documented in chapter 5 of FDI's
《通信协议》: send ``#fconfig``, the module STOPS navigating and stops
emitting frames, every ``#f`` command is answered in plain text, and
``#fdeconfig`` (or a confirmed ``#freboot``) puts it back on the air.  That
is the whole of this module.

Two facts shape the code.  The console is the ONLY documented way to move a
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
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time


#: What the console appends to every command, and what it answers with.
#: These are the literal bytes the vendor's own FDILinkTool puts on the wire
#: (``#fconfig\r\n`` / ``#fdeconfig\r\n``), not a guess at the line ending.
LINE_END = "\r\n"

#: The module's success token, and the banner it prints on entering.
OK = "*#OK"
CONFIG_BANNER = "Config Mode"

#: Commands the module refuses to run until it is answered ``y``.
CONFIRM_PROMPT = "(y/n)"

#: How long one command may take to answer.  A property round trip is a few
#: milliseconds; a reboot is seconds, and is waited for separately.
REPLY_TIMEOUT_SECONDS = 2.0

#: The module keeps sending for a moment after ``#fconfig`` -- frames already
#: in flight -- so entering waits this long for the line to go quiet.  Even
#: at the top of the rate ladder frames are milliseconds apart and the
#: driver's buffer holds only a few, so this is generous.
ENTER_QUIET_SECONDS = 0.15

#: A reply has no end marker, so it ends when the module has said nothing for
#: this long.  One command is then about this much wall clock, which is what
#: sets how long a whole settings apply takes.  The module answers one
#: command in a single USB frame's worth of time, and prints multi-line
#: replies back to back, so a gap this long is the end of the reply.
REPLY_QUIET_SECONDS = 0.08

#: ``IMU        [40]  100.0Hz`` -- one line per packet the module can emit.
_RATE_LINE = re.compile(
    r"^\s*(?P<name>\S+)\s*\[\s*(?P<id>[0-9A-Fa-f]{1,2})\s*\]\s*"
    r"(?P<hz>[0-9]+(?:\.[0-9]+)?)\s*Hz",
    re.IGNORECASE,
)

#: ``imu_algn_yaw = 30.000000`` -- how the console prints one named value.
_PARAM_LINE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>\S+)")


class ConsoleRefused(RuntimeError):
    """The module answered, and the answer was not what was asked for.

    The setting is still whatever it was: the console reports per command,
    so a refusal here means nothing was written, which is what tells this
    apart from a write whose readback was lost.
    """


@dataclass(frozen=True)
class PacketRate:
    """One packet the module can emit, and how often it is emitting it."""

    name: str
    packet_id: int
    rate_hz: float


class FdiConfigConsole:
    """A conversation with one module, held while its stream is stopped.

    The caller owns the port and is responsible for having parked whatever
    reads it; this class only talks.  Use it as a context manager so the
    module always gets its ``#fdeconfig`` even when a command raises.
    """

    def __init__(self, port, *, reply_timeout: float = REPLY_TIMEOUT_SECONDS) -> None:
        self._port = port
        self._reply_timeout = float(reply_timeout)
        self._entered = False

    # ------------------------------------------------------------ session
    def __enter__(self) -> "FdiConfigConsole":
        self.enter()
        return self

    def __exit__(self, *_exc) -> None:
        self.leave()

    def enter(self) -> None:
        """Stop the stream and take the module into config mode.

        The module answers ``Config Mode``; frames already on the wire keep
        arriving for a moment, so the line is read until it goes quiet and
        everything before the banner is discarded as the tail of the
        stream.
        """

        if self._entered:
            return
        self._port.reset_input_buffer()
        self._write("#fconfig")
        answer = self._read_until_quiet(ENTER_QUIET_SECONDS)
        self._entered = True
        if CONFIG_BANNER.lower() not in answer.lower():
            # The banner is the documented reply, but the stream stopping is
            # what the manual calls the success signal, so a module that
            # went quiet without printing it is still in config mode -- and
            # has to be taken back out.
            if answer.strip():
                self.leave()
                raise ConsoleRefused(
                    "the module did not enter config mode; it answered "
                    f"{answer.strip()[:200]!r}"
                )

    def leave(self) -> None:
        """Put the module back on the air, whatever happened in between."""

        if not self._entered:
            return
        try:
            self._write("#fdeconfig")
            # Waiting for quiet would wait forever: the module answers this
            # one by going back on the air, so the acknowledgement itself is
            # what ends the reply.
            self._read_until_quiet(ENTER_QUIET_SECONDS, until=OK)
        finally:
            self._entered = False

    # ------------------------------------------------------------ commands
    def command(self, text: str) -> str:
        """One command, and everything the module said back."""

        if not self._entered:
            raise ConsoleRefused(
                f"{text!r} is a config-mode command and this console is not in "
                "config mode"
            )
        self._write(text)
        return self._read_until_quiet(REPLY_QUIET_SECONDS)

    def packet_rates(self) -> tuple[PacketRate, ...]:
        """Every packet this module can emit, with its current rate.

        This is the module describing itself: which packets its firmware
        has, and what each is set to.  Nothing here is assumed.
        """

        answer = self.command("#fmsg")
        rates = []
        for line in answer.splitlines():
            found = _RATE_LINE.match(line)
            if found:
                rates.append(
                    PacketRate(
                        found["name"],
                        int(found["id"], 16),
                        float(found["hz"]),
                    )
                )
        if not rates:
            raise ConsoleRefused(
                "the module listed no packets; it answered "
                f"{answer.strip()[:200]!r}"
            )
        return tuple(rates)

    def set_packet_rate(self, packet_id: int, rate_hz: float) -> float:
        """Ask for a rate; answer with the rate the module says it took.

        The module echoes what it actually set, which is how a rate the
        firmware does not offer is caught -- the ladder is per packet and
        per firmware, so asking is the only honest way to know it.
        """

        wanted = float(rate_hz)
        answer = self.command(f"#fmsg {packet_id:02x} {wanted:g}")
        for line in answer.splitlines():
            found = _RATE_LINE.match(line)
            if found and int(found["id"], 16) == int(packet_id):
                return float(found["hz"])
        raise ConsoleRefused(
            f"the module did not confirm packet 0x{packet_id:02x} at {wanted:g} Hz; "
            f"it answered {answer.strip()[:200]!r}"
        )

    def get_parameter(self, name: str) -> str | None:
        """One named parameter's value, or None when this firmware lacks it.

        Parameter names differ between firmwares -- the manual's own example
        names one that the shipped parameter tables do not have -- so a
        missing name is a fact about this module, not an error.
        """

        answer = self.command(f"#fparam get {name}")
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

        self.command(f"#fparam set {name} {value}")
        reading = self.get_parameter(name)
        if reading is None:
            raise ConsoleRefused(
                f"this module has no parameter {name!r}: it would not read "
                "the name back after the write"
            )
        return reading

    def save(self) -> None:
        """Commit to flash, without which every change dies at power-off."""

        answer = self.command("#fsave")
        if OK not in answer:
            raise ConsoleRefused(
                f"the module did not confirm the save; it answered "
                f"{answer.strip()[:200]!r}"
            )

    # -------------------------------------------------------------- lines
    def _write(self, text: str) -> None:
        self._port.write((text + LINE_END).encode("ascii"))
        flush = getattr(self._port, "flush", None)
        if callable(flush):
            flush()

    def _read_until_quiet(self, quiet: float, *, until: str | None = None) -> str:
        """Everything the module says, until it stops or says ``until``.

        The console has no general end-of-reply marker: ``#fmsg`` answers
        with one line per packet, ``#faxis`` with three, ``#fsave`` with
        one.  So a reply is normally "what arrived before the line went
        quiet", bounded by the command timeout.  ``until`` is for the one
        command whose reply is followed by the navigation stream starting
        again, where quiet never comes.
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
                if until is not None and until.encode("ascii") in b"".join(chunks):
                    break
            elif chunks and now - last >= quiet:
                break
        return b"".join(chunks).decode("ascii", "replace")


__all__ = [
    "CONFIG_BANNER",
    "ConsoleRefused",
    "ENTER_QUIET_SECONDS",
    "FdiConfigConsole",
    "LINE_END",
    "OK",
    "PacketRate",
    "REPLY_QUIET_SECONDS",
]
