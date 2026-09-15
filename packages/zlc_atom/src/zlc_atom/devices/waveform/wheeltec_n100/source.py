"""WHEELTEC N100 nine-axis IMU, read off its FDILink serial stream.

The module talks first: from the moment it has power it emits binary
FDILink packets on its serial port at the rate it was configured to (the
factory default is 100 Hz; the operator may have set 200 or 400 with the
vendor tool), and nothing here ever writes to it.  One IMU packet carries
the three gyroscope axes, the three accelerometer axes, the three
magnetometer axes, the die temperature, the barometer pair and the
module's own microsecond clock, so this source publishes four quantities
from one record column set.

What is published is what the packet says, in units the registry knows:
the magnetometer's milligauss become microtesla (one milligauss is a tenth
of a microtesla), the die temperature in celsius becomes kelvin, the rates
and accelerations pass through in radians per second and metres per second
squared.  The sample interval is not assumed from the datasheet: it is
measured off the module's own packet clock when the port is opened, which
is also how a module set to a different rate is found out.

A frame proves itself.  Its 7-byte header carries a CRC8 over the header
and a CRC16 over the payload, and both are checked before a single number
is believed -- so a packet type this driver does not publish is stepped
over by the length the module itself stated, a corrupted frame is dropped
rather than published as a magnetic field, and no table in this file has
to hold a payload length that a firmware revision could falsify.  (The
vendor's own material gives three different lengths for the INS/GPS
packet, which is exactly the trap a length table walks into.)

Settings are the operator's, through Device Control: the module carries an
ASCII configuration console on the same line, and ``console.py`` speaks it.
Entering it stops the stream, so the reader is parked for the round trip
and the packet rate is re-measured afterwards rather than assumed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import struct
import threading
import time
from uuid import uuid4

import numpy as np

from zlc_atom.authoring import (
    AuthoringChoice,
    AuthoringField,
    TunableField,
    TuneRefused,
)
from zlc_atom.devices.waveform.contract import (
    WaveformAcquisitionMode,
    WaveformCaptureTerminalRecord,
    WaveformOutput,
    WaveformRecord,
    WaveformRecordQueue,
    WaveformWorkingPoint,
)

from .console import LINE_END, FdiConfigConsole


FRAME_HEAD = 0xFC
FRAME_TAIL = 0xFD
IMU_PACKET = 0x40
#: The payload length of the one packet this driver reads.  Every other
#: packet's length comes off the wire, from the header byte the CRC8 proves.
IMU_PAYLOAD_LENGTH = 56
#: head, type, length, serial, header CRC8, payload CRC16 high, low.
HEADER_LENGTH = 7


def _crc8_table() -> tuple[int, ...]:
    """CRC-8/MAXIM, the module's header check (reflected polynomial 0x8C)."""

    table = []
    for index in range(256):
        value = index
        for _ in range(8):
            value = (value >> 1) ^ 0x8C if value & 1 else value >> 1
        table.append(value & 0xFF)
    return tuple(table)


def _crc16_table() -> tuple[int, ...]:
    """CRC-16/XMODEM, the module's payload check (polynomial 0x1021)."""

    table = []
    for index in range(256):
        value = index << 8
        for _ in range(8):
            value = ((value << 1) ^ 0x1021) & 0xFFFF if value & 0x8000 else (value << 1) & 0xFFFF
        table.append(value)
    return tuple(table)


CRC8_TABLE = _crc8_table()
CRC16_TABLE = _crc16_table()


def header_crc8(header: bytes) -> int:
    """The check over start, type, length and sequence number."""

    value = 0
    for byte in header:
        value = CRC8_TABLE[value ^ byte]
    return value


def payload_crc16(payload: bytes) -> int:
    """The check over the payload bytes."""

    value = 0
    for byte in payload:
        value = ((value << 8) & 0xFF00) ^ CRC16_TABLE[((value >> 8) ^ byte) & 0xFF]
    return value & 0xFFFF
#: gyroscope xyz (rad/s), accelerometer xyz (m/s^2), magnetometer xyz (mG),
#: IMU temperature (degC), pressure, pressure temperature, timestamp (us).
_IMU_PAYLOAD = struct.Struct("<12fq")
DEFAULT_BAUD = 921600

#: The outer edge of what Device Control will let an operator TYPE.  This
#: is the documented ceiling for this product -- 用户手册 and the FAQ both
#: answer "数据包发布的最大频率为400HZ" -- and NOT the 1000 Hz the same
#: manual quotes, which is the internal sensor sampling rate and reaches no
#: packet.  What this particular module accepts is still its own answer;
#: this only keeps a typo from asking for something no firmware has.
MAX_PACKET_RATE_HZ = 400.0

#: Which KINDS of named parameter belong to an operator rather than to the
#: factory.  Filters shape what the sensors report -- a notch on the mains
#: frequency is the reason this matters for a magnetic measurement -- and
#: the AID switches decide which sensors the attitude solution fuses.  The
#: scale factors, cross-axis terms and biases beside them are calibration,
#: and are deliberately not offered.
OPERATOR_PARAMETER_PREFIXES = ("FILT_", "AID_")

#: How this firmware spells a packet's rate: an INDEX into this ladder,
#: written to a named parameter, never a number of hertz.  The manual's
#: ``#fmsg 40 100`` is answered ``*#OK`` and changes nothing; the vendor's
#: own parameter cache gives ``MSG_IMU`` the values 0..9 against exactly
#: these rungs, and its ground station writes that parameter.  Index 0 is
#: "no output" and is deliberately not offered: this bench recognises the
#: module by its packets, so a module told to stop sending them is a module
#: it can no longer find.
PACKET_RATE_LADDER_HZ = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0)

#: The parameter that holds the rate of the one packet this driver reads.
IMU_RATE_PARAMETER = "MSG_IMU"

#: The settings offered to an operator, tried BY NAME because this
#: firmware's bare ``#fparam`` lists nothing either.  A name the module does
#: not answer to is simply not offered, so this is a menu to try rather than
#: a claim about what any particular firmware has.  The rate is first; the
#: filters shape what the sensors report, which is why a mains notch
#: matters for a magnetic measurement; the AID switches decide which sensors
#: the attitude solution fuses.  Factory calibration sits in the same
#: namespace and is deliberately absent.
OFFERED_PARAMETERS = (
    IMU_RATE_PARAMETER,
    "FILT_LPF_ENABLED",
    "FILT_LPF_CUTOFF_FREQUENCY",
    "FILT_NOTCH_ENABLED",
    "FILT_NOTCH_CENTER_FREQUENCY",
    "FILT_NOTCH_CUTOFF_FREQUENCY",
    "FILT_NOTCH2_ENABLED",
    "FILT_NOTCH2_CENTER_FREQUENCY",
    "FILT_NOTCH2_CUTOFF_FREQUENCY",
    "AID_MAG_V_MAGNETIC",
    "AID_INIT_YAW_USE_MAG",
)


def rate_ladder_index(rate_hz: float) -> int:
    """The ladder rung an asked-for rate belongs to, or a refusal.

    Only the rungs are offerable, so a value off the ladder is refused here
    -- before anything reaches the module -- rather than sent and silently
    read as something else.
    """

    wanted = float(rate_hz)
    for index, rung in enumerate(PACKET_RATE_LADDER_HZ):
        if index and abs(rung - wanted) < 1e-6:
            return index
    offered = ", ".join(f"{rung:g}" for rung in PACKET_RATE_LADDER_HZ[1:])
    raise TuneRefused(
        f"{wanted:g} Hz is not one of this module's rates; it offers {offered} Hz"
    )


#: A parameter's NAME says what kind of value it holds.  The module gives
#: every one of them as a bare decimal, which would put a frequency and a
#: switch in the same nondescript number box; the vendor's naming is
#: consistent enough to do better, and unlike a list of parameter names a
#: suffix convention survives a firmware revision.  A name ending in
#: ``_FREQUENCY`` is hertz, and a switch is a switch -- the fusion
#: parameters are called switches in the vendor's own manual, and the
#: filter ones say ``_ENABLED``.
_FREQUENCY_SUFFIX = "_FREQUENCY"
_SWITCH_SUFFIX = "_ENABLED"
_SWITCH_PREFIX = "AID_"


def parameter_shape(name: str) -> tuple[str, str | None]:
    """The ``(kind, unit)`` a named parameter should be offered as."""

    upper = str(name).upper()
    if upper == IMU_RATE_PARAMETER:
        return "rate", "Hz"
    if upper.endswith(_SWITCH_SUFFIX) or upper.startswith(_SWITCH_PREFIX):
        return "switch", None
    if upper.endswith(_FREQUENCY_SUFFIX):
        return "float", "Hz"
    return "float", None


def _as_number(value: object) -> float | None:
    """The module's answer as a number, or None when it is not one."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: object) -> str:
    """One value as the console spells it: whole numbers without a point."""

    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


_MILLIGAUSS_PER_MICROTESLA = 10.0
_CELSIUS_TO_KELVIN = 273.15

#: The four quantities one packet carries, over the record's ten columns.
N100_OUTPUTS = (
    WaveformOutput("magnetic_field", "uT", ("x", "y", "z"), (0, 1, 2)),
    WaveformOutput("angular_rate", "rad*s^-1", ("x", "y", "z"), (3, 4, 5)),
    WaveformOutput("acceleration", "m*s^-2", ("x", "y", "z"), (6, 7, 8)),
    WaveformOutput("temperature", "K", ("imu",), (9,)),
)
_COLUMNS = 10


def drain_imu_samples(buffer: bytearray) -> list[tuple[float, tuple[float, ...]]]:
    """Take every complete packet off the front of ``buffer``.

    Returns ``(timestamp_seconds, samples)`` for each IMU packet, in arrival
    order, with the samples already in the published units and column
    order; every other packet type is stepped over by the length its own
    header states.  A frame is accepted only once its header CRC8 and its
    payload CRC16 both check, so a float containing 0xFC cannot start a
    false frame and a corrupted packet is dropped instead of published.
    Bytes that do not frame a packet are dropped one at a time until a
    frame lines up again, and a trailing partial frame stays in the buffer
    for the next read to complete.
    """

    found: list[tuple[float, tuple[float, ...]]] = []
    at = 0
    total = len(buffer)
    while True:
        head = buffer.find(FRAME_HEAD, at)
        if head < 0:
            at = total
            break
        if total - head < HEADER_LENGTH:
            at = head
            break
        if buffer[head + 4] != header_crc8(bytes(buffer[head:head + 4])):
            at = head + 1
            continue
        kind = buffer[head + 1]
        size = buffer[head + 2]
        end = head + HEADER_LENGTH + size + 1
        if total < end:
            at = head
            break
        if buffer[end - 1] != FRAME_TAIL:
            at = head + 1
            continue
        payload = bytes(buffer[head + HEADER_LENGTH:head + HEADER_LENGTH + size])
        if payload_crc16(payload) != (buffer[head + 5] << 8 | buffer[head + 6]):
            at = head + 1
            continue
        if kind == IMU_PACKET and size == IMU_PAYLOAD_LENGTH:
            values = _IMU_PAYLOAD.unpack(payload)
            found.append(
                (
                    values[12] * 1e-6,
                    (
                        values[6] / _MILLIGAUSS_PER_MICROTESLA,
                        values[7] / _MILLIGAUSS_PER_MICROTESLA,
                        values[8] / _MILLIGAUSS_PER_MICROTESLA,
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[9] + _CELSIUS_TO_KELVIN,
                    ),
                )
            )
        at = end
    del buffer[:at]
    return found


@dataclass(frozen=True)
class WheeltecN100Config:
    """Which port the module is on, and the line rate it was configured to."""

    port: str
    baud: int = DEFAULT_BAUD
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        port = str(self.port).strip()
        if not port:
            raise ValueError("serial port is required")
        if isinstance(self.baud, bool) or int(self.baud) <= 0:
            raise ValueError("baud must be a positive integer")
        if float(self.timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be positive")
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "baud", int(self.baud))
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


def _other_ports_on_this_machine() -> str:
    """The rest of the serial ports, named, so the next guess is informed."""

    try:
        from serial.tools import list_ports

        listed = [
            f"{item.device} ({item.description})"
            for item in sorted(list_ports.comports(), key=lambda item: item.device)
        ]
    except Exception:  # noqa: BLE001 -- this is already an error message
        return ""
    return ("This machine has: " + "; ".join(listed)) if listed else ""


def wake_from_config_mode(port) -> None:
    """Take a module that was left in its console back on the air.

    Config mode is a state in the MODULE, not in this process.  A session
    that opened the console and then died -- crashed, killed, unplugged
    mid-command -- leaves the module silent, and a silent N100 is one that
    discovery cannot see and that no amount of re-scanning will find,
    because discovery recognises this module BY its stream.  The module
    would sit like that until someone power-cycled it.

    So anything that expects to hear an N100 and hears nothing says this
    first and listens again.  It costs one line on the wire and it is
    harmless to a module that is already streaming: ``#fdeconfig`` outside
    config mode is a command the module does not act on.
    """

    port.write(b"#fdeconfig" + LINE_END.encode("ascii"))
    flush = getattr(port, "flush", None)
    if callable(flush):
        flush()


def _open_serial(port: str, baud: int):
    import serial

    return serial.Serial(port, baud, timeout=0.05, write_timeout=1.0)


#: How many of a silent port's bytes to keep as evidence: enough to show
#: a frame header, a line of text, or the shape of a wrong baud rate.
_EVIDENCE_BYTES = 96

#: How many times to listen again while an undone module comes back.  A
#: warm restart takes seconds, and each round is one rate-measuring window.
_UNDO_LISTEN_ROUNDS = 4

#: How long to wait for whole packets after the console closes.  A module
#: that is going to resume does it at once; this is the margin, not the
#: expectation.
_STREAM_BACK_SECONDS = 1.5

#: How many packets the rate is measured over when the port opens, and how
#: long to wait for them: at the slowest configurable rate (1 Hz) this is
#: not enough, and a module set that slow is not a magnetometer anyone is
#: sampling with -- the error says which port stayed silent.
_RATE_SAMPLE_PACKETS = 40
_RATE_SAMPLE_SECONDS = 2.0


class WheeltecN100WaveformSource:
    """Every IMU packet off the port is one record of ten samples."""

    def __init__(self, config: WheeltecN100Config, *, serial_port=None) -> None:
        self.config = config
        self._serial = _open_serial(config.port, config.baud) if serial_port is None else serial_port
        self._stop = threading.Event()
        # The reader owns the port; a settings round trip takes it away for
        # the duration.  ``_park`` asks, ``_parked`` acknowledges, so the
        # console never writes while the reader is mid-read.
        self._park = threading.Event()
        self._parked = threading.Event()
        self._settings_lock = threading.RLock()
        self._device_session_id = uuid4().hex
        self._settings_epoch = 0
        self._stamps: deque[float] = deque(maxlen=_RATE_SAMPLE_PACKETS)
        self._stamps_ready = threading.Event()
        self._sample_interval: float | None = None
        # How many of the packets counted so far carried a magnetic field
        # that had actually moved since the packet before.  See
        # ``magnetic_update_interval``.
        #: The first bytes this port produced, kept only to explain a
        #: silence.  "No packets arrived" has at least four causes -- wrong
        #: port, wrong baud, a module still in its console, a module with
        #: no power -- and the bytes tell them apart, so they are reported
        #: rather than left for the operator to guess between.
        self._first_bytes = bytearray()
        #: When the last whole IMU packet arrived.  Leaving the console is
        #: judged by this: a module that is navigating sends packets, and
        #: nothing else it does proves it.
        self._last_packet_at = 0.0
        self._magnetic_field: tuple[float, float, float] | None = None
        self._magnetic_changes = 0
        self._magnetic_packets = 0
        #: What the module last said its settings were.  Empty until it has
        #: been asked, and empty for good on a module whose firmware does
        #: not answer the configuration console -- such a module streams
        #: perfectly well and simply has nothing an operator can turn.
        self._settings: dict[str, object] = {}
        self._settings_refusal: str | None = None
        #: Whether the module enumerated its own packets, or the IMU rate
        #: below was measured off the stream because it would not.
        self._packets_listed = False
        #: The last console command and its verbatim reply, for the record.
        self._last_exchange: tuple[str, str] = ("", "")
        self._records = WaveformRecordQueue(
            "the N100", join_timeout_seconds=config.timeout_seconds
        )
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"zlc-n100-{config.port}",
            daemon=True,
        )
        try:
            self._reader.start()
            self._hear_the_module()
            self._read_settings_once()
        except BaseException:
            self.close()
            raise

    def _hear_the_module(self) -> None:
        """Time the stream, waking a module that was left in its console.

        The first silence is not yet a fault: the module may be sitting in
        a configuration console some earlier session never closed, where it
        says nothing at all.  One ``#fdeconfig`` is what tells those two
        silences apart, and the second listen is the one that decides.
        """

        try:
            self._await_rate()
            return
        except RuntimeError:
            pass
        self._park_reader()
        try:
            wake_from_config_mode(self._serial)
        finally:
            self._release_reader()
        self._stamps.clear()
        self._stamps_ready.clear()
        self._sample_interval = None
        self._await_rate()

    def _read_settings_once(self) -> None:
        """Ask the module for its settings, and let a refusal be a fact.

        A module that will not answer its configuration console still
        streams; what it does not have is anything for an operator to turn.
        Recording WHY is what turns a blank Device Control page into an
        answer.
        """

        try:
            self._settings = self._in_console(self._read_settings, stream_back=False)
        except Exception as refusal:  # noqa: BLE001 -- reported, not raised
            self._settings = {}
            self._settings_refusal = f"{type(refusal).__name__}: {refusal}"
        # Config mode stopped the stream either way, so the interval that
        # stamps the records is timed again before anyone reads.  This is
        # the one part that may NOT be shrugged off: a module that went
        # quiet and did not come back is unusable, and saying so with the
        # console's own refusal attached is the difference between a device
        # that fails for a stated reason and one that fails for none.  It
        # is deliberately not a ``finally``, which would have replaced a
        # console refusal with this one and lost the first.
        try:
            self._remeasure_rate()
        except BaseException as silent:
            # One more #fdeconfig before giving up: the console's own exit
            # may be what went missing.
            try:
                self._park_reader()
                try:
                    wake_from_config_mode(self._serial)
                finally:
                    self._release_reader()
                self._remeasure_rate()
                return
            except BaseException:
                pass
            raise RuntimeError(
                f"{self.config.port} stopped streaming when its configuration "
                "console was opened and did not resume"
                + (
                    f" (the console said: {self._settings_refusal})"
                    if self._settings_refusal
                    else ""
                )
            ) from silent

    # ------------------------------------------------------------- reading
    def _read_loop(self) -> None:
        buffer = bytearray()
        try:
            while not self._stop.is_set():
                if self._park.is_set():
                    # The console has the port.  Whatever is half-read is
                    # not a frame any more once the module has been in and
                    # out of config mode, so the buffer goes too.
                    buffer.clear()
                    self._parked.set()
                    self._stop.wait(0.02)
                    continue
                self._parked.clear()
                waiting = self._serial.in_waiting
                chunk = self._serial.read(waiting if waiting else 1)
                if not chunk:
                    continue
                buffer += chunk
                if len(self._first_bytes) < _EVIDENCE_BYTES:
                    self._first_bytes += chunk[
                        : _EVIDENCE_BYTES - len(self._first_bytes)
                    ]
                samples = drain_imu_samples(buffer)
                if samples:
                    self._accept(samples)
        except BaseException as error:  # noqa: BLE001 -- surfaced to the reader of records
            self._records.fail(error)
            self._stamps_ready.set()

    def _accept(self, samples: list[tuple[float, tuple[float, ...]]]) -> None:
        received = time.time_ns()
        self._last_packet_at = time.monotonic()
        for stamp, values in samples:
            self._stamp(stamp)
            self._watch_magnetic(values[0:3])
            # The module's own packet clock is the record's time: it ticks
            # with the sampling, where the host's clock ticks with the
            # serial delivery.
            self._records.push(
                np.asarray(values, dtype=np.float32).reshape(1, _COLUMNS), stamp, received
            )

    def _stamp(self, stamp: float) -> None:
        """The packet interval off the module's own clock, measured on the reader.

        The first packets after open are timed against each other and the
        median of their intervals is the rate; once enough have been seen
        the rate is settled and the stamps are no longer kept.
        """

        if self._stamps_ready.is_set():
            return
        self._stamps.append(stamp)
        if len(self._stamps) >= 2:
            intervals = np.diff(np.asarray(self._stamps, dtype=np.float64))
            advancing = intervals[intervals > 0.0]
            if advancing.size:
                self._sample_interval = float(np.median(advancing))
        if len(self._stamps) >= _RATE_SAMPLE_PACKETS:
            self._stamps_ready.set()

    def _watch_magnetic(self, field: tuple[float, ...]) -> None:
        """Count how often the magnetic field is a NEW reading.

        The three magnetometer axes ride inside the IMU packet, so they are
        SENT at the packet rate whether or not the magnetometer has taken a
        new measurement in between.  Nothing the vendor ships states the
        magnetometer's own output rate -- the specification table is a
        verbatim lift from another manufacturer's part -- so the only
        honest answer is the module's: count the packets whose field
        differs from the one before.  A field that repeats every other
        packet is a magnetometer running at half the packet rate, and
        raising the packet rate past it buys nothing but duplicates.
        """

        if self._stamps_ready.is_set():
            return
        current = (float(field[0]), float(field[1]), float(field[2]))
        if self._magnetic_field is not None:
            self._magnetic_packets += 1
            if current != self._magnetic_field:
                self._magnetic_changes += 1
        self._magnetic_field = current

    @property
    def magnetic_update_interval(self) -> float | None:
        """Seconds between genuinely new magnetic readings, or None if unknown.

        This is the packet interval multiplied by how many packets pass per
        change.  It answers the question the datasheet does not: whether the
        magnetic field this bench plots is really arriving as fast as the
        packets are.
        """

        interval = self._sample_interval
        if interval is None or self._magnetic_changes == 0:
            return None
        return interval * self._magnetic_packets / self._magnetic_changes

    def _what_the_port_said(self) -> str:
        """Read the silence: what did arrive, and what that usually means.

        Four different faults produce "no packets": the wrong port, the
        wrong baud, a module left in its configuration console, and a module
        with no power.  They look nothing alike on the wire, so the bytes
        are reported and named instead of being replaced by a list of
        things to go and check.
        """

        seen = bytes(self._first_bytes)
        if not seen:
            return (
                "Not one byte arrived. The port opened, so it exists and nothing "
                "else holds it, but nothing is talking on it: either this is not "
                "the module's port, or the module has no power. "
                + _other_ports_on_this_machine()
            )
        shown = seen[:48].hex(" ")
        if all(32 <= byte < 127 or byte in (9, 10, 13) for byte in seen):
            return (
                f"{len(seen)} bytes arrived and every one of them is text: "
                f"{seen[:96].decode('ascii', 'replace')!r}. An N100 in its "
                "configuration console answers in text and streams nothing; so "
                "does a different kind of device on this port."
            )
        if FRAME_HEAD in seen:
            return (
                f"{len(seen)} bytes arrived and they do contain FDILink frame "
                f"headers, but no whole packet framed up, which means the frames "
                f"are arriving damaged: {shown}"
            )
        return (
            f"{len(seen)} bytes arrived and none of them framed up as FDILink: "
            f"{shown}. Bytes with no frame header are what the wrong baud rate "
            f"looks like -- this module ships at {DEFAULT_BAUD} but its rate can "
            "be changed and is kept in its flash -- or what a different device "
            "on this port looks like."
        )

    def _await_rate(self) -> None:
        self._stamps_ready.wait(_RATE_SAMPLE_SECONDS)
        failure = self._records.failure
        if failure is not None:
            raise RuntimeError(f"reading {self.config.port} failed") from failure
        if len(self._stamps) < 2:
            raise RuntimeError(
                f"no FDILink IMU packets arrived on {self.config.port} at "
                f"{self.config.baud} baud within {_RATE_SAMPLE_SECONDS:g} s. "
                + self._what_the_port_said()
            )
        if self._sample_interval is None:
            raise RuntimeError(
                f"the module on {self.config.port} stamps its packets with a clock "
                "that does not advance"
            )

    # ------------------------------------------------------------ contract
    @property
    def timeout(self) -> float:
        return self.config.timeout_seconds

    @property
    def identity(self) -> str:
        return f"wheeltec-n100:{self.config.port}"

    @property
    def outputs(self) -> tuple[WaveformOutput, ...]:
        return N100_OUTPUTS

    @property
    def record_samples(self) -> int:
        return 1

    def working_point(self) -> WaveformWorkingPoint:
        interval = self._sample_interval
        if interval is None:
            raise RuntimeError("the N100 packet rate was never measured")
        return WaveformWorkingPoint(
            WaveformAcquisitionMode.FREE_RUNNING,
            interval,
            {
                "port": self.config.port,
                "baud": self.config.baud,
                "packet_rate_hz": round(1.0 / interval, 3),
                "magnetic_update_hz": (
                    None
                    if self.magnetic_update_interval is None
                    else round(1.0 / self.magnetic_update_interval, 3)
                ),
                "settings": dict(self._settings),
                "settings_refusal": self._settings_refusal,
                "packets_listed": self._packets_listed,
                "last_console_exchange": self._last_exchange,
            },
        )


    # -------------------------------------------------------------- knobs
    def _park_reader(self) -> None:
        """Take the port away from the reader, or say why it could not be."""

        self._park.set()
        if not self._parked.wait(self.config.timeout_seconds):
            self._park.clear()
            raise RuntimeError(
                f"the reader on {self.config.port} did not release the port; "
                "its settings cannot be read or written while it holds it"
            )

    def _release_reader(self) -> None:
        self._park.clear()
        self._parked.clear()

    def _in_console(self, work, *, stream_back: bool = True):
        """Run ``work(console)`` with the module in its configuration console.

        Entering stops the stream, so this refuses while a capture is armed:
        a capture that lost its packets mid-flight would publish a gap no
        reader could tell from the module having gone quiet.

        And EVERY command verifies the way out, not just the one that
        changes the sample rate.  Config mode is a state in the module: a
        parameter write, or a save, that reported success while leaving the
        module silent would put it exactly where a crashed session used to
        -- invisible to discovery, unusable until power-cycled -- with the
        operator told the write had worked.
        """

        with self._settings_lock:
            if self._records.armed:
                raise RuntimeError(
                    "the N100 stops streaming while its settings are read or "
                    "written, so a capture must be finished first"
                )
            self._park_reader()
            try:
                with FdiConfigConsole(self._serial) as console:
                    try:
                        answer = work(console)
                    finally:
                        self._last_exchange = console.last_exchange
            finally:
                self._release_reader()
            if stream_back:
                self._require_stream_back()
            return answer

    def _require_stream_back(self) -> None:
        """Wait for whole packets again, asking once more if they do not come.

        Whole packets, not a frame header: this module emits a 1 Hz
        heartbeat whose first byte is a frame header too, so a header alone
        would have read a silent module as a navigating one.
        """

        for attempt in range(2):
            mark = time.monotonic()
            deadline = mark + _STREAM_BACK_SECONDS
            while time.monotonic() < deadline:
                if self._last_packet_at >= mark:
                    return
                time.sleep(0.005)
            if attempt == 0:
                self._park_reader()
                try:
                    wake_from_config_mode(self._serial)
                finally:
                    self._release_reader()
        raise RuntimeError(
            f"the module on {self.config.port} did not start sending again "
            "after its configuration console was closed, so it is still in "
            "config mode and nothing can see it. " + self._what_the_port_said()
        )

    def _remeasure_rate(self) -> None:
        """Forget the measured packet interval and time the stream again.

        The interval stamps every record's sample axis, so a rate change
        that left the old number in place would describe the new capture by
        the old rate.  The magnetic update count goes with it: at a new
        packet rate the repeat factor is a different number.
        """

        self._stamps.clear()
        self._stamps_ready.clear()
        self._sample_interval = None
        self._magnetic_field = None
        self._magnetic_changes = 0
        self._magnetic_packets = 0
        self._first_bytes.clear()
        self._hear_the_module()

    def _read_settings(self, console) -> dict[str, object]:
        """Every offered parameter this module actually answers to.

        Asked by name: this firmware's bare ``#fparam`` lists nothing, just
        as its bare ``#fmsg`` lists no packets, so the only way to find out
        what it has is to ask for each one.  What it will not answer to, it
        does not have, and is not offered.
        """

        settings: dict[str, object] = {}
        for name in self._parameter_names(console):
            reading = _as_number(console.get_parameter(name))
            if reading is not None:
                settings[name] = reading
        self._packets_listed = IMU_RATE_PARAMETER in settings
        return settings

    def _parameter_names(self, console) -> tuple[str, ...]:
        """The parameters to try, plus any the module volunteers.

        A firmware that DOES answer a bare ``#fparam`` gets its own list
        read as well, filtered to the kinds that belong to an operator, so
        a module with more than this bench knows about is not cut down to
        it.
        """

        found = list(OFFERED_PARAMETERS)
        try:
            answer = console.query("#fparam")
        except Exception:  # noqa: BLE001 -- the tried names still stand
            return tuple(found)
        for line in answer.splitlines():
            name = line.split("=")[0].strip()
            if (
                name.upper().startswith(OPERATOR_PARAMETER_PREFIXES)
                and name not in found
            ):
                found.append(name)
        return tuple(found)

    def _field_for(self, name: str, value: object) -> TunableField:
        kind, unit = parameter_shape(name)
        if kind == "rate":
            index = int(_as_number(value) or 0)
            rungs = PACKET_RATE_LADDER_HZ
            current = rungs[index] if 0 <= index < len(rungs) else 0.0
            return TunableField(
                metadata=AuthoringField(
                    name,
                    "choice",
                    "Packet rate",
                    None,
                    unit="Hz",
                    choices=tuple(
                        AuthoringChoice(f"{rung:g}", f"{rung:g} Hz")
                        for rung in rungs[1:]
                    ),
                    description=(
                        "how often the module sends the packet this bench "
                        "reads. The module takes a rung of its own ladder, "
                        "not a number of hertz, so only the rungs are offered"
                    ),
                ),
                # A module sitting on rung 0 is sending nothing, which this
                # bench cannot see; show the rung it is on rather than
                # inventing one.
                current=f"{current:g}" if current else f"{rungs[1]:g}",
                live_write=True,
                dependency_group=(name,),
            )
        if kind == "switch":
            return TunableField(
                metadata=AuthoringField(
                    name,
                    "choice",
                    name,
                    None,
                    choices=(
                        AuthoringChoice("0", "Off"),
                        AuthoringChoice("1", "On"),
                    ),
                    description="a module switch, read and written with #fparam",
                ),
                current=str(int(float(_as_number(value) or 0))),
                live_write=True,
                dependency_group=(name,),
            )
        return TunableField(
            metadata=AuthoringField(
                name,
                "float",
                name,
                None,
                minimum=0.0,
                unit=unit,
                description="a module parameter, read and written with #fparam",
            ),
            current=float(_as_number(value) or 0.0),
            live_write=True,
            dependency_group=(name,),
        )

    def tunable_fields(self) -> tuple[TunableField, ...]:
        """What the module said it had, the last time it was asked.

        Reading these costs a round trip through the configuration console,
        which stops the stream -- so this answers from the last reading and
        ``refresh_tunable_fields`` is what goes back to the module.
        """

        with self._settings_lock:
            return tuple(
                self._field_for(name, value) for name, value in self._settings.items()
            )

    def refresh_tunable_fields(self) -> tuple[TunableField, ...]:
        """Ask the module again, stopping its stream for the round trip."""

        with self._settings_lock:
            self._settings = self._in_console(self._read_settings)
            return self.tunable_fields()

    def tunable_values(self) -> dict[str, object]:
        with self._settings_lock:
            return {
                field.metadata.name: field.current for field in self.tunable_fields()
            }

    def settings_provenance(self) -> dict[str, object]:
        with self._settings_lock:
            return {
                "device_session_id": self._device_session_id,
                "settings_epoch": self._settings_epoch,
            }

    def tune(self, name: str, value: object) -> object:
        """Write one setting and answer with what the module read back.

        The rate is the one that has to be watched: it is written as a
        ladder INDEX, and a module that took an index this driver did not
        intend can stop sending altogether.  So it is written, re-timed off
        the stream, and undone if the stream does not come back.
        """

        selected = str(name)
        with self._settings_lock:
            if selected not in self._settings:
                offered = ", ".join(repr(key) for key in sorted(self._settings))
                raise ValueError(
                    f"this module has no setting {selected!r}; it offers "
                    f"{offered or 'none -- its configuration console did not answer'}"
                )
            is_rate = selected.upper() == IMU_RATE_PARAMETER
            previous = self._settings.get(selected)
            spelling = (
                str(rate_ladder_index(float(value))) if is_rate else _as_text(value)
            )

            def write(console):
                return _as_number(console.set_parameter(selected, spelling))

            # The rate's own re-timing is a stronger check than "packets came
            # back", so for that one it is the check that runs.
            taken = self._in_console(write, stream_back=not is_rate)
            self._settings_epoch += 1
            if is_rate:
                try:
                    self._remeasure_rate()
                except BaseException as silenced:
                    self._put_back(selected, previous, silenced, value)
                if taken is None:
                    taken = float(spelling)
            if taken is None:
                raise TuneRefused(
                    f"the module acknowledged {selected} but would not read it "
                    "back, so this bench will not record a value it did not read"
                )
            self._settings[selected] = taken
            return self._field_for(selected, taken).current

    def _put_back(
        self, name: str, previous: object, silenced: BaseException, wanted: object
    ) -> None:
        """Undo a write that stopped the module sending, and say what happened.

        Writing the previous value back is tried first, but it cannot be
        relied on: if the module read the new value in a spelling this
        driver did not intend, it will read the old one the same way.  The
        undo that does NOT depend on the spelling is a restart, because
        nothing here has been saved to flash -- so the module comes back on
        the configuration it booted with.
        """

        def write_back(console):
            if previous is None:
                raise RuntimeError("nothing to put back")
            return console.set_parameter(name, _as_text(previous))

        restored = False
        for undo in (write_back, lambda console: console.reboot()):
            try:
                self._in_console(undo, stream_back=False)
            except BaseException:  # noqa: BLE001 -- the next undo is the answer
                pass
            for _ in range(_UNDO_LISTEN_ROUNDS):
                try:
                    self._remeasure_rate()
                    restored = True
                    break
                except BaseException:  # noqa: BLE001 -- keep listening
                    continue
            if restored:
                break
        said = self._last_exchange[1].strip()
        if restored:
            raise TuneRefused(
                f"asking this module for {wanted} stopped it sending; "
                f"{name} has been put back and it is streaming again. The "
                f"module answered {said[:120]!r}."
            )
        raise RuntimeError(
            f"asking this module for {wanted} stopped it sending, and "
            f"neither writing {name} back nor restarting brought the stream "
            f"back. The module answered "
            f"{said[:120]!r}. Power-cycle the module: nothing was written to "
            "its flash, so a cold start restores the configuration it had."
        ) from silenced

    def save_settings(self) -> str:
        """Commit the module's settings to its flash, and say what it answered.

        Nothing reports what is in flash, so this cannot be verified the way
        a written setting is read back; the module's own reply is handed
        on rather than judged here.
        """

        return self._in_console(lambda console: console.save())

    def arm(self, records: int | None, *, buffer_record_count: int) -> None:
        self._records.arm(records, buffer_record_count=buffer_record_count)

    def read_records(
        self, n: int, *, timeout: float, exact: bool
    ) -> list[WaveformRecord]:
        return self._records.read(n, timeout=timeout, exact=exact)

    def finish_record_capture(self) -> WaveformCaptureTerminalRecord:
        return self._records.finish()

    def capture_state(self) -> bool:
        return self._records.armed

    def close(self) -> None:
        """Stop reading and release the port; the handle goes only once it has.

        Taken under the settings lock: a configuration round trip owns the
        port for its duration, and closing the handle out from under it
        would leave the module in config mode -- silent, with nobody left
        to send it ``#fdeconfig``.
        """

        with self._settings_lock:
            self._stop.set()
            try:
                if self._records.armed:
                    self._records.finish()
            finally:
                if (
                    self._reader.is_alive()
                    and self._reader is not threading.current_thread()
                ):
                    self._reader.join(timeout=self.config.timeout_seconds)
                self._serial.close()


def _listen_for_packets(port, listen_seconds: float) -> int:
    """How many whole IMU packets frame up on this port in a moment."""

    buffer = bytearray()
    packets = 0
    deadline = time.monotonic() + float(listen_seconds)
    while time.monotonic() < deadline and packets < 2:
        waiting = port.in_waiting
        chunk = port.read(waiting if waiting else 1)
        if chunk:
            buffer += chunk
            packets += len(drain_imu_samples(buffer))
    return packets


def discover_n100(*, baud: int = DEFAULT_BAUD, listen_seconds: float = 0.5) -> tuple[str, ...]:
    """Every serial port with an FDILink IMU talking on it, found by listening.

    The module has no identification query; what identifies it is the
    stream itself, so each port that can be opened is listened to for a
    moment and kept if whole IMU packets frame up on it.  A port another
    program holds cannot be opened and is passed over, which is the ordinary
    case on a bench whose pulse board owns one.

    A port that says nothing is asked once more, after ``#fdeconfig``: a
    module left in its configuration console is silent, and silence is
    exactly what this function reads as "not an N100".  Without that line a
    module whose console was opened and never closed would disappear from
    every scan until someone power-cycled it.
    """

    from serial.tools import list_ports

    found: list[str] = []
    for info in sorted(list_ports.comports(), key=lambda item: item.device):
        try:
            port = _open_serial(info.device, baud)
        except Exception:
            continue
        try:
            packets = _listen_for_packets(port, listen_seconds)
            if packets < 2:
                wake_from_config_mode(port)
                packets = _listen_for_packets(port, listen_seconds)
        except Exception:
            continue
        finally:
            try:
                port.close()
            except Exception:
                pass
        if packets >= 2:
            found.append(info.device)
    return tuple(found)


__all__ = [
    "DEFAULT_BAUD",
    "IMU_RATE_PARAMETER",
    "OFFERED_PARAMETERS",
    "PACKET_RATE_LADDER_HZ",
    "parameter_shape",
    "rate_ladder_index",
    "MAX_PACKET_RATE_HZ",
    "N100_OUTPUTS",
    "OPERATOR_PARAMETER_PREFIXES",
    "WheeltecN100Config",
    "WheeltecN100WaveformSource",
    "discover_n100",
    "drain_imu_samples",
    "header_crc8",
    "payload_crc16",
    "wake_from_config_mode",
]
