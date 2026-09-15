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

from zlc_atom.authoring import AuthoringField, TunableField
from zlc_atom.devices.waveform.contract import (
    WaveformAcquisitionMode,
    WaveformCaptureTerminalRecord,
    WaveformOutput,
    WaveformRecord,
    WaveformRecordQueue,
    WaveformWorkingPoint,
)

from .console import FdiConfigConsole


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

#: The highest rate any shipped firmware's ladder reaches, used only as the
#: outer edge of what Device Control will let an operator TYPE.  What this
#: particular module accepts is its own answer: ``#fmsg`` echoes the rung it
#: actually took, and that echo is what the bench records.  An N-class
#: module tops out at 400 Hz on the IMU packet and 200 Hz on everything
#: else; a larger firmware reaches 1000.
MAX_PACKET_RATE_HZ = 1000.0

#: Which KINDS of named parameter belong to an operator rather than to the
#: factory.  Filters shape what the sensors report -- a notch on the mains
#: frequency is the reason this matters for a magnetic measurement -- and
#: the AID switches decide which sensors the attitude solution fuses.  The
#: scale factors, cross-axis terms and biases beside them are calibration,
#: and are deliberately not offered.
OPERATOR_PARAMETER_PREFIXES = ("FILT_", "AID_")

#: How a packet's rate is named among the settings.  The module's own packet
#: id is in the name, because the NAME differs between firmwares while the
#: id is the protocol's.
_PACKET_RATE_PREFIX = "packet_"
_PACKET_RATE_SUFFIX = "_rate_hz"


def packet_rate_field(packet_id: int) -> str:
    """The settings name for one packet's transmit rate."""

    return f"{_PACKET_RATE_PREFIX}{int(packet_id):02x}{_PACKET_RATE_SUFFIX}"


def packet_id_of(name: str) -> int | None:
    """The packet a settings name belongs to, or None for a named parameter."""

    text = str(name)
    if not (text.startswith(_PACKET_RATE_PREFIX) and text.endswith(_PACKET_RATE_SUFFIX)):
        return None
    digits = text[len(_PACKET_RATE_PREFIX):-len(_PACKET_RATE_SUFFIX)]
    try:
        return int(digits, 16)
    except ValueError:
        return None


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


def _open_serial(port: str, baud: int):
    import serial

    return serial.Serial(port, baud, timeout=0.05, write_timeout=1.0)


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
        self._magnetic_field: tuple[float, float, float] | None = None
        self._magnetic_changes = 0
        self._magnetic_packets = 0
        #: What the module last said its settings were.  Empty until it has
        #: been asked, and empty for good on a module whose firmware does
        #: not answer the configuration console -- such a module streams
        #: perfectly well and simply has nothing an operator can turn.
        self._settings: dict[str, object] = {}
        self._settings_refusal: str | None = None
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
            self._await_rate()
            self._read_settings_once()
        except BaseException:
            self.close()
            raise

    def _read_settings_once(self) -> None:
        """Ask the module for its settings, and let a refusal be a fact.

        A module that will not answer its configuration console still
        streams; what it does not have is anything for an operator to turn.
        Recording WHY is what turns a blank Device Control page into an
        answer.
        """

        try:
            self._settings = self._in_console(self._read_settings)
        except Exception as refusal:  # noqa: BLE001 -- reported, not raised
            self._settings = {}
            self._settings_refusal = f"{type(refusal).__name__}: {refusal}"
        finally:
            # Config mode stopped the stream either way, so the interval
            # that stamps the records is timed again before anyone reads.
            self._remeasure_rate()

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
                samples = drain_imu_samples(buffer)
                if samples:
                    self._accept(samples)
        except BaseException as error:  # noqa: BLE001 -- surfaced to the reader of records
            self._records.fail(error)
            self._stamps_ready.set()

    def _accept(self, samples: list[tuple[float, tuple[float, ...]]]) -> None:
        received = time.time_ns()
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

    def _await_rate(self) -> None:
        self._stamps_ready.wait(_RATE_SAMPLE_SECONDS)
        failure = self._records.failure
        if failure is not None:
            raise RuntimeError(f"reading {self.config.port} failed") from failure
        if len(self._stamps) < 2:
            raise RuntimeError(
                f"no FDILink IMU packets arrived on {self.config.port} at "
                f"{self.config.baud} baud within {_RATE_SAMPLE_SECONDS:g} s: check "
                "the port, the module's power and that its line rate is "
                f"{self.config.baud}"
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

    def _in_console(self, work):
        """Run ``work(console)`` with the module in its configuration console.

        Entering stops the stream, so this refuses while a capture is armed:
        a capture that lost its packets mid-flight would publish a gap no
        reader could tell from the module having gone quiet.
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
                    return work(console)
            finally:
                self._release_reader()

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
        self._await_rate()

    def _read_settings(self, console) -> dict[str, object]:
        """Everything this module says it has: packet rates and parameters."""

        settings: dict[str, object] = {}
        for packet in console.packet_rates():
            settings[packet_rate_field(packet.packet_id)] = packet.rate_hz
        for name in self._parameter_names(console):
            reading = console.get_parameter(name)
            number = _as_number(reading)
            if number is not None:
                settings[name] = number
        return settings

    def _parameter_names(self, console) -> tuple[str, ...]:
        """The parameters worth an operator's attention, as the module lists them.

        The module is asked to print its parameters and the answer is
        filtered by PREFIX, not against a list of names: names differ
        between firmwares -- the protocol manual's own worked example names
        one that the shipped parameter tables do not have -- so a pinned
        list would be a second source of truth that a firmware revision
        falsifies.  The prefixes say which KINDS of setting belong to the
        operator: the digital filters that shape what the sensors report,
        and the fusion switches that decide which sensors the attitude
        solution uses.
        """

        answer = console.command("#fparam")
        found: list[str] = []
        for line in answer.splitlines():
            name = line.split("=")[0].strip()
            if name.upper().startswith(OPERATOR_PARAMETER_PREFIXES) and name not in found:
                found.append(name)
        return tuple(found)

    def _field_for(self, name: str, value: object) -> TunableField:
        packet = packet_id_of(name)
        if packet is not None:
            return TunableField(
                metadata=AuthoringField(
                    name,
                    "float",
                    f"Packet 0x{packet:02X} rate",
                    None,
                    minimum=0.0,
                    maximum=MAX_PACKET_RATE_HZ,
                    unit="Hz",
                    description=(
                        "how often the module sends this packet; 0 turns it "
                        "off.  The module answers with the rate it actually "
                        "took, which is its own ladder and not a table here"
                    ),
                ),
                current=float(value),
                live_write=True,
                dependency_group=(name,),
            )
        return TunableField(
            metadata=AuthoringField(
                name,
                "float",
                name,
                None,
                description="a module parameter, read and written with #fparam",
            ),
            current=float(value),
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

        The module is the authority on what it took: a rate its firmware
        does not offer comes back as the nearest rung it does offer, and
        that returned number -- not the request -- is what the bench
        records.
        """

        selected = str(name)
        with self._settings_lock:
            if selected not in self._settings:
                offered = ", ".join(repr(key) for key in sorted(self._settings))
                raise ValueError(
                    f"this module has no setting {selected!r}; it offers "
                    f"{offered or 'none -- its configuration console did not answer'}"
                )
            packet = packet_id_of(selected)

            def write(console):
                if packet is not None:
                    return console.set_packet_rate(packet, float(value))
                return _as_number(console.set_parameter(selected, _as_text(value)))

            taken = self._in_console(write)
            if taken is None:
                raise RuntimeError(
                    f"the module took {selected} but would not say what to"
                )
            self._settings[selected] = taken
            self._settings_epoch += 1
            if packet == IMU_PACKET:
                # This is the rate the records are stamped at, so it is
                # measured off the stream again rather than believed.
                self._remeasure_rate()
            return taken

    def save_settings(self) -> None:
        """Commit the module's settings to its flash, to survive power-off."""

        self._in_console(lambda console: console.save())

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
        """Stop reading and release the port; the handle goes only once it has."""

        self._stop.set()
        try:
            if self._records.armed:
                self._records.finish()
        finally:
            if self._reader.is_alive() and self._reader is not threading.current_thread():
                self._reader.join(timeout=self.config.timeout_seconds)
            self._serial.close()


def discover_n100(*, baud: int = DEFAULT_BAUD, listen_seconds: float = 0.5) -> tuple[str, ...]:
    """Every serial port with an FDILink IMU talking on it, found by listening.

    The module has no identification query; what identifies it is the
    stream itself, so each port that can be opened is listened to for a
    moment and kept if whole IMU packets frame up on it.  A port another
    program holds cannot be opened and is passed over, which is the ordinary
    case on a bench whose pulse board owns one.
    """

    from serial.tools import list_ports

    found: list[str] = []
    for info in sorted(list_ports.comports(), key=lambda item: item.device):
        try:
            port = _open_serial(info.device, baud)
        except Exception:
            continue
        try:
            buffer = bytearray()
            packets = 0
            deadline = time.monotonic() + float(listen_seconds)
            while time.monotonic() < deadline and packets < 2:
                waiting = port.in_waiting
                chunk = port.read(waiting if waiting else 1)
                if chunk:
                    buffer += chunk
                    packets += len(drain_imu_samples(buffer))
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
    "MAX_PACKET_RATE_HZ",
    "N100_OUTPUTS",
    "OPERATOR_PARAMETER_PREFIXES",
    "WheeltecN100Config",
    "WheeltecN100WaveformSource",
    "discover_n100",
    "drain_imu_samples",
    "header_crc8",
    "packet_id_of",
    "packet_rate_field",
    "payload_crc16",
]
