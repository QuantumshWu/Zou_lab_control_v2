"""WHEELTEC N100 nine-axis IMU, read off its FDILink serial stream.

The module talks first: from the moment it has power it emits binary
FDILink packets on its serial port at the rate it was configured to (the
factory default is 100 Hz).  One IMU packet carries the three gyroscope
axes, the three accelerometer axes, the three magnetometer axes, the die
temperature, the barometer pair and the module's own microsecond clock, so
this source publishes four quantities from one record column set.

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

Opening this source asks the module NOTHING.  The rate is measured off
the stream that is already arriving -- those packets are what every record
gets stamped with, so they are the truth about the rate, and a trip
through the console would cost seconds to be told something less true.

The console is entered only to WRITE, when the operator changes the rate
in Device Control.  ``console.py`` speaks it; entering stops the stream,
so the reader is parked for the round trip, and what the module comes back
DOING is read off the stream afterwards rather than asked for.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import struct
import threading
import time
from uuid import uuid4

import numpy as np

from zlc_atom.devices import RecordQueue
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
    WaveformWorkingPoint,
)

from .console import IMU_PACKET_NAME, LINE_END, FdiConfigConsole


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

#: How this firmware spells a packet's rate: an INDEX into this ladder,
#: written to a named parameter, never a number of hertz.  The manual's
#: ``#fmsg 40 100`` is answered ``*#OK`` and changes nothing; the vendor's
#: own parameter cache gives ``MSG_IMU`` the values 0..9 against exactly
#: these rungs, and its ground station writes that parameter.
#:
#: The ladder is also the ceiling, so nothing else holds one: 400 Hz is the
#: top rung and the documented maximum -- 用户手册 and the FAQ both answer
#: "数据包发布的最大频率为400HZ" -- while the 1000 Hz the same manual
#: quotes is the internal sampling rate and reaches no packet.  Rung 0 is
#: "no output"; ``offered_rungs`` says why it and the slow rungs are not
#: offered.
PACKET_RATE_LADDER_HZ = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0)

#: The slowest rate the IMU packet may be set to.  Discovery recognises
#: this module by hearing whole packets, and it listens for a moment --
#: below this, a scan can pass over a module that is working perfectly,
#: and since applying a rate writes it to flash, the module would be
#: invisible from then on, power cycle included.  Rung 0 is excluded for
#: the same reason; these are the same hazard at different speeds.
SLOWEST_DISCOVERABLE_HZ = 5.0


def offered_rungs(standing_at: float | None = None) -> tuple[float, ...]:
    """The rates the IMU packet may be set to, and always the one it is ON.

    The slow end of the ladder is withheld, and every rung withheld is a
    way of losing the module: this bench recognises an N100 by hearing
    whole packets and it listens for a moment, so a packet turned off --
    rung 0 -- or slowed below what a scan can hear is a module no scan will
    find again.  Applying a rate writes it to FLASH, so that is permanent,
    power cycle included.  One comparison covers both, because rung 0 is
    simply the slowest rung of all.

    But policy says where an operator may MOVE the module, never what the
    module is DOING.  A module standing on a rung this driver would not
    choose -- the vendor's ground station can set one, and rungs 1 and 2 Hz
    are on the module's own ladder -- is still running that rate, and a
    panel that cannot render it does not open AT ALL: the whole Device
    Control window fails, taking every knob that answered with it.  So what
    the module is on is always offerable, whatever policy thinks of it.
    """

    allowed = [
        rung for rung in PACKET_RATE_LADDER_HZ if rung >= SLOWEST_DISCOVERABLE_HZ
    ]
    if standing_at is not None and not any(
        abs(rung - float(standing_at)) < 1e-6 for rung in allowed
    ):
        allowed.append(float(standing_at))
    return tuple(sorted(allowed))


def rate_ladder_index(rate_hz: float, *, standing_at: float | None = None) -> int:
    """The ladder rung an asked-for rate belongs to, or a refusal.

    Only the rungs are offerable, so a value off the ladder is refused here
    -- before anything reaches the module -- rather than sent and silently
    read as something else.  ``standing_at`` is the rate the module is on,
    which is always allowed: putting a module back where it already was
    cannot be a new way of losing it, and the undo path depends on that.
    """

    wanted = float(rate_hz)
    allowed = offered_rungs(standing_at)
    for index, rung in enumerate(PACKET_RATE_LADDER_HZ):
        if abs(rung - wanted) < 1e-6 and rung in allowed:
            return index
    offered = ", ".join("off" if rung == 0 else f"{rung:g}" for rung in allowed)
    raise TuneRefused(
        f"{wanted:g} Hz is not one of this module's rates; it offers {offered}"
    )


def _as_number(value: object) -> float | None:
    """The module's answer as a number, or None when it is not one."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _spelling_of(value: object, *, standing_at: object = None) -> str:
    """How the rate is written on the wire: its rung's index, never hertz.

    The settings map holds it in hertz, because that is what an operator
    means by a sample rate, so this is the one place the two spellings
    meet.  It is used by the write and by the undo alike, which is why
    ``standing_at`` matters: an undo puts back a rate the module was
    demonstrably running, and refusing it there would leave the module on
    the value that silenced it while reporting that the undo was tried.
    """

    return str(rate_ladder_index(float(value), standing_at=_as_number(standing_at)))


#: How far a measured rate may sit from a rung and still BE that rung.
#: A quarter, and no absolute floor: the rungs are a factor of two or more
#: apart, so a quarter of each never reaches its neighbour -- 1 Hz owns
#: 0.75..1.25 and 2 Hz owns 1.5..2.5, with a gap between them.  A floor of
#: one hertz, which is what this had, made those two bands OVERLAP, and
#: since the search ran up the ladder a stream measured at 2 Hz was named
#: 1 Hz.  Nothing needs a floor: the measurement is the median of intervals
#: off the module's own microsecond clock, not a stopwatch.
_RUNG_TOLERANCE = 0.25


def _rung_the_stream_is_on(measured_hz: float) -> float | None:
    """Which rung a measured stream rate is standing on, or None.

    The NEAREST rung, and only when the measurement is near enough to be
    that rung rather than the gap between two.  That is what lets the
    STREAM answer for the rate after a restart: the packets arriving are
    what this bench stamps its records with, and asking the console instead
    would cost another trip through it to be told something less true.
    """

    nearest: tuple[float, float] | None = None
    for rung in PACKET_RATE_LADDER_HZ:
        if rung <= 0.0:
            continue
        gap = abs(measured_hz - rung)
        if nearest is None or gap < nearest[0]:
            nearest = (gap, rung)
    if nearest is None:
        return None
    gap, rung = nearest
    return rung if gap <= rung * _RUNG_TOLERANCE else None


def _apply(console, name: str, spelling: str) -> None:
    """Write one setting so that it actually takes effect.

    Measured on a real module: writing the parameter alone changes nothing
    -- ``#fparam get`` returns the new value and the module goes on sending
    at the old rate, because the running configuration is not the parameter
    table.  It takes hold when the table is committed to flash and the
    module restarts, which is what the vendor's own ground station does
    behind its Save and Restart buttons.

        #fparam set MSG_IMU 7     ->  *#OK
        #fparam get MSG_IMU       ->  MSG_IMU=7      (written, not yet live)
        #fsave                    ->  *#OK
        #freboot                  ->  (y/n)
        y                         ->  back in about 2.5 s, now at 100 Hz

    ``set_parameter`` reads the value back before the restart, because that
    is where a readback can still be had, and it proves only that the TABLE
    took it.  What the module then DOES is the caller's to check, and it is
    checked off the stream rather than asked for.
    """

    console.set_parameter(name, spelling)
    try:
        console.save()
    except BaseException:
        # The table now holds a value that was never committed, and this
        # module commits the WHOLE table: the next #fsave anybody runs --
        # the operator's own Save button included -- would put it in flash
        # and the restart after that would make it live, with nothing ever
        # having shown it.  A restart WITHOUT a save is exactly how this
        # firmware discards an uncommitted table, so that is the undo.
        try:
            console.reboot()
        except BaseException:  # noqa: BLE001 -- the save's refusal is the news
            pass
        raise
    console.reboot()


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


def drain_imu_samples(
    buffer: bytearray, *, on_frame: Callable[[int | None, tuple | None], None] | None = None,
) -> list[tuple[float, tuple[float, ...]]]:
    """Take every complete packet off the front of ``buffer``.

    Returns ``(timestamp_seconds, samples)`` for each IMU packet, in arrival
    order, with the samples already in the published units and column
    order. When ``on_frame`` is supplied, deliver each validated frame there
    instead, including non-IMU frames with no sample and damaged frames with
    no serial, so an active capture can reject data loss. Every other packet
    type is stepped over by the length its own
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
            if on_frame is not None:
                on_frame(None, None)
            at = head + 1
            continue
        kind = buffer[head + 1]
        size = buffer[head + 2]
        end = head + HEADER_LENGTH + size + 1
        if total < end:
            at = head
            break
        if buffer[end - 1] != FRAME_TAIL:
            if on_frame is not None:
                on_frame(None, None)
            at = head + 1
            continue
        payload = bytes(buffer[head + HEADER_LENGTH:head + HEADER_LENGTH + size])
        if payload_crc16(payload) != (buffer[head + 5] << 8 | buffer[head + 6]):
            if on_frame is not None:
                on_frame(None, None)
            at = head + 1
            continue
        if kind == IMU_PACKET and size == IMU_PAYLOAD_LENGTH:
            values = _IMU_PAYLOAD.unpack(payload)
            stamp = values[12] * 1e-6
            sample = (
                stamp,
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
            if on_frame is None:
                found.append(sample)
            else:
                on_frame(buffer[head + 3], sample)
        elif on_frame is not None:
            on_frame(buffer[head + 3], None)
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
_UNDO_LISTEN_ROUNDS = 2

#: How long to wait for whole packets after the console closes.  A module
#: that is going to resume does it at once; this is the margin, not the
#: expectation.
_STREAM_BACK_SECONDS = 1.5

#: And after a RESTART, which is what makes a setting take effect.  A real
#: module came back in 2.5 s; this is that with room to spare, and no more
#: -- every second here is a second an operator waits on a settings page,
#: and a second the undo path waits again before it gives up.
_RESTART_SECONDS = 8.0

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
        self._capture_lock = threading.Lock()
        self._last_frame_serial: int | None = None
        self._last_imu_stamp: float | None = None
        self._device_session_id = uuid4().hex
        self._settings_epoch = 0
        self._stamps: deque[float] = deque(maxlen=_RATE_SAMPLE_PACKETS)
        self._stamps_ready = threading.Event()
        self._sample_interval: float | None = None
        # How many of the packets counted so far carried a magnetic field
        # that had actually moved since the packet before.  See
        # ``magnetic_change_interval``.
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
        #: The last console command and its verbatim reply, for the record.
        self._last_exchange: tuple[str, str] = ("", "")
        self._records = RecordQueue(
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
            self._note_the_rate()
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

    def _note_the_rate(self) -> None:
        """Take the one setting off the STREAM, which is already timed.

        The rate is the only thing this bench turns, and the packets
        arriving say what it is: ``_hear_the_module`` has just measured the
        interval, because it stamps every record, and the ladder's rungs
        are a factor of two apart so a measurement names one exactly.

        Asking the console instead would cost what the console costs -- the
        module refuses commands that crowd each other, so every one of them
        is seconds apart -- and it would stop the stream to do it, and what
        it would come back with is what the module MEANS to send.  The
        packets are what this bench actually records.
        """

        interval = self._sample_interval
        measured = 1.0 / interval if interval else 0.0
        rung = _rung_the_stream_is_on(measured)
        if rung is None:
            self._settings = {}
            self._settings_refusal = (
                f"the module on {self.config.port} is sending {measured:.1f} Hz, "
                "which is not one of its rates, so there is nothing here to turn"
            )
            return
        self._settings = {IMU_PACKET_NAME: rung}
        self._settings_refusal = ""

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
                with self._capture_lock:
                    drain_imu_samples(buffer, on_frame=self._accept_frame)
        except BaseException as error:  # noqa: BLE001 -- surfaced to the reader of records
            self._records.fail(error)
            self._stamps_ready.set()

    def _accept_frame(self, serial: int | None, sample: tuple | None) -> None:
        """A capture must not hide missing frames or a reset device clock."""

        if self._records.accepting:
            if serial is None:
                self._records.fail(RuntimeError("N100 damaged FDILink frame during capture"))
                return
            previous = self._last_frame_serial
            if previous is not None and serial != (previous + 1) & 0xFF:
                self._records.fail(RuntimeError(f"N100 frame sequence gap: expected {(previous + 1) & 0xFF}, received {serial}"))
                return
            self._last_frame_serial = serial
        if sample is None:
            return
        stamp, values = sample
        if self._records.accepting:
            previous_stamp = self._last_imu_stamp
            if previous_stamp is not None:
                elapsed = stamp - previous_stamp
                if elapsed <= 0:
                    self._records.fail(RuntimeError(f"N100 device timestamp did not advance: {previous_stamp} -> {stamp}"))
                    return
                # More than half a packet period beyond the measured interval
                # is no longer the next sample. Do not bridge that gap.
                if self._sample_interval is not None and elapsed > 1.5 * self._sample_interval:
                    self._records.fail(RuntimeError(f"N100 device timestamp gap: {elapsed:g} s at {1 / self._sample_interval:g} Hz"))
                    return
            self._last_imu_stamp = stamp
        received = time.time_ns()
        self._last_packet_at = time.monotonic()
        self._stamp(stamp)
        self._watch_magnetic(values[0:3])
        if self._records.accepting:
            self._records.push(WaveformRecord(
                np.asarray(values, dtype=np.float32).reshape(1, _COLUMNS),
                self._records.produced_count, stamp, received,
            ))

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
        """Count how often the published magnetic field CHANGES.

        The three magnetometer axes ride inside the IMU packet, so they are
        SENT at the packet rate whether or not anything new went into them.
        Nothing the vendor ships states the magnetometer's own output rate
        -- its specification table is a verbatim lift from another
        manufacturer's part -- so what is counted here is the one thing
        this bench can actually see: how often the three published values
        differ from the ones before.

        That is NOT the magnetometer's rate, in either direction, and it
        must not be read as a bound on it.  Two independent conversions can
        land on the same quantised value, so a repeat does not mean no new
        sample was taken; and whatever filtering or fusion sits between the
        sensor and the packet can make the published number move on packets
        where no conversion happened at all, so a change does not mean one
        did.  It is the rate of change of what is published, and that is
        all it is.
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
    def magnetic_change_interval(self) -> float | None:
        """Seconds between packets whose published magnetic field DIFFERS.

        An OBSERVATION, not a rate of anything in the instrument.  It is
        neither an upper nor a lower bound on how fast the magnetometer
        converts: a repeated value may be two conversions landing on the
        same quantised number, and a changed value may be filtering moving
        the published figure on a packet where nothing was converted.  It
        must not be reported as an ODR, and nothing about rise time can be
        read out of it.

        What it does answer is the question the datasheet does not, and the
        only one this bench is entitled to ask: whether the field being
        PLOTTED is carrying new numbers as fast as the packets arrive.  For
        drawing a field, a repeated value is no new information whatever
        produced it.

        It is counted over the opening window only -- the packets the rate
        is timed from -- so it describes how the module was behaving when
        it was opened, not how it is behaving now.
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
                "magnetic_change_hz": (
                    None
                    if self.magnetic_change_interval is None
                    else round(1.0 / self.magnetic_change_interval, 3)
                ),
                "settings": dict(self._settings),
                "settings_refusal": self._settings_refusal,
                "last_console_exchange": self._last_exchange,
            },
            time_basis="device_clock",
        )


    # -------------------------------------------------------------- knobs
    def _park_reader(self) -> None:
        """Take the port away from the reader, or say why it could not be."""

        self._park.set()
        if self._parked.wait(self.config.timeout_seconds):
            return
        self._park.clear()
        # A reader that died never acknowledges the park, and saying it is
        # holding the port sends the operator after a thread that holds
        # nothing -- while the serial failure that actually stopped it sits
        # unread on the queue.
        failure = self._records.failure
        if failure is not None:
            raise RuntimeError(
                f"reading {self.config.port} failed, so its settings cannot "
                "be read or written"
            ) from failure
        if not self._reader.is_alive():
            raise RuntimeError(
                f"nothing is reading {self.config.port} any more, so this "
                "source is closed and its settings cannot be reached"
            )
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

    def _require_stream_back(self, window: float = _STREAM_BACK_SECONDS) -> None:
        """Wait for whole packets again, asking once more if they do not come.

        Whole packets, not a frame header: this module emits a 1 Hz
        heartbeat whose first byte is a frame header too, so a header alone
        would have read a silent module as a navigating one.
        """

        # Two goes, always.  This used to be one whenever the window was
        # long, which is only ever the restart path -- the very path where a
        # module is most likely to need telling to come back out.
        attempts = 2
        for attempt in range(attempts):
            mark = time.monotonic()
            deadline = mark + window
            while time.monotonic() < deadline:
                if self._last_packet_at >= mark:
                    return
                time.sleep(0.005)
            if attempt + 1 < attempts:
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

    def _field_for(self, name: str, value: object) -> TunableField:
        """The rate, as a choice of the rungs its module actually offers."""

        # Held in hertz, because that is what a sample rate means to an
        # operator; WRITTEN as the rung's index, which is what the module
        # takes.
        current = float(_as_number(value) or 0.0)
        # Always including the rung the module is standing on: a rate this
        # driver would not have chosen is still the rate it found, and a
        # choice list that cannot express it fails the whole Device Control
        # window rather than one row.
        rungs = offered_rungs(current)
        return TunableField(
            metadata=AuthoringField(
                name,
                "choice",
                "Acquisition rate",
                None,
                unit="Hz",
                choices=tuple(
                    AuthoringChoice(f"{rung:g}", f"{rung:g} Hz") for rung in rungs
                ),
                description=(
                    "how often the module sends the packet this bench reads, "
                    "and therefore this bench's sample rate. It cannot be "
                    f"turned off or set below {SLOWEST_DISCOVERABLE_HZ:g} Hz: "
                    "a scan finds this module by those frames. The module "
                    "takes a rung of its own ladder, not a number of hertz, "
                    "so only the rungs are offered, and a change costs a "
                    "save and a restart"
                ),
            ),
            current=f"{current:g}",
            live_write=True,
            dependency_group=(name,),
        )

    def tunable_fields(self) -> tuple[TunableField, ...]:
        """The rate, as the stream last showed it.

        There is deliberately no ``refresh_tunable_fields`` here, and that
        is not an omission.  A refresh exists for an instrument whose front
        panel somebody can turn by hand behind the software's back; this
        module has no front panel, its settings move only through the
        console, and nothing can reach that console while this driver holds
        the port.  So this answer is already the current one, and a refresh
        would stop the stream for seconds to be told what it already knows
        -- which is exactly what made opening the settings page take ten of
        them.

        ``zlc_atom.authoring.refresh_tunable_fields`` falls back to this
        when a device declares no refresh, so the panel gets its answer at
        once.
        """

        with self._settings_lock:
            return tuple(
                self._field_for(name, value) for name, value in self._settings.items()
            )

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
            previous = self._settings.get(selected)
            spelling = _spelling_of(value, standing_at=previous)

            self._in_console(
                lambda console: _apply(console, selected, spelling),
                stream_back=False,
            )
            self._settings_epoch += 1
            try:
                self._require_stream_back(_RESTART_SECONDS)
            except BaseException as silenced:
                self._put_back(selected, previous, silenced, value)
            # The stream is the answer, and it costs nothing: the interval
            # has to be re-timed anyway, because it stamps every record.
            # Going back into the console to ask would cost another trip --
            # entering, a command, leaving, ten seconds of an operator's
            # time -- to be told what the module MEANS to send rather than
            # what it is sending.
            self._remeasure_rate()
            measured = 1.0 / self._sample_interval if self._sample_interval else 0.0
            taken = _rung_the_stream_is_on(measured)
            if taken is None:
                raise TuneRefused(
                    f"the module came back sending {measured:.1f} Hz, which is "
                    "not one of its rates; this bench will not stamp records "
                    "with a rate it cannot name"
                )
            wanted = _as_number(value)
            if wanted is not None and abs(taken - wanted) > 1e-6:
                raise TuneRefused(
                    f"asked this module for {wanted:g} Hz and it came back "
                    f"sending {taken:g} Hz"
                )
            self._settings[selected] = taken
            return self._field_for(selected, taken).current

    def _put_back(
        self, name: str, previous: object, silenced: BaseException, wanted: object
    ) -> None:
        """Undo a write that stopped the module sending, and say what happened.

        The bad value is already IN FLASH: applying a setting saves before
        it restarts, so there is nothing for a bare restart to discard -- it
        would reload the very value that silenced the module, and so would
        a power cycle.  The only undo is to write the previous value the
        same five-step way, and if that will not take, to say plainly that
        the module needs its rate put back by other means rather than to
        recommend a power cycle that boots straight back into silence.
        """

        def write_back(console):
            if previous is None:
                raise RuntimeError("nothing to put back")
            # In the settings a rate is held in HERTZ, because that is how
            # the module reports it; on the wire it is a rung's index.  The
            # undo has to spell it the same way the write did, or it asks
            # for rung 10 when it means 10 Hz -- off the end of the ladder,
            # which is another way of saying "stop sending".
            _apply(console, name, _spelling_of(previous, standing_at=previous))

        restored = False
        for _ in range(_UNDO_LISTEN_ROUNDS):
            try:
                self._in_console(write_back, stream_back=False)
            except BaseException:  # noqa: BLE001 -- keep trying, then report
                pass
            try:
                self._require_stream_back(_RESTART_SECONDS)
                self._remeasure_rate()
                restored = True
                break
            except BaseException:  # noqa: BLE001 -- one more go
                continue
        said = self._last_exchange[1].strip()
        if restored:
            raise TuneRefused(
                f"asking this module for {wanted} stopped it sending; "
                f"{name} has been put back and it is streaming again. The "
                f"module answered {said[:120]!r}."
            )
        raise RuntimeError(
            f"asking this module for {wanted} stopped it sending, and writing "
            f"{name} back did not bring the stream returned. The module "
            f"answered {said[:120]!r}. A POWER CYCLE WILL NOT HELP: applying "
            "a setting writes it to flash before restarting, so the module "
            f"boots into this same state. Put {name} back with the vendor's "
            "ground station, or over its serial console by hand."
        ) from silenced

    def arm(self, records: int | None, *, buffer_record_count: int) -> None:
        """Take the queue, under the lock a settings round trip holds.

        Without the lock, "a capture must be finished first" is decided
        against a queue that can arm a moment later: the console then stops
        the stream under a capture that believes it owns it, or a write
        that has already reached flash is reported as never having
        happened.
        """

        with self._settings_lock:
            with self._capture_lock:
                self._records.arm(records, buffer_record_count=buffer_record_count)
                self._last_frame_serial = None
                self._last_imu_stamp = None
                if not self._reader.is_alive():
                    self._records.fail(RuntimeError("the N100 receive thread is not running"))
                else:
                    self._records.mark_ready()
            try:
                self._records.wait_ready(self.timeout)
            except BaseException:
                self._records.finish()
                raise

    def read_records(
        self, n: int, *, timeout: float, exact: bool
    ) -> list[WaveformRecord]:
        return self._records.read(n, timeout=timeout, exact=exact)

    def finish_record_capture(self) -> WaveformCaptureTerminalRecord:
        with self._capture_lock:
            produced = self._records.finish()
            return WaveformCaptureTerminalRecord(produced, True, not self._records.pending_count, True)

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
                if self._reader.is_alive():
                    raise RuntimeError("N100 receive thread did not stop; serial port retained")
                self._serial.close()
                self._records.close()


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

    A port that says nothing is asked once more, after ``#fdeconfig`` -- a
    module left in its configuration console is silent, and silence is
    exactly what this function otherwise reads as "not an N100".  But that
    line is written ONLY once listening alone has found nothing at all,
    because a scan runs against every serial port on the bench: the pulse
    board's side-channel, an SLM, whatever else is idle.  Twelve ASCII
    bytes into one of those is not something to do on the off-chance, so it
    is done only when the alternative is not finding the module.
    """

    from serial.tools import list_ports

    found: list[str] = []
    silent: list[str] = []
    for info in sorted(list_ports.comports(), key=lambda item: item.device):
        try:
            port = _open_serial(info.device, baud)
        except Exception:
            continue
        try:
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
        else:
            silent.append(info.device)
    if found:
        return tuple(found)
    # Nothing answered on its own.  NOW it is worth asking a silent port
    # whether it is a module someone left in its console.
    for device in silent:
        try:
            port = _open_serial(device, baud)
        except Exception:
            continue
        try:
            wake_from_config_mode(port)
            if _listen_for_packets(port, listen_seconds) >= 2:
                found.append(device)
        except Exception:
            continue
        finally:
            try:
                port.close()
            except Exception:
                pass
    return tuple(found)


__all__ = [
    "DEFAULT_BAUD",
    "SLOWEST_DISCOVERABLE_HZ",
    "offered_rungs",
    "PACKET_RATE_LADDER_HZ",
    "rate_ladder_index",
    "N100_OUTPUTS",
    "WheeltecN100Config",
    "WheeltecN100WaveformSource",
    "discover_n100",
    "drain_imu_samples",
    "header_crc8",
    "payload_crc16",
    "wake_from_config_mode",
]
