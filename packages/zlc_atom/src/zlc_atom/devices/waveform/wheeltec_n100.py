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
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import struct
import threading
import time

import numpy as np

from zlc_atom.devices.waveform.contract import (
    WaveformAcquisitionMode,
    WaveformCaptureTerminalRecord,
    WaveformOutput,
    WaveformRecord,
    WaveformWorkingPoint,
)


FRAME_HEAD = 0xFC
FRAME_TAIL = 0xFD
IMU_PACKET = 0x40
#: Payload length of every FDILink state packet the stream may carry, by
#: type byte.  The reader steps over whole packets it does not publish
#: instead of scanning their bytes for a head, which is where a float that
#: happens to contain 0xFC would otherwise start a false frame.
PACKET_PAYLOAD_LENGTHS = {
    IMU_PACKET: 56,
    0x41: 48,
    0x42: 72,
    0x5C: 32,
    0x50: 100,
}
#: head, type, length, serial, header CRC8, payload CRC16 high, low.
HEADER_LENGTH = 7
#: gyroscope xyz (rad/s), accelerometer xyz (m/s^2), magnetometer xyz (mG),
#: IMU temperature (degC), pressure, pressure temperature, timestamp (us).
_IMU_PAYLOAD = struct.Struct("<12fq")
DEFAULT_BAUD = 921600

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
    order; other packet types are stepped over.  Bytes that do not frame a
    packet -- a partial packet at the end, or noise -- are dropped one at a
    time until a frame lines up again, and a trailing partial frame stays
    in the buffer for the next read to complete.
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
        kind = buffer[head + 1]
        expected = PACKET_PAYLOAD_LENGTHS.get(kind)
        if expected is None or buffer[head + 2] != expected:
            at = head + 1
            continue
        end = head + HEADER_LENGTH + expected + 1
        if total < end:
            at = head
            break
        if buffer[end - 1] != FRAME_TAIL:
            at = head + 1
            continue
        if kind == IMU_PACKET:
            values = _IMU_PAYLOAD.unpack_from(buffer, head + HEADER_LENGTH)
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
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._stamps: deque[float] = deque(maxlen=_RATE_SAMPLE_PACKETS)
        self._sample_interval: float | None = None
        self._queue: deque[WaveformRecord] = deque()
        self._armed = False
        self._accepting = False
        self._expected: int | None = None
        self._buffer_record_count = 1
        self._next_ordinal = 0
        self._produced_count = 0
        self._reader_error: BaseException | None = None
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"zlc-n100-{config.port}",
            daemon=True,
        )
        try:
            self._reader.start()
            self._measure_rate()
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------- reading
    def _read_loop(self) -> None:
        buffer = bytearray()
        try:
            while not self._stop.is_set():
                waiting = self._serial.in_waiting
                chunk = self._serial.read(waiting if waiting else 1)
                if not chunk:
                    continue
                buffer += chunk
                samples = drain_imu_samples(buffer)
                if samples:
                    self._accept(samples)
        except BaseException as error:  # noqa: BLE001 -- surfaced to the reader of records
            with self._condition:
                self._reader_error = error
                self._accepting = False
                self._condition.notify_all()

    def _accept(self, samples: list[tuple[float, tuple[float, ...]]]) -> None:
        received = time.time_ns()
        with self._condition:
            for stamp, values in samples:
                self._stamps.append(stamp)
                if not self._accepting:
                    continue
                record = WaveformRecord(
                    np.asarray(values, dtype=np.float32).reshape(1, _COLUMNS),
                    self._next_ordinal,
                    received,
                )
                self._next_ordinal += 1
                self._produced_count += 1
                while len(self._queue) >= self._buffer_record_count:
                    self._queue.popleft()
                self._queue.append(record)
                if self._expected is not None and self._produced_count >= self._expected:
                    self._accepting = False
            self._condition.notify_all()

    def _measure_rate(self) -> None:
        """The packet interval off the module's own clock, once, at open."""

        deadline = time.monotonic() + _RATE_SAMPLE_SECONDS
        with self._condition:
            while len(self._stamps) < _RATE_SAMPLE_PACKETS:
                if self._reader_error is not None:
                    raise RuntimeError(
                        f"reading {self.config.port} failed"
                    ) from self._reader_error
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            stamps = np.asarray(self._stamps, dtype=np.float64)
        if stamps.size < 2:
            raise RuntimeError(
                f"no FDILink IMU packets arrived on {self.config.port} at "
                f"{self.config.baud} baud within {_RATE_SAMPLE_SECONDS:g} s: check "
                "the port, the module's power and that its line rate is "
                f"{self.config.baud}"
            )
        intervals = np.diff(stamps)
        interval = float(np.median(intervals[intervals > 0.0]))
        if not np.isfinite(interval) or interval <= 0.0:
            raise RuntimeError(
                f"the module on {self.config.port} stamps its packets with a clock "
                "that does not advance"
            )
        self._sample_interval = interval

    # ------------------------------------------------------------ contract
    @property
    def timeout(self) -> float:
        return self.config.timeout_seconds

    @property
    def identity(self) -> str:
        return f"wheeltec-n100:{self.config.port}"

    def working_point(self) -> WaveformWorkingPoint:
        interval = self._sample_interval
        if interval is None:
            raise RuntimeError("the N100 packet rate was never measured")
        return WaveformWorkingPoint(
            WaveformAcquisitionMode.FREE_RUNNING,
            interval,
            1,
            N100_OUTPUTS,
            {
                "port": self.config.port,
                "baud": self.config.baud,
                "packet_rate_hz": round(1.0 / interval, 3),
            },
        )

    def arm(
        self, records: int | None, *, buffer_record_count: int, timeout: float
    ) -> None:
        del timeout
        buffer_count = int(buffer_record_count)
        if buffer_count <= 0:
            raise ValueError("buffer_record_count must be positive")
        expected = None if records is None else int(records)
        if expected is not None and (expected <= 0 or buffer_count != expected):
            raise ValueError("a finite arm buffers exactly the records it expects")
        with self._condition:
            if self._reader_error is not None:
                raise RuntimeError("the N100 reader has failed") from self._reader_error
            if self._armed:
                raise RuntimeError("the N100 is already armed")
            self._queue.clear()
            self._armed = True
            self._accepting = True
            self._expected = expected
            self._buffer_record_count = buffer_count
            self._next_ordinal = 0
            self._produced_count = 0

    def read_records(
        self, n: int, *, timeout: float, exact: bool
    ) -> list[WaveformRecord]:
        requested = int(n)
        if requested <= 0:
            raise ValueError("n must be positive")
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while len(self._queue) < requested:
                if self._reader_error is not None:
                    raise RuntimeError("the N100 reader has failed") from self._reader_error
                if not self._armed or not self._accepting:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if exact and len(self._queue) < requested:
                raise TimeoutError("the N100 did not deliver the requested exact records")
            count = requested if exact else min(requested, len(self._queue))
            return [self._queue.popleft() for _ in range(count)]

    def finish_record_capture(self) -> WaveformCaptureTerminalRecord:
        with self._condition:
            self._accepting = False
            self._armed = False
            self._expected = None
            terminal = WaveformCaptureTerminalRecord(
                self._produced_count, True, not self._queue, True
            )
            self._condition.notify_all()
            if self._reader_error is not None:
                raise RuntimeError("the N100 reader has failed") from self._reader_error
            return terminal

    def capture_state(self) -> bool:
        with self._condition:
            return self._armed

    def close(self) -> None:
        """Stop reading and release the port; the handle goes only once it has."""

        self._stop.set()
        with self._condition:
            self._accepting = False
            self._armed = False
            self._condition.notify_all()
        if self._reader.is_alive() and self._reader is not threading.current_thread():
            self._reader.join(timeout=2.0)
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
    "N100_OUTPUTS",
    "WheeltecN100Config",
    "WheeltecN100WaveformSource",
    "discover_n100",
    "drain_imu_samples",
]
