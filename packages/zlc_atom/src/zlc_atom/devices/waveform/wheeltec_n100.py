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
    WaveformRecordQueue,
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
        self._stop = threading.Event()
        self._stamps: deque[float] = deque(maxlen=_RATE_SAMPLE_PACKETS)
        self._stamps_ready = threading.Event()
        self._sample_interval: float | None = None
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
            self._records.fail(error)
            self._stamps_ready.set()

    def _accept(self, samples: list[tuple[float, tuple[float, ...]]]) -> None:
        received = time.time_ns()
        for stamp, values in samples:
            self._stamp(stamp)
            self._records.push(
                np.asarray(values, dtype=np.float32).reshape(1, _COLUMNS), received
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
    "N100_OUTPUTS",
    "WheeltecN100Config",
    "WheeltecN100WaveformSource",
    "discover_n100",
    "drain_imu_samples",
]
