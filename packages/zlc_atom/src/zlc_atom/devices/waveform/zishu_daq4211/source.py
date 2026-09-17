"""A ZishuTech DAQ-4211 acquisition card as a waveform source.

The card is eight single-ended 16-bit inputs sampled SYNCHRONOUSLY at one
rate, over USB, with a software trigger and no external one (UG0015 §5.2).
So one record is one block of that stream: ``record_samples`` per channel,
every channel taken at the same instant, and the card's own sample clock is
the record's time.

What the card cannot know is what is WIRED to each input.  A Stefan Mayer
FLC 100 answers ±1 V per 50 µT about a 2.5 V reference, and three of them
are a magnetic field in microtesla, not three volts: so each authored
channel says which quantity it carries, in which unit, and the volts-to-unit
conversion of the thing plugged into it.  That is a fact about the bench,
which is exactly what a device configuration is for -- and it is why this
source publishes ``magnetic_field`` in µT beside the N100's, rather than
volts nobody can compare.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Sequence

import numpy as np

from zlc_atom.devices import RecordQueue
from zlc_atom.devices.waveform.contract import (
    WaveformAcquisitionMode,
    WaveformCaptureTerminalRecord,
    WaveformOutput,
    WaveformRecord,
    WaveformWorkingPoint,
    validate_waveform_outputs,
)

from ._libdaq2 import library


#: The card's analog-input module, from its hardware guide's property table.
ADC_MODULE = "ADC"

#: The model this driver is written against: same protocol across the family,
#: different channel count and rate ceiling per model, so the card it may
#: claim to have found is named.
SUPPORTED_MODEL = "DAQ-4211"

#: ``InputRange`` as the card numbers it (UG0015 §8.1).
INPUT_RANGES = {10.0: 0, 5.0: 1}

#: The USB stream is aggregated by the driver into packets of about this
#: much time, so a read shorter than one is answered no sooner than one.
#: It sets the floor on how promptly a capture can be stopped.
_AGGREGATION_SECONDS = 0.05


@dataclass(frozen=True)
class ChannelReading:
    """What one analog input is wired to, in the unit it is read in.

    ``unit_per_volt`` and ``offset_volts`` are the SENSOR's, not the card's:
    volts on the pin become ``(volts - offset_volts) * unit_per_volt`` of
    the stated unit.  An FLC 100 read single-ended against its own 2.5 V
    reference is ``50.0`` µT per volt about ``2.5``; a bare voltage tap is
    ``1.0`` about ``0.0``.
    """

    channel: int
    quantity: str
    unit: str
    label: str
    unit_per_volt: float = 1.0
    offset_volts: float = 0.0

    def __post_init__(self) -> None:
        channel = int(self.channel)
        if channel < 0:
            raise ValueError("an analog input is numbered from zero")
        object.__setattr__(self, "channel", channel)
        for name in ("quantity", "unit", "label"):
            text = str(getattr(self, name)).strip()
            if not text:
                raise ValueError(f"a channel reading needs its {name}")
            object.__setattr__(self, name, text)
        scale = float(self.unit_per_volt)
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("unit_per_volt must be finite and non-zero")
        offset = float(self.offset_volts)
        if not np.isfinite(offset):
            raise ValueError("offset_volts must be finite")
        object.__setattr__(self, "unit_per_volt", scale)
        object.__setattr__(self, "offset_volts", offset)


@dataclass(frozen=True)
class ZishuDaq4211Config:
    """Which card, how fast, how long a record is, and what is plugged in."""

    serial: str
    readings: tuple[ChannelReading, ...]
    sample_rate_hz: int = 10000
    record_samples: int = 100
    input_range_volts: float = 10.0
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "serial", str(self.serial).strip())
        readings = tuple(self.readings)
        if not readings or any(
            not isinstance(value, ChannelReading) for value in readings
        ):
            raise TypeError("a DAQ-4211 is authored with its channel readings")
        channels = [value.channel for value in readings]
        if len(set(channels)) != len(channels):
            raise ValueError("two readings name one analog input")
        rate = int(self.sample_rate_hz)
        if rate < 1:
            raise ValueError("sample_rate_hz must be at least 1")
        samples = int(self.record_samples)
        if samples < 1:
            raise ValueError("record_samples must be at least 1")
        window = float(self.input_range_volts)
        if window not in INPUT_RANGES:
            offered = ", ".join(f"±{value:g} V" for value in sorted(INPUT_RANGES))
            raise ValueError(f"this card takes {offered}, not ±{window:g} V")
        timeout = float(self.timeout_seconds)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        object.__setattr__(self, "readings", readings)
        object.__setattr__(self, "sample_rate_hz", rate)
        object.__setattr__(self, "record_samples", samples)
        object.__setattr__(self, "input_range_volts", window)
        object.__setattr__(self, "timeout_seconds", timeout)

    @property
    def channel_count(self) -> int:
        """How many inputs the card must enable: it enables 0..N-1 together."""

        return 1 + max(value.channel for value in self.readings)


def outputs_of(readings: Sequence[ChannelReading]) -> tuple[WaveformOutput, ...]:
    """One published quantity per distinct quantity name, in authored order."""

    order: list[str] = []
    grouped: dict[str, list[ChannelReading]] = {}
    for reading in readings:
        if reading.quantity not in grouped:
            order.append(reading.quantity)
            grouped[reading.quantity] = []
        grouped[reading.quantity].append(reading)
    outputs = []
    for quantity in order:
        rows = grouped[quantity]
        units = {row.unit for row in rows}
        if len(units) != 1:
            raise ValueError(
                f"the channels reading {quantity!r} disagree about its unit: "
                f"{sorted(units)}"
            )
        outputs.append(
            WaveformOutput(
                quantity,
                rows[0].unit,
                tuple(row.label for row in rows),
                tuple(row.channel for row in rows),
            )
        )
    return validate_waveform_outputs(outputs)


class ZishuDaq4211WaveformSource:
    """The card's continuous stream, cut into records of one working point."""

    def __init__(self, config: ZishuDaq4211Config, *, daq=None) -> None:
        self.config = config
        self._daq = library() if daq is None else daq
        self._outputs = outputs_of(config.readings)
        self._columns = config.channel_count
        # Volts to the authored unit, per record column.  Columns no reading
        # names are enabled (the card counts from zero) and simply published
        # by nobody, so their scale never matters.
        self._scale = np.ones(self._columns, dtype=np.float64)
        self._offset = np.zeros(self._columns, dtype=np.float64)
        for reading in config.readings:
            self._scale[reading.channel] = reading.unit_per_volt
            self._offset[reading.channel] = reading.offset_volts
        self._records = RecordQueue(
            f"the {SUPPORTED_MODEL} {config.serial}",
            join_timeout_seconds=config.timeout_seconds + 1.0,
        )
        self._daq.open(config.serial)
        self._sample_rate = float(config.sample_rate_hz)

    # ------------------------------------------------------------ contract
    @property
    def timeout(self) -> float:
        return self.config.timeout_seconds

    @property
    def identity(self) -> str:
        return f"zishu-daq:{self.config.serial}"

    @property
    def outputs(self) -> tuple[WaveformOutput, ...]:
        return self._outputs

    @property
    def record_samples(self) -> int:
        return self.config.record_samples

    def working_point(self) -> WaveformWorkingPoint:
        """The rate the card is clocking at, and what each column means.

        The rate is the one read back when the card was armed, which is the
        same number that stamps the records: a working point that asked the
        card again would describe a capture by a rate the capture was not
        taken at, and would spend a USB round trip per capture saying so.
        """

        rate = self._sample_rate
        return WaveformWorkingPoint(
            WaveformAcquisitionMode.FREE_RUNNING,
            1.0 / rate,
            {
                "serial": self.config.serial,
                "model": SUPPORTED_MODEL,
                "module": ADC_MODULE,
                "sample_rate_hz": rate,
                "record_samples": self.config.record_samples,
                "input_range_volts": self.config.input_range_volts,
                "channels": self._columns,
                "readings": [
                    {
                        "channel": reading.channel,
                        "quantity": reading.quantity,
                        "unit": reading.unit,
                        "label": reading.label,
                        "unit_per_volt": reading.unit_per_volt,
                        "offset_volts": reading.offset_volts,
                    }
                    for reading in self.config.readings
                ],
            },
            time_basis="sample_clock",
        )

    def arm(self, records: int | None, *, buffer_record_count: int) -> None:
        """Configure the card, start its task, and read the stream it makes."""

        if self._records.armed:
            raise RuntimeError(f"the {SUPPORTED_MODEL} is already armed")
        serial, module = self.config.serial, ADC_MODULE
        self._daq.set_int(serial, module, "InputRange", INPUT_RANGES[self.config.input_range_volts])
        self._daq.set_int(serial, module, "Channels", self._columns)
        self._daq.set_int(serial, module, "Frequency", int(self.config.sample_rate_hz))
        # Zero is the card's word for "until stopped": a capture of N records
        # is this driver's count, not a limit the card has to be told.
        self._daq.set_int(serial, module, "Cycles", 0)
        self._daq.set_text(serial, module, "SampleMode", "Continuous")
        self._daq.sync_channel_setting(serial, module)
        # What the card will actually clock at: it divides its own timebase,
        # so the rate it reports back -- not the authored one -- is what the
        # records are stamped with and what the working point publishes.
        clocked = int(self._daq.get_int(serial, module, "Frequency"))
        if clocked < 1:
            raise RuntimeError(
                f"the {SUPPORTED_MODEL} {serial} reports a sample rate of "
                f"{clocked} Hz after being asked for "
                f"{self.config.sample_rate_hz} Hz"
            )
        self._sample_rate = float(clocked)
        self._daq.clear_buffer(serial, module)
        self._records.arm(
            records,
            buffer_record_count=buffer_record_count,
            worker=lambda stop: threading.Thread(
                target=self._acquire,
                args=(stop,),
                name=f"zlc-zishu-daq-{serial}",
                daemon=True,
            ),
        )
        try:
            self._records.wait_ready(self.timeout)
        except BaseException:
            self.finish_record_capture()
            raise

    def _acquire(self, stop: threading.Event) -> None:
        serial, module = self.config.serial, ADC_MODULE
        block = self.config.record_samples * self._columns
        # Long enough that a read normally comes back full, short enough that
        # a stop is noticed within one: the USB stream is delivered in packets
        # of tens of milliseconds whatever is asked for.
        slice_ms = int(
            max(_AGGREGATION_SECONDS, block / (self._sample_rate * self._columns)) * 1000.0
        ) + 50
        carry = np.zeros(0, dtype=np.float64)
        produced = 0
        try:
            self._daq.command(serial, module, "StartTask")
            self._daq.command(serial, module, "SoftTrigger")
            self._records.mark_ready()
            while not stop.is_set() and self._records.accepting:
                arrived = self._daq.read_analog(
                    serial, module, block - carry.size, slice_ms
                )
                if arrived.size:
                    carry = np.concatenate((carry, arrived))
                while carry.size >= block and not stop.is_set() and self._records.accepting:
                    values = carry[:block].reshape(
                        self.config.record_samples, self._columns
                    )
                    carry = carry[block:]
                    samples = (values - self._offset) * self._scale
                    # The card's own sample clock: every record is exactly
                    # its length after the one before it, whatever the USB
                    # packets did on the way here.
                    seconds = produced * self.config.record_samples / self._sample_rate
                    self._records.push(WaveformRecord(
                        np.asarray(samples, dtype=np.float32), produced, seconds, 0,
                    ))
                    produced += 1
        except BaseException as error:  # noqa: BLE001 -- surfaced to the reader of records
            self._records.fail(error)
        finally:
            try:
                self._daq.command(serial, module, "Stop")
                self._daq.command(serial, module, "StopTask")
            except BaseException as error:  # noqa: BLE001 -- the capture is over either way
                self._records.fail(error)

    def read_records(
        self, n: int, *, timeout: float, exact: bool
    ) -> list[WaveformRecord]:
        return self._records.read(n, timeout=timeout, exact=exact)

    def finish_record_capture(self) -> WaveformCaptureTerminalRecord:
        produced = self._records.finish()
        return WaveformCaptureTerminalRecord(produced, True, not self._records.pending_count, True)

    def capture_state(self) -> bool:
        return self._records.armed

    def close(self) -> None:
        try:
            if self._records.armed:
                self.finish_record_capture()
        finally:
            self._daq.close(self.config.serial)


def discover_daq4211(daq=None) -> tuple[str, ...]:
    """The serial of every card of this model attached to this machine."""

    bus = library() if daq is None else daq
    return tuple(
        serial
        for serial in bus.device_serials()
        if SUPPORTED_MODEL in bus.device_model(serial)
    )


__all__ = [
    "ADC_MODULE",
    "INPUT_RANGES",
    "SUPPORTED_MODEL",
    "ChannelReading",
    "ZishuDaq4211Config",
    "ZishuDaq4211WaveformSource",
    "discover_daq4211",
    "outputs_of",
]
