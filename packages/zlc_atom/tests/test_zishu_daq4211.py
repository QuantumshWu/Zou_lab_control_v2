"""The DAQ-4211 card: what is wired to a pin is what the signal says it is."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zlc_atom.devices.waveform.zishu_daq4211.device_types import (
    DEVICE_TYPES,
    ZISHU_DAQ4211_SCHEMA,
    authored_config,
)
from zlc_atom.devices.waveform.zishu_daq4211.source import (
    ADC_MODULE,
    SUPPORTED_MODEL,
    ZishuDaq4211WaveformSource,
    discover_daq4211,
)


FLC_100 = tuple(
    {
        "channel": index,
        "quantity": "magnetic_field",
        "unit": "uT",
        "label": axis,
        # A Stefan Mayer FLC 100: ±1 V per 50 uT, about its own 2.5 V reference.
        "unit_per_volt": 50.0,
        "offset_volts": 2.5,
    }
    for index, axis in enumerate("xyz")
)


class _FakeCard:
    """libdaq2 with one card on it: the properties, the task, and a stream.

    It answers reads the way the library does -- interleaved volts, never
    more than asked, and short whenever the stream has not caught up -- so
    the driver's own blocking, reshaping and conversion are what the test
    exercises.
    """

    def __init__(self, *, serial: str = "0362504001", volts=None) -> None:
        self.serial = serial
        self.log: list[str] = []
        self.properties: dict[str, object] = {}
        self.opened = False
        self.running = False
        self.produced = 0
        self._volts = volts if volts is not None else (lambda index, channel: 0.0)

    # ----------------------------------------------------------- devices
    def device_serials(self) -> tuple[str, ...]:
        return (self.serial, "9999999999")

    def device_model(self, serial: str) -> str:
        return SUPPORTED_MODEL if serial == self.serial else "DAQ-7100"

    def open(self, serial: str) -> None:
        assert serial == self.serial
        self.opened = True
        self.log.append("open")

    def close(self, serial: str) -> None:
        self.opened = False
        self.log.append("close")

    # ------------------------------------------- properties and commands
    def command(self, serial: str, module: str, command: str) -> None:
        assert module == ADC_MODULE
        self.log.append(command)
        if command == "StartTask":
            self.running = True
        elif command == "StopTask":
            self.running = False

    def set_int(self, serial: str, module: str, name: str, value: int) -> None:
        self.properties[name] = int(value)
        self.log.append(f"{name}={value}")

    def get_int(self, serial: str, module: str, name: str) -> int:
        return int(self.properties[name])

    def set_text(self, serial: str, module: str, name: str, value: str) -> None:
        self.properties[name] = value
        self.log.append(f"{name}={value}")

    def sync_channel_setting(self, serial: str, module: str) -> None:
        self.log.append("sync")

    def clear_buffer(self, serial: str, module: str) -> None:
        self.log.append("clear")
        self.produced = 0

    def read_analog(self, serial: str, module: str, samples: int, timeout_ms: int):
        if not self.running:
            return np.zeros(0, dtype=np.float64)
        channels = int(self.properties["Channels"])
        # Half a request at a time: the real library answers short whenever
        # its packet is not full yet.
        count = max(channels, (int(samples) // (2 * channels)) * channels)
        values = np.empty(count, dtype=np.float64)
        for position in range(count):
            index, channel = divmod(self.produced + position, channels)
            values[position] = self._volts(index, channel)
        self.produced += count
        return values


def _source(card: _FakeCard, **overrides) -> ZishuDaq4211WaveformSource:
    values = {
        "serial": card.serial,
        "sample_rate_hz": 1000,
        "record_samples": 4,
        "input_range": "5",
        "readings": FLC_100,
        "timeout_seconds": 1.0,
    }
    values.update(overrides)
    return ZishuDaq4211WaveformSource(authored_config(values), daq=card)


def test_the_card_publishes_what_its_pins_are_wired_to() -> None:
    """Three FLC 100 heads on AI0..AI2 are ONE three-component field in uT.

    The card knows volts.  What those volts mean -- 50 uT per volt about a
    2.5 V reference -- is the bench's statement, written on the device, so
    the measurement publishes microtesla beside the N100's rather than
    volts nobody can compare.
    """

    # 2.5 V is zero field; +0.02 V is +1 uT.
    card = _FakeCard(volts=lambda index, channel: 2.5 + 0.02 * (channel + 1))
    source = _source(card)
    try:
        (field,) = source.outputs
        assert field.name == "magnetic_field" and field.unit == "uT"
        assert field.channel_labels == ("x", "y", "z") and field.columns == (0, 1, 2)
        assert source.record_samples == 4

        source.arm(2, buffer_record_count=2)
        assert card.properties == {
            "InputRange": 1,
            "Channels": 3,
            "Frequency": 1000,
            "Cycles": 0,
            "SampleMode": "Continuous",
        }
        assert "sync" in card.log and card.log.index("clear") < card.log.index("StartTask")
        assert "SoftTrigger" in card.log

        first, second = source.read_records(2, timeout=2.0, exact=True)
        assert first.samples.shape == (4, 3)
        assert first.samples[0] == pytest.approx((1.0, 2.0, 3.0))
        # The card's own sample clock stamps the records: four samples at
        # 1 kHz is 4 ms, whatever the USB packets did on the way here.
        assert first.time_seconds == pytest.approx(0.0)
        assert second.time_seconds == pytest.approx(0.004)
        assert [record.source_ordinal for record in (first, second)] == [0, 1]

        point = source.working_point()
        assert point.sample_interval_seconds == pytest.approx(0.001)
        assert point.settings["model"] == SUPPORTED_MODEL
        assert point.settings["input_range_volts"] == 5.0
        assert point.settings["readings"][0]["unit_per_volt"] == 50.0

        terminal = source.finish_record_capture()
        assert terminal.produced_count >= 2 and terminal.joined
        assert card.log[-2:] == ["Stop", "StopTask"]
        assert source.capture_state() is False
    finally:
        source.close()
    assert card.opened is False


def test_a_record_is_cut_from_the_stream_however_the_reads_land() -> None:
    """The library answers short and unaligned; the records still line up.

    Each sample carries its own position in the interleaved stream, so a
    record that began mid-answer would show it immediately.
    """

    card = _FakeCard(volts=lambda index, channel: 2.5 + 0.02 * (100 * index + channel))
    source = _source(card, record_samples=3)
    try:
        source.arm(None, buffer_record_count=8)
        records = []
        deadline = time.monotonic() + 5.0
        while len(records) < 3 and time.monotonic() < deadline:
            records.extend(source.read_records(3, timeout=0.5, exact=False))
        assert len(records) >= 3
        # Each sample says which sample of the stream it is, so a record cut
        # on the wrong boundary -- or one that lost a sample to a short read
        # -- shows up as a row that does not follow the one above it.  Which
        # records survive is the ring's business: a free-running source
        # keeps the newest.
        for record in records[:3]:
            values = np.asarray(record.samples, dtype=np.float64)
            first = values[0, 0]
            expected = first + np.array(
                [[100 * row + channel for channel in range(3)] for row in range(3)],
                dtype=np.float64,
            )
            assert values == pytest.approx(expected)
        starts = [float(np.asarray(record.samples)[0, 0]) for record in records[:3]]
        assert np.allclose(np.diff(starts), 300.0), "three samples per record, in order"
    finally:
        source.close()


def test_only_this_model_is_offered_and_its_channels_are_declared_once() -> None:
    card = _FakeCard()
    assert discover_daq4211(card) == (card.serial,)

    (descriptor,) = DEVICE_TYPES
    assert descriptor.type_id == "waveform.zishu_daq4211"
    assert descriptor.domain == "waveform" and descriptor.capabilities == ("waveform.source",)
    assert descriptor.authoring_schema is ZISHU_DAQ4211_SCHEMA

    with pytest.raises(ValueError, match="two readings name one analog input"):
        authored_config(
            {
                "serial": "x",
                "readings": (
                    {"channel": 0, "quantity": "voltage", "unit": "V", "label": "a",
                     "unit_per_volt": 1.0, "offset_volts": 0.0},
                    {"channel": 0, "quantity": "voltage", "unit": "V", "label": "b",
                     "unit_per_volt": 1.0, "offset_volts": 0.0},
                ),
            }
        )
    with pytest.raises(ValueError, match="disagree about its unit"):
        authored_config(
            {
                "serial": "x",
                "readings": (
                    {"channel": 0, "quantity": "field", "unit": "uT", "label": "a",
                     "unit_per_volt": 1.0, "offset_volts": 0.0},
                    {"channel": 1, "quantity": "field", "unit": "V", "label": "b",
                     "unit_per_volt": 1.0, "offset_volts": 0.0},
                ),
            }
        ).readings and ZishuDaq4211WaveformSource(
            authored_config(
                {
                    "serial": "x",
                    "readings": (
                        {"channel": 0, "quantity": "field", "unit": "uT", "label": "a",
                         "unit_per_volt": 1.0, "offset_volts": 0.0},
                        {"channel": 1, "quantity": "field", "unit": "V", "label": "b",
                         "unit_per_volt": 1.0, "offset_volts": 0.0},
                    ),
                }
            ),
            daq=_FakeCard(serial="x"),
        )
