"""The N100's settings: the module is asked, and the module's answer wins."""

from __future__ import annotations

import struct
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zlc_atom.devices.waveform.wheeltec_n100 import (
    FRAME_HEAD,
    FRAME_TAIL,
    IMU_PACKET,
    ConsoleRefused,
    FdiConfigConsole,
    WheeltecN100Config,
    WheeltecN100WaveformSource,
    header_crc8,
    packet_rate_field,
    payload_crc16,
)


def _frame(kind: int, payload: bytes, serial: int = 0) -> bytes:
    head = bytes((FRAME_HEAD, kind, len(payload), serial))
    check = payload_crc16(payload)
    return (
        head
        + bytes((header_crc8(head), check >> 8, check & 0xFF))
        + payload
        + bytes((FRAME_TAIL,))
    )


class _FakeModule:
    """An N100 as its serial port sees it: a stream, and a console.

    It streams IMU frames until ``#fconfig`` arrives, answers the console in
    the module's own formats, and starts streaming again on ``#fdeconfig``.
    Rates are quantized to the ladder the real firmware offers, so a request
    off the ladder comes back as the rung it actually took -- which is the
    whole reason the driver reads its writes back.
    """

    LADDER = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0)

    def __init__(
        self, *, rate_hz: float = 100.0, parameters=None, magnetic_repeat: int = 1
    ) -> None:
        self._lock = threading.RLock()
        self.streaming = True
        #: How many packets carry each magnetic reading -- a magnetometer
        #: running slower than the packet rate repeats itself this many
        #: times, which is exactly what the driver counts.
        self.magnetic_repeat = int(magnetic_repeat)
        self.rates = {IMU_PACKET: rate_hz, 0x41: 100.0}
        self.names = {IMU_PACKET: "IMU", 0x41: "AHRS"}
        self.parameters = dict(
            parameters
            if parameters is not None
            else {
                "FILT_LPF_ENABLED": "0",
                "FILT_NOTCH_ENABLED": "0",
                "FILT_NOTCH_CENTER_FREQUENCY": "0",
                "AID_MAG_V_MAGNETIC": "1",
                "IMU_ACC_SCALE_X": "1.000000",
            }
        )
        self.commands: list[str] = []
        self._out = bytearray()
        self._pending = bytearray()
        self._packets = 0
        self.closed = False

    # ------------------------------------------------------------- serial
    @property
    def in_waiting(self) -> int:
        with self._lock:
            self._fill()
            return len(self._out)

    def read(self, size: int = 1) -> bytes:
        with self._lock:
            self._fill()
            taken = bytes(self._out[:size])
            del self._out[:size]
            return taken

    def write(self, data: bytes) -> int:
        with self._lock:
            self._pending += data
            while b"\r\n" in self._pending:
                line, _, rest = bytes(self._pending).partition(b"\r\n")
                self._pending = bytearray(rest)
                self._answer(line.decode("ascii", "replace").strip())
            return len(data)

    def flush(self) -> None:
        return None

    def reset_input_buffer(self) -> None:
        with self._lock:
            self._out.clear()

    def close(self) -> None:
        self.closed = True

    # ------------------------------------------------------------- module
    def _fill(self) -> None:
        if not self.streaming or self._out:
            return
        interval = 1.0 / max(self.rates[IMU_PACKET], 1.0)
        self._packets += 1
        step = float((self._packets - 1) // self.magnetic_repeat)
        payload = struct.pack(
            "<12fq",
            0.0, 0.0, 0.0,
            0.0, 0.0, 9.8,
            200.0 + step, -50.0, 450.0,
            26.85, 1013.0, 24.5,
            int(self._packets * interval * 1e6),
        )
        self._out += _frame(IMU_PACKET, payload, self._packets & 0xFF)

    def _say(self, text: str) -> None:
        self._out += (text + "\r\n").encode("ascii")

    def _answer(self, line: str) -> None:
        self.commands.append(line)
        if line == "#fconfig":
            self.streaming = False
            self._out.clear()
            self._say("Config Mode")
        elif line == "#fdeconfig":
            self.streaming = True
            self._say("*#OK")
        elif line == "#fsave":
            self._say("*#OK")
        elif line == "#fmsg":
            for packet, rate in self.rates.items():
                self._say(f"{self.names[packet]}      [{packet:02X}]   {rate:.1f}Hz")
        elif line.startswith("#fmsg "):
            _, which, wanted = line.split()
            packet = int(which, 16)
            asked = float(wanted)
            took = min(self.LADDER, key=lambda rung: abs(rung - asked))
            self.rates[packet] = took
            self._say(f"{self.names[packet]}      [{packet:02X}]   {took:.1f}Hz")
        elif line == "#fparam":
            for name, value in self.parameters.items():
                self._say(f"{name} = {value}")
        elif line.startswith("#fparam get "):
            name = line.split()[-1]
            if name in self.parameters:
                self._say(f"{name} = {self.parameters[name]}")
            else:
                self._say("*#ERR unknown parameter")
        elif line.startswith("#fparam set "):
            _, _, name, value = line.split()
            if name in self.parameters:
                self.parameters[name] = value
            self._say("*#OK")
        else:
            self._say("*#ERR")


def _source(module: _FakeModule) -> WheeltecN100WaveformSource:
    return WheeltecN100WaveformSource(
        WheeltecN100Config(port="COM_TEST", baud=921600, timeout_seconds=2.0),
        serial_port=module,
    )


def test_the_console_asks_the_module_what_it_can_do() -> None:
    """The packet list and the rate ladder are the module's, not this code's."""

    module = _FakeModule()
    with FdiConfigConsole(module) as console:
        assert module.streaming is False, "config mode stops the stream"
        listed = console.packet_rates()
        assert {packet.packet_id: packet.rate_hz for packet in listed} == {
            IMU_PACKET: 100.0,
            0x41: 100.0,
        }
        assert [packet.name for packet in listed] == ["IMU", "AHRS"]

        # A rate off the module's ladder comes back as the rung it took.
        assert console.set_packet_rate(IMU_PACKET, 200.0) == 200.0
        assert console.set_packet_rate(IMU_PACKET, 137.0) == 100.0

        assert console.get_parameter("AID_MAG_V_MAGNETIC") == "1"
        assert console.get_parameter("NO_SUCH_PARAMETER") is None
        assert console.set_parameter("FILT_NOTCH_ENABLED", "1") == "1"
        with pytest.raises(ConsoleRefused, match="no parameter"):
            console.set_parameter("NO_SUCH_PARAMETER", "1")
        console.save()
    assert module.streaming is True, "leaving config mode puts it back on the air"
    assert module.commands[0] == "#fconfig" and module.commands[-1] == "#fdeconfig"


def test_device_control_moves_the_packet_rate_and_the_records_follow() -> None:
    """Raising the IMU rate is one write, and the sample axis is re-measured.

    The record's sample interval is what the dataset's time axis is built
    from, so a rate change that left the old interval in place would date
    every later shot by the rate it is no longer running at.
    """

    module = _FakeModule(rate_hz=10.0)
    source = _source(module)
    try:
        rate_field = packet_rate_field(IMU_PACKET)
        values = source.tunable_values()
        assert values[rate_field] == 10.0
        assert source.working_point().sample_interval_seconds == pytest.approx(0.1)

        # Only the operator's kinds of parameter are offered; the factory's
        # calibration coefficients sitting beside them are not.
        assert "AID_MAG_V_MAGNETIC" in values
        assert "FILT_NOTCH_CENTER_FREQUENCY" in values
        assert "IMU_ACC_SCALE_X" not in values

        assert source.tune(rate_field, 200.0) == 200.0
        assert module.rates[IMU_PACKET] == 200.0
        assert source.working_point().sample_interval_seconds == pytest.approx(0.005)
        assert module.streaming is True, "the module is left streaming"

        # A rate the firmware does not have is reported as what it took.
        assert source.tune(rate_field, 137.0) == 100.0
        assert source.tunable_values()[rate_field] == 100.0

        # A parameter's NAME says what it is: a frequency is offered in
        # hertz and a switch as a switch, though the module spells both as
        # bare decimals.
        fields = {field.metadata.name: field.metadata for field in source.tunable_fields()}
        assert fields["FILT_NOTCH_CENTER_FREQUENCY"].unit == "Hz"
        assert [choice.value for choice in fields["AID_MAG_V_MAGNETIC"].choices] == ["0", "1"]
        assert fields[rate_field].unit == "Hz"

        assert source.tune("FILT_NOTCH_CENTER_FREQUENCY", 50.0) == 50.0
        assert module.parameters["FILT_NOTCH_CENTER_FREQUENCY"] == "50"
        assert source.tune("AID_MAG_V_MAGNETIC", "0") == "0"
        assert module.parameters["AID_MAG_V_MAGNETIC"] == "0"

        before = source.settings_provenance()["settings_epoch"]
        source.save_settings()
        assert "#fsave" in module.commands
        assert source.settings_provenance()["settings_epoch"] == before

        with pytest.raises(ValueError, match="no setting"):
            source.tune("NOT_A_SETTING", 1.0)
    finally:
        source.close()
    assert module.closed is True


def test_settings_cannot_move_under_a_running_capture() -> None:
    """Config mode stops the stream, so an armed capture refuses the write.

    A capture whose packets stopped mid-flight would publish a gap that no
    reader could tell from the module having gone quiet.
    """

    module = _FakeModule()
    source = _source(module)
    try:
        source.arm(None, buffer_record_count=4)
        assert source.read_records(1, timeout=2.0, exact=True)
        with pytest.raises(RuntimeError, match="capture must be finished"):
            source.tune(packet_rate_field(IMU_PACKET), 200.0)
        with pytest.raises(RuntimeError, match="capture must be finished"):
            source.refresh_tunable_fields()
        assert module.rates[IMU_PACKET] == 100.0, "nothing was written"
        source.finish_record_capture()
        assert source.tune(packet_rate_field(IMU_PACKET), 200.0) == 200.0
    finally:
        source.close()


def test_a_module_whose_console_stays_silent_still_streams() -> None:
    """No console is a fact about the module, not a reason to refuse it.

    A firmware that does not answer ``#fconfig`` still emits perfectly good
    packets; what it does not have is anything an operator can turn, and
    the working point says why.
    """

    class _Mute(_FakeModule):
        def _answer(self, line: str) -> None:
            self.commands.append(line)

    module = _Mute()
    source = _source(module)
    try:
        assert source.tunable_fields() == ()
        assert source.tunable_values() == {}
        point = source.working_point()
        assert point.settings["settings"] == {}
        # It kept streaming through #fconfig, which is exactly how a module
        # without the console announces itself.
        assert "did not enter config mode" in point.settings["settings_refusal"]
        assert point.sample_interval_seconds == pytest.approx(0.01)
        source.arm(None, buffer_record_count=4)
        assert source.read_records(1, timeout=2.0, exact=True)
    finally:
        source.close()


def test_the_module_says_how_fast_its_magnetic_field_actually_moves() -> None:
    """A magnetometer slower than the packet rate is caught by counting repeats.

    Nothing the vendor ships states the magnetometer's own output rate --
    its specification table is a verbatim lift from another manufacturer's
    part -- so raising the packet rate could buy nothing but duplicate
    readings. The driver counts how many packets pass per genuinely new
    field, which turns that unknown into a number off this module.
    """

    packets_per_reading = 4
    module = _FakeModule(rate_hz=200.0, magnetic_repeat=packets_per_reading)
    source = _source(module)
    try:
        point = source.working_point()
        assert point.settings["packet_rate_hz"] == pytest.approx(200.0, rel=0.05)
        assert point.settings["magnetic_update_hz"] == pytest.approx(
            200.0 / packets_per_reading, rel=0.15
        )
    finally:
        source.close()

    # And a magnetometer that keeps up reports the packet rate itself.
    quick = _FakeModule(rate_hz=200.0, magnetic_repeat=1)
    source = _source(quick)
    try:
        point = source.working_point()
        assert point.settings["magnetic_update_hz"] == pytest.approx(200.0, rel=0.05)
    finally:
        source.close()
