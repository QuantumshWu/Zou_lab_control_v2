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

from zlc_atom.authoring import TuneRefused
from zlc_atom.devices.waveform.wheeltec_n100.console import CONFIRM_PROMPT, OK
from zlc_atom.devices.waveform.wheeltec_n100.source import _listen_for_packets
from zlc_atom.devices.waveform.wheeltec_n100 import (
    FRAME_HEAD,
    wake_from_config_mode,
    FRAME_TAIL,
    IMU_PACKET,
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
        self.booted_rates = dict(self.rates)
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
        # Rate zero is the ladder's "No Output": the module goes quiet, which
        # is what makes a wrong rate write dangerous rather than merely wrong.
        if not self.streaming or self._out or self.rates[IMU_PACKET] <= 0.0:
            return
        interval = 1.0 / self.rates[IMU_PACKET]
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
            # What a REAL module answers -- not the "Config Mode" the manual
            # prints.  Judging entry by that banner is what broke on the
            # bench; entry is the stream stopping.
            self._say(OK)
        elif line == "#fdeconfig":
            self.streaming = True
            self._say("*#OK")
        elif line == "#fsave":
            self._say("*#OK")
        elif line == "#freboot":
            self._say(CONFIRM_PROMPT)
        elif line == "y":
            # A restart discards everything not written to flash, which is
            # what makes it the undo that needs no spelling.
            self.rates = dict(self.booted_rates)
            self.streaming = True
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
        with pytest.raises(TuneRefused, match="no parameter"):
            console.set_parameter("NO_SUCH_PARAMETER", "1")
        console.save()

        # It is the bench's own text-command link, the same shape a Rigol
        # and a Tektronix are driven through.
        from zlc_atom.devices.visa import ScpiLink

        assert all(
            callable(getattr(console, name, None))
            for name in ScpiLink.__protocol_attrs__
        ), sorted(ScpiLink.__protocol_attrs__)
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
        assert "never entered config mode" in point.settings["settings_refusal"]
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


def test_a_module_that_never_comes_back_says_so_in_those_words() -> None:
    """Opening the console is the one step that can leave a module silent.

    Reading the settings when the port opens costs a trip through config
    mode, and config mode stops the stream. A module that does not start
    again is genuinely unusable, and the device must fail saying THAT --
    naming the console as what silenced it -- rather than with the generic
    "no packets arrived", which would send its operator to check a cable
    that is fine.
    """

    class _NeverReturns(_FakeModule):
        def _answer(self, line: str) -> None:
            super()._answer(line)
            if line == "#fdeconfig":
                self.streaming = False  # it acknowledged, and stayed quiet

    module = _NeverReturns()
    with pytest.raises(RuntimeError, match="configuration console") as refusal:
        _source(module)
    assert "did not resume" in str(refusal.value)
    assert module.closed is True, "a module that failed to open still lets the port go"


def test_a_module_left_in_its_console_is_found_and_opened_again() -> None:
    """Config mode outlives the process that opened it, so it must be undone.

    A session that opened the console and died leaves the module silent,
    and this bench recognises an N100 BY its stream -- so without a
    #fdeconfig it would vanish from every scan and refuse every open until
    somebody power-cycled it. One line on the wire is the difference.
    """

    module = _FakeModule()
    module.streaming = False       # exactly where a dead session leaves it
    module._answer("#fconfig")     # ...and the module thinks it is in config mode

    assert _listen_for_packets(module, 0.05) == 0, "a stuck module says nothing"
    wake_from_config_mode(module)
    assert _listen_for_packets(module, 0.05) >= 2, "one #fdeconfig brings it back"

    # And opening it works without the operator touching anything.
    stuck = _FakeModule()
    stuck.streaming = False
    stuck._answer("#fconfig")
    source = _source(stuck)
    try:
        assert source.working_point().settings["packet_rate_hz"] > 0
        assert "#fdeconfig" in stuck.commands
    finally:
        source.close()


def test_entry_is_the_stream_stopping_not_a_banner() -> None:
    """A module that answers something else has still entered; one that
    keeps streaming has not, whatever it printed.

    The manual prints "Config Mode" as the reply to #fconfig and a real
    module answers "*#OK". Reading entry off the banner failed on the first
    real module it met. The manual's other sentence is the one that holds:
    if the data stops, config mode was entered.
    """

    class _Terse(_FakeModule):
        def _answer(self, line: str) -> None:
            self.commands.append(line)
            if line == "#fconfig":
                self.streaming = False
                self._out.clear()          # enters, and says nothing at all
            else:
                super()._answer(line)

    terse = _Terse()
    with FdiConfigConsole(terse) as console:
        assert terse.streaming is False
        assert console.greeting == ""

    class _Deaf(_FakeModule):
        def _answer(self, line: str) -> None:
            self.commands.append(line)     # prints nothing, keeps streaming

    with pytest.raises(RuntimeError, match="kept streaming through #fconfig"):
        FdiConfigConsole(_Deaf()).enter()


def test_a_module_that_lists_nothing_still_offers_the_knob_that_matters() -> None:
    """#fmsg is documented to print every packet. A real module prints *#OK.

    That is a fact about the firmware, not a fault, and it must not cost the
    operator the one setting they came for: this driver reads the IMU
    packet, so it already knows that packet's rate -- it timed it off the
    stream. Asked first, measured where the answer does not come.
    """

    class _Terse(_FakeModule):
        def _answer(self, line: str) -> None:
            if line == "#fmsg":
                self.commands.append(line)
                self._say(OK)          # acknowledges, lists nothing
                return
            super()._answer(line)

    module = _Terse(rate_hz=50.0)
    source = _source(module)
    try:
        rate_field = packet_rate_field(IMU_PACKET)
        values = source.tunable_values()
        assert values[rate_field] == pytest.approx(50.0, rel=0.05), (
            "the rate came off the stream, since the module would not say it"
        )
        point = source.working_point()
        assert point.settings["packets_listed"] is False
        assert point.settings["last_console_exchange"][1].strip() != "", (
            "the module's own words are kept, so the next surprise is one lookup away"
        )
        # And the knob still writes.
        assert source.tune(rate_field, 200.0) == 200.0
        assert module.rates[IMU_PACKET] == 200.0
    finally:
        source.close()


def test_a_rate_that_was_set_is_never_reported_as_refused() -> None:
    """The module may acknowledge a write without echoing it back.

    This firmware answers a bare #fmsg with nothing but *#OK, so requiring
    an echo would have turned a rate that WAS set into a refusal, and the
    operator would have been told the knob did not move while the records
    quietly arrived at the new rate. The stream is what settles it.
    """

    class _Silent(_FakeModule):
        """Sets the rate, says only *#OK, and never lists anything."""

        def _answer(self, line: str) -> None:
            if line.startswith("#fmsg"):
                self.commands.append(line)
                parts = line.split()
                if len(parts) == 3:
                    took = min(self.LADDER, key=lambda r: abs(r - float(parts[2])))
                    self.rates[int(parts[1], 16)] = took
                self._say(OK)
                return
            super()._answer(line)

    module = _Silent(rate_hz=50.0)
    source = _source(module)
    try:
        rate_field = packet_rate_field(IMU_PACKET)
        taken = source.tune(rate_field, 200.0)
        assert module.rates[IMU_PACKET] == 200.0, "the module really did take it"
        assert taken == pytest.approx(200.0, rel=0.05), (
            "and the bench reports the rate it measured, not a refusal"
        )
        assert source.working_point().sample_interval_seconds == pytest.approx(
            0.005, rel=0.05
        )
    finally:
        source.close()


def test_a_write_that_silences_the_module_is_put_back() -> None:
    """The archive does not settle what #fmsg's second argument is.

    The manual says literal hertz; the vendor's own parameter tables spell
    rates as ladder indices where 0 means "no output". A wrong spelling can
    therefore turn the packet off. This bench must never leave an
    operator's module mute because it guessed: the stream is checked after
    every rate write, and a write that stopped it is written back.
    """

    class _TurnsOffOnOutOfRange(_FakeModule):
        """Reads #fmsg's argument as a LADDER INDEX, as the GUI's tables do."""

        LADDER = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0)

        def _answer(self, line: str) -> None:
            if line.startswith("#fmsg "):
                self.commands.append(line)
                _, which, value = line.split()
                index = int(float(value))
                rate = self.LADDER[index] if 0 <= index < len(self.LADDER) else 0.0
                self.rates[int(which, 16)] = rate
                self._say(OK)
                return
            super()._answer(line)

    module = _TurnsOffOnOutOfRange(rate_hz=50.0)
    # It is at ladder index 6 == 50 Hz, which this driver knows as 50.0.
    source = _source(module)
    try:
        rate_field = packet_rate_field(IMU_PACKET)
        with pytest.raises(TuneRefused, match="stopped it sending"):
            source.tune(rate_field, 100.0)   # read as index 100 -> out of range -> off
        assert module.rates[IMU_PACKET] == 50.0, "the module was put back"
        assert module.streaming is True
        # And the module is usable: the records keep coming.
        source.arm(None, buffer_record_count=4)
        assert source.read_records(1, timeout=3.0, exact=True)
        source.finish_record_capture()
    finally:
        source.close()
