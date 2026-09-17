"""The N100's settings: what the module DOES is the only thing believed.

Every fake here answers the way the real module was recorded answering,
which is not the way its manual says.  ``#fconfig`` is answered ``*#OK``,
not ``Config Mode``.  A bare ``#fmsg`` enumerates every packet as
``MSG_IMU[40]   10.0Hz``, while ``#fmsg 40 100`` changes nothing at all.
A bare ``#fparam`` is ``*#ERROR``.  ``#fparam get MSG_IMU`` answers
``MSG_IMU=4`` -- no spaces, no ``*#OK`` -- and that 4 beside the 10.0Hz is
what says a rate is stored as a ladder index.
"""

from __future__ import annotations

import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zlc_atom.authoring import TuneRefused
from zlc_atom.devices.waveform.wheeltec_n100 import (
    FRAME_HEAD,
    FRAME_TAIL,
    IMU_PACKET,
    IMU_PACKET_NAME,
    PACKET_RATE_LADDER_HZ,
    FdiConfigConsole,
    WheeltecN100Config,
    WheeltecN100WaveformSource,
    header_crc8,
    payload_crc16,
    rate_ladder_index,
    wake_from_config_mode,
)
from zlc_atom.devices.waveform.wheeltec_n100 import console as console_module
from zlc_atom.devices.waveform.wheeltec_n100.console import CONFIRM_PROMPT, OK
from zlc_atom.devices.waveform.wheeltec_n100.source import _listen_for_packets


def _frame(kind: int, payload: bytes, serial: int = 0) -> bytes:
    head = bytes((FRAME_HEAD, kind, len(payload), serial))
    check = payload_crc16(payload)
    return (
        head
        + bytes((header_crc8(head), check >> 8, check & 0xFF))
        + payload
        + bytes((FRAME_TAIL,))
    )


#: How long this fake goes on ignoring commands after #fconfig. The real
#: module's acknowledgement arrives after its frames have drained, and it
#: refuses commands that crowd the one before, so a driver that fires the
#: next line straight away must be caught here. The window the driver waits
#: is scaled down for the suite, so this is too.
_CONFIG_SETTLE = 0.15


class _FakeModule:
    """An N100 as its serial port sees it: a stream, and a console."""

    def __init__(
        self, *, rate_hz: float = 100.0, parameters=None, magnetic_repeat: int = 1
    ) -> None:
        self._lock = threading.RLock()
        self.streaming = True
        self.magnetic_repeat = int(magnetic_repeat)
        self.parameters = dict(
            parameters
            if parameters is not None
            else {
                IMU_PACKET_NAME: str(PACKET_RATE_LADDER_HZ.index(rate_hz)),
                "MSG_AHRS": "0",
                "FILT_LPF_ENABLED": "0",
                "FILT_NOTCH_ENABLED": "0",
                "FILT_NOTCH_CENTER_FREQUENCY": "0",
                "AID_MAG_V_MAGNETIC": "1",
                "IMU_ACC_SCALE_X": "1.000000",
            }
        )
        #: Three copies, because the module has three.  ``parameters`` is
        #: the table #fparam writes; ``saved`` is what #fsave put in
        #: flash; ``running`` is what the module is actually DOING, and
        #: it is reloaded from flash only by a restart.  Measured on a
        #: real module: after #fparam set the table holds the new value
        #: and the stream keeps its old rate.
        self.saved = dict(self.parameters)
        self.running = dict(self.parameters)
        #: How long the module takes to START answering.  A real one
        #: takes its time over #fmsg, which prints some 1900 bytes.
        self.reply_delay = 0.0
        #: How long after acknowledging #fconfig the module actually
        #: starts obeying #f commands.  It prints *#OK first and changes
        #: state afterwards, so anything sent in between is read by a
        #: module that is still navigating, and dropped.  The probe that
        #: worked on the bench hid this by listening a fixed 1.2 s after
        #: every command.
        self.config_settle = _CONFIG_SETTLE
        self._listening_at = 0.0
        self._due: deque = deque()
        self.commands: list[str] = []
        self._out = bytearray()
        self._pending = bytearray()
        self._packets = 0
        self.closed = False

    @property
    def rate_hz(self) -> float:
        """The rate the ladder index names; index 0 is no output at all."""

        return self._rate_of(IMU_PACKET_NAME)

    def _rate_of(self, name: str) -> float:
        """From what the module is RUNNING, not from the table."""

        try:
            index = int(float(self.running.get(name, 0)))
        except ValueError:
            return 0.0
        rungs = PACKET_RATE_LADDER_HZ
        return rungs[index] if 0 <= index < len(rungs) else 0.0

    # ------------------------------------------------------------- serial
    @property
    def in_waiting(self) -> int:
        with self._lock:
            self._release()
            self._fill()
            return len(self._out)

    def read(self, size: int = 1) -> bytes:
        with self._lock:
            self._release()
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
        if self._due or not self.streaming or self._out or self.rate_hz <= 0.0:
            return
        interval = 1.0 / self.rate_hz
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
        self._due.append(
            (time.monotonic() + self.reply_delay, (text + "\r\n").encode("ascii"))
        )
        self._release()

    def _release(self) -> None:
        now = time.monotonic()
        while self._due and self._due[0][0] <= now:
            self._out += self._due.popleft()[1]

    def _answer(self, line: str) -> None:
        self.commands.append(line)
        if line not in ("#fconfig", "#fdeconfig"):
            if time.monotonic() < self._listening_at:
                return                 # still switching: the line is lost
        if line == "#fconfig":
            self.streaming = False
            self._out.clear()
            # What a REAL module answers -- not the "Config Mode" the manual
            # prints. Judging entry by that banner is what broke on the
            # bench; entry is the module starting to ANSWER.
            self._say(OK)
            self._listening_at = time.monotonic() + self.config_settle
        elif line == "#fdeconfig":
            self.streaming = True
            self._say(OK)
        elif line == "#fsave":
            self.saved = dict(self.parameters)
            self._say(OK)
        elif line == "#freboot":
            self._say(CONFIRM_PROMPT)
        elif line == "y":
            # The restart is what makes a saved setting take effect, and
            # what discards one that was never saved.
            self.parameters = dict(self.saved)
            self.running = dict(self.saved)
            self.streaming = True
        elif line == "#fmsg":
            # The module enumerating itself, verbatim in this layout:
            #     MSG_IMU[40]   10.0Hz
            for name, packet_id in (
                (IMU_PACKET_NAME, IMU_PACKET), ("MSG_AHRS", 0x41)
            ):
                self._say(f"{name}[{packet_id:02x}]   {self._rate_of(name):.1f}Hz")
        elif line.startswith("#fmsg "):
            # The SET form changes nothing on this firmware: a rate never
            # arrives as a number of hertz.
            self._say(OK)
        elif line == "#fparam":
            # A bare #fparam is an error: parameters cannot be enumerated.
            self._say("*#ERROR")
        elif line.startswith("#fparam get "):
            name = line.split()[-1]
            if name in self.parameters:
                # NAME=value -- no spaces, and no *#OK after it.
                self._say(f"{name}={self.parameters[name]}")
            else:
                # Its own word for no, recorded off the bench -- and said
                # twice over by a module that answered a pair of #fparam
                # gets sent back to back with two of them.
                self._say("*#ERROR")
        elif line.startswith("#fparam set "):
            _, _, name, value = line.split()
            if name in self.parameters:
                self.parameters[name] = value
            self._say(OK)
        else:
            self._say("*#ERROR")



@pytest.fixture(autouse=True)
def _console_windows_for_a_fake(monkeypatch):
    """Shrink the listening windows, which a fake does not need.

    The real ones are a hardware requirement -- the module answers *#ERROR
    to a command that arrives inside them -- and the fake has nothing to
    say about that, so paying them here would buy a slower suite and
    nothing else. The product default is untouched.
    """

    monkeypatch.setattr(console_module, "SPACING_SECONDS", 0.05)
    monkeypatch.setattr(console_module, "REPLY_QUIET_SECONDS", 0.02)
    monkeypatch.setattr(console_module, "STILL_NAVIGATING_SECONDS", 0.05)
    monkeypatch.setitem(globals(), "_CONFIG_SETTLE", 0.005)


def _source(module: _FakeModule) -> WheeltecN100WaveformSource:
    return WheeltecN100WaveformSource(
        WheeltecN100Config(port="COM_TEST", baud=921600, timeout_seconds=2.0),
        serial_port=module,
    )


# ------------------------------------------------------------------ console
def test_the_console_is_this_bench_s_own_text_link() -> None:
    """Entering is the stream stopping; every value comes back read."""

    module = _FakeModule()
    with FdiConfigConsole(module) as console:
        assert module.streaming is False, "config mode stops the stream"

        assert console.get_parameter("AID_MAG_V_MAGNETIC") == "1"
        assert console.get_parameter("NO_SUCH_PARAMETER") is None
        assert console.set_parameter("FILT_NOTCH_ENABLED", "1") == "1"
        with pytest.raises(TuneRefused, match="no parameter"):
            console.set_parameter("NO_SUCH_PARAMETER", "1")
        console.save()
    assert module.streaming is True, "leaving puts it back on the air"
    assert module.commands[0] == "#fconfig" and module.commands[-1] == "#fdeconfig"


def test_entry_is_the_stream_stopping_not_a_banner() -> None:
    """The manual prints "Config Mode"; a real module answers "*#OK".

    Reading entry off the banner failed on the first real module it met.
    The manual's other sentence is the one that holds: if the data stops,
    config mode was entered.
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
        # It printed nothing at all, and it is in config mode: what says so
        # is that it answers the question every module answers.
        assert console.get_parameter(IMU_PACKET_NAME) is not None

    class _Deaf(_FakeModule):
        def _answer(self, line: str) -> None:
            self.commands.append(line)     # prints nothing, keeps streaming

    with pytest.raises(RuntimeError, match="kept streaming through #fconfig"):
        FdiConfigConsole(_Deaf()).enter()


# -------------------------------------------------------------- the knobs
def test_the_rate_is_a_ladder_rung_written_as_its_index() -> None:
    """The module takes an INDEX, not hertz, and only the rungs exist.

    "#fmsg 40 100" is answered "*#OK" and changes nothing on this firmware,
    which is how a module sitting at 10 Hz stayed at 10 Hz while the bench
    reported the write as applied. The rate lives in MSG_IMU, and 100 Hz is
    rung 7.
    """

    assert rate_ladder_index(100.0) == 7
    assert rate_ladder_index(10.0) == 4
    assert rate_ladder_index(400.0) == 9
    with pytest.raises(TuneRefused, match="not one of this module's rates"):
        rate_ladder_index(137.0)
    with pytest.raises(TuneRefused):
        # Rung 0 stops the packets this bench finds the module by...
        rate_ladder_index(0.0)
    with pytest.raises(TuneRefused, match="not one of this module's rates"):
        # ...and 1 Hz hides them from a scan that listens for half a second,
        # which applying would then write to flash. Same hazard, slower.
        rate_ladder_index(1.0)
    assert rate_ladder_index(1.0, standing_at=1.0) == 1, (
        "unless the module is already ON it: putting a rate back where it "
        "was cannot be a new way of losing the module, and the undo needs it"
    )

    module = _FakeModule(rate_hz=10.0)
    source = _source(module)
    try:
        values = source.tunable_values()
        assert values[IMU_PACKET_NAME] == "10", "reported in hertz by #fmsg"
        assert source.working_point().sample_interval_seconds == pytest.approx(0.1)

        # Only the rungs are offered, so a wrong number cannot be typed.
        offered = {
            field.metadata.name: field.metadata for field in source.tunable_fields()
        }[IMU_PACKET_NAME]
        assert [choice.value for choice in offered.choices] == [
            "5", "10", "20", "50", "100", "200", "400"
        ], "1 and 2 Hz are too slow for a scan to hear, so they are not offered"

        assert source.tune(IMU_PACKET_NAME, "100") == "100"
        assert module.parameters[IMU_PACKET_NAME] == "7", "written as its index"
        assert module.saved[IMU_PACKET_NAME] == "7", "and committed to flash"
        assert module.rate_hz == 100.0, "and live, which takes the restart"
        assert "#fsave" in module.commands and "#freboot" in module.commands
        assert source.working_point().sample_interval_seconds == pytest.approx(
            0.01, rel=0.05
        )
        assert module.streaming is True
    finally:
        source.close()


def test_settings_cannot_move_under_a_running_capture() -> None:
    """Config mode stops the stream, so an armed capture refuses the write."""

    module = _FakeModule()
    source = _source(module)
    try:
        source.arm(1, buffer_record_count=4)
        assert source.read_records(1, timeout=3.0, exact=True)
        with pytest.raises(RuntimeError, match="capture must be finished"):
            source.tune(IMU_PACKET_NAME, "200")
        # Reading the rate never touches the console -- the stream says it --
        # so there is nothing here for a capture to refuse.
        assert source.tunable_values()[IMU_PACKET_NAME] == "100"
        assert module.rate_hz == 100.0, "nothing was written"
        source.finish_record_capture()
        assert source.tune(IMU_PACKET_NAME, "200") == "200"
    finally:
        source.close()

    # Capture boundaries reset stream continuity; all frame kinds participate
    # in the serial counter, and native device timestamps remain untouched.
    class _ControlledModule(_FakeModule):
        read_error = None

        def read(self, size=1):
            if self.read_error is not None:
                raise self.read_error
            return super().read(size)

        def _fill(self):
            if self._packets < 40:
                super()._fill()

    def packet(serial, stamp):
        return _frame(IMU_PACKET, struct.pack("<12fq", *([0.0] * 12), stamp), serial)

    module = _ControlledModule(rate_hz=400)
    source = _source(module)
    try:
        assert source.working_point().time_basis == "device_clock"
        source.arm(2, buffer_record_count=4)
        with module._lock:
            module._out += (packet(255, 1_000_000) + _frame(0x41, bytes(48), 0)
                            + packet(1, 1_002_500))
        records = source.read_records(2, timeout=1, exact=True)
        assert [record.source_ordinal for record in records] == [0, 1]
        assert [record.time_seconds for record in records] == pytest.approx([1.0, 1.0025])
        assert source.finish_record_capture().produced_count == 2
        source.arm(1, buffer_record_count=1)
        with module._lock:
            module._out += packet(18, 8_000_000)
        record, = source.read_records(1, timeout=1, exact=True)
        assert record.source_ordinal == 0 and record.time_seconds == 8.0
        source.finish_record_capture()
    finally:
        source.close()

    damaged = bytearray(packet(1, 1_002_500))
    damaged[20] ^= 1
    for next_frame, error in (
        (packet(2, 1_002_500), "sequence gap"),
        (packet(1, 1_007_500), "timestamp gap"),
        (packet(1, 1_000_000), "timestamp did not advance"),
        (damaged, "damaged FDILink"),
    ):
        module = _ControlledModule(rate_hz=400)
        source = _source(module)
        try:
            source.arm(None, buffer_record_count=4)
            with module._lock:
                module._out += packet(0, 1_000_000) + next_frame
            with pytest.raises(RuntimeError, match=error):
                source.read_records(2, timeout=1, exact=True)
            # Even the valid first record cannot hide the later stream fault.
            with pytest.raises(RuntimeError, match=error):
                source.read_records(1, timeout=0, exact=True)
            with pytest.raises(RuntimeError):
                source.finish_record_capture()
            assert source._reader.is_alive(), "capture data errors do not close the serial device"
            source.arm(1, buffer_record_count=4)
            with module._lock:
                module._out += packet(18, 8_000_000)
            record, = source.read_records(1, timeout=1, exact=True)
            assert record.source_ordinal == 0 and record.time_seconds == 8.0
            assert source.finish_record_capture().produced_count == 1
        finally:
            source.close()

    module = _ControlledModule(rate_hz=400)
    source = _source(module)
    try:
        source.arm(None, buffer_record_count=4)
        module.read_error = OSError("serial cable removed")
        source._reader.join(timeout=1)
        assert not source._reader.is_alive()
        with pytest.raises(RuntimeError, match="serial cable removed"):
            source.read_records(1, timeout=0, exact=True)
        with pytest.raises(RuntimeError, match="serial cable removed"):
            source.finish_record_capture()
        with pytest.raises(RuntimeError, match="receive thread is not running"):
            source.arm(1, buffer_record_count=4)
    finally:
        source.close()

    # The shared FIFO retains every accepted record and fails instead of
    # replacing its oldest entry, even when the caller asks for only one.
    import numpy as np
    from zlc_atom.devices import RecordQueue
    from zlc_atom.devices.waveform.contract import WaveformRecord
    queue = RecordQueue("test capture", join_timeout_seconds=1)
    queue.arm(3, buffer_record_count=1)
    queue.mark_ready()
    queue.wait_ready(0)
    for ordinal in range(3):
        assert queue.push(WaveformRecord(np.asarray([[ordinal]]), ordinal, ordinal * 0.0025))
        record, = queue.read(1, timeout=0, exact=True)
        assert record.source_ordinal == ordinal and record.samples[0, 0] == ordinal
    assert queue.finish() == 3
    queue.arm(None, buffer_record_count=1)
    assert queue.push(WaveformRecord(np.asarray([[1]]), 0, 1))
    assert not queue.push(WaveformRecord(np.asarray([[2]]), 1, 2))
    assert queue.produced_count == 1
    with pytest.raises(RuntimeError, match="buffer overflow"):
        queue.read(1, timeout=0, exact=True)
    with pytest.raises(RuntimeError):
        queue.finish()
    queue.arm(None, buffer_record_count=2)
    assert queue.failure is None
    assert queue.push(WaveformRecord(np.asarray([[3]]), 0, 3))
    assert queue.finish() == 1 and queue.pending_count == 1
    assert queue.read(1, timeout=0, exact=True)[0].samples[0, 0] == 3
    queue.arm(None, buffer_record_count=2)
    assert not queue.push(WaveformRecord(np.asarray([[4]]), 1, 4))
    with pytest.raises(RuntimeError, match="record ordinal"):
        queue.read(1, timeout=0, exact=False)
    with pytest.raises(RuntimeError):
        queue.finish()


def test_a_write_that_silences_the_module_is_put_back() -> None:
    """Index 0 is "no output", and a module told to stop sending is lost.

    Discovery recognises this module only by its packets, so a write that
    silenced it would make it invisible until power-cycled. The stream is
    checked after every rate write, and a write that stopped it is undone
    by restarting -- which needs no knowledge of how the module spelled it,
    since nothing here was written to flash.
    """

    class _MisreadsTheIndex(_FakeModule):
        """One rung this firmware reads as "no output" instead.

        A module that turned EVERY write into rung 0 could not be recovered
        by anything, since its flash would hold the silence too; what is
        modelled here is the recoverable case, where writing the previous
        value back works.
        """

        def _answer(self, line: str) -> None:
            if line == "#fparam set %s 7" % IMU_PACKET_NAME:
                self.commands.append(line)
                self.parameters[IMU_PACKET_NAME] = "0"   # no output
                self._say(OK)
                return
            super()._answer(line)

    module = _MisreadsTheIndex(rate_hz=50.0)
    source = _source(module)
    try:
        with pytest.raises(TuneRefused, match="stopped it sending"):
            source.tune(IMU_PACKET_NAME, "100")
        assert module.rate_hz == 50.0, "the restart put it back"
        assert module.streaming is True
        source.arm(1, buffer_record_count=4)
        assert source.read_records(1, timeout=3.0, exact=True)
        source.finish_record_capture()
    finally:
        source.close()


# ------------------------------------------------------ leaving the console
def test_every_way_out_of_the_console_is_checked_by_whole_packets() -> None:
    """A parameter write that left the module silent used to report success."""

    class _StaysQuiet(_FakeModule):
        """Acknowledges everything, and never comes back on the air.

        Both ways out matter: a settings write leaves through the restart,
        and a plain read leaves through #fdeconfig.
        """

        armed = False

        def _answer(self, line: str) -> None:
            super()._answer(line)
            if line in ("#fdeconfig", "y") and self.armed:
                self.streaming = False

    module = _StaysQuiet()
    source = _source(module)
    try:
        module.armed = True
        # The write reaches the module; what fails is that it never comes
        # back on the air, and the driver says so rather than reporting the
        # write as done.
        with pytest.raises(RuntimeError, match="stopped it sending"):
            source.tune(IMU_PACKET_NAME, 50.0)
    finally:
        source.close()


def test_a_heartbeat_is_not_a_navigating_module() -> None:
    """The module emits a 1 Hz heartbeat whose first byte is a frame header.

    Judging "the stream is back" by a frame header would read that
    heartbeat as navigation resumed, which is how a silent module gets
    reported as a working one. It takes a WHOLE IMU packet.
    """

    class _HeartbeatOnly(_FakeModule):
        armed = False

        def _answer(self, line: str) -> None:
            super()._answer(line)
            if line in ("#fdeconfig", "y") and self.armed:
                self.streaming = False
                self._out += bytes((FRAME_HEAD, 0xF0))   # the documented heartbeat

    module = _HeartbeatOnly()
    source = _source(module)
    try:
        module.armed = True
        # The write reaches the module; what fails is that it never comes
        # back on the air, and the driver says so rather than reporting the
        # write as done.
        with pytest.raises(RuntimeError, match="stopped it sending"):
            source.tune(IMU_PACKET_NAME, 50.0)
    finally:
        source.close()


def test_a_module_left_in_its_console_is_found_and_opened_again() -> None:
    """Config mode outlives the process that opened it, so it must be undone."""

    module = _FakeModule()
    module.streaming = False       # exactly where a dead session leaves it
    module._answer("#fconfig")

    assert _listen_for_packets(module, 0.05) == 0, "a stuck module says nothing"
    wake_from_config_mode(module)
    assert _listen_for_packets(module, 0.05) >= 2, "one #fdeconfig brings it back"

    stuck = _FakeModule()
    stuck.streaming = False
    stuck._answer("#fconfig")
    source = _source(stuck)
    try:
        assert source.working_point().settings["packet_rate_hz"] > 0
        assert "#fdeconfig" in stuck.commands
    finally:
        source.close()


def test_a_module_whose_console_stays_silent_still_streams() -> None:
    """No console is a fact about the module, not a reason to refuse it.

    And the rate is still known: it is read off the STREAM, which is where
    the sample interval that stamps every record comes from anyway. Opening
    this device sends no console command at all, so a module whose console
    never answers is fully usable and its one knob is shown -- which is
    what this test always meant by the sentence above.
    """

    class _Mute(_FakeModule):
        def _answer(self, line: str) -> None:
            self.commands.append(line)

    module = _Mute()
    source = _source(module)
    try:
        assert source.tunable_values() == {IMU_PACKET_NAME: "100"}
        point = source.working_point()
        assert point.settings["settings"] == {IMU_PACKET_NAME: 100.0}
        assert not point.settings["settings_refusal"]
        assert not module.commands, "opening it asked the console nothing"
        assert point.sample_interval_seconds == pytest.approx(0.01)
        source.arm(1, buffer_record_count=4)
        assert source.read_records(1, timeout=3.0, exact=True)
    finally:
        source.close()


# ------------------------------------------------------------- the magnetics
def test_the_module_says_how_often_its_magnetic_field_changes() -> None:
    """Counting repeats bounds the magnetometer's rate from above.

    Nothing the vendor ships states the magnetometer's own output rate --
    its specification table is a verbatim lift from another manufacturer's
    part -- so raising the packet rate could buy nothing but duplicate
    readings. What is counted is how often the published field CHANGES,
    which is the slower of the magnetometer's sampling and the field's own
    movement, so it is an upper bound on the first rather than a
    measurement of it.
    """

    packets_per_reading = 4
    module = _FakeModule(rate_hz=200.0, magnetic_repeat=packets_per_reading)
    source = _source(module)
    try:
        point = source.working_point()
        assert point.settings["packet_rate_hz"] == pytest.approx(200.0, rel=0.05)
        assert point.settings["magnetic_change_hz"] == pytest.approx(
            200.0 / packets_per_reading, rel=0.15
        )
    finally:
        source.close()

    quick = _FakeModule(rate_hz=200.0, magnetic_repeat=1)
    source = _source(quick)
    try:
        point = source.working_point()
        assert point.settings["magnetic_change_hz"] == pytest.approx(200.0, rel=0.05)
    finally:
        source.close()


def test_a_module_that_takes_its_time_is_still_answering() -> None:
    """A pause before a reply is not a reply that will not come.

    "#fmsg" prints some 1900 bytes and takes its time about starting. The
    console used to give up after its quiet window even when NOTHING had
    arrived yet, read that as "the module said nothing", and hand back an
    empty settings list -- which is why Device Control opened empty against
    a module that was answering perfectly well.
    """

    module = _FakeModule(rate_hz=10.0)
    module.reply_delay = 0.6          # a long pause before it starts
    # The window has to outlast what the module takes to start talking:
    # that is the rule on the real one too, where it is two seconds.
    console_module.REPLY_QUIET_SECONDS = 0.3
    console_module.REPLY_TIMEOUT_SECONDS = 4.0
    source = _source(module)
    try:
        values = source.tunable_values()
        assert values[IMU_PACKET_NAME] == "10"
    finally:
        source.close()


def test_a_slow_console_never_reads_the_previous_reply() -> None:
    """A reply that arrives late must not be read as the next one's answer.

    The console has no sequence numbers. Returning from a command before
    its answer arrives puts that answer inside the NEXT command's window,
    and every command after it reads the previous one's reply -- so every
    "#fparam get X" comes back about some other parameter and is scored as
    "this firmware does not have X". That is how Device Control came up
    empty against a module that was answering every question correctly.
    """

    module = _FakeModule(rate_hz=10.0)
    module.reply_delay = 0.9          # far longer than the quiet window
    # The window has to outlast what the module takes to start talking:
    # that is the rule on the real one too, where it is two seconds.
    console_module.REPLY_QUIET_SECONDS = 0.3
    console_module.REPLY_TIMEOUT_SECONDS = 4.0
    source = _source(module)
    try:
        values = source.tunable_values()
        assert values[IMU_PACKET_NAME] == "10"
    finally:
        source.close()


def test_a_heartbeat_does_not_block_entering_config_mode() -> None:
    """The 1 Hz heartbeat is what a module sends when it is NOT navigating.

    Counting it as "still streaming" made the two halves of this driver read
    the same two bytes in opposite directions: leaving the console refuses a
    heartbeat as proof the stream is back, while entering used to accept it
    as proof the stream never stopped. A module whose tick fell on the wrong
    side of #fconfig was then refused.
    """

    class _Ticks(_FakeModule):
        def _answer(self, line: str) -> None:
            super()._answer(line)
            if line == "#fconfig":
                # Not navigating any more -- and saying so the way the
                # module does, once a second.
                self._out += bytes((FRAME_HEAD, 0xF0))

    module = _Ticks(rate_hz=10.0)
    with FdiConfigConsole(module) as console:
        assert module.streaming is False, "it entered, heartbeat and all"
        assert console.get_parameter(IMU_PACKET_NAME) == "4", "rung 4 is 10 Hz"


def test_a_refused_save_is_not_a_save() -> None:
    """*#ERROR is this console's own word for no, and a save is built on."""

    class _RefusesSave(_FakeModule):
        def _answer(self, line: str) -> None:
            if line == "#fsave":
                self.commands.append(line)
                self._say("*#ERROR")
                return
            super()._answer(line)

    module = _RefusesSave()
    source = _source(module)
    try:
        with pytest.raises((TuneRefused, RuntimeError)):
            source.tune(IMU_PACKET_NAME, 50.0)
        assert source.tunable_values()[IMU_PACKET_NAME] == "100", (
            "the settings must not claim a value the module would not keep"
        )
    finally:
        source.close()


def test_somebody_else_s_reply_is_not_a_missing_parameter() -> None:
    """Absence is what the module SAYS, not what a regex fails to find.

    Reading "no such parameter" off "nothing matched" is what emptied Device
    Control: any stray text satisfies it, so a knob the panel was showing a
    moment ago vanishes with no reason recorded.
    """

    class _AnswersLate(_FakeModule):
        """Replies to a get with the tail of some earlier command."""

        stray = ""

        def _answer(self, line: str) -> None:
            if line.startswith("#fparam get ") and self.stray:
                self.commands.append(line)
                self._say(self.stray)
                return
            super()._answer(line)

    module = _AnswersLate()
    with FdiConfigConsole(module) as console:
        # The module's own way of saying it has not got one.
        assert console.get_parameter("NO_SUCH_PARAMETER") is None

        for stray in ("*#OK", "(y/n)", "MSG_ODOMETER[6f]   0.0Hz", "Config Mode"):
            module.stray = stray
            with pytest.raises(RuntimeError, match="neither the value nor"):
                console.get_parameter(IMU_PACKET_NAME)
