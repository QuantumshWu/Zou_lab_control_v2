"""A waveform source is a camera for time: one record, one shot, one measurement."""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zlc_atom.devices.simulation.waveform import (
    VirtualWaveformConfig,
    VirtualWaveformSource,
)
from zlc_atom.devices.waveform.contract import WaveformOutput
from zlc_atom.devices.waveform.tek_scope import (
    TIME_PER_DIV_FIELD,
    TekScopeConfig,
    TekScopeWaveformSource,
    volts_per_div_field,
)
from zlc_atom.devices.waveform.wheeltec_n100 import (
    FRAME_HEAD,
    FRAME_TAIL,
    IMU_PACKET,
    N100_OUTPUTS,
    drain_imu_samples,
    header_crc8,
    payload_crc16,
)
from zlc_atom.nodes.waveform_measurement import (
    LOGIC_NODE,
    MIN_READ_INTERVAL_SECONDS,
    WaveformMeasurementNode,
    WaveformMeasurementRequest,
)
from zlc_data import PRIMARY_INDEX, SHOT_TIME, AxisId
from zlc_runtime.host import NodeHost
from zlc_runtime.plane import SignalDataPlane


def _frame(kind: int, payload: bytes, serial: int = 0) -> bytes:
    """One FDILink frame, checks and all, exactly as the module sends it.

    The two checks are not decoration: the reader refuses a frame whose
    header CRC8 or payload CRC16 disagrees, so a test that filled them with
    zeroes would be testing a stream no module produces.
    """

    head = bytes((FRAME_HEAD, kind, len(payload), serial))
    check = payload_crc16(payload)
    return (
        head
        + bytes((header_crc8(head), check >> 8, check & 0xFF))
        + payload
        + bytes((FRAME_TAIL,))
    )


def _imu_packet(
    *,
    gyro: tuple[float, float, float],
    accel: tuple[float, float, float],
    mag_milligauss: tuple[float, float, float],
    celsius: float,
    microseconds: int,
    serial: int = 0,
) -> bytes:
    payload = struct.pack(
        "<12fq", *gyro, *accel, *mag_milligauss, celsius, 1013.0, 24.5, microseconds
    )
    return _frame(IMU_PACKET, payload, serial)


def test_the_fdilink_stream_parses_into_samples_in_published_units() -> None:
    """Whole IMU packets come out in order; noise, other packets and a torn
    tail do not: the tail waits for the bytes that complete it."""

    first = _imu_packet(
        gyro=(0.1, 0.2, 0.3),
        accel=(0.0, 0.0, 9.8),
        mag_milligauss=(200.0, -50.0, 450.0),
        celsius=26.85,
        microseconds=1_000_000,
    )
    second = _imu_packet(
        gyro=(0.0, 0.0, 0.0),
        accel=(0.0, 0.0, 9.8),
        mag_milligauss=(201.0, -49.0, 449.0),
        celsius=26.85,
        microseconds=1_002_500,
    )
    # Two packets this driver does not publish: an AHRS frame whose length
    # it does know, and a local-magnetic-field frame whose length it has
    # never been told.  Both are stepped over by the length the module
    # itself stated, so neither needs a table here.
    ahrs = _frame(0x41, bytes(48), serial=1)
    local_field = _frame(0x6E, struct.pack("<3f", 1.0, 2.0, 3.0), serial=2)
    buffer = bytearray(b"\xfc\x99" + first + ahrs + local_field + second[:20])
    samples = drain_imu_samples(buffer)

    assert [stamp for stamp, _values in samples] == [1.0]
    values = samples[0][1]
    assert values[0:3] == pytest.approx((20.0, -5.0, 45.0))
    assert values[3:6] == pytest.approx((0.1, 0.2, 0.3))
    assert values[6:9] == pytest.approx((0.0, 0.0, 9.8))
    assert values[9] == pytest.approx(300.0)
    assert bytes(buffer) == second[:20]

    buffer += second[20:]
    (stamp, values), = drain_imu_samples(buffer)
    assert stamp == pytest.approx(1.0025)
    assert values[0] == pytest.approx(20.1)
    assert not buffer


def test_a_frame_that_does_not_check_out_is_not_a_magnetic_field() -> None:
    """One flipped bit anywhere in a packet drops it, rather than publishing it.

    A serial line at 100 Hz and up carries far more frames than anyone
    inspects, and the one number this bench takes from them is a magnetic
    field. A frame whose payload lost a bit would otherwise arrive as a
    perfectly plausible reading, which is the worst kind of wrong.
    """

    good = _imu_packet(
        gyro=(0.0, 0.0, 0.0),
        accel=(0.0, 0.0, 9.8),
        mag_milligauss=(200.0, -50.0, 450.0),
        celsius=26.85,
        microseconds=5_000_000,
    )
    assert len(drain_imu_samples(bytearray(good))) == 1

    for position in (1, 3, 8, 30, len(good) - 2):
        torn = bytearray(good)
        torn[position] ^= 0x01
        assert drain_imu_samples(bytearray(torn)) == [], (
            f"a bit flipped at byte {position} was published as data"
        )


def _imu_like_source(rate_hz: float) -> VirtualWaveformSource:
    """Packets whose x field counts the packet, so a shot says which packet it was."""

    def samples(times: np.ndarray) -> np.ndarray:
        out = np.zeros((times.size, 10), dtype=np.float32)
        out[:, 0] = np.round(times * rate_hz)
        out[:, 1] = -5.0
        out[:, 2] = 45.0
        out[:, 8] = 9.8
        out[:, 9] = 300.0
        return out

    return VirtualWaveformSource(
        VirtualWaveformConfig(rate_hz, 1, N100_OUTPUTS), sample_source=samples
    )


def _scope_like_source() -> VirtualWaveformSource:
    outputs = (WaveformOutput("voltage", "V", ("CH1", "CH2"), (0, 1)),)
    return VirtualWaveformSource(
        VirtualWaveformConfig(10000.0, 100, outputs),
        sample_source=lambda times: np.zeros((times.size, 2), dtype=np.float32),
    )


def _host(node: WaveformMeasurementNode, plane: SignalDataPlane, wake: Event) -> NodeHost:
    return NodeHost(
        node,
        plane,
        wake.set,
        instance_id=node.instance_id,
        kind="measurement",
        dataset_output_declarations=node.dataset_output_declarations,
    )


def _drive(host: NodeHost, until, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        host.poll()
        result = until()
        if result is not None:
            return result
        time.sleep(0.005)
    return None


def test_one_measurement_publishes_what_its_source_carries() -> None:
    """The outputs and the preview are the bound instrument's, not a node's.

    An IMU packet is four quantities and is watched as a rolling trace of
    shots; a scope acquisition is one voltage and is watched as the trace
    it is.  Until a draft binds a source it publishes nothing, and the
    cadence it may be read at has a floor.
    """

    imu = _imu_like_source(500.0)
    scope = _scope_like_source()
    try:
        assert LOGIC_NODE.outputs_for({}, {}) == ()
        imu_outputs = LOGIC_NODE.outputs_for({}, {"sampler": imu})
        assert [output.name for output in imu_outputs] == [
            "magnetic_field",
            "angular_rate",
            "acceleration",
            "temperature",
        ]
        assert all(output.index_by_source for output in imu_outputs)
        (preview,) = LOGIC_NODE.previews_for({}, {"sampler": imu})
        assert preview.output == imu_outputs[0] and preview.plot_kind == "rolling"
        (voltage,) = LOGIC_NODE.outputs_for({}, {"sampler": scope})
        assert voltage.name == "voltage" and voltage.contract_id == "waveform.voltage"
        (preview,) = LOGIC_NODE.previews_for({}, {"sampler": scope})
        assert preview.plot_kind == "curve"
        (requirement,) = LOGIC_NODE.device_requirements
        assert requirement.fields_frozen_by(imu) == ()
        with pytest.raises(ValueError, match="at least"):
            WaveformMeasurementRequest("imu", repeat=0, read_interval_seconds=0.0)
        field = next(
            field
            for field in LOGIC_NODE.authoring_schema.fields
            if field.name == "read_interval_seconds"
        )
        assert field.minimum == MIN_READ_INTERVAL_SECONDS
    finally:
        imu.close()
        scope.close()


def test_every_packet_is_a_shot_and_a_rolling_window_keeps_the_last_ones() -> None:
    """Read faster than the source produces, each record publishes at once
    as (1) x () x (channel) per quantity, in its own unit; a history lease
    makes the plane keep the last N shots by the measurement's own
    sequence, which is what a Rolling panel reads."""

    plane = SignalDataPlane()
    source = _imu_like_source(500.0)
    node = WaveformMeasurementNode(
        sampler=source,
        request=WaveformMeasurementRequest("imu", repeat=0, read_interval_seconds=0.001),
        signal_plane=plane,
        producer="imu-live",
    )
    wake = Event()
    host = _host(node, plane, wake)
    history = None
    try:
        host.start()
        assert host.wait_ready(5), "the monitor did not report ready"
        assert source.capture_state()
        key = host.signal_key("magnetic_field")
        value = _drive(host, lambda: plane.freeze().value(key))
        assert value is not None, "the hosted monitor published no shot"
        schema = value.snapshot.block.schema
        assert schema.physical_shape == (1, 1, 3)
        (channel_axis,) = schema.cell_domain.axes
        assert channel_axis.coordinate_labels == ("x", "y", "z")
        assert schema.value_schema.value_unit == "uT"
        assert np.asarray(value.snapshot.block.values)[0, 0, 1] == pytest.approx(-5.0)
        assert value.run_record["named_devices"] == {"sampler": "imu"}
        assert value.run_record["parameters"]["read_interval_seconds"] == 0.001
        assert value.run_record["device_snapshots"]["sampler"]["record_samples"] == 1
        temperature = plane.freeze().value(host.signal_key("temperature"))
        assert temperature is not None
        assert temperature.snapshot.block.schema.value_schema.value_unit == "K"
        assert temperature.snapshot.block.schema.physical_shape == (1, 1, 1)

        assert plane.supports_indexed_history(key)
        history = plane.acquire_indexed_history(key, 8)
        first = plane.latest_publication(key).event_ref.sequence
        _drive(host, lambda: (
            True if plane.latest_publication(key).event_ref.sequence >= first + 20 else None
        ))
        publication = plane.latest_publication(key)
        snapshot, _record = plane.current_dataset_view(key, publication)
        source_index = snapshot.block.schema.point_domain.axis(
            AxisId("zlc_data.primary-index")
        )
        assert source_index.role == PRIMARY_INDEX
        assert source_index.coordinates == tuple(range(-7, 1))
        # The window holds eight consecutive shots: their x fields are the
        # packet numbers, eight of them in a row.
        packets = np.asarray(snapshot.block.values)[0, :, 0]
        assert np.all(np.diff(packets) == 1.0)
        # And when each was taken, on the source's own clock, seconds from
        # the run's first shot: 2 ms apart at 500 Hz, beside the index.
        shot_time = snapshot.block.schema.point_domain.axis(AxisId("zlc_data.shot-time"))
        assert shot_time.role == SHOT_TIME and shot_time.unit == "s"
        times = np.asarray(shot_time.coordinates, dtype=float)
        assert np.allclose(np.diff(times), 0.002, atol=1e-6)
        assert times[0] == pytest.approx(0.002 * (packets[0] - 0.0), abs=1e-6)
        assert value.run_record["started_at_ns"] > 0

        host.cancel("test completed")
        assert _drive(host, lambda: True if host.observation.terminal else None)
        assert source.capture_state() is False
        publication = plane.latest_publication(key)
        assert publication is not None
        assert plane.retains(key, publication)
    finally:
        if history is not None:
            history.close()
        if host.observation.running:
            host.cancel("test cleanup")
            _drive(host, lambda: True if not host.observation.running else None)
        host.shutdown()
        source.close()
        plane.close()


def test_a_finite_measurement_takes_its_shots_at_its_own_cadence() -> None:
    for interval, expect_every_packet in ((0.001, True), (0.004, False)):
        plane = SignalDataPlane()
        source = _imu_like_source(500.0)
        node = WaveformMeasurementNode(
            sampler=source,
            request=WaveformMeasurementRequest(
                "imu", repeat=4, read_interval_seconds=interval
            ),
            signal_plane=plane,
            producer="imu-finite",
        )
        wake = Event()
        host = _host(node, plane, wake)
        try:
            host.start()
            assert host.wait_ready(5)
            assert _drive(host, lambda: True if host.observation.terminal else None)
            key = host.signal_key("magnetic_field")
            publication = plane.latest_publication(key)
            assert publication is not None
            value = publication.value(key)
            assert value.coverage.written_cells == 4 and value.coverage.total_cells == 4
            dataset = plane.current_dataset(key, publication)
            assert dataset.block.schema.physical_shape == (4, 1, 3)
            packets = np.asarray(dataset.block.values)[:, 0, 0]
            if expect_every_packet:
                # Read every 1 ms off a 500 Hz source: each due time waits
                # for the next packet, so the shots are the packets.
                assert packets.tolist() == [0.0, 1.0, 2.0, 3.0]
            else:
                # Read every 4 ms off a 500 Hz source: the three intervals
                # between four shots span about six packets, each shot a new
                # packet.  A cadence paced by a timed lock wait would sit on
                # the 15 ms OS timer tick and span twenty.
                assert np.all(np.diff(packets) >= 1)
                assert 5 <= packets[-1] - packets[0] <= 10
            assert source.capture_state() is False
        finally:
            host.shutdown()
            source.close()
            plane.close()


class _ScopeInstrument:
    """A Tektronix scope reduced to the registers this driver reads and writes."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.time_per_div = 1e-3
        self.volts_per_div = {1: 1.0, 2: 0.5}
        self.source = 1
        self.record_length = 8
        self.acquired = 0
        self.armed = False

    def write(self, command: str) -> None:
        self.log.append(command)
        upper = command.upper()
        if upper.startswith(":DATA:SOURCE CH"):
            self.source = int(upper.rsplit("CH", 1)[1])
        elif upper.startswith(":HORIZONTAL:SCALE "):
            requested = float(command.split()[-1])
            # A scope snaps to its 1-2-5 sequence.
            decade = 10.0 ** np.floor(np.log10(requested))
            self.time_per_div = float(
                min((1.0, 2.0, 5.0), key=lambda step: abs(step * decade - requested)) * decade
            )
        elif ":SCALE " in upper and upper.startswith(":CH"):
            channel = int(upper[3 : upper.index(":", 1)])
            self.volts_per_div[channel] = float(command.split()[-1])
        elif upper == ":ACQUIRE:STATE RUN":
            self.armed = True

    def query(self, command: str) -> str:
        self.log.append(command)
        upper = command.upper()
        if upper == "*IDN?":
            return "TEKTRONIX,DPO4104B,C012345,CF:91.1CT FV:v3.24"
        if upper == ":HORIZONTAL:RECORDLENGTH?":
            return str(self.record_length)
        if upper == ":HORIZONTAL:SCALE?":
            return f"{self.time_per_div:.4E}"
        if upper.startswith(":CH") and upper.endswith(":SCALE?"):
            return f"{self.volts_per_div[int(upper[3:4])]:.4E}"
        if upper == ":WFMOUTPRE:XINCR?":
            return f"{self.time_per_div * 10.0 / self.record_length:.6E}"
        if upper == ":WFMOUTPRE:YMULT?":
            return f"{self.volts_per_div[self.source] / 25.0:.6E}"
        if upper == ":WFMOUTPRE:YOFF?":
            return "-2.0"
        if upper == ":WFMOUTPRE:YZERO?":
            return "0.0"
        if upper == ":ACQUIRE:STATE?":
            if self.armed:
                self.armed = False
                self.acquired += 1
            return "0"
        raise AssertionError(f"unexpected query {command!r}")

    def query_int16(self, command: str) -> np.ndarray:
        self.log.append(command)
        assert command.upper() == ":CURVE?"
        return np.arange(self.record_length, dtype=np.int16) * self.source

    def close(self) -> None:
        self.log.append("<closed>")


def test_the_tek_scope_driver_scales_curves_and_snaps_its_knobs() -> None:
    instrument = _ScopeInstrument()
    scope = TekScopeWaveformSource(
        TekScopeConfig(resource="USB0::0x0699::0x0401::C012345::INSTR", channels=(1, 2)),
        link=instrument,
    )
    try:
        assert scope.identity == "tek-scope:C012345"
        assert ":DATA:ENCDG RIBINARY" in [entry.upper() for entry in instrument.log]
        assert scope.record_samples == 8
        (voltage,) = scope.outputs
        assert voltage.channel_labels == ("CH1", "CH2") and voltage.unit == "V"
        point = scope.working_point()
        assert point.sample_interval_seconds == pytest.approx(1e-3 * 10.0 / 8)
        assert point.settings[volts_per_div_field(2)] == 0.5

        # The knob answers with the scope's own step, not the request.
        assert scope.tune(TIME_PER_DIV_FIELD, 3e-3) == pytest.approx(2e-3)
        fields = {field.metadata.name: field for field in scope.tunable_fields()}
        assert fields[TIME_PER_DIV_FIELD].current == pytest.approx(2e-3)
        assert set(fields) == {TIME_PER_DIV_FIELD, "ch1_volts_per_div", "ch2_volts_per_div"}
        # A capture takes the instrument as it stands: every knob is frozen.
        (requirement,) = LOGIC_NODE.device_requirements
        assert set(requirement.fields_frozen_by(scope)) == set(fields)

        scope.arm(2, buffer_record_count=2)
        with pytest.raises(RuntimeError):
            scope.tune(TIME_PER_DIV_FIELD, 1e-3)
        records = scope.read_records(2, timeout=2.0, exact=True)
        assert [record.source_ordinal for record in records] == [0, 1]
        samples = records[0].samples
        assert samples.shape == (8, 2)
        # volts = (raw - YOFf) * YMUlt + YZEro, per channel.
        assert samples[:, 0] == pytest.approx((np.arange(8) + 2.0) * (1.0 / 25.0))
        assert samples[:, 1] == pytest.approx((np.arange(8) * 2 + 2.0) * (0.5 / 25.0))
        terminal = scope.finish_record_capture()
        assert terminal.produced_count == 2 and terminal.joined
        assert instrument.acquired == 2
        assert scope.capture_state() is False
    finally:
        scope.close()
    assert instrument.log[-1] == "<closed>"
