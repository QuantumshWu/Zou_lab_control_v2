from __future__ import annotations

from threading import Event, Thread
import time

import numpy as np
import pytest

from tests.fakes import FakePlane
from zlc_atom.authoring import AuthoringSchema
from zlc_atom.install import (
    DeviceCatalogSnapshot,
    DeviceSpec,
    DeviceTypeDescriptor,
    InstalledLeaf,
    create_installation,
)
from zlc_atom.data import snapshot_from_array
from zlc_atom.devices.camera import CameraFrameRecord
from zlc_atom.nodes.camera_measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
    MonitorCapture,
)


def _one_camera_window_program():
    from zlc_pulse import (
        PulsePeriod,
        PulseSequence,
        compile_sequence,
        load_streamer_config,
        pulse_target_from_xdc,
    )

    config = load_streamer_config()
    target = pulse_target_from_xdc()
    trigger_lane = target.by_key["emCCD"].lanes[0]
    trigger_index = target.raw_lanes.index(trigger_lane)
    high = [0] * len(target.raw_lanes)
    high[trigger_index] = 1
    sequence = PulseSequence(
        "one_camera_window",
        target,
        1e9 / config["clock_hz"],
        (
            PulsePeriod("expose", 0.02, "s", tuple(high)),
            PulsePeriod(
                "close",
                1e9 / config["clock_hz"],
                "ns",
                (0,) * len(high),
            ),
        ),
    )
    return compile_sequence(sequence, config["params"], config["clock_hz"])


def test_repeat_zero_monitor_replaces_latest_only_with_a_complete_camera_cycle() -> None:
    installation = create_installation("virtual")
    plane = FakePlane()
    try:
        measurement = CameraMeasurementNode(
            camera=installation.device("camera"),
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=(2, 3, 20, 16),
                repeat=0,
                frames_per_cycle=3,
                photoelectrons=False,
            ),
            signal_plane=plane,
        )
        monitor = measurement.monitor()
        assert isinstance(monitor, MonitorCapture)
        camera = installation.device("camera")

        # Continuous capture keeps four whole cycles.  Let that 12-frame raw
        # buffer advance once before the consumer reads: it now contains
        # ordinals 1..12.  The leading 1/2 cannot be combined with 3, and the
        # retained capacity must still leave 3/4/5 available as the next shot.
        camera.trigger(13, frame=np.zeros((96, 128), dtype=np.uint16))
        deadline = time.monotonic() + 5.0
        while camera.produced_count < 13 and time.monotonic() < deadline:
            time.sleep(0.001)
        for _ in range(3):
            monitor.poll()
        assert monitor.latest_record is not None
        assert monitor.latest_record.source_ordinal == 3
        signal_key = measurement.signal_key("frames")
        plane.freeze()
        assert plane.latest_publication(signal_key) is None

        # Physical ordinals 4/5 complete the already retained 3/4/5 shot.
        publication = None
        deadline = time.monotonic() + 1.0
        while publication is None and time.monotonic() < deadline:
            monitor.poll()
            plane.freeze()
            publication = plane.latest_publication(signal_key)
        assert publication is not None
        assert set(publication.signals) == {signal_key}
        value = publication.value(signal_key)
        assert value is not None
        # One monitor snapshot: (repeat=1, frame points=cycle, y, x).
        assert value.snapshot.block.values.shape == (1, 3, 16, 20)
        assert value.snapshot.block.values.dtype.str == "<u2"
        assert measurement.request.roi_xywh == (2, 3, 20, 16)
        assert measurement.actual_working_point is not None
        assert measurement.actual_working_point.roi_origin_yx == (3, 2)
        assert any(call[0] == "begin_generation" for call in plane.calls)
        front = plane.freeze()
        assert signal_key in front.signals
        terminal = monitor.close()
        assert terminal.source_stopped and terminal.joined
        assert measurement.camera.capture_state() is False
        # The close ends the run; the sealed monitor publication is
        # retained for the panels (and derivations) that still show it.
        assert plane.latest_publication(signal_key) is not None
        assert not plane.is_generation_live(signal_key)
    finally:
        plane.close()
        installation.close()


def test_repeated_freezes_share_one_schema_and_retain_the_frame_bytes() -> None:
    """A live monitor freezes at up to 10 Hz; the schema is a configuration fact.

    One DatasetSchema instance per ordered Point/Cell axis declaration and
    array shape/dtype
    keeps schema identity stable, so the per-instance fingerprint cache and
    every downstream schema-fingerprint consumer hit instead of re-hashing per
    freeze -- and the bytes-backed camera frame is retained as a view rather
    than copied again at the publication boundary.
    """

    from zlc_data import SPATIAL_X, SPATIAL_Y

    record = CameraFrameRecord(
        np.arange(48, dtype="<u2").reshape(6, 8), 0, host_received_at_ns=1
    )
    view = np.asarray(record.image)[None, ...]
    first = snapshot_from_array(
        view, producer="cam", signal="frames",
        cell_axes=(SPATIAL_Y, SPATIAL_X), generation="g", revision=1,
    )
    second = snapshot_from_array(
        view, producer="cam", signal="frames",
        cell_axes=(SPATIAL_Y, SPATIAL_X), generation="g", revision=2,
    )
    assert first.block.schema is second.block.schema
    assert first.ref.schema_fingerprint == second.ref.schema_fingerprint
    assert (
        second.block.values.__array_interface__["data"][0]
        == record.image.__array_interface__["data"][0]
    )

    # Another shape (a reconfigure) is another configuration: another schema.
    other = snapshot_from_array(
        np.zeros((1, 6, 9), dtype="<u2"), producer="cam", signal="frames",
        cell_axes=(SPATIAL_Y, SPATIAL_X), generation="g", revision=3,
    )
    assert other.block.schema is not first.block.schema
    assert other.ref.schema_fingerprint != first.ref.schema_fingerprint


def test_snapshot_array_axes_are_positional_and_allow_repeated_roles() -> None:
    from zlc_data import AxisId, AxisSpec, SCAN_POINT, SITE

    slow = AxisSpec(AxisId("scan.slow"), "slow", SCAN_POINT, 2, (10, 20))
    fast = AxisSpec(AxisId("scan.fast"), "fast", SCAN_POINT, 3, (1, 2, 3))
    left = AxisSpec(AxisId("cell.left"), "left", SITE, 2)
    right = AxisSpec(AxisId("cell.right"), "right", SITE, 4)
    values = np.arange(1 * 2 * 3 * 2 * 4).reshape(1, 2, 3, 2, 4)

    snapshot = snapshot_from_array(
        values,
        producer="ordered",
        signal="values",
        point_axes=(slow, fast),
        cell_axes=(left, right),
        generation="g",
        revision=1,
    )

    schema = snapshot.block.schema
    assert schema.point_domain.axes == (slow, fast)
    assert schema.cell_domain.axes == (left, right)
    assert tuple(schema.point_domain.codes(slow.axis_id)) == (0, 0, 0, 1, 1, 1)
    assert tuple(schema.point_domain.codes(fast.axis_id)) == (0, 1, 2, 0, 1, 2)
    assert snapshot.block.values.shape == (1, 6, 2, 4)
    np.testing.assert_array_equal(snapshot.block.values, values.reshape(1, 6, 2, 4))

    with pytest.raises(TypeError, match="ordered sequence"):
        snapshot_from_array(
            np.zeros((1, 2)),
            producer="ordered",
            signal="old-mapping",
            point_axes={SCAN_POINT: slow},  # type: ignore[arg-type]
            generation="g",
            revision=1,
        )


def test_a_validity_mask_that_is_not_bool_is_refused_not_truthed() -> None:
    """A status code is not a validity; Data's bool contract is the only door.

    The wrapper cast whatever it was handed with ``dtype=bool`` before the
    Dataset saw it, so an int mask of ``[0, 2]`` -- an SDK status, a count
    handed over by mistake -- became ``[False, True]`` and status 2 entered
    the valid data plane.  The mask now reaches the one Data constructor as
    it is, and that constructor refuses anything but bool.
    """

    from zlc_data import SCAN_POINT

    values = np.array([[10.0, 20.0]])
    with pytest.raises(TypeError, match="validity mask dtype must be bool"):
        snapshot_from_array(
            values,
            producer="p",
            signal="s",
            point_axes=(SCAN_POINT,),
            generation="g",
            revision=1,
            validity=np.array([[0, 2]], dtype=np.int64),
        )
    accepted = snapshot_from_array(
        values,
        producer="p",
        signal="s",
        point_axes=(SCAN_POINT,),
        generation="g",
        revision=1,
        validity=np.array([[False, True]]),
    )
    assert accepted.expanded_validity().reshape(-1).tolist() == [False, True]


def test_direct_monitor_disarms_when_empty_generation_retire_fails() -> None:
    installation = create_installation("virtual")
    plane = FakePlane()
    try:
        camera = installation.device("camera")
        measurement = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=None,
                repeat=0,
                frames_per_cycle=1,
            ),
            signal_plane=plane,
        )
        monitor = measurement.monitor()

        def fail_retire(_node: object) -> None:
            raise RuntimeError("synthetic retire failure")

        plane.retire = fail_retire  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="synthetic retire failure"):
            monitor.close()
        assert camera.capture_state() is False
    finally:
        plane.close()
        installation.close()


def test_finite_measurement_collects_only_external_triggers() -> None:
    installation = create_installation("virtual")
    plane = FakePlane()
    try:
        measurement = CameraMeasurementNode(
            camera=installation.device("camera"),
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=None,
                repeat=3,
                frames_per_cycle=1,
            ),
            signal_plane=plane,
        )
        result_box: list[object] = []
        worker = Thread(
            target=lambda: result_box.append(measurement.measure()),
            daemon=True,
        )
        worker.start()
        deadline = time.monotonic() + 1.0
        while not installation.device("camera").capture_state() and time.monotonic() < deadline:
            time.sleep(0.001)
        sequencer = installation.device("sequencer")
        sequencer.load(_one_camera_window_program())
        for _ in range(3):
            sequencer.fire(run_repeats=1, scan_repeats=1)
            sequencer.wait_done(1.0)
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert len(result_box) == 1
        result = result_box[0]
        assert len(result.frames) == 3  # type: ignore[union-attr]
    finally:
        plane.close()
        installation.close()


def _refusing_once(camera: object) -> list[int]:
    """Make the camera's next disarm fail once; later ones run the real one.

    Set on the instance, so the adapter is still the same CameraAdapter the
    node type-checked.  Returns the attempt log.
    """

    disarm = camera.finish_record_capture
    attempts: list[int] = []

    def refuse_once():
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise OSError("controlled SDK could not disarm")
        return disarm()

    camera.finish_record_capture = refuse_once  # type: ignore[method-assign]
    return attempts


def test_a_monitor_disarm_the_device_refused_is_retried_and_never_made_up() -> None:
    """Only a terminal the device produced is cached and handed back.

    ``closed`` used to be set before the device was asked, so a disarm the
    device refused left the second close answering ``(0, True, True, True)``
    -- an all-clear over a camera still armed -- and the generation had been
    detached before the device stopped.  A refused disarm keeps the capture
    open and the generation live; the next close retries the same disarm and
    detaches only once the device has stopped.
    """

    installation = create_installation("virtual")
    plane = FakePlane()
    camera = installation.device("camera")
    try:
        measurement = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=None,
                repeat=0,
                frames_per_cycle=1,
            ),
            signal_plane=plane,
        )
        monitor = measurement.monitor()
        attempts = _refusing_once(camera)

        with pytest.raises(OSError, match="could not disarm"):
            monitor.close()
        assert monitor.terminal is None
        assert camera.capture_state() is True, "reported closed over a camera still armed"
        assert not any(call[0] == "retire" for call in plane.calls), (
            "the generation was detached before the device had stopped"
        )

        terminal = monitor.close()
        assert attempts == [0, 1], "the second close did not retry the disarm"
        assert terminal.source_stopped and terminal.joined
        assert terminal.produced_count == 0
        assert camera.capture_state() is False
        assert [call[0] for call in plane.calls].count("retire") == 1
        assert monitor.close() is terminal
        assert attempts == [0, 1]
    finally:
        camera.__dict__.pop("finish_record_capture", None)
        plane.close()
        installation.close()


def test_a_finite_disarm_the_device_refused_leaves_the_capture_open_for_the_next_close() -> None:
    """A finish the device refused is not a finish; the next close retries it.

    The capture used to mark itself closed before asking the device, so the
    second close raised "closed without terminal evidence" and no owner could
    ever retry the disarm of a camera the device still held armed.
    """

    installation = create_installation("virtual")
    plane = FakePlane()
    camera = installation.device("camera")
    measurement = None
    try:
        measurement = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=None,
                repeat=1,
                frames_per_cycle=1,
            ),
            signal_plane=plane,
        )
        capture = measurement.prepare()
        camera.trigger(1, frame=np.zeros((96, 128), dtype=np.uint16))
        cycle = capture.next_cycle()
        assert cycle is not None and len(cycle) == 1
        attempts = _refusing_once(camera)

        with pytest.raises(OSError, match="could not disarm"):
            capture.close()
        assert capture.terminal is None and not capture.closed
        assert camera.capture_state() is True

        terminal = capture.close()
        assert attempts == [0, 1], "the second close did not retry the disarm"
        assert terminal.produced_count == 1 and terminal.source_stopped
        assert camera.capture_state() is False
        assert capture.close() is terminal
        assert attempts == [0, 1]
    finally:
        camera.__dict__.pop("finish_record_capture", None)
        if measurement is not None:
            plane.retire(measurement)
        plane.close()
        installation.close()


def test_a_direct_stop_half_way_through_a_cycle_keeps_the_complete_cycles_it_took() -> None:
    """Stop keeps the measured prefix, sealed short; the partial cycle is the device's.

    A direct finite capture asked to stop after one complete cycle and one
    frame of the next used to fail its terminal check -- the camera had
    honestly produced three frames for two accounted -- and then withdrew
    the complete cycle it had already published.
    """

    installation = create_installation("virtual")
    plane = FakePlane()
    try:
        camera = installation.device("camera")
        measurement = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=None,
                repeat=2,
                frames_per_cycle=2,
            ),
            signal_plane=plane,
        )
        stop = Event()
        capture = measurement.prepare(should_stop=stop.is_set)
        result_box: list[object] = []
        worker = Thread(
            target=lambda: result_box.append(capture.collect()),
            daemon=True,
        )
        worker.start()
        signal_key = measurement.signal_key("frames")
        frame = np.zeros((96, 128), dtype=np.uint16)
        camera.trigger(2, frame=frame)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            plane.freeze()
            if plane.latest_publication(signal_key) is not None:
                break
            time.sleep(0.005)
        assert plane.latest_publication(signal_key) is not None
        camera.trigger(1, frame=frame)
        while camera.produced_count < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert camera.produced_count == 3
        stop.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert result_box, "the stopped capture raised instead of keeping its prefix"
        result = result_box[0]
        assert result is not None
        assert result.cycle_count == 1  # type: ignore[union-attr]
        assert result.terminal.produced_count == 3  # type: ignore[union-attr]
        assert capture.stopped
        plane.freeze()
        assert plane.latest_publication(signal_key) is not None, (
            "the complete cycle was withdrawn"
        )
        assert not plane.is_generation_live(signal_key)
        assert not any(call[0] == "retire" for call in plane.calls)
    finally:
        plane.close()
        installation.close()


def test_installation_isolates_one_factory_failure_and_closes_successful_leaves() -> None:
    closed: list[str] = []

    def good_factory(_context, key, _values):
        return InstalledLeaf(
            key,
            "test.good",
            object(),
            {},
            closer=lambda: closed.append(key),
        )

    def bad_factory(_context, _key, _values):
        raise RuntimeError("synthetic startup failure")

    descriptors = (
        DeviceTypeDescriptor("test.good", "test", AuthoringSchema(()), (), factory=good_factory),
        DeviceTypeDescriptor("test.bad", "test", AuthoringSchema(()), (), factory=bad_factory),
    )
    installation = create_installation(
        (DeviceSpec("good", "test.good"), DeviceSpec("bad", "test.bad")),
        catalog=DeviceCatalogSnapshot(descriptors, ()),
    )
    assert set(installation.devices) == {"good"}
    assert isinstance(installation.failures["bad"], RuntimeError)
    installation.close()
    assert closed == ["good"]


def test_missing_dependency_is_a_graph_error_not_a_partial_device() -> None:
    descriptor = DeviceTypeDescriptor(
        "test.dependent",
        "test",
        AuthoringSchema(()),
        (),
        dependencies=("test.missing",),
        factory=lambda _context, key, _values: InstalledLeaf(key, "test.dependent", object(), {}),
    )
    with pytest.raises(ValueError, match="missing dependencies"):
        create_installation(
            (DeviceSpec("dependent", "test.dependent"),),
            catalog=DeviceCatalogSnapshot((descriptor,), ()),
        )
