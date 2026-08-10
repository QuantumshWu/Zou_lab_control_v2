from __future__ import annotations

from threading import Thread
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
        signal_keys = tuple(
            measurement.signal_key(f"frame_{index}") for index in range(3)
        )
        plane.freeze()
        assert plane.latest_publication(signal_keys[0]) is None

        # Physical ordinals 4/5 complete the already retained 3/4/5 shot.
        publication = None
        deadline = time.monotonic() + 1.0
        while publication is None and time.monotonic() < deadline:
            monitor.poll()
            plane.freeze()
            publication = plane.latest_publication(signal_keys[0])
        assert publication is not None
        assert set(publication.signals) == set(signal_keys)
        assert all(plane.latest_publication(key) is publication for key in signal_keys)
        for key in signal_keys:
            value = publication.value(key)
            assert value is not None
            assert value.snapshot.block.values.shape[-2:] == (16, 20)
            assert value.snapshot.block.values.dtype.str == "<u2"
        assert measurement.request.roi_xywh == (2, 3, 20, 16)
        assert measurement.actual_working_point is not None
        assert measurement.actual_working_point.roi_origin_yx == (3, 2)
        assert any(call[0] == "reserve" for call in plane.calls)
        assert any(call[0] == "mark_changed" for call in plane.calls)
        front = plane.freeze()
        assert all(key in front.signals for key in signal_keys)
        terminal = monitor.close()
        assert terminal.source_stopped and terminal.joined
        assert monitor.slot.closed
        assert measurement.camera.capture_state() is False
        assert all(plane.latest_publication(key) is None for key in signal_keys)
    finally:
        plane.close()
        installation.close()


def test_direct_monitor_disarms_when_live_detach_fails() -> None:
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
        monitor = measurement.monitor(buffer_frames=1)

        def fail_detach(_node: object) -> None:
            raise RuntimeError("synthetic detach failure")

        plane.detach_live = fail_detach  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="synthetic detach failure"):
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
            sequencer.fire()
            sequencer.wait_done(1.0)
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert len(result_box) == 1
        result = result_box[0]
        assert len(result.frames) == 3  # type: ignore[union-attr]
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
