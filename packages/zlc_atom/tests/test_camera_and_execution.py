from __future__ import annotations

import numpy as np
import pytest
import time

from zlc_atom.devices.camera import (
    CameraAdapter,
    CameraFrameRecord,
    DcamCameraAdapter,
    PylonCameraAdapter,
)
from zlc_atom.devices.camera.binding import bind_camera
from zlc_atom.devices.simulation import (
    SimulationWorld,
    VirtualCamera,
    VirtualCameraConfig,
    VirtualPulseStreamer,
)
from zlc_atom.nodes.calibration.pulse import resolve_pulse
from zlc_atom.execution import (
    DeviceIdentityEvidenceKind,
    DeviceBroker,
    PhysicalDeviceIdentity,
    ResourceKey,
    bind_verified_device,
)
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE


def test_real_and_virtual_cameras_share_one_runtime_contract() -> None:
    world = SimulationWorld()
    virtual = VirtualCamera(frame_source=world.render_frame)
    adapters = (
        virtual,
        object.__new__(DcamCameraAdapter),
        object.__new__(PylonCameraAdapter),
    )
    assert all(isinstance(adapter, CameraAdapter) for adapter in adapters)


def test_camera_binding_rejects_an_object_outside_the_camera_contract() -> None:
    with pytest.raises(TypeError, match="canonical CameraAdapter"):
        bind_camera(object(), "bad", object(), "bad", "camera.bad")  # type: ignore[arg-type]


def test_virtual_camera_preserves_trigger_to_frame_causality_and_drops_monitor_history() -> None:
    world = SimulationWorld()
    camera = VirtualCamera(
        frame_source=lambda ordinal, exposure: world.render_frame(
            ordinal,
            exposure_seconds=exposure,
        )
    )
    camera.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=1.0)
    camera.trigger(3)
    deadline = time.monotonic() + 1.0
    while camera.produced_count < 3 and time.monotonic() < deadline:
        time.sleep(0.001)
    records = camera.read_frame_records(1, timeout=0.1, exact=True)
    assert len(records) == 1
    assert records[0].source_ordinal == 2
    assert camera.capture_state() is True
    terminal = camera.finish_record_capture()
    assert terminal.source_stopped and terminal.joined is True


def test_virtual_measurement_configuration_returns_actual_crop_and_is_idle_only() -> None:
    full = np.arange(80, dtype=np.uint16).reshape(8, 10)
    exposures: list[float] = []

    def source(_ordinal: int, exposure: float) -> np.ndarray:
        exposures.append(exposure)
        return full

    camera = VirtualCamera(
        VirtualCameraConfig(frame_shape_yx=(8, 10)),
        frame_source=source,
    )
    point = camera.configure_measurement(
        exposure_seconds=0.007,
        roi_xywh=(3, 2, 4, 3),
    )
    assert point.exposure_seconds == pytest.approx(0.007)
    assert point.sensor_shape_yx == (8, 10)
    assert point.roi_origin_yx == (2, 3)
    assert point.roi_shape_yx == (3, 4)
    assert point.frame_shape_yx == (3, 4)

    camera.arm(1, source_group_sizes=(1,), buffer_frame_count=1, timeout=1.0)
    with pytest.raises(RuntimeError, match="while armed"):
        camera.configure_measurement(exposure_seconds=0.01, roi_xywh=None)
    camera.trigger()
    record = camera.read_frame_records(1, timeout=1.0, exact=True)[0]
    np.testing.assert_array_equal(record.image, full[2:5, 3:7])
    assert exposures == [pytest.approx(0.007)]
    camera.finish_record_capture()


def test_external_gate_can_only_shorten_the_camera_working_point() -> None:
    world = SimulationWorld(seed=3)
    camera = VirtualCamera(
        VirtualCameraConfig(frame_shape_yx=world.geometry.image_shape_yx),
        frame_source=lambda ordinal, exposure: world.render_frame(
            ordinal,
            exposure_seconds=exposure,
        ),
    )
    world.register_camera(camera)
    camera.configure_measurement(exposure_seconds=0.013, roi_xywh=(4, 5, 8, 6))
    rendered: list[tuple[float, np.ndarray]] = []
    render = world.render_frame

    def record_render(
        ordinal: int,
        *,
        exposure_seconds: float,
        probe_seconds: float | None = None,
        occupancy: object | None = None,
    ) -> np.ndarray:
        rendered.append((float(exposure_seconds), np.asarray(occupancy, dtype=bool)))
        return render(
            ordinal,
            exposure_seconds=exposure_seconds,
            probe_seconds=probe_seconds,
            occupancy=occupancy,
        )

    world.render_frame = record_render  # type: ignore[method-assign]
    camera.arm(3, source_group_sizes=(3,), buffer_frame_count=3, timeout=1.0)
    sequencer = VirtualPulseStreamer(world=world)
    sequencer.open()
    pulse = resolve_pulse(
        IMAGING_PULSE_RESOURCE.value,
        path=IMAGING_PULSE_RESOURCE.path,
        board=sequencer.describe(),
        api_values={
            "reference_probe_duration_before": 0.1,
            "readout_probe_duration": 0.002,
            "reference_probe_duration_after": 0.2,
        },
    )
    sequencer.load(pulse.program, source=pulse.sequence)
    sequencer.fire()
    records = camera.read_frame_records(3, timeout=1.0, exact=True)
    camera.finish_record_capture()
    sequencer.close()

    assert [exposure for exposure, _occupancy in rendered] == [
        pytest.approx(0.013),
        pytest.approx(0.002),
        pytest.approx(0.013),
    ]
    assert all(np.array_equal(occupancy, rendered[0][1]) for _exposure, occupancy in rendered)
    assert [record.image.shape for record in records] == [(6, 8)] * 3


def test_virtual_camera_clips_into_its_declared_dtype() -> None:
    """The producer clips in place into the configured unsigned pixel format.

    The one per-frame copy is the CameraFrameRecord bytes snapshot, made on
    the producer thread; the reused clip buffer therefore must never leak
    into a published record.
    """

    source = np.array([[300, -5], [7, 260]], dtype=np.int64)
    camera = VirtualCamera(
        VirtualCameraConfig(frame_shape_yx=(2, 2), frame_dtype="|u1"),
        frame_source=lambda _ordinal, _exposure: source,
    )
    assert camera.frame_dtype == np.dtype("|u1")
    camera.arm(2, source_group_sizes=(2,), buffer_frame_count=2, timeout=1.0)
    camera.trigger(2)
    first, second = camera.read_frame_records(2, timeout=1.0, exact=True)
    camera.finish_record_capture()
    for record in (first, second):
        assert record.image.dtype.str == "|u1"
        np.testing.assert_array_equal(
            record.image, np.array([[255, 0], [7, 255]], dtype=np.uint8)
        )
        with pytest.raises(ValueError):
            record.image[0, 0] = 1
    assert (
        first.image.__array_interface__["data"][0]
        != second.image.__array_interface__["data"][0]
    )

    with pytest.raises(ValueError, match="unsigned integer"):
        VirtualCamera(
            VirtualCameraConfig(frame_shape_yx=(2, 2), frame_dtype="<f8"),
            frame_source=lambda _ordinal, _exposure: source,
        )


def test_camera_frame_record_copies_reusable_storage() -> None:
    source = np.zeros((2, 2), dtype=np.uint16)
    record = CameraFrameRecord(source, 0, host_received_at_ns=1)
    source.fill(7)
    assert np.all(record.image == 0)
    with pytest.raises(ValueError):
        record.image[0, 0] = 4


def test_broker_helper_is_the_single_identity_binding_ritual() -> None:
    broker = DeviceBroker()
    binding, proof = bind_verified_device(
        broker,
        key=ResourceKey.parse("device/test"),
        identity_probe=lambda: PhysicalDeviceIdentity("test", DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT),
        capability_probe=lambda: {"test": "capability"},
    )
    assert proof.snapshot["test"] == "capability"
    assert binding.capabilities["test"] == "capability"
