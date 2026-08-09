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
from zlc_atom.devices.simulation import SimulationWorld, VirtualCamera, VirtualCameraConfig
from zlc_atom.execution import (
    DeviceIdentityEvidenceKind,
    DeviceBroker,
    PhysicalDeviceIdentity,
    ResourceClaim,
    ResourceKey,
    SafetyOperation,
    RunPlan,
    bind_verified_device,
    run_plan,
)


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
        occupancy: object | None = None,
    ) -> np.ndarray:
        rendered.append((float(exposure_seconds), np.asarray(occupancy, dtype=bool)))
        return render(
            ordinal,
            exposure_seconds=exposure_seconds,
            occupancy=occupancy,
        )

    world.render_frame = record_render  # type: ignore[method-assign]
    camera.arm(3, source_group_sizes=(3,), buffer_frame_count=3, timeout=1.0)
    world.fire(3, frame_exposures=(0.1, 0.002, 0.2))
    records = camera.read_frame_records(3, timeout=1.0, exact=True)
    camera.finish_record_capture()

    assert [exposure for exposure, _occupancy in rendered] == [
        pytest.approx(0.013),
        pytest.approx(0.002),
        pytest.approx(0.013),
    ]
    assert all(np.array_equal(occupancy, rendered[0][1]) for _exposure, occupancy in rendered)
    assert [record.image.shape for record in records] == [(6, 8)] * 3


def test_camera_frame_record_copies_reusable_storage() -> None:
    source = np.zeros((2, 2), dtype=np.uint16)
    record = CameraFrameRecord(source, 0, host_received_at_ns=1)
    source.fill(7)
    assert np.all(record.image == 0)
    with pytest.raises(ValueError):
        record.image[0, 0] = 4


def test_broker_helper_is_the_single_identity_binding_ritual() -> None:
    broker = DeviceBroker()
    calls: list[str] = []
    binding, proof = bind_verified_device(
        broker,
        key=ResourceKey.parse("device/test"),
        identity_probe=lambda: PhysicalDeviceIdentity("test", DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT),
        execute_command=lambda command: calls.append(str(command)),
        capability_probe=lambda: {"test": "capability"},
        interrupt_operations={SafetyOperation.STOP: lambda: calls.append("stop")},
    )
    assert broker.require_capability(proof, "test") == "capability"
    plan = RunPlan(
        name="broker-helper",
        resource_claims=(ResourceClaim(ResourceKey.parse("device/test")),),
        bound_devices=(binding,),
        preflight=lambda _context: None,
        execute=lambda context, _prepared: context.device(ResourceKey.parse("device/test")).execute("command"),
        cleanup=lambda _context, _prepared, _error: None,
        finalize=lambda _prepared, executed: executed,
    )
    assert run_plan(plan, broker=broker).result(timeout=1) is None
    assert calls == ["command"]
    with pytest.raises(RuntimeError, match="raw BoundDevice"):
        broker.execute(binding, "outside-run")
