from __future__ import annotations

import numpy as np
import pytest
import time

from zlc_atom.devices.camera import CameraFrameRecord, VirtualCamera
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
from zlc_atom.devices.camera.world import SimulationWorld


def test_virtual_camera_preserves_trigger_to_frame_causality_and_drops_monitor_history() -> None:
    world = SimulationWorld()
    camera = VirtualCamera(frame_source=lambda ordinal: world.render_frame(ordinal))
    camera.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=1.0)
    camera.trigger(3)
    deadline = time.monotonic() + 1.0
    while camera.produced_count < 3 and time.monotonic() < deadline:
        time.sleep(0.001)
    records = camera.read_frame_records(1, timeout=0.1, exact=True)
    assert len(records) == 1
    assert records[0].source_ordinal == 2
    assert camera.capture_state() == (True, 2)
    terminal = camera.finish_record_capture()
    assert terminal.source_stopped and terminal.joined is True


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
