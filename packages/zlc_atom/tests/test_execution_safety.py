from __future__ import annotations

import threading
import time

import pytest

from zlc_atom.execution import (
    DeviceBroker,
    DeviceIdentityEvidenceKind,
    IdentityProof,
    PhysicalDeviceIdentity,
    ResourceClaim,
    ResourceKey,
    RunPlan,
    SafetyInterrupt,
    SafetyOperation,
    bind_verified_device,
    run_plan,
)


def _binding(broker: DeviceBroker, calls: list[str]):
    return bind_verified_device(
        broker,
        key=ResourceKey.parse("device/safety"),
        identity_probe=lambda: PhysicalDeviceIdentity(
            "safety-device",
            DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        ),
        execute_command=lambda command: calls.append(f"execute:{command}"),
        capability_probe=lambda: {"test": "capability"},
        interrupt_operations={SafetyOperation.STOP: lambda: calls.append("stop")},
    )[0]


def test_identity_proof_is_opaque_and_single_use() -> None:
    broker = DeviceBroker()
    with pytest.raises(TypeError):
        IdentityProof(PhysicalDeviceIdentity("forged", DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT), "nonce")  # type: ignore[call-arg]
    proof = broker.verify_identity(
        lambda: PhysicalDeviceIdentity(
            "once",
            DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        )
    )
    binding = broker.bind(
        key=ResourceKey.parse("device/once"),
        identity=proof,
        execute_command=lambda _command: None,
        capability_probe=lambda: {},
    )
    assert binding.key == ResourceKey.parse("device/once")
    with pytest.raises(RuntimeError, match="consumed"):
        broker.bind(
            key=ResourceKey.parse("device/once-again"),
            identity=proof,
            execute_command=lambda _command: None,
            capability_probe=lambda: {},
        )


def test_run_lease_seals_execution_before_interrupt_and_revokes_after_cancel() -> None:
    broker = DeviceBroker()
    calls: list[str] = []
    binding = _binding(broker, calls)
    started = threading.Event()
    held_device: list[object] = []

    def execute(context, _prepared):
        device = context.device(binding.key)
        held_device.append(device)
        device.execute("before-cancel")
        started.set()
        while not context.cancel_requested:
            time.sleep(0.001)
        with pytest.raises(RuntimeError, match="no active execution"):
            device.execute("after-cancel")
        return "never-published"

    plan = RunPlan(
        name="cancel-seal",
        resource_claims=(ResourceClaim(binding.key),),
        bound_devices=(binding,),
        preflight=lambda _context: None,
        execute=execute,
        cleanup=lambda _context, _prepared, _error: None,
        finalize=lambda _prepared, result: result,
        interrupt_operations=(SafetyInterrupt(binding.key, SafetyOperation.STOP),),
    )
    handle = run_plan(plan, broker=broker)
    assert started.wait(1.0)
    assert handle.cancel("test cancellation") is True
    with pytest.raises(RuntimeError, match="cancelled"):
        handle.result(timeout=1.0)
    assert calls == ["execute:before-cancel", "stop"]
    with pytest.raises(RuntimeError, match="raw BoundDevice"):
        broker.execute(binding, "held-raw-binding")
    with pytest.raises(RuntimeError, match="no active execution"):
        held_device[0].execute("post-revoke")
    broker.shutdown()


def test_broker_shutdown_rejects_active_run_and_accepts_after_revoke() -> None:
    broker = DeviceBroker()
    calls: list[str] = []
    binding = _binding(broker, calls)
    started = threading.Event()
    release = threading.Event()

    def execute(context, _prepared):
        started.set()
        release.wait(1.0)
        context.device(binding.key).execute("one-shot")
        return None

    plan = RunPlan(
        name="shutdown-active",
        resource_claims=(ResourceClaim(binding.key),),
        bound_devices=(binding,),
        preflight=lambda _context: None,
        execute=execute,
        cleanup=lambda _context, _prepared, _error: None,
        finalize=lambda _prepared, result: result,
    )
    handle = run_plan(plan, broker=broker)
    assert started.wait(1.0)
    with pytest.raises(RuntimeError, match="active"):
        broker.shutdown()
    release.set()
    assert handle.result(timeout=1.0) is None
    broker.shutdown()
