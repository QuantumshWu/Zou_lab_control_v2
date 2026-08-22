from __future__ import annotations

import gc
import weakref

import pytest

from zlc_atom.execution import (
    DeviceBroker,
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
    bind_verified_device,
)
from zlc_atom.execution.ports import IdentityProof


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
        capability_probe=lambda: {},
    )
    assert binding.key == ResourceKey.parse("device/once")
    with pytest.raises(RuntimeError, match="consumed"):
        broker.bind(
            key=ResourceKey.parse("device/once-again"),
            identity=proof,
            capability_probe=lambda: {},
        )


def test_unbind_releases_only_the_exact_broker_minted_physical_binding() -> None:
    broker = DeviceBroker()
    identity = PhysicalDeviceIdentity(
        "camera:serial-1",
        DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
    )

    def bind(key: str):
        return bind_verified_device(
            broker,
            key=ResourceKey.parse(f"device/{key}"),
            identity_probe=lambda: identity,
            capability_probe=dict,
        )[0]

    first = bind("camera")
    assert first.physical_identity is identity
    with pytest.raises(RuntimeError, match="already bound"):
        bind("duplicate")

    assert broker.unbind(first) is True
    assert broker.unbind(first) is False, "close retry is idempotent"
    with pytest.raises(RuntimeError, match="unknown"):
        broker.verify_capability(first)

    second = bind("camera-again")
    foreign = DeviceBroker()
    with pytest.raises(RuntimeError, match="unknown"):
        foreign.unbind(second)
    assert broker.verify_capability(second).binding is second
    assert broker.unbind(second) is True


def test_unbound_binding_tombstone_is_collected_without_retaining_broker() -> None:
    broker = DeviceBroker()
    binding = bind_verified_device(
        broker,
        key=ResourceKey.parse("device/transient"),
        identity_probe=lambda: PhysicalDeviceIdentity(
            "transient",
            DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        ),
        capability_probe=dict,
    )[0]
    token = binding._broker_token
    observed = weakref.ref(binding)
    assert broker.unbind(binding) is True
    assert token in broker._known_bindings

    del binding
    gc.collect()
    assert observed() is None
    assert token not in broker._known_bindings
