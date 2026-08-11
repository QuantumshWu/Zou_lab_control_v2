from __future__ import annotations

import pytest

from zlc_atom.execution import (
    DeviceBroker,
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
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
