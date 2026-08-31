"""Installation binding shared by real and simulated RfSource implementations."""

from __future__ import annotations

from zlc_atom.devices.rf.contract import RfSource
from zlc_atom.execution import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
    bind_verified_device,
)
from zlc_atom.install.descriptors import InstalledLeaf


def bind_rf_source(
    context,
    key: str,
    source: RfSource,
    identity: str,
    type_id: str,
) -> InstalledLeaf:
    if not isinstance(source, RfSource):
        raise TypeError("rf source must implement the canonical RfSource contract")
    try:
        binding, proof = bind_verified_device(
            context.broker,
            key=ResourceKey.parse(f"device/{key}"),
            identity_probe=lambda: PhysicalDeviceIdentity(
                identity,
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            ),
            capability_probe=lambda: {
                "rf.source": source,
            },
        )
    except BaseException:
        close = getattr(source, "close", None)
        if callable(close):
            close()
        raise
    return InstalledLeaf(
        key,
        type_id,
        source,
        dict(proof.snapshot),
        binding=binding,
        closer=getattr(source, "close", None),
    )


__all__ = ["bind_rf_source"]
