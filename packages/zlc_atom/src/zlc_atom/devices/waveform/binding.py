"""Installation binding shared by real and simulated WaveformSource implementations."""

from __future__ import annotations

from zlc_atom.devices.waveform.contract import WaveformSource
from zlc_atom.execution import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
    bind_verified_device,
)
from zlc_atom.install.descriptors import InstalledLeaf


def bind_waveform_source(
    context,
    key: str,
    source: WaveformSource,
    identity: str,
    type_id: str,
) -> InstalledLeaf:
    if not isinstance(source, WaveformSource):
        raise TypeError("source must implement the canonical WaveformSource contract")
    try:
        binding, proof = bind_verified_device(
            context.broker,
            key=ResourceKey.parse(f"device/{key}"),
            identity_probe=lambda: PhysicalDeviceIdentity(
                identity,
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            ),
            capability_probe=lambda: {
                "waveform.source": source,
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


__all__ = ["bind_waveform_source"]
