"""Installation binding shared by physical and simulated sequencers."""

from __future__ import annotations

from zlc_atom.devices.sequencer.device import SequencerDevice
from zlc_atom.execution import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
    SafetyOperation,
    bind_verified_device,
)
from zlc_atom.install.descriptors import InstalledLeaf


def bind_sequencer(
    context,
    key: str,
    device: SequencerDevice,
    identity: str,
    type_id: str,
) -> InstalledLeaf:
    if not isinstance(device, SequencerDevice):
        raise TypeError("sequencer must use the canonical SequencerDevice")
    binding, proof = bind_verified_device(
        context.broker,
        key=ResourceKey.parse(f"device/{key}"),
        identity_probe=lambda: PhysicalDeviceIdentity(
            identity,
            DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        ),
        execute_command=lambda command: device.fire(forever=bool(command))
        if command is not None
        else device.fire(),
        capability_probe=lambda: {"sequencer.streamer": device},
        close_session=lambda _command: device.close(),
        interrupt_operations={SafetyOperation.STOP: lambda: device.close()},
    )
    return InstalledLeaf(
        key,
        type_id,
        device,
        dict(proof.snapshot),
        binding=binding,
        closer=device.close,
    )


__all__ = ["bind_sequencer"]
