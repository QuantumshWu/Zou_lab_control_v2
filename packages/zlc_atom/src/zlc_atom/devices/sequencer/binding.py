"""Installation binding shared by physical and simulated sequencers."""

from __future__ import annotations

from zlc_atom.devices.sequencer.device import SequencerDevice
from zlc_atom.execution import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
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
        capability_probe=lambda: {"sequencer.streamer": device},
    )
    return InstalledLeaf(
        key,
        type_id,
        device,
        dict(proof.snapshot),
        binding=binding,
        closer=device.close,
    )


def open_sequencer_control(session, device_key: str, window_ratio=None, owner=None):
    """Open PulseGUI for one named sequencer in an existing experiment."""

    from zlc_workbench.apps.pulse_editor import create_bound_window

    return create_bound_window(
        workspace=session.workspace,
        sequence=None,
        sequencer=session.installation.device(str(device_key)),
        device_use=session.device_use,
        path="",
        window_ratio=window_ratio,
        owner=owner,
    )


__all__ = ["bind_sequencer", "open_sequencer_control"]
