"""Installation binding shared by real and simulated CameraAdapter implementations."""

from __future__ import annotations

from zlc_atom.devices.camera.contract import CameraAdapter
from zlc_atom.execution import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
    SafetyOperation,
    bind_verified_device,
)
from zlc_atom.install.descriptors import InstalledLeaf


def _reject_untyped_command(_command: object) -> None:
    raise TypeError("camera operations must use the CameraAdapter capability")


def bind_camera(
    context,
    key: str,
    camera: CameraAdapter,
    identity: str,
    type_id: str,
) -> InstalledLeaf:
    if not isinstance(camera, CameraAdapter):
        raise TypeError("camera must implement the canonical CameraAdapter contract")
    binding, proof = bind_verified_device(
        context.broker,
        key=ResourceKey.parse(f"device/{key}"),
        identity_probe=lambda: PhysicalDeviceIdentity(
            identity,
            DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        ),
        execute_command=_reject_untyped_command,
        capability_probe=lambda: {
            "camera.adapter": camera,
            "camera.working_point": camera.capture_working_point(),
        },
        close_session=lambda _command: camera.finish_record_capture()
        if camera.capture_state()
        else None,
        interrupt_operations={
            SafetyOperation.DISARM: lambda: camera.finish_record_capture()
        },
    )
    return InstalledLeaf(
        key,
        type_id,
        camera,
        dict(proof.snapshot),
        binding=binding,
        closer=getattr(camera, "close", None),
    )


__all__ = ["bind_camera"]
