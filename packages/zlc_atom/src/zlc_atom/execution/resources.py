"""Small resource and physical-identity contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, order=True)
class ResourceKey:
    segments: tuple[str, ...]

    def __post_init__(self) -> None:
        segments = tuple(str(segment) for segment in self.segments)
        if not segments or any(not segment or "/" in segment for segment in segments):
            raise ValueError("resource key segments must be non-empty and slash-free")
        object.__setattr__(self, "segments", segments)

    @classmethod
    def parse(cls, value: str) -> "ResourceKey":
        if not isinstance(value, str) or not value:
            raise ValueError("resource key must be non-empty text")
        return cls(tuple(value.split("/")))

    def __str__(self) -> str:
        return "/".join(self.segments)


class DeviceIdentityEvidenceKind(str, Enum):
    HARDWARE_IDENTITY_READBACK = "HARDWARE_IDENTITY_READBACK"
    INSTALLATION_ASSERTED_ENDPOINT = "INSTALLATION_ASSERTED_ENDPOINT"


@dataclass(frozen=True, order=True)
class PhysicalDeviceIdentity:
    stable_device_identity: str
    evidence_kind: DeviceIdentityEvidenceKind

    def __post_init__(self) -> None:
        if not self.stable_device_identity:
            raise ValueError("stable_device_identity must be non-empty")
        if not isinstance(self.evidence_kind, DeviceIdentityEvidenceKind):
            raise TypeError("evidence_kind must be DeviceIdentityEvidenceKind")


@dataclass(frozen=True, order=True)
class DeviceBindingStamp:
    physical_identity: PhysicalDeviceIdentity
    binding_instance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.physical_identity, PhysicalDeviceIdentity):
            raise TypeError("physical_identity must be PhysicalDeviceIdentity")
        if not self.binding_instance_id or "/" in self.binding_instance_id:
            raise ValueError("binding_instance_id must be non-empty and slash-free")


__all__ = [
    "DeviceBindingStamp",
    "DeviceIdentityEvidenceKind",
    "PhysicalDeviceIdentity",
    "ResourceKey",
]
