"""Small resource and physical-identity contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
import time


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


@dataclass(frozen=True)
class ResourceClaim:
    key: ResourceKey

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResourceKey):
            raise TypeError("ResourceClaim.key must be ResourceKey")


@dataclass(frozen=True)
class ResourceBusy:
    requested: ResourceClaim
    conflicting_run_id: str
    conflicting_claim: ResourceClaim


class ResourceLease:
    def __init__(self, arbiter: "ResourceArbiter", run_id: str, capability: object) -> None:
        self._arbiter = arbiter
        self.run_id = run_id
        self._capability = capability
        self._released = False

    def release(self) -> bool:
        if self._released:
            return False
        self._arbiter._release(self.run_id, self._capability)
        self._released = True
        return True

    @property
    def released(self) -> bool:
        return self._released


class ResourceArbiter:
    """Atomic in-process ownership table used by the run engine."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._owners: dict[str, tuple[object, tuple[ResourceClaim, ...]]] = {}
        self._closed = False

    def acquire_all(self, run_id: str, claims: tuple[ResourceClaim, ...]) -> ResourceLease | tuple[ResourceBusy, ...]:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        normalized = tuple(sorted(claims, key=lambda claim: claim.key))
        if len({claim.key for claim in normalized}) != len(normalized):
            raise ValueError("one run cannot claim the same resource twice")
        with self._condition:
            if self._closed:
                raise RuntimeError("ResourceArbiter is closed")
            blockers = tuple(
                ResourceBusy(requested, owner, held)
                for requested in normalized
                for owner, (_capability, held_claims) in sorted(self._owners.items())
                for held in held_claims
                if held.key == requested.key
            )
            if blockers:
                return blockers
            capability = object()
            self._owners[run_id] = (capability, normalized)
            return ResourceLease(self, run_id, capability)

    def _release(self, run_id: str, capability: object) -> None:
        with self._condition:
            owner = self._owners.get(run_id)
            if owner is None or owner[0] is not capability:
                raise RuntimeError("resource lease is no longer active")
            del self._owners[run_id]
            self._condition.notify_all()

    def wait_until_released(self, run_ids: tuple[str, ...], *, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while any(run_id in self._owners for run_id in run_ids):
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(remaining)
            return True

    def shutdown(self) -> None:
        with self._condition:
            if self._owners:
                raise RuntimeError("cannot shut down ResourceArbiter with active leases")
            self._closed = True


__all__ = [
    "DeviceBindingStamp",
    "DeviceIdentityEvidenceKind",
    "PhysicalDeviceIdentity",
    "ResourceArbiter",
    "ResourceBusy",
    "ResourceClaim",
    "ResourceKey",
    "ResourceLease",
]
