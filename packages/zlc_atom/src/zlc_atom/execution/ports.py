"""Installation-time device identity and capability binding."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import uuid
from typing import Callable, Mapping

from .capabilities import CAPABILITY_TYPES
from .resources import (
    DeviceBindingStamp,
    PhysicalDeviceIdentity,
    ResourceKey,
)


_IDENTITY_PROOF_TOKEN = object()


class IdentityProof:
    """Broker-minted physical identity used once while installing a device."""

    __slots__ = ("_broker", "_identity", "_nonce")

    def __init__(
        self,
        token: object,
        *,
        broker: "DeviceBroker",
        identity: PhysicalDeviceIdentity,
        nonce: object,
    ) -> None:
        if token is not _IDENTITY_PROOF_TOKEN:
            raise PermissionError("IdentityProof can only be minted by DeviceBroker")
        self._broker = broker
        self._identity = identity
        self._nonce = nonce

    @property
    def identity(self) -> PhysicalDeviceIdentity:
        return self._identity


@dataclass(frozen=True)
class CapabilityProof:
    binding: "BoundDevice"
    snapshot: Mapping[str, object]


@dataclass(frozen=True)
class BoundDevice:
    key: ResourceKey
    stamp: DeviceBindingStamp
    capabilities: Mapping[str, object]
    _broker_token: object


CapabilityProbe = Callable[[], Mapping[str, object]]
IdentityProbe = Callable[[], PhysicalDeviceIdentity]


class DeviceBroker:
    """Reject duplicate physical devices and verify declared capabilities."""

    CAPABILITY_TYPES = CAPABILITY_TYPES

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._verified_identities: dict[object, IdentityProof] = {}
        self._binding_tokens: set[object] = set()
        self._physical_ids: set[str] = set()

    def verify_identity(self, probe: IdentityProbe) -> IdentityProof:
        identity = probe()
        if not isinstance(identity, PhysicalDeviceIdentity):
            raise TypeError("identity probe must return PhysicalDeviceIdentity")
        nonce = object()
        proof = IdentityProof(
            _IDENTITY_PROOF_TOKEN,
            broker=self,
            identity=identity,
            nonce=nonce,
        )
        with self._lock:
            self._verified_identities[nonce] = proof
        return proof

    def bind(
        self,
        *,
        key: ResourceKey,
        identity: IdentityProof,
        capability_probe: CapabilityProbe,
    ) -> BoundDevice:
        if (
            not isinstance(key, ResourceKey)
            or not isinstance(identity, IdentityProof)
            or identity._broker is not self
        ):
            raise TypeError("bind requires a ResourceKey and IdentityProof")
        if not callable(capability_probe):
            raise TypeError("capability_probe must be callable")
        snapshot = dict(capability_probe())
        token = object()
        stamp = DeviceBindingStamp(identity.identity, uuid.uuid4().hex)
        binding = BoundDevice(key, stamp, snapshot, token)
        with self._lock:
            if self._verified_identities.get(identity._nonce) is not identity:
                raise RuntimeError(
                    "verified device identity was already consumed or belongs "
                    "to another broker"
                )
            stable_id = stamp.physical_identity.stable_device_identity
            if stable_id in self._physical_ids:
                raise RuntimeError(f"physical device {stable_id!r} is already bound")
            self._verified_identities.pop(identity._nonce)
            self._physical_ids.add(stable_id)
            self._binding_tokens.add(token)
        return binding

    def verify_capability(self, binding: BoundDevice) -> CapabilityProof:
        self._require_binding(binding)
        snapshot = dict(binding.capabilities)
        for token, expected in self.CAPABILITY_TYPES.items():
            if token in snapshot and not isinstance(snapshot[token], expected):
                raise TypeError(
                    f"capability {token!r} must have type {expected.__name__}"
                )
        return CapabilityProof(binding, snapshot)

    def _require_binding(self, binding: BoundDevice) -> None:
        if not isinstance(binding, BoundDevice):
            raise TypeError("binding must be BoundDevice")
        if binding._broker_token not in self._binding_tokens:
            raise RuntimeError("device binding is unknown")


def bind_verified_device(
    broker: DeviceBroker,
    *,
    key: ResourceKey,
    identity_probe: IdentityProbe,
    capability_probe: CapabilityProbe,
) -> tuple[BoundDevice, CapabilityProof]:
    """Bind one physical identity and verify its declared capabilities."""

    identity = broker.verify_identity(identity_probe)
    binding = broker.bind(
        key=key,
        identity=identity,
        capability_probe=capability_probe,
    )
    return binding, broker.verify_capability(binding)


__all__ = [
    "BoundDevice",
    "CapabilityProof",
    "DeviceBroker",
    "IdentityProof",
    "bind_verified_device",
]
