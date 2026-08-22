"""Installation-time device identity and capability binding."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import uuid
import weakref
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
    _broker_ref: weakref.ReferenceType["DeviceBroker"]

    @property
    def physical_identity(self) -> PhysicalDeviceIdentity:
        """The real resource this logical binding owns."""

        return self.stamp.physical_identity

    @property
    def broker(self) -> "DeviceBroker":
        """The identity owner required to retire this exact binding."""

        owner = self._broker_ref()
        if owner is None:
            raise RuntimeError("device binding broker no longer exists")
        return owner


CapabilityProbe = Callable[[], Mapping[str, object]]
IdentityProbe = Callable[[], PhysicalDeviceIdentity]


class DeviceBroker:
    """Reject duplicate physical devices and verify declared capabilities."""

    CAPABILITY_TYPES = CAPABILITY_TYPES

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._verified_identities: dict[object, IdentityProof] = {}
        # Keep the exact minted object, not only its opaque token.  Otherwise a
        # caller that copied a token into another BoundDevice could release the
        # real binding while presenting different identity evidence.
        self._active_bindings: dict[object, BoundDevice] = {}
        # A legitimate second unbind is an idempotent close retry.  Remember
        # bindings minted here so that retry can return False while a binding
        # from another broker (or a forged one) is still rejected.
        # Weak after unbind: remembering that a retry is legitimate must not
        # retain the closed adapter through BoundDevice.capabilities forever.
        self._known_bindings: dict[object, weakref.ReferenceType[BoundDevice]] = {}
        self._physical_ids: dict[str, object] = {}

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
        binding = BoundDevice(key, stamp, snapshot, token, weakref.ref(self))
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
            self._physical_ids[stable_id] = token
            self._active_bindings[token] = binding

            broker_ref = weakref.ref(self)

            def forget(
                reference: weakref.ReferenceType[BoundDevice],
                *,
                binding_token: object = token,
                owner_ref: weakref.ReferenceType[DeviceBroker] = broker_ref,
            ) -> None:
                # The weakref owns this callback, so capturing ``self`` here
                # would create broker -> weakref -> callback -> broker and turn
                # the tombstone cleanup into the leak it is meant to prevent.
                owner = owner_ref()
                if owner is None:
                    return
                with owner._lock:
                    if (
                        binding_token not in owner._active_bindings
                        and owner._known_bindings.get(binding_token) is reference
                    ):
                        owner._known_bindings.pop(binding_token, None)

            self._known_bindings[token] = weakref.ref(binding, forget)
        return binding

    def unbind(self, binding: BoundDevice) -> bool:
        """Release one exact physical binding after its device has closed.

        Returns ``True`` for the transition and ``False`` for an idempotent
        retry.  Unknown, copied, or foreign bindings are never treated as an
        already-completed close: accepting one would let it release somebody
        else's physical device identity.
        """

        if not isinstance(binding, BoundDevice):
            raise TypeError("binding must be BoundDevice")
        token = binding._broker_token
        with self._lock:
            known = self._known_bindings.get(token)
            if known is None or known() is not binding:
                raise RuntimeError("device binding is unknown")
            if self._active_bindings.get(token) is not binding:
                return False
            stable_id = binding.physical_identity.stable_device_identity
            if self._physical_ids.get(stable_id) is not token:
                raise RuntimeError("device binding physical identity is inconsistent")
            self._active_bindings.pop(token)
            self._physical_ids.pop(stable_id)
            return True

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
        with self._lock:
            if self._active_bindings.get(binding._broker_token) is not binding:
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
