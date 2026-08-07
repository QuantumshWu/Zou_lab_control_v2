"""Device broker and interrupt contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
import uuid
from typing import Any, Callable, Mapping

from .capabilities import CAPABILITY_TYPES
from .resources import DeviceBindingStamp, DeviceIdentityEvidenceKind, PhysicalDeviceIdentity, ResourceKey


class SafetyOperation(str, Enum):
    DISARM = "DISARM"
    STOP = "STOP"
    SAFE = "SAFE"


@dataclass(frozen=True)
class SafetyInterrupt:
    key: ResourceKey
    operation: SafetyOperation


_IDENTITY_PROOF_TOKEN = object()


class IdentityProof:
    """Opaque broker-minted identity evidence used exactly once by ``bind``."""

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
        object.__setattr__(self, "_broker", broker)
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_nonce", nonce)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("IdentityProof is immutable")

    @property
    def identity(self) -> PhysicalDeviceIdentity:
        return self._identity


@dataclass(frozen=True)
class CapabilityProof:
    binding: "BoundDevice"
    snapshot: Mapping[str, object]
    nonce: str


@dataclass(frozen=True)
class BoundDevice:
    key: ResourceKey
    stamp: DeviceBindingStamp
    capabilities: Mapping[str, object]
    _broker_token: object

    def require_capability(self, token: str, expected_type: type | None = None) -> object:
        try:
            value = self.capabilities[token]
        except KeyError as exc:
            raise KeyError(f"device {self.key} lacks capability {token!r}") from exc
        if expected_type is not None and not isinstance(value, expected_type):
            raise TypeError(f"capability {token!r} is not {expected_type.__name__}")
        return value


CapabilityProbe = Callable[[], Mapping[str, object]]
CommandCallback = Callable[[object], object]
CloseCallback = Callable[[object], object]
IdentityProbe = Callable[[], PhysicalDeviceIdentity]


class DeviceBroker:
    """Own physical binding identity and inject all device callbacks."""

    CAPABILITY_TYPES = CAPABILITY_TYPES

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._closed = False
        self._endpoints: dict[object, tuple[CommandCallback, CapabilityProbe, CloseCallback | None, Mapping[SafetyOperation, Callable[[], object]]]] = {}
        self._verified_identities: dict[object, IdentityProof] = {}
        self._active: dict[object, str] = {}
        self._physical_ids: set[str] = set()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DeviceBroker is closed")

    def verify_identity(self, probe: IdentityProbe) -> IdentityProof:
        self._ensure_open()
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
        with self._condition:
            self._ensure_open()
            self._verified_identities[nonce] = proof
        return proof

    def bind(
        self,
        *,
        key: ResourceKey,
        identity: IdentityProof,
        execute_command: CommandCallback,
        capability_probe: CapabilityProbe,
        close_session: CloseCallback | None = None,
        interrupt_operations: Mapping[SafetyOperation, Callable[[], object]] | None = None,
    ) -> BoundDevice:
        self._ensure_open()
        if not isinstance(key, ResourceKey) or not isinstance(identity, IdentityProof) or identity._broker is not self:
            raise TypeError("bind requires a ResourceKey and IdentityProof")
        if not callable(execute_command) or not callable(capability_probe):
            raise TypeError("device callbacks must be callable")
        token = object()
        interrupt_map = dict(interrupt_operations or {})
        if any(not isinstance(operation, SafetyOperation) or not callable(callback) for operation, callback in interrupt_map.items()):
            raise TypeError("interrupt_operations must map SafetyOperation to callables")
        snapshot = dict(capability_probe())
        stamp = DeviceBindingStamp(identity.identity, uuid.uuid4().hex)
        binding = BoundDevice(key, stamp, snapshot, token)
        with self._condition:
            self._ensure_open()
            pending = self._verified_identities.get(identity._nonce)
            if pending is not identity:
                raise RuntimeError("verified device identity was already consumed or belongs to another broker")
            if stamp.physical_identity.stable_device_identity in self._physical_ids:
                raise RuntimeError(f"physical device {stamp.physical_identity.stable_device_identity!r} is already bound")
            self._endpoints[token] = (execute_command, capability_probe, close_session, interrupt_map)
            self._physical_ids.add(stamp.physical_identity.stable_device_identity)
            self._verified_identities.pop(identity._nonce, None)
        return binding

    def verify_capability(self, binding: BoundDevice) -> CapabilityProof:
        endpoint = self._endpoint(binding)
        with self._condition:
            if binding._broker_token in self._active:
                raise RuntimeError("cannot probe capability while device is active in a Run")
        snapshot = dict(endpoint[1]())
        for token, expected in self.CAPABILITY_TYPES.items():
            if token in snapshot and not isinstance(snapshot[token], expected):
                raise TypeError(f"capability {token!r} must have type {expected.__name__}")
        return CapabilityProof(binding, snapshot, uuid.uuid4().hex)

    def require_capability(self, proof: CapabilityProof, token: str, expected_type: type | None = None) -> object:
        if not isinstance(proof, CapabilityProof) or proof.binding._broker_token not in self._endpoints:
            raise RuntimeError("capability proof is unknown or revoked")
        try:
            value = proof.snapshot[token]
        except KeyError as exc:
            raise KeyError(f"device {proof.binding.key} lacks capability {token!r}") from exc
        if expected_type is not None and not isinstance(value, expected_type):
            raise TypeError(f"capability {token!r} is not {expected_type.__name__}")
        return value

    def execute(self, binding: BoundDevice, command: object) -> object:
        del binding, command
        raise RuntimeError("raw BoundDevice execution is forbidden; use a RunContext device lease")

    def close(self, binding: BoundDevice, command: object = None) -> object:
        del binding, command
        raise RuntimeError("raw BoundDevice cleanup is forbidden; use a RunContext device lease")

    def interrupt(self, binding: BoundDevice, operation: SafetyOperation) -> object:
        del binding, operation
        raise RuntimeError("raw BoundDevice interrupts are forbidden; use a RunContext device lease")

    def open_run(self, run_id: str, bindings: tuple[BoundDevice, ...]) -> "_DeviceRunLease":
        """Atomically lease installation bindings to one Run."""

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be non-empty text")
        normalized = tuple(bindings)
        if len({binding._broker_token for binding in normalized}) != len(normalized):
            raise ValueError("a Run cannot lease one device binding twice")
        with self._condition:
            self._ensure_open()
            for binding in normalized:
                self._endpoint(binding)
                owner = self._active.get(binding._broker_token)
                if owner is not None:
                    raise RuntimeError(f"device binding {binding.key} is already active in Run {owner}")
            for binding in normalized:
                self._active[binding._broker_token] = run_id
        return _DeviceRunLease(self, run_id, normalized)

    def _endpoint_for_run(self, run_id: str, binding: BoundDevice) -> tuple[CommandCallback, CapabilityProbe, CloseCallback | None, Mapping[SafetyOperation, Callable[[], object]]]:
        endpoint = self._endpoint(binding)
        with self._condition:
            if self._active.get(binding._broker_token) != run_id:
                raise RuntimeError(f"device binding {binding.key} is not active for Run {run_id}")
        return endpoint

    def _release_run(self, run_id: str, bindings: tuple[BoundDevice, ...]) -> None:
        with self._condition:
            for binding in bindings:
                if self._active.get(binding._broker_token) == run_id:
                    self._active.pop(binding._broker_token, None)
            self._condition.notify_all()

    def _endpoint(self, binding: BoundDevice) -> tuple[CommandCallback, CapabilityProbe, CloseCallback | None, Mapping[SafetyOperation, Callable[[], object]]]:
        if not isinstance(binding, BoundDevice):
            raise TypeError("binding must be BoundDevice")
        try:
            return self._endpoints[binding._broker_token]
        except KeyError as exc:
            raise RuntimeError("device binding is unknown or revoked") from exc

    def shutdown(self) -> None:
        with self._condition:
            if self._active:
                raise RuntimeError("cannot shut down broker with active devices")
            self._closed = True


class _DeviceRunLease:
    """Run-scoped device authority; revocation is idempotent and fail-closed."""

    def __init__(self, broker: DeviceBroker, run_id: str, bindings: tuple[BoundDevice, ...]) -> None:
        self._broker = broker
        self._run_id = run_id
        self._bindings = bindings
        self._revoked = False
        self._lock = threading.Lock()

    def _endpoint(self, binding: BoundDevice):
        with self._lock:
            if self._revoked:
                raise RuntimeError("device broker lease is revoked")
            if binding not in self._bindings:
                raise RuntimeError("device binding is not owned by this Run")
        return self._broker._endpoint_for_run(self._run_id, binding)

    def execute(self, binding: BoundDevice, command: object) -> object:
        return self._endpoint(binding)[0](command)

    def close(self, binding: BoundDevice, command: object = None) -> object:
        callback = self._endpoint(binding)[2]
        if callback is None:
            return None
        return callback(command)

    def interrupt(self, binding: BoundDevice, operation: SafetyOperation) -> object:
        try:
            callback = self._endpoint(binding)[3][operation]
        except KeyError as exc:
            raise RuntimeError(f"device {binding.key} has no {operation.value} interrupt") from exc
        return callback()

    def revoke(self) -> None:
        with self._lock:
            if self._revoked:
                return
            self._revoked = True
        self._broker._release_run(self._run_id, self._bindings)


def bind_verified_device(
    broker: DeviceBroker,
    *,
    key: ResourceKey,
    identity_probe: IdentityProbe,
    execute_command: CommandCallback,
    capability_probe: CapabilityProbe,
    close_session: CloseCallback | None = None,
    interrupt_operations: Mapping[SafetyOperation, Callable[[], object]] | None = None,
) -> tuple[BoundDevice, CapabilityProof]:
    """The one shared verify-identity → bind → verify-capability ritual."""

    proof = broker.verify_identity(identity_probe)
    binding = broker.bind(
        key=key,
        identity=proof,
        execute_command=execute_command,
        capability_probe=capability_probe,
        close_session=close_session,
        interrupt_operations=interrupt_operations,
    )
    return binding, broker.verify_capability(binding)


def bind_trivial_device(
    broker: DeviceBroker,
    *,
    key: ResourceKey,
    identity: str,
    execute_command: CommandCallback,
    capabilities: Mapping[str, object] | None = None,
) -> tuple[BoundDevice, CapabilityProof]:
    """Ten-line-friendly path for a device with no safety ceremony."""

    snapshot = dict(capabilities or {})
    return bind_verified_device(
        broker,
        key=key,
        identity_probe=lambda: PhysicalDeviceIdentity(identity, DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT),
        execute_command=execute_command,
        capability_probe=lambda: snapshot,
    )


__all__ = [
    "BoundDevice",
    "CapabilityProof",
    "DeviceBroker",
    "IdentityProof",
    "SafetyInterrupt",
    "SafetyOperation",
    "bind_trivial_device",
    "bind_verified_device",
]
