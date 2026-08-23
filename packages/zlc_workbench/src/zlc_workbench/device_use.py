"""One session-scoped truth for who may drive an installed device.

Logic nodes and PulseGUI share exact device objects from one ExperimentSession.
This module coordinates those callers; it never wraps a device or executes a
hardware command.  Read-only access needs no lease.  Exclusive claims conflict
when they name the same installed key or the same object identity.  Key
identity remains stable while a live installation replaces its device object.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import threading
import time


@dataclass(frozen=True)
class DeviceClaim:
    """One resolved device use frozen into a candidate run."""

    argument_name: str
    device_key: str
    device: object = field(compare=False)
    protected_fields: tuple[str, ...] = ()
    exclusive: bool = True

    def __post_init__(self) -> None:
        argument = str(self.argument_name).strip()
        key = str(self.device_key).strip()
        protected = tuple(str(value).strip() for value in self.protected_fields)
        if not argument or not key:
            raise ValueError("device claim requires argument and installed key")
        if any(not value for value in protected):
            raise ValueError("protected device fields must be non-empty text")
        if len(set(protected)) != len(protected):
            raise ValueError("protected device fields must be unique")
        if type(self.exclusive) is not bool:
            raise TypeError("device claim exclusive must be bool")
        object.__setattr__(self, "argument_name", argument)
        object.__setattr__(self, "device_key", key)
        object.__setattr__(self, "protected_fields", protected)


def _conflict(left: Sequence[DeviceClaim], right: Sequence[DeviceClaim]) -> bool:
    return any(
        first.exclusive
        and second.exclusive
        and (first.device is second.device or first.device_key == second.device_key)
        for first in left
        for second in right
    )


def _same_device(left: Sequence[DeviceClaim], right: Sequence[DeviceClaim]) -> bool:
    return any(
        first.device is second.device or first.device_key == second.device_key
        for first in left
        for second in right
    )


def _claims_keys(claims: Sequence[DeviceClaim]) -> frozenset[str]:
    return frozenset(str(claim.device_key) for claim in claims)


class DeviceUseBusy(RuntimeError):
    """A complete candidate was refused because another owner is driving."""

    def __init__(self, blockers: Sequence[str]) -> None:
        labels = tuple(dict.fromkeys(str(value) for value in blockers))
        self.blockers = labels
        super().__init__(f"device is in use by {', '.join(labels)}")


class DeviceLease:
    """An active set of claims; release is idempotent."""

    def __init__(
        self,
        coordinator: "DeviceUseCoordinator",
        owner: object,
        label: str,
        kind: str,
        claims: tuple[DeviceClaim, ...],
        stop: Callable[[str], None] | None,
    ) -> None:
        self._coordinator = coordinator
        self.owner = owner
        self.label = str(label)
        self.kind = str(kind)
        self.claims = claims
        self.stop = stop

    def release(self) -> bool:
        return self._coordinator._release(self)


class DeviceMaintenance:
    """A key barrier which stops Logic and excludes new device users."""

    def __init__(
        self,
        coordinator: "DeviceUseCoordinator",
        owner: object,
        label: str,
        device_keys: frozenset[str],
    ) -> None:
        self._coordinator = coordinator
        self.owner = owner
        self.label = str(label)
        self.device_keys = device_keys

    @property
    def waiting_for(self) -> tuple[str, ...]:
        return self._coordinator._maintenance_waiting_for(self)

    def wait(self, timeout: float | None = None) -> None:
        self._coordinator._wait_maintenance(self, timeout)

    def release(self) -> bool:
        return self._coordinator._release_maintenance(self)


class LogicReservation:
    """A validated Logic candidate holding its whole claim set before preemption."""

    def __init__(
        self,
        coordinator: "DeviceUseCoordinator",
        owner: object,
        label: str,
        claims: tuple[DeviceClaim, ...],
        stop: Callable[[str], None],
        superseded: Callable[[], None],
    ) -> None:
        self._coordinator = coordinator
        self.owner = owner
        self.label = str(label)
        self.claims = claims
        self.stop = stop
        self.superseded = superseded

    @property
    def waiting_for(self) -> tuple[str, ...]:
        return self._coordinator._waiting_for(self)

    def commit(self) -> DeviceLease:
        return self._coordinator._commit(self)

    def abort(self) -> bool:
        return self._coordinator._abort(self)


class DeviceUseCoordinator:
    """Atomic in-process admission shared by one ExperimentSession."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._leases: dict[object, DeviceLease] = {}
        self._reservations: dict[object, LogicReservation] = {}
        self._maintenance: dict[object, DeviceMaintenance] = {}
        self._device_owner_revisions: dict[str, int] = {}

    def logic_owners(self, device_key: str) -> tuple[str, ...]:
        key = str(device_key).strip()
        if not key:
            raise ValueError("logic owners require a device key")
        with self._lock:
            return tuple(
                lease.label
                for lease in self._leases.values()
                if lease.kind == "logic"
                and any(claim.device_key == key for claim in lease.claims)
            )

    def field_policy(
        self,
        device_key: str,
        field_names: Sequence[str],
        *,
        dependency_groups: Sequence[Sequence[str]] = (),
    ) -> tuple[int, tuple[str, ...], dict[str, tuple[str, ...]]]:
        """Project active Logic blockers without changing exclusive admission."""

        key = str(device_key).strip()
        fields = tuple(str(value).strip() for value in field_names)
        if not key or any(not value for value in fields):
            raise ValueError("field policy requires a device key and field names")
        if len(set(fields)) != len(fields):
            raise ValueError("field policy field names must be unique")
        neighbors: dict[str, set[str]] = {}
        for raw_group in dependency_groups:
            group = tuple(str(value).strip() for value in raw_group)
            if any(not value for value in group):
                raise ValueError("device field dependency names must be non-empty")
            members = set(group)
            for name in members:
                neighbors.setdefault(name, set()).update(members - {name})

        def closure(name: str) -> set[str]:
            reached = {name}
            pending = [name]
            while pending:
                current = pending.pop()
                for dependency in neighbors.get(current, ()):
                    if dependency not in reached:
                        reached.add(dependency)
                        pending.append(dependency)
            return reached

        with self._lock:
            blockers: dict[str, list[str]] = {name: [] for name in fields}
            requested = set(fields)
            owners: list[str] = []
            for lease in self._leases.values():
                if lease.kind != "logic":
                    continue
                for claim in lease.claims:
                    if claim.device_key != key:
                        continue
                    if lease.label not in owners:
                        owners.append(lease.label)
                    for protected in claim.protected_fields:
                        for name in closure(protected) & requested:
                            if lease.label not in blockers[name]:
                                blockers[name].append(lease.label)
            return (
                self._device_owner_revisions.get(key, 0),
                tuple(owners),
                {name: tuple(labels) for name, labels in blockers.items()},
            )

    def _maintenance_blockers(
        self, claims: Sequence[DeviceClaim]
    ) -> tuple[str, ...]:
        keys = _claims_keys(claims)
        return tuple(
            barrier.label
            for barrier in self._maintenance.values()
            if keys & barrier.device_keys
        )

    def prepare_logic(
        self,
        owner: object,
        label: str,
        claims: Sequence[DeviceClaim],
        *,
        stop: Callable[[str], None],
        superseded: Callable[[], None],
    ) -> LogicReservation:
        """Reserve a built candidate, then ask only conflicting Logic owners to stop."""

        frozen = tuple(claims)
        with self._lock:
            maintenance = self._maintenance_blockers(frozen)
            if maintenance:
                raise DeviceUseBusy(maintenance)
            active_logic = tuple(
                lease
                for lease in self._leases.values()
                if lease.kind == "logic"
                and (lease.owner is owner or _conflict(frozen, lease.claims))
            )
            command_blockers = tuple(
                lease
                for lease in self._leases.values()
                if lease.kind != "logic"
                and (lease.owner is owner or _same_device(frozen, lease.claims))
            )
            if command_blockers:
                raise DeviceUseBusy(tuple(lease.label for lease in command_blockers))
            active_blockers = active_logic

            replaced = tuple(
                reservation
                for reservation in self._reservations.values()
                if reservation.owner is owner or _conflict(frozen, reservation.claims)
            )
            for reservation in replaced:
                self._reservations.pop(reservation.owner, None)

            reservation = LogicReservation(
                self,
                owner,
                label,
                frozen,
                stop,
                superseded,
            )
            self._reservations[owner] = reservation

        try:
            for previous in replaced:
                previous.superseded()
            for lease in active_blockers:
                if lease.stop is not None:
                    lease.stop(f"{label} needs its exclusive device")
        except BaseException:
            reservation.abort()
            raise
        return reservation

    def acquire_command(
        self,
        owner: object,
        label: str,
        claims: Sequence[DeviceClaim],
    ) -> DeviceLease:
        """Acquire a Pulse/editor command without preempting any Logic owner."""

        frozen = tuple(claims)
        with self._lock:
            existing = self._leases.get(owner)
            if existing is not None:
                raise DeviceUseBusy((existing.label,))
            blockers = self._maintenance_blockers(frozen) + tuple(
                lease.label
                for lease in self._leases.values()
                if _same_device(frozen, lease.claims)
            ) + tuple(
                reservation.label
                for reservation in self._reservations.values()
                if _same_device(frozen, reservation.claims)
            )
            if blockers:
                raise DeviceUseBusy(blockers)
            lease = DeviceLease(self, owner, label, "command", frozen, None)
            self._leases[owner] = lease
            return lease

    def acquire_field_command(
        self,
        owner: object,
        label: str,
        claim: DeviceClaim,
        *,
        dependency_groups: Sequence[Sequence[str]] = (),
        expected_owner_revision: int,
        allow_while_logic: bool,
    ) -> DeviceLease:
        """Acquire one unclaimed field while an unrelated Logic may stay active."""

        if not isinstance(claim, DeviceClaim) or len(claim.protected_fields) != 1:
            raise TypeError("field command requires one DeviceClaim protected field")
        if type(expected_owner_revision) is not int or expected_owner_revision < 0:
            raise TypeError("expected owner revision must be a non-negative integer")
        if type(allow_while_logic) is not bool:
            raise TypeError("allow_while_logic must be bool")
        field = claim.protected_fields[0]
        with self._lock:
            if owner in self._leases:
                raise DeviceUseBusy((self._leases[owner].label,))
            maintenance = self._maintenance_blockers((claim,))
            if maintenance:
                raise DeviceUseBusy(maintenance)
            revision, owners, policy = self.field_policy(
                claim.device_key,
                (field,),
                dependency_groups=dependency_groups,
            )
            if revision != expected_owner_revision:
                raise DeviceUseBusy(("device owners changed; review access again",))
            if policy[field]:
                raise DeviceUseBusy(policy[field])
            if owners and not allow_while_logic:
                raise DeviceUseBusy(owners)
            blockers = tuple(
                lease.label
                for lease in self._leases.values()
                if lease.kind != "logic" and _conflict((claim,), lease.claims)
            ) + tuple(
                reservation.label
                for reservation in self._reservations.values()
                if _same_device((claim,), reservation.claims)
            )
            if blockers:
                raise DeviceUseBusy(blockers)
            lease = DeviceLease(
                self,
                owner,
                label,
                "field_command",
                (claim,),
                None,
            )
            self._leases[owner] = lease
            return lease

    def _waiting_for(self, reservation: LogicReservation) -> tuple[str, ...]:
        with self._lock:
            if self._reservations.get(reservation.owner) is not reservation:
                return ()
            return tuple(
                dict.fromkeys(
                    lease.label
                    for lease in self._leases.values()
                    if lease.owner is reservation.owner
                    or _conflict(reservation.claims, lease.claims)
                )
            )

    def _commit(self, reservation: LogicReservation) -> DeviceLease:
        with self._lock:
            if self._reservations.get(reservation.owner) is not reservation:
                raise RuntimeError("logic reservation is no longer active")
            blockers = self._maintenance_blockers(reservation.claims) + tuple(
                lease.label
                for lease in self._leases.values()
                if lease.owner is reservation.owner
                or (
                    _conflict(reservation.claims, lease.claims)
                    if lease.kind == "logic"
                    else _same_device(reservation.claims, lease.claims)
                )
            )
            if blockers:
                raise DeviceUseBusy(blockers)
            self._reservations.pop(reservation.owner)
            self._condition.notify_all()
            lease = DeviceLease(
                self,
                reservation.owner,
                reservation.label,
                "logic",
                reservation.claims,
                reservation.stop,
            )
            self._leases[reservation.owner] = lease
            for key in _claims_keys(lease.claims):
                self._device_owner_revisions[key] = (
                    self._device_owner_revisions.get(key, 0) + 1
                )
            return lease

    def _abort(self, reservation: LogicReservation) -> bool:
        with self._lock:
            if self._reservations.get(reservation.owner) is not reservation:
                return False
            self._reservations.pop(reservation.owner)
            self._condition.notify_all()
            return True

    def _release(self, lease: DeviceLease) -> bool:
        with self._lock:
            if self._leases.get(lease.owner) is not lease:
                return False
            self._leases.pop(lease.owner)
            if lease.kind == "logic":
                for key in _claims_keys(lease.claims):
                    self._device_owner_revisions[key] = (
                        self._device_owner_revisions.get(key, 0) + 1
                    )
            self._condition.notify_all()
            return True

    def begin_maintenance(
        self,
        owner: object,
        label: str,
        device_keys: Sequence[str],
    ) -> DeviceMaintenance:
        """Exclude new users, refuse commands, and ask conflicting Logic to stop."""

        keys = frozenset(str(key) for key in device_keys if str(key))
        if not keys:
            raise ValueError("device maintenance requires at least one device key")
        with self._lock:
            if owner in self._maintenance:
                raise DeviceUseBusy((self._maintenance[owner].label,))
            overlapping = tuple(
                barrier.label
                for barrier in self._maintenance.values()
                if keys & barrier.device_keys
            )
            if overlapping:
                raise DeviceUseBusy(overlapping)
            command_blockers = tuple(
                lease.label
                for lease in self._leases.values()
                if lease.kind != "logic"
                and keys & _claims_keys(lease.claims)
            )
            if command_blockers:
                raise DeviceUseBusy(command_blockers)
            replaced = tuple(
                reservation
                for reservation in self._reservations.values()
                if keys & _claims_keys(reservation.claims)
            )
            for reservation in replaced:
                self._reservations.pop(reservation.owner, None)
            active_logic = tuple(
                lease
                for lease in self._leases.values()
                if lease.kind == "logic" and keys & _claims_keys(lease.claims)
            )
            barrier = DeviceMaintenance(self, owner, label, keys)
            self._maintenance[owner] = barrier
            self._condition.notify_all()
        try:
            for reservation in replaced:
                reservation.superseded()
            for lease in active_logic:
                if lease.stop is not None:
                    lease.stop(f"{label} needs {', '.join(sorted(keys))}")
        except BaseException:
            barrier.release()
            raise
        return barrier

    def _maintenance_waiting_for(
        self, barrier: DeviceMaintenance
    ) -> tuple[str, ...]:
        with self._lock:
            if self._maintenance.get(barrier.owner) is not barrier:
                return ()
            return tuple(
                lease.label
                for lease in self._leases.values()
                if barrier.device_keys & _claims_keys(lease.claims)
            )

    def _wait_maintenance(
        self, barrier: DeviceMaintenance, timeout: float | None
    ) -> None:
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while True:
                waiting = self._maintenance_waiting_for(barrier)
                if not waiting:
                    return
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "device maintenance still waits for " + ", ".join(waiting)
                    )
                self._condition.wait(remaining)

    def _release_maintenance(self, barrier: DeviceMaintenance) -> bool:
        with self._condition:
            if self._maintenance.get(barrier.owner) is not barrier:
                return False
            self._maintenance.pop(barrier.owner)
            self._condition.notify_all()
            return True

    def assert_idle(self) -> None:
        with self._lock:
            labels = tuple(
                lease.label for lease in self._leases.values()
            ) + tuple(
                reservation.label for reservation in self._reservations.values()
            ) + tuple(
                barrier.label for barrier in self._maintenance.values()
            )
        if labels:
            raise RuntimeError(f"device use still active: {', '.join(labels)}")


__all__ = [
    "DeviceClaim",
    "DeviceLease",
    "DeviceMaintenance",
    "DeviceUseBusy",
    "DeviceUseCoordinator",
    "LogicReservation",
]
