"""One session-scoped truth for who may drive an installed device.

Logic nodes and PulseGUI share exact device objects from one ExperimentSession.
This module coordinates those callers; it never wraps a device or executes a
hardware command.  Read-only access needs no lease.  Exclusive claims conflict
only when they name the same object by identity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import threading


@dataclass(frozen=True)
class DeviceClaim:
    """One resolved device use frozen into a candidate run."""

    argument_name: str
    device_key: str
    device: object = field(compare=False)
    access: object


def _exclusive(claim: DeviceClaim) -> bool:
    return str(getattr(claim.access, "value", claim.access)) == "exclusive"


def _conflict(left: Sequence[DeviceClaim], right: Sequence[DeviceClaim]) -> bool:
    return any(
        first.device is second.device and _exclusive(first) and _exclusive(second)
        for first in left
        for second in right
    )


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
        self._leases: dict[object, DeviceLease] = {}
        self._reservations: dict[object, LogicReservation] = {}

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
            active_blockers = tuple(
                lease
                for lease in self._leases.values()
                if lease.owner is owner or _conflict(frozen, lease.claims)
            )
            command_blockers = tuple(
                lease for lease in active_blockers if lease.kind != "logic"
            )
            if command_blockers:
                raise DeviceUseBusy(tuple(lease.label for lease in command_blockers))

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
            blockers = tuple(
                lease.label
                for lease in self._leases.values()
                if _conflict(frozen, lease.claims)
            ) + tuple(
                reservation.label
                for reservation in self._reservations.values()
                if _conflict(frozen, reservation.claims)
            )
            if blockers:
                raise DeviceUseBusy(blockers)
            lease = DeviceLease(self, owner, label, "command", frozen, None)
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
            blockers = tuple(
                lease.label
                for lease in self._leases.values()
                if lease.owner is reservation.owner
                or _conflict(reservation.claims, lease.claims)
            )
            if blockers:
                raise DeviceUseBusy(blockers)
            self._reservations.pop(reservation.owner)
            lease = DeviceLease(
                self,
                reservation.owner,
                reservation.label,
                "logic",
                reservation.claims,
                reservation.stop,
            )
            self._leases[reservation.owner] = lease
            return lease

    def _abort(self, reservation: LogicReservation) -> bool:
        with self._lock:
            if self._reservations.get(reservation.owner) is not reservation:
                return False
            self._reservations.pop(reservation.owner)
            return True

    def _release(self, lease: DeviceLease) -> bool:
        with self._lock:
            if self._leases.get(lease.owner) is not lease:
                return False
            self._leases.pop(lease.owner)
            return True

    def assert_idle(self) -> None:
        with self._lock:
            labels = tuple(
                lease.label for lease in self._leases.values()
            ) + tuple(
                reservation.label for reservation in self._reservations.values()
            )
        if labels:
            raise RuntimeError(f"device use still active: {', '.join(labels)}")


__all__ = [
    "DeviceClaim",
    "DeviceLease",
    "DeviceUseBusy",
    "DeviceUseCoordinator",
    "LogicReservation",
]
