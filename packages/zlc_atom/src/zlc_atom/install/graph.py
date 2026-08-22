"""Dependency-ordered installation composition with reverse close."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
import threading
from typing import Any, Mapping

from .descriptors import CAPABILITY_TYPES, DeviceTypeDescriptor, InstallationFactoryContext, InstalledLeaf
from .discovery import DeviceCatalogSnapshot, discover_device_catalog
from .configuration import _freeze_plain


@dataclass(frozen=True)
class DeviceSpec:
    key: str
    type_id: str
    config: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.key or not self.type_id:
            raise ValueError("DeviceSpec requires key and type_id")
        if self.config is not None and not isinstance(self.config, Mapping):
            raise TypeError("DeviceSpec config must be a mapping or None")
        object.__setattr__(
            self,
            "config",
            _freeze_plain(dict(self.config or {}), "device config"),
        )


class Installation:
    def __init__(
        self,
        devices: Mapping[str, InstalledLeaf],
        *,
        world: object | None,
        failures: Mapping[str, BaseException] | None = None,
        broker: object | None = None,
    ) -> None:
        from zlc_atom.execution import DeviceBroker

        if broker is not None and not isinstance(broker, DeviceBroker):
            raise TypeError("installation broker must be DeviceBroker or None")
        prepared = dict(devices)
        if any(not isinstance(leaf, InstalledLeaf) for leaf in prepared.values()):
            raise TypeError("installation devices must be InstalledLeaf values")
        mismatched = {key for key, leaf in prepared.items() if key != leaf.key}
        if mismatched:
            raise ValueError(
                f"installation mapping keys differ from leaf identity: {sorted(mismatched)}"
            )
        if broker is None and any(leaf.binding is not None for leaf in prepared.values()):
            raise ValueError("bound installation leaves require their DeviceBroker")
        if broker is not None:
            for leaf in prepared.values():
                if leaf.binding is not None:
                    broker.verify_capability(leaf.binding)
        self._lock = threading.RLock()
        self._devices = prepared
        self.world = world
        self._failures = dict(failures or {})
        self._broker = broker
        self._revision = 0
        self._borrow_tokens: set[object] = set()
        self._closing = False
        self._closed = False

    @property
    def devices(self) -> Mapping[str, InstalledLeaf]:
        with self._lock:
            return dict(self._devices)

    @property
    def broker(self) -> object | None:
        """The identity owner shared by successor installations."""

        return self._broker

    @property
    def revision(self) -> int:
        """Monotonic ownership revision for optimistic reconcile admission."""

        with self._lock:
            return self._revision

    def _borrow_snapshot(
        self,
        expected_revision: int,
    ) -> tuple[object, dict[str, InstalledLeaf]]:
        """Pin one exact ownership snapshot for successor factory construction."""

        if type(expected_revision) is not int or expected_revision < 0:
            raise TypeError("borrowed_revision must be a non-negative integer")
        with self._lock:
            if self._revision != expected_revision:
                raise RuntimeError("borrowed installation ownership revision changed")
            if self._closing or self._closed:
                raise RuntimeError("borrowed installation is closing or closed")
            token = object()
            self._borrow_tokens.add(token)
            return token, dict(self._devices)

    def _release_borrow(self, token: object) -> None:
        with self._lock:
            if token not in self._borrow_tokens:
                raise RuntimeError("installation borrow token is no longer active")
            self._borrow_tokens.remove(token)

    @property
    def failures(self) -> Mapping[str, BaseException]:
        """Per-leaf startup failures; independent leaves remain usable."""

        with self._lock:
            return dict(self._failures)

    def device(self, key: str) -> object:
        with self._lock:
            try:
                return self._devices[key].device
            except KeyError as exc:
                raise KeyError(f"no installed device {key!r}") from exc

    def capability(self, token: str, *, key: str | None = None) -> object:
        with self._lock:
            candidates = (
                tuple(self._devices.values())
                if key is None
                else (self._devices[key],)
            )
            for leaf in candidates:
                if token in leaf.capabilities:
                    value = leaf.capabilities[token]
                    expected = CAPABILITY_TYPES.get(token)
                    if expected is not None and not isinstance(value, expected):
                        raise TypeError(f"capability {token!r} has the wrong type")
                    return value
        raise KeyError(f"no installed device provides capability {token!r}")

    def transfer_leaves_to(
        self,
        target: "Installation",
        keys: tuple[str, ...] | list[str],
        *,
        source_revision: int,
        target_revision: int,
    ) -> tuple[str, ...]:
        """Atomically move unchanged leaf ownership into one successor.

        Both revisions are required because reconcile commonly prepares a
        successor off-thread.  A stale completion must not detach devices from
        a source or overwrite a target that changed while it was preparing.
        The two installations must share the same broker.  A world-affine leaf
        must also retain its factory world; an independent physical leaf may
        cross that boundary because it closes over no simulation state.
        """

        if not isinstance(target, Installation):
            raise TypeError("transfer target must be Installation")
        if target is self:
            raise ValueError("installation cannot transfer leaves to itself")
        if type(source_revision) is not int or source_revision < 0:
            raise TypeError("source_revision must be a non-negative integer")
        if type(target_revision) is not int or target_revision < 0:
            raise TypeError("target_revision must be a non-negative integer")
        if not isinstance(keys, (tuple, list)):
            raise TypeError("transfer keys must be a tuple or list")
        if any(type(key) is not str for key in keys):
            raise TypeError("transfer keys must contain text")
        normalized = tuple(keys)
        if (
            not normalized
            or any(not key for key in normalized)
            or len(set(normalized)) != len(normalized)
        ):
            raise ValueError("transfer keys must be unique non-empty text")
        # One global order prevents A->B and B->A reconciles from deadlocking.
        first, second = (
            (self, target) if id(self._lock) < id(target._lock) else (target, self)
        )
        with first._lock:
            with second._lock:
                if self._revision != source_revision:
                    raise RuntimeError("source installation ownership revision changed")
                if target._revision != target_revision:
                    raise RuntimeError("target installation ownership revision changed")
                if self._closing or self._closed:
                    raise RuntimeError("source installation is closing or closed")
                if target._closing or target._closed:
                    raise RuntimeError("target installation is closing or closed")
                if self._borrow_tokens:
                    raise RuntimeError("source installation leaves are borrowed")
                if target._borrow_tokens:
                    raise RuntimeError("target installation leaves are borrowed")
                if self._broker is not target._broker:
                    raise RuntimeError("installation transfer requires one DeviceBroker")
                wanted = set(normalized)
                ordered = tuple(key for key in self._devices if key in wanted)
                missing = wanted - set(ordered)
                if missing:
                    raise KeyError(f"source installation has no leaves {sorted(missing)}")
                occupied = wanted.intersection(target._devices, target._failures)
                if occupied:
                    raise ValueError(
                        f"target installation already owns keys {sorted(occupied)}"
                    )
                if self.world is not target.world:
                    world_bound = tuple(
                        key
                        for key in ordered
                        if self._devices[key].world_affinity is not None
                    )
                    if world_bound:
                        raise RuntimeError(
                            "installation transfer crosses worlds for bound leaves "
                            f"{list(world_bound)}"
                        )
                broker = self._broker
                if broker is not None:
                    for key in ordered:
                        binding = self._devices[key].binding
                        if binding is not None:
                            broker.verify_capability(binding)
                moved = {key: self._devices.pop(key) for key in ordered}
                # Retained dependencies precede newly built leaves, preserving
                # reverse-close order for the successor.
                target._devices = {**moved, **target._devices}
                self._revision += 1
                target._revision += 1
                return ordered

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._closing:
                raise RuntimeError("installation close is already in progress")
            if self._borrow_tokens:
                raise RuntimeError("installation leaves are borrowed")
            self._closing = True
            pending = tuple(reversed(tuple(self._devices.items())))
        errors: list[BaseException] = []
        closed_keys: list[str] = []
        for key, leaf in pending:
            try:
                leaf.close()
                if leaf.binding is not None:
                    assert self._broker is not None
                    self._broker.unbind(leaf.binding)
            except BaseException as error:
                errors.append(error)
                # Insertion order is topological and close order is its
                # reverse.  Everything left in the loop was constructed before
                # this failed leaf and may be a dependency it still needs in
                # order to retry close safely.
                break
            else:
                closed_keys.append(key)
        with self._lock:
            for key in closed_keys:
                # A transfer cannot run while _closing is true, so identity is
                # unchanged from the snapshot above.
                self._devices.pop(key, None)
            transitioned = bool(closed_keys)
            if not self._devices and not errors:
                self._closed = True
                transitioned = True
            if transitioned:
                self._revision += 1
            self._closing = False
        if errors:
            raise BaseExceptionGroup("installation close failed", errors)

    def __enter__(self) -> "Installation":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def tunable_devices(installation: Installation) -> dict[str, object]:
    """Every installed device that volunteers scan-tunable fields, by key.

    A device volunteers by exposing ``tunable_fields()`` -- authoring fields
    whose bounds are BOTH declared, because an unbounded knob is not
    scannable -- and ``tune(name, value)`` to move one of them at runtime.
    Duck-typed like the optional ``close``: a device without runtime knobs
    simply does not appear, which is an honest absence rather than a stub.
    """

    return {
        key: leaf.device
        for key, leaf in installation.devices.items()
        if callable(getattr(leaf.device, "tunable_fields", None))
        and callable(getattr(leaf.device, "tune", None))
    }


def _topological(
    specs: tuple[DeviceSpec, ...],
    descriptors: Mapping[str, DeviceTypeDescriptor],
    *,
    borrowed_types: frozenset[str] = frozenset(),
) -> tuple[DeviceSpec, ...]:
    by_type = {spec.type_id: spec for spec in specs}
    missing = {
        dependency
        for spec in specs
        for dependency in descriptors[spec.type_id].dependencies
        if dependency not in by_type and dependency not in borrowed_types
    }
    if missing:
        raise ValueError(f"device dependency graph has missing dependencies: {sorted(missing)}")
    pending = list(specs)
    result: list[DeviceSpec] = []
    while pending:
        progressed = False
        for spec in tuple(pending):
            descriptor = descriptors[spec.type_id]
            if all(
                dependency in borrowed_types
                or any(done.type_id == dependency for done in result)
                for dependency in descriptor.dependencies
            ):
                pending.remove(spec)
                result.append(spec)
                progressed = True
        if not progressed:
            raise ValueError("device dependency graph contains a cycle or missing dependency")
    return tuple(result)


def _world_from_apparatus(
    specs: tuple[DeviceSpec, ...],
    descriptors: Mapping[str, DeviceTypeDescriptor],
    simulation: Mapping[str, Any],
) -> object:
    """Build one shared world from the installation's root simulation truth."""

    from zlc_atom.devices.simulation import SimulationWorld, SimulationWorldConfig

    resolvers = {
        descriptor.world_config
        for spec in specs
        for descriptor in (descriptors[spec.type_id],)
        if descriptor.world_config is not None
    }
    if not resolvers:
        if simulation:
            raise ValueError("installation simulation requires a virtual device")
        return None
    if len(resolvers) != 1:
        raise ValueError("one installation cannot have different world owners")
    config = next(iter(resolvers))(dict(simulation))
    if not isinstance(config, SimulationWorldConfig):
        raise TypeError("simulation world resolver returned the wrong type")
    return SimulationWorld(config)


def _admit_factory_leaf(
    leaf: object,
    spec: DeviceSpec,
    descriptor: DeviceTypeDescriptor,
    broker: object,
) -> InstalledLeaf:
    """Validate the complete identity a factory claims before ownership passes."""

    from zlc_atom.execution import DeviceBroker, ResourceKey

    if not isinstance(broker, DeviceBroker):
        raise TypeError("factory admission requires DeviceBroker")
    if not isinstance(leaf, InstalledLeaf):
        raise TypeError(f"factory {descriptor.type_id} did not return InstalledLeaf")
    if leaf.key != spec.key:
        raise ValueError(
            f"factory {descriptor.type_id} returned leaf key {leaf.key!r}, "
            f"expected {spec.key!r}"
        )
    if leaf.type_id != descriptor.type_id:
        raise ValueError(
            f"factory {descriptor.type_id} returned leaf type {leaf.type_id!r}"
        )
    if not isinstance(leaf.capabilities, Mapping):
        raise TypeError("installed leaf capabilities must be a mapping")
    verified: Mapping[str, object] = {}
    if leaf.binding is not None:
        expected_key = ResourceKey.parse(f"device/{spec.key}")
        if leaf.binding.key != expected_key:
            raise ValueError(
                f"factory {descriptor.type_id} bound {leaf.binding.key}, "
                f"expected {expected_key}"
            )
        verified = broker.verify_capability(leaf.binding).snapshot
    for token in descriptor.capabilities:
        if token not in leaf.capabilities:
            raise RuntimeError(
                f"factory {descriptor.type_id} did not provide declared "
                f"capability {token!r}"
            )
        expected = CAPABILITY_TYPES[token]
        value = leaf.capabilities[token]
        if not isinstance(value, expected):
            raise TypeError(
                f"factory {descriptor.type_id} capability {token!r} has wrong type"
            )
        if leaf.binding is not None and verified.get(token) is not value:
            raise RuntimeError(
                f"factory {descriptor.type_id} capability {token!r} differs "
                "from its verified binding"
            )
    return leaf


def _close_factory_leaf(leaf: InstalledLeaf) -> tuple[BaseException, ...]:
    """Close then unbind one rejected/rolled-back leaf, preserving every error."""

    try:
        leaf.close()
    except BaseException as error:
        return (error,)
    binding = leaf.binding
    if binding is None:
        return ()
    try:
        binding.broker.unbind(binding)
    except BaseException as error:
        return (error,)
    return ()


def _rollback_factory_leaves(
    installed: Mapping[str, InstalledLeaf],
) -> tuple[tuple[BaseException, ...], tuple[InstalledLeaf, ...]]:
    ordered = tuple(installed.values())
    for index in range(len(ordered) - 1, -1, -1):
        leaf = ordered[index]
        failed = _close_factory_leaf(leaf)
        if failed:
            # Everything before this leaf may be a dependency it still needs
            # for a later close retry.  Recovery owns the whole intact prefix.
            return failed, ordered[: index + 1]
    return (), ()


class InstallationRecovery:
    """Strong ownership of leaves that could not be closed during composition."""

    def __init__(self, leaves: Iterable[InstalledLeaf]) -> None:
        prepared = tuple(leaves)
        if not prepared or any(not isinstance(leaf, InstalledLeaf) for leaf in prepared):
            raise TypeError("installation recovery requires InstalledLeaf values")
        self._lock = threading.RLock()
        self._leaves = list(prepared)
        self._closing = False

    @property
    def leaves(self) -> tuple[InstalledLeaf, ...]:
        with self._lock:
            return tuple(self._leaves)

    def close(self) -> None:
        with self._lock:
            if not self._leaves:
                return
            if self._closing:
                raise RuntimeError("installation recovery close is already in progress")
            self._closing = True
        error: BaseException | None = None
        try:
            while True:
                with self._lock:
                    if not self._leaves:
                        break
                    leaf = self._leaves[-1]
                failed = _close_factory_leaf(leaf)
                if failed:
                    error = BaseExceptionGroup(
                        "installation recovery close failed", list(failed)
                    )
                    break
                with self._lock:
                    if not self._leaves or self._leaves[-1] is not leaf:
                        raise RuntimeError("installation recovery ownership changed")
                    self._leaves.pop()
        finally:
            with self._lock:
                self._closing = False
        if error is not None:
            raise error


class InstallationCompositionError(BaseExceptionGroup):
    """Composition failed and ``recovery`` still owns unclosed leaves."""

    def __new__(
        cls,
        message: str,
        errors: tuple[BaseException, ...],
        recovery: InstallationRecovery,
    ) -> "InstallationCompositionError":
        instance = super().__new__(cls, message, errors)
        instance.recovery = recovery
        return instance

    def __init__(
        self,
        message: str,
        errors: tuple[BaseException, ...],
        recovery: InstallationRecovery,
    ) -> None:
        super().__init__(message, errors)

    def derive(
        self,
        errors: Iterable[BaseException],
    ) -> "InstallationCompositionError":
        return InstallationCompositionError(
            self.message, tuple(errors), self.recovery
        )


_BLUEPRINT_TOKEN = object()


@dataclass(frozen=True)
class InstallationBlueprint:
    """Side-effect-free, dependency-ordered input to device factory execution."""

    _token: object
    specs: tuple[DeviceSpec, ...]
    catalog: DeviceCatalogSnapshot
    world: object | None
    borrowed_from: Installation | None = None
    borrowed_revision: int | None = None
    borrowed_keys: tuple[str, ...] = ()
    borrowed_types: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self._token is not _BLUEPRINT_TOKEN:
            raise PermissionError(
                "InstallationBlueprint can only be minted by preflight_installation"
            )

    def dependent_keys(self, affected_types: Iterable[str]) -> frozenset[str]:
        """Target leaves transitively constructed from any affected type."""

        affected = {str(value) for value in affected_types}
        descriptors = {
            descriptor.type_id: descriptor for descriptor in self.catalog.available
        }
        result: set[str] = set()
        changed = True
        while changed:
            changed = False
            for spec in self.specs:
                if spec.key in result:
                    continue
                if affected.intersection(descriptors[spec.type_id].dependencies):
                    result.add(spec.key)
                    affected.add(spec.type_id)
                    changed = True
        return frozenset(result)


def preflight_installation(
    specs: tuple[DeviceSpec, ...] | list[DeviceSpec] | str,
    *,
    world: object | None = None,
    simulation: Mapping[str, Any] | None = None,
    catalog: DeviceCatalogSnapshot | None = None,
    borrowed_from: Installation | None = None,
    borrowed_revision: int | None = None,
) -> InstallationBlueprint:
    """Validate structure/topology and resolve the world without opening devices."""

    snapshot = catalog if catalog is not None else discover_device_catalog()
    if not isinstance(snapshot, DeviceCatalogSnapshot):
        raise TypeError("catalog must be DeviceCatalogSnapshot or None")
    simulation_supplied = simulation is not None
    if isinstance(specs, str):
        from .templates import installation_config_from_template

        config = installation_config_from_template(snapshot, specs)
        specs = config.specs()
        if simulation is None:
            simulation = config.simulation
    if simulation is not None and not isinstance(simulation, Mapping):
        raise TypeError("simulation must be a mapping or None")
    frozen_simulation = _freeze_plain(
        dict(simulation or {}), "installation simulation"
    )
    normalized = tuple(
        spec if isinstance(spec, DeviceSpec) else DeviceSpec(**spec)
        for spec in specs
    )  # type: ignore[arg-type]
    keys = tuple(spec.key for spec in normalized)
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise ValueError(f"duplicate device key(s): {sorted(duplicates)}")
    by_type = {value.type_id: value for value in snapshot.available}
    unknown = {spec.type_id for spec in normalized} - set(by_type)
    if unknown:
        raise KeyError(f"unknown device types: {sorted(unknown)}")

    borrow_token = None
    borrowed: dict[str, InstalledLeaf] = {}
    if borrowed_from is None:
        if borrowed_revision is not None:
            raise ValueError("borrowed_revision requires borrowed_from")
    else:
        if not isinstance(borrowed_from, Installation):
            raise TypeError("borrowed_from must be Installation or None")
        if borrowed_revision is None:
            raise ValueError("borrowed_from requires borrowed_revision")
        borrow_token, borrowed = borrowed_from._borrow_snapshot(borrowed_revision)
    try:
        overlap = set(borrowed).intersection(keys)
        if overlap:
            raise ValueError(
                f"borrowed leaves duplicate device spec key(s): {sorted(overlap)}"
            )
        world_bound = tuple(
            key for key, leaf in borrowed.items() if leaf.world_affinity is not None
        )
        if borrowed_from is not None and not simulation_supplied and world is None:
            resolved_world = borrowed_from.world
            if resolved_world is None and any(
                by_type[spec.type_id].world_config is not None for spec in normalized
            ):
                resolved_world = _world_from_apparatus(
                    normalized, by_type, frozen_simulation
                )
        elif world is None:
            if world_bound:
                raise ValueError(
                    "world-bound borrowed leaves cannot accompany a new simulation world"
                )
            resolved_world = _world_from_apparatus(
                normalized, by_type, frozen_simulation
            )
        else:
            if simulation_supplied:
                raise ValueError("pass an explicit world or simulation config, not both")
            incompatible = tuple(
                key
                for key, leaf in borrowed.items()
                if leaf.world_affinity is not None
                and leaf.world_affinity is not world
            )
            if incompatible:
                raise ValueError(
                    "borrowed leaves belong to another world: "
                    f"{list(incompatible)}"
                )
            resolved_world = world
        borrowed_types = frozenset(leaf.type_id for leaf in borrowed.values())
        ordered = _topological(
            normalized,
            by_type,
            borrowed_types=borrowed_types,
        )
        return InstallationBlueprint(
            _BLUEPRINT_TOKEN,
            ordered,
            snapshot,
            resolved_world,
            borrowed_from,
            borrowed_revision,
            tuple(borrowed),
            borrowed_types,
        )
    finally:
        if borrow_token is not None:
            assert borrowed_from is not None
            borrowed_from._release_borrow(borrow_token)


def _raise_composition_failure(
    message: str,
    original: BaseException,
    *cleanup_groups: tuple[BaseException, ...],
    recovery_leaves: tuple[InstalledLeaf, ...] = (),
) -> None:
    failures = (original, *(item for group in cleanup_groups for item in group))
    if recovery_leaves:
        raise InstallationCompositionError(
            message,
            failures,
            InstallationRecovery(recovery_leaves),
        ) from None
    if len(failures) == 1:
        raise original
    raise BaseExceptionGroup(message, list(failures)) from None


def create_installation(
    specs: tuple[DeviceSpec, ...] | list[DeviceSpec] | str | InstallationBlueprint,
    *,
    world: object | None = None,
    simulation: Mapping[str, Any] | None = None,
    broker: object | None = None,
    catalog: DeviceCatalogSnapshot | None = None,
    connect_pulse: object | None = None,
    borrowed_from: Installation | None = None,
    borrowed_revision: int | None = None,
) -> Installation:
    """Open only the leaves described by one proven installation blueprint."""

    from zlc_atom.execution import DeviceBroker

    if isinstance(specs, InstallationBlueprint):
        if (
            world is not None
            or simulation is not None
            or catalog is not None
            or borrowed_from is not None
            or borrowed_revision is not None
        ):
            raise ValueError(
                "an InstallationBlueprint already owns world, catalog, and borrow input"
            )
        blueprint = specs
    else:
        blueprint = preflight_installation(
            specs,
            world=world,
            simulation=simulation,
            catalog=catalog,
            borrowed_from=borrowed_from,
            borrowed_revision=borrowed_revision,
        )

    owner = blueprint.borrowed_from
    borrow_token = None
    borrowed: dict[str, InstalledLeaf] = {}
    if owner is not None:
        assert blueprint.borrowed_revision is not None
        borrow_token, borrowed = owner._borrow_snapshot(
            blueprint.borrowed_revision
        )
    try:
        if tuple(borrowed) != blueprint.borrowed_keys or frozenset(
            leaf.type_id for leaf in borrowed.values()
        ) != blueprint.borrowed_types:
            raise RuntimeError("borrowed installation leaves changed after preflight")
        if owner is not None:
            if owner.broker is None:
                raise RuntimeError("borrowed installation has no DeviceBroker")
            if broker is None:
                broker = owner.broker
            elif broker is not owner.broker:
                raise RuntimeError("successor requires the borrowed installation broker")
        if broker is None:
            broker = DeviceBroker()
        elif not isinstance(broker, DeviceBroker):
            raise TypeError("broker must be DeviceBroker or None")
        for leaf in borrowed.values():
            if leaf.binding is not None:
                broker.verify_capability(leaf.binding)

        by_type = {
            descriptor.type_id: descriptor
            for descriptor in blueprint.catalog.available
        }
        installed: dict[str, InstalledLeaf] = {}
        failures: dict[str, BaseException] = {}
        successful_types = set(blueprint.borrowed_types)
        no_leaf = object()
        for spec in blueprint.specs:
            descriptor = by_type[spec.type_id]
            unavailable = tuple(
                dependency
                for dependency in descriptor.dependencies
                if dependency not in successful_types
            )
            if unavailable:
                failures[spec.key] = RuntimeError(
                    "device dependencies unavailable: " + ", ".join(unavailable)
                )
                continue
            candidate: object = no_leaf
            try:
                context = InstallationFactoryContext(
                    blueprint.world,
                    broker,
                    {**borrowed, **installed},
                    connect_pulse,
                )
                candidate = descriptor.factory(  # type: ignore[misc]
                    context, spec.key, spec.config
                )
                leaf = _admit_factory_leaf(candidate, spec, descriptor, broker)
                if leaf.world_affinity is not None:
                    raise ValueError("device factory must not assign world_affinity")
                leaf = replace(
                    leaf,
                    world_affinity=(
                        blueprint.world
                        if descriptor.world_config is not None
                        else None
                    ),
                )
            except BaseException as original:
                if candidate is no_leaf:
                    failures[spec.key] = original
                    continue
                if not isinstance(candidate, InstalledLeaf):
                    rollback, rollback_remaining = _rollback_factory_leaves(installed)
                    installed.clear()
                    _raise_composition_failure(
                        f"device {spec.key!r} factory result was not ownable",
                        original,
                        rollback,
                        recovery_leaves=rollback_remaining,
                    )
                cleanup = _close_factory_leaf(candidate)
                if cleanup:
                    rollback, rollback_remaining = _rollback_factory_leaves(installed)
                    installed.clear()
                    _raise_composition_failure(
                        f"device {spec.key!r} admission and cleanup failed",
                        original,
                        cleanup,
                        rollback,
                        recovery_leaves=(*rollback_remaining, candidate),
                    )
                failures[spec.key] = original
                continue
            installed[spec.key] = leaf
            successful_types.add(spec.type_id)
        try:
            return Installation(
                installed,
                world=blueprint.world,
                failures=failures,
                broker=broker,
            )
        except BaseException as original:
            rollback, rollback_remaining = _rollback_factory_leaves(installed)
            _raise_composition_failure(
                "installation ownership admission failed",
                original,
                rollback,
                recovery_leaves=rollback_remaining,
            )
    finally:
        if borrow_token is not None:
            assert owner is not None
            owner._release_borrow(borrow_token)


__all__ = [
    "DeviceSpec",
    "Installation",
    "InstallationBlueprint",
    "InstallationCompositionError",
    "InstallationRecovery",
    "create_installation",
    "preflight_installation",
]
