"""Dependency-ordered installation composition with reverse close."""

from __future__ import annotations

from dataclasses import dataclass
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
    ) -> None:
        self._devices = dict(devices)
        self.world = world
        self._failures = dict(failures or {})
        self._closed = False

    @property
    def devices(self) -> Mapping[str, InstalledLeaf]:
        return dict(self._devices)

    @property
    def failures(self) -> Mapping[str, BaseException]:
        """Per-leaf startup failures; independent leaves remain usable."""

        return dict(self._failures)

    def device(self, key: str) -> object:
        try:
            return self._devices[key].device
        except KeyError as exc:
            raise KeyError(f"no installed device {key!r}") from exc

    def capability(self, token: str, *, key: str | None = None) -> object:
        candidates = self._devices.values() if key is None else (self._devices[key],)
        for leaf in candidates:
            if token in leaf.capabilities:
                value = leaf.capabilities[token]
                expected = CAPABILITY_TYPES.get(token)
                if expected is not None and not isinstance(value, expected):
                    raise TypeError(f"capability {token!r} has the wrong type")
                return value
        raise KeyError(f"no installed device provides capability {token!r}")

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for leaf in reversed(tuple(self._devices.values())):
            try:
                leaf.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("installation close failed", errors)
        self._closed = True

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


def _topological(specs: tuple[DeviceSpec, ...], descriptors: Mapping[str, DeviceTypeDescriptor]) -> tuple[DeviceSpec, ...]:
    by_type = {spec.type_id: spec for spec in specs}
    missing = {
        dependency
        for spec in specs
        for dependency in descriptors[spec.type_id].dependencies
        if dependency not in by_type
    }
    if missing:
        raise ValueError(f"device dependency graph has missing dependencies: {sorted(missing)}")
    pending = list(specs)
    result: list[DeviceSpec] = []
    while pending:
        progressed = False
        for spec in tuple(pending):
            descriptor = descriptors[spec.type_id]
            if all(any(done.type_id == dep for done in result) for dep in descriptor.dependencies):
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


def create_installation(
    specs: tuple[DeviceSpec, ...] | list[DeviceSpec] | str,
    *,
    world: object | None = None,
    simulation: Mapping[str, Any] | None = None,
    broker: object | None = None,
    catalog: DeviceCatalogSnapshot | None = None,
    connect_pulse: object | None = None,
) -> Installation:
    from zlc_atom.execution import DeviceBroker
    simulation_supplied = simulation is not None
    snapshot = catalog if catalog is not None else discover_device_catalog()
    if not isinstance(snapshot, DeviceCatalogSnapshot):
        raise TypeError("catalog must be DeviceCatalogSnapshot or None")
    if isinstance(specs, str):
        from .templates import installation_config_from_template

        config = installation_config_from_template(snapshot, specs)
        specs = config.specs()
        if simulation is None:
            simulation = config.simulation
    if simulation is not None and not isinstance(simulation, Mapping):
        raise TypeError("simulation must be a mapping or None")
    simulation = _freeze_plain(dict(simulation or {}), "installation simulation")
    normalized = tuple(spec if isinstance(spec, DeviceSpec) else DeviceSpec(**spec) for spec in specs)  # type: ignore[arg-type]
    seen_keys: set[str] = set()
    duplicate_keys: set[str] = set()
    for spec in normalized:
        if spec.key in seen_keys:
            duplicate_keys.add(spec.key)
        seen_keys.add(spec.key)
    if duplicate_keys:
        raise ValueError(f"duplicate device key(s): {sorted(duplicate_keys)}")
    found = snapshot.available
    by_type = {value.type_id: value for value in found}
    unknown = {spec.type_id for spec in normalized} - set(by_type)
    if unknown:
        raise KeyError(f"unknown device types: {sorted(unknown)}")
    if world is None:
        world = _world_from_apparatus(normalized, by_type, simulation)
    elif simulation_supplied:
        raise ValueError("pass an explicit world or simulation config, not both")
    if broker is None:
        broker = DeviceBroker()
    ordered = _topological(normalized, by_type)
    installed: dict[str, InstalledLeaf] = {}
    failures: dict[str, BaseException] = {}
    successful_types: set[str] = set()
    for spec in ordered:
        descriptor = by_type[spec.type_id]
        unavailable = tuple(
            dependency
            for dependency in descriptor.dependencies
            if dependency not in successful_types
        )
        if unavailable:
            failures[spec.key] = RuntimeError(
                f"device dependencies unavailable: {', '.join(unavailable)}"
            )
            continue
        leaf: InstalledLeaf | None = None
        try:
            context = InstallationFactoryContext(world, broker, dict(installed), connect_pulse)
            leaf = descriptor.factory(context, spec.key, spec.config)  # type: ignore[misc]
            if not isinstance(leaf, InstalledLeaf):
                raise TypeError(f"factory {descriptor.type_id} did not return InstalledLeaf")
            for token in descriptor.capabilities:
                if token not in leaf.capabilities:
                    raise RuntimeError(f"factory {descriptor.type_id} did not provide declared capability {token!r}")
                expected = CAPABILITY_TYPES[token]
                if not isinstance(leaf.capabilities[token], expected):
                    raise TypeError(f"factory {descriptor.type_id} capability {token!r} has wrong type")
        except BaseException as error:
            if leaf is not None:
                try:
                    leaf.close()
                except BaseException as close_error:
                    error = ExceptionGroup(
                        f"device {spec.key!r} startup failed and cleanup failed",
                        [error, close_error],
                    )
            failures[spec.key] = error
            continue
        installed[spec.key] = leaf
        successful_types.add(spec.type_id)
    return Installation(installed, world=world, failures=failures)


__all__ = ["DeviceSpec", "Installation", "create_installation"]
