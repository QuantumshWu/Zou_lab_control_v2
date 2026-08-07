"""Dependency-ordered installation composition with reverse close."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .descriptors import CAPABILITY_TYPES, DeviceTypeDescriptor, InstallationFactoryContext, InstalledLeaf
from .discovery import discover_device_types


@dataclass(frozen=True)
class DeviceSpec:
    key: str
    type_id: str
    config: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.key or not self.type_id:
            raise ValueError("DeviceSpec requires key and type_id")
        object.__setattr__(self, "config", dict(self.config or {}))


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
        self._closed = True
        if errors:
            raise ExceptionGroup("installation close failed", errors)

    def __enter__(self) -> "Installation":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


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


def create_installation(
    specs: tuple[DeviceSpec, ...] | list[DeviceSpec] | str,
    *,
    world: object | None = None,
    broker: object | None = None,
    descriptors: tuple[DeviceTypeDescriptor, ...] | None = None,
    connect_pulse: object | None = None,
) -> Installation:
    from zlc_atom.execution import DeviceBroker
    from zlc_atom.devices.camera.world import SimulationWorld

    if isinstance(specs, str):
        from .templates import INSTALLATION_TEMPLATES
        try:
            specs = INSTALLATION_TEMPLATES[specs]
        except KeyError as exc:
            raise KeyError(f"unknown installation template {specs!r}") from exc
    normalized = tuple(spec if isinstance(spec, DeviceSpec) else DeviceSpec(**spec) for spec in specs)  # type: ignore[arg-type]
    if world is None:
        world = SimulationWorld(seed=0)
    if broker is None:
        broker = DeviceBroker()
    found = tuple(descriptors or discover_device_types())
    by_type = {value.type_id: value for value in found}
    unknown = {spec.type_id for spec in normalized} - set(by_type)
    if unknown:
        raise KeyError(f"unknown device types: {sorted(unknown)}")
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
