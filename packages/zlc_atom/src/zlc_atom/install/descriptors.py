"""Data-only device type declarations and the capability type table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.execution.capabilities import CAPABILITY_TYPES
from zlc_atom.execution.ports import BoundDevice
from zlc_atom.install.configuration import DeviceInstanceConfig


@dataclass(frozen=True)
class InstalledLeaf:
    key: str
    type_id: str
    device: object
    capabilities: Mapping[str, object]
    binding: BoundDevice | None = None
    closer: Callable[[], None] | None = None

    def close(self) -> None:
        if self.closer is not None:
            self.closer()


@dataclass(frozen=True)
class InstallationFactoryContext:
    # Backend-owned composition state.  A device family validates the concrete
    # value it consumes; the generic installation descriptor does not depend on
    # any simulation implementation.
    world: object | None
    broker: object
    devices: Mapping[str, InstalledLeaf]
    #: How to reach a pulse server, supplied by the composition root.
    #:
    #: This package declares WHERE a board is -- host, port, timeout, written
    #: down in the device configuration so it can be saved and reopened -- and
    #: never HOW to dial it, because importing the pulse client here would break
    #: the boundary that keeps the domain independent of the device packages.
    connect_pulse: Callable[..., object] | None = None


@dataclass(frozen=True)
class DeviceTypeDescriptor:
    type_id: str
    domain: str
    authoring_schema: AuthoringSchema
    capabilities: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    factory: Callable[[InstallationFactoryContext, str, Mapping[str, Any]], InstalledLeaf] | None = None
    world_config: Callable[[Mapping[str, Any]], object] | None = None
    discover: Callable[[], tuple[DeviceInstanceConfig, ...]] | None = None

    def __post_init__(self) -> None:
        if not self.type_id or not self.domain:
            raise ValueError("device type requires type_id and domain")
        if not isinstance(self.authoring_schema, AuthoringSchema):
            raise TypeError("authoring_schema must be AuthoringSchema")
        capabilities = tuple(self.capabilities)
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("device capabilities must be unique")
        unknown = set(capabilities) - set(CAPABILITY_TYPES)
        if unknown:
            raise ValueError(f"device type uses unknown capability tokens: {sorted(unknown)}")
        dependencies = tuple(self.dependencies)
        if self.type_id in dependencies:
            raise ValueError("device type cannot depend on itself")
        if self.factory is None or not callable(self.factory):
            raise TypeError("device type factory must be callable")
        if self.world_config is not None and not callable(self.world_config):
            raise TypeError("device type world_config must be callable or None")
        if self.discover is not None and not callable(self.discover):
            raise TypeError("device type discover must be callable or None")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "dependencies", dependencies)


__all__ = ["CAPABILITY_TYPES", "DeviceTypeDescriptor", "InstallationFactoryContext", "InstalledLeaf"]
