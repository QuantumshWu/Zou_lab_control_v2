"""Data-only device type declarations and the capability type table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.execution.capabilities import CAPABILITY_TYPES
from zlc_atom.execution.ports import BoundDevice
from zlc_atom.execution.resources import PhysicalDeviceIdentity
from zlc_atom.install.configuration import DeviceInstanceConfig


@dataclass(frozen=True)
class InstalledLeaf:
    key: str
    type_id: str
    device: object
    capabilities: Mapping[str, object]
    binding: BoundDevice | None = None
    closer: Callable[[], None] | None = None
    #: The factory world this leaf's implementation closes over.  Assigned by
    #: create_installation from descriptor.world_config, never by the family.
    #: None means the leaf is independent physical infrastructure and may be
    #: retained while a simulation world is replaced.
    world_affinity: object | None = None

    @property
    def physical_identity(self) -> PhysicalDeviceIdentity | None:
        """The verified real resource, or None for an unbound synthetic leaf."""

        return None if self.binding is None else self.binding.physical_identity

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
    control_factory: Callable[..., object] | None = None
    #: How a peer bench should author THIS device when it is published on the
    #: fabric: authored parameters -> (peer type_id, peer parameters).  A type
    #: that serves its own protocol (a local pulse board, a local SLM) is not
    #: reachable under its own type_id -- the peer installs the CLIENT type
    #: against this machine's endpoint, and only the family that defines both
    #: sides knows that mapping.  None means the device is announced as
    #: itself, which is right for plain endpoint clients and tunables.
    announce: Callable[[Mapping[str, Any]], tuple[str, dict[str, Any]]] | None = None

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
        if self.control_factory is not None and not callable(self.control_factory):
            raise TypeError("device type control_factory must be callable or None")
        if self.announce is not None and not callable(self.announce):
            raise TypeError("device type announce must be callable or None")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "dependencies", dependencies)


__all__ = ["CAPABILITY_TYPES", "DeviceTypeDescriptor", "InstallationFactoryContext", "InstalledLeaf"]
