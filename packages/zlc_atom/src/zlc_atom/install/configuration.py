"""An apparatus, written down.

What is in the lab is not knowledge the code should carry: which camera, which
serial, what exposure, which board at which address, and which one shared
simulation world a virtual apparatus uses.  That belongs in a file the operator
edits and today's session starts from.

The document is deliberately unforgiving about its shape.  A device entry has
exactly four keys and a document exactly three, and anything else is rejected
rather than ignored -- a configuration that silently drops a key it does not
recognise is how an experiment runs all afternoon with a setting that was never
applied.

The former camera-owned simulation fields are deliberately not kept as a
second grammar: an old document is refused with the root field it must add.
What did NOT come with the split is a content digest: an apparatus file is
meant to be edited by hand.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any


__all__ = [
    "DEVICE_ENTRY_KEYS",
    "DOCUMENT_FORMAT",
    "DeviceInstanceConfig",
    "InstallationConfig",
    "load_installation_config",
    "save_installation_config",
]


DOCUMENT_FORMAT = "zlc.installation"
DEVICE_ENTRY_KEYS = frozenset({"instance_id", "role", "type_id", "parameters"})
_DOCUMENT_KEYS = frozenset({"format", "simulation", "devices"})


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{what} must be non-empty text with no surrounding space")
    return value


def _plain(value: object, what: str) -> Any:
    """Reject anything a JSON file cannot hold, instead of mangling it."""

    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {_text(key, f"{what} key"): _plain(item, what) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, what) for item in value]
    raise TypeError(f"{what} cannot hold {type(value).__name__}; a configuration is plain data")


def _freeze_plain(value: object, what: str) -> object:
    """Validate, deep-own, and freeze one installation-configuration value."""

    plain = _plain(value, what)

    def freeze(item: object) -> object:
        if type(item) is dict:
            return MappingProxyType(
                {key: freeze(child) for key, child in item.items()}
            )
        if type(item) is list:
            return tuple(freeze(child) for child in item)
        return item

    return freeze(plain)


@dataclass(frozen=True)
class DeviceInstanceConfig:
    """One named device in an apparatus: what it is, and how it is set up."""

    instance_id: str
    role: str
    type_id: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        instance_id = _text(self.instance_id, "device instance_id")
        if "/" in instance_id:
            raise ValueError("device instance_id cannot contain '/'")
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "role", _text(self.role, "device role"))
        object.__setattr__(self, "type_id", _text(self.type_id, "device type_id"))
        if not isinstance(self.parameters, Mapping):
            raise TypeError("device parameters must be a mapping")
        object.__setattr__(
            self,
            "parameters",
            _freeze_plain(dict(self.parameters), "device parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "role": self.role,
            "type_id": self.type_id,
            "parameters": _plain(self.parameters, "device parameters"),
        }

    @classmethod
    def from_dict(cls, value: object) -> "DeviceInstanceConfig":
        if not isinstance(value, Mapping) or set(value) != DEVICE_ENTRY_KEYS:
            raise ValueError(
                f"a device entry has exactly {sorted(DEVICE_ENTRY_KEYS)}, "
                f"got {sorted(value) if isinstance(value, Mapping) else type(value).__name__}"
            )
        return cls(
            instance_id=value["instance_id"],
            role=value["role"],
            type_id=value["type_id"],
            parameters=value["parameters"],
        )


@dataclass(frozen=True)
class InstallationConfig:
    """Every device and the one shared simulation config in an apparatus."""

    devices: tuple[DeviceInstanceConfig, ...]
    simulation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        devices = tuple(self.devices)
        if not all(isinstance(item, DeviceInstanceConfig) for item in devices):
            raise TypeError("devices must all be DeviceInstanceConfig")
        for what, seen in (("instance_id", "instance_id"), ("role", "role")):
            values = [getattr(item, seen) for item in devices]
            duplicates = {value for value in values if values.count(value) > 1}
            if duplicates:
                raise ValueError(f"duplicate device {what}: {sorted(duplicates)}")
        object.__setattr__(self, "devices", devices)
        if not isinstance(self.simulation, Mapping):
            raise TypeError("installation simulation must be a mapping")
        object.__setattr__(
            self,
            "simulation",
            _freeze_plain(dict(self.simulation), "installation simulation"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": DOCUMENT_FORMAT,
            "simulation": _plain(self.simulation, "installation simulation"),
            "devices": [item.to_dict() for item in self.devices],
        }

    @classmethod
    def from_dict(cls, value: object) -> "InstallationConfig":
        if not isinstance(value, Mapping):
            raise TypeError("installation document must be a mapping")
        if value.get("format") != DOCUMENT_FORMAT:
            raise ValueError(f"unknown installation format {value.get('format')!r}")
        if set(value) != _DOCUMENT_KEYS:
            raise ValueError(f"an installation document has exactly {sorted(_DOCUMENT_KEYS)}")
        devices = value["devices"]
        if not isinstance(devices, Sequence) or isinstance(devices, (str, bytes)):
            raise TypeError("installation devices must be a list")
        return cls(
            tuple(DeviceInstanceConfig.from_dict(item) for item in devices),
            simulation=value["simulation"],
        )

    def specs(self) -> tuple[dict[str, Any], ...]:
        """The device specs for ``create_installation``; simulation is separate."""

        return tuple(
            {
                "key": item.instance_id,
                "type_id": item.type_id,
                "config": _plain(item.parameters, "device parameters"),
            }
            for item in self.devices
        )


def save_installation_config(config: InstallationConfig, path: str | os.PathLike[str]) -> Path:
    """Write the apparatus atomically, so a crash cannot leave half a file."""

    from zlc_durable import atomic_write_text

    target = Path(path)
    if not target.parent.is_dir():
        raise NotADirectoryError(f"directory does not exist: {target.parent}")
    document = json.dumps(config.to_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    atomic_write_text(target, document)
    return target


def load_installation_config(path: str | os.PathLike[str]) -> InstallationConfig:
    """Read an apparatus back, refusing anything malformed.

    Duplicate JSON keys and NaN are rejected outright: both are ways a file can
    look valid and mean something other than what it says.
    """

    def _no_duplicates(pairs):
        seen: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                raise ValueError(f"duplicate key {key!r} in installation document")
            seen[key] = value
        return seen

    def _no_constants(name):
        raise ValueError(f"installation document contains {name}")

    text = Path(path).read_text(encoding="utf-8")
    return InstallationConfig.from_dict(
        json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_no_constants)
    )
