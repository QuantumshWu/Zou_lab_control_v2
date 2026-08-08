"""Rglob-based device type discovery.

A lab machine is normally missing at least one vendor runtime -- the DCAM SDK
is not installed on the bench this is written on -- so a device family that
cannot import is the ordinary case, not the exceptional one.  Discovery used
to import every module bare: one missing SDK and the apparatus editor could
not open AT ALL, with a traceback about a driver in place of the window you
open to fix your configuration.

So a family that will not import is REPORTED rather than raised.  The editor
offers it greyed with the reason beside it, which is the difference between
"this bench cannot use a qCMOS today" and "the device manager is broken".
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from .descriptors import DeviceTypeDescriptor


@dataclass(frozen=True)
class UnavailableDeviceTypes:
    """One device family this machine cannot offer, and why."""

    module: str
    reason: str

    @property
    def family(self) -> str:
        """The part a person recognises: 'camera', 'sequencer'."""

        parts = self.module.split(".")
        return parts[-2] if len(parts) > 1 else self.module


@dataclass(frozen=True)
class DeviceCatalogSnapshot:
    """One indivisible view of what this Python process can and cannot build."""

    available: tuple[DeviceTypeDescriptor, ...]
    unavailable: tuple[UnavailableDeviceTypes, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "available", tuple(self.available))
        object.__setattr__(self, "unavailable", tuple(self.unavailable))


def _modules() -> tuple[str, ...]:
    root = Path(__file__).resolve().parents[1] / "devices"
    names: list[str] = []
    for path in root.rglob("device_types.py"):
        relative = path.relative_to(root).with_suffix("")
        names.append("zlc_atom.devices." + ".".join(relative.parts))
    return tuple(sorted(names))


def _walk() -> tuple[tuple[DeviceTypeDescriptor, ...], tuple[UnavailableDeviceTypes, ...]]:
    values: list[DeviceTypeDescriptor] = []
    missing: list[UnavailableDeviceTypes] = []
    for module_name in _modules():
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            # Any failure, not just ImportError: a vendor package that imports
            # but cannot find its DLL raises whatever it likes, and the editor
            # must open either way.
            missing.append(
                UnavailableDeviceTypes(module_name, f"{type(error).__name__}: {error}")
            )
            continue
        descriptor = getattr(module, "DEVICE_TYPE", None)
        descriptors = getattr(module, "DEVICE_TYPES", None)
        candidates = tuple(descriptors) if descriptors is not None else (descriptor,)
        if any(not isinstance(value, DeviceTypeDescriptor) for value in candidates):
            raise TypeError(f"{module_name} must export DeviceTypeDescriptor values")
        values.extend(candidates)
    by_id = tuple(value.type_id for value in values)
    if len(set(by_id)) != len(by_id):
        raise ValueError("device type ids must be unique")
    return (
        tuple(sorted(values, key=lambda value: value.type_id)),
        tuple(missing),
    )


def discover_device_catalog() -> DeviceCatalogSnapshot:
    """Discover available and unavailable device families in one filesystem walk."""

    available, unavailable = _walk()
    return DeviceCatalogSnapshot(available, unavailable)


__all__ = [
    "DeviceCatalogSnapshot",
    "UnavailableDeviceTypes",
    "discover_device_catalog",
]
