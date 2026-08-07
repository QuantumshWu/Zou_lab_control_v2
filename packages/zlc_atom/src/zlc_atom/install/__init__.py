"""Installation graph and automatic device-type discovery."""

from .descriptors import (
    CAPABILITY_TYPES,
    DeviceTypeDescriptor,
    InstallationFactoryContext,
    InstalledLeaf,
)
from .discovery import discover_device_types
from .graph import DeviceSpec, Installation, create_installation
from .templates import INSTALLATION_TEMPLATES

__all__ = [
    "CAPABILITY_TYPES",
    "DeviceSpec",
    "DeviceTypeDescriptor",
    "INSTALLATION_TEMPLATES",
    "Installation",
    "InstallationFactoryContext",
    "InstalledLeaf",
    "create_installation",
    "discover_device_types",
]
