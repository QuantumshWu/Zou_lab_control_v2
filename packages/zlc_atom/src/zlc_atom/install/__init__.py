"""Installation graph and automatic device-type discovery."""

from .descriptors import (
    CAPABILITY_TYPES,
    DeviceTypeDescriptor,
    InstallationFactoryContext,
    InstalledLeaf,
)
from .discovery import (
    DeviceCatalogSnapshot,
    UnavailableDeviceTypes,
    discover_device_catalog,
)
from .graph import (
    DeviceSpec,
    Installation,
    InstallationBlueprint,
    InstallationCompositionError,
    InstallationRecovery,
    create_installation,
    preflight_installation,
    tunable_devices,
)
from .templates import installation_config_from_template, installation_template_names

__all__ = [
    "CAPABILITY_TYPES",
    "DeviceSpec",
    "DeviceCatalogSnapshot",
    "DeviceTypeDescriptor",
    "Installation",
    "InstallationBlueprint",
    "InstallationCompositionError",
    "InstallationRecovery",
    "InstallationFactoryContext",
    "InstalledLeaf",
    "UnavailableDeviceTypes",
    "create_installation",
    "preflight_installation",
    "tunable_devices",
    "discover_device_catalog",
    "installation_config_from_template",
    "installation_template_names",
]
