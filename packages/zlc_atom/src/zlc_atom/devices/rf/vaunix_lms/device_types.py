"""The Vaunix Lab Brick this bench can install, and how to find one."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.rf.binding import bind_rf_source
from zlc_atom.devices.rf.contract import (
    WINDOW_AUTHORING_FIELDS,
    validate_window_values,
)
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from .source import CtypesLmsLibrary, VaunixLmsConfig, VaunixLmsRfSource


VAUNIX_LMS_SCHEMA = AuthoringSchema(
    (
        # No serial yet is a VACANCY, not the number zero: a default that
        # violates its own minimum poisons every draft projection.
        AuthoringField(
            "serial", "int", "Serial number", None, required=True, minimum=1
        ),
        *WINDOW_AUTHORING_FIELDS,
    ),
    validator=validate_window_values,
)


def _vaunix_factory(context, key: str, values: dict) -> InstalledLeaf:
    authored = VAUNIX_LMS_SCHEMA.project_values(values)
    config = VaunixLmsConfig(
        serial=int(authored["serial"]),
        frequency_low_hz=authored["frequency_low_hz"],
        frequency_high_hz=authored["frequency_high_hz"],
        power_low_dbm=authored["power_low_dbm"],
        power_high_dbm=authored["power_high_dbm"],
    )
    source = VaunixLmsRfSource(config)
    return bind_rf_source(
        context,
        key,
        source,
        f"vaunix-lms:{config.serial}",
        "rf.vaunix_lms",
    )


def _discover_vaunix() -> tuple[DeviceInstanceConfig, ...]:
    """Every attached Lab Brick, by serial -- a count read, no opens.

    A missing vendor DLL raises the INSTRUCTION (which file, into which
    folder) rather than an empty result: the scan strip is exactly where
    an operator wondering "why no bricks?" is looking.
    """

    from zlc_atom.devices.vendor import resolve_vendor_file

    library = CtypesLmsLibrary(
        resolve_vendor_file(
            __file__, "vnx_fmsynth.dll", what="the Vaunix LMS SDK (64-bit)"
        )
    )
    return tuple(
        DeviceInstanceConfig(
            instance_id=f"lms_{serial}",
            role=f"lms_{serial}",
            type_id="rf.vaunix_lms",
            parameters=VAUNIX_LMS_SCHEMA.project_values({"serial": int(serial)}),
        )
        for serial in library.device_serials()
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "rf.vaunix_lms",
        "rf",
        VAUNIX_LMS_SCHEMA,
        ("rf.source",),
        factory=_vaunix_factory,
        discover=_discover_vaunix,
    ),
)

__all__ = ["DEVICE_TYPES", "VAUNIX_LMS_SCHEMA"]
