"""Automatically discovered RF source device types."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.rf.binding import bind_rf_source
from zlc_atom.devices.rf.rigol_dg4000 import RigolDg4000Config, RigolDg4000RfSource
from zlc_atom.devices.rf.vaunix_lms import (
    CtypesLmsLibrary,
    VaunixLmsConfig,
    VaunixLmsRfSource,
)
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

#: Bounds are AUTHORED because they are bench policy, not instrument fact:
#: the form has to offer a finite scan range before anything is open, and
#: the window an experiment allows is usually narrower than the datasheet's.
RIGOL_DG4000_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "resource",
            "str",
            "VISA resource",
            "",
            required=True,
        ),
        AuthoringField(
            "timeout_seconds",
            "float",
            "VISA timeout (s)",
            5.0,
            minimum=0.1,
            unit="s",
        ),
    )
)

VAUNIX_LMS_SCHEMA = AuthoringSchema(
    (
        # No serial yet is a VACANCY, not the number zero: a default that
        # violates its own minimum poisons every draft projection.
        AuthoringField(
            "serial", "int", "Serial number", None, required=True, minimum=1
        ),
    )
)


def _rigol_factory(context, key: str, values: dict) -> InstalledLeaf:
    authored = RIGOL_DG4000_SCHEMA.project_values(values)
    config = RigolDg4000Config(
        resource=str(authored["resource"]),
        timeout_seconds=float(authored["timeout_seconds"]),
    )
    source = RigolDg4000RfSource(config)
    return bind_rf_source(
        context,
        key,
        source,
        f"rigol-dg4000:{config.resource}",
        "rf.rigol_dg4000",
    )


def _vaunix_factory(context, key: str, values: dict) -> InstalledLeaf:
    authored = VAUNIX_LMS_SCHEMA.project_values(values)
    config = VaunixLmsConfig(
        serial=int(authored["serial"]),
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
        "rf.rigol_dg4000",
        "rf",
        RIGOL_DG4000_SCHEMA,
        ("rf.source",),
        factory=_rigol_factory,
    ),
    DeviceTypeDescriptor(
        "rf.vaunix_lms",
        "rf",
        VAUNIX_LMS_SCHEMA,
        ("rf.source",),
        factory=_vaunix_factory,
        discover=_discover_vaunix,
    ),
)

__all__ = [
    "DEVICE_TYPES",
    "RIGOL_DG4000_SCHEMA",
    "VAUNIX_LMS_SCHEMA",
]
