"""The virtual RF source this bench can install."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.rf.contract import (
    WINDOW_AUTHORING_FIELDS,
    validate_window_values,
)
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf


#: The virtual brick reuses the REAL RF policy fields: an absent edge and an
#: authored edge mean exactly the same thing on simulation and hardware.
VIRTUAL_RF_SCHEMA = AuthoringSchema(
    (
        AuthoringField("serial", "int", "Serial number", 1001, minimum=1),
        *WINDOW_AUTHORING_FIELDS,
    ),
    validator=validate_window_values,
)


def _rf_factory(context, key: str, values: dict) -> InstalledLeaf:
    from zlc_atom.devices.rf.binding import bind_rf_source
    from zlc_atom.devices.rf.vaunix_lms import VaunixLmsConfig
    from .source import virtual_rf_source

    authored = VIRTUAL_RF_SCHEMA.project_values(values)
    config = VaunixLmsConfig(
        serial=int(authored["serial"]),
        frequency_low_hz=authored["frequency_low_hz"],
        frequency_high_hz=authored["frequency_high_hz"],
        power_low_dbm=authored["power_low_dbm"],
        power_high_dbm=authored["power_high_dbm"],
    )
    return bind_rf_source(
        context,
        key,
        virtual_rf_source(config),
        f"virtual-rf:{config.serial}",
        "rf.virtual",
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "rf.virtual",
        "rf",
        VIRTUAL_RF_SCHEMA,
        ("rf.source",),
        factory=_rf_factory,
    ),
)

__all__ = ["DEVICE_TYPES", "VIRTUAL_RF_SCHEMA"]
