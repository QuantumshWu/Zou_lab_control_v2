"""The Rigol DG4000 this bench can install, and how to find one."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.rf.contract import (
    WINDOW_AUTHORING_FIELDS,
    validate_window_values,
)
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf, bind_leaf

from .source import RigolDg4000Config, RigolDg4000RfSource


#: Bounds are optional bench policy, not instrument facts.  Omitting an edge
#: means no policy limit on that side; it can still be set or cleared later in
#: Device Control through the same shared RF contract.
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
        *WINDOW_AUTHORING_FIELDS,
    ),
    validator=validate_window_values,
)


def _rigol_factory(context, key: str, values: dict) -> InstalledLeaf:
    authored = RIGOL_DG4000_SCHEMA.project_values(values)
    config = RigolDg4000Config(
        resource=str(authored["resource"]),
        timeout_seconds=float(authored["timeout_seconds"]),
        frequency_low_hz=authored["frequency_low_hz"],
        frequency_high_hz=authored["frequency_high_hz"],
        power_low_dbm=authored["power_low_dbm"],
        power_high_dbm=authored["power_high_dbm"],
    )
    source = RigolDg4000RfSource(config)
    return bind_leaf(
        context,
        key,
        "rf.rigol_dg4000",
        source,
        f"rigol-dg4000:{config.resource}",
        "rf.source",
    )


def _discover_rigol() -> tuple[DeviceInstanceConfig, ...]:
    """Every DG4000 that answers on this machine, named by what it answered.

    The serial is the instrument's own, off ``*IDN?``, so unplugging one and
    scanning again offers the same card rather than a differently numbered
    stranger.  When an instrument gives no serial the resource it was found
    at is the name -- still stable, still that instrument, just longer.
    A VISA that lists nothing to ask is raised by the bus probe itself.
    """

    from .source import discover_dg4000

    def named(sighting) -> str:
        tail = sighting.serial or "".join(
            character if character.isalnum() else "_"
            for character in sighting.resource
        )
        return f"dg4000_{tail}"

    return tuple(
        DeviceInstanceConfig(
            instance_id=named(sighting),
            role=named(sighting),
            type_id="rf.rigol_dg4000",
            parameters=RIGOL_DG4000_SCHEMA.project_values(
                {"resource": sighting.resource}
            ),
        )
        for sighting in discover_dg4000()
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "rf.rigol_dg4000",
        "rf",
        RIGOL_DG4000_SCHEMA,
        ("rf.source",),
        factory=_rigol_factory,
        discover=_discover_rigol,
    ),
)

__all__ = ["DEVICE_TYPES", "RIGOL_DG4000_SCHEMA"]
