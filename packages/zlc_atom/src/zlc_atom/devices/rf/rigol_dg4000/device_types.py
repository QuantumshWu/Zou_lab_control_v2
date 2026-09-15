"""The Rigol DG4000 this bench can install, and how to find one."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.rf.binding import bind_rf_source
from zlc_atom.devices.rf.contract import (
    WINDOW_AUTHORING_FIELDS,
    validate_window_values,
)
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

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
    return bind_rf_source(
        context,
        key,
        source,
        f"rigol-dg4000:{config.resource}",
        "rf.rigol_dg4000",
    )


def _discover_rigol() -> tuple[DeviceInstanceConfig, ...]:
    """Every DG4000 that answers on this machine, named by what it answered.

    The serial is the instrument's own, off ``*IDN?``, so unplugging one and
    scanning again offers the same card rather than a differently numbered
    stranger.  When an instrument gives no serial the resource it was found
    at is the name -- still stable, still that instrument, just longer.
    """

    from zlc_atom.devices import visa
    from zlc_atom.devices.visa import PROBED_RESOURCE_PREFIXES

    from .source import discover_dg4000

    # "Found nothing" is only an answer if something was asked.  VISA's own
    # list is far blinder than an operator expects: a LAN instrument appears
    # only once it has been added in NI MAX, and a USB one only once its
    # USB-TMC driver is bound -- so a Rigol sitting there, plugged in and
    # working, can simply not be in the list.  Saying nothing then reports
    # "no Rigol here" about a bench that has one.
    manager = visa.visa_resources()
    listed = tuple(str(name) for name in manager.list_resources())
    probeable = visa.probeable_resources(listed)
    if not probeable:
        raise RuntimeError(
            "VISA lists nothing to ask: no "
            f"{' or '.join(PROBED_RESOURCE_PREFIXES)} resource is registered "
            f"on this machine (it lists: {', '.join(listed) or 'nothing'}). "
            "A LAN instrument has to be added in NI MAX -- or skip that and "
            "type its TCPIP0::<address>::INSTR in by hand, which needs no "
            "install; a USB one is invisible to VISA until a USB-TMC driver "
            "is bound to it, which is what installing NI-VISA (or Rigol "
            "UltraSigma) does."
        )

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
