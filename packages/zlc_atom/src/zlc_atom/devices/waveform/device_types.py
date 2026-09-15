"""Automatically discovered waveform source device types."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.waveform.binding import bind_waveform_source
from zlc_atom.devices.waveform.tek_scope import (
    TekScopeConfig,
    TekScopeWaveformSource,
    discover_tek_scopes,
)
from zlc_atom.devices.waveform.wheeltec_n100 import (
    DEFAULT_BAUD,
    WheeltecN100Config,
    WheeltecN100WaveformSource,
    discover_n100,
)
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf


#: Which port, at the line rate the module was configured to.  The packet
#: rate is not authored: it is measured off the stream when the port opens.
WHEELTEC_N100_SCHEMA = AuthoringSchema(
    (
        AuthoringField("port", "str", "Serial port", "", required=True),
        AuthoringField("baud", "int", "Baud", DEFAULT_BAUD, minimum=1200),
        AuthoringField(
            "timeout_seconds", "float", "Record timeout (s)", 2.0, minimum=0.05
        ),
    )
)

#: Where the scope is and which channels a record carries, by number.
TEK_SCOPE_SCHEMA = AuthoringSchema(
    (
        AuthoringField("resource", "str", "VISA resource", "", required=True),
        AuthoringField("channels", "str", "Channels (e.g. 1,2)", "1", required=True),
        AuthoringField("timeout_seconds", "float", "Timeout (s)", 5.0, minimum=0.1),
    )
)


def scope_channels(text: object) -> tuple[int, ...]:
    """The channel numbers an operator typed, e.g. ``"1, 3"``."""

    parts = [part.strip() for part in str(text).split(",")]
    try:
        return tuple(int(part) for part in parts if part)
    except ValueError as error:
        raise ValueError(
            f"channels must be channel numbers separated by commas, not {text!r}"
        ) from error


def _n100_factory(context, key: str, values: dict) -> InstalledLeaf:
    authored = WHEELTEC_N100_SCHEMA.project_values(values)
    config = WheeltecN100Config(
        port=str(authored["port"]),
        baud=int(authored["baud"]),
        timeout_seconds=float(authored["timeout_seconds"]),
    )
    source = WheeltecN100WaveformSource(config)
    return bind_waveform_source(
        context, key, source, source.identity, "waveform.wheeltec_n100"
    )


def _tek_scope_factory(context, key: str, values: dict) -> InstalledLeaf:
    authored = TEK_SCOPE_SCHEMA.project_values(values)
    config = TekScopeConfig(
        resource=str(authored["resource"]),
        channels=scope_channels(authored["channels"]),
        timeout_seconds=float(authored["timeout_seconds"]),
    )
    source = TekScopeWaveformSource(config)
    return bind_waveform_source(
        context, key, source, source.identity, "waveform.tek_scope"
    )


def _discover_n100() -> tuple[DeviceInstanceConfig, ...]:
    ports = discover_n100()
    if not ports:
        raise RuntimeError(
            "no serial port carries an FDILink IMU stream: plug the N100 in "
            f"over USB and make sure its line rate is {DEFAULT_BAUD}; a port "
            "another program holds open cannot be listened to"
        )

    def named(port: str) -> str:
        return "n100_" + "".join(c if c.isalnum() else "_" for c in port)

    return tuple(
        DeviceInstanceConfig(
            instance_id=named(port),
            role=named(port),
            type_id="waveform.wheeltec_n100",
            parameters=WHEELTEC_N100_SCHEMA.project_values({"port": port}),
        )
        for port in ports
    )


def _discover_tek_scopes() -> tuple[DeviceInstanceConfig, ...]:
    from zlc_atom.devices.rf.rigol_dg4000 import identity_fields

    def named(resource: str, identity: str) -> str:
        fields = identity_fields(identity)
        tail = fields[2] if len(fields) > 2 and fields[2] else "".join(
            c if c.isalnum() else "_" for c in resource
        )
        return f"scope_{tail}"

    return tuple(
        DeviceInstanceConfig(
            instance_id=named(resource, identity),
            role=named(resource, identity),
            type_id="waveform.tek_scope",
            parameters=TEK_SCOPE_SCHEMA.project_values({"resource": resource}),
        )
        for resource, identity in discover_tek_scopes()
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "waveform.wheeltec_n100",
        "waveform",
        WHEELTEC_N100_SCHEMA,
        ("waveform.source",),
        factory=_n100_factory,
        discover=_discover_n100,
    ),
    DeviceTypeDescriptor(
        "waveform.tek_scope",
        "waveform",
        TEK_SCOPE_SCHEMA,
        ("waveform.source",),
        factory=_tek_scope_factory,
        discover=_discover_tek_scopes,
    ),
)

__all__ = [
    "DEVICE_TYPES",
    "TEK_SCOPE_SCHEMA",
    "WHEELTEC_N100_SCHEMA",
    "scope_channels",
]
