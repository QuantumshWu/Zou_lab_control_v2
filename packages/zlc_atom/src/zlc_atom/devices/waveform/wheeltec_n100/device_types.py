"""The WHEELTEC N100 this bench can install, and how to find one."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.waveform.binding import bind_waveform_source
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from .source import (
    DEFAULT_BAUD,
    WheeltecN100Config,
    WheeltecN100WaveformSource,
    discover_n100,
)


#: Which port, at the line rate the module was configured to.  The packet
#: rate is NOT authored: it is a setting that lives in the module's own
#: flash, so authoring it here would be a second place it is written down
#: and a bench that disagreed with its own hardware after a power cycle.
#: It is measured off the stream when the port opens, and moved from Device
#: Control, where the module answers with the rate it actually took.
WHEELTEC_N100_SCHEMA = AuthoringSchema(
    (
        AuthoringField("port", "str", "Serial port", "", required=True),
        AuthoringField("baud", "int", "Baud", DEFAULT_BAUD, minimum=1200),
        AuthoringField(
            "timeout_seconds", "float", "Record timeout (s)", 2.0, minimum=0.05
        ),
    )
)


def _factory(context, key: str, values: dict) -> InstalledLeaf:
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


def _discover() -> tuple[DeviceInstanceConfig, ...]:
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


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "waveform.wheeltec_n100",
        "waveform",
        WHEELTEC_N100_SCHEMA,
        ("waveform.source",),
        factory=_factory,
        discover=_discover,
    ),
)

__all__ = ["DEVICE_TYPES", "WHEELTEC_N100_SCHEMA"]
