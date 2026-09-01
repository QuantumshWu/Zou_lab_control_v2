"""Automatically discovered remote devices, through the fabric.

"Scan hardware" on PC1 runs this family's ``discover`` beside the vendor
SDK scans: one broadcast finds every announcer on the subnet, one TCP call
each lists what they publish, and every record becomes a one-click add.

A record whose origin family has its own server (the pulse streamer, the
SLM) comes back as THAT family's type with its endpoint parameters
pre-filled -- connecting to it is the existing client doing what it always
did, minus the typing.  A record served by the fabric's generic tunable
plane becomes a ``remote.tunable`` device: the tunable quartet over the
wire, which is all a scan axis or a control panel ever asked of it.
"""

from __future__ import annotations

import os

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.remote.fabric import (
    DEFAULT_FABRIC_PORT,
    RemoteTunableDevice,
    discover_announcers,
    list_remote_devices,
)
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

#: Peers a broadcast cannot reach (a different subnet), named once here
#: instead of per device: comma-separated hostnames or addresses.
FABRIC_TUNABLE_TYPE = "remote.tunable"
FABRIC_PEERS_ENVIRONMENT = "ZLC_FABRIC_PEERS"

REMOTE_TUNABLE_SCHEMA = AuthoringSchema(
    (
        AuthoringField("host", "str", "Fabric host", "", required=True),
        AuthoringField(
            "port",
            "int",
            "Fabric port",
            DEFAULT_FABRIC_PORT,
            minimum=1,
            maximum=65535,
        ),
        AuthoringField(
            "instance_id", "str", "Published instance", "", required=True
        ),
    )
)


def _remote_tunable_factory(context, key: str, values: dict) -> InstalledLeaf:
    from zlc_atom.execution import (
        DeviceIdentityEvidenceKind,
        PhysicalDeviceIdentity,
        ResourceKey,
        bind_verified_device,
    )

    authored = REMOTE_TUNABLE_SCHEMA.project_values(values)
    device = RemoteTunableDevice(
        host=str(authored["host"]),
        port=int(authored["port"]),
        instance_id=str(authored["instance_id"]),
    )
    binding, proof = bind_verified_device(
        context.broker,
        key=ResourceKey.parse(f"device/{key}"),
        identity_probe=lambda: PhysicalDeviceIdentity(
            f"fabric:{authored['host']}:{authored['port']}"
            f"/{authored['instance_id']}",
            DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        ),
        capability_probe=lambda: {},
    )
    return InstalledLeaf(
        key,
        FABRIC_TUNABLE_TYPE,
        device,
        dict(proof.snapshot),
        binding=binding,
        closer=device.close,
    )


def _fabric_peers() -> tuple[str, ...]:
    named = os.environ.get(FABRIC_PEERS_ENVIRONMENT, "")
    return tuple(
        peer.strip() for peer in named.split(",") if peer.strip()
    )


def _discover_fabric() -> tuple[DeviceInstanceConfig, ...]:
    """Everything every reachable announcer publishes, as one-click adds."""

    entries: list[DeviceInstanceConfig] = []
    for host, port in discover_announcers(extra_hosts=_fabric_peers()):
        for record in list_remote_devices(host, port):
            instance = str(record.get("instance_id", ""))
            if not instance:
                continue
            if record.get("tunable"):
                entries.append(
                    DeviceInstanceConfig(
                        instance_id=f"remote_{instance}",
                        role=str(record.get("role") or instance),
                        type_id=FABRIC_TUNABLE_TYPE,
                        parameters=REMOTE_TUNABLE_SCHEMA.project_values(
                            {
                                "host": host,
                                "port": int(port),
                                "instance_id": instance,
                            }
                        ),
                    )
                )
                continue
            # A device with its own server: offer the ORIGIN family with
            # its endpoint parameters pre-filled.  Connecting is the
            # existing client doing what it always did, minus the typing.
            parameters = record.get("parameters")
            type_id = str(record.get("type_id", ""))
            if not type_id or not isinstance(parameters, dict):
                continue
            entries.append(
                DeviceInstanceConfig(
                    instance_id=f"remote_{instance}",
                    role=str(record.get("role") or instance),
                    type_id=type_id,
                    parameters=dict(parameters),
                )
            )
    return tuple(entries)


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        FABRIC_TUNABLE_TYPE,
        "remote",
        REMOTE_TUNABLE_SCHEMA,
        (),
        factory=_remote_tunable_factory,
        addable=False,
        discover=_discover_fabric,
    ),
)

__all__ = [
    "DEVICE_TYPES",
    "FABRIC_PEERS_ENVIRONMENT",
    "REMOTE_TUNABLE_SCHEMA",
]
