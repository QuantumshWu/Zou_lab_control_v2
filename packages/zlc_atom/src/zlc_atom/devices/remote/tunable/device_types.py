"""Automatically discovered remote devices, through the fabric.

"Scan hardware" on PC1 runs this family's ``discover`` beside the vendor
SDK scans: one broadcast finds every announcer on the subnet, one TCP call
each, all at once, lists what they publish, and every record becomes a
one-click add.

A record whose origin family has its own server (the pulse streamer, the
SLM) comes back as THAT family's type with its endpoint parameters
pre-filled -- connecting to it is the existing client doing what it always
did, minus the typing.  A record served by the fabric's generic tunable
plane becomes a ``remote.tunable`` device: the tunable quartet over the
wire, which is all a scan axis or a control panel ever asked of it.
"""

from __future__ import annotations

import ipaddress
import os
import threading
import time

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.remote.fabric import (
    DEFAULT_FABRIC_PORT,
    FABRIC_TUNABLE_TYPE,
    RemoteTunableDevice,
    discover_announcers,
    list_remote_devices,
)
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf, bind_leaf
from zlc_pulse.endpoint import dialled_address

#: Peers a broadcast cannot reach (a different subnet), named once here
#: instead of per device: comma-separated hostnames or addresses.
FABRIC_PEERS_ENVIRONMENT = "ZLC_FABRIC_PEERS"

#: How long the announcers a broadcast found may take to list, all asked at
#: once.  Inside the scan's deadline for the whole family, which drops every
#: device the family found when it is missed: asked one after another, two
#: announcers that answered the broadcast and then stalled cost a request
#: timeout each, and every live announcer's devices went with them.  Counted
#: from before the broadcast: a named peer's name is resolved as it is
#: probed, seconds for one no resolver answers, and that time comes out of
#: the listing instead of going on top of it.  Names that take all of it
#: leave the announcers no time to list, and the scan says so of each.
_LISTING_DEADLINE_SECONDS = 15.0

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
    authored = REMOTE_TUNABLE_SCHEMA.project_values(values)
    device = RemoteTunableDevice(
        host=str(authored["host"]),
        port=int(authored["port"]),
        instance_id=str(authored["instance_id"]),
    )
    # Named by the machine it reaches, not by how the host is written: one
    # published knob reached as PC2, pc2 and its address is one knob, and
    # the broker refuses a second leaf on it by name.  The device has just
    # dialled that host, so the name is resolved already.
    return bind_leaf(
        context,
        key,
        FABRIC_TUNABLE_TYPE,
        device,
        f"fabric:{dialled_address(str(authored['host']))}:{authored['port']}"
        f"/{authored['instance_id']}",
        None,
    )


def _fabric_peers() -> tuple[str, ...]:
    named = os.environ.get(FABRIC_PEERS_ENVIRONMENT, "")
    return tuple(
        peer.strip() for peer in named.split(",") if peer.strip()
    )


def _discover_fabric() -> tuple[tuple[DeviceInstanceConfig, ...], tuple[str, ...]]:
    """Everything every reachable announcer publishes, as one-click adds.

    And, by name, every named peer that could not be probed or was slow to
    resolve, and every announcer that answered the broadcast but could not
    be listed: of another version, or gone, refused or silent by the time
    its list was asked for.  One such announcer used to end the scan, and
    every other announcer's devices with it.  A device another announcer
    already offers by the same name is named too, not offered: two benches
    that each published their default ``rf`` showed as one card, and the
    second could not be added, with no line saying why.  The one offered
    is the lowest address's.
    """

    # Before any named peer's name is resolved: see the deadline's note.
    deadline = time.monotonic() + _LISTING_DEADLINE_SECONDS
    announcers, notes = discover_announcers(extra_hosts=_fabric_peers())
    notes = list(notes)
    # Probing named peers took the whole window (the notes name each slow or
    # unresolvable one): an announcer not listed then was never given time.
    probing_took_all = time.monotonic() >= deadline
    answers: dict[tuple[str, int], object] = {}
    answers_lock = threading.Lock()

    def ask(announcer: tuple[str, int]) -> None:
        try:
            answer: object = list_remote_devices(*announcer)
        except Exception as error:  # noqa: BLE001 -- reported per announcer
            answer = error
        with answers_lock:
            answers[announcer] = answer

    # A daemon each: one that misses the deadline is left to end on its
    # own, because a call to a silent peer cannot be cancelled.
    threads = [
        threading.Thread(
            target=ask,
            args=(announcer,),
            name=f"zlc-fabric-list-{announcer[0]}:{announcer[1]}",
            daemon=True,
        )
        for announcer in announcers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    with answers_lock:
        listed = dict(answers)
    entries: list[DeviceInstanceConfig] = []
    offered: dict[str, str] = {}
    # Lowest address first, so a name two announcers publish is offered from
    # the same one on every scan: the order the broadcast answers arrived in
    # changes from one scan to the next, and the one card with it.
    for host, port in sorted(
        announcers,
        key=lambda announcer: (ipaddress.ip_address(announcer[0]), announcer[1]),
    ):
        if (host, port) not in listed:
            notes.append(
                f"the announcer at {host}:{port} was not listed: probing the "
                f"named peers took the whole {_LISTING_DEADLINE_SECONDS:g}s"
                if probing_took_all else
                f"the announcer at {host}:{port} did not list within "
                f"{_LISTING_DEADLINE_SECONDS:g}s"
            )
            continue
        records = listed[(host, port)]
        if isinstance(records, Exception):
            notes.append(f"the announcer at {host}:{port} did not list: {records}")
            continue
        for record in records:
            instance = str(record.get("instance_id", ""))
            if not instance:
                continue
            if instance in offered:
                notes.append(
                    f"{instance!r} at {host}:{port} is not offered: the "
                    f"announcer at {offered[instance]} publishes one by that "
                    "name, or is the same machine reached at another address"
                )
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
                offered[instance] = f"{host}:{port}"
                continue
            # A device with its own server: offer the ORIGIN family with
            # its endpoint parameters pre-filled.  Connecting is the
            # existing client doing what it always did, minus the typing.
            # Its server listens on every interface of the machine that
            # just answered here, so the address this bench reached it at
            # is the host -- never one the serving machine guessed for
            # itself, which a second NIC or a rig LAN without a gateway
            # gets wrong.
            parameters = record.get("parameters")
            type_id = str(record.get("type_id", ""))
            if not type_id or not isinstance(parameters, dict):
                continue
            entries.append(
                DeviceInstanceConfig(
                    instance_id=f"remote_{instance}",
                    role=str(record.get("role") or instance),
                    type_id=type_id,
                    parameters={**parameters, "host": host},
                )
            )
            offered[instance] = f"{host}:{port}"
    return tuple(entries), tuple(notes)


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
