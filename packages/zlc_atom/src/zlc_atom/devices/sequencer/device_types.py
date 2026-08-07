"""Automatically discovered sequencer device types."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.sequencer.protocol import PulseStreamer
from zlc_atom.devices.sequencer.virtual import SequencerDevice, VirtualSequencer
from zlc_atom.execution import DeviceIdentityEvidenceKind, PhysicalDeviceIdentity, ResourceKey, SafetyOperation, bind_verified_device
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf
from zlc_atom.devices.camera.world import SimulationWorld


#: The virtual sequencer needs nothing said about it: it has no endpoint.
VIRTUAL_SEQUENCER_SCHEMA = AuthoringSchema(())

#: A real board is reached over the network, by the pulse server that owns it.
#: Writing the endpoint down is what lets an apparatus configuration be saved
#: and reopened tomorrow; demanding an injected connection object meant it never
#: could be, because a live socket is not something a JSON file can hold.
HARDWARE_SEQUENCER_SCHEMA = AuthoringSchema(
    (
        AuthoringField("host", "str", "Pulse server host", "127.0.0.1"),
        AuthoringField("port", "int", "Pulse server port", 18861, minimum=1, maximum=65535),
        AuthoringField("request_timeout", "float", "Request timeout seconds", 30.0, minimum=0.1),
    )
)


def _bind(context, key: str, device: SequencerDevice, identity: str, type_id: str) -> InstalledLeaf:
    binding, proof = bind_verified_device(
        context.broker,
        key=ResourceKey.parse(f"device/{key}"),
        identity_probe=lambda: PhysicalDeviceIdentity(identity, DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT),
        execute_command=lambda command: device.fire(forever=bool(command)) if command is not None else device.fire(),
        capability_probe=lambda: {"sequencer.streamer": device},
        close_session=lambda _command: device.close(),
        interrupt_operations={SafetyOperation.STOP: lambda: device.close()},
    )
    return InstalledLeaf(key, type_id, device, dict(proof.snapshot), binding=binding, closer=device.close)


def _virtual_factory(context, key: str, values: dict) -> InstalledLeaf:
    del values
    if not isinstance(context.world, SimulationWorld):
        raise TypeError("sequencer.virtual requires the installation SimulationWorld")
    world = context.world
    device = VirtualSequencer(world=world)
    device.open()
    return _bind(context, key, device, f"virtual-sequencer:{key}", "sequencer.virtual")


def _hardware_factory(context, key: str, values: dict) -> InstalledLeaf:
    """Reach the real board, from a written-down endpoint or an injected one.

    An already-open streamer may be supplied (a test, or a session that opened
    the connection itself).  Otherwise the endpoint in the configuration is
    dialled -- which is the whole point of writing it down.
    """

    streamer = values.get("streamer")
    if streamer is None:
        authored = HARDWARE_SEQUENCER_SCHEMA.freeze(
            {name: value for name, value in values.items() if name != "streamer"}
        )
        dial = getattr(context, "connect_pulse", None)
        if not callable(dial):
            raise TypeError(
                "sequencer.hardware needs either an open streamer or a way to "
                "reach one: pass connect_pulse to create_installation (the "
                "composition root owns the client; this package owns only the "
                f"endpoint {authored['host']}:{authored['port']})"
            )
        streamer = dial(
            str(authored["host"]),
            int(authored["port"]),
            request_timeout=float(authored["request_timeout"]),
        )
    if not isinstance(streamer, PulseStreamer):
        raise TypeError("sequencer.hardware needs an object satisfying the PulseStreamer contract")
    device = SequencerDevice(streamer)
    device.open()
    return _bind(context, key, device, f"sequencer:{key}", "sequencer.hardware")


DEVICE_TYPES = (
    DeviceTypeDescriptor("sequencer.hardware", "sequencer", HARDWARE_SEQUENCER_SCHEMA, ("sequencer.streamer",), factory=_hardware_factory),
    DeviceTypeDescriptor("sequencer.virtual", "sequencer", VIRTUAL_SEQUENCER_SCHEMA, ("sequencer.streamer",), factory=_virtual_factory),
)

__all__ = ["DEVICE_TYPES", "HARDWARE_SEQUENCER_SCHEMA", "VIRTUAL_SEQUENCER_SCHEMA"]
