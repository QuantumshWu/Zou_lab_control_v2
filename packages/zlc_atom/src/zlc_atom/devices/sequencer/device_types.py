"""Automatically discovered sequencer device types."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.sequencer.binding import bind_sequencer
from zlc_atom.devices.sequencer.device import SequencerDevice
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf
from zlc_pulse import PulseStreamer, RemotePulseStreamer


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


def _hardware_factory(context, key: str, values: dict) -> InstalledLeaf:
    """Reach the real board, from a written-down endpoint or an injected one.

    An already-open streamer may be supplied (a test, or a session that opened
    the connection itself).  Otherwise the endpoint in the configuration is
    dialled -- which is the whole point of writing it down.
    """

    streamer = values.get("streamer")
    if streamer is None:
        authored = HARDWARE_SEQUENCER_SCHEMA.project_values(
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
    if not isinstance(streamer, (PulseStreamer, RemotePulseStreamer)):
        raise TypeError("sequencer.hardware needs a zlc_pulse device")
    device = SequencerDevice(streamer)
    device.open()
    return bind_sequencer(context, key, device, f"sequencer:{key}", "sequencer.hardware")


DEVICE_TYPES = (
    DeviceTypeDescriptor("sequencer.hardware", "sequencer", HARDWARE_SEQUENCER_SCHEMA, ("sequencer.streamer",), factory=_hardware_factory),
)

__all__ = ["DEVICE_TYPES", "HARDWARE_SEQUENCER_SCHEMA"]
