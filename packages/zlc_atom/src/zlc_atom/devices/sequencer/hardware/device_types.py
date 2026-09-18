"""The networked pulse board this bench can install."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.sequencer.binding import bind_sequencer, open_sequencer_control
from zlc_atom.devices.sequencer.device import SequencerDevice
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf
from zlc_pulse import (
    DEFAULT_REQUEST_TIMEOUT,
    PulseStreamer,
    RemotePulseStreamer,
)


#: A real board is reached over the network, by the pulse server that owns it.
#: Writing the endpoint down is what lets an apparatus configuration be saved
#: and reopened tomorrow; demanding an injected connection object meant it never
#: could be, because a live socket is not something a JSON file can hold.
HARDWARE_SEQUENCER_SCHEMA = AuthoringSchema(
    (
        AuthoringField("host", "str", "Pulse server host", "127.0.0.1"),
        AuthoringField("port", "int", "Pulse server port", 18861, minimum=1, maximum=65535),
        AuthoringField("config_file", "str", "Config file (optional)", ""),
    )
)


def _hardware_factory(context, key: str, values: dict) -> InstalledLeaf:
    """Reach the real board at the endpoint the configuration writes down.

    The streamer is this factory's to own from here: on success the leaf's
    closer closes it, and a failure between open and bind -- the broker
    refusing a physical identity that is already bound, say -- closes it
    here, because a device that never became a leaf has nobody else to
    close it.
    """

    authored = HARDWARE_SEQUENCER_SCHEMA.project_values(values)
    dial = getattr(context, "connect_pulse", None)
    if not callable(dial):
        raise TypeError(
            "sequencer.hardware needs a way to reach its board: pass "
            "connect_pulse to create_installation (the composition root owns "
            "the client; this package owns only the "
            f"endpoint {authored['host']}:{authored['port']})"
        )
    streamer = dial(
        str(authored["host"]),
        int(authored["port"]),
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    if not isinstance(streamer, (PulseStreamer, RemotePulseStreamer)):
        raise TypeError("sequencer.hardware needs a zlc_pulse device")
    device = SequencerDevice(streamer)
    try:
        device.open()
        return bind_sequencer(
            context, key, device, f"sequencer:{key}", "sequencer.hardware",
            config_file=authored["config_file"],
        )
    except BaseException as error:
        try:
            device.close()
        except BaseException as close_error:
            error.add_note(
                "closing the streamer also reported: "
                f"{type(close_error).__name__}: {close_error}"
            )
        raise


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "sequencer.hardware",
        "sequencer",
        HARDWARE_SEQUENCER_SCHEMA,
        ("sequencer.streamer",),
        factory=_hardware_factory,
        control_factory=open_sequencer_control,
    ),
)

__all__ = ["DEVICE_TYPES", "HARDWARE_SEQUENCER_SCHEMA"]
