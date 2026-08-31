"""Automatically discovered sequencer device types."""

from __future__ import annotations

from dataclasses import replace

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
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
            request_timeout=DEFAULT_REQUEST_TIMEOUT,
        )
    if not isinstance(streamer, (PulseStreamer, RemotePulseStreamer)):
        raise TypeError("sequencer.hardware needs a zlc_pulse device")
    device = SequencerDevice(streamer)
    device.open()
    return bind_sequencer(context, key, device, f"sequencer:{key}", "sequencer.hardware")


#: The machine the board is plugged into serves it FROM the bench process:
#: no .bat, no second console -- initialize the device and the server is up,
#: narrating on the ``zlc_pulse.remote`` logger where the bench can show it.
#: The bench's own leaf dials the same loopback endpoint every remote client
#: would, so there is exactly one owner of the hardware: the server.
LOCAL_SEQUENCER_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "backend",
            "choice",
            "Board transport",
            "auto",
            choices=(
                AuthoringChoice("auto", "Auto (probe UART, fall back to JTAG)"),
                AuthoringChoice("uart", "UART"),
                AuthoringChoice("jtag-axi", "JTAG-to-AXI (Vivado)"),
                AuthoringChoice("memory", "Memory mock (no hardware)"),
            ),
        ),
        AuthoringField("uart_port", "str", "UART port (blank = probe)", ""),
        AuthoringField("port", "int", "Serve on port", 18861, minimum=1, maximum=65535),
    )
)


def _local_factory(context, key: str, values: dict) -> InstalledLeaf:
    """Open the plugged-in board, serve it, and join as the loopback client."""

    from zlc_pulse import LocalPulseService

    authored = LOCAL_SEQUENCER_SCHEMA.project_values(
        {name: value for name, value in values.items() if name != "streamer"}
    )
    dial = getattr(context, "connect_pulse", None)
    if not callable(dial):
        raise TypeError(
            "sequencer.local needs a way to dial its own server: pass "
            "connect_pulse to create_installation (the composition root owns "
            "the client; this package owns the board and the server)"
        )
    service = LocalPulseService(
        values.get("streamer"),
        backend=str(authored["backend"]),
        uart_port=str(authored["uart_port"]).strip() or None,
        port=int(authored["port"]),
    )
    try:
        streamer = dial(
            "127.0.0.1", service.port, request_timeout=DEFAULT_REQUEST_TIMEOUT
        )
        device = SequencerDevice(streamer)
        device.open()
        leaf = bind_sequencer(
            context, key, device, f"sequencer:{key}", "sequencer.local"
        )
    except BaseException:
        service.close()
        raise

    def _close(device=device, service=service) -> None:
        try:
            device.close()
        finally:
            service.close()

    return replace(leaf, closer=_close)


def _announce_local(parameters) -> tuple[str, dict]:
    """A peer reaches this board as an ordinary hardware client."""

    return "sequencer.hardware", {
        "host": "127.0.0.1",
        "port": int(parameters["port"]),
    }


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "sequencer.hardware",
        "sequencer",
        HARDWARE_SEQUENCER_SCHEMA,
        ("sequencer.streamer",),
        factory=_hardware_factory,
        control_factory=open_sequencer_control,
    ),
    DeviceTypeDescriptor(
        "sequencer.local",
        "sequencer",
        LOCAL_SEQUENCER_SCHEMA,
        ("sequencer.streamer",),
        factory=_local_factory,
        control_factory=open_sequencer_control,
        announce=_announce_local,
    ),
)

__all__ = ["DEVICE_TYPES", "HARDWARE_SEQUENCER_SCHEMA", "LOCAL_SEQUENCER_SCHEMA"]
