"""The locally served pulse board this bench can install."""

from __future__ import annotations

from dataclasses import replace

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices.sequencer.binding import bind_sequencer, open_sequencer_control
from zlc_atom.devices.sequencer.device import SequencerDevice
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf
from zlc_pulse import DEFAULT_REQUEST_TIMEOUT


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
        AuthoringField("config_file", "str", "Config file (optional)", ""),
    )
)


def _local_factory(context, key: str, values: dict) -> InstalledLeaf:
    """Open the plugged-in board, serve it to this machine, and join as the
    loopback client.  The server admits a peer only once the device is
    published: ``admit_peers`` on the leaf is the bench's switch for that.
    """

    from zlc_pulse import LocalPulseService

    authored = LOCAL_SEQUENCER_SCHEMA.project_values(values)
    dial = getattr(context, "connect_pulse", None)
    if not callable(dial):
        raise TypeError(
            "sequencer.local needs a way to dial its own server: pass "
            "connect_pulse to create_installation (the composition root owns "
            "the client; this package owns the board and the server)"
        )
    service = LocalPulseService(
        backend=str(authored["backend"]),
        uart_port=str(authored["uart_port"]).strip() or None,
        port=int(authored["port"]),
        peers=False,
    )
    device = None
    try:
        streamer = dial(
            "127.0.0.1", service.port, request_timeout=DEFAULT_REQUEST_TIMEOUT
        )
        device = SequencerDevice(streamer)
        device.open()
        leaf = bind_sequencer(
            context, key, device, f"sequencer:{key}", "sequencer.local",
            config_file=authored["config_file"],
        )
    except BaseException:
        # The loopback client before its server, the order the closer keeps.
        try:
            if device is not None:
                device.close()
        finally:
            service.close()
        raise

    def _close(device=device, service=service) -> None:
        try:
            device.close()
        finally:
            service.close()

    return replace(leaf, closer=_close, admit_peers=service.admit_peers)


def _announce_local(parameters) -> tuple[str, dict]:
    """A peer reaches this board as an ordinary hardware client."""

    return "sequencer.hardware", {
        "host": "127.0.0.1",
        "port": int(parameters["port"]),
    }


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "sequencer.local",
        "sequencer",
        LOCAL_SEQUENCER_SCHEMA,
        ("sequencer.streamer",),
        factory=_local_factory,
        control_factory=open_sequencer_control,
        announce=_announce_local,
        log_channels=("zlc_pulse.remote",),
    ),
)

__all__ = ["DEVICE_TYPES", "LOCAL_SEQUENCER_SCHEMA"]
