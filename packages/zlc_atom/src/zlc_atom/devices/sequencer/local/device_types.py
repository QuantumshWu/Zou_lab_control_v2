"""The locally served pulse board this bench can install."""

from __future__ import annotations

from dataclasses import replace
import logging

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices.sequencer.binding import (
    bind_sequencer,
    open_sequencer_control,
    pulse_board_identity,
)
from zlc_atom.devices.sequencer.device import SequencerDevice
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf
from zlc_pulse import DEFAULT_PORT, DEFAULT_REQUEST_TIMEOUT


#: Where the in-process server narrates, and where this leaf says what
#: withdrawing it could not do: the board's log is one story.
_NARRATION = "zlc_pulse.remote"
_LOG = logging.getLogger(_NARRATION)

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
        AuthoringField("port", "int", "Serve on port", DEFAULT_PORT, minimum=1, maximum=65535),
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
        leaf = bind_sequencer(
            context, key, device,
            pulse_board_identity("127.0.0.1", service.port),
            "sequencer.local",
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

    def _admit_peers(admitted: bool, device=device, service=service) -> None:
        service.admit_peers(admitted)
        if admitted:
            return
        # The board goes to whichever client connected last, so a peer that
        # used the published board took it from this machine's client, and
        # withdrawing has just dropped that peer.  The loopback client only
        # learns it lost the board on its next request; ask, and join again
        # if so, or this bench could not drive its own board until the
        # device was initialised again.  An untouched loopback owner answers
        # and keeps whatever it is running.  Best effort: the door is already
        # shut, and a rejoin the board refuses (its takeover SAFE failed, the
        # server faulted) must not stop the withdrawal that asked for it.
        # Nor is it quick on a board that stopped answering: the rejoin
        # queues behind the dropped peer's AUTO-SAFE and then SAFEs twice
        # itself, each up to its deadline -- the bench's device worker waits
        # it out, never its GUI thread.
        try:
            try:
                device.snapshot()
            except ConnectionError:
                device.open()
        except Exception as error:  # noqa: BLE001 -- told, not raised
            _LOG.warning(
                "LOCAL CLIENT NOT REJOINED error=%s: %s -- initialize the "
                "devices again to drive this board from here",
                type(error).__name__,
                str(error).replace(chr(10), " "),
            )

    return replace(leaf, closer=_close, admit_peers=_admit_peers)


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
        log_channels=(_NARRATION,),
    ),
)

__all__ = ["DEVICE_TYPES", "LOCAL_SEQUENCER_SCHEMA"]
