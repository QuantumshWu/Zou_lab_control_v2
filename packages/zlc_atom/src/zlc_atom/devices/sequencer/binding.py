"""Installation binding shared by physical and simulated sequencers."""

from __future__ import annotations

from zlc_atom.devices.sequencer.device import SequencerDevice
from zlc_atom.execution import (
    PhysicalDeviceIdentity,
    ResourceKey,
    bind_verified_device,
)
from zlc_atom.install.descriptors import InstalledLeaf
from zlc_pulse.endpoint import dialled_address


def pulse_board_identity(host: str, port: int) -> str:
    """The board a pulse server at ``host:port`` serves, as the broker names it.

    A server owns one board, so its endpoint IS that board: two leaves
    dialling one server command one board, whatever each is called.  The
    host is named by the machine it reaches (``dialled_address``): every
    spelling of this machine is this machine, and a name is the address it
    resolves to.  Asked once per leaf, at Init.
    """

    return f"pulse-server:{dialled_address(host)}:{int(port)}"


def bind_sequencer(
    context,
    key: str,
    device: SequencerDevice,
    identity: str,
    type_id: str,
    *,
    config_file: str = "",
) -> InstalledLeaf:
    """Bind ``device`` as leaf ``key``, claim its board, and only then open it.

    OPENING a pulse board takes it: its server hands the board to whichever
    client commands it last, and SAFEs it on the way.  Claimed at admission,
    as other devices are, a second leaf naming a board another leaf holds
    had SAFEd the owner's run -- a Logic running on it -- by the time it was
    refused.  So the board is claimed before the first command goes out,
    and a duplicate is refused here without having touched it.  (Of two such
    leaves in one Init, which one is refused is then the order they reach
    the broker in; either way the board is never taken from the other.)
    The caller still owns ``device``, and closes it if this raises.
    """

    if not isinstance(device, SequencerDevice):
        raise TypeError("sequencer must use the canonical SequencerDevice")
    if config_file.strip():
        device.load_config_file(config_file.strip())
    binding, proof = bind_verified_device(
        context.broker,
        key=ResourceKey.parse(f"device/{key}"),
        identity_probe=lambda: PhysicalDeviceIdentity(identity),
        capability_probe=lambda: {"sequencer.streamer": device},
    )
    try:
        context.broker.claim(binding)
        device.open()
    except BaseException:
        # Nothing was opened for this binding to stand for: a refused claim
        # never sent a command, and a failed open has disconnected.
        context.broker.unbind(binding)
        raise
    return InstalledLeaf(
        key,
        type_id,
        device,
        dict(proof.snapshot),
        binding=binding,
        closer=device.close,
    )


def open_sequencer_control(session, device_key: str, window_ratio=None, render=None):
    """Open PulseGUI for one named sequencer in an existing experiment.

    ``render`` is the application's Edit/Save render child, where the
    preview is drawn; without one the editor starts its own.
    """

    from zlc_workbench.apps.pulse_editor import create_bound_window

    return create_bound_window(
        workspace=session.workspace,
        sequence=None,
        sequencer=session.installation.device(str(device_key)),
        device_use=session.device_use,
        device_label=session.device_labels.get(str(device_key), str(device_key)),
        path="",
        window_ratio=window_ratio,
        render=render,
    )


__all__ = ["bind_sequencer", "open_sequencer_control", "pulse_board_identity"]
