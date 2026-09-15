"""The virtual pulse sequencer this bench can install."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.devices.sequencer.binding import bind_sequencer, open_sequencer_control
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from ..authoring import simulation_world_config
from ..world import SimulationWorld
from .device import VirtualSequencer


# A virtual sequencer has no physical endpoint or device-local parameters.
VIRTUAL_SEQUENCER_SCHEMA = AuthoringSchema(())


def _sequencer_factory(context, key: str, values: dict) -> InstalledLeaf:
    VIRTUAL_SEQUENCER_SCHEMA.project_values(values)
    if not isinstance(context.world, SimulationWorld):
        raise TypeError("sequencer.virtual requires the installation SimulationWorld")
    device = VirtualSequencer(world=context.world)
    device.open()
    return bind_sequencer(
        context,
        key,
        device,
        f"virtual-sequencer:{key}",
        "sequencer.virtual",
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "sequencer.virtual",
        "sequencer",
        VIRTUAL_SEQUENCER_SCHEMA,
        ("sequencer.streamer",),
        factory=_sequencer_factory,
        control_factory=open_sequencer_control,
        world_config=simulation_world_config,
    ),
)

__all__ = ["DEVICE_TYPES", "VIRTUAL_SEQUENCER_SCHEMA"]
