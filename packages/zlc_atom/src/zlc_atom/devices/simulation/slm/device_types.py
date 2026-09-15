"""The virtual SLM this bench can install."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.devices.slm import bind_slm, open_slm_control
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from ..authoring import simulation_world_config
from ..world import SimulationWorld
from .device import VirtualSLM


# The virtual panel geometry belongs to its one SimulationWorld.
VIRTUAL_SLM_SCHEMA = AuthoringSchema(())


def _slm_factory(context, key: str, values: dict) -> InstalledLeaf:
    VIRTUAL_SLM_SCHEMA.project_values(values)
    if not isinstance(context.world, SimulationWorld):
        raise TypeError("slm.virtual requires the installation SimulationWorld")
    return bind_slm(
        context,
        key,
        VirtualSLM(context.world, identity="virtual-slm"),
        "slm.virtual",
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "slm.virtual",
        "slm",
        VIRTUAL_SLM_SCHEMA,
        ("slm.phase",),
        factory=_slm_factory,
        control_factory=open_slm_control,
        world_config=simulation_world_config,
    ),
)

__all__ = ["DEVICE_TYPES", "VIRTUAL_SLM_SCHEMA"]
