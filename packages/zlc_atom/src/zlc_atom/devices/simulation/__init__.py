"""The simulated world every virtual device stands in.

Each virtual device is a folder beside this module, discovered like any
other device; what they SHARE is the world and the one schema that builds
it.  Nothing here imports those folders: deleting one deletes its devices
and leaves the world, and the bench, intact.
"""

from .world import (
    DEFAULT_MOT_FIELD_OPTIMUM_DAC,
    DEFAULT_SIMULATION_GRID_SHAPE_YX,
    DEFAULT_SIMULATION_IMAGE_SHAPE_YX,
    DEFAULT_SIMULATION_SITE_SPACING_PIXELS,
    DEFAULT_SIMULATION_SLM_SHAPE_YX,
    SimulationGeometry,
    SimulationWorld,
    SimulationWorldConfig,
)

__all__ = [
    "DEFAULT_MOT_FIELD_OPTIMUM_DAC",
    "DEFAULT_SIMULATION_GRID_SHAPE_YX",
    "DEFAULT_SIMULATION_IMAGE_SHAPE_YX",
    "DEFAULT_SIMULATION_SITE_SPACING_PIXELS",
    "DEFAULT_SIMULATION_SLM_SHAPE_YX",
    "SimulationGeometry",
    "SimulationWorld",
    "SimulationWorldConfig",
]
