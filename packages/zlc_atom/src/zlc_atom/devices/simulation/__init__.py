"""Public API for the installation-owned virtual apparatus."""

from .camera import VirtualCamera, VirtualCameraConfig
from .sequencer import CAMERA_TRIGGER_CHANNEL, VirtualPulseStreamer, VirtualSequencer
from .world import (
    DEFAULT_SIMULATION_GRID_SHAPE_YX,
    DEFAULT_SIMULATION_IMAGE_SHAPE_YX,
    DEFAULT_SIMULATION_SITE_SPACING_PIXELS,
    SimulationGeometry,
    SimulationWorld,
    SimulationWorldConfig,
)

__all__ = [
    "CAMERA_TRIGGER_CHANNEL",
    "DEFAULT_SIMULATION_GRID_SHAPE_YX",
    "DEFAULT_SIMULATION_IMAGE_SHAPE_YX",
    "DEFAULT_SIMULATION_SITE_SPACING_PIXELS",
    "SimulationGeometry",
    "SimulationWorld",
    "SimulationWorldConfig",
    "VirtualCamera",
    "VirtualCameraConfig",
    "VirtualPulseStreamer",
    "VirtualSequencer",
]
