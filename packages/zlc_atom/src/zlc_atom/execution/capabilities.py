"""Canonical capability declarations shared by installation and execution."""

from __future__ import annotations

from zlc_atom.devices.camera.contract import CameraAdapter, CameraWorkingPoint
from zlc_atom.devices.sequencer.protocol import PulseStreamer
from zlc_atom.devices.camera.world import SimulationWorld


CAPABILITY_TYPES: dict[str, type] = {
    "camera.adapter": CameraAdapter,
    "camera.working_point": CameraWorkingPoint,
    "sequencer.streamer": PulseStreamer,
    "simulation.world": SimulationWorld,
}


__all__ = ["CAPABILITY_TYPES"]
