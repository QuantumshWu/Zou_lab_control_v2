"""Canonical capability declarations shared by installation and execution."""

from __future__ import annotations

from zlc_atom.devices.camera.contract import CameraAdapter, CameraWorkingPoint
from zlc_atom.devices.sequencer import SequencerDevice


CAPABILITY_TYPES: dict[str, type] = {
    "camera.adapter": CameraAdapter,
    "camera.working_point": CameraWorkingPoint,
    "sequencer.streamer": SequencerDevice,
}


__all__ = ["CAPABILITY_TYPES"]
