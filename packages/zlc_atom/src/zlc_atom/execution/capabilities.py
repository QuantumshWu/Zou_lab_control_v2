"""Canonical capability declarations shared by installation and execution."""

from __future__ import annotations

from zlc_atom.devices.camera.contract import CameraAdapter
from zlc_atom.devices.sequencer import SequencerDevice
from zlc_atom.devices.slm import SlmAdapter


CAPABILITY_TYPES: dict[str, type] = {
    "camera.adapter": CameraAdapter,
    "sequencer.streamer": SequencerDevice,
    "slm.phase": SlmAdapter,
}


__all__ = ["CAPABILITY_TYPES"]
