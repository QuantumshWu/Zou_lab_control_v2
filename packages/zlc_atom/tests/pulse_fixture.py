"""The pulse documents the tests run against, and the calibration one compiled.

No pulse document lives in the product: pulses are workspace files the
operator names.  The tests own theirs, here beside them, and read them
through ``pulse_document``.
"""

from __future__ import annotations

import json
from pathlib import Path


from zlc_atom.nodes.calibration import LOGIC_NODE as CALIBRATION_LOGIC_NODE
from zlc_atom.devices.simulation.sequencer import CAMERA_TRIGGER_CHANNEL
from zlc_atom.nodes.calibration.pulse import resolve_pulse
from zlc_pulse import PulseSequence, sequence_from_tree


PULSE_ROOT = Path(__file__).resolve().parent / "pulses"


def pulse_document(name: str) -> bytes:
    """One test-owned pulse document, exactly as the file holds it."""

    return (PULSE_ROOT / name).read_bytes()


def pulse_sequence(name: str) -> PulseSequence:
    """One test-owned pulse document, read as the pulse it authors."""

    return sequence_from_tree(json.loads(pulse_document(name).decode("utf-8")))


#: The port the virtual board gates its camera from -- the simulated
#: world's own fact, and the only place in this project that needs one.
CAMERA_CHANNEL = CAMERA_TRIGGER_CHANNEL
#: How many camera windows the template beside the calibration node plays.
#: A fact about THIS fixture's pulse, stated here: a measurement is told how
#: many frames it reads, it does not interrogate a pulse to find out.
CALIBRATION_FRAMES_PER_CYCLE = 3
IMAGING_PULSE_RESOURCE = CALIBRATION_LOGIC_NODE.workspace_resources[0].resolve(
    PULSE_ROOT / "imaging_template.json"
)


def build_calibration_pulse(
    sequencer: object,
    *,
    reference_exposure_seconds: float = 0.020,
    readout_exposure_seconds: float = 0.005,
) -> object:
    resolved = resolve_pulse(
        IMAGING_PULSE_RESOURCE.value,
        path=IMAGING_PULSE_RESOURCE.path,
        sequencer=sequencer,
        api_values={
            "reference_probe_duration_before": reference_exposure_seconds,
            "readout_probe_duration": readout_exposure_seconds,
            "reference_probe_duration_after": reference_exposure_seconds,
        },
    )
    return resolved.program


__all__ = [
    "CAMERA_CHANNEL",
    "IMAGING_PULSE_RESOURCE",
    "PULSE_ROOT",
    "build_calibration_pulse",
    "pulse_document",
    "pulse_sequence",
]
