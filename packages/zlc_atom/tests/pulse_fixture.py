"""The pulse documents the tests run against, the calibration one compiled,
and the calibration request and bench that run it.

No pulse document lives in the product: pulses are workspace files the
operator names.  The tests own theirs, here beside them, and read them
through ``pulse_document``.

The workbench tests import this module by its bare name, so it imports
nothing from ``tests``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path


from zlc_atom.install import create_installation
from zlc_atom.nodes.calibration import (
    CalibrationRequest,
    CalibrationTask,
    LOGIC_NODE as CALIBRATION_LOGIC_NODE,
    ReadoutModelKind,
)
from zlc_atom.devices.simulation.sequencer import CAMERA_TRIGGER_CHANNEL
from zlc_atom.nodes.calibration.pulse import resolve_pulse
from zlc_pulse import PulseSequence, sequence_from_tree
from zlc_runtime import SignalDataPlane


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
            "duration:long_before": reference_exposure_seconds,
            "duration:short": readout_exposure_seconds,
            "duration:long_after": reference_exposure_seconds,
        },
    )
    return resolved.program


def calibration_request(*, repeats: int = 30) -> CalibrationRequest:
    """The box-model calibration of the imaging template's three windows."""

    return CalibrationRequest(
        camera_key="camera",
        sequencer_key="sequencer",
        pulse_template=IMAGING_PULSE_RESOURCE.path.name,
        repeats=repeats,
        reference_exposure_seconds=0.02,
        readout_exposure_seconds=0.005,
        camera_exposure_seconds=0.02,
        reference_before_field="duration:long_before",
        readout_field="duration:short",
        reference_after_field="duration:long_after",
        default_model_kind=ReadoutModelKind.BOX,
        threshold_method="gaussian",
        box_half_width=1,
        psf_half_width=3,
        psf_padding=3,
        detection_spot_sigma=1.0,
        detection_sigma=6.0,
    )


@contextmanager
def calibration_task(request: CalibrationRequest) -> Iterator[CalibrationTask]:
    """A calibration task on a fresh virtual bench, closed when the block ends.

    The installation is the owner of the devices the task borrows, and the
    plane of what it publishes; a task closes neither, so the helper that
    built the bench is the one that takes it down again.
    """

    plane = SignalDataPlane()
    try:
        installation = create_installation("virtual")
        try:
            yield CalibrationTask(
                camera=installation.device("camera"),
                sequencer=installation.device("sequencer"),
                request=request,
                pulse_sequence=IMAGING_PULSE_RESOURCE.value,
                pulse_path=IMAGING_PULSE_RESOURCE.path,
                signal_plane=plane,
            )
        finally:
            installation.close()
    finally:
        plane.close()


__all__ = [
    "CAMERA_CHANNEL",
    "IMAGING_PULSE_RESOURCE",
    "PULSE_ROOT",
    "build_calibration_pulse",
    "calibration_request",
    "calibration_task",
    "pulse_document",
    "pulse_sequence",
]
