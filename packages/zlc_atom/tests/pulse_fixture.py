"""The shipped calibration template compiled at its normal working point."""

from __future__ import annotations

from pathlib import Path

from zlc_pulse.device import BoardDescription

from zlc_atom.nodes.calibration import LOGIC_NODE as CALIBRATION_LOGIC_NODE
from zlc_atom.nodes.calibration.pulse import (
    CAMERA_TRIGGER_PORT,
    resolve_pulse,
)


PULSE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "zlc_atom"
    / "nodes"
    / "calibration"
)
CAMERA_CHANNEL = CAMERA_TRIGGER_PORT
IMAGING_PULSE_RESOURCE = CALIBRATION_LOGIC_NODE.workspace_resources[0].resolve(
    PULSE_ROOT / "imaging_template.json"
)


def build_calibration_pulse(
    board: BoardDescription,
    *,
    reference_exposure_seconds: float = 0.020,
    readout_exposure_seconds: float = 0.005,
) -> tuple[object, object]:
    resolved = resolve_pulse(
        IMAGING_PULSE_RESOURCE.value,
        path=IMAGING_PULSE_RESOURCE.path,
        board=board,
        api_values={
            "reference_probe_duration_before": reference_exposure_seconds,
            "readout_probe_duration": readout_exposure_seconds,
            "reference_probe_duration_after": reference_exposure_seconds,
        },
    )
    return resolved.program, resolved.metadata


__all__ = [
    "CAMERA_CHANNEL",
    "IMAGING_PULSE_RESOURCE",
    "PULSE_ROOT",
    "build_calibration_pulse",
]
