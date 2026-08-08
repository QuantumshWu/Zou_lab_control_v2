"""The shipped calibration template compiled at its normal working point."""

from __future__ import annotations

from pathlib import Path

from zlc_pulse.device import BoardDescription

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


def build_calibration_pulse(
    board: BoardDescription,
    *,
    reference_exposure_seconds: float = 0.020,
    readout_exposure_seconds: float = 0.005,
) -> tuple[object, object]:
    resolved = resolve_pulse(
        "imaging_template.json",
        board=board,
        search_paths=(PULSE_ROOT,),
        api_values={
            "reference_probe_duration_before": reference_exposure_seconds,
            "readout_probe_duration": readout_exposure_seconds,
            "reference_probe_duration_after": reference_exposure_seconds,
        },
    )
    return resolved.program, resolved.metadata


__all__ = [
    "CAMERA_CHANNEL",
    "PULSE_ROOT",
    "build_calibration_pulse",
]
