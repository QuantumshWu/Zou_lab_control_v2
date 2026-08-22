"""Saved calibration artifact contract at the node composition boundary."""

from __future__ import annotations

from zlc_atom.nodes._framework.descriptor import ArtifactCodec

from .calibration import TrapCalibration


CALIBRATION_ARTIFACT_CODEC = ArtifactCodec(
    TrapCalibration.CONTRACT_ID,
    "Calibration artifacts (*.json)",
    (".json",),
    TrapCalibration.load,
)


__all__ = ["CALIBRATION_ARTIFACT_CODEC"]
