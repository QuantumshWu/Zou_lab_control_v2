"""Saved calibration artifact contract at the node composition boundary."""

from __future__ import annotations

from zlc_atom.nodes._framework.descriptor import ArtifactCodec

from .calibration import TrapCalibration


def _describe(calibration: TrapCalibration) -> str:
    """The exposure it was trained at, its sites and its default model."""

    exposure = calibration.frame_contract.exposure_seconds
    return " · ".join((
        *(() if exposure is None else (f"{exposure * 1e3:g} ms",)),
        f"{calibration.n_sites} sites",
        calibration.default_model_kind.value,
    ))


CALIBRATION_ARTIFACT_CODEC = ArtifactCodec(
    TrapCalibration.CONTRACT_ID,
    "Calibration artifacts (*.json)",
    (".json",),
    TrapCalibration.load,
    _describe,
)


__all__ = ["CALIBRATION_ARTIFACT_CODEC"]
