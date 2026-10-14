"""Saved calibration artifact contract at the node composition boundary."""

from __future__ import annotations

from zlc_atom.nodes._framework.descriptor import ArtifactCodec

from .calibration import READOUT_MODEL_CHOICES, TrapCalibration


def _describe(calibration: TrapCalibration) -> str:
    """The exposure it was trained at, its sites and its default model.

    The model is named as the Readout model choice names it, so a file's
    line and the choice beside it speak one vocabulary.
    """

    exposure = calibration.frame_contract.exposure_seconds
    return " · ".join((
        *(() if exposure is None else (f"{exposure * 1e3:g} ms",)),
        f"{calibration.n_sites} sites",
        "default " + next(choice.label for choice in READOUT_MODEL_CHOICES
                          if choice.value == calibration.default_model_kind.value),
    ))


CALIBRATION_ARTIFACT_CODEC = ArtifactCodec(
    TrapCalibration.CONTRACT_ID,
    "Calibration artifacts (*.json)",
    (".json",),
    TrapCalibration.load,
    _describe,
)


__all__ = ["CALIBRATION_ARTIFACT_CODEC"]
