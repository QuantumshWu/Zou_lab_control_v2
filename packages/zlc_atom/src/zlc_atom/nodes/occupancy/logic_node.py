"""Discoverable occupancy processor descriptor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    ArtifactInputSpec,
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
    OutputSpec,
    ResolvedArtifact,
)
from zlc_atom.nodes.calibration import (
    CALIBRATION_ARTIFACT_CODEC,
    DEFAULT_READOUT_MODEL_CHOICE,
    TrapCalibration,
    readout_model_kind_from_choice,
)
from zlc_plot import ImagePointOverlay

from .processor import OccupancyProcessor


OCCUPANCY_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "model_kind",
            "choice",
            "Readout model",
            DEFAULT_READOUT_MODEL_CHOICE,
            choices=(
                AuthoringChoice("default", "Calibration default"),
                AuthoringChoice("box", "Box"),
                AuthoringChoice("psf", "Per-site PSF"),
                AuthoringChoice("uniform_psf", "Uniform PSF"),
            ),
        ),
    )
)


def _build(
    *,
    calibration: ResolvedArtifact,
    source_signal: str,
    **values: object,
) -> OccupancyProcessor:
    authored = OCCUPANCY_SCHEMA.project_values(values)
    if (
        not isinstance(calibration, ResolvedArtifact)
        or calibration.contract_id != CALIBRATION_ARTIFACT_CODEC.contract_id
        or not isinstance(calibration.value, TrapCalibration)
    ):
        raise TypeError("calibration must be a resolved calibration artifact")
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    return OccupancyProcessor(
        calibration.value,
        calibration_path=calibration.path,
        source_signal=selected_source,
        model_kind=readout_model_kind_from_choice(authored["model_kind"]),
    )


LOGIC_NODE = LogicNodeDescriptor(
    "occupancy",
    NodeKind.PROCESSOR,
    OCCUPANCY_SCHEMA,
    input_specs=(
        DatasetInputSpec("frames", None),
        ArtifactInputSpec(
            "calibration_path",
            "Calibration artifact",
            CALIBRATION_ARTIFACT_CODEC,
            argument_name="calibration",
        ),
    ),
    outputs=(
        OutputSpec("counts", "occupancy.counts.v1"),
        OutputSpec("occupied", "occupancy.occupied.v1"),
        OutputSpec("valid", "occupancy.valid.v1"),
        OutputSpec("rate", "occupancy.rate.v1"),
        OutputSpec("survival", "occupancy.survival.v1"),
        OutputSpec("survival_rate", "occupancy.survival_rate.v1"),
        OutputSpec("frame_judged", "occupancy.frame_judged.v1"),
        OutputSpec("site_overlay", ImagePointOverlay.CONTRACT_ID),
    ),
    build=_build,
)

__all__ = ["LOGIC_NODE", "OCCUPANCY_SCHEMA"]
