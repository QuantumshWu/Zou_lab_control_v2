"""Discoverable occupancy processor descriptor."""

from __future__ import annotations

from collections.abc import Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    ArtifactInputSpec,
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
    ResolvedArtifact,
)
from zlc_atom.nodes.calibration import (
    CALIBRATION_ARTIFACT_CODEC,
    DEFAULT_READOUT_MODEL_CHOICE,
    READOUT_MODEL_CHOICES,
    TrapCalibration,
    readout_model_kind_from_choice,
)

from .processor import OCCUPANCY_OUTPUTS, OccupancyProcessor


OCCUPANCY_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "model_kind",
            "choice",
            "Readout model",
            DEFAULT_READOUT_MODEL_CHOICE,
            choices=READOUT_MODEL_CHOICES,
        ),
    )
)


def _resolved_calibration(artifact: object, what: str) -> ResolvedArtifact:
    if (
        not isinstance(artifact, ResolvedArtifact)
        or artifact.contract_id != CALIBRATION_ARTIFACT_CODEC.contract_id
        or not isinstance(artifact.value, TrapCalibration)
    ):
        raise TypeError(f"{what} must be a resolved calibration artifact")
    return artifact


def _build(
    *,
    calibration: ResolvedArtifact,
    calibration_by_frame: Mapping[int, ResolvedArtifact] | None = None,
    source_signal: str,
    **values: object,
) -> OccupancyProcessor:
    authored = OCCUPANCY_SCHEMA.project_values(values)
    shared = _resolved_calibration(calibration, "calibration")
    by_frame = {
        int(frame): _resolved_calibration(artifact, f"frame {frame} calibration")
        for frame, artifact in dict(calibration_by_frame or {}).items()
    }
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    return OccupancyProcessor(
        shared.value,
        calibration_by_frame={frame: artifact.value for frame, artifact in by_frame.items()},
        calibration_path=shared.path,
        calibration_paths_by_frame={frame: artifact.path for frame, artifact in by_frame.items()},
        source_signal=selected_source,
        model_kind=readout_model_kind_from_choice(authored["model_kind"]),
    )


LOGIC_NODE = LogicNodeDescriptor(
    "occupancy",
    NodeKind.PROCESSOR,
    OCCUPANCY_SCHEMA,
    input_specs=(
        DatasetInputSpec("frames", None, "exact"),
        # Every frame reads with the calibration named here unless a frame
        # is given its own: a load frame and a readout frame taken under
        # different exposures are read with calibrations trained on each.
        ArtifactInputSpec(
            "calibration_path",
            "Calibration artifact",
            CALIBRATION_ARTIFACT_CODEC,
            argument_name="calibration",
            per_frame=True,
        ),
    ),
    outputs=OCCUPANCY_OUTPUTS,
    build=_build,
)

__all__ = ["LOGIC_NODE", "OCCUPANCY_SCHEMA"]
