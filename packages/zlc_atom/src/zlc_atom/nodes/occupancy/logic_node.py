"""Discoverable occupancy processor descriptor."""

from __future__ import annotations

from collections.abc import Sequence

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    ArtifactInputSpec,
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
    ResolvedArtifact,
)
from zlc_atom.nodes._framework.frames import FrameClaim
from zlc_atom.nodes.calibration import (
    CALIBRATION_ARTIFACT_CODEC,
    DEFAULT_READOUT_MODEL_CHOICE,
    READOUT_MODEL_CHOICES,
    TrapCalibration,
    readout_model_kind_from_choice,
)

from .processor import OCCUPANCY_OUTPUTS, CalibrationReading, OccupancyProcessor


OCCUPANCY_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "model_kind",
            "choice",
            "Readout model",
            DEFAULT_READOUT_MODEL_CHOICE,
            choices=READOUT_MODEL_CHOICES,
            segmented=True,
            description="Default reads each file with its calibration's own default model",
        ),
    )
)


def _reading(claim: object) -> CalibrationReading:
    artifact = getattr(claim, "value", None)
    if (
        not isinstance(claim, FrameClaim)
        or not isinstance(artifact, ResolvedArtifact)
        or artifact.contract_id != CALIBRATION_ARTIFACT_CODEC.contract_id
        or not isinstance(artifact.value, TrapCalibration)
    ):
        raise TypeError("occupancy calibrations must claim frames with resolved calibration artifacts")
    return CalibrationReading(artifact.value, claim.frames, artifact.path)


def _build(
    *,
    calibrations: Sequence[FrameClaim],
    source_signal: str,
    post_select=None,
    post_select_record=None,
    **values: object,
) -> OccupancyProcessor:
    authored = OCCUPANCY_SCHEMA.project_values(values)
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    # A row claiming no frame is kept, not read: only its file goes on, for
    # the run record -- whether or not that file could be decoded.
    return OccupancyProcessor(
        tuple(_reading(claim) for claim in calibrations if claim.frames),
        parked=tuple(
            None if claim.value is None else claim.value.path
            for claim in calibrations if not claim.frames
        ),
        source_signal=selected_source,
        model_kind=readout_model_kind_from_choice(authored["model_kind"]),
        post_select=post_select,
        post_select_record=post_select_record,
    )


def _editor_factory(parent=None):
    from .editor import CalibrationsEditor

    return CalibrationsEditor(parent)


def _describe_output_frames(values, output_name, source):
    """Every Occupancy output preserves its camera's readout-event axis, name and labels."""
    return source if any(item.name == output_name for item in OCCUPANCY_OUTPUTS) else None


LOGIC_NODE = LogicNodeDescriptor(
    "occupancy",
    NodeKind.PROCESSOR,
    OCCUPANCY_SCHEMA,
    description=(
        "Read camera image data using saved Calibrations and publish per-site counts, "
        "occupied verdicts and frame_judged diagnostics while preserving frame and scan "
        "axes. Each calibration claims the frames it reads: 'first' (frame 0), 'last', "
        "positive numbers for middle frames between them (a middle number the cycle no "
        "longer has between its ends is absent: kept, noted and reading nothing, never the "
        "last frame), and 'rest' (every frame no other calibration names); one calibration "
        "with ['rest'] reads every frame. A calibration with frames [] is kept but not read, "
        "and its file is not checked against the others. Use occupied for loading-rate "
        "plots, camera overlays or Frame Survival; the processor follows the source and "
        "resumes when compatible data returns."
    ),
    input_specs=(
        DatasetInputSpec("frames", None, "exact"),
        # Each calibration claims the frames it reads -- frame numbers from
        # 0, first, last, rest -- so a load frame and a readout frame taken under
        # different exposures are read with calibrations trained on each.
        ArtifactInputSpec(
            "calibrations",
            "Calibrations",
            CALIBRATION_ARTIFACT_CODEC,
            frame_claims=True,
        ),
    ),
    ui_contributions=(_editor_factory,),
    outputs=OCCUPANCY_OUTPUTS,
    describe_output_frames=_describe_output_frames,
    build=_build,
)

__all__ = ["LOGIC_NODE", "OCCUPANCY_SCHEMA"]
