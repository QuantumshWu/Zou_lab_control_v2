"""Discoverable direct qCMOS-to-SLM feedback Task."""

from __future__ import annotations

from zlc_pulse import PulseSequence
from zlc_plot import Reduction

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.slm.solver import load_target
from zlc_atom.nodes._framework.descriptor import (
    ArtifactCodec,
    ArtifactInputSpec,
    ArtifactOutputSpec,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodePreviewSpec,
    NodeKind,
    ResolvedArtifact,
    ResolvedWorkspaceResource,
    WorkspaceResourceSpec,
)
from zlc_atom.nodes.calibration import CALIBRATION_ARTIFACT_CODEC, TrapCalibration
from zlc_atom.nodes.calibration.pulse import load_calibration_pulse_template
from zlc_atom.nodes.camera_measurement.measurement import CAMERA_FRAMES_OUTPUT

from .task import (
    CANDIDATE_PHASE_OUTPUT,
    READOUT_FRAME_COORDINATE,
    SLM_PHASE_ARTIFACT_CONTRACT,
    SlmFeedbackTask,
    UNIFORMITY_HISTORY_OUTPUT,
)


TARGET_CODEC = ArtifactCodec(
    "zlc.slm.target.v1", "SLM targets (*.json)", (".json",), load_target
)
_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    "zlc.pulse.v1/slm-feedback",
    "pulses",
    (".json",),
    load_calibration_pulse_template,
    argument_name="pulse_resource",
)

SLM_FEEDBACK_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse_template", "resource", "Imaging pulse", "imaging_template.json", required=True
        ),
        AuthoringField(
            "shots_per_candidate", "int", "qCMOS shots per candidate", 100, minimum=10
        ),
        AuthoringField(
            "validation_shots", "int", "Independent qCMOS validation shots", 100, minimum=10
        ),
        AuthoringField("max_updates", "int", "Maximum feedback updates", 120, minimum=1),
    )
)


def _build(
    *,
    camera: object,
    camera_key: str,
    sequencer: object,
    sequencer_key: str,
    slm: object,
    slm_key: str,
    signal_plane: object,
    calibration: ResolvedArtifact,
    target: ResolvedArtifact,
    pulse_resource: ResolvedWorkspaceResource,
    artifact_directory: object,
    **values: object,
) -> SlmFeedbackTask:
    authored = SLM_FEEDBACK_SCHEMA.project_values(values)
    if not isinstance(calibration, ResolvedArtifact) or not isinstance(
        calibration.value, TrapCalibration
    ):
        raise TypeError("calibration must be a resolved calibration artifact")
    if not isinstance(target, ResolvedArtifact):
        raise TypeError("target must be a resolved SLM target")
    if not isinstance(pulse_resource, ResolvedWorkspaceResource) or not isinstance(
        pulse_resource.value, PulseSequence
    ):
        raise TypeError("pulse_resource must be a resolved imaging pulse")
    return SlmFeedbackTask(
        camera=camera,
        camera_key=camera_key,
        sequencer=sequencer,
        sequencer_key=sequencer_key,
        slm=slm,
        slm_key=slm_key,
        signal_plane=signal_plane,
        calibration=calibration.value,
        calibration_path=calibration.path,
        target=target.value,
        target_path=target.path,
        pulse_sequence=pulse_resource.value,
        pulse_path=pulse_resource.path,
        shots_per_candidate=int(authored["shots_per_candidate"]),
        validation_shots=int(authored["validation_shots"]),
        max_updates=int(authored["max_updates"]),
        artifact_directory=artifact_directory,
    )


LOGIC_NODE = LogicNodeDescriptor(
    "slm_feedback",
    NodeKind.TASK,
    SLM_FEEDBACK_SCHEMA,
    input_specs=(
        ArtifactInputSpec(
            "calibration_path", "Calibration artifact", CALIBRATION_ARTIFACT_CODEC, argument_name="calibration"
        ),
        ArtifactInputSpec("target_path", "SLM target", TARGET_CODEC, argument_name="target"),
    ),
    outputs=(
        CANDIDATE_PHASE_OUTPUT,
        UNIFORMITY_HISTORY_OUTPUT,
    ),
    node_previews=(
        NodePreviewSpec(
            CAMERA_FRAMES_OUTPUT,
            "image",
            semantic={
                "fate:frame": READOUT_FRAME_COORDINATE,
                "fate:repeat": "reduce",
                "reduction": Reduction.MEAN,
            },
            producer="camera",
        ),
        NodePreviewSpec(CANDIDATE_PHASE_OUTPUT, "image"),
        NodePreviewSpec(UNIFORMITY_HISTORY_OUTPUT, "curve"),
    ),
    artifact_outputs=(
        ArtifactOutputSpec("artifact_path", SLM_PHASE_ARTIFACT_CONTRACT),
    ),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera"),
        DeviceRequirement("sequencer.streamer", "sequencer"),
        DeviceRequirement("slm.phase", "slm"),
    ),
    build=_build,
    workspace_resources=(_PULSE_RESOURCE,),
)

__all__ = ["LOGIC_NODE", "SLM_FEEDBACK_SCHEMA", "TARGET_CODEC"]
