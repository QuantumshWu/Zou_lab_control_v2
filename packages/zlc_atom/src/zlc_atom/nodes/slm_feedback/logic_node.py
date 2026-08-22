"""Discoverable direct qCMOS-to-SLM feedback Task."""

from __future__ import annotations

from zlc_pulse import PulseSequence
from zlc_plot import Reduction

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices.slm.solver import load_science_context
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


_SCIENCE_CONTEXT_CODEC = ArtifactCodec(
    SLM_PHASE_ARTIFACT_CONTRACT,
    "SLM Science Contexts (*.npz)",
    (".npz",),
    load_science_context,
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
            "feedback_mode",
            "choice",
            "Feedback mode",
            "qcmos_bright_dark",
            choices=(
                AuthoringChoice(
                    "qcmos_bright_dark",
                    "qCMOS fluorescence (bright - dark)",
                ),
            ),
        ),
        AuthoringField(
            "pulse_template", "resource", "Imaging pulse", "", required=True
        ),
        AuthoringField(
            "exposure_seconds",
            "float",
            "Camera exposure seconds",
            None,
            required=True,
            minimum=1e-9,
        ),
        AuthoringField(
            "shots_per_candidate", "int", "qCMOS shots per candidate", 500, minimum=10
        ),
        AuthoringField(
            "validation_shots",
            "int",
            "Maximum independent validation shots",
            3000,
            minimum=10,
        ),
        AuthoringField("max_updates", "int", "Maximum feedback updates", 12, minimum=1),
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
    science_context: ResolvedArtifact,
    pulse_resource: ResolvedWorkspaceResource,
    artifact_directory: object,
    **values: object,
) -> SlmFeedbackTask:
    authored = SLM_FEEDBACK_SCHEMA.project_values(values)
    if not isinstance(calibration, ResolvedArtifact) or not isinstance(
        calibration.value, TrapCalibration
    ):
        raise TypeError("calibration must be a resolved calibration artifact")
    if not isinstance(science_context, ResolvedArtifact):
        raise TypeError("science_context must be a resolved Science Context artifact")
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
        science_context=science_context.value,
        science_context_path=science_context.path,
        pulse_sequence=pulse_resource.value,
        pulse_path=pulse_resource.path,
        feedback_mode=str(authored["feedback_mode"]),
        exposure_seconds=float(authored["exposure_seconds"]),
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
        ArtifactInputSpec(
            "science_context_path",
            "SLM Science Context",
            _SCIENCE_CONTEXT_CODEC,
            argument_name="science_context",
        ),
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

__all__ = [
    "LOGIC_NODE",
    "SLM_FEEDBACK_SCHEMA",
]
