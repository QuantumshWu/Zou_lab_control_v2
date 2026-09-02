"""Discoverable direct qCMOS-to-SLM feedback Task."""

from __future__ import annotations

from zlc_pulse import PulseSequence
from zlc_plot import AxisRef, Reduction
from zlc_plot.semantics import fate_field_name

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
    OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT,
    SITE_SIGNAL_HISTORY_OUTPUT,
    SLM_PHASE_ARTIFACT_CONTRACT,
    SlmFeedbackTask,
    TARGET_SHARE_HISTORY_OUTPUT,
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
    "zlc.pulse/slm-feedback",
    "pulses",
    (".json",),
    load_calibration_pulse_template,
    argument_name="pulse_resource",
)


def _validate_feedback(values: dict[str, object]) -> None:
    factors = tuple(float(value) for value in values["probe_factors"])
    if len(set(factors)) != len(factors) or any(
        value <= 0.0 or value == 1.0 for value in factors
    ):
        raise ValueError("probe_factors must be unique positive numbers excluding 1")

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
            0.1,
            minimum=1e-9,
        ),
        AuthoringField(
            "shots_per_candidate", "int", "qCMOS shots per candidate", 100, minimum=10
        ),
        AuthoringField(
            "probe_factors",
            "numeric_tuple",
            "Single-population probe factors",
            (0.5, 2.0),
        ),
        # LOOP gain: the fraction of each site's residual removed per
        # candidate, once the plant slope has been measured.  Not a weight
        # multiplier -- the task divides by the measured response so that
        # 0.3 means 30% of the residual regardless of how hard the traps
        # answer; before a slope is trusted it steps at half this.
        AuthoringField(
            "feedback_gain",
            "float",
            "Loop gain (residual fraction removed per candidate)",
            0.3,
            minimum=0.0,
        ),
        AuthoringField(
            "maximum_weight_change",
            "float",
            "Maximum ordinary weight change",
            0.5,
            minimum=0.0,
        ),
        AuthoringField("max_updates", "int", "Maximum feedback updates", 12, minimum=1),
    ),
    validator=_validate_feedback,
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
        probe_factors=tuple(float(value) for value in authored["probe_factors"]),
        feedback_gain=float(authored["feedback_gain"]),
        maximum_weight_change=float(authored["maximum_weight_change"]),
        max_updates=int(authored["max_updates"]),
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
        OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT,
        SITE_SIGNAL_HISTORY_OUTPUT,
        TARGET_SHARE_HISTORY_OUTPUT,
    ),
    node_previews=(
        NodePreviewSpec(
            CAMERA_FRAMES_OUTPUT,
            "image",
            semantic={
                fate_field_name(AxisRef.repeat()): "reduce",
                "reduction": Reduction.MEAN,
            },
            producer="camera",
        ),
        NodePreviewSpec(OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT, "curve"),
        NodePreviewSpec(
            SITE_SIGNAL_HISTORY_OUTPUT,
            "curve",
            semantic={
                fate_field_name(AxisRef.point("slm_feedback.candidate")): "x",
                fate_field_name(AxisRef.data("calibration.site")): "group",
            },
        ),
        NodePreviewSpec(
            TARGET_SHARE_HISTORY_OUTPUT,
            "curve",
            semantic={
                fate_field_name(AxisRef.point("slm_feedback.candidate")): "x",
                fate_field_name(AxisRef.data("calibration.site")): "group",
            },
        ),
    ),
    artifact_outputs=(
        ArtifactOutputSpec("artifact_path", SLM_PHASE_ARTIFACT_CONTRACT),
    ),
    device_requirements=(
        DeviceRequirement(
            "camera.adapter",
            "camera",
            (
                "exposure_seconds",
                "roi_x",
                "roi_y",
                "roi_width",
                "roi_height",
                "trigger_source",
                "readout_speed",
                "offset_counts",
                "electrons_per_count",
            ),
        ),
        DeviceRequirement("sequencer.streamer", "sequencer", ("program",)),
        DeviceRequirement(
            "slm.phase",
            "slm",
            (
                "phase",
                "wavelength_nm",
                "display_name",
                "width",
                "height",
                "correction_path",
                "flip_x",
                "flip_y",
            ),
        ),
    ),
    build=_build,
    workspace_resources=(_PULSE_RESOURCE,),
)

__all__ = [
    "LOGIC_NODE",
    "SLM_FEEDBACK_SCHEMA",
]
