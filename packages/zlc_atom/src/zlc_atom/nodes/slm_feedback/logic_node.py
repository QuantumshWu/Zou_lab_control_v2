"""SLM feedback from camera populations or a configured acquisition's fit."""

from __future__ import annotations

from zlc_pulse import PulseSequence
from zlc_plot import AxisRef, Reduction
from zlc_plot.semantics import fate_field_name

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS
from zlc_atom.devices.slm.solver import load_science_context
from zlc_atom.nodes._framework.descriptor import (
    ArtifactCodec,
    ArtifactInputSpec,
    ArtifactOutputSpec,
    DeviceRequirement,
    DatasetInputSpec,
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
from zlc_runtime.selection_bridge import FIT_PARAMETER_CONTRACT

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
_CAMERA_MODES = ("feedback_mode", ("qcmos_bright_dark", "qcmos_loading_rate"))
_EXTERNAL_MODE = ("feedback_mode", ("ramsey_frequency",))


def _validate_feedback(values: dict[str, object]) -> None:
    if values["feedback_mode"] == "ramsey_frequency":
        return
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
                AuthoringChoice(
                    "qcmos_loading_rate",
                    "qCMOS loading rate (bright shots / all shots)",
                ),
                AuthoringChoice(
                    "ramsey_frequency", "Ramsey frequency (external acquisition)"
                ),
            ),
        ),
        AuthoringField(
            "pulse_template", "resource", "Imaging pulse", "", required=True,
            enabled_when=_CAMERA_MODES,
        ),
        AuthoringField(
            "acquisition_logic", "str", "Acquisition logic", "", required=True,
            enabled_when=_EXTERNAL_MODE,
            description="Choose a finite acquisition; run it to completion for each SLM phase without changing its settings.",
        ),
        AuthoringField(
            "exposure_seconds",
            "float",
            "Camera exposure seconds",
            0.1,
            minimum=1e-9,
            enabled_when=_CAMERA_MODES,
        ),
        AuthoringField(
            "shots_per_candidate", "int", "qCMOS shots per candidate", 100, minimum=10,
            enabled_when=_CAMERA_MODES,
        ),
        AuthoringField(
            "probe_factors",
            "numeric_tuple",
            "Single-population probe factors",
            (0.5, 2.0),
            enabled_when=_CAMERA_MODES,
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
    camera: object = None,
    camera_key: str = "",
    sequencer: object = None,
    sequencer_key: str = "",
    slm: object,
    slm_key: str,
    signal_plane: object,
    calibration: ResolvedArtifact,
    science_context: ResolvedArtifact,
    pulse_resource: ResolvedWorkspaceResource | None = None,
    source_signal: str = "",
    restart_logic: object = None,
    check_signal_input: object = None,
    save_figure_artifact: object = None,
    **values: object,
) -> SlmFeedbackTask:
    authored = SLM_FEEDBACK_SCHEMA.project_values(values)
    if not isinstance(calibration, ResolvedArtifact) or not isinstance(
        calibration.value, TrapCalibration
    ):
        raise TypeError("calibration must be a resolved calibration artifact")
    if not isinstance(science_context, ResolvedArtifact):
        raise TypeError("science_context must be a resolved Science Context artifact")
    external = authored["feedback_mode"] == "ramsey_frequency"
    if not external and (not isinstance(pulse_resource, ResolvedWorkspaceResource) or not isinstance(
        pulse_resource.value, PulseSequence
    )):
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
        pulse_sequence=None if external else pulse_resource.value,
        pulse_path=None if external else pulse_resource.path,
        acquisition_logic=str(authored.get("acquisition_logic", "")),
        source_signal=source_signal,
        restart_logic=restart_logic,
        check_signal_input=check_signal_input,
        feedback_mode=str(authored["feedback_mode"]),
        exposure_seconds=None if external else float(authored["exposure_seconds"]),
        shots_per_candidate=0 if external else int(authored["shots_per_candidate"]),
        probe_factors=() if external else tuple(float(value) for value in authored["probe_factors"]),
        feedback_gain=float(authored["feedback_gain"]),
        maximum_weight_change=float(authored["maximum_weight_change"]),
        max_updates=int(authored["max_updates"]),
        save_figure_artifact=save_figure_artifact,
    )


LOGIC_NODE = LogicNodeDescriptor(
    "slm_feedback",
    NodeKind.TASK,
    SLM_FEEDBACK_SCHEMA,
    input_specs=(
        DatasetInputSpec("frequency", FIT_PARAMETER_CONTRACT, "latest", enabled_when=_EXTERNAL_MODE),
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
    node_previews=(),
    declare_previews=lambda values, devices: _PREVIEWS[1:] if values.get("feedback_mode") == "ramsey_frequency" else _PREVIEWS,
    artifact_outputs=(
        ArtifactOutputSpec("artifact_path", SLM_PHASE_ARTIFACT_CONTRACT),
    ),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera", CAMERA_PROTECTED_FIELDS, enabled_when=_CAMERA_MODES),
        DeviceRequirement("sequencer.streamer", "sequencer", ("program",), enabled_when=_CAMERA_MODES),
        DeviceRequirement(
            "slm.phase", "slm",
            ("phase", "wavelength_nm", "display_name", "width", "height", "correction_path", "flip_x", "flip_y"),
        ),
    ),
    build=_build,
    workspace_resources=(_PULSE_RESOURCE,),
    acquisition_input="acquisition_logic",
    acquisition_completion="terminal",
)

_PREVIEWS = (
        NodePreviewSpec(
            CAMERA_FRAMES_OUTPUT,
            "image",
            semantic={
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
                fate_field_name(AxisRef.cell_data("calibration.site")): "group",
            },
        ),
        NodePreviewSpec(
            TARGET_SHARE_HISTORY_OUTPUT,
            "curve",
            semantic={
                fate_field_name(AxisRef.point("slm_feedback.candidate")): "x",
                fate_field_name(AxisRef.cell_data("calibration.site")): "group",
            },
        ),
)

__all__ = [
    "LOGIC_NODE",
    "SLM_FEEDBACK_SCHEMA",
]
