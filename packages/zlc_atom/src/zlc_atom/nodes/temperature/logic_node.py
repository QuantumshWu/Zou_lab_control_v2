"""The temperature Task: release-recapture over the trap-off time.

What the operator authors is the template, the release times to play, how many
whole sweeps of them to take, and which readout model to judge the frames
with.  Everything else this measurement needs, it already knows or already
holds: it takes the camera and the sequencer as its own devices, and the
exposure and ROI come from the calibration whose thresholds will judge the
frames -- a third copy on this form would be a number nothing enforces.

The release times are authored with the SAME editor the scan nodes use: from,
to, points, against the port's own hard limits.  A release plan is a scan
plan; typing "0.5, 1, 2, 3" into a text box was a second way to say the same
thing, and it was the way that could not tell you the board refuses 0.
"""

from __future__ import annotations

from zlc_pulse import PulseSequence
from zlc_plot import AxisRef, Reduction
from zlc_plot.semantics import fate_field_name

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS
from zlc_atom.nodes._framework.descriptor import (
    ArtifactInputSpec,
    ArtifactOutputSpec,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    NodePreviewSpec,
    ResolvedArtifact,
    ResolvedWorkspaceResource,
)
from zlc_atom.nodes.calibration import (
    CALIBRATION_ARTIFACT_CODEC,
    DEFAULT_READOUT_MODEL_CHOICE,
    READOUT_MODEL_CHOICES,
    TrapCalibration,
    readout_model_kind_from_choice,
)
from zlc_atom.nodes.scan import (
    SCAN_OUTPUT,
    SCAN_PULSE_CONTRACT,
    STEPPED_PULSE_RESOURCE,
    api_overrides_from_authored,
    apply_api_overrides,
    plan_from_authored,
)

from .task import (
    SURVIVAL_OUTPUT,
    TEMPERATURE_ARTIFACT_CONTRACT,
    TemperatureTask,
)


#: Select the actual release field of the chosen Pulse before authoring its range.
DEFAULT_RELEASE_PLAN = '{"axes": []}'


TEMPERATURE_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse_template",
            "resource",
            "Pulse template",
            "",
            required=True,
        ),
        # The release plan, in the unit the template's own release parameter
        # declares; the plan is bound against that parameter's hard limits,
        # so a release the board cannot play is refused before anything arms.
        AuthoringField(
            "plan",
            "text",
            "Release plan",
            DEFAULT_RELEASE_PLAN,
            required=True,
        ),
        # The pulse's other API fields, set once for this run.  The release
        # itself is swept by the plan and never appears here.
        AuthoringField(
            "api_values",
            "text",
            "API values",
            "",
        ),
        AuthoringField(
            "repeats",
            "int",
            "Repeats (full sweeps)",
            20,
            minimum=1,
        ),
        AuthoringField(
            # Empty means "the one the calibration measured its thresholds
            # at", which is the sensible default and was, until now, the only
            # possibility.  Whether another exposure is physically comparable
            # is the operator's judgement, not this node's: a calibration
            # describes what it did, it does not licence the next run.
            "exposure_seconds",
            "float",
            "Exposure seconds (blank = as calibrated)",
            None,
            required=False,
            minimum=1e-9,
        ),
        AuthoringField(
            "model_kind",
            "choice",
            "Readout model",
            DEFAULT_READOUT_MODEL_CHOICE,
            choices=READOUT_MODEL_CHOICES,
        ),
    )
)


def _build(
    *,
    sequencer: object,
    sequencer_key: str,
    camera: object,
    camera_key: str,
    signal_plane: object,
    calibration: ResolvedArtifact,
    pulse_resource: ResolvedWorkspaceResource,
    save_figure_artifact: object = None,
    **values: object,
) -> TemperatureTask:
    authored = TEMPERATURE_SCHEMA.project_values(values)
    if (
        not isinstance(pulse_resource, ResolvedWorkspaceResource)
        or pulse_resource.contract_id != SCAN_PULSE_CONTRACT
        or not isinstance(pulse_resource.value, PulseSequence)
    ):
        raise TypeError("pulse_resource must be a resolved scan template")
    if (
        not isinstance(calibration, ResolvedArtifact)
        or calibration.contract_id != CALIBRATION_ARTIFACT_CODEC.contract_id
        or not isinstance(calibration.value, TrapCalibration)
    ):
        raise TypeError("calibration must be a resolved calibration artifact")
    return TemperatureTask(
        sequencer=sequencer,
        sequencer_key=sequencer_key,
        camera=camera,
        camera_key=camera_key,
        signal_plane=signal_plane,
        sequence=apply_api_overrides(
            pulse_resource.value,
            api_overrides_from_authored(authored["api_values"]),
        ),
        pulse_path=pulse_resource.path,
        calibration=calibration.value,
        calibration_path=calibration.path,
        plan=plan_from_authored(authored["plan"]),
        repeats=int(authored["repeats"]),
        exposure_seconds=(
            None
            if authored["exposure_seconds"] is None
            else float(authored["exposure_seconds"])
        ),
        model_kind=readout_model_kind_from_choice(authored["model_kind"]),
        save_figure_artifact=save_figure_artifact,
    )


def _editor_factory(parent=None):
    from zlc_atom.nodes.scan.editor import scan_plan_editor_factory

    # The operator selects one time-valued API field from this Pulse; no
    # template-specific alias determines which period releases the atoms.
    return scan_plan_editor_factory(
        parent,
        device_ports=False,
    )


LOGIC_NODE = LogicNodeDescriptor(
    "temperature",
    NodeKind.TASK,
    TEMPERATURE_SCHEMA,
    input_specs=(
        ArtifactInputSpec(
            "calibration_path",
            "Calibration artifact",
            CALIBRATION_ARTIFACT_CODEC,
            argument_name="calibration",
        ),
    ),
    outputs=(
        SCAN_OUTPUT,
        SURVIVAL_OUTPUT,
    ),
    # The watched curve is a projection of the per-shot, per-site survival
    # truth, not a second accumulated rate Dataset.
    node_previews=(
        NodePreviewSpec(
            SURVIVAL_OUTPUT,
            "curve",
            semantic={
                fate_field_name(AxisRef.point("temperature.t_off")): "x",
                fate_field_name(AxisRef.cell_data("calibration.site")): "reduce",
                "reduction": Reduction.MEAN,
            },
        ),
    ),
    artifact_outputs=(
        ArtifactOutputSpec("artifact_path", TEMPERATURE_ARTIFACT_CONTRACT),
    ),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera", CAMERA_PROTECTED_FIELDS),
        DeviceRequirement("sequencer.streamer", "sequencer", ("program",)),
    ),
    build=_build,
    ui_contributions=(_editor_factory,),
    workspace_resources=(STEPPED_PULSE_RESOURCE,),
)


__all__ = ["DEFAULT_RELEASE_PLAN", "LOGIC_NODE", "TEMPERATURE_SCHEMA"]
