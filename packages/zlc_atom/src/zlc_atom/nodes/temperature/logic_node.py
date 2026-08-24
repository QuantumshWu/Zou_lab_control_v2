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
from zlc_plot import Reduction

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
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
    TrapCalibration,
    readout_model_kind_from_choice,
)
from zlc_atom.nodes.scan import (
    PULSE_PARAM_FAMILY,
    SCAN_OUTPUT,
    SCAN_PULSE_CONTRACT,
    STEPPED_PULSE_RESOURCE,
    plan_from_authored,
)

from .task import (
    SURVIVAL_OUTPUT,
    TEMPERATURE_ARTIFACT_CONTRACT,
    T_OFF_PARAMETER,
    TemperatureTask,
)


#: Where a recapture curve lives for micro-kelvin atoms in a micron-sized
#: trap: a few to a few tens of microseconds, in the template's own unit.
#: The form opens on this so the first Start measures something, and the
#: editor's own from/to/points is how it is changed.
DEFAULT_RELEASE_PLAN = (
    '{"axes": [{"port": "pulse:param:t_off", "values": '
    "[0.004, 0.009, 0.014, 0.019, 0.025, 0.03, 0.035, 0.04]}]}"
)


TEMPERATURE_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse_template",
            "resource",
            "Pulse template",
            "temperature_template.json",
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
    sequencer: object,
    sequencer_key: str,
    camera: object,
    camera_key: str,
    signal_plane: object,
    calibration: ResolvedArtifact,
    pulse_resource: ResolvedWorkspaceResource,
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
        sequence=pulse_resource.value,
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
    )


def _editor_factory(parent=None):
    from zlc_atom.nodes.scan.editor import scan_plan_editor_factory

    # One axis, and it is the release: this Task pairs the two probe windows
    # of a cycle, which is a statement about t_off and nothing else.  No
    # device ports either -- the board advances this plan from its own table.
    return scan_plan_editor_factory(
        parent,
        device_ports=False,
        only_port=PULSE_PARAM_FAMILY + T_OFF_PARAMETER,
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
                "fate:t_off": "x",
                "fate:repeat": "reduce",
                "fate:Site": "reduce",
                "reduction": Reduction.MEAN,
            },
        ),
    ),
    artifact_outputs=(
        ArtifactOutputSpec("artifact_path", TEMPERATURE_ARTIFACT_CONTRACT),
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
    ),
    build=_build,
    ui_contributions=(_editor_factory,),
    workspace_resources=(STEPPED_PULSE_RESOURCE,),
)


__all__ = ["DEFAULT_RELEASE_PLAN", "LOGIC_NODE", "TEMPERATURE_SCHEMA"]
