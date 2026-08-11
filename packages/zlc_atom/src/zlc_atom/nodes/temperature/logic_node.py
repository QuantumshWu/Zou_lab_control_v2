"""The temperature Task: release-recapture over the trap-off time.

What the operator authors is the template, the release times to play, how many
whole sweeps of them to take, which readout model to judge the frames with,
and the one apparatus fact a recapture curve cannot measure about itself --
how far an atom may drift and still be recaptured.

There is no exposure here on purpose.  The readout exposure is the one the
CALIBRATION measured its thresholds at, and it reaches this Task inside the
artifact; the camera's own exposure belongs to whoever owns the camera, which
is the monitor node publishing the frames this Task watches.  A third copy on
this form would be a number nothing enforces.
"""

from __future__ import annotations

from zlc_pulse import PulseSequence

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    ArtifactInputSpec,
    ArtifactOutputSpec,
    DatasetInputSpec,
    DeviceAccess,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    NodePreviewSpec,
    OutputSpec,
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
    SCAN_OUTPUT,
    SCAN_PULSE_CONTRACT,
    SCAN_PULSE_RESOURCE,
)

from .task import (
    SURVIVAL_OUTPUT,
    SURVIVAL_RATE_OUTPUT,
    TEMPERATURE_ARTIFACT_CONTRACT,
    T_OFF_PARAMETER,
    TemperatureTask,
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
        # The values, in the unit the template's own release parameter
        # declares; the plan is bound against that parameter's hard limits,
        # so a release the board cannot play is refused before anything arms.
        AuthoringField(
            "t_off",
            "text",
            "Release times",
            "0.5, 1, 2, 3",
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
        # The trap's own reach, which no recapture curve can measure about
        # itself: it is what turns a 1/e time into a speed, and a speed into
        # a temperature.
        AuthoringField(
            "capture_radius_um",
            "float",
            "Recapture radius (um)",
            1.0,
            minimum=1e-3,
        ),
    )
)


def _release_times(payload: object) -> tuple[float, ...]:
    """The authored release times, as numbers, in the template's unit."""

    if isinstance(payload, (tuple, list)):
        parts: tuple[object, ...] = tuple(payload)
    else:
        parts = tuple(
            piece
            for piece in str(payload or "").replace(",", " ").split()
            if piece
        )
    if not parts:
        raise ValueError(
            f'the {T_OFF_PARAMETER} release times are empty; they read like '
            '"0.5, 1, 2, 3"'
        )
    return tuple(float(part) for part in parts)


def _build(
    *,
    sequencer: object,
    signal_plane: object,
    source_signal: str,
    calibration: ResolvedArtifact,
    pulse_resource: ResolvedWorkspaceResource,
    artifact_directory: object,
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
    publication = signal_plane.latest_publication(source_signal)
    if publication is None or not signal_plane.is_generation_live(source_signal):
        raise ValueError(
            "the release-recapture Task reads the probe frames off a LIVE "
            f"signal, and {source_signal!r} is not live on this bench"
        )
    return TemperatureTask(
        sequencer=sequencer,
        signal_plane=signal_plane,
        signal_name=str(source_signal),
        source_generation=publication.event_ref.generation,
        sequence=pulse_resource.value,
        calibration=calibration.value,
        calibration_path=calibration.path,
        t_off_values=_release_times(authored["t_off"]),
        repeats=int(authored["repeats"]),
        model_kind=readout_model_kind_from_choice(authored["model_kind"]),
        # Micrometres are what an operator reads off a trap; the Task works in
        # SI, so the form's unit is converted exactly once, here.
        capture_radius_m=float(authored["capture_radius_um"]) * 1e-6,
        artifact_directory=artifact_directory,
    )


LOGIC_NODE = LogicNodeDescriptor(
    "temperature",
    NodeKind.TASK,
    TEMPERATURE_SCHEMA,
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
        OutputSpec(SCAN_OUTPUT.name, SCAN_OUTPUT.contract_id),
        OutputSpec(SURVIVAL_OUTPUT.name, SURVIVAL_OUTPUT.contract_id),
        OutputSpec(SURVIVAL_RATE_OUTPUT.name, SURVIVAL_RATE_OUTPUT.contract_id),
    ),
    # Three outputs, and the one an operator opens the node to watch is the
    # curve: the recapture fraction against the release time.  Per-site
    # survival and the frames themselves are what you go looking for once it
    # surprises you.
    node_previews=(NodePreviewSpec(SURVIVAL_RATE_OUTPUT.name),),
    artifact_outputs=(
        ArtifactOutputSpec("artifact_path", TEMPERATURE_ARTIFACT_CONTRACT),
    ),
    device_requirements=(
        DeviceRequirement(
            "sequencer.streamer",
            "sequencer",
            DeviceAccess.EXCLUSIVE,
        ),
    ),
    build=_build,
    workspace_resources=(SCAN_PULSE_RESOURCE,),
)


__all__ = ["LOGIC_NODE", "TEMPERATURE_SCHEMA"]
