"""Discoverable calibration task with one live preview and saved results."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.photoelectrons import (
    PHOTOELECTRONS,
    photoelectron_switch,
    resolve_photoelectron_availability,
)
from zlc_atom.nodes._framework.descriptor import (
    ArtifactOutputSpec,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    ResolvedWorkspaceResource,
    NodePreviewSpec,
    WorkspaceResourceSpec,
)
from zlc_pulse import PulseSequence

from .artifact import CALIBRATION_ARTIFACT_CODEC
from .calibration import ReadoutModelKind
from .outputs import CAPTURE_PREVIEW_DECLARATION
from .pulse import load_calibration_pulse_template
from .task import (
    FRAMES_FROM_CAMERA,
    FRAMES_FROM_FOLDER,
    CalibrationRequest,
    CalibrationTask,
)


_CALIBRATION_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    "zlc.pulse/calibration",
    "pulses",
    (".json",),
    load_calibration_pulse_template,
    argument_name="pulse_resource",
)
def _validate_calibration(values: dict[str, object]) -> None:
    if float(values["readout_exposure_seconds"]) > float(
        values["reference_exposure_seconds"]
    ):
        raise ValueError("readout exposure cannot exceed reference exposure")


CALIBRATION_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse_template",
            "resource",
            "Calibration pulse template",
            "imaging_template.json",
            required=True,
        ),
        AuthoringField("repeats", "int", "Samples", 200, minimum=1),
        AuthoringField(
            "reference_exposure_seconds",
            "float",
            "Reference exposure seconds",
            0.02,
            minimum=1e-9,
        ),
        AuthoringField(
            "readout_exposure_seconds",
            "float",
            "Readout exposure seconds",
            0.005,
            minimum=1e-9,
        ),
        # Which API slot of the chosen pulse carries each exposure, BY NUMBER
        # -- slot 1 is the first parameter the pulse declares, whatever its
        # author called it.  That is what lets an operator point this task at
        # their own imaging pulse: three slots is three slots.
        AuthoringField(
            "reference_before_slot",
            "int",
            "Reference exposure slot (before)",
            1,
            minimum=1,
        ),
        AuthoringField(
            "readout_slot",
            "int",
            "Readout exposure slot",
            2,
            minimum=1,
        ),
        AuthoringField(
            "reference_after_slot",
            "int",
            "Reference exposure slot (after)",
            3,
            minimum=1,
        ),
        AuthoringField(
            "default_model_kind",
            "choice",
            "Default readout model",
            # The matched filter, per site: it weights each pixel by how much
            # signal that pixel actually carries, which is what the readout
            # is for.  Box stays offered -- it is the one to fall back to
            # when the PSF fit is in doubt -- but it is not the default.
            ReadoutModelKind.PER_SITE_PSF.value,
            choices=(
                AuthoringChoice(ReadoutModelKind.BOX.value, "Box"),
                AuthoringChoice(
                    ReadoutModelKind.PER_SITE_PSF.value,
                    "Per-site PSF",
                ),
                AuthoringChoice(
                    ReadoutModelKind.UNIFORM_PSF.value,
                    "Uniform PSF",
                ),
            ),
        ),
        AuthoringField(
            "threshold_method",
            "choice",
            "Threshold method",
            "empirical",
            choices=(
                AuthoringChoice("empirical", "Empirical"),
                AuthoringChoice("gaussian", "Gaussian"),
            ),
        ),
        AuthoringField(
            "box_half_width",
            "int",
            "Box half-width",
            1,
            minimum=0,
        ),
        AuthoringField(
            "box_reducer",
            "choice",
            "Box reducer",
            "mean",
            choices=(
                AuthoringChoice("mean", "Mean"),
                AuthoringChoice("sum", "Sum"),
                AuthoringChoice("median", "Median"),
                AuthoringChoice("max", "Maximum"),
            ),
        ),
        AuthoringField(
            "psf_half_width",
            "int",
            "PSF half-width",
            3,
            minimum=0,
        ),
        AuthoringField(
            "psf_padding",
            "int",
            "PSF background padding",
            3,
            minimum=1,
        ),
        AuthoringField(
            "detection_spot_sigma",
            "float",
            "Detection spot sigma",
            1.0,
            minimum=1e-9,
        ),
        AuthoringField(
            "detection_sigma",
            "float",
            "Detection threshold sigma",
            6.0,
            minimum=1e-9,
        ),
        # Where the frames come from, and whether they are kept.  Detection
        # settings are the part of a calibration an operator retunes most, and
        # retuning them used to mean holding the bench for another few hundred
        # samples; a folder of saved ones is calibrated again in seconds.
        AuthoringField(
            "frame_source",
            "choice",
            "Frames",
            FRAMES_FROM_CAMERA,
            choices=(
                AuthoringChoice(FRAMES_FROM_CAMERA, "Acquire new frames"),
                AuthoringChoice(FRAMES_FROM_FOLDER, "Calibrate saved frames"),
            ),
        ),
        AuthoringField(
            "saved_frames_path",
            "folder",
            "Saved frames folder",
            # Where runs are filed: saved samples live under the day folder of
            # the run that took them, so this is the place to start looking.
            "data",
            enabled_when=("frame_source", (FRAMES_FROM_FOLDER,)),
        ),
        AuthoringField(
            "save_frames",
            "bool",
            "Save every sample",
            False,
            enabled_when=("frame_source", (FRAMES_FROM_CAMERA,)),
        ),
        photoelectron_switch(enabled_when=("frame_source", (FRAMES_FROM_CAMERA,))),
    ),
    validator=_validate_calibration,
)


def _build(
    *,
    camera: object,
    camera_key: str,
    sequencer: object,
    sequencer_key: str,
    pulse_resource: ResolvedWorkspaceResource,
    signal_plane: object,
    **values: object,
) -> CalibrationTask:
    authored = CALIBRATION_SCHEMA.project_values(values)
    if (
        not isinstance(pulse_resource, ResolvedWorkspaceResource)
        or pulse_resource.contract_id
        != _CALIBRATION_PULSE_RESOURCE.contract_id
        or not isinstance(pulse_resource.value, PulseSequence)
    ):
        raise TypeError("pulse_resource must be a resolved calibration pulse")
    return CalibrationTask(
        camera=camera,  # type: ignore[arg-type]
        sequencer=sequencer,
        request=CalibrationRequest(
            camera_key=camera_key,
            sequencer_key=sequencer_key,
            pulse_template=pulse_resource.path.name,
            repeats=int(authored["repeats"]),
            reference_exposure_seconds=float(
                authored["reference_exposure_seconds"]
            ),
            readout_exposure_seconds=float(authored["readout_exposure_seconds"]),
            reference_before_slot=int(authored["reference_before_slot"]),
            readout_slot=int(authored["readout_slot"]),
            reference_after_slot=int(authored["reference_after_slot"]),
            default_model_kind=ReadoutModelKind(authored["default_model_kind"]),
            threshold_method=str(authored["threshold_method"]),
            box_half_width=int(authored["box_half_width"]),
            box_reducer=str(authored["box_reducer"]),
            psf_half_width=int(authored["psf_half_width"]),
            psf_padding=int(authored["psf_padding"]),
            detection_spot_sigma=float(authored["detection_spot_sigma"]),
            detection_sigma=float(authored["detection_sigma"]),
            save_frames=bool(authored["save_frames"]),
            photoelectrons=bool(authored[PHOTOELECTRONS]),
            frame_source=str(authored["frame_source"]),
            saved_frames_path=str(authored["saved_frames_path"]),
        ),
        pulse_sequence=pulse_resource.value,
        pulse_path=pulse_resource.path,
        signal_plane=signal_plane,
    )


LOGIC_NODE = LogicNodeDescriptor(
    "calibration",
    NodeKind.TASK,
    CALIBRATION_SCHEMA,
    outputs=(CAPTURE_PREVIEW_DECLARATION,),
    # Long reference, short readout, long reference: three frames whose whole
    # point is to be compared with each other, so they are faceted rather than
    # reduced.  Pinned to "image" they were averaged into one picture; left
    # unpinned they were averaged too, because a plotting package reads shape
    # and this is physics.
    node_previews=(
        NodePreviewSpec(CAPTURE_PREVIEW_DECLARATION, "facet_grid"),
    ),
    artifact_outputs=(
        ArtifactOutputSpec("artifact_path", CALIBRATION_ARTIFACT_CODEC.contract_id),
    ),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera"),
        DeviceRequirement("sequencer.streamer", "sequencer"),
    ),
    build=_build,
    workspace_resources=(_CALIBRATION_PULSE_RESOURCE,),
    # Whether it can read that way is the camera's answer: calibration keeps
    # no conversion of its own, so a bench that has not configured one cannot
    # switch this on here either.
    resolve_field_availability=resolve_photoelectron_availability,
)


__all__ = ["CALIBRATION_SCHEMA", "LOGIC_NODE"]
