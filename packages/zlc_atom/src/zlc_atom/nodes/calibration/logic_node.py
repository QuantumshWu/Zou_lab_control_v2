"""Discoverable calibration task with one live preview and saved results."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    ArtifactOutputSpec,
    DeviceAccess,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    OutputSpec,
    ResolvedWorkspaceResource,
    TaskPreviewSpec,
    WorkspaceResourceSpec,
)
from zlc_pulse import PulseSequence

from .calibration import ReadoutModelKind
from .outputs import CAPTURE_PREVIEW_DECLARATION
from .pulse import load_calibration_pulse_template
from .task import CalibrationRequest, CalibrationTask


_ROI_FIELDS = ("roi_x", "roi_y", "roi_width", "roi_height")
_CALIBRATION_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    "zlc.pulse.v1/calibration",
    "pulses",
    (".json",),
    load_calibration_pulse_template,
    argument_name="pulse_resource",
)


def _validate_calibration(values: dict[str, object]) -> None:
    roi = tuple(values[name] for name in _ROI_FIELDS)
    if any(value is None for value in roi) and not all(value is None for value in roi):
        raise ValueError("calibration ROI requires all four fields or none")
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
        AuthoringField("repeats", "int", "Samples", 30, minimum=1),
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
        AuthoringField("roi_x", "int", "ROI x", None, required=False, minimum=0),
        AuthoringField("roi_y", "int", "ROI y", None, required=False, minimum=0),
        AuthoringField(
            "roi_width", "int", "ROI width", None, required=False, minimum=1
        ),
        AuthoringField(
            "roi_height", "int", "ROI height", None, required=False, minimum=1
        ),
        AuthoringField(
            "default_model_kind",
            "choice",
            "Default readout model",
            ReadoutModelKind.BOX.value,
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
            "detection_min_distance",
            "int",
            "Detection minimum distance",
            3,
            minimum=1,
        ),
        AuthoringField(
            "detection_sigma",
            "float",
            "Detection threshold sigma",
            6.0,
            minimum=1e-9,
        ),
        AuthoringField(
            "timeout_seconds",
            "float",
            "Timeout seconds",
            2.0,
            minimum=0.001,
        ),
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
    artifact_directory: object,
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
    roi_values = tuple(authored[name] for name in _ROI_FIELDS)
    roi = (
        None
        if all(value is None for value in roi_values)
        else tuple(int(value) for value in roi_values)
    )
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
            roi_xywh=roi,  # type: ignore[arg-type]
            default_model_kind=ReadoutModelKind(authored["default_model_kind"]),
            threshold_method=str(authored["threshold_method"]),
            box_half_width=int(authored["box_half_width"]),
            box_reducer=str(authored["box_reducer"]),
            psf_half_width=int(authored["psf_half_width"]),
            psf_padding=int(authored["psf_padding"]),
            detection_spot_sigma=float(authored["detection_spot_sigma"]),
            detection_min_distance=int(authored["detection_min_distance"]),
            detection_sigma=float(authored["detection_sigma"]),
            timeout_seconds=float(authored["timeout_seconds"]),
        ),
        pulse_sequence=pulse_resource.value,
        pulse_path=pulse_resource.path,
        artifact_directory=artifact_directory,  # type: ignore[arg-type]
    )


LOGIC_NODE = LogicNodeDescriptor(
    "calibration",
    NodeKind.TASK,
    CALIBRATION_SCHEMA,
    outputs=(
        OutputSpec(
            CAPTURE_PREVIEW_DECLARATION.name,
            CAPTURE_PREVIEW_DECLARATION.contract_id,
        ),
    ),
    task_previews=(TaskPreviewSpec("capture_preview", "image"),),
    artifact_outputs=(
        ArtifactOutputSpec("artifact_path", "calibration.readout.v1"),
    ),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera", DeviceAccess.EXCLUSIVE),
        DeviceRequirement(
            "sequencer.streamer", "sequencer", DeviceAccess.EXCLUSIVE
        ),
    ),
    build=_build,
    workspace_resources=(_CALIBRATION_PULSE_RESOURCE,),
)


__all__ = ["CALIBRATION_SCHEMA", "LOGIC_NODE"]
