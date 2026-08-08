"""Discoverable artifact-only calibration task descriptor."""

from __future__ import annotations

from pathlib import Path

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    ArtifactOutputSpec,
    DeviceAccess,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
)

from .task import CalibrationRequest, CalibrationTask


_ROI_FIELDS = ("roi_x", "roi_y", "roi_width", "roi_height")


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
            "text",
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
            "integration_method",
            "choice",
            "Integration method",
            "box",
            choices=("box", "psf", "uniform_psf"),
        ),
        AuthoringField(
            "threshold_method",
            "choice",
            "Threshold method",
            "empirical",
            choices=("empirical", "gaussian"),
        ),
        AuthoringField(
            "integration_half_width",
            "int",
            "Integration half-width",
            1,
            minimum=0,
        ),
        AuthoringField(
            "reducer",
            "choice",
            "Box reducer",
            "mean",
            choices=("mean", "sum", "median", "max"),
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
    pulse_search_paths: object,
    artifact_directory: object,
    **values: object,
) -> CalibrationTask:
    authored = CALIBRATION_SCHEMA.freeze(values)
    roi_values = tuple(authored[name] for name in _ROI_FIELDS)
    roi = (
        None
        if all(value is None for value in roi_values)
        else tuple(int(value) for value in roi_values)
    )
    if isinstance(pulse_search_paths, (str, Path)):
        paths = (pulse_search_paths,)
    else:
        paths = tuple(pulse_search_paths)  # type: ignore[arg-type]
    return CalibrationTask(
        camera=camera,  # type: ignore[arg-type]
        sequencer=sequencer,
        request=CalibrationRequest(
            camera_key=camera_key,
            sequencer_key=sequencer_key,
            pulse_template=str(authored["pulse_template"]),
            repeats=int(authored["repeats"]),
            reference_exposure_seconds=float(
                authored["reference_exposure_seconds"]
            ),
            readout_exposure_seconds=float(authored["readout_exposure_seconds"]),
            roi_xywh=roi,  # type: ignore[arg-type]
            integration_method=str(authored["integration_method"]),
            threshold_method=str(authored["threshold_method"]),
            integration_half_width=int(authored["integration_half_width"]),
            reducer=str(authored["reducer"]),
            detection_spot_sigma=float(authored["detection_spot_sigma"]),
            detection_min_distance=int(authored["detection_min_distance"]),
            detection_sigma=float(authored["detection_sigma"]),
            timeout_seconds=float(authored["timeout_seconds"]),
        ),
        pulse_search_paths=paths,
        artifact_directory=artifact_directory,  # type: ignore[arg-type]
    )


LOGIC_NODE = LogicNodeDescriptor(
    "calibration",
    NodeKind.TASK,
    CALIBRATION_SCHEMA,
    outputs=(),
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
)


__all__ = ["CALIBRATION_SCHEMA", "LOGIC_NODE"]
