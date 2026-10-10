"""A camera-compatible acquisition with internal first-frame rearrangement."""
from dataclasses import replace

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS
from zlc_atom.devices.camera.authoring import CAMERA_MEASUREMENT_SCHEMA, _validate_measurement, _IMAGE_AREA_TO_ROI
from zlc_atom.devices.camera.photoelectrons import resolve_photoelectron_availability
from zlc_atom.devices.slm.rearrangement import REARRANGEMENT_FIELDS, rearrangement_defaults
from zlc_atom.devices.slm.solver import SCIENCE_CONTEXT_ARTIFACT_CONTRACT, load_science_context, load_target
from zlc_atom.nodes._framework.descriptor import (
    ArtifactCodec, ArtifactInputSpec, DeviceRequirement, LogicNodeDescriptor, NodeKind, NodePreviewSpec,
)
from zlc_atom.nodes.calibration import CALIBRATION_ARTIFACT_CODEC
from zlc_atom.nodes.camera.measurement import CameraMeasurementRequest, CAMERA_FRAMES_OUTPUT

from .measurement import RearrangementMeasurement


_MOVEMENT_FIELDS = {"target_rows", "target_columns", "phase_method", "frame_mode", "motion_frames",
                    "max_camera_step", "frame_rate_hz", "minimum_separation", "nominal_playback_seconds",
                    "intensity_error_percent", "motion_intensity_error_percent"}
REARRANGEMENT_MEASUREMENT_SCHEMA = AuthoringSchema(
    tuple(replace(field, default=2, minimum=2,
                  description="First frame starts rearrangement; the last frame restores the initial SLM state. Pulse timing and any verification image remain operator-controlled.")
          if field.name == "frames_per_cycle" else field for field in CAMERA_MEASUREMENT_SCHEMA.fields)
    + tuple(field for field in REARRANGEMENT_FIELDS if field.name in _MOVEMENT_FIELDS),
    validator=_validate_measurement,
)


def _build(*, camera, camera_key, slm, slm_key, signal_plane, calibration, science_context,
           end_target=None, **values):
    authored = REARRANGEMENT_MEASUREMENT_SCHEMA.project_values(values)
    destination = None
    if end_target is not None:
        destination, kind = end_target.value
        if kind != "spots":
            raise ValueError("The end Target must be a spots Target")
    roi = tuple(authored[name] for name in ("roi_x", "roi_y", "roi_width", "roi_height"))
    return RearrangementMeasurement(camera=camera, slm=slm, slm_key=slm_key, signal_plane=signal_plane,
        request=CameraMeasurementRequest(camera_key=camera_key, exposure_seconds=authored["exposure_seconds"],
            roi_xywh=None if all(value is None for value in roi) else roi,
            repeat=authored["repeat"], frames_per_cycle=authored["frames_per_cycle"],
            photoelectrons=authored["photoelectrons"]),
        calibration=calibration.value, calibration_path=calibration.path,
        science_context=science_context.value, science_context_path=science_context.path,
        target_intensity=destination, target_path=None if end_target is None else str(end_target.path),
        intensity_tolerance=1. + authored["intensity_error_percent"] / 100.,
        motion_intensity_tolerance=1. + authored["motion_intensity_error_percent"] / 100.,
        **{key: value for key, value in authored.items() if key in _MOVEMENT_FIELDS
           - {"intensity_error_percent", "motion_intensity_error_percent", "nominal_playback_seconds"}})


LOGIC_NODE = LogicNodeDescriptor("rearrangement_measurement", NodeKind.MEASUREMENT,
    REARRANGEMENT_MEASUREMENT_SCHEMA, reports_ready=True,
    outputs=(CAMERA_FRAMES_OUTPUT,), node_previews=(NodePreviewSpec(CAMERA_FRAMES_OUTPUT, "facet_grid"),),
    input_specs=(
        ArtifactInputSpec("calibration_path", "Calibration", CALIBRATION_ARTIFACT_CODEC, argument_name="calibration"),
        ArtifactInputSpec("science_context_path", "SLM Science Context",
            ArtifactCodec(SCIENCE_CONTEXT_ARTIFACT_CONTRACT, "SLM Science Contexts (*.npz)", (".npz",), load_science_context),
            argument_name="science_context"),
        ArtifactInputSpec("end_target_path", "End Target (blank = grid)",
            ArtifactCodec("zlc.slm.target", "SLM Targets (*.json)", (".json",), load_target),
            required=False, argument_name="end_target"),
    ),
    device_requirements=(DeviceRequirement("camera.adapter", "camera", CAMERA_PROTECTED_FIELDS),
                         DeviceRequirement("slm.phase", "slm", ("phase", "wavelength_nm", "display_name", "width", "height"))),
    resolve_defaults=rearrangement_defaults, build=_build,
    selection_mappings=(_IMAGE_AREA_TO_ROI,), resolve_field_availability=resolve_photoelectron_availability,
)

__all__ = ["LOGIC_NODE", "REARRANGEMENT_MEASUREMENT_SCHEMA"]
