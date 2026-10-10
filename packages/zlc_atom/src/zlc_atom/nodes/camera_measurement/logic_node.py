"""Discoverable camera measurement descriptor."""

from __future__ import annotations

from zlc_atom.devices.camera import CAMERA_PROTECTED_FIELDS
from zlc_atom.devices.camera.photoelectrons import (
    PHOTOELECTRONS,
    resolve_photoelectron_availability,
)
from zlc_atom.nodes._framework.descriptor import (
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    NodePreviewSpec,
)

from zlc_atom.nodes.camera.measurement import (
    CAMERA_FRAMES_OUTPUT,
    CameraMeasurementNode,
    CameraMeasurementRequest,
)


from zlc_atom.devices.camera.authoring import CAMERA_MEASUREMENT_SCHEMA, _ROI_FIELDS, _IMAGE_AREA_TO_ROI


def _build(
    *,
    camera: object,
    camera_key: str,
    signal_plane: object,
    **values: object,
) -> CameraMeasurementNode:
    authored = CAMERA_MEASUREMENT_SCHEMA.project_values(values)
    roi_means = tuple(
        authored[name] for name in _ROI_FIELDS
    )
    roi = (
        None
        if all(value is None for value in roi_means)
        else tuple(int(value) for value in roi_means)
    )
    return CameraMeasurementNode(
        camera=camera,  # type: ignore[arg-type]
        request=CameraMeasurementRequest(
            camera_key=camera_key,
            exposure_seconds=float(authored["exposure_seconds"]),
            roi_xywh=roi,  # type: ignore[arg-type]
            repeat=int(authored["repeat"]),
            frames_per_cycle=int(authored["frames_per_cycle"]),
            photoelectrons=bool(authored[PHOTOELECTRONS]),
        ),
        signal_plane=signal_plane,
    )


LOGIC_NODE = LogicNodeDescriptor(
    "camera_measurement",
    NodeKind.MEASUREMENT,
    CAMERA_MEASUREMENT_SCHEMA,
    reports_ready=True,
    # One output whatever the cycle size: the frames live on the dataset's
    # READOUT_EVENT axis, so the signal vocabulary no longer changes with
    # the acquisition configuration.
    outputs=(CAMERA_FRAMES_OUTPUT,),
    node_previews=(NodePreviewSpec(CAMERA_FRAMES_OUTPUT, "facet_grid"),),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera", CAMERA_PROTECTED_FIELDS),
    ),
    build=_build,
    selection_mappings=(_IMAGE_AREA_TO_ROI,),
    # Whether this camera can be read that way is the camera's answer, not
    # this node's: the conversion is configured on the device, and a bench
    # that has not written it down cannot switch this on.
    resolve_field_availability=resolve_photoelectron_availability,
)

__all__ = ["CAMERA_MEASUREMENT_SCHEMA", "LOGIC_NODE"]
