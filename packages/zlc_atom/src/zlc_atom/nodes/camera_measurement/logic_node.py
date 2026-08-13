"""Discoverable camera measurement descriptor."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral

from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_runtime import SelectionRange, SelectionState

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DeviceAccess,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    NodePreviewSpec,
    OutputSpec,
    SelectionMapping,
)

from .measurement import (
    CAMERA_FRAMES_OUTPUT,
    CameraMeasurementNode,
    CameraMeasurementRequest,
)


_ROI_FIELDS = ("roi_x", "roi_y", "roi_width", "roi_height")


def _validate_measurement(values: dict[str, object]) -> None:
    roi = tuple(values[name] for name in _ROI_FIELDS)
    if any(value is None for value in roi) and not all(value is None for value in roi):
        raise ValueError("camera ROI requires all four fields or none for the full sensor")


def _spatial_range(selection: SelectionState, role: str) -> SelectionRange:
    matches = tuple(
        value
        for value in selection.ranges
        if value.axis == role or value.axis.endswith(f".{role}")
    )
    if len(matches) != 1:
        raise ValueError(f"image area must contain exactly one {role!r} range")
    return matches[0]


def _selected_sensor_interval(
    value: SelectionRange,
    role: str,
    *,
    binning: int,
    sensor_size: int,
) -> tuple[int, int]:
    # A frame's axes carry the sensor pixels it covers, so a region drawn on
    # it is already stated in sensor pixels and needs no conversion: it is the
    # ROI.  This used to add the current ROI's origin, because the picture was
    # indexed from zero -- once the frame says where it is, adding the origin
    # again shifts every region by the origin, which is why a region drawn
    # after one ROI change landed somewhere else on the next.
    # Producer ROI follows the actual canvas rectangle, including viewport or
    # Area bounds outside the current frame.  Only the physical sensor clips
    # it; data-derived ROI/Fit signals keep their separate data-domain slicing.
    # Each coordinate names the FIRST sensor pixel of its (possibly binned)
    # sample, so the selected rectangle ends one whole sample past the last.
    start = max(0, math.ceil(value.lower))
    stop = min(sensor_size, math.floor(value.upper) + binning)
    if stop <= start:
        raise ValueError(f"{role} selection does not intersect the camera sensor")
    return start, stop


def _current_sensor_shape(context: Mapping[str, object]) -> tuple[int, int]:
    raw = context.get("sensor_shape_yx")
    try:
        values = tuple(raw)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("sensor_shape_yx must contain two positive integers") from error
    if (
        len(values) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in values)
        or any(int(value) <= 0 for value in values)
    ):
        raise ValueError("sensor_shape_yx must contain two positive integers")
    return int(values[0]), int(values[1])


def _current_origin(context: Mapping[str, object]) -> tuple[int, int]:
    raw = context.get("roi_origin_yx")
    if raw is None:
        raise ValueError("camera image selection requires current roi_origin_yx")
    try:
        values = tuple(raw)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("roi_origin_yx must contain two non-negative integers") from error
    if (
        len(values) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in values)
        or any(int(value) < 0 for value in values)
    ):
        raise ValueError("roi_origin_yx must contain two non-negative integers")
    origin_y, origin_x = (int(value) for value in values)
    return origin_x, origin_y


def _current_binning(context: Mapping[str, object]) -> tuple[int, int]:
    raw = context.get("binning_yx")
    if raw is None:
        raise ValueError("camera image selection requires current binning_yx")
    try:
        values = tuple(raw)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("binning_yx must contain two positive integers") from error
    if (
        len(values) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in values)
        or any(int(value) <= 0 for value in values)
    ):
        raise ValueError("binning_yx must contain two positive integers")
    return int(values[0]), int(values[1])


def _image_area_to_roi_patch(
    selection: SelectionState,
    draft: Mapping[str, object],
    context: Mapping[str, object],
) -> dict[str, int]:
    x_range = _spatial_range(selection, SPATIAL_X.value)
    y_range = _spatial_range(selection, SPATIAL_Y.value)
    del draft
    sensor_height, sensor_width = _current_sensor_shape(context)
    binning_y, binning_x = _current_binning(context)
    x_start, x_stop = _selected_sensor_interval(
        x_range,
        SPATIAL_X.value,
        binning=binning_x,
        sensor_size=sensor_width,
    )
    y_start, y_stop = _selected_sensor_interval(
        y_range,
        SPATIAL_Y.value,
        binning=binning_y,
        sensor_size=sensor_height,
    )
    return {
        "roi_x": x_start,
        "roi_y": y_start,
        "roi_width": x_stop - x_start,
        "roi_height": y_stop - y_start,
    }


def _applied_roi_values(context: Mapping[str, object]) -> dict[str, int]:
    """The crop of the sensor this run is actually taking."""

    origin_x, origin_y = _current_origin(context)
    raw = context.get("roi_shape_yx")
    if raw is None:
        raise ValueError("camera ROI readback requires current roi_shape_yx")
    height, width = (int(value) for value in tuple(raw))  # type: ignore[arg-type]
    if height <= 0 or width <= 0:
        raise ValueError("roi_shape_yx must contain two positive integers")
    return {
        "roi_x": origin_x,
        "roi_y": origin_y,
        "roi_width": width,
        "roi_height": height,
    }


_IMAGE_AREA_TO_ROI = SelectionMapping(
    plot_kind="image",
    selector_kind="area",
    draft_fields=_ROI_FIELDS,
    map_patch=_image_area_to_roi_patch,
    applied_values=_applied_roi_values,
)


CAMERA_MEASUREMENT_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "exposure_seconds", "float", "Exposure seconds", 0.1, minimum=1e-9
        ),
        AuthoringField("roi_x", "int", "ROI x", None, required=False, minimum=0),
        AuthoringField("roi_y", "int", "ROI y", None, required=False, minimum=0),
        AuthoringField("roi_width", "int", "ROI width", None, required=False, minimum=1),
        AuthoringField("roi_height", "int", "ROI height", None, required=False, minimum=1),
        AuthoringField("repeat", "int", "Repeat", 0, minimum=0),
        AuthoringField("frames_per_cycle", "int", "Frames per cycle", 1, minimum=1),
    ),
    validator=_validate_measurement,
)


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
        ),
        signal_plane=signal_plane,
    )


def _frames_preview(_values) -> tuple[NodePreviewSpec, ...]:
    """What Start puts on screen: the cycle this run is about to take.

    A cycle's frames are its point rows, one picture each, and that is true
    of a cycle of one as much as a cycle of three.  Naming the kind by the
    count made the panel change shape when an operator changed the count --
    restart a measurement with three frames where there had been one and the
    panel that was an image had to be rebuilt as a grid, losing everything
    set on it.  One kind, whatever the count.
    """

    return (NodePreviewSpec(CAMERA_FRAMES_OUTPUT.name, "facet_grid"),)


LOGIC_NODE = LogicNodeDescriptor(
    "camera_measurement",
    NodeKind.MEASUREMENT,
    CAMERA_MEASUREMENT_SCHEMA,
    # One output whatever the cycle size: the frames live on the dataset's
    # READOUT_EVENT axis, so the signal vocabulary no longer changes with
    # the acquisition configuration.
    outputs=(OutputSpec(CAMERA_FRAMES_OUTPUT.name, CAMERA_FRAMES_OUTPUT.contract_id),),
    resolve_node_previews=_frames_preview,
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera", DeviceAccess.EXCLUSIVE),
    ),
    build=_build,
    selection_mappings=(_IMAGE_AREA_TO_ROI,),
)

__all__ = ["CAMERA_MEASUREMENT_SCHEMA", "LOGIC_NODE"]
