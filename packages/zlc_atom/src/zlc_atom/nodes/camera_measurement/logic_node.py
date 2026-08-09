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
    OutputSpec,
    SelectionMapping,
)

from .measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
    camera_frame_output_declarations,
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


def _selected_pixel_interval(value: SelectionRange, role: str) -> tuple[int, int]:
    # SelectionRange is a closed coordinate interval.  Camera spatial axes are
    # implicit pixel centres (0..n-1), so this is the same interval resolution
    # used by zlc_data selection derivation: [ceil(low), floor(high)].
    start = math.ceil(value.lower)
    stop = math.floor(value.upper) + 1
    if stop <= start:
        raise ValueError(f"{role} selection contains no image pixels")
    return start, stop


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
    x_start, x_stop = _selected_pixel_interval(x_range, SPATIAL_X.value)
    y_start, y_stop = _selected_pixel_interval(y_range, SPATIAL_Y.value)
    del draft
    origin_x, origin_y = _current_origin(context)
    binning_y, binning_x = _current_binning(context)
    roi_x = origin_x + x_start * binning_x
    roi_y = origin_y + y_start * binning_y
    if roi_x < 0 or roi_y < 0:
        raise ValueError("image area maps outside the sensor origin")
    return {
        "roi_x": roi_x,
        "roi_y": roi_y,
        "roi_width": (x_stop - x_start) * binning_x,
        "roi_height": (y_stop - y_start) * binning_y,
    }


_IMAGE_AREA_TO_ROI = SelectionMapping(
    plot_kind="image",
    selector_kind="area",
    draft_fields=_ROI_FIELDS,
    map_patch=_image_area_to_roi_patch,
)


CAMERA_MEASUREMENT_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "exposure_seconds", "float", "Exposure seconds", 0.02, minimum=1e-9
        ),
        AuthoringField("roi_x", "int", "ROI x", None, required=False, minimum=0),
        AuthoringField("roi_y", "int", "ROI y", None, required=False, minimum=0),
        AuthoringField("roi_width", "int", "ROI width", None, required=False, minimum=1),
        AuthoringField("roi_height", "int", "ROI height", None, required=False, minimum=1),
        AuthoringField("repeat", "int", "Repeat", 1, minimum=0),
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
    roi_values = tuple(
        authored[name] for name in _ROI_FIELDS
    )
    roi = (
        None
        if all(value is None for value in roi_values)
        else tuple(int(value) for value in roi_values)
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


def _outputs(values: Mapping[str, object]) -> tuple[OutputSpec, ...]:
    return tuple(
        OutputSpec(value.name, value.contract_id)
        for value in camera_frame_output_declarations(
            int(values["frames_per_cycle"])
        )
    )


LOGIC_NODE = LogicNodeDescriptor(
    "camera_measurement",
    NodeKind.MEASUREMENT,
    CAMERA_MEASUREMENT_SCHEMA,
    resolve_outputs=_outputs,
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera", DeviceAccess.EXCLUSIVE),
    ),
    build=_build,
    selection_mappings=(_IMAGE_AREA_TO_ROI,),
)

__all__ = ["CAMERA_MEASUREMENT_SCHEMA", "LOGIC_NODE"]
