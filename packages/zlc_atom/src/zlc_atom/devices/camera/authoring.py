"""Shared camera acquisition fields and sensor-coordinate selection mapping."""
from collections.abc import Mapping
import math
from numbers import Integral

from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_runtime import SelectionRange, SelectionState
from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.photoelectrons import photoelectron_switch
from zlc_atom.nodes._framework.descriptor import SelectionMapping


_ROI_FIELDS = ("roi_x", "roi_y", "roi_width", "roi_height")


def _validate_measurement(values: dict[str, object]) -> None:
    roi = tuple(values[name] for name in _ROI_FIELDS)
    if any(value is None for value in roi) and not all(value is None for value in roi):
        raise ValueError("camera ROI requires all four fields or none for the full sensor")


def _spatial_range(selection: SelectionState, role: str) -> SelectionRange | None:
    matches = tuple(
        value
        for value in selection.ranges
        if value.axis == role or value.axis.endswith(f".{role}")
    )
    return matches[0] if len(matches) == 1 else None


def _selected_sensor_interval(
    value: SelectionRange,
    *,
    binning: int,
    sensor_size: int,
) -> tuple[int, int] | None:
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
        # A region drawn entirely off the sensor -- in the band beside the
        # picture, or past its edge -- names no crop.  That is an answer,
        # not a fault: raised, it escaped a worker thread while a gesture
        # was in flight and took the gesture's reply with it.
        return None
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
) -> dict[str, int] | None:
    x_range = _spatial_range(selection, SPATIAL_X.value)
    y_range = _spatial_range(selection, SPATIAL_Y.value)
    if x_range is None or y_range is None:
        # An image whose axes are not the sensor plane -- a frame or a site
        # put on one of them -- has an Area that names no crop.  It still
        # derives its ROI; raised here, it failed the gesture as an internal
        # error.
        return None
    del draft
    sensor_height, sensor_width = _current_sensor_shape(context)
    binning_y, binning_x = _current_binning(context)
    horizontal = _selected_sensor_interval(
        x_range,
        binning=binning_x,
        sensor_size=sensor_width,
    )
    vertical = _selected_sensor_interval(
        y_range,
        binning=binning_y,
        sensor_size=sensor_height,
    )
    if horizontal is None or vertical is None:
        return None
    x_start, x_stop = horizontal
    y_start, y_stop = vertical
    return {
        "roi_x": x_start,
        "roi_y": y_start,
        "roi_width": x_stop - x_start,
        "roi_height": y_stop - y_start,
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
            "exposure_seconds", "float", "Exposure seconds", 0.1, minimum=1e-9
        ),
        AuthoringField("roi_x", "int", "ROI x", None, required=False, minimum=0),
        AuthoringField("roi_y", "int", "ROI y", None, required=False, minimum=0),
        AuthoringField("roi_width", "int", "ROI width", None, required=False, minimum=1),
        AuthoringField("roi_height", "int", "ROI height", None, required=False, minimum=1),
        AuthoringField("repeat", "int", "Repeat", 0, minimum=0),
        AuthoringField("frames_per_cycle", "int", "Frames per cycle", 1, minimum=1),
        photoelectron_switch(),
    ),
    validator=_validate_measurement,
)
