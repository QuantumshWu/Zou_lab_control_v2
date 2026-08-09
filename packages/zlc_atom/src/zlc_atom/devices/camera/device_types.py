"""Automatically discovered camera device types."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.binding import bind_camera
from zlc_atom.devices.camera.dcam import DcamCameraAdapter, DcamCameraConfig
from zlc_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf


#: A physical sensor: which one, how long, how fast, and which corner of it.
#: ROI is four independently optional numbers, and leaving them unset means the
#: full sensor -- an operator who only wants a shorter exposure should not have
#: to state the geometry as well.
DCAM_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField("device_index", "int", "Device index", 0, minimum=0),
        AuthoringField("exposure_seconds", "float", "Exposure seconds", 0.02, minimum=1e-9),
        AuthoringField("readout_speed", "int", "Readout speed", 1, minimum=1),
        AuthoringField(
            "binning",
            "int",
            "Binning",
            1,
            choices=tuple(
                AuthoringChoice(value, f"{value} × {value}")
                for value in (1, 2, 4, 8, 16)
            ),
        ),
        AuthoringField("timeout_seconds", "float", "Timeout seconds", 2.0, minimum=1e-3),
        AuthoringField("roi_x", "int", "ROI x", None, required=False, minimum=0),
        AuthoringField("roi_y", "int", "ROI y", None, required=False, minimum=0),
        AuthoringField("roi_width", "int", "ROI width", None, required=False, minimum=1),
        AuthoringField("roi_height", "int", "ROI height", None, required=False, minimum=1),
    )
)


def _roi_xywh(authored: dict) -> tuple[int, int, int, int] | None:
    """The ROI the operator asked for, or None meaning the full sensor."""

    corners = (
        authored.get("roi_x"),
        authored.get("roi_y"),
        authored.get("roi_width"),
        authored.get("roi_height"),
    )
    if all(value is None for value in corners):
        return None
    if any(value is None for value in corners):
        raise ValueError(
            "an ROI needs all four of roi_x, roi_y, roi_width and roi_height, "
            "or none of them for the full sensor"
        )
    return tuple(int(value) for value in corners)


#: A Basler: which one by serial, how long, how it is gated, and what it sends.
PYLON_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField("serial", "str", "Serial number", "", required=False),
        AuthoringField("exposure_seconds", "float", "Exposure seconds", 0.005, minimum=1e-9),
        AuthoringField("trigger_source", "str", "Trigger source", "Software"),
        AuthoringField("pixel_format", "str", "Pixel format", "Mono8"),
        AuthoringField("timeout_seconds", "float", "Timeout seconds", 2.0, minimum=1e-3),
        AuthoringField("roi_x", "int", "ROI x", None, required=False, minimum=0),
        AuthoringField("roi_y", "int", "ROI y", None, required=False, minimum=0),
        AuthoringField("roi_width", "int", "ROI width", None, required=False, minimum=1),
        AuthoringField("roi_height", "int", "ROI height", None, required=False, minimum=1),
    )
)


def _pylon_factory(context, key: str, values: dict) -> InstalledLeaf:
    """Open a Basler from a written-down configuration.

    An already-attached camera object may be injected for tests; otherwise the
    serial in the configuration selects one, or the first attached camera is
    taken when no serial is given.
    """

    authored = PYLON_CAMERA_SCHEMA.project_values(
        {name: value for name, value in values.items() if name != "camera"}
    )
    camera = PylonCameraAdapter(
        PylonCameraConfig(
            serial=str(authored["serial"] or ""),
            exposure_seconds=float(authored["exposure_seconds"]),
            trigger_source=str(authored["trigger_source"]),
            pixel_format=str(authored["pixel_format"]),
            roi_xywh=_roi_xywh(authored),
            timeout_seconds=float(authored["timeout_seconds"]),
        ),
        camera=values.get("camera"),
    )
    camera.open()
    return bind_camera(context, key, camera, f"pylon-camera:{key}", "camera.pylon")


def _dcam_factory(context, key: str, values: dict) -> InstalledLeaf:
    """Open a Hamamatsu qCMOS from a written-down configuration.

    A live driver object may be passed in for tests, but it is NOT required:
    demanding one meant a saved apparatus configuration could never be reopened,
    because a driver is not something a JSON file can hold.  With no driver the
    adapter opens the SDK itself from the authored device index, which is what a
    configuration is for.
    """

    driver = values.get("driver")
    authored = DCAM_CAMERA_SCHEMA.project_values(
        {name: value for name, value in values.items() if name != "driver"}
    )
    camera = DcamCameraAdapter(
        DcamCameraConfig(
            exposure_seconds=float(authored["exposure_seconds"]),
            readout_speed=int(authored["readout_speed"]),
            binning=int(authored["binning"]),
            roi_xywh=_roi_xywh(authored),
            device_index=int(authored["device_index"]),
            timeout_seconds=float(authored["timeout_seconds"]),
        ),
        driver=driver,
    )
    return bind_camera(context, key, camera, f"dcam-camera:{key}", "camera.dcam")


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "camera.dcam",
        "camera",
        DCAM_CAMERA_SCHEMA,
        ("camera.adapter", "camera.working_point"),
        factory=_dcam_factory,
    ),
    DeviceTypeDescriptor(
        "camera.pylon",
        "camera",
        PYLON_CAMERA_SCHEMA,
        ("camera.adapter", "camera.working_point"),
        factory=_pylon_factory,
    ),
)

__all__ = [
    "DCAM_CAMERA_SCHEMA",
    "DEVICE_TYPES",
    "PYLON_CAMERA_SCHEMA",
]
