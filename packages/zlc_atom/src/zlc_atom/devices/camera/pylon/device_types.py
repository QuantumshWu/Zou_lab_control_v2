"""The Basler this bench can install, and how to find one."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.binding import bind_camera
from zlc_atom.devices.camera.roi_grid import authored_roi_xywh
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from .adapter import PylonCameraAdapter, PylonCameraConfig


#: A Basler: which one by serial, how long, and which external line gates finite
#: acquisition.  Mono8 and the SDK timeout are adapter policy, not authoring.
PYLON_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField("serial", "str", "Serial number", "", required=True),
        AuthoringField("exposure_seconds", "float", "Exposure seconds", 0.1, minimum=1e-9),
        # No bounds written down: the sensor's own gain limits differ by model
        # and pixel format, and the camera refuses what it cannot do.
        AuthoringField("gain_db", "float", "Gain (dB)", 0.0),
        AuthoringField("trigger_source", "str", "Trigger source", "Line1"),
        AuthoringField("roi_x", "int", "ROI x", None, required=False, minimum=0),
        AuthoringField("roi_y", "int", "ROI y", None, required=False, minimum=0),
        AuthoringField("roi_width", "int", "ROI width", None, required=False, minimum=1),
        AuthoringField("roi_height", "int", "ROI height", None, required=False, minimum=1),
        # Unset by default, which is the honest answer for most machine-vision
        # sensors: no conversion stated, so its frames are the counts they are.
        AuthoringField(
            "offset_counts",
            "float",
            "Offset (counts)",
            None,
            required=False,
        ),
        AuthoringField(
            "electrons_per_count",
            "float",
            "Electrons per count",
            None,
            required=False,
            minimum=1e-12,
        ),
    )
)


def _discover_pylon() -> tuple[DeviceInstanceConfig, ...]:
    from pypylon import pylon

    factory = pylon.TlFactory.GetInstance()
    return tuple(
        DeviceInstanceConfig(
            instance_id=f"pylon_{serial.replace('/', '_')}",
            role=f"pylon_{serial.replace('/', '_')}",
            type_id="camera.pylon",
            parameters=PYLON_CAMERA_SCHEMA.project_values({"serial": serial}),
        )
        for info in factory.EnumerateDevices()
        for serial in (str(info.GetSerialNumber()),)
    )


def _pylon_factory(context, key: str, values: dict) -> InstalledLeaf:
    """Open a Basler from a written-down configuration.

    The serial in the configuration selects exactly one camera.  That serial
    is also the physical identity the broker guards: the logical key is what
    an apparatus calls the device, and two keys naming one serial are one
    camera, which the broker can only refuse if it is told the serial.
    """

    authored = PYLON_CAMERA_SCHEMA.project_values(values)
    camera = PylonCameraAdapter(
        PylonCameraConfig(
            serial=str(authored["serial"]),
            exposure_seconds=float(authored["exposure_seconds"]),
            gain_db=float(authored["gain_db"]),
            trigger_source=str(authored["trigger_source"]),
            roi_xywh=authored_roi_xywh(authored),
            offset_counts=authored["offset_counts"],
            electrons_per_count=authored["electrons_per_count"],
        ),
    )
    camera.open()
    return bind_camera(
        context,
        key,
        camera,
        f"pylon-camera:serial={camera.config.serial}",
        "camera.pylon",
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "camera.pylon",
        "camera",
        PYLON_CAMERA_SCHEMA,
        ("camera.adapter",),
        factory=_pylon_factory,
        discover=_discover_pylon,
    ),
)

__all__ = ["DEVICE_TYPES", "PYLON_CAMERA_SCHEMA"]
