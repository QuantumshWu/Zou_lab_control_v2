"""Automatically discovered camera device types."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.binding import bind_camera
from zlc_atom.devices.camera.dcam import DcamCameraAdapter, DcamCameraConfig
from zlc_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
from zlc_atom.devices.camera._dcam_driver import DcamSdkDriver
from zlc_atom.devices.camera.endpoint import DEFAULT_HOST, DEFAULT_PORT
from zlc_atom.devices.camera.remote import RemoteCameraAdapter
from zlc_atom.install.configuration import DeviceInstanceConfig
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


#: A Basler: which one by serial, how long, and which external line gates finite
#: acquisition.  Mono8 and the SDK timeout are adapter policy, not authoring.
PYLON_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField("serial", "str", "Serial number", "", required=True),
        AuthoringField("exposure_seconds", "float", "Exposure seconds", 0.1, minimum=1e-9),
        AuthoringField("trigger_source", "str", "Trigger source", "Line1"),
        AuthoringField("roi_x", "int", "ROI x", None, required=False, minimum=0),
        AuthoringField("roi_y", "int", "ROI y", None, required=False, minimum=0),
        AuthoringField("roi_width", "int", "ROI width", None, required=False, minimum=1),
        AuthoringField("roi_height", "int", "ROI height", None, required=False, minimum=1),
    )
)


def _discover_dcam() -> tuple[DeviceInstanceConfig, ...]:
    driver = DcamSdkDriver()
    owned = driver.initialize()
    try:
        return tuple(
            DeviceInstanceConfig(
                instance_id=f"dcam_{index}",
                role=f"dcam_{index}",
                type_id="camera.dcam",
                parameters=DCAM_CAMERA_SCHEMA.project_values({"device_index": index}),
            )
            for index in range(driver.device_count)
        )
    finally:
        if owned:
            driver.uninitialize()


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

    An already-attached camera object may be injected for tests; otherwise the
    serial in the configuration selects exactly one camera.
    """

    authored = PYLON_CAMERA_SCHEMA.project_values(
        {name: value for name, value in values.items() if name != "camera"}
    )
    camera = PylonCameraAdapter(
        PylonCameraConfig(
            serial=str(authored["serial"]),
            exposure_seconds=float(authored["exposure_seconds"]),
            trigger_source=str(authored["trigger_source"]),
            roi_xywh=_roi_xywh(authored),
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
            binning=1,
            roi_xywh=_roi_xywh(authored),
            device_index=int(authored["device_index"]),
        ),
        driver=driver,
    )
    return bind_camera(context, key, camera, f"dcam-camera:{key}", "camera.dcam")


#: A camera reached over the network, by the camera server that owns it.
#: Writing the endpoint down is what lets an apparatus configuration be saved
#: and reopened tomorrow, exactly as the hardware sequencer does; which driver
#: sits behind the endpoint is the server machine's business, not authoring's.
REMOTE_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField("host", "str", "Camera server host", DEFAULT_HOST),
        AuthoringField("port", "int", "Camera server port", DEFAULT_PORT, minimum=1, maximum=65535),
    )
)


def _remote_factory(context, key: str, values: dict) -> InstalledLeaf:
    """Dial a camera server from a written-down endpoint.

    Unlike the sequencer, no injected dialler is needed: the remote client
    lives in this same package, so the endpoint alone is a complete request.
    """

    authored = REMOTE_CAMERA_SCHEMA.project_values(values)
    camera = RemoteCameraAdapter(str(authored["host"]), int(authored["port"]))
    camera.open()
    return bind_camera(context, key, camera, f"remote-camera:{key}", "camera.remote")


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "camera.dcam",
        "camera",
        DCAM_CAMERA_SCHEMA,
        ("camera.adapter", "camera.working_point"),
        factory=_dcam_factory,
        discover=_discover_dcam,
    ),
    DeviceTypeDescriptor(
        "camera.pylon",
        "camera",
        PYLON_CAMERA_SCHEMA,
        ("camera.adapter", "camera.working_point"),
        factory=_pylon_factory,
        discover=_discover_pylon,
    ),
    DeviceTypeDescriptor(
        "camera.remote",
        "camera",
        REMOTE_CAMERA_SCHEMA,
        ("camera.adapter", "camera.working_point"),
        factory=_remote_factory,
    ),
)

__all__ = [
    "DCAM_CAMERA_SCHEMA",
    "DEVICE_TYPES",
    "PYLON_CAMERA_SCHEMA",
    "REMOTE_CAMERA_SCHEMA",
]
