"""Automatically discovered camera device types."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.binding import bind_camera
from zlc_atom.devices.camera.dcam import DcamCameraAdapter, DcamCameraConfig
from zlc_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
from zlc_atom.devices.camera._dcam_driver import DcamSdkDriver
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
        # What one count is worth, from this sensor's datasheet: the
        # ORCA-Quest's ultra-quiet readout is 0.107 electrons per count over
        # an offset of 200, which is also what the virtual sensor applies
        # going the other way.  It is authored rather than read back because
        # it is a property of the SENSOR, not of the session, and a
        # measurement has to be able to offer the unit before anything is
        # open.  Cleared, this camera publishes counts and nothing else.
        AuthoringField(
            "offset_counts",
            "float",
            "Offset (counts)",
            200.0,
            required=False,
        ),
        AuthoringField(
            "electrons_per_count",
            "float",
            "Electrons per count",
            0.107,
            required=False,
            minimum=1e-12,
        ),
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


def _discover_dcam() -> tuple[DeviceInstanceConfig, ...]:
    # A count read.  Scanning used to start the whole vendor runtime and tear
    # it down again per button press, which on a bench with a qCMOS attached
    # is most of what "scan hardware" cost -- and it collided with any camera
    # that was open at the time.  The runtime belongs to the process now.
    driver = DcamSdkDriver()
    driver.initialize()
    return tuple(
        DeviceInstanceConfig(
            instance_id=f"dcam_{index}",
            role=f"dcam_{index}",
            type_id="camera.dcam",
            parameters=DCAM_CAMERA_SCHEMA.project_values({"device_index": index}),
        )
        for index in range(driver.device_count)
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
            gain_db=float(authored["gain_db"]),
            trigger_source=str(authored["trigger_source"]),
            roi_xywh=_roi_xywh(authored),
            offset_counts=authored["offset_counts"],
            electrons_per_count=authored["electrons_per_count"],
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
            offset_counts=authored["offset_counts"],
            electrons_per_count=authored["electrons_per_count"],
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
)

__all__ = [
    "DCAM_CAMERA_SCHEMA",
    "DEVICE_TYPES",
    "PYLON_CAMERA_SCHEMA",
]
