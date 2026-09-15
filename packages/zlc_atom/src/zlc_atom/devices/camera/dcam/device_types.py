"""The Hamamatsu qCMOS this bench can install, and how to find one."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.binding import bind_camera
from zlc_atom.devices.camera.roi_grid import authored_roi_xywh
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from ._dcam_driver import DcamSdkDriver
from .adapter import DcamCameraAdapter, DcamCameraConfig


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


def _dcam_factory(context, key: str, values: dict) -> InstalledLeaf:
    """Open a Hamamatsu qCMOS from a written-down configuration.

    The adapter opens the SDK itself from the authored device index, which is
    what a configuration is for: a saved apparatus has to be reopenable
    tomorrow, and a driver is not something a JSON file can hold.

    DCAM addresses a camera by its index in the runtime's enumeration, so that
    index -- scoped to this process's DCAM runtime -- is the physical identity
    the broker guards; two keys opening one index are one camera.
    """

    authored = DCAM_CAMERA_SCHEMA.project_values(values)
    device_index = int(authored["device_index"])
    camera = DcamCameraAdapter(
        DcamCameraConfig(
            exposure_seconds=float(authored["exposure_seconds"]),
            readout_speed=int(authored["readout_speed"]),
            binning=1,
            roi_xywh=authored_roi_xywh(authored),
            device_index=device_index,
            offset_counts=authored["offset_counts"],
            electrons_per_count=authored["electrons_per_count"],
        ),
    )
    return bind_camera(
        context,
        key,
        camera,
        f"dcam-camera:index={device_index}",
        "camera.dcam",
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "camera.dcam",
        "camera",
        DCAM_CAMERA_SCHEMA,
        ("camera.adapter",),
        factory=_dcam_factory,
        discover=_discover_dcam,
    ),
)

__all__ = ["DCAM_CAMERA_SCHEMA", "DEVICE_TYPES"]
