"""Automatically discovered camera device types."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.contract import CameraAcquisitionMode
from zlc_atom.devices.camera.dcam import DcamCameraAdapter, DcamCameraConfig
from zlc_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
from zlc_atom.devices.camera.virtual import VirtualCamera, VirtualCameraConfig
from zlc_atom.execution import DeviceIdentityEvidenceKind, PhysicalDeviceIdentity, ResourceKey, SafetyOperation, bind_verified_device
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf
from .world import SimulationWorld


# Each camera type declares the parameters IT has.  One shared schema meant the
# virtual camera's vocabulary -- a grid of simulated atoms and a random seed --
# was the only thing any camera could be configured with, so a real sensor could
# not be given a device index, an ROI, a binning or a readout speed.  An operator
# could see the qCMOS in the catalogue and had no way to set it up.

VIRTUAL_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField("frame_shape_yx", "pair", "Frame shape (Y,X)", (32, 48)),
        AuthoringField("grid_shape_yx", "pair", "Grid shape (Y,X)", (2, 3)),
        AuthoringField("seed", "int", "Seed", 0, minimum=0),
        AuthoringField("exposure_seconds", "float", "Exposure seconds", 0.02, minimum=1e-9),
    )
)

#: A physical sensor: which one, how long, how fast, and which corner of it.
#: ROI is four independently optional numbers, and leaving them unset means the
#: full sensor -- an operator who only wants a shorter exposure should not have
#: to state the geometry as well.
DCAM_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField("device_index", "int", "Device index", 0, minimum=0),
        AuthoringField("exposure_seconds", "float", "Exposure seconds", 0.02, minimum=1e-9),
        AuthoringField("readout_speed", "int", "Readout speed", 1, minimum=1),
        AuthoringField("binning", "int", "Binning", 1, minimum=1),
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

    authored = PYLON_CAMERA_SCHEMA.freeze(
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
    return _bind_camera(context, key, camera, f"pylon-camera:{key}", "camera.pylon")


def _bind_camera(context, key: str, camera: object, identity: str, type_id: str) -> InstalledLeaf:
    broker = context.broker
    binding, proof = bind_verified_device(
        broker,
        key=ResourceKey.parse(f"device/{key}"),
        identity_probe=lambda: PhysicalDeviceIdentity(identity, DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT),
        execute_command=lambda command: getattr(camera, "trigger")(int(command or 1)),
        capability_probe=lambda: {
            "camera.adapter": camera,
            "camera.working_point": camera.capture_working_point(),
        },
        close_session=lambda _command: getattr(camera, "finish_record_capture")() if getattr(camera, "capture_state")() else None,
        interrupt_operations={SafetyOperation.DISARM: lambda: getattr(camera, "finish_record_capture")()},
    )
    capabilities = dict(proof.snapshot)
    return InstalledLeaf(key, type_id, camera, capabilities, binding=binding, closer=getattr(camera, "close", None))


def _virtual_factory(context, key: str, values: dict) -> InstalledLeaf:
    if not isinstance(context.world, SimulationWorld):
        raise TypeError("camera.virtual requires the installation SimulationWorld")
    world = context.world
    authored = VIRTUAL_CAMERA_SCHEMA.freeze(values)
    geometry = world.geometry
    config = VirtualCameraConfig(
        frame_shape_yx=tuple(values.get("frame_shape_yx", geometry.image_shape_yx)),
        grid_shape_yx=tuple(values.get("grid_shape_yx", geometry.grid_shape_yx)),
        seed=int(authored["seed"]),
        exposure_seconds=float(authored["exposure_seconds"]),
    )
    if config.frame_shape_yx != geometry.image_shape_yx or config.grid_shape_yx != geometry.grid_shape_yx:
        raise ValueError("camera virtual geometry must share SimulationWorld geometry")
    camera = VirtualCamera(
        config,
        frame_source=lambda ordinal, exposure: world.render_frame(
            ordinal,
            exposure_seconds=exposure,
        ),
    )
    world.register_camera(camera)
    return _bind_camera(context, key, camera, f"virtual-camera:{key}", "camera.virtual")


def _dcam_factory(context, key: str, values: dict) -> InstalledLeaf:
    """Open a Hamamatsu qCMOS from a written-down configuration.

    A live driver object may be passed in for tests, but it is NOT required:
    demanding one meant a saved apparatus configuration could never be reopened,
    because a driver is not something a JSON file can hold.  With no driver the
    adapter opens the SDK itself from the authored device index, which is what a
    configuration is for.
    """

    driver = values.get("driver")
    authored = DCAM_CAMERA_SCHEMA.freeze(
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
    return _bind_camera(context, key, camera, f"dcam-camera:{key}", "camera.dcam")


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
    DeviceTypeDescriptor(
        "camera.virtual",
        "camera",
        VIRTUAL_CAMERA_SCHEMA,
        ("camera.adapter", "camera.working_point"),
        factory=_virtual_factory,
    ),
)

__all__ = [
    "DCAM_CAMERA_SCHEMA",
    "DEVICE_TYPES",
    "PYLON_CAMERA_SCHEMA",
    "VIRTUAL_CAMERA_SCHEMA",
]
