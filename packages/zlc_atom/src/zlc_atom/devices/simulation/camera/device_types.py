"""The virtual cameras this bench can install: the site sensor and the MOT monitor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.binding import bind_camera
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from ..authoring import simulation_world_config
from ..world import DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX, SimulationWorld
from .adapter import VirtualCamera, VirtualCameraConfig


VIRTUAL_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "exposure_seconds",
            "float",
            "Exposure seconds",
            0.02,
            minimum=1e-9,
        ),
    )
)

VIRTUAL_MOT_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "frame_shape_yx",
            "pair",
            "Frame shape (Y,X)",
            DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX,
        ),
        AuthoringField(
            "exposure_seconds",
            "float",
            "Exposure seconds",
            0.1,
            minimum=1e-9,
        ),
    )
)


def _camera_factory(context, key: str, values: dict) -> InstalledLeaf:
    if not isinstance(context.world, SimulationWorld):
        raise TypeError("camera.virtual requires the installation SimulationWorld")
    authored = VIRTUAL_CAMERA_SCHEMA.project_values(values)
    world = context.world
    geometry = world.geometry
    config = VirtualCameraConfig(
        frame_shape_yx=geometry.image_shape_yx,
        exposure_seconds=float(authored["exposure_seconds"]),
        # The world's own numbers, read from it rather than restated here:
        # it converts electrons to counts to make the frame, and this is the
        # same conversion said in the direction a reader needs it.
        offset_counts=world.offset_counts,
        electrons_per_count=world.conversion_e_per_count,
    )
    camera = VirtualCamera(
        config,
        frame_source=lambda exposure: world.render_frame(
            exposure_seconds=exposure,
        ),
    )
    world.register_camera(camera)
    return bind_camera(
        context,
        key,
        camera,
        f"virtual-camera:{key}",
        "camera.virtual",
    )


def _mot_camera_factory(context, key: str, values: dict) -> InstalledLeaf:
    if not isinstance(context.world, SimulationWorld):
        raise TypeError("camera.virtual_mot requires the installation SimulationWorld")
    authored = VIRTUAL_MOT_CAMERA_SCHEMA.project_values(values)
    world = context.world
    # The real MOT monitor is a Basler read out as Mono8: the pylon adapter
    # declares and enforces uint8 frames, so the virtual stand-in does too.
    config = VirtualCameraConfig(
        frame_shape_yx=tuple(authored["frame_shape_yx"]),
        exposure_seconds=float(authored["exposure_seconds"]),
        frame_dtype="|u1",
    )

    def render(*, exposure_seconds: float, occupancy=None):
        return world.render_mot_frame(
            exposure_seconds=exposure_seconds,
            occupancy=occupancy,
            frame_shape_yx=config.frame_shape_yx,
        )

    camera = VirtualCamera(
        config,
        frame_source=lambda exposure: render(exposure_seconds=exposure),
        free_running=True,
    )
    return bind_camera(
        context,
        key,
        camera,
        f"virtual-mot-camera:{key}",
        "camera.virtual_mot",
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "camera.virtual",
        "camera",
        VIRTUAL_CAMERA_SCHEMA,
        ("camera.adapter",),
        factory=_camera_factory,
        world_config=simulation_world_config,
    ),
    DeviceTypeDescriptor(
        "camera.virtual_mot",
        "camera",
        VIRTUAL_MOT_CAMERA_SCHEMA,
        ("camera.adapter",),
        factory=_mot_camera_factory,
        world_config=simulation_world_config,
    ),
)

__all__ = ["DEVICE_TYPES", "VIRTUAL_CAMERA_SCHEMA", "VIRTUAL_MOT_CAMERA_SCHEMA"]
