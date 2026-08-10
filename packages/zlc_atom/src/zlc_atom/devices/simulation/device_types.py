"""Automatically discovered virtual apparatus device types."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.binding import bind_camera
from zlc_atom.devices.sequencer.binding import bind_sequencer
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from .camera import VirtualCamera, VirtualCameraConfig
from .sequencer import VirtualSequencer
from .world import (
    DEFAULT_SIMULATION_GRID_SHAPE_YX,
    DEFAULT_SIMULATION_IMAGE_SHAPE_YX,
    DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX,
    SimulationGeometry,
    SimulationWorld,
    SimulationWorldConfig,
)


VIRTUAL_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "frame_shape_yx",
            "pair",
            "Frame shape (Y,X)",
            DEFAULT_SIMULATION_IMAGE_SHAPE_YX,
        ),
        AuthoringField(
            "grid_shape_yx",
            "pair",
            "Grid shape (Y,X)",
            DEFAULT_SIMULATION_GRID_SHAPE_YX,
        ),
        AuthoringField("seed", "int", "Seed", 0, minimum=0),
        AuthoringField(
            "exposure_seconds",
            "float",
            "Exposure seconds",
            0.02,
            minimum=1e-9,
        ),
    )
)

# A virtual sequencer has no physical endpoint or device-local parameters.
VIRTUAL_SEQUENCER_SCHEMA = AuthoringSchema(())

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
            0.05,
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
        frame_shape_yx=tuple(authored["frame_shape_yx"]),
        exposure_seconds=float(authored["exposure_seconds"]),
    )
    if config.frame_shape_yx != geometry.image_shape_yx:
        raise ValueError("virtual camera frame must share SimulationWorld geometry")
    camera = VirtualCamera(
        config,
        frame_source=lambda ordinal, exposure: world.render_frame(
            ordinal,
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


def _camera_world_config(values: dict) -> SimulationWorldConfig:
    authored = VIRTUAL_CAMERA_SCHEMA.project_values(values)
    return SimulationWorldConfig(
        SimulationGeometry(
            grid_shape_yx=tuple(authored["grid_shape_yx"]),
            image_shape_yx=tuple(authored["frame_shape_yx"]),
        ),
        seed=int(authored["seed"]),
    )


def _mot_camera_factory(context, key: str, values: dict) -> InstalledLeaf:
    if not isinstance(context.world, SimulationWorld):
        raise TypeError("camera.virtual_mot requires the installation SimulationWorld")
    authored = VIRTUAL_MOT_CAMERA_SCHEMA.project_values(values)
    world = context.world
    config = VirtualCameraConfig(
        frame_shape_yx=tuple(authored["frame_shape_yx"]),
        exposure_seconds=float(authored["exposure_seconds"]),
    )

    def render(ordinal: int, *, exposure_seconds: float, occupancy=None):
        return world.render_mot_frame(
            ordinal,
            exposure_seconds=exposure_seconds,
            occupancy=occupancy,
            frame_shape_yx=config.frame_shape_yx,
        )

    camera = VirtualCamera(
        config,
        frame_source=lambda ordinal, exposure: render(
            ordinal,
            exposure_seconds=exposure,
        ),
        free_running=True,
    )
    return bind_camera(
        context,
        key,
        camera,
        f"virtual-mot-camera:{key}",
        "camera.virtual_mot",
    )


def _mot_camera_world_config(values: dict) -> None:
    VIRTUAL_MOT_CAMERA_SCHEMA.project_values(values)
    return None


def _sequencer_factory(context, key: str, values: dict) -> InstalledLeaf:
    VIRTUAL_SEQUENCER_SCHEMA.project_values(values)
    if not isinstance(context.world, SimulationWorld):
        raise TypeError("sequencer.virtual requires the installation SimulationWorld")
    device = VirtualSequencer(world=context.world)
    device.open()
    return bind_sequencer(
        context,
        key,
        device,
        f"virtual-sequencer:{key}",
        "sequencer.virtual",
    )


def _sequencer_world_config(values: dict) -> None:
    """Request the default shared world without claiming camera geometry."""

    VIRTUAL_SEQUENCER_SCHEMA.project_values(values)
    return None


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "camera.virtual",
        "camera",
        VIRTUAL_CAMERA_SCHEMA,
        ("camera.adapter", "camera.working_point"),
        factory=_camera_factory,
        world_config=_camera_world_config,
    ),
    DeviceTypeDescriptor(
        "sequencer.virtual",
        "sequencer",
        VIRTUAL_SEQUENCER_SCHEMA,
        ("sequencer.streamer",),
        factory=_sequencer_factory,
        world_config=_sequencer_world_config,
    ),
    DeviceTypeDescriptor(
        "camera.virtual_mot",
        "camera",
        VIRTUAL_MOT_CAMERA_SCHEMA,
        ("camera.adapter", "camera.working_point"),
        factory=_mot_camera_factory,
        world_config=_mot_camera_world_config,
    ),
)


__all__ = [
    "DEVICE_TYPES",
    "VIRTUAL_CAMERA_SCHEMA",
    "VIRTUAL_MOT_CAMERA_SCHEMA",
    "VIRTUAL_SEQUENCER_SCHEMA",
]
