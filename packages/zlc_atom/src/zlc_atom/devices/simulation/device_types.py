"""Automatically discovered virtual apparatus device types."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.camera.binding import bind_camera
from zlc_atom.devices.sequencer.binding import bind_sequencer, open_sequencer_control
from zlc_atom.devices.slm import bind_slm, open_slm_control
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from .camera import VirtualCamera, VirtualCameraConfig
from .sequencer import VirtualSequencer
from .slm import VirtualSLM
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
            "exposure_seconds",
            "float",
            "Exposure seconds",
            0.02,
            minimum=1e-9,
        ),
    )
)

SIMULATION_WORLD_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "image_shape_yx",
            "pair",
            "Image shape (Y,X)",
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
            "world_profile",
            "str",
            "Simulation world profile (JSON)",
            "",
        ),
    )
)

# A virtual sequencer has no physical endpoint or device-local parameters.
VIRTUAL_SEQUENCER_SCHEMA = AuthoringSchema(())

# The virtual panel geometry belongs to its one SimulationWorld.
VIRTUAL_SLM_SCHEMA = AuthoringSchema(())

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


def _simulation_world_config(values: dict) -> SimulationWorldConfig:
    authored = SIMULATION_WORLD_SCHEMA.project_values(values)
    geometry = SimulationGeometry(
        grid_shape_yx=tuple(authored["grid_shape_yx"]),
        image_shape_yx=tuple(authored["image_shape_yx"]),
    )
    profile_value = authored["world_profile"]
    if not isinstance(profile_value, str):
        raise TypeError("Simulation world profile (JSON) must be text")
    profile = profile_value.strip()
    if profile:
        return SimulationWorldConfig.from_profile(
            profile,
            geometry=geometry,
            seed=int(authored["seed"]),
        )
    return SimulationWorldConfig(geometry, seed=int(authored["seed"]))


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


def _slm_factory(context, key: str, values: dict) -> InstalledLeaf:
    VIRTUAL_SLM_SCHEMA.project_values(values)
    if not isinstance(context.world, SimulationWorld):
        raise TypeError("slm.virtual requires the installation SimulationWorld")
    return bind_slm(
        context,
        key,
        VirtualSLM(context.world, identity="virtual-slm"),
        "slm.virtual",
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "camera.virtual",
        "camera",
        VIRTUAL_CAMERA_SCHEMA,
        ("camera.adapter",),
        factory=_camera_factory,
        world_config=_simulation_world_config,
    ),
    DeviceTypeDescriptor(
        "sequencer.virtual",
        "sequencer",
        VIRTUAL_SEQUENCER_SCHEMA,
        ("sequencer.streamer",),
        factory=_sequencer_factory,
        world_config=_simulation_world_config,
        control_factory=open_sequencer_control,
    ),
    DeviceTypeDescriptor(
        "camera.virtual_mot",
        "camera",
        VIRTUAL_MOT_CAMERA_SCHEMA,
        ("camera.adapter",),
        factory=_mot_camera_factory,
        world_config=_simulation_world_config,
    ),
    DeviceTypeDescriptor(
        "slm.virtual",
        "slm",
        VIRTUAL_SLM_SCHEMA,
        ("slm.phase",),
        factory=_slm_factory,
        world_config=_simulation_world_config,
        control_factory=open_slm_control,
    ),
)


__all__ = [
    "DEVICE_TYPES",
    "SIMULATION_WORLD_SCHEMA",
    "VIRTUAL_CAMERA_SCHEMA",
    "VIRTUAL_MOT_CAMERA_SCHEMA",
    "VIRTUAL_SEQUENCER_SCHEMA",
    "VIRTUAL_SLM_SCHEMA",
]
