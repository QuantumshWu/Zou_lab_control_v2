"""What every virtual device is told about the one world it stands in.

The apparatus has ONE simulated world, and the installation builds it from
whichever virtual device was authored with it (``install.graph`` refuses an
apparatus whose devices disagree about the owner).  So the schema and the
resolver are named once, here, and every virtual device folder declares THIS
function -- the same object, which is what lets the installation recognise
one world behind several devices.
"""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema

from .world import (
    DEFAULT_SIMULATION_GRID_SHAPE_YX,
    DEFAULT_SIMULATION_IMAGE_SHAPE_YX,
    SimulationGeometry,
    SimulationWorldConfig,
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


def simulation_world_config(values: dict) -> SimulationWorldConfig:
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


__all__ = ["SIMULATION_WORLD_SCHEMA", "simulation_world_config"]
