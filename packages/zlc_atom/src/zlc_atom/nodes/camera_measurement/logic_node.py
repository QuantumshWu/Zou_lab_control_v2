"""Discoverable camera measurement descriptor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import DeviceRequirement, LogicNodeDescriptor, NodeKind, OutputSpec

from .measurement import CameraMeasurementNode


CAMERA_MEASUREMENT_SCHEMA = AuthoringSchema(
    (
        AuthoringField("repeat", "int", "Repeat", 1, minimum=1),
        AuthoringField("frames_per_cycle", "int", "Frames per cycle", 1, minimum=1),
        AuthoringField("timeout", "float", "Timeout seconds", 2.0, minimum=0.001),
    )
)


def _build(*, camera: object, signal_plane: object, **values: object) -> CameraMeasurementNode:
    authored = CAMERA_MEASUREMENT_SCHEMA.freeze(values)
    return CameraMeasurementNode(
        camera=camera,  # type: ignore[arg-type]
        signal_plane=signal_plane,
        timeout=float(authored["timeout"]),
        # Every authored field reaches the node.  Freezing repeat and
        # frames_per_cycle and then dropping them left a hosted run with no
        # acquisition to perform and no record of what was asked for.
        repeat=int(authored["repeat"]),
        frames_per_cycle=int(authored["frames_per_cycle"]),
    )


LOGIC_NODE = LogicNodeDescriptor(
    "camera_measurement",
    NodeKind.MEASUREMENT,
    CAMERA_MEASUREMENT_SCHEMA,
    outputs=(OutputSpec("frames", "camera.frames.v1"),),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera"),
    ),
    build=_build,
)

__all__ = ["CAMERA_MEASUREMENT_SCHEMA", "LOGIC_NODE"]
