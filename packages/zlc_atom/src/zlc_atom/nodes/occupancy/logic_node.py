"""Discoverable occupancy processor descriptor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.nodes._framework.descriptor import ArtifactInputSpec, DatasetInputSpec, LogicNodeDescriptor, NodeKind, OutputSpec

from .processor import OccupancyProcessor


def _build(*, calibration: object, signal_plane: object | None = None, source_signal: str | None = None) -> OccupancyProcessor:
    return OccupancyProcessor(
        calibration,  # type: ignore[arg-type]
        signal_plane=signal_plane,
        source_signal=source_signal,
    )


LOGIC_NODE = LogicNodeDescriptor(
    "occupancy",
    NodeKind.PROCESSOR,
    AuthoringSchema(()),
    input_specs=(
        DatasetInputSpec("frames", "camera.frames.v1"),
        ArtifactInputSpec("calibration", "calibration.readout.v1", allow_saved_reference=True),
    ),
    outputs=(
        OutputSpec("counts", "occupancy.counts.v1"),
        OutputSpec("occupied", "occupancy.occupied.v1"),
        OutputSpec("rate", "occupancy.rate.v1"),
    ),
    build=_build,
)

__all__ = ["LOGIC_NODE"]
