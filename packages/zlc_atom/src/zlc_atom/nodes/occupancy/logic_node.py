"""Discoverable occupancy processor descriptor."""

from __future__ import annotations

from pathlib import Path

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    ArtifactInputSpec,
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
    OutputSpec,
)
from zlc_atom.nodes.calibration import (
    DEFAULT_READOUT_MODEL_CHOICE,
    READOUT_MODEL_CHOICES,
    TrapCalibration,
    readout_model_kind_from_choice,
)

from .processor import OccupancyProcessor


OCCUPANCY_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "model_kind",
            "choice",
            "Readout model",
            DEFAULT_READOUT_MODEL_CHOICE,
            choices=READOUT_MODEL_CHOICES,
        ),
    )
)


def _build(
    *,
    calibration_path: object,
    source_signal: str,
    signal_plane: object | None = None,
    **values: object,
) -> OccupancyProcessor:
    authored = OCCUPANCY_SCHEMA.freeze(values)
    text = str(calibration_path).strip()
    if not text:
        raise ValueError("calibration_path must be non-empty")
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    path = Path(text).expanduser().resolve()
    return OccupancyProcessor(
        TrapCalibration.load(path),
        calibration_path=path,
        signal_plane=signal_plane,
        source_signal=selected_source,
        model_kind=readout_model_kind_from_choice(authored["model_kind"]),
    )


LOGIC_NODE = LogicNodeDescriptor(
    "occupancy",
    NodeKind.PROCESSOR,
    OCCUPANCY_SCHEMA,
    input_specs=(
        DatasetInputSpec("frames", "camera.frames.v1"),
        ArtifactInputSpec("calibration_path", "calibration.readout.v1"),
    ),
    outputs=(
        OutputSpec("counts", "occupancy.counts.v1"),
        OutputSpec("occupied", "occupancy.occupied.v1"),
        OutputSpec("valid", "occupancy.valid.v1"),
        OutputSpec("rate", "occupancy.rate.v1"),
        OutputSpec("frame_judged", "occupancy.frame_judged.v1"),
    ),
    build=_build,
)

__all__ = ["LOGIC_NODE", "OCCUPANCY_SCHEMA"]
