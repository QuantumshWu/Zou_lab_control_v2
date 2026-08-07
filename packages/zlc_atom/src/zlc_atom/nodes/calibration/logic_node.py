"""Discoverable calibration task descriptor."""

from __future__ import annotations

from pathlib import Path

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import DeviceRequirement, LogicNodeDescriptor, NodeKind, OutputSpec

from .task import CalibrationTask


CALIBRATION_SCHEMA = AuthoringSchema(
    (
        AuthoringField("grid_shape_yx", "pair", "Grid shape (Y,X)", (2, 3)),
        AuthoringField("roi_radius", "int", "Box radius", 1, minimum=0),
        AuthoringField("method", "choice", "Readout method", "box", choices=("box", "psf")),
        AuthoringField("reducer", "choice", "Box reducer", "mean", choices=("mean", "sum", "median", "max")),
        AuthoringField("repeats", "int", "Calibration brackets", 30, minimum=1),
        AuthoringField("pulse", "text", "Calibration pulse", "calibration", required=True),
    )
)


def _build(
    *,
    camera: object,
    sequencer: object,
    signal_plane: object,
    pulse_search_paths: object | None = None,
    expected_centers_xy: object | None = None,
    **values: object,
) -> CalibrationTask:
    authored = CALIBRATION_SCHEMA.freeze(values)
    if pulse_search_paths is None:
        paths = (Path.cwd() / "pulses",)
    elif isinstance(pulse_search_paths, (str, Path)):
        paths = (pulse_search_paths,)
    else:
        paths = tuple(pulse_search_paths)  # type: ignore[arg-type]
    return CalibrationTask(
        camera=camera,  # type: ignore[arg-type]
        sequencer=sequencer,
        signal_plane=signal_plane,
        grid_shape=tuple(authored["grid_shape_yx"]),  # type: ignore[arg-type]
        roi_radius=int(authored["roi_radius"]),
        method=str(authored["method"]),
        reducer=str(authored["reducer"]),
        repeats=int(authored["repeats"]),
        pulse_name=str(authored["pulse"]),
        pulse_search_paths=paths,
        expected_centers_xy=expected_centers_xy,
    )


LOGIC_NODE = LogicNodeDescriptor(
    "calibration",
    NodeKind.TASK,
    CALIBRATION_SCHEMA,
    outputs=(
        OutputSpec("calibration", "calibration.readout.v1"),
        OutputSpec("report", "calibration.report.v1"),
    ),
    device_requirements=(
        DeviceRequirement("camera.adapter", "camera"),
        DeviceRequirement("sequencer.streamer", "sequencer"),
    ),
    build=_build,
)

__all__ = ["CALIBRATION_SCHEMA", "LOGIC_NODE"]
