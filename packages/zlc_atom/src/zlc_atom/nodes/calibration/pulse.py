"""Resolve one project-owned calibration pulse JSON into a compiled program."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from zlc_pulse import (
    PulseSequence,
    compile_sequence,
    resolve_api_parameters,
    sequence_from_tree,
)
from zlc_pulse.device import BoardDescription


CALIBRATION_API_PARAMETER_IDS = (
    "reference_probe_duration_before",
    "readout_probe_duration",
    "reference_probe_duration_after",
)
CAMERA_TRIGGER_PORT = "emCCD"


@dataclass(frozen=True)
class ResolvedPulse:
    """One project JSON pulse resolved at the requested calibration point."""

    name: str
    path: Path
    sequence: PulseSequence
    program: object
    metadata: Mapping[str, Any]


def load_calibration_pulse_template(path: str | Path) -> PulseSequence:
    """Decode one exact JSON file and require calibration-template semantics."""

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".json" or not source.is_file():
        raise ValueError("calibration pulse template must be an existing JSON file")
    sequence = sequence_from_tree(
        json.loads(source.read_text(encoding="utf-8"))
    )
    _validate_calibration_sequence(sequence)
    return sequence


def _validate_calibration_sequence(sequence: PulseSequence) -> None:
    if not isinstance(sequence, PulseSequence):
        raise TypeError("calibration pulse must be PulseSequence")
    if sequence.slots:
        raise ValueError("calibration pulse cannot declare scan slots")
    parameter_ids = tuple(
        parameter.parameter_id for parameter in sequence.api_parameters
    )
    if parameter_ids != CALIBRATION_API_PARAMETER_IDS:
        raise ValueError(
            "calibration pulse API parameters must be "
            f"{CALIBRATION_API_PARAMETER_IDS!r}, got {parameter_ids!r}"
        )


def arm_sequencer(sequencer: object, pulse: ResolvedPulse) -> None:
    """Apply and load one resolved pulse, including its inspectable source."""

    if not isinstance(pulse, ResolvedPulse):
        raise TypeError("pulse must be ResolvedPulse")
    channel = pulse.metadata.get("camera_trigger_channel")
    if channel is not None and hasattr(sequencer, "camera_trigger_channel"):
        sequencer.camera_trigger_channel = str(channel)
    sequencer.load(pulse.program, source=pulse.sequence)


def resolve_pulse(
    sequence: PulseSequence,
    *,
    path: str | Path,
    board: BoardDescription,
    api_values: Mapping[str, float],
) -> ResolvedPulse:
    """Resolve and compile the already-decoded exact workspace resource."""

    if not isinstance(board, BoardDescription):
        raise TypeError("board must be BoardDescription")
    _validate_calibration_sequence(sequence)
    source = Path(path).expanduser().resolve()
    values = dict(api_values)
    if set(values) != set(CALIBRATION_API_PARAMETER_IDS) or len(values) != len(
        CALIBRATION_API_PARAMETER_IDS
    ):
        raise ValueError(
            "calibration API values must contain exactly "
            f"{CALIBRATION_API_PARAMETER_IDS!r}"
        )
    resolved = resolve_api_parameters(
        sequence,
        {
            parameter_id: float(values[parameter_id])
            for parameter_id in CALIBRATION_API_PARAMETER_IDS
        },
    )
    if resolved.target != board.target:
        raise ValueError(
            "calibration pulse target is incompatible with the connected board"
        )
    program = compile_sequence(resolved, board.geometry, board.clock_hz)
    exposures = program.camera_window_exposures(CAMERA_TRIGGER_PORT)
    metadata = {
        "camera_trigger_channel": CAMERA_TRIGGER_PORT,
        "camera_windows": program.camera_window_count(CAMERA_TRIGGER_PORT),
        "frame_exposures": exposures,
        "frame_semantics": (
            "reference_long_before",
            "short_readout",
            "reference_long_after",
        ),
        "reference_frame_indices": (0, 2),
        "short_frame_index": 1,
        "repeat_forever": program.repeat_forever,
    }
    return ResolvedPulse(
        sequence.name,
        source,
        resolved,
        program,
        metadata,
    )


__all__ = [
    "CALIBRATION_API_PARAMETER_IDS",
    "CAMERA_TRIGGER_PORT",
    "ResolvedPulse",
    "arm_sequencer",
    "load_calibration_pulse_template",
    "resolve_pulse",
]
