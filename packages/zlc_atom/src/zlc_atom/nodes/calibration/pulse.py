"""Resolve one project-owned calibration pulse JSON into a compiled program."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from zlc_pulse import (
    PulseSequence,
    resolve_api_parameters,
)
from zlc_pulse.codec import read_pulse_document
from zlc_pulse.device import BoardDescription


@dataclass(frozen=True)
class ResolvedPulse:
    """One project JSON pulse resolved at the requested calibration point."""

    name: str
    path: Path
    sequence: PulseSequence
    program: object


def load_calibration_pulse_template(path: str | Path) -> PulseSequence:
    """Decode the chosen Pulse; the task separately chooses its API fields."""

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".json" or not source.is_file():
        raise ValueError("calibration pulse template must be an existing JSON file")
    sequence, _editor = read_pulse_document(source)
    return sequence


def arm_sequencer(sequencer: object, pulse: ResolvedPulse) -> None:
    """Apply and load one resolved pulse, including its inspectable source.

    It used to also tell the sequencer which port gates the camera.  Only the
    SIMULATED board needs that -- its world has to know which edge produces a
    frame -- and it is a fact about the apparatus, not about a measurement: on
    a real bench that port is a wire, and nothing in software is told about it.
    The virtual sequencer owns its own answer.
    """

    if not isinstance(pulse, ResolvedPulse):
        raise TypeError("pulse must be ResolvedPulse")
    sequencer.load(pulse.program, source=pulse.sequence)


def resolve_pulse(
    sequence: PulseSequence,
    *,
    path: str | Path,
    sequencer: object,
    api_values: Mapping[str, float],
) -> ResolvedPulse:
    """Resolve and compile the already-decoded exact workspace resource.

    API values are resolved here. Saved Config values remain the device's
    responsibility at LOAD/Fire; execution records read its applied state.

    ``api_values`` addresses the selected API fields by their stable field
    references, independently of declaration order or period names. Every other
    parameter keeps the value its author gave it, which is what lets one
    imaging pulse carry a MOT duration nobody here has an opinion about.

    Nothing about the camera is said here at all.  What the frames mean is the
    protocol's own arithmetic on the fields it drove, and a task that re-derived
    it from windows and exposures made itself depend on the shape of a document
    the operator writes.
    """

    board = sequencer.describe()
    if not isinstance(board, BoardDescription):
        raise TypeError("board must be BoardDescription")
    if not isinstance(sequence, PulseSequence):
        raise TypeError("calibration pulse must be PulseSequence")
    source = Path(path).expanduser().resolve()
    resolved = resolve_api_parameters(replace(
        sequence,
        bindings=tuple(replace(binding, scan=False) for binding in sequence.bindings),
    ), api_values)
    if resolved.target != board.target:
        raise ValueError(
            "calibration pulse target is incompatible with the connected board"
        )
    resolved, program = sequencer.compile_pulse(
        resolved, board.geometry, board.clock_hz
    )
    # Named by its file: the operator chose the file, and a document's own
    # name is whatever it was called when first drawn.
    return ResolvedPulse(
        Path(source).stem,
        source,
        resolved,
        program,
    )


__all__ = [
    "ResolvedPulse",
    "arm_sequencer",
    "load_calibration_pulse_template",
    "resolve_pulse",
]
