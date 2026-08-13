"""Resolve one project-owned calibration pulse JSON into a compiled program."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from zlc_pulse import (
    PulseSequence,
    authored_api_values,
    compile_sequence,
    resolve_api_parameters,
    sequence_from_tree,
)
from zlc_pulse.device import BoardDescription


#: The port a board usually gates its camera from.  A default for an authoring
#: field, not a fact about anybody's board: a node that wrote it into its own
#: source could only ever run the one pulse whose author had picked that name.
DEFAULT_CAMERA_TRIGGER_PORT = "emCCD"


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
    """What calibration needs of ANY pulse, said without naming one.

    It used to demand three API parameters carrying three exact identifiers in
    one exact order -- the private naming of the template that shipped with the
    node.  An operator's own imaging pulse could not be used at all, however
    many API slots it had, because the names were not the ones this file
    happened to contain.  What the protocol actually requires is timing it can
    set and camera windows it can read; WHICH parameters those are is the
    operator's choice, made in the task's own form.
    """

    if not isinstance(sequence, PulseSequence):
        raise TypeError("calibration pulse must be PulseSequence")
    if sequence.slots:
        raise ValueError("calibration pulse cannot declare scan slots")


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
    trigger_port: str = DEFAULT_CAMERA_TRIGGER_PORT,
) -> ResolvedPulse:
    """Resolve and compile the already-decoded exact workspace resource.

    ``api_values`` names the parameters this run owns, by the identifiers the
    PULSE declares -- a caller that owns slots addresses them by number and
    reads the identifier off the declaration in that order.  Every other
    parameter keeps the value its author gave it, which is what lets one
    imaging pulse carry a MOT duration nobody here has an opinion about.

    Nothing about the camera is read back out of the compiled program.  What
    the frames mean is the protocol's own arithmetic on the slots it drove,
    and a task that re-derived it from windows and exposures made itself
    depend on the shape of a document the operator writes.
    """

    if not isinstance(board, BoardDescription):
        raise TypeError("board must be BoardDescription")
    _validate_calibration_sequence(sequence)
    source = Path(path).expanduser().resolve()
    port = str(trigger_port).strip()
    if not port:
        raise ValueError("a camera trigger port is required")
    declared = authored_api_values(sequence)
    unknown = tuple(name for name in api_values if name not in declared)
    if unknown:
        raise ValueError(
            f"pulse {sequence.name!r} declares no API parameter "
            f"{unknown!r}; it offers {tuple(declared)!r}"
        )
    values = dict(declared)
    values.update({name: float(value) for name, value in api_values.items()})
    resolved = resolve_api_parameters(sequence, values)
    if resolved.target != board.target:
        raise ValueError(
            "calibration pulse target is incompatible with the connected board"
        )
    program = compile_sequence(resolved, board.geometry, board.clock_hz)
    metadata = {
        "camera_trigger_channel": port,
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
    "DEFAULT_CAMERA_TRIGGER_PORT",
    "ResolvedPulse",
    "arm_sequencer",
    "load_calibration_pulse_template",
    "resolve_pulse",
]
