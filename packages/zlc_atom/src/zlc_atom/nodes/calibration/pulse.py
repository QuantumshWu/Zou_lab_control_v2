"""Resolve one project-owned calibration pulse JSON into a compiled program."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from zlc_pulse import (
    compile_sequence,
    load_streamer_config,
    resolve_scan_point,
    sequence_from_tree,
)


CALIBRATION_SLOT_IDS = (
    "reference_before",
    "readout",
    "reference_after",
)
CAMERA_TRIGGER_PORT = "emCCD"


def arm_sequencer(
    sequencer: object,
    program: object,
    metadata: Mapping[str, Any],
) -> None:
    """Apply the pulse's logical camera line, then load its program."""

    channel = metadata.get("camera_trigger_channel")
    if channel is not None and hasattr(sequencer, "camera_trigger_channel"):
        sequencer.camera_trigger_channel = str(channel)
    sequencer.load(program)


@dataclass(frozen=True)
class ResolvedPulse:
    """One project JSON pulse resolved at the requested calibration point."""

    name: str
    path: Path
    program: object
    metadata: Mapping[str, Any]


def _template_filename(value: object) -> str:
    name = str(value).strip()
    if not name or Path(name).name != name:
        raise ValueError("pulse template must be a plain JSON filename")
    suffix = Path(name).suffix.lower()
    if suffix == "":
        return f"{name}.json"
    if suffix != ".json":
        raise ValueError("pulse template must be a plain JSON filename")
    return name


def resolve_pulse(
    template: str,
    *,
    search_paths: Sequence[str | Path],
    slot_values: Mapping[str, float],
) -> ResolvedPulse:
    """Load one exact project JSON, resolve its three slots, and compile it."""

    filename = _template_filename(template)
    if isinstance(search_paths, (str, Path)):
        search_paths = (search_paths,)
    roots = tuple(Path(value).expanduser().resolve() for value in search_paths)
    candidates = tuple(root / filename for root in roots)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        attempted = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            f"pulse template {filename!r} was not found; searched these paths:\n"
            f"{attempted}"
        )

    tree = json.loads(path.read_text(encoding="utf-8"))
    sequence = sequence_from_tree(tree)
    slot_ids = tuple(slot.slot_id for slot in sequence.slots)
    if slot_ids != CALIBRATION_SLOT_IDS:
        raise ValueError(
            "calibration pulse slots must be "
            f"{CALIBRATION_SLOT_IDS!r}, got {slot_ids!r}"
        )
    values = dict(slot_values)
    if set(values) != set(CALIBRATION_SLOT_IDS) or len(values) != len(
        CALIBRATION_SLOT_IDS
    ):
        raise ValueError(
            "calibration slot values must contain exactly "
            f"{CALIBRATION_SLOT_IDS!r}"
        )
    resolved = resolve_scan_point(
        sequence,
        tuple(float(values[slot_id]) for slot_id in CALIBRATION_SLOT_IDS),
    )
    config = load_streamer_config()
    if config["source"] is None:
        raise RuntimeError(
            "no streamer config was found, so the deployed board geometry is "
            "unknown"
        )
    program = compile_sequence(resolved, config["params"], config["clock_hz"])
    exposures = program.camera_window_exposures(CAMERA_TRIGGER_PORT)
    reference_indices = tuple(
        index
        for index, slot_id in enumerate(slot_ids)
        if slot_id.startswith("reference_")
    )
    readout_index = slot_ids.index("readout")
    metadata = {
        "camera_trigger_channel": CAMERA_TRIGGER_PORT,
        "camera_windows": program.camera_window_count(CAMERA_TRIGGER_PORT),
        "frame_exposures": exposures,
        "frame_semantics": tuple(
            "short_readout"
            if slot_id == "readout"
            else f"reference_long_{slot_id.removeprefix('reference_')}"
            for slot_id in slot_ids
        ),
        "reference_frame_indices": reference_indices,
        "short_frame_index": readout_index,
        "repeat_forever": program.repeat_forever,
    }
    return ResolvedPulse(sequence.name, path.resolve(), program, metadata)


__all__ = [
    "CALIBRATION_SLOT_IDS",
    "CAMERA_TRIGGER_PORT",
    "ResolvedPulse",
    "arm_sequencer",
    "resolve_pulse",
]
