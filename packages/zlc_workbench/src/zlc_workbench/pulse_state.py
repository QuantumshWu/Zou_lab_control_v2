"""The one complete authoring record persisted by the Pulse Editor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
import os
from pathlib import Path
from typing import Any

from zlc_durable import write_readable_json
from zlc_pulse import (
    MAXIMUM_REPEAT_COUNT,
    PulseSequence,
    scan_columns_for,
    sequence_from_tree,
    sequence_to_tree,
    validate_scan_table,
)
from zlc_pulse.codec import parse_pulse_tree_json, split_pulse_document_tree


@dataclass(frozen=True, slots=True)
class PulseEditorState:
    """Everything authored in one Pulse Editor, independent of its file path."""

    sequence: PulseSequence | None = None
    visible_ports: frozenset[str] | None = None
    scan_source: str = ""
    scan_rows: tuple[tuple[float, ...], ...] = ()
    scan_source_dirty: bool = False
    scan_repeats: int = 0

    def __post_init__(self) -> None:
        if self.sequence is not None and not isinstance(self.sequence, PulseSequence):
            raise TypeError("sequence must be PulseSequence or None")
        visible = self.visible_ports
        if visible is not None and isinstance(visible, (str, bytes, Mapping)):
            raise TypeError("visible_ports must be a collection of port names")
        if visible is not None and any(not isinstance(key, str) for key in visible):
            raise TypeError("visible_ports must contain strings")
        if visible:
            if self.sequence is None:
                raise ValueError("visible_ports requires a sequence")
            unknown = frozenset(visible).difference(self.sequence.target.by_key)
            if unknown:
                raise ValueError(
                    f"visible_ports names unknown port(s): {', '.join(sorted(unknown))}"
                )
        object.__setattr__(
            self,
            "visible_ports",
            None if visible is None else frozenset(visible),
        )
        if not isinstance(self.scan_source, str):
            raise TypeError("scan_source must be a string")
        rows: list[tuple[float, ...]] = []
        for row in self.scan_rows:
            if isinstance(row, (str, bytes, Mapping)) or not isinstance(row, Sequence):
                raise TypeError("each scan row must be a sequence of numbers")
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in row):
                raise TypeError("each scan value must be a number")
            rows.append(tuple(float(value) for value in row))
        object.__setattr__(
            self,
            "scan_rows",
            tuple(rows),
        )
        if isinstance(self.scan_repeats, bool) or not isinstance(self.scan_repeats, int):
            raise TypeError("scan_repeats must be an integer")
        repeats = self.scan_repeats
        if repeats < 0:
            raise ValueError("scan_repeats must be non-negative")
        if repeats > MAXIMUM_REPEAT_COUNT:
            raise ValueError("scan_repeats does not fit the hardware 32-bit count")
        if not isinstance(self.scan_source_dirty, bool):
            raise TypeError("scan_source_dirty must be a boolean")


def state_from_tree(tree: Mapping[str, Any]) -> PulseEditorState:
    """Decode the pulse and its sole ``editor`` section as one candidate."""

    sequence_tree, raw = split_pulse_document_tree(tree)
    sequence = sequence_from_tree(sequence_tree)
    visible = raw.get("visible_ports")
    source = raw.get("scan_source", "")
    rows = raw.get("scan_rows", ())
    dirty = raw.get("scan_source_dirty", False)
    repeats = raw.get("scan_repeats", 0)
    candidate = PulseEditorState(
        sequence=sequence,
        visible_ports=None if visible is None else frozenset(visible),
        scan_source=source,
        scan_rows=tuple(tuple(row) for row in rows),
        scan_source_dirty=dirty,
        scan_repeats=repeats,
    )
    if candidate.scan_rows:
        validate_scan_table(candidate.scan_rows, scan_columns_for(sequence))
    return candidate


def state_to_tree(state: PulseEditorState) -> dict[str, Any]:
    """Encode exactly the state the Workbench owns."""

    if not isinstance(state, PulseEditorState):
        raise TypeError("state must be PulseEditorState")
    if state.sequence is None:
        raise ValueError("a pulse file requires a sequence")
    tree = dict(sequence_to_tree(state.sequence))
    tree["editor"] = {
        "visible_ports": (
            None
            if state.visible_ports is None
            else [
                port.key
                for port in state.sequence.target.ports
                if port.key in state.visible_ports
            ]
        ),
        "scan_source": state.scan_source,
        "scan_rows": [list(row) for row in state.scan_rows],
        "scan_source_dirty": state.scan_source_dirty,
        "scan_repeats": state.scan_repeats,
    }
    return tree


def read_pulse(path: str | os.PathLike[str]) -> PulseEditorState:
    """Read one complete ``zlc.pulse`` Workbench authoring state."""

    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError(f"pulse files must be JSON: {source}")
    tree = parse_pulse_tree_json(source.read_text(encoding="utf-8"))
    return state_from_tree(tree)


def write_pulse(path: str | os.PathLike[str], state: PulseEditorState) -> None:
    """Write one complete authoring state through the sole Workbench codec."""

    write_readable_json(Path(path), state_to_tree(state))


__all__ = ["PulseEditorState", "read_pulse", "state_from_tree", "state_to_tree", "write_pulse"]
