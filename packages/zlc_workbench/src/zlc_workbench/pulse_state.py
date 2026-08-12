"""The one complete authoring record persisted by the Pulse Editor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from numbers import Real
import os
from pathlib import Path
from typing import Any

from zlc_durable import write_readable_json
from zlc_pulse import (
    PulseSequence,
    scan_columns_for,
    sequence_from_tree,
    sequence_to_tree,
    validate_scan_table,
)


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
        if not isinstance(self.scan_source_dirty, bool):
            raise TypeError("scan_source_dirty must be a boolean")


_EDITOR_FIELDS = (
    "visible_ports",
    "scan_source",
    "scan_rows",
    "scan_source_dirty",
    "scan_repeats",
)


def state_from_tree(tree: Mapping[str, Any]) -> PulseEditorState:
    """Decode the pulse and its sole ``editor`` section as one candidate."""

    sequence = sequence_from_tree(tree)
    encoded = tree["editor"] if "editor" in tree else {}
    if not isinstance(encoded, Mapping):
        raise TypeError("pulse editor state must be an object")
    raw = dict(encoded)
    legacy_use_loaded = raw.pop("scan_use_loaded", None)
    unknown = tuple(key for key in raw if key not in _EDITOR_FIELDS)
    if unknown:
        raise ValueError(f"unknown pulse editor field(s): {', '.join(map(str, unknown))}")

    visible = raw.get("visible_ports")
    if visible is not None and (
        isinstance(visible, (str, bytes)) or not isinstance(visible, Sequence)
    ):
        raise TypeError("editor.visible_ports must be null or a list")
    if visible is not None and any(not isinstance(key, str) for key in visible):
        raise TypeError("editor.visible_ports must contain strings")
    source = raw.get("scan_source", "")
    if not isinstance(source, str):
        raise TypeError("editor.scan_source must be a string")
    rows = raw.get("scan_rows", ())
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Sequence):
        raise TypeError("editor.scan_rows must be a table")
    for row in rows:
        if isinstance(row, (str, bytes, Mapping)) or not isinstance(row, Sequence):
            raise TypeError("each editor.scan_rows row must be a list")
    legacy_api_columns = False
    if "api_parameters" not in tree:
        old_slots = tuple(tree.get("slots", ()))
        legacy_api_columns = any(
            str(slot["slot_id"]).startswith("api_") for slot in old_slots
        )
        scan_indices = tuple(
            index
            for index, slot in enumerate(old_slots)
            if not str(slot["slot_id"]).startswith("api_")
        )
        rows = (
            tuple(
                tuple(row[index] for index in scan_indices)
                for row in rows
            )
            if scan_indices
            else ()
        )
    dirty = raw.get(
        "scan_source_dirty",
        bool(source)
        and (
            legacy_api_columns
            or legacy_use_loaded is True
            or not bool(rows)
        ),
    )
    if not isinstance(dirty, bool):
        raise TypeError("editor.scan_source_dirty must be a boolean")
    repeats = raw.get("scan_repeats", 0)
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise TypeError("editor.scan_repeats must be an integer")
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
    """Encode exactly the state the Workbench owns, with no legacy fields."""

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
    """Read one complete ``zlc.pulse.v1`` Workbench authoring state."""

    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError(f"pulse files must be JSON: {source}")
    tree = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(tree, Mapping):
        raise TypeError(f"pulse must be a JSON object: {source}")
    return state_from_tree(tree)


def write_pulse(path: str | os.PathLike[str], state: PulseEditorState) -> None:
    """Write one complete authoring state through the sole Workbench codec."""

    write_readable_json(Path(path), state_to_tree(state))


__all__ = ["PulseEditorState", "read_pulse", "state_from_tree", "state_to_tree", "write_pulse"]
