"""The one complete authoring record persisted by the Pulse Editor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
from zlc_pulse.codec import (
    PULSE_EDITOR_DEFAULTS,
    check_pulse_editor_fields,
    read_pulse_document,
    split_pulse_document_tree,
)


@dataclass(frozen=True, slots=True)
class PulseEditorState:
    """Everything authored in one Pulse Editor, independent of its file path."""

    sequence: PulseSequence | None = None
    # Each default is the pulse codec's, the one place they are decided --
    # what a file that leaves a field out means is what a new editor starts
    # with.  A scan's repeat count of 0 runs it until Stop.
    visible_ports: frozenset[str] | None = PULSE_EDITOR_DEFAULTS["visible_ports"]
    scan_source: str = PULSE_EDITOR_DEFAULTS["scan_source"]
    scan_rows: tuple[tuple[float, ...], ...] = PULSE_EDITOR_DEFAULTS["scan_rows"]
    scan_source_dirty: bool = PULSE_EDITOR_DEFAULTS["scan_source_dirty"]
    scan_repeats: int = PULSE_EDITOR_DEFAULTS["scan_repeats"]

    def __post_init__(self) -> None:
        if self.sequence is not None and not isinstance(self.sequence, PulseSequence):
            raise TypeError("sequence must be PulseSequence or None")
        # The codec's rule for these fields, not a copy of it; what is
        # checked here besides is what only the sequence can answer.
        check_pulse_editor_fields(
            visible_ports=self.visible_ports,
            scan_source=self.scan_source,
            scan_rows=self.scan_rows,
            scan_source_dirty=self.scan_source_dirty,
            scan_repeats=self.scan_repeats,
        )
        visible = self.visible_ports
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
        object.__setattr__(
            self,
            "scan_rows",
            tuple(tuple(map(float, row)) for row in self.scan_rows),
        )


def state_from_tree(tree: Mapping[str, Any]) -> PulseEditorState:
    """Decode the pulse and its sole ``editor`` section as one candidate."""

    sequence_tree, raw = split_pulse_document_tree(tree)
    return editor_state_from(sequence_from_tree(sequence_tree), raw)


def editor_state_from(
    sequence: PulseSequence, raw: Mapping[str, Any]
) -> PulseEditorState:
    """One authoring state from a decoded pulse and its editor section.

    Apart from :func:`state_from_tree`, because a pulse read from a FILE has
    already been decoded by the time it gets here, and decoding it a second
    time from the tree would
    throw that refresh away.
    """

    fields = {name: raw.get(name, default) for name, default in PULSE_EDITOR_DEFAULTS.items()}
    visible = fields["visible_ports"]
    candidate = PulseEditorState(
        sequence=sequence,
        visible_ports=None if visible is None else frozenset(visible),
        scan_source=fields["scan_source"],
        scan_rows=tuple(tuple(row) for row in fields["scan_rows"]),
        scan_source_dirty=fields["scan_source_dirty"],
        scan_repeats=fields["scan_repeats"],
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
    return editor_state_from(*read_pulse_document(source))


def write_pulse(path: str | os.PathLike[str], state: PulseEditorState) -> None:
    """Write one complete authoring state through the sole Workbench codec."""

    write_readable_json(Path(path), state_to_tree(state))


__all__ = [
    "PulseEditorState",
    "editor_state_from",
    "read_pulse",
    "state_from_tree",
    "state_to_tree",
    "write_pulse",
]
