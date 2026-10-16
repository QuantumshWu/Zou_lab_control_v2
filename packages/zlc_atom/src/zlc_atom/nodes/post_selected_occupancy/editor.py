"""Post-selected occupancy's condition, under Occupancy's calibrations.

A section of its own: the Mode strip, then one "Keep where" row in the
same control column -- frame A, the relation strip, frame B in Relation
mode; each frame with its Any | 1 | 0 strip in Values mode.  The frame
combos offer the frames the source carries (the console's
``source_frames``), named as Occupancy's columns name them: First and Last
by what they are, the middle frames by number.  The title row's summary
reads the condition.  A draft problem is a red sentence under the rows; a
frame this source lacks, or a cycle of one frame, an amber one: said, as
Occupancy and Frame Survival say what the source lacks, and the run waits.
The other mode's values stay in the draft as they are.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    FluentComboBox, FluentSegmentedButtons, FluentSettingRow, card_title_row, error_note_label,
    muted_note_label, row_gap, set_status_ink, show_problem, signals_blocked,
)

from zlc_atom.nodes._framework.frames import FIRST, LAST, frame_token

from .logic_node import (
    ANY, CONDITION_FIELDS, FRAME_A, FRAME_B, MODE, RELATION, VALUE_A, VALUE_B, condition, source_problem,
)

_CONDITION = "Keep where"


def _choices(field) -> tuple[tuple[object, str], ...]:
    return tuple((choice.value, choice.label) for choice in field.choices)


class PostSelectionEditor(QtWidgets.QWidget):
    """The two frames post-selected on, and the relation or values they must have."""

    draft_changed = QtCore.pyqtSignal(object)
    managed_fields = tuple(field.name for field in CONDITION_FIELDS)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(row_gap())
        self.summary = muted_note_label("")
        header, self.title = card_title_row("Post-selection", self.summary)
        layout.addLayout(header)
        self.mode = FluentSegmentedButtons(_choices(MODE), self)
        self.mode_row = FluentSettingRow(MODE.label, self.mode)
        layout.addWidget(self.mode_row)
        controls = QtWidgets.QWidget(self)
        line = QtWidgets.QHBoxLayout(controls)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(row_gap())
        self.frame_a, self.frame_b = FluentComboBox(controls), FluentComboBox(controls)
        self.value_a = FluentSegmentedButtons(_choices(VALUE_A), controls)
        self.value_b = FluentSegmentedButtons(_choices(VALUE_B), controls)
        self.relation = FluentSegmentedButtons(_choices(RELATION), controls)
        self._strips = {MODE.name: self.mode, RELATION.name: self.relation,
                        VALUE_A.name: self.value_a, VALUE_B.name: self.value_b}
        for name, strip in self._strips.items():
            strip.activated.connect(lambda value, name=name: self._write(name, value))
        for field, combo in ((FRAME_A, self.frame_a), (FRAME_B, self.frame_b)):
            combo.activated.connect(lambda _row, field=field, combo=combo: self._write(field.name, combo.currentData()))
        # A frame and its value stand closer than the two frames do.
        for combo, strip in ((self.frame_a, self.value_a), (self.frame_b, self.value_b)):
            slot = QtWidgets.QHBoxLayout()
            slot.setSpacing(row_gap() // 2)
            slot.addWidget(combo)
            slot.addWidget(strip)
            line.addLayout(slot)
            if combo is self.frame_a:
                line.addWidget(self.relation)
        controls.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.condition_row = FluentSettingRow(_CONDITION, controls)
        layout.addWidget(self.condition_row)
        self.problem = error_note_label()
        layout.addWidget(self.problem)
        self.note = error_note_label()
        set_status_ink(self.note, "warning")
        layout.addWidget(self.note)
        #: The draft's problem this section states, for the page header.
        self._draft_problem = ""

    def row_labels(self) -> tuple[str, ...]:
        return (MODE.label, _CONDITION)

    def set_label_width(self, width: int) -> None:
        self.mode_row.set_label(MODE.label, width=int(width))
        self.condition_row.set_label(_CONDITION, width=int(width))

    def update_projection(self, projection) -> None:
        values = projection["form_values"]
        frames = projection.get("source_frames")
        for name, strip in self._strips.items():
            strip.set_current(values[name])
        relation = values[MODE.name] == "relation"
        self.relation.setVisible(relation)
        self.value_a.setVisible(not relation)
        self.value_b.setVisible(not relation)
        for field, combo in ((FRAME_A, self.frame_a), (FRAME_B, self.frame_b)):
            _offer(combo, frames, str(values[field.name]))
        try:
            kept = condition(values)
        except ValueError as error:
            kept, self._draft_problem = None, str(error)
        else:
            self._draft_problem = ""
        self.summary.setText("" if kept is None else _summary(kept))
        show_problem(self.problem, self._draft_problem)
        show_problem(self.note, "" if kept is None or frames is None else source_problem(kept, len(frames)))

    def _write(self, name: str, value) -> None:
        # Marked at once, as a popup shows its pick; the projection the edit
        # comes back as marks it again, or puts back what was refused.
        if name in self._strips:
            self._strips[name].set_current(value)
        self.draft_changed.emit({"values": {name: value}})

    def shown_problems(self) -> dict[str, str]:
        return {self._draft_problem: self.title.text()} if self._draft_problem else {}

    def set_mutation_enabled(self, enabled: bool) -> None:
        self.setEnabled(bool(enabled))


def _offer(combo: FluentComboBox, frames, written: str) -> None:
    """The frames ``frames`` carries, as Occupancy's columns name them, and
    ``written`` marked -- offered as written too when the source lacks it."""

    count = None if frames is None else len(frames)
    items = [("First" if count is None else f"First · {frames[0]}", FIRST),
             *(() if count is None else ((frames[k], str(k)) for k in range(1, count - 1))),
             ("Last" if count is None else f"Last · {frames[-1]}", LAST)]
    try:
        token = frame_token(written)
        current = FIRST if token == 0 else str(token)
    except ValueError:
        current = written
    if all(data != current for _text, data in items):
        items.append((f"{written} · not in this source", current))
    if getattr(combo, "_offered", None) != items:
        with signals_blocked(combo):
            combo.clear()
            for text, data in items:
                combo.addItem(text, data)
        combo._offered = items
    with signals_blocked(combo):
        combo.setCurrentIndex(next(row for row, (_text, data) in enumerate(items) if data == current))


def _summary(kept) -> str:
    """The condition as the title row reads it: ``kept where frame first = frame last``."""

    names = tuple(f"frame {token}" for token in kept["frames"])
    if kept["mode"] == "relation":
        relation = dict(_choices(RELATION))[kept["relation"]]
        return f"kept where {names[0]} {relation} {names[1]}"
    verdicts = dict(_choices(VALUE_A))
    return "kept where " + " and ".join(
        f"{name} is {verdicts[value]}" for name, value in zip(names, kept["values"]) if value != ANY)


__all__ = ["PostSelectionEditor"]
