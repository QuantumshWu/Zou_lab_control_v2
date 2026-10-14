"""Occupancy's calibrations: one row per file, the frames it reads as columns.

The operator says which file reads which frames the way they think it --
this one the first frame, that one the last, another the rest -- by marking
cells: "First" is frame 0 whatever the count, "Last" and "Rest" follow the
frame count, a frame number keeps its position.  A frame reached through
First, Last or Rest shows as a dashed card in its file's row, and as an
empty dashed outline in the other rows when it is a First or a Last (only
that column moves it: a click there says so); a frame nobody reads is an
empty red card in every row, and one two files reach, or a number the source
does not carry, is a red-edged card with the reason below.  Nothing here changes a claim
because the frame count changed: what was authored stays until the operator
edits it.  A file is removed by the × that ends its row and replaced by
clicking its name.  The readout model -- which model each file is read
with -- is chosen under the files it applies to.  The file names are the
editor's label column, shared with the form: the title, the model's label
and the names end where the form's labels do (the longest name widens the
column for all of them), so Add, the first card, the model buttons and the
form's fields start on one edge.  The console resolves the
files and projects how the source's frames read them; this widget only
draws that and writes the edits back.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    RED_DARK,
    TEXT,
    FluentButton,
    FluentLabel,
    FluentSectionLabel,
    FluentSegmentedButtons,
    FluentSelectionGrid,
    fluent_open_path,
    muted_note_label,
    scaled_px,
)

from zlc_atom.nodes._framework.frames import FIRST, LAST, REST, canonical_frames, reach_frames
from zlc_atom.nodes.calibration import READOUT_MODEL_CHOICES

#: The artifact input this editor manages.
CALIBRATIONS = "calibrations"
_NAMED = ((FIRST, "First"), (LAST, "Last"), (REST, "Rest"))
_GAP = None
_HINT = "Click a cell to give that frame to a file · Last and Rest follow the frame count"
_EMPTY = "Add a calibration file; the first one reads every frame"


class CalibrationsEditor(QtWidgets.QWidget):
    """Calibration files as rows, frames as columns; marked cells are claims."""

    draft_changed = QtCore.pyqtSignal(object)
    managed_fields = ("model_kind",)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QHBoxLayout()
        # The label-to-field spacing of a setting row: with the title in the
        # label column, Add starts on the form's field edge.
        header.setSpacing(scaled_px(6, minimum=4))
        self.title = FluentSectionLabel("Calibrations")
        # Centred on the button beside it, not dropped below it like an own-line section.
        self.title.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title)
        self.add_button = FluentButton("Add calibration…")
        self.add_button.clicked.connect(self._browse_new)
        header.addWidget(self.add_button)
        header.addStretch(1)
        self.summary = muted_note_label("")
        header.addWidget(self.summary)
        layout.addLayout(header)
        self.grid = FluentSelectionGrid(
            self, toggle=True,
            row_actions=(("remove", "×", "Remove this calibration"),),
            label_action=("replace", "Click to choose another file"),
        )
        # File names read as a list: they start on the editor's content edge.
        self.grid.verticalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.grid.selection_edited.connect(self._marked)
        self.grid.row_action.connect(self.row_action)
        layout.addWidget(self.grid, 0, QtCore.Qt.AlignLeft)
        self.note = FluentLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {RED_DARK}; background: transparent; border: none;")
        layout.addWidget(self.note)
        self.hint = muted_note_label(_HINT)
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.model_row = QtWidgets.QWidget(self)
        model_layout = QtWidgets.QHBoxLayout(self.model_row)
        model_layout.setContentsMargins(0, scaled_px(6), 0, 0)
        model_layout.setSpacing(scaled_px(6, minimum=4))
        # A field label like "Frames" above it, not a second hint line.
        self.model_label = FluentLabel("Readout model")
        self.model_label.setStyleSheet(f"color: {TEXT}; background: transparent; border: none;")
        model_layout.addWidget(self.model_label)
        self.model = FluentSegmentedButtons(
            tuple((choice.value, choice.label) for choice in READOUT_MODEL_CHOICES), self.model_row)
        self.model.activated.connect(
            lambda value: self.draft_changed.emit({"values": {"model_kind": value}}))
        model_layout.addWidget(self.model)
        model_layout.addStretch(1)
        layout.addWidget(self.model_row)
        self._entries: tuple[dict, ...] = ()
        self._columns: tuple = ()
        self._shown: tuple = ()
        self._base_dir = ""
        self._filter = "All files (*)"

    def label_width_hint(self) -> int:
        """The label column this editor's title, model label and file names need."""

        names = self.grid.label_width_hint() if self._entries else 0
        return max(names, self.title.sizeHint().width(), self.model_label.sizeHint().width())

    def set_label_width(self, width: int) -> None:
        """The editor's label column, shared with the form's "Frames" row."""

        self.title.setFixedWidth(int(width))
        self.model_label.setFixedWidth(int(width))
        self.grid.set_label_width(int(width))

    # ---------------------------------------------------------------- reading

    def update_projection(self, projection) -> None:
        claims = projection["artifact_claims"][CALIBRATIONS]
        self._entries = tuple(
            {"path": entry["path"], "frames": tuple(entry["frames"])}
            for entry in claims["entries"]
        )
        self._base_dir = str(claims["base_dir"])
        self._filter = str(claims["file_filter"])
        frames = projection.get("source_frames")
        self.model.set_current(projection["form_values"]["model_kind"])
        self._draw(frames, claims["files"], claims["problems"])

    def _draw(self, frames, files, problems) -> None:
        entries = self._entries
        count = None if frames is None else len(frames)
        claimed = sorted({
            token for entry in entries for token in entry["frames"] if isinstance(token, int)
        })
        present = () if count is None else tuple(range(count))
        # A number the source does not carry keeps a column of its own, so
        # the claim stays visible -- and one click away from being removed.
        ghosts = tuple(number for number in claimed if count is None or number >= count)
        self._columns = (*(token for token, _label in _NAMED), _GAP, *present, *ghosts)
        labels = (
            *(label for _token, label in _NAMED), "",
            *(frames[number] for number in present), *(str(number) for number in ghosts),
        )
        reach = None if count is None or not entries else reach_frames(
            tuple(entry["frames"] for entry in entries), count)
        cells: dict[tuple[int, int], str] = {}
        selected: list[tuple[int, int]] = []
        states: dict[tuple[int, int], str] = {}
        tips: dict[tuple[int, int], str] = {}
        names = tuple(_name(entry) for entry in entries)
        for row, entry in enumerate(entries):
            for column, token in enumerate(self._columns):
                if token is _GAP:
                    continue
                label = labels[column]
                if isinstance(token, int) and token not in present:
                    if token not in entry["frames"]:
                        continue
                    states[(row, column)] = "warning"
                    tips[(row, column)] = (
                        f"frame {token} is not in this source; click to drop the claim"
                        if count is not None else f"frame {token}"
                    )
                elif isinstance(token, int) and reach is not None:
                    readers = reach.readers[token]
                    mine = next((how for reader, how in readers if reader == row), None)
                    end = None if mine is not None else _held_by(entries, row, token, count)
                    if end is not None:
                        # Another file's First or Last: that column moves it,
                        # the number does not.
                        holder, how = end
                        word = how.capitalize()
                        states[(row, column)] = "held"
                        tips[(row, column)] = (
                            f"{names[holder]} reads frame {token} as {word}; click {word} "
                            "in this row to read it here")
                    elif len(readers) > 1 and mine is not None:
                        states[(row, column)] = "warning"
                        tips[(row, column)] = "read by " + " and ".join(
                            f"{names[reader]} ({how})" for reader, how in readers)
                    elif not readers:
                        states[(row, column)] = "missing"
                        tips[(row, column)] = (
                            f"frame {token} has no calibration; click to give it to this file")
                    elif mine in (FIRST, LAST, REST):
                        states[(row, column)] = "derived"
                        tips[(row, column)] = f"read as {mine}"
                cells[(row, column)] = label
                if token in entry["frames"]:
                    selected.append((row, column))
        notes = tuple(
            "cannot be read" if item is None and entry["path"] else
            "no file yet" if item is None else item["summary"]
            for entry, item in zip(entries, files)
        )
        row_tips = tuple(
            (entry["path"] or "no file yet")
            + ("" if reach is None else
               "\nreads " + (", ".join(str(frame) for frame in reach.frames_of(index)) or "no frame"))
            for index, entry in enumerate(entries)
        )
        shown = (names, labels, tuple(sorted(cells.items())), notes, row_tips)
        if shown != self._shown:
            self._shown = shown
            self.grid.set_grid(
                names, labels, cells, row_notes=notes, row_tips=row_tips,
                narrow_columns=(self._columns.index(_GAP),),
            )
        self.grid.set_cell_states(states, tips)
        if self.grid.selected_cells() != tuple(sorted(selected)):
            self.grid.set_selected(selected)
        self.grid.setVisible(bool(entries))
        self.model_row.setVisible(bool(entries))
        self.hint.setText(_HINT if entries else _EMPTY)
        if not entries:
            self.summary.setText("No calibration yet")
        elif count is None:
            self.summary.setText("Waiting for the source frames")
        elif problems:
            self.summary.setText(f"{count} frame{'s' if count != 1 else ''}")
        elif len(entries) == 1:
            self.summary.setText(f"{names[0]} reads every frame")
        else:
            self.summary.setText(f"{count} frames · each read once")
        self.note.setText("\n".join(problems))
        self.note.setVisible(bool(problems))

    # ---------------------------------------------------------------- editing

    def _marked(self, cells) -> None:
        before = set(self.grid_cells_claimed())
        after = set(cells)
        frames = [list(entry["frames"]) for entry in self._entries]
        for row, column in sorted(before - after):
            token = self._columns[column]
            if token in frames[row]:
                frames[row].remove(token)
        for row, column in sorted(after - before):
            token = self._columns[column]
            # A frame, First, Last or Rest belongs to one file: taking it here takes
            # it from whichever file had it.  First and 0 are one frame in every
            # cycle, so taking either takes both.
            same = (FIRST, 0) if token in (FIRST, 0) else (token,)
            for index, other in enumerate(frames):
                if index == row:
                    continue
                for taken in same:
                    if taken in other:
                        other.remove(taken)
            frames[row].append(token)
        self._write(tuple(
            {"path": entry["path"], "frames": tuple(claimed)}
            for entry, claimed in zip(self._entries, frames)
        ))

    def grid_cells_claimed(self) -> tuple[tuple[int, int], ...]:
        """The cells the draft claims, as last drawn."""

        return tuple(
            (row, column)
            for row, entry in enumerate(self._entries)
            for column, token in enumerate(self._columns)
            if token is not _GAP and token in entry["frames"]
        )

    def row_action(self, row: int, action: object) -> None:
        if action == "remove":
            self._write(tuple(entry for index, entry in enumerate(self._entries) if index != row))
        elif action == "replace":
            path = fluent_open_path(self, "Replace calibration", self._start(row), self._filter)
            if path:
                self.replace_file(row, path)

    def add_file(self, path: str) -> None:
        """Another calibration; the first one reads every frame, a later one none yet."""

        taken = any(REST in entry["frames"] for entry in self._entries)
        self._write((*self._entries, {"path": str(path), "frames": () if taken else (REST,)}))

    def replace_file(self, row: int, path: str) -> None:
        self._write(tuple(
            {"path": str(path), "frames": entry["frames"]} if index == row else entry
            for index, entry in enumerate(self._entries)
        ))

    def _browse_new(self) -> None:
        path = fluent_open_path(self, "Add calibration", self._start(None), self._filter)
        if path:
            self.add_file(path)

    def _start(self, row) -> str:
        if row is not None and self._entries[row]["path"]:
            return self._entries[row]["path"]
        return self._base_dir

    def _write(self, entries) -> None:
        self.draft_changed.emit({"artifact_inputs": {CALIBRATIONS: tuple(
            {"path": entry["path"], "frames": canonical_frames(entry["frames"])} for entry in entries
        )}})

    def set_mutation_enabled(self, enabled: bool) -> None:
        self.setEnabled(bool(enabled))


def _held_by(entries, row: int, frame: int, count: int) -> tuple[int, str] | None:
    """The other file reading ``frame`` as its First or Last, and which.

    Read from the raw claims: a file that also names the frame by number
    still holds it through First or Last, and only that column moves it.
    """

    for holder, entry in enumerate(entries):
        if holder == row:
            continue
        if frame == 0 and FIRST in entry["frames"]:
            return holder, FIRST
        if frame == count - 1 and LAST in entry["frames"]:
            return holder, LAST
    return None


def _name(entry) -> str:
    return Path(entry["path"]).name if entry["path"] else "(no file)"


__all__ = ["CalibrationsEditor"]
