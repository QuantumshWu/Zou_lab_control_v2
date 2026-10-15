"""Occupancy's calibrations: one row per file, the frames it reads as columns.

The operator says which file reads which frames the way they think it --
this one the first frame, that one the last, another the rest -- by marking
cells.  The ends of the cycle are named by what they are: "First" is frame
0 whatever the count and "Last" follows the count, so they have their own
columns and no numbered column repeats them; the numbered columns are the
MIDDLE frames, which keep their number.  "Rest" takes whatever no file
names.  A frame reached through Rest shows as a dashed card in its file's
row (on First or Last when it is an end); a frame nobody reads is an empty
red card in every row, and one two files reach, or a middle number the
source no longer has between its ends, is a red-edged card with the reason
below.  Nothing here changes a claim
because the frame count changed: what was authored stays until the operator
edits it.  A file is removed by the × that ends its row and replaced by
clicking its name.  A file row is one line, as tall as every other
control: what the file is (its exposure, sites and model) is in the name's
tooltip with its path.  The readout model -- which model each file is read
with, set once -- is a segmented choice under the files it applies to: the
four models are this node's closed set.

The section opens like every editor's section: its title, then at the right
end its summary and "Add calibration…".  The file names stand in the
editor's label column, shared with the form, so the first card, the model
choice and the form's fields start on one edge.  The column is set by the
labels that stay ("Readout model", the form's) and a fixed room for file
names, never by a file name itself, so it does not move as files come and
go.  A row is named as every sentence about it names it
(:func:`claim_names`: the file's stem, or "row N" before it has a file),
elided in the middle when it is wider than the column; the tooltip reads
the whole path.  The console resolves the
files and projects how the source's frames read them; this widget only
draws that and writes the edits back.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.fluent import (
    FONT,
    TEXT,
    FluentButton,
    FluentLabel,
    FluentSegmentedButtons,
    FluentSelectionGrid,
    error_note_label,
    fluent_font_size,
    fluent_open_path,
    muted_note_label,
    retire_widget,
    row_gap,
    scaled_px,
    setting_label_width,
    show_note,
)
from zlc_ui.status import sentence
from zlc_ui.fluent.column_set import section_header

from zlc_atom.nodes._framework.frames import FIRST, LAST, REST, canonical_frames, claim_names, reach_frames
from zlc_atom.nodes.calibration import DEFAULT_READOUT_MODEL_CHOICE, READOUT_MODEL_CHOICES

#: The artifact input this editor manages.
CALIBRATIONS = "calibrations"
_NAMED = ((FIRST, "First"), (LAST, "Last"), (REST, "Rest"))
#: Design px of label column a calibration file name gets (see label_width_floor).
_FILE_NAME_ROOM = 150
_GAP = None
_HINT = ("Click a cell to give that frame to a file · First and Last follow the cycle's ends, "
         "middle frames keep their number")
_EMPTY = "Add a calibration file; the first one reads every frame"


class CalibrationsEditor(QtWidgets.QWidget):
    """Calibration files as rows, frames as columns; marked cells are claims."""

    draft_changed = QtCore.pyqtSignal(object)
    managed_fields = ("model_kind",)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(row_gap())
        self.summary = muted_note_label("")
        self.add_button = FluentButton("Add calibration…")
        self.add_button.clicked.connect(self._browse_new)
        _title, header = section_header("Calibrations", summary=self.summary, actions=(self.add_button,))
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
        # One red block per problem, half a row gap apart: a problem that
        # wraps is still told from the next one.
        self.problems = QtWidgets.QWidget(self)
        self._problem_layout = QtWidgets.QVBoxLayout(self.problems)
        self._problem_layout.setContentsMargins(0, 0, 0, 0)
        self._problem_layout.setSpacing(row_gap() // 2)
        self._problem_labels: list[QtWidgets.QLabel] = []
        self.problems.hide()
        layout.addWidget(self.problems)
        self.model_row = QtWidgets.QWidget(self)
        model_layout = QtWidgets.QHBoxLayout(self.model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(row_gap())
        # A field label like "Frames" above it, not a second hint line.
        self.model_label = FluentLabel("Readout model")
        self.model_label.setStyleSheet(f"color: {TEXT}; background: transparent; border: none;")
        model_layout.addWidget(self.model_label)
        self.model = FluentSegmentedButtons(
            tuple((choice.value, choice.label) for choice in READOUT_MODEL_CHOICES), self.model_row)
        self.model.button(DEFAULT_READOUT_MODEL_CHOICE).setToolTip(
            "Read each file with its calibration's own default model")
        self.model.activated.connect(self._model_picked)
        model_layout.addWidget(self.model)
        model_layout.addStretch(1)
        layout.addWidget(self.model_row)
        # The section's hint is its last line, under everything it explains.
        self.hint = muted_note_label(_HINT)
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self._entries: tuple[dict, ...] = ()
        self._columns: tuple = ()
        self._shown: tuple = ()
        self._base_dir = ""
        self._filter = "All files (*)"
        #: The label column the file names are fitted to; None until the host sets it.
        self._label_width: int | None = None
        #: What the grid was last drawn from, to draw it again for a new column.
        self._drawn = None

    def label_width_floor(self) -> int:
        """The room a file name gets in the label column: a stem the way
        calibrations are named (``background_5ms``, ``readout_50ms``) shows
        whole; a longer one is elided in the middle.  Fixed, so the column
        is the same with no file, one or ten."""

        return scaled_px(_FILE_NAME_ROOM)

    def row_labels(self) -> tuple[str, ...]:
        """The labels that set the editor's label column: the model's label.
        The file names stand in the column too, fitted to it (:func:`_fitted`),
        so a long one never moves the form's fields."""

        return (self.model_label.text(),)

    def set_label_width(self, width: int) -> None:
        """The editor's label column, shared with the form's "Frames" row."""

        width = int(width)
        self.model_label.setFixedWidth(width)
        self.grid.set_label_width(width)
        if width != self._label_width:
            self._label_width = width
            if self._drawn is not None:
                self._draw(*self._drawn)

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
        self._drawn = (frames, claims["files"], claims["problems"])
        self._draw(*self._drawn)

    def _draw(self, frames, files, problems) -> None:
        entries = self._entries
        count = None if frames is None else len(frames)
        claimed = sorted({
            token for entry in entries for token in entry["frames"] if isinstance(token, int)
        })
        # The numbered columns are the middle frames; the ends are First and Last.
        middle = () if count is None else tuple(range(1, count - 1))
        # A middle number the source does not carry between its ends keeps a
        # column of its own, so the claim stays visible -- and one click away
        # from being removed.
        ghosts = tuple(number for number in claimed if number not in middle)
        self._columns = (*(token for token, _label in _NAMED), _GAP, *middle, *ghosts)
        ends = {FIRST: 0, LAST: None if count is None else count - 1}
        labels = (
            *(label if count is None else f"{label} · {frames[ends[token]]}"
              if token in ends else label for token, label in _NAMED), "",
            *(frames[number] for number in middle), *(str(number) for number in ghosts),
        )
        reach = None if count is None or not entries else reach_frames(
            tuple(entry["frames"] for entry in entries), count)
        cells: dict[tuple[int, int], str] = {}
        selected: list[tuple[int, int]] = []
        states: dict[tuple[int, int], str] = {}
        tips: dict[tuple[int, int], str] = {}
        names = claim_names(entries)
        for row, entry in enumerate(entries):
            for column, token in enumerate(self._columns):
                if token is _GAP:
                    continue
                label = labels[column]
                frame = (ends[token] if token in ends else
                         token if isinstance(token, int) and token in middle else None)
                if isinstance(token, int) and token not in middle:
                    if token not in entry["frames"]:
                        continue
                    states[(row, column)] = "warning"
                    tips[(row, column)] = (
                        f"middle frame {token} is not in this source; click to drop the claim"
                        if count is not None else f"frame {token}"
                    )
                elif frame is not None and reach is not None:
                    readers = reach.readers[frame]
                    mine = next((how for reader, how in readers if reader == row), None)
                    if len(readers) > 1 and mine is not None:
                        states[(row, column)] = "warning"
                        tips[(row, column)] = "read by " + " and ".join(
                            f"{names[reader]} ({how})" for reader, how in readers)
                    elif not readers:
                        states[(row, column)] = "missing"
                        tips[(row, column)] = (
                            f"frame {frame} has no calibration; click to give it to this file")
                    elif mine == REST and token not in entry["frames"]:
                        states[(row, column)] = "derived"
                        tips[(row, column)] = f"frame {frame} is read as rest"
                cells[(row, column)] = label
                if token in entry["frames"]:
                    selected.append((row, column))
        # What each file is rides with its path in the name's tooltip: a
        # second line under every name made each row two lines tall beside
        # one-line controls, and floated the name above its cards' text.
        row_tips = tuple(
            "\n".join(part for part in (
                entry["path"] or "no file yet",
                "cannot be read" if item is None and entry["path"] else "" if item is None else item["summary"],
                "" if reach is None else
                "reads " + (", ".join(str(frame) for frame in reach.frames_of(index)) or "no frame"),
            ) if part)
            for index, (entry, item) in enumerate(zip(entries, files))
        )
        fitted = tuple(_fitted(name, self._label_width) for name in names)
        shown = (fitted, labels, tuple(sorted(cells.items())), row_tips)
        if shown != self._shown:
            self._shown = shown
            self.grid.set_grid(
                fitted, labels, cells, row_tips=row_tips,
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
        self._show_problems(problems)

    def _show_problems(self, problems) -> None:
        """Each problem as its own sentence in its own block, ending in a full
        stop.  Not capitalised: a problem may start with a file's stem."""

        labels = self._problem_labels
        while len(labels) > len(problems):
            retire_widget(labels.pop())
        while len(labels) < len(problems):
            labels.append(error_note_label())
            self._problem_layout.addWidget(labels[-1])
        for label, problem in zip(labels, problems):
            show_note(label, sentence(problem))
        self.problems.setVisible(bool(problems))

    # ---------------------------------------------------------------- editing

    def _model_picked(self, value) -> None:
        # Marked at once, as a popup shows its pick; the projection the edit
        # comes back as marks it again, or puts back what was refused.
        self.model.set_current(value)
        self.draft_changed.emit({"values": {"model_kind": value}})

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
            # A middle frame, First, Last or Rest belongs to one file: taking
            # it here takes it from whichever file had it.
            for index, other in enumerate(frames):
                if index != row and token in other:
                    other.remove(token)
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


def _fitted(name: str, width: int | None) -> str:
    """``name`` as the label column shows it: whole while it fits the room
    any label has in a ``width`` column (the column less the slack
    :func:`setting_label_width` leaves after its widest label), else elided
    in the middle, keeping the start and the end that tell files apart
    (``readout_50ms`` whole, ``backg…d_5ms``)."""

    if width is None:
        return name
    metrics = QtGui.QFontMetrics(QtGui.QFont(FONT, fluent_font_size()))
    advance = metrics.horizontalAdvance(name)
    room = width - (setting_label_width((name,), minimum=0) - advance)
    return name if advance <= room else metrics.elidedText(name, QtCore.Qt.ElideMiddle, room)


__all__ = ["CalibrationsEditor"]
