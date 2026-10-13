"""Discoverable frame-survival processor descriptor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
)

from .processor import SURVIVAL_OUTPUTS, FrameSurvivalProcessor, _checked_pairs

#: What the operator picks: a question that follows the frames the source
#: carries, or the pairs they marked themselves.
_SELECTIONS = (
    AuthoringChoice("all", "All forward pairs"),
    AuthoringChoice("consecutive", "Consecutive frames (k→k+1)"),
    AuthoringChoice("from_first", "From the first frame (0→k)"),
    AuthoringChoice("to_last", "To the last frame (k→last)"),
    AuthoringChoice("custom", "Selected pairs"),
)


def _validate(values):
    if values["selection"] == "custom":
        _checked_pairs(tuple((row["start"], row["end"]) for row in values["pairs"]))
    elif values["pairs"]:
        raise ValueError("Selected pairs apply only to the 'Selected pairs' choice; choose it or clear them")

FRAME_SURVIVAL_SCHEMA = AuthoringSchema((
    AuthoringField("selection", "choice", "Published pairs", "all", required=True, choices=_SELECTIONS,
                   description="A named choice follows the frames the source carries; "
                               "Selected pairs publishes exactly the marked pairs."),
    AuthoringField("pairs", "rows", "Selected pairs", (), columns=(
        AuthoringField("start", "int", "Start frame", 0, required=True, minimum=0),
        AuthoringField("end", "int", "End frame", 1, required=True, minimum=0),
    )),
), validator=_validate)


def _build(*, source_signal: str, **values: object) -> FrameSurvivalProcessor:
    authored = FRAME_SURVIVAL_SCHEMA.project_values(values)
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    selection = authored["selection"]
    return FrameSurvivalProcessor(
        source_signal=selected_source,
        pairs=selection if selection != "custom" else tuple(
            (row["start"], row["end"]) for row in authored["pairs"]),
    )


def _editor_factory(parent=None):
    from PyQt5 import QtCore, QtGui, QtWidgets
    from zlc_data import READOUT_EVENT
    from zlc_ui.fluent import ACCENT, ACCENT_TINT, BG, GRAPHITE, TEXT, FluentButton, FluentLabel, FluentTableView
    from .processor import PAIR_RULES, _rule_pairs

    labels = {choice.value: choice.label for choice in _SELECTIONS}

    class PairMatrix(QtGui.QStandardItemModel):
        """Pair labels to select, never cells to write: the table's paste and
        clear gestures must not rename a pair."""

        def setData(self, index, value, role=QtCore.Qt.EditRole):  # noqa: N802 - Qt API
            return False

        def clear_indices(self, cells):
            return None

        def set_block_data(self, row, column, cells):
            return None

    short = {"all": "All", "consecutive": "Consecutive", "from_first": "From first", "to_last": "To last"}

    class SurvivalForm(QtWidgets.QWidget):
        """Mark pairs on a start × end matrix, or pick a named selection.

        The matrix is an ordinary table selection: a click selects one pair,
        a drag selects a rectangle, Ctrl adds or removes, and a row or column
        header selects every pair from or to that frame.  Marking anything
        makes the selection explicit; a preset button returns to a rule.
        """

        draft_changed = QtCore.pyqtSignal(object)
        managed_fields = ("selection", "pairs")

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            header = QtWidgets.QHBoxLayout()
            self.presets = {}
            for rule in PAIR_RULES:
                button = FluentButton(short[rule], color=GRAPHITE)
                button.clicked.connect(lambda _checked=False, rule=rule: self.draft_changed.emit(
                    {"values": {"selection": rule, "pairs": ()}}))
                header.addWidget(button)
                self.presets[rule] = button
            header.addStretch(1)
            self.summary = FluentLabel("")
            header.addWidget(self.summary)
            layout.addLayout(header)
            self.model = PairMatrix(self)
            self.table = FluentTableView(self)
            self.table.setModel(self.model)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.set_content_rows_limit(16)
            # The pairs fill the editor's width rather than leaving it to blank margin.
            self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            # The selection IS the published set, so it stays visible when the
            # table does not have focus; a backward cell reads as no cell at all.
            self.table.setStyleSheet(self.table.styleSheet() + (
                f"QTableView::item:selected {{ background: {ACCENT_TINT}; color: {TEXT}; }}"
                f"QTableView::item:disabled, QTableView::item:disabled:hover {{ background: {BG}; }}"))
            layout.addWidget(self.table)
            self.note = FluentLabel("")
            self.note.setWordWrap(True)
            layout.addWidget(self.note)
            self._frames = None
            self._reconciling = False
            self.table.selectionModel().selectionChanged.connect(self._marked)

        def _marked(self, *_args):
            if self._reconciling or self._frames is None:
                return
            pairs = sorted((index.row(), index.column() + 1) for index in self.table.selectedIndexes())
            self.draft_changed.emit({"values": {
                "selection": "custom",
                "pairs": tuple({"start": start, "end": end} for start, end in pairs),
            }})

        def _rebuild(self, frames):
            self._reconciling = True
            try:
                self.model.clear()
                self._frames = frames
                if frames is None:
                    return
                count = len(frames)
                self.model.setRowCount(count - 1)
                self.model.setColumnCount(count - 1)
                self.model.setVerticalHeaderLabels([f"{frames[start]} →" for start in range(count - 1)])
                self.model.setHorizontalHeaderLabels([f"→ {frames[end]}" for end in range(1, count)])
                for start in range(count - 1):
                    for end in range(1, count):
                        forward = end > start
                        item = QtGui.QStandardItem(f"{frames[start]}→{frames[end]}" if forward else "")
                        item.setTextAlignment(QtCore.Qt.AlignCenter)
                        item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                                      if forward else QtCore.Qt.NoItemFlags)
                        self.model.setItem(start, end - 1, item)
            finally:
                self._reconciling = False

        def update_projection(self, projection):
            values = projection["form_values"]
            schema = projection.get("source_schema")
            axes = () if schema is None else tuple(
                axis for axis in schema.point_domain.axes if axis.role == READOUT_EVENT)
            axis = axes[0] if len(axes) == 1 and axes[0].size >= 2 else None
            frames = None if axis is None else tuple(
                value if isinstance(value, str) else f"{value:g}"
                for value in (axis.coordinate_at(code) for code in range(axis.size)))
            if frames != self._frames:
                self._rebuild(frames)
            selection = values["selection"]
            explicit = tuple((row["start"], row["end"]) for row in values["pairs"]
                             if row["start"] is not None and row["end"] is not None)
            for rule, button in self.presets.items():
                button.set_color(ACCENT if rule == selection else GRAPHITE)
            self.table.setEnabled(frames is not None)
            if frames is None:
                self.summary.setText(labels[selection])
                self.note.setText("Waiting for the source frames")
                return
            count = len(frames)
            wanted = (_rule_pairs(selection, count) if selection != "custom"
                      else tuple(pair for pair in explicit if pair[1] < count))
            absent = () if selection != "custom" else tuple(pair for pair in explicit if pair[1] >= count)
            marked = {(index.row(), index.column() + 1) for index in self.table.selectedIndexes()}
            if marked != set(wanted):
                chosen = QtCore.QItemSelection()
                for start, end in wanted:
                    index = self.model.index(start, end - 1)
                    chosen.select(index, index)
                self._reconciling = True
                try:
                    self.table.selectionModel().select(chosen, QtCore.QItemSelectionModel.ClearAndSelect)
                finally:
                    self._reconciling = False
            self.summary.setText(f"{labels[selection]} · {len(wanted)} of {count * (count - 1) // 2} pairs")
            self.note.setText("" if not absent else "Not in this source: " + ", ".join(
                f"{start}-{end}" for start, end in absent))

        def set_mutation_enabled(self, enabled):
            self.setEnabled(enabled)

    return SurvivalForm(parent)


LOGIC_NODE = LogicNodeDescriptor(
    "frame_survival",
    NodeKind.PROCESSOR,
    FRAME_SURVIVAL_SCHEMA,
    description=(
        "Turn Occupancy occupied data into survival for the selected earlier/later "
        "frame pairs, retaining site and scan axes. Each value is the later verdict; "
        "validity requires the earlier frame occupied and both frames judgeable, so "
        "Mean over repeats gives conditional survival. Feed survival to Seamless Scan "
        "and use a site Facet with native Exponential fitting for per-site lifetimes."
    ),
    input_specs=(DatasetInputSpec("occupied", None, "exact"),),
    outputs=SURVIVAL_OUTPUTS,
    ui_contributions=(_editor_factory,),
    build=_build,
)

__all__ = ["FRAME_SURVIVAL_SCHEMA", "LOGIC_NODE"]
