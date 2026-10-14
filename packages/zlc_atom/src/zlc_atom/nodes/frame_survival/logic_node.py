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
    from PyQt5 import QtCore, QtWidgets
    from zlc_data import READOUT_EVENT
    from zlc_ui.fluent import (
        FluentLabel, FluentSegmentedButtons, FluentSelectionGrid, muted_note_label,
    )
    from .processor import PAIR_RULES, _rule_pairs

    labels = {choice.value: choice.label for choice in _SELECTIONS}
    short = {"all": "All", "consecutive": "Consecutive", "from_first": "From first", "to_last": "To last"}

    class SurvivalForm(QtWidgets.QWidget):
        """Mark pairs on a start x end grid, or pick a named selection.

        The grid takes the ordinary table gestures: a click selects one pair,
        a drag a rectangle, Ctrl adds or removes, a row or column header every
        pair from or to that frame.  Marking anything makes the selection
        explicit; a preset returns to a rule.
        """

        draft_changed = QtCore.pyqtSignal(object)
        managed_fields = ("selection", "pairs")

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            header = QtWidgets.QHBoxLayout()
            self.presets = FluentSegmentedButtons(tuple((rule, short[rule]) for rule in PAIR_RULES), self)
            self.presets.activated.connect(lambda rule: self.draft_changed.emit(
                {"values": {"selection": rule, "pairs": ()}}))
            header.addWidget(self.presets)
            header.addStretch(1)
            self.summary = muted_note_label("")
            header.addWidget(self.summary)
            layout.addLayout(header)
            self.table = FluentSelectionGrid(self)
            self.table.selection_edited.connect(self._marked)
            layout.addWidget(self.table, 0, QtCore.Qt.AlignLeft)
            self.note = FluentLabel("")
            self.note.setWordWrap(True)
            layout.addWidget(self.note)
            self._frames = None

        @property
        def model(self):
            return self.table.model()

        def _marked(self, cells):
            if self._frames is None:
                return
            self.draft_changed.emit({"values": {
                "selection": "custom",
                "pairs": tuple({"start": row, "end": column + 1} for row, column in cells),
            }})

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
                self._frames = frames
                count = 0 if frames is None else len(frames)
                self.table.set_grid(
                    tuple(f"{frames[start]} →" for start in range(count - 1)),
                    tuple(f"→ {frames[end]}" for end in range(1, count)),
                    {(start, end - 1): f"{frames[start]}→{frames[end]}"
                     for start in range(count - 1) for end in range(start + 1, count)},
                )
            selection = values["selection"]
            explicit = tuple((row["start"], row["end"]) for row in values["pairs"]
                             if row["start"] is not None and row["end"] is not None)
            self.presets.set_current(selection)
            self.table.setEnabled(frames is not None)
            if frames is None:
                self.summary.setText(labels[selection])
                self.note.setText("Waiting for the source frames")
                return
            count = len(frames)
            wanted = (_rule_pairs(selection, count) if selection != "custom"
                      else tuple(pair for pair in explicit if pair[1] < count))
            absent = () if selection != "custom" else tuple(pair for pair in explicit if pair[1] >= count)
            cells = tuple(sorted((start, end - 1) for start, end in wanted))
            if self.table.selected_cells() != cells:
                self.table.set_selected(cells)
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
        "Mean over repeats gives conditional survival. Choose the pairs with selection: "
        "all, consecutive (k->k+1), from_first (0->k) and to_last (k->last) follow the "
        "frames the source carries; custom publishes exactly the listed pairs rows. "
        "Feed survival to Seamless Scan and use a site Facet with native Exponential "
        "fitting for per-site lifetimes."
    ),
    input_specs=(DatasetInputSpec("occupied", None, "exact"),),
    outputs=SURVIVAL_OUTPUTS,
    ui_contributions=(_editor_factory,),
    build=_build,
)

__all__ = ["FRAME_SURVIVAL_SCHEMA", "LOGIC_NODE"]
