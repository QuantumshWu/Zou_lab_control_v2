"""Discoverable frame-survival processor descriptor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
)
from zlc_atom.nodes._framework.frames import frame_identity

from .processor import (
    SURVIVAL_OUTPUTS, FrameSurvivalProcessor, _checked_pairs, _checked_overrides,
    _rule_pairs, _selected_pairs, pair_label,
)

#: What the operator picks: a question that follows the frames the source
#: carries, no pair at all, or the pairs they marked themselves.  The labels
#: are the words on the editor's preset strip and in its summary, so they are
#: short; what each rule pairs (k→k+1, 0→k, k→last) is in the field's
#: description.  None and Selected are two choices, not one: None asks for no
#: pair, Selected for the list being marked -- empty while it is begun -- so
#: the strip marks the one the operator picked and either can follow the other.
_SELECTIONS = (
    AuthoringChoice("all", "All"),
    AuthoringChoice("consecutive", "Consecutive"),
    AuthoringChoice("from_first", "From first"),
    AuthoringChoice("to_last", "To last"),
    AuthoringChoice("none", "None"),
    AuthoringChoice("custom", "Selected"),
)
#: The choices that name their pairs rather than follow a rule.
_EXPLICIT = ("none", "custom")


def _validate(values):
    overrides = _checked_overrides(tuple((row["start"], row["end"], row["selected"])
                                        for row in values["pairs"]))
    if values["selection"] == "none" or (
        values["selection"] == "custom" and not any(selected for _start, _end, selected in overrides)
    ):
        raise ValueError("Pick a rule or select at least one pair.")
    if values["selection"] == "custom":
        _checked_pairs(tuple((start, end) for start, end, selected in overrides if selected))
        if any(not selected for _start, _end, selected in overrides):
            raise ValueError("Selected (custom) lists only the pairs it includes; remove rows marked not selected.")

FRAME_SURVIVAL_SCHEMA = AuthoringSchema((
    AuthoringField("selection", "choice", "Published pairs", "all", required=True, choices=_SELECTIONS,
                   description="A named choice follows the frames the source carries: All forward pairs, "
                               "Consecutive (k→k+1), From first (0→k) or To last (k→last); None asks for "
                               "no pair; Selected lists them. Grid edits override only the changed pairs; first/last follow "
                               "the cycle."),
    AuthoringField("pairs", "rows", "Pair selections", (), columns=(
        AuthoringField("start", "text", "Start frame", "0", required=True),
        AuthoringField("end", "text", "End frame", "last", required=True),
        AuthoringField("selected", "bool", "Selected", True),
    )),
), validator=_validate)


def _build(*, source_signal: str, **values: object) -> FrameSurvivalProcessor:
    authored = FRAME_SURVIVAL_SCHEMA.project_values(values)
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    selection = authored["selection"]
    overrides = tuple((row["start"], row["end"], row["selected"]) for row in authored["pairs"])
    return FrameSurvivalProcessor(
        source_signal=selected_source,
        pairs=selection if selection != "custom" else tuple(
            (start, end) for start, end, _selected in overrides),
        pair_overrides=overrides if selection != "custom" else (),
    )


def _editor_factory(parent=None):
    from PyQt5 import QtCore, QtWidgets
    from zlc_ui.fluent import (
        FluentSegmentedButtons, FluentSelectionGrid, error_note_label, muted_note_label, row_gap, show_problem,
    )
    from zlc_ui.fluent.column_set import section_header
    from .processor import PAIR_RULES

    labels = {choice.value: choice.label for choice in _SELECTIONS}

    class SurvivalForm(QtWidgets.QWidget):
        """Mark pairs on a start x end grid, or pick a named selection.

        The grid takes the ordinary table gestures: a click selects one pair,
        a drag a rectangle, Ctrl adds or removes, a row or column header every
        pair from or to that frame. Edits retain the current rule and record
        only exceptions; None starts an explicit list.

        The grid reads the way Occupancy's does: a pair the RULE selects is
        a derived card (the light wash, dashed), a pair the operator added
        against the rule is a chosen card, and one they left out is an
        empty card -- so what the rule decided and what was decided by hand
        are told apart at a glance.  The strip is the selection's closed
        set and always marks the one picked: a rule, None (no pair), or
        Selected (the marked list, empty while it is begun).  Selected,
        picked under a rule, keeps the rule's pairs as the list; picked
        under None it begins an empty one.  A pair marked under None begins
        a Selected list with it.

        "Pairs" heads the section in the editor's label column, so the strip
        and the grid's first card start on the form's field edge.
        """

        draft_changed = QtCore.pyqtSignal(object)
        managed_fields = ("selection", "pairs")

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(row_gap())
            self.presets = FluentSegmentedButtons(
                (*((rule, labels[rule]) for rule in PAIR_RULES), ("none", labels["none"]), ("custom", labels["custom"])),
                self)
            self.presets.activated.connect(self._preset)
            self.summary = muted_note_label("")
            self.title, header = section_header("Pairs", lead=(self.presets,), summary=self.summary)
            layout.addLayout(header)
            self.table = FluentSelectionGrid(self)
            self.table.selection_edited.connect(self._marked)
            layout.addWidget(self.table, 0, QtCore.Qt.AlignLeft)
            self.note = error_note_label()
            layout.addWidget(self.note)
            self._frames = None
            self._selection = "all"
            self._overrides = ()
            self._wanted = ()

        @property
        def model(self):
            return self.table.model()

        def row_labels(self):
            """The labels this form puts in the editor's label column."""

            header = self.table.verticalHeader()
            return (self.title.text(), *(
                str(self.model.headerData(row, QtCore.Qt.Vertical)) for row in range(header.count())))

        def set_label_width(self, width):
            """The editor's label column: the strip and the first card start on its field edge."""

            self.title.setFixedWidth(int(width))
            self.table.set_label_width(int(width))

        def _preset(self, choice):
            if choice == self._selection:
                return
            pairs = ()
            if choice == "custom" and self._frames is not None:
                # What is selected now becomes the list: a rule's pairs, or
                # none at all after None.
                count = len(self._frames)
                pairs = tuple({"start": str(frame_identity(start, count)), "end": str(frame_identity(end, count)),
                               "selected": True} for start, end in self._wanted)
            self.draft_changed.emit({"values": {"selection": choice, "pairs": pairs}})

        def _marked(self, cells):
            if self._frames is None:
                return
            count = len(self._frames)
            previous = set(self._wanted)
            current = {(row, column + 1) for row, column in cells}
            edits = {(start, end): selected for start, end, selected in self._overrides}
            baseline = set() if self._selection in _EXPLICIT else set(_rule_pairs(self._selection, count))
            for pair in previous ^ current:
                identity = tuple(frame_identity(index, count) for index in pair)
                if (pair in current) == (pair in baseline):
                    edits.pop(identity, None)
                else:
                    edits[identity] = pair in current
            self.draft_changed.emit({"values": {
                # A pair marked under None begins a Selected list.
                "selection": "custom" if self._selection == "none" else self._selection,
                "pairs": tuple({"start": str(start), "end": str(end), "selected": selected}
                               for (start, end), selected in edits.items()),
            }})

        def update_projection(self, projection):
            values = projection["form_values"]
            # The console's one answer to "which frames does the source carry".
            frames = projection.get("source_frames")
            frames = None if frames is None or len(frames) < 2 else tuple(frames)
            if frames != self._frames:
                self._frames = frames
                count = 0 if frames is None else len(frames)
                self.table.set_grid(
                    tuple(f"{frames[start]} →" for start in range(count - 1)),
                    tuple(f"→ {frames[end]}" for end in range(1, count)),
                    {(start, end - 1): pair_label(frames[start], frames[end])
                     for start in range(count - 1) for end in range(start + 1, count)},
                )
            selection = values["selection"]
            self._selection = selection
            self._overrides = _checked_overrides(tuple(
                (row["start"], row["end"], row.get("selected", True)) for row in values["pairs"]
                if row["start"] not in (None, "") and row["end"] not in (None, "")))
            self.presets.set_current(selection)
            self.table.setEnabled(frames is not None)
            if frames is None:
                self.table.set_cell_states({})
                self._wanted = ()
                self.summary.setText("Waiting for the source frames")
                show_problem(self.note, "")
                return
            count = len(frames)
            listed = () if selection == "none" else tuple(
                (start, end) for start, end, selected in self._overrides if selected)
            wanted, absent = _selected_pairs(
                selection if selection not in _EXPLICIT else listed,
                self._overrides if selection not in _EXPLICIT else (), count)
            self._wanted = wanted
            # What the rule selects by itself is derived; what is selected
            # beyond it was marked by hand and stays a chosen card.  Either
            # way the pair is in the table's selection, so the gestures
            # (and the draft they edit) are the table's own.
            rule = set() if selection in _EXPLICIT else set(_rule_pairs(selection, count))
            derived = tuple(pair for pair in wanted if pair in rule)
            self.table.set_cell_states(
                {(start, end - 1): "derived" for start, end in derived},
                {(start, end - 1): f"{pair_label(frames[start], frames[end])} follows {labels[selection]}; "
                                   "Ctrl-click to leave it out" for start, end in derived},
            )
            cells = tuple(sorted((start, end - 1) for start, end in wanted))
            if self.table.selected_cells() != cells:
                self.table.set_selected(cells)
            # Named with the strip's own word for its current segment.
            self.summary.setText(
                f"{self.presets.button(selection).text()} · {len(wanted)} of {count * (count - 1) // 2} pairs")
            show_problem(self.note, "" if not absent else "not in this source: " + ", ".join(
                pair_label(start, end) for start, end, _selected in absent))

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
        "frames the source carries; none asks for no pair (Start waits for a rule or a pair). "
        "Pair rows override the rule with selected=true/false; "
        "custom lists only included pairs. Endpoints use text '0'/'last'; numbered "
        "middle frames keep their index when the cycle changes. "
        "Feed survival to Seamless Scan and use a site Facet with native Exponential "
        "fitting for per-site lifetimes."
    ),
    input_specs=(DatasetInputSpec("occupied", None, "exact"),),
    outputs=SURVIVAL_OUTPUTS,
    ui_contributions=(_editor_factory,),
    build=_build,
)

__all__ = ["FRAME_SURVIVAL_SCHEMA", "LOGIC_NODE"]
