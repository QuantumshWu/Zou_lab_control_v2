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
    PAIR_RULES, SURVIVAL_OUTPUTS, FrameSurvivalProcessor, _checked_overrides, _checked_pairs, _rule_pairs,
    _selected_pairs, pair_label,
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
#: A rule's exception: selected adds the pair to the rule, unselected leaves
#: it out.  Every row is in use, so every row is checked.
_OVERRIDE_COLUMNS = (
    AuthoringField("start", "text", "Start frame", "0", required=True),
    AuthoringField("end", "text", "End frame", "last", required=True),
    AuthoringField("selected", "bool", "Selected", True),
)
#: A row of Selected's own list: a selected row is published and checked;
#: one kept unselected is parked -- kept as written, its ends filled in or
#: not, and never checked, against the rows in use or at all.
_LIST_COLUMNS = (
    AuthoringField("start", "text", "Start frame", "0"),
    AuthoringField("end", "text", "End frame", "last"),
    AuthoringField("selected", "bool", "Selected", True),
)
_OVERRIDES = AuthoringField(
    "pairs", "rows", "Rule overrides", (), columns=_OVERRIDE_COLUMNS, enabled_when=("selection", PAIR_RULES),
    description="The rule's exceptions: selected=true adds a pair to it, false leaves one out.")
_LIST = AuthoringField(
    "custom_pairs", "rows", "Selected pairs", (), columns=_LIST_COLUMNS, enabled_when=("selection", ("custom",)),
    description="Selected's own list: selected=true pairs are published; selected=false ones are parked -- "
                "kept as written, never checked and never published. The list is kept while a rule is "
                "picked, so Selected comes back to it.")


def _overrides(rows):
    """A rule's exceptions as checked ``(start, end, selected)``."""

    return _checked_overrides(tuple((row["start"], row["end"], row["selected"]) for row in rows))


def _published(rows):
    """The pairs Selected's list publishes, checked: its selected rows, and only those."""

    for index, row in enumerate(rows, 1):
        for column in _LIST_COLUMNS[:2]:
            if row["selected"] and not row[column.name].strip():
                raise ValueError(f"{_LIST.label} row {index}: {column.label} must be filled in")
    selected = tuple((row["start"], row["end"]) for row in rows if row["selected"])
    return _checked_pairs(selected) if selected else ()


def _parked(rows):
    """The pairs Selected's list keeps unselected, as written."""

    return tuple((row["start"], row["end"]) for row in rows if not row["selected"])


def _validate(values):
    # Only the list the selection reads is checked: the other one is kept as
    # it was written (enabled_when), for when its choice is picked again.  Of
    # Selected's list only the selected rows are: a parked row never stops Start.
    selection = values["selection"]
    if selection in PAIR_RULES:
        _overrides(values["pairs"])
    elif selection != "custom" or not _published(values["custom_pairs"]):
        raise ValueError("Pick a rule or select at least one pair.")


FRAME_SURVIVAL_SCHEMA = AuthoringSchema((
    AuthoringField("selection", "choice", "Published pairs", "all", required=True, choices=_SELECTIONS,
                   description="A named choice follows the frames the source carries: All forward pairs, "
                               "Consecutive (k→k+1), From first (0→k) or To last (k→last); None asks for "
                               "no pair; Selected publishes its own list. Grid edits override only the changed "
                               "pairs; first/last follow the cycle."),
    _OVERRIDES,
    _LIST,
), validator=_validate)


def _build(*, source_signal: str, **values: object) -> FrameSurvivalProcessor:
    authored = FRAME_SURVIVAL_SCHEMA.project_values(values)
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    selection = authored["selection"]
    if selection != "custom":
        return FrameSurvivalProcessor(source_signal=selected_source, pairs=selection,
                                      pair_overrides=_overrides(authored["pairs"]))
    listed = authored["custom_pairs"]
    return FrameSurvivalProcessor(source_signal=selected_source, pairs=_published(listed),
                                  parked_pairs=_parked(listed))


def _editor_factory(parent=None):
    from PyQt5 import QtCore, QtWidgets
    from zlc_ui.fluent import (
        FONT, ORANGE_DARK, FluentSegmentedButtons, FluentSelectionGrid, error_note_label, fluent_font_size,
        muted_note_label, row_gap, show_note,
    )
    from zlc_ui.fluent.column_set import section_header
    from zlc_ui.status import sentence

    labels = {choice.value: choice.label for choice in _SELECTIONS}

    def authored(rows):
        """A list as the grid reads it.  The list the selection does not
        read is kept unchecked, so it may be malformed: the grid then shows
        it empty, and once its choice is picked the draft issue says why."""

        try:
            return _checked_overrides(tuple(
                (row.get("start"), row.get("end"), row.get("selected", True)) for row in rows
                if row.get("start") not in (None, "") and row.get("end") not in (None, "")))
        except ValueError:
            return ()

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
        Selected (the marked list, empty while it is begun).  Selected has a
        list of its own, kept while a rule is picked: picked again, it comes
        back.  With no list to come back to, Selected picked under a rule
        keeps the rule's pairs as the list; picked under None it begins an
        empty one.  None clears every mark, Selected's list with them, and a
        pair marked under None begins a Selected list with it.

        A pair kept but not published this time is said, never refused or
        dropped: one whose middle frame the source does not carry is named in
        an amber note, one a Selected list keeps unselected in a muted one.

        "Pairs" heads the section in the editor's label column, so the strip
        and the grid's first card start on the form's field edge.
        """

        draft_changed = QtCore.pyqtSignal(object)
        managed_fields = ("selection", "pairs", "custom_pairs")

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
            # The error line's shape in the amber of a note: said, not refused.
            self.note = error_note_label()
            self.note.setStyleSheet(
                f'color: {ORANGE_DARK}; background: transparent; border: none; font: {fluent_font_size()}pt "{FONT}";')
            layout.addWidget(self.note)
            self.parked = muted_note_label("")
            self.parked.setWordWrap(True)
            self.parked.setVisible(False)
            layout.addWidget(self.parked)
            self._frames = None
            self._selection = "all"
            self._overrides = ()
            self._listed = ()
            self._keeps_list = False
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
            # A choice on the strip starts afresh: a rule's exceptions go.
            change = {"selection": choice, "pairs": ()}
            if choice == "none":
                # None clears every mark at once, Selected's list with them.
                change["custom_pairs"] = ()
            elif choice == "custom" and not self._keeps_list and self._frames is not None:
                # No list to come back to: what is selected now becomes it --
                # a rule's pairs, or none at all after None.
                count = len(self._frames)
                change["custom_pairs"] = tuple(
                    {"start": str(frame_identity(start, count)), "end": str(frame_identity(end, count)),
                     "selected": True} for start, end in self._wanted)
            self.draft_changed.emit({"values": change})

        def _marked(self, cells):
            if self._frames is None:
                return
            count = len(self._frames)
            rule = self._selection in PAIR_RULES
            previous = set(self._wanted)
            current = {(row, column + 1) for row, column in cells}
            # Under a rule the edits are its exceptions; under Selected they
            # are its list, where a pair unmarked leaves the list and one it
            # keeps unselected stays; under None they begin a list.
            edits = {(start, end): selected for start, end, selected in (
                self._overrides if rule else self._listed if self._selection == "custom" else ())}
            baseline = set(_rule_pairs(self._selection, count)) if rule else set()
            for pair in previous ^ current:
                identity = tuple(frame_identity(index, count) for index in pair)
                if (pair in current) == (pair in baseline):
                    edits.pop(identity, None)
                else:
                    edits[identity] = pair in current
            rows = tuple({"start": str(start), "end": str(end), "selected": selected}
                         for (start, end), selected in edits.items())
            self.draft_changed.emit({"values": {"selection": self._selection, "pairs": rows} if rule else {
                # A pair marked under None begins a Selected list.
                "selection": "custom", "custom_pairs": rows}})

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
            self._overrides = authored(values["pairs"])
            self._listed = authored(values["custom_pairs"])
            self._keeps_list = bool(values["custom_pairs"])
            self.presets.set_current(selection)
            self.table.setEnabled(frames is not None)
            if frames is None:
                self.table.set_cell_states({})
                self._wanted = ()
                self.summary.setText("Waiting for the source frames")
                show_note(self.note, "")
                show_note(self.parked, "")
                return
            count = len(frames)
            listed = self._listed if selection == "custom" else ()
            wanted, absent = _selected_pairs(
                selection if selection in PAIR_RULES else tuple(
                    (start, end) for start, end, selected in listed if selected),
                self._overrides if selection in PAIR_RULES else (), count)
            self._wanted = wanted
            # What the rule selects by itself is derived; what is selected
            # beyond it was marked by hand and stays a chosen card.  Either
            # way the pair is in the table's selection, so the gestures
            # (and the draft they edit) are the table's own.
            rule = set(_rule_pairs(selection, count)) if selection in PAIR_RULES else set()
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
            parked = tuple((start, end) for start, end, selected in listed if not selected)
            show_note(self.note, "" if not absent else sentence("kept, not in this source: " + ", ".join(
                pair_label(start, end) for start, end, _selected in absent)))
            show_note(self.parked, "" if not parked else sentence("kept, not selected: " + ", ".join(
                pair_label(start, end) for start, end in parked)))

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
        "Under a rule, pairs rows override it with selected=true/false; custom publishes "
        "its own custom_pairs list, whose selected=false rows are parked: kept as written, "
        "never checked and never published. That list is kept while a rule is picked. "
        "Endpoints use text "
        "'0'/'last'; numbered middle frames keep their index when the cycle changes, and "
        "a selected pair whose middle frame the source lacks is kept and left out of the "
        "run, which fails only when no selected pair is left. "
        "Feed survival to Seamless Scan and use a site Facet with native Exponential "
        "fitting for per-site lifetimes."
    ),
    input_specs=(DatasetInputSpec("occupied", None, "exact"),),
    outputs=SURVIVAL_OUTPUTS,
    ui_contributions=(_editor_factory,),
    build=_build,
)

__all__ = ["FRAME_SURVIVAL_SCHEMA", "LOGIC_NODE"]
