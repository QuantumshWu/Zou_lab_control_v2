"""Discoverable frame-survival processor descriptor."""

from __future__ import annotations

from dataclasses import replace

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
                "kept as written, never checked and never published. The list is kept while a rule or None "
                "is picked, so Selected comes back to it.")


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
        FluentSegmentedButtons, FluentSelectionGrid, card_title_row, error_note_label, muted_note_label, row_gap,
        set_status_ink, show_note, show_problem,
    )
    from zlc_ui.status import sentence

    labels = {choice.value: choice.label for choice in _SELECTIONS}
    # Each list projected on its own, as the draft projects it under its own choice.
    lists = {field.name: AuthoringSchema((replace(field, enabled_when=None),)) for field in (_OVERRIDES, _LIST)}

    def read(field, value):
        """One list as the grid reads it: its rows as the draft projects them
        and the pairs it uses, checked -- or why the grid cannot read it.  A
        list holds anything it was given while its choice is not picked (it
        is kept unchecked), or when the lab API wrote it."""

        try:
            rows = lists[field.name].project_values({field.name: value})[field.name]
            return rows, (_overrides(rows) if field is _OVERRIDES else _published(rows)), ""
        except (TypeError, ValueError) as error:
            return None, None, f"{field.label} cannot be marked on the grid: {error}"

    def named(row):
        """The pair a parked row names, or None: a parked row is never checked."""

        try:
            return _checked_pairs(((row["start"], row["end"]),))[0]
        except ValueError:
            return None

    class SurvivalForm(QtWidgets.QWidget):
        """Mark pairs on a start x end grid, or pick a named selection.

        The grid takes the ordinary table gestures; under a rule, edits are
        recorded as the rule's exceptions.  It reads the way Occupancy's
        does: a pair the RULE selects is a derived card, one added against
        the rule a chosen card, one left out an empty card.  The strip is
        the selection's closed set (``_SELECTIONS``); Selected keeps its own
        list (``_LIST``) while a rule or None is picked.  With no list to
        come back to, Selected picked under a rule keeps the rule's pairs as
        the list, under None it begins an empty one, and a pair marked under
        None joins the list and picks Selected.  In the list a pair unmarked
        leaves it and one marked is added, or selected in place when it is
        parked; every other row stays as written.  A pair kept but not
        published is said (amber when the source lacks its middle frame,
        muted when parked); a list the grid cannot read is said in red and
        the grid is locked.
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
            header, self.title = card_title_row("Pairs", self.summary, lead=(self.presets,))
            layout.addLayout(header)
            self.table = FluentSelectionGrid(self)
            self.table.selection_edited.connect(self._marked)
            layout.addWidget(self.table, 0, QtCore.Qt.AlignLeft)
            self.error = error_note_label()
            layout.addWidget(self.error)
            # The error line's shape in the warning ink: said, not refused.
            self.note = error_note_label()
            set_status_ink(self.note, "warning")
            layout.addWidget(self.note)
            self.parked = muted_note_label("")
            self.parked.setWordWrap(True)
            self.parked.setVisible(False)
            layout.addWidget(self.parked)
            self._frames = None
            self._selection = "all"
            self._overrides = ()
            self._rows = ()
            self._row_pairs = ()
            self._listed = ()
            self._keeps_list = False
            self._problem = ""
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
            # Selected's list stays, for Selected to come back to.
            change = {"selection": choice, "pairs": ()}
            if choice == "custom" and not self._keeps_list and self._frames is not None:
                # No list to come back to: what is selected now becomes it --
                # a rule's pairs, or none at all after None.
                count = len(self._frames)
                change["custom_pairs"] = tuple(
                    {"start": str(frame_identity(start, count)), "end": str(frame_identity(end, count)),
                     "selected": True} for start, end in self._wanted)
            self.draft_changed.emit({"values": change})

        def _marked(self, cells):
            if self._frames is None or self._problem:
                return
            count = len(self._frames)
            current = {(row, column + 1) for row, column in cells}

            def identity(pair):
                return tuple(frame_identity(index, count) for index in pair)

            if self._selection in PAIR_RULES:
                # The edits are the rule's exceptions.
                edits = {(start, end): selected for start, end, selected in self._overrides}
                baseline = set(_rule_pairs(self._selection, count))
                for pair in set(self._wanted) ^ current:
                    if (pair in current) == (pair in baseline):
                        edits.pop(identity(pair), None)
                    else:
                        edits[identity(pair)] = pair in current
                self.draft_changed.emit({"values": {"selection": self._selection, "pairs": tuple(
                    {"start": str(start), "end": str(end), "selected": selected}
                    for (start, end), selected in edits.items())}})
                return
            # The edits are Selected's list.  Under None none of it is shown,
            # so nothing of it is unmarked: a marked pair joins it.
            previous = set(self._listed)
            if self._selection == "none":
                current |= previous
            removed = {identity(pair) for pair in previous - current}
            added = [identity(pair) for pair in sorted(current - previous)]
            rows = []
            for row, pair in zip(self._rows, self._row_pairs):
                if row["selected"] and pair in removed:
                    continue
                if not row["selected"] and pair in added:
                    added.remove(pair)
                    row = {**row, "selected": True}
                rows.append(row)
            rows.extend({"start": str(start), "end": str(end), "selected": True} for start, end in added)
            self.draft_changed.emit({"values": {"selection": "custom", "custom_pairs": tuple(rows)}})

        def update_projection(self, projection):
            values = projection["form_values"]
            # The console's one answer to "which frames does the source carry".
            carried = projection.get("source_frames")
            frames = None if carried is None or len(carried) < 2 else tuple(carried)
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
            rule = selection in PAIR_RULES
            _rows, self._overrides, overrides_problem = read(_OVERRIDES, values["pairs"])
            self._rows, published, list_problem = read(_LIST, values["custom_pairs"])
            # A list the grid cannot read is still a list to come back to.
            self._keeps_list = self._rows is None or bool(self._rows)
            # Gestures write a rule's exceptions, or Selected's list -- under
            # None too: one the grid cannot read is said, never written over.
            self._problem = overrides_problem if rule else list_problem
            show_problem(self.error, self._problem)
            self.presets.set_current(selection)
            self.table.setEnabled(frames is not None and not self._problem)
            if frames is None or self._problem:
                self.table.set_cell_states({})
                self.table.set_selected(())
                self._wanted = ()
                # A source of one frame per cycle is known, not awaited: say
                # why there is no pair to mark.  A list the grid cannot read
                # keeps the strip's name; the red line under it says why.
                self.summary.setText(
                    "Waiting for the source frames" if carried is None
                    else "One frame per cycle: no pairs" if frames is None
                    else self.presets.button(selection).text()
                )
                show_note(self.note, "")
                show_note(self.parked, "")
                return
            count = len(frames)
            # Under a rule Selected's list is not the grid's, and may be unread.
            rows = self._rows or ()
            selected = iter(published or ())
            self._row_pairs = tuple(next(selected) if row["selected"] else named(row) for row in rows)
            self._listed, listed_absent = _selected_pairs(
                tuple(pair for row, pair in zip(rows, self._row_pairs) if row["selected"]), (), count)
            wanted, absent = (
                _selected_pairs(selection, self._overrides, count) if rule
                else (self._listed, listed_absent) if selection == "custom" else ((), ()))
            self._wanted = wanted
            # What the rule selects by itself is derived; what is selected
            # beyond it was marked by hand and stays a chosen card.  Either
            # way the pair is in the table's selection, so the gestures
            # (and the draft they edit) are the table's own.
            baseline = set(_rule_pairs(selection, count)) if rule else set()
            derived = tuple(pair for pair in wanted if pair in baseline)
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
            # A parked row naming a pair the list publishes is not said: the
            # selected row wins.
            shown = set(published or ())
            parked = tuple(row for row, pair in zip(rows, self._row_pairs)
                           if not row["selected"] and pair not in shown) if selection == "custom" else ()
            show_note(self.note, "" if not absent else sentence("kept, not in this source: " + ", ".join(
                pair_label(start, end) for start, end, _selected in absent)))
            show_note(self.parked, "" if not parked else sentence("kept, not selected: " + ", ".join(
                pair_label(row["start"] or "(no start)", row["end"] or "(no end)") for row in parked)))

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
        "never checked and never published. That list is kept while a rule or none is "
        "picked. Endpoints use text "
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
