"""Derive's variables, edited as the cards they are, each line with its typed answer.

This leaf edits plain draft values.  Its answers come from
``typed_variables`` -- the evaluation's own plans applied to the schemas the
Runtime already published -- so it neither evaluates variables nor asks a
device or Runtime to materialize data on the Qt thread.

A variable is a card that reads as a vertical formula: its name, "=" and
first operand on the head line, with Publish and the card's ×; one line per
later term, the term's operator strip standing right before its operand and
ending where the name field ends; then its statistic.  Every line of every
card is a line of ONE column set -- the operand under the operand, ``at``
under ``at``, the labels under the labels, Publish and every × on one x
each -- and the operand column is the page's control column: the bundle
combo's text, the Inputs' names, every operand, result and problem read
down one x.  Fixed sets (an operator kind, the statistics) are segmented
strips; whatever comes from the data (members, variables, axes and
domains, labels, ``over``) is a combo, several values a checkable one --
a pick's values led by All (no pick), First and Last (its ends, whatever
its length).  A
problem stands under the line it is about, as one sentence
(``show_problem``); a parked helper's in muted ink, because it blocks
nothing.  A line's two wide commands -- the statistic strip with its
``over``, the add commands -- run on to a second row when the page is too
narrow for them, rather than squeeze.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Mapping
from html import escape

from PyQt5 import QtCore, QtGui, QtWidgets, sip

from zlc_runtime import stable_signal_key
from zlc_ui.fluent import (
    AXIS_GROUP_COLORS, FluentButton, FluentColumnLine, FluentColumnSet,
    FluentComboBox, FluentFrame, FluentLabel, FluentLineEdit, FluentSegmentedButtons, FluentSwitch, card_pad,
    error_note_label, field_text_inset, hold_place, muted_note_label, retire_widget, row_gap,
    scaled_px, section_gap, set_status_ink, show_problem, signals_blocked, status_cue,
)
from zlc_ui.fluent.column_set import section_header
from zlc_workbench.topology import signal_text

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.frames import FIRST, LAST

from .expression import (
    CATEGORIES, DOMAINS, NO_PUBLISHED, OPERATORS, STATISTICS, TERM_DEFAULTS, VARIABLE_COLUMNS, axis_groups,
    changing_axes, compact_variable, describe_schema, error_stage, frames_kept, pick_values, term_text,
    typed_variables, where_notes,
)

#: The rows as a draft holds them, completed column by column; no validator,
#: because a problem is shown under its line, not refused.
_ROWS = AuthoringSchema((AuthoringField("variables", "rows", "Variables", (), columns=VARIABLE_COLUMNS),))
_STRIPS = dict(CATEGORIES)
_FOOTER = "Averages over shots with error bars, per-site probabilities and fits belong in a plot panel."
_EMPTY = ("No variables yet. A variable picks values of an input, combines them line by line, can mask and "
          "reduce them; Publish makes it a signal.")
#: The operand combo's constants.  A member or variable name never starts
#: with "#", so no name reads like these keys.
_NUMBER, _TRUE, _FALSE = "#number", "#true", "#false"
#: "No pick": the axis combo's first row, and a pick's first value.
_ALL = ""
#: What a pick's values list starts with: All, then the axis' ends, written
#: in the draft as Occupancy's frame claims write a cycle's.
_ENDS = (("All", _ALL), ("First", FIRST), ("Last", LAST))
_CONSTANTS = {_NUMBER: "Number", _TRUE: "true", _FALSE: "false"}
_COLOURS = dict(zip(DOMAINS, AXIS_GROUP_COLORS))
#: The most labels one labels list offers; a longer axis says how many it has.
_LABEL_LIMIT = 2048
#: The least room a variable's name gets (design px); the lead column is as
#: wide as the widest operator strip when that is wider.
_NAME_WIDTH = 150
#: The operand column (design px), pinned: a name typed on one card never
#: moves the columns of every card.  A longer name elides; its tooltip
#: says it whole.
_OPERAND_WIDTH = 140
#: The columns of every card line: the name (or a later line's operator
#: strip), the "=" gutter, the operand, ``at`` and its axis (or a number),
#: the labels (or a number's unit), the rest of the line, Publish and ×.
_COLUMNS = (("lead", "fit"), ("gutter", "fit"), ("operand", "fit"), ("at", "fit"), ("labels", "fit"),
            ("tail", "stretch"), ("publish", "fit"), ("remove", "glyph"))
_ADD = (("arithmetic", "+ Arithmetic"), ("comparison", "+ Comparison"), ("logic", "+ Logic"),
        ("where", "+ Where"), ("statistic", "+ Statistic"))
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _set_text(widget, text: str) -> None:
    if widget.text() != text:
        widget.setText(text)


def _set_tip(widget, text: str) -> None:
    if widget.toolTip() != text:
        widget.setToolTip(text)


def _visible(widget: QtWidgets.QWidget, shown: bool, hold: bool = True) -> None:
    """Show or hide ``widget``; hidden with ``hold`` it keeps its place in
    its line, so its column keeps its width whichever lines use it."""

    policy = widget.sizePolicy()
    if policy.retainSizeWhenHidden() != hold:
        policy.setRetainSizeWhenHidden(hold)
        widget.setSizePolicy(policy)
    if widget.isHidden() == shown:
        widget.setVisible(shown)


def _alive(widget: QtWidgets.QWidget) -> bool:
    return not sip.isdeleted(widget) and widget.isVisible()


def _boolean(schema) -> bool:
    return schema.value_schema.dtype.kind == "b"


def _flavour(schema) -> str | None:
    """"b" for booleans, "n" for numbers, None while not known."""
    return None if schema is None else "b" if _boolean(schema) else "n"


def _item(text: str, data) -> tuple:
    """A row of a list: ``(text, key, bold, colour, enabled)``."""
    return (text, data, False, "", True)


def _header(text: str, colour: str = "", data=None, enabled: bool = False) -> tuple:
    """A list's bold group title; a choice of its own when ``enabled``."""
    return (text, data, True, colour, enabled)


def _fill(combo: FluentComboBox, items, current=None) -> None:
    """Offer ``items`` (:func:`_item`, :func:`_header`), refilled only when
    they change -- a cached list is the same object, so a long one is not
    compared row by row -- and mark ``current`` (nothing when None) on a
    one-value combo."""

    items = tuple(items)
    offered = getattr(combo, "_derive_offered", None)
    if offered is not items and offered != items:
        with signals_blocked(combo):
            combo.clear()
            for text, data, bold, colour, enabled in items:
                combo.addItem(text, data)
                if not (bold or colour or not enabled):
                    continue
                item = combo.model().item(combo.count() - 1)
                item.setEnabled(enabled)
                if bold:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                if colour:
                    item.setForeground(QtGui.QBrush(QtGui.QColor(colour)))
        combo._derive_offered = items
    if current is not None or combo.currentIndex() >= 0:
        index = -1 if current is None else next(
            (row for row in range(combo.count())
             if combo.itemData(row) == current and combo.model().item(row).isEnabled()), -1)
        if index != combo.currentIndex():
            with signals_blocked(combo):
                combo.setCurrentIndex(index)


@functools.lru_cache(maxsize=64)
def _label_items(at: str, keys: tuple | None, alternates: tuple) -> tuple:
    """A pick's choices as list rows: All, First and Last, then under the
    axis' (domain's) name -- which keeps the collapsed text from reading a
    run across the ends -- its values, each with its alternate
    coordinates' (``1 · time 0.1 s``), at most ``_LABEL_LIMIT`` of them;
    none while nothing says them.  Kept: a long axis is offered again on
    every projection."""

    keys = keys or ()
    items = tuple(
        _item(key + "".join(f" · {name} {labels[index]}" + (f" {unit}" if unit else "")
                            for name, unit, labels in alternates), key)
        for index, key in enumerate(keys[:_LABEL_LIMIT])
    )
    if len(keys) > _LABEL_LIMIT:
        items += ((f"{_LABEL_LIMIT} of {len(keys)} shown", None, False, status_cue("idle")[1], False),)
    return (*(_item(text, key) for text, key in _ENDS), _header(at), *items)


def _alternates(schema, at: str) -> tuple:
    """``(name, unit, labels)`` of each alternate coordinate of the axis ``at``."""
    for _word, domain, axes in axis_groups(schema):
        for axis in axes:
            if axis.name == at:
                return tuple((other.name, other.unit or "", other.labels()) for other in domain.axes
                             if other.coordinate_of == axis.axis_id)
    return ()


def _remove_button(parent, tooltip: str) -> FluentButton:
    """A line's own ×: the quiet glyph in the destructive role."""

    button = FluentButton("×", parent, role="destructive")
    button.setToolTip(tooltip)
    return button


def _spacer() -> QtWidgets.QWidget:
    """The room before a word or strip that ends on the name field's edge."""

    spacer = QtWidgets.QWidget()
    spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    return spacer


def _note_gap() -> int:
    """How far a note stands under the line it belongs to: half a row gap,
    so a note reads as its line's and not as the next line's."""

    return row_gap() // 2


def _card_column(card: QtWidgets.QWidget) -> tuple[QtWidgets.QVBoxLayout, QtWidgets.QVBoxLayout]:
    """``(lines, notes)``: the layout a card's lines go in, a row gap apart,
    and the card's column that holds them and, under the last line, its
    notes.  A note under the last line stands a note gap under it, as a
    line's own problem does under its line, so the result and the problem
    that replaces it stand at one y and the card keeps its height."""

    notes = QtWidgets.QVBoxLayout(card)
    pad = card_pad()
    notes.setContentsMargins(pad, pad, pad, pad)
    notes.setSpacing(_note_gap())
    lines = QtWidgets.QVBoxLayout()
    lines.setContentsMargins(0, 0, 0, 0)
    lines.setSpacing(row_gap())
    notes.addLayout(lines)
    return lines, notes


def _flow_box(pieces) -> QtWidgets.QWidget:
    """A box that lays ``pieces`` left to right in as few rows as its width
    allows (:func:`_flow`), as wide as the cell it is put in."""

    box = QtWidgets.QWidget()
    box.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
    rows = QtWidgets.QVBoxLayout(box)
    rows.setContentsMargins(0, 0, 0, 0)
    rows.setSpacing(row_gap())
    box._flow_pieces, box._flow_rows = tuple(pieces), None
    for piece in box._flow_pieces:
        piece.setParent(box)
    _flow(box)
    return box


def _flow(box: QtWidgets.QWidget) -> None:
    """Lay the box's pieces in rows: each row takes pieces while the next
    one's natural width still fits beside them; a hidden piece takes no
    room, so it never opens a row of its own."""

    gap, width = row_gap(), box.width()
    rows, used = [[]], 0
    for piece in box._flow_pieces:
        shown = not piece.isHidden()
        need = piece.sizeHint().width() if shown else 0
        if shown and rows[-1] and used + gap + need > width:
            rows.append([])
            used = 0
        if shown:
            used += (gap if used else 0) + need
        rows[-1].append(piece)
    key = tuple(len(row) for row in rows)
    if box._flow_rows == key:
        return
    box._flow_rows = key
    layout = box.layout()
    while layout.count():
        layout.takeAt(0)
    for row in rows:
        line = QtWidgets.QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(gap)
        for piece in row:
            line.addWidget(piece)
        line.addStretch(1)
        layout.addLayout(line)
    box.updateGeometry()


def _operand_key(term: Mapping[str, object]):
    if term["number"] is not None:
        return _NUMBER
    if term["boolean"]:
        return f"#{term['boolean']}"
    return term["of"] or None


def _reveal(*widgets: QtWidgets.QWidget) -> None:
    """Scroll the page the ``widgets`` stand in until they show, the last
    one first of all.  A card just added has no place on the page until
    the page is laid out again, so that is done first."""

    parent = widgets[0].parentWidget()
    while parent is not None and not isinstance(parent, QtWidgets.QScrollArea):
        parent = parent.parentWidget()
    if parent is None:
        return
    # Each layout that grows asks its parent's again: one pass per level.
    for _level in range(4):
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.LayoutRequest)
    for widget in widgets:
        parent.ensureWidgetVisible(widget, 0, row_gap())


class _VariableCard(FluentFrame):
    """One variable: ``name = operand at axis labels``, its later terms, its statistic."""

    def __init__(self, editor: "DeriveEditor"):
        super().__init__(editor)
        self._editor = editor
        self.row: dict[str, object] = {}
        self._answer: Mapping[str, object] = {}
        self._context: Mapping[str, object] = {}
        #: The draft problem this card states under one of its lines, in
        #: the draft's words, when it blocks Start; "" otherwise.
        self.problem = ""
        #: The statistic line opened by "+ Statistic" and not filled yet.
        self._statistic_open = False
        self._lines, notes = _card_column(self)
        self.head = self._term_line(None)
        self._lines.addWidget(self.head["block"])
        #: One per term, in order: the head line, then a line per later term.
        self.terms_ui = [self.head]
        self.statistic = self._statistic_line()
        self._lines.addWidget(self.statistic["block"])
        self.preview = muted_note_label("")
        self.preview.setWordWrap(True)
        editor.columns.follow(self.preview, "operand", inset=field_text_inset())
        notes.addWidget(self.preview)
        notes.addSpacing(row_gap() - _note_gap())
        add = FluentColumnLine(editor.columns, self)
        self.add = {}
        for kind, text in _ADD:
            button = FluentButton(text, self)
            button.clicked.connect(lambda _checked=False, kind=kind: self._add(kind))
            self.add[kind] = button
        self._add_box = _flow_box(self.add.values())
        self._add_box.installEventFilter(self)
        # The add commands run on over the empty Publish and × columns.
        add.place("operand", self._add_box, to="remove", fill=True)
        notes.addWidget(add)

    # -- construction -------------------------------------------------------

    def _block(self) -> tuple[QtWidgets.QWidget, FluentColumnLine, QtWidgets.QLabel]:
        """A line, and the block that keeps its problem under it."""

        block = QtWidgets.QWidget(self)
        rows = QtWidgets.QVBoxLayout(block)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(_note_gap())
        line = FluentColumnLine(self._editor.columns, block)
        rows.addWidget(line)
        error = error_note_label()
        self._editor.columns.follow(error, "operand", inset=field_text_inset())
        rows.addWidget(error)
        return block, line, error

    def _term_line(self, kind: str | None) -> dict[str, object]:
        """The head line (``kind`` None) or a later term's line of one operator kind."""

        block, line, error = self._block()
        ui: dict[str, object] = {"kind": kind, "block": block, "error": error, "pending": None}
        if kind is None:
            name = FluentLineEdit("")
            name.setPlaceholderText("name")
            # A name is written when it is finished, not per keystroke: the
            # variables that read it follow it then, and a published one's
            # signal is renamed once.
            name.editingFinished.connect(lambda: self._rename(name))
            line.place("lead", name, fill=True)
            line.place("gutter", FluentLabel("="))
            ui["name"] = name
        else:
            # The strip stands right before its operand, ending where the
            # name field ends: the card reads down as one formula.
            line.place("lead", _spacer(), fill=True)
            strip = FluentSegmentedButtons(tuple((op, OPERATORS[op][2]) for op in _STRIPS[kind]), self)
            strip.activated.connect(lambda op, ui=ui: self._term_edit(ui, op=op))
            line.place("lead", strip)
            ui["strip"] = strip
        operand = FluentComboBox()
        operand.setPlaceholderText("choose")
        operand.activated.connect(lambda _index, ui=ui: self._operand_chosen(ui))
        line.place("operand", operand, fill=True)
        at = hold_place(FluentLabel("at"))
        axis = hold_place(FluentComboBox())
        axis.activated.connect(lambda _index, ui=ui: self._axis_chosen(ui))
        # A number has no axis to pick: its box takes the pick's place.
        number = FluentLineEdit("")
        number.setPlaceholderText("number")
        number.textEdited.connect(lambda text, ui=ui: self._number_edited(ui, text))
        number.editingFinished.connect(lambda ui=ui: self._number_finished(ui))
        line.place("at", at)
        line.place("at", axis, number, fill=True)
        labels = hold_place(FluentComboBox())
        labels.set_checkable(True)
        labels.setPlaceholderText("choose")
        labels.checked_edited.connect(lambda values, ui=ui: self._labels_edited(ui, tuple(values)))
        unit = muted_note_label("")
        line.place("labels", labels, fill=True)
        line.place("labels", unit)
        if kind is None:
            publish = FluentSwitch("Publish")
            publish.clicked.connect(lambda checked: self._set(publish=bool(checked)))
            line.place("publish", publish)
            ui["publish"] = publish
        remove = _remove_button(self, "Remove this variable" if kind is None else "Remove this line")
        remove.clicked.connect(lambda _checked=False, ui=ui: self._remove(ui))
        line.place("remove", remove)
        ui.update(operand=operand, at=at, axis=axis, number=number, labels=labels, unit=unit, remove=remove)
        return ui

    def _statistic_line(self) -> dict[str, object]:
        block, line, error = self._block()
        line.place("lead", _spacer(), fill=True)
        line.place("lead", FluentLabel("statistic"))
        strip = FluentSegmentedButtons(tuple((name, name) for name in STATISTICS), self)
        strip.activated.connect(self._statistic_chosen)
        reduce = QtWidgets.QWidget()
        pair = QtWidgets.QHBoxLayout(reduce)
        pair.setContentsMargins(0, 0, 0, 0)
        pair.setSpacing(row_gap())
        word = FluentLabel("over")
        over = FluentComboBox()
        over.set_checkable(True, headers_select=True)
        over.setPlaceholderText("choose")
        over.checked_edited.connect(lambda values: self._set(over=tuple(values)))
        pair.addWidget(word)
        pair.addWidget(over)
        box = _flow_box((strip, reduce))
        box.installEventFilter(self)
        # The strip and its "over" run on over the empty Publish column, and
        # "over" goes under the strip when the page is too narrow for both.
        line.place("operand", box, to="publish", fill=True)
        line.stand_on_first_row()
        remove = _remove_button(self, "Remove this statistic")
        remove.clicked.connect(self._close_statistic)
        line.place("remove", remove)
        return {"block": block, "error": error, "strip": strip, "reduce": reduce, "word": word, "over": over,
                "box": box, "remove": remove, "pending": None}

    def _retire(self, ui: Mapping[str, object]) -> None:
        self._lines.removeWidget(ui["block"])
        self._editor.columns.forget(ui["block"])
        retire_widget(ui["block"])

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        kind = event.type()
        if kind in (QtCore.QEvent.Resize, QtCore.QEvent.LayoutRequest) and hasattr(watched, "_flow_pieces"):
            _flow(watched)
        elif kind == QtCore.QEvent.Hide:
            # A cascaded labels list closed.  What it ticked, if anything,
            # was written as it closed (``_labels_edited``); one that closed
            # with nothing ticked leaves the pick as it was.
            for ui in self.terms_ui:
                if ui["pending"] is not None and watched is ui["labels"].view().window():
                    QtCore.QTimer.singleShot(0, lambda ui=ui: self._cancel_pending(ui))
        return super().eventFilter(watched, event)

    # -- edits -------------------------------------------------------------

    def _set(self, **changes) -> None:
        self.row.update(changes)
        self._editor.card_edited()

    def _rename(self, field: FluentLineEdit) -> None:
        if field.isModified():
            field.setModified(False)
            self._editor.rename(self, field.text().strip())

    def _term(self, ui) -> dict[str, object]:
        """The term ``ui`` edits; the head line of a variable with no term yet writes its first."""
        if not self.row["terms"]:
            self.row["terms"].append(dict(TERM_DEFAULTS))
        return self.row["terms"][self.terms_ui.index(ui)]

    def _term_edit(self, ui, **changes) -> None:
        if ui in self.terms_ui:
            self._term(ui).update(changes)
            self._editor.card_edited()

    def _operand_chosen(self, ui) -> None:
        term = self._term(ui)
        key = ui["operand"].currentData()
        if key == _NUMBER:
            if term["number"] is None:
                term.update(of="", number=0.0, boolean="", at="", labels=())
            self._editor.card_edited()
            QtCore.QTimer.singleShot(0, lambda: self._focus_number(ui))
            return
        if key in (_TRUE, _FALSE):
            term.update(of="", number=None, boolean=key[1:], at="", labels=())
        else:
            # The pick stays when the new operand has its axis (domain),
            # or while nothing says which axes it has.
            schema = self._context["schema_of"](key)
            keeps = schema is None or (term["at"] and pick_values(schema, term["at"]) is not None)
            term.update(of=key, number=None, boolean="")
            if not keeps:
                term.update(at="", labels=())
        self._editor.card_edited()

    def _focus_number(self, ui) -> None:
        if ui in self.terms_ui and _alive(ui["number"]):
            ui["number"].setFocus()
            ui["number"].selectAll()

    def _number_edited(self, ui, text: str) -> None:
        """The number as typed, once it reads as a finite one."""
        try:
            value = float(text)
        except ValueError:
            return
        if math.isfinite(value):
            self._term_edit(ui, number=value)

    def _number_finished(self, ui) -> None:
        """Text that never read as a number gives way to the stored one."""
        if ui not in self.terms_ui:
            return
        term = self._term(ui)
        try:
            readable = math.isfinite(float(ui["number"].text()))
        except ValueError:
            readable = False
        if not readable and term["number"] is not None:
            _set_text(ui["number"], term_text(term))

    def _axis_chosen(self, ui) -> None:
        """``All`` clears the pick.  An axis -- or a domain -- opens its
        labels to pick; the pick is written once they are picked, so the
        card, its readers and the page say nothing in between.  The axis
        the line already picks opens its labels again."""
        axis = ui["axis"].currentData() or _ALL
        if not axis:
            ui["pending"] = None
            self._term_edit(ui, at="", labels=())
            return
        if axis != self._term(ui)["at"]:
            ui["pending"] = axis
            self._editor.refresh()
        QtCore.QTimer.singleShot(0, lambda: self._cascade(ui))

    def _cascade(self, ui) -> None:
        labels = ui["labels"]
        if ui not in self.terms_ui or sip.isdeleted(labels):
            return
        if _alive(labels) and labels.count():
            labels.showPopup()
            labels.view().window().installEventFilter(self)
        elif ui["pending"] is not None:
            # Nothing to pick from: the axis is written, its problem says so.
            at, ui["pending"] = ui["pending"], None
            self._term_edit(ui, at=at, labels=())

    def _labels_edited(self, ui, values: tuple) -> None:
        """The values ticked; All among them is no pick, as the axis
        combo's All is."""
        if ui not in self.terms_ui:
            return
        if _ALL in values:
            ui["pending"] = None
            self._term_edit(ui, at="", labels=())
        elif ui["pending"] is not None:
            at, ui["pending"] = ui["pending"], None
            self._term_edit(ui, at=at, labels=values)
        else:
            self._term_edit(ui, labels=values)

    def _cancel_pending(self, ui) -> None:
        if ui in self.terms_ui and ui["pending"] is not None:
            ui["pending"] = None
            self._editor.refresh()

    def _statistic_chosen(self, statistic: str) -> None:
        """The statistic; with nothing to reduce over yet, every Cell axis
        of the value -- or, with none, the list of its axes to pick."""
        self.row["statistic"] = statistic
        reducible = self._answer.get("reducible")
        cascade = False
        if not self.row["over"] and reducible is not None:
            groups = axis_groups(reducible)
            cells = tuple(axis.name for word, _domain, axes in groups if word == "Cell" for axis in axes)
            if cells:
                self.row["over"] = cells
            else:
                cascade = bool(groups)
        self._editor.card_edited()
        if cascade:
            QtCore.QTimer.singleShot(0, self._open_over)

    def _open_over(self) -> None:
        over = self.statistic["over"]
        if _alive(over) and over.count():
            over.showPopup()

    def _close_statistic(self) -> None:
        self._statistic_open = False
        if self.row["statistic"] or self.row["over"]:
            self._set(statistic="", over=())
        else:
            self._editor.refresh()

    def _remove(self, ui) -> None:
        if ui is self.head:
            self._editor.remove_card(self)
        elif ui in self.terms_ui:
            del self.row["terms"][self.terms_ui.index(ui)]
            self._editor.card_edited()

    def _add(self, kind: str) -> None:
        """A line of ``kind`` after the last, written the way most formulas
        want it; a pick it starts asks for its labels at once."""
        if kind == "statistic":
            self._statistic_open = True
            self._editor.refresh()
            return
        if not self.row["terms"]:
            self.row["terms"].append(dict(TERM_DEFAULTS))
        head = self.row["terms"][0]
        schema_of = self._context["schema_of"]
        term = dict(TERM_DEFAULTS, op=_STRIPS[kind][0])
        # What the new line stands beside: the line above it, at the
        # tightest binding a comparison or an arithmetic step has.
        last = self.row["terms"][-1]
        infos = self._answer.get("terms") or ()
        beside = ("n" if last["number"] is not None else "b" if last["boolean"]
                  else _flavour(infos[-1]["operand"]) if len(infos) == len(self.row["terms"]) else None)
        if kind == "arithmetic" or (kind == "comparison" and beside != "b"):
            pool = [name for name in self._context["names"] if _flavour(schema_of(name)) != "b"]
            of = head["of"] if head["of"] in pool else pool[0] if pool else ""
            if of:
                term["of"] = of
            else:
                term["number"] = 0.0
        else:
            # A verdict: a boolean variable this one may read, else a member.
            pool = [name for name in (*self._context["variables"], *self._context["members"])
                    if _flavour(schema_of(name)) == "b"]
            term["of"] = head["of"] if kind == "comparison" and head["of"] in pool else pool[0] if pool else ""
        schema = schema_of(term["of"]) if term["of"] else None
        pending = None
        if schema is not None and head["at"] and pick_values(schema, head["at"]) is not None:
            if kind != "where":
                pending = head["at"]
            elif all(label in (FIRST, LAST) or label in pick_values(schema, head["at"])
                     for label in head["labels"]):
                # The mask at the value's own pick.
                term.update(at=head["at"], labels=tuple(head["labels"]))
        self.row["terms"].append(term)
        self._editor.card_edited()
        if pending is not None and len(self.terms_ui) == len(self.row["terms"]):
            ui = self.terms_ui[-1]
            ui["pending"] = pending
            self._editor.refresh()
            QtCore.QTimer.singleShot(0, lambda: self._cascade(ui))

    # -- projection --------------------------------------------------------

    def _shape(self, terms) -> None:
        """One line per term, each of its operator's kind."""
        for k in range(1, len(terms)):
            kind = OPERATORS.get(terms[k]["op"], ("arithmetic",))[0]
            if k < len(self.terms_ui) and self.terms_ui[k]["kind"] == kind:
                continue
            ui = self._term_line(kind)
            if k < len(self.terms_ui):
                self._retire(self.terms_ui[k])
                self.terms_ui[k] = ui
            else:
                self.terms_ui.append(ui)
            self._lines.insertWidget(k, ui["block"])
        while len(self.terms_ui) > max(1, len(terms)):
            self._retire(self.terms_ui.pop())

    def show_row(self, row: Mapping[str, object], answer: Mapping[str, object],
                 context: Mapping[str, object]) -> None:
        self.row = {**row, "terms": [dict(term) for term in row["terms"]]}
        self._answer, self._context = answer, context
        terms = self.row["terms"] or [dict(TERM_DEFAULTS)]
        infos = answer["terms"] or ({"operand": None, "left": None, "number_unit": None},)
        self._shape(terms)
        name = self.head["name"]
        if not name.isModified():
            _set_text(name, str(row["name"]))
        publish = self.head["publish"]
        with signals_blocked(publish):
            publish.setChecked(bool(row["publish"]))
        _set_tip(publish, f"Published as {context['signal_text'](row['name'])}" if row["publish"]
                 else "Internal: other variables read it; nothing publishes it")
        for ui, term, info in zip(self.terms_ui, terms, infos):
            self._show_term(ui, term, info)
        self._show_statistic(answer.get("reducible"))
        for box in (self._add_box, self.statistic["box"]):
            _flow(box)
        if any(ui["pending"] is not None for ui in self.terms_ui):
            # A labels list is open on a pick not written yet: what the
            # card says under its lines stays as it was until it closes.
            return

        error, failure = answer["error"], context["failure"]
        if error is not None:
            stage, detail = error.stage, error.detail
            muted = not answer["live"] and stage != "name"
        elif failure is not None:
            (stage, detail), muted = failure, False
        else:
            stage, detail, muted = None, "", False
        said = self._line_of(stage)
        for ui in (*self.terms_ui, self.statistic):
            # Red when it blocks; a parked helper's in the idle ink of a note.
            set_status_ink(ui["error"], "idle" if muted else "error")
            show_problem(ui["error"], detail if ui is said else "")
        self.problem = str(error) if error is not None and not muted else ""
        # A parked helper's problem blocks nothing; its result says why.
        self.preview.setVisible(stage is None or muted)
        if stage is None:
            _set_text(self.preview, self._preview(answer, context["typed"]))
        elif muted:
            _set_text(self.preview, "→ internal · read by nothing, so not evaluated")

    def _line_of(self, stage: str | None):
        """The line a problem at ``stage`` stands under."""
        if stage is None:
            return None
        if stage == "statistic" and not self.statistic["block"].isHidden():
            return self.statistic
        if stage.startswith("term "):
            k = int(stage.split()[1]) - 1
            if k < len(self.terms_ui):
                return self.terms_ui[k]
        return self.head

    def _show_term(self, ui, term, info) -> None:
        context = self._context
        schema_of = context["schema_of"]
        kind = ui["kind"]
        if kind is not None:
            # Every comparison takes two booleans (false < true); the
            # operand list offers a boolean only booleans to compare with.
            ui["strip"].set_current(term["op"] or None)
        operand = ui["operand"]
        _fill(operand, self._operands(kind, term, info), _operand_key(term))
        _set_tip(operand, term["of"])

        number = term["number"] is not None
        at = ui["pending"] or term["at"]
        schema = schema_of(term["of"]) if term["of"] else None
        groups = axis_groups(schema) if schema is not None else ()
        # Before the source publishes, the readout-event axis its draft
        # declares -- by the name it gives it -- is the one pick offered,
        # with the labels this operand keeps of it.
        frame_axis = context["frame_axis"]
        frames = context["frames_of"](term["of"]) if term["of"] and schema is None else None
        picks = bool(term["of"]) and (bool(groups) or bool(frames) or bool(at))
        _visible(ui["at"], picks, hold=not number)
        _visible(ui["axis"], picks, hold=not number)
        _visible(ui["number"], number, hold=False)
        _visible(ui["labels"], picks and bool(at), hold=not number)
        _visible(ui["unit"], number, hold=False)
        if picks:
            changing = context["changing"]
            items = [_item("All", _ALL)]
            for word, _domain, axes in groups:
                # An axis each publication carries its own part of is never
                # a pick; a domain is one when every axis of it may be.
                free = [axis for axis in axes if axis.axis_id not in changing]
                if not free:
                    continue
                whole = len(axes) > 1 and len(free) == len(axes)
                items.append(_header(word, _COLOURS[word], word if whole else None, whole))
                items.extend(_item(f"    {axis.name}", axis.name) for axis in free)
            if frames:
                items.extend((_header(DOMAINS[1], _COLOURS[DOMAINS[1]]),
                              _item(f"    {frame_axis}", frame_axis)))
            if at and not any(data == at and enabled for _t, data, _b, _c, enabled in items):
                items.append(_item(at, at))
            _fill(ui["axis"], items, at)
        if picks and at:
            if schema is not None:
                keys, alternates = pick_values(schema, at), _alternates(schema, at)
            else:
                keys, alternates = (frames if frames and at == frame_axis else None), ()
            offered = _label_items(at, keys, alternates)
            shown = () if ui["pending"] is not None else tuple(term["labels"])
            known = {*(keys or ()), FIRST, LAST}
            extra = tuple(_item(label, label) for label in shown if label not in known)
            _fill(ui["labels"], offered + extra if extra else offered)
            ui["labels"].set_checked_data(shown)
        if number:
            if not ui["number"].hasFocus():
                _set_text(ui["number"], term_text(term))
            _set_text(ui["unit"], info["number_unit"] or "")

    def _operands(self, kind, term, info) -> list[tuple]:
        """What the line's operand may be: the inputs, the variables this one
        may read (never itself or a reader of it, so no cycle can be made),
        and the constants its operator takes -- once typed, only those of
        the type the operator wants."""
        context = self._context
        schema_of = context["schema_of"]
        if kind is None:
            want, constants = None, (_NUMBER,)
        elif kind == "arithmetic":
            want, constants = "n", (_NUMBER,)
        elif kind == "comparison":
            left = info["left"]
            want, constants = ((None, (_NUMBER, _TRUE, _FALSE)) if left is None
                               else ("b", (_TRUE, _FALSE)) if _boolean(left) else ("n", (_NUMBER,)))
        else:
            want, constants = "b", ()
        if not context["typed"]:
            want = None

        def offered(name):
            return want is None or _flavour(schema_of(name)) in (want, None)

        inputs = [name for name in context["members"] if offered(name)]
        variables = [name for name in context["variables"] if offered(name)]
        current = _operand_key(term)
        if current is not None and current.startswith("#"):
            constants = (*constants, *((current,) if current not in constants else ()))
        elif current is not None and current not in inputs and current not in variables:
            (inputs if current in context["members"] else variables).append(current)
        items = []
        for title, keys in (("Inputs", inputs), ("Variables", variables), ("Constants", constants)):
            if keys:
                items.append(_header(title))
                items.extend(_item(f"    {_CONSTANTS.get(key, key)}", key) for key in keys)
        return items

    def _show_statistic(self, reducible) -> None:
        row, line = self.row, self.statistic
        shown = self._statistic_open or bool(row["statistic"]) or bool(row["over"])
        if row["statistic"] or row["over"]:
            self._statistic_open = False
        line["block"].setVisible(shown)
        # The last command: nothing after it moves when it goes.
        _visible(self.add["statistic"], not shown, hold=False)
        strip = line["strip"]
        strip.set_current(row["statistic"] or None)
        boolean = reducible is not None and _boolean(reducible)
        for name in STATISTICS:
            withheld = reducible is not None and (
                name in ("min", "max", "std") if boolean else name in ("any", "all"))
            strip.button(name).setEnabled(not withheld or name == row["statistic"])
        items = []
        for word, _domain, axes in (() if reducible is None else axis_groups(reducible)):
            items.append(_header(word, _COLOURS[word], None, True))
            items.extend(_item(axis.name, axis.name) for axis in axes)
        known = {data for _text, data, _bold, _colour, _enabled in items}
        items.extend(_item(name, name) for name in row["over"] if name not in known)
        _visible(line["reduce"], bool(items), hold=False)
        _fill(line["over"], items)
        line["over"].set_checked_data(row["over"])

    def _preview(self, answer, typed: bool) -> str:
        if answer["waits_for"]:
            return f"→ waits for {answer['waits_for']}"
        schema = answer["schema"]
        if schema is None or not typed:
            return "→ checked when the source publishes"
        row = self.row
        notes = [describe_schema(schema)]
        if answer["grouping"]:
            notes.append(answer["grouping"])
        reducible = answer["reducible"]
        if row["statistic"] and reducible is not None and _boolean(reducible):
            notes.extend(note for note in ({"sum": "number true", "mean": "fraction true",
                                            "count": "valid samples"}.get(row["statistic"]),) if note)
        elif row["statistic"] == "count":
            notes.append("valid samples")
        notes.extend(where_notes(row["terms"]))
        if not row["publish"]:
            notes.append(f"internal · read by {', '.join(answer['read_by'])}" if answer["live"]
                         else "internal · read by nothing, so not evaluated")
        return "→ " + " · ".join(notes)

    def focus_chain(self) -> list[QtWidgets.QWidget]:
        """The card's controls in the order they read: line by line, each
        line left to right, then the add commands."""
        chain: list[QtWidgets.QWidget] = []
        for ui in self.terms_ui:
            lead = [ui["name"]] if ui["kind"] is None else ui["strip"].findChildren(QtWidgets.QAbstractButton)
            chain.extend((*lead, ui["operand"], ui["axis"], ui["number"], ui["labels"]))
            if ui["kind"] is None:
                chain.append(ui["publish"])
            chain.append(ui["remove"])
        line = self.statistic
        chain.extend((*line["strip"].findChildren(QtWidgets.QAbstractButton), line["over"], line["remove"]))
        chain.extend(self.add.values())
        return chain


class DeriveEditor(QtWidgets.QWidget):
    """The concrete Derive contribution to the existing Logic editor."""

    draft_changed = QtCore.pyqtSignal(object)
    managed_fields = ("variables",)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._loading = False
        self._enabled = True
        self._cards: list[_VariableCard] = []
        self._members: tuple[str, ...] = ()
        #: Each member's typing schema, None until it publishes.
        self._inputs: dict[str, object] = {}
        #: The readout-event labels the source's draft says every member
        #: carries (``source_frames``) and that axis' name, as the source
        #: names it (``source_frame_axis``); None while nothing says, and
        #: for one frame, which may as well be no axis: there is no pick to
        #: offer.
        self._frames: tuple[str, ...] | None = None
        self._frame_axis: str | None = None
        #: The axes no pick may name (``changing_axes``), over every member.
        self._changing: dict[object, tuple[str, ...]] = {}
        self._node = ""
        #: What the operator calls this node: a published variable's
        #: signal is read as ``<name>/<variable>``.
        self._node_name = ""
        self._failure: tuple[str, str, str] | None = None
        self._draft_problem = ""
        #: The columns every line of every card shares.  The "=", the lead
        #: and the operand are pinned, so where the operands stand is known
        #: before the first card is built (the label column above is fitted
        #: to it), and a name typed anywhere moves no column.
        self.columns = FluentColumnSet(_COLUMNS, self)
        self._gutter = FluentLabel("=").sizeHint().width()
        self.columns.set_width("gutter", self._gutter)
        self.columns.set_width("operand", scaled_px(_OPERAND_WIDTH))
        widths = [scaled_px(_NAME_WIDTH)]
        for _kind, operators in CATEGORIES:
            probe = FluentSegmentedButtons(tuple((op, OPERATORS[op][2]) for op in operators))
            widths.append(probe.sizeHint().width())
            retire_widget(probe)
        self._lead = max(widths)
        self.columns.set_width("lead", self._lead)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(row_gap())

        # A row of the editor's form: "Inputs" in its label column, the
        # members' names where the bundle combo's text starts above them.
        inputs = QtWidgets.QHBoxLayout()
        inputs.setSpacing(row_gap())
        self.inputs_title = FluentLabel("Inputs")
        inputs.addWidget(self.inputs_title, 0, QtCore.Qt.AlignTop)
        self.inputs = QtWidgets.QGridLayout()
        self.inputs.setContentsMargins(field_text_inset(), 0, 0, 0)
        # Words in columns, not bordered cells: the columns stand a section
        # gap apart, or a member and its type read as one phrase.
        self.inputs.setHorizontalSpacing(section_gap())
        self.inputs.setVerticalSpacing(row_gap())
        inputs.addLayout(self.inputs)
        inputs.addStretch(1)
        layout.addLayout(inputs)
        self._input_rows: list[tuple[QtWidgets.QLabel, ...]] = []

        self.add_variable = FluentButton("Add variable", self)
        self.add_variable.clicked.connect(self._add)
        self.variables_title, header = section_header("Variables", actions=(self.add_variable,))
        # A new section under the Inputs row: the section gap above its title.
        layout.addSpacing(section_gap() - row_gap())
        layout.addLayout(header)
        self.note = error_note_label("")
        layout.addWidget(self.note)
        self.empty = muted_note_label(_EMPTY)
        self.empty.setWordWrap(True)
        layout.addWidget(self.empty)
        self.cards = QtWidgets.QVBoxLayout()
        self.cards.setSpacing(row_gap())
        layout.addLayout(self.cards)
        self.footer = muted_note_label(_FOOTER)
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)

    def row_labels(self) -> tuple[str, ...]:
        """The labels this editor puts in the editor's label column."""

        return (self.inputs_title.text(),)

    def label_width_floor(self) -> int:
        """The label column that leaves a card's lead -- a name, or the
        widest operator strip -- its room: the cards' operands stand on the
        column's controls (:meth:`set_label_width`), so the column is at
        least as wide as a card line's lead, "=" and their gaps reach."""

        return self._label_width_for(self._lead)

    def set_label_width(self, width: int) -> None:
        """The editor's label column, shared with the bundle row above.

        The cards' lead column takes what the label column leaves before
        its controls, so the bundle's text, the Inputs' names and every
        operand, result and problem under them read down one x."""

        self.inputs_title.setFixedWidth(int(width))
        self.columns.set_width("lead", int(width) - self._label_width_for(0))

    def _label_width_for(self, lead: int) -> int:
        """The label column whose controls stand where the operands do when
        the lead column is ``lead`` wide: a card line runs from the card's
        pad through the lead, the "=" and their gaps to its operand, and a
        control stands a row gap past the label column."""

        return card_pad() + lead + 2 * self.columns.gap + self._gutter - row_gap()

    def shown_problems(self) -> dict[str, str]:
        """The draft's blocking problems this editor states under the lines
        they are about, and the one under its header, each with its
        section: the page header counts them rather than saying one again."""

        section = self.variables_title.text()
        shown = {card.problem: section for card in self._cards if card.problem}
        if self._draft_problem:
            shown[self._draft_problem] = section
        return shown

    def set_mutation_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self.add_variable.setEnabled(self._enabled)
        for card in self._cards:
            card.setEnabled(self._enabled)

    # -- edits -------------------------------------------------------------

    def card_edited(self) -> None:
        """A card wrote its row: the draft is every card's row, written
        compactly, and the cards show it at once."""

        if not self._loading:
            self._commit(tuple(compact_variable(card.row) for card in self._cards))

    def _commit(self, variables) -> None:
        self.draft_changed.emit({"values": {"variables": variables}})
        self._show(variables)

    def refresh(self) -> None:
        """Show the cards' rows again, the draft unchanged."""

        if not self._loading:
            self._show(tuple(compact_variable(card.row) for card in self._cards))

    def rename(self, card: _VariableCard, name: str) -> None:
        """``card``'s variable named ``name``.  What read it by its old
        name reads it by the new one, when the new one is free -- a name
        no other variable or input has -- and the card was the variable
        that name meant (the first so named)."""

        old = str(card.row["name"])
        if name == old:
            return
        others = {str(other.row["name"]) for other in self._cards if other is not card}
        first = next((other for other in self._cards if other.row["name"] == old), None) is card
        follows = (bool(old) and first and bool(_NAME.fullmatch(name))
                   and name not in others and name not in self._members)
        card.row["name"] = name
        if follows:
            for other in self._cards:
                for term in other.row["terms"]:
                    if term["of"] == old:
                        term["of"] = name
        self.card_edited()

    def remove_card(self, card: _VariableCard) -> None:
        self._cards.remove(card)
        self._retire(card)
        self.card_edited()

    def _retire(self, card: _VariableCard) -> None:
        self.cards.removeWidget(card)
        self.columns.forget(card)
        retire_widget(card)

    def _add(self) -> None:
        """A published variable reading the first input, its name selected to type over."""
        taken = {card.row["name"] for card in self._cards}
        ordinal = 1
        while f"variable_{ordinal}" in taken:
            ordinal += 1
        term = {"of": self._members[0]} if self._members else {}
        self._commit((*(compact_variable(card.row) for card in self._cards),
                      {"name": f"variable_{ordinal}", "terms": (term,)}))
        card = self._cards[-1]
        QtCore.QTimer.singleShot(0, lambda: self._focus_name(card))

    @staticmethod
    def _focus_name(card: _VariableCard) -> None:
        name = card.head["name"]
        if _alive(name):
            # The whole card into view, its name at least: on a long page
            # the new card stands below what shows.
            _reveal(card, name)
            name.setFocus()
            name.selectAll()

    # -- projection --------------------------------------------------------

    def _show_inputs(self, bundle) -> None:
        while len(self._input_rows) > len(bundle):
            for label in self._input_rows.pop():
                retire_widget(label)
        for index, (name, schema, _snapshot) in enumerate(bundle):
            if index == len(self._input_rows):
                labels = (FluentLabel(""), muted_note_label(""), FluentLabel(""))
                labels[2].setTextFormat(QtCore.Qt.RichText)
                for column, label in enumerate(labels):
                    self.inputs.addWidget(label, index, column)
                self._input_rows.append(labels)
            member, kind, axes = self._input_rows[index]
            _set_text(member, str(name))
            if schema is None:
                _set_text(kind, "waiting for its first publication")
                _set_text(axes, "")
                tooltip = ""
            else:
                value = schema.value_schema
                _set_text(kind, f"{value.dtype} · unit {value.value_unit or '1'}")
                groups, lines = [], []
                for word, _domain, domain_axes in axis_groups(schema):
                    colour = _COLOURS[word]
                    for axis in domain_axes:
                        groups.append(f'<span style="color: {colour}">{escape(axis.name)} {axis.size}</span>')
                        labels = axis.labels()
                        shown = ", ".join(labels) if len(labels) <= 8 else f"{labels[0]} … {labels[-1]}"
                        moving = " · changes with each publication" if axis.axis_id in self._changing else ""
                        lines.append(f"{axis.name} — {axis.axis_id.value} · {word} · {axis.role.value}"
                                     f" · {shown}{moving}")
                _set_text(axes, " × ".join(groups) or "one value")
                tooltip = "\n".join(lines)
            for label in (member, kind, axes):
                _set_tip(label, tooltip)

    def update_projection(self, projection: Mapping[str, object]) -> None:
        values = projection.get("form_values") or {}
        bundle = tuple(projection.get("input_bundle") or ())
        self._node = str(projection.get("node_id") or "")
        self._node_name = str(projection.get("name") or self._node)
        self._members = tuple(str(name) for name, _schema, _snapshot in bundle)
        self._inputs = {str(name): schema for name, schema, _snapshot in bundle}
        frames = projection.get("source_frames")
        self._frames = None if frames is None or len(frames) < 2 else tuple(frames)
        self._frame_axis = None if self._frames is None else projection.get("source_frame_axis")
        self._changing = {}
        for _name, schema, snapshot in bundle:
            if schema is not None and snapshot is not None:
                self._changing.update(changing_axes(schema, snapshot.block.schema))
        self._show_inputs(bundle)
        self._failure = error_stage(str(projection.get("error") or ""))
        self._show(values.get("variables") or ())

    def _show(self, variables) -> None:
        """Type the variables and show each on its card, in card order."""
        try:
            rows = _ROWS.draft_values({"variables": variables})["variables"]
            problem = ""
        except (TypeError, ValueError) as error:
            rows, problem = (), str(error)
        if not problem and rows and not any(row["publish"] for row in rows):
            problem = NO_PUBLISHED
        self._draft_problem = problem
        show_problem(self.note, problem)
        typed = bool(self._inputs) and all(schema is not None for schema in self._inputs.values())
        answers = typed_variables(rows, self._inputs, self._changing)
        members, schemas = self._members, self._inputs
        results: dict[str, object] = {}
        reads: dict[str, set[str]] = {}
        for row, answer in zip(rows, answers):
            results.setdefault(row["name"], answer["schema"])
            reads.setdefault(row["name"], set()).update(term["of"] for term in row["terms"] if term["of"])

        def schema_of(name):
            return schemas.get(name) if name in members else results.get(name)

        # Untyped, each name's frames as the source's draft says them.
        drafted = {}
        if not typed and self._frames is not None:
            for row, frames in zip(rows, frames_kept(rows, self._frame_axis, self._frames, members)):
                drafted.setdefault(row["name"], frames)

        def frames_of(name):
            return self._frames if name in members else drafted.get(name)

        def readers(name):
            found, frontier = set(), [name]
            while frontier:
                current = frontier.pop()
                for other, read in reads.items():
                    if current in read and other not in found:
                        found.add(other)
                        frontier.append(other)
            return found

        self._loading = True
        try:
            while len(self._cards) > len(rows):
                self._retire(self._cards.pop())
            while len(self._cards) < len(rows):
                card = _VariableCard(self)
                card.setEnabled(self._enabled)
                self._cards.append(card)
                self.cards.addWidget(card)
            for card, row, answer in zip(self._cards, rows, answers):
                closed = readers(row["name"]) | {row["name"]}
                variables = tuple(dict.fromkeys(
                    other["name"] for other in rows
                    if other["name"] and other["name"] not in closed and other["name"] not in members))
                failure = self._failure
                card.show_row(row, answer, {
                    "members": members, "variables": variables, "names": (*members, *variables),
                    "schema_of": schema_of, "frames_of": frames_of, "frame_axis": self._frame_axis,
                    "typed": typed, "signal_text": self._published_text,
                    "changing": self._changing,
                    "failure": failure[1:] if failure is not None and failure[0] == row["name"] else None,
                })
            self.columns.refresh()
            self.empty.setVisible(not rows)
            self._order_focus()
        finally:
            self._loading = False

    def _published_text(self, variable: str) -> str:
        """What a person calls the signal ``variable`` publishes, as every
        status and picker calls it (:func:`signal_text`)."""

        return signal_text(stable_signal_key(self._node, variable), {self._node: self._node_name})

    def _order_focus(self) -> None:
        """Tab walks the page as it reads: Add variable, then each card
        line by line -- a line built later does not go to the end."""
        chain = [self.add_variable]
        for card in self._cards:
            chain.extend(card.focus_chain())
        chain = [widget for widget in chain if widget.focusPolicy() & QtCore.Qt.TabFocus]
        for before, after in zip(chain, chain[1:]):
            QtWidgets.QWidget.setTabOrder(before, after)


__all__ = ["DeriveEditor"]
