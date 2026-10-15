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
strips; whatever comes from the data (members, variables, axes, labels,
``over``) is a combo, several values a checkable one.  A problem stands
under the line it is about, as one sentence (``show_problem``); a parked
helper's in muted ink, because it blocks nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from html import escape

from PyQt5 import QtCore, QtGui, QtWidgets, sip

from zlc_data.axis import SCALAR
from zlc_ui.fluent import (
    AXIS_GROUP_COLORS, FONT, MUTED_INK, RED_DARK, FluentButton, FluentColumnLine, FluentColumnSet,
    FluentComboBox, FluentFrame, FluentLabel, FluentLineEdit, FluentSegmentedButtons, FluentSwitch, card_pad,
    error_note_label, field_text_inset, fluent_font_size, hold_place, muted_note_label, retire_widget, row_gap,
    scaled_px, section_gap, show_problem, signals_blocked,
)
from zlc_ui.fluent.column_set import section_header

from zlc_atom.authoring import AuthoringField, AuthoringSchema

from .expression import (
    CATEGORIES, NO_PUBLISHED, OPERATORS, STATISTICS, TERM_COLUMNS, VARIABLE_COLUMNS, compact_variable,
    describe_schema, error_stage, term_text, typed_variables, where_notes,
)

#: The rows as a draft holds them, completed column by column; no validator,
#: because a problem is shown under its line, not refused.
_ROWS = AuthoringSchema((AuthoringField("variables", "rows", "Variables", (), columns=VARIABLE_COLUMNS),))
#: A term with nothing written: every column at its default.
_TERM = {column.name: () if column.value_type == "text_tuple" else column.default for column in TERM_COLUMNS}
_STRIPS = dict(CATEGORIES)
_FOOTER = "Averages over shots with error bars, per-site probabilities and fits belong in a plot panel."
_EMPTY = ("No variables yet. A variable picks values of an input, combines them line by line, can mask and "
          "reduce them; Publish makes it a signal.")
#: The operand combo's constants.  A member or variable is an identifier,
#: so no name reads like these keys.
_NUMBER, _TRUE, _FALSE = "#number", "#true", "#false"
_CONSTANTS = {_NUMBER: "Number", _TRUE: "true", _FALSE: "false"}
_DOMAINS = (("Repeat", "repeat_domain"), ("Point", "point_domain"), ("Cell", "cell_domain"))
#: The most labels one labels list offers; a longer axis says how many it has.
_LABEL_LIMIT = 2048
#: The least room a variable's name gets (design px); the lead column is as
#: wide as the widest operator strip when that is wider.
_NAME_WIDTH = 150
#: The columns of every card line: the name (or a later line's operator
#: strip), the "=" gutter, the operand, ``at`` and its axis (or a number),
#: the labels (or a number's unit), the rest of the line, Publish and ×.
_COLUMNS = (("lead", "fit"), ("gutter", "fit"), ("operand", "fit"), ("at", "fit"), ("labels", "fit"),
            ("tail", "stretch"), ("publish", "fit"), ("remove", "glyph"))
_ADD = (("arithmetic", "+ Arithmetic"), ("comparison", "+ Comparison"), ("logic", "+ Logic"),
        ("where", "+ Where"), ("statistic", "+ Statistic"))


def _set_text(widget, text: str) -> None:
    if widget.text() != text:
        widget.setText(text)


def _visible(widget: QtWidgets.QWidget, shown: bool, hold: bool = True) -> None:
    """Show or hide ``widget``; hidden with ``hold`` it keeps its place in
    its line, so its column keeps its width whichever lines use it."""

    policy = widget.sizePolicy()
    if policy.retainSizeWhenHidden() != hold:
        policy.setRetainSizeWhenHidden(hold)
        widget.setSizePolicy(policy)
    if widget.isHidden() == shown:
        widget.setVisible(shown)


def _ink(label: QtWidgets.QLabel, colour: str) -> None:
    """A problem's ink: red when it blocks, muted when it does not."""

    if label.property("ink") != colour:
        label.setProperty("ink", colour)
        label.setStyleSheet(
            f'color: {colour}; background: transparent; border: none; font: {fluent_font_size()}pt "{FONT}";')


def _alive(widget: QtWidgets.QWidget) -> bool:
    return not sip.isdeleted(widget) and widget.isVisible()


def _boolean(schema) -> bool:
    return schema.value_schema.dtype.kind == "b"


def _flavour(schema) -> str | None:
    """"b" for booleans, "n" for numbers, None while not known."""
    return None if schema is None else "b" if _boolean(schema) else "n"


def _domains_of(schema) -> list[tuple[str, str, object, tuple]]:
    """(domain word, its colour, the domain, its primary named axes) of each
    domain that has some; the one-value placeholder axis is never one."""
    groups = []
    for (word, attribute), colour in zip(_DOMAINS, AXIS_GROUP_COLORS):
        domain = getattr(schema, attribute)
        axes = tuple(axis for axis in domain.axes if axis.coordinate_of is None and axis.role != SCALAR)
        if axes:
            groups.append((word, colour, domain, axes))
    return groups


def _axis_named(schema, name: str):
    """(axis, its domain) of the one primary axis named ``name``, or None."""
    found = [(axis, domain) for _word, _colour, domain, axes in _domains_of(schema)
             for axis in axes if axis.name == name]
    return found[0] if len(found) == 1 else None


def _item(text: str, data) -> tuple:
    return (text, data, "", "")


def _header(text: str, colour: str = "") -> tuple:
    return (text, None, "header", colour)


def _fill(combo: FluentComboBox, items, current=None, *, headers_enabled: bool = False) -> None:
    """Offer ``items`` -- ``(text, data, kind, colour)``, a header bold and
    keyed None, a note muted -- refilled only when they change, and mark
    ``current`` (nothing when None) on a one-value combo."""

    items = tuple(items)
    shown = tuple((combo.itemText(row), combo.itemData(row)) for row in range(combo.count()))
    if shown != tuple((text, data) for text, data, _kind, _colour in items):
        with signals_blocked(combo):
            combo.clear()
            for text, data, kind, colour in items:
                combo.addItem(text, data)
                if not kind:
                    continue
                item = combo.model().item(combo.count() - 1)
                item.setEnabled(kind == "header" and headers_enabled)
                if kind == "header":
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                if colour:
                    item.setForeground(QtGui.QBrush(QtGui.QColor(colour)))
    if current is not None or combo.currentIndex() >= 0:
        with signals_blocked(combo):
            combo.setCurrentIndex(-1 if current is None else combo.findData(current))


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


def _operand_key(term: Mapping[str, object]):
    if term["number"] is not None:
        return _NUMBER
    if term["boolean"]:
        return f"#{term['boolean']}"
    return term["of"] or None


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
        hold_place(self.add["statistic"])
        add.place("operand", *self.add.values(), to="tail")
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
        ui: dict[str, object] = {"kind": kind, "block": block, "error": error, "hold": False}
        if kind is None:
            name = FluentLineEdit("")
            name.setPlaceholderText("name")
            name.textEdited.connect(lambda text: self._set(name=text))
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
        labels.set_checkable(True, placeholder="choose")
        labels.checked_edited.connect(lambda values, ui=ui: self._term_edit(ui, labels=tuple(values)))
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
        word = FluentLabel("over")
        over = FluentComboBox()
        over.set_checkable(True, placeholder="choose", headers_select=True)
        over.checked_edited.connect(lambda values: self._set(over=tuple(values)))
        line.place("operand", strip, word, over, to="tail")
        remove = _remove_button(self, "Remove the statistic")
        remove.clicked.connect(self._close_statistic)
        line.place("remove", remove)
        return {"block": block, "error": error, "strip": strip, "word": word, "over": over, "remove": remove,
                "hold": False}

    def _retire(self, ui: Mapping[str, object]) -> None:
        self._lines.removeWidget(ui["block"])
        self._editor.columns.forget(ui["block"])
        retire_widget(ui["block"])

    # -- edits -------------------------------------------------------------

    def _set(self, **changes) -> None:
        self.row.update(changes)
        self._editor.card_edited()

    def _term(self, ui) -> dict[str, object]:
        """The term ``ui`` edits; the head line of a variable with no term yet writes its first."""
        if not self.row["terms"]:
            self.row["terms"].append(dict(_TERM))
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
            # The pick stays when the new operand has its axis, or while
            # nothing says which axes it has.
            schema = self._context["schema_of"](key)
            keeps = schema is None or (term["at"] and _axis_named(schema, term["at"]) is not None)
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
        """The axis, its labels cleared; then its labels to pick, the
        line's problem held back while that list is open."""
        axis = ui["axis"].currentData() or ""
        if axis == self._term(ui)["at"]:
            return
        ui["hold"] = bool(axis)
        self._term_edit(ui, at=axis, labels=())
        if axis:
            QtCore.QTimer.singleShot(0, lambda: self._cascade(ui))

    def _cascade(self, ui) -> None:
        labels = ui["labels"]
        if ui in self.terms_ui and _alive(labels) and labels.count():
            labels.showPopup()
            labels.view().window().installEventFilter(self)
            return
        if ui in self.terms_ui:
            ui["hold"] = False
            self._editor.refresh()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        # The cascaded labels list closed: its line says its problem again,
        # once the list's own close has written what was ticked.
        if event.type() == QtCore.QEvent.Hide:
            for ui in self.terms_ui:
                if ui["hold"] and watched is ui["labels"].view().window():
                    ui["hold"] = False
                    QtCore.QTimer.singleShot(0, self._editor.refresh)
        return super().eventFilter(watched, event)

    def _statistic_chosen(self, statistic: str) -> None:
        """The statistic; with nothing to reduce over yet, every Cell axis
        of the value -- or, with none, the list of its axes to pick."""
        self.row["statistic"] = statistic
        reducible = self._answer.get("reducible")
        cascade = False
        if not self.row["over"] and reducible is not None:
            cells = tuple(axis.name for word, _colour, _domain, axes in _domains_of(reducible)
                          if word == "Cell" for axis in axes)
            if cells:
                self.row["over"] = cells
            else:
                cascade = bool(_domains_of(reducible))
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
            self.row["terms"].append(dict(_TERM))
        head = self.row["terms"][0]
        schema_of = self._context["schema_of"]
        term = dict(_TERM, op=_STRIPS[kind][0])
        if kind in ("arithmetic", "comparison"):
            pool = [name for name in self._context["names"]
                    if kind == "comparison" or _flavour(schema_of(name)) != "b"]
            of = head["of"] if head["of"] in pool else pool[0] if pool else ""
            if of:
                term["of"] = of
            else:
                term["number"] = 0.0
        else:
            # A verdict: a boolean variable this one may read, else a member.
            pool = [name for name in (*self._context["variables"], *self._context["members"])
                    if _flavour(schema_of(name)) == "b"]
            term["of"] = pool[0] if pool else ""
        schema = schema_of(term["of"]) if term["of"] else None
        found = None if schema is None or not head["at"] else _axis_named(schema, head["at"])
        if found is not None:
            if kind != "where":
                term["at"] = head["at"]
            elif all(label in found[0].labels() for label in head["labels"]):
                # The mask at the value's own pick.
                term.update(at=head["at"], labels=tuple(head["labels"]))
        self.row["terms"].append(term)
        self._editor.card_edited()
        if term["at"] and not term["labels"] and len(self.terms_ui) == len(self.row["terms"]):
            ui = self.terms_ui[-1]
            ui["hold"] = True
            show_problem(ui["error"], "")
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
        terms = self.row["terms"] or [dict(_TERM)]
        infos = answer["terms"] or ({"operand": None, "left": None, "number_unit": None},)
        self._shape(terms)
        _set_text(self.head["name"], str(row["name"]))
        publish = self.head["publish"]
        with signals_blocked(publish):
            publish.setChecked(bool(row["publish"]))
        tip = (f"Published as @logic/{context['node']}/{row['name']}" if row["publish"]
               else "Internal: other variables read it; nothing publishes it")
        if publish.toolTip() != tip:
            publish.setToolTip(tip)
        for ui, term, info in zip(self.terms_ui, terms, infos):
            self._show_term(ui, term, info)
        self._show_statistic(answer.get("reducible"))

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
            _ink(ui["error"], MUTED_INK if muted else RED_DARK)
            show_problem(ui["error"], detail if ui is said and not ui["hold"] else "")
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
            strip = ui["strip"]
            strip.set_current(term["op"] or None)
            if kind == "comparison":
                # Booleans compare only with = and ≠; one the draft holds
                # stays marked, and the line's problem says why.
                boolean_left = info["left"] is not None and _boolean(info["left"])
                for op in (">", "<", ">=", "<="):
                    strip.button(op).setEnabled(not boolean_left or op == term["op"])
        _fill(ui["operand"], self._operands(kind, term, info), _operand_key(term))

        number = term["number"] is not None
        schema = schema_of(term["of"]) if term["of"] else None
        groups = _domains_of(schema) if schema is not None else []
        found = None if schema is None or not term["at"] else _axis_named(schema, term["at"])
        picks = bool(term["of"]) and (bool(groups) or bool(term["at"]))
        _visible(ui["at"], picks, hold=not number)
        _visible(ui["axis"], picks, hold=not number)
        _visible(ui["number"], number, hold=False)
        _visible(ui["labels"], picks and bool(term["at"]), hold=not number)
        _visible(ui["unit"], number, hold=False)
        if picks:
            items = [_item("All", "")]
            for word, colour, _domain, axes in groups:
                items.append(_header(word, colour))
                items.extend(_item(f"    {axis.name}", axis.name) for axis in axes)
            if term["at"] and found is None:
                items.append(_item(term["at"], term["at"]))
            _fill(ui["axis"], items, term["at"])
        if picks and term["at"]:
            labels = ui["labels"]
            if found is None:
                offered = [_item(label, label) for label in term["labels"]]
            else:
                offered = self._label_items(*found)
            offered.extend(_item(label, label) for label in term["labels"]
                           if label not in {data for _text, data, _kind, _colour in offered})
            _fill(labels, offered)
            labels.set_checked_data(term["labels"])
        if number:
            if not ui["number"].hasFocus():
                _set_text(ui["number"], term_text(term))
            _set_text(ui["unit"], info["number_unit"] or "")

    @staticmethod
    def _label_items(axis, domain) -> list[tuple]:
        """The axis' labels, each with its alternate coordinates' (``1 · time 0.1 s``)."""
        labels = axis.labels()
        alternates = [(other, other.labels()) for other in domain.axes if other.coordinate_of == axis.axis_id]
        items = [
            _item(label + "".join(f" · {other.name} {texts[index]}" + (f" {other.unit}" if other.unit else "")
                                  for other, texts in alternates), label)
            for index, label in enumerate(labels[:_LABEL_LIMIT])
        ]
        if len(labels) > _LABEL_LIMIT:
            items.append((f"{_LABEL_LIMIT} of {len(labels)} shown", None, "note", MUTED_INK))
        return items

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
        _visible(self.add["statistic"], not shown)
        strip = line["strip"]
        strip.set_current(row["statistic"] or None)
        boolean = reducible is not None and _boolean(reducible)
        for name in STATISTICS:
            withheld = reducible is not None and (
                name in ("min", "max", "std") if boolean else name in ("any", "all"))
            strip.button(name).setEnabled(not withheld or name == row["statistic"])
        items = []
        for word, colour, _domain, axes in ([] if reducible is None else _domains_of(reducible)):
            items.append(_header(word, colour))
            items.extend(_item(axis.name, axis.name) for axis in axes)
        known = {data for _text, data, _kind, _colour in items}
        items.extend(_item(name, name) for name in row["over"] if name not in known)
        offered = bool(items)
        line["word"].setVisible(offered)
        line["over"].setVisible(offered)
        _fill(line["over"], items, headers_enabled=True)
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
        #: The members' typing schemas once every member has published, else None.
        self._typing: dict[str, object] | None = None
        self._node = ""
        self._failure: tuple[str, str, str] | None = None
        self._draft_problem = ""
        #: The columns every line of every card shares.  The "=" and the
        #: lead are pinned, so where the operands stand is known before the
        #: first card is built: the label column above is fitted to it.
        self.columns = FluentColumnSet(_COLUMNS, self)
        self._gutter = FluentLabel("=").sizeHint().width()
        self.columns.set_width("gutter", self._gutter)
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
                for word, colour, _domain, domain_axes in _domains_of(schema):
                    for axis in domain_axes:
                        groups.append(f'<span style="color: {colour}">{escape(axis.name)} {axis.size}</span>')
                        labels = axis.labels()
                        shown = ", ".join(labels) if len(labels) <= 8 else f"{labels[0]} … {labels[-1]}"
                        lines.append(f"{axis.name} — {axis.axis_id.value} · {word} · {axis.role.value}"
                                     f" · {shown}")
                _set_text(axes, " × ".join(groups) or "one value")
                tooltip = "\n".join(lines)
            for label in (member, kind, axes):
                if label.toolTip() != tooltip:
                    label.setToolTip(tooltip)

    def update_projection(self, projection: Mapping[str, object]) -> None:
        values = projection.get("form_values") or {}
        bundle = tuple(projection.get("input_bundle") or ())
        self._node = str(projection.get("node_id") or "")
        self._show_inputs(bundle)
        self._members = tuple(str(name) for name, _schema, _snapshot in bundle)
        schemas = {str(name): schema for name, schema, _snapshot in bundle}
        self._typing = schemas if bundle and all(schema is not None for schema in schemas.values()) else None
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
        show_problem(self.note, problem[:1].upper() + problem[1:])
        answers = typed_variables(rows, self._typing)
        members, schemas = self._members, self._typing or {}
        typed: dict[str, object] = {}
        reads: dict[str, set[str]] = {}
        for row, answer in zip(rows, answers):
            typed.setdefault(row["name"], answer["schema"])
            reads.setdefault(row["name"], set()).update(term["of"] for term in row["terms"] if term["of"])

        def schema_of(name):
            return schemas.get(name) if name in members else typed.get(name)

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
                    "schema_of": schema_of, "typed": self._typing is not None, "node": self._node,
                    "failure": failure[1:] if failure is not None and failure[0] == row["name"] else None,
                })
            self.columns.refresh()
            self.empty.setVisible(not rows)
        finally:
            self._loading = False


__all__ = ["DeriveEditor"]
