"""Derive's outputs, edited as the rows they are, with each row's typed answer.

This leaf edits plain draft values.  Its previews come from
``typed_outputs`` -- the evaluation's own plans applied to the schemas the
Runtime already published -- so it neither evaluates outputs nor asks a
device or Runtime to materialize data on the Qt thread.

Every line of every output card is a line of ONE column set: the name (or a
stage's word) in the lead column, the operand under the operand, ``at`` under
``at``, the card's add commands in fixed slots and every × on one x, so a
card reads as a row of a table rather than a sentence packed from its own
left edge.  A problem or a result stands under its line, from the operand's
text.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    AXIS_GROUP_COLORS, CARD_PAD, FluentButton, FluentCheckBox, FluentCodeEdit, FluentColumnLine, FluentColumnSet,
    FluentComboBox, FluentFrame, FluentLabel, FluentLineEdit, FluentSectionLabel, FluentSegmentedButtons,
    error_note_label, field_text_inset, hold_place, line_gap, muted_note_label, popup_gap, retire_widget,
    scaled_px, show_note, signals_blocked,
)
from zlc_ui.fluent.fluent import _FluentRoundedMenu

from .expression import (
    OUTPUT_COLUMNS, ROW_DEFAULTS, STATISTICS, compact_row, complete_row, describe_schema, error_stage,
    event_labels, number, over_axes, readout_axis, references, typed_outputs, within_shot_axes,
)

_OPERATION_LABELS = {choice.value: choice.label for column in OUTPUT_COLUMNS if column.name == "op"
                     for choice in column.choices if choice.value}
_NUMBER = "#number"
#: How a combo shows the items that are not names: the number a combine
#: takes instead of a signal, and "every frame" at an ``at``.
_SHOWN = {_NUMBER: "number", "": "All"}
_FOOTER = "Averages over shots, per-site probabilities and fits belong in a plot panel."
#: The width of an output's name (design px), and so of the lead column the
#: stage words under it share.
_NAME_WIDTH = 150
#: The columns of every card line: the name or stage word, the "=" gutter,
#: the operand, the ``at`` pin (or a combine's number), the rest of the line,
#: the card's add commands and the line's ×.
_COLUMNS = (("lead", "fit"), ("gutter", "fit"), ("operand", "fit"), ("at", "fit"),
            ("tail", "stretch"), ("actions", "slots"), ("remove", "glyph"))


def _set_text(widget, text: str) -> None:
    if widget.text() != text:
        widget.setText(text)


def _pinned(name: str, label: str) -> str:
    return f"{name}@{label}" if label else name


def _labels_of(schema) -> tuple[str, ...] | None:
    """The readout-event labels ``at`` offers on this schema; None if it has none."""
    if schema is None:
        return None
    try:
        axis = readout_axis(schema)
    except ValueError:
        return None
    return None if axis is None else event_labels(axis)


def _remove_button(parent, tooltip: str) -> FluentButton:
    """A line's own ×: the quiet glyph that turns red under the pointer."""

    button = FluentButton("×", parent, role="destructive")
    button.setToolTip(tooltip)
    return button


def _text_word(text: str) -> QtWidgets.QLabel:
    """A word standing in a column of fields, its ink where their text starts."""

    word = FluentLabel(text)
    word.setContentsMargins(field_text_inset(), 0, 0, 0)
    return word


def _card_column(card: QtWidgets.QWidget) -> QtWidgets.QVBoxLayout:
    column = QtWidgets.QVBoxLayout(card)
    pad = scaled_px(CARD_PAD)
    column.setContentsMargins(pad, pad, pad, pad)
    column.setSpacing(line_gap())
    return column


def _under_lines(columns: FluentColumnSet, label: QtWidgets.QLabel) -> QtWidgets.QLabel:
    """A note under a card's lines, from the operand's text to the card's edge."""

    columns.follow(label, "operand", inset=field_text_inset())
    return label


class _OutputCard(FluentFrame):
    """One structured output: ``name = value at …``, then its optional stages."""

    kind = "structured"

    def __init__(self, editor: "DeriveEditor"):
        super().__init__(editor)
        self._editor = editor
        self.row = dict(ROW_DEFAULTS)
        #: Stages the operator opened and has not filled yet.
        self._opened: set[str] = set()
        column = _card_column(self)

        line, block = self._line(column)
        self.name = FluentLineEdit("")
        self.name.setPlaceholderText("name")
        self.name.textEdited.connect(lambda text: self._edit(name=text))
        line.place("lead", self.name, fill=True)
        line.place("gutter", FluentLabel("="))
        self.value = FluentComboBox()
        self.value.activated.connect(lambda _index: self._edit(value=self.value.currentData()))
        line.place("operand", self.value, fill=True)
        # The frames come from the data -- one per readout event -- so they
        # are a list to pick from, "All" first.  A value with no frames
        # leaves the pin's place empty rather than pulling the line together.
        self.at_label = hold_place(FluentLabel("at"))
        self.at = hold_place(FluentComboBox())
        self.at.activated.connect(lambda _index: self._edit(at=self.at.currentData()))
        line.place("at", self.at_label)
        line.place("at", self.at, fill=True)
        # One slot each: a stage in use leaves its slot empty, so the others
        # stay where the operator last found them.
        self.add = {}
        for stage in ("where", "combine", "statistic"):
            button = hold_place(FluentButton(f"+ {stage}", self))
            button.clicked.connect(lambda _checked=False, stage=stage: self._open(stage))
            self.add[stage] = button
        line.place("actions", *self.add.values())
        self.remove = _remove_button(self, "Remove this output")
        self.remove.clicked.connect(lambda: self._editor.remove_card(self))
        line.place("remove", self.remove)
        self.value_error = self._error_under(block)

        self.lines = {}
        self.errors = {}
        where = self._stage_line(column, "where")
        self.mask = FluentComboBox()
        self.mask.activated.connect(lambda _index: self._edit(where=self.mask.currentData()))
        where.place("operand", self.mask, fill=True)
        self.where_at = FluentComboBox()
        self.where_at.activated.connect(lambda _index: self._edit(where_at=self.where_at.currentData()))
        where.place("at", FluentLabel("at"))
        where.place("at", self.where_at, fill=True)
        self.where_is = FluentSegmentedButtons(((True, "true"), (False, "false")), self)
        self.where_is.activated.connect(lambda keep: self._edit(where_is=keep))
        where.place("tail", FluentLabel("is"), self.where_is, to="actions")

        # The operand stands under the value it combines with; the operation
        # follows it, where the line has room for all of them.
        combine = self._stage_line(column, "combine")
        self.other = FluentComboBox()
        self.other.activated.connect(self._other_chosen)
        combine.place("operand", self.other, fill=True)
        self.with_at_label = FluentLabel("at")
        self.with_at = FluentComboBox()
        self.with_at.activated.connect(lambda _index: self._edit(with_at=self.with_at.currentData()))
        # A number has no frame to pin: it takes the pin's place.
        self.constant = FluentLineEdit("")
        self.constant.setPlaceholderText("number")
        self.constant.textEdited.connect(lambda text: self._edit(**{"with": text.strip()}))
        combine.place("at", self.with_at_label)
        combine.place("at", self.with_at, self.constant, fill=True)
        self.op = FluentSegmentedButtons(tuple(_OPERATION_LABELS.items()), self)
        self.op.activated.connect(lambda op: self._edit(op=op))
        combine.place("tail", FluentLabel("using"), self.op, to="actions")

        statistic = self._stage_line(column, "statistic")
        self.statistic = FluentSegmentedButtons(tuple((name, name) for name in STATISTICS), self)
        self.statistic.activated.connect(lambda name: self._edit(statistic=name))
        over = QtWidgets.QWidget(self)
        self.over_host = QtWidgets.QHBoxLayout(over)
        self.over_host.setContentsMargins(0, 0, 0, 0)
        self.over_host.setSpacing(line_gap())
        statistic.place("operand", self.statistic, FluentLabel("over"), over, to="actions")
        self.over: dict[str, QtWidgets.QCheckBox] = {}

        self.preview = _under_lines(editor.columns, muted_note_label(""))
        self.preview.setWordWrap(True)
        column.addWidget(self.preview)

    # -- construction -------------------------------------------------------

    def _line(self, column) -> tuple[FluentColumnLine, QtWidgets.QWidget]:
        """A line and the block that keeps its error under it."""

        block = QtWidgets.QWidget(self)
        rows = QtWidgets.QVBoxLayout(block)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(line_gap() // 2)
        line = FluentColumnLine(self._editor.columns, block)
        rows.addWidget(line)
        column.addWidget(block)
        return line, block

    def _error_under(self, block) -> QtWidgets.QLabel:
        label = _under_lines(self._editor.columns, error_note_label())
        block.layout().addWidget(label)
        return label

    def _stage_line(self, column, stage) -> FluentColumnLine:
        line, block = self._line(column)
        # The stage's word is where the name's text is: one left edge in the
        # card, the hierarchy told by the word, not by an indent.
        line.place("lead", _text_word(stage))
        close = _remove_button(self, f"Remove the {stage} step")
        close.clicked.connect(lambda: self._close(stage))
        line.place("remove", close)
        self.lines[stage] = block
        self.errors[stage] = self._error_under(block)
        return line

    # -- edits -------------------------------------------------------------

    def _edit(self, **changes) -> None:
        self.row.update(changes)
        self._editor.card_edited()

    def _open(self, stage) -> None:
        self._opened.add(stage)
        if stage == "where":
            # The mask most rows want: a verdict, at the value's own event.
            masks = self._editor.boolean_names(self)
            if masks and not self.row["where"]:
                mask = masks[0]
                labels = _labels_of(self._editor.schema_of(self, mask)) or ()
                self._edit(where=mask, where_at=self.row["at"] if self.row["at"] in labels else "")
                return
        self.lines[stage].show()
        self.add[stage].hide()

    def _close(self, stage) -> None:
        self._opened.discard(stage)
        # An opened step left empty changes no column: nothing re-projects it.
        self.lines[stage].hide()
        self.add[stage].show()
        columns = {"where": {"where": "", "where_at": "", "where_is": True},
                   "combine": {"op": "", "with": "", "with_at": ""},
                   "statistic": {"statistic": "", "over": ""}}[stage]
        self._edit(**columns)

    def _other_chosen(self, _index) -> None:
        chosen = self.other.currentData()
        if chosen == _NUMBER:
            self.constant.show()
            self.constant.setFocus()
            self._edit(**{"with": self.constant.text().strip(), "with_at": ""})
        else:
            self._edit(**{"with": chosen, "with_at": ""})

    def _over_toggled(self) -> None:
        self._edit(over=", ".join(name for name, box in self.over.items() if box.isChecked()))

    # -- projection --------------------------------------------------------

    @staticmethod
    def _fill(combo, items, current) -> None:
        """Offer ``items`` (and a ``current`` they lack), refilled only when they change."""
        items = tuple(items)
        if current and current not in items:
            items = (*items, current)
        if tuple(combo.itemData(index) for index in range(combo.count())) != items:
            with signals_blocked(combo):
                combo.clear()
                for item in items:
                    combo.addItem(_SHOWN.get(item, item), item)
        with signals_blocked(combo):
            combo.setCurrentIndex(items.index(current) if current in items else -1)

    def show_row(self, row: Mapping[str, object], context: Mapping[str, object]) -> None:
        self.row = dict(row)
        names, schema_of = context["names"], context["schema_of"]
        answer = context["answer"]
        _set_text(self.name, row["name"])
        self._fill(self.value, names, row["value"])
        labels = _labels_of(schema_of(row["value"]))
        if labels is None and row["value"] in context["frames_of"]:
            # Before the source publishes, its producer says which frames come.
            labels = context["frames_of"][row["value"]]
        if row["at"] and not labels:
            labels = (row["at"],)
        self.at_label.setVisible(bool(labels))
        self.at.setVisible(bool(labels))
        if labels:
            self._fill(self.at, ("", *labels), row["at"])

        shown = {"where": bool(row["where"]), "combine": bool(row["op"] or row["with"]),
                 "statistic": bool(row["statistic"] or row["over"])}
        self._opened -= {stage for stage, filled in shown.items() if filled}
        for stage, filled in shown.items():
            self.lines[stage].setVisible(filled or stage in self._opened)
            self.add[stage].setVisible(not (filled or stage in self._opened))

        self._fill(self.mask, context["booleans"], row["where"])
        mask_labels = (_labels_of(schema_of(row["where"])) or context["frames_of"].get(row["where"])
                       or ((row["where_at"],) if row["where_at"] else ()))
        self._fill(self.where_at, ("", *mask_labels), row["where_at"])
        self.where_is.set_current(bool(row["where_is"]))

        self.op.set_current(row["op"] or None)
        # A number being typed stays in its box until it reads as one.
        constant = number(row["with"]) is not None or (
            self.other.currentData() == _NUMBER and (not row["with"] or self.constant.hasFocus()))
        self._fill(self.other, (*names, _NUMBER), _NUMBER if constant else row["with"])
        self.constant.setVisible(constant)
        if constant:
            _set_text(self.constant, row["with"])
        with_labels = () if constant else (
            _labels_of(schema_of(row["with"])) or context["frames_of"].get(row["with"])
            or ((row["with_at"],) if row["with_at"] else ()))
        self.with_at_label.setVisible(bool(with_labels))
        self.with_at.setVisible(bool(with_labels))
        if with_labels:
            self._fill(self.with_at, ("", *with_labels), row["with_at"])

        self.statistic.set_current(row["statistic"] or None)
        reducible = answer["reducible"]
        offered = tuple(axis.name for axis in within_shot_axes(reducible)) if reducible is not None else ()
        chosen = over_axes(row["over"])
        offered = (*offered, *(name for name in chosen if name not in offered))
        if tuple(self.over) != offered:
            for box in self.over.values():
                retire_widget(box)
            self.over = {}
            for name in offered:
                box = FluentCheckBox(name)
                box.toggled.connect(self._over_toggled)
                self.over_host.addWidget(box)
                self.over[name] = box
        for name, box in self.over.items():
            with signals_blocked(box):
                box.setChecked(name in chosen)

        # A run's failure names a row the schemas could not type; a row they
        # did type answers for itself.
        if answer["error"] is not None:
            stage, detail = answer["error"].stage, answer["error"].detail
        elif context["failure"] is not None and answer["schema"] is None:
            _output, stage, detail = context["failure"]
        else:
            stage, detail = None, ""
        show_note(self.value_error, detail if stage in ("name", "value") else "")
        for name, label in self.errors.items():
            show_note(label, detail if stage == name else "")
        self.preview.setVisible(stage is None)
        if stage is None:
            self.preview.setText(self._preview(row, answer, context))

    @staticmethod
    def _preview(row, answer, context) -> str:
        schema = answer["schema"]
        if schema is None:
            waiting = [name for _stage, name in references(row) if name in context["failed"]]
            if waiting:
                return f"→ waits for {waiting[0]}"
            if context["typed"]:
                return "→ known after Start: it reads a code output"
            return "→ checked when the source publishes"
        notes = [describe_schema(schema)]
        reduced = answer["reducible"]
        if row["statistic"] and reduced is not None and reduced.value_schema.dtype.kind == "b":
            notes.append({"sum": "number true", "mean": "fraction true",
                          "count": "valid samples"}.get(row["statistic"], row["statistic"]))
        elif row["statistic"] == "count":
            notes.append("valid samples")
        if row["where"]:
            notes.append(f"valid where {_pinned(row['where'], row['where_at'])} is "
                         f"{'true' if row['where_is'] else 'false'}")
        return "→ " + " · ".join(notes)


class _CodeCard(FluentFrame):
    """One code output: ``name = code``, Python over ``a.<member>``."""

    kind = "code"

    def __init__(self, editor: "DeriveEditor"):
        super().__init__(editor)
        self._editor = editor
        self.row = dict(ROW_DEFAULTS)
        column = _card_column(self)
        line = FluentColumnLine(editor.columns, self)
        self.name = FluentLineEdit("")
        self.name.setPlaceholderText("name")
        self.name.textEdited.connect(lambda text: self._edit(name=text))
        line.place("lead", self.name, fill=True)
        line.place("gutter", FluentLabel("="))
        line.place("operand", _text_word("code"))
        # A command whose list is the data's -- every member and axis name --
        # so the names wait behind it in a menu instead of lining the card.
        # It is the card's command, so it stands in the card's command slot.
        self.insert = hold_place(FluentButton("Insert", self))
        self.insert.setToolTip("Insert a member or axis name at the cursor")
        self.insert.clicked.connect(self._show_inserts)
        line.place("actions", self.insert)
        self.remove = _remove_button(self, "Remove this output")
        self.remove.clicked.connect(lambda: self._editor.remove_card(self))
        line.place("remove", self.remove)
        column.addWidget(line)
        self.code = FluentCodeEdit("")
        self.code.setPlaceholderText("result = …  (ordinary Python over a.<member>, np and earlier outputs)")
        self.code.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.code.setTabStopDistance(self.code.fontMetrics().horizontalAdvance("    "))
        self.code.textChanged.connect(self._code_changed)
        column.addWidget(self.code)
        self._inserts: tuple[str, ...] = ()
        self.example = _under_lines(editor.columns, muted_note_label(""))
        self.example.setWordWrap(True)
        self.example.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        column.addWidget(self.example)
        self.status = _under_lines(editor.columns, muted_note_label(""))
        self.status.setWordWrap(True)
        column.addWidget(self.status)
        self.error = _under_lines(editor.columns, error_note_label())
        column.addWidget(self.error)
        self._fit_height()

    def _edit(self, **changes) -> None:
        self.row.update(changes)
        self._editor.card_edited()

    def _code_changed(self) -> None:
        self._fit_height()
        if self.code.toPlainText() != self.row["code"]:
            self._edit(code=self.code.toPlainText())

    def _fit_height(self) -> None:
        lines = min(14, max(2, self.code.document().blockCount()))
        margins = self.code.contentsMargins()
        height = (lines * self.code.fontMetrics().lineSpacing() + 2 * self.code.document().documentMargin()
                  + margins.top() + margins.bottom() + scaled_px(10))
        if self.code.height() != int(height):
            self.code.setFixedHeight(int(height))

    def _show_inserts(self) -> None:
        """This bundle's names below Insert, built for this click."""
        menu = _FluentRoundedMenu(self)
        offered = {menu.addAction(text): text for text in self._inserts}
        anchor = self.insert.mapToGlobal(QtCore.QPoint(0, self.insert.height() + popup_gap()))
        try:
            chosen = offered.get(menu.exec_(anchor))
        finally:
            retire_widget(menu)
        if chosen is not None:
            self._insert(chosen)

    def _insert(self, text) -> None:
        self.code.insertPlainText(str(text))
        self.code.setFocus()

    def show_row(self, row: Mapping[str, object], context: Mapping[str, object]) -> None:
        self.row = dict(row)
        _set_text(self.name, row["name"])
        if self.code.toPlainText() != row["code"]:
            with signals_blocked(self.code):
                self.code.setPlainText(row["code"])
            self._fit_height()
        self._inserts = tuple(context["inserts"])
        self.insert.setVisible(bool(self._inserts))
        # The example is for an empty box; written code speaks for itself.
        _set_text(self.example, context["example"])
        self.example.setVisible(bool(context["example"]) and not row["code"].strip())
        answer, failure = context["answer"], context["failure"]
        error = "" if not row["code"].strip() else (
            answer["error"].detail if answer["error"] is not None
            else failure[2] if failure is not None else "")
        show_note(self.error, error)
        published = context["published"]
        self.status.setText(
            "Write Python; assign the Dataset answer to result." if not row["code"].strip()
            else f"→ {describe_schema(published)}" if published is not None and not error
            else "→ known after Start: code is checked when it runs")
        self.status.setVisible(not error)


class DeriveEditor(QtWidgets.QWidget):
    """The concrete Derive contribution to the existing Logic editor."""

    draft_changed = QtCore.pyqtSignal(object)
    managed_fields = ("outputs",)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._loading = False
        self._cards: list[_OutputCard | _CodeCard] = []
        self._contexts: dict[int, Mapping[str, object]] = {}
        self._members: tuple[str, ...] = ()
        self._schemas: dict[str, object] = {}
        #: The columns every line of every card shares.
        self.columns = FluentColumnSet(_COLUMNS, self)
        self.columns.set_width("lead", scaled_px(_NAME_WIDTH))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(line_gap())

        # A row of the editor's form: "Inputs" in its label column, the
        # members' names where the bundle combo's text starts above them.
        inputs = QtWidgets.QHBoxLayout()
        inputs.setSpacing(line_gap())
        self.inputs_title = FluentLabel("Inputs")
        inputs.addWidget(self.inputs_title, 0, QtCore.Qt.AlignTop)
        self.inputs = QtWidgets.QGridLayout()
        self.inputs.setContentsMargins(field_text_inset(), 0, 0, 0)
        self.inputs.setHorizontalSpacing(scaled_px(14))
        self.inputs.setVerticalSpacing(scaled_px(2))
        inputs.addLayout(self.inputs)
        inputs.addStretch(1)
        layout.addLayout(inputs)
        self._input_rows: list[tuple[QtWidgets.QLabel, ...]] = []

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(line_gap())
        title = FluentSectionLabel("Outputs")
        # Centred on the buttons beside it, not dropped below them like an own-line section.
        title.setContentsMargins(0, 0, 0, 0)
        header.addWidget(title)
        header.addStretch(1)
        self.add_output = FluentButton("Add output", self)
        self.add_output.clicked.connect(lambda: self._add("structured"))
        header.addWidget(self.add_output)
        self.add_code = FluentButton("Add code output", self)
        self.add_code.clicked.connect(lambda: self._add("code"))
        header.addWidget(self.add_code)
        layout.addLayout(header)
        self.empty = muted_note_label(
            "No outputs yet. An output picks a value, pins it at a frame, and can mask, combine and reduce it.")
        self.empty.setWordWrap(True)
        layout.addWidget(self.empty)
        self.cards = QtWidgets.QVBoxLayout()
        self.cards.setSpacing(line_gap())
        layout.addLayout(self.cards)
        self.footer = muted_note_label(_FOOTER)
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)

    def row_labels(self) -> tuple[str, ...]:
        """The labels this editor puts in the editor's label column."""

        return (self.inputs_title.text(),)

    def set_label_width(self, width: int) -> None:
        """The editor's label column, shared with the bundle row above."""

        self.inputs_title.setFixedWidth(int(width))

    def set_mutation_enabled(self, enabled: bool) -> None:
        self.add_output.setEnabled(enabled)
        self.add_code.setEnabled(enabled)
        for card in self._cards:
            card.setEnabled(enabled)

    # -- what the cards ask -------------------------------------------------

    def schema_of(self, card, name):
        context = self._contexts.get(id(card))
        return None if context is None else context["schema_of"](name)

    def boolean_names(self, card) -> tuple[str, ...]:
        context = self._contexts.get(id(card))
        return () if context is None else tuple(context["booleans"])

    # -- edits -------------------------------------------------------------

    def card_edited(self) -> None:
        if not self._loading:
            self.draft_changed.emit({"values": {
                "outputs": tuple(compact_row(card.row) for card in self._cards),
            }})

    def remove_card(self, card) -> None:
        self._cards.remove(card)
        self._contexts.pop(id(card), None)
        self._retire(card)
        self.columns.refresh()
        self.empty.setVisible(not self._cards)
        self.card_edited()

    def _retire(self, card) -> None:
        self.columns.forget(card)
        retire_widget(card)

    def _card(self, kind):
        card = _CodeCard(self) if kind == "code" else _OutputCard(self)
        card.setEnabled(self.add_output.isEnabled())
        return card

    def _add(self, kind) -> None:
        taken = {card.row["name"] for card in self._cards}
        ordinal = 1
        while f"output_{ordinal}" in taken:
            ordinal += 1
        card = self._card(kind)
        card.row = dict(ROW_DEFAULTS, name=f"output_{ordinal}")
        if kind != "code" and self._members:
            card.row["value"] = self._members[0]
        self._cards.append(card)
        self.cards.addWidget(card)
        self.columns.refresh()
        self.empty.hide()
        self.card_edited()
        (card.code if kind == "code" else card.name).setFocus()

    # -- projection --------------------------------------------------------

    def _show_inputs(self, bundle) -> None:
        while len(self._input_rows) > len(bundle):
            for label in self._input_rows.pop():
                retire_widget(label)
        for index, (name, schema, snapshot) in enumerate(bundle):
            if index == len(self._input_rows):
                labels = (FluentLabel(""), muted_note_label(""), FluentLabel(""))
                labels[2].setTextFormat(QtCore.Qt.RichText)
                for column, label in enumerate(labels):
                    self.inputs.addWidget(label, index, column)
                self._input_rows.append(labels)
            member, kind, axes = self._input_rows[index]
            schema = snapshot.block.schema if snapshot is not None else schema
            _set_text(member, str(name))
            if schema is None:
                _set_text(kind, "waiting for its first publication")
                _set_text(axes, "")
                tooltip = ""
            else:
                value = schema.value_schema
                _set_text(kind, f"{value.dtype} · unit {value.value_unit or '1'}")
                groups, lines = [], []
                for colour, domain_label, domain in zip(
                        AXIS_GROUP_COLORS, ("Repeat", "Point", "Cell"),
                        (schema.repeat_domain, schema.point_domain, schema.cell_domain)):
                    for axis in domain.axes:
                        if axis.coordinate_of is not None or axis.role.value == "scalar":
                            continue
                        groups.append(f'<span style="color: {colour}">{escape(axis.name)} {axis.size}</span>')
                        labels = event_labels(axis)
                        shown = ", ".join(labels) if len(labels) <= 8 else f"{labels[0]} … {labels[-1]}"
                        lines.append(f"{axis.name} — {axis.axis_id.value} · {domain_label} · {axis.role.value}"
                                     f" · {shown}")
                _set_text(axes, " × ".join(groups) or "one value")
                tooltip = "\n".join(lines)
            for label in (member, kind, axes):
                if label.toolTip() != tooltip:
                    label.setToolTip(tooltip)

    def update_projection(self, projection: Mapping[str, object]) -> None:
        values = projection.get("form_values") or {}
        rows = tuple(complete_row(row) for row in tuple(values.get("outputs") or ()) if isinstance(row, Mapping))
        bundle = tuple(projection.get("input_bundle") or ())
        self._show_inputs(bundle)
        members = tuple(str(name) for name, _schema, _snapshot in bundle)
        schemas = {str(name): (snapshot.block.schema if snapshot is not None else schema)
                   for name, schema, snapshot in bundle}
        typing_inputs = schemas if bundle and all(schema is not None for schema in schemas.values()) else None
        self._members, self._schemas = members, schemas
        answers = typed_outputs(rows, typing_inputs)
        published = {str(name): (snapshot.block.schema if snapshot is not None else schema)
                     for name, schema, snapshot in tuple(projection.get("output_bundle") or ())}
        failure = error_stage(str(projection.get("error") or ""))
        frames = projection.get("source_frames")
        frames_of = {} if typing_inputs is not None or not frames else {name: tuple(frames) for name in members}
        failed = {row["name"] for row, answer in zip(rows, answers) if answer["error"] is not None}
        inserts, example = self._code_help(members, schemas)

        self._loading = True
        try:
            for index, row in enumerate(rows):
                structured = any(row[column] != ROW_DEFAULTS[column] for column in ROW_DEFAULTS
                                 if column not in ("name", "code"))
                kind = ("code" if row["code"] else "structured" if structured
                        else self._cards[index].kind if index < len(self._cards) else "structured")
                if index < len(self._cards) and self._cards[index].kind != kind:
                    self._contexts.pop(id(self._cards[index]), None)
                    self._retire(self._cards[index])
                    self._cards[index] = self._card(kind)
                    self.cards.insertWidget(index, self._cards[index])
                elif index == len(self._cards):
                    self._cards.append(self._card(kind))
                    self.cards.addWidget(self._cards[-1])
                earlier = {earlier_row["name"]: answer["schema"]
                           for earlier_row, answer in zip(rows[:index], answers[:index])}
                names = (*members, *(name for name in earlier if name and name not in members))

                def schema_of(name, earlier=earlier):
                    return earlier[name] if name in earlier else schemas.get(name)

                context = {
                    "names": names, "schema_of": schema_of,
                    "booleans": tuple(name for name in names
                                      if (schema := schema_of(name)) is not None
                                      and schema.value_schema.dtype.kind == "b"),
                    "answer": answers[index], "failed": failed,
                    "failure": failure if failure is not None and failure[0] == row["name"] else None,
                    "published": published.get(row["name"]), "inserts": inserts, "example": example,
                    "frames_of": frames_of, "typed": typing_inputs is not None,
                }
                card = self._cards[index]
                self._contexts[id(card)] = context
                card.show_row(row, context)
            while len(self._cards) > len(rows):
                card = self._cards.pop()
                self._contexts.pop(id(card), None)
                self._retire(card)
            self.columns.refresh()
            self.empty.setVisible(not rows)
        finally:
            self._loading = False

    @staticmethod
    def _code_help(members, schemas):
        """What Insert offers and one example, written with this bundle's real names."""
        inserts, axes = [], []
        for name in members:
            inserts.append(f"a.{name}")
            schema = schemas.get(name)
            for axis in () if schema is None else (
                    *schema.point_domain.axes, *schema.cell_domain.axes):
                if axis.coordinate_of is None and axis.role.value != "scalar" and axis.name not in axes:
                    axes.append(axis.name)
        inserts.extend(repr(name) for name in axes)
        numeric = next((name for name in members if schemas.get(name) is not None
                        and schemas[name].value_schema.dtype.kind != "b"), None)
        boolean = next((name for name in members if schemas.get(name) is not None
                        and schemas[name].value_schema.dtype.kind == "b"), None)
        if numeric is None:
            return tuple(inserts), ""
        schema = schemas[numeric]
        try:
            axis = readout_axis(schema)
        except ValueError:
            axis = None

        def pick(name):
            if axis is None:
                return f"a.{name}"
            coordinate = axis.coordinate_at(min(1, axis.size - 1))
            key = f"{axis.name}={coordinate!r}" if axis.name.isidentifier() else f"{{{axis.name!r}: {coordinate!r}}}"
            return f"a.{name}.sel({key})"

        cell = next((a.name for a in within_shot_axes(schema) if a is not axis), None)
        expression = pick(numeric)
        if boolean is not None:
            expression += f".where({pick(boolean)})"
        if cell is not None:
            expression += f".mean({cell!r})"
        return tuple(inserts), (f"e.g. result = {expression}   ·   isel/sel select, where masks, "
                                "mean sum count min max std any all reduce named axes")


__all__ = ["DeriveEditor"]
