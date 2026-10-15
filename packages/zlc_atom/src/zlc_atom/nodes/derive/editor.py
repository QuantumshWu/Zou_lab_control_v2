"""Derive's inputs and variables on the Edit page, each with its typed answer.

This leaf reads plain draft values.  Its answers come from
``typed_variables`` -- the evaluation's own plans applied to the schemas
the Runtime already published -- so it neither evaluates variables nor
asks a device or Runtime to materialize data on the Qt thread.  The
variables stand as written, one line each with its answer or its problem
under it; they are written through the node's draft (``logic.update``).
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    AXIS_GROUP_COLORS, FluentLabel, error_note_label, field_text_inset, muted_note_label, retire_widget,
    row_gap, section_gap, show_problem,
)
from zlc_ui.fluent.column_set import section_header

from zlc_atom.authoring import AuthoringField, AuthoringSchema

from .expression import (
    NO_PUBLISHED, OPERATORS, STATISTICS, VARIABLE_COLUMNS, describe_schema, error_stage, term_text,
    typed_variables,
)

_FOOTER = ("Averages over shots with error bars, per-site probabilities and fits belong in a plot panel.")
#: The rows as a draft holds them, completed column by column; no validator,
#: because a problem is shown under its variable, not refused.
_ROWS = AuthoringSchema((AuthoringField("variables", "rows", "Variables", (), columns=VARIABLE_COLUMNS),))


def _set_text(widget, text: str) -> None:
    if widget.text() != text:
        widget.setText(text)


def _formula(row) -> str:
    """``kept = counts where same``, with its statistic and whether it publishes."""
    terms = row["terms"]
    text = " ".join(
        term_text(term) if not index or term["op"] not in OPERATORS
        else f"{OPERATORS[term['op']][2]} {term_text(term)}"
        for index, term in enumerate(terms)
    ) or "…"
    if row["statistic"] in STATISTICS:
        text = f"{row['statistic']} over {', '.join(row['over']) or '…'} of {text}"
    return f"{row['name'] or '…'} = {text}" + ("" if row["publish"] else "   (internal)")


class DeriveEditor(QtWidgets.QWidget):
    """The concrete Derive contribution to the existing Logic editor."""

    draft_changed = QtCore.pyqtSignal(object)
    managed_fields = ("variables",)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._problems: dict[str, str] = {}
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
        self.inputs.setHorizontalSpacing(section_gap())
        self.inputs.setVerticalSpacing(row_gap())
        inputs.addLayout(self.inputs)
        inputs.addStretch(1)
        layout.addLayout(inputs)
        self._input_rows: list[tuple[QtWidgets.QLabel, ...]] = []

        self.variables_title, header = section_header("Variables")
        layout.addSpacing(section_gap() - row_gap())
        layout.addLayout(header)
        self.note = error_note_label("")
        layout.addWidget(self.note)
        self.empty = muted_note_label("No variables yet.")
        layout.addWidget(self.empty)
        self.lines = QtWidgets.QVBoxLayout()
        self.lines.setSpacing(row_gap())
        layout.addLayout(self.lines)
        self._lines: list[tuple[QtWidgets.QLabel, QtWidgets.QLabel, QtWidgets.QLabel]] = []
        self.footer = muted_note_label(_FOOTER)
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)

    def row_labels(self) -> tuple[str, ...]:
        """The labels this editor puts in the editor's label column."""

        return (self.inputs_title.text(),)

    def set_label_width(self, width: int) -> None:
        self.inputs_title.setFixedWidth(int(width))

    def shown_problems(self) -> dict[str, str]:
        """The draft's problems this editor states in red under the
        variables they are about: the page header counts them."""

        return dict(self._problems)

    def set_mutation_enabled(self, enabled: bool) -> None:
        del enabled

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
                for colour, domain_label, domain in zip(
                        AXIS_GROUP_COLORS, ("Repeat", "Point", "Cell"),
                        (schema.repeat_domain, schema.point_domain, schema.cell_domain)):
                    for axis in domain.axes:
                        if axis.coordinate_of is not None or axis.role.value == "scalar":
                            continue
                        groups.append(f'<span style="color: {colour}">{escape(axis.name)} {axis.size}</span>')
                        labels = axis.labels()
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
        bundle = tuple(projection.get("input_bundle") or ())
        self._show_inputs(bundle)
        try:
            rows = _ROWS.draft_values({"variables": values.get("variables") or ()})["variables"]
            draft_problem = ""
        except (TypeError, ValueError) as error:
            rows, draft_problem = (), str(error)
        if not draft_problem and rows and not any(row["publish"] for row in rows):
            draft_problem = NO_PUBLISHED
        schemas = {str(name): schema for name, schema, _snapshot in bundle}
        typing = schemas if bundle and all(schema is not None for schema in schemas.values()) else None
        answers = typed_variables(rows, typing)
        failure = error_stage(str(projection.get("error") or ""))
        show_problem(self.note, draft_problem[:1].upper() + draft_problem[1:])
        self._problems = {draft_problem: self.variables_title.text()} if draft_problem else {}
        while len(self._lines) > len(rows):
            for label in self._lines.pop():
                retire_widget(label)
        for index, (row, answer) in enumerate(zip(rows, answers)):
            if index == len(self._lines):
                line = (FluentLabel(""), muted_note_label(""), error_note_label(""))
                for label in line[1:]:
                    label.setContentsMargins(field_text_inset(), 0, 0, 0)
                    label.setWordWrap(True)
                for label in line:
                    self.lines.addWidget(label)
                self._lines.append(line)
            formula, result, problem = self._lines[index]
            _set_text(formula, _formula(row))
            error = answer["error"]
            text = (str(error.detail) if error is not None
                    else failure[2] if failure is not None and failure[0] == row["name"] else "")
            show_problem(problem, text)
            if error is not None and (answer["live"] or error.stage == "name"):
                self._problems[str(error)] = self.variables_title.text()
            notes = [] if answer["schema"] is None else [describe_schema(answer["schema"])]
            if answer["grouping"]:
                notes.append(answer["grouping"])
            if not row["publish"]:
                notes.append(f"internal · read by {', '.join(answer['read_by'])}" if answer["live"]
                             else "internal · read by nothing, so not evaluated")
            preview = (f"→ waits for {answer['waits_for']}" if answer["waits_for"]
                       else "→ checked when the source publishes" if typing is None
                       else "→ " + " · ".join(notes))
            _set_text(result, preview)
            result.setVisible(not text)
        self.empty.setVisible(not rows)


__all__ = ["DeriveEditor"]
