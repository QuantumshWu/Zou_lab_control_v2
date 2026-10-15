"""Lines that share named columns: one x for each kind of cell, down a section.

A section made of lines -- Derive's variable cards, the scan plan's axis rows --
reads as one table only when every line puts the same kind of cell at the
same x: every operand under every operand, every ``at`` under every ``at``,
every × on one x.  Packed line by line, each line laid its cells out from its
own left edge, so the same control stood at a different x on every line, and
a command slid sideways whenever the one beside it was hidden.

:class:`FluentColumnSet` names the columns once; a :class:`FluentColumnLine`
is one line of them.  A line holds one cell per column, in the set's order,
even where it has nothing to put there (the empty cell keeps the next
column's x), and :meth:`FluentColumnSet.refresh` fixes every cell of a column
to one width, decided by the column's kind:

``"fit"``
    the widest thing any line puts there;
``"stretch"``
    whatever the line has left over (one such column per set);
``"slots"``
    one slot per widget, every slot as wide as the widest widget, standing at
    the column's trailing edge -- repeated commands keep one x each, and the
    last one sits beside the line's ×;
``"glyph"``
    a square as tall as the controls (a line's ×).

A widget hidden with ``retainSizeWhenHidden`` keeps its place and its share
of the width, as Qt's own layouts keep it, so a used "+ Where" leaves its
slot empty instead of letting "+ Combine" slide into it; any other hidden
widget takes no room.  A line that is itself hidden (a closed stage) still
counts, so opening it moves nothing.  A cell may span to a later column
(:meth:`FluentColumnLine.place` ``to=``): a statistic strip starts at the
operand's x and runs on through the columns after it.  :meth:`set_width`
pins a column (an output name's width, an editor's label column), and
:meth:`follow` keeps a row under the lines -- an error, a result -- starting
at a column's x, so a long message adds lines below instead of squeezing
into what is left beside the controls.

:func:`section_header` is the line a section of such an editor opens with:
its title, the section's own control, and at the right end its summary and
the commands that add a row to it -- the same shape in every editor, so
"Add …" is found at one end of the page whichever editor is open.
"""

from __future__ import annotations

from collections.abc import Sequence

from PyQt5 import QtCore, QtWidgets, sip

from ..status import sentence
from .fluent import FluentLabel, FluentSectionLabel, control_height, row_gap, set_status_ink

#: What decides a column's width; see the module docstring.
COLUMN_KINDS = ("fit", "stretch", "slots", "glyph")


def section_header(
    title: str,
    *,
    lead: Sequence[QtWidgets.QWidget] = (),
    summary: QtWidgets.QWidget | None = None,
    actions: Sequence[QtWidgets.QWidget] = (),
) -> tuple[FluentSectionLabel, QtWidgets.QHBoxLayout]:
    """THE header line of a section inside an editor, and its title.

    The title comes first; ``lead`` -- the section's own control, such as a
    preset strip -- follows it, on the form's field edge once the owner holds
    the title to the editor's label column.  The right end belongs to the
    section's ``summary`` and then its ``actions``, the commands that add a
    row to the section, so the last command ends on the page's right edge
    in every editor.  The title is centred on what stands beside it; the
    section gap above the line is the gap of the layout it stands in.
    """

    label = FluentSectionLabel(title)
    line = QtWidgets.QHBoxLayout()
    line.setContentsMargins(0, 0, 0, 0)
    line.setSpacing(row_gap())
    line.addWidget(label)
    for widget in lead:
        line.addWidget(widget)
    line.addStretch(1)
    if summary is not None:
        line.addWidget(summary)
    for widget in actions:
        line.addWidget(widget)
    return label, line


def error_note_label(text: str = "") -> QtWidgets.QLabel:
    """The one error line: the ``error`` status ink, word-wrapped, hidden
    while it says nothing.  It stands under what it concerns (see
    :meth:`FluentColumnSet.follow`), never in the width left beside it, and
    says it as a sentence (:func:`show_problem`)."""

    label = FluentLabel(text)
    set_status_ink(label, "error")
    label.setWordWrap(True)
    label.setVisible(bool(text))
    return label


def show_note(label: QtWidgets.QLabel, text: str) -> None:
    """Write a note line and show it only while it says something: an empty
    note still took a text line and a gap, and opened a hole in the page."""

    if label.text() != text:
        label.setText(text)
    label.setVisible(bool(text))


def show_problem(label: QtWidgets.QLabel, text: str) -> None:
    """Write a problem into its :func:`error_note_label` as one sentence
    (:func:`zlc_ui.status.sentence`: a full stop at its end, never a capital
    added at its start), shown only while there is one.  Every editor's
    problem under its row is written through here, so none reads as a
    fragment beside another that ends in a full stop."""

    show_note(label, sentence(text))


def hold_place(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Keep ``widget``'s place in its line while it is hidden, so what
    stands after it does not move when it goes (a used "+ Where", an ``at``
    with no frames to offer)."""

    policy = widget.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    widget.setSizePolicy(policy)
    return widget


def _counts(widget: QtWidgets.QWidget) -> bool:
    """Whether a widget takes its room in a line: shown, or hidden and holding its place."""

    return not widget.isHidden() or widget.sizePolicy().retainSizeWhenHidden()


class FluentColumnSet(QtCore.QObject):
    """Named columns shared by every :class:`FluentColumnLine` built on it.

    ``columns`` is ``(name, kind)`` in order, kinds from :data:`COLUMN_KINDS`.
    The owner calls :meth:`refresh` after it changed what its lines hold
    (rows built or refilled, a picker mounted) and :meth:`forget` before it
    retires a line, so a line on its way out no longer widens a column.
    """

    def __init__(self, columns: Sequence[tuple[str, str]], parent=None) -> None:
        super().__init__(parent)
        self.names = tuple(str(name) for name, _kind in columns)
        self._kinds = {str(name): str(kind) for name, kind in columns}
        unknown = set(self._kinds.values()) - set(COLUMN_KINDS)
        if unknown:
            raise ValueError(f"unknown column kinds {sorted(unknown)!r}")
        self.gap = row_gap()
        self._pinned: dict[str, int] = {}
        self._lines: list[FluentColumnLine] = []
        self._followers: list[tuple[QtWidgets.QWidget, str, int]] = []
        self._widths: dict[str, int] = {}
        self._slots: dict[str, int] = {}
        self._x: dict[str, int] = {}

    def kind(self, column: str) -> str:
        return self._kinds[column]

    def set_width(self, column: str, width: int | None) -> None:
        """Pin ``column`` to ``width`` (None measures it again)."""

        if width is None:
            self._pinned.pop(column, None)
        else:
            self._pinned[column] = int(width)
        self.refresh()

    def column_x(self, column: str) -> int:
        """Where ``column`` starts, from the lines' left edge -- for a column
        at or before the stretch column, whose x does not depend on the
        line's width."""

        return self._x[column]

    def follow(self, widget: QtWidgets.QWidget, column: str, *, inset: int = 0) -> None:
        """Keep ``widget`` -- a row laid under the lines, sharing their left
        edge -- starting at ``column`` (plus ``inset``) through its left
        contents margin, on every refresh."""

        self._followers.append((widget, column, int(inset)))
        if column in self._x:
            self._place_follower(widget, column, int(inset))

    def forget(self, widget: QtWidgets.QWidget) -> None:
        """Stop measuring the lines and followers that are, or are inside, ``widget``."""

        def kept(item: QtWidgets.QWidget) -> bool:
            return not sip.isdeleted(item) and item is not widget and not widget.isAncestorOf(item)

        self._lines = [line for line in self._lines if kept(line)]
        self._followers = [entry for entry in self._followers if kept(entry[0])]

    def refresh(self) -> None:
        """Measure every column over every line, then fix every cell to it."""

        self._lines = [line for line in self._lines if not sip.isdeleted(line)]
        self._followers = [entry for entry in self._followers if not sip.isdeleted(entry[0])]
        widths: dict[str, int] = {}
        slots: dict[str, int] = {}
        for name in self.names:
            kind = self._kinds[name]
            if kind == "stretch":
                continue
            if kind in ("slots", "glyph"):
                needs = [line._slot_need(name, kind) for line in self._lines]
                count = max((count for count, _width in needs), default=0)
                slot = max((width for _count, width in needs), default=0)
                slots[name] = slot
                measured = count * slot + self.gap * max(0, count - 1) if slot else 0
            else:
                measured = max((line._need(name) for line in self._lines), default=0)
            widths[name] = self._pinned.get(name, measured)
        self._widths, self._slots = widths, slots
        self._x = {}
        x = 0
        for name in self.names:
            self._x[name] = x
            if self._kinds[name] == "stretch":
                break
            if widths[name] > 0:
                x += widths[name] + self.gap
        for line in self._lines:
            line._apply()
        for widget, column, inset in self._followers:
            self._place_follower(widget, column, inset)

    def _place_follower(self, widget: QtWidgets.QWidget, column: str, inset: int) -> None:
        left = self._x[column] + inset
        margins = widget.contentsMargins()
        if margins.left() != left:
            widget.setContentsMargins(left, margins.top(), margins.right(), margins.bottom())


class FluentColumnLine(QtWidgets.QWidget):
    """One line of a :class:`FluentColumnSet`: a cell per column, in the set's order."""

    def __init__(self, columns: FluentColumnSet, parent=None) -> None:
        super().__init__(parent)
        self._columns = columns
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(columns.gap)
        self._cells: dict[str, QtWidgets.QWidget] = {}
        self._held: dict[str, list[QtWidgets.QWidget]] = {name: [] for name in columns.names}
        self._fills: set[int] = set()
        #: A spanning cell's first column -> its last; the columns it covers.
        self._spans: dict[str, str] = {}
        self._covered: set[str] = set()
        for name in columns.names:
            cell = QtWidgets.QWidget(self)
            box = QtWidgets.QHBoxLayout(cell)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(columns.gap)
            # Commands stand at a slot column's trailing edge; everything
            # else starts at its column's x and leaves the rest empty.
            box.addStretch(1)
            layout.addWidget(cell)
            self._cells[name] = cell
        # The room a line without a stretch column has over, after its cells.
        layout.addStretch(0)
        columns._lines.append(self)

    def place(self, column: str, *widgets: QtWidgets.QWidget, to: str | None = None,
              fill: bool = False) -> None:
        """Put ``widgets`` in ``column``, in order.

        ``fill`` widgets share the cell's width between them (a combo as wide
        as every operand combo) and ask for only their minimum when the
        column is measured; others keep their natural width.  ``to`` makes
        the cell run on through that later column, which this line then
        leaves out; a spanning cell takes no part in measuring.
        """

        cell = self._cells[column]
        box = cell.layout()
        if to is not None:
            names = self._columns.names
            self._spans[column] = to
            self._covered.update(names[names.index(column) + 1:names.index(to) + 1])
        slots = self._columns.kind(column) in ("slots", "glyph")
        for widget in widgets:
            if slots:
                box.addWidget(widget)
            else:
                box.insertWidget(box.count() - 1, widget, 1 if fill else 0)
            if fill:
                self._fills.add(id(widget))
            self._held[column].append(widget)
        if fill and not slots:
            box.setStretch(box.count() - 1, 0)

    def stand_on_first_row(self) -> None:
        """Stand every cell on the line's first row.  A cell whose commands
        run on to a second row when the line is narrow grows downward,
        while the word and the × beside it stay level with that first row
        instead of centring on the whole height."""

        height = control_height()
        layout = self.layout()
        for cell in self._cells.values():
            cell.setMinimumHeight(height)
            layout.setAlignment(cell, QtCore.Qt.AlignTop)

    def _natural(self, widget: QtWidgets.QWidget) -> int:
        hint = widget.minimumSizeHint() if id(widget) in self._fills else widget.sizeHint()
        width = max(hint.width(), widget.minimumWidth(), 0)
        return min(width, widget.maximumWidth())

    def _need(self, column: str) -> int:
        if column in self._covered or column in self._spans:
            return 0
        sizes = [self._natural(widget) for widget in self._held[column] if _counts(widget)]
        return sum(sizes) + self._columns.gap * max(0, len(sizes) - 1)

    def _slot_need(self, column: str, kind: str) -> tuple[int, int]:
        widgets = [widget for widget in self._held[column] if _counts(widget)]
        if kind == "glyph":
            return len(widgets), max((widget.minimumHeight() for widget in widgets), default=0)
        return len(widgets), max((widget.sizeHint().width() for widget in widgets), default=0)

    def _apply(self) -> None:
        columns = self._columns
        names = columns.names
        layout = self.layout()
        for index, name in enumerate(names):
            cell = self._cells[name]
            if name in self._covered:
                cell.setVisible(False)
                continue
            span = names[index:names.index(self._spans.get(name, name)) + 1]
            stretches = any(columns.kind(item) == "stretch" for item in span)
            if stretches:
                if cell.minimumWidth() != 0 or cell.maximumWidth() != QtWidgets.QWIDGETSIZE_MAX:
                    cell.setMinimumWidth(0)
                    cell.setMaximumWidth(QtWidgets.QWIDGETSIZE_MAX)
                width = None
            else:
                shown = [columns._widths[item] for item in span if columns._widths[item] > 0]
                width = sum(shown) + columns.gap * max(0, len(shown) - 1)
                if cell.minimumWidth() != width or cell.maximumWidth() != width:
                    cell.setFixedWidth(width)
            # An empty column takes no room and no gap on any line.
            cell.setVisible(width is None or width > 0 or name in self._spans)
            stretch = 1 if stretches else 0
            if layout.stretch(index) != stretch:
                layout.setStretch(index, stretch)
            if columns.kind(name) in ("slots", "glyph"):
                slot = columns._slots[name]
                for widget in self._held[name]:
                    if slot and (widget.minimumWidth() != slot or widget.maximumWidth() != slot):
                        widget.setFixedWidth(slot)


__all__ = [
    "COLUMN_KINDS",
    "FluentColumnLine",
    "FluentColumnSet",
    "error_note_label",
    "hold_place",
    "section_header",
    "show_note",
    "show_problem",
]
