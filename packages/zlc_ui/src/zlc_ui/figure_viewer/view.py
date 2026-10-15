"""Pure saved-figure browser shell.

The presenter owns archive IO, metadata projection and the plot widget.  This
view only provides the file/path intent, generic info projection, and an
atomic QWidget mount point for the presenter-owned surface.
"""

from __future__ import annotations

import math

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.console.board_view import ConsoleBoardView
from zlc_ui.console.panel_card_view import (
    PanelCardView,
    _escaped,
    data_structure_fragments,
    new_panel_card,
)
from zlc_ui.console.panel_editor_view import (
    PanelEditorView,
    new_panel_editor,
    panel_editor_projection,
)
from zlc_ui.console.task_console_view import plot_choice_text
from zlc_ui.form.qt_form import being_edited
from zlc_ui.fluent import (
    retire_widget,
    card_pad,
    control_height,
    row_gap,
    section_gap,
    ElidedLabel,
    FluentPageBody,
    ORANGE_TINT,
    FluentButton,
    FluentComboBox,
    FluentCycleComboBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSegmentedButtons,
    FluentSettingRow,
    FluentSpinBox,
    FluentSwitch,
    FluentTableView,
    FluentTabWidget,
    InfoPane,
    add_row_switch,
    card_title_row,
    fluent_save_path,
    fluent_scrollbar_thickness,
    muted_note_label,
    muted_summary_label,
    setting_label_width,
    signals_blocked,
    window_pad,
)
from zlc_ui.status import sentence


#: The editor's fixed row labels.  Its label column is the widest of these
#: and of the axis rows' labels, which carry each axis's size and unit.
_STATIC_LABELS = (
    "Name", "Type", "Value unit", "Note", "Editing", "Length", "Unit", "Domain", "Table shows",
)

#: The fewest rows a value table shows, in a window too short for its card:
#: the page then scrolls to the rest of the card.
_LEAST_TABLE_ROWS = 3

#: The data editor's two pages: the Dataset and Axes cards, and the Data
#: card.  Each page is what its one value table scrolls in.
_EDITOR_PAGES = (("dataset", "Dataset and axes"), ("data", "Data"))


def _surface_right_inset() -> int:
    """How far in from the right surface's right edge its rows end.

    One card pad, as on its left -- or the bar gutter a scrolled page keeps,
    where that is the wider: the data editor's bar lies in the surface's
    padding, so its cards and its action bar end where the data bar over the
    tabs and the board's commands end."""

    return max(card_pad(), fluent_scrollbar_thickness())


def _two_column_grid() -> QtWidgets.QGridLayout:
    """Two equal columns of setting rows, a section gap apart: at a row's
    gap the second column's label read as a word trailing the first
    column's field."""

    grid = QtWidgets.QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(section_gap())
    grid.setVerticalSpacing(row_gap())
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    return grid


def _plain_scalar(value: object) -> str:
    """Format one already-projected table scalar without interpreting it.

    A numpy scalar is written by its own type: ``str(np.float32(2.6749))`` is
    ``2.6749``, the shortest text that reads back as that float32.  Widening
    it first with ``.item()`` printed the double ``2.6749000549316406`` --
    digits nobody stored -- and the column elided all of them.
    """

    return "" if value is None else str(value)


def _matrix_item(values: object, row: int, column: int) -> object:
    """Read a two-dimensional plain/array projection lazily."""

    try:
        return values[row, column]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        row_value = values[row]  # type: ignore[index]
        if isinstance(row_value, (str, bytes)):
            return row_value if column == 0 else ""
        try:
            return row_value[column]
        except (IndexError, KeyError, TypeError):
            return row_value if column == 0 else ""


class _VirtualTextTableModel(QtCore.QAbstractTableModel):
    """Visible-cell-only editable view over a presenter-owned 2-D projection."""

    edits_requested = QtCore.pyqtSignal(object)
    rejected = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._shape = (0, 0)
        self._values: object = ()
        self._validity: object | None = None
        self._row_headers: object = ()
        self._column_headers: object = ()
        self._editable = True
        self._finite_values = False
        self._blank_hint = ""
        self._pending: dict[tuple[int, int], str] = {}

    def set_projection(self, projection: object) -> None:
        data = dict(projection or {})
        shape = tuple(int(value) for value in tuple(data.get("shape", (0, 0))))
        if len(shape) != 2 or any(value < 0 for value in shape):
            raise ValueError("table projection shape must contain two nonnegative sizes")
        reset = shape != self._shape
        row_headers = data.get("row_headers", ())
        column_headers = data.get("column_headers", ())
        row_headers_changed = row_headers != self._row_headers
        column_headers_changed = column_headers != self._column_headers
        if reset:
            self.beginResetModel()
        self._shape = shape
        self._values = data.get("values", ())
        self._validity = data.get("validity")
        self._row_headers = row_headers
        self._column_headers = column_headers
        self._editable = bool(data.get("editable", True))
        self._finite_values = bool(data.get("finite_values", False))
        self._blank_hint = str(data.get("blank_hint", ""))
        self._pending.clear()
        if reset:
            self.endResetModel()
            return
        changed = data.get("changed_cells")
        if shape[0] and shape[1] and (changed is None or changed):
            first = (0, 0) if changed is None else (
                min(row for row, _column in changed), min(column for _row, column in changed)
            )
            last = (shape[0] - 1, shape[1] - 1) if changed is None else (
                max(row for row, _column in changed), max(column for _row, column in changed)
            )
            self.dataChanged.emit(
                self.index(*first),
                self.index(*last),
                (
                    QtCore.Qt.DisplayRole,
                    QtCore.Qt.EditRole,
                    QtCore.Qt.BackgroundRole,
                    QtCore.Qt.ToolTipRole,
                ),
            )
        if column_headers_changed:
            self.headerDataChanged.emit(QtCore.Qt.Horizontal, 0, max(0, shape[1] - 1))
        if row_headers_changed:
            self.headerDataChanged.emit(QtCore.Qt.Vertical, 0, max(0, shape[0] - 1))

    def rowCount(self, _parent=QtCore.QModelIndex()) -> int:  # noqa: N802
        return self._shape[0]

    def columnCount(self, _parent=QtCore.QModelIndex()) -> int:  # noqa: N802
        return self._shape[1]

    def _projected_text(self, row: int, column: int) -> str:
        pending = self._pending.get((row, column))
        if pending is not None:
            return pending
        if self._validity is not None and not bool(
            _matrix_item(self._validity, row, column)
        ):
            return ""
        value = _matrix_item(self._values, row, column)
        if self._finite_values:
            try:
                if not math.isfinite(float(value)):
                    return ""
            except (TypeError, ValueError):
                return ""
        return _plain_scalar(value)

    def data(self, index, role=QtCore.Qt.DisplayRole):  # noqa: A003
        if not index.isValid():
            return None
        row, column = index.row(), index.column()
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            return self._projected_text(row, column)
        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        if role == QtCore.Qt.BackgroundRole and (row, column) in self._pending:
            return QtGui.QBrush(QtGui.QColor(ORANGE_TINT))
        if role == QtCore.Qt.ToolTipRole:
            # The whole stored value, for a column narrowed by hand.
            return self._projected_text(row, column) or self._blank_hint or None
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):  # noqa: N802
        if role not in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole):
            return None
        labels = self._column_headers if orientation == QtCore.Qt.Horizontal else self._row_headers
        try:
            return _plain_scalar(labels[section])  # type: ignore[index]
        except (IndexError, KeyError, TypeError):
            return str(int(section))

    def flags(self, index):
        flags = super().flags(index)
        return flags | QtCore.Qt.ItemIsEditable if self._editable else flags

    def setData(self, index, value, role=QtCore.Qt.EditRole):  # noqa: N802
        if role != QtCore.Qt.EditRole or not index.isValid() or not self._editable:
            return False
        if str(value) == self._projected_text(index.row(), index.column()):
            self._pending.pop((index.row(), index.column()), None)
            return True
        self.set_block_data(index.row(), index.column(), ((str(value),),))
        return True

    def set_block_data(self, row: int, column: int, values: object) -> None:
        rows = tuple(tuple(fields) for fields in tuple(values))
        first_row = int(row)
        first_column = int(column)
        if (
            first_row < 0
            or first_column < 0
            or first_row + len(rows) > self._shape[0]
            or any(first_column + len(fields) > self._shape[1] for fields in rows)
        ):
            self.rejected.emit(
                "Paste exceeds the current table; resize axes first"
            )
            return
        edits = []
        for row_offset, fields in enumerate(rows):
            for column_offset, value in enumerate(fields):
                target = (first_row + row_offset, first_column + column_offset)
                text = str(value)
                self._pending[target] = text
                edits.append((target[0], target[1], text))
        self._publish_edits(edits)

    def clear_indices(self, indices: object) -> None:
        edits = []
        for row, column in tuple(indices):
            target = (int(row), int(column))
            if 0 <= target[0] < self._shape[0] and 0 <= target[1] < self._shape[1]:
                self._pending[target] = ""
                edits.append((target[0], target[1], ""))
        self._publish_edits(edits)

    def _publish_edits(self, edits: object) -> None:
        cells = tuple(edits)
        if not cells:
            return
        rows = tuple(value[0] for value in cells)
        columns = tuple(value[1] for value in cells)
        self.dataChanged.emit(
            self.index(min(rows), min(columns)),
            self.index(max(rows), max(columns)),
            (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.BackgroundRole),
        )
        self.edits_requested.emit(cells)


def _fill_choice_combo(combo: FluentComboBox, rows: object, current: object) -> None:
    desired = []
    for row in tuple(rows or ()):
        values = tuple(row)
        if len(values) not in (2, 3):
            raise ValueError("choice rows must be (key, label[, enabled])")
        key, label = values[:2]
        desired.append((key, str(label), bool(values[2]) if len(values) == 3 else True))
    with signals_blocked(combo):
        for index, (key, label, enabled) in enumerate(desired):
            if index >= combo.count():
                combo.addItem(label, key)
            item = combo.model().item(index)
            if item.text() != label:
                item.setText(label)
            previous = item.data(QtCore.Qt.UserRole)
            if type(previous) is not type(key) or previous != key:
                item.setData(key, QtCore.Qt.UserRole)
            if item.isEnabled() != enabled:
                item.setEnabled(enabled)
        if combo.count() > len(desired):
            combo.model().removeRows(len(desired), combo.count() - len(desired))
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else (-1 if not combo.count() else 0))


class _DataEditorView(QtWidgets.QWidget):
    """Fluent working-copy editor; all Dataset meaning remains in its presenter.

    Two pages, chosen at the editor's top: the Dataset and Axes cards, with
    the selected axis's values, and the Data card, with the value table.
    Each page holds one value table and the table is what scrolls: it is
    sized to the room its page leaves (_fit_table_to_page), so the page
    itself has nothing to scroll -- a page and its table each scrolling
    stood two vertical bars side by side, the wheel moving the one under
    the pointer.  Only a window too short for a page's controls and the
    fewest table rows scrolls the page as well.  The working copy's own
    commands and their messages stand under both pages.
    """

    intent_requested = QtCore.pyqtSignal(object)

    def __init__(self, projection: object, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._updating = False
        self._selected_axis = ""
        self._adding_axis = False
        #: The axis draft's domain and the component the table shows: what
        #: the two segmented choices mark, which report clicks and hold no
        #: value to read back.
        self._domain: object = None
        self._component: object = None
        self._save_suggested = "figure.npz"
        self._axis_view_widgets: dict[
            str,
            tuple[
                QtWidgets.QWidget,
                QtWidgets.QLabel,
                FluentCycleComboBox,
            ],
        ] = {}

        #: Every labelled row of the editor and its label: the three cards
        #: share ONE label column (see _set_label_width).
        self._label_rows: list[tuple[FluentSettingRow, str]] = []
        self._label_width = setting_label_width(_STATIC_LABELS, minimum=68)

        gutter = fluent_scrollbar_thickness()
        pad = card_pad()
        outer = QtWidgets.QVBoxLayout(self)
        # The page is as wide as the surface; its rows stand one card pad in
        # on the left, and on the right the surface's inset less the gutter
        # the scroll area keeps for its bar, so the bar lies in the surface's
        # padding and the cards end where the data bar over the tabs ends.
        # One card pad under the tab strip, as every tab page of the window.
        outer.setContentsMargins(pad, pad, _surface_right_inset() - gutter, 0)
        outer.setSpacing(0)
        # Which page: a closed choice heading the editor where the Monitor
        # page has its commands, its first card one card pad under it as the
        # board's first card is under those -- switching tabs moves nothing.
        self._page = _EDITOR_PAGES[0][0]
        self.page_choice = FluentSegmentedButtons(_EDITOR_PAGES)
        outer.addWidget(self.page_choice, 0, QtCore.Qt.AlignLeft)
        outer.addSpacing(pad)
        self.editor_scroll = FluentScrollArea(self)
        self.editor_scroll.setMinimumSize(0, 0)
        # The same opaque, surface-white page body the console's Edit page
        # scrolls: transparent content showed the viewport's own grey.
        content = FluentPageBody()
        root = QtWidgets.QVBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(row_gap())
        # The gutter is held whether or not the bar shows: the cards and the
        # action bar end on one edge whichever page is shown, and a window
        # short enough to scroll a page does not narrow every card.
        self.editor_scroll.set_width_bounded_widget(content)
        outer.addWidget(self.editor_scroll, 1)

        dataset_group = FluentGroupBox("Dataset", self, title_strip_px=0)
        self.dataset_group = dataset_group
        dataset_layout = QtWidgets.QVBoxLayout(dataset_group)
        dataset_layout.setContentsMargins(pad, pad, pad, pad)
        dataset_layout.setSpacing(row_gap())
        # Where this working copy came from, on the card's title line.
        self.source_note = muted_summary_label()
        dataset_layout.addLayout(card_title_row("Dataset", summary=self.source_note)[0])
        self.name_edit = FluentLineEdit()
        self.dtype_combo = FluentComboBox()
        self.dtype_combo.setMinimumContentsLength(9)
        self.unit_edit = FluentLineEdit()
        self.note_edit = FluentLineEdit()
        self.note_edit.setPlaceholderText("Optional reason for this manual edit")
        dataset_grid = _two_column_grid()
        dataset_grid.addWidget(self._labelled_row("Name", self.name_edit), 0, 0)
        dataset_grid.addWidget(self._labelled_row("Type", self.dtype_combo), 0, 1)
        dataset_grid.addWidget(self._labelled_row("Value unit", self.unit_edit), 1, 0)
        dataset_grid.addWidget(self._labelled_row("Note", self.note_edit), 1, 1)
        dataset_layout.addLayout(dataset_grid)
        root.addWidget(dataset_group)

        axes_group = FluentGroupBox("Axes", content, title_strip_px=0)
        self.axes_group = axes_group
        axes_layout = QtWidgets.QVBoxLayout(axes_group)
        axes_layout.setContentsMargins(pad, pad, pad, pad)
        axes_layout.setSpacing(row_gap())
        # Creating an axis is the card's own command, on its title line, with
        # what the fields below are doing -- editing the chosen axis or
        # drafting a new one -- said beside it.  Choosing which axis to edit
        # is a different sentence: the labelled Editing row, whose Apply and
        # Delete act on the axis it holds.
        self.add_axis_button = FluentButton("Add axis")
        self.add_axis_button.setToolTip("Create a new axis")
        self.axis_mode_label = muted_summary_label("Edit selected axis")
        axes_layout.addLayout(
            card_title_row("Axes", self.add_axis_button, summary=self.axis_mode_label)[0]
        )
        self.axis_combo = FluentComboBox()
        self.axis_combo.setMinimumContentsLength(13)
        self.apply_axis_button = FluentButton("Apply axis")
        self.remove_axis_button = FluentButton("Delete", role="destructive")
        self.remove_axis_button.setToolTip(
            "Delete selected axis and keep its current Scope slice"
        )
        self.axis_choice_row = self._labelled_row(
            "Editing",
            self.axis_combo,
            after=((self.apply_axis_button, 0), (self.remove_axis_button, 0)),
        )
        axes_layout.addWidget(self.axis_choice_row)
        # The presenter's closed sets -- the three domains here, the
        # components below -- are each one segmented choice, built with the
        # projection the editor opens on and only marked after that.
        opening = dict(projection or {})
        domains = tuple(
            (key, str(label)) for key, label in tuple(opening.get("domain_choices", ()))
        )
        self._domain_keys = tuple(key for key, _label in domains)
        self.domain_choice = FluentSegmentedButtons(domains)
        self.axis_size_spin = FluentSpinBox()
        self.axis_size_spin.setRange(1, 2_147_483_647)
        self.axis_size_spin.setKeyboardTracking(False)
        self.axis_name_edit = FluentLineEdit()
        self.axis_unit_edit = FluentLineEdit()
        self._axis_rows: dict[str, FluentSettingRow] = {}
        axis_form = _two_column_grid()
        for field, label, control, row_index, column_index in (
            ("name", "Name", self.axis_name_edit, 0, 0),
            ("size", "Length", self.axis_size_spin, 0, 1),
            ("unit", "Unit", self.axis_unit_edit, 1, 0),
            ("domain", "Domain", self.domain_choice, 1, 1),
        ):
            row = self._labelled_row(label, control)
            self._axis_rows[field] = row
            axis_form.addWidget(row, row_index, column_index)
        # The values are a block of their own under a heading, a section gap
        # below the fields, and as wide as the first column of the grid:
        # one axis is one column of values, read down by index.  The heading
        # keeps a whole control-high cell at the foot of its row, the row
        # taller by what a section gap has over the grid's row gap: a margin
        # inside the heading would come out of its cell and set its words,
        # and the table under them, too high.
        values_heading = FluentSectionLabel("Axis values")
        axis_form.addWidget(values_heading, 2, 0, QtCore.Qt.AlignBottom)
        axis_form.setRowMinimumHeight(2, control_height() + section_gap() - row_gap())
        self.axis_value_model = _VirtualTextTableModel(self)
        self.axis_value_table = FluentTableView(self)
        self.axis_value_table.setModel(self.axis_value_model)
        self.axis_value_table.setTabKeyNavigation(True)
        self.axis_value_table.horizontalHeader().setDefaultAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )
        self.axis_value_table.horizontalHeader().setStretchLastSection(True)
        self.axis_value_table.set_value_columns()
        self.axis_value_table.set_content_rows_limit(12)
        axis_form.addWidget(self.axis_value_table, 3, 0)
        axes_layout.addLayout(axis_form)
        root.addWidget(axes_group)

        data_group = FluentGroupBox("Data", content, title_strip_px=0)
        self.data_group = data_group
        data_layout = QtWidgets.QVBoxLayout(data_group)
        data_layout.setContentsMargins(pad, pad, pad, pad)
        data_layout.setSpacing(row_gap())
        data_layout.addLayout(card_title_row("Data")[0])
        self.structure_label = FluentLabel("")
        self.structure_label.setTextFormat(QtCore.Qt.RichText)
        self.structure_label.setWordWrap(True)
        data_layout.addWidget(self.structure_label)
        self._axis_view_holder = QtWidgets.QWidget(data_group)
        self._axis_view_holder.setStyleSheet("background: transparent;")
        self._axis_view_layout = QtWidgets.QGridLayout(self._axis_view_holder)
        self._axis_view_layout.setContentsMargins(0, 0, 0, 0)
        self._axis_view_layout.setVerticalSpacing(row_gap())
        data_layout.addWidget(self._axis_view_holder)
        # One sentence, one row: what the table below is showing, and the
        # switch that decides whether the Dataset stores a sigma at all --
        # named by that, not by the component it offers: a segment and a
        # switch both called Sigma on one row were one word for two things.
        # A setting row is a width CONSUMER, so the pair travels inside its
        # control slot rather than beside it in a box that would collapse it.
        self.component_choice = FluentSegmentedButtons(
            tuple(
                (row[0], str(row[1]))
                for row in tuple(dict(opening.get("table", {})).get("component_choices", ()))
            )
        )
        self.sigma_switch = FluentSwitch("Store sigma")
        component_controls = QtWidgets.QWidget(data_group)
        component_controls.setStyleSheet("background: transparent;")
        component_layout = QtWidgets.QHBoxLayout(component_controls)
        component_layout.setContentsMargins(0, 0, 0, 0)
        component_layout.setSpacing(row_gap())
        component_layout.addWidget(self.component_choice)
        add_row_switch(component_layout, self.sigma_switch, trail=False)
        component_layout.addStretch(1)
        data_layout.addWidget(self._labelled_row("Table shows", component_controls))
        self.blank_help_label = muted_note_label("")
        data_layout.addWidget(self.blank_help_label)
        self.value_model = _VirtualTextTableModel(self)
        self.value_table = FluentTableView(self)
        self.value_table.setModel(self.value_model)
        self.value_table.setTabKeyNavigation(True)
        self.value_table.horizontalHeader().setDefaultAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )
        self.value_table.horizontalHeader().setStretchLastSection(True)
        self.value_table.set_value_columns()
        self.value_table.set_content_rows_limit(12)
        data_layout.addWidget(self.value_table)
        root.addWidget(data_group)
        # Under a page's cards, what the window has to spare: a table that
        # is whole ends its card, not the page.
        root.addStretch(1)

        # Apply, save and discard act on the whole working copy, so they
        # close the editor under both pages: Discard at the left edge, the
        # two steps that keep the edit at the right, in the order they are
        # taken, ending where the cards end, a section gap under them.  The
        # same rule as every other Save, in two steps: while the fields hold
        # edits not yet applied, Apply & preview is the primary (with its
        # star); once applied and not yet saved, Save as… is.  What the
        # working copy last said -- an axis refused, a file saved -- is the
        # line over these commands, read from either page, and shown only
        # while it has something to say.
        self.message_label = muted_note_label("")
        self.message_label.setWordWrap(True)
        self.message_label.setVisible(False)
        self.apply_button = FluentButton("Apply && preview")
        self.save_button = FluentButton("Save as…")
        self.discard_button = FluentButton("Discard", role="destructive")
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(row_gap())
        actions.addWidget(self.discard_button)
        actions.addStretch(1)
        actions.addWidget(self.apply_button)
        actions.addWidget(self.save_button)
        foot = QtWidgets.QVBoxLayout()
        foot.setContentsMargins(0, section_gap(), gutter, 0)
        foot.setSpacing(row_gap())
        foot.addWidget(self.message_label)
        foot.addLayout(actions)
        outer.addLayout(foot)

        #: Each page's value table and the card it closes.
        self._page_tables = {
            "dataset": (self.axis_value_table, axes_group),
            "data": (self.value_table, data_group),
        }
        self._table_rows: dict[FluentTableView, int] = {}
        # After the layouts the change set off have settled.
        self._table_fit = QtCore.QTimer(self)
        self._table_fit.setSingleShot(True)
        self._table_fit.setInterval(0)
        self._table_fit.timeout.connect(self._fit_table_to_page)
        # A horizontal bar that comes or goes -- columns widened to their
        # measured values, a window narrowed -- is a row the page loses or
        # gains.
        for table, _card in self._page_tables.values():
            table.horizontalScrollBar().rangeChanged.connect(
                lambda _low, _high: self._table_fit.start()
            )

        self.name_edit.editingFinished.connect(lambda: self._dataset_field("name", self.name_edit.text()))
        self.dtype_combo.activated.connect(lambda _index: self._dataset_field("dtype", self.dtype_combo.currentData()))
        self.unit_edit.editingFinished.connect(lambda: self._dataset_field("unit", self.unit_edit.text()))
        self.note_edit.editingFinished.connect(lambda: self._dataset_field("note", self.note_edit.text()))
        self.axis_combo.activated.connect(self._select_axis)
        self.add_axis_button.clicked.connect(self._begin_add_axis)
        self.remove_axis_button.clicked.connect(
            lambda: self._emit("delete_axis", axis_id=self._selected_axis)
        )
        self.apply_axis_button.clicked.connect(self._commit_axis)
        self.axis_value_model.edits_requested.connect(
            lambda cells: self._emit(
                "set_axis_values", axis_id=self._selected_axis, cells=cells
            )
        )
        self.axis_value_model.rejected.connect(self._show_local_rejection)
        self.domain_choice.activated.connect(self._show_domain)
        self.component_choice.activated.connect(self._component_picked)
        self.sigma_switch.toggled.connect(lambda enabled: self._emit("toggle_sigma", enabled=bool(enabled)))
        self.value_model.edits_requested.connect(
            lambda cells: self._emit("set_cells", component=self._component, cells=cells)
        )
        self.value_model.rejected.connect(self._show_local_rejection)
        self.apply_button.clicked.connect(lambda: self._emit("apply_preview", note=self.note_edit.text()))
        self.save_button.clicked.connect(self._save_as)
        self.discard_button.clicked.connect(lambda: self._emit("discard"))
        self.page_choice.activated.connect(self._show_page)
        self._show_page(self._page)
        self.update_projection(projection)

    def _labelled_row(
        self,
        label: str,
        control: QtWidgets.QWidget,
        *,
        after: tuple[tuple[QtWidgets.QWidget, int], ...] = (),
    ) -> FluentSettingRow:
        """A setting row on the editor's one label column."""

        row = FluentSettingRow(label, control, label_width=self._label_width, after=after)
        self._label_rows.append((row, label))
        return row

    def _set_label_width(self, axis_labels: tuple[str, ...]) -> None:
        """One label column for the three cards: wide enough for every fixed
        label and every axis row's name, size and unit, so no axis row cuts
        off the size its suffix is there to show.  Capped at a third of a
        typical card, past which an axis name elides into its tooltip."""

        width = min(
            setting_label_width(_STATIC_LABELS + axis_labels, minimum=68),
            max(setting_label_width(_STATIC_LABELS, minimum=68), window_pad(16)),
        )
        if width == self._label_width:
            return
        self._label_width = width
        for row, label in self._label_rows:
            row.set_label(label, width=width)
        for holder, label, _mode in self._axis_view_widgets.values():
            holder.set_label(label.text(), width=width)

    def _show_page(self, page: object) -> None:
        """Show one page's cards, from their top."""

        self._page = str(page)
        self.page_choice.set_current(self._page)
        on_data = self._page == "data"
        self.dataset_group.setVisible(not on_data)
        self.axes_group.setVisible(not on_data)
        self.data_group.setVisible(on_data)
        self.editor_scroll.verticalScrollBar().setValue(0)
        self._table_fit.start()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._table_fit.start()

    def _fit_table_to_page(self) -> None:
        """Let the shown page's value table be as tall as the page can show
        it whole.

        Sized to the room the viewport leaves under the cards above it and
        its card's foot, a table is whole when its rows fit -- its card
        ends under its last row -- and otherwise a viewport of exactly that
        many rows that scrolls itself, its column names and horizontal bar
        on screen with it, and the page has nothing left to scroll.  Below
        the fewest rows a table still shows, the page scrolls the rest.
        """

        # The line over the commands comes and goes: settle the editor's
        # own layout before reading the viewport it leaves.
        self.layout().activate()
        viewport = self.editor_scroll.viewport().height()
        if viewport <= 0:
            return
        page = self.editor_scroll.widget()
        page.layout().activate()
        table, card = self._page_tables[self._page]
        card.layout().activate()
        above = table.mapTo(page, QtCore.QPoint()).y()
        below = card.layout().contentsMargins().bottom()
        rows = table.verticalHeader()
        pitch = rows.sectionSize(0) if rows.count() else rows.defaultSectionSize()
        # The horizontal bar only while the columns overflow, as the table
        # counts it in its own height: one stretched Value column never
        # shows it, and the row it was held for is a row of values.
        bar = table.horizontalScrollBar()
        chrome = (
            table.horizontalHeader().sizeHint().height()
            + (bar.sizeHint().height() if bar.maximum() > bar.minimum() else 0)
            + 2 * table.frameWidth()
        )
        fit = max(_LEAST_TABLE_ROWS, (viewport - above - below - chrome) // max(1, pitch))
        if self._table_rows.get(table) != fit:
            self._table_rows[table] = fit
            table.set_content_rows_limit(fit, viewport_rows=fit)

    def is_dirty(self) -> bool:
        return bool(getattr(self, "_dirty", False))

    def _emit(self, operation: str, **values: object) -> None:
        if not self._updating:
            self.intent_requested.emit({"op": str(operation), **values})

    def _dataset_field(self, field: str, value: object) -> None:
        self._emit("set_dataset_field", field=str(field), value=value)

    def _show_local_rejection(self, message: str) -> None:
        self._show_message(message)

    def _show_message(self, message: object) -> None:
        # One sentence, whoever words it: this editor's own refusals and
        # what its presenter says alike.
        text = sentence(str(message))
        self.message_label.setText(text)
        self.message_label.setVisible(bool(text))
        # A line over the commands is room the page no longer has.
        self._table_fit.start()

    def _show_domain(self, domain: object) -> None:
        # Part of the axis draft: sent with its name and length by Apply or
        # Create, not on its own.
        self._domain = domain
        self.domain_choice.set_current(domain)

    def _show_component(self, component: object) -> None:
        self._component = component
        self.component_choice.set_current(component)

    def _component_picked(self, component: object) -> None:
        self._show_component(component)
        self._emit("set_component", component=component)

    def _select_axis(self, _index: int) -> None:
        self._adding_axis = False
        self._emit("select_axis", axis_id=self.axis_combo.currentData())

    def _begin_add_axis(self) -> None:
        self._adding_axis = True
        self._selected_axis = ""
        with signals_blocked(self.axis_combo):
            self.axis_combo.setCurrentIndex(-1)
        self.axis_name_edit.clear()
        self.axis_size_spin.setValue(1)
        self.axis_unit_edit.clear()
        self.axis_value_model.set_projection(
            {"shape": (0, 1), "values": (), "column_headers": ("Value",)}
        )
        self._show_domain(self._domain_keys[0] if self._domain_keys else None)
        self.axis_mode_label.setText("New axis")
        self.apply_axis_button.setText("Create axis")
        self._set_axis_mode_controls(has_axis=False)
        self.axis_name_edit.setFocus()

    def _set_axis_mode_controls(self, *, has_axis: bool) -> None:
        """Enable exactly the controls the current axis mode can use.

        Add and a projection are two entries into one mode, so both come
        through here.  Add has no presenter round trip of its own -- nothing
        is authored until Create -- so it cannot wait for a projection to
        do this: a New axis has a name, length, unit and domain to write
        and nothing to delete or list values for, and a Dataset with no
        axis yet still gets to create its first one.
        """

        editing = has_axis or self._adding_axis
        for control in (
            self.axis_name_edit,
            self.axis_size_spin,
            self.axis_unit_edit,
            self.domain_choice,
            self.apply_axis_button,
        ):
            control.setEnabled(editing)
        self.remove_axis_button.setEnabled(has_axis and not self._adding_axis)
        self.axis_value_table.setEnabled(has_axis and not self._adding_axis)

    def _commit_axis(self) -> None:
        name = self.axis_name_edit.text().strip()
        if not name:
            self._show_local_rejection("Axis name cannot be blank")
            return
        values = {
            "name": name,
            "length": int(self.axis_size_spin.value()),
            "unit": self.axis_unit_edit.text().strip(),
            "domain": self._domain,
        }
        if self._adding_axis:
            # Still adding until the projection shows the axis made
            # (update_projection): a refused Create keeps the draft and the
            # button that creates it.
            self._emit("add_axis", **values)
        elif self._selected_axis:
            self._emit("edit_axis", axis_id=self._selected_axis, **values)

    def _save_as(self) -> None:
        path = fluent_save_path(
            self,
            "Save edited figure",
            self._save_suggested,
            "Saved figure archives (*.npz)",
        )
        if path:
            self._emit("save_as", path=path, note=self.note_edit.text())

    def _set_structure(self, structure: object) -> None:
        shape, names = data_structure_fragments(structure)

        def html(fragments: object) -> str:
            return "".join(
                _escaped(text)
                if colour is None
                else f'<span style="color:{colour}">{_escaped(text)}</span>'
                for text, colour, _elide in tuple(fragments)
            )

        self.structure_label.setText(
            html(shape) + "<br>" + html(names)
        )

    def _axis_view_mode_changed(
        self, axis_id: str, combo: FluentCycleComboBox
    ) -> None:
        if combo.isCycleSelected():
            self._emit(
                "set_scope",
                axis_id=str(axis_id),
                index=combo.cyclePosition(),
            )
        else:
            self._emit(
                "set_table_axis",
                axis_id=str(axis_id),
                mode=combo.currentData(),
            )

    def _set_axis_view_controls(self, rows: object) -> None:
        projected = tuple(dict(raw) for raw in tuple(rows or ()))
        wanted = {str(row.get("axis_id", "")) for row in projected}
        for axis_id in tuple(self._axis_view_widgets):
            if axis_id in wanted:
                continue
            widget, *_unused = self._axis_view_widgets.pop(axis_id)
            self._axis_view_layout.removeWidget(widget)
            retire_widget(widget)
        for position, row in enumerate(projected):
            axis_id = str(row.get("axis_id", ""))
            controls = self._axis_view_widgets.get(axis_id)
            if controls is None:
                label = ElidedLabel("")
                mode = FluentCycleComboBox()
                mode.setMinimumContentsLength(9)
                holder = FluentSettingRow(
                    label, mode, label_width=self._label_width,
                    parent=self._axis_view_holder,
                )
                mode.activated.connect(
                    lambda _index, aid=axis_id, control=mode: self._axis_view_mode_changed(
                        aid, control
                    )
                )
                controls = (holder, label, mode)
                self._axis_view_widgets[axis_id] = controls
            holder, label, mode = controls
            placed = self._axis_view_layout.indexOf(holder)
            if placed < 0 or self._axis_view_layout.getItemPosition(placed) != (position, 0, 1, 1):
                self._axis_view_layout.addWidget(holder, position, 0)
            unit = str(row.get("unit") or "")
            suffix = f" ({int(row.get('size', 1))})" + (f" [{unit}]" if unit else "")
            text = str(row.get("name", axis_id)) + suffix
            if label.text() != text:
                holder.set_label(text)
            with signals_blocked(mode):
                if not mode.count():
                    mode.addItem("Rows ↓", "rows")
                    mode.addItem("Columns →", "columns")
                mode.setCycleChoices("Scope", row["scope_choices"])
                selected_mode = str(row.get("mode", "scope"))
                if selected_mode == "scope":
                    mode.setCyclePosition(int(row.get("index", 0)))
                else:
                    mode.setCurrentIndex(mode.findData(selected_mode))
        self._set_label_width(
            tuple(label.text() for _holder, label, _mode in self._axis_view_widgets.values())
        )
        self._axis_view_holder.setVisible(bool(projected))

    def update_projection(self, projection: object) -> None:
        data = dict(projection or {})
        dataset = dict(data.get("dataset", {}))
        axes = tuple(dict(axis) for axis in tuple(data.get("axes", ())))
        selected_axis = str(data.get("selected_axis", ""))
        selected = next((axis for axis in axes if str(axis.get("id", "")) == selected_axis), None)
        if selected is None and axes:
            selected = axes[0]
            selected_axis = str(selected.get("id", ""))

        def project(edit: QtWidgets.QLineEdit, text: str) -> None:
            # A preview landing re-projects the editor; the box the operator
            # is typing in is theirs until they leave it.
            if not being_edited(edit):
                edit.setText(text)

        self._updating = True
        try:
            project(self.name_edit, str(dataset.get("name", "")))
            _fill_choice_combo(self.dtype_combo, dataset.get("dtype_choices", ()), dataset.get("dtype"))
            project(self.unit_edit, str(dataset.get("unit", "")))
            project(self.note_edit, str(dataset.get("note", "")))
            self.source_note.setText(str(dataset.get("source", "")))
            self._save_suggested = str(data.get("save_suggested", "figure.npz"))

            axis_rows = tuple(
                (
                    str(axis.get("id", "")),
                    f"{axis.get('domain_label', axis.get('domain', 'Axis'))} · {axis.get('name', axis.get('id', ''))}",
                )
                for axis in axes
            )
            listed = {self.axis_combo.itemData(index) for index in range(self.axis_combo.count())}
            _fill_choice_combo(self.axis_combo, axis_rows, selected_axis)
            if self._adding_axis and selected_axis and selected_axis not in listed:
                # Create landed: the presenter selects the axis it made,
                # which the list did not hold.  Cleared when Create was
                # pressed, a refusal re-projected the selected axis over the
                # draft and turned the button into its Apply -- or, with no
                # projection at all, left a Create button that did nothing.
                self._adding_axis = False
            if self._adding_axis:
                self.axis_combo.setCurrentIndex(-1)
            self._selected_axis = selected_axis
            if not self._adding_axis:
                self._show_domain(None if selected is None else selected.get("domain"))
                project(
                    self.axis_name_edit,
                    "" if selected is None else str(selected.get("name", "")),
                )
                if not being_edited(self.axis_size_spin):
                    self.axis_size_spin.setValue(
                        1 if selected is None else max(1, int(selected.get("size", 1)))
                    )
                project(
                    self.axis_unit_edit,
                    "" if selected is None else str(selected.get("unit", "")),
                )
                self.axis_mode_label.setText("Edit selected axis")
                self.apply_axis_button.setText("Apply axis")
            self.axis_value_model.set_projection(data.get("axis_values", {}))
            table = dict(data.get("table", {}))
            self._set_structure(table.get("structure", ()))
            self._set_axis_view_controls(table.get("axes", ()))
            # Sigma is offered only while the draft carries one; it keeps its
            # place, so switching Sigma on does not widen the choice beside it.
            for row in tuple(table.get("component_choices", ())):
                self.component_choice.button(row[0]).setEnabled(
                    bool(row[2]) if len(row) == 3 else True
                )
            self._show_component(table.get("component"))
            with signals_blocked(self.sigma_switch):
                self.sigma_switch.setChecked(bool(table.get("sigma_enabled", False)))
            self.value_model.set_projection(table)
            self.blank_help_label.setText(str(table.get("blank_help", "")))
            self._show_message(data.get("message", ""))
            self._dirty = bool(data.get("dirty", False))
            modified = bool(data.get("modified", False))
            self.apply_button.set_dirty(modified)
            self.save_button.set_dirty(self._dirty and not modified)
            self.discard_button.setEnabled(self._dirty)
            self._set_axis_mode_controls(has_axis=selected is not None)
            self.apply_button.setEnabled(bool(data.get("can_apply", True)))
            self.save_button.setEnabled(bool(data.get("can_save", False)))
        finally:
            self._updating = False
        # The rows above a table -- the Data card's axis rows, its wrapped
        # structure line -- come and go with the projection.
        self._table_fit.start()


class FigureViewerView(QtWidgets.QWidget):
    path_committed = QtCore.pyqtSignal(str)
    new_data_requested = QtCore.pyqtSignal()
    edit_data_requested = QtCore.pyqtSignal(str)
    data_editor_intent = QtCore.pyqtSignal(str, object)
    data_editor_closed = QtCore.pyqtSignal(str)
    add_panel_requested = QtCore.pyqtSignal(str)
    panel_state_changed = QtCore.pyqtSignal(str, object)
    panel_remove_requested = QtCore.pyqtSignal(str)
    panel_edit_requested = QtCore.pyqtSignal(str)
    panel_order_committed = QtCore.pyqtSignal(tuple)
    panel_editor_closed = QtCore.pyqtSignal(str)
    panel_snapshot_refresh_requested = QtCore.pyqtSignal(str)
    panel_save_figure_requested = QtCore.pyqtSignal(str, str)
    panel_plot_error = QtCore.pyqtSignal(str, str)
    save_screenshot_requested = QtCore.pyqtSignal()
    #: An action a Devices row offered was pressed (the action's id).
    info_action_requested = QtCore.pyqtSignal(str)
    close_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None, *, path_base_dir: str = "") -> None:
        super().__init__(parent)
        self.setObjectName("FigureViewerView")
        self.setStyleSheet("background: transparent;")
        self._cards: dict[str, PanelCardView] = {}
        self._editors: dict[str, QtWidgets.QWidget] = {}
        self._data_editors: dict[str, _DataEditorView] = {}
        self._panel_sizes: tuple[str, ...] = ()
        self._panel_default_size = ""
        self._panel_intervals: tuple[int, ...] = ()
        self._panel_default_interval = 0
        self._grid_cell_kinds: tuple[str, ...] = ()
        self._closing = False
        self._info_tabs: tuple = ()
        self._flow_graph: object = {"nodes": (), "edges": ()}
        root = QtWidgets.QHBoxLayout(self)

        self.info_pane = InfoPane(
            # These formal projection keys also determine the fixed left-pane
            # width, so using invented labels
            # changes the entire FigureViewer split even while the pane looks
            # otherwise correct.
            label_names=("schema_fingerprint", "coordinate_frame"),
            tabs=(
                ("Plot", ()),
                ("Logic", ()),
                ("Devices", ()),
                ("Flow", ()),
                ("Raw", ()),
            ),
            path_label="File",
            path_caption="Open a saved figure archive (.npz)",
            file_filter="Saved figure archives (*.npz)",
            path_base_dir=str(path_base_dir),
            initial_status="Open a saved figure archive (.npz).",
            graph_tabs=("Flow",),
            parent=self,
        )
        self.info_pane.path_committed.connect(self.path_committed)
        self.info_pane.action_requested.connect(self.info_action_requested)
        # The window pads every side by one WINDOW_PAD and the two panes
        # stand one WINDOW_PAD apart; whatever side inset the pane keeps
        # itself counts toward those, so the right surface has the same
        # margin at the window's edge as the pane has at its own.
        pad = window_pad()
        pane = self.info_pane.layout()
        inset = pane.contentsMargins()
        root.setContentsMargins(pad - inset.left(), pad, pad, pad)
        root.setSpacing(pad - inset.right())
        root.addWidget(self.info_pane, 0)

        # The right half is one white Fluent work surface.  Cards remain
        # visibly separate inside it instead of dissolving into the grey
        # top-level background.  Its tab strip runs from side to side, as
        # the left pane's does, and its first tab's underline starts on the
        # one card-pad edge every row of the surface starts on (the strip's
        # page inset); each row keeps that inset itself.
        holder = FluentFrame(parent=self, bordered=False)
        holder.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._surface_layout = QtWidgets.QVBoxLayout(holder)
        side = card_pad()
        inset = _surface_right_inset()
        self._surface_layout.setContentsMargins(0, 0, 0, side)
        # The two panes end on one line as well: the pane's status reads on
        # the line of the data editor's action bar, a control high and one
        # card pad above the surface's foot, so its band is that row with a
        # card pad above and below it.
        self.info_pane.status.setFixedHeight(control_height() + 2 * side)
        # The two panes are built alike: a header band, the gap, a tab strip.
        # The data band is as tall as the pane's path band and the gap is
        # the pane's own, so the two first rows share one centre line and
        # the two tab strips one top.
        self._surface_layout.setSpacing(pane.spacing())
        path_band = self.info_pane.path_edit.parentWidget()

        self._panel_bar = QtWidgets.QWidget(holder)
        self._panel_bar.setStyleSheet("background: transparent;")
        self._panel_bar.setFixedHeight(
            max(path_band.minimumHeight(), min(path_band.maximumHeight(), path_band.sizeHint().height()))
        )
        # The button beside a chooser acts ON what the chooser holds; the
        # one after it does something else entirely: Edit data answers the
        # box to its left, New data does not.
        self.data_combo = FluentComboBox()
        self.data_combo.setMinimumContentsLength(15)
        self.data_combo.setEnabled(False)
        self.edit_data_button = FluentButton("Edit data")
        self.edit_data_button.setEnabled(False)
        self.edit_data_button.clicked.connect(self._edit_selected_data)
        self.new_data_button = FluentButton("New data")
        self.new_data_button.clicked.connect(self.new_data_requested)
        data_label = FluentLabel("Data")
        bar_layout = QtWidgets.QHBoxLayout(self._panel_bar)
        bar_layout.setContentsMargins(side, 0, inset, 0)
        # One row of controls, as tall as they are, centred in the band: a
        # row stretched to the band centred its label on the band and the
        # controls on themselves, half a pixel apart.
        bar_layout.addWidget(
            FluentSettingRow(
                data_label,
                self.data_combo,
                label_width=data_label.sizeHint().width(),
                after=((self.edit_data_button, 0), (self.new_data_button, 0)),
            ),
            0,
            QtCore.Qt.AlignVCenter,
        )
        self._surface_layout.addWidget(self._panel_bar)

        self.tabs = FluentTabWidget(holder)
        # Every page of this surface starts its content one card pad in.
        self.tabs.set_page_inset(side)
        # The strip stands over Monitor alone too: editor tabs come and go
        # beside it, and the board under it -- level with the left pane's
        # filter -- does not move when the first opens or the last closes.
        self.tabs.set_lone_bar_visible(True)
        self.tabs.tab_close_requested.connect(self._tab_close_clicked)
        monitor = QtWidgets.QWidget()
        monitor.setStyleSheet("background: white;")
        monitor_layout = QtWidgets.QVBoxLayout(monitor)
        # One card pad under the tab strip, as every tab page of the window.
        # The board runs from side to side and keeps its own card-pad gap at
        # its edges, so its first card stands one pad under the commands and
        # starts on their edge.
        monitor_layout.setContentsMargins(0, side, 0, 0)
        monitor_layout.setSpacing(0)
        # Adding a panel and saving the board's picture act on the board
        # alone, so they head the board's own tab; in a second bar above
        # every tab they pushed this pane's tab strip a row below the
        # left pane's.
        self.kind_combo = FluentComboBox()
        self.kind_combo.setMinimumContentsLength(12)
        self.add_panel_button = FluentButton("Add panel")
        self.add_panel_button.clicked.connect(self._add_selected_panel)
        self.add_panel_button.setEnabled(False)
        self.save_image_button = FluentButton("Save image…")
        self.save_image_button.clicked.connect(self.save_screenshot_requested)
        # A chooser and the command that reads it, spaced as the data row's.
        board_actions = QtWidgets.QHBoxLayout()
        board_actions.setContentsMargins(side, 0, inset, 0)
        board_actions.setSpacing(row_gap())
        board_actions.addWidget(self.kind_combo)
        board_actions.addWidget(self.add_panel_button)
        board_actions.addStretch(1)
        board_actions.addWidget(self.save_image_button)
        monitor_layout.addLayout(board_actions)
        self.board = ConsoleBoardView()
        self.board.order_committed.connect(self.panel_order_committed)
        # An empty board says its next step where its first card will
        # stand, as the console's does: the words start where a card's
        # title does and end where its text would, and their top is a
        # card's top -- the board's gap under the commands.
        self.board_hint = muted_note_label("")
        self.board_hint.setWordWrap(True)
        words = self.board.text_left()
        self.board_hint.setContentsMargins(
            words, words - PanelCardView.strip_text_inset(), words, 0
        )
        monitor_layout.addWidget(self.board_hint)
        self.scroll = FluentScrollArea()
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setWidget(self.board)
        monitor_layout.addWidget(self.scroll, 1)
        self.tabs.add_permanent_tab(monitor, "Monitor")
        self._surface_layout.addWidget(self.tabs, 1)
        root.addWidget(holder, 1)
        self._sync_monitor_empty()

    def set_archive_info(
        self,
        tabs: tuple[tuple[str, tuple[tuple[str, object], ...]], ...],
        graph: object,
    ) -> None:
        """Replace one archive's rows and Flow in the same owner turn."""

        previous_tabs = self._info_tabs
        previous_graph = self._flow_graph
        self._info_tabs = tuple(tabs)
        self._flow_graph = graph
        try:
            self._render_info()
        except BaseException:
            self._info_tabs = previous_tabs
            self._flow_graph = previous_graph
            self._render_info()
            raise
        self._sync_monitor_empty()

    def _render_info(self) -> None:
        rows = dict(self._info_tabs)
        self.info_pane.set_tabs(tuple(
            (title, () if title == "Flow" else tuple(rows.get(title, ())))
            for title in ("Plot", "Logic", "Devices", "Flow", "Raw")
        ))
        self.info_pane.set_graph("Flow", self._flow_graph)

    def set_panel_kinds(self, kinds: object, default_kind: str = "") -> None:
        rows = tuple((str(key), str(label or key)) for key, label in tuple(kinds))
        current = self.kind_combo.currentData()
        self.kind_combo.clear()
        for key, label in rows:
            # Worded as the console's chooser words it: the box says what
            # Add panel will add.
            self.kind_combo.addItem(plot_choice_text(label), key)
        if current is not None:
            index = self.kind_combo.findData(current)
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
        if self.kind_combo.currentIndex() < 0 and default_kind:
            index = self.kind_combo.findData(str(default_kind))
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
        self.add_panel_button.setEnabled(bool(rows))

    def set_editable_data_choices(
        self, choices: object, *, current: str = ""
    ) -> None:
        """Project archive/manual Dataset choices without interpreting them."""

        rows = tuple((str(key), str(label or key)) for key, label in tuple(choices))
        selected = str(current or self.data_combo.currentData() or "")
        _fill_choice_combo(self.data_combo, rows, selected)
        enabled = bool(rows)
        self.data_combo.setEnabled(enabled)
        self.edit_data_button.setEnabled(enabled)
        self._sync_monitor_empty()

    def _edit_selected_data(self) -> None:
        key = self.data_combo.currentData()
        if isinstance(key, str) and key:
            self.edit_data_requested.emit(key)

    def set_panel_sizes(self, sizes: object, default_size: str) -> None:
        self._panel_sizes = tuple(str(value) for value in tuple(sizes))
        self._panel_default_size = str(default_size)
        for card in self._cards.values():
            card.set_size_choices(self._panel_sizes, self._panel_default_size)

    def set_panel_intervals(
        self, intervals: object, default_interval: int
    ) -> None:
        self._panel_intervals = tuple(int(value) for value in tuple(intervals))
        self._panel_default_interval = int(default_interval)
        for card in self._cards.values():
            card.set_interval_choices(
                self._panel_intervals,
                self._panel_default_interval,
            )

    def set_grid_cell_kinds(self, kinds: object) -> None:
        self._grid_cell_kinds = tuple(str(value) for value in tuple(kinds))
        for card in self._cards.values():
            card.set_cell_kind_choices(self._grid_cell_kinds)

    def _add_selected_panel(self) -> None:
        kind = self.kind_combo.currentData()
        if isinstance(kind, str):
            self.add_panel_requested.emit(kind)

    def add_panel(self, panel_id: str, title: str) -> None:
        key = str(panel_id)
        if key in self._cards:
            return
        self._cards[key] = new_panel_card(self, key, title)
        self.board.set_cards(tuple(self._cards.values()))
        self._sync_monitor_empty()

    def remove_panel(self, panel_id: str) -> None:
        key = str(panel_id)
        self.close_panel_editor(key)
        self._cards.pop(key, None)
        self.board.set_cards(tuple(self._cards.values()))
        self._sync_monitor_empty()

    def set_panel_order(self, order: object) -> None:
        wanted = [str(key) for key in tuple(order) if str(key) in self._cards]
        wanted += [key for key in self._cards if key not in wanted]
        self._cards = {key: self._cards[key] for key in wanted}
        self.board.set_cards(tuple(self._cards.values()))

    def set_panel_signal_choices(
        self,
        panel_id: str,
        groups: object,
        *,
        current: str = "",
        overlay_groups: object = (),
        overlay_current: str = "",
    ) -> None:
        self._cards[str(panel_id)].set_signal_choices(
            groups,
            current=str(current),
            overlay_groups=overlay_groups,
            overlay_current=str(overlay_current),
        )

    def set_panel_publishers(self, _publishers: object) -> None:
        """FigureViewer has no Logic-row chrome; signals remain in Setting."""

    def panel_ids(self) -> tuple[str, ...]:
        return tuple(self._cards)

    def set_panel_selectors_enabled(self, panel_id: str, enabled: bool) -> None:
        key = str(panel_id)
        self._cards[key].set_selectors_enabled(bool(enabled))
        editor = self._editors.get(key)
        if isinstance(editor, PanelEditorView):
            editor.set_selectors_enabled(bool(enabled))

    def present_panel_front(self, panel_id: str, front: object) -> bool:
        card = self._cards.get(str(panel_id))
        surface = None if card is None else card.surface
        present = getattr(surface, "present_front", None)
        return bool(callable(present) and present(front))

    def set_panel_projection(
        self, panel_id: str, state: object, surface: object
    ) -> None:
        self._cards[str(panel_id)].set_panel_projection(state, surface)

    def set_panel_surface(
        self, panel_id: str, widget: QtWidgets.QWidget | None
    ) -> None:
        if widget is not None and not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("figure surface must be QWidget or None")
        self._cards[str(panel_id)].set_surface(widget)

    def panel_surface(self, panel_id: str) -> QtWidgets.QWidget | None:
        """The surface a panel currently shows, so its owner can reuse it."""

        return self._cards[str(panel_id)].surface

    def set_panel_status(self, panel_id: str, text: str, *, error: bool) -> None:
        self._cards[str(panel_id)].set_status(str(text), error=bool(error))

    def _sync_monitor_empty(self) -> None:
        """What an empty board's next step is: a figure to open, until
        there is a figure or data to plot; a panel to add, after."""

        has_data = bool(self._info_tabs) or self.data_combo.count() > 0
        self.board_hint.setText(
            "No panels yet. Choose a Plot above and click Add panel."
            if has_data
            else "Open a saved figure to begin, or make data with New data."
        )
        self.board_hint.setVisible(not self._cards)

    def open_panel_editor(
        self, panel_id: str, projection: object, title: str
    ) -> None:
        key = str(panel_id)
        editor = self._editors.get(key)
        if editor is not None:
            editor.update_projection(panel_editor_projection(self, projection))
            self.tabs.setCurrentWidget(editor)
        else:
            # On this surface's page inset (its tab widget's), not the
            # console's: the Edit page starts where the Monitor and data
            # pages do.
            editor = new_panel_editor(self, key, projection, page_inset=card_pad())
            self._editors[key] = editor
            self.tabs.add_closable_tab(editor, str(title), focus=True)
        editor.set_selectors_enabled(self._cards[key].selectors_enabled)

    def open_data_editor(
        self, editor_id: str, projection: object, title: str
    ) -> None:
        key = str(editor_id)
        existing = self._data_editors.get(key)
        if existing is not None:
            existing.update_projection(projection)
            self.tabs.setCurrentWidget(existing)
            return
        editor = _DataEditorView(projection, self)
        editor.intent_requested.connect(
            lambda intent, eid=key: self.data_editor_intent.emit(eid, intent)
        )
        self._data_editors[key] = editor
        editor.setProperty("data_editor_title", str(title))
        self.tabs.add_closable_tab(editor, str(title), focus=True)
        self._sync_data_editor_title(editor)

    def close_data_editor(self, editor_id: str) -> bool:
        key = str(editor_id)
        editor = self._data_editors.pop(key, None)
        if editor is None:
            return False
        index = self.tabs.indexOf(editor)
        if index >= 0:
            self.tabs.removeTab(index)
        retire_widget(editor)
        return True

    def update_data_editor(self, editor_id: str, projection: object) -> bool:
        editor = self._data_editors.get(str(editor_id))
        if editor is None:
            return False
        document = dict(projection or {})
        dataset = dict(document.get("dataset") or {})
        name = str(dataset.get("name") or "").strip()
        if name:
            editor.setProperty("data_editor_title", f"Data · {name}")
        editor.update_projection(projection)
        self._sync_data_editor_title(editor)
        return True

    def _sync_data_editor_title(self, editor: _DataEditorView) -> None:
        index = self.tabs.indexOf(editor)
        if index < 0:
            return
        base = str(editor.property("data_editor_title") or "Data")
        self.tabs.setTabText(index, base + (" *" if editor.is_dirty() else ""))

    def has_data_editor(self, editor_id: str) -> bool:
        return str(editor_id) in self._data_editors

    def focus_data_editor(self, editor_id: str) -> bool:
        editor = self._data_editors.get(str(editor_id))
        if editor is None:
            return False
        self.tabs.setCurrentWidget(editor)
        return True

    def close_panel_editor(self, panel_id: str) -> bool:
        key = str(panel_id)
        editor = self._editors.pop(key, None)
        if editor is None:
            return False
        # Unmount first: the host's widget is the host's, and is not retired
        # with the page that showed it.
        editor.set_surface(None)
        index = self.tabs.indexOf(editor)
        if index >= 0:
            self.tabs.removeTab(index)
        retire_widget(editor)
        return True

    def update_panel_editor(self, panel_id: str, projection: object) -> bool:
        editor = self._editors.get(str(panel_id))
        if not isinstance(editor, PanelEditorView):
            return False
        editor.update_projection(panel_editor_projection(self, projection))
        return True

    def has_panel_editor(self, panel_id: str) -> bool:
        return str(panel_id) in self._editors

    def set_panel_snapshot_status(self, panel_id: str, status: object) -> None:
        editor = self._editors.get(str(panel_id))
        if isinstance(editor, PanelEditorView):
            editor.set_snapshot_status(dict(status))

    def show_panel_editor(
        self, panel_id: str, widget: QtWidgets.QWidget | None
    ) -> None:
        editor = self._editors[str(panel_id)]
        assert isinstance(editor, PanelEditorView)
        editor.set_surface(widget)

    def focus_panel_editor(self, panel_id: str) -> bool:
        editor = self._editors.get(str(panel_id))
        if editor is None:
            return False
        self.tabs.setCurrentWidget(editor)
        return True

    def _tab_close_clicked(self, editor: QtWidgets.QWidget) -> None:
        data_editor_id = next(
            (key for key, value in self._data_editors.items() if value is editor),
            "",
        )
        if data_editor_id:
            self.data_editor_closed.emit(data_editor_id)
            return
        panel_id = next(
            (key for key, value in self._editors.items() if value is editor),
            "",
        )
        if panel_id:
            self.panel_editor_closed.emit(panel_id)

    def set_status(self, text: str, *, error: bool = False) -> None:
        if error:
            self.info_pane.status.show_message(str(text), severity="error")
        else:
            self.info_pane.set_status(str(text))

    def show_status(self, text: str, severity: str) -> None:
        """Present the same ConsolePresenter status channel as TaskConsole."""

        self.info_pane.status.show_message(str(text), severity=str(severity))

    def set_title(self, text: str) -> None:
        self.setWindowTitle(str(text))

    def set_path(self, path: str) -> None:
        """Show which file is open.

        A viewer whose File field stays empty after opening something is a
        viewer you cannot tell apart from one that opened nothing -- and the
        first thing anyone checks when a figure looks wrong is which file it is.
        """

        self.info_pane.path_edit.blockSignals(True)
        self.info_pane.path_edit.setText(str(path))
        self.info_pane.path_edit.blockSignals(False)

    def finish_close(self) -> None:
        self._closing = True
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._closing:
            event.ignore()
            self.close_requested.emit()
            return
        super().closeEvent(event)


__all__ = ["FigureViewerView"]
