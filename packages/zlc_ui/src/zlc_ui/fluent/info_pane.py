"""Generic read-only information surface with an optional path field.

A record is a tree, and it is read as one.  Every tab is a two-column
tree of names and values -- a run's parameters under the run, a device's
snapshot under the device, the whole saved document under its sections --
with a filter above it that finds a name or a value anywhere in the tab,
and Copy on any row.  A pane that printed each record as a Python literal
in a text box was showing the data structure, not the data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum

from PyQt5 import QtCore, QtGui, QtWidgets

from .fluent import (
    FluentButton,
    FluentFrame,
    FluentLineEdit,
    FluentPathEdit,
    FluentSectionLabel,
    FluentStatusStrip,
    FluentTabWidget,
    _FluentRoundedMenu,
    apply_fluent_scrollbars,
    fluent_font_size,
    retire_widget,
    scaled_px,
    setting_label_width,
    window_pad,
)
from .style import (
    ACCENT,
    ACCENT_TINT,
    BG,
    DIVIDER,
    FONT,
    GRAPHITE,
    GREY,
    ORANGE,
    ORANGE_DARK,
    ORANGE_TINT,
    RADIUS,
    SURFACE,
    TEXT,
)


InfoRow = tuple[str, object]
InfoTab = tuple[str, tuple[InfoRow, ...]]

#: A flat list longer than this is read as its count and range, not spelled
#: out: a coordinate axis of 96 pixels is worth "96 numbers, 0 to 95".
_LIST_PREVIEW = 8


def _plain(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _is_scalar(value: object) -> bool:
    return _plain(value) is None or isinstance(_plain(value), (str, bool, int, float))


def _is_flat_list(value: object) -> bool:
    return isinstance(value, (list, tuple)) and all(_is_scalar(item) for item in value)


def _is_composite(value: object) -> bool:
    return isinstance(value, Mapping) or (
        isinstance(value, (list, tuple)) and not _is_flat_list(value)
    )


def value_text(value: object) -> str:
    """One value as the pane shows it beside its name.

    A number is a number, a switch is a word, a flat list is one line, and
    a long list of numbers is its count and range.  A mapping shows its
    scalar fields inline -- the summary of the record under it -- and a
    list of records says how many there are.
    """

    value = _plain(value)
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    if _is_flat_list(value):
        items = tuple(_plain(item) for item in value)
        if not items:
            return "(none)"
        numbers = [
            item for item in items
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ]
        if len(items) > _LIST_PREVIEW and len(numbers) == len(items):
            return (
                f"{len(items)} numbers, {value_text(min(numbers))} "
                f"to {value_text(max(numbers))}"
            )
        shown = ", ".join(value_text(item) for item in items[:_LIST_PREVIEW])
        if len(items) > _LIST_PREVIEW:
            return f"{shown}, … ({len(items)} in all)"
        return shown
    if isinstance(value, Mapping):
        inline = "; ".join(
            f"{key}: {value_text(item)}"
            for key, item in value.items()
            if not _is_composite(item)
        )
        return inline or f"{len(value)} fields"
    if isinstance(value, (list, tuple)):
        return f"{len(value)} items"
    return str(value)


def copy_text(value: object, indent: str = "") -> str:
    """The whole value, for the clipboard: every number of a long list,
    every field of a record, one name per line, nested by indentation."""

    value = _plain(value)
    if _is_flat_list(value):
        return ", ".join(value_text(item) for item in value) or "(none)"
    if isinstance(value, Mapping):
        lines = []
        for key, item in value.items():
            if _is_composite(item):
                lines.append(f"{indent}{key}:")
                lines.append(copy_text(item, indent + "  "))
            else:
                lines.append(f"{indent}{key}: {copy_text(item)}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        lines = []
        for index, item in enumerate(value):
            if _is_composite(item):
                lines.append(f"{indent}[{index}]:")
                lines.append(copy_text(item, indent + "  "))
            else:
                lines.append(f"{indent}[{index}]: {copy_text(item)}")
        return "\n".join(lines)
    return value_text(value)


def _is_action(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"text", "action"}
        and isinstance(value["text"], str)
        and isinstance(value["action"], str)
    )


class _WrapAnywhereDelegate(QtWidgets.QStyledItemDelegate):
    """Cells wrap anywhere, as the readout fields did: a path or a digest
    has no space to break at, and cut off it is not a value."""

    def __init__(self, tree: QtWidgets.QTreeWidget) -> None:
        super().__init__(tree)
        self._tree = tree
        self._pad_x = scaled_px(5, minimum=3)
        self._pad_y = scaled_px(2, minimum=1)

    def _laid_out(self, option, index, width: int) -> QtGui.QTextDocument:
        document = QtGui.QTextDocument()
        document.setDefaultFont(option.font)
        text_option = QtGui.QTextOption()
        text_option.setWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        document.setDefaultTextOption(text_option)
        document.setDocumentMargin(0)
        document.setPlainText(str(index.data(QtCore.Qt.DisplayRole) or ""))
        document.setTextWidth(max(1, width - 2 * self._pad_x))
        return document

    def _cell_width(self, index) -> int:
        """The width the cell is painted at: its column, less the
        indentation a nested name sits behind.  Measured at any other
        width, a value wraps to more lines than its row was given."""

        width = self._tree.columnWidth(index.column())
        if index.column() == 0:
            depth = 1
            parent = index.parent()
            while parent.isValid():
                depth += 1
                parent = parent.parent()
            width -= self._tree.indentation() * depth
        return max(1, width)

    def sizeHint(self, option, index):  # noqa: N802 - Qt naming
        styled = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        document = self._laid_out(styled, index, self._cell_width(index))
        return QtCore.QSize(
            int(document.idealWidth()) + 2 * self._pad_x,
            int(document.size().height()) + 2 * self._pad_y,
        )

    def paint(self, painter, option, index):  # noqa: N802 - Qt naming
        styled = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        text_color = styled.palette.color(QtGui.QPalette.Text)
        # The style paints the row -- its ground, its selection -- and the
        # words are laid out here, where they can wrap.
        styled.text = ""
        widget = styled.widget
        style = widget.style() if widget is not None else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, styled, painter, widget)
        document = self._laid_out(styled, index, styled.rect.width())
        painter.save()
        painter.setClipRect(styled.rect)
        painter.translate(styled.rect.left() + self._pad_x, styled.rect.top() + self._pad_y)
        context = QtGui.QAbstractTextDocumentLayout.PaintContext()
        context.palette.setColor(QtGui.QPalette.Text, text_color)
        document.documentLayout().draw(painter, context)
        painter.restore()


class InfoTree(QtWidgets.QTreeWidget):
    """Names and values as a tree: records open under their name, a filter
    finds any name or value, and Copy takes a row's whole value."""

    #: A row's action was pressed; carries the action the row named.
    action_requested = QtCore.pyqtSignal(str)

    def __init__(self, *, name_width: int, parent=None) -> None:
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setIndentation(scaled_px(14, minimum=10))
        # A value wraps under its name rather than being cut off: a path, a
        # program, a sentence of a record are read whole.
        self.setUniformRowHeights(False)
        self.setTextElideMode(QtCore.Qt.ElideNone)
        self.setItemDelegate(_WrapAnywhereDelegate(self))
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(False)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._name_floor = int(name_width)
        header = self.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        header.setMinimumSectionSize(scaled_px(60, minimum=48))
        header.resizeSection(0, self._name_floor)
        # A column that moved changes where every value wraps: the rows are
        # measured again at the widths they will be painted at.
        header.sectionResized.connect(lambda *_: self.scheduleDelayedItemsLayout())
        radius = scaled_px(RADIUS, minimum=2)
        self.setStyleSheet(
            f"""
            QTreeView {{
                background: {SURFACE};
                color: {TEXT};
                border: 1px solid {DIVIDER};
                border-radius: {radius}px;
                font: {fluent_font_size()}pt "{FONT}";
                outline: none;
                selection-background-color: {ACCENT_TINT};
                selection-color: {TEXT};
                show-decoration-selected: 1;
            }}
            QTreeView::item {{
                border: none;
            }}
            QTreeView::item:hover:!selected {{
                background: {BG};
            }}
            QTreeView::branch {{
                background: {SURFACE};
            }}
            QTreeView::branch:selected {{
                background: {ACCENT_TINT};
            }}
            """
        )
        apply_fluent_scrollbars(self)

    # ------------------------------------------------------------- rows

    def set_rows(self, rows: Iterable[InfoRow]) -> None:
        """Replace every row.  Top-level rows open; what is under them waits
        to be asked for, so a tab is a list of its subjects first."""

        self.clear()
        for label, value in rows:
            if _is_action(value):
                self._add_action(str(label), value)
                continue
            item = self._add_entry(None, str(label), value)
            item.setExpanded(True)
        self._fit_name_column()

    def _fit_name_column(self) -> None:
        """The name column is as wide as its widest name, up to half the
        view: a name broken across lines is a name that cannot be read."""

        metrics = self.fontMetrics()
        slack = 2 * scaled_px(5, minimum=3) + scaled_px(6, minimum=4)
        widest = self._name_floor

        def measure(item: QtWidgets.QTreeWidgetItem, depth: int) -> None:
            nonlocal widest
            widest = max(
                widest,
                metrics.horizontalAdvance(item.text(0))
                + self.indentation() * depth
                + slack,
            )
            for index in range(item.childCount()):
                measure(item.child(index), depth + 1)

        for item in self._top_level_items():
            measure(item, 1)
        limit = max(self._name_floor, self.viewport().width() // 2)
        self.header().resizeSection(0, min(widest, limit))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._fit_name_column()
        self.scheduleDelayedItemsLayout()

    def _add_action(self, label: str, value: Mapping[str, str]) -> None:
        item = QtWidgets.QTreeWidgetItem([label, ""])
        self.addTopLevelItem(item)
        button = FluentButton(str(value["text"]), color=ACCENT)
        # The value column's width, like every value beside it: a button
        # sized to its own text overflowed the column and was clipped
        # mid-word the moment the text was a real name.
        button.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        button.setMinimumWidth(0)
        button.clicked.connect(
            lambda _checked=False, action=str(value["action"]): (
                self.action_requested.emit(action)
            )
        )
        self.setItemWidget(item, 1, button)
        item.setData(1, QtCore.Qt.UserRole, str(value["text"]))

    def _add_entry(
        self, parent: QtWidgets.QTreeWidgetItem | None, name: str, value: object
    ) -> QtWidgets.QTreeWidgetItem:
        shown = value_text(value)
        item = QtWidgets.QTreeWidgetItem([name, shown])
        if parent is None:
            self.addTopLevelItem(item)
        else:
            parent.addChild(item)
        whole = copy_text(value)
        item.setData(1, QtCore.Qt.UserRole, whole)
        if whole != shown:
            item.setToolTip(1, whole)
        if _is_composite(value):
            # The inline summary is a reading of what is below, not a
            # value of its own: set in the muted ink so the eye tells
            # them apart.
            item.setForeground(1, QtGui.QBrush(QtGui.QColor(GRAPHITE)))
            entries = (
                value.items()
                if isinstance(value, Mapping)
                else ((f"[{index}]", item_value) for index, item_value in enumerate(value))
            )
            for key, child in entries:
                self._add_entry(item, str(key), child)
        return item

    def _top_level_items(self) -> tuple[QtWidgets.QTreeWidgetItem, ...]:
        return tuple(self.topLevelItem(index) for index in range(self.topLevelItemCount()))

    # ----------------------------------------------------------- filter

    def apply_filter(self, text: str) -> None:
        """Show the rows whose name or value contains ``text`` anywhere
        below them, opened down to the match; an empty filter shows every
        row, opened one level as at first."""

        needle = str(text).strip().lower()
        for item in self._top_level_items():
            self._filter_item(item, needle, top=True)

    def _filter_item(
        self, item: QtWidgets.QTreeWidgetItem, needle: str, *, top: bool
    ) -> bool:
        children = [item.child(index) for index in range(item.childCount())]
        # Every child is judged, not just the first that matches: a hidden
        # sibling is a row the operator cannot find.
        below = [self._filter_item(child, needle, top=False) for child in children]
        own = not needle or needle in item.text(0).lower() or needle in str(
            item.data(1, QtCore.Qt.UserRole) or item.text(1)
        ).lower()
        shown = own or any(below)
        item.setHidden(not shown)
        item.setExpanded(any(below) if needle else top)
        return shown

    # -------------------------------------------------------------- copy

    def show_row(self, label: str) -> bool:
        """Bring the top-level row named ``label`` into view, opened."""

        wanted = str(label)
        for item in self._top_level_items():
            if item.text(0) == wanted:
                item.setHidden(False)
                item.setExpanded(True)
                self.setCurrentItem(item)
                self.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtTop)
                return True
        return False

    def row_value(self, item: QtWidgets.QTreeWidgetItem | None = None) -> str:
        current = self.currentItem() if item is None else item
        if current is None:
            return ""
        return str(current.data(1, QtCore.Qt.UserRole) or current.text(1))

    def row_name(self, item: QtWidgets.QTreeWidgetItem | None = None) -> str:
        """The row's name from the top: ``devices.camera.exposure_seconds``."""

        current = self.currentItem() if item is None else item
        parts: list[str] = []
        while current is not None:
            parts.append(current.text(0))
            current = current.parent()
        return ".".join(reversed(parts))

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.matches(QtGui.QKeySequence.Copy) and self.currentItem() is not None:
            QtWidgets.QApplication.clipboard().setText(self.row_value())
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        item = self.itemAt(event.pos())
        if item is not None:
            self.setCurrentItem(item)
        menu = _FluentRoundedMenu(self)
        clipboard = QtWidgets.QApplication.clipboard()
        copy_value = menu.addAction("Copy value")
        copy_value.setEnabled(item is not None)
        copy_value.triggered.connect(lambda: clipboard.setText(self.row_value()))
        copy_name = menu.addAction("Copy name")
        copy_name.setEnabled(item is not None)
        copy_name.triggered.connect(lambda: clipboard.setText(self.row_name()))
        menu.addSeparator()
        menu.addAction("Expand all").triggered.connect(self.expandAll)
        menu.addAction("Collapse all").triggered.connect(self.collapseAll)
        try:
            menu.exec_(event.globalPos())
        finally:
            retire_widget(menu)
        event.accept()


class _RowsTab(QtWidgets.QWidget):
    """One tab: a filter over its tree."""

    def __init__(self, rows: Iterable[InfoRow], *, name_width: int, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(self)
        gap = scaled_px(6, minimum=4)
        layout.setContentsMargins(0, gap, 0, 0)
        layout.setSpacing(gap)
        self.filter_edit = FluentLineEdit()
        self.filter_edit.setPlaceholderText("filter names and values")
        self.filter_edit.setClearButtonEnabled(True)
        layout.addWidget(self.filter_edit)
        self.tree = InfoTree(name_width=name_width)
        self.tree.set_rows(rows)
        layout.addWidget(self.tree, 1)
        self.filter_edit.textChanged.connect(self.tree.apply_filter)


class _FlowView(QtWidgets.QGraphicsView):
    """The flow's picture; a click on a card names its node."""

    node_activated = QtCore.pyqtSignal(str)

    def __init__(self, scene: QtWidgets.QGraphicsScene, parent=None) -> None:
        super().__init__(scene, parent)
        self._pressed_at: QtCore.QPoint | None = None

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._pressed_at = event.pos() if event.button() == QtCore.Qt.LeftButton else None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().mouseReleaseEvent(event)
        pressed = self._pressed_at
        self._pressed_at = None
        if pressed is None or event.button() != QtCore.Qt.LeftButton:
            return
        # A press that travelled was a pan of the picture, not a choice.
        if (event.pos() - pressed).manhattanLength() > QtWidgets.QApplication.startDragDistance():
            return
        for item in self.items(event.pos()):
            node_id = item.data(0)
            if node_id:
                self.node_activated.emit(str(node_id))
                return


class InfoPane(QtWidgets.QWidget):
    """A reusable path header plus injected read-only tabs.

    The pane does not know what the rows represent.  A tab is supplied as
    ``(title, ((label, value), ...))`` and values are formatted only for
    display.  Path and status copy are also injected so the widget can be
    reused by different presenters without importing their data model.
    """

    path_committed = QtCore.pyqtSignal(str)
    #: A row's action was pressed; carries the action the row named.
    action_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        label_names: Iterable[str],
        tabs: tuple[InfoTab, ...] = (),
        path_label: str = "Path",
        path_caption: str = "Choose a path",
        file_filter: str = "All files (*)",
        path_base_dir: str = "",
        initial_status: str = "",
        graph_tabs: Iterable[str] = (),
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        # The declared names set the pane's one stable split.  Archive labels
        # are content, not geometry authority: loading another file must not
        # move the divider or resize the top-level window.
        self._declared_labels = tuple(str(name) for name in label_names)
        self._label_width = setting_label_width(self._declared_labels)
        self._graph_tab_titles = frozenset(str(title) for title in graph_tabs)
        self._graph_tabs: dict[str, _FlowView] = {}
        #: Per graph tab, which row each node stands for: ``(tab, label)``.
        self._graph_rows: dict[str, dict[str, tuple[str, str]]] = {}
        self._rows_tabs: dict[str, _RowsTab] = {}
        self._fixed_pane_width: int | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(window_pad(1), 0, window_pad(1), 0)
        layout.setSpacing(window_pad(0.5))

        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(scaled_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(scaled_px(12), scaled_px(6), scaled_px(12), scaled_px(6))
        header.setSpacing(scaled_px(8, minimum=5))
        header.addWidget(FluentSectionLabel(str(path_label)))

        self.path_edit = FluentPathEdit(
            "",
            mode="file",
            caption=str(path_caption),
            file_filter=str(file_filter),
            base_dir=str(path_base_dir),
            refreshable=True,
        )
        self.path_edit.selected.connect(self.path_committed.emit)
        # Refresh is "open this same path again": the file was written or
        # edited elsewhere, and re-picking it in the dialog is the only thing
        # the operator could do about it.
        self.path_edit.refresh_requested.connect(
            lambda: self.path_committed.emit(self.path_edit.text())
        )
        self.path_edit.edit.editingFinished.connect(self._commit_path_draft)
        header.addWidget(self.path_edit, 1)
        layout.addWidget(header_frame)

        self.info_tabs = FluentTabWidget()
        self.info_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.set_tabs(tabs)
        layout.addWidget(self.info_tabs, 1)

        self.status = FluentStatusStrip()
        if initial_status:
            self.status.show_message(str(initial_status))
        layout.addWidget(self.status)
        self._apply_pane_width()

    def _apply_pane_width(self) -> None:
        """Wide enough for the label column AND for its own tab bar.

        The pane used to be sized by the settings labels alone.  Five tabs did
        not fit in that width, so the tab bar elided its titles and grew the
        scroll arrows nobody asked for -- a navigation control that has to be
        scrolled to be seen is one an operator does not know exists.
        """

        if self._fixed_pane_width is not None:
            self.setFixedWidth(self._fixed_pane_width)
            return
        label_pane_width = self._label_width + scaled_px(280, minimum=220)
        bar = self.info_tabs.tabBar()
        # The bar caps its own tab widths when they overflow, so its laid-out
        # widths are the ones it already settled for.  Sizing to those would
        # freeze whatever it gave up; ask instead what it wanted.
        natural = getattr(bar, "natural_width", None)
        tabs_width = (
            natural()
            if callable(natural)
            else sum(bar.tabSizeHint(index).width() for index in range(bar.count()))
        )
        corner = self.info_tabs.cornerWidget(QtCore.Qt.TopRightCorner)
        corner_width = 0 if corner is None else corner.sizeHint().width()
        margins = self.layout().contentsMargins()
        needed = tabs_width + corner_width + margins.left() + margins.right()
        self._fixed_pane_width = max(label_pane_width, needed)
        self.setFixedWidth(self._fixed_pane_width)

    @QtCore.pyqtSlot()
    def _commit_path_draft(self) -> None:
        self.path_committed.emit(self.path_edit.text())

    def set_tabs(self, tabs: tuple[InfoTab, ...]) -> None:
        """Replace all tabs from plain ``(title, rows)`` values."""

        normalized = tuple(
            (str(title), tuple((str(label), value) for label, value in rows))
            for title, rows in tabs
        )
        titles = [title for title, _rows in normalized]
        if len(set(titles)) != len(titles):
            raise ValueError("info tab titles must be unique")
        # WHICH TAB THE OPERATOR IS READING is theirs, not the refresh's.
        # Every tab is destroyed and rebuilt below -- the rows change, the
        # titles almost never do -- and the rebuilt stack starts at the
        # first tab, so a refresh threw anyone reading Devices back to Plot.
        showing = self.info_tabs.tabText(self.info_tabs.currentIndex())
        while self.info_tabs.count():
            widget = self.info_tabs.widget(0)
            self.info_tabs.removeTab(0)
            if widget is not None:
                widget.deleteLater()
        self._rows_tabs.clear()
        self._graph_tabs.clear()
        self._graph_rows.clear()
        for title, rows in normalized:
            if title in self._graph_tab_titles:
                self._add_graph_tab(title)
            else:
                self._add_rows_tab(title, rows)
        for index in range(self.info_tabs.count()):
            if self.info_tabs.tabText(index) == showing:
                self.info_tabs.setCurrentIndex(index)
                break
        self._apply_pane_width()

    def _add_rows_tab(self, title: str, rows: tuple[InfoRow, ...]) -> None:
        tab = _RowsTab(rows, name_width=self._label_width)
        tab.tree.action_requested.connect(self.action_requested)
        self.info_tabs.add_permanent_tab(tab, title)
        self._rows_tabs[title] = tab

    def show_row(self, title: str, label: str) -> bool:
        """Open the tab ``title`` on its top-level row ``label``."""

        tab = self._rows_tabs.get(str(title))
        if tab is None:
            return False
        # A filter that hides the row would hide the answer to a question
        # the operator just asked of the picture.
        if tab.filter_edit.text():
            tab.filter_edit.clear()
        if not tab.tree.show_row(label):
            return False
        self.info_tabs.setCurrentWidget(tab)
        return True

    def _add_graph_tab(self, title: str) -> None:
        scene = QtWidgets.QGraphicsScene()
        view = _FlowView(scene)
        view.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        view.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing
        )
        view.setFrameShape(QtWidgets.QFrame.NoFrame)
        view.setBackgroundBrush(QtGui.QColor(SURFACE))
        view.setStyleSheet(f"QGraphicsView {{ border-top: 1px solid {DIVIDER}; }}")
        apply_fluent_scrollbars(view)
        view.node_activated.connect(
            lambda node_id, graph=title: self._activate_graph_node(graph, node_id)
        )
        self.info_tabs.add_permanent_tab(view, title)
        self._graph_tabs[title] = view

    def _activate_graph_node(self, graph: str, node_id: str) -> None:
        row = self._graph_rows.get(graph, {}).get(node_id)
        if row is not None:
            self.show_row(*row)

    def set_graph(self, title: str, graph: object) -> None:
        """Replace one node/edge graph from domain-free plain data.

        A node may name the row it stands for, ``("Logic", label)``: a click
        on its card opens that row, so the picture is a map of the tabs.
        """

        key = str(title)
        try:
            view = self._graph_tabs[key]
        except KeyError as error:
            raise KeyError(f"info pane has no graph tab {key!r}") from error
        if not isinstance(graph, Mapping) or set(graph) != {"nodes", "edges"}:
            raise TypeError("info graph must contain nodes and edges")
        raw_nodes, raw_edges = graph["nodes"], graph["edges"]
        if not isinstance(raw_nodes, tuple) or not isinstance(raw_edges, tuple):
            raise TypeError("info graph nodes and edges must be tuples")
        nodes: dict[str, Mapping[str, object]] = {}
        order: list[str] = []
        rows: dict[str, tuple[str, str]] = {}
        for raw in raw_nodes:
            if not isinstance(raw, Mapping) or set(raw) != {
                "id", "kind", "title", "subtitle", "root", "tooltip", "row"
            }:
                raise ValueError("info graph node fields differ")
            node_id = str(raw["id"])
            kind = str(raw["kind"])
            if not node_id or node_id in nodes or kind not in {"logic", "device"}:
                raise ValueError("info graph node identity is invalid")
            if type(raw["root"]) is not bool:
                raise TypeError("info graph node root flag must be bool")
            row = raw["row"]
            if row is not None:
                if (
                    not isinstance(row, tuple)
                    or len(row) != 2
                    or not all(isinstance(part, str) and part for part in row)
                ):
                    raise TypeError("info graph node row must be (tab, label) or None")
                rows[node_id] = (str(row[0]), str(row[1]))
            nodes[node_id] = raw
            order.append(node_id)
        edges: list[Mapping[str, object]] = []
        for raw in raw_edges:
            if not isinstance(raw, Mapping) or set(raw) != {
                "source", "target", "kind", "label"
            }:
                raise ValueError("info graph edge fields differ")
            source, target, kind = (
                str(raw["source"]), str(raw["target"]), str(raw["kind"])
            )
            if source not in nodes or target not in nodes or kind not in {
                "causal", "device"
            }:
                raise ValueError("info graph edge is invalid")
            if kind == "causal" and (
                nodes[source]["kind"] != "logic"
                or nodes[target]["kind"] != "logic"
            ):
                raise ValueError("causal graph edges must connect Logic nodes")
            if kind == "device" and (
                nodes[source]["kind"] != "device"
                or nodes[target]["kind"] != "logic"
            ):
                raise ValueError("device graph edges must point into Logic nodes")
            edges.append(raw)
        self._graph_rows[key] = rows

        scene = view.scene()
        assert scene is not None
        scene.clear()
        view._flow_node_rects = {}
        view._flow_edge_paths = ()
        view._flow_edge_count = 0
        if not nodes:
            empty = scene.addSimpleText("No saved Logic flow", QtGui.QFont(FONT, fluent_font_size()))
            empty.setBrush(QtGui.QColor(GREY))
            empty.setPos(scaled_px(18), scaled_px(18))
            scene.setSceneRect(empty.boundingRect().adjusted(-12, -12, 24, 24))
            return

        logic = [node_id for node_id in order if nodes[node_id]["kind"] == "logic"]
        causal = [edge for edge in edges if edge["kind"] == "causal"]
        predecessors = {node_id: [] for node_id in logic}
        successors = {node_id: [] for node_id in logic}
        for edge in causal:
            source, target = str(edge["source"]), str(edge["target"])
            predecessors[target].append(source)
            successors[source].append(target)
        pending = {node_id: len(predecessors[node_id]) for node_id in logic}
        ready = [node_id for node_id in logic if pending[node_id] == 0]
        topological: list[str] = []
        rank = {node_id: 1 for node_id in ready}
        while ready:
            node_id = ready.pop(0)
            topological.append(node_id)
            for target in successors[node_id]:
                rank[target] = max(rank.get(target, 1), rank[node_id] + 1)
                pending[target] -= 1
                if pending[target] == 0:
                    ready.append(target)
        if len(topological) != len(logic):
            raise ValueError("info graph contains a causal cycle")
        for node_id in order:
            if nodes[node_id]["kind"] == "device":
                rank[node_id] = 0

        card_height = scaled_px(62, minimum=50)
        horizontal_gap = scaled_px(16, minimum=12)
        vertical_gap = scaled_px(58, minimum=44)
        margin = scaled_px(18, minimum=14)
        layers: dict[int, list[str]] = {}
        for node_id in order:
            layers.setdefault(rank[node_id], []).append(node_id)
        def normalized_positions() -> dict[str, float]:
            return {
                node_id: (index + 0.5) / len(layer)
                for layer in layers.values()
                for index, node_id in enumerate(layer)
            }

        def reorder(layer_rank: int, *, from_predecessors: bool) -> None:
            layer = layers[layer_rank]
            positions = normalized_positions()
            original = {node_id: index for index, node_id in enumerate(layer)}

            def score(node_id: str) -> tuple[float, int]:
                neighbours = [
                    str(edge["source"] if from_predecessors else edge["target"])
                    for edge in edges
                    if str(edge["target"] if from_predecessors else edge["source"])
                    == node_id
                    and rank[
                        str(edge["source"] if from_predecessors else edge["target"])
                    ]
                    != layer_rank
                ]
                barycenter = (
                    sum(positions[item] for item in neighbours) / len(neighbours)
                    if neighbours
                    else positions[node_id]
                )
                return barycenter, original[node_id]

            layer.sort(key=score)

        layer_ranks = sorted(layers)
        for _sweep in range(2):
            for layer_rank in layer_ranks[1:]:
                reorder(layer_rank, from_predecessors=True)
            for layer_rank in reversed(layer_ranks[:-1]):
                reorder(layer_rank, from_predecessors=False)
        pane_margins = self.layout().contentsMargins()
        settled_view_width = (
            int(view.viewport().width())
            if self._fixed_pane_width is None
            else self._fixed_pane_width
            - pane_margins.left()
            - pane_margins.right()
            - scaled_px(8, minimum=6)
        )
        settled_view_width -= view.verticalScrollBar().sizeHint().width()
        available_width = max(
            scaled_px(220, minimum=190),
            settled_view_width - 2 * margin,
        )
        minimum_card_width = scaled_px(105, minimum=90)
        preferred_card_width = scaled_px(210, minimum=176)
        layer_geometry: dict[int, tuple[float, float]] = {}
        for layer_rank, layer in layers.items():
            card_width = max(
                minimum_card_width,
                min(
                    preferred_card_width,
                    (available_width - (len(layer) - 1) * horizontal_gap)
                    / len(layer),
                ),
            )
            layer_width = (
                len(layer) * card_width
                + max(0, len(layer) - 1) * horizontal_gap
            )
            layer_geometry[layer_rank] = card_width, layer_width
        widest = max(
            float(available_width),
            *(width for _card, width in layer_geometry.values()),
        )
        rects: dict[str, QtCore.QRectF] = {}
        next_y = float(margin)
        for layer_rank, layer in sorted(layers.items()):
            card_width, layer_width = layer_geometry[layer_rank]
            x = margin + (widest - layer_width) / 2.0
            for node_id in layer:
                rects[node_id] = QtCore.QRectF(
                    x, next_y, card_width, card_height
                )
                x += card_width + horizontal_gap
            next_y += card_height + vertical_gap

        long_edges = [
            edge
            for edge in edges
            if rank[str(edge["target"])] - rank[str(edge["source"])] > 1
        ]
        lane_base = margin + widest + scaled_px(22, minimum=16)
        lane_step = scaled_px(12, minimum=9)
        long_lanes = {id(edge): lane_base + index * lane_step for index, edge in enumerate(long_edges)}
        incoming: dict[str, list[int]] = {node_id: [] for node_id in nodes}
        outgoing: dict[str, list[int]] = {node_id: [] for node_id in nodes}
        edge_paths: list[tuple[str, str, QtGui.QPainterPath]] = []
        for edge_index, edge in enumerate(edges):
            outgoing[str(edge["source"])].append(edge_index)
            incoming[str(edge["target"])].append(edge_index)
        for _source, indices in outgoing.items():
            indices.sort(
                key=lambda index: rects[str(edges[index]["target"])].center().x()
            )
        for _target, indices in incoming.items():
            indices.sort(
                key=lambda index: rects[str(edges[index]["source"])].center().x()
            )
        for edge_index, edge in enumerate(edges):
            source, target = str(edge["source"]), str(edge["target"])
            source_rect, target_rect = rects[source], rects[target]
            source_edges = outgoing[source]
            target_edges = incoming[target]
            source_port = (source_edges.index(edge_index) + 1) / (
                len(source_edges) + 1
            )
            target_port = (target_edges.index(edge_index) + 1) / (
                len(target_edges) + 1
            )
            start = QtCore.QPointF(
                source_rect.left() + source_port * source_rect.width(),
                source_rect.bottom(),
            )
            end = QtCore.QPointF(
                target_rect.left() + target_port * target_rect.width(),
                target_rect.top(),
            )
            path = QtGui.QPainterPath(start)
            if id(edge) in long_lanes:
                lane = long_lanes[id(edge)]
                shoulder = scaled_px(16, minimum=12)
                path.lineTo(start.x(), start.y() + shoulder)
                path.lineTo(lane, start.y() + shoulder)
                path.lineTo(lane, end.y() - shoulder)
                path.lineTo(end.x(), end.y() - shoulder)
                path.lineTo(end)
            else:
                middle = (start.y() + end.y()) / 2.0
                path.cubicTo(start.x(), middle, end.x(), middle, end.x(), end.y())
            edge_paths.append((source, target, QtGui.QPainterPath(path)))
            is_device = edge["kind"] == "device"
            pen = QtGui.QPen(QtGui.QColor(ORANGE if is_device else GRAPHITE))
            pen.setWidthF(float(scaled_px(1.4, minimum=1)))
            if is_device:
                pen.setStyle(QtCore.Qt.DashLine)
            item = scene.addPath(path, pen)
            item.setZValue(-2)
            label = str(edge["label"])
            item.setToolTip(label or ("device use" if is_device else "causal input"))
            arrow = scaled_px(6, minimum=5)
            polygon = QtGui.QPolygonF(
                (
                    end,
                    QtCore.QPointF(end.x() - arrow, end.y() - arrow * 1.35),
                    QtCore.QPointF(end.x() + arrow, end.y() - arrow * 1.35),
                )
            )
            head = scene.addPolygon(polygon, pen, QtGui.QBrush(pen.color()))
            head.setZValue(-1)

        for node_id in order:
            node, rect = nodes[node_id], rects[node_id]
            is_device = node["kind"] == "device"
            outline = ORANGE if is_device else ACCENT if node["root"] else DIVIDER
            card_path = QtGui.QPainterPath()
            radius = scaled_px(5, minimum=4)
            card_path.addRoundedRect(rect, radius, radius)
            card = scene.addPath(
                card_path,
                QtGui.QPen(QtGui.QColor(outline), scaled_px(2 if node["root"] else 1)),
                QtGui.QBrush(QtGui.QColor(ORANGE_TINT if is_device else SURFACE)),
            )
            card.setToolTip(str(node["tooltip"]))
            # The card answers a click with its node; the words on it are
            # painted over it and take no click of their own.
            card.setData(0, node_id)
            if node_id in rows:
                card.setCursor(QtCore.Qt.PointingHandCursor)
            inset = scaled_px(10, minimum=8)
            badge_font = QtGui.QFont(FONT, max(6, fluent_font_size() - 4))
            badge_font.setBold(True)
            badge = scene.addSimpleText("DEVICE" if is_device else "LOGIC", badge_font)
            badge.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            badge.setBrush(QtGui.QColor(ORANGE_DARK if is_device else GREY))
            badge.setPos(rect.left() + inset, rect.top() + scaled_px(5, minimum=4))
            title_font = QtGui.QFont(FONT, max(8, fluent_font_size() - 1))
            title_font.setBold(True)
            metrics = QtGui.QFontMetrics(title_font)
            title = metrics.elidedText(
                str(node["title"]),
                QtCore.Qt.ElideMiddle,
                int(rect.width()) - 2 * inset,
            )
            title_item = scene.addSimpleText(title, title_font)
            title_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            title_item.setBrush(QtGui.QColor(TEXT))
            title_item.setPos(rect.left() + inset, rect.top() + scaled_px(19, minimum=15))
            detail_font = QtGui.QFont(FONT, max(7, fluent_font_size() - 3))
            detail_metrics = QtGui.QFontMetrics(detail_font)
            detail = detail_metrics.elidedText(
                str(node["subtitle"]),
                QtCore.Qt.ElideRight,
                int(rect.width()) - 2 * inset,
            )
            detail_item = scene.addSimpleText(detail, detail_font)
            detail_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            detail_item.setBrush(QtGui.QColor(GREY))
            detail_item.setPos(rect.left() + inset, rect.top() + scaled_px(40, minimum=32))

        scene_width = widest + 2 * margin
        if long_lanes:
            scene_width = max(scene_width, max(long_lanes.values()) + margin)
        scene_height = max(rect.bottom() for rect in rects.values()) + margin
        scene.setSceneRect(0, 0, scene_width, scene_height)
        view._flow_node_rects = dict(rects)
        view._flow_edge_paths = tuple(edge_paths)
        view._flow_edge_count = len(edges)

    def set_status(self, text: str) -> None:
        self.status.show_message(str(text))


__all__ = ["InfoPane", "InfoRow", "InfoTab", "InfoTree", "copy_text", "value_text"]
