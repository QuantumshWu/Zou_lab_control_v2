"""Generic read-only information surface with an optional path field."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from pprint import pformat

from PyQt5 import QtCore, QtWidgets

from .fluent import (
    FluentFrame,
    FluentPathEdit,
    FluentReadoutMultiline,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentStatusStrip,
    FluentTabWidget,
    apply_fluent_scrollbars,
    fluent_font_size,
    scaled_px,
    setting_label_width,
    window_pad,
)
from .style import ACCENT, CARD_PAD, DIVIDER, FONT, SURFACE, TEXT


InfoRow = tuple[str, object]
InfoTab = tuple[str, tuple[InfoRow, ...]]


class InfoPane(QtWidgets.QWidget):
    """A reusable path header plus injected read-only tabs.

    The pane does not know what the rows represent.  A tab is supplied as
    ``(title, ((label, value), ...))`` and values are formatted only for
    display.  Path and status copy are also injected so the widget can be
    reused by different presenters without importing their data model.
    """

    path_committed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        label_names: Iterable[str],
        tabs: tuple[InfoTab, ...] = (),
        path_label: str = "Path",
        path_caption: str = "Choose a path",
        file_filter: str = "All files (*)",
        initial_status: str = "",
        tree_tabs: Iterable[str] = (),
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        # The declared names set the pane's one stable split.  Archive labels
        # are content, not geometry authority: loading another file must not
        # move the divider or resize the top-level window.
        self._declared_labels = tuple(str(name) for name in label_names)
        self._label_width = setting_label_width(self._declared_labels)
        self._tree_tab_titles = frozenset(str(title) for title in tree_tabs)
        self._tree_tabs: dict[str, QtWidgets.QTreeWidget] = {}
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
        )
        self.path_edit.selected.connect(self.path_committed.emit)
        self.path_edit.edit.editingFinished.connect(self._commit_path_draft)
        header.addWidget(self.path_edit, 1)
        layout.addWidget(header_frame)

        self.info_tabs = FluentTabWidget()
        self.info_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self._tab_layouts: dict[str, QtWidgets.QVBoxLayout] = {}
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
        while self.info_tabs.count():
            widget = self.info_tabs.widget(0)
            self.info_tabs.removeTab(0)
            if widget is not None:
                widget.deleteLater()
        self._tab_layouts.clear()
        self._tree_tabs.clear()
        for title, rows in normalized:
            if title in self._tree_tab_titles:
                self._add_tree_tab(title)
            else:
                self._add_rows_tab(title, rows)
        self._apply_pane_width()

    def _add_rows_tab(self, title: str, rows: tuple[InfoRow, ...]) -> None:
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        body.setStyleSheet("background: transparent;")
        tab_layout = QtWidgets.QVBoxLayout(body)
        margin = scaled_px(CARD_PAD, minimum=6)
        tab_layout.setContentsMargins(margin, margin, margin, margin)
        tab_layout.setSpacing(scaled_px(3, minimum=2))
        tab_layout.setAlignment(QtCore.Qt.AlignTop)
        self._fill_rows(tab_layout, rows)
        scroll.setWidget(body)
        self.info_tabs.add_permanent_tab(scroll, title)
        self._tab_layouts[title] = tab_layout

    def _fill_rows(self, layout: QtWidgets.QVBoxLayout, rows: tuple[InfoRow, ...]) -> None:
        for key, value in rows:
            text = self._readout_text(value)
            field = FluentReadoutMultiline(text)
            field.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Fixed,
            )
            layout.addWidget(
                FluentSettingRow(
                    str(key),
                    field,
                    label_width=self._label_width,
                )
            )

    def _add_tree_tab(self, title: str) -> None:
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderHidden(True)
        tree.setRootIsDecorated(True)
        tree.setUniformRowHeights(True)
        tree.setIndentation(scaled_px(18, minimum=14))
        tree.setStyleSheet(
            f"QTreeWidget {{ background: {SURFACE}; color: {TEXT}; "
            f"border: none; font: {fluent_font_size()}pt '{FONT}'; }}"
            f"QTreeWidget::item {{ padding: {scaled_px(3, minimum=2)}px; }}"
            f"QTreeWidget::item:selected {{ background: {ACCENT}; color: white; }}"
            f"QTreeWidget {{ border-top: 1px solid {DIVIDER}; }}"
        )
        apply_fluent_scrollbars(tree)
        self.info_tabs.add_permanent_tab(tree, title)
        self._tree_tabs[title] = tree

    def set_tree(self, title: str, branches: object) -> None:
        """Replace one real expandable tree from plain `(label, children)` data."""

        key = str(title)
        try:
            tree = self._tree_tabs[key]
        except KeyError as error:
            raise KeyError(f"info pane has no tree tab {key!r}") from error
        if not isinstance(branches, tuple):
            raise TypeError("info tree branches must be a tuple")
        tree.clear()

        def add(parent: QtWidgets.QTreeWidgetItem | None, values: tuple) -> None:
            for branch in values:
                if not isinstance(branch, tuple) or len(branch) != 2:
                    raise ValueError("info tree branch must contain label and children")
                label, children = branch
                if not isinstance(label, str) or not isinstance(children, tuple):
                    raise TypeError("info tree branch label/children are malformed")
                item = QtWidgets.QTreeWidgetItem((label,))
                if parent is None:
                    tree.addTopLevelItem(item)
                else:
                    parent.addChild(item)
                add(item, children)
                item.setExpanded(True)

        add(None, branches)

    def set_status(self, text: str) -> None:
        self.status.show_message(str(text))

    @staticmethod
    def _readout_text(value: object) -> str:
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, Mapping):
            return pformat(dict(value), sort_dicts=False, width=80)
        if isinstance(value, (tuple, list)):
            return ", ".join(
                str(item.value if isinstance(item, Enum) else item)
                for item in value
            )
        return str(value)


__all__ = ["InfoPane", "InfoRow", "InfoTab"]
