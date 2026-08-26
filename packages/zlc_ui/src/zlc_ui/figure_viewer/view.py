"""Pure saved-figure browser shell.

The presenter owns archive IO, metadata projection and the plot widget.  This
view only provides the file/path intent, generic info projection, and an
atomic QWidget mount point for the presenter-owned surface.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.console.board_view import ConsoleBoardView
from zlc_ui.console.panel_card_view import PanelCardView
from zlc_ui.console.panel_editor_view import PanelEditorView
from zlc_ui.fluent import (
    ACCENT,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentLabel,
    FluentScrollArea,
    FluentSectionLabel,
    FluentTabWidget,
    InfoPane,
    window_pad,
)


class FigureViewerView(QtWidgets.QWidget):
    path_committed = QtCore.pyqtSignal(str)
    add_panel_requested = QtCore.pyqtSignal(str)
    panel_state_changed = QtCore.pyqtSignal(str, object)
    panel_remove_requested = QtCore.pyqtSignal(str)
    panel_edit_requested = QtCore.pyqtSignal(str)
    panel_order_committed = QtCore.pyqtSignal(tuple)
    panel_editor_closed = QtCore.pyqtSignal(str)
    save_image_requested = QtCore.pyqtSignal()
    close_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("FigureViewerView")
        self.setStyleSheet("background: transparent;")
        self._cards: dict[str, PanelCardView] = {}
        self._editors: dict[str, QtWidgets.QWidget] = {}
        self._panel_sizes: tuple[str, ...] = ()
        self._panel_default_size = ""
        self._grid_cell_kinds: tuple[str, ...] = ()
        self._closing = False
        self._info_tabs: tuple = ()
        self._lineage_tree: tuple = ()
        root = QtWidgets.QHBoxLayout(self)
        # InfoPane owns the left/right window inset.  The host supplies only
        # the top/bottom frame.
        root.setContentsMargins(0, window_pad(1), 0, window_pad(1))
        root.setSpacing(window_pad(0.5))

        self.info_pane = InfoPane(
            # These formal projection keys also determine the fixed left-pane
            # width, so using invented labels
            # changes the entire FigureViewer split even while the pane looks
            # otherwise correct.
            label_names=("schema_fingerprint", "coordinate_frame"),
            tabs=(
                ("Plot", ()),
                ("Measurement", ()),
                ("Device", ()),
                ("Flow", ()),
                ("Raw", ()),
            ),
            path_label="File",
            path_caption="Open a saved figure archive (.npz)",
            file_filter="Saved figure archives (*.npz)",
            initial_status="Open a current saved Figure (.npz).",
            tree_tabs=("Flow",),
            parent=self,
        )
        self.info_pane.path_committed.connect(self.path_committed)
        root.addWidget(self.info_pane, 0)

        # The right half is one white Fluent work surface.  Cards remain
        # visibly separate inside it instead of dissolving into the grey
        # top-level background.
        holder = FluentFrame(parent=self, bordered=False)
        holder.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._surface_layout = QtWidgets.QVBoxLayout(holder)
        self._surface_layout.setContentsMargins(
            window_pad(0.75), window_pad(0.5), window_pad(0.75), window_pad(0.75)
        )
        self._surface_layout.setSpacing(window_pad(0.5))

        self._dataset_bar = QtWidgets.QWidget(holder)
        self._dataset_bar.setStyleSheet("background: transparent;")
        self._dataset_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        bar_layout = QtWidgets.QHBoxLayout(self._dataset_bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(window_pad(0.5))
        bar_layout.addWidget(FluentSectionLabel("Saved panels"))
        self.kind_combo = FluentComboBox()
        self.kind_combo.setMinimumContentsLength(12)
        bar_layout.addWidget(self.kind_combo)
        self.add_panel_button = FluentButton("Add panel", color=ACCENT)
        self.add_panel_button.clicked.connect(self._add_selected_panel)
        self.add_panel_button.setEnabled(False)
        bar_layout.addWidget(self.add_panel_button)
        bar_layout.addStretch(1)
        self.save_image_button = FluentButton("Save image", color=ACCENT)
        self.save_image_button.clicked.connect(self.save_image_requested)
        self.save_image_button.setEnabled(False)
        bar_layout.addWidget(self.save_image_button)
        self._surface_layout.addWidget(self._dataset_bar)

        self.tabs = FluentTabWidget(holder)
        self.tabs.tab_close_requested.connect(self._editor_close_clicked)
        monitor = QtWidgets.QWidget()
        monitor.setStyleSheet("background: white;")
        monitor_layout = QtWidgets.QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(0, 0, 0, 0)
        self.board = ConsoleBoardView()
        self.board.order_committed.connect(self.panel_order_committed)
        self.scroll = FluentScrollArea()
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setWidget(self.board)
        monitor_layout.addWidget(self.scroll, 1)
        self.tabs.add_permanent_tab(monitor, "Monitor")
        self._surface_layout.addWidget(self.tabs, 1)

        self._placeholder = FluentFrame(parent=holder)
        placeholder_layout = QtWidgets.QVBoxLayout(self._placeholder)
        placeholder_layout.addStretch(1)
        placeholder_label = FluentLabel("Open a saved figure to begin", self._placeholder)
        placeholder_label.setAlignment(QtCore.Qt.AlignCenter)
        placeholder_layout.addWidget(placeholder_label)
        placeholder_layout.addStretch(1)
        monitor_layout.addWidget(self._placeholder, 1)
        root.addWidget(holder, 1)
        self._sync_monitor_empty()

    def set_info(self, tabs: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]) -> None:
        self._info_tabs = tuple(tabs)
        self._render_info()

    def set_lineage_tree(self, tree: object) -> None:
        """Show an exact causal tree through the view's plain-data seam."""

        if not isinstance(tree, tuple):
            raise TypeError("lineage tree must be a tuple")
        self._lineage_tree = tree
        # The tree has its own stable widget.  Rebuilding all five info tabs
        # after set_info() just to populate Flow caused a visible second flash
        # and discarded the user's current tab for no data-model reason.
        self.info_pane.set_tree("Flow", tree)

    def _render_info(self) -> None:
        rows = dict(self._info_tabs)
        self.info_pane.set_tabs(tuple(
            (title, () if title == "Flow" else tuple(rows.get(title, ())))
            for title in ("Plot", "Measurement", "Device", "Flow", "Raw")
        ))
        self.info_pane.set_tree("Flow", self._lineage_tree)

    def set_panel_kinds(self, kinds: object) -> None:
        rows = tuple((str(key), str(label or key)) for key, label in tuple(kinds))
        current = self.kind_combo.currentData()
        self.kind_combo.clear()
        for key, label in rows:
            self.kind_combo.addItem(label, key)
        if current is not None:
            index = self.kind_combo.findData(current)
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
        self.add_panel_button.setEnabled(bool(rows))

    def set_panel_sizes(self, sizes: object, default_size: str) -> None:
        self._panel_sizes = tuple(str(value) for value in tuple(sizes))
        self._panel_default_size = str(default_size)
        for card in self._cards.values():
            card.set_size_choices(self._panel_sizes, self._panel_default_size)

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
        if not self._panel_sizes:
            raise RuntimeError("FigureViewer panel sizes were not projected")
        card = PanelCardView(key, str(title))
        card.set_size_choices(self._panel_sizes, self._panel_default_size)
        if self._grid_cell_kinds:
            card.set_cell_kind_choices(self._grid_cell_kinds)
        card.remove_requested.connect(
            lambda _=None, pid=key: self.panel_remove_requested.emit(pid)
        )
        card.edit_requested.connect(
            lambda _=None, pid=key: self.panel_edit_requested.emit(pid)
        )
        card.state_changed.connect(
            lambda patch, pid=key: self.panel_state_changed.emit(pid, patch)
        )
        self._cards[key] = card
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

    def set_panel_datasets(
        self,
        panel_id: str,
        datasets: tuple[tuple[str, str], ...],
        current: str = "",
    ) -> None:
        incoming = tuple((str(key), str(label or key)) for key, label in datasets)
        self._cards[str(panel_id)].set_signal_choices(
            (("this archive", tuple((label, key) for key, label in incoming)),)
            if incoming
            else (),
            current=str(current),
        )

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

    def set_panel_status(self, panel_id: str, text: str, *, error: bool) -> None:
        self._cards[str(panel_id)].set_status(str(text), error=bool(error))

    def _sync_monitor_empty(self) -> None:
        has_panels = bool(self._cards)
        self.scroll.setVisible(has_panels)
        self._placeholder.setVisible(not has_panels)
        self.save_image_button.setEnabled(has_panels)

    def open_panel_editor(
        self, panel_id: str, projection: object, title: str
    ) -> None:
        key = str(panel_id)
        existing = self._editors.get(key)
        if existing is not None:
            self.tabs.setCurrentWidget(existing)
            return
        incoming = dict(projection)
        incoming["size_choices"] = self._panel_sizes
        editor = PanelEditorView(key, incoming)
        editor.state_changed.connect(
            lambda patch, pid=key: self.panel_state_changed.emit(pid, patch)
        )
        self._editors[key] = editor
        self.tabs.add_closable_tab(editor, str(title), focus=True)

    def close_panel_editor(self, panel_id: str) -> bool:
        key = str(panel_id)
        editor = self._editors.pop(key, None)
        if editor is None:
            return False
        index = self.tabs.indexOf(editor)
        if index >= 0:
            self.tabs.removeTab(index)
        editor.setParent(None)
        editor.deleteLater()
        return True

    def update_panel_editor(self, panel_id: str, projection: object) -> bool:
        editor = self._editors.get(str(panel_id))
        if not isinstance(editor, PanelEditorView):
            return False
        incoming = dict(projection)
        incoming["size_choices"] = self._panel_sizes
        editor.update_projection(incoming)
        return True

    def has_panel_editor(self, panel_id: str) -> bool:
        return str(panel_id) in self._editors

    def _editor_close_clicked(self, editor: QtWidgets.QWidget) -> None:
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
