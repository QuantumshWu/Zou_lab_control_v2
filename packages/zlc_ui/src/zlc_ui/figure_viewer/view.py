"""Pure saved-figure browser shell.

The presenter owns archive IO, metadata projection and the plot widget.  This
view only provides the file/path intent, generic info projection, and an
atomic QWidget mount point for the presenter-owned surface.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.console.panel_card_view import PanelCardView
from zlc_ui.fluent import (
    ACCENT,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentLabel,
    InfoPane,
    scaled_px,
    window_pad,
)


class FigureViewerView(QtWidgets.QWidget):
    path_committed = QtCore.pyqtSignal(str)
    dataset_picked = QtCore.pyqtSignal(str)
    save_image_requested = QtCore.pyqtSignal()
    close_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("FigureViewerView")
        self.setStyleSheet("background: transparent;")
        self._figure_surface: QtWidgets.QWidget | None = None
        self._retiring_surfaces: set[QtWidgets.QWidget] = set()
        self._closing = False
        root = QtWidgets.QHBoxLayout(self)
        # The InfoPane owns the left/right window inset, exactly like the v1
        # FigureInfoPane.  The host supplies only the top/bottom frame.
        root.setContentsMargins(0, window_pad(1), 0, window_pad(1))
        root.setSpacing(window_pad(0.5))

        self.info_pane = InfoPane(
            # These are the v1 formal projection keys.  They are also what
            # determines the fixed left-pane width, so using invented labels
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
            parent=self,
        )
        self.info_pane.path_committed.connect(self.path_committed)
        root.addWidget(self.info_pane, 0)

        # The v1 host is a transparent QWidget.  The presenter-owned surface
        # supplies its own paint/border when mounted; the empty shell must not
        # invent a second card around it.
        holder = QtWidgets.QWidget(self)
        holder.setStyleSheet("background: transparent;")
        holder.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._surface_layout = QtWidgets.QVBoxLayout(holder)
        self._surface_layout.setContentsMargins(0, 0, window_pad(1), 0)
        self._surface_layout.setSpacing(window_pad(0.5))

        # Panel Save Fig writes one dataset, while notebook-created archives may
        # contain several.  Keep the generic viewer picker without suggesting
        # that TaskConsole packages the whole board.
        self._dataset_bar = QtWidgets.QWidget(holder)
        self._dataset_bar.setStyleSheet("background: transparent;")
        bar_layout = QtWidgets.QHBoxLayout(self._dataset_bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(window_pad(0.5))
        # Which dataset is drawn is asked ON THE CARD now, where every other
        # per-figure decision already is.  A second picker up here would be the
        # same question in two places, and the card's is the one that also
        # carries the size, the title and the way into the plot's own controls.
        bar_layout.addStretch(1)
        self.save_image_button = FluentButton("Save image", color=ACCENT)
        self.save_image_button.clicked.connect(self.save_image_requested)
        self.save_image_button.setEnabled(False)
        bar_layout.addWidget(self.save_image_button)
        self._dataset_bar.hide()
        self._surface_layout.addWidget(self._dataset_bar)
        # The v1 empty surface is a white FluentFrame inside the transparent
        # host.  Putting the label directly on the host leaks the outer
        # window's grey background through the whole plot area and creates a
        # large, misleading outer-background mismatch.
        self._placeholder = FluentFrame(parent=holder)
        placeholder_layout = QtWidgets.QVBoxLayout(self._placeholder)
        placeholder_layout.addStretch(1)
        placeholder_label = FluentLabel("Open a saved figure to begin", self._placeholder)
        placeholder_label.setAlignment(QtCore.Qt.AlignCenter)
        placeholder_layout.addWidget(placeholder_label)
        placeholder_layout.addStretch(1)
        self._surface_layout.addWidget(self._placeholder, 1)
        # The figure is a PANEL now, not a bare picture: it carries its own
        # signal picker, size, title and the way into the plot's controls --
        # the same card the console uses, in the mode that admits a saved file
        # will not deliver again.
        self.figure_card = PanelCardView("figure", "figure", parent=holder)
        self.figure_card.set_live(False)
        self.figure_card.signal_picked.connect(self.dataset_picked)
        self.figure_card.hide()
        self._surface_layout.addWidget(self.figure_card, 1)
        root.addWidget(holder, 1)

    def set_info(self, tabs: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]) -> None:
        self.info_pane.set_tabs(tabs)

    def set_datasets(
        self, datasets: tuple[tuple[str, str], ...], current: str = ""
    ) -> None:
        """Offer the datasets this archive holds, and say which is drawn.

        Each entry is (key, label): the key is what the archive stores and what
        this emits back, the label is what the operator reads.  They used to be
        the same string, which meant a saved console figure offered "panel-1"
        and "panel-2" -- an operator had to guess which of them was the camera.

        Hidden when there is nothing to choose between: one dataset needs no
        picker, and none means nothing is open.
        """

        incoming = tuple((str(key), str(label or key)) for key, label in datasets)
        # Grouped the way the card asks for them: one archive is one source.
        self.figure_card.set_signal_choices(
            (("this archive", tuple((label, key) for key, label in incoming)),)
            if incoming
            else (),
            current=str(current),
        )
        self._dataset_bar.setVisible(bool(incoming))
        self.save_image_button.setEnabled(bool(incoming))

    @property
    def dataset_combo(self):
        """The card's picker.  Kept as a name so a test can still find it."""

        return self.figure_card.signal_combo

    def set_figure_size(self, size: str) -> None:
        self.figure_card.set_panel_size(str(size))

    def set_figure_surface(self, widget: QtWidgets.QWidget | None) -> None:
        if widget is not None and not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("figure surface must be QWidget or None")
        previous = self._figure_surface
        if widget is previous:
            if widget is not None:
                widget.show()
            return
        # The new surface is installed before the old one is retired, so a
        # presenter can swap a finished canvas without a white intermediate.
        # Into the CARD, which is what carries the picker, the size and the
        # way into the plot's own controls.  It used to go straight into the
        # column as a bare picture, and a bare picture is all it could ever be.
        self.figure_card.set_surface(widget)
        self._figure_surface = widget
        self.figure_card.setVisible(widget is not None)
        self._placeholder.setVisible(widget is None)

    def set_status(self, text: str, *, error: bool = False) -> None:
        self.info_pane.set_status(str(text))
        if error:
            self.info_pane.status.show_message(str(text), severity="error")

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
