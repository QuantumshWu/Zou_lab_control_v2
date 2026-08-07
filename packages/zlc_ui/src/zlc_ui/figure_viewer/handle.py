"""The whole of what a figure viewer offers the outside: one object, no widgets.

Same rule as every window here, and for the same reason: a host that can hold
the view can reach into it, and then what is on screen has two owners.  This
one had a second way in as well -- the presenter was handed a `surface_of`
callable so it could turn a drawing host into a QWidget itself, which is a
composition root building UI with an extra step in front of it.

The host arrives whole and is asked for its widget here, where widgets belong.
"""

from __future__ import annotations

from typing import Any

from PyQt5 import QtCore

from .view import FigureViewerView


class FigureViewerHandle(QtCore.QObject):
    """One figure viewer, as the outside sees it."""

    # -- the window ------------------------------------------------------
    close_requested = QtCore.pyqtSignal()
    closed = QtCore.pyqtSignal()

    # -- what the operator asks for --------------------------------------
    path_committed = QtCore.pyqtSignal(str)
    dataset_picked = QtCore.pyqtSignal(str)
    save_image_requested = QtCore.pyqtSignal()
    #: The figure is a PANEL now: its size and its title are decisions about
    #: this figure, asked on the card where every other one already is.
    figure_size_picked = QtCore.pyqtSignal(str)
    figure_title_committed = QtCore.pyqtSignal(str)
    figure_edit_requested = QtCore.pyqtSignal()

    def __init__(self, window: Any, view: FigureViewerView) -> None:
        super().__init__()
        self._window = window
        self._view = view
        view.path_committed.connect(self.path_committed)
        view.dataset_picked.connect(self.dataset_picked)
        view.save_image_requested.connect(self.save_image_requested)
        view.close_requested.connect(self.close_requested)
        view.figure_card.size_picked.connect(self.figure_size_picked)
        view.figure_card.title_committed.connect(self.figure_title_committed)
        view.figure_card.edit_requested.connect(self.figure_edit_requested)
        if window is not None and hasattr(window, "closed"):
            window.closed.connect(self.closed)

    # ------------------------------------------------------------ the window

    def close(self) -> None:
        self._view.finish_close()
        if self._window is not None:
            self._window.close()

    def finish_close(self) -> None:
        self._view.finish_close()

    def is_visible(self) -> bool:
        target = self._window if self._window is not None else self._view
        return bool(target.isVisible())

    def window_size(self) -> tuple[int, int]:
        target = self._window if self._window is not None else self._view
        size = target.size()
        return int(size.width()), int(size.height())

    def window_title(self) -> str:
        target = self._window if self._window is not None else self._view
        return str(target.windowTitle())

    # ------------------------------------------------------------- the page

    def set_title(self, text: str) -> None:
        self._view.set_title(text)

    def set_path(self, path: str) -> None:
        self._view.set_path(path)

    def set_status(self, text: str, *, error: bool = False) -> None:
        self._view.set_status(text, error=error)

    def set_info(self, tabs: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]) -> None:
        self._view.set_info(tabs)

    def set_datasets(self, datasets: tuple[tuple[str, str], ...], current: str = "") -> None:
        self._view.set_datasets(datasets, current)

    def run_host_dialog(self, opener, host, *, title: str):
        """Run a dialog the DRAWING package owns, centred on this window."""

        return opener(host, self._view, title=str(title))

    def set_figure_title(self, text: str) -> None:
        self._view.set_figure_title(text)

    def set_figure_size(self, size: str) -> None:
        self._view.set_figure_size(size)

    def show_figure(self, host: Any | None) -> None:
        """Draw what this host is holding, or clear the page when None.

        The host makes its own widget.  A callable that turned one into the
        other used to be injected from outside, which put the composition root
        back in the business of constructing Qt objects -- one level of
        indirection away from doing it by hand, and no further from the wall.
        """

        self._view.set_figure_surface(None if host is None else host.qt_widget())


__all__ = ["FigureViewerHandle"]
