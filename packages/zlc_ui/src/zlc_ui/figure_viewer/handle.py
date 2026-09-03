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
    panel_plot_error = QtCore.pyqtSignal(str, str)
    save_image_requested = QtCore.pyqtSignal()

    def __init__(self, window: Any, view: FigureViewerView) -> None:
        super().__init__()
        self._window = window
        self._view = view
        dpr_target = window if window is not None else view
        self._device_pixel_ratio = float(dpr_target.devicePixelRatioF())
        native_window = dpr_target.windowHandle()
        if native_window is not None:
            native_window.screenChanged.connect(self._screen_changed)
        view.path_committed.connect(self.path_committed)
        view.new_data_requested.connect(self.new_data_requested)
        view.edit_data_requested.connect(self.edit_data_requested)
        view.data_editor_intent.connect(self.data_editor_intent)
        view.data_editor_closed.connect(self.data_editor_closed)
        view.add_panel_requested.connect(self.add_panel_requested)
        view.panel_state_changed.connect(self.panel_state_changed)
        view.panel_remove_requested.connect(self.panel_remove_requested)
        view.panel_edit_requested.connect(self.panel_edit_requested)
        view.panel_order_committed.connect(self.panel_order_committed)
        view.panel_editor_closed.connect(self.panel_editor_closed)
        view.panel_plot_error.connect(self.panel_plot_error)
        view.save_image_requested.connect(self.save_image_requested)
        view.close_requested.connect(self.close_requested)
        if window is not None and hasattr(window, "closed"):
            window.closed.connect(self.closed)

    # ------------------------------------------------------------ the window

    def close(self) -> None:
        if self._window is not None:
            self._window.close()
        else:
            self._view.finish_close()

    def close_later(self) -> None:
        """Retry the guarded close after the current owner callback returns."""

        QtCore.QTimer.singleShot(0, self.close)

    def set_close_guard(self, guard) -> None:
        if self._window is None or not hasattr(self._window, "set_close_guard"):
            raise RuntimeError("figure viewer has no top-level close guard")
        self._window.set_close_guard(guard)

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

    def device_pixel_ratio(self) -> float:
        """The latest GUI-owned screen scale, safe for a projection worker."""

        return self._device_pixel_ratio

    def _screen_changed(self, screen: object) -> None:
        ratio = getattr(screen, "devicePixelRatio", None)
        if callable(ratio):
            self._device_pixel_ratio = float(ratio())

    # ------------------------------------------------------------- the page

    def set_title(self, text: str) -> None:
        self._view.set_title(text)

    def set_path(self, path: str) -> None:
        self._view.set_path(path)

    def set_status(self, text: str, *, error: bool = False) -> None:
        self._view.set_status(text, error=error)

    def set_archive_info(
        self,
        tabs: tuple[tuple[str, tuple[tuple[str, object], ...]], ...],
        graph: object,
    ) -> None:
        self._view.set_archive_info(tabs, graph)

    def set_panel_sizes(self, sizes: object, default_size: str) -> None:
        self._view.set_panel_sizes(sizes, default_size)

    def set_editable_data_choices(
        self, choices: object, *, current: str = ""
    ) -> None:
        self._view.set_editable_data_choices(choices, current=current)

    def open_data_editor(
        self,
        editor_id: str,
        projection: object,
        *,
        title: str = "",
    ) -> None:
        self._view.open_data_editor(
            editor_id,
            projection,
            str(title or f"Data · {editor_id}"),
        )

    def close_data_editor(self, editor_id: str) -> bool:
        return self._view.close_data_editor(editor_id)

    def update_data_editor(self, editor_id: str, projection: object) -> bool:
        return self._view.update_data_editor(editor_id, projection)

    def has_data_editor(self, editor_id: str) -> bool:
        return self._view.has_data_editor(editor_id)

    def focus_data_editor(self, editor_id: str) -> bool:
        return self._view.focus_data_editor(editor_id)

    def set_panel_kinds(self, kinds: object, default_kind: str = "") -> None:
        self._view.set_panel_kinds(kinds, default_kind)

    def set_panel_intervals(
        self, intervals: object, default_interval: int
    ) -> None:
        self._view.set_panel_intervals(intervals, default_interval)

    def set_grid_cell_kinds(self, kinds: object) -> None:
        self._view.set_grid_cell_kinds(kinds)

    def add_panel(self, panel_id: str, title: str) -> None:
        self._view.add_panel(panel_id, title)

    def remove_panel(self, panel_id: str) -> None:
        self._view.remove_panel(panel_id)

    def set_panel_order(self, order: object) -> None:
        self._view.set_panel_order(order)

    def set_panel_signal_choices(self, panel_id: str, *args, **kwargs) -> None:
        self._view.set_panel_signal_choices(panel_id, *args, **kwargs)

    def set_panel_publishers(self, publishers: object) -> None:
        self._view.set_panel_publishers(publishers)

    def panel_ids(self) -> tuple[str, ...]:
        return self._view.panel_ids()

    def set_panel_selectors_enabled(self, panel_id: str, enabled: bool) -> None:
        self._view.set_panel_selectors_enabled(panel_id, enabled)

    def set_panel_mutation_enabled(self, panel_id: str, enabled: bool) -> None:
        self._view.set_panel_mutation_enabled(panel_id, enabled)

    def present_panel_front(self, panel_id: str, front: object) -> bool:
        return self._view.present_panel_front(panel_id, front)

    def set_panel_projection(
        self, panel_id: str, state: object, surface: object
    ) -> None:
        self._view.set_panel_projection(panel_id, state, surface)

    def set_panel_status(
        self, panel_id: str, text: str, *, error: bool = False
    ) -> None:
        self._view.set_panel_status(panel_id, text, error=error)

    def show_panel(self, panel_id: str, host: Any | None) -> None:
        self._view.set_panel_surface(
            panel_id, None if host is None else host.qt_widget()
        )

    def open_panel_editor(
        self, panel_id: str, projection: Any, *, title: str = ""
    ) -> None:
        state = dict(projection).get("state", {}) if isinstance(projection, dict) else {}
        resolved_title = str(title or f"Edit Panel · {dict(state).get('title', panel_id)}")
        self._view.open_panel_editor(panel_id, projection, resolved_title)

    def close_panel_editor(self, panel_id: str) -> bool:
        return self._view.close_panel_editor(panel_id)

    def update_panel_editor(self, panel_id: str, projection: object) -> bool:
        return self._view.update_panel_editor(panel_id, projection)

    def has_panel_editor(self, panel_id: str) -> bool:
        return self._view.has_panel_editor(panel_id)

    def show_panel_editor(self, panel_id: str, host: Any | None) -> None:
        self._view.show_panel_editor(
            panel_id,
            None if host is None else host.qt_widget(),
        )

    def focus_panel_editor(self, panel_id: str) -> bool:
        return self._view.focus_panel_editor(panel_id)


__all__ = ["FigureViewerHandle"]
