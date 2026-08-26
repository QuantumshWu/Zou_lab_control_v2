"""The whole of what a task console offers the outside: one object, no widgets.

This window's seam was the worst of the four, and in an instructive way: the
outside was handed FACTORIES -- make a card, make a logic row, ask which signal
-- and then handed the built widgets back.  A composition root that constructs
widgets is assembling a UI whether or not the constructor came from somewhere
else, and every per-card wire (title, size, interval, remove, edit) was strung
across the wall one card at a time.

Here the cards and the rows belong to the window.  What crosses is which panels
exist, what each is called, and what each is showing; and back the other way,
what the operator did TO a named panel.  A drawn panel arrives as its host,
never as a widget, for the reason the other windows give: this package may not
import the package that draws.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import FluentDialogWindow, fluent_open_path, fluent_save_path

from .logic_row_view import LogicRowView
from .logic_editor_view import LogicEditorView
from .panel_card_view import PanelCardView
from .panel_editor_view import PanelEditorView
from .point_review_view import PointReviewView
from .signal_chooser import choose_signal
from .task_console_view import TaskConsoleView


class TaskConsoleHandle(QtCore.QObject):
    """One task console, as the outside sees it."""

    # -- the window ------------------------------------------------------
    close_requested = QtCore.pyqtSignal()
    closed = QtCore.pyqtSignal()

    # -- the board -------------------------------------------------------
    add_panel_requested = QtCore.pyqtSignal(str)
    add_logic_requested = QtCore.pyqtSignal(str)
    pause_toggled = QtCore.pyqtSignal(bool)
    selectors_toggled = QtCore.pyqtSignal(bool)
    save_layout_requested = QtCore.pyqtSignal()
    load_layout_requested = QtCore.pyqtSignal()
    save_screenshot_requested = QtCore.pyqtSignal()
    stop_task_requested = QtCore.pyqtSignal()
    panel_order_committed = QtCore.pyqtSignal(tuple)

    # -- one named panel -------------------------------------------------
    panel_remove_requested = QtCore.pyqtSignal(str)
    panel_edit_requested = QtCore.pyqtSignal(str)
    #: A refusal the panel's mounted plot surface reported.  The widget's
    #: ``errorOccurred`` was connected by nothing in the console, so pointer
    #: currency-guard refusals vanished; the card relays it here with the
    #: panel's name attached.
    panel_plot_error = QtCore.pyqtSignal(str, str)
    panel_state_changed = QtCore.pyqtSignal(str, object)
    panel_snapshot_refresh_requested = QtCore.pyqtSignal(str)
    panel_save_figure_requested = QtCore.pyqtSignal(str, str)
    panel_editor_closed = QtCore.pyqtSignal(str)

    # -- one named logic node --------------------------------------------
    logic_start_requested = QtCore.pyqtSignal(str)
    logic_auto_preview_changed = QtCore.pyqtSignal(str, bool)
    logic_stop_requested = QtCore.pyqtSignal(str)
    logic_edit_requested = QtCore.pyqtSignal(str)
    logic_remove_requested = QtCore.pyqtSignal(str)
    logic_draft_changed = QtCore.pyqtSignal(str, object)
    panel_publisher_edit_requested = QtCore.pyqtSignal(str)
    panel_publisher_draft_changed = QtCore.pyqtSignal(str, object)

    def __init__(
        self,
        window: Any,
        view: TaskConsoleView,
        *,
        plot_surface: Any | None = None,
    ) -> None:
        super().__init__()
        self._window = window
        self._view = view
        dpr_target = window if window is not None else view
        self._device_pixel_ratio = float(dpr_target.devicePixelRatioF())
        native_window = dpr_target.windowHandle()
        if native_window is not None:
            native_window.screenChanged.connect(self._screen_changed)
        # The composition root's panel-widget policy: host -> QWidget.  The
        # console injects a staging widget factory here so the board can
        # present same-shot groups atomically; without one the host supplies
        # its own default widget.  Widgets are reused per host (a host owns
        # one surface), matching the host's own widget caching.
        self._plot_surface = plot_surface if callable(plot_surface) else None
        self._cards: dict[str, PanelCardView] = {}
        self._rows: dict[str, LogicRowView] = {}
        self._panel_publisher_rows: dict[str, LogicRowView] = {}
        self._logic_editors: dict[str, LogicEditorView] = {}
        self._panel_publisher_editors: dict[str, LogicEditorView] = {}
        self._panel_editors: dict[str, PanelEditorView] = {}
        self._panel_intervals: tuple[int, ...] = ()
        self._panel_default_interval = 0
        self._panel_sizes: tuple[str, ...] = ()
        self._panel_default_size = ""
        self._grid_cell_kinds: tuple[str, ...] = ()
        self._task_takeover = False
        for name in (
            "add_panel_requested", "add_logic_requested", "pause_toggled",
            "selectors_toggled", "save_layout_requested",
            "load_layout_requested", "save_screenshot_requested",
            "stop_task_requested", "panel_order_committed",
        ):
            getattr(view, name).connect(getattr(self, name))
        if hasattr(view, "close_requested"):
            view.close_requested.connect(self.close_requested)
        if window is not None and hasattr(window, "closed"):
            window.closed.connect(self.closed)
        view.editor_close_requested.connect(self._editor_close_requested)

    # ------------------------------------------------------------ the window

    def close(self) -> None:
        if self._window is not None:
            self._window.close()
        elif hasattr(self._view, "finish_close"):
            self._view.finish_close()

    def close_later(self) -> None:
        """Retry the top-level close after the current owner turn finishes."""

        QtCore.QTimer.singleShot(0, self.close)

    @property
    def owner_window(self):
        """The window anything opened FROM this console belongs to.

        Named here because ownership is the console's to state: a window it
        opens is not a peer that the desktop may stack anything between.
        """

        return self._window

    def set_close_guard(self, guard) -> None:
        """Refuse the close until the host says its workers are down.

        A console owns a camera, a display beat and a session; closing the
        window before those let go leaves the process alive with nothing on
        screen.  The guard answers True only once they have, and the window
        stays up for another try otherwise -- so what a host must supply is a
        promise, not a widget, and it belongs on the port like everything else.
        """

        if self._window is not None:
            self._window.set_close_guard(guard)

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

    # ------------------------------------------------------------- the board

    def set_panel_kinds(self, kinds: tuple[tuple[str, str], ...], current: str = "") -> None:
        self._view.set_panel_kinds(kinds, current)

    def set_panel_intervals(self, intervals: object, default_interval: int) -> None:
        """Project the scheduler's finite refresh policy to every panel view."""

        values = tuple(int(value) for value in tuple(intervals or ()))
        if not values:
            raise ValueError("panel interval choices must not be empty")
        default = int(default_interval)
        if default not in values:
            raise ValueError("default panel interval must be one of the choices")
        self._panel_intervals = values
        self._panel_default_interval = default
        for card in self._cards.values():
            card.set_interval_choices(values, default)

    def set_panel_sizes(self, sizes: object, default_size: str) -> None:
        """Project the plotting owner's finite size policy to every panel view."""

        values = tuple(str(value) for value in tuple(sizes or ()))
        if not values or len(set(values)) != len(values):
            raise ValueError("panel size choices must be unique and non-empty")
        default = str(default_size)
        if default not in values:
            raise ValueError("default panel size must be one of the choices")
        self._panel_sizes = values
        self._panel_default_size = default
        for card in self._cards.values():
            card.set_size_choices(values, default)

    def set_grid_cell_kinds(self, kinds: object) -> None:
        """Project the grid cell vocabulary to every panel's settings control."""

        values = tuple(str(value) for value in tuple(kinds or ()))
        if not values or len(set(values)) != len(values):
            raise ValueError("grid cell kinds must be unique and non-empty")
        self._grid_cell_kinds = values
        for card in self._cards.values():
            card.set_cell_kind_choices(values)

    def set_logic_kinds(
        self, kinds: tuple[tuple[str, str, str, str], ...]
    ) -> None:
        self._view.set_logic_kinds(kinds)

    def set_paused(self, paused: bool) -> None:
        self._view.set_paused(paused)

    def set_selectors(self, enabled: bool) -> None:
        self._view.set_selectors(enabled)

    def set_summary(self, text: str) -> None:
        self._view.set_summary(text)

    def show_status(self, text: str, severity: str) -> None:
        self._view.show_status(text, severity)

    def set_task_takeover(self, active: bool) -> None:
        """Project one application admission gate onto every existing view."""

        self._task_takeover = bool(active)
        self._view.set_task_takeover(self._task_takeover)
        for row in self._rows.values():
            row.set_task_takeover(self._task_takeover)
        for editor in self._logic_editors.values():
            editor.set_mutation_enabled(not self._task_takeover)

    def choose_signal(self, rows) -> str | None:
        """Ask which signal to show; None when the operator declines."""

        return choose_signal(rows, self._view)

    def ask_save_path(self, caption: str, suggested: str, filter: str) -> str:
        """Ask where to write, offering a name; "" when they decline."""

        return fluent_save_path(self._view, caption, suggested, filter)

    def ask_open_path(self, caption: str, start: str, filter: str) -> str:
        """Ask for a file to open; "" when the operator declines."""

        return fluent_open_path(self._view, caption, start, filter)

    def save_screenshot(self, path: str) -> str:
        """Write one screenshot of the complete TaskConsole window."""

        target = self._window if self._window is not None else self._view
        pixmap = target.grab()
        if pixmap.isNull() or not pixmap.save(str(path), "PNG"):
            raise OSError(f"could not save TaskConsole screenshot to {path!r}")
        return str(path)

    def show_warning(self, title: str, text: str) -> None:
        """Say what was refused, in the one modal this project owns."""

        from ..fluent import fluent_message

        fluent_message(self._view, str(title), str(text), kind="warning")

    def review_points(
        self,
        surface,
        points,
        *,
        title: str,
        message: str = "",
        confirm_label: str = "Continue",
        initial_excluded=(),
    ) -> tuple[str, ...] | None:
        """Run the complete Fluent point-review view around one plot QWidget."""

        try:
            review = PointReviewView(
                surface,
                points,
                message=str(message),
                confirm_label=str(confirm_label),
                initial_excluded=tuple(initial_excluded),
            )
        except BaseException:
            surface.close_adapter()
            raise
        surface.toggle_requested.connect(review.toggle_point)
        surface.selection_requested.connect(review.select_points)
        review.state_changed.connect(surface.set_state)
        surface.set_state(review.excluded_ids, review.selected_ids)
        parent = self._window if self._window is not None else self._view
        try:
            result = review.exec_(parent, title=str(title))
            return (
                review.excluded_ids
                if result == FluentDialogWindow.Accepted
                else None
            )
        finally:
            surface.close_adapter()

    def manual_axis_setting(self, *, title: str, message: str) -> bool:
        """Stand aside while a hand moves a knob; False when it declines."""

        from ..fluent import fluent_confirm

        parent = self._window if self._window is not None else self._view
        return bool(
            fluent_confirm(
                parent,
                str(title),
                str(message),
                confirm_text="Continue",
                cancel_text="Stop",
                kind="info",
            )
        )

    # -------------------------------------------------------------- panels

    def add_panel(self, panel_id: str, title: str) -> None:
        """Put one panel on the board, and wire it to this port.

        The card is made here.  It used to be made outside and passed in, so
        every one of its six intents had to be re-wired by whoever made it --
        six wires per panel across the wall, and a panel whose maker forgot one
        was a control that silently did nothing.
        """

        key = str(panel_id)
        card = self._cards.get(key)
        if card is None:
            card = PanelCardView(key, str(title))
            if not self._panel_sizes:
                raise RuntimeError("panel size choices were not projected")
            card.set_size_choices(self._panel_sizes, self._panel_default_size)
            card.remove_requested.connect(
                lambda _=None, pid=key: self.panel_remove_requested.emit(pid)
            )
            card.edit_requested.connect(
                lambda _=None, pid=key: self.panel_edit_requested.emit(pid)
            )
            card.state_changed.connect(
                lambda patch, pid=key: self.panel_state_changed.emit(pid, patch)
            )
            card.plot_error.connect(
                lambda message, pid=key: self.panel_plot_error.emit(
                    pid, str(message)
                )
            )
            self._cards[key] = card
            if self._panel_intervals:
                card.set_interval_choices(
                    self._panel_intervals,
                    self._panel_default_interval,
                )
            if self._grid_cell_kinds:
                card.set_cell_kind_choices(self._grid_cell_kinds)
            card.set_editing_enabled(True)
        self._view.set_cards(tuple(self._cards.values()))

    def remove_panel(self, panel_id: str) -> None:
        key = str(panel_id)
        self.close_panel_editor(key)
        self._cards.pop(key, None)
        self._view.set_cards(tuple(self._cards.values()))

    def panel_ids(self) -> tuple[str, ...]:
        return tuple(self._cards)

    def set_panel_order(self, order) -> None:
        """Put the cards in this order, which IS the panel order.

        Where the cards are decides what a saved figure contains and in what
        sequence.  With the cards on this side of the wall, saying "the panels
        are in this order" and saying "the cards are" became two statements,
        and only the first was being made -- a board that rearranged itself
        back on the next redraw.
        """

        wanted = [str(panel_id) for panel_id in order if str(panel_id) in self._cards]
        wanted += [key for key in self._cards if key not in wanted]
        self._cards = {key: self._cards[key] for key in wanted}
        self._view.set_cards(tuple(self._cards.values()))

    def show_panel(self, panel_id: str, host: Any | None) -> None:
        """Draw what this host holds on a named panel, given the host itself."""

        card = self._cards[str(panel_id)]
        card.set_surface(None if host is None else self._surface_for(card, host))

    def _surface_for(self, card: PanelCardView, host: Any) -> Any:
        """One widget per host: reuse the mounted surface when it is his."""

        current = card.surface
        if current is not None and getattr(current, "host", None) is host:
            return current
        if self._plot_surface is not None:
            return self._plot_surface(host)
        return host.qt_widget()

    def present_panel_front(self, panel_id: str, front: Any) -> bool:
        """Put one completed immutable front on a panel's staged widget.

        Duck-typed like the interaction gate: a plot widget owns
        ``present_front``, a placeholder standing in for one does not.  The
        return says whether pixels actually changed (a stale front is refused
        by the widget's own monotonic-sequence gate).
        """

        card = self._cards.get(str(panel_id))
        surface = None if card is None else card.surface
        present = getattr(surface, "present_front", None)
        if not callable(present):
            return False
        return bool(present(front))

    def set_panel_signal_choices(self, panel_id: str, *args: Any, **kwargs: Any) -> None:
        self._cards[str(panel_id)].set_signal_choices(*args, **kwargs)

    def set_panel_projection(
        self,
        panel_id: str,
        state: object,
        parameter_surface: object,
    ) -> None:
        """Project one atomic Setting state/schema replacement."""

        self._cards[str(panel_id)].set_panel_projection(state, parameter_surface)

    def set_panel_status(self, panel_id: str, text: str, *, error: bool) -> None:
        self._cards[str(panel_id)].set_status(text, error=error)

    def set_panel_mutation_enabled(self, panel_id: str, enabled: bool) -> None:
        """Combine one panel's owner gate with the application Task gate."""

        key = str(panel_id)
        effective = bool(enabled)
        card = self._cards.get(key)
        if card is not None:
            card.set_editing_enabled(effective)
        editor = self._panel_editors.get(key)
        if editor is not None:
            editor.set_mutation_enabled(effective)

    def set_panel_selectors_enabled(self, panel_id: str, enabled: bool) -> None:
        key = str(panel_id)
        self._cards[key].set_selectors_enabled(enabled)
        editor = self._panel_editors.get(key)
        if editor is not None:
            editor.set_selectors_enabled(enabled)

    # --------------------------------------------------------- panel editors

    def open_panel_editor(self, panel_id: str, projection: Any) -> None:
        key = str(panel_id)
        incoming = dict(projection)
        incoming["interval_choices"] = self._panel_intervals
        incoming["size_choices"] = self._panel_sizes
        editor = self._panel_editors.get(key)
        if editor is None:
            editor = PanelEditorView(key, incoming)
            editor.set_mutation_enabled(True)
            editor.state_changed.connect(
                lambda patch, pid=key: self.panel_state_changed.emit(pid, patch)
            )
            editor.snapshot_refresh_requested.connect(
                lambda _=None, pid=key: self.panel_snapshot_refresh_requested.emit(pid)
            )
            editor.producer_edit_requested.connect(
                lambda node_id: self.logic_edit_requested.emit(str(node_id))
            )
            editor.save_figure_requested.connect(
                lambda path, pid=key: self.panel_save_figure_requested.emit(pid, str(path))
            )
            self._panel_editors[key] = editor
            state = incoming.get("state") or {}
            title = (
                str(dict(state).get("title") or key)
                if isinstance(state, Mapping)
                else key
            )
            self._view.add_editor_tab(editor, f"Edit Panel · {title}")
        else:
            editor.update_projection(incoming)
            self._view.focus_editor_tab(editor)
        editor.set_selectors_enabled(self._cards[key].selectors_enabled)

    def update_panel_editor(self, panel_id: str, projection: Any) -> bool:
        editor = self._panel_editors.get(str(panel_id))
        if editor is None:
            return False
        incoming = dict(projection)
        incoming["interval_choices"] = self._panel_intervals
        incoming["size_choices"] = self._panel_sizes
        editor.update_projection(incoming)
        return True

    def show_panel_editor(self, panel_id: str, host: Any | None) -> None:
        """Mount a plotting host without exposing its QWidget to Workbench."""

        editor = self._panel_editors[str(panel_id)]
        editor.set_surface(None if host is None else host.qt_widget())

    def focus_panel_editor(self, panel_id: str) -> bool:
        editor = self._panel_editors.get(str(panel_id))
        return False if editor is None else self._view.focus_editor_tab(editor)

    def close_panel_editor(self, panel_id: str) -> bool:
        editor = self._panel_editors.pop(str(panel_id), None)
        if editor is None:
            return False
        editor.set_surface(None)
        return self._view.remove_editor_tab(editor)

    # ---------------------------------------------------------- logic rows

    def _show_logic_rows(self) -> None:
        self._view.set_logic_rows(
            tuple(self._rows.values()) + tuple(self._panel_publisher_rows.values())
        )

    def add_logic_row(
        self, node_id: str, kind: str, offers_preview: bool = True
    ) -> None:
        key = str(node_id)
        row = self._rows.get(key)
        if row is None:
            row = LogicRowView(key, str(kind))
            row.start_requested.connect(
                lambda _=None, nid=key: self.logic_start_requested.emit(nid)
            )
            row.set_preview_offered(offers_preview)
            row.auto_preview_changed.connect(
                lambda enabled, nid=key: (
                    self.logic_auto_preview_changed.emit(nid, bool(enabled))
                )
            )
            row.stop_requested.connect(
                lambda _=None, nid=key: self.logic_stop_requested.emit(nid)
            )
            row.edit_requested.connect(
                lambda _=None, nid=key: self.logic_edit_requested.emit(nid)
            )
            row.remove_requested.connect(
                lambda _=None, nid=key: self.logic_remove_requested.emit(nid)
            )
            self._rows[key] = row
            row.set_task_takeover(self._task_takeover)
        self._show_logic_rows()

    def remove_logic_row(self, node_id: str) -> None:
        key = str(node_id)
        self.close_logic_editor(key)
        row = self._rows.pop(key, None)
        if row is not None:
            row.setParent(None)
        self._show_logic_rows()

    def logic_row_ids(self) -> tuple[str, ...]:
        return tuple(self._rows)

    def set_logic_state(self, node_id: str, state: str, status_text: str = "") -> None:
        self._rows[str(node_id)].set_state(state, status_text)

    def set_logic_commands(
        self, node_id: str, *, can_start: bool, can_stop: bool
    ) -> None:
        self._rows[str(node_id)].set_commands(
            can_start=can_start,
            can_stop=can_stop,
        )

    def set_logic_auto_preview(self, node_id: str, enabled: bool) -> None:
        self._rows[str(node_id)].set_auto_preview(bool(enabled))

    def set_logic_publishes(self, node_id: str, rows: tuple[tuple[str, str, str], ...]) -> None:
        self._rows[str(node_id)].set_publishes(rows)

    def set_panel_publishers(
        self,
        publishers: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...],
    ) -> None:
        """Show panel-owned signals beside, not inside, LogicNode rows."""

        incoming: dict[str, LogicRowView] = {}
        for panel_id, published in publishers:
            key = str(panel_id)
            row = self._panel_publisher_rows.get(key)
            if row is None:
                row = LogicRowView(key, "panel publisher")
                row.dot.hide()
                row.status_label.hide()
                for button in (
                    row.start_button,
                    # No node behind this row, so nothing here can be started
                    # and nothing decides what a Start would open.
                    row.preview_switch,
                    row.stop_button,
                    row.remove_button,
                ):
                    button.hide()
                row.edit_requested.connect(
                    lambda _=None, pid=key: self.panel_publisher_edit_requested.emit(pid)
                )
            row.set_publishes(published)
            row.edit_button.setEnabled(True)
            incoming[key] = row
        for key, row in tuple(self._panel_publisher_rows.items()):
            if key not in incoming:
                self.close_panel_publisher_editor(key)
                row.setParent(None)
        self._panel_publisher_rows = incoming
        self._show_logic_rows()

    def open_panel_publisher_editor(self, panel_id: str, projection: Any) -> None:
        key = str(panel_id)
        editor = self._panel_publisher_editors.get(key)
        if editor is None:
            editor = LogicEditorView(key, projection, show_actions=False)
            editor.draft_changed.connect(
                lambda patch, pid=key: self.panel_publisher_draft_changed.emit(pid, patch)
            )
            self._panel_publisher_editors[key] = editor
            title = str(dict(projection).get("api_name") or key)
            self._view.add_editor_tab(editor, f"Edit · {title}")
        else:
            editor.update_projection(projection)
            self._view.focus_editor_tab(editor)
        editor.set_mutation_enabled(
            not bool(dict(projection).get("science_locked"))
        )

    def update_panel_publisher_editor(self, panel_id: str, projection: Any) -> bool:
        editor = self._panel_publisher_editors.get(str(panel_id))
        if editor is None:
            return False
        editor.update_projection(projection)
        editor.set_mutation_enabled(
            not bool(dict(projection).get("science_locked"))
        )
        return True

    def has_panel_publisher_editor(self, panel_id: str) -> bool:
        return str(panel_id) in self._panel_publisher_editors

    def focus_panel_publisher_editor(self, panel_id: str) -> bool:
        editor = self._panel_publisher_editors.get(str(panel_id))
        return False if editor is None else self._view.focus_editor_tab(editor)

    def close_panel_publisher_editor(self, panel_id: str) -> bool:
        editor = self._panel_publisher_editors.pop(str(panel_id), None)
        return False if editor is None else self._view.remove_editor_tab(editor)

    # --------------------------------------------------------- logic editors

    def open_logic_editor(self, node_id: str, projection: Any) -> None:
        key = str(node_id)
        editor = self._logic_editors.get(key)
        if editor is None:
            editor = LogicEditorView(key, projection)
            editor.set_mutation_enabled(not self._task_takeover)
            editor.draft_changed.connect(
                lambda patch, nid=key: self.logic_draft_changed.emit(nid, patch)
            )
            editor.start_requested.connect(
                lambda _=None, nid=key: self.logic_start_requested.emit(nid)
            )
            # The row's switch and the editor's are two views of one binding,
            # so they raise the SAME intent rather than each owning a state.
            editor.auto_preview_changed.connect(
                lambda enabled, nid=key: (
                    self.logic_auto_preview_changed.emit(nid, bool(enabled))
                )
            )
            editor.stop_requested.connect(
                lambda _=None, nid=key: self.logic_stop_requested.emit(nid)
            )
            editor.remove_requested.connect(
                lambda _=None, nid=key: self.logic_remove_requested.emit(nid)
            )
            self._logic_editors[key] = editor
            title = str(dict(projection).get("api_name") or key)
            self._view.add_editor_tab(editor, f"Edit · {title}")
        else:
            editor.update_projection(projection)
            self._view.focus_editor_tab(editor)

    def update_logic_editor(self, node_id: str, projection: Any) -> bool:
        editor = self._logic_editors.get(str(node_id))
        if editor is None:
            return False
        editor.update_projection(projection)
        return True

    def has_logic_editor(self, node_id: str) -> bool:
        return str(node_id) in self._logic_editors

    def focus_logic_editor(self, node_id: str) -> bool:
        editor = self._logic_editors.get(str(node_id))
        return False if editor is None else self._view.focus_editor_tab(editor)

    def close_logic_editor(self, node_id: str) -> bool:
        editor = self._logic_editors.pop(str(node_id), None)
        return False if editor is None else self._view.remove_editor_tab(editor)

    def _editor_close_requested(self, editor: object) -> None:
        for node_id, candidate in tuple(self._logic_editors.items()):
            if candidate is editor:
                self.close_logic_editor(node_id)
                return
        for panel_id, candidate in tuple(self._panel_publisher_editors.items()):
            if candidate is editor:
                self.close_panel_publisher_editor(panel_id)
                return
        for panel_id, candidate in tuple(self._panel_editors.items()):
            if candidate is editor:
                self.panel_editor_closed.emit(panel_id)
                self.close_panel_editor(panel_id)
                return


__all__ = ["TaskConsoleHandle"]
