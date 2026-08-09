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

from PyQt5 import QtCore

from .logic_row_view import LogicRowView
from .logic_editor_view import LogicEditorView
from .panel_card_view import PanelCardView
from .panel_editor_view import PanelEditorView
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
    panel_state_changed = QtCore.pyqtSignal(str, object)
    panel_snapshot_refresh_requested = QtCore.pyqtSignal(str)
    panel_producer_apply_requested = QtCore.pyqtSignal(str)
    panel_save_figure_requested = QtCore.pyqtSignal(str)
    panel_editor_closed = QtCore.pyqtSignal(str)

    # -- one named logic node --------------------------------------------
    logic_start_requested = QtCore.pyqtSignal(str)
    logic_stop_requested = QtCore.pyqtSignal(str)
    logic_edit_requested = QtCore.pyqtSignal(str)
    logic_remove_requested = QtCore.pyqtSignal(str)
    logic_draft_changed = QtCore.pyqtSignal(str, object)

    def __init__(self, window: Any, view: TaskConsoleView) -> None:
        super().__init__()
        self._window = window
        self._view = view
        self._cards: dict[str, PanelCardView] = {}
        self._rows: dict[str, LogicRowView] = {}
        self._logic_editors: dict[str, LogicEditorView] = {}
        self._panel_editors: dict[str, PanelEditorView] = {}
        self._panel_intervals: tuple[int, ...] = ()
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
        if hasattr(self._view, "finish_close"):
            self._view.finish_close()
        if self._window is not None:
            self._window.close()

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

    # ------------------------------------------------------------- the board

    def set_panel_kinds(self, kinds: tuple[tuple[str, str], ...], current: str = "") -> None:
        self._view.set_panel_kinds(kinds, current)

    def set_panel_intervals(self, intervals: object) -> None:
        """Project the scheduler's finite refresh policy to every panel view."""

        values = tuple(int(value) for value in tuple(intervals or ()))
        if not values:
            raise ValueError("panel interval choices must not be empty")
        self._panel_intervals = values
        for card in self._cards.values():
            card.set_interval_choices(values)

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
        editing = not self._task_takeover
        self._view.set_task_takeover(self._task_takeover)
        for card in self._cards.values():
            card.set_editing_enabled(editing)
        for row in self._rows.values():
            row.set_task_takeover(self._task_takeover)
        for editor in self._logic_editors.values():
            editor.set_mutation_enabled(editing)
        for editor in self._panel_editors.values():
            editor.set_mutation_enabled(editing)

    def choose_signal(self, rows) -> str | None:
        """Ask which signal to show; None when the operator declines."""

        return choose_signal(rows, self._view)

    def edit_values(self, spec, values, *, title: str):
        """Show a modal form over a spec, and return what was set."""

        from ..form import edit_values as _edit_values

        return _edit_values(spec, values, self._view, title=str(title))

    def ask_save_path(self, caption: str, start_dir: str, filter: str) -> str:
        """Ask where to write; "" when the operator declines."""

        from PyQt5 import QtWidgets

        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self._view, str(caption), str(start_dir), str(filter)
        )
        return str(path)

    def ask_open_path(self, caption: str, start_dir: str, filter: str) -> str:
        """Ask for a file to open; "" when the operator declines."""

        from PyQt5 import QtWidgets

        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self._view, str(caption), str(start_dir), str(filter)
        )
        return str(path)

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

    def run_host_dialog(self, opener, host, *, title: str):
        """Run a dialog the DRAWING package owns, centred on this window.

        The opener is a function, not a widget: this package may not import
        the one that draws, and the outside may not hold the widget that
        would parent its dialog.  So the function crosses inward and is given
        this window to sit on, and nothing crosses the other way.
        """

        return opener(host, self._view, title=str(title))

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
            if self._panel_intervals:
                card.set_interval_choices(self._panel_intervals)
            card.set_editing_enabled(not self._task_takeover)
        self._view.set_cards(tuple(self._cards.values()))

    def remove_panel(self, panel_id: str) -> None:
        key = str(panel_id)
        self.close_panel_editor(key)
        card = self._cards.pop(key, None)
        if card is not None:
            card.setParent(None)
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
        card.set_surface(None if host is None else host.qt_widget())

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
        effective = bool(enabled) and not self._task_takeover
        card = self._cards.get(key)
        if card is not None:
            card.set_editing_enabled(effective)
        editor = self._panel_editors.get(key)
        if editor is not None:
            editor.set_mutation_enabled(effective)

    def set_panel_selectors_enabled(self, panel_id: str, enabled: bool) -> None:
        self._cards[str(panel_id)].set_selectors_enabled(enabled)

    # --------------------------------------------------------- panel editors

    def open_panel_editor(self, panel_id: str, projection: Any) -> None:
        key = str(panel_id)
        incoming = dict(projection)
        incoming["interval_choices"] = self._panel_intervals
        editor = self._panel_editors.get(key)
        if editor is None:
            editor = PanelEditorView(key, incoming)
            editor.set_mutation_enabled(not self._task_takeover)
            editor.state_changed.connect(
                lambda patch, pid=key: self.panel_state_changed.emit(pid, patch)
            )
            editor.snapshot_refresh_requested.connect(
                lambda _=None, pid=key: self.panel_snapshot_refresh_requested.emit(pid)
            )
            editor.producer_draft_changed.connect(
                lambda node_id, patch: self.logic_draft_changed.emit(str(node_id), patch)
            )
            editor.producer_apply_requested.connect(
                lambda _=None, pid=key: self.panel_producer_apply_requested.emit(pid)
            )
            editor.save_figure_requested.connect(
                lambda _=None, pid=key: self.panel_save_figure_requested.emit(pid)
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

    def update_panel_editor(self, panel_id: str, projection: Any) -> bool:
        editor = self._panel_editors.get(str(panel_id))
        if editor is None:
            return False
        incoming = dict(projection)
        incoming["interval_choices"] = self._panel_intervals
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

    def add_logic_row(self, node_id: str, kind: str) -> None:
        key = str(node_id)
        row = self._rows.get(key)
        if row is None:
            row = LogicRowView(key, str(kind))
            row.start_requested.connect(
                lambda _=None, nid=key: self.logic_start_requested.emit(nid)
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
        self._view.set_logic_rows(tuple(self._rows.values()))

    def remove_logic_row(self, node_id: str) -> None:
        key = str(node_id)
        self.close_logic_editor(key)
        row = self._rows.pop(key, None)
        if row is not None:
            row.setParent(None)
        self._view.set_logic_rows(tuple(self._rows.values()))

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

    def set_logic_publishes(self, node_id: str, rows: tuple[tuple[str, str, str], ...]) -> None:
        self._rows[str(node_id)].set_publishes(rows)

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

    def focus_logic_editor(self, node_id: str) -> bool:
        editor = self._logic_editors.get(str(node_id))
        return False if editor is None else self._view.focus_editor_tab(editor)

    def close_logic_editor(self, node_id: str) -> bool:
        editor = self._logic_editors.pop(str(node_id), None)
        return False if editor is None else self._view.remove_editor_tab(editor)

    def _editor_close_requested(self, editor: object) -> None:
        if self._task_takeover:
            self._view.focus_editor_tab(editor)
            return
        for node_id, candidate in tuple(self._logic_editors.items()):
            if candidate is editor:
                self.close_logic_editor(node_id)
                return
        for panel_id, candidate in tuple(self._panel_editors.items()):
            if candidate is editor:
                self.panel_editor_closed.emit(panel_id)
                self.close_panel_editor(panel_id)
                return


__all__ = ["TaskConsoleHandle"]
