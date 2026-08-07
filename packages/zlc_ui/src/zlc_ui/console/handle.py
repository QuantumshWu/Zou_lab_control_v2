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

from typing import Any

from PyQt5 import QtCore

from .logic_row_view import LogicRowView
from .panel_card_view import PanelCardView
from .signal_chooser import choose_signal
from .task_console_view import TaskConsoleView


class TaskConsoleHandle(QtCore.QObject):
    """One task console, as the outside sees it."""

    # -- the window ------------------------------------------------------
    close_requested = QtCore.pyqtSignal()
    closed = QtCore.pyqtSignal()

    # -- the board -------------------------------------------------------
    add_panel_requested = QtCore.pyqtSignal(str)
    add_logic_requested = QtCore.pyqtSignal()
    pause_toggled = QtCore.pyqtSignal(bool)
    selectors_toggled = QtCore.pyqtSignal(bool)
    save_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()
    save_image_requested = QtCore.pyqtSignal()
    save_board_requested = QtCore.pyqtSignal()
    load_board_requested = QtCore.pyqtSignal()
    panel_order_committed = QtCore.pyqtSignal(tuple)

    # -- one named panel -------------------------------------------------
    panel_signal_picked = QtCore.pyqtSignal(str, str)
    panel_size_picked = QtCore.pyqtSignal(str, str)
    panel_update_ms_picked = QtCore.pyqtSignal(str, int)
    panel_title_committed = QtCore.pyqtSignal(str, str)
    panel_remove_requested = QtCore.pyqtSignal(str)
    panel_edit_requested = QtCore.pyqtSignal(str)

    # -- one named logic node --------------------------------------------
    logic_start_requested = QtCore.pyqtSignal(str)
    logic_stop_requested = QtCore.pyqtSignal(str)
    logic_edit_requested = QtCore.pyqtSignal(str)
    logic_remove_requested = QtCore.pyqtSignal(str)

    def __init__(self, window: Any, view: TaskConsoleView) -> None:
        super().__init__()
        self._window = window
        self._view = view
        self._cards: dict[str, PanelCardView] = {}
        self._rows: dict[str, LogicRowView] = {}
        for name in (
            "add_panel_requested", "add_logic_requested", "pause_toggled",
            "selectors_toggled", "save_requested", "load_requested",
            "save_image_requested", "save_board_requested",
            "load_board_requested", "panel_order_committed",
        ):
            getattr(view, name).connect(getattr(self, name))
        if hasattr(view, "close_requested"):
            view.close_requested.connect(self.close_requested)
        if window is not None and hasattr(window, "closed"):
            window.closed.connect(self.closed)

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

    def set_paused(self, paused: bool) -> None:
        self._view.set_paused(paused)

    def set_selectors(self, enabled: bool) -> None:
        self._view.set_selectors(enabled)

    def set_summary(self, text: str) -> None:
        self._view.set_summary(text)

    def show_status(self, text: str, severity: str) -> None:
        self._view.show_status(text, severity)

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
            card.signal_picked.connect(
                lambda name, pid=key: self.panel_signal_picked.emit(pid, str(name))
            )
            card.size_picked.connect(
                lambda size, pid=key: self.panel_size_picked.emit(pid, str(size))
            )
            card.update_ms_picked.connect(
                lambda ms, pid=key: self.panel_update_ms_picked.emit(pid, int(ms))
            )
            card.title_committed.connect(
                lambda text, pid=key: self.panel_title_committed.emit(pid, str(text))
            )
            card.remove_requested.connect(
                lambda _=None, pid=key: self.panel_remove_requested.emit(pid)
            )
            card.edit_requested.connect(
                lambda _=None, pid=key: self.panel_edit_requested.emit(pid)
            )
            self._cards[key] = card
        self._view.set_cards(tuple(self._cards.values()))

    def remove_panel(self, panel_id: str) -> None:
        card = self._cards.pop(str(panel_id), None)
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

    def set_panel_update_ms(self, panel_id: str, interval_ms: int) -> None:
        self._cards[str(panel_id)].set_update_ms(interval_ms)

    def set_panel_size(self, panel_id: str, size: str) -> None:
        self._cards[str(panel_id)].set_panel_size(size)

    def set_panel_status(self, panel_id: str, text: str, *, error: bool) -> None:
        self._cards[str(panel_id)].set_status(text, error=error)

    def set_panel_selectors_enabled(self, panel_id: str, enabled: bool) -> None:
        self._cards[str(panel_id)].set_selectors_enabled(enabled)

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
        self._view.set_logic_rows(tuple(self._rows.values()))

    def remove_logic_row(self, node_id: str) -> None:
        row = self._rows.pop(str(node_id), None)
        if row is not None:
            row.setParent(None)
        self._view.set_logic_rows(tuple(self._rows.values()))

    def logic_row_ids(self) -> tuple[str, ...]:
        return tuple(self._rows)

    def set_logic_state(self, node_id: str, state: str, status_text: str = "") -> None:
        self._rows[str(node_id)].set_state(state, status_text)

    def set_logic_publishes(self, node_id: str, rows: tuple[tuple[str, str, str], ...]) -> None:
        self._rows[str(node_id)].set_publishes(rows)


__all__ = ["TaskConsoleHandle"]
