"""Pure v1-style TaskConsole chrome.

The presenter still owns cards, logic rows and persistence.  This module only
recreates the v1 window shell: the flat 48 px header, persistent status strip,
and the permanent ``Monitor`` / ``Logic`` tabs.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT,
    ElidedLabel,
    GREEN,
    GREY,
    ORANGE,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentLabel,
    FluentLineEdit,
    FluentScrollArea,
    FluentStatusDot,
    FluentSwitch,
    FluentTabWidget,
    fluent_widget_stylesheet,
    scaled_px,
    window_pad,
)

from .board_view import ConsoleBoardView
from .logic_row_view import LogicRowView
from .panel_card_view import PanelCardView
from .status_strip import StatusStrip


class TaskConsoleView(QtWidgets.QWidget):
    """A presenter-friendly shell with the original v1 arrangement."""

    #: The v1 header has one combined chooser and one Add Panel button.  These
    #: two signals are the typed result of that one visible action.
    add_panel_requested = QtCore.pyqtSignal(str)
    add_logic_requested = QtCore.pyqtSignal(str)
    pause_toggled = QtCore.pyqtSignal(bool)
    selectors_toggled = QtCore.pyqtSignal(bool)
    #: Two deliberately separate header products: a stopped, reusable pipeline
    #: layout and one ordinary screenshot of the current GUI.  Scientific
    #: figure data is saved from a panel's Edit tab, never from this header.
    save_layout_requested = QtCore.pyqtSignal()
    load_layout_requested = QtCore.pyqtSignal()
    save_screenshot_requested = QtCore.pyqtSignal()
    #: Re-raised from the board, because a presenter talks to the view it was
    #: given.  Where the operator put the cards is where the panels ARE, and a
    #: figure saved in a different order is a figure of a board nobody saw.
    panel_order_committed = QtCore.pyqtSignal(tuple)
    editor_close_requested = QtCore.pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("TaskConsoleView")
        self.setWindowTitle("TaskConsole@Zou lab")
        self.setStyleSheet(fluent_widget_stylesheet())

        outer = QtWidgets.QVBoxLayout(self)
        margin = window_pad(1)
        outer.setContentsMargins(margin, margin, margin, margin)
        outer.setSpacing(window_pad(0.5))

        # One semantic row, as in v1: identity on the left, telemetry as the
        # only flexible middle cell, and commands pinned to the right.  The
        # telemetry elides when space is tight; commands never wrap into a
        # second pseudo-toolbar whose alignment depends on label size hints.
        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(scaled_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(scaled_px(12), scaled_px(6), scaled_px(12), scaled_px(6))
        header.setSpacing(scaled_px(7, minimum=4))

        self.status_dot = FluentStatusDot(size=16)
        self.status_dot.set_color(GREEN)
        self.name_edit = FluentLineEdit("task")
        self.name_edit.setPlaceholderText("task name")
        self.name_edit.setFixedWidth(scaled_px(150, minimum=110))
        self.summary_label = ElidedLabel("")
        self.summary = self.summary_label
        self.summary_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")

        self.kind_combo = FluentComboBox()
        self._panel_kind_choices: tuple[tuple[str, str], ...] = ()
        self._logic_kind_choices: tuple[tuple[str, str, str, str], ...] = ()
        # v1's toolbar token: keep the full kind label and its arrow in the
        # same fixed slot on every screen scale.
        self.kind_combo.setFixedWidth(scaled_px(170, minimum=130))
        self.add_panel_button = FluentButton("Add Panel", color=ACCENT)
        self.add_panel_button.setEnabled(False)

        # Rendered as a push button rather than a toggle, so the button carries
        # the command and the label carries the state.  It emits the state it
        # is asking FOR, which is what makes the command reversible.
        self.pause_switch = FluentButton("Pause", color=ORANGE)
        self.pause_button = self.pause_switch
        self._paused = False
        self.selectors_switch = FluentSwitch("Selectors")
        self.save_screenshot_button = FluentButton("Save image", color=ACCENT)
        self.save_screenshot_button.setToolTip("Save one screenshot of TaskConsole")
        self.save_layout_button = FluentButton("Save", color=ACCENT)
        self.save_layout_button.setToolTip("Save the layout and signal wiring")
        self.load_layout_button = FluentButton("Load", color=ORANGE)
        self.load_layout_button.setToolTip("Load a saved layout and signal wiring")

        for widget in (
            self.status_dot,
            self.name_edit,
        ):
            header.addWidget(widget, 0, QtCore.Qt.AlignVCenter)
        header.addWidget(self.summary_label, 1, QtCore.Qt.AlignVCenter)
        for widget in (
            self.kind_combo,
            self.add_panel_button,
            self.selectors_switch,
            self.pause_switch,
            self.save_screenshot_button,
            self.save_layout_button,
            self.load_layout_button,
        ):
            header.addWidget(widget, 0, QtCore.Qt.AlignVCenter)
        outer.addWidget(header_frame)

        # v1 keeps this status surface between the header and tabs.  It is not
        # a bottom log panel and therefore never shifts the board vertically.
        self.status_strip = StatusStrip()
        outer.addWidget(self.status_strip)

        self.tabs = FluentTabWidget()
        self.tabs.tab_close_requested.connect(self.editor_close_requested.emit)
        monitor_page = QtWidgets.QWidget()
        monitor_page.setStyleSheet("background: transparent;")
        monitor_layout = QtWidgets.QVBoxLayout(monitor_page)
        monitor_layout.setContentsMargins(0, 0, 0, 0)
        self.board = ConsoleBoardView()
        self.board.order_committed.connect(self.panel_order_committed.emit)
        self.scroll = FluentScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setWidget(self.board)
        monitor_layout.addWidget(self.scroll, 1)
        self.tabs.add_permanent_tab(monitor_page, "Monitor")

        logic_page = QtWidgets.QWidget()
        logic_page.setStyleSheet("background: transparent;")
        logic_outer = QtWidgets.QVBoxLayout(logic_page)
        logic_outer.setContentsMargins(0, 0, 0, 0)
        self.logic_scroll = FluentScrollArea()
        self.logic_body = QtWidgets.QWidget()
        self.logic_body.setStyleSheet("background: transparent;")
        self.logic_layout = QtWidgets.QVBoxLayout(self.logic_body)
        self.logic_layout.setContentsMargins(scaled_px(10, minimum=6), scaled_px(10, minimum=6), scaled_px(10, minimum=6), scaled_px(10, minimum=6))
        self.logic_layout.setSpacing(scaled_px(8, minimum=5))
        self.logic_hint = FluentLabel(
            "No logic nodes yet.  Add a Measurement / Processor / Task from the header "
            "(it starts STOPPED); open its Edit to set parameters and Start it."
        )
        self.logic_hint.setWordWrap(True)
        self.logic_hint.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.logic_layout.addWidget(self.logic_hint)
        self.logic_layout.addStretch(1)
        self._logic_rows: tuple[LogicRowView, ...] = ()
        self.logic_scroll.set_width_bounded_widget(self.logic_body)
        logic_outer.addWidget(self.logic_scroll, 1)
        self.logic_page = self.logic_scroll
        self.tabs.add_permanent_tab(logic_page, "Logic")
        outer.addWidget(self.tabs, 1)

        self.add_panel_button.clicked.connect(self._add_current_selection)
        self.pause_switch.clicked.connect(
            lambda _checked=False: self.pause_toggled.emit(not self._paused)
        )
        self.selectors_switch.toggled.connect(self.selectors_toggled.emit)
        self.save_screenshot_button.clicked.connect(self.save_screenshot_requested.emit)
        self.save_layout_button.clicked.connect(self.save_layout_requested.emit)
        self.load_layout_button.clicked.connect(self.load_layout_requested.emit)

    def set_panel_kinds(self, kinds: tuple[tuple[str, str], ...], current: str = "") -> None:
        """Replace the Plot entries in the combined v1 chooser."""

        self._panel_kind_choices = tuple(
            (str(key), str(label or key)) for key, label in kinds
        )
        selected = ("plot", str(current)) if current else None
        self._rebuild_kind_combo(selected)

    def set_logic_kinds(
        self, kinds: tuple[tuple[str, str, str, str], ...]
    ) -> None:
        """Replace the ``(api_name, kind, publishes, blocked)`` Logic entries."""

        layer_order = {"measurement": 0, "processor": 1, "task": 2}
        self._logic_kind_choices = tuple(
            sorted(
                (
                    (str(api_name), str(kind), str(publishes), str(blocked))
                    for api_name, kind, publishes, blocked in kinds
                ),
                key=lambda item: (layer_order.get(item[1], 3), item[0]),
            )
        )
        self._rebuild_kind_combo()

    def _rebuild_kind_combo(self, selected: tuple[str, str] | None = None) -> None:
        wanted = selected if selected is not None else self.kind_combo.currentData()
        self.kind_combo.blockSignals(True)
        self.kind_combo.clear()
        for key, label in self._panel_kind_choices:
            self.kind_combo.addItem(f"Plot: {label}", ("plot", key))
        for api_name, kind, publishes, blocked in self._logic_kind_choices:
            title = api_name.replace("_", " ").strip().title()
            layer = kind.replace("_", " ").strip().title() or "Logic"
            self.kind_combo.addItem(f"{layer}: {title}", ("logic", api_name))
            details = blocked or f"Publishes: {publishes}"
            self.kind_combo.setItemData(
                self.kind_combo.count() - 1, details, QtCore.Qt.ToolTipRole
            )
        if wanted is not None:
            index = next(
                (
                    item
                    for item in range(self.kind_combo.count())
                    if self.kind_combo.itemData(item) == wanted
                ),
                -1,
            )
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
        self.kind_combo.blockSignals(False)
        self.add_panel_button.setEnabled(self.kind_combo.count() > 0)

    def _add_current_selection(self) -> None:
        selected = self.kind_combo.currentData()
        if selected is None:
            return
        family, key = selected
        if family == "plot":
            self.add_panel_requested.emit(str(key))
        elif family == "logic":
            self.add_logic_requested.emit(str(key))

    def set_cards(self, cards: tuple[PanelCardView, ...]) -> None:
        self.board.set_cards(tuple(cards))

    def set_paused(self, paused: bool) -> None:
        """Show the state, so the button can ask for the other one."""

        self._paused = bool(paused)
        self.pause_switch.setText("Resume" if self._paused else "Pause")

    def set_selectors(self, enabled: bool) -> None:
        """Show whether selections derive, without asking for it again.

        The presenter owns this answer too.  A switch that set itself would
        disagree with the presenter the first time the presenter declined.
        """

        self.selectors_switch.blockSignals(True)
        self.selectors_switch.setChecked(bool(enabled))
        self.selectors_switch.blockSignals(False)

    def set_logic_rows(self, rows: tuple[LogicRowView, ...]) -> None:
        incoming = tuple(rows)
        for row in incoming:
            if not isinstance(row, LogicRowView):
                raise TypeError("logic rows must be LogicRowView instances")

        incoming_ids = {id(row) for row in incoming}
        for row in self._logic_rows:
            if id(row) not in incoming_ids:
                self.logic_layout.removeWidget(row)
                row.setParent(None)

        for row in self._logic_rows:
            self.logic_layout.removeWidget(row)
        for row in incoming:
            row.setParent(self.logic_body)
            self.logic_layout.insertWidget(self.logic_layout.count() - 1, row)
        self._logic_rows = incoming
        self.logic_hint.setVisible(not incoming)

    def show_status(self, text: str, severity: str) -> None:
        self.status_strip.show_status(text, severity)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(str(text))

    def add_editor_tab(self, editor: QtWidgets.QWidget, title: str) -> None:
        self.tabs.add_closable_tab(editor, str(title), focus=True)

    def focus_editor_tab(self, editor: QtWidgets.QWidget) -> bool:
        index = self.tabs.indexOf(editor)
        if index < 0:
            return False
        self.tabs.setCurrentIndex(index)
        return True

    def remove_editor_tab(self, editor: QtWidgets.QWidget) -> bool:
        index = self.tabs.indexOf(editor)
        if index < 0:
            return False
        self.tabs.removeTab(index)
        editor.setParent(None)
        editor.deleteLater()
        return True


__all__ = ["TaskConsoleView"]
