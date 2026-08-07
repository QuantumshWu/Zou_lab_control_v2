"""Pure v1-style TaskConsole chrome.

The presenter still owns cards, logic rows and persistence.  This module only
recreates the v1 window shell: the flat 48 px header, persistent status strip,
and the permanent ``Monitor`` / ``Logic`` tabs.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    FluentFlowRow,
    ACCENT,
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

    #: Which KIND of panel to add -- the one chosen beside the button.
    add_panel_requested = QtCore.pyqtSignal(str)
    add_logic_requested = QtCore.pyqtSignal()
    pause_toggled = QtCore.pyqtSignal(bool)
    selectors_toggled = QtCore.pyqtSignal(bool)
    save_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()
    save_image_requested = QtCore.pyqtSignal()
    #: The BOARD, not the data on it: which panels, drawn how, and what is
    #: producing them.  An arrangement that took an afternoon used to be lost
    #: with the window, because saving here meant saving numbers.
    save_board_requested = QtCore.pyqtSignal()
    load_board_requested = QtCore.pyqtSignal()
    #: Re-raised from the board, because a presenter talks to the view it was
    #: given.  Where the operator put the cards is where the panels ARE, and a
    #: figure saved in a different order is a figure of a board nobody saw.
    panel_order_committed = QtCore.pyqtSignal(tuple)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("TaskConsoleView")
        self.setWindowTitle("TaskConsole@Zou lab")
        self.setStyleSheet(fluent_widget_stylesheet())

        outer = QtWidgets.QVBoxLayout(self)
        margin = window_pad(1)
        outer.setContentsMargins(margin, margin, margin, margin)
        outer.setSpacing(window_pad(0.5))

        # A WRAPPING row, not a widening one.  A plain horizontal row's minimum
        # width is the sum of everything in it, so nine controls set the
        # window's minimum and the tenth pushed it past the shared screen-fit
        # size -- adding a button became a window-geometry decision.  The frame
        # takes its height from the row rather than fixing it, so a second line
        # appears when one is needed and never otherwise.
        header_frame = FluentFrame(bordered=False)
        header = FluentFlowRow(header_frame, spacing=scaled_px(7, minimum=4))
        header.setContentsMargins(scaled_px(12), scaled_px(6), scaled_px(12), scaled_px(6))

        self.status_dot = FluentStatusDot(size=16)
        self.status_dot.set_color(GREEN)
        self.name_edit = FluentLineEdit("task")
        self.name_edit.setPlaceholderText("task name")
        self.name_edit.setFixedWidth(scaled_px(150, minimum=110))
        self.summary_label = FluentLabel("")
        self.summary = self.summary_label
        self.summary_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")

        self.kind_combo = FluentComboBox()
        # v1's toolbar token: keep the full kind label and its arrow in the
        # same fixed slot on every screen scale.
        self.kind_combo.setFixedWidth(scaled_px(170, minimum=130))
        self.add_panel_button = FluentButton("Add Panel", color=ACCENT)

        # Rendered as a push button rather than a toggle, so the button carries
        # the command and the label carries the state.  It emits the state it
        # is asking FOR, which is what makes the command reversible.
        self.pause_switch = FluentButton("Pause", color=ORANGE)
        self.pause_button = self.pause_switch
        self._paused = False
        self.selectors_switch = FluentSwitch("Selectors")
        self.save_image_button = FluentButton("Save image", color=ACCENT)
        self.save_button = FluentButton("Save data", color=ACCENT)
        self.load_button = FluentButton("Open figure", color=ORANGE)
        # Two different acts, and they were one word apart: this pair writes and
        # restores the BOARD; the pair above writes the numbers on it and opens
        # a figure someone saved earlier.
        self.save_board_button = FluentButton("Save board", color=ACCENT)
        self.load_board_button = FluentButton("Load board", color=ORANGE)

        for widget in (
            self.status_dot,
            self.name_edit,
        ):
            header.addWidget(widget)
        header.addWidget(self.summary_label)
        for widget in (
            self.kind_combo,
            self.add_panel_button,
            self.selectors_switch,
            self.pause_switch,
            self.save_image_button,
            self.save_button,
            self.load_button,
            self.save_board_button,
            self.load_board_button,
        ):
            header.addWidget(widget)
        outer.addWidget(header_frame)

        # v1 keeps this status surface between the header and tabs.  It is not
        # a bottom log panel and therefore never shifts the board vertically.
        self.status_strip = StatusStrip()
        outer.addWidget(self.status_strip)

        self.tabs = FluentTabWidget()
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

        self.add_panel_button.clicked.connect(
            lambda: self.add_panel_requested.emit(str(self.kind_combo.currentData() or ""))
        )
        self.pause_switch.clicked.connect(
            lambda _checked=False: self.pause_toggled.emit(not self._paused)
        )
        self.selectors_switch.toggled.connect(self.selectors_toggled.emit)
        self.save_button.clicked.connect(self.save_requested.emit)
        self.load_button.clicked.connect(self.load_requested.emit)
        self.save_image_button.clicked.connect(self.save_image_requested.emit)
        self.save_board_button.clicked.connect(self.save_board_requested.emit)
        self.load_board_button.clicked.connect(self.load_board_requested.emit)

    def set_panel_kinds(self, kinds: tuple[tuple[str, str], ...], current: str = "") -> None:
        """What kinds of panel this board can add, as (key, label).

        The combo held one hardcoded item and nothing read it: pressing Add
        Panel opened a signal chooser instead, so the control beside the button
        described a choice the button did not make.  What kinds exist belongs
        to the plotting package and arrives from whoever knows both.
        """

        incoming = tuple((str(key), str(label or key)) for key, label in kinds)
        self.kind_combo.blockSignals(True)
        self.kind_combo.clear()
        for key, label in incoming:
            self.kind_combo.addItem(label, key)
        if current:
            index = self.kind_combo.findData(str(current))
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
        self.kind_combo.blockSignals(False)
        self.add_panel_button.setEnabled(bool(incoming))

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


__all__ = ["TaskConsoleView"]
