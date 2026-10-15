"""TaskConsole chrome.

The presenter still owns cards, logic rows and persistence.  This module only
builds the window shell: the flat 48 px header, persistent status strip,
and the permanent ``Monitor`` / ``Logic`` tabs.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    retire_widget,
    detach_widget,
    ElidedLabel,
    MUTED_LABEL_STYLE,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentLabel,
    FluentScrollArea,
    FluentStatusDot,
    FluentSwitch,
    add_row_switch,
    card_pad,
    fit_button_words,
    FluentTabWidget,
    fluent_widget_stylesheet,
    muted_note_label,
    row_gap,
    scaled_px,
    window_pad,
)

from zlc_ui.status import name_words

from .board_view import ConsoleBoardView
from .logic_row_view import STATUS_DOT, LogicRowView, mount_logic_page
from .panel_card_view import PanelCardView
from .status_strip import StatusStrip

#: What the header's Add command says, by the family of the choice beside
#: it: a Plot adds a panel to the Monitor, a measurement, processor or task
#: adds a node to the Logic list.
_ADD_WORDS = {"plot": "Add panel", "logic": "Add node"}


def plot_choice_text(label: str) -> str:
    """How a plot kind reads in a chooser that adds a panel of it: one
    wording in every window that offers one."""

    return f"Plot: {label}"


class TaskConsoleView(QtWidgets.QWidget):
    """A presenter-friendly shell with the product arrangement."""

    #: The header has one combined chooser and one Add Panel button.  These
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
    #: The whole board at once: every panel and every logic node.
    clear_board_requested = QtCore.pyqtSignal()
    save_screenshot_requested = QtCore.pyqtSignal()
    stop_task_requested = QtCore.pyqtSignal()
    #: Re-raised from the board, because a presenter talks to the view it was
    #: given.  Where the operator put the cards is where the panels ARE, and a
    #: figure saved in a different order is a figure of a board nobody saw.
    panel_order_committed = QtCore.pyqtSignal(tuple)
    editor_close_requested = QtCore.pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("TaskConsoleView")
        self.setWindowTitle("TaskConsole@Zou lab")
        self.setStyleSheet(fluent_widget_stylesheet(self.objectName()))

        outer = QtWidgets.QVBoxLayout(self)
        margin = window_pad(1)
        outer.setContentsMargins(margin, margin, margin, margin)
        outer.setSpacing(window_pad(0.5))

        # One semantic row: identity on the left, telemetry as the
        # only flexible middle cell, and commands pinned to the right.  The
        # telemetry elides when space is tight; commands never wrap into a
        # second pseudo-toolbar whose alignment depends on label size hints.
        # Keep this status surface between the header and tabs.  It is not
        # a bottom log panel and therefore never shifts the board vertically.
        self.status_strip = StatusStrip(action_text="Stop task")
        self.status_strip.action_clicked.connect(self.stop_task_requested.emit)
        # Every content edge of the window is one x: the header, the strip
        # and every tab page start (and end) a window pad in.  The header's
        # dot and word are laid out as the strip's dot and line right under
        # them -- its side margin, one dot size, its gap -- so the two dots
        # stand on one x and the two texts start on one x.
        self.status_strip.set_inset(True, pad=window_pad())
        strip_layout = self.status_strip.layout()
        side = strip_layout.contentsMargins().left()

        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(scaled_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(side, scaled_px(6), side, scaled_px(6))
        header.setSpacing(row_gap())

        # The console's state: the worst of its logic nodes' (set_console_state).
        self.status_dot = FluentStatusDot(size=STATUS_DOT)
        # The window's identity, as a word: nothing reads a name typed here,
        # so an edit box in its place was an input that changed nothing.  At
        # its own width, so the summary follows the word instead of a blank
        # column a hundred pixels wide.
        self.name_label = FluentLabel("task")
        self.summary_label = ElidedLabel("")
        self.summary_label.setStyleSheet(MUTED_LABEL_STYLE)

        self.kind_combo = FluentComboBox()
        self._task_takeover = False
        self._panel_kind_choices: tuple[tuple[str, str], ...] = ()
        self._logic_kind_choices: tuple[tuple[str, str, str], ...] = ()
        # Keep the full kind label and its arrow in the
        # same fixed slot on every screen scale.
        self.kind_combo.setFixedWidth(scaled_px(170, minimum=130))
        # The command names what it adds, a panel or a node, from the family
        # of the choice beside it; its slot fits either, so the commands
        # after it never move.
        self.add_button = FluentButton(_ADD_WORDS["plot"])
        fit_button_words(self.add_button, tuple(_ADD_WORDS.values()))
        self.add_button.setEnabled(False)

        # A toggle command: its label names the next press, and it wears the
        # active face while the console is paused -- both projected by the
        # presenter (set_paused), never set by the click.  It emits the state
        # it is asking FOR, which is what makes the command reversible.
        self.pause_button = FluentButton("Pause")
        # Its slot fits both words, so Resume moves nothing beside it.
        fit_button_words(self.pause_button, ("Pause", "Resume"))
        self._paused = False
        # An on/off setting is a switch, even in a row of commands: a
        # checkable button there read as one more command.
        self.selectors_switch = FluentSwitch("Selectors")
        # The file commands ask where before they act, so each carries the
        # ellipsis; reading one is Open, as in every window of the suite.
        self.save_screenshot_button = FluentButton("Save image…")
        self.save_screenshot_button.setToolTip("Save one screenshot of TaskConsole")
        self.save_layout_button = FluentButton("Save…")
        self.save_layout_button.setToolTip("Save the layout and signal wiring")
        self.load_layout_button = FluentButton("Open…")
        self.load_layout_button.setToolTip("Open a saved layout and signal wiring")
        self.clear_board_button = FluentButton("Clear", role="destructive")
        self.clear_board_button.setToolTip("Remove every panel and logic node from the board")

        identity = QtWidgets.QHBoxLayout()
        identity.setSpacing(strip_layout.spacing())
        identity.addWidget(self.status_dot, 0, QtCore.Qt.AlignVCenter)
        identity.addWidget(self.name_label, 0, QtCore.Qt.AlignVCenter)
        header.addLayout(identity)
        header.addWidget(self.summary_label, 1, QtCore.Qt.AlignVCenter)
        header.addWidget(self.kind_combo, 0, QtCore.Qt.AlignVCenter)
        header.addWidget(self.add_button, 0, QtCore.Qt.AlignVCenter)
        # The setting is a group of its own between the commands, so its
        # label does not run into Pause.
        add_row_switch(header, self.selectors_switch)
        for widget in (
            self.pause_button,
            self.save_screenshot_button,
            self.save_layout_button,
            self.load_layout_button,
            self.clear_board_button,
        ):
            header.addWidget(widget, 0, QtCore.Qt.AlignVCenter)
        outer.addWidget(header_frame)
        outer.addWidget(self.status_strip)

        # Every page of the tabs starts its content a window pad in -- the
        # Logic list and every Edit page (mount_logic_page), and the board,
        # whose own card-pad edge makes up the rest -- and the first tab's
        # underline starts there too, so switching pages moves nothing.
        page = window_pad()
        lead = page - card_pad()
        self.tabs = FluentTabWidget()
        self.tabs.set_page_inset(page)
        self.tabs.tab_close_requested.connect(self.editor_close_requested.emit)
        monitor_page = QtWidgets.QWidget()
        monitor_page.setStyleSheet("background: transparent;")
        monitor_layout = QtWidgets.QVBoxLayout(monitor_page)
        monitor_layout.setContentsMargins(lead, lead, 0, 0)
        self.board = ConsoleBoardView()
        self.board.order_committed.connect(self.panel_order_committed.emit)
        # An empty Monitor said nothing at all; it says how a panel arrives.
        self.monitor_hint = muted_note_label(
            "No panels yet. Choose a Plot in the header and click Add panel, or turn on "
            "Plot on a Logic row before starting it."
        )
        self.monitor_hint.setWordWrap(True)
        # Where the board's first card will stand: its words start where a
        # card's title does and end where its text would, and its top is a
        # card's top -- the board's gap under the page's own lead.
        words = self.board.text_left()
        self.monitor_hint.setContentsMargins(
            words, words - PanelCardView.strip_text_inset(), words + lead, 0
        )
        monitor_layout.addWidget(self.monitor_hint)
        self.scroll = FluentScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setWidget(self.board)
        self.scroll.reserve_vertical_gutter()
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
        # Cards stack a form gap apart, like rows; the page edges are every
        # Logic page's (mount_logic_page).
        self.logic_layout.setSpacing(row_gap())
        # Says how a node arrives, in the words of the header it arrives by,
        # as the Monitor's note does for a panel.
        self.logic_hint = muted_note_label(
            "No logic nodes yet. Choose a measurement, processor or task in the "
            "header and click Add node; it starts stopped. Open its Edit page to "
            "set parameters, then press Start."
        )
        self.logic_hint.setWordWrap(True)
        self.logic_layout.addWidget(self.logic_hint)
        self.logic_layout.addStretch(1)
        self._logic_rows: tuple[LogicRowView, ...] = ()
        mount_logic_page(self.logic_scroll, self.logic_body)
        logic_outer.addWidget(self.logic_scroll, 1)
        self.tabs.add_permanent_tab(logic_page, "Logic")
        outer.addWidget(self.tabs, 1)

        self.add_button.clicked.connect(self._add_current_selection)
        self.kind_combo.currentIndexChanged[int].connect(
            lambda _index: self._update_add_button()
        )
        self.pause_button.clicked.connect(
            lambda _checked=False: self.pause_toggled.emit(not self._paused)
        )
        self.selectors_switch.toggled.connect(self.selectors_toggled.emit)
        self.save_screenshot_button.clicked.connect(self.save_screenshot_requested.emit)
        self.save_layout_button.clicked.connect(self.save_layout_requested.emit)
        self.load_layout_button.clicked.connect(self.load_layout_requested.emit)
        self.clear_board_button.clicked.connect(self.clear_board_requested.emit)

    def set_panel_kinds(self, kinds: tuple[tuple[str, str], ...], current: str = "") -> None:
        """Replace the Plot entries in the combined chooser."""

        self._panel_kind_choices = tuple(
            (str(key), str(label or key)) for key, label in kinds
        )
        selected = ("plot", str(current)) if current else None
        self._rebuild_kind_combo(selected)

    def set_logic_kinds(
        self, kinds: tuple[tuple[str, str, str], ...]
    ) -> None:
        """Replace the ``(api_name, kind, publishes)`` Logic entries."""

        layer_order = {"measurement": 0, "processor": 1, "task": 2}
        self._logic_kind_choices = tuple(
            sorted(
                (
                    (str(api_name), str(kind), str(publishes))
                    for api_name, kind, publishes in kinds
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
            self.kind_combo.addItem(plot_choice_text(label), ("plot", key))
        # A node TYPE is listed by its api_name -- as a plot is by its kind
        # key; a node itself is called by the Name it is given.
        for api_name, kind, publishes in self._logic_kind_choices:
            layer = name_words(kind) or "Logic"
            self.kind_combo.addItem(f"{layer}: {api_name}", ("logic", api_name))
            self.kind_combo.setItemData(
                self.kind_combo.count() - 1,
                f"Publishes: {publishes}",
                QtCore.Qt.ToolTipRole,
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
        self._update_add_button()

    def _update_add_button(self) -> None:
        selected = self.kind_combo.currentData()
        family = "plot" if selected is None else selected[0]
        self.add_button.setText(_ADD_WORDS[family])
        blocked_logic = self._task_takeover and family == "logic"
        self.add_button.setEnabled(
            self.kind_combo.count() > 0 and not blocked_logic
        )

    def _add_current_selection(self) -> None:
        selected = self.kind_combo.currentData()
        if selected is None:
            return
        family, key = selected
        if family == "plot":
            self.add_panel_requested.emit(str(key))
        elif family == "logic" and not self._task_takeover:
            self.add_logic_requested.emit(str(key))

    def set_cards(self, cards: tuple[PanelCardView, ...]) -> None:
        self.board.set_cards(tuple(cards))
        self.monitor_hint.setVisible(not cards)

    def set_paused(self, paused: bool) -> None:
        """Show the state, so the button can ask for the other one."""

        self._paused = bool(paused)
        self.pause_button.setText("Resume" if self._paused else "Pause")
        self.pause_button.set_active(self._paused)

    def set_console_state(self, word: str) -> None:
        """The header dot: ``running`` while a node runs, ``error`` while
        one has failed, ``idle`` otherwise (``STATUS_CUES``)."""

        self.status_dot.set_state(word)

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
                detach_widget(row)

        for row in self._logic_rows:
            self.logic_layout.removeWidget(row)
        for row in incoming:
            row.setParent(self.logic_body)
            self.logic_layout.insertWidget(self.logic_layout.count() - 1, row)
        self._logic_rows = incoming
        self.logic_hint.setVisible(not incoming)

    def show_status(self, text: str, severity: str) -> None:
        self.status_strip.show_status(text, severity)

    def set_task_takeover(self, active: bool) -> None:
        """Project the console-wide Task command gate onto header chrome."""

        self._task_takeover = bool(active)
        # A running Task owns the board: neither another board nor an empty
        # one may replace it from under the task.
        self.load_layout_button.setEnabled(not self._task_takeover)
        self.clear_board_button.setEnabled(not self._task_takeover)
        self._update_add_button()
        self.status_strip.set_action_visible(self._task_takeover)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(str(text))

    def add_editor_tab(self, editor: QtWidgets.QWidget, title: str) -> None:
        self.tabs.add_closable_tab(editor, str(title), focus=True)

    def set_editor_tab_title(self, editor: QtWidgets.QWidget, title: str) -> None:
        """Retitle an open editor's tab when what it edits is renamed."""

        index = self.tabs.indexOf(editor)
        if index >= 0 and self.tabs.tabText(index) != str(title):
            self.tabs.setTabText(index, str(title))

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
        retire_widget(editor)
        return True


__all__ = ["TaskConsoleView", "plot_choice_text"]
