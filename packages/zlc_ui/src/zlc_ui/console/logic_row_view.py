"""Pure operator chrome for one console logic row."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    GREEN,
    GREY,
    MUTED_INK,
    RED,
    RED_DARK,
    ElidedLabel,
    FluentButton,
    FluentFrame,
    FluentLabel,
    FluentStatusDot,
    popup_gap,
    retire_widget,
    scaled_px,
    setting_label_width,
)
from zlc_ui.fluent import PublishedItemsLegend
from zlc_ui.fluent.fluent import _FluentRoundedMenu

#: Every kind a row can carry, so the kind column is one width on every row
#: and the status beside it starts at one x down the page.
_KINDS = ("(measurement)", "(processor)", "(task)", "(panel publisher)")


class LogicRowView(FluentFrame):
    """A lifecycle-free row that emits only operator intent.

    One lifecycle button says the next thing the node can do -- Start while
    it is stopped, Stop while it runs -- so a row no longer carries two
    commands of which one is always dead.  Restart and Remove, the rare and
    the destructive, sit behind the row's ``⋯``.
    """

    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    edit_requested = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal()
    auto_preview_changed = QtCore.pyqtSignal(bool)

    _STATE_COLORS = {"idle": GREY, "running": GREEN, "error": RED}

    def __init__(self, title: str = "Logic", kind: str = "logic", parent=None) -> None:
        super().__init__(parent)
        self.title = str(title)
        self.kind = str(kind)
        self._state = "idle"
        self._status_text = "idle"
        self._status_colour = ""
        self._can_start = False
        self._can_stop = False
        self._task_takeover = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(scaled_px(12), scaled_px(8), scaled_px(12), scaled_px(8))
        outer.setSpacing(scaled_px(6, minimum=4))

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(scaled_px(8, minimum=5))
        self.dot = FluentStatusDot(size=14, color=GREY)
        self.name_label = ElidedLabel(self.title)
        # Name and kind are fixed columns: every row's kind and status start
        # at the same x, instead of floating with the length of the name.
        self.name_label.setFixedWidth(scaled_px(200, minimum=150))
        self.kind_label = FluentLabel(f"({self.kind})")
        self.kind_label.setStyleSheet(f"color: {MUTED_INK}; background: transparent; border: none;")
        self.kind_label.setFixedWidth(setting_label_width(_KINDS))
        self.status_label = ElidedLabel("idle")
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        #: Beside Start because it says what Start will DO: with it on, the
        #: node's declared preview opens as an ordinary panel, already wired
        #: to the signal.  A toggle CHIP, pressed in when on, so it reads as
        #: a state among the commands and not as one more of them.
        self.preview_switch = FluentButton("Plot")
        self.preview_switch.setCheckable(True)
        self.preview_switch.setToolTip(
            "Open this node's plot panel when it starts"
        )
        self.preview_switch.toggled.connect(self.auto_preview_changed.emit)
        self.run_button = FluentButton("Stop")
        stop_width = self.run_button.sizeHint().width()
        self.run_button.setText("Start")
        # Wide enough for either word, so the row does not shift when it flips.
        self.run_button.setMinimumWidth(max(stop_width, self.run_button.sizeHint().width()))
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run_clicked)
        self.edit_button = FluentButton("Edit")
        self.edit_button.clicked.connect(self.edit_requested.emit)
        self.more_button = FluentButton("⋯")
        self.more_button.setFixedWidth(self.more_button.minimumHeight())
        self.more_button.setToolTip("More: Restart, Remove")
        self.more_button.clicked.connect(self._show_more)
        self.restart_action = QtWidgets.QAction("Restart", self)
        self.restart_action.triggered.connect(self.start_requested.emit)
        self.remove_action = QtWidgets.QAction("Remove", self)
        self.remove_action.triggered.connect(self.remove_requested.emit)
        top.addWidget(self.dot)
        top.addWidget(self.name_label)
        top.addWidget(self.kind_label)
        top.addWidget(self.status_label, 1)
        for button in (
            self.preview_switch,
            self.run_button,
            self.edit_button,
            self.more_button,
        ):
            top.addWidget(button)
        outer.addLayout(top)

        self.publishes_label = PublishedItemsLegend()
        outer.addWidget(self.publishes_label)
        self._project_commands()

    def _running(self) -> bool:
        return self._state == "running"

    def _run_clicked(self) -> None:
        if self._running():
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def _show_more(self) -> None:
        """The row's rare commands, below its ``⋯`` -- built for this click."""

        menu = _FluentRoundedMenu(self)
        if self._running():
            menu.addAction(self.restart_action)
        menu.addAction(self.remove_action)
        button = self.more_button
        try:
            menu.exec_(button.mapToGlobal(QtCore.QPoint(0, button.height() + popup_gap())))
        finally:
            retire_widget(menu)

    def set_preview_offered(self, offered: bool) -> None:
        """Show the switch only for a node that declares something to open.

        A processor answers a signal it does not own; nothing about starting
        it says which of its answers belongs on this board, so there is no
        preference to express and no switch to express it with.
        """

        self.preview_switch.setVisible(bool(offered))

    def set_auto_preview(self, enabled: bool) -> None:
        """Show the owner's stored preference without re-emitting it.

        The preference belongs to the node's binding; this widget holds no
        default of its own, so a rebuilt row cannot disagree with the board.
        """

        blocked = self.preview_switch.blockSignals(True)
        try:
            self.preview_switch.setChecked(bool(enabled))
        finally:
            self.preview_switch.blockSignals(blocked)

    def set_state(self, state: str, status_text: str = "") -> None:
        state = str(state)
        if state not in self._STATE_COLORS:
            raise ValueError("state must be idle, running, or error")
        self._state = state
        self._status_text = str(status_text or state)
        self.dot.set_color(self._STATE_COLORS[state])
        self.status_label.setText(self._status_text)
        # Written only when it CHANGES.  The colour has two values and a
        # status tick arrives ten times a second, so almost every write said
        # what the last one said -- and a stylesheet write re-parses the
        # sheet, repolishes the widget and schedules a repaint regardless.
        colour = RED_DARK if state == "error" else MUTED_INK
        if colour != self._status_colour:
            self._status_colour = colour
            self.status_label.setStyleSheet(
                f"color: {colour}; background: transparent; border: none;"
            )
        self._project_commands()

    def set_commands(self, *, can_start: bool, can_stop: bool) -> None:
        """Project presenter-owned command admission onto this row."""

        self._can_start = bool(can_start)
        self._can_stop = bool(can_stop)
        self._project_commands()

    def set_task_takeover(self, active: bool) -> None:
        """Leave the row readable while the status strip owns all commands."""

        self._task_takeover = bool(active)
        self._project_commands()

    def _project_commands(self) -> None:
        enabled = not self._task_takeover
        running = self._running()
        self.run_button.setText("Stop" if running else "Start")
        self.run_button.setEnabled(enabled and (self._can_stop if running else self._can_start))
        # It says what the next Start will do, so a row that cannot Start
        # cannot re-aim Start either.
        self.preview_switch.setEnabled(enabled)
        self.edit_button.setEnabled(enabled)
        self.more_button.setEnabled(enabled)
        self.restart_action.setEnabled(enabled and running and self._can_start)
        self.remove_action.setEnabled(enabled)
        # A running Task's Stop is the status strip's; the row does not
        # offer a second one beside it.
        self.run_button.setVisible(not (self._task_takeover and self.kind == "task"))

    def set_publishes(self, rows: tuple[tuple[str, str, str], ...]) -> None:
        self.publishes_label.set_rows(tuple(tuple(str(value) for value in row) for row in rows))


__all__ = ["LogicRowView"]
