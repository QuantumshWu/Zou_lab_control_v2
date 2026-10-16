"""Pure operator chrome for one console logic row.

Also what the row shares with its node's Edit page and with the Logic list
around it -- the status words, the Start words, the gaps -- so the list and
every Edit page are built from one set of numbers.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ElidedLabel,
    FluentButton,
    FluentFrame,
    FluentStatusDot,
    FluentSwitch,
    add_row_switch,
    card_pad,
    fit_button_words,
    muted_note_label,
    scaled_px,
    set_status_ink,
    setting_label_width,
    status_dot_gap,
)
from zlc_ui.fluent import PublishedItemsLegend
from zlc_ui.fluent import row_gap
from zlc_ui.status import SUMMARY_SEPARATOR

#: Every kind a row can carry, so the kind column is one width on every row
#: and the status beside it starts at one x down the page.
_KINDS = ("(measurement)", "(processor)", "(task)", "(panel publisher)")

#: Every word the Start slot can say, on the row and on the node's Edit page.
START_LABELS = ("Start", "Restart", "Starting…")

#: Each node state as the status word it is drawn with (``STATUS_CUES``).
#: ONE table for the row and the node's Edit page.  A draft is
#: drawn as calmly as a stopped node: what keeps it from starting is
#: something still to fill in, not a fault, and amber is what reads as
#: "something is wrong".  A run wears the plain ink of something under way,
#: a run that is not what the operator asked for -- its Restart refused --
#: wears amber, and only a failure is red.
STATE_WORDS = {
    "idle": "idle",
    "draft": "idle",
    "running": "running",
    "warning": "warning",
    "error": "error",
}


def start_label(*, pending: bool, active: bool) -> str:
    """What Start does now: one rule, so the row and its Edit page agree.

    ``active`` is the presenter's ``can_stop`` -- running, following its
    source (a failed follower included) or queued -- which is exactly when
    a Start replaces something instead of beginning from nothing.
    """

    if pending:
        return "Starting…"
    return "Restart" if active else "Start"


class LogicRowView(FluentFrame):
    """A lifecycle-free row that emits only operator intent.

    Start and Stop are two fixed slots that are always there: Start reads
    Restart while the node is active and Starting… while its start waits,
    and Stop is enabled whenever there is anything to stop.  One button
    that flipped between Start and Stop left a failed follower with nothing
    to stop, and a Restart that appeared only after a Start moved the
    controls beside it.
    """

    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    edit_requested = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal()
    auto_preview_changed = QtCore.pyqtSignal(bool)

    def __init__(self, title: str = "Logic", kind: str = "logic", parent=None) -> None:
        super().__init__(parent)
        self.kind = str(kind)
        self._status_text = "idle"
        #: The status word the row shows now; the console's own dot is the
        #: worst of its rows'.
        self.state_word = "idle"
        self._can_start = False
        self._can_stop = False
        self._pending = False
        self._task_takeover = False
        self._owns_task = False
        outer = QtWidgets.QVBoxLayout(self)
        # A card's one padding on all four sides, and the form's gap between
        # the command line and the legend under it.  The commands are spaced
        # by that gap too, as on the node's Edit page, so the same
        # Plot | Start | Stop group has the same air in both places.
        pad = card_pad()
        outer.setContentsMargins(pad, pad, pad, pad)
        outer.setSpacing(row_gap())

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(row_gap())
        self.dot = FluentStatusDot()
        # Elided in the MIDDLE: names that differ at their end ("... shot 1",
        # "... shot 2") still read differently in the column.
        self.name_label = ElidedLabel(str(title), mode=QtCore.Qt.ElideMiddle)
        # Name and kind are fixed columns: every row's kind and status start
        # at the same x, instead of floating with the length of the name.
        self.name_label.setFixedWidth(scaled_px(200, minimum=150))
        self.kind_label = muted_note_label(f"({self.kind})")
        self.kind_label.setFixedWidth(setting_label_width(_KINDS))
        self.status_label = ElidedLabel("idle")
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        #: Beside Start because it says what Start will DO: with it on, the
        #: node's declared preview opens as an ordinary panel, already wired
        #: to the signal.  A SWITCH, a setting and not one more command.
        self.preview_switch = FluentSwitch("Plot")
        self.preview_switch.setToolTip(
            "Open this node's plot panel when it starts"
        )
        self.preview_switch.toggled.connect(self.auto_preview_changed.emit)
        self.start_button = FluentButton("Start")
        fit_button_words(self.start_button, START_LABELS)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button = FluentButton("Stop", role="destructive")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.edit_button = FluentButton("Edit")
        self.edit_button.clicked.connect(self.edit_requested.emit)
        # On the row, where it is looked for: the quiet ghost x that turns
        # red under the pointer, as on every other list.  Removing a node
        # takes two clicks: the first arms the white-on-red confirm.
        self.remove_button = FluentButton(
            "×", role="destructive", confirm_twice="remove this node"
        )
        self.remove_button.setFixedWidth(self.remove_button.minimumHeight())
        self.remove_button.confirmed.connect(self.remove_requested.emit)
        top.addWidget(self.dot)
        # A dot's words stand a status gap after it, as on the node's Edit
        # page and in the status strip: one status look wherever it is.
        top.addSpacing(status_dot_gap() - row_gap())
        top.addWidget(self.name_label)
        top.addWidget(self.kind_label)
        top.addWidget(self.status_label, 1)
        # Air after the "Plot" label, so it does not read as part of Start.
        add_row_switch(top, self.preview_switch, lead=False)
        for button in (
            self.start_button,
            self.stop_button,
            self.edit_button,
            self.remove_button,
        ):
            top.addWidget(button)
        outer.addLayout(top)

        self.publishes_label = PublishedItemsLegend()
        outer.addWidget(self.publishes_label)
        self._project_commands()

    def set_name(self, text: str, details: tuple[str, ...] = ()) -> None:
        """Show what the node is called; elided in its column, whole in the
        tooltip with ``details`` after it -- what the node is and its
        internal ID, which a name the operator chose no longer says.  The
        row is not where it is renamed: its Edit page is."""

        self.name_label.setText(str(text))
        self.name_label.setToolTip(
            SUMMARY_SEPARATOR.join((str(text), *(str(item) for item in details if item)))
        )

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
        if state not in STATE_WORDS:
            raise ValueError(f"state must be one of {tuple(STATE_WORDS)}")
        self._status_text = str(status_text or state)
        self.state_word = STATE_WORDS[state]
        self.dot.set_state(self.state_word)
        self.status_label.setText(self._status_text)
        set_status_ink(self.status_label, self.state_word)

    def set_commands(self, *, can_start: bool, can_stop: bool, pending: bool = False) -> None:
        """Project presenter-owned command admission onto this row.

        ``pending`` is a Start queued behind a device release: the slot says
        Starting… until it runs, and Stop cancels it.
        """

        self._can_start = bool(can_start)
        self._can_stop = bool(can_stop)
        self._pending = bool(pending)
        self._project_commands()

    def set_task_takeover(self, active: bool, *, owns_task: bool = False) -> None:
        """Leave the row readable while a running Task owns the console.

        The Task's own row keeps its Stop -- the same endpoint as the status
        strip's "Stop task" -- so the run can be ended where it is listed.
        """

        self._task_takeover = bool(active)
        self._owns_task = self._task_takeover and bool(owns_task)
        self._project_commands()

    def _project_commands(self) -> None:
        enabled = not self._task_takeover
        self.start_button.setText(start_label(pending=self._pending, active=self._can_stop))
        self.start_button.setEnabled(enabled and self._can_start and not self._pending)
        self.stop_button.setEnabled(self._can_stop and (enabled or self._owns_task))
        # It says what the next Start will do, so a row that cannot Start
        # cannot re-aim Start either.
        self.preview_switch.setEnabled(enabled)
        # Edit stays: a page opened under a Task is read-only but its Name,
        # which is not part of what the Task runs.
        self.remove_button.setEnabled(enabled)

    def set_publishes(self, rows: tuple[tuple[str, str, str], ...]) -> None:
        self.publishes_label.set_rows(tuple(tuple(str(value) for value in row) for row in rows))


__all__ = [
    "START_LABELS",
    "STATE_WORDS",
    "LogicRowView",
    "start_label",
]
