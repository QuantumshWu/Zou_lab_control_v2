"""Pure task-console view composition."""

from .board_view import ConsoleBoardView
from .logic_row_view import LogicRowView
from .panel_card_view import PanelCardView
from .signal_chooser import SignalChooser, choose_signal
from .handle import TaskConsoleHandle
from .status_strip import StatusStrip
from .task_console_view import TaskConsoleView

__all__ = [
    "ConsoleBoardView",
    "LogicRowView",
    "PanelCardView",
    "SignalChooser",
    "choose_signal",
    "StatusStrip",
    "TaskConsoleHandle",
    "TaskConsoleView",
]
