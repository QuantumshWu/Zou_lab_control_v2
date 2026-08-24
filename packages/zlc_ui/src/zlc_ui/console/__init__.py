"""Pure task-console view composition."""

from .board_view import ConsoleBoardView
from .logic_row_view import LogicRowView
from .logic_editor_view import LogicEditorView
from .panel_card_view import PanelCardView
from .panel_editor_view import PanelEditorView
from .point_review_view import PointReviewView
from .signal_chooser import SignalChooser, choose_signal
from .handle import TaskConsoleHandle
from .status_strip import StatusStrip
from .task_console_view import TaskConsoleView

__all__ = [
    "ConsoleBoardView",
    "LogicRowView",
    "LogicEditorView",
    "PanelCardView",
    "PanelEditorView",
    "PointReviewView",
    "SignalChooser",
    "choose_signal",
    "StatusStrip",
    "TaskConsoleHandle",
    "TaskConsoleView",
]
