"""Pure operator chrome for one console logic row."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    ElidedLabel,
    FluentButton,
    FluentFrame,
    FluentLabel,
    FluentStatusDot,
    scaled_px,
)
from zlc_ui.fluent import PublishedItemsLegend


class LogicRowView(FluentFrame):
    """A lifecycle-free row that emits only operator intent."""

    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    edit_requested = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal()

    _STATE_COLORS = {"idle": GREY, "running": GREEN, "error": RED}

    def __init__(self, title: str = "Logic", kind: str = "logic", parent=None) -> None:
        super().__init__(parent)
        self.title = str(title)
        self.kind = str(kind)
        self._state = "idle"
        self._status_text = "idle"
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(scaled_px(12), scaled_px(8), scaled_px(12), scaled_px(8))
        outer.setSpacing(scaled_px(4, minimum=3))

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(scaled_px(8, minimum=5))
        self.dot = FluentStatusDot(size=14, color=GREY)
        self.name_label = ElidedLabel(self.title)
        self.name_label.setMinimumWidth(0)
        self.name_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.kind_label = FluentLabel(f"({self.kind})")
        self.kind_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.status_label = ElidedLabel("idle")
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.start_button = FluentButton("Start", color=GREEN)
        self.stop_button = FluentButton("Stop", color=ORANGE)
        self.edit_button = FluentButton("Edit", color=ACCENT)
        self.remove_button = FluentButton("Remove", color=GREY)
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.edit_button.clicked.connect(self.edit_requested.emit)
        self.remove_button.clicked.connect(self.remove_requested.emit)
        top.addWidget(self.dot)
        top.addWidget(self.name_label, 1)
        top.addWidget(self.kind_label)
        top.addWidget(self.status_label, 2)
        for button in (self.start_button, self.stop_button, self.edit_button, self.remove_button):
            top.addWidget(button)
        outer.addLayout(top)

        self.publishes_label = PublishedItemsLegend()
        outer.addWidget(self.publishes_label)

    def set_state(self, state: str, status_text: str = "") -> None:
        state = str(state)
        if state not in self._STATE_COLORS:
            raise ValueError("state must be idle, running, or error")
        self._state = state
        self._status_text = str(status_text or state)
        self.dot.set_color(self._STATE_COLORS[state])
        self.status_label.setText(self._status_text)
        self.status_label.setStyleSheet(
            f"color: {RED if state == 'error' else GREY}; background: transparent; border: none;"
        )
        running = state == "running"
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def set_publishes(self, rows: tuple[tuple[str, str, str], ...]) -> None:
        self.publishes_label.set_rows(tuple(tuple(str(value) for value in row) for row in rows))


__all__ = ["LogicRowView"]
