"""Extension-cost proof: one new file can define a new visual card."""

from __future__ import annotations

from PyQt5 import QtWidgets


class SyntheticCardView(QtWidgets.QFrame):
    """A deliberately tiny card that knows nothing about the console host."""

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background: #FFF4CC; border: 1px solid #D69A6E; }")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(str(label)))
        layout.addWidget(QtWidgets.QLabel("new card, plain Qt only"))
