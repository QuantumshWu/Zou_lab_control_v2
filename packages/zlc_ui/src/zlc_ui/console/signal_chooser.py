"""Choosing which signal to open a panel on.

This does not live in the header.  The single-row header is already full at the window
width the launcher targets, and a picker squeezed in beside it collapsed to
"fra..." while clipping the control next to it -- a chooser you cannot read is
not a chooser.  Asked for at the moment it is needed, it can be as wide as the
names it has to show.

It renders rows it is given and returns the one that was picked.  What the rows
mean, which order they are in and whether any of them can be drawn are all
decided elsewhere; this is a list and a button.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT,
    GREY,
    FluentButton,
    FluentCardDialog,
    FluentLabel,
    FluentSectionLabel,
    scaled_px,
    window_pad,
)


__all__ = ["SignalChooser", "choose_signal"]


class SignalChooser(FluentCardDialog):
    """A modal list of offerable signals, grouped by what produced them.

    On the shared Fluent card, like every other modal here: a widget
    background of its own would paint a square over the card's rounded
    corners, so the card supplies the ground and this supplies a title
    the frameless chrome has nowhere else to put.
    """

    def __init__(self, rows, parent=None) -> None:
        super().__init__(parent, title="Add panel")

        layout = QtWidgets.QVBoxLayout(self)
        margin = window_pad(1)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(window_pad(0.5))
        layout.addWidget(FluentSectionLabel("Add panel"))

        self.list = QtWidgets.QListWidget()
        self.list.setMinimumWidth(scaled_px(420, minimum=320))
        self.list.setMinimumHeight(scaled_px(240, minimum=180))
        self.list.setStyleSheet(
            "QListWidget { background: white; border: 1px solid #e1e1e1; "
            "border-radius: 4px; } "
            f"QListWidget::item {{ padding: {scaled_px(6, minimum=4)}px; }} "
            f"QListWidget::item:selected {{ background: {ACCENT}; color: white; }}"
        )
        self._rows = tuple(rows)
        producer = None
        for name, label, state, group, derived_from in self._rows:
            if group != producer:
                producer = group
                heading = QtWidgets.QListWidgetItem(str(group))
                heading.setFlags(QtCore.Qt.NoItemFlags)
                heading.setForeground(QtWidgets.QApplication.palette().mid())
                self.list.addItem(heading)
            text = f"    {label}" if state == "live" else f"    {label}  ({state})"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, str(name))
            tooltip = str(name)
            if derived_from:
                tooltip = f"{tooltip}\ncut from {derived_from}"
            item.setToolTip(tooltip)
            self.list.addItem(item)
        layout.addWidget(self.list, 1)

        if not self._rows:
            empty = FluentLabel("Nothing has published yet.")
            empty.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            layout.addWidget(empty)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = FluentButton("Cancel", color=GREY)
        self.accept_button = FluentButton("Add Panel", color=ACCENT)
        self.accept_button.setEnabled(False)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.accept_button)
        layout.addLayout(buttons)

        self.list.itemSelectionChanged.connect(self._selection_changed)
        self.list.itemDoubleClicked.connect(lambda _item: self.accept())
        self.cancel_button.clicked.connect(self.reject)
        self.accept_button.clicked.connect(self.accept)

        first = self._first_selectable()
        if first is not None:
            self.list.setCurrentRow(first)

    def _first_selectable(self) -> int | None:
        for index in range(self.list.count()):
            if self.list.item(index).flags() & QtCore.Qt.ItemIsSelectable:
                return index
        return None

    def _selection_changed(self) -> None:
        self.accept_button.setEnabled(self.chosen() is not None)

    def chosen(self) -> str | None:
        """The signal name currently selected, or None."""

        item = self.list.currentItem()
        if item is None or not item.flags() & QtCore.Qt.ItemIsSelectable:
            return None
        value = item.data(QtCore.Qt.UserRole)
        return None if value is None else str(value)


def choose_signal(rows, parent=None) -> str | None:
    """Ask which signal to show; None when the operator declines."""

    dialog = SignalChooser(rows, parent)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return dialog.chosen()
