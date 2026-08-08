"""A modal must not paint the window behind it with widgets that are leaving.

``deleteLater`` runs when the event loop it was called from returns to its top
level.  A modal dialog never returns there -- it opens a NESTED loop -- so a
rebuild that retires widgets and then reports what it did leaves the retired
widgets alive for exactly as long as the message is up, and the nested loop
repaints the window with both sets on it.  On screen that is two sets of text
on top of each other, for the instant the dialog is open.

Measured on a pulse reopen before the fix: 6 old period cards alive beside the
4 new ones.
"""

from __future__ import annotations

import pytest


def test_a_dialog_opens_only_after_retired_widgets_are_gone() -> None:
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets

    from zlc_ui.fluent import retire_pending_widgets
    from zlc_ui.qt import ensure_qt_app

    app = ensure_qt_app(["retire-before-modal"])
    holder = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(holder)
    doomed = [QtWidgets.QLabel(f"old {index}", holder) for index in range(6)]
    for label in doomed:
        layout.addWidget(label)

    for label in doomed:
        layout.removeWidget(label)
        label.setParent(None)
        label.deleteLater()

    import sip

    assert not any(sip.isdeleted(label) for label in doomed), (
        "deleteLater is supposed to DEFER -- if these were already gone this "
        "test would pass without proving anything"
    )
    for _ in range(20):
        app.processEvents()
    assert not any(sip.isdeleted(label) for label in doomed), (
        "processEvents alone left them alive, which is the situation a modal "
        "opens into"
    )

    retire_pending_widgets()

    assert all(sip.isdeleted(label) for label in doomed), (
        "a dialog would have painted over widgets that were on their way out"
    )
    holder.deleteLater()
