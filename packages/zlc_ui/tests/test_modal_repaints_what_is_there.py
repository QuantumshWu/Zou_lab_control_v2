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


def _card_dialogs(parent):
    """One of each modal that wears the shared frameless card."""

    from zlc_ui.fluent import FluentInputDialog
    from zlc_ui.fluent.fluent import _FluentMessageDialog

    return (
        ("message", _FluentMessageDialog(parent, "Pulse", "saved", kind="info")),
        (
            "confirm",
            _FluentMessageDialog(
                parent, "Pulse", "discard?", kind="warning", confirm=True
            ),
        ),
        ("input", FluentInputDialog("Value", 1.0, parent, title="Set")),
    )


def test_a_modal_always_opens_where_its_buttons_can_be_clicked() -> None:
    """A frameless card is placed by hand, so it can be placed off the screen.

    A platform frame is placed by the window manager, which never does that.
    This one was centred on its parent with no clamp: a window near the right
    or bottom edge put the WHOLE card past the edge -- and it is modal, so
    nothing else would take a click and no button could be reached.  Saving a
    pulse with the window near an edge was an unrecoverable freeze.
    """

    pytest.importorskip("PyQt5")
    from PyQt5 import QtCore, QtWidgets

    from zlc_ui.qt import ensure_qt_app

    app = ensure_qt_app(["modal-placement"])
    screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
    parent = QtWidgets.QWidget()
    parent.resize(600, 400)

    corners = {
        "middle": (screen.x() + 300, screen.y() + 200),
        "right edge": (screen.x() + screen.width() - 120, screen.y() + 100),
        "bottom edge": (screen.x() + 100, screen.y() + screen.height() - 120),
        "off the left": (screen.x() - 400, screen.y() + 100),
    }
    for where, (x, y) in corners.items():
        parent.move(x, y)
        parent.show()
        app.processEvents()
        for name, dialog in _card_dialogs(parent):
            dialog.adjustSize()
            dialog.move(dialog._opening_position())
            dialog.show()
            app.processEvents()
            frame = dialog.frameGeometry()
            assert screen.contains(frame), (
                f"the {name} card opened at {frame} with the parent {where}, "
                f"which is outside {screen}"
            )
            buttons = dialog.findChildren(QtWidgets.QAbstractButton)
            assert buttons, f"the {name} card has no button to reach"
            for button in buttons:
                rect = QtCore.QRect(
                    button.mapToGlobal(QtCore.QPoint(0, 0)), button.size()
                )
                assert screen.contains(rect), (
                    f"{name}: {button.text()!r} is off the screen with the "
                    f"parent {where}"
                )
            dialog.close()
            dialog.deleteLater()
        app.processEvents()
    parent.close()


def test_a_frameless_card_can_be_moved_because_it_is_its_own_title_bar() -> None:
    """Taking the frame away takes the drag handle with it.

    Clamping decides where a card OPENS; this is what lets the operator move
    it afterwards -- over the text it is asking about, or off a monitor edge
    the clamp had to push it away from.  The card itself is the handle: Qt
    delivers the press here only once every child has declined it, so the
    buttons still click and the message body still selects.
    """

    pytest.importorskip("PyQt5")
    from PyQt5 import QtCore, QtGui, QtWidgets

    from zlc_ui.qt import ensure_qt_app
    from zlc_ui.fluent.fluent import _FluentMessageDialog

    app = ensure_qt_app(["modal-drag"])
    parent = QtWidgets.QWidget()
    parent.resize(600, 400)
    parent.move(300, 200)
    parent.show()
    app.processEvents()

    dialog = _FluentMessageDialog(parent, "Pulse", "saved", kind="info")
    dialog.adjustSize()
    dialog.move(dialog._opening_position())
    dialog.show()
    app.processEvents()

    start = dialog.frameGeometry().topLeft()
    grab = start + QtCore.QPoint(dialog.width() // 2, 8)
    shift = QtCore.QPoint(60, 40)
    for kind, at, buttons in (
        (QtCore.QEvent.MouseButtonPress, grab, QtCore.Qt.LeftButton),
        (QtCore.QEvent.MouseMove, grab + shift, QtCore.Qt.LeftButton),
        (QtCore.QEvent.MouseButtonRelease, grab + shift, QtCore.Qt.NoButton),
    ):
        QtWidgets.QApplication.sendEvent(
            dialog,
            QtGui.QMouseEvent(
                kind,
                dialog.mapFromGlobal(at),
                at,
                QtCore.Qt.LeftButton,
                buttons,
                QtCore.Qt.NoModifier,
            ),
        )
        app.processEvents()

    assert dialog.frameGeometry().topLeft() - start == shift, (
        "dragging the card did not move it, so a card the clamp could not "
        "place well is a card nobody can move"
    )
    dialog.close()
    parent.close()


def test_the_full_dialog_window_opens_on_a_screen_too() -> None:
    """The other modal this project owns is placed by hand as well.

    ``FluentDialogWindow`` is nine tenths of the screen and centres on the
    window that opened it, so an anchor near an edge put its title bar above
    the top of the monitor -- and the title bar is the only thing that could
    have moved it back.
    """

    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets

    from zlc_ui.fluent import FluentDialogWindow, screen_fit_window_size
    from zlc_ui.fluent.fluent import _placed_on_screen
    from zlc_ui.qt import ensure_qt_app

    app = ensure_qt_app(["dialog-window-placement"])
    screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
    anchor = QtWidgets.QWidget()
    anchor.resize(600, 400)

    for where, (x, y) in {
        "middle": (screen.x() + 300, screen.y() + 200),
        "right edge": (screen.x() + screen.width() - 120, screen.y() + 100),
        "off the left": (screen.x() - 500, screen.y() + 100),
    }.items():
        anchor.move(x, y)
        anchor.show()
        app.processEvents()
        dialog = FluentDialogWindow(
            widget=QtWidgets.QWidget(), title="Review", anchor=anchor
        )
        # The placement of exec_, without entering its loop.
        dialog.resize(screen_fit_window_size(0.9))
        frame = dialog.frameGeometry()
        frame.moveCenter(anchor.window().frameGeometry().center())
        dialog.move(_placed_on_screen(frame.topLeft(), dialog.size(), widget=anchor))
        dialog.show()
        app.processEvents()
        assert screen.contains(dialog.frameGeometry()), (
            f"the dialog window opened at {dialog.frameGeometry()} with the "
            f"anchor {where}, which is outside {screen}"
        )
        dialog.close()
        app.processEvents()
    anchor.close()
