"""A widget leaves the screen through one door, and never as a window.

Adding a widget to a layout under a visible parent QUEUES a show of it.
``setParent(None)`` on a widget that was not explicitly hidden clears the
explicit-hide mark, so a widget added and unparented in the same pass is
shown by that queued event after it has lost its parent -- as a top-level
window, on the desktop, for the one frame before its deferred delete runs.
Thirteen retirements in this project wrote that idiom by hand; the small
window that flashed at Add axis, at a pulse reopen and at a closed editor
tab was every one of them.  The two verbs in zlc_ui.fluent are now the only
way out, and this file is what keeps it that way.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
PRODUCT_ROOTS = tuple(
    REPO / "packages" / package / "src"
    for package in ("zlc_ui", "zlc_atom", "zlc_workbench", "zlc_plot")
)
THE_ONLY_PLACE = REPO / "packages/zlc_ui/src/zlc_ui/fluent/fluent.py"


def test_no_product_code_unparents_a_widget_by_hand() -> None:
    offenders = sorted(
        str(path.relative_to(REPO))
        for root in PRODUCT_ROOTS
        for path in root.rglob("*.py")
        if path != THE_ONLY_PLACE and "setParent(None)" in path.read_text(encoding="utf-8")
    )
    assert offenders == [], (
        "a widget is retired with retire_widget() or moved with detach_widget(); "
        f"these unparent by hand: {offenders}"
    )


def _window_shows(app):
    """Every Show delivered to a widget that is a window, as it happens."""

    from PyQt5 import QtCore, QtWidgets

    seen: list[str] = []

    class Filter(QtCore.QObject):
        def eventFilter(self, watched, event):
            if (
                event.type() == QtCore.QEvent.Show
                and isinstance(watched, QtWidgets.QWidget)
                and watched.isWindow()
            ):
                seen.append(type(watched).__name__)
            return False

    holder = Filter(app)
    app.installEventFilter(holder)
    return seen, holder


def _pump(app, passes: int = 20) -> None:
    from PyQt5 import QtCore

    for _ in range(passes):
        app.processEvents(QtCore.QEventLoop.AllEvents, 10)
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.fixture
def stage():
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets
    from zlc_ui.qt import ensure_qt_app

    app = ensure_qt_app(["widget-retirement"])
    parent = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(parent)
    parent.show()
    _pump(app)
    yield app, parent, layout
    parent.close()
    parent.deleteLater()
    _pump(app)


def test_the_hand_written_idiom_really_does_map_a_window(stage) -> None:
    """The mechanism this file exists for, demonstrated -- so the guard bites."""

    from PyQt5 import QtWidgets

    app, parent, layout = stage
    seen, holder = _window_shows(app)
    child = QtWidgets.QLabel("row", parent)
    layout.addWidget(child)
    child.setParent(None)
    child.deleteLater()
    _pump(app)
    app.removeEventFilter(holder)
    assert "QLabel" in seen, "Qt no longer shows an unparented child; revisit this file"


def test_a_retired_widget_is_never_shown_as_a_window(stage) -> None:
    from PyQt5 import QtWidgets, sip
    from zlc_ui.fluent import retire_widget

    app, parent, layout = stage
    seen, holder = _window_shows(app)
    child = QtWidgets.QLabel("row", parent)
    layout.addWidget(child)
    retire_widget(child)
    _pump(app)
    app.removeEventFilter(holder)
    assert seen == [], seen
    assert sip.isdeleted(child), "retired means gone, not merely hidden"


def test_a_detached_widget_is_hidden_until_its_next_host_shows_it(stage) -> None:
    from PyQt5 import QtWidgets
    from zlc_ui.fluent import detach_widget

    app, parent, layout = stage
    seen, holder = _window_shows(app)
    child = QtWidgets.QLabel("surface", parent)
    layout.addWidget(child)
    detach_widget(child)
    _pump(app)
    assert seen == [], seen
    assert child.parent() is None and child.isHidden()
    layout.addWidget(child)
    child.show()
    _pump(app)
    app.removeEventFilter(holder)
    assert child.isVisible() and not child.isWindow()
    assert seen == [], "a re-hosted widget is a child again, not a window"
