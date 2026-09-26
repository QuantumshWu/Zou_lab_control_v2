"""Every window opens through zlc_ui's one launcher.

zlc_ui owns what a window IS here: the frameless Fluent chrome, the shared
display scale resolved from the screen, the screen-fit initial size, centring,
retention on the app registry, and the close handshake with a body that guards
its own close.  Its launcher docstring says so, and says why -- hand-copied
launchers drift, and one had already silently dropped ensure_qt_app and the
shared scale.

These apps drifted anyway: they called ensure_qt_app and then view.show(), so
the windows arrived with no chrome, at a size nobody had computed, and -- since
the bodies refuse their own close and wait to be told -- could not be closed at
all.

So one test asserts the rule mechanically, and one checks the result.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

APPS = Path(__file__).resolve().parents[1] / "src" / "zlc_workbench" / "apps"

#: Ways of putting a window on screen that bypass the launcher's lifecycle.
FORBIDDEN_CALLS = {"show", "resize", "setWindowTitle", "setFixedSize", "adjustSize"}


def test_no_app_opens_a_window_by_hand() -> None:
    """The mechanical half: nothing here may size or show a top-level window."""

    offenders: list[str] = []
    launchers: set[str] = set()
    for path in sorted(APPS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS:
                    offenders.append(f"{path.name}: .{node.func.attr}()")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"open_fluent_window", "launch_fluent_window"}:
                    launchers.add(path.name)
    assert offenders == [], offenders
    # pulse_editor.py is not here any more, and that is the direction of
    # travel: it opens its window with zlc_ui's own one-call entry
    # (open_pulse_editor), which owns the launcher on the other side of the
    # wall.  The three still listed call the launcher themselves because they
    # still build their bodies themselves; each moves as its handle lands.
    # Empty, and that is the finish line: every window here is opened by
    # zlc_ui's own one-call entry, which owns the launcher on the other side.
    assert launchers == set()


@pytest.mark.parametrize(
    ("opener", "content_minimum"),
    [
        pytest.param("pulse_editor", True, id="pulse editor"),
        pytest.param("figure_viewer", False, id="figure viewer"),
    ],
)
def test_a_sealed_window_is_reached_only_through_its_handle(
    opener: str, content_minimum: bool
) -> None:
    """The same three facts, asked of a handle instead of a widget.

    What comes back is deliberately NOT a QWidget: an outside layer that can
    hold one will sooner or later assemble a UI, which is the one job it does
    not have.  So the size rule, the title and the working X are all questions
    the handle answers, and there is nothing else on it to reach through.

    The console is left out only because opening it opens devices; its entry is
    exercised by the app tests and by the acceptance capture.

    create_window is the shape zlc_ui's acceptance capture opens, and it is
    opened here the way the capture opens it: with the ratio the capture
    passes.  An app with only a blocking main() cannot be inspected at all.
    """

    from PyQt5 import QtWidgets

    from zlc_ui.fluent import WINDOW_SCREEN_FRACTION, screen_fit_window_size
    from zlc_ui.qt import ensure_qt_app

    application = ensure_qt_app(["window-geometry"])
    module = __import__(f"zlc_workbench.apps.{opener}", fromlist=["create_window"])

    window = module.create_window(window_ratio=WINDOW_SCREEN_FRACTION)
    try:
        assert not isinstance(window, QtWidgets.QWidget), "a widget escaped zlc_ui"
        assert window.window_title().endswith("@Zou lab")
        target = screen_fit_window_size(WINDOW_SCREEN_FRACTION)
        width, height = window.window_size()
        assert height == target.height()
        if content_minimum:
            # Qt never shrinks a window below its content's minimum: on the
            # 800x600 offscreen screen the pulse editor's schedule toolbar is
            # wider than the fit width, so the window opens at that minimum --
            # still on the screen, never narrower than the shared rule.
            screen = application.primaryScreen().availableGeometry()
            assert target.width() <= width <= screen.width(), (width, target.width())
        else:
            assert width == target.width()
        assert window.is_visible()

        window.close()
        # A hardware-owning window retires off the Qt thread and only commits
        # close on the completion turn; wait for that formal handshake rather
        # than treating one arbitrary event-loop turn as the lifecycle API.
        deadline = time.monotonic() + 2.0
        while window.is_visible() and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(0.005)
        assert not window.is_visible(), "the window could not be closed"
    finally:
        application.processEvents()


def test_qt_worker_refuses_to_claim_closed_while_vendor_work_is_hung() -> None:
    from threading import Event

    from zlc_ui.qt import ensure_qt_app
    from zlc_workbench.board import attach_qt_worker

    application = ensure_qt_app(["device-worker-close"])
    release = Event()
    delivered: list[str] = []
    run, close = attach_qt_worker("device-worker-test")
    run(
        lambda: release.wait(5.0) or "timed-out",
        lambda value: delivered.append(str(value)),
        lambda error: delivered.append(str(error)),
    )
    assert close() is False
    release.set()
    deadline = time.monotonic() + 2.0
    while not delivered and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    assert delivered == ["True"]
    assert close() is True
    with pytest.raises(RuntimeError, match="closed"):
        run(lambda: None, lambda _value: None, lambda _error: None)


def test_a_qt_driven_beat_never_ends_the_process() -> None:
    """The one hop where a callable becomes a slot is a total boundary.

    Every view signal in the console is wrapped for this reason and the
    beat -- which runs continuously and touches everything -- was
    connected raw.  An exception leaving a Qt slot is not reported
    anywhere: PyQt calls qFatal() and the session is gone, panels,
    experiment and traceback with it.  A raising beat must cost a log
    line and the next tick.
    """

    from PyQt5 import QtCore

    from zlc_ui import ensure_qt_app
    from zlc_workbench.board import attach_qt

    ensure_qt_app()
    beats: list[int] = []

    def beat() -> None:
        beats.append(len(beats))
        raise LookupError("signal 'x' is not retained")

    timer = attach_qt(beat, interval_ms=1)
    try:
        deadline = time.monotonic() + 5.0
        while len(beats) < 3 and time.monotonic() < deadline:
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.002)
        assert len(beats) >= 3, (
            "the timer stopped at the first raising beat"
        )
    finally:
        timer.stop()
