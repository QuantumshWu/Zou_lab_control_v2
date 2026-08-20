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
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

APPS = Path(__file__).resolve().parents[1] / "src" / "zlc_workbench" / "apps"

#: Ways of putting a window on screen that bypass the launcher's lifecycle.
FORBIDDEN_CALLS = {"show", "resize", "setWindowTitle", "setFixedSize", "adjustSize"}


def test_every_app_offers_the_one_window_entry() -> None:
    """create_window is the shape zlc_ui's acceptance capture opens.

    An app with only a blocking main() cannot be inspected by the one capture
    API at all, so its window is whatever the screenshot script felt like --
    which is how this session produced pictures of windows nobody would see.
    """

    for name in ("task_console", "pulse_editor", "figure_viewer"):
        module = __import__(f"zlc_workbench.apps.{name}", fromlist=["create_window"])
        entry = getattr(module, "create_window", None)
        assert callable(entry), f"{name} has no create_window"
        # It must accept the ratio the capture API passes.
        import inspect

        assert "window_ratio" in inspect.signature(entry).parameters, name


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
    "opener",
    [
        pytest.param("pulse_editor", id="pulse editor"),
        pytest.param("figure_viewer", id="figure viewer"),
    ],
)
def test_a_sealed_window_is_reached_only_through_its_handle(opener: str) -> None:
    """The same three facts, asked of a handle instead of a widget.

    What comes back is deliberately NOT a QWidget: an outside layer that can
    hold one will sooner or later assemble a UI, which is the one job it does
    not have.  So the size rule, the title and the working X are all questions
    the handle answers, and there is nothing else on it to reach through.

    The console is left out only because opening it opens devices; its entry is
    exercised by the app tests and by the acceptance capture.
    """

    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets

    from zlc_ui.fluent import WINDOW_SCREEN_FRACTION, screen_fit_window_size
    from zlc_ui.qt import ensure_qt_app

    application = ensure_qt_app(["window-geometry"])
    module = __import__(f"zlc_workbench.apps.{opener}", fromlist=["create_window"])

    window = module.create_window()
    try:
        assert not isinstance(window, QtWidgets.QWidget), "a widget escaped zlc_ui"
        assert window.window_title().endswith("@Zou lab")
        target = screen_fit_window_size(WINDOW_SCREEN_FRACTION)
        assert window.window_size() == (target.width(), target.height())
        assert window.is_visible()

        window.close()
        # A hardware-owning window retires off the Qt thread and only commits
        # close on the completion turn; wait for that formal handshake rather
        # than treating one arbitrary event-loop turn as the lifecycle API.
        import time

        deadline = time.monotonic() + 2.0
        while window.is_visible() and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(0.005)
        assert not window.is_visible(), "the window could not be closed"
    finally:
        application.processEvents()


def test_the_console_window_can_refuse_its_own_close() -> None:
    """A console owns a camera, a beat and a session; the X waits for them.

    The guard is what makes that true, and the port has to carry it: the app
    installed one on what became a handle, and nothing here opens the console
    window, so the AttributeError waited until someone ran the launcher.  It
    is checked against the handle rather than the window, because the handle
    is all the app can reach.
    """

    pytest.importorskip("PyQt5")
    from zlc_ui import ensure_qt_app, open_task_console

    application = ensure_qt_app(["console-close-guard"])
    console = open_task_console(window_ratio=0.4)
    try:
        refusals = []

        def _guard() -> bool:
            refusals.append(True)
            return len(refusals) > 1

        console.set_close_guard(_guard)
        console.close()
        application.processEvents()
        assert refusals, "the guard was never asked"
        assert console.is_visible(), "a refused close must leave the window up"

        console.close()
        application.processEvents()
        assert not console.is_visible(), "and the second answer must let it go"
    finally:
        application.processEvents()


def test_qt_worker_refuses_to_claim_closed_while_vendor_work_is_hung() -> None:
    from threading import Event
    import time

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
