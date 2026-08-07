"""Nothing outside zlc_ui may hold a widget, or reach past a window's handle.

This is the rule the delay-column defect came from.  A presenter that could
reach ``view.schedule_view.channel_panel`` pushed a value into one panel and
left the one beside it alone, so "which rows exist" quietly became two facts in
two places and the delay column went on showing rows that Hide Off had already
taken out of the cards.  Whatever the outside can hold, the outside will
assemble, and assembling a UI is the one job a composition root does not have.

So it is checked rather than remembered, and checked in the two ways it can be
broken: by importing a piece of the GUI package, and by naming a Qt class.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Qt lives in exactly two places here: the shim that moves a wake onto the
#: owner thread, and the timer that drives the display beat.  Both are about
#: THREADS and events, not about widgets -- neither builds, holds or shows one.
QT_IS_ALLOWED = {"board.py"}


def _python_files() -> list[Path]:
    return sorted(path for path in SRC.rglob("*.py") if "__pycache__" not in path.parts)


def test_no_module_here_reaches_into_the_gui_package() -> None:
    """zlc_ui is one facade wide: a window, a handle, and the wiring vocabulary."""

    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("zlc_ui."):
                    offenders.append(f"{path.name}: from {module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("zlc_ui."):
                        offenders.append(f"{path.name}: import {alias.name}")
    assert offenders == [], (
        "these reach past the facade; whatever they need belongs on it: "
        f"{offenders}"
    )


def test_no_module_here_builds_or_holds_a_qt_widget() -> None:
    """A composition root that can construct a widget will assemble a UI.

    Importing PyQt5 at all is the check, because there is no widget-free half
    of it worth carving out: the two modules that legitimately need Qt need it
    for threads and timers, and they are named above.
    """

    offenders: list[str] = []
    for path in _python_files():
        if path.name in QT_IS_ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = ()
            if isinstance(node, ast.ImportFrom):
                names = ((node.module or ""),)
            elif isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            for name in names:
                if name == "PyQt5" or name.startswith("PyQt5."):
                    offenders.append(f"{path.name}: {name}")
    assert offenders == [], (
        "Qt outside the GUI package: a window is opened with one call and "
        f"driven through its handle. {offenders}"
    )


def test_every_window_is_opened_by_one_call_and_returns_a_handle() -> None:
    """Four windows, four entries, and not a widget among them."""

    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")

    from PyQt5 import QtWidgets

    import zlc_ui

    for name in (
        "open_pulse_editor",
        "open_figure_viewer",
        "open_device_manager",
        "open_task_console",
    ):
        assert hasattr(zlc_ui, name), f"{name} is not on the facade"

    application = zlc_ui.ensure_qt_app(["seam"])
    for opener in (zlc_ui.open_pulse_editor, zlc_ui.open_figure_viewer):
        handle = opener(window_ratio=0.4)
        try:
            assert not isinstance(handle, QtWidgets.QWidget), "a widget escaped"
            # What a host may ask of a window, and the whole of it.
            for member in ("close", "is_visible", "window_size", "window_title"):
                assert callable(getattr(handle, member)), member
        finally:
            handle.close()
            application.processEvents()
