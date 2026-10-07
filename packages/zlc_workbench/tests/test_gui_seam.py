"""Nothing here may name a Qt class, or reach past a window's handle.

A plot's surface is the one widget that crosses this layer, and only as
zlc_plot hands it out -- ``host.qt_widget`` for a panel card,
``ImagePointReviewSurface`` for a site review -- to be passed to the handle
whole: nothing here builds a widget of its own or lays one out.

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

#: Qt lives in exactly one module here, board.py: the shims that move a wake
#: or a worker's answer onto the owner thread, and the timer that drives the
#: display beat.  They are about THREADS and events, not about widgets --
#: none of them builds, holds or shows one.
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


def test_no_module_here_imports_qt_outside_board() -> None:
    """A composition root that can construct a widget will assemble a UI.

    Importing PyQt5 at all is the check, because there is no widget-free half
    of it worth carving out: the one module that legitimately needs Qt needs
    it for threads and timers, and it is named above.  It does not see the
    widgets zlc_plot builds and this layer holds on their way to a handle --
    ``host.qt_widget``, ``ImagePointReviewSurface`` -- which are the permitted
    ones: zlc_plot builds them whole and this layer only passes them on.
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
