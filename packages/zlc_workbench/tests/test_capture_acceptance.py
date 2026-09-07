"""The acceptance capture opens each window the way its launcher does."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import zou_lab_control  # noqa: F401  (layer path bootstrap)

from zlc_workbench.tools.capture_acceptance import _opener


def test_the_figure_capture_forwards_the_explicit_workspace(monkeypatch) -> None:
    """A workspace named on the command line reaches the figure window.

    The console, pulse and device branches forwarded it; the figure branch
    passed only the path, so the viewer discovered and prepared a workspace
    of its own -- and its day folder and save location were not the ones
    the capture had been told to use.
    """

    import zlc_workbench.apps.figure_viewer as viewer

    opened: dict = {}
    monkeypatch.setattr(
        viewer, "create_window", lambda **kwargs: opened.update(kwargs)
    )
    arguments = SimpleNamespace(
        view="figure",
        path=Path("example.npz"),
        workspace=Path("explicit-workspace"),
        template=None,
        pulse=None,
        connect=None,
    )
    _opener(arguments, 0.5)()
    assert opened == {
        "path": Path("example.npz"),
        "workspace": Path("explicit-workspace"),
        "window_ratio": 0.5,
    }
