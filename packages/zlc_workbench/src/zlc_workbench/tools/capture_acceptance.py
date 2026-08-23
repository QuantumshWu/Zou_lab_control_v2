"""Capture one workbench window for visual acceptance, on a real screen.

    zlc capture --view console --template virtual
    zlc capture --view pulse --pulse imaging_template.json
    zlc capture --view figure --path run.npz
    zlc capture --view device --workspace D:/experiment

This is an adapter, not a capture implementation.  zlc_ui owns the only one:
``zlc_ui.acceptance.capture_window`` opens the same ``create_window`` entry a
human uses, checks the window is exactly the shared screen-fit size, and writes
a physical-DPR crop.  It rejects the offscreen backend on purpose -- an
offscreen canvas cannot reproduce the monitor the picture is meant to inspect,
and a screenshot taken any other way is a picture of a window nobody will see.

A wrong launcher therefore fails loudly here instead of producing a plausible
image at a size somebody typed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _opener(arguments: argparse.Namespace, ratio: float):
    """The window entry for one view, with this run's arguments bound."""

    if arguments.view == "console":
        from ..apps.task_console import create_window

        return lambda: create_window(
            workspace=arguments.workspace,
            template=arguments.template,
            window_ratio=ratio,
        )
    if arguments.view == "pulse":
        from ..apps.pulse_editor import create_window

        return lambda: create_window(
            workspace=arguments.workspace,
            pulse=arguments.pulse,
            connect=arguments.connect,
            window_ratio=ratio,
        )
    if arguments.view == "device":
        from ..apps.device_manager import create_window

        return lambda: create_window(
            workspace=arguments.workspace,
            window_ratio=ratio,
        )
    from ..apps.figure_viewer import create_window

    return lambda: create_window(path=arguments.path, window_ratio=ratio)


def main(argv: list[str] | None = None) -> int:
    from zlc_ui import WINDOW_SCREEN_FRACTION, capture_window

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view", choices=("console", "pulse", "figure", "device"), required=True
    )
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--template", default=None)
    parser.add_argument("--pulse", default=None)
    parser.add_argument("--connect", default=None)
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--ratio", type=float, default=WINDOW_SCREEN_FRACTION)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--screen-output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    output = arguments.output or Path(f"acceptance-{arguments.view}-window.png")
    report = capture_window(
        _opener(arguments, float(arguments.ratio)),
        ratio=float(arguments.ratio),
        output=output,
        desktop_output=arguments.screen_output,
    )
    values = report.as_dict()
    from importlib.metadata import version
    import zou_lab_control
    import zlc_ui
    import zlc_workbench

    print(f"product_version={version('zou-lab-control')}")
    print(f"product_bootstrap={Path(zou_lab_control.__file__).resolve()}")
    print(f"ui_layer={Path(zlc_ui.__file__).resolve()}")
    print(f"workbench_layer={Path(zlc_workbench.__file__).resolve()}")
    for key in (
        "platform",
        "screen_logical",
        "screen_dpr",
        "target_logical",
        "window_logical",
        "window_fraction",
        "grab_physical",
        "content_top_margin",
        "output",
    ):
        print(f"{key}={values[key]}")
    return 0
