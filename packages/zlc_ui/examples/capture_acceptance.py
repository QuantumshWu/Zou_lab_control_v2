"""Capture one public zlc_ui window entry for human acceptance.

Run this on the real desktop, without ``QT_QPA_PLATFORM=offscreen``::

    python examples/capture_acceptance.py --view console
    python examples/capture_acceptance.py --view figure
    python examples/capture_acceptance.py --view pulse
    python examples/capture_acceptance.py --view device
    python examples/capture_acceptance.py --view gallery

The sizing, validation, DPR handling, and screen-proportion composition live
in :mod:`zlc_ui.acceptance`; this file is only a small CLI/example adapter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import zlc_ui.acceptance as _acceptance  # noqa: E402
import zlc_ui.fluent as _fluent  # noqa: E402
from zlc_ui import ensure_qt_app  # noqa: E402


def open_acceptance_window(
    view: str,
    *,
    ratio: float = _fluent.WINDOW_SCREEN_FRACTION,
):
    """Open one formal acceptance state through the public human entry."""

    app = ensure_qt_app(["zlc-ui-acceptance", str(view)])
    creators = {
        "console": ("demo_console", "TaskConsole@Zou lab"),
        "figure": ("demo_figure_viewer", "FigureViewer@Zou lab"),
        "pulse": ("demo_pulse_editor", "PulseGUI@Zou lab"),
        "device": ("demo_device_manager", "DeviceManager@Zou lab"),
        "gallery": ("gallery", "zlc_ui control gallery"),
    }
    try:
        module_name, _title = creators[str(view)]
    except KeyError as error:
        raise ValueError(f"unknown acceptance view {view!r}") from error
    module = __import__(module_name)
    creator = getattr(module, "create_window", None)
    if not callable(creator):
        raise TypeError(f"{module_name}.create_window is not callable")
    return app, creator(window_ratio=float(ratio))


def _print_report(report: _acceptance.AcceptanceCapture) -> None:
    values = report.as_dict()
    print(f"platform={values['platform']}")
    print(f"available_logical={values['available_logical']}")
    print(f"screen_logical={values['screen_logical']}")
    print(f"screen_dpr={values['screen_dpr']}")
    print(f"target_logical={values['target_logical'][0]}x{values['target_logical'][1]}")
    print(f"window_logical={values['window_logical'][0]}x{values['window_logical'][1]}")
    print(
        "window_fraction="
        f"{values['window_fraction'][0]:.4f}x{values['window_fraction'][1]:.4f}"
    )
    print(f"window_frame={values['window_frame']}")
    print(f"grab_physical={values['grab_physical'][0]}x{values['grab_physical'][1]}")
    print(f"desktop_relative_physical={values['desktop_relative_physical']}")
    print(f"title_bar_logical={values['title_bar_logical']}")
    print(f"content_top_logical={values['content_top_logical']}")
    print(f"content_top_margin={values['content_top_margin']}")
    print(f"output={values['output']}")
    print(f"desktop_relative_output={values['desktop_relative_output']}")
    print("acceptance_mode=real-screen")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        choices=("console", "figure", "pulse", "device", "gallery"),
        required=True,
    )
    parser.add_argument("--ratio", type=float, default=_fluent.WINDOW_SCREEN_FRACTION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--screen-output", type=Path)
    args = parser.parse_args(argv)

    output = args.output or ROOT / "artifacts" / f"acceptance-{args.view}-window.png"
    screen_output = args.screen_output or ROOT / "artifacts" / f"acceptance-{args.view}-screen.png"
    report = _acceptance.capture_window(
        lambda: open_acceptance_window(args.view, ratio=float(args.ratio))[1],
        ratio=float(args.ratio),
        output=output,
        desktop_output=screen_output,
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
