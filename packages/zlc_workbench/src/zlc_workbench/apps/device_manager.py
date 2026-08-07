"""Edit this bench's apparatus.

    python -m zlc_workbench.apps.device_manager --workspace D:/experiment

Which devices exist and how each is set up is the one thing a session cannot
start without, and it was the one thing with no window.  This writes
``apparatus.json`` in the workspace, the same file every other entry point
reads.

It opens no device.  Writing down that the bench has a camera at index 2 is a
different act from reaching for it, and an apparatus has to be editable from a
laptop with no hardware attached.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edit a workspace's apparatus.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help=(
            "directory holding pulses/, data/ and apparatus.json "
            "(default: found at or above this one)"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="build everything and exit without opening a window",
    )
    return parser


def apparatus_path(workspace=None) -> Path:
    """The apparatus this window edits, found the way every entry finds it."""

    from ..session import Workspace

    space = Workspace(workspace) if workspace is not None else Workspace.discover()
    return space.apparatus


def build(view: object, path: Path) -> object:
    """One presenter over one apparatus file, with the view it drives."""

    from ..device_manager import DeviceManagerPresenter

    return DeviceManagerPresenter(view, path, confirm_overwrite=lambda _path: True)


def create_window(*, workspace=None, window_ratio=None):
    """Open the apparatus editor and return its window."""

    from zlc_ui import open_device_manager

    path = apparatus_path(workspace)
    # One call, one handle: this layer never names a widget class.
    window = open_device_manager(title="Devices@Zou lab", window_ratio=window_ratio)
    window.presenter = build(window, path)
    return window


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    from zlc_ui import ensure_qt_app

    application = ensure_qt_app([])

    try:
        path = apparatus_path(arguments.workspace)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"apparatus: {path}", flush=True)

    if arguments.check:
        from zlc_ui import open_device_manager

        # The same composition, through the same entry: a smoke test.
        presenter = build(open_device_manager(window_ratio=0.4), path)
        print(
            f"device manager ready: {len(presenter.devices)} device(s), "
            f"{len(presenter.types)} type(s) offerable"
        )
        return 0

    create_window(workspace=arguments.workspace)
    return int(application.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
