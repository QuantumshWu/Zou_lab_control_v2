"""Edit this bench's apparatus.

    python -m zlc_workbench.apps.device_manager --workspace D:/experiment

Which devices exist and how each is set up is the one thing a session cannot
start without, and it was the one thing with no window.  This writes
``apparatus.json`` in the workspace, the same file every other entry point
reads.

The standalone editor opens no device.  The unified experiment entry injects
the Init/Shutdown callbacks that retain the one shared session; editing the
same apparatus still remains possible on a laptop with no hardware attached.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zlc_atom.install import DeviceCatalogSnapshot


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


def build(
    view: object,
    path: Path,
    *,
    catalog: DeviceCatalogSnapshot | None = None,
    initial_config: object | None = None,
    initialize_session=None,
    on_initialized=None,
    shutdown_session=None,
) -> object:
    """One presenter over one apparatus file, with the view it drives."""

    from ..device_manager import DeviceManagerPresenter

    return DeviceManagerPresenter(
        view,
        path,
        catalog=catalog,
        confirm_overwrite=lambda _path: True,
        initial_config=initial_config,
        initialize_session=initialize_session,
        on_initialized=on_initialized,
        shutdown_session=shutdown_session,
    )


def create_window(
    *,
    workspace=None,
    window_ratio=None,
    catalog: DeviceCatalogSnapshot | None = None,
    initial_config: object | None = None,
    initialize_session=None,
    on_initialized=None,
    shutdown_session=None,
):
    """Open the apparatus editor and return its window."""

    from zlc_ui import open_device_manager

    path = apparatus_path(workspace)
    # One call, one handle: this layer never names a widget class.
    window = open_device_manager(title="Devices@Zou lab", window_ratio=window_ratio)
    window.presenter = build(
        window,
        path,
        catalog=catalog,
        initial_config=initial_config,
        initialize_session=initialize_session,
        on_initialized=on_initialized,
        shutdown_session=shutdown_session,
    )
    window.closed.connect(window.presenter.close)
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
