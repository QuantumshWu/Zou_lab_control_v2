"""Open a saved figure.

    zlc figure_viewer --path D:/experiment/data/2026_08_05/run.npz

The archive is the whole input.  This window needs no session, no devices and
no apparatus file: a figure saved on the bench opens on a laptop months later,
which is the only reason the record was written into the file in the first
place.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browse a saved figure archive.")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="archive to open at startup (otherwise use the window's File field)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="open and report without showing a window (a startup smoke test)",
    )
    return parser


def build(
    view: object,
    *,
    workspace: object | None = None,
    run_off_thread,
    close_worker,
    request_close,
) -> object:
    """Wire one viewer window, so a host embedding it does not repeat this."""

    from ..panel_sizes import install as install_panel_sizes

    install_panel_sizes()
    from datetime import date
    from types import SimpleNamespace

    from zlc_durable import day_folder
    from zlc_runtime import SignalDataPlane
    from ..console import ConsolePresenter
    from ..device_use import DeviceUseCoordinator
    from ..panel_catalog import task_console_fitting_spec
    from ..viewer import FigureViewerPresenter

    if workspace is None:
        from ..session import Workspace

        space = Workspace.discover().prepare()
        workspace = SimpleNamespace(
            root=space.root,
            data=space.data,
            today=day_folder(space.data, date.today()),
        )

    plane = SignalDataPlane()
    session = SimpleNamespace(
        signal_plane=plane,
        workspace=workspace,
        installation=SimpleNamespace(devices={}, revision=0),
        device_use=DeviceUseCoordinator(),
        day_folder_path=lambda: workspace.today,
        resolve_device_setting_records=lambda _records: (),
    )

    def make_host(plot_input, state):
        from .task_console import build_panel_host

        return build_panel_host(
            plot_input,
            state,
            device_pixel_ratio=float(view.device_pixel_ratio()),
        )

    def spec_for(snapshot, kind: str = "", cell_kind: str = ""):
        return task_console_fitting_spec(
            snapshot.block.schema,
            kind,
            cell_kind,
        )

    panels = ConsolePresenter(
        session,
        view,
        make_host=make_host,
        spec_for=spec_for,
        run_off_thread=run_off_thread,
        close_worker=lambda: True,
        panel_only=True,
    )

    return FigureViewerPresenter(
        view,
        run_off_thread=run_off_thread,
        close_worker=close_worker,
        request_close=request_close,
        panel_presenter=panels,
        signal_plane=plane,
    )


def create_window(
    *,
    path=None,
    workspace=None,
    window_ratio: float | None = None,
):
    """Open the viewer the way a human does, and return its window.

    The non-blocking public entry: what the launcher runs, what a notebook
    calls, and what zlc_ui's acceptance capture opens.
    """

    from datetime import date
    from types import SimpleNamespace

    from zlc_durable import day_folder
    from zlc_ui import open_figure_viewer
    from zlc_workbench.board import (
        attach_qt,
        attach_qt_owner_turn,
        attach_qt_worker,
    )
    from ..session import Workspace

    space = (
        Workspace.discover()
        if workspace is None
        else Workspace(workspace)
    ).prepare()
    today = day_folder(space.data, date.today())
    viewer_workspace = SimpleNamespace(
        root=space.root,
        data=space.data,
        today=today,
    )

    # One call, one handle: this layer never names a widget class.
    window = open_figure_viewer(
        title="FigureViewer@Zou lab",
        window_ratio=window_ratio,
        path_base_dir=str(today),
    )
    run_off_thread, close_worker = attach_qt_worker("zlc-figure-viewer")
    try:
        window.presenter = build(
            window,
            workspace=viewer_workspace,
            run_off_thread=run_off_thread,
            close_worker=close_worker,
            request_close=window.close_later,
        )
    except BaseException:
        close_worker()
        window.close()
        raise
    window.set_close_guard(window.presenter.close)
    panel_presenter = window.presenter._panel_presenter
    panel_presenter.board.wake.set_notify(
        attach_qt_owner_turn(window.presenter.commit_surfaces)
    )
    window.presenter.timer = attach_qt(
        window.presenter.beat,
        interval_ms=panel_presenter.board.base_interval_ms,
    )
    if path is not None:
        window.presenter.open(str(path))
    return window


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    if arguments.check:
        try:
            if arguments.path is None:
                print("figure viewer ready: no archive given")
                return 0
            from zlc_data.figure_archive import read_archive
            from ..viewer import describe_archive

            description = describe_archive(*read_archive(arguments.path))
            print(
                f"figure ready: {description.name!r}, "
                f"{len(description.datasets)} dataset(s), "
                f"{sum(len(rows) for _title, rows in description.tabs)} record row(s)"
            )
            return 0
        except Exception as error:
            print(f"error: could not read {arguments.path}: {error}", file=sys.stderr)
            return 2

    from zlc_ui import ensure_qt_app

    application = ensure_qt_app([])

    window = create_window(path=arguments.path)
    del window
    return int(application.exec_())
