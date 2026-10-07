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
    workspace: object,
    run_off_thread,
    close_worker,
    request_close,
    monitor_render,
    editor_render,
    close_render_processes,
) -> object:
    """Wire one viewer window, so a host embedding it does not repeat this."""

    from ..panel_sizes import install as install_panel_sizes

    install_panel_sizes()
    from datetime import date
    from types import SimpleNamespace

    from zlc_durable import day_folder_path
    from zlc_runtime import SignalDataPlane
    from ..console import ConsolePresenter
    from ..device_use import DeviceUseCoordinator
    from functools import partial

    from ..pulse_preview import build_pulse_preview_host, resize_pulse_preview_host
    from ..viewer import FigureViewerPresenter
    from .task_console import panel_host_factories

    plane = SignalDataPlane()
    session = SimpleNamespace(
        signal_plane=plane,
        workspace=workspace,
        installation=SimpleNamespace(devices={}, revision=0),
        device_use=DeviceUseCoordinator(),
        day_folder_path=lambda: day_folder_path(workspace.data, date.today()),
        resolve_device_setting_records=lambda _records: (),
    )

    make_monitor_host, make_editor_host = panel_host_factories(
        view, monitor_render, editor_render
    )
    panels = ConsolePresenter(
        session,
        view,
        make_monitor_host=make_monitor_host,
        make_editor_host=make_editor_host,
        save_figure_artifact=editor_render.save_figure_artifact,
        close_render_processes=close_render_processes,
        panel_only=True,
    )

    return FigureViewerPresenter(
        view,
        run_off_thread=run_off_thread,
        close_worker=close_worker,
        request_close=request_close,
        panel_presenter=panels,
        signal_plane=plane,
        build_figure_host=editor_render.build_host,
        save_figure_artifact=editor_render.save_figure_artifact,
        confirm_discard=getattr(view, "confirm_discard", None),
        make_pulse_preview=partial(build_pulse_preview_host, build_host=editor_render.build_host),
        resize_pulse_preview=resize_pulse_preview_host,
    )


def create_window(
    *,
    path=None,
    workspace=None,
    window_ratio: float | None = None,
    monitor_render=None,
    editor_render=None,
):
    """Open the viewer the way a human does, and return its window.

    The non-blocking public entry: what the launcher runs, what a notebook
    calls, and what zlc_ui's acceptance capture opens.
    """

    from datetime import date
    from types import SimpleNamespace

    from zlc_durable import day_folder
    import zlc_plot as plot
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
    viewer_workspace = SimpleNamespace(root=space.root, data=space.data)

    if (monitor_render is None) != (editor_render is None):
        raise ValueError(
            "monitor_render and editor_render must be supplied together"
        )
    owns_render_processes = monitor_render is None
    if owns_render_processes:
        monitor_render = plot.RenderProcessPool("zlc-monitor-render")
        try:
            editor_render = plot.RenderProcess("zlc-edit-save-render")
        except BaseException:
            monitor_render.release(timeout=30.0)
            raise

    assert monitor_render is not None and editor_render is not None
    if not owns_render_processes:
        monitor_render.retain()
        try:
            editor_render.retain()
        except BaseException:
            monitor_render.release(timeout=0.0)
            raise

    # One call, one handle: this layer never names a widget class.  The
    # panels are a console board, so they get the console's staging policy.
    from .task_console import render_processes_closer, staged_panel_surface

    close_render_processes = render_processes_closer(monitor_render, editor_render)

    try:
        window = open_figure_viewer(
            title="FigureViewer@Zou lab",
            window_ratio=window_ratio,
            path_base_dir=str(today),
            plot_surface=staged_panel_surface,
        )
    except BaseException:
        close_render_processes(30.0)
        raise
    run_off_thread, close_worker = attach_qt_worker("zlc-figure-viewer")
    try:
        window.presenter = build(
            window,
            workspace=viewer_workspace,
            run_off_thread=run_off_thread,
            close_worker=close_worker,
            request_close=window.close_later,
            monitor_render=monitor_render,
            editor_render=editor_render,
            close_render_processes=close_render_processes,
        )
    except BaseException:
        close_worker()
        close_render_processes(30.0)
        window.close()
        raise
    window.set_close_guard(window.presenter._guarded(window.presenter.close))
    panel_presenter = window.presenter._panel_presenter
    panel_presenter.board.wake.set_notify(
        attach_qt_owner_turn(window.presenter._guarded(window.presenter.commit_surfaces))
    )
    window.presenter.timer = attach_qt(
        window.presenter._guarded(window.presenter.beat),
        interval_ms=panel_presenter.board.base_interval_ms,
        board=panel_presenter.board,
    )
    if path is not None:
        window.presenter._guarded(window.presenter.open)(str(path))
    return window


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    if arguments.check:
        try:
            if arguments.path is None:
                print("figure viewer ready: no archive given")
                return 0
            # What the window reads when it opens, every Dataset and its
            # recipe included -- a check of less passed archives that then
            # did not open.
            from ..viewer import read_figure_archive

            _resolved, description, *_rest = read_figure_archive(arguments.path)
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
