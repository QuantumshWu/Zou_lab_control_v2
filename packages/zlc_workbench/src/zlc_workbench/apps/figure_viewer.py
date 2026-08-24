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
    run_off_thread,
    close_worker,
    request_close,
) -> object:
    """Wire one viewer window, so a host embedding it does not repeat this."""

    from ..panel_sizes import install as install_panel_sizes

    install_panel_sizes()
    import zlc_plot as plot
    from ..viewer import FigureViewerPresenter

    def make_host(plot_input, name: str, recipe):
        del name
        return plot.open_figure_host(plot_input, recipe)

    return FigureViewerPresenter(
        view,
        make_host=make_host,
        run_off_thread=run_off_thread,
        close_worker=close_worker,
        request_close=request_close,
    )


def create_window(*, path=None, window_ratio: float | None = None):
    """Open the viewer the way a human does, and return its window.

    The non-blocking public entry: what the launcher runs, what a notebook
    calls, and what zlc_ui's acceptance capture opens.
    """

    from zlc_ui import open_figure_viewer
    from zlc_workbench.board import attach_qt_worker

    # One call, one handle: this layer never names a widget class.
    window = open_figure_viewer(
        title="FigureViewer@Zou lab", window_ratio=window_ratio
    )
    run_off_thread, close_worker = attach_qt_worker("zlc-figure-viewer")
    try:
        window.presenter = build(
            window,
            run_off_thread=run_off_thread,
            close_worker=close_worker,
            request_close=window.close_later,
        )
    except BaseException:
        close_worker()
        window.close()
        raise
    window.set_close_guard(window.presenter.close)
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
