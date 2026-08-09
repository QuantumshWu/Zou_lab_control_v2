"""Open a saved figure.

    python -m zlc_workbench.apps.figure_viewer --path D:/experiment/data/2026_08_05/run.npz

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


def build(view: object) -> object:
    """Wire one viewer window, so a host embedding it does not repeat this."""

    from dataclasses import replace

    import zlc_plot as plot
    from zlc_plot.primitives import ImageFrame

    from ..panel_spec import fitting_panel_spec
    from ..viewer import FigureViewerPresenter

    def make_host(plot_input, name: str, kind: str, cell_kind: str):
        # Which plot suits a dataset is a question about the dataset, and the
        # saved schema answers it.  This window used to answer it itself, in a
        # two-branch copy of a rule zlc_plot already owns per kind -- so a
        # dataset that was neither a plain image nor a point table opened as an
        # exception instead of a plot.  Asking zlc_plot means one answer.
        snapshot = plot_input.snapshot if isinstance(plot_input, ImageFrame) else plot_input
        spec = fitting_panel_spec(
            snapshot.block.schema, kind, cell_kind
        )
        if spec is None:
            raise ValueError(
                f"nothing in {name!r} can be drawn as {kind or 'an inferred plot'}"
            )
        # Only the title: axis names come from the payload's own schema when a
        # label is left unset, and the signal's name is the one thing the data
        # does not know about itself.
        return plot.RasterPlotHost.from_plot(
            plot_input, replace(spec, labels=replace(spec.labels, title=name))
        )

    return FigureViewerPresenter(
        view,
        make_host=make_host,
        # The plot's own controls belong to the plotting package; the window
        # parents the dialog, and this only says which plot and what to call it.
        edit_figure=lambda host, title: view.run_host_dialog(
            plot.edit_plot_display, host, title=f"{title} display"
        ),
    )


def create_window(*, path=None, window_ratio: float | None = None):
    """Open the viewer the way a human does, and return its window.

    The non-blocking public entry: what the launcher runs, what a notebook
    calls, and what zlc_ui's acceptance capture opens.
    """

    from zlc_ui import open_figure_viewer

    # One call, one handle: this layer never names a widget class.
    window = open_figure_viewer(
        title="FigureViewer@Zou lab", window_ratio=window_ratio
    )
    window.presenter = build(window)
    window.closed.connect(window.presenter.close)
    if path is not None:
        window.presenter.open(str(path))
    return window


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    from zlc_ui import ensure_qt_app, open_figure_viewer

    application = ensure_qt_app([])

    if arguments.check:
        # The same composition, through the same entry: a smoke test, not
        # acceptance.  It used to build the body class directly, which is the
        # one shape this layer must not know.
        presenter = build(open_figure_viewer(window_ratio=0.4))
        try:
            if arguments.path is None:
                print("figure viewer ready: no archive given")
                return 0
            description = presenter.open(str(arguments.path))
            if description is None:
                print(f"error: could not read {arguments.path}", file=sys.stderr)
                return 2
            print(
                f"figure ready: {description.name!r}, "
                f"{len(description.datasets)} dataset(s), "
                f"{sum(len(rows) for _title, rows in description.tabs)} record row(s)"
            )
            return 0
        finally:
            presenter.close()

    window = create_window(path=arguments.path)
    if arguments.path is not None and window.presenter.description is None:
        print(f"error: could not read {arguments.path}", file=sys.stderr)
        return 2
    del window
    return int(application.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
