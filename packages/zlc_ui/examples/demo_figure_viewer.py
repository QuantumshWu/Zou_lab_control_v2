"""Human acceptance demo for the pure figure-viewer shell."""

from __future__ import annotations

import argparse
import os

# The facade and nothing else: this file is the tutorial for the surface an
# outside host may use, so it is written under the rule that surface enforces.
from zlc_ui import ensure_qt_app, open_figure_viewer


def populate(viewer) -> None:
    def remember(name: str):
        def handler(*payload) -> None:
            text = f"{name}{payload!r}"
            print(text, flush=True)
        return handler

    viewer.path_committed.connect(remember("path_committed"))
    viewer.add_panel_requested.connect(remember("add_panel_requested"))
    viewer.panel_state_changed.connect(remember("panel_state_changed"))
    viewer.save_image_requested.connect(remember("save_image_requested"))
    viewer.close_requested.connect(remember("close_requested"))
    # What a host puts on the page: an archive's datasets and its info tabs.
    viewer.set_title("FigureViewer - run.npz")
    viewer.set_path("D:/data/2026_08_05/run.npz")
    viewer.set_panel_sizes(("2x2", "4x4"), "2x2")
    viewer.set_grid_cell_kinds(("curve", "image", "histogram"))
    viewer.set_panel_kinds((("image", "Image"), ("curve", "Curve")))
    viewer.add_panel("saved-panel-1", "camera · frame")
    viewer.set_panel_datasets(
        "saved-panel-1",
        (("panel-1", "camera · frame"), ("panel-2", "fit · centre")),
        "panel-1",
    )
    viewer.set_panel_projection(
        "saved-panel-1",
        {
            "signal": "panel-1", "kind": "image", "cell_kind": "",
            "size": "2x2", "interval_ms": 400, "title": "camera · frame",
            "semantic": {}, "display": {}, "fit": {}, "overlay_signal": "",
        },
        {"semantic": (), "display": (), "fit": (), "data_structure": (), "data_scope": ()},
    )
    viewer.set_info((("Measurement", (("shots", 32), ("pulse", "calibration"))),))
    viewer.set_status("fake archive: nothing was read from disk")
    print("filled: 2 datasets, 1 info tab, listening on 5 signals", flush=True)


def create_window(
    argv: list[str] | None = None,
    *,
    window_ratio: float | None = None,
):
    """Open the viewer the way an outside host does, and fill it with fakes."""

    ensure_qt_app(["zlc-ui-figure-demo", *(argv or [])])
    viewer = open_figure_viewer(title="FigureViewer@Zou lab", window_ratio=window_ratio)
    populate(viewer)
    return viewer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    app = ensure_qt_app(["zlc-ui-figure-demo", *(argv or [])])
    create_window(argv)
    app.processEvents()
    if args.once or os.environ.get("ZLC_UI_FIGURE_ONESHOT") == "1" or app.platformName().strip().lower() == "offscreen":
        app.processEvents()
        return 0
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
