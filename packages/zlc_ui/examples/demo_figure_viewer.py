"""Human acceptance demo for the pure figure-viewer shell."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The facade and nothing else: this file is the tutorial for the surface an
# outside host may use, so it is written under the rule that surface enforces.
from zlc_ui import ensure_qt_app, open_figure_viewer  # noqa: E402


def populate(viewer) -> None:
    def remember(name: str):
        def handler(*payload) -> None:
            text = f"{name}{payload!r}"
            print(text, flush=True)
        return handler

    viewer.path_committed.connect(remember("path_committed"))
    viewer.dataset_picked.connect(remember("dataset_picked"))
    viewer.save_image_requested.connect(remember("save_image_requested"))
    viewer.close_requested.connect(remember("close_requested"))
    # What a host puts on the page: an archive's datasets and its info tabs.
    viewer.set_title("FigureViewer - run.npz")
    viewer.set_path("D:/data/2026_08_05/run.npz")
    viewer.set_datasets((("panel-1", "camera · frame"), ("panel-2", "fit · centre")), "panel-1")
    viewer.set_info((("Measurement", (("shots", 32), ("pulse", "calibration"))),))
    viewer.set_status("fake archive: nothing was read from disk")
    print("filled: 2 datasets, 1 info tab, listening on 4 signals", flush=True)


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
