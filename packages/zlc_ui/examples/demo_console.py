"""End-to-end fake-presenter demo for the pure console views."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PyQt5 import QtCore, QtWidgets  # noqa: E402

if __package__:
    from .synthetic_card import SyntheticCardView  # noqa: E402
else:
    from synthetic_card import SyntheticCardView  # noqa: E402

# zlc_ui itself: the facade and nothing else.  This file is the tutorial for
# the surface an outside host may use, so it is written under that rule.
from zlc_ui import ensure_qt_app, open_task_console  # noqa: E402


class _FakeHost:
    """What a drawing package hands over: a host that owns its own widget."""

    def __init__(self, widget) -> None:
        self._widget = widget

    def qt_widget(self):
        return self._widget


class FakePresenter:
    """Answer every intent the handle raises, and echo it.

    A host connects ONCE, at the port: a card's controls arrive already named
    by the panel they belong to.  They used to be re-wired per card by whoever
    built the widget, which is how a card could look configurable and not be.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def log(name: str, *payload: object) -> None:
        print(f"{name}{payload!r}", flush=True)

    def wire(self, console) -> None:
        for name in (
            "add_panel_requested", "add_logic_requested", "pause_toggled",
            "selectors_toggled", "save_layout_requested",
            "load_layout_requested", "save_screenshot_requested",
            "panel_order_committed",
            "panel_remove_requested",
            "panel_edit_requested", "logic_start_requested",
            "logic_stop_requested", "logic_edit_requested",
            "logic_remove_requested", "logic_draft_changed",
            "panel_state_changed", "panel_snapshot_refresh_requested",
            "panel_producer_restart_requested", "panel_save_figure_requested",
        ):
            callback = lambda *payload, label=name: print(
                f"{label}{payload!r}", flush=True
            )
            getattr(console, name).connect(callback)


def populate(console, *, wire_intents: bool = True) -> None:
    from zlc_ui.board import panel_display_size

    demo_sizes = ("1x4", "2x2", "4x2")
    console.set_panel_sizes(demo_sizes, "2x2")
    console.set_summary("3 fake cards · mixed 1x4 / 2x2 / 4x2 · 2 fake logic rows")
    console.show_status("ready", "idle")

    for index in range(1, 4):
        panel_id = f"panel-{index}"
        console.add_panel(panel_id, f"Fake card {index}")
        console.set_panel_projection(
            panel_id,
            {
                "signal": "",
                "kind": "image",
                "size": demo_sizes[index - 1],
                "interval_ms": 100,
                "title": f"Fake card {index}",
                "semantic": {},
                "display": {},
                "fit": {},
                "overlay_signal": "",
            },
            {
                "semantic": (),
                "display": (),
                "fit": (),
            },
        )
        console.set_panel_signal_choices(
            panel_id,
            (("Fake source", (("Temperature", "temperature"), ("Count", "count"))),),
        )
        # The host, not its widget: this side never holds one.  The widget
        # inside is a third party's -- the extension proof: anything that is a
        # QWidget can be what a host draws with.
        # A card is as big as what is mounted in it, so the stand-in is
        # sized like the plot it stands in for -- otherwise every demo card
        # comes out the same size and the mixed-size board shows nothing.
        synthetic = SyntheticCardView(f"Synthetic surface {index}")
        synthetic.setFixedSize(*panel_display_size(demo_sizes[index - 1]))
        console.show_panel(panel_id, _FakeHost(synthetic))
        console.set_panel_status(panel_id, "waiting for fake data", error=False)

    console.add_logic_row("measurement-1", "measurement")
    console.set_logic_state("measurement-1", "running", "fake stream running")
    console.set_logic_publishes(
        "measurement-1", (("temperature", "1", "fake temperature"),)
    )
    console.add_logic_row("processor-1", "processor")
    console.set_logic_state("processor-1", "idle", "waiting")
    console.set_logic_publishes(
        "processor-1", (("smoothed", "1", "fake derived value"),)
    )

    if wire_intents:
        presenter = FakePresenter()
        presenter.wire(console)
        console._demo_presenter = presenter
    print("console_demo_ready", flush=True)


def create_window(
    argv: list[str] | None = None,
    *,
    window_ratio: float | None = None,
):
    """Open the console the way an outside host does, and fill it with fakes."""

    ensure_qt_app(["zlc-ui-demo", *(argv or [])])
    console = open_task_console(title="TaskConsole@Zou lab", window_ratio=window_ratio)
    populate(console)
    return console


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    app = ensure_qt_app(["zlc-ui-demo", *(argv or [])])
    window = create_window(argv)
    app.processEvents()
    if args.once or os.environ.get("ZLC_UI_DEMO_ONESHOT") == "1" or app.platformName().strip().lower() == "offscreen":
        # The one-shot path has no app.exec_() shutdown cycle.  Close the
        # frameless native shell explicitly so Windows Qt does not tear it
        # down during interpreter finalization after the signal smoke test.
        window.close()
        app.processEvents()
        return 0
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
