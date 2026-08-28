"""The third layer: the whole console, not just a plot.

``run_session`` times ``PlotSession.update_data``; ``run_host`` adds the
raster worker, the front promotion and the Qt widget.  Neither of them
carries what the operator actually runs: a Runtime signal plane, a logic
node producing on its own thread, the presenter's beat, panel cards, the
selection bridge and every other panel in the window.  Numbers taken at the
lower layers are floors, not forecasts -- this layer is what the console
costs.

Everything is composed through the product's own entry points
(``ExperimentSession.open``, ``build_console``, the view's signals, real
QMouseEvents on the card's widget), so product logic changes carry this
bench with them instead of leaving it measuring a fiction.

Run:  python -m bench.plot_perf.run_console [--kind image] [--size 2x2]
                                            [--seconds 8] [--gesture]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile
import time

# The operator's console runs on the real display.  Measured offscreen it is
# 826x609 at device pixel ratio 1 -- one ninth of the pixels -- so this layer
# clears any inherited offscreen choice before Qt is touched.
os.environ.pop("QT_QPA_PLATFORM", None)

from .common import (
    SIZE_PRESET,
    Pointer,
    Presented,
    ROOT,
    pump,
    pump_until,
    stats,
    write_result,
)
from . import guards, probe


def _console_paths() -> None:
    """Make the checkout authoritative, then expose the pulse fixtures.

    ``zou_lab_control`` is the product's own bootstrap: it puts this
    checkout's packages ahead of any installed copy.  Skipping it is how a
    bench ends up measuring the stale standalone package directories that
    pip's editable installs still point at.
    """

    import zou_lab_control  # noqa: F401

    tests = str(ROOT / "packages" / "zlc_workbench" / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)


#: Seams worth timing on a panel's renderer.  Absent ones are skipped, so
#: one list covers every kind and a new kind only has to add its own.
RENDERER_SEAMS = (
    "present",
    "_compose_frame",
    "_native_draw",
    "_update_image_artist",
    "_image_rgba_front",
    "_view_filling_rgba_front",
    "_update_image_chrome",
    "_mutate_image_artists",
    "_update_selectors",
    "_cached_image_range",
    "_blit_exact_rgba_image",
    "_dynamic_artists",
    "_box_sized_rgba_front",
    "_update_height_bars_artist",
    "_update_height_bars_chrome",
    "_thin_overlapping_chrome",
    "_height_bars_occluded_polyline",
    "_update_curve",
    "_update_histogram",
    "_update_horizontal_histogram",
    "_update_fit",
    "_settle_owned_boxes",
    "_resolve_image_limits",
)


class ConsoleBench:
    """One console, one panel under test, measured the way it is run."""

    def __init__(self, *, allow_low_density: bool = False):
        _console_paths()
        from zlc_ui.qt import ensure_qt_app

        self.app = ensure_qt_app(["zlc-console-bench"])
        self.allow_low_density = allow_low_density
        self.report: dict = {}
        self._tmp = pathlib.Path(tempfile.mkdtemp(prefix="console-bench-"))

    # ------------------------------------------------------------ lifecycle
    def start(self, *, camera: str = "mot_camera", exposure: float = 0.01):
        from pulse_fixtures import PULSE_NAME, write_ordinary_pulse
        from zlc_workbench.apps.task_console import build_console
        from zlc_workbench.logic import stable_signal_key
        from zlc_workbench.session import ExperimentSession

        self.session = ExperimentSession.open(self._tmp, template="virtual")
        write_ordinary_pulse(self._tmp)
        self.view, self.presenter = build_console(self.session)
        self.session.load_pulse(PULSE_NAME)

        self.node = self.presenter.add_logic("camera_measurement")
        self.view.logic_draft_changed.emit(
            self.node,
            {
                "device_keys": {"camera": camera},
                "values": {
                    "exposure_seconds": exposure,
                    "repeat": 0,
                    "frames_per_cycle": 1,
                },
            },
        )
        self.presenter.start_logic(self.node)
        self._until(
            lambda: self.session.installation.device(camera).capture_state(),
            "camera armed",
        )
        self.signal = stable_signal_key(self.node, "frames")
        self.session.fire(shots=1)
        self._until(
            lambda: self.session.signal_plane.freeze().publication(self.signal)
            is not None,
            "first publication",
        )
        window = self.view._window if self.view._window is not None else self.view._view
        # A REAL window -- the console's own, with its real device pixel
        # ratio, which is the whole point of this layer.  Sized rather than
        # maximised so the card geometry is the same on every run: maximised
        # depends on the screen it lands on, and two runs then measure two
        # different pictures.
        window.resize(1600, 1000)
        window.show()
        self._pump(1.2)
        self.view._view.tabs.setCurrentIndex(0)
        self.presenter.set_deriving(True)
        self.report["signal"] = self.signal
        return self

    def _pump(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.presenter.beat()
            self.app.processEvents()
            time.sleep(0.002)

    def _until(self, predicate, what: str, timeout: float = 60.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.presenter.beat()
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.004)
        raise guards.HarnessError(f"timed out waiting for {what}")

    # --------------------------------------------------------------- panels
    def add_panel(self, kind: str, *, size: str = SIZE_PRESET, display=None):
        """Add one panel through the view's own signals, and settle it."""

        semantic = "image" if kind == "height_bars" else kind
        panel = self.presenter.add_selected_panel(semantic)
        self.view.panel_state_changed.emit(
            panel.panel_id, {"signal": self.signal, "size": size}
        )
        self._until(lambda: panel.host is not None, f"{kind} host")
        if kind == "height_bars":
            self.view.panel_state_changed.emit(
                panel.panel_id, {"display": {"presentation": "height_bars"}}
            )
        elif display:
            self.view.panel_state_changed.emit(panel.panel_id, {"display": display})
        self._pump(3.0)
        self._until(
            lambda: self.surface(panel) is not None, f"{kind} card surface"
        )
        return panel

    def surface(self, panel):
        card = self.view._cards.get(panel.panel_id)
        return None if card is None else card.surface

    def renderer(self, panel):
        return panel.host._session._renderer

    # ---------------------------------------------------------- measurement
    def instrument(self, panel, *, seams=RENDERER_SEAMS) -> list[str]:
        """Bind self-time probes to THIS panel's renderer only."""

        return probe.watch(self.renderer(panel), *seams)

    def density(self, panel) -> dict:
        renderer = self.renderer(panel)
        if self.allow_low_density:
            return guards.display_density(renderer)
        return guards.require_real_density(renderer)

    def live(self, panel, seconds: float = 8.0) -> dict:
        """Time live frames with the producer and the beat as the product runs them."""

        presented = Presented(self.surface(panel))
        guards.free_running(self.session)
        self._pump(1.0)
        probe.reset()
        presented.count = 0
        presented.stamps.clear()
        with guards.ProductBeat(self.app, self.presenter) as beat:
            elapsed = beat.run(seconds, tick=presented.poll)
        gaps = [
            b - a for a, b in zip(presented.stamps, presented.stamps[1:])
        ]
        return {
            "window_s": round(elapsed, 2),
            "frames": presented.count,
            "frames_per_second": round(presented.count / elapsed, 1),
            "frame_gap": stats(gaps),
            "seams": probe.rows(elapsed),
        }

    def gesture(self, panel, *, kind: str, moves: int = 8, trials: int = 6) -> dict:
        """Press-move-release with REAL Qt events, and prove it landed."""

        from PyQt5 import QtCore, QtGui

        widget = self.surface(panel)
        pointer = Pointer(widget, self.app, QtCore, QtGui)
        presented = Presented(widget)
        orbit = kind == "height_bars"
        button = QtCore.Qt.MiddleButton if orbit else QtCore.Qt.LeftButton

        def owned():
            if orbit:
                camera = self.renderer(panel).height_bars_camera
                return None if camera is None else (
                    round(camera.azimuth_deg, 3),
                    round(camera.elevation_deg, 3),
                )
            return guards.committed_region(panel)

        before = owned()
        latencies: list[float] = []
        for _ in range(trials):
            pointer.press(0.35, 0.40, button=button)
            pump(self.app, 0.05)
            for step in range(1, moves + 1):
                sent = time.perf_counter()
                baseline = presented.count
                pointer.move(0.35 + 0.05 * step, 0.40 + 0.03 * step)
                got = pump_until(
                    self.app,
                    lambda: (presented.poll() or presented.count > baseline),
                    1.0,
                )
                if got and presented.stamps:
                    latencies.append(presented.stamps[-1] - sent)
            pointer.release(0.35 + 0.05 * moves, 0.40 + 0.03 * moves, button=button)
            pump(self.app, 0.15)
        guards.require_effect(
            before, owned(), "the camera" if orbit else "the committed region"
        )
        return {
            "gesture": "orbit" if orbit else "area",
            "moves": moves * trials,
            "answered": len(latencies),
            "hand_to_picture": stats(latencies),
        }

    def close(self) -> None:
        try:
            self.session.sequencer.safe()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="image")
    parser.add_argument("--size", default=SIZE_PRESET)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--gesture", action="store_true")
    parser.add_argument("--allow-low-density", action="store_true")
    parser.add_argument("--top", type=int, default=18)
    args = parser.parse_args()

    bench = ConsoleBench(allow_low_density=args.allow_low_density).start()
    panel = bench.add_panel(args.kind, size=args.size)
    guards.require_panels(bench.presenter, len(bench.presenter.panels))
    payload = {
        "kind": args.kind,
        "size": args.size,
        "panels_in_console": len(bench.presenter.panels),
        "density": bench.density(panel),
    }
    bench.instrument(panel)
    payload["live"] = bench.live(panel, args.seconds)
    if args.gesture:
        payload["gesture"] = bench.gesture(panel, kind=args.kind)
    bench.close()

    print("kind=%s size=%s  %s px  DPR %s  %s panels in the console" % (
        args.kind,
        args.size,
        payload["density"]["figure_px"],
        payload["density"]["device_pixel_ratio"],
        payload["panels_in_console"],
    ))
    live = payload["live"]
    print("live: %d frames in %.1f s = %.1f/s, gap %s" % (
        live["frames"], live["window_s"], live["frames_per_second"],
        live["frame_gap"],
    ))
    if "gesture" in payload:
        print("gesture: %s" % (payload["gesture"],))
    print()
    print(probe.report(live["window_s"], top=args.top))
    path = write_result(payload, f"console-{args.kind}-{args.size}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
