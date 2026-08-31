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
    axis_by_role,
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
# The seams a frame is made of, DERIVED from the renderer rather than typed
# out here.  A hand-kept list goes blind exactly when it matters: it had no
# _update_rolling, so a rolling panel's 31.6 ms per frame sat in
# _compose_frame's self-time with no child to blame, and a plot kind added
# tomorrow would be invisible the same way.  Every ``_update_*`` the product
# defines is a seam by construction; these are the ones whose names do not
# follow that shape.
_COMPOSE_SEAMS = (
    "present",
    "_compose_frame",
    "_native_draw",
    "_image_rgba_front",
    "_view_filling_rgba_front",
    "_mutate_image_artists",
    "_cached_image_range",
    "_blit_exact_rgba_image",
    "_dynamic_artists",
    "_raster_facet_curve_command",
    "_raster_prepared_error_bars",
    "_raster_curve_lines",
    "_raster_prepared_images",
    "_raster_facet_fit_annotations",
    "_thin_overlapping_chrome",
    "_height_bars_occluded_polyline",
    "_settle_owned_boxes",
    "_resolve_image_limits",
)


def renderer_seams(renderer_type=None) -> tuple[str, ...]:
    """Every timeable seam on the renderer, product-derived."""

    if renderer_type is None:
        from zlc_plot.rendering import MatplotlibRenderer as renderer_type
    updates = tuple(
        sorted(
            name
            for name in vars(renderer_type)
            if name.startswith("_update_")
            and callable(vars(renderer_type)[name])
        )
    )
    missing = tuple(
        name for name in _COMPOSE_SEAMS if not hasattr(renderer_type, name)
    )
    if missing:
        raise HarnessSeamError(
            "the bench names seams the renderer no longer has: %s. A probe "
            "that binds nothing reports zero and reads like free work."
            % ", ".join(missing)
        )
    return _COMPOSE_SEAMS + updates


def every_gap_ms(stamps, origin: float) -> dict:
    """EVERY interval between presented frames, in order, since the panel opened.

    Not a summary and not a bucket average.  The producer runs at 10 Hz, so
    the colour-limit bar -- which is redrawn on every frame -- should move
    every 100 ms for as long as the console is open.  What an operator
    reports is that it starts out doing exactly that and then slows down,
    and neither a single median over the whole run nor a five-second bucket
    average can show that: both are means over the very interval where the
    change happens.  A trend is only visible if nothing is averaged.
    """

    if not stamps:
        return {"first_frame_ms": None, "gaps_ms": []}
    return {
        "first_frame_ms": round(1e3 * (stamps[0] - origin), 1),
        "gaps_ms": [
            round(1e3 * (b - a), 1) for a, b in zip(stamps, stamps[1:])
        ],
    }


class HarnessSeamError(RuntimeError):
    """The bench's idea of the renderer no longer matches the renderer."""


class _PanelFronts:
    """Count presented fronts by RESOLVING the surface every time.

    A card's surface widget is replaced when the panel is remounted, and a
    captured reference then reports the same front for ever: four panels
    measured that way each said "1 frame in 10 s" while the console was
    presenting fifty-six.  The panel is the identity that survives; the
    widget is not.
    """

    def __init__(self, bench, panel):
        self._bench = bench
        self._panel = panel
        self.count = 0
        self.stamps: list[float] = []
        self._last = None

    def poll(self) -> bool:
        widget = self._bench.surface(self._panel)
        front = None if widget is None else getattr(widget, "presented_front", None)
        if front is not None and front is not self._last:
            self._last = front
            self.count += 1
            self.stamps.append(time.perf_counter())
            return True
        return False

    def reset(self) -> None:
        # Baseline the picture that is ALREADY on screen.  Clearing the count
        # while leaving ``_last`` as None counts that old front on the first
        # poll of the measurement window, inflating every run by one frame.
        widget = self._bench.surface(self._panel)
        self._last = (
            None if widget is None else getattr(widget, "presented_front", None)
        )
        self.count = 0
        self.stamps.clear()


class ConsoleBench:
    """One console, one panel under test, measured the way it is run."""

    def __init__(self, *, allow_low_density: bool = False):
        _console_paths()
        from zlc_ui.qt import ensure_qt_app

        self.app = ensure_qt_app(["zlc-console-bench"])
        self.allow_low_density = allow_low_density
        self.report: dict = {}
        # The kind the SCENARIO asked for, per panel.  A height-bar panel and
        # an image panel both read back "image" from the product, so reading
        # the kind off the panel loses what was declared.
        self._kinds: dict[str, str] = {}
        self._labels: dict[str, str] = {}
        self._tmp = pathlib.Path(tempfile.mkdtemp(prefix="console-bench-"))

    # ------------------------------------------------------------ lifecycle
    def start(
        self,
        *,
        camera: str = "mot_camera",
        exposure: float = 0.01,
        clear_preview_panels: bool = True,
    ):
        from pulse_fixtures import PULSE_NAME, write_ordinary_pulse
        from zlc_workbench.apps.task_console import build_console
        from zlc_workbench.logic import stable_signal_key
        from zlc_workbench.session import ExperimentSession

        self.session = ExperimentSession.open(self._tmp, template="virtual")
        write_ordinary_pulse(self._tmp)
        self.view, self.presenter = build_console(self.session)
        self.reports: list[tuple[str, str]] = []
        original_report = self.presenter._report

        def capture(message, severity="info", **kwargs):
            # Everything the console would put in front of the operator.  A
            # bench that drops these reports a BROKEN panel as a slow one:
            # a panel erroring every frame publishes almost nothing, and the
            # number that comes back looks like latency.
            self.reports.append((str(severity), str(message)))
            return original_report(message, severity=severity, **kwargs)

        self.presenter._report = capture
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
        # ratio, which is the whole point of this layer.
        #
        # AND THE PRODUCT'S OWN SIZE.  This pinned 1600x1000 "so the card
        # geometry is the same on every run", which meant every measurement
        # was taken at a size the operator never sees.  Card size decides
        # the Setting frame's height cap, the square field's box and how
        # much of a frame is dynamic, so that is not a detail -- a whole
        # class of behaviour was being measured in a regime that does not
        # occur.  Two runs on one machine are comparable anyway: it is the
        # same product opening on the same screen.
        window.show()
        self._pump(1.2)
        self.view._view.tabs.setCurrentIndex(0)
        self.presenter.set_deriving(True)
        self.report["signal"] = self.signal
        if clear_preview_panels:
            self.report["preview_panels_closed"] = self._clear_panels()
        return self

    def _clear_panels(self) -> tuple[str, ...]:
        """Start from zero panels, so the scenario is DECLARED not inherited.

        A logic node declares preview panels and the console opens them --
        correct for an operator, and an uncontrolled second live panel for a
        benchmark.  Leaving it there made every "one panel" measurement a
        two-panel measurement: the seam table came back at 29-59 per cent
        cpu because the thread was competing with a panel nobody asked for.
        Detecting it was not enough; require_panels only made it visible.
        """

        closed = []
        for panel_id in tuple(self.presenter.panels):
            try:
                self.presenter.remove_panel(panel_id)
                closed.append(panel_id)
            except Exception:
                continue
        if closed:
            self._pump(1.5)
        return tuple(closed)

    def _pump(self, seconds: float) -> None:
        """Let the console run, AT THE RATE THE PRODUCT RUNS IT.

        These two used to call ``presenter.beat()`` in a tight loop, once
        every two milliseconds -- three to five hundred hertz against the
        board's own hundred-millisecond timer.  ProductBeat exists because
        of exactly that mistake and its docstring says so; it was fixed
        where frames were counted and left standing everywhere else, which
        is most of a run.

        What it produced was a console that, before acquisition started,
        published at the full source rate: measured, 23.3 revisions a
        second reaching the screen 23.3 times a second, against 9.2 once
        the beat took over.  An operator watching the real product never
        sees that burst, because the real product never beats that fast.
        The bench was showing its own hand and calling it startup.
        """

        with guards.ProductBeat(self.app, self.presenter) as beat:
            beat.run(seconds)

    def report_cursor(self) -> int:
        """Mark the operator-visible report stream before one action.

        Presenter work is asynchronous: a call can accept a request and the
        worker can reject it one owner turn later.  A benchmark that remembers
        only the return value loses the actual failure and eventually calls it
        a timeout.  The cursor makes that interval explicit without clearing
        reports needed by the final run summary.
        """

        return len(self.reports)

    def reports_since(self, cursor: int) -> tuple[tuple[str, str], ...]:
        """Every status emitted after ``cursor``, in product order."""

        selected = int(cursor)
        if selected < 0 or selected > len(self.reports):
            raise ValueError("report cursor is outside this benchmark run")
        return tuple(self.reports[selected:])

    def errors_since(self, cursor: int = 0) -> tuple[str, ...]:
        """Every operator-visible error after one report cursor."""

        return tuple(
            message
            for severity, message in self.reports_since(cursor)
            if severity == "error"
        )

    def require_no_errors(self, cursor: int, what: str) -> None:
        """Fail one measured interval with the exact product errors it saw."""

        errors = self.errors_since(cursor)
        if errors:
            raise guards.HarnessError(
                f"{what} failed: " + " | ".join(errors)
            )

    def _until(
        self,
        predicate,
        what: str,
        timeout: float = 60.0,
        *,
        report_cursor: int | None = None,
        allow_errors: bool = False,
    ) -> None:
        """Wait for one product condition, with its error channel attached.

        A new presenter ``error`` is a terminal answer to the action being
        awaited, not background text to ignore until the wall clock expires.
        Callers deliberately exercising a refusal can opt out; ordinary mount,
        fit, history, selector, device and save waits all fail immediately with
        the exact operator-visible sentence.
        """

        # With no explicit action cursor, ANY error in this benchmark run
        # invalidates the wait.  This also catches a synchronous presenter
        # error emitted in the narrow interval between a trigger and its
        # following ``_until`` call; starting at the call itself lost it.
        cursor = 0 if report_cursor is None else int(report_cursor)
        outcome: dict[str, object] = {}

        def answered() -> bool:
            if not allow_errors:
                errors = self.errors_since(cursor)
                if errors:
                    outcome["errors"] = errors
                    return True
            if predicate():
                outcome["success"] = True
                return True
            return False

        with guards.ProductBeat(self.app, self.presenter) as beat:
            completed = beat.run_until(answered, timeout)
        if outcome.get("success") is True:
            return
        errors = tuple(outcome.get("errors", ()))
        if errors:
            raise guards.HarnessError(
                f"{what} failed: " + " | ".join(str(error) for error in errors)
            )
        reports = self.reports_since(cursor)
        detail = (
            ""
            if not reports
            else "; reports: "
            + " | ".join(f"{severity}: {message}" for severity, message in reports)
        )
        if not completed:
            raise guards.HarnessError(f"timed out waiting for {what}{detail}")
        raise guards.HarnessError(f"{what} did not complete{detail}")

    def perform(self, what: str, trigger, complete, *, timeout: float = 60.0):
        """Submit one asynchronous action and wait for success or real error."""

        cursor = self.report_cursor()
        result = trigger()
        if result is False:
            reports = self.reports_since(cursor)
            detail = " | ".join(
                f"{severity}: {message}" for severity, message in reports
            )
            raise guards.HarnessError(
                f"{what} was refused" + (f": {detail}" if detail else "")
            )
        self._until(
            complete,
            what,
            timeout,
            report_cursor=cursor,
        )
        return result

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
        self._name(panel, kind)
        return panel

    def _name(self, panel, kind: str) -> str:
        """The name this panel reports under.  ONE PANEL, ONE NAME.

        The probe prefix, the live rows and the render-cost roll-up all
        join on this string.  It used to be the declared kind alone, so a
        chain -- two panels of the same kind, which is exactly what the
        documented ``--chain image`` builds with the default ``--kind
        image`` -- merged both renderers' self-times into one probe key,
        emitted two rows called "image", and then divided the summed wall
        time by whichever frame count survived the dict.  About double, and
        the one thing a chain exists to separate, gone.
        """

        self._kinds[panel.panel_id] = kind
        taken = {
            name
            for panel_id, name in self._labels.items()
            if panel_id != panel.panel_id
        }
        name, ordinal = kind, 1
        while name in taken:
            ordinal += 1
            name = f"{kind}#{ordinal}"
        self._labels[panel.panel_id] = name
        return name

    def label(self, panel) -> str:
        """What this panel is called in every report this run prints."""

        return self._labels.get(panel.panel_id) or panel.panel_id

    def surface(self, panel):
        card = self.view._cards.get(panel.panel_id)
        return None if card is None else card.surface

    def renderer(self, panel):
        return panel.host._session._renderer

    # ---------------------------------------------------------- measurement
    def instrument(
        self,
        panel,
        *,
        seams=None,
        module_seams: bool = True,
    ) -> list[str]:
        """Bind self-time probes to THIS panel's renderer, and to the
        module-level work a frame does outside it.

        A renderer-only tap hides the front store: the source reduction, the
        front store and the block mean all live in ``_image_raster`` as plain
        functions, and they were a third of some frames while every visible
        row said the renderer was cheap.  Module functions have exactly one
        instance, so watching them is safe where watching a class is not.
        """

        # Four panels reporting under one class name is the class-level tap
        # again, one level up: _native_draw fired 85 times across a layout
        # and there was no way to say whose frames those were.
        label = "%s[%s]" % (type(self.renderer(panel)).__name__, self.label(panel))
        renderer = self.renderer(panel)
        if seams is None:
            seams = renderer_seams(type(renderer))
        bound = probe.watch(renderer, *seams, prefix=label)
        # The three things compose does that are NOT renderer methods: the
        # full-figure capture, the restore, and the per-artist draws.  A
        # rolling panel spent 32 ms per frame in compose's own body with
        # every named child under one millisecond, and there was no way to
        # say which of the three it was.  Capture and restore are canvas
        # methods, so the instance tap reaches them; what is left after
        # subtracting them is the artist loop.
        bound += probe.watch(
            renderer.figure.canvas,
            "copy_from_bbox",
            "restore_region",
            prefix="%s.canvas" % label,
        )
        import zlc_plot._image_raster as raster
        import zlc_plot._height3d_raster as h3d

        if module_seams and not getattr(ConsoleBench, "_module_seams_bound", False):
            for module, names in (
                (raster, ("prepare_image_front", "_area_mean", "_reduce_blocks")),
                (h3d, ("render_height_bars", "_stroke_rims")),
            ):
                for name in names:
                    try:
                        probe.watch_module(module, name)
                    except Exception:
                        continue
            ConsoleBench._module_seams_bound = True
        return bound

    def instrument_pipeline(self, panel) -> dict[str, list[str]]:
        """Bind low-overhead stage probes to one complete panel pipeline.

        These are instance taps: four panels may run the same classes, but
        every row remains attributable to the exact panel that paid it.
        Inclusive (gross) timing is retained by ``probe`` for stage latency;
        nested self timing still adds without double-counting for hotspot
        attribution.
        """

        label = self.label(panel)
        port = panel.port
        host = panel.host
        session = host._session
        widget = self.surface(panel)
        if port is None or widget is None:
            raise guards.HarnessError(f"{label} has no live port/widget to instrument")

        bound = {
            "port": probe.watch(
                port,
                "prepare",
                "accept",
                "_put_on_screen",
                prefix=f"PanelPort[{label}]",
            ),
            "port_callbacks": probe.watch_attribute(
                port,
                "_project_input",
                "_present",
                prefix=f"PanelPort[{label}]",
            ),
            "host": probe.watch(
                host,
                "update_data",
                "_enqueue_data_frame",
                "_begin_data_frame",
                "_on_frame_prepared",
                "_on_frame_solved",
                "_dispatch_frame_commit",
                "_on_frame_committed",
                prefix=f"RasterHost[{label}]",
            ),
            "session": probe.watch(
                session,
                "prepare_live_frame",
                "_prepare_live_frame_worker",
                "solve_live_frame",
                "_solve_live_pair",
                "_solve_started_fit_parts",
                "_fit_facet_batch",
                "_solve_fit_selection",
                "commit_live_frame",
                "_accept_pair_fit",
                "_present_projection_transaction",
                "_update_renderer",
                "describe_display",
                prefix=f"PlotSession[{label}]",
            ),
            "fit_engine": probe.watch(
                session._fit_engine,
                "fit",
                "fit_batch",
                prefix=f"FitEngine[{label}]",
            ),
            "qt": probe.watch(
                widget,
                "present_front",
                "_install_front",
                prefix=f"QtWidget[{label}]",
            ),
        }
        return bound

    def density(self, panel) -> dict:
        renderer = self.renderer(panel)
        if self.allow_low_density:
            return guards.display_density(renderer)
        return guards.require_real_density(renderer)

    def live(self, panel, seconds: float = 8.0) -> dict:
        """Time live frames with the producer and the beat as the product runs them."""

        presented = _PanelFronts(self, panel)
        guards.free_running(self.session)
        self._pump(1.0)
        probe.reset()
        presented.reset()
        cursor = self.report_cursor()
        origin = time.perf_counter()
        with guards.ProductBeat(self.app, self.presenter) as beat:
            elapsed = beat.run(seconds, tick=presented.poll)
        self.require_no_errors(cursor, f"{self.label(panel)} live window")
        gaps = [
            b - a for a, b in zip(presented.stamps, presented.stamps[1:])
        ]
        # A "stall" is a gap the operator can see: the console publishes on
        # its beat, so anything past two beats is a frame that did not
        # appear when the next one was due.
        beat_s = self.presenter.board.base_interval_ms / 1000.0
        stalls = [gap for gap in gaps if gap > 2 * beat_s]
        return {
            "window_s": round(elapsed, 2),
            "frames": presented.count,
            "frames_per_second": round(presented.count / elapsed, 1),
            "beat_ms": round(beat_s * 1e3, 1),
            "frame_gap": stats(gaps),
            # EVERY interval, in order.  See every_gap_ms().
            "timeline": every_gap_ms(presented.stamps, origin),
            "stalls_over_two_beats": len(stalls),
            "worst_gaps_ms": [
                round(gap * 1e3, 1) for gap in sorted(gaps, reverse=True)[:5]
            ],
            "seams": probe.rows(elapsed),
        }

    def add_panels(self, kinds, *, size: str = SIZE_PRESET) -> list:
        """Declare a whole layout at once, and assert it is what arrived."""

        panels = [self.add_panel(kind, size=size) for kind in kinds]
        guards.require_panels(self.presenter, len(panels))
        return panels

    def live_all(
        self,
        panels,
        seconds: float = 10.0,
        *,
        window_start=None,
    ) -> dict:
        """Every panel measured in ONE window, plus what the process paid.

        A panel measured alone is a floor.  Panels share one machine, one
        beat and one plane, and the thing an operator sees is what they do
        TOGETHER -- measured here rather than inferred from adding up solo
        numbers, which is how a per-panel cost gets reported as if it
        composed linearly.
        """

        # Every row this returns, and every seam the probe rolled up, is
        # keyed by a panel's name.  Two panels answering to one name merge
        # both and then divide by one of their frame counts.
        guards.require_distinct_labels(self.label(panel) for panel in panels)


        import psutil

        process = psutil.Process()
        counters = [_PanelFronts(self, panel) for panel in panels]
        guards.free_running(self.session)
        self._pump(1.0)
        probe.reset()
        for counter in counters:
            counter.reset()
        if window_start is not None:
            window_start()

        cursor = self.report_cursor()
        cpu_before = process.cpu_times()
        rss_before = process.memory_info().rss

        def tick():
            for counter in counters:
                counter.poll()

        origin = time.perf_counter()
        with guards.ProductBeat(self.app, self.presenter) as beat:
            elapsed = beat.run(seconds, tick=tick)
        self.require_no_errors(cursor, "multi-panel live window")

        cpu_after = process.cpu_times()
        used = (
            (cpu_after.user - cpu_before.user)
            + (cpu_after.system - cpu_before.system)
        )
        beat_s = self.presenter.board.base_interval_ms / 1000.0
        rows = []
        for panel, counter in zip(panels, counters):
            gaps = [
                b - a for a, b in zip(counter.stamps, counter.stamps[1:])
            ]
            rows.append(
                {
                    # The label the probe prefixes with, so the seam
                    # roll-up joins to these rows.
                    "panel": self.label(panel),
                    "frames": counter.count,
                    "frames_per_second": round(counter.count / elapsed, 1),
                    "frame_gap": stats(gaps),
                    "stalls_over_two_beats": sum(
                        1 for gap in gaps if gap > 2 * beat_s
                    ),
                    # EVERY interval, in order.  See every_gap_ms().
                    "timeline": every_gap_ms(counter.stamps, origin),
                }
            )
        return {
            "window_s": round(elapsed, 2),
            "panels": rows,
            "beat_ms": round(beat_s * 1e3, 1),
            "cpu_percent_of_one_core": round(100.0 * used / elapsed, 0),
            "rss_mb_before": round(rss_before / 2**20, 1),
            "rss_mb_after": round(process.memory_info().rss / 2**20, 1),
            "seams": probe.rows(elapsed),
        }

    def derive(self, panel, output: str = "roi_frame", *, fraction: float = 0.4):
        """Draw a region and publish one derived signal from it.

        The product's own path: a real drag on the card, then the panel's
        published-output switch.  ``require_effect`` proves the drag landed
        -- a region that never committed derives nothing, and the chain
        would then be measuring an ordinary panel while claiming to measure
        a chain.
        """

        from PyQt5 import QtCore, QtGui

        widget = self.surface(panel)
        pointer = Pointer(widget, self.app, QtCore, QtGui)
        before = guards.committed_region(panel)
        low = 0.5 - fraction / 2.0
        high = 0.5 + fraction / 2.0
        pointer.press(low, low, button=QtCore.Qt.LeftButton)
        pump(self.app, 0.05)
        pointer.move(high, high)
        pump(self.app, 0.05)
        pointer.release(high, high, button=QtCore.Qt.LeftButton)
        self._pump(1.0)
        guards.require_effect(before, guards.committed_region(panel), "the region")

        self.presenter.update_panel_published_outputs(panel.panel_id, {output: True})
        deadline = time.monotonic() + 30.0
        name = None
        while time.monotonic() < deadline:
            self._pump(0.3)
            names = [
                item
                for item in self.session.signal_plane.freeze().names()
                if item.endswith(output) and panel.panel_id in item
            ]
            if names:
                name = names[0]
                break
        if name is None:
            raise guards.HarnessError(
                "%s published no %s: the chain has no upstream to measure"
                % (panel.panel_id, output)
            )
        return name

    def add_panel_on(self, signal: str, kind: str, *, size: str = SIZE_PRESET):
        """A panel on a DERIVED signal, not on the camera."""

        semantic = "image" if kind == "height_bars" else kind
        panel = self.presenter.add_selected_panel(semantic)
        self.view.panel_state_changed.emit(
            panel.panel_id, {"signal": signal, "size": size}
        )
        self._until(lambda: panel.host is not None, f"{kind} host on {signal}")
        if kind == "height_bars":
            self.view.panel_state_changed.emit(
                panel.panel_id, {"display": {"presentation": "height_bars"}}
            )
        self._pump(3.0)
        self._until(
            lambda: self.surface(panel) is not None, f"{kind} card on {signal}"
        )
        self._name(panel, kind)
        return panel

    def edit_setting(self, panel, section: str, strict: bool = True, **values) -> dict:
        """Change a panel parameter the way the Setting form changes it.

        The form does not send the one field the operator touched: it sends
        the WHOLE section, because ``parameter_edit_values`` re-reads every
        widget.  A bench that sends one key exercises a path the product
        never takes -- and one that never checks the value took reads a
        refused edit as a product defect.

        Returns what the edit cost: the wait until the next presented front.
        """

        if section not in {"display", "semantic", "fit"}:
            raise ValueError("a panel edits display, semantic or fit")
        current = dict(getattr(panel.state, section, {}) or {})
        current.update(values)
        fronts = _PanelFronts(self, panel)
        fronts.poll()
        cursor = self.report_cursor()
        started = time.perf_counter()
        self.view.panel_state_changed.emit(panel.panel_id, {section: current})
        answered = None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            self.presenter.beat()
            self.app.processEvents()
            errors = self.errors_since(cursor)
            if errors:
                raise guards.HarnessError(
                    f"{section} edit failed: " + " | ".join(errors)
                )
            if fronts.poll():
                answered = time.perf_counter() - started
                break
            time.sleep(0.002)
        self._pump(0.5)
        applied = dict(getattr(panel.state, section, {}) or {})
        refused = {
            name: applied.get(name)
            for name, wanted in values.items()
            if applied.get(name) != wanted
        }
        if refused and strict:
            # A refused edit's timing describes the OLD picture, so a caller
            # that asked for one value and measured another is measuring
            # nothing.  A sweep over every field is the exception: the
            # product legitimately refuses some of them (a classifier with
            # no threshold to classify), and that is a fact to record, not
            # a reason to abandon the other twelve fields.
            raise guards.HarnessError(
                "%s did not take: asked %r, panel holds %r. The edit was "
                "refused, and any timing here describes the old picture."
                % (section, dict(values), refused)
            )
        return {
            "section": section,
            "values": dict(values),
            "refused": refused or None,
            "answered": answered is not None,
            "to_next_front_ms": None if answered is None else round(answered * 1e3, 2),
        }

    # Strings the bench is allowed to invent a different value for.  Every
    # other string field is an enum whose vocabulary belongs to the product,
    # and guessing a member is how a bench reports a refused edit as a
    # product defect.
    _FREE_TEXT = ("title", "x_label", "y_label", "value_label")
    _COLORMAPS = ("gray", "magma", "viridis")

    @classmethod
    def _different_value(cls, field: str, current):
        """A valid, different value for this field -- or why not.

        Derived from what the panel is holding, so a display field added
        with a new plot kind is exercised without being typed out here.
        A hand-kept menu named bin_count under the wrong section and
        invented two fields that do not exist.
        """

        if isinstance(current, bool):
            return not current, None
        if field in cls._FREE_TEXT:
            return "edited while live", None
        if field == "colormap":
            other = [name for name in cls._COLORMAPS if name != current]
            return (other[0], None) if other else (None, "no other colormap")
        if isinstance(current, int):
            return (max(2, current // 2) if current > 3 else current + 8), None
        if isinstance(current, float):
            return (current * 1.25 if current else 1.25), None
        if current is None:
            return None, "unset, so the bench cannot tell the type"
        return None, "%s is an enum owned by the product" % type(current).__name__

    def edit_run(self, panel, *, kind: str) -> dict:
        """Every display field this panel holds, edited while it is live.

        These are the changes that feel instant or do not, and none of them
        move a frame-rate number: the panel keeps its cadence either way,
        what changes is how long the operator stares at the OLD picture
        after committing the form.
        """

        held = dict(getattr(panel.state, "display", {}) or {})
        rows, skipped = [], []
        for field in sorted(held):
            value, refusal = self._different_value(field, held[field])
            if refusal is not None:
                skipped.append({"field": field, "why": refusal})
                continue
            rows.append(
                self.edit_setting(panel, "display", strict=False, **{field: value})
            )
            # Put it back before the next one.  Without this each field is
            # timed on a panel already deformed by every field before it,
            # and the panel refused a threshold classifier because an
            # earlier edit in the same sweep had made it cumulative.
            self.edit_setting(
                panel, "display", strict=False, **{field: held[field]}
            )
        # A refused edit redraws nothing and must not be averaged in with
        # the ones that did.
        answered = [
            row["to_next_front_ms"]
            for row in rows
            if row["answered"] and not row["refused"]
        ]
        return {
            "kind": kind,
            "edits": rows,
            # Never a silent cap: what was not exercised, and why.
            "skipped": skipped,
            "answered": len(answered),
            "of": len(rows),
            "worst_ms": max(answered) if answered else None,
        }

    def attribute_stalls(self, panel, seconds: float = 12.0) -> dict:
        """Who lost the frame: the producer, the beat, or the render?

        A panel publishes on its beat, so a visible stutter is one cycle that
        produced no picture.  Three clocks say which of them stopped:

        * the SOURCE's revision stamps -- was there new data to draw?
        * the BEAT's tick stamps -- did the cycle fire, and on time?
        * the front install stamps -- did pixels follow?

        Without all three the answer is a guess.  With them, a gap is
        attributable: no new revision is the producer, a late tick is the
        event loop, and a tick with data and no front is the render.
        """

        presented = _PanelFronts(self, panel)
        guards.free_running(self.session)
        self._pump(1.0)

        beats: list[float] = []
        revisions: list[tuple[float, int]] = []
        rate = guards.SourceRate(self.session, self.signal)
        original_beat = self.presenter.beat

        def timed_beat():
            beats.append(time.perf_counter())
            return original_beat()

        self.presenter.beat = timed_beat
        try:
            presented.reset()
            last_revision = None

            def tick():
                nonlocal last_revision
                presented.poll()
                revision = rate._revision()
                if revision is not None and revision != last_revision:
                    last_revision = revision
                    revisions.append((time.perf_counter(), revision))

            with guards.ProductBeat(self.app, self.presenter) as beat:
                elapsed = beat.run(seconds, tick=tick)
        finally:
            self.presenter.beat = original_beat

        beat_s = self.presenter.board.base_interval_ms / 1000.0
        gaps = [
            (a, b - a) for a, b in zip(presented.stamps, presented.stamps[1:])
        ]
        slips = []
        for start, gap in gaps:
            if gap <= 1.5 * beat_s:
                continue
            end = start + gap
            ticks = [t for t in beats if start < t <= end]
            fresh = [t for t, _r in revisions if start < t <= end]
            longest_tick_gap = max(
                (b - a for a, b in zip([start] + ticks, ticks + [end])),
                default=gap,
            )
            if not fresh:
                blame = "producer: no new revision in the gap"
            elif longest_tick_gap > 1.5 * beat_s:
                blame = "event loop: the beat itself did not fire on time"
            else:
                blame = "render: beats fired with data and no front followed"
            slips.append(
                {
                    "gap_ms": round(gap * 1e3, 1),
                    "beats_in_gap": len(ticks),
                    "new_revisions_in_gap": len(fresh),
                    "longest_tick_gap_ms": round(longest_tick_gap * 1e3, 1),
                    "blame": blame,
                }
            )
        beat_gaps = [b - a for a, b in zip(beats, beats[1:])]
        only_gaps = [gap for _start, gap in gaps]
        # How far the CONTENT moved between two presented frames.  A camera
        # producing three frames per beat means the panel shows every third
        # one, and a steady 9 fps of every-third looks smooth.  The same 9
        # fps of two-then-five does not, and nothing in a frame rate, a gap
        # distribution or a stall count can tell the two apart -- the frame
        # arrived on time either way, carrying a picture from further ahead
        # or further behind than the last one.
        strides = []
        stamps = presented.stamps
        for start, end in zip(stamps, stamps[1:]):
            strides.append(
                sum(1 for moment, _r in revisions if start < moment <= end)
            )
        # Everything live() reports, plus the attribution -- so this can BE
        # the measurement window instead of following one.  Two windows put
        # the hitch in the first and the attribution in the second, and
        # divided two windows of seam time by one window's frames.
        return {
            "window_s": round(elapsed, 2),
            "beat_ms": round(beat_s * 1e3, 1),
            "frames": presented.count,
            "frames_per_second": round(presented.count / elapsed, 1),
            "frame_gap": stats(only_gaps),
            "stalls_over_two_beats": sum(
                1 for gap in only_gaps if gap > 2 * beat_s
            ),
            "worst_gaps_ms": [
                round(gap * 1e3, 1) for gap in sorted(only_gaps, reverse=True)[:5]
            ],
            "source_revisions": len(revisions),
            "beat_ticks": len(beats),
            "beat_tick_gap": stats(beat_gaps),
            "content_stride": stats([float(value) for value in strides]),
            "content_stride_counts": {
                str(value): strides.count(value) for value in sorted(set(strides))
            },
            "slips": slips,
        }

    #: How long a hand takes between putting the button down and starting
    #: to move it.  Not a constant: the point is to land the press at every
    #: phase of the producer's cycle, because that phase decides whether it
    #: waits out a frame that is already running.  Milliseconds, walked in
    #: order so two runs see the same hand.
    _REACTION_MS = (17, 143, 61, 210, 34, 96, 178, 8, 122, 249, 45, 79)

    #: A hand goes up, down, left and right.  One direction for eight steps
    #: is one sign of one rounding, and it walks the view off the data --
    #: after which the frame is letterboxed and the compose is doing
    #: different work from the one being asked about.  Widget-normalized,
    #: small enough that the cursor stays well inside the axes.
    _WALK = (
        (0.00, -0.06), (0.05, 0.00), (0.00, 0.07), (-0.06, 0.00),
        (0.04, 0.05), (-0.05, -0.04), (0.06, -0.03), (-0.03, 0.06),
        (-0.04, -0.05), (0.06, 0.02), (0.00, -0.07), (-0.05, 0.03),
        (0.03, 0.06), (-0.06, -0.02), (0.05, -0.04), (-0.02, 0.01),
    )

    def _hand_timeline(self, widget, on_submit, on_answer):
        """Stamp every pointer submission and every answer, and pair NOTHING.

        The hand's round trip is the only quantity that can be attributed
        to the hand at all: neither "a front was presented" nor any field
        of the front's identity separates the gesture's frame from the
        camera's -- a pan advances only ``sequence``, and so does every
        producer frame (measured: 64 moves, display revision delta 0 on
        every one, while the view limits moved on every one).

        But the round trip cannot be recovered for a move.  The host
        coalesces pointer work, so a hand moving faster than the console
        answers has moves that are correctly never answered, and any
        first-in-first-out pairing slips by one on each of them and
        inflates everything after.  So this reports the two streams and
        lets the caller take only what needs no pairing.
        """

        submit = widget._submit_pointer

        def wrapped_submit(action, *args, **kwargs):
            on_submit(action, time.perf_counter())
            return submit(action, *args, **kwargs)

        def observe(result):
            on_answer(result[1], time.perf_counter())

        widget._submit_pointer = wrapped_submit
        # A SECOND RECEIVER on the signal, not a replacement for the slot.
        # Rebinding ``widget._finish_pointer`` measured nothing at all --
        # 120 events submitted, none answered -- because the connection
        # made at construction holds the ORIGINAL bound method, and an
        # instance attribute set afterwards is not what Qt calls.  Slots
        # run in connection order, so this one runs after the widget has
        # installed the answer, which is when the operator would see it.
        widget._gesture_ready.connect(observe)

        def restore():
            widget._submit_pointer = submit
            try:
                widget._gesture_ready.disconnect(observe)
            except TypeError:
                pass

        return restore

    def gesture(
        self,
        panel,
        *,
        kind: str,
        motion: str = "auto",
        moves: int = 60,
        trials: int = 12,
    ) -> dict:
        """Press-move-release with REAL Qt events, and prove it landed.

        Three quantities, not one.  A drag whose button is already down is
        smooth; what an operator complains about is the beginning, so the
        press and the first move are reported SEPARATELY from the steady
        state they are supposed to resemble.  Pooling them buries the
        twelve first moves under eighty-four later ones, where no median
        and barely a p90 can find them.
        """

        from PyQt5 import QtCore, QtGui
        from zlc_plot import NumericRange

        widget = self.surface(panel)
        pointer = Pointer(widget, self.app, QtCore, QtGui)
        # THROUGH THE QUEUE.  A synchronous sendEvent runs the widget's
        # handler on the calling thread, so the press skips every wait an
        # operator's press cannot skip.  The clock therefore starts when
        # the event is POSTED, not when the handler gets round to it.
        pointer.post = True
        if motion == "auto":
            motion = "orbit" if kind == "height_bars" else "pan"
        if motion not in {"orbit", "pan", "area", "clim"}:
            raise ValueError("gesture motion must be orbit, pan, area or clim")
        # The operator's gesture is the middle button: on height bars it
        # turns the camera, on everything else it pans the view.  The left
        # button draws an area, which is a different complaint.
        button = (
            QtCore.Qt.LeftButton
            if motion in {"area", "clim"}
            else QtCore.Qt.MiddleButton
        )

        def color_limit_front():
            front = getattr(widget, "presented_front", None)
            transform = None if front is None else axis_by_role(front, "distribution")
            limits = None if front is None else front.interaction.color_limits
            if transform is None or limits is None:
                raise guards.HarnessError("image surface has no color-limit rail")
            return transform, limits

        def color_limit_point(transform, value: float) -> list[float]:
            left, top, right, bottom = transform.bounds
            low, high = transform.y_limits
            fraction = (float(value) - high) / (low - high)
            return [
                left + 0.5 * (right - left),
                top + fraction * (bottom - top),
            ]

        def owned():
            if motion == "orbit":
                camera = self.renderer(panel).height_bars_camera
                return None if camera is None else (
                    round(camera.azimuth_deg, 3),
                    round(camera.elevation_deg, 3),
                )
            if motion == "pan":
                value = panel.host.describe_display().result().value.limits
                return (
                    float(value.x.low), float(value.x.high),
                    float(value.y.low), float(value.y.high),
                )
            if motion == "clim":
                _transform, value = color_limit_front()
                return (float(value.low), float(value.high))
            return guards.committed_region(panel)

        home = None
        if motion == "pan":
            # A HOME VIEW HAS NOWHERE TO PAN.  Fully zoomed out the frame
            # already shows everything, so the drag is clamped to a no-op
            # and the run ends with "the gesture left the view limits
            # unchanged" -- the harness correctly reporting that it
            # measured nothing.
            #
            # PAST full, not merely to it: the operator zooms in and then
            # drags, and the whole drag stays full of data.  The criterion
            # is stated and checked rather than a notch count, because the
            # home view of an image IS the data extent.
            home = owned()
            span = (home[1] - home[0], home[3] - home[2])
            notches = 0
            while notches < 60:
                now = owned()
                if (now[1] - now[0]) <= span[0] / 12.0 and (
                    now[3] - now[2]
                ) <= span[1] / 12.0:
                    break
                pointer.wheel(0.5, 0.5, -1)
                pump(self.app, 0.08)
                notches += 1
            self._pump(0.5)
        zoomed = None if home is None else owned()

        def still_full_of_data() -> bool:
            """Is the frame still covered by the data it is showing?"""

            if home is None:
                return True
            now = owned()
            return (
                now[0] >= home[0] - 1e-9
                and now[1] <= home[1] + 1e-9
                and now[2] >= home[2] - 1e-9
                and now[3] <= home[3] + 1e-9
            )

        presses: list[float] = []
        first_moves: list[float] = []
        steady_gaps: list[float] = []
        submitted_moves = [0]
        answered_moves = [0]
        trial_state = {
            "press_at": None,
            "first_move_at": None,
            "last_answer": None,
            "reaction_ms": 0.0,
        }
        # (reaction, first move) per trial.  THE decisive pair: the press
        # does a full compose the operator sees nothing for, and whether
        # that lands on the critical path depends entirely on whether the
        # hand starts moving before it finishes.  A hand that waits out
        # its own reaction time gets it for free; one that moves at once
        # queues behind it.
        by_reaction: list[tuple[float, float]] = []

        def on_submit(action, when):
            # The submission stamp is NOT the start.  It is taken inside
            # the widget's handler, which is after the queue wait -- the
            # very interval an operator experiences as the press not
            # landing.  The starts are stamped at post time below.
            if action == "move":
                submitted_moves[0] += 1

        def on_answer(action, when):
            if action == "press":
                started = trial_state["press_at"]
                if started is not None:
                    presses.append(when - started)
                    trial_state["press_at"] = None
                return
            if action != "move":
                return
            answered_moves[0] += 1
            started = trial_state["first_move_at"]
            if started is not None and trial_state["last_answer"] is None:
                # THE FIRST ANSWER AFTER THE FIRST MOVE.  Exact without
                # pairing: only one move task can be pending, so this is
                # that move's answer whatever was coalesced behind it.
                first_moves.append(when - started)
                by_reaction.append(
                    (
                        round(trial_state["reaction_ms"], 1),
                        round((when - started) * 1e3, 1),
                    )
                )
            elif trial_state["last_answer"] is not None:
                # What a hand already moving actually experiences: how long
                # between one update of the picture and the next.
                steady_gaps.append(when - trial_state["last_answer"])
            trial_state["last_answer"] = when

        before = owned()
        left_the_data = 0
        restore = self._hand_timeline(widget, on_submit, on_answer)
        try:
            # THE CONSOLE KEEPS RUNNING WHILE THE HAND DOES.  Without this
            # the beat never ticks for the length of the gesture, so no live
            # frame is ever started and the press competes with nothing --
            # which is the whole mechanism under investigation.
            with guards.ProductBeat(self.app, self.presenter):
                for trial in range(trials):
                    if zoomed is not None:
                        # BACK TO WHERE THE ZOOM PUT IT.  A pan commits, so
                        # without this each trial starts further from the
                        # centre than the last and the later ones walk the
                        # view past the data edge -- measured, five of
                        # twelve did, and a letterboxed frame is different
                        # work from the one being asked about.
                        panel.host.set_viewport(
                            NumericRange(zoomed[0], zoomed[1]),
                            NumericRange(zoomed[2], zoomed[3]),
                        ).result()
                        self._pump(0.25)
                    trial_state["first_move_at"] = None
                    trial_state["last_answer"] = None
                    trial_state["reaction_ms"] = float(
                        self._REACTION_MS[trial % len(self._REACTION_MS)]
                    )
                    # NOT pumped to quiescence first.  A press that only
                    # ever arrives on an idle machine is a press that never
                    # waits for anything.
                    if motion == "clim":
                        color_transform, color_limits = color_limit_front()
                        at = color_limit_point(color_transform, color_limits.high)
                    else:
                        color_transform = color_limits = None
                        at = [0.5, 0.5]
                    trial_state["press_at"] = time.perf_counter()
                    pointer.press(at[0], at[1], button=button)
                    pump(
                        self.app,
                        self._REACTION_MS[trial % len(self._REACTION_MS)] / 1e3,
                    )
                    for step in range(moves):
                        if motion == "clim":
                            assert color_transform is not None
                            assert color_limits is not None
                            fraction = 0.01 + 0.002 * (step % 10)
                            at = color_limit_point(
                                color_transform,
                                color_limits.high - color_limits.span * fraction,
                            )
                        else:
                            dx, dy = self._WALK[step % len(self._WALK)]
                            at[0] += dx
                            at[1] += dy
                        if trial_state["first_move_at"] is None:
                            trial_state["first_move_at"] = time.perf_counter()
                        pointer.move(at[0], at[1])
                        # A hand moves at 60-125 Hz.
                        pump(self.app, 0.012)
                    pointer.release(at[0], at[1], button=button)
                    pump(self.app, 0.15)
                    # ONCE PER TRIAL, not once per move.  `owned()` is a
                    # blocking CONTROL round trip to the worker; calling it
                    # between moves serializes the very queue being timed,
                    # so the probe would have been the load.
                    if not still_full_of_data():
                        left_the_data += 1
        finally:
            restore()
        guards.require_effect(
            before,
            owned(),
            {
                "orbit": "the camera",
                "pan": "the view limits",
                "area": "the committed region",
                "clim": "the color limits",
            }[motion],
        )
        first = stats(first_moves)
        later = stats(steady_gaps)
        return {
            "gesture": motion,
            "trials": trials,
            "panels_in_console": len(self.presenter.panels),
            # What the hand waits for before the gesture has caught.
            "press": stats(presses),
            "first_move": first,
            # What a hand already moving experiences: the interval between
            # one update of the picture and the next.
            "steady_gap": later,
            "start_penalty": (
                None
                if not first_moves or not steady_gaps
                else round(first["median_ms"] / later["median_ms"], 2)
            ),
            # Moves the host coalesced away.  Correct behaviour, not a
            # fault: the newest supersedes them.  Reported because the
            # ratio says how much faster the hand moved than the console
            # could answer.
            "first_move_by_reaction_ms": sorted(by_reaction),
            "moves_submitted": submitted_moves[0],
            "moves_answered": answered_moves[0],
            # Non-zero means a walk wandered off the data and part of
            # this distribution is about a letterboxed frame.
            "trials_that_left_the_data": left_the_data,
        }

    # ------------------------------------------------------------- teardown
    def __enter__(self) -> "ConsoleBench":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        """Shut the console down the way the product does, and finish.

        A bench that only stops the pulse leaves everything else standing:
        the panels' raster workers, the logic node's thread, the save
        worker ``build_console`` attaches, the window, and the session's
        device claims.  None of those are daemon threads, so the process
        does not exit and the console stays on screen -- which is exactly
        what happened until this method existed, and it had to be killed
        from the task list.

        The sequence is the product's own, the same one the console guard
        test uses: quiet the hardware, release the claims, drive the
        presenter's asynchronous close to completion, close the view, then
        close the session.  ``presenter.close`` deliberately never blocks
        on the Qt owner, so it is PUMPED to completion rather than awaited.
        """

        presenter = getattr(self, "presenter", None)
        if presenter is None:
            return
        session = getattr(self, "session", None)

        # 1. Quiet the hardware and release its claims, or session.close()
        #    refuses on device_use.assert_idle().
        if session is not None:
            try:
                session.sequencer.safe()
            except Exception:
                pass
        node = getattr(self, "node", None)
        if node is not None:
            try:
                presenter.stop_logic(node)
            except Exception:
                pass

        # 2. The presenter's close is a guard, not a call: first ask begins
        #    the shutdown, and it reports ready only once it has finished.
        try:
            presenter.close()
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                presenter.beat()
                self.app.processEvents()
                if presenter.close():
                    break
                time.sleep(0.005)
        except Exception:
            pass

        # 3. The window, past its own close guard -- the console asks the
        #    operator before closing, and a bench is not an operator.
        view = getattr(self, "view", None)
        if view is not None:
            try:
                view.set_close_guard(lambda: True)
                view.close()
                self.app.processEvents()
            except Exception:
                pass

        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        self.presenter = None

    def problems(self) -> tuple[tuple[str, str], ...]:
        """Operator-visible warnings and errors seen during the run."""

        return tuple(
            (severity, message)
            for severity, message in getattr(self, "reports", ())
            if severity in {"warning", "error"}
        )

    def surviving_threads(self, grace_s: float = 3.0) -> tuple[str, ...]:
        """Non-daemon threads that OUTLIVE close: what keeps the process up.

        Some product pools shut down with ``wait=False`` on purpose -- the
        GUI thread must not block inside a node callback -- so a worker can
        still be finishing its last call when ``close()`` returns.  That is
        a thread ending, not a thread leaking, and reporting the two the
        same way turns the guard into noise.  Anything still standing after
        the grace window is the failure this check was written for: the
        console stayed on screen until it was killed from the task list.
        """

        import threading

        def alive() -> set:
            return {
                thread
                for thread in threading.enumerate()
                if thread is not threading.main_thread() and not thread.daemon
            }

        deadline = time.monotonic() + max(0.0, grace_s)
        left = alive()
        while left and time.monotonic() < deadline:
            time.sleep(0.05)
            left = alive()
        return tuple(sorted(thread.name for thread in left))


def render_cost(seams, frames_by_panel: dict) -> list[dict]:
    """Per-frame render cost per panel, rolled up from the self-times.

    Self-times sum to the frame by construction, so grouping them by the
    panel prefix gives what one panel costs to draw once -- which is the
    number the session and host layers report, and the only way to put a
    console panel beside a standalone plot.  The frame gap cannot do it:
    the console is beat-paced, so every panel reports the beat.
    """

    totals: dict[str, list] = {}
    renders: dict[str, int] = {}
    for row in seams:
        seam = row["seam"]
        # Pipeline probes use the same ``[panel]`` identity, but they are not
        # renderer children.  Accept only the renderer/canvas prefix; otherwise
        # FitEngine time is counted once as fit and again as "render cost".
        if not seam.startswith("MatplotlibRenderer[") or "]" not in seam:
            continue
        panel = seam[seam.index("[") + 1:seam.index("]")]
        if seam == f"MatplotlibRenderer[{panel}].present":
            renders[panel] = int(row["calls"])
        entry = totals.setdefault(panel, [0.0, 0.0])
        entry[0] += row["self_ms_total"]
        entry[1] += row["self_ms_per_call"] * row["calls"] * row["cpu_share"]
    out = []
    for panel, (wall, cpu) in sorted(totals.items(), key=lambda i: -i[1][0]):
        frames = max(1, renders.get(panel, frames_by_panel.get(panel, 0)))
        out.append({
            "panel": panel,
            "frames": frames,
            "presented_frames": int(frames_by_panel.get(panel, 0)),
            "wall_ms_per_frame": round(wall / frames, 2),
            "cpu_ms_per_frame": round(cpu / frames, 2),
        })
    return out


def _print_render_cost(payload: dict, seconds: float, frames_by_panel: dict) -> None:
    rows = render_cost(probe.rows(seconds), frames_by_panel)
    if not rows:
        return
    payload["render_cost"] = rows
    print()
    print("render cost per frame (self-times rolled up per panel)")
    for row in rows:
        print("   %-12s %7.2f ms wall   %7.2f ms cpu   over %d frames"
              % (row["panel"], row["wall_ms_per_frame"],
                 row["cpu_ms_per_frame"], row["frames"]))


def _print_problems(payload: dict) -> None:
    problems = payload.get("problems") or []
    print()
    if not problems:
        print("the console reported no warnings or errors")
        return
    print("THE CONSOLE REPORTED %d problem(s) -- read these before any "
          "number above:" % len(problems))
    seen = set()
    for item in problems:
        line = "   %-8s %s" % (item["severity"], item["message"])
        if line in seen:
            continue
        seen.add(line)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="image")
    parser.add_argument("--panels", default="",
                        help="comma-separated kinds measured TOGETHER, e.g. "
                             "image,curve,histogram,rolling")
    parser.add_argument("--chain", default="",
                        help="derive from the camera panel and put this kind "
                             "on the derived signal, e.g. --chain image")
    parser.add_argument("--size", default=SIZE_PRESET)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--gesture", action="store_true")
    parser.add_argument("--motion", default="auto",
                        choices=("auto", "pan", "area", "orbit", "clim"),
                        help="which hand to drive: the middle-button view "
                             "pan (default for every 2-D kind), the orbit "
                             "on height bars, the left-button area, or clim")
    parser.add_argument("--edits", action="store_true",
                        help="time the small Setting-form edits (title, bins, "
                             "clim) that an operator makes mid-run")
    parser.add_argument("--stalls", action="store_true",
                        help="attribute every slipped cycle to producer, "
                             "event loop or render")
    parser.add_argument("--allow-low-density", action="store_true")
    parser.add_argument("--top", type=int, default=18)
    args = parser.parse_args()

    bench = ConsoleBench(allow_low_density=args.allow_low_density)
    layout = [item.strip() for item in args.panels.split(",") if item.strip()]
    with bench:
        # A PERFORMANCE run pins the window: two runs are only comparable if
        # the card geometry is identical, and the product's own size follows
        # whatever screen it lands on.  Anything looking at behaviour uses
        # the default, which is the product's.
        bench.start()
        if layout:
            panels = bench.add_panels(layout, size=args.size)
            payload = {
                "scenario": "layout",
                "kinds": layout,
                "size": args.size,
                "density": bench.density(panels[0]),
            }
            for panel in panels:
                bench.instrument(panel)
            payload["together"] = bench.live_all(panels, args.seconds)
            if args.gesture:
                # WITH THE SIBLINGS LIVE.  The arbiter exists because a
                # hand competes with the other panels for one machine, so
                # a pointer measurement taken alone cannot see the thing
                # the arbiter was built for.
                payload["gesture"] = bench.gesture(
                    panels[0], kind=layout[0], motion=args.motion
                )
        elif args.chain:
            source = bench.add_panel(args.kind, size=args.size)
            guards.require_panels(bench.presenter, 1)
            signal = bench.derive(source)
            downstream = bench.add_panel_on(signal, args.chain, size=args.size)
            guards.require_panels(bench.presenter, 2)
            payload = {
                "scenario": "chain",
                "kinds": [args.kind, args.chain],
                "derived_signal": signal,
                "size": args.size,
                "density": bench.density(source),
            }
            bench.instrument(source)
            bench.instrument(downstream)
            payload["together"] = bench.live_all([source, downstream], args.seconds)
        else:
            panel = bench.add_panel(args.kind, size=args.size)
            # The scenario declares ONE panel; anything else is contention
            # the numbers would silently carry.
            guards.require_panels(bench.presenter, 1)
            payload = {
                "scenario": "solo",
                "kind": args.kind,
                "size": args.size,
                "panels_in_console": len(bench.presenter.panels),
                "density": bench.density(panel),
            }
            bench.instrument(panel)
            if args.stalls:
                # One window, measured and attributed.
                payload["stalls"] = bench.attribute_stalls(panel, args.seconds)
                payload["live"] = payload["stalls"]
            else:
                payload["live"] = bench.live(panel, args.seconds)
            if args.edits:
                payload["edits"] = bench.edit_run(panel, kind=args.kind)
            if args.gesture:
                payload["gesture"] = bench.gesture(
                    panel, kind=args.kind, motion=args.motion
                )
    if "edits" in payload:
        block = payload["edits"]
        print("")
        print("Setting-form edits while live  (%d of %d redrew)"
              % (block["answered"], block["of"]))
        for row in sorted(block["edits"],
                          key=lambda item: -(item["to_next_front_ms"] or 0.0)):
            field = ", ".join("%s=%r" % item for item in row["values"].items())
            print("   %-44s %s"
                  % (field,
                     "REFUSED by the panel (holds %r)" % row["refused"]
                     if row["refused"] else
                     "no redraw" if not row["answered"]
                     else "%7.1f ms to the new picture" % row["to_next_front_ms"]))
        for item in block["skipped"]:
            print("   %-44s not exercised: %s" % (item["field"], item["why"]))
    payload["threads_left_running"] = list(bench.surviving_threads())
    payload["problems"] = [
        {"severity": severity, "message": message}
        for severity, message in bench.problems()
    ]

    print("%s  size=%s  %s px  DPR %s" % (
        payload.get("scenario"),
        args.size,
        payload["density"]["figure_px"],
        payload["density"]["device_pixel_ratio"],
    ))
    if "together" in payload:
        together = payload["together"]
        print("%d panels over %.1f s, beat %s ms" % (
            len(together["panels"]), together["window_s"], together["beat_ms"]))
        if payload.get("derived_signal"):
            print("derived signal: %s" % payload["derived_signal"])
        for row in together["panels"]:
            print("   %-12s %5.1f fps  gap med %6.1f p90 %6.1f max %6.1f  "
                  "stalls %d" % (
                      row["panel"], row["frames_per_second"],
                      row["frame_gap"].get("median_ms", 0.0),
                      row["frame_gap"].get("p90_ms", 0.0),
                      row["frame_gap"].get("max_ms", 0.0),
                      row["stalls_over_two_beats"]))
        print("process: %.0f%% of one core,  RSS %.1f -> %.1f MB" % (
            together["cpu_percent_of_one_core"],
            together["rss_mb_before"], together["rss_mb_after"]))
        print()
        print(probe.report(together["window_s"], top=args.top))
        _print_render_cost(
            payload,
            together["window_s"],
            {row["panel"]: row["frames"] for row in together["panels"]},
        )
        _print_problems(payload)
        left = payload["threads_left_running"]
        print()
        print("non-daemon threads still alive after close: %s"
              % (", ".join(left) if left else "none"))
        path = write_result(payload, "console-%s-%s" % (
            payload.get("scenario"), args.size))
        print(f"wrote {path}")
        return
    live = payload["live"]
    print("live: %d frames in %.1f s = %.1f/s (beat %s ms)" % (
        live["frames"], live["window_s"], live["frames_per_second"],
        live["beat_ms"],
    ))
    print("      gap %s" % (live["frame_gap"],))
    print("      %d stalls past two beats; worst gaps %s ms" % (
        live["stalls_over_two_beats"], live["worst_gaps_ms"],
    ))
    if "stalls" in payload:
        st = payload["stalls"]
        print()
        print("%d frames, %d source revisions, %d beat ticks in %.1f s"
              % (st["frames"], st["source_revisions"], st["beat_ticks"],
                 st["window_s"]))
        print("beat tick gap %s" % (st["beat_tick_gap"],))
        print("frame gap     %s" % (st["frame_gap"],))
        print("content stride (source revisions skipped between two shown "
              "frames): %s" % (st["content_stride_counts"],))
        if st["slips"]:
            print("slipped cycles:")
            for slip in st["slips"]:
                print("   %6.1f ms  beats=%d revisions=%d  worst tick gap "
                      "%6.1f ms  -> %s"
                      % (slip["gap_ms"], slip["beats_in_gap"],
                         slip["new_revisions_in_gap"],
                         slip["longest_tick_gap_ms"], slip["blame"]))
        else:
            print("no cycle slipped past 1.5 beats")
    if "gesture" in payload:
        print("gesture: %s" % (payload["gesture"],))
    print()
    print(probe.report(live["window_s"], top=args.top))
    _print_render_cost(payload, live["window_s"], {args.kind: live["frames"]})
    _print_problems(payload)
    left = payload["threads_left_running"]
    print()
    print("non-daemon threads still alive after close: %s"
          % (", ".join(left) if left else "none"))
    path = write_result(payload, f"console-{args.kind}-{args.size}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
