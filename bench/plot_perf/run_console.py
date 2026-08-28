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
    "_box_sized_rgba_front",
    "_mutate_image_artists",
    "_cached_image_range",
    "_blit_exact_rgba_image",
    "_dynamic_artists",
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
        self._tmp = pathlib.Path(tempfile.mkdtemp(prefix="console-bench-"))

    # ------------------------------------------------------------ lifecycle
    def start(
        self,
        *,
        camera: str = "mot_camera",
        exposure: float = 0.01,
        clear_preview_panels: bool = True,
        window_size: tuple[int, int] | None = None,
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
        # AND THE PRODUCT'S OWN SIZE BY DEFAULT.  This used to pin 1600x1000
        # unconditionally, "so the card geometry is the same on every run".
        # It also meant every acceptance measurement was taken at a size the
        # operator never sees -- and card size decides the Setting frame's
        # height cap, so a whole class of frame behaviour was measured in a
        # regime that does not occur.  A benchmark that needs two runs to be
        # comparable asks for a size explicitly; looking at the product does
        # not.
        if window_size is not None:
            window.resize(int(window_size[0]), int(window_size[1]))
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
        self._kinds[panel.panel_id] = kind
        return panel

    def surface(self, panel):
        card = self.view._cards.get(panel.panel_id)
        return None if card is None else card.surface

    def renderer(self, panel):
        return panel.host._session._renderer

    # ---------------------------------------------------------- measurement
    def instrument(self, panel, *, seams=None) -> list[str]:
        """Bind self-time probes to THIS panel's renderer, and to the
        module-level work a frame does outside it.

        A renderer-only tap hides the front store: the source reduction, the
        pyramid and the block mean all live in ``_image_raster`` as plain
        functions, and they were a third of some frames while every visible
        row said the renderer was cheap.  Module functions have exactly one
        instance, so watching them is safe where watching a class is not.
        """

        # Four panels reporting under one class name is the class-level tap
        # again, one level up: _native_draw fired 85 times across a layout
        # and there was no way to say whose frames those were.
        label = "%s[%s]" % (
            type(self.renderer(panel)).__name__,
            self._kinds.get(panel.panel_id) or panel.panel_id,
        )
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

        if not getattr(ConsoleBench, "_module_seams_bound", False):
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
        with guards.ProductBeat(self.app, self.presenter) as beat:
            elapsed = beat.run(seconds, tick=presented.poll)
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

    def live_all(self, panels, seconds: float = 10.0) -> dict:
        """Every panel measured in ONE window, plus what the process paid.

        A panel measured alone is a floor.  Panels share one machine, one
        beat and one plane, and the thing an operator sees is what they do
        TOGETHER -- measured here rather than inferred from adding up solo
        numbers, which is how a per-panel cost gets reported as if it
        composed linearly.
        """

        import psutil

        process = psutil.Process()
        counters = [_PanelFronts(self, panel) for panel in panels]
        guards.free_running(self.session)
        self._pump(1.0)
        probe.reset()
        for counter in counters:
            counter.reset()

        cpu_before = process.cpu_times()
        rss_before = process.memory_info().rss

        def tick():
            for counter in counters:
                counter.poll()

        with guards.ProductBeat(self.app, self.presenter) as beat:
            elapsed = beat.run(seconds, tick=tick)

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
                    "panel": self._kinds.get(panel.panel_id) or panel.state.kind,
                    "frames": counter.count,
                    "frames_per_second": round(counter.count / elapsed, 1),
                    "frame_gap": stats(gaps),
                    "stalls_over_two_beats": sum(
                        1 for gap in gaps if gap > 2 * beat_s
                    ),
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
        self._kinds[panel.panel_id] = kind
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
        started = time.perf_counter()
        self.view.panel_state_changed.emit(panel.panel_id, {section: current})
        answered = None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            self.presenter.beat()
            self.app.processEvents()
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
    for row in seams:
        seam = row["seam"]
        if "[" not in seam or "]" not in seam:
            continue
        panel = seam[seam.index("[") + 1:seam.index("]")]
        entry = totals.setdefault(panel, [0.0, 0.0])
        entry[0] += row["self_ms_total"]
        entry[1] += row["self_ms_per_call"] * row["calls"] * row["cpu_share"]
    out = []
    for panel, (wall, cpu) in sorted(totals.items(), key=lambda i: -i[1][0]):
        frames = max(1, frames_by_panel.get(panel, 0))
        out.append({
            "panel": panel,
            "frames": frames,
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
        bench.start(window_size=(1600, 1000))
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
                payload["gesture"] = bench.gesture(panel, kind=args.kind)
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
