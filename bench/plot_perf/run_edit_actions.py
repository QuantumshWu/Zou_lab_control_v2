"""What the operator waits for on a deep-history grid: Edit, scroll, Refresh, Save.

The four-panel console of ``run_mot_roi_chain`` with the ROI history leased
``--window`` deep (1000 by default): a camera grid, an ROI histogram over
the window, an ROI grid faceted by rows whose curve cells reduce the whole
window and fit, and an ROI curve.  Then the actions an operator takes on
the grid, timed the way they are felt:

* Edit      -- open the panel's Edit tab, until its frozen surface shows.
* scroll    -- drag the Edit page's scroll bar top to bottom in steps.
* Refresh   -- Refresh snapshot, until the editor shows the newer freeze.
* Save fig  -- Save figure, until the console reports the files written.
* close     -- close the Edit tab.

Two clocks per action: wall time to the visible answer, and the longest
single turn of the Qt event loop while waiting -- the stall a human feels
as the window freezing, whatever thread the work is on.

Run: python -m bench.plot_perf.run_edit_actions --window 1000
"""

from __future__ import annotations

import argparse
import os
import pathlib
import tempfile
import time

from . import guards
from .common import provenance, write_result
from .run_console import ConsoleBench
from .run_mot_roi_chain import _assign_roles, _draw_mot_roi, _source_index_size


class _OwnerSampler:
    """Where the GUI thread's Python time goes, sampled from a helper thread.

    cProfile on Python 3.12+ records every thread, so a profile started on
    the owner cannot say what the OWNER was doing.  Sampling the main
    thread's frame every few milliseconds can: each sample is charged to
    the innermost frame that belongs to this repository, and the busiest
    stacks are kept whole.  A sample inside the event pump with nothing
    of ours above it is idle.
    """

    def __init__(self, interval: float = 0.004) -> None:
        import collections
        import sys
        import threading

        self._sys = sys
        self._interval = interval
        self._main = threading.main_thread().ident
        self._stop = threading.Event()
        self.frames: collections.Counter = collections.Counter()
        self.stacks: collections.Counter = collections.Counter()
        self.log: list[tuple[float, tuple[str, str], tuple]] = []
        self.samples = 0
        self._thread = threading.Thread(target=self._run, name="owner-sampler", daemon=True)

    def __enter__(self) -> "_OwnerSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @staticmethod
    def _ours(frame) -> tuple[str, str] | None:
        """The innermost frame of this repository on one thread, if any."""

        while frame is not None:
            name = frame.f_code.co_filename
            if "zlc_" in name and "run_edit_actions" not in name:
                return (name, frame.f_code.co_name)
            frame = frame.f_back
        return None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            frames = self._sys._current_frames()
            frame = frames.get(self._main)
            if frame is None:
                continue
            stack = []
            while frame is not None and len(stack) < 30:
                code = frame.f_code
                stack.append((code.co_filename, code.co_name))
                frame = frame.f_back
            self.samples += 1
            own = next(
                (entry for entry in stack if "zlc_" in entry[0] and "run_edit_actions" not in entry[0]),
                None,
            )
            busy = []
            for ident, other in frames.items():
                if ident == self._main or self._ours(other) is None:
                    continue
                code = other.f_code
                busy.append((code.co_filename, code.co_name))
            charged = own or ("idle", "event pump")
            self.log.append((time.perf_counter(), charged, tuple(busy)))
            self.frames[charged] += 1
            if own is None:
                continue
            trimmed = tuple(
                (pathlib.Path(name).name, func)
                for name, func in stack
                if "run_edit_actions" not in name and "processEvents" not in func
            )[:7]
            self.stacks[trimmed] += 1

    def during(self, window: tuple[float, float]) -> list[tuple[str, int]]:
        """What the owner ran inside one time window, busiest first."""

        import collections

        begin, end = window
        inside = [entry for entry in self.log if begin <= entry[0] <= end]
        counts: collections.Counter = collections.Counter(
            charged for _stamp, charged, _others in inside
        )
        busy_samples = sum(1 for _stamp, _charged, others in inside if others)
        elsewhere: collections.Counter = collections.Counter(
            f"{pathlib.Path(name).name}:{func}"
            for _stamp, _charged, others in inside
            for name, func in others
        )
        return [
            (f"{pathlib.Path(name).name}:{func}", count)
            for (name, func), count in counts.most_common(8)
        ] + [
            (f"samples with other threads in our Python: {busy_samples}/{len(inside)}", 0)
        ] + [
            (f"other thread in {name}", count) for name, count in elsewhere.most_common(6)
        ]

    def summary(self, top: int = 12) -> dict:
        def short(entry):
            name, func = entry
            return f"{pathlib.Path(name).name}:{func}"

        total = max(1, self.samples)
        return {
            "samples": self.samples,
            "interval_ms": self._interval * 1000.0,
            "owner_busy_fraction": round(
                1.0 - self.frames[("idle", "event pump")] / total, 3
            ),
            "top_frames": [
                (short(entry), count) for entry, count in self.frames.most_common(top)
            ],
            "top_stacks": [
                (" < ".join(f"{name}:{func}" for name, func in stack), count)
                for stack, count in self.stacks.most_common(6)
            ],
        }


class _LoopClock:
    """Longest single pass of the event loop between two marks."""

    def __init__(self, app) -> None:
        from PyQt5 import QtCore

        self._app = app
        self._flag = QtCore.QEventLoop.AllEvents
        self.turns: list[float] = []
        self.longest: tuple[float, float] = (0.0, 0.0)

    def pump(self) -> None:
        begin = time.perf_counter()
        self._app.processEvents(self._flag, 20)
        end = time.perf_counter()
        self.turns.append(end - begin)
        if end - begin > self.longest[1] - self.longest[0]:
            self.longest = (begin, end)

    def summary(self) -> dict:
        turns = sorted(self.turns)
        if not turns:
            return {"turns": 0}
        stalls = [turn for turn in turns if turn > 0.1]
        return {
            "turns": len(turns),
            "longest_turn_ms": round(turns[-1] * 1000.0, 1),
            "p95_turn_ms": round(turns[int(0.95 * (len(turns) - 1))] * 1000.0, 1),
            "stalls_over_100ms": len(stalls),
            "stalled_ms": round(sum(stalls) * 1000.0, 1),
        }


def _wait(bench: ConsoleBench, clock: _LoopClock, predicate, what: str, timeout: float) -> float:
    """Pump the product beat until ``predicate``; return the wall time."""

    from PyQt5 import QtCore

    cursor = bench.report_cursor()
    timer = QtCore.QTimer()
    timer.setInterval(int(bench.presenter.board.base_interval_ms))
    timer.timeout.connect(bench.presenter.beat)
    timer.start()
    began = time.perf_counter()
    try:
        deadline = began + timeout
        while True:
            clock.pump()
            errors = bench.errors_since(cursor)
            if errors:
                raise guards.HarnessError(f"{what} failed: " + " | ".join(errors))
            if predicate():
                return time.perf_counter() - began
            if time.perf_counter() > deadline:
                raise guards.HarnessError(f"timed out waiting for {what}")
    finally:
        timer.stop()


def _editor_front(binding):
    host = binding.editor_host
    return (
        host is not None
        and binding.editor_configuration is None
        and getattr(host, "front", None) is not None
    )


def _editor_view(bench: ConsoleBench, panel):
    return bench.view._panel_editors[str(panel.panel_id)]


def _timed_action(bench: ConsoleBench, what: str, trigger, predicate, *, timeout: float = 120.0) -> dict:
    """Time one action; with ZLC_EDIT_PROFILE set, profile the owner thread too."""

    clock = _LoopClock(bench.app)
    profile = None
    if os.environ.get("ZLC_EDIT_PROFILE"):
        import cProfile

        profile = cProfile.Profile()
        profile.enable()
    with _OwnerSampler() as sampler:
        began = time.perf_counter()
        result = trigger()
        triggered = time.perf_counter()
        trigger_ms = (triggered - began) * 1000.0
        if result is False:
            raise guards.HarnessError(f"{what} was refused")
        wall = _wait(bench, clock, predicate, what, timeout)
    row = {
        "what": what,
        "trigger_ms": round(trigger_ms, 1),
        "wall_ms": round(wall * 1000.0, 1),
        **clock.summary(),
        "owner": sampler.summary(),
        "longest_turn_frames": sampler.during(clock.longest),
        "trigger_frames": sampler.during((began, triggered)),
    }
    if profile is not None:
        import io
        import pstats

        profile.disable()
        cumulative = io.StringIO()
        pstats.Stats(profile, stream=cumulative).sort_stats("cumulative").print_stats(
            r"zlc_|bench", 40
        )
        internal = io.StringIO()
        pstats.Stats(profile, stream=internal).sort_stats("tottime").print_stats(25)
        callers = io.StringIO()
        pstats.Stats(profile, stream=callers).sort_stats("cumulative").print_callers(
            r"_compose_frame|_capture_front|evaluate_processor|device_types.py.*render|_raster_capture_rgba_bytes|_image_arrays|describe_semantics|_panel_projection"
        )
        row["owner_profile"] = cumulative.getvalue()
        row["owner_profile_tottime"] = internal.getvalue()
        row["owner_profile_callers"] = callers.getvalue()
    return row


def _scroll(bench: ConsoleBench, editor, steps: int = 20, *, hide: str = "") -> dict:
    """Scroll the editor top to bottom and back; ``hide`` names what to hide.

    ``"surface"`` hides the plot widget, ``"forms"`` hides everything else
    in the scrolled column -- the difference between the three passes is
    the attribution a stack sample cannot give, because the paint is Qt's.
    """

    from PyQt5 import QtWidgets

    hidden: list = []
    if hide == "surface":
        hidden = [editor.surface_holder]
    elif hide == "forms":
        column = editor.surface_holder.parentWidget()
        hidden = [
            child
            for child in column.findChildren(QtWidgets.QWidget)
            if child.parentWidget() is column and child is not editor.surface_holder
        ]
    for widget in hidden:
        widget.hide()
    bench.app.processEvents()
    bar = editor.scroll.verticalScrollBar()
    clock = _LoopClock(bench.app)
    per_step: list[float] = []
    top, bottom = bar.minimum(), bar.maximum()
    try:
        with _OwnerSampler() as sampler:
            for index in list(range(steps + 1)) + list(range(steps, -1, -1)):
                value = top + (bottom - top) * index // steps
                began = time.perf_counter()
                bar.setValue(value)
                clock.pump()
                clock.pump()
                per_step.append((time.perf_counter() - began) * 1000.0)
    finally:
        for widget in hidden:
            widget.show()
        bench.app.processEvents()
    per_step.sort()
    return {
        "what": "scroll" + (f" (no {hide})" if hide else ""),
        "range": [int(top), int(bottom)],
        "steps": len(per_step),
        "median_step_ms": round(per_step[len(per_step) // 2], 1),
        "worst_step_ms": round(per_step[-1], 1),
        **clock.summary(),
        "owner": sampler.summary(),
        "longest_turn_frames": sampler.during(clock.longest),
    }


def run(*, window: int, exposure: float, save_dir: pathlib.Path) -> dict:
    bench = ConsoleBench()
    payload: dict = {
        "scenario": "edit-actions-on-deep-history-grid",
        "provenance": provenance(),
        "requested": {"window": window, "exposure_seconds": exposure},
        "actions": [],
    }
    with bench:
        bench.start(camera="mot_camera", exposure=exposure, clear_preview_panels=False)
        (camera,) = tuple(bench.presenter.panels.values())
        bench._labels[camera.panel_id] = "camera-grid"
        bench._kinds[camera.panel_id] = "facet_grid:image"
        bench._until(
            lambda: camera.host is not None and bench.surface(camera) is not None
            and bench.surface(camera).presented_front is not None,
            "camera preview",
        )
        roi = _draw_mot_roi(bench, camera)
        roi_signal = roi["signal"]

        histogram = bench.add_panel_on(roi_signal, "histogram", size="2x2")
        bench._labels[histogram.panel_id] = f"histogram-{window}"
        bench.edit_setting(histogram, "display", window=window)

        grid = bench.presenter.add_selected_panel("facet_grid")
        bench.view.panel_state_changed.emit(
            grid.panel_id, {"cell_kind": "curve", "signal": roi_signal, "size": "2x2"}
        )
        bench._until(
            lambda: grid.state.cell_kind == "curve" and grid.host is not None
            and bench.surface(grid) is not None
            and bench.surface(grid).presented_front is not None,
            "curve-cell grid",
        )
        bench._name(grid, "facet_grid:curve")
        bench._labels[grid.panel_id] = f"grid-rows-over-{window}"
        bench._pump(2.0)
        _assign_roles(
            bench, grid,
            {"spatial-y": "facet", "spatial-x": "x", "source index": "reduced"},
        )
        bench.edit_setting(grid, "fit", model="gaussian_offset")

        curve = bench.add_panel_on(roi_signal, "curve", size="2x2")
        bench._labels[curve.panel_id] = "curve"
        _assign_roles(
            bench, curve,
            {"source index": "x", "spatial-x": "reduced", "spatial-y": "reduced"},
        )

        began = time.perf_counter()
        bench._until(
            lambda: _source_index_size(bench, roi_signal) >= window,
            f"{window}-shot ROI history",
            timeout=600.0,
        )
        payload["history_fill_s"] = round(time.perf_counter() - began, 1)
        bench._pump(3.0)
        guards.require_panels(bench.presenter, 4)
        payload["density"] = {bench.label(p): bench.density(p) for p in (camera, histogram, grid, curve)}

        actions = payload["actions"]
        presenter = bench.presenter
        for label, panel in (("grid", grid), ("histogram", histogram)):
            actions.append(_timed_action(
                bench, f"{label}: edit",
                lambda p=panel: presenter.edit_panel(p.panel_id),
                lambda p=panel: _editor_front(p),
            ))
            editor = _editor_view(bench, panel)
            bench._pump(1.0)
            scrolled = _scroll(bench, editor)
            scrolled["what"] = f"{label}: scroll"
            actions.append(scrolled)
            for hide in ("surface", "forms"):
                scrolled = _scroll(bench, editor, hide=hide)
                scrolled["what"] = f"{label}: scroll (no {hide})"
                actions.append(scrolled)
            before = panel.frozen_data
            actions.append(_timed_action(
                bench, f"{label}: refresh snapshot",
                lambda p=panel: presenter.refresh_panel_snapshot(p.panel_id),
                lambda p=panel, b=before: p.frozen_data is not b and _editor_front(p),
            ))
            target = save_dir / f"{label}-window{window}"
            cursor = bench.report_cursor()
            actions.append(_timed_action(
                bench, f"{label}: save fig",
                lambda p=panel, t=target: presenter.save_panel_figure(p.panel_id, str(t)),
                lambda c=cursor: any(
                    "panel saved" in message for _severity, message in bench.reports_since(c)
                ),
                timeout=600.0,
            ))
            image = target.with_suffix(".png")
            archive = target.with_suffix(".npz")
            actions[-1]["png_bytes"] = image.stat().st_size if image.exists() else None
            actions[-1]["npz_bytes"] = archive.stat().st_size if archive.exists() else None
            if os.environ.get("ZLC_EDIT_PROFILE"):
                # The same save once more, synchronously and profiled, so
                # the worker's share (archive encode, export render) is
                # attributed by function rather than guessed from the wall.
                import cProfile
                import io
                import pstats

                from zlc_workbench.panel_save import save_panel_figure

                profile = cProfile.Profile()
                began = time.perf_counter()
                profile.enable()
                save_panel_figure(
                    save_dir / f"{label}-window{window}-profiled",
                    state=panel.state,
                    frozen=panel.frozen_data,
                )
                profile.disable()
                out = io.StringIO()
                pstats.Stats(profile, stream=out).sort_stats("cumulative").print_stats(
                    r"zlc_|numpy|zipfile|zlib|matplotlib", 45
                )
                actions.append({
                    "what": f"{label}: save fig (sync, profiled)",
                    "wall_ms": round((time.perf_counter() - began) * 1000.0, 1),
                    "owner_profile": out.getvalue(),
                    "owner_profile_tottime": "",
                })
            actions.append(_timed_action(
                bench, f"{label}: close editor",
                lambda p=panel: presenter.close_panel_editor(p.panel_id),
                lambda p=panel: p.editor_host is None,
            ))
        payload["problems"] = bench.problems()
    return payload


def _print(payload: dict) -> None:
    print(f"history fill: {payload.get('history_fill_s')} s")
    print(f"{'action':28s} {'trigger':>8s} {'wall':>9s} {'longest turn':>13s} {'stalls>100ms':>13s}")
    for row in payload["actions"]:
        print(
            f"{row['what']:28s} {row.get('trigger_ms', '-'):>8} {row.get('wall_ms', row.get('worst_step_ms', '-')):>9} "
            f"{row.get('longest_turn_ms', '-'):>13} {row.get('stalls_over_100ms', '-'):>13}"
            + (f"   png {row['png_bytes']} npz {row['npz_bytes']}" if "npz_bytes" in row else "")
            + (f"   median step {row['median_step_ms']} ms" if "median_step_ms" in row else "")
        )
    if payload.get("problems"):
        print("problems:", payload["problems"])
    for row in payload["actions"]:
        owner = row.get("owner")
        if owner:
            print("")
            print(
                f"==== owner thread during {row['what']}: busy "
                f"{owner['owner_busy_fraction']} of {owner['samples']} samples"
            )
            for name, count in owner["top_frames"][:8]:
                print(f"   {count:5d}  {name}")
            for stack, count in owner["top_stacks"][:4]:
                print(f"   {count:5d}  {stack}")
            if row.get("longest_turn_frames"):
                print(f"   longest turn ({row.get('longest_turn_ms')} ms) ran:", row["longest_turn_frames"])
            if row.get("trigger_frames"):
                print(f"   synchronous trigger ({row.get('trigger_ms')} ms) ran:", row["trigger_frames"])
    for row in payload["actions"]:
        if "owner_profile" in row:
            print("")
            print("==== owner-thread profile:", row["what"])
            print(row["owner_profile"][:7000])
            print(row["owner_profile_tottime"][:3500])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--exposure", type=float, default=0.02)
    parser.add_argument("--save-dir", default="")
    arguments = parser.parse_args()
    save_dir = pathlib.Path(arguments.save_dir or tempfile.mkdtemp(prefix="edit-actions-"))
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = run(window=arguments.window, exposure=arguments.exposure, save_dir=save_dir)
    _print(payload)
    print("wrote", write_result(payload, f"edit_actions_w{arguments.window}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
