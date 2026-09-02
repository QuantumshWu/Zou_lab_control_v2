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
        # Threads alive now are the live panels' and the runtime's; one that
        # appears later belongs to the action (a new host's raster worker).
        self._known = frozenset(self._sys._current_frames())
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
                busy.append((code.co_filename, code.co_name, ident in self._known))
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

    _WAITING = ("threading.py", "_base.py", "queue.py", "selectors.py", "socket.py")

    def during(
        self, window: tuple[float, float], *, top_others: int = 6
    ) -> list[tuple[str, int]]:
        """What the owner ran inside one time window, busiest first.

        The other threads' innermost frames are ranked too, with the
        frames of a thread that is merely waiting left out: what is left
        is work that competed for the interpreter during the window.
        """

        import collections

        begin, end = window
        inside = [entry for entry in self.log if begin <= entry[0] <= end]
        counts: collections.Counter = collections.Counter(
            charged for _stamp, charged, _others in inside
        )
        busy_samples = sum(
            1
            for _stamp, _charged, others in inside
            if any(pathlib.Path(name).name not in self._WAITING for name, _func, _old in others)
        )
        elsewhere: collections.Counter = collections.Counter(
            f"{pathlib.Path(name).name}:{func}"
            for _stamp, _charged, others in inside
            for name, func, _old in others
            if pathlib.Path(name).name not in self._WAITING
        )
        new_threads: collections.Counter = collections.Counter(
            f"{pathlib.Path(name).name}:{func}"
            for _stamp, _charged, others in inside
            for name, func, old in others
            if not old and pathlib.Path(name).name not in self._WAITING
        )
        new_samples = sum(
            1
            for _stamp, _charged, others in inside
            if any(not old for _name, _func, old in others)
        )
        return [
            (f"{pathlib.Path(name).name}:{func}", count)
            for (name, func), count in counts.most_common(8)
        ] + [
            (f"samples with other threads working in our Python: {busy_samples}/{len(inside)}", 0)
        ] + [
            (f"other thread in {name}", count)
            for name, count in elsewhere.most_common(top_others)
        ] + [
            (f"samples with the action's own new threads alive: {new_samples}/{len(inside)}", 0)
        ] + [
            (f"new thread in {name}", count)
            for name, count in new_threads.most_common(top_others)
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


class _HostTimeline:
    """Every raster operation submitted while active, per host, with when it
    was submitted, started on the worker, and ended.

    Hosts that existed before the window opened are the live panels'; the
    ones that appear inside it are the action's own -- the Edit surface, an
    export host -- and their chain of operations, with the gaps between
    them, is where a wall time goes that no thread is busy for.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._known: frozenset = frozenset()
        self._original = None

    def __enter__(self) -> "_HostTimeline":
        from zlc_plot import raster

        self._raster = raster
        self._original = raster.RasterPlotHost._submit
        timeline = self
        original = self._original

        def submit(host, callback, **kwargs):
            record = {
                "host": id(host),
                "name": str(kwargs.get("coalesce_key") or getattr(callback, "__qualname__", repr(callback)))[:60],
                "mode": getattr(kwargs.get("mode"), "name", "?"),
                "submitted": time.perf_counter(),
                "started": None,
                "ended": None,
                "done": None,
            }
            timeline.records.append(record)

            def timed():
                record["started"] = time.perf_counter()
                try:
                    return callback()
                finally:
                    record["ended"] = time.perf_counter()

            future = original(host, timed, **kwargs)
            add = getattr(future, "add_done_callback", None)
            if callable(add):
                add(lambda _f: record.__setitem__("done", time.perf_counter()))
            return future

        raster.RasterPlotHost._submit = submit
        return self

    def __exit__(self, *_exc) -> None:
        self._raster.RasterPlotHost._submit = self._original

    def new_hosts(self, since: float) -> list[int]:
        seen: list[int] = []
        for record in self.records:
            if record["submitted"] >= since and record["host"] not in seen:
                seen.append(record["host"])
        return seen

    def rows(self, host: int, origin: float) -> list[str]:
        """This host's operations in order, as ``+ms  mode  name  start run gap``."""

        out = []
        last_end = None
        for record in self.records:
            if record["host"] != host:
                continue
            sub = (record["submitted"] - origin) * 1000.0
            start = None if record["started"] is None else (record["started"] - origin) * 1000.0
            end = None if record["ended"] is None else (record["ended"] - origin) * 1000.0
            gap = "" if last_end is None or start is None else f" gap {start - last_end:6.1f}"
            if end is not None:
                last_end = end
            run = "" if start is None or end is None else f" run {end - start:6.1f}"
            out.append(
                f"      +{sub:7.1f} ms  {record['mode']:<12} {record['name']:<40}"
                f"{' start +%.1f' % start if start is not None else ' (never ran)'}{run}{gap}"
            )
        return out


class _EventWatch:
    """The slowest Qt events delivered while installed, by type and receiver.

    A stack sample of the owner thread inside ``processEvents`` says only
    "Qt".  An application event filter sees every delivered event enter and
    leave (``eventFilter`` runs before the receiver; the time until the
    next event enters is that event's cost, to within the filter's own
    work), so the turn can be charged to a paint of a particular widget, a
    timer, a queued method call.
    """

    def __init__(self, app) -> None:
        from PyQt5 import QtCore

        self._QtCore = QtCore
        self._app = app
        self.slowest: list[tuple[float, str, str]] = []
        self._filter = None

    def __enter__(self) -> "_EventWatch":
        QtCore = self._QtCore
        watch = self

        class Filter(QtCore.QObject):
            def __init__(self):
                super().__init__()
                self._open = None

            def eventFilter(self, receiver, event):  # noqa: N802 - Qt API
                now = time.perf_counter()
                if self._open is not None:
                    began, kind, who = self._open
                    watch._record(now - began, kind, who)
                try:
                    kind = str(event.type())
                    who = f"{receiver.metaObject().className()}:{receiver.objectName() or ''}"
                except Exception:
                    kind, who = "?", "?"
                self._open = (now, kind, who)
                return False

        self._filter = Filter()
        self._app.installEventFilter(self._filter)
        _EventWatch.current = self
        return self

    current: "_EventWatch | None" = None

    def close_open(self) -> None:
        """The pump returned: the open event's cost ends here, not at the next pump."""

        if self._filter is not None and self._filter._open is not None:
            began, kind, who = self._filter._open
            self._record(time.perf_counter() - began, kind, who)
            self._filter._open = None

    def __exit__(self, *_exc) -> None:
        if self._filter is not None:
            self._app.removeEventFilter(self._filter)
            self._filter = None
        if _EventWatch.current is self:
            _EventWatch.current = None

    def _record(self, cost: float, kind: str, who: str) -> None:
        if cost < 0.004:
            return
        self.slowest.append((cost, kind, who))
        if len(self.slowest) > 400:
            self.slowest.sort(reverse=True)
            del self.slowest[200:]

    def summary(self, top: int = 8) -> list[str]:
        self.slowest.sort(reverse=True)
        return [f"{cost * 1000.0:6.1f} ms  {kind}  {who}" for cost, kind, who in self.slowest[:top]]


class _RelayTimer:
    """Every owner-turn relay's slot, timed: which turn a queued call is.

    A queued meta-call on the owner is one of two things here -- the
    board's completion turn, or a Qt worker's delivery drain -- and both
    reach the owner through ``attach_qt_owner_turn``.  Wrapping that
    factory before the console is built names and times every slot it
    creates; the workers are named by the ``attach_qt_worker`` call that
    made them.
    """

    log: list[tuple[float, str, float]] = []
    _worker: list[str] = []

    @classmethod
    def install(cls) -> None:
        from zlc_workbench import board

        if getattr(board.attach_qt_owner_turn, "_bench_timed", False):
            return
        original_turn = board.attach_qt_owner_turn
        original_worker = board.attach_qt_worker

        def attach_qt_owner_turn(turn):
            name = getattr(turn, "__qualname__", repr(turn))
            if "drain" in name and cls._worker:
                name = f"worker drain [{cls._worker[-1]}]"

            def timed(*args, **kwargs):
                began = time.perf_counter()
                try:
                    return turn(*args, **kwargs)
                finally:
                    end = time.perf_counter()
                    cls.log.append((end, name, end - began))

            timed.__qualname__ = name
            return original_turn(timed)

        def attach_qt_worker(name, *args, **kwargs):
            cls._worker.append(str(name))
            try:
                return original_worker(name, *args, **kwargs)
            finally:
                cls._worker.pop()

        attach_qt_owner_turn._bench_timed = True
        board.attach_qt_owner_turn = attach_qt_owner_turn
        board.attach_qt_worker = attach_qt_worker

    @classmethod
    def summary(cls, window: tuple[float, float]) -> list[str]:
        import collections

        begin, end = window
        totals: dict = collections.defaultdict(lambda: [0, 0.0, 0.0])
        for stamp, name, cost in cls.log:
            if begin <= stamp <= end:
                entry = totals[name]
                entry[0] += 1
                entry[1] += cost
                entry[2] = max(entry[2], cost)
        rows = sorted(totals.items(), key=lambda item: -item[1][1])
        return [
            f"{name:<44} calls {count:4d}  total {total * 1000.0:7.1f} ms  worst {worst * 1000.0:6.1f} ms"
            for name, (count, total, worst) in rows[:8]
        ]


class _OwnerSteps:
    """Wall time of each step the owner's turns run, while installed.

    The beat and the completion-driven commit are the two slots a queued
    event delivers to; both are sequences of presenter and board steps.
    Wrapping the bound methods for the window charges every slot to its
    steps -- count, total and worst -- which a stack sample cannot do for
    a step that spends its time in a C call holding the interpreter.
    """

    STEPS = (
        ("presenter", "commit_surfaces"),
        ("presenter", "beat"),
        ("presenter", "_settle_panel_hosts"),
        ("presenter", "_drain_panel_interactions"),
        ("presenter", "_poll_retired_plot_hosts"),
        ("presenter", "_reconcile_panel_derivations"),
        ("presenter", "_report_panel_errors"),
        ("presenter", "poll_logic"),
        ("presenter", "_refresh_signal_choices"),
        ("board", "tick"),
        ("board", "commit"),
    )

    def __init__(self, bench: ConsoleBench) -> None:
        self._bench = bench
        self._originals: list = []
        self.stats: dict[str, list[float]] = {}

    def __enter__(self) -> "_OwnerSteps":
        for owner_name, method in self.STEPS:
            owner = self._bench.presenter if owner_name == "presenter" else self._bench.presenter.board
            original = getattr(owner, method, None)
            if not callable(original):
                continue
            label = f"{owner_name}.{method}"
            self.stats[label] = []

            def timed(*args, _original=original, _label=label, **kwargs):
                began = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.stats[_label].append(time.perf_counter() - began)

            setattr(owner, method, timed)
            self._originals.append((owner, method, original))
        return self

    def __exit__(self, *_exc) -> None:
        for owner, method, original in self._originals:
            try:
                delattr(owner, method)
            except AttributeError:
                setattr(owner, method, original)
            if getattr(owner, method, None) is not original and not callable(getattr(type(owner), method, None)):
                setattr(owner, method, original)
        self._originals.clear()

    def summary(self) -> list[str]:
        rows = []
        for label, samples in self.stats.items():
            if not samples:
                continue
            rows.append((sum(samples), label, len(samples), max(samples)))
        rows.sort(reverse=True)
        return [
            f"{label:<38} calls {count:4d}  total {total * 1000.0:7.1f} ms  worst {worst * 1000.0:6.1f} ms"
            for total, label, count, worst in rows[:9]
        ]


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
        watch = _EventWatch.current
        if watch is not None:
            watch.close_open()
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


def _host_label(bench: ConsoleBench, host_id: int, index: int) -> str:
    """Which panel a raster host belongs to, and whether it is the card or Edit."""

    for panel in bench.presenter.panels.values():
        label = bench._labels.get(panel.panel_id, panel.panel_id)
        if id(panel.host) == host_id:
            return f"{label} (live card)"
        if id(panel.editor_host) == host_id:
            return f"{label} (Edit surface)"
        entry = panel.editor_configuration
        if entry is not None and id(entry[0]) == host_id:
            return f"{label} (Edit surface, staging)"
    return f"host {index} (retired or export)"


def _profile_session_build(bench: ConsoleBench, panel, label: str) -> dict:
    """Build the Edit surface's PlotSession on this thread, profiled.

    The raster worker builds it before its first operation runs -- the
    whole of the wait between the trigger and the surface's first frame --
    and a profile taken there is every thread's.  Built here, from the same
    frozen input through the same spec projection, its self times are its
    own.
    """

    import cProfile
    import io
    import pstats

    import zlc_plot as plot
    from zlc_plot.session import PlotSession
    from zlc_workbench.panel_catalog import task_console_fitting_spec
    from zlc_workbench.panel_state import project_panel_state

    frozen = panel.frozen_data
    plot_input = frozen.plot_input
    snapshot = getattr(plot_input, "snapshot", plot_input)
    state = panel.state
    spec = task_console_fitting_spec(snapshot.block.schema, state.kind, state.cell_kind)
    projection = project_panel_state(snapshot.block.schema, spec, state)
    profile = cProfile.Profile()
    began = time.perf_counter()
    profile.enable()
    session = PlotSession(
        plot_input,
        projection.spec,
        size=state.size,
        parameters=dict(projection.parameters),
        device_pixel_ratio=bench.app.devicePixelRatio() if hasattr(bench.app, "devicePixelRatio") else 1.0,
    )
    profile.disable()
    built = time.perf_counter() - began
    text = io.StringIO()
    pstats.Stats(profile, stream=text).sort_stats("tottime").print_stats(40)
    cumulative = io.StringIO()
    pstats.Stats(profile, stream=cumulative).sort_stats("cumulative").print_stats("zlc_", 30)
    session.close()
    return {
        "what": f"{label}: session build (main thread, profiled)",
        "wall_ms": round(built * 1000.0, 1),
        "trigger_ms": round(built * 1000.0, 1),
        "build_profile": text.getvalue(),
        "build_profile_cumulative": cumulative.getvalue(),
    }


def _steady_state(bench: ConsoleBench, seconds: float) -> dict:
    """The live cards' per-shot pipeline at depth, with no action in flight.

    Every raster operation of every host over ``seconds`` of pumping,
    reduced per host to the number of data frames, the median and worst
    prepare and commit run times, and the median gap a committed frame
    waits before the next begins -- the cadence each card actually
    sustains, which is what an operator sees as a laggy panel.
    """

    import statistics

    clock = _LoopClock(bench.app)
    # The console beats on its own timer in the product; the bench stands
    # that timer up for the window, or nothing is staged at all.
    with guards.ProductBeat(bench.app, bench.presenter), _OwnerSampler() as sampler, _HostTimeline() as timeline, _OwnerSteps(bench) as steps:
        began = time.perf_counter()
        while time.perf_counter() - began < seconds:
            clock.pump()
        finished = time.perf_counter()
    hosts: dict = {}
    for index, host in enumerate(timeline.new_hosts(began)):
        label = _host_label(bench, host, index)
        prepares = []
        commits = []
        starts = []
        for record in timeline.records:
            if record["host"] != host or record["started"] is None or record["ended"] is None:
                continue
            run = (record["ended"] - record["started"]) * 1000.0
            if "stage_prepare" in record["name"]:
                prepares.append(run)
                starts.append(record["started"])
            elif "stage_commit" in record["name"]:
                commits.append(run)
        if not commits:
            continue
        cadence = (
            [(b - a) * 1000.0 for a, b in zip(starts, starts[1:])] if len(starts) > 1 else []
        )
        hosts[label] = {
            "frames": len(commits),
            "prepare_ms_median": round(statistics.median(prepares), 1) if prepares else None,
            "commit_ms_median": round(statistics.median(commits), 1),
            "commit_ms_max": round(max(commits), 1),
            "frame_interval_ms_median": round(statistics.median(cadence), 1) if cadence else None,
        }
    return {
        "what": f"steady state ({seconds:.0f} s, no action)",
        "trigger_ms": 0.0,
        "wall_ms": round((finished - began) * 1000.0, 1),
        **clock.summary(),
        "owner": sampler.summary(),
        "longest_turn_frames": sampler.during(clock.longest),
        "hosts": hosts,
        "owner_steps": steps.summary(),
        "relay_turns": _RelayTimer.summary((began, finished)),
    }


def _timed_action(bench: ConsoleBench, what: str, trigger, predicate, *, timeout: float = 120.0) -> dict:
    """Time one action; with ZLC_EDIT_PROFILE set, profile the owner thread too."""

    clock = _LoopClock(bench.app)
    profile = None
    mode = os.environ.get("ZLC_EDIT_PROFILE", "")
    if mode:
        import cProfile

        profile = cProfile.Profile()
        if mode != "trigger":
            profile.enable()
    with _OwnerSampler() as sampler, _HostTimeline() as timeline, _EventWatch(bench.app) as events, _OwnerSteps(bench) as steps:
        began = time.perf_counter()
        # "trigger" profiles the synchronous call alone: it runs on the
        # owner with the workers mostly waiting, so its self times are the
        # owner's own -- Qt's C++ included, as PyQt method entries.
        if mode == "trigger":
            profile.enable()
        result = trigger()
        if mode == "trigger":
            profile.disable()
        triggered = time.perf_counter()
        trigger_ms = (triggered - began) * 1000.0
        if result is False:
            raise guards.HarnessError(f"{what} was refused")
        wall = _wait(bench, clock, predicate, what, timeout)
        finished = time.perf_counter()
    row = {
        "what": what,
        "trigger_ms": round(trigger_ms, 1),
        "wall_ms": round(wall * 1000.0, 1),
        **clock.summary(),
        "owner": sampler.summary(),
        "longest_turn_frames": sampler.during(clock.longest),
        "trigger_frames": sampler.during((began, triggered)),
        "action_frames": sampler.during((began, finished), top_others=12),
        "host_timelines": {
            _host_label(bench, host, index): timeline.rows(host, began)
            for index, host in enumerate(timeline.new_hosts(began))
        },
        "slowest_events": events.summary(),
        "owner_steps": steps.summary(),
        "relay_turns": _RelayTimer.summary((began, finished)),
    }
    if mode == "trigger":
        import io
        import pstats

        text = io.StringIO()
        pstats.Stats(profile, stream=text).sort_stats("tottime").print_stats(45)
        row["trigger_profile"] = text.getvalue()
        profile = None
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


def run(
    *, window: int, exposure: float, save_dir: pathlib.Path, screenshots: bool = False
) -> dict:
    _RelayTimer.install()
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
        actions.append(_steady_state(bench, 6.0))
        presenter = bench.presenter
        for label, panel in (("grid", grid), ("histogram", histogram)):
            actions.append(_timed_action(
                bench, f"{label}: edit",
                lambda p=panel: presenter.edit_panel(p.panel_id),
                lambda p=panel: _editor_front(p),
            ))
            editor = _editor_view(bench, panel)
            bench._pump(1.0)
            if os.environ.get("ZLC_EDIT_PROFILE") == "build":
                actions.append(_profile_session_build(bench, panel, label))
            if screenshots:
                shot = save_dir / f"{label}-editor.png"
                editor.grab().save(str(shot))
                print(f"editor page: {shot}")
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
        if row.get("hosts"):
            print("")
            print(f"==== {row['what']}: per-host pipeline")
            print("   %-38s %7s %9s %9s %9s %11s" % ("host", "frames", "prep med", "commit md", "commit mx", "interval md"))
            for label, stats in row["hosts"].items():
                print("   %-38s %7d %9s %9s %9s %11s" % (
                    label[:38], stats["frames"], stats["prepare_ms_median"],
                    stats["commit_ms_median"], stats["commit_ms_max"],
                    stats["frame_interval_ms_median"],
                ))
        if row.get("trigger_profile"):
            print(f"   ---- {row['what']}: trigger profile ({row.get('trigger_ms')} ms, tottime)")
            print("\n".join(row["trigger_profile"].splitlines()[:60]))
        if row.get("build_profile"):
            print(f"   ---- {row['what']} ({row.get('wall_ms')} ms, tottime)")
            print("\n".join(row["build_profile"].splitlines()[:55]))
            print("   ---- cumulative, this repository")
            print("\n".join(row["build_profile_cumulative"].splitlines()[:45]))
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
            if row.get("action_frames"):
                print(f"   whole action ({row.get('wall_ms')} ms), other threads worked in:", row["action_frames"][-13:])
            if row.get("slowest_events"):
                print("   slowest Qt events delivered during the action:")
                for line in row["slowest_events"]:
                    print("     ", line)
            if row.get("owner_steps"):
                print("   owner turn steps during the action:")
                for line in row["owner_steps"]:
                    print("     ", line)
            if row.get("relay_turns"):
                print("   owner-turn relays during the action:")
                for line in row["relay_turns"]:
                    print("     ", line)
            for index, rows in (row.get("host_timelines") or {}).items():
                print(f"   new host {index}: {len(rows)} operations (ms after the trigger)")
                for line in rows[:40]:
                    print(line)
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
    parser.add_argument("--screenshots", action="store_true",
                        help="save each editor page as PNG in the save dir")
    arguments = parser.parse_args()
    save_dir = pathlib.Path(arguments.save_dir or tempfile.mkdtemp(prefix="edit-actions-"))
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = run(
        window=arguments.window,
        exposure=arguments.exposure,
        save_dir=save_dir,
        screenshots=arguments.screenshots,
    )
    _print(payload)
    print("wrote", write_result(payload, f"edit_actions_w{arguments.window}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
