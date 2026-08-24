"""Full-pipeline benchmark: RasterPlotHost + Qt widget + real pointer events.

Per case it measures:

* first_render  -- host construction to the first presented front
* live          -- serialized per-revision latency AND free-running Hz
* interactions  -- per-gesture serialized latency (each event waits for its
                   presented front) and, for drags, free-running preview Hz
* live+drag     -- presented Hz while data streams and a drag is active
* fit           -- live-fit latency per revision (when the case asks)

Run:  python -m bench.plot_perf.run_host [--only substring] [--label name]
"""
from __future__ import annotations

import argparse
import math
import time
import traceback

from .common import (
    Pointer,
    Presented,
    axis_by_role,
    axis_center,
    pump,
    pump_until,
    qt_env,
    stats,
    write_result,
)
from .cases import Case, catalog

FRONT_TIMEOUT = 6.0
SIZE_PRESET = "4x4"


def _curve_on_point(transform, dims0: int = 10):
    """A display point ON the mean curve of the lattice profile."""

    peak = (dims0 - 1) / 2.0
    x = float(math.floor(peak))
    y = math.exp(-((x - peak) ** 2) / 4.0)
    return transform.display_to_normalized(x, y)


def _inside(transform, fx: float, fy: float):
    left, top, right, bottom = transform.bounds
    return left + fx * (right - left), top + fy * (bottom - top)


class HostBench:
    def __init__(self, case: Case):
        self.case = case
        self.report: dict = {"case": case.name, "notes": case.notes}
        self.app, self.QtCore, self.QtGui, self.QtWidgets = qt_env()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        from zlc_plot import Qt5PlotWidget
        from zlc_plot.raster import RasterPlotHost

        self.feed = self.case.feed()
        self.report["points"] = self.feed.size
        spec = self.case.spec()
        t0 = time.perf_counter()
        self.host = RasterPlotHost.from_plot(self.feed.next(), spec)
        self.host.configure(size=SIZE_PRESET)
        self.widget = Qt5PlotWidget(self.host)
        self.widget.show()
        self.presented = Presented(self.widget)
        ok = pump_until(
            self.app,
            lambda: self.presented.poll() or self.presented.count > 0,
            timeout=30.0,
        )
        first = time.perf_counter() - t0
        # The size preset may land in a second front; wait for quiescence.
        pump(self.app, 0.3)
        self.presented.poll()
        self.report["first_render_ms"] = round(first * 1e3, 1) if ok else None
        self.pointer = Pointer(self.widget, self.app, self.QtCore, self.QtGui)
        if self.case.fit:
            done = self.host.configure(fit=dict(self.case.fit))
            pump_until(self.app, done.done, timeout=20.0)
            pump(self.app, 0.5)
            self.presented.poll()

    def close(self) -> None:
        try:
            self.widget.close_adapter()
        finally:
            self.host.close()

    def front(self):
        return self.widget.presented_front

    # ------------------------------------------------------------ live feed
    def bench_live(self) -> None:
        # Warm.
        for _ in range(2):
            self.host.update_data(self.feed.next())
            self.presented.wait_next(self.app, FRONT_TIMEOUT)
        # Serialized: one revision, wait for its presentation.
        waits = []
        for _ in range(12):
            self.host.update_data(self.feed.next())
            got = self.presented.wait_next(self.app, FRONT_TIMEOUT)
            if got is not None:
                waits.append(got)
        self.report["live_serialized"] = stats(waits)
        # Free-running: submit at 200 Hz for 3 s, count presented fronts.
        start = time.perf_counter()
        submitted = 0
        baseline = self.presented.count
        next_submit = start
        while time.perf_counter() - start < 3.0:
            now = time.perf_counter()
            if now >= next_submit:
                self.host.update_data(self.feed.next())
                submitted += 1
                next_submit = now + 0.005
            self.app.processEvents()
            self.presented.poll()
            time.sleep(0.0003)
        elapsed = time.perf_counter() - start
        pump(self.app, 0.4)
        self.presented.poll()
        shown = self.presented.count - baseline
        self.report["live_free_running"] = {
            "submitted_hz": round(submitted / elapsed, 1),
            "presented_hz": round(shown / elapsed, 1),
        }

    # ---------------------------------------------------------- interactions
    def _serialized_moves(self, positions, presses=None) -> dict:
        """Send each move, wait for its front; returns latency stats."""

        waits, missed = [], 0
        for nx, ny in positions:
            before = self.presented.count
            self.pointer.move(nx, ny)
            got = self.presented.wait_next(self.app, FRONT_TIMEOUT)
            if got is None:
                missed += 1
            else:
                waits.append(got)
        result = stats(waits)
        if missed:
            result["no_front"] = missed
        return result

    def _spray_moves(self, positions, seconds: float) -> dict:
        """Send moves continuously; count presented fronts (preview Hz)."""

        baseline = self.presented.count
        start = time.perf_counter()
        index = 0
        sent = 0
        while time.perf_counter() - start < seconds:
            nx, ny = positions[index % len(positions)]
            index += 1
            self.pointer.move(nx, ny)
            sent += 1
            self.app.processEvents()
            self.presented.poll()
            time.sleep(0.003)
        elapsed = time.perf_counter() - start
        pump(self.app, 0.3)
        self.presented.poll()
        return {
            "sent_hz": round(sent / elapsed, 1),
            "presented_hz": round((self.presented.count - baseline) / elapsed, 1),
        }

    def bench_interactions(self, *, live: bool = False) -> None:
        front = self.front()
        if front is None:
            return
        tags = self.case.interactions
        out = self.report.setdefault("interactions", {})
        for tag in tags:
            try:
                handler = getattr(self, f"_do_{tag}")
                out[tag] = handler()
            except Exception as error:
                out[tag] = {"error": f"{type(error).__name__}: {error}"}

    def _main_axis(self):
        front = self.front()
        for role in ("main", "curve", "image", "histogram", "rolling"):
            found = axis_by_role(front, role)
            if found is not None:
                return found
        # Facet overview: use the first cell.
        for item in front.interaction.axes:
            if item.role.startswith("facet"):
                return item
        return front.interaction.axes[0]

    def _do_hover_series(self) -> dict:
        axis = self._main_axis()
        try:
            on = _curve_on_point(axis)
        except Exception:
            on = axis_center(axis)
        off = _inside(axis, 0.06, 0.08)
        sequence = [on, off] * 6
        serialized = self._serialized_moves(sequence)
        spray = self._spray_moves([on, off], 1.5)
        self.pointer.move(*off)
        pump(self.app, 0.2)
        return {"serialized": serialized, "spray": spray}

    def _do_click_series(self) -> dict:
        axis = self._main_axis()
        try:
            on = _curve_on_point(axis)
        except Exception:
            on = axis_center(axis)
        off = _inside(axis, 0.06, 0.08)
        waits, missed = [], 0
        for target in [on, off] * 4:
            self.pointer.press(*target)
            self.pointer.release(*target)
            got = self.presented.wait_next(self.app, FRONT_TIMEOUT)
            waits.append(got) if got is not None else None
            if got is None:
                missed += 1
        result = stats(waits)
        if missed:
            result["no_front"] = missed
        return result

    def _do_click_main(self) -> dict:
        return self._do_click_series()

    def _drag(self, axis, path) -> dict:
        start = path[0]
        self.pointer.press(*start)
        self.presented.wait_next(self.app, 2.0)
        serialized = self._serialized_moves(path[1:])
        # Free-running preview inside the same drag.
        spray = self._spray_moves(path[1:], 1.5)
        self.pointer.release(*path[-1])
        self.presented.wait_next(self.app, 2.0)
        pump(self.app, 0.2)
        return {"serialized": serialized, "spray": spray}

    def _do_drag_main(self) -> dict:
        axis = self._main_axis()
        path = [
            _inside(axis, 0.3 + 0.05 * i, 0.45 + 0.02 * i) for i in range(9)
        ]
        result = self._drag(axis, path)
        front = self.front()
        result["selectors_after"] = [
            str(state.kind.value) for state in front.interaction.selectors
        ]
        return result

    def _do_drag_colorbar(self) -> dict:
        front = self.front()
        axis = axis_by_role(front, "colorbar")
        if axis is None:
            return {"error": "no colorbar axis"}
        path = [_inside(axis, 0.5, 0.75 - 0.05 * i) for i in range(9)]
        return self._drag(axis, path)

    def _do_wheel_main(self) -> dict:
        axis = self._main_axis()
        center = axis_center(axis)
        waits, missed = [], 0
        for step in (1, 1, -1, -1, 1, -1):
            self.pointer.wheel(*center, step)
            got = self.presented.wait_next(self.app, FRONT_TIMEOUT)
            waits.append(got) if got is not None else None
            if got is None:
                missed += 1
        result = stats(waits)
        if missed:
            result["no_front"] = missed
        return result

    def _do_dclick_cell(self) -> dict:
        front = self.front()
        cells = [
            item
            for item in front.interaction.axes
            if item.role.startswith("facet")
        ]
        if not cells:
            return {"error": "no facet cells"}
        target = axis_center(cells[len(cells) // 2])
        waits, missed = [], 0
        for _ in range(4):
            self.pointer.dclick(*target)
            got = self.presented.wait_next(self.app, FRONT_TIMEOUT)
            waits.append(got) if got is not None else None
            if got is None:
                missed += 1
            pump(self.app, 0.2)
            self.presented.poll()
        result = stats(waits)
        if missed:
            result["no_front"] = missed
        return result

    # -------------------------------------------------------- live + drag
    def bench_live_drag(self) -> None:
        if "drag_main" not in self.case.interactions:
            return
        axis = self._main_axis()
        path = [
            _inside(axis, 0.3 + 0.04 * i, 0.4 + 0.02 * (i % 3))
            for i in range(12)
        ]
        self.pointer.press(*path[0])
        pump(self.app, 0.1)
        baseline = self.presented.count
        start = time.perf_counter()
        submitted = 0
        index = 0
        next_submit = start
        while time.perf_counter() - start < 3.0:
            now = time.perf_counter()
            if now >= next_submit:
                self.host.update_data(self.feed.next())
                submitted += 1
                next_submit = now + 0.02
            self.pointer.move(*path[index % len(path)])
            index += 1
            self.app.processEvents()
            self.presented.poll()
            time.sleep(0.002)
        elapsed = time.perf_counter() - start
        self.pointer.release(*path[-1])
        pump(self.app, 0.4)
        self.presented.poll()
        self.report["live_plus_drag"] = {
            "submitted_hz": round(submitted / elapsed, 1),
            "presented_hz": round(
                (self.presented.count - baseline) / elapsed, 1
            ),
        }

    def run(self) -> dict:
        self.start()
        try:
            self.bench_live()
            self.bench_interactions()
            self.bench_live_drag()
        finally:
            self.close()
        return self.report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--label", default="host")
    arguments = parser.parse_args()
    results = []
    for case in catalog():
        if arguments.only and arguments.only not in case.name:
            continue
        print(f"=== {case.name} ===", flush=True)
        try:
            report = HostBench(case).run()
        except Exception:
            report = {"case": case.name, "fatal": traceback.format_exc()}
        results.append(report)
        print(report, flush=True)
    path = write_result({"results": results}, arguments.label)
    print("written:", path)


if __name__ == "__main__":
    main()
