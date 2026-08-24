"""Full-pipeline benchmark: RasterPlotHost + Qt widget + real pointer events.

Per case it measures:

* first_render  -- host construction to the first presented front
* live          -- serialized per-revision latency (capacity) and a
                   25 Hz producer-tracking run inside a real Qt event loop
* interactions  -- per-gesture serialized latency (each event waits for its
                   presented front) and, for drags, free-running preview Hz
* live+drag     -- presented Hz while data streams and a drag is active
* fit           -- live-fit behaviour per revision (when the case asks)

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
PRODUCER_HZ = 25.0


def _curve_on_point(transform, dims0: int = 10):
    """A display point ON the mean curve of the lattice profile."""

    peak = (dims0 - 1) / 2.0
    x = float(math.floor(peak))
    y = math.exp(-((x - peak) ** 2) / 4.0)
    return transform.display_to_normalized(x, y)


def _inside(transform, fx: float, fy: float):
    left, top, right, bottom = transform.bounds
    return left + fx * (right - left), top + fy * (bottom - top)


def _value_point(transform, value: float, fx: float = 0.5):
    """Aim at a VALUE on a vertical distribution axis (value along y)."""

    left, top, right, bottom = transform.bounds
    y0, y1 = transform.y_limits
    ty = (value - y1) / (y0 - y1)
    return left + fx * (right - left), top + ty * (bottom - top)


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
        self.report["first_render_ms"] = round(first * 1e3, 1) if ok else None
        self.settle()
        self.pointer = Pointer(self.widget, self.app, self.QtCore, self.QtGui)
        if self.case.fit:
            done = self.host.configure(fit=dict(self.case.fit))
            pump_until(self.app, done.done, timeout=20.0)
            self.settle()

    def settle(self, quiet: float = 0.35, limit: float = 8.0) -> None:
        """Pump until no new front has been presented for ``quiet`` seconds."""

        deadline = time.perf_counter() + limit
        last_change = time.perf_counter()
        baseline = self.presented.count
        while time.perf_counter() < deadline:
            self.app.processEvents()
            if self.presented.poll():
                last_change = time.perf_counter()
            if time.perf_counter() - last_change >= quiet:
                return
            time.sleep(0.001)

    def close(self) -> None:
        try:
            self.widget.close_adapter()
        finally:
            self.host.close()

    def front(self):
        return self.widget.presented_front

    # ------------------------------------------------------------ live feed
    def bench_live(self) -> None:
        for _ in range(2):
            self.host.update_data(self.feed.next())
            self.presented.wait_next(self.app, FRONT_TIMEOUT)
        waits = []
        for _ in range(12):
            self.host.update_data(self.feed.next())
            got = self.presented.wait_next(self.app, FRONT_TIMEOUT)
            if got is not None:
                waits.append(got)
        self.report["live_serialized"] = stats(waits)
        # Producer-tracking: a 25 Hz feed inside a real Qt event loop.
        QtCore = self.QtCore
        state = {"submitted": 0}
        feed_timer = QtCore.QTimer()
        feed_timer.setInterval(max(1, int(1000 / PRODUCER_HZ)))

        def tick():
            self.host.update_data(self.feed.next())
            state["submitted"] += 1

        feed_timer.timeout.connect(tick)
        poller = QtCore.QTimer()
        poller.setInterval(1)
        poller.timeout.connect(self.presented.poll)
        stop = QtCore.QTimer()
        stop.setSingleShot(True)
        stop.setInterval(3000)
        stop.timeout.connect(self.app.quit)
        baseline = self.presented.count
        start = time.perf_counter()
        feed_timer.start(); poller.start(); stop.start()
        self.app.exec_()
        feed_timer.stop(); poller.stop()
        elapsed = time.perf_counter() - start
        self.settle()
        shown = self.presented.count - baseline
        self.report["live_25hz"] = {
            "submitted_hz": round(state["submitted"] / elapsed, 1),
            "presented_hz": round(shown / elapsed, 1),
        }

    # ---------------------------------------------------------- interactions
    def _serialized_moves(self, positions) -> dict:
        waits, missed = [], 0
        for nx, ny in positions:
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

    def bench_interactions(self) -> None:
        front = self.front()
        if front is None:
            return
        out = self.report.setdefault("interactions", {})
        for tag in self.case.interactions:
            try:
                self.settle()
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
        for item in front.interaction.axes:
            if item.role.startswith("facet"):
                return item
        return front.interaction.axes[0]

    def _find_hover_target(self, axis):
        """A pointer position that actually changes pixels, found by probing."""

        try:
            candidate = _curve_on_point(axis)
        except Exception:
            candidate = axis_center(axis)
        off = _inside(axis, 0.04, 0.06)
        probes = [candidate] + [
            _inside(axis, fx, fy)
            for fy in (0.5, 0.35, 0.65, 0.2, 0.8)
            for fx in (0.5, 0.3, 0.7)
        ]
        for position in probes:
            self.pointer.move(*position)
            got = self.presented.wait_next(self.app, 1.0)
            if got is not None:
                self.pointer.move(*off)
                self.presented.wait_next(self.app, 1.0)
                return position, off
        return None, off

    def _do_hover_series(self) -> dict:
        axis = self._main_axis()
        on, off = self._find_hover_target(axis)
        if on is None:
            return {"error": "no hover-responsive position found"}
        serialized = self._serialized_moves([on, off] * 6)
        spray = self._spray_moves([on, off], 1.5)
        self.pointer.move(*off)
        pump(self.app, 0.2)
        return {"serialized": serialized, "spray": spray}

    def _do_click_series(self) -> dict:
        axis = self._main_axis()
        on, off = self._find_hover_target(axis)
        if on is None:
            return {"error": "no hover-responsive position found"}
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
        axis = self._main_axis()
        positions = [
            _inside(axis, 0.4, 0.5),
            _inside(axis, 0.6, 0.5),
        ]
        waits, missed = [], 0
        for target in positions * 3:
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

    def _drag(self, path, button=None) -> dict:
        self.pointer.press(*path[0], button)
        self.presented.wait_next(self.app, 2.0)
        serialized = self._serialized_moves(path[1:])
        spray = self._spray_moves(path[1:], 1.5)
        self.pointer.release(*path[-1], button)
        self.presented.wait_next(self.app, 2.0)
        pump(self.app, 0.2)
        return {"serialized": serialized, "spray": spray}

    def _do_drag_main(self) -> dict:
        axis = self._main_axis()
        path = [
            _inside(axis, 0.3 + 0.05 * i, 0.45 + 0.02 * i) for i in range(9)
        ]
        result = self._drag(path)
        front = self.front()
        result["selectors_after"] = [
            str(state.kind.value) for state in front.interaction.selectors
        ]
        return result

    def _do_pan_drag(self) -> dict:
        axis = self._main_axis()
        path = [
            _inside(axis, 0.55 - 0.03 * i, 0.5 - 0.015 * i) for i in range(9)
        ]
        result = self._drag(path, button=self.QtCore.Qt.MiddleButton)
        # Restore the viewport for later interactions.
        self.host.dispatch(lambda: None)
        return result

    def _do_drag_clim(self) -> dict:
        front = self.front()
        axis = axis_by_role(front, "distribution")
        if axis is None:
            return {"error": "no distribution axis"}
        limits = front.interaction.color_limits
        if limits is None:
            return {"error": "no color limits on front"}
        span = limits.high - limits.low
        start = _value_point(axis, limits.high)
        path = [start] + [
            _value_point(axis, limits.high - span * 0.04 * (i + 1))
            for i in range(8)
        ]
        return self._drag(path)

    def _do_drag_threshold(self) -> dict:
        from zlc_plot.selectors import SelectorKind, SelectorState

        front = self.front()
        distribution_axis = axis_by_role(front, "distribution")
        if distribution_axis is not None:
            v0, v1 = distribution_axis.y_limits
        else:
            v0, v1 = self._main_axis().x_limits
        value = v0 + 0.5 * (v1 - v0)
        done = self.host.configure(
            selectors=(SelectorState(SelectorKind.THRESHOLD, float(value)),)
        )
        pump_until(self.app, done.done, timeout=10.0)
        self.settle()
        front = self.front()
        installed = [
            str(state.kind.value) for state in front.interaction.selectors
        ]
        if "threshold" not in installed:
            return {"error": f"threshold not installed: {installed}"}
        distribution = axis_by_role(front, "distribution")
        if distribution is not None:
            # Image kinds paint the threshold on the value distribution:
            # a horizontal guide at the value height; drag it vertically.
            start = _value_point(distribution, value)
            y0, y1 = distribution.y_limits
            span = abs(y1 - y0)
            path = [start] + [
                _value_point(distribution, value + span * 0.03 * (i + 1))
                for i in range(8)
            ]
        else:
            axis = self._main_axis()
            nx, _ = axis.display_to_normalized(value, 0.0)
            _, top, _, bottom = axis.bounds
            ny = (top + bottom) / 2.0
            step = (axis.bounds[2] - axis.bounds[0]) * 0.03
            path = [(nx, ny)] + [(nx + step * (i + 1), ny) for i in range(8)]
        result = self._drag(path)
        done = self.host.configure(selectors=())
        pump_until(self.app, done.done, timeout=10.0)
        return result

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
            self.settle()
            self.pointer.dclick(*target)
            got = self.presented.wait_next(self.app, FRONT_TIMEOUT)
            waits.append(got) if got is not None else None
            if got is None:
                missed += 1
        self.settle()
        result = stats(waits)
        if missed:
            result["no_front"] = missed
        return result

    # -------------------------------------------------------- live + drag
    def bench_live_drag(self) -> None:
        if "drag_main" not in self.case.interactions:
            return
        self.settle()
        axis = self._main_axis()
        path = [
            _inside(axis, 0.3 + 0.04 * i, 0.4 + 0.02 * (i % 3))
            for i in range(12)
        ]
        QtCore = self.QtCore
        state = {"submitted": 0, "index": 0}
        feed_timer = QtCore.QTimer()
        feed_timer.setInterval(max(1, int(1000 / PRODUCER_HZ)))

        def tick():
            self.host.update_data(self.feed.next())
            state["submitted"] += 1

        feed_timer.timeout.connect(tick)
        mover = QtCore.QTimer()
        mover.setInterval(8)

        def wiggle():
            state["index"] += 1
            self.pointer.move(*path[state["index"] % len(path)])

        mover.timeout.connect(wiggle)
        poller = QtCore.QTimer()
        poller.setInterval(1)
        poller.timeout.connect(self.presented.poll)
        stop = QtCore.QTimer()
        stop.setSingleShot(True)
        stop.setInterval(3000)
        stop.timeout.connect(self.app.quit)
        self.pointer.press(*path[0])
        pump(self.app, 0.1)
        baseline = self.presented.count
        start = time.perf_counter()
        feed_timer.start(); mover.start(); poller.start(); stop.start()
        self.app.exec_()
        feed_timer.stop(); mover.stop(); poller.stop()
        elapsed = time.perf_counter() - start
        self.pointer.release(*path[-1])
        self.settle()
        self.report["live_plus_drag"] = {
            "submitted_hz": round(state["submitted"] / elapsed, 1),
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
