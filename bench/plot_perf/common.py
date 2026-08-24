"""Shared plumbing for the plot performance matrix.

Everything here measures the worktree copy of the packages: the bootstrap
prepends this worktree's package sources to sys.path before any zlc import.
Layer conventions:

* "session" numbers time PlotSession.update_data alone (projection + Agg
  render), the pure compute cost.
* "host" numbers run the full user-facing pipeline: RasterPlotHost worker,
  RGBA capture, front promotion, Qt widget presentation (QImage), and
  pointer gestures synthesized as real QMouseEvents on the widget.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def bootstrap() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    for pkg in ("zlc_plot", "zlc_data"):
        path = str(ROOT / "packages" / pkg / "src")
        if path not in sys.path:
            sys.path.insert(0, path)
    tests = str(ROOT / "packages" / "zlc_plot" / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)


bootstrap()

import numpy as np  # noqa: E402

from data_factory import (  # noqa: E402
    Axis,
    DatasetSchema,
    DatasetSnapshot,
    PointTable,
    PointTopology,
)
from zlc_data import AxisId  # noqa: E402


# ---------------------------------------------------------------- datasets
class SnapshotFeed:
    """Pre-generated value buffers cycled with fresh revisions.

    Generating 20M random floats per revision would dominate the timing;
    the feed pays that cost once and hands out snapshots at buffer-swap
    cost, the way a producer hands the panel already-materialized blocks.
    """

    def __init__(self, schema: DatasetSchema, buffers: list[np.ndarray]):
        self.schema = schema
        self._buffers = buffers
        self._revision = 0

    @property
    def size(self) -> int:
        return int(np.prod(self._buffers[0].shape))

    def next(self) -> DatasetSnapshot:
        self._revision += 1
        buffer = self._buffers[self._revision % len(self._buffers)]
        return DatasetSnapshot(self.schema, buffer, revision=self._revision)


def lattice_feed(
    *,
    repeats: int = 20,
    rows: int = 1000,
    frames: int = 3,
    sites: int = 34,
    dims: tuple[int, ...] = (10, 10, 10),
    buffers: int = 3,
    seed: int = 0,
) -> SnapshotFeed:
    """The bench-shaped signal: (R)x(rows)x(F)x(S) with a scan topology.

    Values carry a gaussian profile over the ``ax`` coordinate so curves
    have shape (hover targets exist away from the mean line) and fits have
    something to converge on.
    """

    assert int(np.prod(dims)) == rows
    rng = np.random.default_rng(seed)
    cells = []
    for i in range(rows):
        remainder = i
        cell = []
        for size in dims:
            cell.append(remainder % size)
            remainder //= size
        cells.append(tuple(cell))
    names = ("ax", "ay", "az")[: len(dims)]
    columns = {
        name: np.asarray([float(cell[j]) for cell in cells])
        for j, name in enumerate(names)
    }
    schema = DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        PointTable.from_columns(columns),
        data_axes=(
            Axis.create("frame", values=[float(i) for i in range(frames)]),
            Axis.create("site", values=[float(i) for i in range(sites)]),
        ),
        dtype=np.float64,
        generation="bench",
        point_topology=PointTopology(
            tuple(AxisId(name) for name in names),
            tuple(tuple(float(v) for v in range(size)) for size in dims),
            tuple(cells),
        ),
    )
    ax_column = columns["ax"]
    profile = np.exp(-((ax_column - (dims[0] - 1) / 2.0) ** 2) / 4.0)
    shape = (repeats, rows, frames, sites)
    stack = [
        profile[None, :, None, None] + rng.normal(scale=0.25, size=shape)
        for _ in range(buffers)
    ]
    return SnapshotFeed(schema, stack)


def camera_feed(
    *, height: int = 2048, width: int = 2048, buffers: int = 3, seed: int = 1
) -> SnapshotFeed:
    """A camera-monitor-shaped dense image: (1, 1, H, W) uint16 frames."""

    rng = np.random.default_rng(seed)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"shot": np.asarray([0.0])}),
        data_axes=(
            Axis.create("y", values=[float(i) for i in range(height)]),
            Axis.create("x", values=[float(i) for i in range(width)]),
        ),
        dtype=np.uint16,
        generation="bench-camera",
    )
    yy, xx = np.mgrid[0:height, 0:width]
    blob = 3000.0 * np.exp(
        -(((yy - height / 2) ** 2) + ((xx - width / 2) ** 2))
        / (2 * (height / 8.0) ** 2)
    )
    stack = [
        np.clip(
            blob + rng.normal(scale=120.0, size=(height, width)) + 400.0,
            0,
            65535,
        ).astype(np.uint16)[None, None]
        for _ in range(buffers)
    ]
    return SnapshotFeed(schema, stack)


# ------------------------------------------------------------ Qt plumbing
def qt_env():
    """Import Qt lazily so session-layer runs never touch it."""

    from zlc_plot import ensure_qt5_application

    app = ensure_qt5_application([])
    from PyQt5 import QtCore, QtGui, QtWidgets

    return app, QtCore, QtGui, QtWidgets


def pump(app, seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.0005)


def pump_until(app, predicate, timeout: float) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.0005)
    return bool(predicate())


class Presented:
    """Counts widget-presented fronts (the pixels the user actually sees)."""

    def __init__(self, widget):
        self._widget = widget
        self.count = 0
        self.stamps: list[float] = []
        self._last = None

    def poll(self) -> bool:
        front = self._widget.presented_front
        if front is not None and front is not self._last:
            self._last = front
            self.count += 1
            self.stamps.append(time.perf_counter())
            return True
        return False

    def wait_next(self, app, timeout: float) -> float | None:
        """Pump until one more front is presented; return the wait or None."""

        start = time.perf_counter()
        baseline = self.count
        deadline = start + timeout
        while time.perf_counter() < deadline:
            app.processEvents()
            self.poll()
            if self.count > baseline:
                return time.perf_counter() - start
            time.sleep(0.0003)
        return None


class Pointer:
    """Real QMouseEvents in widget-logical coordinates."""

    def __init__(self, widget, app, QtCore, QtGui):
        self._widget = widget
        self._app = app
        self._QtCore = QtCore
        self._QtGui = QtGui
        self._buttons = QtCore.Qt.NoButton

    def _point(self, nx: float, ny: float):
        return self._QtCore.QPointF(
            nx * self._widget.width(), ny * self._widget.height()
        )

    def _send(self, kind, nx, ny, button):
        QtCore, QtGui = self._QtCore, self._QtGui
        event = QtGui.QMouseEvent(
            kind,
            self._point(nx, ny),
            button,
            self._buttons,
            QtCore.Qt.NoModifier,
        )
        self._app.sendEvent(self._widget, event)

    def press(self, nx, ny, button=None):
        QtCore = self._QtCore
        chosen = button or QtCore.Qt.LeftButton
        self._buttons = self._buttons | chosen
        self._send(self._QtGui.QMouseEvent.MouseButtonPress, nx, ny, chosen)

    def move(self, nx, ny):
        self._send(
            self._QtGui.QMouseEvent.MouseMove, nx, ny, self._QtCore.Qt.NoButton
        )

    def release(self, nx, ny, button=None):
        QtCore = self._QtCore
        chosen = button or QtCore.Qt.LeftButton
        self._buttons = self._buttons & ~chosen
        self._send(self._QtGui.QMouseEvent.MouseButtonRelease, nx, ny, chosen)

    def dclick(self, nx, ny):
        QtCore = self._QtCore
        self.press(nx, ny)
        self.release(nx, ny)
        self._buttons = self._buttons | QtCore.Qt.LeftButton
        self._send(
            self._QtGui.QMouseEvent.MouseButtonDblClick,
            nx,
            ny,
            QtCore.Qt.LeftButton,
        )
        self.release(nx, ny)

    def wheel(self, nx, ny, steps: int):
        QtCore, QtGui = self._QtCore, self._QtGui
        event = QtGui.QWheelEvent(
            self._point(nx, ny),
            self._widget.mapToGlobal(self._point(nx, ny).toPoint()),
            QtCore.QPoint(),
            QtCore.QPoint(0, steps * 120),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
            QtCore.Qt.ScrollUpdate,
            False,
        )
        self._app.sendEvent(self._widget, event)


def axis_by_role(front, role: str, cell: int | None = None):
    for item in front.interaction.axes:
        if item.role == role and (cell is None or item.cell_index == cell):
            return item
    return None


def axis_center(transform) -> tuple[float, float]:
    left, top, right, bottom = transform.bounds
    return (left + right) / 2.0, (top + bottom) / 2.0


def stats(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0}
    array = np.asarray(samples, dtype=float)
    return {
        "n": int(array.size),
        "median_ms": round(float(np.median(array)) * 1e3, 2),
        "p90_ms": round(float(np.percentile(array, 90)) * 1e3, 2),
        "best_ms": round(float(array.min()) * 1e3, 2),
    }


def write_result(payload: dict, label: str) -> pathlib.Path:
    out = ROOT / "bench" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{label}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
