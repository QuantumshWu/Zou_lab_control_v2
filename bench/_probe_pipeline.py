"""Smoke probe: worktree imports, offscreen Qt host, pointer roundtrip."""
import os, sys, time, pathlib
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = pathlib.Path(__file__).resolve().parents[1]
for pkg in ("zlc_plot", "zlc_data"):
    sys.path.insert(0, str(ROOT / "packages" / pkg / "src"))
sys.path.insert(0, str(ROOT / "packages" / "zlc_plot" / "tests"))

import numpy as np
import zlc_plot, zlc_data
print("zlc_plot from", zlc_plot.__file__)

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable, PointTopology
from zlc_data import AxisId
from zlc_plot import AxisRef, CurvePlot, Qt5PlotWidget, ensure_qt5_application
from zlc_plot.raster import RasterPlotHost

app = ensure_qt5_application([])
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtTest import QTest

R, P, F, S = 20, 1000, 3, 34
rng = np.random.default_rng(0)
cells = [(i % 10, (i // 10) % 10, i // 100) for i in range(P)]
schema = DatasetSchema.create(
    Axis.create("repeat", size=R),
    PointTable.from_columns({
        "ax": np.asarray([float(c[0]) for c in cells]),
        "ay": np.asarray([float(c[1]) for c in cells]),
        "az": np.asarray([float(c[2]) for c in cells]),
    }),
    data_axes=(Axis.create("frame", values=[0.0, 1.0, 2.0]),
               Axis.create("site", values=[float(i) for i in range(S)])),
    dtype=np.float64, generation="bench",
    point_topology=PointTopology(
        (AxisId("ax"), AxisId("ay"), AxisId("az")),
        (tuple(float(i) for i in range(10)),) * 3,
        tuple((c[0], c[1], c[2]) for c in cells),
    ),
)
rev = [0]
def snap():
    rev[0] += 1
    return DatasetSnapshot(schema, rng.random((R, P, F, S)), revision=rev[0])

t0 = time.perf_counter()
host = RasterPlotHost.from_plot(snap(), CurvePlot(AxisRef.point("ax")))
widget = Qt5PlotWidget(host)
widget.show()
fronts = []
host.subscribe_front(lambda f: fronts.append(time.perf_counter()))
deadline = time.monotonic() + 15
while widget.presented_front is None and time.monotonic() < deadline:
    app.processEvents(); time.sleep(0.002)
t_first = time.perf_counter() - t0
print(f"first front: {t_first*1e3:.0f} ms; widget size {widget.width()}x{widget.height()}")
front = widget.presented_front
print("front type:", type(front).__name__, "identity:", front.identity.host_id[:8])

# one live update through the host
n0 = len(fronts)
t0 = time.perf_counter()
fut = host.update_data(snap())
while len(fronts) == n0 and time.monotonic() < time.monotonic() + 0.0 or (len(fronts) == n0 and (time.perf_counter() - t0) < 10):
    app.processEvents(); time.sleep(0.001)
print(f"live update -> new front: {(time.perf_counter()-t0)*1e3:.1f} ms")

# pointer hover roundtrip
n0 = len(fronts)
t0 = time.perf_counter()
QTest.mouseMove(widget, QPoint(widget.width() // 2, widget.height() // 2))
while len(fronts) == n0 and (time.perf_counter() - t0) < 5:
    app.processEvents(); time.sleep(0.001)
got = len(fronts) > n0
print(f"hover -> {'front' if got else 'NO front'} in {(time.perf_counter()-t0)*1e3:.1f} ms")

widget.close_adapter(); host.close()
print("OK")
