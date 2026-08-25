"""End-to-end probe: the height-bar scene through the real host+widget."""
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench.plot_perf.common import (  # noqa: E402
    Pointer, Presented, axis_by_role, pump, pump_until, qt_env,
)
import numpy as np  # noqa: E402
from data_factory import (  # noqa: E402
    Axis, DatasetSchema, DatasetSnapshot, PointTable, PointTopology,
)
from zlc_data import AxisId  # noqa: E402
from zlc_plot import AxisRef, ImagePlot, Qt5PlotWidget  # noqa: E402
from zlc_plot.raster import RasterPlotHost  # noqa: E402
from PIL import Image  # noqa: E402

OUT = pathlib.Path(
    r"C:\Users\eadri\AppData\Local\Temp\claude"
    r"\C--Users-eadri-Dropbox-WorkCode-Github-Zou-lab-control-v1-claude"
    r"\9df53e90-97f8-4e66-ae86-8894c910f2c2\scratchpad"
)

app, QtCore, QtGui, _ = qt_env()
rows = 100
cells = [(i % 10, i // 10) for i in range(rows)]
schema = DatasetSchema.create(
    Axis.create("repeat", size=5),
    PointTable.from_columns({
        "ax": np.asarray([float(c[0]) for c in cells]),
        "ay": np.asarray([float(c[1]) for c in cells]),
    }),
    data_axes=(Axis.create("site", values=[0.0, 1.0]),),
    dtype=np.float64,
    point_topology=PointTopology(
        (AxisId("ax"), AxisId("ay")),
        (tuple(float(i) for i in range(10)),) * 2,
        tuple(cells),
    ),
)
rng = np.random.default_rng(0)
xx, yy = np.meshgrid(np.arange(10), np.arange(10))
profile = np.exp(-((xx - 4.5) ** 2 + (yy - 4.5) ** 2) / 8.0)
values = profile.reshape(-1)[None, :, None] + rng.normal(
    scale=0.05, size=(5, rows, 2)
)
host = RasterPlotHost.from_plot(
    DatasetSnapshot(schema, values, revision=1),
    ImagePlot(AxisRef.point_dimension("ax"), AxisRef.point_dimension("ay")),
)
host.configure(size="4x4")
widget = Qt5PlotWidget(host)
widget.show()
presented = Presented(widget)
pump_until(app, lambda: presented.poll() or presented.count > 0, 30.0)
pump(app, 0.4); presented.poll()

# install an area selector FIRST so we can check it survives the 3D trip
from zlc_plot.selectors import SelectorKind, SelectorState, RectangleRange, NumericRange
done = host.configure(selectors=(
    SelectorState(SelectorKind.AREA, RectangleRange(NumericRange(2.0, 6.0), NumericRange(2.0, 6.0))),
))
pump_until(app, done.done, 10.0); pump(app, 0.4); presented.poll()
before_kinds = [s.kind.value for s in widget.presented_front.interaction.selectors]

t0 = time.perf_counter()
done = host.set_parameter("presentation", "height_bars")
pump_until(app, done.done, 10.0)
presented.wait_next(app, 6.0)
print(f"switch to 3D: {(time.perf_counter()-t0)*1e3:.0f} ms")
pump(app, 0.4); presented.poll()
front = widget.presented_front
print("selectors painted in 3D:", [s.kind.value for s in front.interaction.selectors])

image_axis = axis_by_role(front, "image")
pointer = Pointer(widget, app, QtCore, QtGui)
cx = (image_axis.bounds[0] + image_axis.bounds[2]) / 2
cy = (image_axis.bounds[1] + image_axis.bounds[3]) / 2

# ---- orbit drag: press, timed moves, release
pointer.press(cx, cy)
pump(app, 0.2)
waits = []
for i in range(8):
    n0 = presented.count
    pointer.move(cx + 0.02 * (i + 1), cy + 0.01 * (i + 1))
    got = presented.wait_next(app, 6.0)
    if got is not None:
        waits.append(got * 1e3)
pointer.release(cx + 0.16, cy + 0.08)
pump(app, 0.6); presented.poll()
import statistics
print(f"orbit move latency: median {statistics.median(waits):.1f} ms over {len(waits)} moves")
front = widget.presented_front
print("selectors after drag:", [s.kind.value for s in front.interaction.selectors])

# camera should have committed
import concurrent.futures
description = host.describe_display().result(timeout=10).value
params = {name: description.display_state[name] for name in ("camera_azimuth", "camera_elevation", "camera_zoom")}
print("camera after drag:", {k: round(float(params[k]), 1) for k in ("camera_azimuth", "camera_elevation", "camera_zoom")})

# ---- wheel zoom
n0 = presented.count
pointer.wheel(cx, cy, 2)
presented.wait_next(app, 6.0)
pump(app, 0.4)
description = host.describe_display().result(timeout=10).value
print("zoom after wheel:", round(float(description.display_state["camera_zoom"]), 3))

Image.fromarray(widget.presented_front.buffer.as_rgba()).save(OUT / "h3d_host.png")

# ---- back to heatmap: selector must return
t0 = time.perf_counter()
done = host.set_parameter("presentation", "heatmap")
pump_until(app, done.done, 10.0)
presented.wait_next(app, 6.0)
pump(app, 0.4); presented.poll()
front = widget.presented_front
after_kinds = [s.kind.value for s in front.interaction.selectors]
print("selectors back on heatmap:", before_kinds, "->", after_kinds)
Image.fromarray(front.buffer.as_rgba()).save(OUT / "h3d_host_back.png")

# ---- live update rate in 3D
done = host.set_parameter("presentation", "height_bars")
pump_until(app, done.done, 10.0); presented.wait_next(app, 6.0); pump(app, 0.3)
revision = [1]
def snap():
    revision[0] += 1
    fresh = profile.reshape(-1)[None, :, None] + rng.normal(scale=0.05, size=(5, rows, 2))
    return DatasetSnapshot(schema, fresh, revision=revision[0])
waits = []
for _ in range(10):
    host.update_data(snap())
    got = presented.wait_next(app, 6.0)
    if got is not None:
        waits.append(got * 1e3)
print(f"live revision in 3D: median {statistics.median(waits):.1f} ms")

widget.close_adapter(); host.close()
print("OK")
