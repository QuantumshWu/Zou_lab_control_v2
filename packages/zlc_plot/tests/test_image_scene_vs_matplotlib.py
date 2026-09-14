"""The image scene kernel against matplotlib's own nearest ``imshow``, byte
for byte.

An image on screen is a nearest resample of its array through Agg, in
1/256 pixel fixed point along each row through matplotlib's own image
transform, blitted at its box's corner rounded half up, clipped as Agg
clips it, and coloured through ``Normalize`` and the colormap's 256 slots
in the array's promoted dtype.  The kernel that paints a prepared image
scene reproduces every one of those steps, and this is where that is held
to:
random arrays, extents (either way round), views (whole, zoomed in, zoomed
out), fractional axes boxes, both origins and three dtypes, each drawn by
matplotlib and by the kernel, must not differ in one pixel.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib import colormaps  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from zlc_plot import _raster_kernels as kernels  # noqa: E402
from zlc_plot.rendering import MatplotlibRenderer, _normalize_arithmetic  # noqa: E402

WIDTH, HEIGHT = 240, 200


def _scene(rng, dtype):
    rows = int(rng.integers(2, 40))
    columns = int(rng.integers(2, 40))
    if dtype == np.uint16:
        values = rng.integers(0, 5000, (rows, columns)).astype(dtype)
        vmin, vmax = float(rng.integers(0, 800)), float(rng.integers(3500, 5200))
    else:
        values = (rng.random((rows, columns)) * 100.0).astype(dtype)
        vmin, vmax = float(rng.random() * 15.0), float(60.0 + rng.random() * 50.0)
    valid = rng.random((rows, columns)) > 0.08
    left, right = sorted(rng.uniform(-3.0, 3.0, 2))
    bottom, top = sorted(rng.uniform(-3.0, 3.0, 2))
    if rng.random() < 0.3:
        left, right = right, left
    if rng.random() < 0.3:
        bottom, top = top, bottom
    extent = (float(left), float(right), float(bottom), float(top))
    x_low, x_high = min(left, right), max(left, right)
    y_low, y_high = min(bottom, top), max(bottom, top)
    kind = rng.integers(0, 3)
    if kind == 0:  # the whole picture fills the view
        xlim, ylim = (x_low, x_high), (y_low, y_high)
    elif kind == 1:  # zoomed in: a view inside the picture
        xlim = tuple(sorted(rng.uniform(x_low, x_high, 2)))
        ylim = tuple(sorted(rng.uniform(y_low, y_high, 2)))
    else:  # zoomed out: the picture inside the view
        xlim = (x_low - rng.uniform(0.0, 2.0), x_high + rng.uniform(0.0, 2.0))
        ylim = (y_low - rng.uniform(0.0, 2.0), y_high + rng.uniform(0.0, 2.0))
    if xlim[0] == xlim[1] or ylim[0] == ylim[1]:
        xlim, ylim = (x_low, x_high), (y_low, y_high)
    if rng.random() < 0.5:
        ylim = (ylim[1], ylim[0])
    if rng.random() < 0.2:
        xlim = (xlim[1], xlim[0])
    box = (
        float(rng.uniform(4.0, 60.0)),
        float(rng.uniform(4.0, 50.0)),
        float(rng.uniform(40.0, 160.0)),
        float(rng.uniform(40.0, 130.0)),
    )
    origin = "upper" if rng.random() < 0.5 else "lower"
    return values, valid, extent, xlim, ylim, box, origin, vmin, vmax


def _figure(box):
    figure = Figure(figsize=(WIDTH / 100.0, HEIGHT / 100.0), dpi=100)
    FigureCanvasAgg(figure)
    figure.patch.set_facecolor("white")
    left, bottom, width, height = box
    axes = figure.add_axes((left / WIDTH, bottom / HEIGHT, width / WIDTH, height / HEIGHT))
    axes.set_axis_off()
    axes.patch.set_visible(False)
    return figure, axes


@pytest.mark.parametrize("dtype", (np.uint16, np.float32, np.float64))
@pytest.mark.parametrize("seed", range(8))
def test_the_scene_paints_what_imshow_paints(dtype, seed: int) -> None:
    rng = np.random.default_rng(seed * 7 + {np.uint16: 1, np.float32: 2, np.float64: 3}[dtype])
    values, valid, extent, xlim, ylim, box, origin, vmin, vmax = _scene(rng, dtype)
    cmap = colormaps["viridis"]

    figure, axes = _figure(box)
    shown = np.ma.array(values, mask=~valid)
    axes.imshow(
        shown,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin=origin,
        extent=extent,
        aspect="auto",
    )
    axes.set_xlim(*xlim)
    axes.set_ylim(*ylim)
    figure.canvas.draw()
    reference = np.array(figure.canvas.buffer_rgba(), copy=True)

    geometry = MatplotlibRenderer._image_scene_geometry(
        axes, extent, values.shape[0], values.shape[1], HEIGHT, origin == "upper"
    )
    canvas = np.full((HEIGHT, WIDTH, 4), 255, dtype=np.uint8)
    if geometry is not None:
        blit, clip, affine = geometry
        lut = cmap((np.arange(256, dtype=float) + 0.5) / 256.0, bytes=True)
        single, vmin32, span32, vmin64, span64 = _normalize_arithmetic(values.dtype, vmin, vmax)
        kernels.raster_prepared_images(
            kernels.readable(values[np.newaxis]),
            kernels.readable(valid[np.newaxis]),
            True,
            kernels.readable(np.asarray([blit], dtype=np.int32)),
            kernels.readable(np.asarray([clip], dtype=np.int32)),
            kernels.readable(np.asarray([affine], dtype=np.float64)),
            kernels.readable(np.asarray(lut, dtype=np.uint8)),
            vmin32,
            span32,
            vmin64,
            span64,
            single,
            canvas,
        )
    differing = np.any(reference[..., :3] != canvas[..., :3], axis=-1)
    count = int(differing.sum())
    if count:
        ys, xs = np.nonzero(differing)
        detail = (
            f"rows {ys.min()}-{ys.max()} cols {xs.min()}-{xs.max()}; "
            f"first {reference[ys[0], xs[0], :3]} vs {canvas[ys[0], xs[0], :3]}; "
            f"extent {extent} xlim {xlim} ylim {ylim} box {box} origin {origin} "
            f"shape {values.shape} geometry {geometry}"
        )
    else:
        detail = ""
    assert count == 0, f"{count} pixels differ from imshow: {detail}"
