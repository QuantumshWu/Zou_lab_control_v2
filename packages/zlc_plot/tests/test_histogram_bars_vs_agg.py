"""The histogram bar kernel against Agg, byte for byte.

Agg snaps an unstroked rectilinear path to the pixel grid and rounds the
clip box to whole pixels, so a histogram bar has no antialiased edge; what
is left is which pixels a bar covers and how the fill colour lands on them,
and both are exact.  The kernel must reproduce them exactly, or the live
picture of a histogram and its export would differ by a level on every bar.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from zlc_plot import _raster_kernels as kernels  # noqa: E402

WIDTH, HEIGHT = 160, 96


def _agg(bars, face, alpha, clip, background):
    """Agg's picture: bars as a PolyCollection clipped to ``clip`` (image
    coordinates: left, top, right, bottom) over a flat background."""

    figure = Figure(figsize=(WIDTH / 100.0, HEIGHT / 100.0), dpi=100)
    figure.patch.set_facecolor(background)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.set_axis_off()
    axes.set_xlim(0.0, WIDTH)
    axes.set_ylim(HEIGHT, 0.0)
    edges, tops, base = bars
    verts = [
        [(edges[i], base), (edges[i], tops[i]), (edges[i + 1], tops[i]), (edges[i + 1], base)]
        for i in range(len(tops))
    ]
    collection = PolyCollection(verts, facecolors=[face], edgecolors="none", alpha=alpha)
    axes.add_collection(collection)
    left, top, right, bottom = clip
    collection.set_clip_box(Bbox([[left, HEIGHT - bottom], [right, HEIGHT - top]]))
    canvas.draw()
    return np.array(canvas.buffer_rgba(), copy=True)


def _kernel(bars, face, alpha, clip, background):
    edges, tops, base = bars
    rgba = np.array(matplotlib.colors.to_rgba(face), dtype=float)
    rgba[3] = alpha
    colour = np.floor(rgba * 255.0 + 0.5).astype(np.uint8)[np.newaxis, :]
    target = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    target[..., :3] = np.floor(np.array(matplotlib.colors.to_rgb(background)) * 255.0 + 0.5).astype(np.uint8)
    target[..., 3] = 255
    left, top, right, bottom = clip
    snapped = np.array(
        [[np.floor(left + 0.5), np.floor(top + 0.5), np.floor(right + 0.5), np.floor(bottom + 0.5)]],
        dtype=np.int32,
    )
    kernels.raster_histogram_bars(
        kernels.readable(np.asarray(edges, dtype=np.float64)),
        kernels.readable(np.append(np.asarray(tops, dtype=np.float64), np.nan)),
        kernels.readable(np.array([base], dtype=np.float64)),
        kernels.readable(np.array([0, len(edges)], dtype=np.int64)),
        kernels.readable(colour),
        kernels.readable(snapped),
        target,
    )
    return target


def test_two_surfaces_keep_their_own_bar_tops() -> None:
    """Surfaces of different bar counts index their tops by their own edges:
    the second cell of a grid drew the first cell's heights shifted by one
    when tops were counted per bar and edges per edge."""

    rng = np.random.default_rng(5)
    first_edges = np.linspace(4.0, 70.0, 7)
    second_edges = np.linspace(90.0, 156.0, 12)
    first_tops = rng.uniform(10.0, 60.0, 6)
    second_tops = rng.uniform(10.0, 60.0, 11)
    base = 80.0
    colour = np.array([[40, 90, 200, 255]], dtype=np.uint8)
    target = np.full((HEIGHT, WIDTH, 4), 255, dtype=np.uint8)
    kernels.raster_histogram_bars(
        kernels.readable(np.concatenate([first_edges, second_edges])),
        kernels.readable(np.concatenate([np.append(first_tops, np.nan), np.append(second_tops, np.nan)])),
        kernels.readable(np.array([base, base])),
        kernels.readable(np.array([0, first_edges.size, first_edges.size + second_edges.size])),
        kernels.readable(np.repeat(colour, 2, axis=0)),
        kernels.readable(np.array([[0, 0, 80, HEIGHT], [80, 0, WIDTH, HEIGHT]], dtype=np.int32)),
        target,
    )
    for edges, tops in ((first_edges, first_tops), (second_edges, second_tops)):
        for index, top in enumerate(tops):
            column = int(np.floor((edges[index] + edges[index + 1]) / 2.0))
            painted = np.flatnonzero(target[:, column, 0] != 255)
            assert painted.size, f"bar at column {column} is missing"
            assert painted.min() == int(np.floor(top + 0.5)), (column, painted.min(), top)
            assert painted.max() == int(np.floor(base + 0.5)) - 1


@pytest.mark.parametrize("seed", range(6))
def test_bars_land_where_agg_lands_them_byte_for_byte(seed: int) -> None:
    rng = np.random.default_rng(seed)
    count = int(rng.integers(3, 24))
    edges = np.sort(rng.uniform(4.0, WIDTH - 4.0, count + 1))
    base = float(rng.uniform(HEIGHT * 0.6, HEIGHT - 3.0))
    tops = rng.uniform(3.0, base, count)
    tops[rng.random(count) < 0.2] = base  # empty bins
    face = tuple(rng.random(3))
    alpha = 1.0 if seed % 3 == 0 else float(rng.uniform(0.1, 0.95))
    background = tuple(rng.random(3)) if seed % 2 else (1.0, 1.0, 1.0)
    # a clip box with fractional edges that cuts the bars on every side
    clip = (
        float(rng.uniform(6.0, 20.0)),
        float(rng.uniform(4.0, 14.0)),
        float(rng.uniform(WIDTH - 24.0, WIDTH - 6.0)),
        float(rng.uniform(base - 6.0, base + 4.0)),
    )
    agg = _agg((edges, tops, base), face, alpha, clip, background)
    ours = _kernel((edges, tops, base), face, alpha, clip, background)
    differing = int(np.count_nonzero((agg[..., :3] != ours[..., :3]).any(axis=-1)))
    assert differing == 0, f"{differing} pixels differ from Agg (seed {seed})"
