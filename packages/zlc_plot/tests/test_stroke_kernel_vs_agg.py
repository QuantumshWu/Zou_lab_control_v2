"""The stroke kernel against Agg, which is what the export draws with.

The band tests check that the kernel's parallel bands agree with its own
serial reading; this one checks the reading itself, against an independent
renderer, on the shapes a live panel actually shows: a steep noisy trace
(a rolling scalar with a hundred shots across the box), gentle slopes, a
dense noise band, and level and vertical runs, at the device pixel ratios
the console draws at.  A stroke that loses the steep runs of a trace --
which is what a column envelope sampled at the column's centre does --
shows a hundred dots where the export shows a line.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from zlc_plot import _raster_kernels as kernels  # noqa: E402
from zlc_plot.rendering import _envelope_decimated  # noqa: E402

WIDTH, HEIGHT = 240, 160


def _agg(xs: np.ndarray, ys: np.ndarray, width_px: float, dpr: float) -> np.ndarray:
    figure = Figure(figsize=(WIDTH * dpr / 100.0, HEIGHT * dpr / 100.0), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.set_axis_off()
    axes.set_xlim(0.0, WIDTH * dpr)
    axes.set_ylim(HEIGHT * dpr, 0.0)
    axes.plot(
        xs,
        ys,
        color="black",
        linewidth=width_px * 72.0 / 100.0,
        solid_joinstyle="round",
        solid_capstyle="butt",
        antialiased=True,
    )
    canvas.draw()
    return np.array(canvas.buffer_rgba(), copy=True)[..., 0].astype(int)


def _kernel(xs: np.ndarray, ys: np.ndarray, width_px: float, dpr: float) -> np.ndarray:
    height, width = int(HEIGHT * dpr), int(WIDTH * dpr)
    target = np.full((height, width, 4), 255, np.uint8)
    vertices = np.column_stack([xs, ys]).astype(np.float64)
    kernels.raster_polylines(
        kernels.readable(vertices),
        kernels.readable(np.array([0, len(xs)], np.int64)),
        kernels.readable(np.array([[0, 0, 0, 255]], np.uint8)),
        kernels.readable(np.array([width_px], np.float64)),
        kernels.readable(np.array([[0, 0, width, height]], np.int32)),
        kernels.readable(np.array([0, 1], np.int64)),
        kernels.stroke_bands(1),
        target,
    )
    return target[..., 0].astype(int)


def _shapes(dpr: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(11)
    count = 60
    xs = np.linspace(12, WIDTH - 12, count) * dpr
    steep = (HEIGHT / 2 + 40 * rng.standard_normal(count)) * dpr
    gentle = (HEIGHT / 2 + 25 * np.sin(np.linspace(0, 2 * np.pi, count))) * dpr
    # A dense trace reaches both renderers THINNED to each pixel column's
    # extremes, which is how the panel hands it to its Line2D and to the
    # kernel alike; the stroke is then a steep zigzag, one column wide.
    raw_x = np.linspace(12, WIDTH - 12, 20000) * dpr
    raw_y = (HEIGHT / 2 + 12 * rng.standard_normal(raw_x.size)) * dpr
    dense_x, dense = _envelope_decimated(
        raw_x, raw_y, (float(raw_x[0]), float(raw_x[-1])), int(WIDTH * dpr)
    )
    level = np.array([12, WIDTH - 12]) * dpr
    return {
        "steep noise": (xs, steep),
        "gentle": (xs, gentle),
        "dense band": (dense_x, dense),
        "level": (level, np.array([HEIGHT / 2 + 0.3, HEIGHT / 2 + 0.3]) * dpr),
        "vertical": (np.array([WIDTH / 2 + 0.3, WIDTH / 2 + 0.3]) * dpr, np.array([12, HEIGHT - 12]) * dpr),
        "diagonal": (np.array([12, WIDTH - 12]) * dpr, np.array([12, HEIGHT - 12]) * dpr),
    }


@pytest.mark.parametrize("dpr", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("width_pt", [1.0, 1.5])
def test_the_kernel_lays_the_ink_agg_lays(dpr: float, width_pt: float) -> None:
    width_px = max(1.0, width_pt * 100.0 * dpr / 72.0)
    for name, (xs, ys) in _shapes(dpr).items():
        agg = _agg(xs, ys, width_px, dpr)
        ours = _kernel(xs, ys, width_px, dpr)
        ink_agg = float((255 - agg).sum())
        ink_ours = float((255 - ours).sum())
        ratio = ink_ours / ink_agg
        mean_delta = float(np.abs(ours - agg).mean())
        # Column by column: a run the kernel lost shows as columns with a
        # fraction of Agg's ink, and a mean over the whole box hides it.
        column_agg = (255 - agg).sum(axis=0)
        column_ours = (255 - ours).sum(axis=0)
        inked = column_agg > 2 * 255
        starved = int(np.count_nonzero(column_ours[inked] < 0.5 * column_agg[inked]))
        assert 0.85 <= ratio <= 1.15, (
            f"{name} @dpr {dpr} {width_pt}pt: the kernel lays {ratio:.2f}x Agg's ink"
        )
        assert starved == 0, (
            f"{name} @dpr {dpr} {width_pt}pt: {starved} columns hold less than "
            "half of Agg's ink"
        )
        # Agg lays a little more ink than the geometry where two bands of a
        # thinned trace come within a fraction of a pixel of each other, so
        # the dense band is held to a looser bound than a single stroke.
        bound = 1.5 if name == "dense band" else 1.0
        assert mean_delta < bound, (
            f"{name} @dpr {dpr} {width_pt}pt: mean |delta| {mean_delta:.2f} levels"
        )
