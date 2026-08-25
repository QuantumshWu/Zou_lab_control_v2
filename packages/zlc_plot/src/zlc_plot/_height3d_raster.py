"""A software rasterizer for the Image kind's height-bar presentation.

One (ny, nx) value grid drawn as shaded, outlined boxes standing on a
white gridded floor between two empty gridded back panes -- the classic
tomography look -- rendered by pure numpy into a uint8 RGBA buffer that
the existing image compose blits.  The bar heights and the pane span are
anchored to the SAME limits as the colour scale, so the z axis and the
colorbar read as one quantity.

The algorithm exploits the one property this camera guarantees: an
orthographic projection whose z axis is vertical ON SCREEN.  Every
pixel COLUMN then shares a single footprint line through the ground
grid, so the whole scene is a per-column front-to-back walk:

* every column's grid crossings are generated in bulk and merged
  arithmetically (two arithmetic progressions interleave in closed
  form -- no sort),
* occlusion is ONE ``np.maximum.accumulate`` down the step axis,
* the spans materialize through one bincount and one cumsum, and
* face outlines fall out of edge detection on a face-id plane -- which
  doubles as the pixel->bar PICK map interactions read.

Cost is O(width * (nx + ny)) + O(width * height).  A grid denser than
the pixels it lands on (a 1000x2000 scan on a 600-px box) pools to the
display resolution first, exactly as the heatmap's front store does,
so the walk never exceeds a few cells per pixel column.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(
            max_workers=min(8, os.cpu_count() or 1),
            thread_name_prefix="h3d_cumsum",
        )
    return _POOL


def _cumsum_axis0(diff: NDArray, out: NDArray) -> None:
    """Column-parallel prefix sum along axis 0, bit-identical to numpy.

    An axis-0 cumsum walks a non-contiguous stride element by element;
    every column is independent, so column blocks fan out over threads
    (numpy releases the GIL) without changing a single addition's order.
    """

    columns = diff.shape[1]
    workers = _pool()._max_workers
    if columns < 4096 or workers < 2:
        np.cumsum(diff, axis=0, out=out)
        return
    block = -(-columns // workers)
    futures = [
        _pool().submit(
            np.cumsum,
            diff[:, start:start + block],
            axis=0,
            out=out[:, start:start + block],
        )
        for start in range(0, columns, block)
    ]
    for future in futures:
        future.result()


@dataclass(frozen=True, slots=True)
class HeightBarCamera:
    """The presentation camera: orbit angles plus a zoom factor."""

    azimuth_deg: float = -55.0
    elevation_deg: float = 30.0
    zoom: float = 1.0

    def __post_init__(self) -> None:
        for name in ("azimuth_deg", "elevation_deg", "zoom"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite number")
        object.__setattr__(self, "azimuth_deg", float(self.azimuth_deg))
        object.__setattr__(
            self,
            "elevation_deg",
            float(min(max(self.elevation_deg, 8.0), 80.0)),
        )
        object.__setattr__(self, "zoom", float(min(max(self.zoom, 0.4), 6.0)))


@dataclass(frozen=True, slots=True)
class HeightBarScene:
    """Everything an interaction needs to read one rendered scene.

    ``project`` speaks FOLDED ground coordinates (the render's own
    frame); ``pick`` and ``cell_corners`` speak the caller's original
    grid indices and undo the azimuth fold and the LOD pooling.
    """

    quadrant: int
    ca: float
    sa: float
    se: float
    ce: float
    scale: float
    x_low: float
    y_high: float
    nx: int
    ny: int
    source_nx: int
    source_ny: int
    pool_x: int
    pool_y: int
    z_unit: float
    value_low: float
    value_high: float
    width: int
    height: int
    id_plane: NDArray[np.int32]
    #: Folded (ny, nx) drawn TOP-surface height per cell in value units
    #: (max(clipped, 0); -inf where no bar).  With the z axis vertical on
    #: screen a pixel's view ray RISES toward the viewer, so "this pixel's
    #: top face occludes a point" is exactly "its top is above the point".
    top_values: NDArray[np.float64]
    #: True when cells are below outline density and the grid drew as a
    #: lit continuous surface; scene chrome skips floor rules then --
    #: a rule at z=0 under a near-zero surface wins the height tie and
    #: leaks through as bright dashes.
    dense: bool

    def project(self, a: float, b: float, value: float) -> tuple[float, float]:
        """Folded ground point + value -> pixel (x, y), y down."""

        sx = a * self.ca + b * self.sa
        sy = (
            -a * self.sa * self.se
            + b * self.ca * self.se
            + value * self.z_unit * self.ce
        )
        return (sx - self.x_low) * self.scale, (self.y_high - sy) * self.scale

    def fold_cell(self, row: int, column: int) -> tuple[int, int]:
        """Original grid indices -> folded (a_cell, b_cell) indices."""

        a, b = int(column) // self.pool_x, int(row) // self.pool_y
        if self.quadrant == 1:
            a = self.nx - 1 - a
        elif self.quadrant == 2:
            a, b = self.nx - 1 - a, self.ny - 1 - b
        elif self.quadrant == 3:
            b = self.ny - 1 - b
        return a, b

    def pick(self, x: float, y: float) -> tuple[int, int] | None:
        """Pixel -> original (row, column) of the bar drawn there."""

        column = int(x)
        row = int(y)
        if not (0 <= row < self.height and 0 <= column < self.width):
            return None
        face = int(self.id_plane[row, column])
        if face < 4:
            return None
        cell = (face - 4) // 4
        a, b = cell % self.nx, cell // self.nx
        if self.quadrant == 1:
            a = self.nx - 1 - a
        elif self.quadrant == 2:
            a, b = self.nx - 1 - a, self.ny - 1 - b
        elif self.quadrant == 3:
            b = self.ny - 1 - b
        return b * self.pool_y, a * self.pool_x

    def cell_corners(
        self, row: int, column: int
    ) -> tuple[tuple[float, float], ...]:
        """The four ground corners of one original cell, in pixels."""

        a, b = self.fold_cell(row, column)
        return tuple(
            self.project(float(a + da), float(b + db), 0.0)
            for da, db in ((0, 0), (1, 0), (1, 1), (0, 1))
        )


def _pooled(
    grid: NDArray[np.float64],
    rgb: NDArray[np.float32],
    finite: NDArray[np.bool_],
    limit: int,
) -> tuple[NDArray, NDArray, NDArray, int, int]:
    """Mean-pool a grid denser than its pixels down to display resolution."""

    ny, nx = grid.shape
    pool_y = max(1, -(-ny // max(limit, 1)))
    pool_x = max(1, -(-nx // max(limit, 1)))
    if pool_x == 1 and pool_y == 1:
        return grid, rgb, finite, 1, 1
    pad_y = (-ny) % pool_y
    pad_x = (-nx) % pool_x
    if pad_y or pad_x:
        grid = np.pad(grid, ((0, pad_y), (0, pad_x)), constant_values=np.nan)
        rgb = np.pad(rgb, ((0, pad_y), (0, pad_x), (0, 0)))
        finite = np.pad(finite, ((0, pad_y), (0, pad_x)))
    blocks_y = grid.shape[0] // pool_y
    blocks_x = grid.shape[1] // pool_x
    shaped = grid.reshape(blocks_y, pool_y, blocks_x, pool_x)
    counts = finite.reshape(blocks_y, pool_y, blocks_x, pool_x).sum(
        axis=(1, 3)
    )
    pooled_h = (
        np.nansum(np.where(np.isfinite(shaped), shaped, 0.0), axis=(1, 3))
        / np.maximum(counts, 1)
    )
    pooled_h = np.where(counts > 0, pooled_h, np.nan)
    pooled_rgb = rgb.reshape(blocks_y, pool_y, blocks_x, pool_x, 3).mean(
        axis=(1, 3), dtype=np.float32
    )
    return pooled_h, pooled_rgb, counts > 0, pool_x, pool_y


_ENGINE = os.environ.get("ZLC_H3D_ENGINE", "auto")


def _scanline_selected() -> bool:
    """Whether the numba scanline engine renders this frame.

    ``ZLC_H3D_ENGINE`` forces ``numpy`` (the reference) or ``numba``;
    ``auto`` uses the scanline engine whenever numba imports.  Both
    engines are bit-identical by contract (test_height_bars pins it).
    """

    if _ENGINE == "numpy":
        return False
    try:
        from . import _height3d_scanline as scanline
    except Exception:
        return False
    return scanline.HAVE_NUMBA


def render_height_bars(
    heights: NDArray[np.floating],
    top_rgb: NDArray[np.floating],
    *,
    camera: HeightBarCamera,
    value_limits: tuple[float, float],
    width: int,
    height: int,
    supersample: int = 1,
    side_shades: tuple[float, float] = (1.0, 1.0),
    edge_darken: float = 0.85,
    background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    zero_rgb: tuple[float, float, float] | None = None,
    z_fraction: float = 0.55,
    pool_pixels_per_cell: float = 2.0,
    edge_min_cell_px: float = 3.0,
    bar_edges: bool = True,
) -> tuple[NDArray[np.uint8], HeightBarScene]:
    """Render the grid as boxes -> ((H, W, 4) uint8 RGBA, scene map).

    NaN heights are absent bars.  Bar heights anchor to ``value_limits``
    (the colour limits): values clip to them exactly as colours saturate,
    a negative low hangs the zero plane mid-scene, and the pane/z-axis
    span IS the limit span.  The floor and the two back panes are plain
    background surfaces: every LINE of the scene -- pane grids, floor
    grid, axes and bar outlines -- is vector chrome drawn by the caller,
    never raster pixels, so nothing here can alias.
    """

    if supersample not in (1, 2, 3, 4):
        raise ValueError("supersample must be 1, 2, 3 or 4")
    render_w = int(width) * supersample
    render_h = int(height) * supersample
    if render_w < 8 or render_h < 8:
        raise ValueError("height-bar raster needs at least 8x8 pixels")

    h_grid = np.asarray(heights, dtype=np.float64)
    rgb_grid = np.asarray(top_rgb, dtype=np.float32)
    if h_grid.ndim != 2 or rgb_grid.shape != (*h_grid.shape, 3):
        raise ValueError("heights must be (ny, nx) and top_rgb (ny, nx, 3)")
    source_ny, source_nx = h_grid.shape
    finite_grid = np.isfinite(h_grid)

    # ---- LOD: a grid denser than the pixels pools to display resolution
    limit = max(8, int(render_w / max(pool_pixels_per_cell, 0.5)))
    h_grid, rgb_grid, finite_grid, pool_x, pool_y = _pooled(
        h_grid, rgb_grid, finite_grid, limit
    )
    ny, nx = h_grid.shape

    # ---- fold the azimuth into [0, 90) by flipping the grid
    azimuth = math.radians(camera.azimuth_deg) % (2.0 * math.pi)
    quadrant = int(azimuth // (math.pi / 2.0))
    local = azimuth - quadrant * (math.pi / 2.0)
    if quadrant == 1:
        h_grid, rgb_grid, finite_grid = (
            h_grid[:, ::-1], rgb_grid[:, ::-1], finite_grid[:, ::-1]
        )
    elif quadrant == 2:
        h_grid, rgb_grid, finite_grid = (
            h_grid[::-1, ::-1], rgb_grid[::-1, ::-1], finite_grid[::-1, ::-1]
        )
    elif quadrant == 3:
        h_grid, rgb_grid, finite_grid = (
            h_grid[::-1, :], rgb_grid[::-1, :], finite_grid[::-1, :]
        )
    ca = max(math.cos(local), 1e-9)
    sa = max(math.sin(local), 1e-9)
    elevation = math.radians(camera.elevation_deg)
    se, ce = math.sin(elevation), math.cos(elevation)

    value_low, value_high = (float(v) for v in value_limits)
    if not (math.isfinite(value_low) and math.isfinite(value_high)):
        value_low, value_high = 0.0, 1.0
    if value_high <= value_low:
        value_high = value_low + 1.0
    magnitude = max(abs(value_low), abs(value_high))
    if magnitude <= 0.0:
        magnitude = 1.0
    z_unit = z_fraction * max(nx, ny) / magnitude
    clipped = np.clip(h_grid, value_low, value_high)
    hz = np.where(finite_grid, clipped, 0.0) * z_unit
    z_top = np.maximum(hz, 0.0) * ce
    z_bot = np.minimum(hz, 0.0) * ce
    pane_high = max(value_high, 0.0) * z_unit * ce
    pane_low = min(value_low, 0.0) * z_unit * ce

    # ---- screen frame (y up during the math), fit-to-box then zoom
    ux, uy = ca, -sa * se
    vx, vy = sa, ca * se
    ground_x = np.array([0.0, nx * ux, ny * vx, nx * ux + ny * vx])
    ground_y = np.array([0.0, nx * uy, ny * vy, nx * uy + ny * vy])
    x_low = float(ground_x.min())
    x_high = float(ground_x.max())
    y_low = float(ground_y.min()) + pane_low
    y_high = float(ground_y.max()) + pane_high
    margin = 0.04 * max(x_high - x_low, y_high - y_low)
    x_low -= margin
    x_high += margin
    y_low -= margin
    y_high += margin
    scale = (
        min(render_w / (x_high - x_low), render_h / (y_high - y_low))
        * camera.zoom
    )
    x_mid = 0.5 * (x_low + x_high)
    y_mid = 0.5 * (y_low + y_high)
    x_low = x_mid - 0.5 * render_w / scale
    y_high = y_mid + 0.5 * render_h / scale

    # ---- per-column footprint lines: a*ca + b*sa = X, direction (-sa, ca)
    X = (np.arange(render_w, dtype=np.float64) + 0.5) / scale + x_low
    a0 = X / ca
    t_enter = np.maximum((a0 - nx) / sa, 0.0)
    t_exit = np.minimum(a0 / sa, ny / ca)
    t_exit = np.maximum(t_exit, t_enter)
    a_at = a0 - t_enter * sa
    b_at = t_enter * ca
    ia0 = np.clip(np.ceil(a_at) - 1.0, 0.0, nx - 1.0)
    ib0 = np.clip(np.floor(b_at), 0.0, ny - 1.0)

    # ---- crossings merged arithmetically (ties resolve a-before-b)
    t_a0 = t_enter + (a_at - ia0) / sa
    t_b0 = t_enter + ((ib0 + 1.0) - b_at) / ca
    # ---- per-cell colour grids and z planes, shared by both engines
    z_top32 = z_top.astype(np.float32)
    z_bot32 = z_bot.astype(np.float32)
    dense_surface = scale < edge_min_cell_px * supersample
    if zero_rgb is None:
        base_grid = rgb_grid
    else:
        base_grid = np.ascontiguousarray(np.broadcast_to(
            np.asarray(zero_rgb, dtype=np.float32), rgb_grid.shape
        ))
    if dense_surface:
        # Sub-outline density: the grid reads as a heightfield, and flat
        # per-cell tops lose all depth.  Light the top faces by the local
        # slope (light from the upper-left of the screen frame) -- per
        # CELL here, gathered per span below: elementwise either way, so
        # the two orders are bit-identical.
        gradient_b, gradient_a = np.gradient(hz)
        slope_field = gradient_b - gradient_a
        slope_field = slope_field / np.sqrt(1.0 + np.square(slope_field))
        lighting = np.clip(
            1.0 + 0.45 * slope_field, 0.6, 1.2
        ).astype(np.float32)
        top_grid = np.clip(rgb_grid * lighting[..., None], 0.0, 1.0)
    else:
        top_grid = rgb_grid

    if _scanline_selected():
        # The numba scanline engine: one pixel written once per column
        # walk, bit-identical to the reference materialization below by
        # the standing contract test.
        from ._height3d_scanline import _materialize

        out = np.empty((render_h, render_w, 4), dtype=np.uint8)
        id_plane = np.empty((render_h, render_w), dtype=np.int32)
        bg32 = np.asarray(background_rgb, dtype=np.float32)
        _materialize(
            a_at,
            t_enter,
            t_exit,
            ia0,
            ib0,
            t_a0,
            t_b0,
            (a0 * uy).astype(np.float32),
            np.ascontiguousarray(z_top32),
            np.ascontiguousarray(z_bot32),
            np.ascontiguousarray(finite_grid),
            np.ascontiguousarray(rgb_grid),
            np.ascontiguousarray(base_grid),
            np.ascontiguousarray(top_grid),
            bool(dense_surface),
            np.float32(side_shades[0]),
            np.float32(side_shades[1]),
            float(sa),
            float(ca),
            np.float32(se),
            float(y_high),
            float(scale),
            np.float32(pane_high),
            np.float32(min(pane_low, 0.0)),
            bg32,
            (bg32 * 255.0).astype(np.uint8),
            out,
            id_plane,
            np.int64(min(32, (os.cpu_count() or 4) * 2)),
        )
    else:
        k = np.arange(nx, dtype=np.float64)[:, None]
        m = np.arange(ny, dtype=np.float64)[:, None]
        t_cross_a = t_a0[None, :] + k / sa
        t_cross_b = t_b0[None, :] + m / ca
        b_before_a = np.clip(np.ceil((t_cross_a - t_b0[None, :]) * ca), 0.0, ny)
        pos_a = (k + b_before_a).astype(np.int64)

        # The b-crossings take the COMPLEMENT of the a slots, in order.  A
        # second float formula for their positions could disagree with
        # ``pos_a`` at an exact tie -- two crossings claiming one slot, the
        # orphaned slot left holding uninitialized memory, and the frame
        # nondeterministic.  The complement is pure integer bookkeeping, so
        # the merge is a permutation BY CONSTRUCTION.
        S = nx + ny
        columns_row = np.arange(render_w, dtype=np.int64)
        used = np.zeros((render_w, S), dtype=bool)
        used[columns_row[None, :], pos_a] = True
        unused_cols, unused_slots = np.nonzero(~used)
        pos_b = unused_slots.reshape(render_w, ny).T
        sorted_t = np.empty((S, render_w), dtype=np.float64)
        sorted_is_b = np.ones((S, render_w), dtype=bool)
        sorted_t[pos_a, columns_row[None, :]] = t_cross_a
        sorted_is_b[pos_a, columns_row[None, :]] = False
        sorted_t[pos_b, columns_row[None, :]] = t_cross_b

        with np.errstate(over="ignore"):
            # Clamped-azimuth crossings can exceed float32 range; they land
            # as inf, which every comparison downstream handles.
            sorted_t = sorted_t.astype(np.float32)
        seg_t0 = np.empty((S, render_w), dtype=np.float32)
        seg_t0[0] = t_enter
        seg_t0[1:] = sorted_t[:-1]
        np.clip(seg_t0, None, t_exit[None, :], out=seg_t0)
        seg_t1 = np.clip(sorted_t, None, t_exit[None, :])

        before_b = np.cumsum(sorted_is_b, axis=0)
        before_b_prior = np.empty_like(before_b)
        before_b_prior[0] = 0
        before_b_prior[1:] = before_b[:-1]
        seg_index = np.arange(S, dtype=np.int64)[:, None]
        ia = ia0[None, :].astype(np.int64) - (seg_index - before_b_prior)
        ib = ib0[None, :].astype(np.int64) + before_b_prior
        inside = (ia >= 0) & (ia < nx) & (ib >= 0) & (ib < ny) & (seg_t1 > seg_t0)
        cell_a = np.clip(ia, 0, nx - 1)
        cell_b = np.clip(ib, 0, ny - 1)

        entered_x = np.empty((S, render_w), dtype=bool)
        entered_x[0] = np.abs(a_at - np.round(a_at)) < 1e-9
        entered_x[1:] = ~sorted_is_b[:-1]

        cell_top = z_top32[cell_b, cell_a]
        cell_bot = z_bot32[cell_b, cell_a]
        cell_ok = finite_grid[cell_b, cell_a] & inside
        g0 = (a0 * uy).astype(np.float32)
        se32 = np.float32(se)
        g_lo = g0[None, :] + seg_t0 * se32
        g_hi = g0[None, :] + seg_t1 * se32

        silhouette = np.where(cell_ok, g_hi + cell_top, -np.inf)
        running = np.maximum.accumulate(silhouette, axis=0)
        watermark = np.empty_like(running)
        watermark[0] = -np.inf
        watermark[1:] = running[:-1]

        # Bar face ids start at 4: 1 is the floor, 2 the panes, 3 reserved --
        # a bar's faces must never collide with the scene chrome.
        face_id = (cell_b * nx + cell_a) * 4 + 4
        shade = np.where(entered_x, side_shades[0], side_shades[1]).astype(
            np.float32
        )
        side_lo = np.maximum(g_lo + cell_bot, watermark)
        side_hi = np.where(cell_ok, g_lo + cell_top, -np.inf)
        top_lo = np.maximum(np.where(cell_ok, g_lo + cell_top, np.inf), watermark)
        top_hi = np.where(cell_ok, g_hi + cell_top, -np.inf)

        cell_rgb = rgb_grid[cell_b, cell_a]
        top_rgb_cells = cell_rgb
        # Side faces interp-shade through the colour scale, the reference
        # figures' MATLAB ``shading interp`` look: the z=0 end of every side
        # face takes the zero-value colour, the far end the bar's own.
        base_rgb_cells = base_grid[cell_b, cell_a]
        positive = z_top32[cell_b, cell_a] > 0.0
        side_top_rgb = np.where(
            positive[..., None], cell_rgb, base_rgb_cells
        ) * shade[..., None]
        side_bottom_rgb = np.where(
            positive[..., None], base_rgb_cells, cell_rgb
        ) * shade[..., None]
        if dense_surface:
            # A dense grid is a lit continuous surface (see the shared
            # per-cell colour grids above): tops and sides gather the same
            # pre-lit colours.
            top_rgb_cells = top_grid[cell_b, cell_a]
            side_top_rgb = top_rgb_cells
            side_bottom_rgb = top_rgb_cells
        background = np.asarray(background_rgb, dtype=np.float32)

        span_cols = [
            np.broadcast_to(columns_row, (S, render_w)).ravel(),
            np.broadcast_to(columns_row, (S, render_w)).ravel(),
            np.broadcast_to(columns_row, (S, render_w)).ravel(),
        ]
        # The floor: the background surface under every cell without a bar.
        floor_lo = np.maximum(g_lo, watermark)
        floor_hi = np.where(inside & ~cell_ok, g_hi, -np.inf)
        span_lo = [side_lo.ravel(), top_lo.ravel(), floor_lo.ravel()]
        span_hi = [side_hi.ravel(), top_hi.ravel(), floor_hi.ravel()]
        # The side spans remember their UNCLIPPED extremes: the gradient's
        # intercept and slope come from the full face, so watermark clipping
        # trims pixels without shifting the shading.
        side_full_lo = (g_lo + cell_bot).ravel()
        side_full_hi = (g_lo + cell_top).ravel()
        span_rgb_low = [
            side_bottom_rgb.reshape(-1, 3),
            top_rgb_cells.reshape(-1, 3),
            np.broadcast_to(background, (S * render_w, 3)),
        ]
        span_rgb_high = [
            side_top_rgb.reshape(-1, 3),
            top_rgb_cells.reshape(-1, 3),
            np.broadcast_to(background, (S * render_w, 3)),
        ]
        span_full_lo = [side_full_lo, top_lo.ravel(), floor_lo.ravel()]
        span_full_hi = [side_full_hi, top_hi.ravel(), floor_hi.ravel()]
        span_id = [
            (face_id + 1 + entered_x.astype(np.int64)).ravel(),
            (face_id + 3).ravel(),
            np.broadcast_to(np.int64(1), (S * render_w,)),
        ]

        # Back panes: background-coloured, from the grid's far silhouette up
        # to the pane top.
        final_watermark = running[-1] if S else np.full(render_w, -np.inf)
        g_exit = g0 + (t_exit.astype(np.float32) * se32)
        pane_top_y = g_exit + np.float32(pane_high)
        pane_bottom = np.maximum(
            g_exit + np.float32(min(pane_low, 0.0)), final_watermark
        )
        pane_visible = t_exit > t_enter
        span_cols.append(columns_row)
        pane_lo_values = np.where(pane_visible, pane_bottom, np.inf)
        pane_hi_values = np.where(pane_visible, pane_top_y, -np.inf)
        span_lo.append(pane_lo_values)
        span_hi.append(pane_hi_values)
        span_rgb_low.append(np.broadcast_to(background, (render_w, 3)))
        span_rgb_high.append(np.broadcast_to(background, (render_w, 3)))
        span_full_lo.append(pane_lo_values)
        span_full_hi.append(pane_hi_values)
        span_id.append(np.broadcast_to(np.int64(2), (render_w,)))

        all_cols = np.concatenate(span_cols)
        all_lo = np.concatenate(span_lo)
        all_hi = np.concatenate(span_hi)
        all_rgb_low = np.concatenate(span_rgb_low)
        all_rgb_high = np.concatenate(span_rgb_high)
        all_full_lo = np.concatenate(span_full_lo)
        all_full_hi = np.concatenate(span_full_hi)
        all_id = np.concatenate(span_id)

        keep = all_hi > all_lo
        cols = all_cols[keep]
        lo = all_lo[keep]
        hi = all_hi[keep]
        color_low = all_rgb_low[keep]
        color_high = all_rgb_high[keep]
        full_lo = all_full_lo[keep]
        full_hi = all_full_hi[keep]
        ids = all_id[keep]
        row_hi = np.clip((y_high - lo) * scale, 0.0, float(render_h)).astype(
            np.int64
        )
        row_lo = np.clip((y_high - hi) * scale, 0.0, float(render_h)).astype(
            np.int64
        )
        keep2 = row_hi > row_lo
        cols = cols[keep2]
        row_lo = row_lo[keep2]
        row_hi = row_hi[keep2]
        color_low = color_low[keep2]
        color_high = color_high[keep2]
        full_lo = full_lo[keep2]
        full_hi = full_hi[keep2]
        ids = ids[keep2]

        # Gradient as intercept+slope in row space, from the UNCLIPPED face:
        # colour(row) = A + s*row, with s = 0 for every constant span.
        full_row_bottom = (y_high - full_lo) * scale   # larger row (z low end)
        full_row_top = (y_high - full_hi) * scale
        span_height = np.maximum(full_row_bottom - full_row_top, 1e-6)
        slope = (color_low - color_high) / span_height[:, None]
        intercept = color_high - slope * full_row_top[:, None]

        stride = render_w
        flat_lo = row_lo * stride + cols
        flat_hi = row_hi * stride + cols
        plane = (render_h + 1) * stride
        channel_index = np.concatenate([
            flat_lo * 3,
            flat_lo * 3 + 1,
            flat_lo * 3 + 2,
            flat_hi * 3,
            flat_hi * 3 + 1,
            flat_hi * 3 + 2,
        ])
        intercept_weights = np.concatenate([
            intercept[:, 0], intercept[:, 1], intercept[:, 2],
            -intercept[:, 0], -intercept[:, 1], -intercept[:, 2],
        ]).astype(np.float64) * 255.0
        slope_weights = np.concatenate([
            slope[:, 0], slope[:, 1], slope[:, 2],
            -slope[:, 0], -slope[:, 1], -slope[:, 2],
        ]).astype(np.float64) * 255.0
        # The three scatter planes are independent of one another: fan the
        # bincounts out over the pool (bit-identical -- no shared sums), and
        # let each cast to its cumsum dtype in its own thread.
        # For the id plane, ONE weighted bincount with +/- weights replaces
        # two full planes; the ids are exact integers, so the merged
        # summation order cannot change a value, and the cumsum runs in
        # int32 -- a quarter of the float64 bytes, no rounding pass.
        id_weights = ids.astype(np.float64)
        futures = [
            _pool().submit(
                lambda w: np.bincount(
                    channel_index, weights=w, minlength=plane * 3
                ).astype(np.float32),
                weights,
            )
            for weights in (intercept_weights, slope_weights)
        ]
        futures.append(_pool().submit(
            lambda: np.bincount(
                np.concatenate([flat_lo, flat_hi]),
                weights=np.concatenate([id_weights, -id_weights]),
                minlength=plane,
            ).astype(np.int32)
        ))
        intercept_diff, slope_diff, id_diff = (f.result() for f in futures)
        filled_full = np.empty((render_h + 1, stride * 3), dtype=np.float32)
        _cumsum_axis0(intercept_diff.reshape(render_h + 1, stride * 3), filled_full)
        filled = filled_full[:render_h].reshape(render_h, stride, 3)
        slope_full = np.empty((render_h + 1, stride * 3), dtype=np.float32)
        _cumsum_axis0(slope_diff.reshape(render_h + 1, stride * 3), slope_full)
        slope_plane = slope_full[:render_h].reshape(render_h, stride, 3)
        # Scale the slope plane by its row IN PLACE: the broadcast product
        # would allocate a full extra colour plane per frame.
        slope_plane *= np.arange(render_h, dtype=np.float32)[:, None, None]
        filled += slope_plane
        id_plane_full = np.empty((render_h + 1, stride), dtype=np.int32)
        _cumsum_axis0(id_diff.reshape(render_h + 1, stride), id_plane_full)
        id_plane = id_plane_full[:render_h]

        covered = id_plane > 0
        np.clip(filled, 0.0, 255.0, out=filled)
        out = np.empty((render_h, render_w, 4), dtype=np.uint8)
        out[..., :3] = filled
        out[..., 3] = covered.astype(np.uint8) * np.uint8(255)
        # Uncovered pixels carry the BACKGROUND colour, not black: the
        # supersample average otherwise mixes black into the scene's border
        # pixels and draws the pane silhouette as a faint dotted outline --
        # the very closing border this scene must not have.
        out[..., :3][~covered] = (
            np.asarray(background_rgb, dtype=np.float32) * 255.0
        ).astype(np.uint8)

    covered = id_plane > 0
    # ---- raster outlines from the id plane (mid-density grids only:
    # small grids hand their outlines to vector chrome, dense grids are
    # a lit continuous surface, and neither takes raster lines)
    if bar_edges and not dense_surface:
        edge = np.zeros((render_h, render_w), dtype=bool)
        edge[:, 1:] |= (id_plane[:, 1:] != id_plane[:, :-1]) & (
            covered[:, 1:] & covered[:, :-1]
        )
        edge[1:, :] |= (id_plane[1:, :] != id_plane[:-1, :]) & (
            covered[1:, :] & covered[:-1, :]
        )
        for _ in range(supersample - 1):
            edge[1:, :] |= edge[:-1, :]
            edge[:, 1:] |= edge[:, :-1]
        edge_rows, edge_cols = np.nonzero(edge)
        edge_ids = id_plane[edge_rows, edge_cols]
        left_ids = id_plane[edge_rows, np.maximum(edge_cols - 1, 0)]
        up_ids = id_plane[np.maximum(edge_rows - 1, 0), edge_cols]
        bar_edge = (edge_ids >= 4) | (left_ids >= 4) | (up_ids >= 4)
        edge_rows = edge_rows[bar_edge]
        edge_cols = edge_cols[bar_edge]
        out[edge_rows, edge_cols, :3] = (
            out[edge_rows, edge_cols, :3].astype(np.float32)
            * np.float32(1.0 - edge_darken)
        ).astype(np.uint8)

    scene_id_plane = id_plane
    if supersample > 1:
        boxed = out.reshape(height, supersample, width, supersample, 4)
        samples = supersample * supersample
        averaged = np.empty((height, width, 4), dtype=np.uint8)

        def _pool_rows(start: int, stop: int) -> None:
            # Rows are independent: each block box-averages its slice
            # alone, so the fan-out cannot change a single sum.
            total = np.zeros((stop - start, width, 4), dtype=np.uint16)
            for row_tap in range(supersample):
                for column_tap in range(supersample):
                    total += boxed[start:stop, row_tap, :, column_tap]
            averaged[start:stop] = (
                (total + samples // 2) // samples
            ).astype(np.uint8)

        workers = _pool()._max_workers
        block = -(-height // workers)
        futures = [
            _pool().submit(_pool_rows, start, min(start + block, height))
            for start in range(0, height, block)
        ]
        for future in futures:
            future.result()
        out = averaged
        scene_id_plane = id_plane[::supersample, ::supersample]
        scale = scale / supersample

    scene = HeightBarScene(
        quadrant=quadrant,
        ca=ca,
        sa=sa,
        se=se,
        ce=ce,
        scale=scale,
        x_low=x_low,
        y_high=y_high,
        nx=nx,
        ny=ny,
        source_nx=source_nx,
        source_ny=source_ny,
        pool_x=pool_x,
        pool_y=pool_y,
        z_unit=z_unit,
        value_low=value_low,
        value_high=value_high,
        width=int(width),
        height=int(height),
        id_plane=np.ascontiguousarray(scene_id_plane.astype(np.int32)),
        top_values=np.where(
            finite_grid, np.maximum(clipped, 0.0), -np.inf
        ),
        dense=bool(dense_surface),
    )
    return out, scene


__all__ = ["HeightBarCamera", "HeightBarScene", "render_height_bars"]
