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
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
    grid_rgb: tuple[float, float, float] = (0.78, 0.78, 0.80),
    background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    zero_rgb: tuple[float, float, float] | None = None,
    z_fraction: float = 0.55,
    wall_ticks: int = 4,
    pool_pixels_per_cell: float = 2.0,
    edge_min_cell_px: float = 3.0,
) -> tuple[NDArray[np.uint8], HeightBarScene]:
    """Render the grid as boxes -> ((H, W, 4) uint8 RGBA, scene map).

    NaN heights are absent bars.  Bar heights anchor to ``value_limits``
    (the colour limits): values clip to them exactly as colours saturate,
    a negative low hangs the zero plane mid-scene, and the pane/z-axis
    span IS the limit span.  The floor and the two back panes are drawn
    background-coloured, carrying grid lines only.
    """

    if supersample not in (1, 2):
        raise ValueError("supersample must be 1 or 2")
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
    k = np.arange(nx, dtype=np.float64)[:, None]
    m = np.arange(ny, dtype=np.float64)[:, None]
    t_cross_a = t_a0[None, :] + k / sa
    t_cross_b = t_b0[None, :] + m / ca
    b_before_a = np.clip(np.ceil((t_cross_a - t_b0[None, :]) * ca), 0.0, ny)
    a_before_b = np.clip(
        np.floor((t_cross_b - t_a0[None, :]) * sa) + 1.0, 0.0, nx
    )
    pos_a = (k + b_before_a).astype(np.int64)
    pos_b = (m + a_before_b).astype(np.int64)

    S = nx + ny
    columns_row = np.arange(render_w, dtype=np.int64)
    sorted_t = np.empty((S, render_w), dtype=np.float64)
    sorted_is_b = np.empty((S, render_w), dtype=bool)
    sorted_t[pos_a, columns_row[None, :]] = t_cross_a
    sorted_is_b[pos_a, columns_row[None, :]] = False
    sorted_t[pos_b, columns_row[None, :]] = t_cross_b
    sorted_is_b[pos_b, columns_row[None, :]] = True

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

    z_top32 = z_top.astype(np.float32)
    z_bot32 = z_bot.astype(np.float32)
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
    dense_surface = scale < edge_min_cell_px * supersample
    # Side faces interp-shade through the colour scale, the reference
    # figures' MATLAB ``shading interp`` look: the z=0 end of every side
    # face takes the zero-value colour, the far end the bar's own.
    if zero_rgb is None:
        base_rgb_cells = cell_rgb
    else:
        base_rgb_cells = np.broadcast_to(
            np.asarray(zero_rgb, dtype=np.float32), cell_rgb.shape
        )
    positive = z_top32[cell_b, cell_a] > 0.0
    side_top_rgb = np.where(
        positive[..., None], cell_rgb, base_rgb_cells
    ) * shade[..., None]
    side_bottom_rgb = np.where(
        positive[..., None], base_rgb_cells, cell_rgb
    ) * shade[..., None]
    if dense_surface:
        # Sub-outline density: the grid reads as a heightfield, and flat
        # per-cell tops lose all depth.  Light the top faces by the local
        # slope (light from the upper-left of the screen frame), the
        # standard heightfield cue, computed once per pooled cell.
        gradient_b, gradient_a = np.gradient(hz)
        # Scale-free slope: hz carries the scene's z factor, so the raw
        # gradient saturates any fixed gain.  g/sqrt(1+g^2) is the sine
        # of the slope angle -- bounded, unit-independent.
        slope = gradient_b - gradient_a
        slope = slope / np.sqrt(1.0 + np.square(slope))
        lighting = np.clip(1.0 + 0.45 * slope, 0.6, 1.2).astype(np.float32)
        top_rgb_cells = np.clip(
            cell_rgb * lighting[cell_b, cell_a][..., None], 0.0, 1.0
        )
        # At this density a cell's visible pixels are mostly its front
        # face slivers: painting those the constant side grey turned the
        # whole surface grey.  A dense grid is a continuous surface, so
        # its sides carry the same lit colour as its tops.
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
    intercept_diff = np.bincount(
        channel_index, weights=intercept_weights, minlength=plane * 3
    ).astype(np.float32)
    slope_diff = np.bincount(
        channel_index, weights=slope_weights, minlength=plane * 3
    ).astype(np.float32)
    filled = np.cumsum(
        intercept_diff.reshape(render_h + 1, stride, 3), axis=0
    )[:render_h]
    slope_plane = np.cumsum(
        slope_diff.reshape(render_h + 1, stride, 3), axis=0
    )[:render_h]
    filled += slope_plane * np.arange(render_h, dtype=np.float32)[
        :, None, None
    ]
    id_weights = ids.astype(np.float64)
    id_diff = np.bincount(
        flat_lo, weights=id_weights, minlength=plane
    ) - np.bincount(flat_hi, weights=id_weights, minlength=plane)
    id_plane = (
        np.cumsum(id_diff.reshape(render_h + 1, stride), axis=0)[:render_h]
        .round()
        .astype(np.int32)
    )

    covered = id_plane > 0
    np.clip(filled, 0.0, 255.0, out=filled)
    out = np.empty((render_h, render_w, 4), dtype=np.uint8)
    out[..., :3] = filled
    out[..., 3] = covered.astype(np.uint8) * np.uint8(255)

    # ---- outlines from the id plane
    edge = np.zeros((render_h, render_w), dtype=bool)
    edge[:, 1:] |= (id_plane[:, 1:] != id_plane[:, :-1]) & (
        covered[:, 1:] & covered[:, :-1]
    )
    edge[1:, :] |= (id_plane[1:, :] != id_plane[:-1, :]) & (
        covered[1:, :] & covered[:-1, :]
    )
    edge_rows, edge_cols = np.nonzero(edge)
    grid_color = (np.asarray(grid_rgb, dtype=np.float32) * 255.0).astype(
        np.uint8
    )
    # Chrome-to-chrome boundaries take the light grid colour; anything
    # touching a bar face darkens into an outline.
    edge_ids = id_plane[edge_rows, edge_cols]
    left_ids = id_plane[edge_rows, np.maximum(edge_cols - 1, 0)]
    up_ids = id_plane[np.maximum(edge_rows - 1, 0), edge_cols]
    bar_edge = (edge_ids >= 4) | (left_ids >= 4) | (up_ids >= 4)
    if dense_surface:
        # Sub-3px bars: an outline per bar would ink the whole surface.
        # The dense grid reads as a lit continuous surface instead; only
        # the chrome boundaries (floor|pane seams) keep their grid line.
        chrome_rows = edge_rows[~bar_edge]
        chrome_cols = edge_cols[~bar_edge]
        out[chrome_rows, chrome_cols, :3] = grid_color[None, :]
    else:
        darkened = (
            out[edge_rows, edge_cols, :3].astype(np.float32)
            * np.float32(1.0 - edge_darken)
        ).astype(np.uint8)
        out[edge_rows, edge_cols, :3] = np.where(
            bar_edge[:, None], darkened, grid_color[None, :]
        )

    # ---- grid lines ON the floor and the panes
    chrome_mask = (id_plane == 1) | (id_plane == 2)
    if chrome_mask.any():
        rows_map, cols_map = np.nonzero(chrome_mask)
        pixel_x = (cols_map + 0.5) / scale + x_low
        pixel_sy = y_high - (rows_map + 0.5) / scale
        on_floor = id_plane[rows_map, cols_map] == 1
        det = ux * vy - vx * uy
        a_coord = (pixel_x * vy - pixel_sy * vx) / det
        b_coord = (pixel_sy * ux - pixel_x * uy) / det
        # Cell-boundary lines: on a dense (pooled) grid rule every POOLED
        # cell, which is the drawn geometry.
        if dense_surface:
            floor_line = np.zeros(rows_map.shape, dtype=bool)
        else:
            near_a = np.abs(a_coord - np.round(a_coord)) * scale * sa < 0.6
            near_b = np.abs(b_coord - np.round(b_coord)) * scale * ca < 0.6
            floor_line = on_floor & (near_a | near_b)
        wall_line = np.zeros(rows_map.shape, dtype=bool)
        if wall_ticks > 0:
            base_y = np.interp(
                cols_map + 0.5, np.arange(render_w) + 0.5, g_exit
            )
            local_z = pixel_sy - base_y
            step = max((pane_high - pane_low) / (wall_ticks + 1), 1e-9)
            wall_line = (~on_floor) & (
                np.abs(local_z - np.round(local_z / step) * step) * 1.0
                < 0.6 / scale
            )
            # Vertical pane rules continue the floor grid up the panes,
            # the reference figure's look.  Which pane a column's far
            # silhouette belongs to decides the coordinate ruled on it.
            left_pane = (a0 / sa) < (ny / ca)  # t_exit bound by a = 0
            pane_of_pixel = left_pane[cols_map]
            exit_b = np.interp(
                cols_map + 0.5,
                np.arange(render_w) + 0.5,
                t_exit * ca,
            )
            exit_a = np.interp(
                cols_map + 0.5,
                np.arange(render_w) + 0.5,
                a0 - t_exit * sa,
            )
            if not dense_surface:
                along = np.where(pane_of_pixel, exit_b, exit_a)
                along_scale = np.where(
                    pane_of_pixel, scale * ca, scale * sa
                )
                vertical = (~on_floor) & (
                    np.abs(along - np.round(along)) * along_scale < 0.6
                )
                wall_line |= vertical
        chosen = floor_line | wall_line
        out[rows_map[chosen], cols_map[chosen], :3] = grid_color[None, :]

    scene_id_plane = id_plane
    if supersample == 2:
        boxed = out.reshape(height, 2, width, 2, 4)
        total = boxed[:, 0, :, 0].astype(np.uint16)
        total += boxed[:, 0, :, 1]
        total += boxed[:, 1, :, 0]
        total += boxed[:, 1, :, 1]
        out = ((total + 2) >> 2).astype(np.uint8)
        scene_id_plane = id_plane[::2, ::2]
        scale = scale / 2.0
        x_low = x_low  # unchanged: scale already halved for pixel mapping

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
    )
    return out, scene


__all__ = ["HeightBarCamera", "HeightBarScene", "render_height_bars"]
