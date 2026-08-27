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
    #: Pooled row count BEFORE the azimuth fold, and whether the source
    #: rows were reversed to stand the grid up the way the heatmap draws
    #: it.  Both live here so every index mapping below speaks the
    #: caller's ORIGINAL (row, column) whatever the renderer did.
    pooled_rows: int
    flip_rows: bool
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
    #: Folded (ny, nx) drawn BASE height per cell in value units
    #: (min(clipped, 0); +inf where no bar).  Top and base together ARE
    #: the box this render drew, which is what the outline traces: a
    #: consumer that re-derived the boxes from the caller's data was a
    #: second definition of the same thing, free to disagree with the
    #: picture it was drawn over.
    base_values: NDArray[np.float64]
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
        """Original grid indices -> folded (a_cell, b_cell) indices.

        Two transforms, in the order the render applied them: the row
        reversal that stands the grid up the way the heatmap draws it,
        then ``np.rot90(grid, quadrant)`` -- odd quadrants swap the axes
        (the folded width is the source height).  Both are written here
        for one index pair.
        """

        col_p, row_p = int(column) // self.pool_x, int(row) // self.pool_y
        if self.flip_rows:
            row_p = self.pooled_rows - 1 - row_p
        if self.quadrant == 0:
            return col_p, row_p
        if self.quadrant == 1:
            return row_p, self.ny - 1 - col_p
        if self.quadrant == 2:
            return self.nx - 1 - col_p, self.ny - 1 - row_p
        return self.nx - 1 - row_p, col_p

    def unfold_cell(self, a: int, b: int) -> tuple[int, int]:
        """Folded (a_cell, b_cell) -> pooled source (row_p, col_p)."""

        a, b = int(a), int(b)
        if self.quadrant == 0:
            row_p, col_p = b, a
        elif self.quadrant == 1:
            row_p, col_p = a, self.ny - 1 - b
        elif self.quadrant == 2:
            row_p, col_p = self.ny - 1 - b, self.nx - 1 - a
        else:
            row_p, col_p = self.nx - 1 - a, b
        if self.flip_rows:
            row_p = self.pooled_rows - 1 - row_p
        return row_p, col_p

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
        row_p, col_p = self.unfold_cell(a, b)
        return row_p * self.pool_y, col_p * self.pool_x

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


def _rim_stamp(width: float) -> tuple[NDArray[np.float32], int]:
    """Coverage weights of a round stroke of ``width`` output pixels.

    A crease lies BETWEEN two pixels, and both of them are marked, so the
    stroke is already one pixel wide before any spreading -- the stamp
    adds the rest and its shoulders.
    """

    half = float(width) * 0.5
    # The radius is where the weights actually stop, not where the width
    # rounds up: a stamp with a zero border costs every pixel a ring of
    # multiplies that can never win the maximum.
    radius = max(1, int(math.ceil(half - 1.0)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    distance = np.hypot(offsets[:, None], offsets[None, :])
    weights = np.clip(half - distance, 0.0, 1.0).astype(np.float32)
    return weights, radius


def _rim_boundary(id_plane: NDArray[np.int32]) -> NDArray[np.bool_]:
    """Pixels beside a visible crease of the DRAWN surface.

    The id plane names the face each pixel shows, so a change between
    neighbours IS a crease -- the exact set the outline traces, already
    occluded, because a hidden face is not in the plane.  Creases between
    two background surfaces (the floor meeting a pane) are the room, not
    the picture, and the room draws its own rules.
    """

    bar = id_plane >= 4
    boundary = np.zeros(id_plane.shape, dtype=bool)
    edge = (id_plane[:, 1:] != id_plane[:, :-1]) & (bar[:, 1:] | bar[:, :-1])
    boundary[:, 1:] |= edge
    boundary[:, :-1] |= edge
    edge = (id_plane[1:, :] != id_plane[:-1, :]) & (bar[1:, :] | bar[:-1, :])
    boundary[1:, :] |= edge
    boundary[:-1, :] |= edge
    return boundary


def _rim_coverage_reference(
    boundary: NDArray[np.bool_],
    weights: NDArray[np.float32],
    radius: int,
) -> NDArray[np.float32]:
    """The specification: coverage is the strongest stamp reaching here."""

    height, width = boundary.shape
    coverage = np.zeros(boundary.shape, dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            weight = weights[dy + radius, dx + radius]
            if weight <= 0.0:
                continue
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            if y0 >= y1 or x0 >= x1:
                continue
            source = boundary[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            target = coverage[y0:y1, x0:x1]
            np.maximum(target, np.where(source, weight, np.float32(0.0)),
                       out=target)
    return coverage


def _stroke_rims(
    out: NDArray[np.uint8],
    id_plane: NDArray[np.int32],
    rim_rgb: tuple[float, float, float],
    width: float,
) -> None:
    """Draw the surface's own creases into the surface, in place.

    Every rim of every box, at the cost of the PIXELS it covers rather
    than the boxes it belongs to.  Drawn as vector chrome it cost about
    forty nanoseconds of Matplotlib stroking per pixel of ink, and the
    ink grows with the bar count: fifty bars a side covered eighteen per
    cent of the canvas and spent nineteen milliseconds a frame, which is
    what made turning a dense scene heavy.  Here the whole pass is two
    sweeps of the id plane and one blend of the pixels it touched.

    The room -- pane grids, floor rules, the axes, the cage -- is still
    drawn by artists.  One owner each: the picture draws its own creases,
    the room draws its own furniture.
    """

    if width <= 0.0:
        return
    weights, radius = _rim_stamp(width)
    target = np.asarray(rim_rgb, dtype=np.float32) * np.float32(255.0)
    if _scanline_selected():
        from ._height3d_scanline import _rim_stroke

        _rim_stroke(
            np.ascontiguousarray(id_plane, dtype=np.int32),
            np.ascontiguousarray(weights),
            np.int64(radius),
            np.int64(min(32, (os.cpu_count() or 4) * 2)),
            out,
            target,
        )
        return
    coverage = _rim_coverage_reference(
        _rim_boundary(id_plane), weights, radius
    )[..., None]
    painted = out[..., :3].astype(np.float32)
    # Every pixel, not the touched ones: a boolean index over a frame this
    # size costs more than the whole stamping pass, and zero coverage
    # rounds each channel back to itself.
    out[..., :3] = (
        painted + (target[None, None, :] - painted) * coverage
        + np.float32(0.5)
    ).astype(np.uint8)


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
    background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    zero_rgb: tuple[float, float, float] | None = None,
    z_fraction: float = 0.55,
    pool_pixels_per_cell: float = 6.0,
    max_cells_across: int = 54,
    rim_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rim_width_px: float = 0.0,
    pool_cache: dict | None = None,
    pool_reference_width: int | None = None,
    origin: str = "lower",
) -> tuple[NDArray[np.uint8], HeightBarScene]:
    """Render the grid as boxes -> ((H, W, 4) uint8 RGBA, scene map).

    NaN heights are absent bars.  Bar heights anchor to ``value_limits``
    (the colour limits): values clip to them exactly as colours saturate,
    a negative low hangs the zero plane mid-scene, and the pane/z-axis
    span IS the limit span.  The floor and the two back panes are plain
    background surfaces: every LINE of the scene -- pane grids, floor
    grid, axes and bar outlines -- is vector chrome drawn by the caller,
    never raster pixels, so nothing here can alias.

    ``origin`` is the image origin the SAME grid is drawn under as a
    heatmap.  A 3D scene of an image is an oblique view of that picture,
    so the row axis must run the way the picture runs: with
    ``origin="upper"`` row 0 is at the TOP of the heatmap, which is the
    far side of the ground once the picture is tipped back, and the rows
    are reversed to put it there.  Rendering the array's own row order
    regardless mirrored the scene against the heatmap -- looking
    straight down at it showed the picture upside down, and the cell at
    the heatmap's bottom right stood at the far corner.
    """

    if supersample not in (1, 2, 3, 4):
        raise ValueError("supersample must be 1, 2, 3 or 4")
    if origin not in ("lower", "upper"):
        raise ValueError("origin must be 'lower' or 'upper'")
    flip_rows = origin == "upper"

    render_w = int(width) * supersample
    render_h = int(height) * supersample
    if render_w < 8 or render_h < 8:
        raise ValueError("height-bar raster needs at least 8x8 pixels")

    h_grid = np.asarray(heights, dtype=np.float64)
    rgb_grid = np.asarray(top_rgb, dtype=np.float32)
    if h_grid.ndim != 2 or rgb_grid.shape != (*h_grid.shape, 3):
        raise ValueError("heights must be (ny, nx) and top_rgb (ny, nx, 3)")
    source_ny, source_nx = h_grid.shape

    # ---- LOD: how many bars this picture has.
    #
    # There is ONE look -- boxes -- so density cannot decide what the
    # scene IS, only how finely it is divided.  Two facts bound that,
    # and both are about the drawn bar rather than the stored grid:
    #
    #   * a cell narrower than a few pixels is not a bar, so the panel's
    #     own width sets a floor on cell size, and
    #   * every bar costs the outline pass a handful of edges, so the BAR
    #     COUNT -- not the pixel count -- is what decides whether a live
    #     frame and a camera drag land inside a frame budget.  Measured on
    #     the console's own panel: 54 cells a side is 14.6 ms of outline
    #     work a frame, and it grows with the square of the count.
    #
    # The cap is where the old surface/box threshold stood, so every scene
    # that already drew as boxes still draws exactly the same boxes; what
    # changes is that a denser grid now pools INTO that look instead of
    # turning into a different one.
    #
    # Pooling depends only on the inputs and these bounds, never the
    # camera; the caller may hand a cache whose validity rides on INPUT
    # IDENTITY -- safe because the caller's own input cache keeps those
    # arrays alive and unchanged.  Deriving it from the transient render
    # width made the pooled grid -- and therefore its cache -- change
    # whenever a drag lowered the preview resolution, so every drag
    # re-pooled the whole scan (millions of cells) instead of reusing it.
    # The reference is the committed surface's width; a preview simply
    # draws that same pooled grid at fewer pixels.
    pool_width = int(render_w if pool_reference_width is None else pool_reference_width)
    limit = max(8, min(
        int(pool_width / max(pool_pixels_per_cell, 0.5)),
        int(max_cells_across),
    ))
    pool_key = (id(h_grid), id(rgb_grid), h_grid.shape, limit)
    if pool_cache is not None and pool_cache.get("key") == pool_key:
        h_grid, rgb_grid, finite_grid, pool_x, pool_y = pool_cache["value"]
    else:
        finite_grid = np.isfinite(h_grid)
        h_grid, rgb_grid, finite_grid, pool_x, pool_y = _pooled(
            h_grid, rgb_grid, finite_grid, limit
        )
        if pool_cache is not None:
            pool_cache["key"] = pool_key
            pool_cache["value"] = (
                h_grid, rgb_grid, finite_grid, pool_x, pool_y
            )

    # ---- the picture's own row direction, AFTER pooling so the pooled
    # blocks stay aligned with the source rows -- and so the pool cache,
    # whose key is the input arrays' identity, keeps hitting.
    pooled_rows = int(h_grid.shape[0])
    if flip_rows:
        h_grid = h_grid[::-1]
        rgb_grid = rgb_grid[::-1]
        finite_grid = finite_grid[::-1]

    # ---- fold the azimuth into [0, 90) by rotating the grid
    azimuth = math.radians(camera.azimuth_deg) % (2.0 * math.pi)
    quadrant = int(azimuth // (math.pi / 2.0))
    local = azimuth - quadrant * (math.pi / 2.0)
    if quadrant:
        # The fold is a ROTATION, not a flip: crossing a quadrant
        # boundary swaps which source axis runs along the projected
        # horizontal, and continuing the orbit smoothly requires the
        # grid to turn with it.  Flipping alone kept the axes in place,
        # so the scene snapped ninety degrees at every boundary.
        h_grid = np.rot90(h_grid, quadrant)
        rgb_grid = np.rot90(rgb_grid, quadrant, axes=(0, 1))
        finite_grid = np.rot90(finite_grid, quadrant)
    # AFTER the fold: odd quadrants swap the folded dimensions.
    ny, nx = h_grid.shape
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
    # The camera DISTANCE must not depend on where the orbit points: a
    # fit against the current footprint breathes as the projected box
    # widens and narrows with the azimuth (and flips the dense-surface
    # threshold with it), which read as the scene lurching nearer and
    # farther -- and lighting toggling -- during a drag.  The scale
    # therefore fits the azimuth-INVARIANT envelope: the footprint's
    # diagonal, which the projected extents reach at their widest.
    # Centring still tracks the actual box, so the scene stays centred
    # while its size holds still.
    diagonal = math.hypot(nx, ny)
    span_x = diagonal
    span_y = diagonal * se + pane_high - pane_low
    margin = 0.04 * max(span_x, span_y)
    scale = (
        min(
            render_w / (span_x + 2.0 * margin),
            render_h / (span_y + 2.0 * margin),
        )
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
    # ---- per-cell colour grids and z planes, shared by both engines.
    # Every plane here depends on the inputs, the limits, the QUADRANT
    # fold and the elevation -- never on the azimuth within a quadrant or
    # the zoom -- so an orbit drag reuses them verbatim: an identity
    # cache, bit-exact by construction.
    # Two cache stages.  Everything the ELEVATION never touches -- the
    # clipped height field, colours, lighting, validity -- caches under
    # the inputs/quadrant/limits alone; the elevation only scales hz
    # into the two z planes, so an elevation drag pays two cheap
    # multiplies instead of the whole derivation (which made vertical
    # orbiting feel dead next to snappy horizontal orbiting).
    # Every input the cached value depends on is named here.  The derived
    # grids are computed AFTER the row reversal, so the row direction is one
    # of those inputs -- a cache shared across two origins would otherwise
    # hand back the other picture's geometry.
    derived_key = (
        pool_key, flip_rows, quadrant, value_low, value_high, zero_rgb,
        float(z_fraction),
    )
    if pool_cache is not None and pool_cache.get("derived_key") == derived_key:
        (
            hz, finite_grid, rgb_grid, base_grid, top_values,
            base_values,
        ) = pool_cache["derived_value"]
    else:
        clipped = np.clip(h_grid, value_low, value_high)
        hz = np.ascontiguousarray(
            np.where(finite_grid, clipped, 0.0) * z_unit
        )
        if zero_rgb is None:
            base_grid = rgb_grid
        else:
            base_grid = np.ascontiguousarray(np.broadcast_to(
                np.asarray(zero_rgb, dtype=np.float32), rgb_grid.shape
            ))
        top_values = np.where(
            finite_grid, np.maximum(clipped, 0.0), -np.inf
        )
        base_values = np.where(
            finite_grid, np.minimum(clipped, 0.0), np.inf
        )
        finite_grid = np.ascontiguousarray(finite_grid)
        rgb_grid = np.ascontiguousarray(rgb_grid)
        base_grid = np.ascontiguousarray(base_grid)
        if pool_cache is not None:
            pool_cache["derived_key"] = derived_key
            pool_cache["derived_value"] = (
                hz, finite_grid, rgb_grid, base_grid, top_values,
                base_values,
            )
    z_key = (derived_key, float(ce))
    if pool_cache is not None and pool_cache.get("derived_z_key") == z_key:
        z_top32, z_bot32 = pool_cache["derived_z_value"]
    else:
        z_top32 = np.ascontiguousarray(
            (np.maximum(hz, 0.0) * ce).astype(np.float32)
        )
        z_bot32 = np.ascontiguousarray(
            (np.minimum(hz, 0.0) * ce).astype(np.float32)
        )
        if pool_cache is not None:
            pool_cache["derived_z_key"] = z_key
            pool_cache["derived_z_value"] = (z_top32, z_bot32)

    if _scanline_selected():
        # The numba analytic engine: exact vertical coverage per column
        # walk, bit-identical to the reference materialization below by
        # the standing contract test.
        from ._height3d_scanline import _materialize

        out = np.empty((render_h // supersample, int(width), 4), dtype=np.uint8)
        # One id per OUTPUT pixel, not per subcolumn.  The picking plane only
        # ever kept the middle tap, so writing all three and then gathering
        # every third column out of a thirty-three megabyte plane cost five
        # milliseconds a frame to throw two thirds of it away.
        id_taps = np.empty((render_h // supersample, int(width)), dtype=np.int32)
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
            np.int64(supersample),
            np.int64(render_h),
            out,
            id_taps,
            np.int64(supersample // 2),
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

        # ---- analytic coverage: span bounds stay FLOAT.  q counts
        # OUTPUT pixels (render rows / taps); a pixel's colour is the
        # exact integral of the piecewise-linear span colours across it,
        # so vertical anti-aliasing is analytic -- no row supersampling.
        rr_top = np.clip((y_high - hi) * scale, 0.0, float(render_h))
        rr_bot = np.clip((y_high - lo) * scale, 0.0, float(render_h))
        keep2 = rr_bot > rr_top
        cols = cols[keep2]
        rr_top = rr_top[keep2]
        rr_bot = rr_bot[keep2]
        color_low = color_low[keep2]
        color_high = color_high[keep2]
        full_lo = full_lo[keep2]
        full_hi = full_hi[keep2]
        ids = ids[keep2]

        # Gradient as intercept+slope in RENDER row space, from the
        # UNCLIPPED face: colour(row) = A + s*row.
        full_row_bottom = (y_high - full_lo) * scale
        full_row_top = (y_high - full_hi) * scale
        span_height = np.maximum(full_row_bottom - full_row_top, 1e-6)
        slope = (color_low - color_high) / span_height[:, None]
        intercept = color_high - slope * full_row_top[:, None]

        taps = supersample
        out_h = render_h // taps
        inv_taps = 1.0 / taps
        q0 = rr_top * inv_taps
        q1 = rr_bot * inv_taps
        # value255(q) = A255 + S255 * q; a full pixel p averages to
        # (A255 + 0.5*S255) + S255 * p -- the same two-plane prefix-sum
        # shape as ever, now over OUTPUT rows.
        s255 = slope * (255.0 * taps)
        a255 = intercept * 255.0
        a_mid = a255 + s255 * 0.5

        c0 = np.ceil(q0)
        c1 = np.floor(q1)
        c0i = c0.astype(np.int64)
        c1i = np.maximum(c1.astype(np.int64), c0i)
        # top partial: pixel floor(q0), covered [q0, min(c0, q1))
        p_top = np.floor(q0).astype(np.int64)
        top_end = np.minimum(c0, q1)
        f_top = np.maximum(top_end - q0, 0.0)
        m_top = (q0 + top_end) * 0.5
        # bottom partial: pixel floor(q1), covered [c1, q1), only when
        # distinct from the top partial's pixel
        p_bot = np.minimum(c1i, out_h - 1)
        f_bot = np.where(c1 >= c0, q1 - c1, 0.0)
        f_bot = np.maximum(f_bot, 0.0)
        m_bot = (c1 + q1) * 0.5

        stride = render_w
        plane = (out_h + 1) * stride
        flat_full0 = c0i * stride + cols
        flat_full1 = c1i * stride + cols
        channel_index = np.concatenate([
            flat_full0 * 3,
            flat_full0 * 3 + 1,
            flat_full0 * 3 + 2,
            flat_full1 * 3,
            flat_full1 * 3 + 1,
            flat_full1 * 3 + 2,
        ])
        a_weights = np.concatenate([
            a_mid[:, 0], a_mid[:, 1], a_mid[:, 2],
            -a_mid[:, 0], -a_mid[:, 1], -a_mid[:, 2],
        ]).astype(np.float64)
        s_weights = np.concatenate([
            s255[:, 0], s255[:, 1], s255[:, 2],
            -s255[:, 0], -s255[:, 1], -s255[:, 2],
        ]).astype(np.float64)
        cov_index = np.concatenate([flat_full0, flat_full1])
        ones = np.ones(cols.shape[0], dtype=np.float64)
        cov_weights = np.concatenate([ones, -ones])
        # partial extras: direct (post-prefix) contributions
        flat_top = np.minimum(p_top, out_h - 1) * stride + cols
        flat_bot = p_bot * stride + cols
        part_top = f_top[:, None] * (a255 + s255 * m_top[:, None])
        part_bot = f_bot[:, None] * (a255 + s255 * m_bot[:, None])
        extra_index = np.concatenate([
            flat_top * 3, flat_top * 3 + 1, flat_top * 3 + 2,
            flat_bot * 3, flat_bot * 3 + 1, flat_bot * 3 + 2,
        ])
        extra_weights = np.concatenate([
            part_top[:, 0], part_top[:, 1], part_top[:, 2],
            part_bot[:, 0], part_bot[:, 1], part_bot[:, 2],
        ]).astype(np.float64)
        extra_cov_index = np.concatenate([flat_top, flat_bot])
        extra_cov_weights = np.concatenate([f_top, f_bot])
        # id: the span covering each pixel CENTRE
        ic0 = np.ceil(q0 - 0.5).astype(np.int64)
        ic1 = np.maximum(np.ceil(q1 - 0.5).astype(np.int64), ic0)
        id_weights = ids.astype(np.float64)
        futures = [
            _pool().submit(
                lambda idx, w, length: np.bincount(
                    idx, weights=w, minlength=length
                ).astype(np.float32),
                *args,
            )
            for args in (
                (channel_index, a_weights, plane * 3),
                (channel_index, s_weights, plane * 3),
                (extra_index, extra_weights, plane * 3),
                (cov_index, cov_weights, plane),
                (extra_cov_index, extra_cov_weights, plane),
            )
        ]
        futures.append(_pool().submit(
            lambda: np.bincount(
                np.concatenate([ic0 * stride + cols, ic1 * stride + cols]),
                weights=np.concatenate([id_weights, -id_weights]),
                minlength=plane,
            ).astype(np.int32)
        ))
        (a_diff, s_diff, extra_rgb, cov_diff, extra_cov, id_diff) = (
            f.result() for f in futures
        )
        filled_full = np.empty((out_h + 1, stride * 3), dtype=np.float32)
        _cumsum_axis0(a_diff.reshape(out_h + 1, stride * 3), filled_full)
        filled = filled_full[:out_h].reshape(out_h, stride, 3)
        slope_full = np.empty((out_h + 1, stride * 3), dtype=np.float32)
        _cumsum_axis0(s_diff.reshape(out_h + 1, stride * 3), slope_full)
        slope_plane = slope_full[:out_h].reshape(out_h, stride, 3)
        slope_plane *= np.arange(out_h, dtype=np.float32)[:, None, None]
        filled += slope_plane
        filled += extra_rgb.reshape(out_h + 1, stride, 3)[:out_h]
        cov_full = np.empty((out_h + 1, stride), dtype=np.float32)
        _cumsum_axis0(cov_diff.reshape(out_h + 1, stride), cov_full)
        coverage = cov_full[:out_h]
        coverage += extra_cov.reshape(out_h + 1, stride)[:out_h]
        np.clip(coverage, 0.0, 1.0, out=coverage)
        id_plane_full = np.empty((out_h + 1, stride), dtype=np.int32)
        _cumsum_axis0(id_diff.reshape(out_h + 1, stride), id_plane_full)
        # The middle tap, taken here so both engines hand the tail the same
        # shape: one id per output pixel.
        id_taps = np.ascontiguousarray(
            id_plane_full[:out_h, (supersample // 2)::supersample]
        )

        # ---- horizontal: average the taps (sequential adds, exact
        # mirror territory), then complete uncovered fractions with the
        # background and convert once, round-half-up.
        rgb_acc = np.zeros((out_h, width, 3), dtype=np.float32)
        cov_acc = np.zeros((out_h, width), dtype=np.float32)
        for tap in range(taps):
            rgb_acc += filled[:, tap::taps, :]
            cov_acc += coverage[:, tap::taps]
        rgb_acc *= np.float32(1.0 / taps)
        cov_acc *= np.float32(1.0 / taps)
        background255 = (
            np.asarray(background_rgb, dtype=np.float32) * np.float32(255.0)
        )
        rgb_acc += (
            (np.float32(1.0) - cov_acc)[..., None] * background255[None, None]
        )
        np.clip(rgb_acc, 0.0, 255.0, out=rgb_acc)
        out = np.empty((out_h, width, 4), dtype=np.uint8)
        out[..., :3] = (rgb_acc + np.float32(0.5)).astype(np.uint8)
        # The frame is FINISHED: the line above already completed every
        # uncovered fraction with ``background_rgb``, so a partly covered
        # pixel already holds its blend.  Reporting the coverage as alpha
        # made the compositor apply it a second time -- an edge pixel came
        # out at bg + (face - bg) * coverage**2, visibly pale -- and denied
        # the exact blit, which then paid Matplotlib's whole image machinery
        # for a picture that was ready.
        out[..., 3] = 255

    taps = supersample
    out_h = render_h // taps
    scene_id_plane = id_taps
    scale = scale / taps
    # The creases of the drawn surface, drawn into it.  They belong to the
    # picture, not to the room, and they cost what they COVER rather than
    # what they belong to -- so a scene with fifty bars a side pays the
    # same two milliseconds as one with ten.
    _stroke_rims(out, scene_id_plane, rim_rgb, float(rim_width_px))

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
        pooled_rows=pooled_rows,
        flip_rows=flip_rows,
        z_unit=z_unit,
        value_low=value_low,
        value_high=value_high,
        width=int(width),
        height=int(height),
        id_plane=np.ascontiguousarray(scene_id_plane, dtype=np.int32),
        top_values=top_values,
        base_values=base_values,
    )
    return out, scene


__all__ = ["HeightBarCamera", "HeightBarScene", "render_height_bars"]
