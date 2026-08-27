"""Numba scanline back end for the height-bar raster: the CPU ceiling.

This module reimplements the reference kernel's materialization tail
(crossing merge -> spans -> gradient scatter -> pixels) as a per-column
scanline walk compiled with numba.  It exists for speed only: the numpy
implementation in ``_height3d_raster`` remains the SPECIFICATION, and a
standing contract test renders both and asserts bit equality -- every
float32/float64 conversion point, every summation order and every
truncation here mirrors the reference operation for operation, so the
two engines cannot drift apart silently.

Why this is faster: the reference materializes through full-plane
passes (scatter, prefix sums, gradient multiply-add) that stream
hundreds of megabytes per frame; the scanline walk touches each output
pixel once, with the per-column bookkeeping in registers and small
per-chunk scratch.

Compilation caches on disk under ``ZLC_NUMBA_CACHE`` (default: the
repository's ``.numba_cache``, see ``bin/warm_numba_cache.bat``): the
first call on a fresh machine compiles once; afterwards every process
loads machine code in milliseconds.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np

if "NUMBA_CACHE_DIR" not in os.environ:
    _repo_root = pathlib.Path(__file__).resolve().parents[4]
    os.environ["NUMBA_CACHE_DIR"] = str(_repo_root / ".numba_cache")

try:  # pragma: no cover - absence is exercised by the engine fallback
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrap(fn):
            return fn

        return wrap

    prange = range  # type: ignore[assignment]


@njit(cache=True, parallel=True, nogil=True)
def _occlusion_samples(  # one edge set, sampled against one scene
    edges,       # f64 (E, 2, 3) folded (a, b, value) endpoints
    id_plane,    # i32 (H, W)
    top_values,  # f64 (ny, nx)
    ca,          # f64
    sa,          # f64
    se,          # f64
    z_unit,      # f64
    ce,          # f64
    x_low,       # f64
    y_high,      # f64
    scale,       # f64
    samples,     # i64 per edge
    xs,          # f64 (E * (samples + 1),) written
    ys,          # f64 (E * (samples + 1),) written
):
    """The occlusion sampler's math, fused per edge.

    Mirrors ``_height_bars_sampled_polyline`` operation for operation
    (pure float64 with integer lookups, so the mirror is direct): the
    projection, the shown box's ahead-and-below test, the inside-solid
    test with viewer-side cells, and the isolated-sample erosion.
    """

    E = edges.shape[0]
    height = id_plane.shape[0]
    width = id_plane.shape[1]
    ny = top_values.shape[0]
    nx = top_values.shape[1]
    z_slack = 0.5 / max(z_unit * ce * scale, 1e-9)
    rise = se / (z_unit * ce)
    stride = samples + 1
    # One edge knows nothing about another: it reads the finished scene
    # and writes its own slice of the output.  The walk was serial while
    # a small ROI drawn as boxes spent twelve milliseconds a frame in it,
    # which is most of what made "not many bars" feel heavy.  Splitting by
    # edge changes no arithmetic -- every sample is computed exactly where
    # it was, in the same order within its edge.
    for e in prange(E):
        a0 = edges[e, 0, 0]
        b0 = edges[e, 0, 1]
        z0 = edges[e, 0, 2]
        da = edges[e, 1, 0] - a0
        db = edges[e, 1, 1] - b0
        dz = edges[e, 1, 2] - z0
        base = e * stride
        # np.linspace computes delta once and multiplies -- mirror that,
        # including the forced exact endpoint.
        delta = 1.0 / (samples - 1.0)
        for n in range(samples):
            if n == samples - 1:
                fraction = 1.0
            else:
                fraction = n * delta
            ga = a0 + da * fraction
            gb = b0 + db * fraction
            gz = z0 + dz * fraction
            sx = ga * ca + gb * sa
            sy = -ga * sa * se + gb * ca * se + gz * z_unit * ce
            px = (sx - x_low) * scale
            py = (y_high - sy) * scale
            column = np.int64(px)
            if column < 0:
                column = 0
            elif column > width - 1:
                column = width - 1
            row = np.int64(py)
            if row < 0:
                row = 0
            elif row > height - 1:
                row = height - 1
            face = id_plane[row, column]
            hidden = False
            if face >= 4:
                shown = (face - 4) >> 2
                shown_a = shown % nx
                shown_b = shown // nx
                sa_i = shown_a
                if sa_i < 0:
                    sa_i = 0
                elif sa_i > nx - 1:
                    sa_i = nx - 1
                sb_i = shown_b
                if sb_i < 0:
                    sb_i = 0
                elif sb_i > ny - 1:
                    sb_i = ny - 1
                shown_top = top_values[sb_i, sa_i]
                enter_a = (shown_a - ga) / sa
                enter_b = (gb - shown_b - 1) / ca
                enter = enter_a
                if enter_b > enter:
                    enter = enter_b
                if enter < 0.0:
                    enter = 0.0
                leave_a = (shown_a + 1 - ga) / sa
                leave_b = (gb - shown_b) / ca
                leave = leave_a
                if leave_b < leave:
                    leave = leave_b
                if leave > 1e-9 and gz + enter * rise < shown_top - z_slack:
                    hidden = True
            if not hidden:
                cell_a = np.int64(np.floor(ga))
                cell_b = np.int64(np.ceil(gb)) - 1
                if 0 <= cell_a < nx and 0 <= cell_b < ny:
                    if gz < top_values[cell_b, cell_a] - z_slack:
                        hidden = True
            if hidden:
                xs[base + n] = np.nan
                ys[base + n] = np.nan
            else:
                xs[base + n] = px / max(width, 1)
                ys[base + n] = 1.0 - py / max(height, 1)
        xs[base + samples] = np.nan
        ys[base + samples] = np.nan
        # isolated-visible erosion, exactly the vectorized rule: a
        # visible sample with hidden neighbours on BOTH sides (edge
        # boundaries count as hidden) becomes hidden.
        for n in range(samples):
            index = base + n
            if not np.isnan(xs[index]):
                left_hidden = n == 0 or np.isnan(xs[index - 1])
                right_hidden = n == samples - 1 or np.isnan(xs[index + 1])
                if left_hidden and right_hidden:
                    # defer to keep neighbour reads pristine
                    ys[index] = np.inf  # mark
        for n in range(samples):
            index = base + n
            if ys[index] == np.inf:
                xs[index] = np.nan
                ys[index] = np.nan


@njit(cache=True, parallel=True, nogil=True)
def _rim_stroke(  # find the creases, stamp them, and blend, in one pass
    id_plane,    # i32 (H, W)
    weights,     # f32 (2R+1, 2R+1)
    radius,      # i64
    bands,       # i64 row bands to split over
    out,         # u8 (H, W, 4) blended in place
    target,      # f32 (3,) rim colour in 0..255
):
    """Mirror of the reference, fused, scattered, and blended in place.

    The reference asks every pixel what reaches it; a crease covers about
    a tenth of the plane, so asking is nine times the work of telling.
    Each band owns its rows and writes only inside them -- it re-reads the
    halo rows rather than sharing them -- so the scatter needs no atomics
    and the maximum over one pixel is still taken by one thread.  A maximum
    over the same finite set in any order is the same number.

    The blend lives here too.  Handing the coverage back and blending the
    pixels it touched through a boolean index cost sixteen milliseconds on
    an 815-pixel preview and forty-three on the committed frame, against
    half a millisecond of actual stamping.
    """

    height = id_plane.shape[0]
    width = id_plane.shape[1]
    band_rows = (height + bands - 1) // bands
    for band in prange(bands):
        lo = band * band_rows
        if lo >= height:
            continue
        hi = lo + band_rows
        if hi > height:
            hi = height
        coverage = np.zeros((hi - lo, width), dtype=np.float32)
        sy0 = lo - radius
        if sy0 < 0:
            sy0 = 0
        sy1 = hi + radius
        if sy1 > height:
            sy1 = height
        for sy in range(sy0, sy1):
            for sx in range(width):
                face = id_plane[sy, sx]
                crease = False
                if sx + 1 < width:
                    other = id_plane[sy, sx + 1]
                    if other != face and (face >= 4 or other >= 4):
                        crease = True
                if not crease and sx > 0:
                    other = id_plane[sy, sx - 1]
                    if other != face and (face >= 4 or other >= 4):
                        crease = True
                if not crease and sy + 1 < height:
                    other = id_plane[sy + 1, sx]
                    if other != face and (face >= 4 or other >= 4):
                        crease = True
                if not crease and sy > 0:
                    other = id_plane[sy - 1, sx]
                    if other != face and (face >= 4 or other >= 4):
                        crease = True
                if not crease:
                    continue
                ty0 = sy - radius
                if ty0 < lo:
                    ty0 = lo
                ty1 = sy + radius + 1
                if ty1 > hi:
                    ty1 = hi
                tx0 = sx - radius
                if tx0 < 0:
                    tx0 = 0
                tx1 = sx + radius + 1
                if tx1 > width:
                    tx1 = width
                for ty in range(ty0, ty1):
                    for tx in range(tx0, tx1):
                        weight = weights[sy - ty + radius, sx - tx + radius]
                        if weight > coverage[ty - lo, tx]:
                            coverage[ty - lo, tx] = weight
        for ty in range(lo, hi):
            for tx in range(width):
                fraction = coverage[ty - lo, tx]
                if fraction <= np.float32(0.0):
                    continue
                for channel in range(3):
                    painted = np.float32(out[ty, tx, channel])
                    value = (
                        painted
                        + (target[channel] - painted) * fraction
                        + np.float32(0.5)
                    )
                    out[ty, tx, channel] = np.uint8(value)


def warm(force: bool = False) -> str:
    """Compile (or load) the kernel's disk cache; returns the outcome.

    The compiled signature is CLOSED: the driver normalizes every dtype
    and layout at the kernel boundary, so one representative render
    covers production forever.  A marker file remembers the toolchain
    versions and this module's source hash -- when they match and the
    cache is populated, warming is a millisecond no-op, which is what
    ``bin\\warm_numba_cache.bat`` relies on to skip rebuilds.
    """

    if not HAVE_NUMBA:
        return "numba is not installed; the numpy reference engine runs"
    import hashlib
    import sys

    import numba

    cache_dir = pathlib.Path(os.environ["NUMBA_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = pathlib.Path(__file__).read_bytes()
    fingerprint = "|".join((
        sys.version.split()[0],
        np.__version__,
        numba.__version__,
        hashlib.sha256(source).hexdigest(),
    ))
    marker = cache_dir / "zlc_height3d.marker"
    populated = any(cache_dir.glob("**/*.nbc"))
    if (
        not force
        and populated
        and marker.exists()
        and marker.read_text(encoding="utf-8") == fingerprint
    ):
        return "cache is current; nothing to do"
    from .import _height3d_raster as raster

    previous = raster._ENGINE
    raster._ENGINE = "numba"
    try:
        heights = np.linspace(0.0, 1.0, 12).reshape(3, 4)
        colors = np.full((3, 4, 3), 0.5, dtype=np.float32)
        raster.render_height_bars(
            heights,
            colors,
            camera=raster.HeightBarCamera(),
            value_limits=(0.0, 1.0),
            width=64,
            height=48,
            supersample=2,
        )
    finally:
        raster._ENGINE = previous
    marker.write_text(fingerprint, encoding="utf-8")
    return "kernel compiled and cached"


def main() -> int:
    """``zlc warm_numba``: compile-or-verify the kernel cache, say which."""

    print(warm())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


@njit(cache=True, parallel=True, nogil=True)
def _derive_planes(  # the reference derivation, fused into one walk
    h_grid,      # f64 (ny, nx)
    finite_grid, # bool (ny, nx)
    value_low,   # f64
    value_high,  # f64
    z_unit,      # f64
    span,        # f64  256 / (value_high - value_low)
    lut,         # f32 (256, 3) the colormap's own table
    hz,          # f64 (ny, nx) written
    top_values,  # f64 (ny, nx) written
    base_values, # f64 (ny, nx) written
    rgb,         # f32 (ny, nx, 3) written
):
    """Everything one cell decides, in one pass.

    Mirrors the reference in ``_height3d_raster`` operation for operation:
    clip, the finite branch, and the colour code the heatmap picks -- the
    scaled value, clipped to the table and truncated, with a value that is
    not a number taking the first entry.  It exists because a LIVE panel
    rebuilds all of this on every shot, and as separate numpy statements
    it is a dozen passes over the grid to compute what one cell knows.
    """

    ny, nx = h_grid.shape
    for row in prange(ny):
        for column in range(nx):
            clipped = h_grid[row, column]
            if clipped < value_low:
                clipped = value_low
            if clipped > value_high:
                clipped = value_high
            if finite_grid[row, column]:
                hz[row, column] = clipped * z_unit
                top_values[row, column] = clipped if clipped > 0.0 else 0.0
                base_values[row, column] = clipped if clipped < 0.0 else 0.0
            else:
                hz[row, column] = 0.0
                top_values[row, column] = -np.inf
                base_values[row, column] = np.inf
            code_value = (clipped - value_low) * span
            if code_value != code_value:
                code = 0
            else:
                if code_value < 0.0:
                    code_value = 0.0
                elif code_value > 255.0:
                    code_value = 255.0
                code = int(code_value)
            rgb[row, column, 0] = lut[code, 0]
            rgb[row, column, 1] = lut[code, 1]
            rgb[row, column, 2] = lut[code, 2]


@njit(cache=True, parallel=True, nogil=True)
def _derive_z_planes(
    hz,          # f64 (ny, nx)
    ce,          # f64
    z_top32,     # f32 (ny, nx) written
    z_bot32,     # f32 (ny, nx) written
):
    """The two elevation-scaled planes, kept apart so a vertical orbit
    pays one cheap pass instead of the whole derivation."""

    ny, nx = hz.shape
    for row in prange(ny):
        for column in range(nx):
            scaled = hz[row, column] * ce
            z_top32[row, column] = np.float32(scaled if scaled > 0.0 else 0.0)
            z_bot32[row, column] = np.float32(scaled if scaled < 0.0 else 0.0)


@njit(cache=True, parallel=True, nogil=True)
def _materialize(  # noqa: C901 - one kernel, mirrored from the reference
    a_at,        # f64 (render_w,)
    t_enter,     # f64 (render_w,)
    t_exit,      # f64 (render_w,)
    ia0_arr,     # f64 (render_w,)
    ib0_arr,     # f64 (render_w,)
    t_a0,        # f64 (render_w,)
    t_b0,        # f64 (render_w,)
    g0_arr,      # f32 (render_w,)
    z_top32,     # f32 (ny, nx)
    z_bot32,     # f32 (ny, nx)
    finite_grid, # bool (ny, nx)
    rgb_grid,    # f32 (ny, nx, 3) bar colours
    base_grid,   # f32 (ny, nx, 3) zero-end colours
    shade_x,     # f32
    shade_y,     # f32
    sa,          # f64
    ca,          # f64
    se32,        # f32
    y_high,      # f64
    scale,       # f64  (render-resolution scale)
    pane_high32, # f32
    pane_low32,  # f32
    background32,  # f32 (3,) background colour in 0..1
    taps,        # i64  horizontal subcolumns per output pixel
    render_h,    # i64  render rows (out_h * taps)
    out,         # u8 (out_h, out_w, 4)   written
    id_taps,     # i32 (out_h, out_w)  written
    mid_tap,     # i64  the subcolumn whose ids the picking plane keeps
    n_chunks,    # i64
):
    """Analytic-coverage materializer, mirroring the numpy reference.

    Vertical anti-aliasing is exact: every span keeps FLOAT bounds and a
    pixel integrates the piecewise-linear span colours crossing it.
    Horizontal anti-aliasing samples ``taps`` subcolumns per output
    pixel.  Accumulation order, every float32 conversion point and the
    round-half-up conversion mirror the reference operation for
    operation -- the standing parity test keeps them bit-equal.
    """

    render_w = a_at.shape[0]
    out_h = out.shape[0]
    out_w = out.shape[1]
    ny = z_top32.shape[0]
    nx = z_top32.shape[1]
    S = nx + ny
    max_spans = 3 * S + 1
    chunk = (out_w + n_chunks - 1) // n_chunks
    inv_taps64 = 1.0 / taps
    t255 = 255.0 * taps
    inv_taps32 = np.float32(1.0 / taps)
    bg255_0 = background32[0] * np.float32(255.0)
    bg255_1 = background32[1] * np.float32(255.0)
    bg255_2 = background32[2] * np.float32(255.0)
    for chunk_index in prange(n_chunks):
        col0 = chunk_index * chunk
        col1 = min(col0 + chunk, out_w)
        if col0 >= col1:
            continue
        sorted_t32 = np.empty(S, dtype=np.float32)
        is_b = np.empty(S, dtype=np.bool_)
        used = np.empty(S, dtype=np.bool_)
        seg_t0 = np.empty(S, dtype=np.float32)
        seg_t1 = np.empty(S, dtype=np.float64)
        seg_ca = np.empty(S, dtype=np.int64)
        seg_cb = np.empty(S, dtype=np.int64)
        seg_inside = np.empty(S, dtype=np.bool_)
        seg_ok = np.empty(S, dtype=np.bool_)
        seg_glo = np.empty(S, dtype=np.float32)
        seg_ghi = np.empty(S, dtype=np.float64)
        seg_wm = np.empty(S, dtype=np.float64)
        seg_ent = np.empty(S, dtype=np.bool_)
        rec_q0 = np.empty(max_spans, dtype=np.float64)
        rec_q1 = np.empty(max_spans, dtype=np.float64)
        rec_a = np.empty((max_spans, 3), dtype=np.float64)
        rec_s = np.empty((max_spans, 3), dtype=np.float64)
        rec_id = np.empty(max_spans, dtype=np.int64)
        diff_a = np.zeros((out_h + 1, 3), dtype=np.float64)
        diff_s = np.zeros((out_h + 1, 3), dtype=np.float64)
        diff_cov = np.zeros(out_h + 1, dtype=np.float64)
        extra_rgb = np.zeros((out_h + 1, 3), dtype=np.float64)
        extra_cov = np.zeros(out_h + 1, dtype=np.float64)
        diff_id = np.zeros(out_h + 1, dtype=np.int64)
        rgb_acc = np.empty((out_h, 3), dtype=np.float32)
        cov_acc = np.empty(out_h, dtype=np.float32)
        for out_col in range(col0, col1):
            for row in range(out_h):
                rgb_acc[row, 0] = 0.0
                rgb_acc[row, 1] = 0.0
                rgb_acc[row, 2] = 0.0
                cov_acc[row] = 0.0
            for tap in range(taps):
                column = out_col * taps + tap
                enter = t_enter[column]
                texit = t_exit[column]
                ia0c = np.int64(ia0_arr[column])
                ib0c = np.int64(ib0_arr[column])
                ta0 = t_a0[column]
                tb0 = t_b0[column]
                g0c = g0_arr[column]
                # ---- crossings: a slots by the closed form, b slots as
                # the integer complement (permutation by construction).
                for s in range(S):
                    used[s] = False
                    is_b[s] = False
                for k in range(nx):
                    t_cross = ta0 + k / sa
                    before = np.ceil((t_cross - tb0) * ca)
                    if before < 0.0:
                        before = 0.0
                    elif before > ny:
                        before = np.float64(ny)
                    slot = k + np.int64(before)
                    sorted_t32[slot] = np.float32(t_cross)
                    used[slot] = True
                cursor = 0
                for m in range(ny):
                    while used[cursor]:
                        cursor += 1
                    sorted_t32[cursor] = np.float32(tb0 + m / ca)
                    used[cursor] = True
                    is_b[cursor] = True
                    cursor += 1
                # ---- segment sweep
                previous32 = np.float32(enter)
                running = np.float64(-np.inf)
                before_b = 0
                for s in range(S):
                    v0 = np.float64(previous32)
                    if v0 > texit:
                        v0 = texit
                    seg_t0[s] = np.float32(v0)
                    v1 = np.float64(sorted_t32[s])
                    if v1 > texit:
                        v1 = texit
                    seg_t1[s] = v1
                    previous32 = sorted_t32[s]
                    cell_ia = ia0c - (s - before_b)
                    cell_ib = ib0c + before_b
                    inside = (
                        cell_ia >= 0
                        and cell_ia < nx
                        and cell_ib >= 0
                        and cell_ib < ny
                        and seg_t1[s] > np.float64(seg_t0[s])
                    )
                    cai = cell_ia
                    if cai < 0:
                        cai = 0
                    elif cai > nx - 1:
                        cai = nx - 1
                    cbi = cell_ib
                    if cbi < 0:
                        cbi = 0
                    elif cbi > ny - 1:
                        cbi = ny - 1
                    seg_ca[s] = cai
                    seg_cb[s] = cbi
                    seg_inside[s] = inside
                    ok = inside and finite_grid[cbi, cai]
                    seg_ok[s] = ok
                    seg_glo[s] = g0c + seg_t0[s] * se32
                    seg_ghi[s] = (
                        np.float64(g0c) + seg_t1[s] * np.float64(se32)
                    )
                    if s == 0:
                        seg_ent[s] = (
                            abs(a_at[column] - np.rint(a_at[column])) < 1e-9
                        )
                    else:
                        seg_ent[s] = not is_b[s - 1]
                    seg_wm[s] = running
                    if ok:
                        sil = seg_ghi[s] + np.float64(z_top32[cbi, cai])
                        if sil > running:
                            running = sil
                    if is_b[s]:
                        before_b += 1
                final_wm = running
                # ---- span records, class-major, keeping FLOAT bounds
                count = 0
                for s in range(S):  # side faces
                    cai = seg_ca[s]
                    cbi = seg_cb[s]
                    cell_top = z_top32[cbi, cai]
                    cell_bot = z_bot32[cbi, cai]
                    lo = np.float64(seg_glo[s] + cell_bot)
                    if lo < seg_wm[s]:
                        lo = seg_wm[s]
                    if seg_ok[s]:
                        hi = np.float64(seg_glo[s] + cell_top)
                    else:
                        hi = np.float64(-np.inf)
                    if not hi > lo:
                        continue
                    rr_top = (y_high - hi) * scale
                    if rr_top < 0.0:
                        rr_top = 0.0
                    elif rr_top > render_h:
                        rr_top = np.float64(render_h)
                    rr_bot = (y_high - lo) * scale
                    if rr_bot < 0.0:
                        rr_bot = 0.0
                    elif rr_bot > render_h:
                        rr_bot = np.float64(render_h)
                    if not rr_bot > rr_top:
                        continue
                    full_lo = seg_glo[s] + cell_bot
                    full_hi = seg_glo[s] + cell_top
                    fr_bottom = (y_high - np.float64(full_lo)) * scale
                    fr_top = (y_high - np.float64(full_hi)) * scale
                    height = fr_bottom - fr_top
                    if height < 1e-6:
                        height = 1e-6
                    if seg_ent[s]:
                        shade = shade_x
                    else:
                        shade = shade_y
                    positive = z_top32[cbi, cai] > np.float32(0.0)
                    for ch in range(3):
                        if positive:
                            c_hi = rgb_grid[cbi, cai, ch] * shade
                            c_lo = base_grid[cbi, cai, ch] * shade
                        else:
                            c_hi = base_grid[cbi, cai, ch] * shade
                            c_lo = rgb_grid[cbi, cai, ch] * shade
                        d = c_lo - c_hi
                        slp = np.float64(d) / height
                        rec_s[count, ch] = slp
                        rec_a[count, ch] = np.float64(c_hi) - slp * fr_top
                    rec_q0[count] = rr_top * inv_taps64
                    rec_q1[count] = rr_bot * inv_taps64
                    rec_id[count] = (
                        (seg_cb[s] * nx + seg_ca[s]) * 4
                        + 4
                        + 1
                        + (1 if seg_ent[s] else 0)
                    )
                    count += 1
                for s in range(S):  # top faces
                    cai = seg_ca[s]
                    cbi = seg_cb[s]
                    cell_top = z_top32[cbi, cai]
                    if seg_ok[s]:
                        lo = np.float64(seg_glo[s] + cell_top)
                    else:
                        lo = np.float64(np.inf)
                    if lo < seg_wm[s]:
                        lo = seg_wm[s]
                    if seg_ok[s]:
                        hi = seg_ghi[s] + np.float64(cell_top)
                    else:
                        hi = np.float64(-np.inf)
                    if not hi > lo:
                        continue
                    rr_top = (y_high - hi) * scale
                    if rr_top < 0.0:
                        rr_top = 0.0
                    elif rr_top > render_h:
                        rr_top = np.float64(render_h)
                    rr_bot = (y_high - lo) * scale
                    if rr_bot < 0.0:
                        rr_bot = 0.0
                    elif rr_bot > render_h:
                        rr_bot = np.float64(render_h)
                    if not rr_bot > rr_top:
                        continue
                    fr_bottom = (y_high - lo) * scale
                    fr_top = (y_high - hi) * scale
                    height = fr_bottom - fr_top
                    if height < 1e-6:
                        height = 1e-6
                    for ch in range(3):
                        c_val = rgb_grid[cbi, cai, ch]
                        d = c_val - c_val
                        slp = np.float64(d) / height
                        rec_s[count, ch] = slp
                        rec_a[count, ch] = np.float64(c_val) - slp * fr_top
                    rec_q0[count] = rr_top * inv_taps64
                    rec_q1[count] = rr_bot * inv_taps64
                    rec_id[count] = (seg_cb[s] * nx + seg_ca[s]) * 4 + 4 + 3
                    count += 1
                for s in range(S):  # floor
                    lo = np.float64(seg_glo[s])
                    if lo < seg_wm[s]:
                        lo = seg_wm[s]
                    if seg_inside[s] and not seg_ok[s]:
                        hi = seg_ghi[s]
                    else:
                        hi = np.float64(-np.inf)
                    if not hi > lo:
                        continue
                    rr_top = (y_high - hi) * scale
                    if rr_top < 0.0:
                        rr_top = 0.0
                    elif rr_top > render_h:
                        rr_top = np.float64(render_h)
                    rr_bot = (y_high - lo) * scale
                    if rr_bot < 0.0:
                        rr_bot = 0.0
                    elif rr_bot > render_h:
                        rr_bot = np.float64(render_h)
                    if not rr_bot > rr_top:
                        continue
                    fr_bottom = (y_high - lo) * scale
                    fr_top = (y_high - hi) * scale
                    height = fr_bottom - fr_top
                    if height < 1e-6:
                        height = 1e-6
                    for ch in range(3):
                        c_val = background32[ch]
                        slp = np.float64(c_val - c_val) / height
                        rec_s[count, ch] = slp
                        rec_a[count, ch] = np.float64(c_val) - slp * fr_top
                    rec_q0[count] = rr_top * inv_taps64
                    rec_q1[count] = rr_bot * inv_taps64
                    rec_id[count] = 1
                    count += 1
                g_exit = g0c + np.float32(texit) * se32
                pane_visible = texit > enter
                if pane_visible:
                    pane_lo = np.float64(g_exit + pane_low32)
                    if pane_lo < final_wm:
                        pane_lo = final_wm
                    pane_hi = np.float64(g_exit + pane_high32)
                else:
                    pane_lo = np.inf
                    pane_hi = -np.inf
                if pane_hi > pane_lo:
                    rr_top = (y_high - pane_hi) * scale
                    if rr_top < 0.0:
                        rr_top = 0.0
                    elif rr_top > render_h:
                        rr_top = np.float64(render_h)
                    rr_bot = (y_high - pane_lo) * scale
                    if rr_bot < 0.0:
                        rr_bot = 0.0
                    elif rr_bot > render_h:
                        rr_bot = np.float64(render_h)
                    if rr_bot > rr_top:
                        fr_bottom = (y_high - pane_lo) * scale
                        fr_top = (y_high - pane_hi) * scale
                        height = fr_bottom - fr_top
                        if height < 1e-6:
                            height = 1e-6
                        for ch in range(3):
                            c_val = background32[ch]
                            slp = np.float64(c_val - c_val) / height
                            rec_s[count, ch] = slp
                            rec_a[count, ch] = (
                                np.float64(c_val) - slp * fr_top
                            )
                        rec_q0[count] = rr_top * inv_taps64
                        rec_q1[count] = rr_bot * inv_taps64
                        rec_id[count] = 2
                        count += 1
                # ---- scatter, mirroring the reference's bincount slot
                # order: every + entry before every - entry per plane.
                for record in range(count):
                    q0 = rec_q0[record]
                    c0 = np.ceil(q0)
                    c0i = np.int64(c0)
                    for ch in range(3):
                        s255 = rec_s[record, ch] * t255
                        a_mid = rec_a[record, ch] * 255.0 + s255 * 0.5
                        diff_a[c0i, ch] += a_mid
                        diff_s[c0i, ch] += s255
                    diff_cov[c0i] += 1.0
                for record in range(count):
                    q0 = rec_q0[record]
                    q1 = rec_q1[record]
                    c0i = np.int64(np.ceil(q0))
                    c1i = np.int64(np.floor(q1))
                    if c1i < c0i:
                        c1i = c0i
                    for ch in range(3):
                        s255 = rec_s[record, ch] * t255
                        a_mid = rec_a[record, ch] * 255.0 + s255 * 0.5
                        diff_a[c1i, ch] += -a_mid
                        diff_s[c1i, ch] += -s255
                    diff_cov[c1i] += -1.0
                # partial extras: all top contributions, then all bottom
                for record in range(count):
                    q0 = rec_q0[record]
                    q1 = rec_q1[record]
                    c0 = np.ceil(q0)
                    p_top = np.int64(np.floor(q0))
                    if p_top > out_h - 1:
                        p_top = out_h - 1
                    top_end = min(c0, q1)
                    f_top = max(top_end - q0, 0.0)
                    m_top = (q0 + top_end) * 0.5
                    for ch in range(3):
                        s255 = rec_s[record, ch] * t255
                        a255 = rec_a[record, ch] * 255.0
                        extra_rgb[p_top, ch] += f_top * (a255 + s255 * m_top)
                    extra_cov[p_top] += f_top
                for record in range(count):
                    q0 = rec_q0[record]
                    q1 = rec_q1[record]
                    c0 = np.ceil(q0)
                    c1 = np.floor(q1)
                    c1i = np.int64(c1)
                    if c1i > out_h - 1:
                        c1i = out_h - 1
                    if c1 >= c0:
                        f_bot = q1 - c1
                        if f_bot < 0.0:
                            f_bot = 0.0
                    else:
                        f_bot = 0.0
                    m_bot = (c1 + q1) * 0.5
                    for ch in range(3):
                        s255 = rec_s[record, ch] * t255
                        a255 = rec_a[record, ch] * 255.0
                        extra_rgb[c1i, ch] += f_bot * (a255 + s255 * m_bot)
                    extra_cov[c1i] += f_bot
                # id diffs at pixel-centre coverage
                for record in range(count):
                    ic0 = np.int64(np.ceil(rec_q0[record] - 0.5))
                    diff_id[ic0] += rec_id[record]
                for record in range(count):
                    ic0 = np.int64(np.ceil(rec_q0[record] - 0.5))
                    ic1 = np.int64(np.ceil(rec_q1[record] - 0.5))
                    if ic1 < ic0:
                        ic1 = ic0
                    diff_id[ic1] += -rec_id[record]
                # ---- where this subcolumn's picture starts.  Every
                # accumulator is zero until the first row a span touches, so
                # the rows above it add zero to zero and write background --
                # exactly what they hold already.  A bar scene is mostly sky,
                # and the walk was paying full height for every one of five
                # thousand subcolumns.  Skipping is bit-exact: adding 0.0
                # changes no float.
                first_row = out_h
                for record in range(count):
                    q0 = rec_q0[record]
                    top = np.int64(np.ceil(q0))
                    floor_top = np.int64(np.floor(q0))
                    if floor_top < top:
                        top = floor_top
                    centre = np.int64(np.ceil(q0 - 0.5))
                    if centre < top:
                        top = centre
                    if top < 0:
                        top = np.int64(0)
                    if top < first_row:
                        first_row = top
                # ---- the walk: float32 prefix sums exactly as the
                # reference's cast-then-cumsum planes.
                acc_a0 = np.float32(0.0)
                acc_a1 = np.float32(0.0)
                acc_a2 = np.float32(0.0)
                acc_s0 = np.float32(0.0)
                acc_s1 = np.float32(0.0)
                acc_s2 = np.float32(0.0)
                acc_c = np.float32(0.0)
                acc_id = np.int64(0)
                if tap == mid_tap:
                    for row in range(first_row):
                        id_taps[row, out_col] = np.int32(0)
                for row in range(first_row, out_h):
                    acc_a0 += np.float32(diff_a[row, 0])
                    acc_a1 += np.float32(diff_a[row, 1])
                    acc_a2 += np.float32(diff_a[row, 2])
                    acc_s0 += np.float32(diff_s[row, 0])
                    acc_s1 += np.float32(diff_s[row, 1])
                    acc_s2 += np.float32(diff_s[row, 2])
                    acc_c += np.float32(diff_cov[row])
                    acc_id += diff_id[row]
                    if tap == mid_tap:
                        id_taps[row, out_col] = np.int32(acc_id)
                    row32 = np.float32(row)
                    v0 = acc_a0 + acc_s0 * row32
                    v0 = v0 + np.float32(extra_rgb[row, 0])
                    v1 = acc_a1 + acc_s1 * row32
                    v1 = v1 + np.float32(extra_rgb[row, 1])
                    v2 = acc_a2 + acc_s2 * row32
                    v2 = v2 + np.float32(extra_rgb[row, 2])
                    c = acc_c + np.float32(extra_cov[row])
                    if c < 0.0:
                        c = np.float32(0.0)
                    elif c > 1.0:
                        c = np.float32(1.0)
                    rgb_acc[row, 0] += v0
                    rgb_acc[row, 1] += v1
                    rgb_acc[row, 2] += v2
                    cov_acc[row] += c
                # ---- re-zero touched rows
                for record in range(count):
                    q0 = rec_q0[record]
                    q1 = rec_q1[record]
                    c0i = np.int64(np.ceil(q0))
                    c1i = np.int64(np.floor(q1))
                    if c1i < c0i:
                        c1i = c0i
                    p_top = np.int64(np.floor(q0))
                    if p_top > out_h - 1:
                        p_top = out_h - 1
                    p_bot = c1i
                    if p_bot > out_h - 1:
                        p_bot = out_h - 1
                    ic0 = np.int64(np.ceil(q0 - 0.5))
                    ic1 = np.int64(np.ceil(q1 - 0.5))
                    if ic1 < ic0:
                        ic1 = ic0
                    for row in (c0i, c1i, p_top, p_bot, ic0, ic1):
                        diff_a[row, 0] = 0.0
                        diff_a[row, 1] = 0.0
                        diff_a[row, 2] = 0.0
                        diff_s[row, 0] = 0.0
                        diff_s[row, 1] = 0.0
                        diff_s[row, 2] = 0.0
                        diff_cov[row] = 0.0
                        extra_rgb[row, 0] = 0.0
                        extra_rgb[row, 1] = 0.0
                        extra_rgb[row, 2] = 0.0
                        extra_cov[row] = 0.0
                        diff_id[row] = 0
            # ---- combine taps, complete with background, convert once
            for row in range(out_h):
                r0 = rgb_acc[row, 0] * inv_taps32
                r1 = rgb_acc[row, 1] * inv_taps32
                r2 = rgb_acc[row, 2] * inv_taps32
                c = cov_acc[row] * inv_taps32
                remainder = np.float32(1.0) - c
                r0 = r0 + remainder * bg255_0
                r1 = r1 + remainder * bg255_1
                r2 = r2 + remainder * bg255_2
                if r0 < 0.0:
                    r0 = np.float32(0.0)
                elif r0 > 255.0:
                    r0 = np.float32(255.0)
                if r1 < 0.0:
                    r1 = np.float32(0.0)
                elif r1 > 255.0:
                    r1 = np.float32(255.0)
                if r2 < 0.0:
                    r2 = np.float32(0.0)
                elif r2 > 255.0:
                    r2 = np.float32(255.0)
                out[row, out_col, 0] = np.uint8(r0 + np.float32(0.5))
                out[row, out_col, 1] = np.uint8(r1 + np.float32(0.5))
                out[row, out_col, 2] = np.uint8(r2 + np.float32(0.5))
                # Mirrors the reference: the frame is finished over the
                # background, so it is opaque.
                out[row, out_col, 3] = np.uint8(255)

