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


@njit(cache=True, nogil=True)
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

    Mirrors ``_height_bars_occluded_polyline`` operation for operation
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
    for e in range(E):
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


if __name__ == "__main__":  # pragma: no cover
    print(warm())


@njit(cache=True, parallel=True, nogil=True)
def _materialize(  # noqa: C901 - one kernel, mirrored from the reference
    a_at,        # f64 (W,)
    t_enter,     # f64 (W,)
    t_exit,      # f64 (W,)
    ia0_arr,     # f64 (W,)
    ib0_arr,     # f64 (W,)
    t_a0,        # f64 (W,)
    t_b0,        # f64 (W,)
    g0_arr,      # f32 (W,)
    z_top32,     # f32 (ny, nx)
    z_bot32,     # f32 (ny, nx)
    finite_grid, # bool (ny, nx)
    rgb_grid,    # f32 (ny, nx, 3) bar colours
    base_grid,   # f32 (ny, nx, 3) zero-end colours
    top_grid,    # f32 (ny, nx, 3) top-face colours (lit when dense)
    dense,       # bool
    shade_x,     # f32
    shade_y,     # f32
    sa,          # f64
    ca,          # f64
    se32,        # f32
    y_high,      # f64
    scale,       # f64
    pane_high32, # f32
    pane_low32,  # f32  (min(pane_low, 0) as f32)
    background32,  # f32 (3,)
    bg_u8,       # u8 (3,)
    out,         # u8 (H, W, 4)  written
    id_plane,    # i32 (H, W)    written
    n_chunks,    # i64
):
    render_w = a_at.shape[0]
    render_h = out.shape[0]
    ny = z_top32.shape[0]
    nx = z_top32.shape[1]
    S = nx + ny
    max_spans = 3 * S + 1
    chunk = (render_w + n_chunks - 1) // n_chunks
    for chunk_index in prange(n_chunks):
        c0 = chunk_index * chunk
        c1 = min(c0 + chunk, render_w)
        if c0 >= c1:
            continue
        # chunk-local scratch; the diff planes are zeroed once and every
        # column re-zeroes only the rows it touched.
        sorted_t32 = np.empty(S, dtype=np.float32)
        is_b = np.empty(S, dtype=np.bool_)
        used = np.empty(S, dtype=np.bool_)
        seg_t0 = np.empty(S, dtype=np.float32)
        # seg_t1 stays float64: the reference's np.clip has no out=, so
        # its result is promoted and NEVER rounded back to float32 --
        # g_hi, the watermark and every hi bound ride that precision.
        seg_t1 = np.empty(S, dtype=np.float64)
        seg_ca = np.empty(S, dtype=np.int64)
        seg_cb = np.empty(S, dtype=np.int64)
        seg_inside = np.empty(S, dtype=np.bool_)
        seg_ok = np.empty(S, dtype=np.bool_)
        seg_glo = np.empty(S, dtype=np.float32)
        seg_ghi = np.empty(S, dtype=np.float64)
        seg_wm = np.empty(S, dtype=np.float64)
        seg_ent = np.empty(S, dtype=np.bool_)
        rec_rlo = np.empty(max_spans, dtype=np.int64)
        rec_rhi = np.empty(max_spans, dtype=np.int64)
        rec_int = np.empty((max_spans, 3), dtype=np.float64)
        rec_slp = np.empty((max_spans, 3), dtype=np.float64)
        rec_id = np.empty(max_spans, dtype=np.int64)
        diff_i = np.zeros((render_h + 1, 3), dtype=np.float64)
        diff_s = np.zeros((render_h + 1, 3), dtype=np.float64)
        diff_id = np.zeros(render_h + 1, dtype=np.int64)
        for column in range(c0, c1):
            enter = t_enter[column]
            texit = t_exit[column]
            ia0c = np.int64(ia0_arr[column])
            ib0c = np.int64(ib0_arr[column])
            ta0 = t_a0[column]
            tb0 = t_b0[column]
            g0c = g0_arr[column]
            # ---- crossings: a slots by the closed form, b slots as the
            # integer complement (permutation by construction).
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
            # ---- segment sweep: bounds, cells, watermark, enter flags
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
                seg_ghi[s] = np.float64(g0c) + seg_t1[s] * np.float64(se32)
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
            # ---- span records in the reference's class-major order
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
                row_bottom = (y_high - lo) * scale
                if row_bottom < 0.0:
                    row_bottom = 0.0
                elif row_bottom > render_h:
                    row_bottom = np.float64(render_h)
                row_top = (y_high - hi) * scale
                if row_top < 0.0:
                    row_top = 0.0
                elif row_top > render_h:
                    row_top = np.float64(render_h)
                r_hi = np.int64(row_bottom)
                r_lo = np.int64(row_top)
                if not r_hi > r_lo:
                    continue
                full_lo = seg_glo[s] + cell_bot
                full_hi = seg_glo[s] + cell_top
                fr_bottom = (y_high - np.float64(full_lo)) * scale
                fr_top = (y_high - np.float64(full_hi)) * scale
                height = fr_bottom - fr_top
                if height < 1e-6:
                    height = 1e-6
                if dense:
                    shade = np.float32(1.0)
                else:
                    if seg_ent[s]:
                        shade = shade_x
                    else:
                        shade = shade_y
                positive = z_top32[cbi, cai] > np.float32(0.0)
                for ch in range(3):
                    if dense:
                        c_hi = top_grid[cbi, cai, ch]
                        c_lo = top_grid[cbi, cai, ch]
                    else:
                        if positive:
                            c_hi = rgb_grid[cbi, cai, ch] * shade
                            c_lo = base_grid[cbi, cai, ch] * shade
                        else:
                            c_hi = base_grid[cbi, cai, ch] * shade
                            c_lo = rgb_grid[cbi, cai, ch] * shade
                    d = c_lo - c_hi
                    slp = np.float64(d) / height
                    rec_slp[count, ch] = slp
                    rec_int[count, ch] = np.float64(c_hi) - slp * fr_top
                rec_rlo[count] = r_lo
                rec_rhi[count] = r_hi
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
                row_bottom = (y_high - lo) * scale
                if row_bottom < 0.0:
                    row_bottom = 0.0
                elif row_bottom > render_h:
                    row_bottom = np.float64(render_h)
                row_top = (y_high - hi) * scale
                if row_top < 0.0:
                    row_top = 0.0
                elif row_top > render_h:
                    row_top = np.float64(render_h)
                r_hi = np.int64(row_bottom)
                r_lo = np.int64(row_top)
                if not r_hi > r_lo:
                    continue
                fr_bottom = (y_high - lo) * scale
                fr_top = (y_high - hi) * scale
                height = fr_bottom - fr_top
                if height < 1e-6:
                    height = 1e-6
                for ch in range(3):
                    c_val = top_grid[cbi, cai, ch]
                    d = c_val - c_val
                    slp = np.float64(d) / height
                    rec_slp[count, ch] = slp
                    rec_int[count, ch] = np.float64(c_val) - slp * fr_top
                rec_rlo[count] = r_lo
                rec_rhi[count] = r_hi
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
                row_bottom = (y_high - lo) * scale
                if row_bottom < 0.0:
                    row_bottom = 0.0
                elif row_bottom > render_h:
                    row_bottom = np.float64(render_h)
                row_top = (y_high - hi) * scale
                if row_top < 0.0:
                    row_top = 0.0
                elif row_top > render_h:
                    row_top = np.float64(render_h)
                r_hi = np.int64(row_bottom)
                r_lo = np.int64(row_top)
                if not r_hi > r_lo:
                    continue
                fr_bottom = (y_high - lo) * scale
                fr_top = (y_high - hi) * scale
                height = fr_bottom - fr_top
                if height < 1e-6:
                    height = 1e-6
                for ch in range(3):
                    c_val = background32[ch]
                    slp = np.float64(c_val - c_val) / height
                    rec_slp[count, ch] = slp
                    rec_int[count, ch] = np.float64(c_val) - slp * fr_top
                rec_rlo[count] = r_lo
                rec_rhi[count] = r_hi
                rec_id[count] = 1
                count += 1
            # pane span (float64 semantics, exactly as the reference's
            # np.where(..., np.inf) upcast)
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
                row_bottom = (y_high - pane_lo) * scale
                if row_bottom < 0.0:
                    row_bottom = 0.0
                elif row_bottom > render_h:
                    row_bottom = np.float64(render_h)
                row_top = (y_high - pane_hi) * scale
                if row_top < 0.0:
                    row_top = 0.0
                elif row_top > render_h:
                    row_top = np.float64(render_h)
                r_hi = np.int64(row_bottom)
                r_lo = np.int64(row_top)
                if r_hi > r_lo:
                    fr_bottom = (y_high - pane_lo) * scale
                    fr_top = (y_high - pane_hi) * scale
                    height = fr_bottom - fr_top
                    if height < 1e-6:
                        height = 1e-6
                    for ch in range(3):
                        c_val = background32[ch]
                        slp = np.float64(c_val - c_val) / height
                        rec_slp[count, ch] = slp
                        rec_int[count, ch] = np.float64(c_val) - slp * fr_top
                    rec_rlo[count] = r_lo
                    rec_rhi[count] = r_hi
                    rec_id[count] = 2
                    count += 1
            # ---- scatter: all + contributions first, then all -, the
            # reference bincount's per-slot summation order.
            for record in range(count):
                r = rec_rlo[record]
                for ch in range(3):
                    diff_i[r, ch] += rec_int[record, ch] * 255.0
                    diff_s[r, ch] += rec_slp[record, ch] * 255.0
                diff_id[r] += rec_id[record]
            for record in range(count):
                r = rec_rhi[record]
                for ch in range(3):
                    diff_i[r, ch] += -rec_int[record, ch] * 255.0
                    diff_s[r, ch] += -rec_slp[record, ch] * 255.0
                diff_id[r] += -rec_id[record]
            # ---- the walk: float32 prefix sums exactly as cast-then-
            # cumsum, one pixel written per row.
            acc_i0 = np.float32(0.0)
            acc_i1 = np.float32(0.0)
            acc_i2 = np.float32(0.0)
            acc_s0 = np.float32(0.0)
            acc_s1 = np.float32(0.0)
            acc_s2 = np.float32(0.0)
            acc_id = np.int64(0)
            for row in range(render_h):
                acc_i0 += np.float32(diff_i[row, 0])
                acc_i1 += np.float32(diff_i[row, 1])
                acc_i2 += np.float32(diff_i[row, 2])
                acc_s0 += np.float32(diff_s[row, 0])
                acc_s1 += np.float32(diff_s[row, 1])
                acc_s2 += np.float32(diff_s[row, 2])
                acc_id += diff_id[row]
                id_plane[row, column] = np.int32(acc_id)
                if acc_id > 0:
                    row32 = np.float32(row)
                    v0 = acc_i0 + acc_s0 * row32
                    if v0 < 0.0:
                        v0 = np.float32(0.0)
                    elif v0 > 255.0:
                        v0 = np.float32(255.0)
                    v1 = acc_i1 + acc_s1 * row32
                    if v1 < 0.0:
                        v1 = np.float32(0.0)
                    elif v1 > 255.0:
                        v1 = np.float32(255.0)
                    v2 = acc_i2 + acc_s2 * row32
                    if v2 < 0.0:
                        v2 = np.float32(0.0)
                    elif v2 > 255.0:
                        v2 = np.float32(255.0)
                    out[row, column, 0] = np.uint8(v0)
                    out[row, column, 1] = np.uint8(v1)
                    out[row, column, 2] = np.uint8(v2)
                    out[row, column, 3] = np.uint8(255)
                else:
                    out[row, column, 0] = bg_u8[0]
                    out[row, column, 1] = bg_u8[1]
                    out[row, column, 2] = bg_u8[2]
                    out[row, column, 3] = np.uint8(0)
            # ---- re-zero only the touched diff rows
            for record in range(count):
                r = rec_rlo[record]
                diff_i[r, 0] = 0.0
                diff_i[r, 1] = 0.0
                diff_i[r, 2] = 0.0
                diff_s[r, 0] = 0.0
                diff_s[r, 1] = 0.0
                diff_s[r, 2] = 0.0
                diff_id[r] = 0
                r = rec_rhi[record]
                diff_i[r, 0] = 0.0
                diff_i[r, 1] = 0.0
                diff_i[r, 2] = 0.0
                diff_s[r, 0] = 0.0
                diff_s[r, 1] = 0.0
                diff_s[r, 2] = 0.0
                diff_id[r] = 0
