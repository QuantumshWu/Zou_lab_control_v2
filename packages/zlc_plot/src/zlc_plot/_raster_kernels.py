"""Numba kernels for hot numeric passes of the display-front pipeline.

Like :mod:`_height3d_scanline`, this module exists for SPEED ONLY.  Each
kernel mirrors a numpy reference that stays where it lives and stays the
specification -- the block-mean in :mod:`_image_raster`, the colour pass and
the box resize in :mod:`rendering`, the uniform histogram in
:mod:`data_view`.  Every kernel here reproduces its reference operation for
operation, in the same dtypes and the same order, and a standing contract
test runs both and asserts bit equality, so the two can never drift apart
silently.

Why they are faster is the same story throughout: numpy answers each of
these questions with several full passes over a megapixel plane (a copy, a
scaled plane, a clipped plane, an index plane, a gather), or with a
general-purpose routine paying for a generality the caller does not need
(``np.add.reduceat`` books a segment for every one of two million output
cells whose blocks are one or two samples wide).  The kernels touch each
output once, in parallel, with the bookkeeping in registers.

Compilation caches on disk exactly as the scanline engine does, in the
directory :mod:`zlc_plot._kernel_cache` owns; ``ZLC_PLOT_KERNELS=numpy``
forces every dispatch back to its reference, which is how the contract test
compares them.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import _kernel_cache

# BEFORE numba is imported: it reads NUMBA_CACHE_DIR when the dispatcher is
# built, so a later assignment is ignored in silence.
_kernel_cache.install()

try:  # pragma: no cover - absence is exercised by the dispatch fallback
    from numba import config, get_num_threads, njit, prange, set_num_threads

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False
    config = None
    set_num_threads = None

    def get_num_threads() -> int:  # type: ignore[misc]
        return 1

    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrap(fn):
            return fn

        return wrap

    prange = range  # type: ignore[assignment]


#: ``numpy`` forces every reference path; ``numba`` demands the kernels;
#: ``auto`` uses them when they compiled.  Mirrors ``ZLC_H3D_ENGINE``.
ENGINE = os.environ.get("ZLC_PLOT_KERNELS", "auto")


def engaged() -> bool:
    """Whether the compiled kernels answer, rather than their references."""

    if ENGINE == "numpy":
        return False
    if ENGINE == "numba":
        return True
    return HAVE_NUMBA


def configure_worker_threads() -> int:
    """Mask one ZLC worker's native team without shrinking the process pool."""

    if not HAVE_NUMBA:
        return 1
    maximum = int(config.NUMBA_NUM_THREADS)
    requested = int(os.environ.get("ZLC_NUMBA_WORKER_THREADS", maximum))
    selected = max(1, min(requested, maximum))
    set_num_threads(selected)
    return selected


#: The most column bands one stroke lane is cut into.  The pool is not one
#: panel's: a four-panel console launches from four workers at once, each
#: masked to the whole pool by default, and cutting a lone lane into all
#: sixteen bands moved the four-panel critical path from 83 to 105 ms --
#: every panel's work inflated under the oversubscription.  Four bands
#: left the critical path where it was (84 ms) and still took the curve
#: panel's compose from 15.1 to 11.9 ms; in isolation a forty-series
#: curve's error bars go 6.0 -> 1.9 ms at four bands against 0.9 at
#: sixteen, a floor not worth the console's ceiling.
_STROKE_BAND_LIMIT = 4


def stroke_bands(lane_count: int) -> int:
    """How many column bands each stroke lane is cut into for this pool.

    Lanes already run in parallel; a lane with fewer peers than the pool has
    threads is split so the pool still has work, up to
    ``_STROKE_BAND_LIMIT`` per lane.  The count is decided HERE, in Python,
    and handed to the kernel: asking numba for its thread count inside a
    kernel makes that kernel uncacheable (a "dynamic global"), and an
    uncacheable stroke kernel is recompiled on every process start --
    seconds before the first curve.
    """

    threads = int(get_num_threads()) if HAVE_NUMBA else 1
    if 0 < lane_count < threads:
        return min(_STROKE_BAND_LIMIT, threads // lane_count)
    return 1


def readable(array: Any) -> Any:
    """A C-contiguous READ-ONLY view of ``array``: one signature, not two.

    NUMBA TYPES MUTABILITY.  ``array(uint16, 2d, C)`` and ``readonly
    array(uint16, 2d, C)`` are different types, so the same kernel compiles
    a second time for each -- and which one a plane is, is an accident of
    where it came from: a published snapshot is sealed, a prepared front is
    sealed, but ``ascontiguousarray`` on a STRIDED read-only view has to
    copy, and a fresh copy is writable.  Which is to say: whether the
    operator had zoomed.  Measured across the image dtypes with and without
    a zoom, 23 compiled signatures of which 10 were the same code again.

    Every array that crosses into a kernel as an INPUT comes through here.
    Outputs do not: they are written.
    """

    view = np.ascontiguousarray(array)
    if view.flags.writeable:
        # A view, never the caller's own array: sealing theirs would be a
        # side effect on a value they still own.
        view = view.view()
        view.setflags(write=False)
    return view


# -------------------------------------------------------------- block means
@njit(cache=True, parallel=True, nogil=True)
def block_mean_valid(values, valid, use_valid, row_starts, column_starts, out, counts):
    """Mean each block, accumulating wide -- and count, when there is a mask.

    ``np.add.reduceat`` books a segment per output cell -- two million of
    them, one or two samples wide -- and that bookkeeping, not the addition,
    is the cost: 9.2 ms for a 1200x1920 float32 plane where this is 0.25.
    The masked path this also owns used to materialise a whole
    ``np.where(valid, values, 0)`` plane and reduce twice more.

    One kernel for both faces because the mask test is loop-invariant --
    the compiler unswitches it, and the merged loop measured FASTER than
    the dedicated whole-plane kernel it replaced (0.22 vs 0.33 ms on a
    1200x1920 float32 plane), bit-identical on both paths.  With
    ``use_valid`` false, ``valid`` and ``counts`` are untouched dummies and
    every sample of a block counts; an empty masked block writes zero and
    a zero count, which the caller masks.

    It accumulates in float64 whatever the plane's dtype and divides
    BEFORE writing, so a floating block's sum never meets ``out``'s dtype: a
    finite float32 plane near its range has block totals past it, and a
    sum written back as float32 came back as an infinite mean of finite
    samples.  For a float32 plane this is not a looser answer than
    ``reduceat`` but a tighter one -- float32 accumulation lands 1e-7
    relative away from a float64 reduction, where this lands on it
    exactly.  For a float64 plane the two differ only by summation order,
    measured at two ulps.

    """

    row_count = row_starts.size
    column_count = column_starts.size
    rows, columns = values.shape
    for i in prange(row_count):
        row_stop = row_starts[i + 1] if i + 1 < row_count else rows
        for j in range(column_count):
            column_stop = column_starts[j + 1] if j + 1 < column_count else columns
            total = np.float64(0.0)
            seen = np.int64(0)
            for r in range(row_starts[i], row_stop):
                for c in range(column_starts[j], column_stop):
                    if use_valid:
                        if not valid[r, c]:
                            continue
                        seen += 1
                    total += np.float64(values[r, c])
            if use_valid:
                counts[i, j] = seen
                out[i, j] = total / np.float64(seen) if seen > 0 else 0.0
            else:
                out[i, j] = total / np.float64(
                    (row_stop - row_starts[i]) * (column_stop - column_starts[j])
                )


# ------------------------------------------------------------------- colour
@njit(cache=True, parallel=True, nogil=True)
def colour_float32(values, lut, vmin, scale, out):
    """Scale, clip, quantise and look up one float plane in a single pass.

    Mirrors ``lut[np.clip((v - vmin) * scale, 0, 255).astype(uint8)]``: the
    same float32 subtract and multiply, the same clip bounds, the same
    truncating cast.
    """

    rows, columns = values.shape
    for i in prange(rows):
        for j in range(columns):
            scaled = (values[i, j] - vmin) * scale
            if scaled < np.float32(0.0):
                scaled = np.float32(0.0)
            elif scaled > np.float32(255.0):
                scaled = np.float32(255.0)
            code = np.uint8(scaled)
            out[i, j, 0] = lut[code, 0]
            out[i, j, 1] = lut[code, 1]
            out[i, j, 2] = lut[code, 2]
            out[i, j, 3] = lut[code, 3]


@njit(cache=True, parallel=True, nogil=True)
def colour_indexed(values, table, out):
    """Look up one integer plane in a whole-domain table.  Mirrors ``table[v]``."""

    rows, columns = values.shape
    for i in prange(rows):
        for j in range(columns):
            code = values[i, j]
            out[i, j, 0] = table[code, 0]
            out[i, j, 1] = table[code, 1]
            out[i, j, 2] = table[code, 2]
            out[i, j, 3] = table[code, 3]


# ------------------------------------------------------------------- gather
@njit(cache=True, parallel=True, nogil=True)
def gather_rows_columns(rgba, row_map, column_map, out):
    """Nearest-neighbour resize of an RGBA plane.

    Mirrors ``rgba[row_map][:, column_map]`` -- the same two index maps,
    without materialising the intermediate plane the chained fancy indexes
    build.
    """

    box_height = row_map.size
    box_width = column_map.size
    for i in prange(box_height):
        row = row_map[i]
        for j in range(box_width):
            column = column_map[j]
            out[i, j, 0] = rgba[row, column, 0]
            out[i, j, 1] = rgba[row, column, 1]
            out[i, j, 2] = rgba[row, column, 2]
            out[i, j, 3] = rgba[row, column, 3]


# ---------------------------------------------------------------- histogram
@njit(cache=True, parallel=True, nogil=True)
def uniform_histogram(
    values,
    valid,
    use_valid,
    facet_codes,
    facet_stride,
    edges,
    bins,
    partials,
    out,
):
    """Count one or more uniform distributions into ``out[group, bin]``.

    ``facet_codes`` maps the physical tensor index to the value-sorted
    Facet cell. It therefore handles duplicate and non-monotonic authored
    coordinates without building one facet code per sample. An empty code
    array is the ungrouped distribution: no per-sample division or modulo.
    Binning retains NumPy's inclusive last edge and its two rounding
    corrections against the actual edges.
    """

    facets = out.shape[0]
    threads = partials.shape[0]
    chunk = (values.size + threads - 1) // threads
    axis_size = facet_codes.size
    first = edges[0]
    last = edges[bins]
    denominator = last - first
    for t in prange(threads):
        stop = min((t + 1) * chunk, values.size)
        for facet in range(facets):
            for b in range(bins):
                partials[t, facet, b] = 0
        for p in range(t * chunk, stop):
            if use_valid and not valid[p]:
                continue
            facet = 0
            if axis_size:
                facet = facet_codes[(p // facet_stride) % axis_size]
            if facet < 0:
                continue
            sample = np.float64(values[p])
            if not (sample >= first and sample <= last):
                continue
            index = np.int64(((sample - first) / denominator) * bins)
            if index == bins:
                index -= 1
            if sample < edges[index]:
                index -= 1
            if sample >= edges[index + 1] and index != bins - 1:
                index += 1
            partials[t, facet, index] += 1
    for facet in range(facets):
        for b in range(bins):
            total = np.int64(0)
            for t in range(threads):
                total += partials[t, facet, b]
            out[facet, b] = total


@njit(cache=True, nogil=True)
def aggregate_axis_codes(
    values,
    valid,
    use_valid,
    axis_codes,
    axis_sizes,
    domain_sizes,
    axis_strides,
    bucket_count,
    operation,
    offsets,
    out,
    counts,
    presence,
):
    """Reduce a tensor by axis-sized codes, preserving row-major order."""

    for bucket in range(bucket_count):
        counts[bucket] = 0
        presence[bucket] = False
        if operation == 2:
            out[bucket] = np.inf
        elif operation == 3:
            out[bucket] = -np.inf
        else:
            out[bucket] = 0.0
    for position in range(values.size):
        bucket = 0
        admitted = True
        for axis in range(axis_sizes.size):
            index = (position // axis_strides[axis]) % axis_sizes[axis]
            code = axis_codes[axis, index]
            if code < 0:
                admitted = False
                break
            bucket = bucket * domain_sizes[axis] + code
        if not admitted:
            continue
        presence[bucket] = True
        if use_valid and not valid[position]:
            continue
        sample = np.float64(values[position])
        if operation == 2:
            # ``np.minimum.reduceat`` propagates NaN and, for equal values,
            # keeps the later operand (observable for signed zero).  The
            # compiled path is the same reduction in the same row-major
            # order, so preserve both details rather than using a plain
            # comparison.
            if sample != sample:
                out[bucket] = sample
            elif out[bucket] == out[bucket] and sample <= out[bucket]:
                out[bucket] = sample
        elif operation == 3:
            if sample != sample:
                out[bucket] = sample
            elif out[bucket] == out[bucket] and sample >= out[bucket]:
                out[bucket] = sample
        elif operation == 4:
            if counts[bucket] == 0:
                out[bucket] = sample
        elif operation == 5:
            delta = sample - offsets[bucket]
            out[bucket] += delta * delta
        else:
            out[bucket] += sample
        counts[bucket] += 1
    if operation == 0:
        for bucket in range(bucket_count):
            if counts[bucket] > 0:
                out[bucket] /= counts[bucket]
    for bucket in range(bucket_count):
        if counts[bucket] == 0:
            out[bucket] = np.nan


def histogram_threads() -> int:
    """How many lanes :func:`uniform_histogram` should be given."""

    if not HAVE_NUMBA:
        return 1
    from numba import get_num_threads

    return int(get_num_threads())


# ------------------------------------------------------- masked tensor reduce
REDUCE_MEAN = 0
REDUCE_SUM = 1
REDUCE_MIN = 2
REDUCE_MAX = 3
REDUCE_FIRST = 4
_MASKED_LEADING_MIN_SAMPLES = 32768


@njit(cache=True, parallel=True, nogil=True)
def masked_leading_float64(values, valid, reduction, out, counts):
    """Reduce ``(pool, outputs)`` once, producing values and counts together.

    The NumPy reference first sums the mask and then walks the value plane
    again under ``where=``.  With holes, every output bucket needs both facts;
    this kernel obtains them in the same leading-axis order in one pass.
    Equality updates for extrema deliberately retain NumPy's last signed zero.
    """

    pool, outputs = values.shape
    for column in prange(outputs):
        count = 0
        accumulator = 0.0
        found = False
        if reduction == REDUCE_MIN:
            accumulator = np.inf
        elif reduction == REDUCE_MAX:
            accumulator = -np.inf
        for row in range(pool):
            if not valid[row, column]:
                continue
            sample = values[row, column]
            count += 1
            if reduction == REDUCE_MEAN or reduction == REDUCE_SUM:
                accumulator += sample
            elif reduction == REDUCE_MIN:
                if sample <= accumulator:
                    accumulator = sample
            elif reduction == REDUCE_MAX:
                if sample >= accumulator:
                    accumulator = sample
            elif not found:
                accumulator = sample
                found = True
        counts[column] = count
        if count == 0:
            out[column] = np.nan
        elif reduction == REDUCE_MEAN:
            out[column] = accumulator / count
        else:
            out[column] = accumulator


def fused_masked_leading_float64(
    values: Any, valid: Any, reduction: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run the fused leading reduction where its arithmetic is exact."""

    if not engaged():
        return None
    source = np.asarray(values)
    marks = np.asarray(valid, dtype=np.bool_)
    if (
        source.dtype != np.float64
        or source.shape != marks.shape
        or source.ndim < 2
        or source.size < _MASKED_LEADING_MIN_SAMPLES
    ):
        return None
    pool = int(source.shape[0])
    shape = source.shape[1:]
    flat_source = np.reshape(source, (pool, -1), order="C")
    flat_marks = np.reshape(marks, (pool, -1), order="C")
    if (
        flat_source.shape[1] < 2
        or not flat_source.flags.c_contiguous
        or not flat_marks.flags.c_contiguous
    ):
        # A copied reorder changes NumPy's floating reduction order for the
        # ordinary (x, pool) tensor layout at pool>=4, in addition to costing
        # 5--10 ms over two million values.  Keep that exact strided question
        # on its NumPy reference.  One output column likewise uses NumPy's
        # pairwise contiguous reduction and gives prange no parallel work.
        return None
    flat = readable(flat_source)
    flat_valid = readable(flat_marks)
    out = np.empty(flat.shape[1], dtype=np.float64)
    counts = np.empty(flat.shape[1], dtype=np.int64)
    masked_leading_float64(flat, flat_valid, int(reduction), out, counts)
    return out.reshape(shape), counts.reshape(shape)

# ------------------------------------------------------------------ extrema
@njit(cache=True, nogil=True)
def prepare_curve_summary(x, y, source_valid, lower, upper, has_band, valid, out):
    """Prepare validity, display bounds and singleton presence in one pass.

    Only derived frame facts are produced: no reordering, thinning or change
    to values/SEM. This serial pass avoids an OpenMP launch for each cell.
    """

    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    pooled_min, pooled_max = np.inf, -np.inf
    isolated = False
    previous = False
    for point in range(x.size):
        xv, yv = x[point], y[point]
        usable = source_valid[point] and np.isfinite(xv) and np.isfinite(yv)
        valid[point] = usable
        if usable:
            low = lower[point] if has_band else yv
            high = upper[point] if has_band else yv
            if not np.isfinite(low):
                low = yv
            if not np.isfinite(high):
                high = yv
            xmin, xmax = min(xmin, xv), max(xmax, xv)
            ymin, ymax = min(ymin, low), max(ymax, high)
            pooled_min = min(pooled_min, yv, low, high)
            pooled_max = max(pooled_max, yv, low, high)
            if not previous:
                following = (point + 1 < x.size and source_valid[point + 1]
                             and np.isfinite(x[point + 1]) and np.isfinite(y[point + 1]))
                if not following:
                    isolated = True
        previous = usable
    out[0], out[1] = xmin, xmax
    out[2], out[3] = ymin, ymax
    out[4] = 1.0 if isolated else 0.0
    out[5], out[6] = pooled_min, pooled_max


@njit(cache=True, parallel=True, nogil=True)
def finite_extrema(values, valid, use_valid, out):
    """One pass for ``(finite count, min, max, all whole)`` over a masked pool.

    Mirrors ``isfinite`` + ``any`` + ``min(where=)`` + ``max(where=)`` +
    ``all(x == floor(x), where=)``: five full reads of a two-million-value
    pool, and a bool plane the size of it, to answer four numbers.  All are
    order-independent, so the parallel partials are the same numbers the
    reductions produce.  The fourth is the histogram's: whether every
    finite sample is a whole number is a fact about every sample, and this
    is the pass that sees every sample.
    """

    threads = out.shape[0] - 1
    chunk = (values.size + threads - 1) // threads
    for t in prange(threads):
        stop = min((t + 1) * chunk, values.size)
        count = 0.0
        low = np.inf
        high = -np.inf
        whole = 1.0
        for p in range(t * chunk, stop):
            if use_valid and not valid[p]:
                continue
            sample = values[p]
            if not np.isfinite(sample):
                continue
            count += 1.0
            if sample < low:
                low = sample
            if sample > high:
                high = sample
            if sample != np.floor(sample):
                whole = 0.0
        out[t, 0] = count
        out[t, 1] = low
        out[t, 2] = high
        out[t, 3] = whole
    total = 0.0
    lowest = np.inf
    highest = -np.inf
    all_whole = 1.0
    for t in range(threads):
        total += out[t, 0]
        if out[t, 1] < lowest:
            lowest = out[t, 1]
        if out[t, 2] > highest:
            highest = out[t, 2]
        if out[t, 3] == 0.0:
            all_whole = 0.0
    out[threads, 0] = total
    out[threads, 1] = lowest
    out[threads, 2] = highest
    out[threads, 3] = all_whole


@njit(cache=True, parallel=True, nogil=True)
def centred_square_sums(values, offset, valid, use_valid, out):
    """Sum ``(x - offset)**2`` into one entry per kept position, in one pass.

    THE SHAPE IS ALWAYS THREE.  Whatever the signal's rank, the axes a
    reduction keeps are one block of it, so the tensor is (everything
    before the block, the block, everything after) and this kernel needs no
    other spelling: one compiled specialization serves a curve over point
    rows, a heatmap over two scan dimensions and a grouped band alike.

    It replaces ``centred = plane - offset`` followed by an einsum.  The
    einsum did fuse the square into the sum, but the CENTRING still
    materialized a whole copy of the tensor first -- 15.6 MB and 4.77 ms of
    a 6.13 ms call on two million samples, where the einsum itself was
    0.65.  Reading each sample once and never writing it is 0.16 ms.  The
    answer moves in its last bits, as any change of summation order does:
    measured 4.6e-15 relative on the shape above.
    """

    outer, keep, inner = values.shape
    for k in prange(keep):
        total = np.float64(0.0)
        for o in range(outer):
            for i in range(inner):
                if use_valid and not valid[o, k, i]:
                    continue
                delta = np.float64(values[o, k, i]) - offset
                total += delta * delta
        out[k] = total


def masked_centred_square_sums(values: Any, offset: float, valid: Any) -> Any:
    """Run :func:`centred_square_sums`, or ``None`` to defer to numpy."""

    if not engaged():
        return None
    view = np.asarray(values)
    if view.ndim != 3 or not view.flags.c_contiguous or not view.size:
        return None
    if view.dtype.kind not in "fiub":
        return None
    use_valid = valid is not None
    if use_valid:
        marks = np.asarray(valid)
        if (
            marks.dtype != np.bool_
            or marks.shape != view.shape
            or not marks.flags.c_contiguous
        ):
            return None
        marks = readable(marks)
    else:
        marks = readable(np.zeros((1, 1, 1), dtype=np.bool_))
    out = np.empty(view.shape[1], dtype=np.float64)
    # The plane is an input like the mask: sealed, so a writable copy from
    # upstream and a published read-only snapshot are one signature.
    centred_square_sums(readable(view), np.float64(offset), marks, use_valid, out)
    return out


def masked_finite_extrema(
    values: Any, valid: Any
) -> tuple[int, float, float, bool] | None:
    """``(count, low, high, integral)`` for a flat float pool, or ``None`` to defer.

    ``integral`` is whether every finite sample is a whole number (vacuously
    so for an empty pool).  The mask stand-in goes through :func:`readable`
    like a real mask: a writable dummy typed a second signature for the
    kernel, and which one a caller got depended on whether it had a mask.
    """

    if not engaged():
        return None
    flat = readable(values).reshape(-1)
    if flat.dtype.kind != "f" or not flat.size:
        return None
    use_valid = valid is not None
    if use_valid:
        mask = readable(np.asarray(valid, dtype=np.bool_)).reshape(-1)
        if mask.size != flat.size:
            return None
    else:
        mask = readable(np.zeros(1, dtype=np.bool_))
    threads = 1
    if HAVE_NUMBA:
        from numba import get_num_threads

        threads = int(get_num_threads())
    out = np.empty((threads + 1, 4), dtype=np.float64)
    finite_extrema(flat, mask, use_valid, out)
    return (
        int(out[threads, 0]),
        float(out[threads, 1]),
        float(out[threads, 2]),
        bool(out[threads, 3] != 0.0),
    )


# -------------------------------------------------------------- polylines
@njit(cache=True, parallel=True, nogil=True)
def raster_error_bars(
    x,
    y_low,
    y_high,
    offsets,
    colours,
    widths,
    cap_widths,
    clips,
    lane_offsets,
    band_count,
    out,
):
    """Raster independent stem/cap error bars with subpixel coverage.

    Each input sample remains one vertical stem and two horizontal caps.  No
    display-column aggregation is permitted: neighbouring measurements may
    overlap on screen, but they never become one invented min/max envelope.
    Axis-aligned rectangle coverage is analytic, so a fractional-DPR or small
    Facet cell retains antialiasing without a supersampled temporary atlas.

    One lane owns one axes.  Facet axes are disjoint and therefore run in
    parallel; grouped series on the same axes remain sequential inside one
    lane, preserving their alpha-composition order without races.  A lane
    with fewer peers than the pool has threads is cut into column BANDS
    that run in parallel too: every band replays every primitive in the
    same painter order, restricted to its own columns, so the sequence of
    blends any one pixel sees is unchanged -- forty grouped series on one
    axes used to stroke on a single core while the pool idled.  The band
    count is the caller's (:func:`stroke_bands`), never a thread query in
    here, which would cost the kernel its on-disk cache.  The cost
    is the blend arithmetic itself: walking a rectangle row-first instead
    of column-first, or short-cutting its unit-coverage interior, measured
    the same on the same frames, so neither is here.
    """

    height, width = out.shape[:2]
    lane_count = lane_offsets.size - 1
    for task in prange(lane_count * band_count):
        lane = task // band_count
        band = task - lane * band_count
        lane_left = width
        lane_right = 0
        for group in range(lane_offsets[lane], lane_offsets[lane + 1]):
            lane_left = min(lane_left, max(0, clips[group, 0]))
            lane_right = max(lane_right, min(width, clips[group, 2]))
        if lane_right <= lane_left:
            continue
        span = lane_right - lane_left
        band_left = lane_left + (span * band) // band_count
        band_right = lane_left + (span * (band + 1)) // band_count
        if band_right <= band_left:
            continue
        for group in range(lane_offsets[lane], lane_offsets[lane + 1]):
            clip_left = max(band_left, clips[group, 0])
            clip_top = max(0, clips[group, 1])
            clip_right = min(band_right, clips[group, 2])
            clip_bottom = min(height, clips[group, 3])
            if clip_right <= clip_left or clip_bottom <= clip_top:
                continue
            radius = max(np.float64(0.5), np.float64(widths[group]) * 0.5)
            cap_half = max(
                np.float64(0.0), np.float64(cap_widths[group]) * 0.5
            )
            alpha_code = np.float64(colours[group, 3]) / np.float64(255.0)

            # Agg's errorbar topology is one LineCollection of all stems
            # followed by the low-cap and high-cap Line2Ds.  Preserve that
            # painter order so overlapping translucent bars accumulate the
            # way the public artist scene does.
            for primitive in range(3):
                if primitive and cap_half <= 0.0:
                    continue
                for point in range(offsets[group], offsets[group + 1]):
                    px = np.float64(x[point])
                    low = np.float64(y_low[point])
                    high = np.float64(y_high[point])
                    if not (
                        np.isfinite(px)
                        and np.isfinite(low)
                        and np.isfinite(high)
                    ):
                        continue
                    if high < low:
                        low, high = high, low
                    if primitive == 0:
                        left = px - radius
                        right = px + radius
                        top = low
                        bottom = high
                    else:
                        cap_y = low if primitive == 1 else high
                        left = px - cap_half
                        right = px + cap_half
                        top = cap_y - radius
                        bottom = cap_y + radius
                    first_column = max(clip_left, int(np.floor(left)))
                    last_column = min(clip_right, int(np.ceil(right)))
                    first_row = max(clip_top, int(np.floor(top)))
                    last_row = min(clip_bottom, int(np.ceil(bottom)))
                    for column in range(first_column, last_column):
                        coverage_x = min(
                            np.float64(column + 1), right
                        ) - max(np.float64(column), left)
                        if coverage_x <= 0.0:
                            continue
                        for row in range(first_row, last_row):
                            coverage_y = min(
                                np.float64(row + 1), bottom
                            ) - max(np.float64(row), top)
                            if coverage_y <= 0.0:
                                continue
                            alpha = alpha_code * min(
                                np.float64(1.0), coverage_x * coverage_y
                            )
                            inverse = np.float64(1.0) - alpha
                            for channel in range(3):
                                value = (
                                    np.float64(colours[group, channel]) * alpha
                                    + np.float64(out[row, column, channel]) * inverse
                                )
                                out[row, column, channel] = np.uint8(
                                    min(
                                        np.float64(255.0),
                                        np.floor(value + np.float64(0.5)),
                                    )
                                )
                            out[row, column, 3] = np.uint8(255)


@njit(cache=True, parallel=True, nogil=True)
def raster_polylines(
    vertices, offsets, colours, widths, clips, lane_offsets, band_count, out
):
    """Stroke monotonic display curves as one antialiased column envelope.

    One lane owns one axes-worth of lines, exactly as the error-bar kernel
    groups its stems: a Facet grid's cells are disjoint pixel boxes, so its
    lanes stroke in parallel without a write race, while the lines INSIDE a
    lane keep their sequential painter order -- overlapping translucent
    strokes accumulate the way the artist scene composes them.  Callers
    prove disjointness (``_polyline_lane_offsets``); anything they cannot
    prove arrives as one lane, which is the old serial behaviour.

    A lane with fewer peers than the pool has threads is cut into column
    bands, each stroking every line of the lane in order over its own
    columns; a column's envelope reads its neighbours up to the stroke
    reach, so each band samples that margin beyond its edge and paints
    none of it.  The column envelopes are the band's own scratch, sized to
    the canvas width, not a per-line plane the caller had to keep.

    The half-thickness the envelope adds at each column offset is the
    same square root for every column of a line; it is taken once per
    offset and read back, not recomputed per pair -- a fifth of the
    serial time, measured.
    """

    height, width = out.shape[:2]
    lane_count = lane_offsets.size - 1
    for task in prange(lane_count * band_count):
        lane = task // band_count
        band = task - lane * band_count
        lane_left = width
        lane_right = 0
        for line in range(lane_offsets[lane], lane_offsets[lane + 1]):
            lane_left = min(lane_left, max(0, clips[line, 0]))
            lane_right = max(lane_right, min(width, clips[line, 2]))
        if lane_right <= lane_left:
            continue
        span = lane_right - lane_left
        band_left = lane_left + (span * band) // band_count
        band_right = lane_left + (span * (band + 1)) // band_count
        if band_right <= band_left:
            continue
        low = np.empty(width, dtype=np.float64)
        high = np.empty(width, dtype=np.float64)
        for line in range(lane_offsets[lane], lane_offsets[lane + 1]):
            start = offsets[line]
            stop = offsets[line + 1]
            if stop - start < 2:
                continue
            clip_left = max(0, clips[line, 0])
            clip_top = max(0, clips[line, 1])
            clip_right = min(width, clips[line, 2])
            clip_bottom = min(height, clips[line, 3])
            if clip_right <= clip_left or clip_bottom <= clip_top:
                continue
            paint_left = max(clip_left, band_left)
            paint_right = min(clip_right, band_right)
            if paint_right <= paint_left:
                continue
            radius = max(np.float64(0.5), np.float64(widths[line]) * 0.5)
            reach = int(np.ceil(radius + np.float64(0.5)))
            fill_left = max(clip_left, paint_left - reach)
            fill_right = min(clip_right, paint_right + reach + 1)
            for column in range(fill_left, fill_right):
                low[column] = np.inf
                high[column] = -np.inf

            for point in range(start, stop - 1):
                x0 = vertices[point, 0]
                y0 = vertices[point, 1]
                x1 = vertices[point + 1, 0]
                y1 = vertices[point + 1, 1]
                if not (
                    np.isfinite(x0)
                    and np.isfinite(y0)
                    and np.isfinite(x1)
                    and np.isfinite(y1)
                ):
                    continue
                dx = x1 - x0
                if abs(dx) < np.float64(1.0e-12):
                    column = int(np.floor(np.float64(0.5) * (x0 + x1)))
                    if fill_left <= column < fill_right:
                        low[column] = min(low[column], y0, y1)
                        high[column] = max(high[column], y0, y1)
                    continue
                first = max(fill_left, int(np.floor(min(x0, x1))))
                last = min(fill_right, int(np.ceil(max(x0, x1))) + 1)
                for column in range(first, last):
                    px = np.float64(column) + np.float64(0.5)
                    along = (px - x0) / dx
                    if along < 0.0 or along > 1.0:
                        continue
                    y = y0 + along * (y1 - y0)
                    low[column] = min(low[column], y)
                    high[column] = max(high[column], y)

            verticals = np.empty(reach + 1, dtype=np.float64)
            for distance in range(reach + 1):
                squared = (
                    (radius + np.float64(0.5))
                    * (radius + np.float64(0.5))
                    - np.float64(distance * distance)
                )
                verticals[distance] = (
                    np.sqrt(squared) if squared > 0.0 else np.float64(-1.0)
                )
            alpha_code = np.float64(colours[line, 3]) / np.float64(255.0)
            for column in range(paint_left, paint_right):
                envelope_low = np.inf
                envelope_high = -np.inf
                for source_column in range(
                    max(clip_left, column - reach),
                    min(clip_right, column + reach + 1),
                ):
                    if not np.isfinite(low[source_column]):
                        continue
                    vertical = verticals[abs(source_column - column)]
                    if vertical < 0.0:
                        continue
                    envelope_low = min(
                        envelope_low, low[source_column] - vertical
                    )
                    envelope_high = max(
                        envelope_high, high[source_column] + vertical
                    )
                if not np.isfinite(envelope_low):
                    continue
                first_row = max(clip_top, int(np.floor(envelope_low - 0.5)))
                last_row = min(clip_bottom, int(np.ceil(envelope_high + 0.5)))
                for row in range(first_row, last_row):
                    py = np.float64(row) + np.float64(0.5)
                    amount = min(
                        np.float64(1.0),
                        py - envelope_low + np.float64(0.5),
                        envelope_high - py + np.float64(0.5),
                    )
                    if amount <= 0.0:
                        continue
                    alpha = alpha_code * amount
                    inverse = np.float64(1.0) - alpha
                    for channel in range(3):
                        value = (
                            np.float64(colours[line, channel]) * alpha
                            + np.float64(out[row, column, channel]) * inverse
                        )
                        out[row, column, channel] = np.uint8(
                            min(np.float64(255.0), np.floor(value + np.float64(0.5)))
                        )
                    out[row, column, 3] = np.uint8(255)


@njit(cache=True, parallel=True, nogil=True)
def raster_prepared_images(
    values,
    valid,
    use_valid,
    boxes,
    views,
    extents,
    lut,
    vmin,
    scale,
    out,
):
    """Map prepared Image surfaces directly into their final canvas boxes."""

    cells, source_rows, source_columns = values.shape
    height, width = out.shape[:2]
    for work in prange(cells * height):
        cell = work // height
        row = work - cell * height
        left = max(0, boxes[cell, 0])
        top = max(0, boxes[cell, 1])
        right = min(width, boxes[cell, 2])
        bottom = min(height, boxes[cell, 3])
        if right <= left or bottom <= top or row < top or row >= bottom:
            continue
        x0 = views[cell, 0]
        x1 = views[cell, 1]
        y0 = views[cell, 2]
        y1 = views[cell, 3]
        source_left = extents[cell, 0]
        source_right = extents[cell, 1]
        source_bottom = extents[cell, 2]
        source_top = extents[cell, 3]
        box_width = right - left
        box_height = bottom - top
        y_fraction = (np.float64(row - top) + 0.5) / box_height
        y_value = y1 + y_fraction * (y0 - y1)
        y_denominator = source_top - source_bottom
        if y_denominator == 0.0:
            continue
        source_y = (source_top - y_value) / y_denominator * source_rows
        source_row = int(np.floor(source_y))
        if source_row < 0 or source_row >= source_rows:
            continue
        for column in range(left, right):
            x_fraction = (np.float64(column - left) + 0.5) / box_width
            x_value = x0 + x_fraction * (x1 - x0)
            x_denominator = source_right - source_left
            if x_denominator == 0.0:
                continue
            source_x = (
                (x_value - source_left) / x_denominator * source_columns
            )
            source_column = int(np.floor(source_x))
            if source_column < 0 or source_column >= source_columns:
                continue
            if use_valid and not valid[cell, source_row, source_column]:
                continue
            scaled = (
                np.float64(values[cell, source_row, source_column]) - vmin
            ) * scale
            if scaled < 0.0:
                scaled = 0.0
            elif scaled > 255.0:
                scaled = 255.0
            code = np.uint8(scaled)
            out[row, column, 0] = lut[code, 0]
            out[row, column, 1] = lut[code, 1]
            out[row, column, 2] = lut[code, 2]
            out[row, column, 3] = lut[code, 3]


@njit(cache=True, nogil=True)
def ellipse_boundary_distance(dx, dy, radius_x, radius_y):
    """The distance from a point to the boundary of an axis-aligned ellipse.

    ``(dx, dy)`` is the point relative to the centre.  The nearest boundary
    point of ``(x / e0)**2 + (y / e1)**2 = 1`` with ``e0 >= e1`` to a
    first-quadrant query ``(y0, y1)`` is ``(e0**2 y0 / (s + e0**2),
    e1**2 y1 / (s + e1**2))`` for the one root ``s`` of

        (e0 y0 / (s + e0**2))**2 + (e1 y1 / (s + e1**2))**2 = 1

    above ``-e1**2``; Eberly's bracket makes bisection on it robust for
    every query, including the ones on and inside the evolute where
    Newton's method wanders.  Written in the normalised variable
    ``t = s / e1**2`` so the bracket is dimensionless.  Bisection halves a
    bracket no wider than the query's own distance, so 64 steps are far
    past double precision and the loop leaves as soon as it stops moving.
    """

    y0 = abs(dx)
    y1 = abs(dy)
    e0 = radius_x
    e1 = radius_y
    if e0 < e1:
        y0, y1 = y1, y0
        e0, e1 = e1, e0
    if y1 > 0.0:
        if y0 > 0.0:
            z0 = y0 / e0
            z1 = y1 / e1
            g = z0 * z0 + z1 * z1 - 1.0
            if g == 0.0:
                return 0.0
            r0 = (e0 / e1) * (e0 / e1)
            n0 = r0 * z0
            s0 = z1 - 1.0
            s1 = 0.0 if g < 0.0 else np.sqrt(n0 * n0 + z1 * z1) - 1.0
            s = 0.0
            for _ in range(64):
                s = 0.5 * (s0 + s1)
                if s == s0 or s == s1:
                    break
                ratio0 = n0 / (s + r0)
                ratio1 = z1 / (s + 1.0)
                g = ratio0 * ratio0 + ratio1 * ratio1 - 1.0
                if g > 0.0:
                    s0 = s
                elif g < 0.0:
                    s1 = s
                else:
                    break
            x0 = r0 * y0 / (s + r0)
            x1 = y1 / (s + 1.0)
            return np.sqrt((x0 - y0) * (x0 - y0) + (x1 - y1) * (x1 - y1))
        return abs(y1 - e1)
    # On the major axis the nearest point is the vertex only outside the
    # evolute's cusp; inside it the foot leaves the axis.
    numerator = e0 * y0
    denominator = e0 * e0 - e1 * e1
    if numerator < denominator:
        fraction = numerator / denominator
        x0 = e0 * fraction
        x1 = e1 * np.sqrt(1.0 - fraction * fraction)
        return np.sqrt((x0 - y0) * (x0 - y0) + x1 * x1)
    return abs(y0 - e0)


@njit(cache=True, parallel=True, nogil=True)
def raster_fit_ellipses(
    geometry,
    ring_colours,
    ring_widths,
    center_colours,
    center_radii,
    clips,
    out,
):
    """Paint independent axis-aligned fit rings and center markers.

    A ring is a stroke of ``ring_widths`` pixels centred on the ellipse's
    boundary, so a pixel's coverage is decided by its centre's distance to
    that boundary.  ``|n - 1| * min(rx, ry)``, with ``n`` the normalised
    radius, is a LOWER bound on it (``n`` is ``1 / min(rx, ry)``-Lipschitz)
    and the exact distance for a circle; a pixel it cannot rule out pays
    for :func:`ellipse_boundary_distance`, and only when the ring is not a
    circle.  Taken as the distance itself, the bound stroked a 5:1 ring
    1.6 px into its interior.
    """

    height, width = out.shape[:2]
    for item in prange(geometry.shape[0]):
        center_x = geometry[item, 0]
        center_y = geometry[item, 1]
        radius_x = max(np.float64(0.5), geometry[item, 2])
        radius_y = max(np.float64(0.5), geometry[item, 3])
        ring_radius = max(np.float64(0.5), ring_widths[item] * 0.5)
        center_radius = max(np.float64(0.5), center_radii[item])
        reach_x = radius_x + ring_radius + 1.0
        reach_y = radius_y + ring_radius + 1.0
        left = max(clips[item, 0], int(np.floor(center_x - reach_x)))
        right = min(clips[item, 2], int(np.ceil(center_x + reach_x)))
        top = max(clips[item, 1], int(np.floor(center_y - reach_y)))
        bottom = min(clips[item, 3], int(np.ceil(center_y + reach_y)))
        scale = min(radius_x, radius_y)
        for row in range(top, bottom):
            py = np.float64(row) + 0.5
            for column in range(left, right):
                px = np.float64(column) + 0.5
                dx = px - center_x
                dy = py - center_y
                normalized = np.sqrt(
                    (dx / radius_x) * (dx / radius_x)
                    + (dy / radius_y) * (dy / radius_y)
                )
                distance = abs(normalized - 1.0) * scale
                if radius_x != radius_y and distance < ring_radius + 0.5:
                    distance = ellipse_boundary_distance(
                        dx, dy, radius_x, radius_y
                    )
                amount = min(1.0, ring_radius + 0.5 - distance)
                if amount > 0.0:
                    alpha = (
                        np.float64(ring_colours[item, 3])
                        / np.float64(255.0)
                        * amount
                    )
                    inverse = 1.0 - alpha
                    for channel in range(3):
                        value = (
                            np.float64(ring_colours[item, channel]) * alpha
                            + np.float64(out[row, column, channel]) * inverse
                        )
                        out[row, column, channel] = np.uint8(
                            min(255.0, np.floor(value + 0.5))
                        )
                    out[row, column, 3] = np.uint8(255)
                center_distance = np.sqrt(dx * dx + dy * dy)
                center_amount = min(1.0, center_radius + 0.5 - center_distance)
                if center_amount > 0.0:
                    alpha = (
                        np.float64(center_colours[item, 3])
                        / np.float64(255.0)
                        * center_amount
                    )
                    inverse = 1.0 - alpha
                    for channel in range(3):
                        value = (
                            np.float64(center_colours[item, channel]) * alpha
                            + np.float64(out[row, column, channel]) * inverse
                        )
                        out[row, column, channel] = np.uint8(
                            min(255.0, np.floor(value + 0.5))
                        )
                    out[row, column, 3] = np.uint8(255)


@njit(cache=True, nogil=True)
def replay_foreground_masks(static_masks, static_rows, static_colors,
                            text_masks, text_rows, text_colors, text_offsets, order, out):
    """Replay independent Agg coverage masks in their original painter order.

    Match Matplotlib's fixed_blender_rgba_plain integer arithmetic, including
    pixfmt's opaque copy shortcut. Coverage is Agg's, never a layer union.
    """
    for entry in range(order.shape[0]):
        kind, index = order[entry]
        if kind == 0:
            masks, rows, colors = static_masks, static_rows, static_colors
            start, stop = index, index + 1
        else:
            masks, rows, colors = text_masks, text_rows, text_colors
            start, stop = text_offsets[index], text_offsets[index + 1]
        for primitive in range(start, stop):
            offset, top, left, height, width = rows[primitive]
            for y in range(height):
                for x in range(width):
                    cover = int(masks[offset + y * width + x])
                    product = int(colors[primitive, 3]) * cover + 128
                    alpha = ((product >> 8) + product) >> 8
                    if alpha == 0:
                        continue
                    py, px = top + y, left + x
                    if alpha == 255:
                        for channel in range(3):
                            out[py, px, channel] = colors[primitive, channel]
                        out[py, px, 3] = 255
                        continue
                    old_alpha = int(out[py, px, 3])
                    denominator = ((alpha + old_alpha) << 8) - alpha * old_alpha
                    for channel in range(3):
                        premultiplied = int(out[py, px, channel]) * old_alpha
                        out[py, px, channel] = (premultiplied * (256 - alpha)
                            + (int(colors[primitive, channel]) << 8) * alpha) // denominator
                    out[py, px, 3] = denominator >> 8


@njit(cache=True, parallel=True, nogil=True)
def transform_curve_batch(
    x,
    y,
    valid,
    affine,
    canvas_height,
    vertices,
):
    """Transform grouped Curve data in one native pass."""

    a, b, c, d, e, f = affine
    for series in prange(y.shape[0]):
        for point in range(y.shape[1]):
            if not valid[series, point]:
                vertices[series, point, 0] = np.nan
                vertices[series, point, 1] = np.nan
                continue
            xv = x[point]
            yv = y[series, point]
            vertices[series, point, 0] = a * xv + c * yv + e
            vertices[series, point, 1] = canvas_height - (b * xv + d * yv + f)

