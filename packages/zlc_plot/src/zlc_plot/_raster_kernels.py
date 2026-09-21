"""Numba kernels for hot numeric passes of the display-front pipeline.

Like :mod:`_height3d_scanline`, this module exists for SPEED ONLY.  Each
kernel mirrors a numpy reference that stays where it lives and stays the
specification -- the uniform histogram and the code aggregations in
:mod:`data_view`, the masked extrema and centred sums beside their own
kernels here.  Every kernel here reproduces its reference operation for
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
    out_second,
    counts,
    presence,
):
    """Reduce a tensor by axis-sized codes, preserving row-major order.

    The axis index of a flat position is ``(position // stride) % size``,
    and evaluating that per element per axis is two integer divisions on
    every one of a two-megapixel frame -- measured at twenty-two
    milliseconds a frame on a mixed-topology image, against three for the
    same frame with no aggregation to do.  A row-major walk knows the same
    indices without dividing for them: each axis advances once every
    ``stride`` positions and wraps at ``size``.  Identical arithmetic,
    identical order, identical output.
    """

    for bucket in range(bucket_count):
        counts[bucket] = 0
        presence[bucket] = False
        if operation == 2:
            out[bucket] = np.inf
        elif operation == 3:
            out[bucket] = -np.inf
        else:
            out[bucket] = 0.0
        if operation == 5:
            out_second[bucket] = 0.0
    axis_count = axis_sizes.size
    axis_index = np.zeros(axis_count, dtype=np.int64)
    axis_tick = np.zeros(axis_count, dtype=np.int64)
    for position in range(values.size):
        if position:
            # Advanced at the TOP, so that every ``continue`` below leaves
            # the walk where the next position expects it.
            for axis in range(axis_count):
                axis_tick[axis] += 1
                if axis_tick[axis] == axis_strides[axis]:
                    axis_tick[axis] = 0
                    axis_index[axis] += 1
                    if axis_index[axis] == axis_sizes[axis]:
                        axis_index[axis] = 0
        bucket = 0
        admitted = True
        for axis in range(axis_count):
            code = axis_codes[axis, axis_index[axis]]
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
            out[bucket] += delta
            out_second[bucket] += delta * delta
        else:
            out[bucket] += sample
        counts[bucket] += 1
    if operation == 0:
        for bucket in range(bucket_count):
            if counts[bucket] > 0:
                out[bucket] /= counts[bucket]
    elif operation == 5:
        for bucket in range(bucket_count):
            if counts[bucket] > 0:
                out[bucket] /= counts[bucket]
                out_second[bucket] /= counts[bucket]
    for bucket in range(bucket_count):
        if counts[bucket] == 0:
            out[bucket] = np.nan
            if operation == 5:
                out_second[bucket] = np.nan


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
def centred_moment_sums(
    values, offsets, valid, use_valid, first_out, second_out
):
    """Sum ``d`` and ``d**2`` per kept position in one pass.

    THE SHAPE IS ALWAYS THREE.  Whatever the signal's rank, the axes a
    reduction keeps are one block of it, so the tensor is (everything
    before the block, the block, everything after) and this kernel needs no
    other spelling: one compiled specialization serves a curve over point
    rows, a heatmap over two scan dimensions and a grouped band alike.

    Each kept position has its own offset: the mean of that output bucket.
    The first centred moment must still be accumulated because a rounded
    first-pass mean does not make ``sum(x - mean)`` exactly zero.  The pair
    replaces ``centred = plane - offsets`` followed by two reductions.  The
    einsum did fuse the square into the sum, but the CENTRING still
    materialized a whole copy of the tensor first -- 15.6 MB and 4.77 ms of
    a 6.13 ms call on two million samples, where the einsum itself was
    0.65.  Reading each sample once and never writing it is 0.16 ms.  The
    answer moves in its last bits, as any change of summation order does:
    measured 4.6e-15 relative on the shape above.
    """

    outer, keep, inner = values.shape
    for k in prange(keep):
        first = np.float64(0.0)
        second = np.float64(0.0)
        offset = offsets[k]
        for o in range(outer):
            for i in range(inner):
                if use_valid and not valid[o, k, i]:
                    continue
                delta = np.float64(values[o, k, i]) - offset
                first += delta
                second += delta * delta
        first_out[k] = first
        second_out[k] = second


def masked_centred_moment_sums(
    values: Any, offsets: Any, valid: Any
) -> Any:
    """Run :func:`centred_moment_sums`, or ``None`` to defer to numpy."""

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
    centres = np.asarray(offsets, dtype=np.float64).reshape(-1)
    if centres.shape != (view.shape[1],):
        return None
    first = np.empty(view.shape[1], dtype=np.float64)
    second = np.empty(view.shape[1], dtype=np.float64)
    # The plane is an input like the mask: sealed, so a writable copy from
    # upstream and a published read-only snapshot are one signature.
    centred_moment_sums(
        readable(view), readable(centres), marks, use_valid, first, second
    )
    return first, second


@njit(cache=True, inline="always")
def _combine_moment_summary(
    left_n,
    left_mean,
    left_m2,
    left_single_sem_square,
    right_n,
    right_mean,
    right_m2,
    right_single_sem_square,
):
    """Chan-combine two adjacent ``(n, mean, M2)`` summaries."""

    if left_n == 0:
        return right_n, right_mean, right_m2, right_single_sem_square
    if right_n == 0:
        return left_n, left_mean, left_m2, left_single_sem_square
    count = left_n + right_n
    delta = right_mean - left_mean
    mean = left_mean + delta * right_n / count
    m2 = (
        left_m2
        + right_m2
        + delta * delta * left_n * right_n / count
    )
    return count, mean, m2, np.nan


@njit(cache=True, nogil=True)
def _trailing_moment_windows(
    counts, means, m2s, single_sem_squares, span
):
    """Replace shot summaries by their trailing-window Chan summaries.

    A span-sized block gets one forward prefix and one backward suffix.  A
    window is then either its block prefix or the combination of exactly one
    preceding suffix and one current prefix.  Each shot is visited a constant
    number of times; no whole-history prefixes are subtracted and variance is
    never recovered as ``Q - S**2/N``.

    The four inputs are writable work planes owned by the caller and become
    the output prefixes/windows.  Only four suffix planes are allocated, so
    memory remains linear without keeping raw, prefix, suffix and output copies.
    """

    total = counts.size
    width = min(span, total)
    suffix_n = np.empty_like(counts)
    suffix_mean = np.empty_like(means)
    suffix_m2 = np.empty_like(m2s)
    suffix_single_sem_square = np.empty_like(single_sem_squares)

    for block_start in range(0, total, width):
        block_stop = min(block_start + width, total)
        count = 0
        mean = 0.0
        m2 = 0.0
        single_sem_square = np.nan
        for index in range(block_stop - 1, block_start - 1, -1):
            count, mean, m2, single_sem_square = _combine_moment_summary(
                counts[index],
                means[index],
                m2s[index],
                single_sem_squares[index],
                count,
                mean,
                m2,
                single_sem_square,
            )
            suffix_n[index] = count
            suffix_mean[index] = mean
            suffix_m2[index] = m2
            suffix_single_sem_square[index] = single_sem_square

        count = 0
        mean = 0.0
        m2 = 0.0
        single_sem_square = np.nan
        for index in range(block_start, block_stop):
            count, mean, m2, single_sem_square = _combine_moment_summary(
                count,
                mean,
                m2,
                single_sem_square,
                counts[index],
                means[index],
                m2s[index],
                single_sem_squares[index],
            )
            counts[index] = count
            means[index] = mean
            m2s[index] = m2
            single_sem_squares[index] = single_sem_square

    for block_start in range(width, total, width):
        # A complete block's final prefix already contains exactly ``span``
        # shots.  Every earlier position also needs the preceding block's
        # suffix; a partial last block has no final-prefix exception.
        block_stop = min(block_start + width - 1, total)
        for index in range(block_start, block_stop):
            left = index - width + 1
            (
                counts[index],
                means[index],
                m2s[index],
                single_sem_squares[index],
            ) = _combine_moment_summary(
                suffix_n[left],
                suffix_mean[left],
                suffix_m2[left],
                suffix_single_sem_square[left],
                counts[index],
                means[index],
                m2s[index],
                single_sem_squares[index],
            )


def trailing_moment_windows(
    counts: Any,
    means: Any,
    m2s: Any,
    single_sem_squares: Any,
    span: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Run the trailing Chan kernel, or ``None`` for the NumPy reference."""

    if not engaged():
        return None
    planes = tuple(
        np.asarray(plane)
        for plane in (counts, means, m2s, single_sem_squares)
    )
    if (
        planes[0].dtype != np.dtype(np.int64)
        or any(plane.dtype != np.dtype(np.float64) for plane in planes[1:])
        or not planes[0].size
        or any(
            plane.ndim != 1
            or plane.shape != planes[0].shape
            or not plane.flags.c_contiguous
            or not plane.flags.writeable
            for plane in planes
        )
    ):
        return None
    width = int(span)
    if width <= 0:
        return None
    # These are caller-owned work/output planes, not immutable inputs; keeping
    # that contract explicit also gives the dispatcher one writable signature.
    _trailing_moment_windows(*planes, width)
    return planes


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
    """Raster each complete error bar once with analytic union coverage.

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

            # All three rectangles share their x centre. Their intersection
            # is a central strip spanning the whole glyph; the remaining
            # width belongs only to the stem or only to the two caps.
            # Sum those disjoint areas, then alpha-blend this BAR once.
            for point in range(offsets[group], offsets[group + 1]):
                px = np.float64(x[point])
                low = np.float64(y_low[point])
                high = np.float64(y_high[point])
                if not (np.isfinite(px) and np.isfinite(low) and np.isfinite(high)):
                    continue
                if high < low:
                    low, high = high, low
                if high <= low:
                    continue
                half = max(radius, cap_half)
                top = low - radius if cap_half > 0.0 else low
                bottom = high + radius if cap_half > 0.0 else high
                first_column = max(clip_left, int(np.floor(px - half)))
                last_column = min(clip_right, int(np.ceil(px + half)))
                first_row = max(clip_top, int(np.floor(top)))
                last_row = min(clip_bottom, int(np.ceil(bottom)))
                for column in range(first_column, last_column):
                    stem_x = max(0.0, min(column + 1.0, px + radius) - max(column, px - radius))
                    cap_x = max(0.0, min(column + 1.0, px + cap_half) - max(column, px - cap_half))
                    shared_x = min(stem_x, cap_x)
                    stem_x -= shared_x
                    cap_x -= shared_x
                    row = first_row
                    while row < last_row:
                        if shared_x == 0.0 and stem_x == 0.0 and row >= low + radius and row + 1.0 <= high - radius:
                            row = int(np.floor(high - radius))
                            continue
                        full_y = max(0.0, min(row + 1.0, bottom) - max(row, top))
                        covered = shared_x * full_y
                        if stem_x > 0.0:
                            covered += stem_x * max(0.0, min(row + 1.0, high) - max(row, low))
                        if cap_x > 0.0:
                            caps_y = full_y
                            if low + radius < high - radius:
                                caps_y = (max(0.0, min(row + 1.0, low + radius) - max(row, low - radius))
                                          + max(0.0, min(row + 1.0, high + radius) - max(row, high - radius)))
                            covered += cap_x * caps_y
                        if covered > 0.0:
                            alpha = alpha_code * min(np.float64(1.0), covered)
                            inverse = np.float64(1.0) - alpha
                            for channel in range(3):
                                value = np.float64(colours[group, channel]) * alpha + np.float64(out[row, column, channel]) * inverse
                                out[row, column, channel] = np.uint8(min(np.float64(255.0), np.floor(value + np.float64(0.5))))
                            out[row, column, 3] = np.uint8(255)
                        row += 1


@njit(cache=True, inline="always")
def _agg_fill_channel(dst, src, alpha):
    """One channel of Agg's plain-RGBA fill over an opaque pixel.

    An opaque source is copied; a translucent one is blended the way
    ``blender_rgba_plain::blend_pix`` blends it, in integer arithmetic with
    the same truncating division, so a bar the kernel paints is the bar Agg
    paints, byte for byte.
    """

    if alpha >= 255:
        return src
    held = dst * 255
    scale = ((alpha + 255) << 8) - alpha * 255
    return (((src << 8) - held) * alpha + (held << 8)) // scale


@njit(cache=True, parallel=True, nogil=True)
def raster_histogram_bars(
    edges, tops, bases, offsets, surface_offsets, colours, clips, out
):
    """Paint histogram bars the way Agg paints an unstroked PolyCollection.

    Agg snaps a rectilinear path to the pixel grid before it fills it, and
    rounds the clip box to whole pixels too, so a bar has no antialiased
    edge at all: each bar is the pixel rectangle from ``floor(x + 1/2)`` to
    ``floor(x' + 1/2)`` across and from the snapped top to the snapped base
    down, cut to the snapped clip box, and every pixel in it takes the fill
    colour through Agg's own blend.  ``edges`` holds each surface's bin
    edges in canvas pixels (one more than its bars), ``tops`` the bar tops
    in canvas rows, one per EDGE so the two index alike (the last of a
    surface is padding), and ``bases`` the baseline row per surface;
    ``offsets`` cut the edge array per surface, ``colours`` is one RGBA per
    surface with the fill alpha folded in, and ``clips`` the surfaces'
    boxes already snapped.
    surface_offsets partitions distributions into disjoint surfaces. Groups
    on one surface blend sequentially; only different surfaces run in parallel.
    """

    height, width = out.shape[:2]
    surface_count = surface_offsets.size - 1
    for surface in prange(surface_count):
        clip_left = max(0, clips[surface, 0])
        clip_top = max(0, clips[surface, 1])
        clip_right = min(width, clips[surface, 2])
        clip_bottom = min(height, clips[surface, 3])
        if clip_right <= clip_left or clip_bottom <= clip_top:
            continue
        for group in range(surface_offsets[surface], surface_offsets[surface + 1]):
            start = offsets[group]
            stop = offsets[group + 1]
            red = int(colours[group, 0])
            green = int(colours[group, 1])
            blue = int(colours[group, 2])
            alpha = int(colours[group, 3])
            base = bases[group]
            if alpha <= 0 or not np.isfinite(base):
                continue
            base_row = int(np.floor(base + np.float64(0.5)))
            for bar in range(start, stop - 1):
                left_edge = edges[bar]
                right_edge = edges[bar + 1]
                top = tops[bar]
                if not (np.isfinite(left_edge) and np.isfinite(right_edge) and np.isfinite(top)):
                    continue
                x0 = int(np.floor(left_edge + np.float64(0.5)))
                x1 = int(np.floor(right_edge + np.float64(0.5)))
                if x1 < x0:
                    x0, x1 = x1, x0
                top_row = int(np.floor(top + np.float64(0.5)))
                y0 = max(min(top_row, base_row), clip_top)
                y1 = min(max(top_row, base_row), clip_bottom)
                x0 = max(x0, clip_left)
                x1 = min(x1, clip_right)
                for row in range(y0, y1):
                    for column in range(x0, x1):
                        out[row, column, 0] = np.uint8(
                            _agg_fill_channel(int(out[row, column, 0]), red, alpha)
                        )
                        out[row, column, 1] = np.uint8(
                            _agg_fill_channel(int(out[row, column, 1]), green, alpha)
                        )
                        out[row, column, 2] = np.uint8(
                            _agg_fill_channel(int(out[row, column, 2]), blue, alpha)
                        )
                        out[row, column, 3] = np.uint8(255)


@njit(cache=True, inline="always")
def _clamped_line_integral(value):
    """The integral of clamp(v, 0, 1) from minus infinity to ``value``."""

    if value <= 0.0:
        return np.float64(0.0)
    if value <= 1.0:
        return np.float64(0.5) * value * value
    return value - np.float64(0.5)


@njit(cache=True, inline="always")
def _slanted_cover(depth, grade):
    """How much of a pixel lies inside an edge ``depth`` past its centre,
    when the edge slants by ``grade`` across the pixel.

    A level edge is the one-pixel ramp, clamp(depth); a slanted one is the
    average of that ramp along the pixel, which is the integral of the
    clamped line between the edge's height at either side.
    """

    if grade <= np.float64(1.0e-9):
        return min(np.float64(1.0), max(np.float64(0.0), depth))
    half = np.float64(0.5) * grade
    return (
        _clamped_line_integral(depth + half) - _clamped_line_integral(depth - half)
    ) / grade


@njit(cache=True, inline="always")
def _disc_cover(px, py, x, y, radius):
    """The one-pixel ramp on a disc of ``radius`` about (x, y)."""

    dx = px - x
    dy = py - y
    return min(
        np.float64(1.0),
        max(np.float64(0.0), radius + np.float64(0.5) - np.sqrt(dx * dx + dy * dy)),
    )


@njit(cache=True, inline="always")
def _segment_cover(
    px, py, x0, y0, x1, y1, x2, y2, radius, cap_start, cap_end, has_next
):
    """What one stroked piece adds to the pixel centred at (px, py).

    THE STROKE OF A PIECE IS A BAND, 2r wide along the piece, whose two
    edges cross a pixel at the piece's own slant: read along the piece's
    MINOR axis -- horizontally for a steep piece at the pixel's row,
    vertically for a shallow one at its column -- each edge is the
    slanted-edge integral over the pixel, and the band's share of the
    pixel is the intersection of the two half-planes, cL + cR - 1.

    The band stops at the piece's ends.  Past the end of a polyline the
    cap PROJECTS: the band runs on for r and is cut square, as Matplotlib
    caps solid lines.  Past a vertex the join is ROUND, and the disc on
    the vertex is added by ONE of the two pieces meeting there -- this
    one, for the wedge outside both bands (beyond this piece's end and
    before the next piece's start); inside either band the band already
    counts.  Contributions ADD and saturate, which is the non-zero winding
    rule Agg fills a stroke's outline with: a smooth polyline's bands meet
    without overlap and sum to the union, and where a path folds back on
    itself the overlapping bands count twice up to full cover, exactly as
    Agg darkens the inner corner of a sharp turn.
    """

    dx = x1 - x0
    dy = y1 - y0
    length2 = dx * dx + dy * dy
    if length2 <= np.float64(0.0):
        return np.float64(0.0)
    along = ((px - x0) * dx + (py - y0) * dy) / length2
    length = np.sqrt(length2)
    beyond = np.float64(0.0)
    if along < np.float64(0.0):
        if not cap_start:
            return np.float64(0.0)
        beyond = -along * length
    elif along > np.float64(1.0):
        if not cap_end:
            if has_next:
                nx = x2 - x1
                ny = y2 - y1
                next2 = nx * nx + ny * ny
                if next2 > np.float64(0.0):
                    along_next = ((px - x1) * nx + (py - y1) * ny) / next2
                    if along_next >= np.float64(0.0):
                        return np.float64(0.0)
            return _disc_cover(px, py, x1, y1, radius)
        beyond = (along - np.float64(1.0)) * length
    if beyond > radius + np.float64(0.5):
        return np.float64(0.0)
    if abs(dy) > abs(dx):
        grade = abs(dx / dy)
        centre = x0 + (py - y0) * dx / dy
        half = radius * np.sqrt(np.float64(1.0) + grade * grade)
        cover = (
            _slanted_cover(px - (centre - half) + np.float64(0.5), grade)
            + _slanted_cover((centre + half) - px + np.float64(0.5), grade)
            - np.float64(1.0)
        )
    else:
        grade = abs(dy / dx)
        centre = y0 + (px - x0) * dy / dx
        half = radius * np.sqrt(np.float64(1.0) + grade * grade)
        cover = (
            _slanted_cover(py - (centre - half) + np.float64(0.5), grade)
            + _slanted_cover((centre + half) - py + np.float64(0.5), grade)
            - np.float64(1.0)
        )
    if cover <= np.float64(0.0):
        return np.float64(0.0)
    if beyond > np.float64(0.0):
        # The square cut of a projecting cap: a one-pixel ramp along the
        # axis, across the band's share.
        cover *= min(np.float64(1.0), radius + np.float64(0.5) - beyond)
    return cover


@njit(cache=True, nogil=True)
def _cut_pieces(
    vertices,
    start,
    stop,
    snap,
    snap_offset,
    piece_x0,
    piece_y0,
    piece_x1,
    piece_y1,
    piece_cap0,
    piece_cap1,
    base,
    write,
):
    """Cut the line [start, stop) into pieces, one per finite segment.

    A piece runs from one finite vertex to the next.  A non-finite vertex
    breaks the path, and the pieces at a break carry the caps: ``cap0`` on
    the first piece of a subpath, ``cap1`` on its last.  A rectilinear
    line's vertices are moved onto the pixel grid first (``snap``).  With
    ``write`` false only the count is taken, which is how the piece store
    is sized exactly before the second walk fills it from ``base``.
    """

    count = 0
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
        if write:
            if snap:
                x0 = np.floor(x0 + np.float64(0.5)) + snap_offset
                y0 = np.floor(y0 + np.float64(0.5)) + snap_offset
                x1 = np.floor(x1 + np.float64(0.5)) + snap_offset
                y1 = np.floor(y1 + np.float64(0.5)) + snap_offset
            piece_x0[base + count] = x0
            piece_y0[base + count] = y0
            piece_x1[base + count] = x1
            piece_y1[base + count] = y1
            piece_cap0[base + count] = point == start or not (
                np.isfinite(vertices[point - 1, 0])
                and np.isfinite(vertices[point - 1, 1])
            )
            piece_cap1[base + count] = point + 2 >= stop or not (
                np.isfinite(vertices[point + 2, 0])
                and np.isfinite(vertices[point + 2, 1])
            )
        count += 1
    return count



@njit(cache=True, parallel=True, nogil=True)
def raster_polylines(
    vertices, offsets, colours, widths, clips, lane_offsets, band_count, out
):
    """Stroke display polylines, piece by piece, antialiased like Agg.

    One lane owns one axes-worth of lines, exactly as the error-bar kernel
    groups its stems: a Facet grid's cells are disjoint pixel boxes, so its
    lanes stroke in parallel without a write race, while the lines INSIDE a
    lane keep their sequential painter order -- overlapping translucent
    strokes accumulate the way the artist scene composes them.  Callers
    prove disjointness (``_polyline_lane_offsets``); anything they cannot
    prove arrives as one lane, which is the serial behaviour.

    EVERY SEGMENT IS A PIECE (``_cut_pieces``).  The kernel strokes the
    line it is given; what a trace denser than the pixel grid looks like
    is not decided here.  The caller decides it once, thinning such a
    trace to each column's extremes before handing it to this kernel and
    to the Line2D Agg draws alike (``rendering._thinned_to_columns``), so
    the two stroke the same polyline.

    AGG SNAPS A RECTILINEAR PATH.  A line whose every segment is level or
    vertical, with fewer than 1024 vertices, has its vertices moved onto
    the pixel grid before it is stroked -- to pixel centres when the
    rounded stroke width is odd, to pixel edges when it is even, so the
    stroke's edges land on pixel edges either way.  The export draws
    through Agg, so this stroke snaps by the same rule.

    A lane with fewer peers than the pool has threads is then cut into
    column bands, each stroking every line of the lane in order over its
    own columns.  Within a line every piece is bucketed by the columns its
    centreline spans, and a column takes its coverage from the pieces
    bucketed within the stroke's reach of it: for each, over the rows it
    can touch at that column, the pixel ADDS what the piece gives it and
    saturates at full cover -- the non-zero winding accumulation Agg fills
    a stroke's outline with -- and is blended once.  A vertical piece
    covers each row of its length by the same amount at a given column,
    so those rows are set, not computed; only its ends are read pixel by
    pixel.
    """

    height, width = out.shape[:2]
    line_count = offsets.size - 1

    # Which lines snap, and by how much.
    snaps = np.zeros(line_count, dtype=np.bool_)
    snap_offsets = np.zeros(line_count, dtype=np.float64)
    for line in prange(line_count):
        start = offsets[line]
        stop = offsets[line + 1]
        snap = 2 <= stop - start < 1024
        if snap:
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
                if x0 != x1 and y0 != y1:
                    snap = False
                    break
        snaps[line] = snap
        snap_offsets[line] = (
            np.float64(0.5)
            if int(np.floor(np.float64(widths[line]) + np.float64(0.5))) % 2 == 1
            else np.float64(0.0)
        )

    # Cut every line into its pieces: a counting walk sizes the store, a
    # filling walk fills it, each line on its own thread.
    piece_offsets = np.zeros(line_count + 1, dtype=np.int64)
    scratch_f = np.empty(1, dtype=np.float64)
    scratch_b = np.empty(1, dtype=np.bool_)
    for line in prange(line_count):
        piece_offsets[line + 1] = _cut_pieces(
            vertices,
            offsets[line],
            offsets[line + 1],
            snaps[line],
            snap_offsets[line],
            scratch_f,
            scratch_f,
            scratch_f,
            scratch_f,
            scratch_b,
            scratch_b,
            0,
            False,
        )
    for line in range(line_count):
        piece_offsets[line + 1] += piece_offsets[line]
    total_pieces = piece_offsets[line_count]
    piece_x0 = np.empty(max(total_pieces, 1), dtype=np.float64)
    piece_y0 = np.empty(max(total_pieces, 1), dtype=np.float64)
    piece_x1 = np.empty(max(total_pieces, 1), dtype=np.float64)
    piece_y1 = np.empty(max(total_pieces, 1), dtype=np.float64)
    piece_cap0 = np.empty(max(total_pieces, 1), dtype=np.bool_)
    piece_cap1 = np.empty(max(total_pieces, 1), dtype=np.bool_)
    for line in prange(line_count):
        _cut_pieces(
            vertices,
            offsets[line],
            offsets[line + 1],
            snaps[line],
            snap_offsets[line],
            piece_x0,
            piece_y0,
            piece_x1,
            piece_y1,
            piece_cap0,
            piece_cap1,
            piece_offsets[line],
            True,
        )

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
        amount = np.zeros(height, dtype=np.float64)
        counts = np.zeros(width + 1, dtype=np.int64)
        for line in range(lane_offsets[lane], lane_offsets[lane + 1]):
            piece_first = piece_offsets[line]
            piece_last = piece_offsets[line + 1]
            if piece_last <= piece_first:
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
            # How far a stroke reaches from its centreline, measured along
            # either axis: a steep band's horizontal half-width is at most
            # r * sqrt(2), a projecting cap runs on r, and the pixel ramp
            # adds a half pixel either side.
            reach = radius * np.float64(1.4142135623730951) + np.float64(1.0)
            reach_columns = int(np.ceil(reach))
            fill_left = max(clip_left, paint_left - reach_columns)
            fill_right = min(clip_right, paint_right + reach_columns + 1)
            if fill_right <= fill_left:
                continue
            keep_left = np.float64(fill_left) - reach
            keep_right = np.float64(fill_right) + reach

            # Bucket this line's pieces by the columns their centrelines
            # span; a piece wholly beyond the band's reach paints nothing.
            piece_count = piece_last - piece_first
            first_column = np.empty(piece_count, dtype=np.int64)
            last_column = np.empty(piece_count, dtype=np.int64)
            for column in range(fill_left, fill_right + 1):
                counts[column] = 0
            for index in range(piece_count):
                x0 = piece_x0[piece_first + index]
                x1 = piece_x1[piece_first + index]
                if max(x0, x1) < keep_left or min(x0, x1) > keep_right:
                    first_column[index] = 1
                    last_column[index] = 0
                    continue
                low_column = max(fill_left, int(np.floor(min(x0, x1))))
                high_column = min(fill_right - 1, int(np.floor(max(x0, x1))))
                first_column[index] = low_column
                last_column[index] = high_column
                for column in range(low_column, high_column + 1):
                    counts[column] += 1
            total = 0
            for column in range(fill_left, fill_right):
                held = counts[column]
                counts[column] = total
                total += held
            counts[fill_right] = total
            entries = np.empty(max(total, 1), dtype=np.int64)
            for index in range(piece_count):
                for column in range(first_column[index], last_column[index] + 1):
                    entries[counts[column]] = index
                    counts[column] += 1
            for column in range(fill_right, fill_left, -1):
                counts[column] = counts[column - 1]
            counts[fill_left] = 0

            alpha_code = np.float64(colours[line, 3]) / np.float64(255.0)
            for column in range(paint_left, paint_right):
                cx = np.float64(column) + np.float64(0.5)
                window_left = cx - reach
                window_right = cx + reach
                rows_low = height
                rows_high = 0
                source_first = max(fill_left, column - reach_columns)
                source_last = min(fill_right - 1, column + reach_columns)
                for source in range(source_first, source_last + 1):
                    for slot in range(counts[source], counts[source + 1]):
                        index = entries[slot]
                        # A piece spanning several columns sits in each of
                        # them; it is read once per target column, from the
                        # first of its columns within reach.
                        if source != max(first_column[index], source_first):
                            continue
                        piece = piece_first + index
                        x0 = piece_x0[piece]
                        y0 = piece_y0[piece]
                        x1 = piece_x1[piece]
                        y1 = piece_y1[piece]
                        cap_start = piece_cap0[piece]
                        cap_end = piece_cap1[piece]
                        has_next = (not cap_end) and piece + 1 < piece_last
                        x2 = piece_x1[piece + 1] if has_next else x1
                        y2 = piece_y1[piece + 1] if has_next else y1
                        dx = x1 - x0
                        if dx == np.float64(0.0):
                            # A VERTICAL PIECE: along its length every row
                            # is covered by the same amount at this column.
                            level = min(
                                np.float64(1.0),
                                max(
                                    np.float64(0.0),
                                    radius + np.float64(0.5) - abs(cx - x0),
                                ),
                            )
                            y_low = min(y0, y1)
                            y_high = max(y0, y1)
                            inner_first = max(
                                clip_top, int(np.ceil(y_low - np.float64(0.5)))
                            )
                            inner_last = min(
                                clip_bottom, int(np.floor(y_high - np.float64(0.5))) + 1
                            )
                            if level > np.float64(0.0):
                                for row in range(inner_first, inner_last):
                                    amount[row] += level
                            row_first = max(clip_top, int(np.floor(y_low - reach)))
                            row_last = min(
                                clip_bottom, int(np.ceil(y_high + reach)) + 1
                            )
                            for row in range(row_first, row_last):
                                if inner_first <= row < inner_last:
                                    continue
                                amount[row] += _segment_cover(
                                    cx,
                                    np.float64(row) + np.float64(0.5),
                                    x0,
                                    y0,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    radius,
                                    cap_start,
                                    cap_end,
                                    has_next,
                                )
                        else:
                            # The rows this piece can touch AT THIS COLUMN:
                            # its own heights across the reach window, and
                            # the stroke's reach beyond them.
                            t_low = (window_left - x0) / dx
                            t_high = (window_right - x0) / dx
                            if t_low > t_high:
                                t_low, t_high = t_high, t_low
                            t_low = min(np.float64(1.0), max(np.float64(0.0), t_low))
                            t_high = min(np.float64(1.0), max(np.float64(0.0), t_high))
                            y_a = y0 + t_low * (y1 - y0)
                            y_b = y0 + t_high * (y1 - y0)
                            row_first = max(
                                clip_top, int(np.floor(min(y_a, y_b) - reach))
                            )
                            row_last = min(
                                clip_bottom, int(np.ceil(max(y_a, y_b) + reach)) + 1
                            )
                            for row in range(row_first, row_last):
                                amount[row] += _segment_cover(
                                    cx,
                                    np.float64(row) + np.float64(0.5),
                                    x0,
                                    y0,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    radius,
                                    cap_start,
                                    cap_end,
                                    has_next,
                                )
                        if row_last > row_first:
                            rows_low = min(rows_low, row_first)
                            rows_high = max(rows_high, row_last)
                for row in range(rows_low, rows_high):
                    covered = min(np.float64(1.0), amount[row])
                    amount[row] = np.float64(0.0)
                    # A share below a millionth is arithmetic noise from
                    # the edge integrals, not ink.
                    if covered <= np.float64(1.0e-6):
                        continue
                    alpha = alpha_code * covered
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


@njit(cache=True, inline="always")
def _agg_iround(value):
    """Agg's ``iround``: half away from zero, then truncation toward zero."""

    if value < 0.0:
        return int(value - 0.5)
    return int(value + 0.5)


@njit(cache=True, inline="always")
def _agg_dda_start(y1, y2, count):
    """The state of Agg's ``dda2_line_interpolator`` at its first pixel.

    The forward-adjusted constructor, in C integer arithmetic: the quotient
    truncates toward zero and the remainder takes the dividend's sign.
    Returns ``(y, lft, rem, mod)``.
    """

    cnt = count if count > 0 else 1
    delta = y2 - y1
    if delta >= 0:
        lft = delta // cnt
    else:
        lft = -((-delta) // cnt)
    rem = delta - lft * cnt
    mod = rem
    if mod <= 0:
        mod += count
        rem += count
        lft -= 1
    mod -= count
    return y1, lft, rem, mod


@njit(cache=True, parallel=True, nogil=True)
def raster_prepared_images(
    values,
    valid,
    use_valid,
    blits,
    clips,
    affines,
    lut,
    vmin32,
    span32,
    vmin64,
    span64,
    single,
    out,
):
    """Paint prepared Image surfaces the way ``imshow`` paints them.

    Matplotlib resamples an image through Agg's nearest filter: a row of the
    output is one span, whose source x runs from the span's first pixel
    centre to its last in 1/256 pixel fixed point along Agg's DDA, and
    whose source y is the row centre rounded to the same fixed point; a
    sample is the floor of each.  The resampled picture is then blitted at
    the truncated corner of its clipped box and cut to the graphics
    context's clip, and each sample's colour is ``Normalize`` and the
    colormap's 256 slots in the data's promoted dtype.  Every one of those
    steps is here, so the picture is matplotlib's byte for byte.

    ``blits`` holds each surface's (left, top, out_width, out_height) on
    the canvas, ``clips`` its clip box rounded as Agg rounds one, and
    ``affines`` the inverse of matplotlib's image transform as Agg holds
    it -- ``(sx, shy, shx, sy, tx, ty)``, mapping a point of the resampled
    picture, x rightward and y UPWARD from its bottom-left corner, to
    array pixel coordinates.  ``single`` says the promoted dtype is
    float32 (``vmin32``/``span32``), else float64.
    """

    cells, source_rows, source_columns = values.shape
    height, width = out.shape[:2]
    for work in prange(cells * height):
        cell = work // height
        row = work - cell * height
        blit_left = blits[cell, 0]
        blit_top = blits[cell, 1]
        out_width = blits[cell, 2]
        out_height = blits[cell, 3]
        v = row - blit_top
        if v < 0 or v >= out_height or out_width <= 0:
            continue
        clip_left = max(0, clips[cell, 0])
        clip_top = max(0, clips[cell, 1])
        clip_right = min(width, clips[cell, 2])
        clip_bottom = min(height, clips[cell, 3])
        if row < clip_top or row >= clip_bottom or clip_right <= clip_left:
            continue
        sx = affines[cell, 0]
        shy = affines[cell, 1]
        shx = affines[cell, 2]
        sy = affines[cell, 3]
        tx = affines[cell, 4]
        ty = affines[cell, 5]
        # The span is this row of the picture, counted upward from its
        # bottom (matplotlib's transform maps to y-up coordinates and the
        # blit flips the buffer), sampled at pixel centres; Agg interpolates
        # both source coordinates along it in 1/256 pixel steps with its DDA.
        y_up = np.float64(out_height - 1 - v) + 0.5
        x_first = _agg_iround((0.5 * sx + y_up * shx + tx) * 256.0)
        y_first = _agg_iround((0.5 * shy + y_up * sy + ty) * 256.0)
        u_last = np.float64(out_width) + 0.5
        x_last = _agg_iround((u_last * sx + y_up * shx + tx) * 256.0)
        y_last = _agg_iround((u_last * shy + y_up * sy + ty) * 256.0)
        x_fixed, x_lft, x_rem, x_mod = _agg_dda_start(x_first, x_last, out_width)
        y_fixed, y_lft, y_rem, y_mod = _agg_dda_start(y_first, y_last, out_width)
        for u in range(out_width):
            if u > 0:
                x_mod += x_rem
                x_fixed += x_lft
                if x_mod > 0:
                    x_mod -= out_width
                    x_fixed += 1
                y_mod += y_rem
                y_fixed += y_lft
                if y_mod > 0:
                    y_mod -= out_width
                    y_fixed += 1
            column = blit_left + u
            if column < clip_left or column >= clip_right:
                continue
            source_column = x_fixed >> 8
            source_row = y_fixed >> 8
            if (
                source_column < 0
                or source_column >= source_columns
                or source_row < 0
                or source_row >= source_rows
            ):
                continue
            if use_valid and not valid[cell, source_row, source_column]:
                continue
            if single:
                # The plane is float32; the limit and the span are float64.
                # NumPy subtracts and divides in float64 and stores each
                # result back into the float32 plane, then scales by the
                # colormap's 256 slots in float32 (exact, a power of two).
                shifted = np.float32(
                    np.float64(np.float32(values[cell, source_row, source_column])) - vmin64
                )
                unit = np.float32(np.float64(shifted) / span64)
                scaled = np.float64(unit * np.float32(256.0))
            else:
                scaled = (
                    (np.float64(values[cell, source_row, source_column]) - vmin64)
                    / span64
                ) * 256.0
            if scaled < 0.0:
                code = 0
            elif scaled >= 256.0:
                code = 255
            else:
                code = int(scaled)
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
def transform_polylines(vertices, offsets, matrices, affine, canvas_height, out):
    """Every polyline's vertices through its own affine, into canvas rows.

    ``matrices`` holds the two rows of each line's 3x3 display affine,
    flattened -- ``(a, c, e, b, d, f)`` in Matplotlib's Affine2D order --
    so a vertex lands where ``transform_affine`` would put it.  A line
    whose ``affine`` flag is off arrives already in display coordinates
    and is only turned into canvas rows.
    """

    for line in prange(offsets.size - 1):
        start = offsets[line]
        stop = offsets[line + 1]
        if affine[line]:
            a = matrices[line, 0]
            c = matrices[line, 1]
            e = matrices[line, 2]
            b = matrices[line, 3]
            d = matrices[line, 4]
            f = matrices[line, 5]
            for point in range(start, stop):
                x = vertices[point, 0]
                y = vertices[point, 1]
                out[point, 0] = a * x + c * y + e
                out[point, 1] = canvas_height - (b * x + d * y + f)
        else:
            for point in range(start, stop):
                out[point, 0] = vertices[point, 0]
                out[point, 1] = canvas_height - vertices[point, 1]


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

