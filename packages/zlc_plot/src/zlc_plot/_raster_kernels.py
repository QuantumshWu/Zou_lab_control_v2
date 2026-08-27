"""Numba kernels for the four hot passes of the display-front pipeline.

Like :mod:`_height3d_scanline`, this module exists for SPEED ONLY.  Each
kernel mirrors a numpy reference that stays where it lives and stays the
specification -- the block-mean in :mod:`_image_raster`, the colour pass and
the box resize in :mod:`rendering`, the uniform histogram in
:mod:`data_view`.  Every kernel here reproduces its reference operation for
operation, in the same dtypes and the same order, and a standing contract
test runs both and asserts bit equality, so the two can never drift apart
silently.

Why they are faster is the same story four times: numpy answers each of
these questions with several full passes over a megapixel plane (a copy, a
scaled plane, a clipped plane, an index plane, a gather), or with a
general-purpose routine paying for a generality the caller does not need
(``np.add.reduceat`` books a segment for every one of two million output
cells whose blocks are one or two samples wide).  The kernels touch each
output once, in parallel, with the bookkeeping in registers.

Compilation caches on disk under ``NUMBA_CACHE_DIR`` exactly as the
scanline engine does; ``ZLC_PLOT_KERNELS=numpy`` forces every dispatch back
to its reference, which is how the contract test compares them.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np

if "NUMBA_CACHE_DIR" not in os.environ:
    _repo_root = pathlib.Path(__file__).resolve().parents[4]
    os.environ["NUMBA_CACHE_DIR"] = str(_repo_root / ".numba_cache")

try:  # pragma: no cover - absence is exercised by the dispatch fallback
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

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


# --------------------------------------------------------------- block sums
#: Largest exactly-representable integer in float32.  A block sum above it
#: would round, and the reference accumulates in float32 too -- so the exact
#: integer sum this kernel computes is the reference's answer only below it.
FLOAT32_EXACT_INTEGER = 1 << 24


def block_sums_are_exact(dtype: Any, row_starts: Any, column_starts: Any,
                         shape: tuple[int, int]) -> bool:
    """Whether every block sum is exact in float32, for any values of *dtype*.

    The reference reduces with ``dtype=float32``; each partial sum is exact
    while it stays under 2**24, and then the whole reduction equals the
    integer sum this kernel computes.  Bounded by the dtype rather than by
    the data so the answer costs no pass over the pixels.
    """

    if dtype.kind not in "u" or dtype.itemsize > 4:
        return False
    rows, columns = shape
    row_block = int(np.diff(np.r_[np.asarray(row_starts), rows]).max())
    column_block = int(np.diff(np.r_[np.asarray(column_starts), columns]).max())
    return (row_block * column_block * int(np.iinfo(dtype).max)
            < FLOAT32_EXACT_INTEGER)


@njit(cache=True, parallel=True, nogil=True)
def block_sum_unsigned(values, row_starts, column_starts, out):
    """Sum each block of an unsigned plane into ``out`` as float32.

    Mirrors ``np.add.reduceat`` twice over: a block's samples are summed in
    row-major order, which for exact integers is every order.
    """

    row_count = row_starts.size
    column_count = column_starts.size
    rows, columns = values.shape
    for i in prange(row_count):
        row_stop = row_starts[i + 1] if i + 1 < row_count else rows
        for j in range(column_count):
            column_stop = column_starts[j + 1] if j + 1 < column_count else columns
            total = np.uint64(0)
            for r in range(row_starts[i], row_stop):
                for c in range(column_starts[j], column_stop):
                    total += np.uint64(values[r, c])
            out[i, j] = np.float32(total)


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
def uniform_histogram(values, edges, bins, partials, out):
    """Count uniformly binned samples in one pass.

    Mirrors numpy's equal-bin path operation for operation: the same
    inclusive range filter, the same ``((a - first) / (last - first)) *
    bins`` index, the same truncating cast, and the same two corrections
    against the real edges that make the answer independent of the last
    ULP.  ``partials`` is a caller-owned ``(threads, bins)`` scratch plane
    so this kernel holds no global state and can be cached.
    """

    first = edges[0]
    last = edges[bins]
    denominator = last - first
    threads = partials.shape[0]
    chunk = (values.size + threads - 1) // threads
    for t in prange(threads):
        stop = min((t + 1) * chunk, values.size)
        for b in range(bins):
            partials[t, b] = 0
        for p in range(t * chunk, stop):
            # numpy filters on the raw dtype and only then casts to the edge
            # dtype; against float64 scalars every integer and float32
            # comparison promotes the same way, so one cast here is both.
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
            partials[t, index] += 1
    for b in range(bins):
        total = np.int64(0)
        for t in range(threads):
            total += partials[t, b]
        out[b] = total


def histogram_threads() -> int:
    """How many lanes :func:`uniform_histogram` should be given."""

    if not HAVE_NUMBA:
        return 1
    from numba import get_num_threads

    return int(get_num_threads())

# ------------------------------------------------------------------ extrema
@njit(cache=True, parallel=True, nogil=True)
def finite_extrema(values, valid, use_valid, out):
    """One pass for ``(finite count, min, max)`` over a masked pool.

    Mirrors ``isfinite`` + ``any`` + ``min(where=)`` + ``max(where=)``: four
    full reads of a two-million-value pool, and a bool plane the size of it,
    to answer three numbers.  Extrema are order-independent, so the parallel
    partials are the same numbers the reductions produce.
    """

    threads = out.shape[0] - 1
    chunk = (values.size + threads - 1) // threads
    for t in prange(threads):
        stop = min((t + 1) * chunk, values.size)
        count = 0.0
        low = np.inf
        high = -np.inf
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
        out[t, 0] = count
        out[t, 1] = low
        out[t, 2] = high
    total = 0.0
    lowest = np.inf
    highest = -np.inf
    for t in range(threads):
        total += out[t, 0]
        if out[t, 1] < lowest:
            lowest = out[t, 1]
        if out[t, 2] > highest:
            highest = out[t, 2]
    out[threads, 0] = total
    out[threads, 1] = lowest
    out[threads, 2] = highest


def masked_finite_extrema(values: Any, valid: Any) -> tuple[int, float, float] | None:
    """``(count, low, high)`` for a flat float pool, or ``None`` to defer."""

    if not engaged():
        return None
    flat = np.ascontiguousarray(values).reshape(-1)
    if flat.dtype.kind != "f" or not flat.size:
        return None
    use_valid = valid is not None
    if use_valid:
        mask = np.ascontiguousarray(np.asarray(valid, dtype=np.bool_)).reshape(-1)
        if mask.size != flat.size:
            return None
    else:
        mask = np.zeros(1, dtype=np.bool_)
    threads = 1
    if HAVE_NUMBA:
        from numba import get_num_threads

        threads = int(get_num_threads())
    out = np.empty((threads + 1, 3), dtype=np.float64)
    finite_extrema(flat, mask, use_valid, out)
    return int(out[threads, 0]), float(out[threads, 1]), float(out[threads, 2])

