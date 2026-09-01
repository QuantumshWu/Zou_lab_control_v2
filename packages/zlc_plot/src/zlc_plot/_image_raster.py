"""One policy-owned display front for regular two-dimensional images.

The immutable source remains authoritative for selection and analysis.  This
module prepares the scalar front handed to Matplotlib through cropping and
optional area reduction; the renderer always presents it with nearest pixels.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from . import _raster_kernels as kernels


@dataclass(frozen=True, slots=True)
class ImageFrontPolicy:
    """Bounded-work policy for one scalar image front.

    Area reduction is worthwhile only when a source dimension materially
    oversamples its physical output dimension.  Smaller differences stay in
    the source dtype and are resolved once by Matplotlib's scalar-data stage.
    """

    minimum_reduction_ratio: float = 1.25


@dataclass(frozen=True, slots=True)
class PreparedImageFront:
    """One scalar display front and the exact data extent it represents."""

    values: np.ndarray | np.ma.MaskedArray
    extent: tuple[float, float, float, float]


#: Untouched stand-ins for the merged block-sum kernel's masked face.
_NO_VALID = np.zeros((1, 1), dtype=np.bool_)
_NO_VALID.setflags(write=False)
_NO_COUNTS = np.zeros((1, 1), dtype=np.int64)


def _all_true(values: np.ndarray) -> bool:
    array = np.asarray(values, dtype=np.bool_)
    if not array.size:
        return True
    if all(stride == 0 for stride in array.strides):
        return bool(array.flat[0])
    return bool(np.all(array))


def _index_window(
    low: float,
    high: float,
    edge_start: float,
    edge_stop: float,
    count: int,
) -> tuple[int, int]:
    step = (edge_stop - edge_start) / count
    first = (low - edge_start) / step
    second = (high - edge_start) / step
    if first > second:
        first, second = second, first
    return (
        max(0, min(int(math.floor(first)), count - 1)),
        max(1, min(int(math.ceil(second)), count)),
    )


def _reduction_starts(
    size: int,
    target: int,
    minimum_ratio: float,
) -> np.ndarray:
    count = (
        target
        if size > target and size / target >= minimum_ratio
        else size
    )
    return np.arange(count, dtype=np.intp) * size // count


def _reduce_blocks(
    values: np.ndarray,
    row_starts: np.ndarray,
    column_starts: np.ndarray,
    dtype: Any,
) -> np.ndarray:
    rows, columns = values.shape
    if row_starts.size == rows:
        return np.add.reduceat(values, column_starts, axis=1, dtype=dtype)
    if column_starts.size == columns:
        return np.add.reduceat(values, row_starts, axis=0, dtype=dtype)
    if row_starts.size * columns <= rows * column_starts.size:
        reduced = np.add.reduceat(values, row_starts, axis=0, dtype=dtype)
        return np.add.reduceat(reduced, column_starts, axis=1, dtype=dtype)
    reduced = np.add.reduceat(values, column_starts, axis=1, dtype=dtype)
    return np.add.reduceat(reduced, row_starts, axis=0, dtype=dtype)


def _area_mean(
    values: np.ndarray,
    valid: np.ndarray,
    row_starts: np.ndarray,
    column_starts: np.ndarray,
) -> np.ndarray | np.ma.MaskedArray:
    all_valid = _all_true(valid)
    mean_dtype = np.result_type(values.dtype, np.float32)
    shape = (row_starts.size, column_starts.size)
    compiled = kernels.engaged()
    counts = None
    if all_valid and compiled and kernels.block_sums_are_exact(
        values.dtype, row_starts, column_starts, values.shape
    ):
        # The compiled kernel's exact integer sum IS this reduction's answer
        # while every partial stays exactly representable, which the judge
        # above establishes from the dtype alone.  ``reduceat`` books a
        # segment per output cell -- two million of them, each one or two
        # samples wide -- and that bookkeeping, not the addition, is the cost.
        summed = np.empty(shape, dtype=np.float32)
        kernels.block_sum_unsigned(
            kernels.readable(values), row_starts, column_starts, summed
        )
    elif all_valid and compiled:
        # Everything the exact-integer judge turns away -- every floating
        # plane, and the wide integers whose partials would round.  The
        # kernel accumulates in float64, so for a float32 plane it is not a
        # looser answer than ``reduceat`` (which accumulates in float32) but
        # a tighter one; measured 9.24 ms -> 0.25 ms on a 1200x1920 plane,
        # and 35.1 -> 0.46 on 2048x2048.
        summed = np.empty(shape, dtype=mean_dtype)
        kernels.block_sum_valid(
            kernels.readable(values),
            _NO_VALID,
            False,
            row_starts,
            column_starts,
            summed,
            _NO_COUNTS,
        )
    elif compiled:
        # Missing samples, in ONE pass.  The reference builds a whole
        # ``np.where(valid, values, 0)`` plane and then reduces twice, once
        # for the sum and once for the count.
        summed = np.empty(shape, dtype=mean_dtype)
        counts = np.empty(shape, dtype=np.int64)
        kernels.block_sum_valid(
            kernels.readable(values),
            kernels.readable(valid),
            True,
            row_starts,
            column_starts,
            summed,
            counts,
        )
    else:
        # NO COMPILED KERNEL.  The reshape mean below is the fastest thing
        # numpy alone can do here and the ragged partition is the general
        # answer; both are slower than the kernels above, so this whole
        # branch is what the interpreter falls back to, never a shortcut
        # taken ahead of them.  Standing above the dispatch, the evenly
        # divisible case -- every power-of-two camera frame, which is to
        # say the common one -- returned from here and the kernels never
        # ran at all: 7.18 ms against 1.01 reducing 2048 to 512, and 4.73
        # against 0.31 reducing it to 256, for the identical answer.
        source = values if all_valid else np.where(valid, values, 0)
        rows, columns = values.shape
        row_block = rows // row_starts.size
        column_block = columns // column_starts.size
        if all_valid and (
            rows == row_block * row_starts.size
            and columns == column_block * column_starts.size
        ):
            return source.reshape(
                row_starts.size,
                row_block,
                column_starts.size,
                column_block,
            ).mean(axis=(1, 3), dtype=mean_dtype)
        summed = _reduce_blocks(source, row_starts, column_starts, mean_dtype)
    if all_valid:
        # IN THE SUM'S OWN DTYPE.  ``np.diff`` answers in the index dtype,
        # and float32 divided by int64 is promoted to float64 for the whole
        # array and demoted again on the way into ``out`` -- two float64
        # passes over one and three quarter million cells, which measured
        # 3.86 ms against 0.55 for the float32 division they stand in for,
        # and cost more than the block sum they divide.
        counts_dtype = summed.dtype
        row_counts = np.diff(np.r_[row_starts, values.shape[0]]).astype(
            counts_dtype, copy=False
        )
        column_counts = np.diff(np.r_[column_starts, values.shape[1]]).astype(
            counts_dtype, copy=False
        )
        np.divide(summed, row_counts[:, np.newaxis], out=summed)
        np.divide(summed, column_counts[np.newaxis, :], out=summed)
        return summed
    if counts is None:
        counts = _reduce_blocks(valid, row_starts, column_starts, np.int64)
    means = np.zeros(summed.shape, dtype=mean_dtype)
    np.divide(summed, counts, out=means, where=counts != 0)
    return means if bool(np.all(counts)) else np.ma.array(
        means, mask=counts == 0, copy=False
    )


def prepare_image_front(
    values: Any,
    validity: Any | None,
    extent: tuple[float, float, float, float],
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    display_pixel_shape: tuple[int, int],
    policy: ImageFrontPolicy,
) -> PreparedImageFront:
    """Crop and, when policy requires it, reduce one regular scalar image.

    ``extent`` follows Matplotlib's ``(left, right, bottom, top)`` order.  The
    first source row is associated with ``top`` because image plots use an
    upper origin.  A source that does not cross the authored reduction ratio
    keeps its dtype and shares source memory.  Only a materially oversampled
    front is promoted to a floating area mean.
    """

    if not isinstance(policy, ImageFrontPolicy):
        raise TypeError("policy must be ImageFrontPolicy")
    source = np.asarray(values)
    if source.ndim != 2 or min(source.shape) < 1:
        raise ValueError("image values must be a non-empty two-dimensional array")
    if source.dtype.kind not in "biuf":
        raise TypeError("image values must be real numeric")
    if validity is None:
        valid = np.broadcast_to(np.asarray(True, dtype=np.bool_), source.shape)
    else:
        valid = np.asarray(validity)
        if valid.dtype != np.dtype(np.bool_):
            raise TypeError("image validity must have boolean dtype")
        try:
            valid = np.broadcast_to(valid, source.shape)
        except ValueError as error:
            raise ValueError(
                "image validity cannot broadcast to the image shape"
            ) from error

    prepared_extent = tuple(float(value) for value in extent)
    if len(prepared_extent) != 4 or not all(
        math.isfinite(value) for value in prepared_extent
    ):
        raise ValueError("image extent must contain four finite values")
    left, right, bottom, top = prepared_extent
    if left == right or bottom == top:
        raise ValueError("image extent spans must be non-degenerate")
    x_view = tuple(float(value) for value in x_limits)
    y_view = tuple(float(value) for value in y_limits)
    if len(x_view) != 2 or len(y_view) != 2 or not all(
        math.isfinite(value) for value in (*x_view, *y_view)
    ):
        raise ValueError("image view limits must contain finite pairs")
    width, height = display_pixel_shape
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or int(width) <= 0
        or int(height) <= 0
    ):
        raise ValueError("display pixel dimensions must be positive integers")
    display_width, display_height = int(width), int(height)

    rows, columns = source.shape
    column_start, column_stop = _index_window(
        *x_view,
        left,
        right,
        columns,
    )
    # Row zero is the image's top edge, even when the displayed Y direction is
    # reversed.  Walking from top to bottom therefore follows source order.
    row_start, row_stop = _index_window(
        *y_view,
        top,
        bottom,
        rows,
    )
    subset = source[row_start:row_stop, column_start:column_stop]
    subset_valid = valid[row_start:row_stop, column_start:column_stop]
    if source.dtype.kind == "f":
        finite = np.isfinite(subset)
        subset_valid = (
            finite
            if _all_true(subset_valid)
            else np.logical_and(subset_valid, finite)
        )

    subset_rows, subset_columns = subset.shape
    row_starts = _reduction_starts(
        subset_rows,
        display_height,
        policy.minimum_reduction_ratio,
    )
    column_starts = _reduction_starts(
        subset_columns,
        display_width,
        policy.minimum_reduction_ratio,
    )
    reduced = row_starts.size < subset_rows or column_starts.size < subset_columns
    if not reduced:
        shown: np.ndarray | np.ma.MaskedArray = (
            subset
            if _all_true(subset_valid)
            else np.ma.array(
                subset,
                mask=np.logical_not(subset_valid),
                copy=False,
            )
        )
    else:
        shown = _area_mean(subset, subset_valid, row_starts, column_starts)
    shown.setflags(write=False)

    x_step = (right - left) / columns
    y_step = (bottom - top) / rows
    shown_extent = (
        left + column_start * x_step,
        left + column_stop * x_step,
        top + row_stop * y_step,
        top + row_start * y_step,
    )
    return PreparedImageFront(
        shown,
        tuple(float(value) for value in shown_extent),
    )


class ImageFrontStore:
    """Prepared-front LRU for one image.

    Wheel zoom and pan change the viewport every step, so the exact-key
    cache alone misses constantly, and each miss reduces the source again.
    This used to hold a mip pyramid as well: a coarser power-of-two level
    whose 2x2 block means compose exactly with the final area reduction,
    so a gesture's preparation cost O(display pixels) instead of O(source
    pixels).

    IT WAS WRITTEN BEFORE THE COMPILED BLOCK SUM EXISTED, and that kernel
    took its reason away -- reducing straight from the source is now 0.7 to
    2 ms, while building one level costs 7 to 14 ms on a floating frame.
    Measured across three dtypes at 2048 square: on a first frame reducing
    directly won by 3.3x and 10.5x; over a zoom the two were within a fifth
    of each other either way, so at best the level paid itself back after a
    hundred and thirty steps and at worst never; panning at a fixed zoom,
    twelve cases across two display densities, it won none and tied one.

    A source narrow enough to sum exactly already bypassed it, by a judge
    written for exactly this reason.  That judge, the levels, their cache
    and the token that invalidated it are all gone with it.
    """

    _LRU_CAPACITY = 6

    def __init__(self) -> None:
        self._fronts: "OrderedDict[tuple, PreparedImageFront]" = OrderedDict()

    def prepare(
        self,
        values: Any,
        validity: Any | None,
        extent: tuple[float, float, float, float],
        *,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        display_pixel_shape: tuple[int, int],
        policy: ImageFrontPolicy,
        revision_token: tuple,
    ) -> PreparedImageFront:
        front_key = (
            revision_token,
            tuple(map(float, extent)),
            tuple(map(float, x_limits)),
            tuple(map(float, y_limits)),
            tuple(map(int, display_pixel_shape)),
            policy,
        )
        cached = self._fronts.get(front_key)
        if cached is not None:
            self._fronts.move_to_end(front_key)
            return cached
        prepared = prepare_image_front(
            values,
            validity,
            extent,
            x_limits=x_limits,
            y_limits=y_limits,
            display_pixel_shape=display_pixel_shape,
            policy=policy,
        )
        self._fronts[front_key] = prepared
        while len(self._fronts) > self._LRU_CAPACITY:
            self._fronts.popitem(last=False)
        return prepared


__all__ = [
    "ImageFrontPolicy",
    "ImageFrontStore",
    "PreparedImageFront",
    "prepare_image_front",
]
