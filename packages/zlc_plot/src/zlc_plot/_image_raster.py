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
    if all_valid:
        rows, columns = values.shape
        row_block = rows // row_starts.size
        column_block = columns // column_starts.size
        if (
            rows == row_block * row_starts.size
            and columns == column_block * column_starts.size
        ):
            # Evenly divisible blocks — the common power-of-two camera case —
            # reduce with one vectorized reshape mean, several times faster
            # than the ragged reduceat partition below.
            return values.reshape(
                row_starts.size,
                row_block,
                column_starts.size,
                column_block,
            ).mean(axis=(1, 3), dtype=mean_dtype)
    source = values if all_valid else np.where(valid, values, 0)
    if (
        all_valid
        and mean_dtype == np.float32
        and kernels.engaged()
        and kernels.block_sums_are_exact(
            values.dtype, row_starts, column_starts, values.shape
        )
    ):
        # The compiled kernel's exact integer sum IS this reduction's answer
        # while every partial stays exactly representable, which the judge
        # above establishes from the dtype alone.  ``reduceat`` books a
        # segment per output cell -- two million of them, each one or two
        # samples wide -- and that bookkeeping, not the addition, is the cost.
        summed = np.empty(
            (row_starts.size, column_starts.size), dtype=np.float32
        )
        kernels.block_sum_unsigned(
            np.ascontiguousarray(source), row_starts, column_starts, summed
        )
    else:
        summed = _reduce_blocks(source, row_starts, column_starts, mean_dtype)
    if all_valid:
        row_counts = np.diff(np.r_[row_starts, values.shape[0]])
        column_counts = np.diff(np.r_[column_starts, values.shape[1]])
        np.divide(summed, row_counts[:, np.newaxis], out=summed)
        np.divide(summed, column_counts[np.newaxis, :], out=summed)
        return summed
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
    """Prepared-front LRU plus a per-revision mip pyramid for one image.

    Wheel zoom and pan change the viewport every step, so the exact-key cache
    alone misses constantly.  The pyramid serves those misses from a coarser
    power-of-two level whose 2x2 block means compose exactly with the final
    area reduction, making gesture-time preparation O(display pixels) instead
    of O(source pixels).  Masked sources bypass the pyramid: per-level count
    planes are not worth their complexity for the rare sparse-validity case.
    """

    _LRU_CAPACITY = 6
    _MAX_LEVEL = 16

    def __init__(self) -> None:
        self._fronts: "OrderedDict[tuple, PreparedImageFront]" = OrderedDict()
        self._pyramid_token: tuple | None = None
        self._pyramid: dict[int, np.ndarray] = {}

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
        source = np.asarray(values)
        level_values = values
        level = 1
        if source.ndim == 2 and source.dtype.kind in "biuf" and (
            validity is None or _all_true(np.asarray(validity))
        ):
            level = self._pick_level(
                source.shape,
                extent,
                x_limits=x_limits,
                y_limits=y_limits,
                display_pixel_shape=display_pixel_shape,
            )
            if level > 1:
                level_values = self._level(source, level, revision_token)
                validity = None
        prepared = prepare_image_front(
            level_values,
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

    def _pick_level(
        self,
        shape: tuple[int, int],
        extent: tuple[float, float, float, float],
        *,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        display_pixel_shape: tuple[int, int],
    ) -> int:
        rows, columns = shape
        left, right, bottom, top = (float(value) for value in extent)
        column_start, column_stop = _index_window(
            float(x_limits[0]), float(x_limits[1]), left, right, columns
        )
        row_start, row_stop = _index_window(
            float(y_limits[0]), float(y_limits[1]), top, bottom, rows
        )
        display_width, display_height = display_pixel_shape
        row_block = (row_stop - row_start) / max(1, int(display_height))
        column_block = (column_stop - column_start) / max(1, int(display_width))
        block = min(row_block, column_block)
        level = 1
        # Halve while at least one source sample per display pixel remains
        # and the source divides evenly.  A residual oversample below the
        # reduction policy's ratio is left to Matplotlib's resample stage —
        # the same treatment fractional-DPR fronts already receive.
        while (
            level * 2 <= self._MAX_LEVEL
            and block / (level * 2) >= 1.0
            and rows % (level * 2) == 0
            and columns % (level * 2) == 0
        ):
            level *= 2
        return level

    def _level(
        self,
        source: np.ndarray,
        level: int,
        revision_token: tuple,
    ) -> np.ndarray:
        if self._pyramid_token != revision_token:
            self._pyramid_token = revision_token
            self._pyramid = {}
        cached = self._pyramid.get(level)
        if cached is not None:
            return cached
        rows, columns = source.shape
        mean_dtype = np.result_type(source.dtype, np.float32)
        # One vectorized pass straight to the requested level: cascading
        # through intermediate halvings would re-read the full plane once
        # per step for identical block means.
        blocks = source.reshape(
            rows // level,
            level,
            columns // level,
            level,
        )
        if source.dtype.kind in "bui" and source.dtype.itemsize <= 2:
            # Integer sources sum exactly in int32 (block sums stay far below
            # 2**31) and the float32 mean of the same block is exact too (a
            # block sum of <=2**20 fits float32's 24-bit mantissa), so summing
            # with SIMD integer adds and scaling once is bit-identical to
            # mean() while roughly halving the pass over a camera frame.
            reduced = blocks.sum(axis=(1, 3), dtype=np.int32).astype(
                mean_dtype
            )
            reduced *= mean_dtype.type(1.0) / mean_dtype.type(level * level)
        else:
            reduced = blocks.mean(axis=(1, 3), dtype=mean_dtype)
        reduced.setflags(write=False)
        self._pyramid[level] = reduced
        return reduced


__all__ = [
    "ImageFrontPolicy",
    "ImageFrontStore",
    "PreparedImageFront",
    "prepare_image_front",
]
