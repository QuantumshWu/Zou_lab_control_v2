"""Optimized separable regular-grid image fitting.

The regular-image solver is isolated from the coordinate-array engine.  The
public fit catalogue supplies the model and unit semantics; this module owns
the stripe/BLAS numerical implementation shared by every separable Gaussian
image model (the built-in radial and anisotropic centers).

The solver never expands per-pixel coordinate grids.  It seeds on a bounded
proxy image, climbs a subsample ladder, and finishes with the exact
full-resolution convergence criterion.  Result arrays are deferred: the
returned :class:`FitResult` retains only the fit input and parameters and
materializes ``fitted_values``/``residuals``/``selected_indices`` on first
access.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
import threading
import time
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import minimize

from .fit import (
    ArrayTuple,
    FitCancelled,
    FitDeadlineExceeded,
    FitModelSpec,
    FitOptions,
    FitResult,
    RegularImageFitInput,
    _covariance_from_information,
    _DeferredFitData,
    _initial_candidates,
    _initial_values,
    _solver_bounds,
    _span,
    _value_range,
)


__all__ = ["fit_regular_separable_image"]


@dataclass(frozen=True, slots=True)
class _RegularImageSummary:
    minimum: float
    maximum: float
    scale: float
    count: int
    all_valid: bool
    normalized_sum: float
    normalized_square_sum: float


@dataclass(frozen=True, slots=True)
class _SolverStatus:
    success: bool
    message: str


_REGULAR_IMAGE_STRIPE_ROWS = 64
_REGULAR_IMAGE_SAMPLE_LIMIT = 257
_REGULAR_IMAGE_LADDER_FACTOR = 4
_REGULAR_IMAGE_MIN_RELATIVE_CONTRAST = 0.05
_REGULAR_IMAGE_MAX_LINE_SEARCH_STEPS = 50
_REGULAR_IMAGE_MAX_NEWTON_STEPS = 8
_REGULAR_IMAGE_FTOL = 1e-10
_REGULAR_IMAGE_GTOL = 1e-8
_REGULAR_IMAGE_STRIPE_WORKERS = 4


_STRIPE_POOL: ThreadPoolExecutor | None = None
_STRIPE_POOL_LOCK = threading.Lock()


def _stripe_pool() -> ThreadPoolExecutor:
    """Shared small pool for stripe sweeps (numpy releases the GIL)."""

    global _STRIPE_POOL
    if _STRIPE_POOL is None:
        with _STRIPE_POOL_LOCK:
            if _STRIPE_POOL is None:
                _STRIPE_POOL = ThreadPoolExecutor(
                    max_workers=min(
                        _REGULAR_IMAGE_STRIPE_WORKERS, os.cpu_count() or 1
                    ),
                    thread_name_prefix="zlc-fit-stripe",
                )
    return _STRIPE_POOL


@dataclass(frozen=True, slots=True)
class _SeparableKernel:
    """Separable structure of one regular-image Gaussian model.

    Both built-in image models factor into per-axis vectors: the basis
    ``exp(-delta**2 / radius**2)`` plus its radius and center derivative
    terms.  ``geometry_terms`` lists, per geometry parameter (every model
    parameter after amplitude and offset, in model order), the outer-product
    terms of the model derivative as ``(y_vector_index, x_vector_index)``
    pairs into ``(basis, radius_term, center_term)``.
    """

    capability: str
    parameter_count: int
    x_radius_index: int
    y_radius_index: int
    geometry_terms: tuple[tuple[tuple[int, int], ...], ...]
    natural_scale_builder: Callable[[float, float, float], tuple[float, ...]]

    def x_vectors(
        self,
        parameters: np.ndarray,
        x_coordinates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _axis_terms(
            x_coordinates,
            float(parameters[-2]),
            float(parameters[self.x_radius_index]),
        )

    def y_vectors(
        self,
        parameters: np.ndarray,
        y_coordinates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _axis_terms(
            y_coordinates,
            float(parameters[-1]),
            float(parameters[self.y_radius_index]),
        )

    def axis_vectors(
        self,
        parameters: np.ndarray,
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        return (
            self.x_vectors(parameters, x_coordinates),
            self.y_vectors(parameters, y_coordinates),
        )


_RADIAL_KERNEL = _SeparableKernel(
    capability="regular_image_radial",
    parameter_count=5,
    x_radius_index=2,
    y_radius_index=2,
    geometry_terms=(((0, 1), (1, 0)), ((0, 2),), ((2, 0),)),
    natural_scale_builder=lambda value_range, x_span, y_span: (
        value_range,
        value_range,
        max(x_span, y_span) / 4.0,
        x_span / 4.0,
        y_span / 4.0,
    ),
)

_ANISOTROPIC_KERNEL = _SeparableKernel(
    capability="regular_image_separable",
    parameter_count=6,
    x_radius_index=2,
    y_radius_index=3,
    geometry_terms=(((0, 1),), ((1, 0),), ((0, 2),), ((2, 0),)),
    natural_scale_builder=lambda value_range, x_span, y_span: (
        value_range,
        value_range,
        x_span / 4.0,
        y_span / 4.0,
        x_span / 4.0,
        y_span / 4.0,
    ),
)

_KERNELS = (_RADIAL_KERNEL, _ANISOTROPIC_KERNEL)


def _kernel_for(model: FitModelSpec) -> _SeparableKernel:
    for kernel in _KERNELS:
        if kernel.capability in model.capabilities:
            if len(model.parameters) != kernel.parameter_count:
                raise ValueError(
                    f"capability {kernel.capability!r} requires "
                    f"{kernel.parameter_count} parameters"
                )
            return kernel
    raise ValueError("this model does not declare a regular-image capability")


def _axis_terms(
    coordinates: np.ndarray,
    center: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = coordinates - center
    basis = np.exp(-(delta**2) / radius**2)
    return basis, basis * (2.0 * delta**2 / radius**3), basis * (2.0 * delta / radius**2)


class _ImageContext:
    """Per-input cache: one float64 promotion plus stripe geometry."""

    __slots__ = ("data", "check", "_float_observations")

    def __init__(
        self,
        data: RegularImageFitInput,
        check: Callable[[], None],
    ) -> None:
        self.data = data
        self.check = check
        self._float_observations: np.ndarray | None = None

    def float_observations(self) -> np.ndarray:
        """Promote the image to float64 exactly once so '@' hits BLAS.

        The cache is forced C-contiguous: projected payloads may hand the
        solver a transposed view, and strided stripes would keep every dot
        and matmul off the fast BLAS paths.
        """

        cached = self._float_observations
        if cached is None:
            cached = np.asarray(self.data.observations)
            if cached.dtype != np.float64 or not cached.flags.c_contiguous:
                cached = np.ascontiguousarray(cached, dtype=np.float64)
            self._float_observations = cached
            return cached
        return cached

    def stripe_bounds(self) -> tuple[tuple[int, int], ...]:
        height = self.data.observations.shape[0]
        return tuple(
            (start, min(height, start + _REGULAR_IMAGE_STRIPE_ROWS))
            for start in range(0, height, _REGULAR_IMAGE_STRIPE_ROWS)
        )

    def stripe_mask(self, start: int, stop: int) -> np.ndarray | None:
        data = self.data
        mask = None if data.valid_mask is None else data.valid_mask[start:stop]
        if data.observations.dtype.kind == "f":
            finite = np.isfinite(self.float_observations()[start:stop])
            mask = finite if mask is None else mask & finite
        if mask is not None and bool(np.all(mask)):
            mask = None
        return mask


def _regular_image_summary(context: _ImageContext) -> _RegularImageSummary:
    """One fused sweep: extrema, count, validity and normalization sums."""

    check = context.check
    data = context.data
    observed = context.float_observations()
    if data.valid_mask is None and data.observations.dtype.kind != "f":
        # All-valid integer images: extrema on the narrow storage dtype,
        # normalization sums as per-stripe BLAS dots on the float cache.
        check()
        minimum = float(np.min(data.observations))
        maximum = float(np.max(data.observations))
        count = int(data.observations.size)
        sums = []
        square_sums = []
        for start, stop in context.stripe_bounds():
            check()
            values = observed[start:stop].reshape(-1)
            sums.append(float(np.sum(values)))
            square_sums.append(float(np.dot(values, values)))
        scale = max(abs(minimum), abs(maximum)) or 1.0
        return _RegularImageSummary(
            minimum,
            maximum,
            scale,
            count,
            True,
            math.fsum(sums) / scale,
            math.fsum(square_sums) / scale**2,
        )
    minimum, maximum, count = math.inf, -math.inf, 0
    all_valid = True
    sums = []
    square_sums = []
    for start, stop in context.stripe_bounds():
        check()
        mask = context.stripe_mask(start, stop)
        all_valid &= mask is None
        values = (
            observed[start:stop].reshape(-1)
            if mask is None
            else observed[start:stop][mask]
        )
        if values.size:
            minimum = min(minimum, float(np.min(values)))
            maximum = max(maximum, float(np.max(values)))
            count += values.size
            sums.append(float(np.sum(values)))
            square_sums.append(float(np.dot(values, values)))
    if count == 0:
        raise ValueError("regular image has no finite valid observations")
    scale = max(abs(minimum), abs(maximum)) or 1.0
    return _RegularImageSummary(
        minimum,
        maximum,
        scale,
        count,
        all_valid,
        math.fsum(sums) / scale,
        math.fsum(square_sums) / scale**2,
    )


def _crop_to_valid_bounds(
    data: RegularImageFitInput,
    check: Callable[[], None],
) -> tuple[RegularImageFitInput, tuple[int, int, int] | None]:
    """Crop a masked image to the valid bounding box.

    A rectangular Area selection becomes an all-valid crop and rejoins the
    separable closed-form objective.  The returned origin ``(row, column,
    full_width)`` recovers the original flat pixel indices for deferred
    result materialization when the caller did not supply explicit indices.
    """

    mask = data.valid_mask
    if mask is None:
        return data, None
    check()
    rows = np.flatnonzero(np.any(mask, axis=1))
    columns = np.flatnonzero(np.any(mask, axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("regular image has no finite valid observations")
    y_start, y_stop = int(rows[0]), int(rows[-1]) + 1
    x_start, x_stop = int(columns[0]), int(columns[-1]) + 1
    height, width = mask.shape
    cropped_mask = mask[y_start:y_stop, x_start:x_stop]
    all_valid = bool(np.all(cropped_mask))
    if (y_stop - y_start, x_stop - x_start) == (height, width):
        if not all_valid:
            return data, None
        full = RegularImageFitInput(
            data.x_coordinates,
            data.y_coordinates,
            data.observations,
            valid_mask=None,
            selected_indices=data.selected_indices,
        )
        return full, None
    selected = (
        None
        if data.selected_indices is None
        else data.selected_indices[y_start:y_stop, x_start:x_stop]
    )
    cropped = RegularImageFitInput(
        data.x_coordinates[x_start:x_stop],
        data.y_coordinates[y_start:y_stop],
        data.observations[y_start:y_stop, x_start:x_stop],
        valid_mask=None if all_valid else cropped_mask,
        selected_indices=selected,
    )
    origin = None if selected is not None else (y_start, x_start, width)
    return cropped, origin


def _bounded_indices(valid: np.ndarray, limit: int) -> np.ndarray:
    indices = np.flatnonzero(valid)
    count = min(indices.size, limit)
    positions = np.linspace(0, indices.size - 1, count, dtype=np.int64)
    return indices[positions]


def _regular_image_subsample(
    data: RegularImageFitInput,
    check: Callable[[], None],
    limit: int,
) -> RegularImageFitInput:
    """Bounded-index subsample used for seeding and the multigrid ladder."""

    height, width = data.observations.shape
    if height <= limit and width <= limit:
        return data
    check()
    mask = data.valid_mask
    if data.observations.dtype.kind == "f":
        finite = np.isfinite(data.observations)
        mask = finite if mask is None else mask & finite
    if mask is None:
        y_index = _bounded_indices(np.ones(height, dtype=np.bool_), limit)
        x_index = _bounded_indices(np.ones(width, dtype=np.bool_), limit)
    else:
        y_index = _bounded_indices(np.any(mask, axis=1), limit)
        x_index = _bounded_indices(np.any(mask, axis=0), limit)
    if y_index.size == 0 or x_index.size == 0:
        raise ValueError("regular image has no finite valid observations")
    observations = np.asarray(
        data.observations[np.ix_(y_index, x_index)], dtype=np.float64
    )
    valid = np.isfinite(observations)
    if data.valid_mask is not None:
        valid &= data.valid_mask[np.ix_(y_index, x_index)]
    return RegularImageFitInput(
        data.x_coordinates[x_index],
        data.y_coordinates[y_index],
        observations,
        valid_mask=None if bool(np.all(valid)) else valid,
    )


def _regular_image_sample(
    data: RegularImageFitInput,
    check: Callable[[], None],
) -> tuple[ArrayTuple, np.ndarray]:
    check()
    selection = data.valid_mask
    if data.observations.dtype.kind == "f":
        finite = np.isfinite(data.observations)
        selection = finite if selection is None else selection & finite
    if selection is None:
        valid_y, valid_x = np.arange(data.y_coordinates.size), np.arange(data.x_coordinates.size)
    else:
        valid_y = np.flatnonzero(np.any(selection, axis=1))
        valid_x = np.flatnonzero(np.any(selection, axis=0))

    y_index, x_index = valid_y, valid_x
    sampled = np.asarray(data.observations[np.ix_(y_index, x_index)], dtype=np.float64)
    valid = np.isfinite(sampled)
    if selection is not None:
        valid &= selection[np.ix_(y_index, x_index)]
    if np.count_nonzero(valid) <= 5:
        raise ValueError("radial center fit needs a spatially coherent selection")
    offset = float(np.median(sampled[valid]))
    filtered = median_filter(np.where(valid, sampled, offset), size=3, mode="nearest")
    x_grid, y_grid = np.meshgrid(data.x_coordinates[x_index], data.y_coordinates[y_index])
    return (x_grid[valid], y_grid[valid]), filtered[valid]


def _axis_stats(
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Sums of each axis vector and dots of each against the basis."""

    basis = vectors[0]
    sums = tuple(float(np.sum(vector)) for vector in vectors)
    basis_dots = tuple(float(np.dot(basis, vector)) for vector in vectors)
    return sums, basis_dots


def _regular_image_linear_objective(
    kernel: _SeparableKernel,
    context: _ImageContext,
    summary: _RegularImageSummary,
    parameters: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Closed-form separable RSS and gradient for all-valid linear fits."""

    amplitude, offset = float(parameters[0]), float(parameters[1])
    data = context.data
    x_vectors, y_vectors = kernel.axis_vectors(
        parameters, data.x_coordinates, data.y_coordinates
    )
    projected = context.float_observations() @ np.column_stack(x_vectors)
    projected = projected / summary.scale

    x_sums, x_basis_dots = _axis_stats(x_vectors)
    y_sums, y_basis_dots = _axis_stats(y_vectors)
    phi_sum = x_sums[0] * y_sums[0]
    phi_square = x_basis_dots[0] * y_basis_dots[0]
    data_phi = float(np.dot(y_vectors[0], projected[:, 0]))
    data_derivatives = np.asarray(
        [
            math.fsum(
                float(np.dot(y_vectors[y_index], projected[:, x_index]))
                for y_index, x_index in terms
            )
            for terms in kernel.geometry_terms
        ]
    )
    derivative_sums = np.asarray(
        [
            math.fsum(
                y_sums[y_index] * x_sums[x_index] for y_index, x_index in terms
            )
            for terms in kernel.geometry_terms
        ]
    )
    phi_derivatives = np.asarray(
        [
            math.fsum(
                y_basis_dots[y_index] * x_basis_dots[x_index]
                for y_index, x_index in terms
            )
            for terms in kernel.geometry_terms
        ]
    )

    amplitude /= summary.scale
    offset /= summary.scale
    terms = (
        amplitude**2 * phi_square,
        2.0 * amplitude * offset * phi_sum,
        offset**2 * summary.count,
        -2.0 * amplitude * data_phi,
        -2.0 * offset * summary.normalized_sum,
        summary.normalized_square_sum,
    )
    rss = math.fsum(terms)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(math.fsum(map(abs, terms)), 1.0)
    if not math.isfinite(rss) or rss < -tolerance:
        raise FloatingPointError("regular-image objective is non-finite")
    rss = max(rss, 0.0)

    residual_phi = amplitude * phi_square + offset * phi_sum - data_phi
    residual_sum = amplitude * phi_sum + offset * summary.count - summary.normalized_sum
    geometry = amplitude * (
        amplitude * phi_derivatives + offset * derivative_sums - data_derivatives
    )
    gradient = np.r_[
        residual_phi / summary.scale,
        residual_sum / summary.scale,
        geometry,
    ]
    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError("regular-image gradient is non-finite")
    return 0.5 * rss, gradient


def _regular_image_linear_information(
    kernel: _SeparableKernel,
    context: _ImageContext,
    parameters: np.ndarray,
    scale: float,
) -> np.ndarray:
    amplitude = float(parameters[0])
    data = context.data
    x_vectors, y_vectors = kernel.axis_vectors(
        parameters, data.x_coordinates, data.y_coordinates
    )
    x_sums = tuple(float(np.sum(vector)) for vector in x_vectors)
    y_sums = tuple(float(np.sum(vector)) for vector in y_vectors)
    x_inner = [
        [float(np.dot(left, right)) for right in x_vectors] for left in x_vectors
    ]
    y_inner = [
        [float(np.dot(left, right)) for right in y_vectors] for left in y_vectors
    ]

    rows: list[tuple[tuple[float, int | None, int | None], ...]] = [
        ((1.0, 0, 0),),
        ((1.0, None, None),),
    ]
    rows.extend(
        tuple((amplitude, y_index, x_index) for y_index, x_index in terms)
        for terms in kernel.geometry_terms
    )

    def axis_inner(
        inner: list[list[float]],
        sums: tuple[float, ...],
        size: int,
        left: int | None,
        right: int | None,
    ) -> float:
        if left is None:
            return float(size) if right is None else sums[right]
        return sums[left] if right is None else inner[left][right]

    count = kernel.parameter_count
    information = np.empty((count, count), dtype=np.float64)
    for row in range(count):
        for column in range(row, count):
            value = math.fsum(
                row_scale
                * column_scale
                * axis_inner(
                    y_inner, y_sums, data.y_coordinates.size, row_y, column_y
                )
                * axis_inner(
                    x_inner, x_sums, data.x_coordinates.size, row_x, column_x
                )
                for row_scale, row_y, row_x in rows[row]
                for column_scale, column_y, column_x in rows[column]
            ) / scale**2
            information[row, column] = information[column, row] = value
    if not np.all(np.isfinite(information)):
        raise FloatingPointError("regular-image information is non-finite")
    return information


def _regular_image_loss_terms(
    residual: np.ndarray,
    loss: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    squared = residual**2
    if loss == "linear":
        first = np.ones_like(residual)
        return squared, first, first
    if loss == "soft_l1":
        root = np.sqrt(1.0 + squared)
        rho = 2.0 * squared / (root + 1.0)
        first, second = 1.0 / root, -0.5 / root**3
    elif loss == "huber":
        outside = squared > 1.0
        root = np.sqrt(squared)
        inverse = np.divide(1.0, root, out=np.ones_like(root), where=root != 0.0)
        rho = np.where(outside, 2.0 * root - 1.0, squared)
        first, second = (
            np.where(outside, inverse, 1.0),
            np.where(outside, -0.5 * inverse**3, 0.0),
        )
    elif loss == "cauchy":
        first = 1.0 / (1.0 + squared)
        rho, second = np.log1p(squared), -(first**2)
    elif loss == "arctan":
        denominator = 1.0 + squared**2
        rho, first = np.arctan(squared), 1.0 / denominator
        second = -2.0 * squared / denominator**2
    else:
        raise RuntimeError(loss)
    information_weight = np.maximum(
        first + 2.0 * second * squared, np.finfo(np.float64).eps
    )
    return rho, first, information_weight


def _regular_image_striped_objective(
    kernel: _SeparableKernel,
    context: _ImageContext,
    parameters: np.ndarray,
    scale: float,
    loss: str,
    collect_information: bool,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Masked/robust objective over row stripes, fanned across a small pool.

    Partial results are combined in stripe order after joining, so the
    accumulation order (and therefore the value) is deterministic regardless
    of worker scheduling.  The offset column is analytic (its derivative
    plane is constant one), so no ones-plane is materialized.
    """

    amplitude, offset = float(parameters[0]), float(parameters[1])
    data = context.data
    parameter_count = kernel.parameter_count
    x_vectors = kernel.x_vectors(parameters, data.x_coordinates)
    context.float_observations()  # materialize once before fan-out

    def stripe_task(
        bounds: tuple[int, int],
    ) -> tuple[
        float,
        float,
        np.ndarray,
        tuple[np.ndarray, np.ndarray, float] | None,
    ]:
        start, stop = bounds
        context.check()
        y_vectors = kernel.y_vectors(parameters, data.y_coordinates[start:stop])
        radial = y_vectors[0][:, None] * x_vectors[0][None, :]
        observed = context.float_observations()[start:stop]
        residual = (amplitude * radial + offset - observed) / scale
        planes = [radial]
        for terms in kernel.geometry_terms:
            plane = None
            for y_index, x_index in terms:
                term = y_vectors[y_index][:, None] * x_vectors[x_index][None, :]
                plane = term if plane is None else plane + term
            planes.append(amplitude * plane)
        mask = context.stripe_mask(start, stop)
        residual = residual.reshape(-1)
        if mask is not None:
            selected = mask.reshape(-1)
            residual = residual[selected]
            planes = [plane.reshape(-1)[selected] for plane in planes]
        else:
            planes = [plane.reshape(-1) for plane in planes]
        rho, first, information_weight = _regular_image_loss_terms(residual, loss)
        cost = 0.5 * float(np.sum(rho))
        square_sum = float(np.dot(residual, residual))
        weighted = first * residual
        gradient = np.empty(parameter_count, dtype=np.float64)
        gradient[0] = float(np.dot(planes[0], weighted)) / scale
        gradient[1] = float(np.sum(weighted)) / scale
        for index, plane in enumerate(planes[1:]):
            gradient[2 + index] = float(np.dot(plane, weighted)) / scale
        information = None
        if collect_information:
            jacobian = np.column_stack(planes) / scale
            weighted_jacobian = information_weight[:, None] * jacobian
            information = (
                jacobian.T @ weighted_jacobian,
                weighted_jacobian.sum(axis=0) / scale,
                float(np.sum(information_weight)) / scale**2,
            )
        return cost, square_sum, gradient, information

    bounds_list = context.stripe_bounds()
    if len(bounds_list) > 1:
        stripe_results = list(_stripe_pool().map(stripe_task, bounds_list))
    else:
        stripe_results = [stripe_task(bounds_list[0])]

    costs: list[float] = []
    square_sums: list[float] = []
    gradient = np.zeros(parameter_count, dtype=np.float64)
    dense_indices = [0, *range(2, parameter_count)]
    core = np.zeros((parameter_count - 1, parameter_count - 1), dtype=np.float64)
    offset_column = np.zeros(parameter_count - 1, dtype=np.float64)
    offset_diagonal = 0.0
    for cost, square_sum, stripe_gradient, stripe_information in stripe_results:
        costs.append(cost)
        square_sums.append(square_sum)
        gradient += stripe_gradient
        if stripe_information is not None:
            core += stripe_information[0]
            offset_column += stripe_information[1]
            offset_diagonal += stripe_information[2]
    cost, rss = math.fsum(costs), math.fsum(square_sums)
    information = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    if collect_information:
        information[np.ix_(dense_indices, dense_indices)] = core
        information[1, dense_indices] = offset_column
        information[dense_indices, 1] = offset_column
        information[1, 1] = offset_diagonal
    finite = (
        math.isfinite(cost)
        and math.isfinite(rss)
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(information))
    )
    if not finite:
        raise FloatingPointError("regular-image objective is non-finite")
    return cost, gradient, rss, information


def _regular_image_result_arrays(
    kernel: _SeparableKernel,
    data: RegularImageFitInput,
    parameters: np.ndarray,
    count: int,
    index_origin: tuple[int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize fitted/residual/index arrays for one deferred result."""

    amplitude, offset = float(parameters[0]), float(parameters[1])
    context = _ImageContext(data, lambda: None)
    x_basis = kernel.x_vectors(parameters, data.x_coordinates)[0]
    fitted = np.empty(count, dtype=np.float64)
    residuals = np.empty(count, dtype=np.float64)
    indices = np.empty(count, dtype=np.int64)
    cursor = 0
    width = data.x_coordinates.size
    for start, stop in context.stripe_bounds():
        y_basis = kernel.y_vectors(parameters, data.y_coordinates[start:stop])[0]
        predicted = amplitude * y_basis[:, None] * x_basis[None, :] + offset
        observed = context.float_observations()[start:stop]
        if data.selected_indices is not None:
            local_indices = data.selected_indices[start:stop]
        elif index_origin is None:
            local_indices = np.arange(
                start * width, stop * width, dtype=np.int64
            ).reshape(observed.shape)
        else:
            row_origin, column_origin, full_width = index_origin
            rows = (
                np.arange(start, stop, dtype=np.int64) + row_origin
            ) * full_width + column_origin
            local_indices = rows[:, None] + np.arange(width, dtype=np.int64)[None, :]
        mask = context.stripe_mask(start, stop)
        if mask is not None:
            predicted = predicted[mask]
            observed = observed[mask]
            local_indices = local_indices[mask]
        local_fitted = predicted.reshape(-1)
        local_observed = observed.reshape(-1)
        local_indices = local_indices.reshape(-1)
        size = local_fitted.size
        fitted[cursor : cursor + size] = local_fitted
        residuals[cursor : cursor + size] = local_observed - local_fitted
        indices[cursor : cursor + size] = local_indices
        cursor += size
    if cursor != count:
        raise RuntimeError("regular-image valid observation count changed during fit")
    return fitted, residuals, indices


def fit_regular_separable_image(
    model: FitModelSpec,
    data: RegularImageFitInput,
    *,
    data_revision: int,
    initial: Mapping[str, float] | Sequence[float] | None,
    warm_start: Mapping[str, float] | Sequence[float] | None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
    options: FitOptions,
    cancelled: Callable[[], bool] | None,
) -> FitResult:
    """Fit a separable Gaussian image model without expanding coordinates."""

    kernel = _kernel_for(model)
    started = time.monotonic()

    def check() -> None:
        if cancelled is not None and cancelled():
            raise FitCancelled("fit cancelled")
        deadline = options.deadline_seconds
        if deadline is not None and time.monotonic() - started > deadline:
            raise FitDeadlineExceeded("fit deadline exceeded")

    data, index_origin = _crop_to_valid_bounds(data, check)
    context = _ImageContext(data, check)
    summary = _regular_image_summary(context)
    if summary.count <= len(model.parameters):
        raise ValueError("fit requires more finite observations than parameters")

    proxy = _regular_image_subsample(data, check, _REGULAR_IMAGE_SAMPLE_LIMIT)
    if proxy is data:
        proxy_context, proxy_summary = context, summary
    else:
        proxy_context = _ImageContext(proxy, check)
        proxy_summary = _regular_image_summary(proxy_context)
    seed_coordinates, seed_values = _regular_image_sample(proxy, check)

    default_bounds = (
        model.bounds_initializer(seed_coordinates, seed_values)
        if model.bounds_initializer is not None
        else None
    )
    lower, upper = _solver_bounds(model, default_bounds, bounds)
    lower_inside, upper_inside = np.nextafter(lower, upper), np.nextafter(upper, lower)

    value_range = _value_range(seed_values)
    x_span, y_span = _span(data.x_coordinates), _span(data.y_coordinates)
    natural_scale = np.asarray(
        kernel.natural_scale_builder(value_range, x_span, y_span)
    )
    loss = options.loss
    full_scale = summary.scale if loss == "linear" else 1.0
    solver_options = {
        "ftol": _REGULAR_IMAGE_FTOL,
        "gtol": _REGULAR_IMAGE_GTOL,
        "maxfun": options.max_nfev,
        "maxiter": options.max_nfev,
        "maxls": _REGULAR_IMAGE_MAX_LINE_SEARCH_STEPS,
    }

    def objective_for(
        stage_context: _ImageContext,
        stage_summary: _RegularImageSummary,
    ) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
        stage_scale = stage_summary.scale if loss == "linear" else 1.0

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            check()
            if stage_summary.all_valid and loss == "linear":
                return _regular_image_linear_objective(
                    kernel, stage_context, stage_summary, parameters
                )
            cost, gradient, _rss, _information = _regular_image_striped_objective(
                kernel, stage_context, parameters, stage_scale, loss, False
            )
            return cost, gradient

        return objective

    full_objective = objective_for(context, summary)
    proxy_objective = (
        full_objective
        if proxy is data
        else objective_for(proxy_context, proxy_summary)
    )

    def solve_stage(
        objective: Callable[[np.ndarray], tuple[float, np.ndarray]],
        parameters: np.ndarray,
    ):
        parameter_scale = np.maximum(np.abs(parameters), natural_scale)

        def scaled_objective(scaled: np.ndarray) -> tuple[float, np.ndarray]:
            cost, gradient = objective(scaled * parameter_scale)
            return cost, gradient * parameter_scale

        solved = minimize(
            scaled_objective,
            parameters / parameter_scale,
            method="L-BFGS-B",
            jac=True,
            bounds=tuple(zip(lower / parameter_scale, upper / parameter_scale)),
            options=solver_options,
        )
        check()
        return solved, np.asarray(solved.x) * parameter_scale

    def full_information(parameters: np.ndarray) -> np.ndarray:
        if summary.all_valid and loss == "linear":
            return _regular_image_linear_information(
                kernel, context, parameters, full_scale
            )
        return _regular_image_striped_objective(
            kernel, context, parameters, full_scale, loss, True
        )[3]

    def gradient_converged(parameters: np.ndarray, gradient: np.ndarray) -> bool:
        """Exact first-order convergence test in the solver's scaled space."""

        parameter_scale = np.maximum(np.abs(parameters), natural_scale)
        return (
            float(np.max(np.abs(gradient * parameter_scale)))
            <= _REGULAR_IMAGE_GTOL
        )

    def newton_polish(
        parameters: np.ndarray,
        cost: float,
        gradient: np.ndarray,
    ) -> tuple[np.ndarray, float] | None:
        """Drive the full-resolution gradient under gtol with Gauss-Newton.

        The separable information matrix is nearly free, so a handful of
        Newton steps replaces an L-BFGS restart that would otherwise spend
        tens of full-image evaluations rebuilding curvature.  Returns the
        polished parameters with their full-resolution cost, or None when
        the quadratic model fails (the caller falls back to the bounded
        L-BFGS refinement).
        """

        current, current_cost, current_gradient = parameters, cost, gradient
        for _step in range(_REGULAR_IMAGE_MAX_NEWTON_STEPS):
            check()
            if gradient_converged(current, current_gradient):
                return current, current_cost
            try:
                step = np.linalg.solve(
                    full_information(current), -current_gradient
                )
            except np.linalg.LinAlgError:
                return None
            candidate = np.clip(current + step, lower_inside, upper_inside)
            try:
                candidate_cost, candidate_gradient = full_objective(candidate)
            except FloatingPointError:
                return None
            improved = math.isfinite(candidate_cost) and (
                candidate_cost < current_cost
                or float(np.max(np.abs(candidate_gradient)))
                < float(np.max(np.abs(current_gradient)))
            )
            if not improved:
                return None
            current, current_cost, current_gradient = (
                candidate,
                candidate_cost,
                candidate_gradient,
            )
        if gradient_converged(current, current_gradient):
            return current, current_cost
        return None

    def full_resolution_gate(
        parameters: np.ndarray,
        status: _SolverStatus,
        evaluated: tuple[float, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, _SolverStatus, float]:
        """Exact full-resolution convergence gate shared by every path.

        Returns the gated parameters, their solver status, and their
        full-resolution cost so competing candidate paths can be compared
        on one exact scale.
        """

        cost, gradient = (
            evaluated if evaluated is not None else full_objective(parameters)
        )
        if gradient_converged(parameters, gradient):
            return parameters, status, cost
        polished = newton_polish(parameters, cost, gradient)
        if polished is not None:
            polished_parameters, polished_cost = polished
            return (
                polished_parameters,
                _SolverStatus(True, "full-resolution Newton refinement converged"),
                polished_cost,
            )
        solved, refined = solve_stage(full_objective, parameters)
        refined_cost, _gradient = full_objective(refined)
        if not math.isfinite(refined_cost) or not np.all(np.isfinite(refined)):
            raise FloatingPointError(
                "full-resolution regular-image refinement is non-finite"
            )
        return (
            refined,
            _SolverStatus(bool(solved.success), str(solved.message)),
            refined_cost,
        )

    def refine_to_full(
        parameters: np.ndarray,
        status: _SolverStatus,
    ) -> tuple[np.ndarray, _SolverStatus, float]:
        """Climb the subsample ladder, then apply the full-resolution gate."""

        if proxy is not data:
            limit = _REGULAR_IMAGE_SAMPLE_LIMIT * _REGULAR_IMAGE_LADDER_FACTOR
            largest = max(data.observations.shape)
            while limit < largest:
                stage = _regular_image_subsample(data, check, limit)
                if stage is data:
                    break
                stage_context = _ImageContext(stage, check)
                stage_summary = _regular_image_summary(stage_context)
                solved, stage_parameters = solve_stage(
                    objective_for(stage_context, stage_summary), parameters
                )
                if np.all(np.isfinite(stage_parameters)):
                    parameters = stage_parameters
                    status = _SolverStatus(
                        bool(solved.success), str(solved.message)
                    )
                limit *= _REGULAR_IMAGE_LADDER_FACTOR
        return full_resolution_gate(parameters, status)

    def build_result(
        parameters: np.ndarray,
        status: _SolverStatus,
    ) -> FitResult:
        if summary.all_valid and loss == "linear":
            information = _regular_image_linear_information(
                kernel, context, parameters, full_scale
            )
            final_cost, _gradient = _regular_image_linear_objective(
                kernel, context, summary, parameters
            )
            normalized_rss = 2.0 * final_cost
        else:
            _cost, _gradient, normalized_rss, information = (
                _regular_image_striped_objective(
                    kernel, context, parameters, full_scale, loss, True
                )
            )
        degrees = max(summary.count - parameters.size, 1)
        normalized_reduced = normalized_rss / degrees
        covariance, covariance_valid = _covariance_from_information(
            information, normalized_reduced, summary.count
        )
        errors = (
            np.sqrt(np.maximum(np.diag(covariance), 0.0))
            if covariance_valid
            else np.full(parameters.size, np.nan)
        )
        result_data, result_count = data, summary.count
        result_parameters = np.asarray(parameters, dtype=np.float64).copy()
        deferred = _DeferredFitData(
            lambda: _regular_image_result_arrays(
                kernel,
                result_data,
                result_parameters,
                result_count,
                index_origin,
            )
        )
        return FitResult(
            model,
            parameters,
            errors,
            covariance,
            deferred,
            deferred,
            deferred,
            data_revision,
            status.success,
            status.message,
            float(normalized_reduced * full_scale**2),
            covariance_valid=covariance_valid,
        )

    def solve_from_cold_seeds() -> tuple[np.ndarray, _SolverStatus, float]:
        """Moment-seeded proxy search refined through the subsample ladder."""

        seeds = _initial_candidates(
            model, seed_coordinates, seed_values, initial, None
        )
        if initial is None:
            strongest = max(abs(float(seed[0])) for seed in seeds)
            seeds = tuple(
                seed
                for seed in seeds
                if abs(float(seed[0]))
                >= strongest * _REGULAR_IMAGE_MIN_RELATIVE_CONTRAST
            )
        seeds = tuple(np.clip(seed, lower_inside, upper_inside) for seed in seeds)
        candidates: list[tuple[float, object, np.ndarray]] = []
        last_error: Exception | None = None
        for seed in seeds:
            check()
            try:
                solved, parameters = solve_stage(proxy_objective, seed)
                cost, _gradient = proxy_objective(parameters)
                if math.isfinite(cost) and np.all(np.isfinite(parameters)):
                    candidates.append((cost, solved, parameters))
            except (FitCancelled, FitDeadlineExceeded):
                raise
            except (ValueError, RuntimeError, FloatingPointError) as error:
                last_error = error
        if not candidates:
            if last_error is not None:
                raise last_error
            raise RuntimeError(
                "regular-image optimizer failed for every initializer"
            )
        _cost, solved, parameters = min(
            candidates, key=lambda item: (not bool(item[1].success), item[0])
        )
        return refine_to_full(
            parameters, _SolverStatus(bool(solved.success), str(solved.message))
        )

    warm_candidate: tuple[np.ndarray, _SolverStatus, float] | None = None
    if warm_start is not None:
        try:
            warm = np.clip(
                _initial_values(model, seed_coordinates, seed_values, warm_start),
                lower_inside,
                upper_inside,
            )
            warm_cost, warm_gradient = full_objective(warm)
            if not math.isfinite(warm_cost) or not np.all(
                np.isfinite(warm_gradient)
            ):
                raise FloatingPointError("warm start objective is non-finite")
            if gradient_converged(warm, warm_gradient):
                # Steady scene: the remembered solution is already an exact
                # full-resolution stationary point, so accept it without
                # paying for a cold seed search.
                return build_result(
                    warm,
                    _SolverStatus(
                        True,
                        "warm start satisfies full-resolution convergence",
                    ),
                )
            # The scene changed under the warm seed.  A bounded Gauss-Newton
            # descent from it tracks small displacements at full-resolution
            # quality, but only as one COMPETING candidate: local descent
            # from a stale basin can settle on a degenerate stationary point
            # far from the blob (center pinned at a frame edge, radius the
            # size of the frame) with solver success, so the moment-seeded
            # cold search below always runs and the better full-resolution
            # cost wins.  A full-image L-BFGS descent from the stale point is
            # deliberately not attempted: it is the path that produced those
            # degenerate accepts and it is redundant next to the cold search.
            polished = newton_polish(warm, warm_cost, warm_gradient)
            if polished is not None:
                polished_parameters, polished_cost = polished
                warm_candidate = (
                    polished_parameters,
                    _SolverStatus(
                        True, "full-resolution Newton refinement converged"
                    ),
                    polished_cost,
                )
        except (FitCancelled, FitDeadlineExceeded):
            raise
        except (ValueError, RuntimeError, FloatingPointError):
            warm_candidate = None  # the cold search stands alone

    cold_candidate: tuple[np.ndarray, _SolverStatus, float] | None
    try:
        cold_candidate = solve_from_cold_seeds()
    except (FitCancelled, FitDeadlineExceeded):
        raise
    except (ValueError, RuntimeError, FloatingPointError):
        if warm_candidate is None:
            raise
        cold_candidate = None

    finalists = [
        candidate
        for candidate in (warm_candidate, cold_candidate)
        if candidate is not None
    ]
    parameters, status, _cost = min(
        finalists, key=lambda item: (not item[1].success, item[2])
    )
    return build_result(parameters, status)
