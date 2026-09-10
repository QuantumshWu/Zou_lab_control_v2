"""Optimized separable regular-grid image fitting.

The regular-image solver is isolated from the coordinate-array engine.  The
public fit catalogue supplies the model and unit semantics; this module owns
the stripe/BLAS numerical implementation shared by every separable Gaussian
image model (the built-in radial and anisotropic centers).

The solver seeds every cell on a bounded proxy and refines the full image
through the same independent TRF.  B1 and multiple cells use one compact-axis,
BLAS-backed objective and the same final-information/covariance pipeline.
Result arrays are deferred: the returned :class:`FitResult` retains only the
fit input and parameters and materializes
``fitted_values``/``residuals``/``selected_indices`` on first access.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Mapping, Sequence

import numpy as np
from numba import njit, prange
from scipy.ndimage import median_filter

from . import _fit_compiled as _compiled_fit
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
    _fixed_parameter_partition,
    _initial_values,
    _solver_bounds,
)


__all__ = ["fit_regular_separable_image"]

_COMPILED_LINEAR_LOSS = int(_compiled_fit.LOSS_CODES["linear"])


_REGULAR_IMAGE_STRIPE_ROWS = 64
_REGULAR_IMAGE_SAMPLE_LIMIT = 129
_REGULAR_IMAGE_FTOL = 1e-10
_REGULAR_IMAGE_GTOL = 1e-8
# Linear proxies select a basin; robust losses and full refinement stay strict.
_REGULAR_IMAGE_PROXY_TOL = 1e-5
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
)

_ANISOTROPIC_KERNEL = _SeparableKernel(
    capability="regular_image_separable",
    parameter_count=6,
    x_radius_index=2,
    y_radius_index=3,
    geometry_terms=(((0, 1),), ((1, 0),), ((0, 2),), ((2, 0),)),
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


@njit(cache=True)
def _prepare_regular_image_refinement(
    _coordinates,
    _observations,
    _valid,
    _seeds,
    _lower,
    _upper,
    _context,
):
    """Python preparation supplies the exact regular-image seeds and bounds."""

    return 0


@njit(cache=True, inline="always")
def _regular_resolution_floor(coordinates, radius_index, lower):
    points = coordinates.shape[1]
    width = 1
    first_y = coordinates[1, 0]
    while width < points and coordinates[1, width] == first_y:
        width += 1
    resolution = math.inf
    for column in range(1, width):
        resolution = min(
            resolution,
            abs(coordinates[0, column] - coordinates[0, column - 1]),
        )
    height = points // width
    for row_index in range(1, height):
        resolution = min(
            resolution,
            abs(
                coordinates[1, row_index * width]
                - coordinates[1, (row_index - 1) * width]
            ),
        )
    if not math.isfinite(resolution) or resolution <= 0.0:
        resolution = np.finfo(np.float64).eps
    lower[radius_index] = max(0.5 * resolution, np.finfo(np.float64).eps)


@njit(cache=True)
def _expanded_regular_coordinates(packed):
    """Expand the bounded proxy only, for the existing point-wise initializers."""

    width, height = int(packed[0, 0]), int(packed[1, 0])
    coordinates = np.empty((2, width * height), dtype=np.float64)
    for row in range(height):
        for column in range(width):
            coordinates[0, row * width + column] = packed[0, column + 1]
            coordinates[1, row * width + column] = packed[1, row + 1]
    return coordinates


@njit(cache=True)
def _prepare_regular_radial_compiled(
    coordinates,
    observations,
    valid,
    seeds,
    lower,
    upper,
    context,
):
    coordinates = _expanded_regular_coordinates(coordinates)
    count = _compiled_fit._prepare_radial(
        coordinates, observations, valid, seeds, lower, upper, context
    )
    _regular_resolution_floor(coordinates, 2, lower)
    return count


@njit(cache=True)
def _prepare_regular_anisotropic_compiled(
    coordinates,
    observations,
    valid,
    seeds,
    lower,
    upper,
    context,
):
    coordinates = _expanded_regular_coordinates(coordinates)
    count = _compiled_fit._prepare_anisotropic(
        coordinates, observations, valid, seeds, lower, upper, context
    )
    _regular_resolution_floor(coordinates, 2, lower)
    _regular_resolution_floor(coordinates, 3, lower)
    return count


@njit(cache=True, inline="always")
def _compiled_axis_terms(
    coordinate: float,
    center: float,
    radius: float,
) -> tuple[float, float, float]:
    delta = coordinate - center
    basis = math.exp(-(delta * delta) / (radius * radius))
    return (
        basis,
        basis * (2.0 * delta * delta / (radius * radius * radius)),
        basis * (2.0 * delta / (radius * radius)),
    )


@njit(cache=True, nogil=True, parallel=True)
def _compiled_regular_centered_context(source, mask, width):
    """Pack physical values and centered moments in one shared stripe pass."""

    cells, points = source.shape
    values = np.empty((cells, points), dtype=np.float64)
    contexts = np.empty((cells, 1, points + 4), dtype=np.float64)
    stripe_size = _REGULAR_IMAGE_STRIPE_ROWS * width
    stripes = (points + stripe_size - 1) // stripe_size
    moments = np.empty((cells, stripes, 3), dtype=np.float64)
    for cell in range(cells):
        reference = 0.0
        for point in range(points):
            value = float(source[cell, point])
            if (mask.size == 0 or mask[cell, point]) and math.isfinite(value):
                reference = value
                break
        contexts[cell, 0, 0] = reference
    for lane in prange(cells * stripes):
        cell, stripe = lane // stripes, lane % stripes
        reference = contexts[cell, 0, 0]
        total, squares, count = 0.0, 0.0, 0
        for point in range(stripe * stripe_size, min((stripe + 1) * stripe_size, points)):
            value = float(source[cell, point])
            values[cell, point] = value
            centered = value - reference
            contexts[cell, 0, point + 4] = centered
            if (mask.size == 0 or mask[cell, point]) and math.isfinite(value):
                total += centered
                squares += centered * centered
                count += 1
        moments[cell, stripe, 0] = total
        moments[cell, stripe, 1] = squares
        moments[cell, stripe, 2] = count
    for cell in range(cells):
        for moment in range(3):
            total = 0.0
            for stripe in range(stripes):
                total += moments[cell, stripe, moment]
            contexts[cell, 0, moment + 1] = total
    return values, contexts


@njit(cache=True)
def _compiled_regular_residual_rss(observations, amplitude, offset, x_basis, y_basis):
    width, height = x_basis.size, y_basis.size
    residuals = np.empty(min(height, _REGULAR_IMAGE_STRIPE_ROWS) * width, dtype=np.float64)
    rss = 0.0
    for start in range(0, height, _REGULAR_IMAGE_STRIPE_ROWS):
        stop = min(height, start + _REGULAR_IMAGE_STRIPE_ROWS)
        for row in range(start, stop):
            for column in range(width):
                residuals[(row - start) * width + column] = (
                    amplitude * y_basis[row] * x_basis[column]
                    + offset - observations[row * width + column]
                )
        stripe = residuals[: (stop - start) * width]
        rss += np.dot(stripe, stripe)
    return rss


@njit(cache=True)
def _compiled_regular_linear_objective(
    observations,
    parameters,
    free_indices,
    gradient,
    information,
    x_vectors,
    y_vectors,
    derivatives,
    radial,
    context,
    information_only=False,
):
    """Closed-form linear residual derivatives for one complete image."""

    height = y_vectors.shape[1]
    width = x_vectors.shape[1]
    amplitude = parameters[0]
    physical_offset = parameters[1]
    offset = physical_offset - (context[0, 0] if context.size else 0.0)
    x_full = np.ones((4, width), dtype=np.float64)
    y_full = np.ones((4, height), dtype=np.float64)
    for vector in range(3):
        x_full[vector] = x_vectors[vector]
        y_full[vector] = y_vectors[vector]
    projected = np.zeros((3, height), dtype=np.float64)
    raw_rss = 0.0
    if not information_only:
        source = context[0, 4:] if context.size else observations
        if derivatives or context.size:
            # One row of the image projected onto each x basis vector, as
            # three matrix-vector products whose results are the rows of
            # ``projected`` -- contiguous, which is what the dot products
            # below and the B1/Bn accumulation read.  Projecting through a
            # (width, 3) basis and transposing the (height, 3) result gave
            # the same numbers as strided views, and np.dot on a strided
            # operand falls off BLAS.  The constant-offset sum already has a
            # scalar owner.
            image = source.reshape(height, width)
            for vector in range(3):
                projected[vector] = image @ x_vectors[vector]
        if context.size:
            x_sum = np.sum(x_vectors[0])
            y_sum = np.sum(y_vectors[0])
            terms = (
                amplitude * amplitude * np.dot(x_vectors[0], x_vectors[0]) * np.dot(y_vectors[0], y_vectors[0]),
                2.0 * amplitude * offset * x_sum * y_sum,
                offset * offset * observations.size,
                -2.0 * amplitude * np.dot(y_vectors[0], projected[0]),
                -2.0 * offset * context[0, 1],
                context[0, 2],
            )
            correction, magnitude = 0.0, 0.0
            for term in terms:
                combined = raw_rss + term
                correction += ((raw_rss - combined) + term if abs(raw_rss) >= abs(term) else (term - combined) + raw_rss)
                raw_rss = combined
                magnitude += abs(term)
            raw_rss += correction
            # Conservative accumulation-roundoff bound, not a fit-quality gate.
            # Near cancellation uses the same direct residual calculation for
            # every cell and every batch size; final RSS is always direct.
            operations = observations.size + width + height + 16
            roundoff = operations * np.finfo(np.float64).eps
            roundoff /= 1.0 - roundoff
            if raw_rss <= roundoff * magnitude:
                raw_rss = _compiled_regular_residual_rss(
                    observations, amplitude, physical_offset, x_vectors[0], y_vectors[0]
                )
        else:
            raw_rss = _compiled_regular_residual_rss(
                observations, amplitude, physical_offset, x_vectors[0], y_vectors[0]
            )
    if not derivatives:
        return 0.5 * raw_rss, raw_rss, math.isfinite(raw_rss)
    observed_sum = context[0, 1] if context.size else np.sum(observations)
    x_sums = np.empty(4, dtype=np.float64)
    y_sums = np.empty(4, dtype=np.float64)
    x_inner = np.empty((4, 4), dtype=np.float64)
    y_inner = np.empty((4, 4), dtype=np.float64)
    data_inner = np.empty((4, 4), dtype=np.float64)
    for left in range(4):
        x_sum = 0.0
        for column in range(width):
            x_sum += x_full[left, column]
        x_sums[left] = x_sum
        y_sum = 0.0
        for row_index in range(height):
            y_sum += y_full[left, row_index]
        y_sums[left] = y_sum
        for right in range(4):
            x_dot = 0.0
            for column in range(width):
                x_dot += x_full[left, column] * x_full[right, column]
            x_inner[left, right] = x_dot
            y_dot = 0.0
            data_dot = 0.0
            for row_index in range(height):
                y_dot += y_full[left, row_index] * y_full[right, row_index]
                if right < 3:
                    data_dot += y_full[left, row_index] * projected[right, row_index]
            y_inner[left, right] = y_dot
            # Only the offset derivative uses the constant x-column, and its
            # y-column is also constant. Other entries here are never read.
            data_inner[left, right] = (observed_sum if left == 3 else 0.0) if right == 3 else data_dot

    parameter_count = parameters.size
    term_count = np.ones(parameter_count, dtype=np.int64)
    term_y = np.zeros((parameter_count, 2), dtype=np.int64)
    term_x = np.zeros((parameter_count, 2), dtype=np.int64)
    term_scale = np.ones((parameter_count, 2), dtype=np.float64)
    term_y[1, 0] = 3
    term_x[1, 0] = 3
    if radial:
        term_count[2] = 2
        term_y[2, 0] = 0
        term_x[2, 0] = 1
        term_y[2, 1] = 1
        term_x[2, 1] = 0
        term_scale[2, 0] = amplitude
        term_scale[2, 1] = amplitude
        term_y[3, 0] = 0
        term_x[3, 0] = 2
        term_scale[3, 0] = amplitude
        term_y[4, 0] = 2
        term_x[4, 0] = 0
        term_scale[4, 0] = amplitude
    else:
        term_y[2, 0] = 0
        term_x[2, 0] = 1
        term_scale[2, 0] = amplitude
        term_y[3, 0] = 1
        term_x[3, 0] = 0
        term_scale[3, 0] = amplitude
        term_y[4, 0] = 0
        term_x[4, 0] = 2
        term_scale[4, 0] = amplitude
        term_y[5, 0] = 2
        term_x[5, 0] = 0
        term_scale[5, 0] = amplitude

    full_gradient = np.empty(parameter_count, dtype=np.float64)
    full_information = np.empty((parameter_count, parameter_count), dtype=np.float64)
    for parameter in range(parameter_count):
        value = 0.0
        for term in range(term_count[parameter]):
            y_index = term_y[parameter, term]
            x_index = term_x[parameter, term]
            scale = term_scale[parameter, term]
            model_dot = (
                amplitude * y_inner[0, y_index] * x_inner[0, x_index]
                + offset * y_sums[y_index] * x_sums[x_index]
            )
            value += scale * (model_dot - data_inner[y_index, x_index])
        full_gradient[parameter] = value
        for other in range(parameter + 1):
            value = 0.0
            for left in range(term_count[parameter]):
                for right in range(term_count[other]):
                    value += (
                        term_scale[parameter, left]
                        * term_scale[other, right]
                        * y_inner[
                            term_y[parameter, left], term_y[other, right]
                        ]
                        * x_inner[
                            term_x[parameter, left], term_x[other, right]
                        ]
                    )
            full_information[parameter, other] = value
            full_information[other, parameter] = value
    for row in range(free_indices.size):
        gradient[row] = full_gradient[free_indices[row]]
        for column in range(free_indices.size):
            information[row, column] = full_information[
                free_indices[row], free_indices[column]
            ]
    finite = (
        math.isfinite(raw_rss)
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(information))
    )
    return 0.5 * raw_rss, raw_rss, finite


@njit(cache=True, nogil=True, parallel=True)
def _compiled_regular_information_batch(x_coordinates, y_coordinates, parameters, radial, observations, complete):
    """Shared axis-Gram information and one final direct physical RSS pass."""

    cells, count = parameters.shape
    matrices = np.full((cells, count, count), np.nan, dtype=np.float64)
    raw_rss = np.full(cells, np.nan, dtype=np.float64)
    usable = np.zeros(cells, dtype=np.bool_)
    x_bases = np.empty((cells, x_coordinates.size), dtype=np.float64)
    y_bases = np.empty((cells, y_coordinates.size), dtype=np.float64)
    free = np.arange(count, dtype=np.int64)
    empty = np.empty(0, dtype=np.float64)
    empty_context = np.empty((0, 0), dtype=np.float64)
    for cell in range(cells):
        if not complete[cell]:
            continue
        values = parameters[cell]
        radius_x = values[2]
        radius_y = values[2] if radial else values[3]
        if not np.all(np.isfinite(values)) or radius_x * radius_x == 0.0 or radius_y * radius_y == 0.0:
            continue
        x_vectors = np.empty((3, x_coordinates.size), dtype=np.float64)
        y_vectors = np.empty((3, y_coordinates.size), dtype=np.float64)
        for column in range(x_coordinates.size):
            terms = _compiled_axis_terms(
                x_coordinates[column], values[-2], radius_x
            )
            for vector in range(3):
                x_vectors[vector, column] = terms[vector]
        for row in range(y_coordinates.size):
            terms = _compiled_axis_terms(
                y_coordinates[row], values[-1], radius_y
            )
            for vector in range(3):
                y_vectors[vector, row] = terms[vector]
        _compiled_regular_linear_objective(
            empty, values, free, np.empty(count), matrices[cell],
            x_vectors, y_vectors, True, radial, empty_context, True,
        )
        x_bases[cell] = x_vectors[0]
        y_bases[cell] = y_vectors[0]
        usable[cell] = True
    stripes = (y_coordinates.size + _REGULAR_IMAGE_STRIPE_ROWS - 1) // _REGULAR_IMAGE_STRIPE_ROWS
    partials = np.empty((cells, stripes), dtype=np.float64)
    for lane in prange(cells * stripes):
        cell, stripe = lane // stripes, lane % stripes
        if not usable[cell]:
            continue
        start = stripe * _REGULAR_IMAGE_STRIPE_ROWS
        stop = min(start + _REGULAR_IMAGE_STRIPE_ROWS, y_coordinates.size)
        partials[cell, stripe] = _compiled_regular_residual_rss(
            observations[cell, start * x_coordinates.size : stop * x_coordinates.size],
            parameters[cell, 0], parameters[cell, 1], x_bases[cell], y_bases[cell, start:stop],
        )
    for cell in range(cells):
        if usable[cell]:
            total = 0.0
            for stripe in range(stripes):
                total += partials[cell, stripe]
            raw_rss[cell] = total
    return matrices, raw_rss


@njit(cache=True)
def _compiled_regular_image_objective(
    coordinates,
    observations,
    valid,
    parameters,
    free_indices,
    weights,
    use_weights,
    poisson,
    loss_code,
    gradient,
    information,
    jacobian_row,
    derivatives,
    radial,
    context,
):
    """Exact regular-grid Gaussian objective with one axis exponential pass."""

    point_count = observations.size
    if point_count == 0 or coordinates.shape[0] != 2:
        return math.inf, math.inf, False
    width, height = int(coordinates[0, 0]), int(coordinates[1, 0])
    if width <= 0 or height <= 0 or width * height != point_count:
        return math.inf, math.inf, False
    radius_x = parameters[2]
    radius_y = parameters[2] if radial else parameters[3]
    center_x = parameters[-2]
    center_y = parameters[-1]
    if radius_x <= 0.0 or radius_y <= 0.0:
        return math.inf, math.inf, False

    x_vectors = np.empty((3, width), dtype=np.float64)
    y_vectors = np.empty((3, height), dtype=np.float64)
    for column in range(width):
        values = _compiled_axis_terms(
            coordinates[0, column + 1], center_x, radius_x
        )
        x_vectors[0, column] = values[0]
        x_vectors[1, column] = values[1]
        x_vectors[2, column] = values[2]
    for row_index in range(height):
        values = _compiled_axis_terms(
            coordinates[1, row_index + 1], center_y, radius_y
        )
        y_vectors[0, row_index] = values[0]
        y_vectors[1, row_index] = values[1]
        y_vectors[2, row_index] = values[2]

    if context.size:
        # Full preparation already counted finite, unmasked values; the public
        # regular-input boundary validates both coordinate axes.
        all_valid = context[0, 3] == point_count
    else:
        all_valid = True
        for point in range(point_count):
            if not valid[point]:
                all_valid = False
                break
    if (
        all_valid
        and not use_weights
        and not poisson
        and loss_code == _COMPILED_LINEAR_LOSS
    ):
        return _compiled_regular_linear_objective(
            observations,
            parameters,
            free_indices,
            gradient,
            information,
            x_vectors,
            y_vectors,
            derivatives,
            radial,
            context,
        )

    if derivatives:
        _compiled_fit.compiled_reset_accumulators(gradient, information)
    cost = 0.0
    raw_rss = 0.0
    full_row = np.empty(parameters.size, dtype=np.float64)
    amplitude = parameters[0]
    offset = parameters[1] - (context[0, 0] if context.size else 0.0)
    source = context[0, 4:] if context.size else observations
    for row_index in range(height):
        y_basis = y_vectors[0, row_index]
        y_radius = y_vectors[1, row_index]
        y_center = y_vectors[2, row_index]
        for column in range(width):
            point = row_index * width + column
            if not valid[point]:
                continue
            x_basis = x_vectors[0, column]
            phi = y_basis * x_basis
            predicted = amplitude * phi + offset
            (
                _raw,
                squared,
                local_cost,
                gradient_factor,
                information_factor,
                finite,
            ) = _compiled_fit.compiled_point_terms(
                predicted,
                source[point],
                poisson,
                weights[point] if use_weights else 1.0,
                use_weights,
                loss_code,
            )
            if not finite or not math.isfinite(predicted):
                return math.inf, math.inf, False
            cost += local_cost
            raw_rss += squared
            if not derivatives:
                continue
            full_row[0] = phi
            full_row[1] = 1.0
            if radial:
                full_row[2] = amplitude * (
                    y_basis * x_vectors[1, column] + y_radius * x_basis
                )
                full_row[3] = amplitude * y_basis * x_vectors[2, column]
                full_row[4] = amplitude * y_center * x_basis
            else:
                full_row[2] = amplitude * y_basis * x_vectors[1, column]
                full_row[3] = amplitude * y_radius * x_basis
                full_row[4] = amplitude * y_basis * x_vectors[2, column]
                full_row[5] = amplitude * y_center * x_basis
            for free_index in range(free_indices.size):
                value = full_row[free_indices[free_index]]
                if not math.isfinite(value):
                    return math.inf, math.inf, False
                jacobian_row[free_index] = value
            _compiled_fit.compiled_accumulate(
                jacobian_row,
                gradient,
                information,
                gradient_factor,
                information_factor,
            )
    if derivatives:
        _compiled_fit.compiled_finish_information(information)
    return cost, raw_rss, math.isfinite(cost) and math.isfinite(raw_rss)


@njit(cache=True)
def _compiled_regular_radial_objective(
    coordinates,
    observations,
    valid,
    parameters,
    free_indices,
    weights,
    use_weights,
    poisson,
    loss_code,
    gradient,
    information,
    jacobian_row,
    derivatives,
    context,
):
    return _compiled_regular_image_objective(
        coordinates,
        observations,
        valid,
        parameters,
        free_indices,
        weights,
        use_weights,
        poisson,
        loss_code,
        gradient,
        information,
        jacobian_row,
        derivatives,
        True,
        context,
    )


@njit(cache=True)
def _compiled_regular_anisotropic_objective(
    coordinates,
    observations,
    valid,
    parameters,
    free_indices,
    weights,
    use_weights,
    poisson,
    loss_code,
    gradient,
    information,
    jacobian_row,
    derivatives,
    context,
):
    return _compiled_regular_image_objective(
        coordinates,
        observations,
        valid,
        parameters,
        free_indices,
        weights,
        use_weights,
        poisson,
        loss_code,
        gradient,
        information,
        jacobian_row,
        derivatives,
        False,
        context,
    )


def _compiled_regular_descriptor(
    kernel: _SeparableKernel,
    *,
    refinement: bool = False,
) -> _compiled_fit.CompiledFitDescriptor:
    if kernel is _RADIAL_KERNEL:
        base = _compiled_fit.radial_gaussian_center_descriptor()
        objective = _compiled_regular_radial_objective
        prepare = _prepare_regular_radial_compiled
    else:
        base = _compiled_fit.anisotropic_gaussian_center_descriptor()
        objective = _compiled_regular_anisotropic_objective
        prepare = _prepare_regular_anisotropic_compiled
    return _compiled_fit.CompiledFitDescriptor(
        prepare=(
            _prepare_regular_image_refinement if refinement else prepare
        ),
        objective=objective,
        value_jacobian=base.value_jacobian,
        context_builder=base.context_builder,
        max_candidates=base.max_candidates,
        cache_key=f"{base.cache_key}-regular-grid-v1",
        coordinate_layout="rectangular-grid",
    )


def _promoted_c_contiguous(values: np.ndarray) -> np.ndarray:
    """Float64 C-contiguous copy; transposed planes copy by column blocks.

    ``ascontiguousarray`` walks a transposed view in cache-hostile order --
    measured ~35 ms for a 2048 squared float64 plane.  Copying column blocks
    rides the source's own fast axis instead, which is the same values in a
    cache-aware visiting order: bit-identical, several times faster.
    """

    if (
        values.ndim == 2
        and values.dtype == np.float64
        and values.T.flags.c_contiguous
    ):
        out = np.empty(values.shape, dtype=np.float64)
        step = 128
        for start in range(0, values.shape[1], step):
            stop = min(values.shape[1], start + step)
            out[:, start:stop] = values[:, start:stop]
        return out
    return np.ascontiguousarray(values, dtype=np.float64)


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

    __slots__ = (
        "data",
        "check",
        "_float_observations",
        "_all_finite",
    )

    def __init__(
        self,
        data: RegularImageFitInput,
        check: Callable[[], None],
    ) -> None:
        self.data = data
        self.check = check
        self._float_observations: np.ndarray | None = None
        self._all_finite: bool | None = None

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
                cached = _promoted_c_contiguous(cached)
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
        if data.observations.dtype.kind == "f" and not self.finite_everywhere():
            finite = np.isfinite(self.float_observations()[start:stop])
            mask = finite if mask is None else mask & finite
        if mask is not None and bool(np.all(mask)):
            mask = None
        return mask

    def finite_everywhere(self) -> bool:
        """One whole-plane finiteness check instead of one per stripe pass.

        The answer is a property of the plane, not of a stripe; asking it
        stripe by stripe re-derived the same fact dozens of times per fit.
        """

        cached = self._all_finite
        if cached is None:
            cached = bool(np.isfinite(self.float_observations()).all())
            self._all_finite = cached
        return cached




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
    observed: np.ndarray | None = None,
) -> RegularImageFitInput:
    """Bounded-index subsample used for seeding and the multigrid ladder.

    ``observed`` is the caller's already-promoted float64 plane of the same
    values; gathering from it skips re-reading the possibly strided source.
    """

    height, width = data.observations.shape
    if height <= limit and width <= limit:
        return data
    check()
    source = data.observations if observed is None else observed
    mask = data.valid_mask
    if data.observations.dtype.kind == "f":
        finite = np.isfinite(source)
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
        source[np.ix_(y_index, x_index)], dtype=np.float64
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
    observed: np.ndarray | None = None,
) -> tuple[ArrayTuple, np.ndarray]:
    check()
    source = data.observations if observed is None else observed
    selection = data.valid_mask
    if data.observations.dtype.kind == "f":
        finite = np.isfinite(source)
        selection = finite if selection is None else selection & finite
    if selection is None:
        valid_y, valid_x = np.arange(data.y_coordinates.size), np.arange(data.x_coordinates.size)
    else:
        valid_y = np.flatnonzero(np.any(selection, axis=1))
        valid_x = np.flatnonzero(np.any(selection, axis=0))

    y_index, x_index = valid_y, valid_x
    sampled = np.asarray(source[np.ix_(y_index, x_index)], dtype=np.float64)
    valid = np.isfinite(sampled)
    if selection is not None:
        valid &= selection[np.ix_(y_index, x_index)]
    if np.count_nonzero(valid) <= 5:
        raise ValueError("radial center fit needs a spatially coherent selection")
    offset = float(np.median(sampled[valid]))
    filtered = median_filter(np.where(valid, sampled, offset), size=3, mode="nearest")
    x_grid, y_grid = np.meshgrid(data.x_coordinates[x_index], data.y_coordinates[y_index])
    return (x_grid[valid], y_grid[valid]), filtered[valid]




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
) -> tuple[float, np.ndarray, float, np.ndarray, int]:
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
        int,
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
        return cost, square_sum, gradient, information, residual.size

    bounds_list = context.stripe_bounds()
    # PlotSession already owns the analysis worker.  A second persistent pool
    # inside one fit oversubscribed 4/8-panel runs and outlived every session;
    # NumPy still performs each stripe's vectorized kernels in native code.
    stripe_results = [stripe_task(bounds) for bounds in bounds_list]

    costs: list[float] = []
    square_sums: list[float] = []
    gradient = np.zeros(parameter_count, dtype=np.float64)
    dense_indices = [0, *range(2, parameter_count)]
    core = np.zeros((parameter_count - 1, parameter_count - 1), dtype=np.float64)
    offset_column = np.zeros(parameter_count - 1, dtype=np.float64)
    offset_diagonal = 0.0
    observation_count = 0
    for cost, square_sum, stripe_gradient, stripe_information, stripe_count in stripe_results:
        observation_count += stripe_count
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
    return cost, gradient, rss, information, observation_count


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


def fit_regular_separable_images(
    model: FitModelSpec,
    items: Sequence[RegularImageFitInput],
    *,
    data_revisions: Sequence[int],
    initial: Mapping[str, float] | Sequence[float] | None,
    warm_starts: Sequence[Mapping[str, float] | Sequence[float] | None],
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
    options: FitOptions,
    cancelled: Callable[[], bool] | None,
) -> tuple[tuple[FitResult | None, ...], tuple[str | None, ...]]:
    """Fit every regular image through one proxy-to-full compiled route."""

    if not items:
        return (), ()
    if len(data_revisions) != len(items) or len(warm_starts) != len(items):
        raise ValueError("regular-image batch metadata must match its cells")
    kernel = _kernel_for(model)
    started = time.monotonic()

    def check() -> None:
        if cancelled is not None and cancelled():
            raise FitCancelled("fit cancelled")
        deadline = options.deadline_seconds
        if deadline is not None and time.monotonic() - started > deadline:
            raise FitDeadlineExceeded("fit deadline exceeded")

    results: list[FitResult | None] = [None] * len(items)
    failures: list[str | None] = [None] * len(items)
    prepared: list[
        tuple[
            RegularImageFitInput,
            tuple[int, int, int] | None,
            RegularImageFitInput,
            _ImageContext,
        ]
        | None
    ] = [None] * len(items)
    for cell, incoming in enumerate(items):
        try:
            check()
            data, index_origin = _crop_to_valid_bounds(incoming, check)
            context = _ImageContext(data, check)
            proxy = _regular_image_subsample(
                data,
                check,
                _REGULAR_IMAGE_SAMPLE_LIMIT,
            )
            prepared[cell] = (data, index_origin, proxy, context)
        except (FitCancelled, FitDeadlineExceeded):
            raise
        except Exception as error:
            failures[cell] = str(error) or type(error).__name__

    base_model_lower, base_model_upper = _solver_bounds(model, None, None)
    requested_lower, requested_upper = _solver_bounds(model, None, bounds)
    requested_mask = np.asarray(
        [parameter.name in (bounds or {}) for parameter in model.parameters],
        dtype=np.bool_,
    )
    fixed_names, free_indices = _fixed_parameter_partition(model, bounds)

    def direct_seed(
        source: Mapping[str, float] | Sequence[float] | None,
        data: RegularImageFitInput,
    ) -> np.ndarray | None:
        if source is None:
            return None
        if isinstance(source, Mapping) and all(
            name in source for name in model.parameter_names
        ):
            values = np.asarray(
                [source[name] for name in model.parameter_names], dtype=np.float64
            )
        elif not isinstance(source, Mapping):
            values = np.asarray(source, dtype=np.float64).reshape(-1)
        else:
            context = _ImageContext(data, check)
            coordinates, observations = _regular_image_sample(
                data, check, context.float_observations()
            )
            values = np.asarray(
                _initial_values(model, coordinates, observations, source),
                dtype=np.float64,
            )
        if values.shape != (len(model.parameters),) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("fit initializer returned invalid parameter values")
        return values

    def refinement_bounds(data: RegularImageFitInput) -> tuple[np.ndarray, np.ndarray]:
        lower = base_model_lower.copy()
        upper = base_model_upper.copy()
        # A radius resolves nothing finer than half the pitch of the axes it
        # spans, and it is decided ONCE per radius: the radial kernel's one
        # radius spans both axes, so its floor is the finer pitch of the two
        # -- the coarse axis does not stop the fine one from resolving the
        # width, and the compiled proxy floor says the same.  Decided axis
        # by axis, whichever axis comes last overwrites the other and a
        # transposed image fits a different radius.
        floors: dict[int, float] = {}
        for parameter_index, coordinates in (
            (kernel.x_radius_index, data.x_coordinates),
            (kernel.y_radius_index, data.y_coordinates),
        ):
            differences = np.abs(np.diff(coordinates))
            resolution = (
                float(np.min(differences))
                if differences.size
                else np.finfo(np.float64).eps
            )
            floors[parameter_index] = min(
                floors.get(parameter_index, math.inf), resolution
            )
        for parameter_index, resolution in floors.items():
            lower[parameter_index] = max(
                0.5 * resolution, np.finfo(np.float64).eps
            )
        lower[-2], upper[-2] = (
            float(np.min(data.x_coordinates)),
            float(np.max(data.x_coordinates)),
        )
        lower[-1], upper[-1] = (
            float(np.min(data.y_coordinates)),
            float(np.max(data.y_coordinates)),
        )
        return lower, upper

    full_information: dict[int, np.ndarray | None] = {}

    def solve_stage(
        stage_items: Mapping[int, RegularImageFitInput],
        seeds: Mapping[int, np.ndarray] | None,
        *,
        refinement: bool,
    ) -> dict[int, tuple[np.ndarray, float, int, bool]]:
        grouped: dict[tuple[bytes, bytes, tuple[int, int]], list[int]] = {}
        for cell, data in stage_items.items():
            key = (
                data.x_coordinates.tobytes(),
                data.y_coordinates.tobytes(),
                data.observations.shape,
            )
            grouped.setdefault(key, []).append(cell)
        solved: dict[int, tuple[np.ndarray, float, int, bool]] = {}
        for cells in grouped.values():
            check()
            first = stage_items[cells[0]]
            height, width = first.observations.shape
            source = np.stack(
                [stage_items[cell].observations.reshape(-1) for cell in cells],
            )
            # The compiled boundary combines these explicit masks with value
            # and coordinate finiteness; do not build that full mask twice.
            masks = [stage_items[cell].valid_mask for cell in cells]
            valid = None
            if any(mask is not None for mask in masks):
                valid = np.stack([
                    np.ones(height * width, dtype=np.bool_)
                    if mask is None else mask.reshape(-1)
                    for mask in masks
                ])
            if refinement:
                # Numba accepts the native integer/f32/f64 camera types. Other
                # real floating storage retains the existing f64 conversion.
                if source.dtype.kind == "f" and source.dtype.itemsize not in (4, 8):
                    source = np.asarray(source, dtype=np.float64)
                values, native_context = _compiled_regular_centered_context(
                    source, np.empty((0, 0), dtype=np.bool_) if valid is None else valid,
                    width,
                )
            else:
                values = np.asarray(source, dtype=np.float64)
                native_context = np.empty((0, 0), dtype=np.float64)
            check()
            if refinement:
                bounds_rows = [refinement_bounds(stage_items[cell]) for cell in cells]
                base_lower = np.stack([row[0] for row in bounds_rows])
                base_upper = np.stack([row[1] for row in bounds_rows])
            else:
                base_lower = base_model_lower
                base_upper = base_model_upper

            authored = None
            authored_flags: bool | np.ndarray = False
            if seeds is not None:
                authored = np.stack([seeds[cell] for cell in cells])[:, None, :]
                authored_flags = True
            elif initial is not None:
                authored = np.stack(
                    [direct_seed(initial, stage_items[cell]) for cell in cells]
                )[:, None, :]
                authored_flags = True
            warm = np.zeros((len(cells), len(model.parameters)), dtype=np.float64)
            warm_flags = np.zeros(len(cells), dtype=np.bool_)
            if not refinement:
                for local, cell in enumerate(cells):
                    warm_seed = direct_seed(warm_starts[cell], stage_items[cell])
                    if warm_seed is not None:
                        warm[local] = warm_seed
                        warm_flags[local] = True
            solve_compiled = (
                _compiled_fit.solve_compiled_single
                if len(cells) == 1
                else _compiled_fit.solve_compiled_batch
            )
            coarse_proxy = not refinement and options.loss == "linear"
            output = solve_compiled(
                _compiled_regular_descriptor(kernel, refinement=refinement),
                (first.x_coordinates, first.y_coordinates),
                values[0] if len(cells) == 1 else values,
                base_lower=base_lower,
                base_upper=base_upper,
                valid=valid,
                context=native_context,
                requested_lower=requested_lower,
                requested_upper=requested_upper,
                requested_mask=requested_mask,
                authored_seeds=authored,
                use_authored=authored_flags,
                warm_seeds=warm,
                use_warm=warm_flags,
                poisson=False,
                loss=options.loss,
                max_nfev=options.max_nfev,
                ftol=_REGULAR_IMAGE_PROXY_TOL if coarse_proxy else _REGULAR_IMAGE_FTOL,
                xtol=_REGULAR_IMAGE_PROXY_TOL if coarse_proxy else _REGULAR_IMAGE_FTOL,
                gtol=_REGULAR_IMAGE_PROXY_TOL if coarse_proxy else _REGULAR_IMAGE_GTOL,
                finalize=False,
                # Regular images retain their original masked/NaN samples;
                # unlike compact point fits, they still need this boundary's
                # finite-input selection.
                all_finite=False,
            )
            direct_rss = None
            if refinement and options.loss == "linear":
                complete = np.asarray([
                    stage_items[cell].valid_mask is None for cell in cells
                ])
                if any(stage_items[cell].observations.dtype.kind == "f" for cell in cells):
                    complete &= np.all(np.isfinite(values), axis=1)
                selected = np.flatnonzero(complete)
                if selected.size:
                    matrices, direct_rss = _compiled_regular_information_batch(
                        first.x_coordinates, first.y_coordinates,
                        output.parameters, kernel is _RADIAL_KERNEL, values, complete,
                    )
                    finite = np.all(np.isfinite(matrices), axis=(1, 2))
                    for local in selected:
                        full_information[cells[local]] = matrices[local] if finite[local] else None
            for local, cell in enumerate(cells):
                solved[cell] = (
                    np.asarray(output.parameters[local], dtype=np.float64).copy(),
                    float(direct_rss[local] if direct_rss is not None and complete[local] else output.raw_rss[local]),
                    int(output.status[local]),
                    bool(output.success[local]),
                )
        return solved


    active = {cell: item[2] for cell, item in enumerate(prepared) if item is not None}
    proxy_solved = solve_stage(active, None, refinement=False)
    full_solved = solve_stage(
        {cell: prepared[cell][0] for cell in active},  # type: ignore[index]
        {cell: proxy_solved[cell][0] for cell in active},
        refinement=True,
    )
    check()

    finished: list[
        tuple[int, np.ndarray, bool, str, int, float, np.ndarray]
    ] = []
    for cell, item in enumerate(prepared):
        if item is None:
            continue
        data, index_origin, _proxy, context = item
        try:
            parameters, raw_rss, status_code, success = full_solved[cell]
            message = _compiled_fit.termination_message(status_code)
            if cell in full_information:
                observation_count = int(data.observations.size)
                information = full_information[cell]
                if information is None:
                    raise FloatingPointError("regular-image information is non-finite")
            else:
                _cost, _gradient, raw_rss, information, observation_count = (
                    _regular_image_striped_objective(
                        kernel, context, parameters, 1.0, options.loss, True,
                    )
                )
            if observation_count <= len(free_indices):
                raise ValueError(
                    "fit requires more finite observations than free parameters"
                )
            degrees = max(observation_count - len(free_indices), 1)
            reduced = raw_rss / degrees
            finished.append((
                cell, parameters, success, message, observation_count,
                reduced, information,
            ))
        except (FitCancelled, FitDeadlineExceeded):
            raise
        except Exception as error:
            failures[cell] = str(error) or type(error).__name__

    check()
    parameter_count = len(model.parameters)
    covariances = np.zeros((len(finished), parameter_count, parameter_count))
    errors_batch = np.zeros((len(finished), parameter_count))
    covariance_valid_batch = np.ones(len(finished), dtype=np.bool_)
    if finished and free_indices:
        free_index = np.asarray(free_indices, dtype=np.int64)
        matrices = np.stack([row[6] for row in finished])
        free_covariances, covariance_valid_batch = _covariance_from_information(
            matrices[:, free_index[:, None], free_index[None, :]],
            np.asarray([row[5] for row in finished]),
            np.asarray([row[4] for row in finished]),
        )
        covariances[:, free_index[:, None], free_index[None, :]] = free_covariances
        errors_batch = np.sqrt(np.maximum(
            np.diagonal(covariances, axis1=1, axis2=2), 0.0
        ))
    check()
    for row, (cell, parameters, success, message, observation_count, reduced, _info) in enumerate(finished):
        data, index_origin, _proxy, _context = prepared[cell]
        try:
            result_parameters = parameters.copy()
            deferred = _DeferredFitData(
                lambda data=data, parameters=result_parameters,
                count=observation_count,
                origin=index_origin: _regular_image_result_arrays(
                    kernel, data, parameters, count, origin
                )
            )
            results[cell] = FitResult(
                model,
                parameters,
                errors_batch[row],
                covariances[row],
                deferred,
                deferred,
                deferred,
                int(data_revisions[cell]),
                success,
                message,
                float(reduced),
                covariance_valid=bool(covariance_valid_batch[row]),
                fixed_parameter_names=fixed_names,
            )
            failures[cell] = None
        except (FitCancelled, FitDeadlineExceeded):
            raise
        except Exception as error:
            failures[cell] = str(error) or type(error).__name__
    return tuple(results), tuple(failures)


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
    """The public single fit is the one-cell form of the batch owner."""

    results, failures = fit_regular_separable_images(
        model,
        (data,),
        data_revisions=(data_revision,),
        initial=initial,
        warm_starts=(warm_start,),
        bounds=bounds,
        options=options,
        cancelled=cancelled,
    )
    result = results[0]
    if result is None:
        raise ValueError(failures[0] or "regular-image fit failed")
    return result


def production_dispatchers() -> tuple[object, ...]:
    """Regular-image roots whose machine code is production state."""

    return (
        _prepare_regular_image_refinement,
        _prepare_regular_radial_compiled,
        _prepare_regular_anisotropic_compiled,
        _compiled_regular_radial_objective,
        _compiled_regular_anisotropic_objective,
        _compiled_regular_information_batch,
        _compiled_regular_centered_context,
    )


def warm_production_cache() -> dict[str, tuple[bool, ...]]:
    """Warm radial/anisotropic single-owner batch work with real inputs."""

    from .fit import FitEngine  # noqa: PLC0415

    engine = FitEngine()
    x = np.linspace(-2.0, 2.0, 19, dtype=np.float64)
    y = np.linspace(-1.5, 1.5, 17, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x, y)
    statuses: dict[str, tuple[bool, ...]] = {}
    for model_id, parameters, storage_dtype in (
        (
            "radial_gaussian_center",
            np.asarray((3.0, 0.2, 0.7, 0.15, -0.1), dtype=np.float64),
            np.dtype(np.uint8),
        ),
        (
            "anisotropic_gaussian_center",
            np.asarray((3.0, 0.2, 0.65, 0.9, 0.15, -0.1), dtype=np.float64),
            np.dtype(np.uint16),
        ),
        (
            "radial_gaussian_center",
            np.asarray((3.0, 0.2, 0.7, 0.15, -0.1), dtype=np.float64),
            np.dtype(np.float32),
        ),
        (
            "anisotropic_gaussian_center",
            np.asarray((3.0, 0.2, 0.65, 0.9, 0.15, -0.1), dtype=np.float64),
            np.dtype(np.float64),
        ),
    ):
        model = engine.registry.get(model_id)
        image = model.evaluate(
            (grid_x.reshape(-1), grid_y.reshape(-1)), parameters
        ).reshape(y.size, x.size)
        if storage_dtype.kind == "u":
            stored = np.clip(
                np.rint(image * 40.0), 0.0, np.iinfo(storage_dtype).max
            ).astype(storage_dtype)
        else:
            stored = image.astype(storage_dtype)
        inputs = tuple(
            RegularImageFitInput(x, y, stored.copy()) for _index in range(4)
        )
        single = engine.fit(model_id, inputs[0])
        if not single.success:
            raise RuntimeError(
                f"cache warm failed for regular single {model_id}: "
                f"{single.message}"
            )
        results, failures = engine.fit_batch(
            model_id,
            inputs,
            (None,) * len(inputs),
        )
        if any(failure is not None for failure in failures):
            raise RuntimeError(
                f"cache warm failed for regular {model_id}: {failures!r}"
            )
        statuses[model_id] = tuple(
            bool(result is not None and result.success) for result in results
        )
        if not all(statuses[model_id]):
            raise RuntimeError(
                f"cache warm failed for regular {model_id}: {statuses[model_id]!r}"
            )
    return statuses
