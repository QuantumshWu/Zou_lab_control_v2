"""Optimized regular-grid radial image fitting.

The regular-image solver is isolated from the coordinate-array engine.  The
public fit catalogue supplies the model and unit semantics; this module owns
only the stripe-based numerical implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping, Sequence

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
    _initial_candidates,
    _readonly,
    _solver_bounds,
    _span,
    _value_range,
)


__all__ = ["fit_regular_radial_image"]


@dataclass(frozen=True, slots=True)
class _RegularImageSummary:
    minimum: float
    maximum: float
    scale: float
    count: int
    all_valid: bool
    normalized_sum: float
    normalized_square_sum: float


_REGULAR_IMAGE_STRIPE_ROWS = 64
_REGULAR_IMAGE_SAMPLE_LIMIT = 257
_REGULAR_IMAGE_MIN_RELATIVE_CONTRAST = 0.05
_REGULAR_IMAGE_MAX_LINE_SEARCH_STEPS = 50
_REGULAR_IMAGE_FTOL = 1e-10
_REGULAR_IMAGE_GTOL = 1e-8


def fit_regular_radial_image(
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
    """Fit the built-in radial Gaussian without expanding coordinate grids."""

    started = time.monotonic()

    def check() -> None:
        if cancelled is not None and cancelled():
            raise FitCancelled("fit cancelled")
        deadline = options.deadline_seconds
        if deadline is not None and time.monotonic() - started > deadline:
            raise FitDeadlineExceeded("fit deadline exceeded")

    summary = _regular_image_summary(data, check)
    if summary.count <= len(model.parameters):
        raise ValueError("fit requires more finite observations than parameters")
    optimization_data = _regular_image_optimization_input(data, check)
    optimization_summary = (
        summary
        if optimization_data is data
        else _regular_image_summary(optimization_data, check)
    )
    seed_coordinates, seed_values = _regular_image_sample(optimization_data, check)
    seeds = _initial_candidates(
        model,
        seed_coordinates,
        seed_values,
        initial,
        warm_start,
    )
    if initial is None:
        strongest = max(abs(float(seed[0])) for seed in seeds)
        seeds = tuple(
            seed
            for seed in seeds
            if abs(float(seed[0])) >= strongest * _REGULAR_IMAGE_MIN_RELATIVE_CONTRAST
        )
    default_bounds = (
        model.bounds_initializer(seed_coordinates, seed_values)
        if model.bounds_initializer is not None
        else None
    )
    lower, upper = _solver_bounds(model, default_bounds, bounds)
    lower_inside, upper_inside = np.nextafter(lower, upper), np.nextafter(upper, lower)
    seeds = tuple(np.clip(seed, lower_inside, upper_inside) for seed in seeds)

    value_range = _value_range(seed_values)
    x_span, y_span = _span(data.x_coordinates), _span(data.y_coordinates)
    natural_scale = np.asarray(
        (value_range, value_range, max(x_span, y_span) / 4.0, x_span / 4.0, y_span / 4.0)
    )
    optimization_scale = (
        optimization_summary.scale if options.loss == "linear" else 1.0
    )

    def optimization_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        check()
        if optimization_summary.all_valid and options.loss == "linear":
            return _regular_image_linear_objective(
                optimization_data, optimization_summary, parameters
            )
        cost, gradient, _rss, _information = _regular_image_striped_objective(
            optimization_data,
            parameters,
            optimization_scale,
            options.loss,
            False,
            check,
        )
        return cost, gradient

    full_scale = summary.scale if options.loss == "linear" else 1.0

    def full_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        check()
        if summary.all_valid and options.loss == "linear":
            return _regular_image_linear_objective(data, summary, parameters)
        cost, gradient, _rss, _information = _regular_image_striped_objective(
            data, parameters, full_scale, options.loss, False, check
        )
        return cost, gradient

    candidates: list[tuple[float, Any, np.ndarray]] = []
    last_error: Exception | None = None
    solver_options = {
        "ftol": _REGULAR_IMAGE_FTOL,
        "gtol": _REGULAR_IMAGE_GTOL,
        "maxfun": options.max_nfev,
        "maxiter": options.max_nfev,
        "maxls": _REGULAR_IMAGE_MAX_LINE_SEARCH_STEPS,
    }
    for seed in seeds:
        check()
        parameter_scale = np.maximum(np.abs(seed), natural_scale)

        def scaled_objective(scaled: np.ndarray) -> tuple[float, np.ndarray]:
            cost, gradient = optimization_objective(scaled * parameter_scale)
            return cost, gradient * parameter_scale

        try:
            solved = minimize(
                scaled_objective,
                seed / parameter_scale,
                method="L-BFGS-B",
                jac=True,
                bounds=tuple(zip(lower / parameter_scale, upper / parameter_scale)),
                options=solver_options,
            )
            check()
            parameters = np.asarray(solved.x) * parameter_scale
            cost, _gradient = optimization_objective(parameters)
            if math.isfinite(cost) and np.all(np.isfinite(parameters)):
                candidates.append((cost, solved, parameters))
        except (FitCancelled, FitDeadlineExceeded):
            raise
        except (ValueError, RuntimeError, FloatingPointError) as error:
            last_error = error
    if not candidates:
        if last_error is not None:
            raise last_error
        raise RuntimeError("regular-image optimizer failed for every initializer")

    _cost, solved, parameters = min(
        candidates, key=lambda item: (not bool(item[1].success), item[0])
    )
    if optimization_data is not data:
        parameter_scale = np.maximum(np.abs(parameters), natural_scale)
        initial_scaled = parameters / parameter_scale
        _full_cost, full_gradient = full_objective(parameters)
        if np.max(np.abs(full_gradient * parameter_scale)) > _REGULAR_IMAGE_GTOL:

            def scaled_full_objective(
                scaled: np.ndarray,
            ) -> tuple[float, np.ndarray]:
                cost, gradient = full_objective(scaled * parameter_scale)
                return cost, gradient * parameter_scale

            refined = minimize(
                scaled_full_objective,
                initial_scaled,
                method="L-BFGS-B",
                jac=True,
                bounds=tuple(
                    zip(lower / parameter_scale, upper / parameter_scale)
                ),
                options=solver_options,
            )
            check()
            refined_parameters = np.asarray(refined.x) * parameter_scale
            refined_cost, _gradient = full_objective(refined_parameters)
            if not math.isfinite(refined_cost) or not np.all(
                np.isfinite(refined_parameters)
            ):
                raise FloatingPointError(
                    "full-resolution regular-image refinement is non-finite"
                )
            solved, parameters = refined, refined_parameters
    if summary.all_valid and options.loss == "linear":
        information = _regular_image_linear_information(
            data, parameters, full_scale
        )
    else:
        _cost, _gradient, _rss, information = _regular_image_striped_objective(
            data, parameters, full_scale, options.loss, True, check
        )
    fitted, residuals, indices = _regular_image_result_arrays(
        data, parameters, summary.count, check
    )
    normalized_rss = float(np.dot(residuals, residuals) / full_scale**2)
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
    fitted, residuals, indices = (
        _readonly(array) for array in (fitted, residuals, indices)
    )
    return FitResult(
        model,
        parameters,
        errors,
        covariance,
        fitted,
        residuals,
        indices,
        data_revision,
        bool(solved.success),
        solved.message,
        float(normalized_reduced * full_scale**2),
        covariance_valid=covariance_valid,
    )


def _regular_image_stripes(
    data: RegularImageFitInput,
    check: Callable[[], None],
):
    height = data.observations.shape[0]
    for start in range(0, height, _REGULAR_IMAGE_STRIPE_ROWS):
        check()
        stop = min(height, start + _REGULAR_IMAGE_STRIPE_ROWS)
        source = np.asarray(data.observations[start:stop])
        observed = source.astype(np.float64, copy=False)
        mask = None if data.valid_mask is None else data.valid_mask[start:stop]
        if source.dtype.kind == "f":
            finite = np.isfinite(source)
            mask = finite if mask is None else mask & finite
        if mask is not None and bool(np.all(mask)):
            mask = None
        yield start, observed, mask


def _regular_image_summary(
    data: RegularImageFitInput,
    check: Callable[[], None],
) -> _RegularImageSummary:
    minimum, maximum, scale, count = math.inf, -math.inf, 0.0, 0
    all_valid = True
    for _start, observed, mask in _regular_image_stripes(data, check):
        all_valid &= mask is None
        values = observed.reshape(-1) if mask is None else observed[mask]
        if values.size:
            low, high = float(np.min(values)), float(np.max(values))
            minimum, maximum = min(minimum, low), max(maximum, high)
            scale = max(scale, abs(low), abs(high))
            count += values.size
    if count == 0:
        raise ValueError("regular image has no finite valid observations")
    scale = scale or 1.0
    sums: list[float] = []
    square_sums: list[float] = []
    for _start, observed, mask in _regular_image_stripes(data, check):
        values = observed.reshape(-1) if mask is None else observed[mask]
        if values.size:
            normalized = values / scale
            sums.append(float(np.sum(normalized)))
            square_sums.append(float(np.dot(normalized, normalized)))
    return _RegularImageSummary(
        minimum, maximum, scale, count, all_valid, math.fsum(sums), math.fsum(square_sums)
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


def _regular_image_optimization_input(
    data: RegularImageFitInput,
    check: Callable[[], None],
) -> RegularImageFitInput:
    height, width = data.observations.shape
    if height <= _REGULAR_IMAGE_SAMPLE_LIMIT and width <= _REGULAR_IMAGE_SAMPLE_LIMIT:
        return data

    valid_y = np.zeros(height, dtype=np.bool_)
    valid_x = np.zeros(width, dtype=np.bool_)
    for start, observed, mask in _regular_image_stripes(data, check):
        stop = start + observed.shape[0]
        if mask is None:
            valid_y[start:stop] = True
            valid_x[:] = True
        else:
            valid_y[start:stop] = np.any(mask, axis=1)
            valid_x |= np.any(mask, axis=0)

    def bounded_indices(valid: np.ndarray) -> np.ndarray:
        indices = np.flatnonzero(valid)
        count = min(indices.size, _REGULAR_IMAGE_SAMPLE_LIMIT)
        positions = np.linspace(0, indices.size - 1, count, dtype=np.int64)
        return indices[positions]

    y_index, x_index = bounded_indices(valid_y), bounded_indices(valid_x)
    if y_index.size == 0 or x_index.size == 0:
        raise ValueError("regular image has no finite valid observations")
    observations = np.asarray(
        data.observations[np.ix_(y_index, x_index)], dtype=np.float64
    )
    valid = np.isfinite(observations)
    if data.valid_mask is not None:
        valid &= data.valid_mask[np.ix_(y_index, x_index)]
    valid_mask = None if bool(np.all(valid)) else valid
    return RegularImageFitInput(
        data.x_coordinates[x_index],
        data.y_coordinates[y_index],
        observations,
        valid_mask=valid_mask,
    )


def _regular_image_linear_objective(
    data: RegularImageFitInput,
    summary: _RegularImageSummary,
    parameters: np.ndarray,
) -> tuple[float, np.ndarray]:
    amplitude, offset, radius, center_x, center_y = map(float, parameters)
    x, x_radius, x_center = _radial_axis_terms(data.x_coordinates, center_x, radius)
    y, y_radius, y_center = _radial_axis_terms(data.y_coordinates, center_y, radius)
    projected = np.einsum(
        "ij,jk->ik",
        data.observations,
        np.column_stack((x, x_radius, x_center)),
        optimize=False,
        dtype=np.float64,
        casting="unsafe",
    ) / summary.scale

    x_sum, y_sum = float(np.sum(x)), float(np.sum(y))
    x_square, y_square = float(np.dot(x, x)), float(np.dot(y, y))
    phi_sum, phi_square = x_sum * y_sum, x_square * y_square
    data_phi = float(np.dot(y, projected[:, 0]))
    data_derivatives = np.asarray(
        (
            np.dot(y, projected[:, 1]) + np.dot(y_radius, projected[:, 0]),
            np.dot(y, projected[:, 2]),
            np.dot(y_center, projected[:, 0]),
        )
    )
    derivative_sums = np.asarray(
        (
            y_sum * np.sum(x_radius) + np.sum(y_radius) * x_sum,
            y_sum * np.sum(x_center),
            np.sum(y_center) * x_sum,
        )
    )
    phi_derivatives = np.asarray(
        (
            y_square * np.dot(x, x_radius) + np.dot(y, y_radius) * x_square,
            y_square * np.dot(x, x_center),
            np.dot(y, y_center) * x_square,
        )
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
    data: RegularImageFitInput,
    parameters: np.ndarray,
    scale: float,
) -> np.ndarray:
    amplitude, _offset, radius, center_x, center_y = map(float, parameters)
    x, x_radius, x_center = _radial_axis_terms(
        data.x_coordinates, center_x, radius
    )
    y, y_radius, y_center = _radial_axis_terms(
        data.y_coordinates, center_y, radius
    )
    terms: tuple[tuple[tuple[float, np.ndarray | None, np.ndarray | None], ...], ...] = (
        ((1.0, y, x),),
        ((1.0, None, None),),
        ((amplitude, y, x_radius), (amplitude, y_radius, x)),
        ((amplitude, y, x_center),),
        ((amplitude, y_center, x),),
    )

    def axis_inner(
        left: np.ndarray | None,
        right: np.ndarray | None,
        size: int,
    ) -> float:
        if left is None:
            return float(size) if right is None else float(np.sum(right))
        return float(np.sum(left)) if right is None else float(np.dot(left, right))

    information = np.empty((5, 5), dtype=np.float64)
    for row, row_terms in enumerate(terms):
        for column in range(row, len(terms)):
            value = math.fsum(
                row_scale
                * column_scale
                * axis_inner(row_y, column_y, data.y_coordinates.size)
                * axis_inner(row_x, column_x, data.x_coordinates.size)
                for row_scale, row_y, row_x in row_terms
                for column_scale, column_y, column_x in terms[column]
            ) / scale**2
            information[row, column] = information[column, row] = value
    if not np.all(np.isfinite(information)):
        raise FloatingPointError("regular-image information is non-finite")
    return information



def _radial_axis_terms(
    coordinates: np.ndarray,
    center: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = coordinates - center
    basis = np.exp(-(delta**2) / radius**2)
    return basis, basis * (2.0 * delta**2 / radius**3), basis * (2.0 * delta / radius**2)


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
    data: RegularImageFitInput,
    parameters: np.ndarray,
    scale: float,
    loss: str,
    collect_information: bool,
    check: Callable[[], None],
) -> tuple[float, np.ndarray, float, np.ndarray]:
    amplitude, offset, radius, center_x, center_y = map(float, parameters)
    x_basis, x_radius, x_center = _radial_axis_terms(
        data.x_coordinates, center_x, radius
    )
    gradient = np.zeros(5, dtype=np.float64)
    information = np.zeros((5, 5), dtype=np.float64)
    costs: list[float] = []
    square_sums: list[float] = []
    for start, observed, mask in _regular_image_stripes(data, check):
        stop = start + observed.shape[0]
        y_basis, y_radius, y_center = _radial_axis_terms(
            data.y_coordinates[start:stop], center_y, radius
        )
        radial = y_basis[:, None] * x_basis[None, :]
        residual = (amplitude * radial + offset - observed) / scale
        derivatives = (
            radial,
            np.ones_like(radial),
            amplitude
            * (y_basis[:, None] * x_radius[None, :] + y_radius[:, None] * x_basis[None, :]),
            amplitude * y_basis[:, None] * x_center[None, :],
            amplitude * y_center[:, None] * x_basis[None, :],
        )
        residual = residual.reshape(-1)
        if mask is not None:
            selected = mask.reshape(-1)
            residual = residual[selected]
            derivatives = tuple(derivative.reshape(-1)[selected] for derivative in derivatives)
        else:
            derivatives = tuple(derivative.reshape(-1) for derivative in derivatives)
        rho, first, information_weight = _regular_image_loss_terms(residual, loss)
        costs.append(0.5 * float(np.sum(rho)))
        square_sums.append(float(np.dot(residual, residual)))
        weighted_residual = first * residual
        gradient += np.asarray(
            [np.dot(derivative, weighted_residual) / scale for derivative in derivatives]
        )
        if collect_information:
            jacobian = np.column_stack(derivatives) / scale
            information += jacobian.T @ (information_weight[:, None] * jacobian)
    cost, rss = math.fsum(costs), math.fsum(square_sums)
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
    data: RegularImageFitInput,
    parameters: np.ndarray,
    count: int,
    check: Callable[[], None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amplitude, offset, radius, center_x, center_y = map(float, parameters)
    x_basis = _radial_axis_terms(data.x_coordinates, center_x, radius)[0]
    fitted = np.empty(count, dtype=np.float64)
    residuals = np.empty(count, dtype=np.float64)
    indices = np.empty(count, dtype=np.int64)
    cursor = 0
    width = data.x_coordinates.size
    for start, observed, mask in _regular_image_stripes(data, check):
        stop = start + observed.shape[0]
        y_basis = _radial_axis_terms(
            data.y_coordinates[start:stop], center_y, radius
        )[0]
        predicted = amplitude * y_basis[:, None] * x_basis[None, :] + offset
        local_indices = (
            np.arange(start * width, stop * width, dtype=np.int64).reshape(observed.shape)
            if data.selected_indices is None
            else data.selected_indices[start:stop]
        )
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
