"""Optimized separable regular-grid image fitting.

The regular-image solver is isolated from the coordinate-array engine.  The
public fit catalogue supplies the model and unit semantics; this module owns
the stripe/BLAS numerical implementation shared by every separable Gaussian
image model (the built-in radial and anisotropic centers).

The solver seeds every cell on a bounded proxy through the common independent
TRF.  Multi-cell refinement evaluates the exact separable full-image objective
in one compiled batch; the one-cell specialization keeps the faster BLAS
axis form.  Both enter through the same public owner and convergence contract.
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
from numba import njit
from scipy.ndimage import median_filter
from scipy.optimize import minimize

from . import _fit_compiled as _compiled_fit
from . import _raster_kernels
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
    _expand_fixed_covariance,
    _fixed_parameter_partition,
    _initial_values,
    _solver_bounds,
    _span,
    _value_range,
)


__all__ = ["fit_regular_separable_image"]

_COMPILED_LINEAR_LOSS = int(_compiled_fit.LOSS_CODES["linear"])


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
_REGULAR_IMAGE_SAMPLE_LIMIT = 129
_REGULAR_IMAGE_MAX_LINE_SEARCH_STEPS = 75
_REGULAR_IMAGE_MAX_NEWTON_STEPS = 8
_REGULAR_IMAGE_FTOL = 1e-10
_REGULAR_IMAGE_GTOL = 1e-8
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


@njit(cache=True, nogil=True)
def _promote_unsigned_summary(
    source: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    """Promote a compact camera plane while deriving its exact moments."""

    target = np.empty(source.shape, dtype=np.float64)
    incoming = source.reshape(-1)
    outgoing = target.reshape(-1)
    minimum = float(incoming[0])
    maximum = minimum
    total = 0.0
    square_total = 0.0
    for index in range(incoming.size):
        value = float(incoming[index])
        outgoing[index] = value
        minimum = min(minimum, value)
        maximum = max(maximum, value)
        total += value
        square_total += value * value
    return target, minimum, maximum, total, square_total


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
def _prepare_regular_radial_compiled(
    coordinates,
    observations,
    valid,
    seeds,
    lower,
    upper,
    context,
):
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
):
    """Closed-form linear residual derivatives for one complete image."""

    height = y_vectors.shape[1]
    width = x_vectors.shape[1]
    amplitude = parameters[0]
    offset = parameters[1]
    projected_count = 4 if derivatives else 1
    projected = np.zeros((projected_count, height), dtype=np.float64)
    raw_rss = 0.0
    for row_index in range(height):
        y_basis = y_vectors[0, row_index]
        for column in range(width):
            point = row_index * width + column
            observed = observations[point]
            x_basis = x_vectors[0, column]
            residual = amplitude * y_basis * x_basis + offset - observed
            raw_rss += residual * residual
            projected[0, row_index] += observed * x_basis
            if derivatives:
                projected[1, row_index] += observed * x_vectors[1, column]
                projected[2, row_index] += observed * x_vectors[2, column]
                projected[3, row_index] += observed
    if not derivatives:
        return 0.5 * raw_rss, raw_rss, math.isfinite(raw_rss)

    x_full = np.ones((4, width), dtype=np.float64)
    y_full = np.ones((4, height), dtype=np.float64)
    for vector in range(3):
        for column in range(width):
            x_full[vector, column] = x_vectors[vector, column]
        for row_index in range(height):
            y_full[vector, row_index] = y_vectors[vector, row_index]
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
                data_dot += y_full[left, row_index] * projected[right, row_index]
            y_inner[left, right] = y_dot
            data_inner[left, right] = data_dot

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
):
    """Exact regular-grid Gaussian objective with one axis exponential pass."""

    point_count = observations.size
    if point_count == 0 or coordinates.shape[0] != 2:
        return math.inf, math.inf, False
    width = 1
    first_y = coordinates[1, 0]
    while width < point_count and coordinates[1, width] == first_y:
        width += 1
    if point_count % width:
        return math.inf, math.inf, False
    height = point_count // width
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
            coordinates[0, column], center_x, radius_x
        )
        x_vectors[0, column] = values[0]
        x_vectors[1, column] = values[1]
        x_vectors[2, column] = values[2]
    for row_index in range(height):
        values = _compiled_axis_terms(
            coordinates[1, row_index * width], center_y, radius_y
        )
        y_vectors[0, row_index] = values[0]
        y_vectors[1, row_index] = values[1]
        y_vectors[2, row_index] = values[2]

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
        )

    if derivatives:
        _compiled_fit.compiled_reset_accumulators(gradient, information)
    cost = 0.0
    raw_rss = 0.0
    full_row = np.empty(parameters.size, dtype=np.float64)
    amplitude = parameters[0]
    offset = parameters[1]
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
                observations[point],
                poisson,
                weights[point],
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
        "_unsigned_summary",
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
        self._unsigned_summary: tuple[float, float, float, float] | None = None

    def float_observations(self) -> np.ndarray:
        """Promote the image to float64 exactly once so '@' hits BLAS.

        The cache is forced C-contiguous: projected payloads may hand the
        solver a transposed view, and strided stripes would keep every dot
        and matmul off the fast BLAS paths.
        """

        cached = self._float_observations
        if cached is None:
            cached = np.asarray(self.data.observations)
            if (
                cached.dtype.kind == "u"
                and cached.dtype.itemsize <= 2
            ):
                # SEALED, not merely contiguous.  ``ascontiguousarray`` hands
                # back whatever it was given when that is already contiguous,
                # so the plane's mutability reached the kernel as an accident
                # of its origin: a published snapshot is read-only, a warmer's
                # or a notebook's fresh copy is writable, and numba compiled
                # and cached the promotion twice per dtype for the difference.
                (
                    cached,
                    minimum,
                    maximum,
                    total,
                    square_total,
                ) = _promote_unsigned_summary(_raster_kernels.readable(cached))
                self._unsigned_summary = (
                    minimum,
                    maximum,
                    total,
                    square_total,
                )
            elif cached.dtype != np.float64 or not cached.flags.c_contiguous:
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


def _regular_image_summary(context: _ImageContext) -> _RegularImageSummary:
    """One fused sweep: extrema, count, validity and normalization sums."""

    check = context.check
    data = context.data
    observed = context.float_observations()
    if data.valid_mask is None and data.observations.dtype.kind != "f":
        # Unsigned camera planes derive all four statistics while promotion
        # fills the float cache.  Other integer sources retain the exact
        # stripe reduction used before this fused camera path.
        check()
        count = int(data.observations.size)
        fused = context._unsigned_summary
        if fused is not None:
            minimum, maximum, total, square_total = fused
        else:
            minimum = float(np.min(data.observations))
            maximum = float(np.max(data.observations))
            sums = []
            square_sums = []
            for start, stop in context.stripe_bounds():
                check()
                values = observed[start:stop].reshape(-1)
                sums.append(float(np.sum(values)))
                square_sums.append(float(np.dot(values, values)))
            total = math.fsum(sums)
            square_total = math.fsum(square_sums)
        scale = max(abs(minimum), abs(maximum)) or 1.0
        return _RegularImageSummary(
            minimum,
            maximum,
            scale,
            count,
            True,
            total / scale,
            square_total / scale**2,
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
            x_grid = np.ascontiguousarray(
                np.broadcast_to(first.x_coordinates, (height, width)).reshape(-1)
            )
            y_grid = np.ascontiguousarray(
                np.broadcast_to(
                    first.y_coordinates[:, None], (height, width)
                ).reshape(-1)
            )
            values = np.stack(
                [stage_items[cell].observations.reshape(-1) for cell in cells]
            )
            valid_rows = []
            all_valid = True
            for cell in cells:
                data = stage_items[cell]
                valid = np.ones(data.observations.shape, dtype=np.bool_)
                if data.valid_mask is not None:
                    valid &= data.valid_mask
                if data.observations.dtype.kind == "f":
                    valid &= np.isfinite(data.observations)
                all_valid &= bool(np.all(valid))
                valid_rows.append(valid.reshape(-1))
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
            output = solve_compiled(
                _compiled_regular_descriptor(kernel, refinement=refinement),
                (x_grid, y_grid),
                values[0] if len(cells) == 1 else values,
                base_lower=base_lower,
                base_upper=base_upper,
                valid=(
                    None
                    if all_valid
                    else valid_rows[0]
                    if len(cells) == 1
                    else np.stack(valid_rows)
                ),
                context=np.empty((0, 0), dtype=np.float64),
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
                ftol=_REGULAR_IMAGE_FTOL,
                xtol=_REGULAR_IMAGE_FTOL,
                gtol=_REGULAR_IMAGE_GTOL,
                finalize=False,
            )
            for local, cell in enumerate(cells):
                solved[cell] = (
                    np.asarray(output.parameters[local], dtype=np.float64).copy(),
                    float(output.raw_rss[local]),
                    int(output.status[local]),
                    bool(output.success[local]),
                )
        return solved

    def refine_cell(
        data: RegularImageFitInput,
        context: _ImageContext,
        parameters: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        _SolverStatus,
        _ImageContext,
        _RegularImageSummary,
        np.ndarray,
    ]:
        summary = _regular_image_summary(context)
        if summary.count <= len(free_indices):
            raise ValueError(
                "fit requires more finite observations than free parameters"
            )
        lower, upper = refinement_bounds(data)
        lower[requested_mask] = requested_lower[requested_mask]
        upper[requested_mask] = requested_upper[requested_mask]
        lower_inside = np.nextafter(lower, upper)
        upper_inside = np.nextafter(upper, lower)
        free_index = np.asarray(free_indices, dtype=np.int64)
        current = np.asarray(parameters, dtype=np.float64).copy()
        if free_indices:
            current[free_index] = np.clip(
                current[free_index], lower_inside[free_index], upper_inside[free_index]
            )
        fixed = np.flatnonzero(lower == upper)
        current[fixed] = lower[fixed]
        scale = summary.scale if options.loss == "linear" else 1.0
        natural_scale = np.asarray(
            kernel.natural_scale_builder(
                max(summary.maximum - summary.minimum, np.finfo(float).eps),
                _span(data.x_coordinates),
                _span(data.y_coordinates),
            ),
            dtype=np.float64,
        )

        def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
            check()
            if summary.all_valid and options.loss == "linear":
                return _regular_image_linear_objective(
                    kernel, context, summary, values
                )
            cost, gradient, _rss, _information = (
                _regular_image_striped_objective(
                    kernel, context, values, scale, options.loss, False
                )
            )
            return cost, gradient

        def information(values: np.ndarray) -> np.ndarray:
            if summary.all_valid and options.loss == "linear":
                return _regular_image_linear_information(
                    kernel, context, values, scale
                )
            return _regular_image_striped_objective(
                kernel, context, values, scale, options.loss, True
            )[3]

        def gradient_norm(values: np.ndarray, gradient: np.ndarray) -> float:
            if not free_indices:
                return 0.0
            parameter_scale = np.maximum(
                np.abs(values[free_index]), natural_scale[free_index]
            )
            return float(
                np.max(np.abs(gradient[free_index] * parameter_scale))
            )

        def converged(values: np.ndarray, gradient: np.ndarray) -> bool:
            return gradient_norm(values, gradient) <= _REGULAR_IMAGE_GTOL

        def newton_polish(
            values: np.ndarray,
            cost: float,
            gradient: np.ndarray,
        ) -> tuple[np.ndarray, float, np.ndarray, bool]:
            for _step in range(_REGULAR_IMAGE_MAX_NEWTON_STEPS):
                if converged(values, gradient):
                    return values, cost, gradient, True
                try:
                    step = np.linalg.solve(
                        information(values)[np.ix_(free_index, free_index)],
                        -gradient[free_index],
                    )
                except np.linalg.LinAlgError:
                    break
                candidate = values.copy()
                candidate[free_index] = np.clip(
                    values[free_index] + step,
                    lower_inside[free_index],
                    upper_inside[free_index],
                )
                candidate_cost, candidate_gradient = objective(candidate)
                improved = math.isfinite(candidate_cost) and (
                    candidate_cost < cost
                    or gradient_norm(candidate, candidate_gradient)
                    < gradient_norm(values, gradient)
                )
                if not improved:
                    break
                values, cost, gradient = (
                    candidate,
                    candidate_cost,
                    candidate_gradient,
                )
            return values, cost, gradient, converged(values, gradient)

        cost, gradient = objective(current)
        current, cost, gradient, polished = newton_polish(
            current, cost, gradient
        )
        status = _SolverStatus(
            polished, "full-resolution Newton refinement converged"
            if polished
            else "full-resolution refinement did not converge"
        )
        if not status.success and free_indices:
            parameter_scale = np.maximum(
                np.abs(current[free_index]), natural_scale[free_index]
            )

            def scaled_objective(values: np.ndarray) -> tuple[float, np.ndarray]:
                complete = current.copy()
                complete[free_index] = values * parameter_scale
                local_cost, local_gradient = objective(complete)
                return local_cost, local_gradient[free_index] * parameter_scale

            solved = minimize(
                scaled_objective,
                current[free_index] / parameter_scale,
                method="L-BFGS-B",
                jac=True,
                bounds=tuple(
                    zip(
                        lower[free_index] / parameter_scale,
                        upper[free_index] / parameter_scale,
                    )
                ),
                options={
                    "ftol": _REGULAR_IMAGE_FTOL,
                    "gtol": _REGULAR_IMAGE_GTOL,
                    "maxfun": options.max_nfev,
                    "maxiter": options.max_nfev,
                    "maxls": _REGULAR_IMAGE_MAX_LINE_SEARCH_STEPS,
                },
            )
            current[free_index] = np.asarray(solved.x) * parameter_scale
            cost, gradient = objective(current)
            current, cost, gradient, polished = newton_polish(
                current, cost, gradient
            )
            status = (
                _SolverStatus(
                    True, "full-resolution Newton refinement converged"
                )
                if polished
                else _SolverStatus(bool(solved.success), str(solved.message))
            )
        final_information = information(current)
        if summary.all_valid and options.loss == "linear":
            raw_rss = 2.0 * cost * scale**2
        else:
            _cost, _gradient, raw_rss, final_information = (
                _regular_image_striped_objective(
                    kernel, context, current, scale, options.loss, True
                )
            )
            if options.loss == "linear":
                raw_rss *= scale**2
        return current, raw_rss, status, context, summary, final_information

    active = {cell: item[2] for cell, item in enumerate(prepared) if item is not None}
    proxy_solved = solve_stage(active, None, refinement=False)
    batch_refinement = len(active) > 1
    full_solved = (
        solve_stage(
            {cell: prepared[cell][0] for cell in active},  # type: ignore[index]
            {cell: proxy_solved[cell][0] for cell in active},
            refinement=True,
        )
        if batch_refinement
        else {}
    )
    check()

    for cell, item in enumerate(prepared):
        if item is None:
            continue
        data, index_origin, _proxy, context = item
        try:
            if batch_refinement:
                parameters, raw_rss, status_code, success = full_solved[cell]
                status = _SolverStatus(
                    success, _compiled_fit.termination_message(status_code)
                )
                complete_linear = (
                    options.loss == "linear"
                    and data.valid_mask is None
                    and (
                        data.observations.dtype.kind != "f"
                        or context.finite_everywhere()
                    )
                )
                if complete_linear:
                    observation_count = int(data.observations.size)
                    information = _regular_image_linear_information(
                        kernel, context, parameters, 1.0
                    )
                else:
                    summary = _regular_image_summary(context)
                    observation_count = summary.count
                    _cost, _gradient, raw_rss, information = (
                        _regular_image_striped_objective(
                            kernel,
                            context,
                            parameters,
                            1.0,
                            options.loss,
                            True,
                        )
                    )
                information_scale = 1.0
            else:
                (
                    parameters,
                    raw_rss,
                    status,
                    context,
                    summary,
                    information,
                ) = refine_cell(data, context, proxy_solved[cell][0])
                observation_count = summary.count
                information_scale = (
                    summary.scale if options.loss == "linear" else 1.0
                )
            if observation_count <= len(free_indices):
                raise ValueError(
                    "fit requires more finite observations than free parameters"
                )
            degrees = max(observation_count - len(free_indices), 1)
            reduced = raw_rss / degrees
            covariance_reduced = reduced / information_scale**2
            if free_indices:
                free_index = np.asarray(free_indices, dtype=np.int64)
                free_covariance, covariance_valid = _covariance_from_information(
                    information[np.ix_(free_index, free_index)],
                    covariance_reduced,
                    observation_count,
                )
                covariance, errors = _expand_fixed_covariance(
                    len(model.parameters),
                    free_indices,
                    free_covariance,
                    covariance_valid,
                )
            else:
                covariance_valid = True
                covariance = np.zeros((len(model.parameters),) * 2)
                errors = np.zeros(len(model.parameters))
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
                errors,
                covariance,
                deferred,
                deferred,
                deferred,
                int(data_revisions[cell]),
                status.success,
                status.message,
                float(reduced),
                covariance_valid=covariance_valid,
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
    """The six regular-image roots whose machine code is production state."""

    return (
        _promote_unsigned_summary,
        _prepare_regular_image_refinement,
        _prepare_regular_radial_compiled,
        _prepare_regular_anisotropic_compiled,
        _compiled_regular_radial_objective,
        _compiled_regular_anisotropic_objective,
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
    ):
        model = engine.registry.get(model_id)
        image = model.evaluate(
            (grid_x.reshape(-1), grid_y.reshape(-1)), parameters
        ).reshape(y.size, x.size)
        maximum = np.iinfo(storage_dtype).max
        stored = np.clip(np.rint(image * 40.0), 0.0, maximum).astype(
            storage_dtype
        )
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
