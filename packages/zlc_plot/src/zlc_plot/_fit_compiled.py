"""Compiled, model-neutral least-squares core used by :mod:`zlc_plot.fit`.

The scalar fit path has two costs that become especially visible in a Facet:
Python repeats model preparation for every cell, then SciPy re-enters Python for
every residual and Jacobian evaluation.  This module removes those crossings
without moving model identity into the solver.  A model supplies one
``CompiledFitDescriptor`` whose callbacks all obey the same array ABI; the
optimizer never branches on a model id.

The callback ABI is deliberately small and write-oriented:

``prepare(coords, observations, valid, seeds, lower, upper, context) -> count``
    Fill *full-parameter* cold seeds and model-derived bounds for one cell.
    ``seeds`` has ``descriptor.max_candidates`` rows.  The common preparation
    owner subsequently applies explicit requested bounds/fixed parameters,
    inserts a warm seed first, chooses authored seeds instead of cold seeds when
    requested, clips, and exactly de-duplicates candidates.

``objective(coords, observations, valid, full_parameters, free_indices,
weights, use_weights, poisson, loss_code, gradient, information, jacobian_row,
with_derivatives) -> (cost, raw_rss, finite)``
    Evaluate the robust objective and, when requested, fill the gradient and
    Gauss--Newton information for *free* parameters.  Model callbacks should
    use ``compiled_point_terms`` and the accumulator helpers below so linear,
    Poisson and robust-loss semantics stay identical.

``value_jacobian(coords, full_parameters) -> (values, full_jacobian)``
    Evaluate the winning model and its analytic full-parameter Jacobian.  The
    common finalizer projects free columns, reconstructs SciPy's robust scaled
    Jacobian, and uses a strict SVD rank test for covariance.

All Numba dispatchers are intentionally declared without explicit signatures.
Importing this module therefore compiles nothing.  The first model actually
used specializes the core lazily; ``cache=True`` stores that specialization in
the repository-owned Numba cache for later processes.  Single fits use the
serial wrapper.  Multi-cell fits use the same numerical routine under
``prange``; cells and candidate trust regions never share numerical state.

This module returns numeric ``CompiledFitOutput`` only.  ``FitResult`` remains
owned by :mod:`zlc_plot.fit`, including model metadata and public validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Callable, Sequence

import numpy as np

from . import _kernel_cache

# Numba reads NUMBA_CACHE_DIR while dispatchers are created.  Install the one
# repository owner before importing numba; doing this afterwards silently puts
# compiled files somewhere else.
_kernel_cache.install()

from numba import njit, prange, types as nb_types  # noqa: E402


EPSILON = np.finfo(np.float64).eps
COUNT_FLOOR = 1.0e-9
RSS_TIE_RELATIVE = 1.0e-10

LOSS_LINEAR = 0
LOSS_SOFT_L1 = 1
LOSS_HUBER = 2
LOSS_CAUCHY = 3
LOSS_ARCTAN = 4
LOSS_CODES = {
    "linear": LOSS_LINEAR,
    "soft_l1": LOSS_SOFT_L1,
    "huber": LOSS_HUBER,
    "cauchy": LOSS_CAUCHY,
    "arctan": LOSS_ARCTAN,
}

# Match scipy.optimize.least_squares positive termination values where useful.
STATUS_INVALID = -2
STATUS_NO_CANDIDATE = -1
STATUS_MAX_NFEV = 0
STATUS_GTOL = 1
STATUS_FTOL = 2
STATUS_XTOL = 3
STATUS_FTOL_XTOL = 4

# ABI types are inert declarations.  Compilation happens in
# ``_ensure_compiled_abi`` on the first real solve, never while importing this
# module.  FunctionType makes every model callback enter the generic kernels
# through one native signature rather than specializing the whole solver for
# each CPUDispatcher object.
_F64_1C = nb_types.Array(nb_types.float64, 1, "C")
_F64_2C = nb_types.Array(nb_types.float64, 2, "C")
_F64_3C = nb_types.Array(nb_types.float64, 3, "C")
_BOOL_1C = nb_types.Array(nb_types.boolean, 1, "C")
_BOOL_2C = nb_types.Array(nb_types.boolean, 2, "C")
_I32_1C = nb_types.Array(nb_types.int32, 1, "C")
_I32_2C = nb_types.Array(nb_types.int32, 2, "C")
_I64_1C = nb_types.Array(nb_types.int64, 1, "C")

_PREPARE_CALLBACK_SIGNATURE = nb_types.int64(
    _F64_2C,
    _F64_1C,
    _BOOL_1C,
    _F64_2C,
    _F64_1C,
    _F64_1C,
    _F64_2C,
)
_PREPARE_FUNCTION_TYPE = nb_types.FunctionType(_PREPARE_CALLBACK_SIGNATURE)

_OBJECTIVE_RETURN = nb_types.Tuple(
    (nb_types.float64, nb_types.float64, nb_types.boolean)
)
_OBJECTIVE_CALLBACK_SIGNATURE = _OBJECTIVE_RETURN(
    _F64_2C,
    _F64_1C,
    _BOOL_1C,
    _F64_1C,
    _I64_1C,
    _F64_1C,
    nb_types.boolean,
    nb_types.boolean,
    nb_types.int64,
    _F64_1C,
    _F64_2C,
    _F64_1C,
    nb_types.boolean,
)
_OBJECTIVE_FUNCTION_TYPE = nb_types.FunctionType(_OBJECTIVE_CALLBACK_SIGNATURE)

_VALUE_JACOBIAN_RETURN = nb_types.Tuple((_F64_1C, _F64_2C))
_VALUE_JACOBIAN_CALLBACK_SIGNATURE = _VALUE_JACOBIAN_RETURN(_F64_2C, _F64_1C)
_VALUE_JACOBIAN_FUNCTION_TYPE = nb_types.FunctionType(
    _VALUE_JACOBIAN_CALLBACK_SIGNATURE
)

_PREPARE_KERNEL_SIGNATURE = nb_types.void(
    _PREPARE_FUNCTION_TYPE,
    _F64_3C,
    _F64_2C,
    _BOOL_2C,
    _F64_3C,
    _F64_2C,
    _F64_2C,
    _F64_2C,
    _F64_2C,
    _BOOL_2C,
    _F64_2C,
    _BOOL_1C,
    _F64_3C,
    _BOOL_1C,
    _BOOL_1C,
    nb_types.int64,
    _F64_3C,
    _F64_2C,
    _F64_2C,
    _I32_1C,
    _I32_1C,
)

_SOLVE_KERNEL_SIGNATURE = nb_types.void(
    _OBJECTIVE_FUNCTION_TYPE,
    _F64_3C,
    _F64_2C,
    _BOOL_2C,
    _F64_3C,
    _I32_1C,
    _F64_2C,
    _F64_2C,
    _I64_1C,
    _F64_2C,
    nb_types.boolean,
    nb_types.boolean,
    nb_types.int64,
    nb_types.int64,
    nb_types.float64,
    nb_types.float64,
    nb_types.float64,
    _BOOL_1C,
    _I32_1C,
    _F64_2C,
    _F64_1C,
    _F64_1C,
    _I32_1C,
    _I32_1C,
    _I32_1C,
    _I32_1C,
    _I32_1C,
    _I32_2C,
    _I32_2C,
    _I32_2C,
    _I32_2C,
)

_FINALIZE_KERNEL_SIGNATURE = nb_types.void(
    _VALUE_JACOBIAN_FUNCTION_TYPE,
    _F64_3C,
    _F64_2C,
    _BOOL_2C,
    _F64_2C,
    _I64_1C,
    _F64_2C,
    nb_types.boolean,
    nb_types.boolean,
    nb_types.int64,
    _F64_2C,
    _F64_2C,
    _F64_3C,
    _F64_2C,
    _F64_1C,
    _BOOL_1C,
)

_ABI_LOCK = threading.RLock()
_SERIAL_ABI_READY = False
_PARALLEL_ABI_READY = False


@dataclass(frozen=True, slots=True)
class CompiledFitDescriptor:
    """Callbacks and stable preparation identity for one compiled model.

    ``coordinate_origin`` names an independent-coordinate axis whose minimum
    is subtracted before callbacks run.  This is the compiled equivalent of an
    anchored decay model.  A caller that has already supplied relative
    coordinates sets ``coordinates_are_canonical=True`` at the solve entry.

    ``context_builder`` is the only Python-level owner of coordinate-dependent
    preparation data.  It receives the canonical tuple of coordinate arrays;
    callers may cache its read-only result by ``cache_key`` plus their exact
    coordinate fingerprint.  The solver itself has no model registry.
    """

    prepare: Any
    objective: Any
    value_jacobian: Any
    context_builder: Callable[[tuple[np.ndarray, ...]], np.ndarray]
    max_candidates: int
    coordinate_origin: int | None = None
    cache_key: str = ""

    def __post_init__(self) -> None:
        if not callable(self.prepare):
            raise TypeError("compiled fit prepare callback must be callable")
        if not callable(self.objective):
            raise TypeError("compiled fit objective callback must be callable")
        if not callable(self.value_jacobian):
            raise TypeError("compiled fit value_jacobian callback must be callable")
        if not callable(self.context_builder):
            raise TypeError("compiled fit context_builder must be callable")
        count = int(self.max_candidates)
        if count <= 0:
            raise ValueError("compiled fit max_candidates must be positive")
        origin = self.coordinate_origin
        if origin is not None and int(origin) < 0:
            raise ValueError("compiled fit coordinate_origin must be non-negative")
        object.__setattr__(self, "max_candidates", count)
        object.__setattr__(self, "coordinate_origin", None if origin is None else int(origin))
        object.__setattr__(self, "cache_key", str(self.cache_key))


@dataclass(frozen=True, slots=True)
class CompiledFitOutput:
    """Complete numeric result of one serial or independent batched solve."""

    parameters: np.ndarray
    standard_errors: np.ndarray
    covariance: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    reduced_chi_square: np.ndarray
    covariance_valid: np.ndarray
    success: np.ndarray
    status: np.ndarray
    cost: np.ndarray
    raw_rss: np.ndarray
    nfev: np.ndarray
    njev: np.ndarray
    iterations: np.ndarray
    winner_seed: np.ndarray
    lane_status: np.ndarray
    lane_nfev: np.ndarray
    lane_njev: np.ndarray
    lane_iterations: np.ndarray
    coordinate_origins: np.ndarray


def termination_message(status: int) -> str:
    """Return the stable public explanation for a compiled termination code."""

    return {
        STATUS_INVALID: "compiled fit encountered non-finite numerical state",
        STATUS_NO_CANDIDATE: "compiled fit has no finite initializer",
        STATUS_MAX_NFEV: "The maximum number of function evaluations is exceeded.",
        STATUS_GTOL: "`gtol` termination condition is satisfied.",
        STATUS_FTOL: "`ftol` termination condition is satisfied.",
        STATUS_XTOL: "`xtol` termination condition is satisfied.",
        STATUS_FTOL_XTOL: (
            "Both `ftol` and `xtol` termination conditions are satisfied."
        ),
    }.get(int(status), f"compiled fit terminated with status {int(status)}")


@njit(cache=True, inline="always")
def _rho(z: float, loss_code: int) -> tuple[float, float, float]:
    if loss_code == LOSS_LINEAR:
        return z, 1.0, 0.0
    if loss_code == LOSS_SOFT_L1:
        total = 1.0 + z
        root = math.sqrt(total)
        return 2.0 * (root - 1.0), 1.0 / root, -0.5 / (total * root)
    if loss_code == LOSS_HUBER:
        if z <= 1.0:
            return z, 1.0, 0.0
        root = math.sqrt(z)
        return 2.0 * root - 1.0, 1.0 / root, -0.5 / (z * root)
    if loss_code == LOSS_CAUCHY:
        denominator = 1.0 + z
        return math.log1p(z), 1.0 / denominator, -1.0 / (denominator * denominator)
    denominator = 1.0 + z * z
    return math.atan(z), 1.0 / denominator, -2.0 * z / (denominator * denominator)


@njit(cache=True, inline="always")
def compiled_point_terms(
    predicted: float,
    observed: float,
    poisson: bool,
    weight: float,
    use_weights: bool,
    loss_code: int,
) -> tuple[float, float, float, float, float, bool]:
    """Residual, objective and derivative factors for descriptor callbacks."""

    if poisson:
        expected = max(predicted, COUNT_FLOOR)
        logarithm = observed * math.log(observed / expected) if observed > 0.0 else 0.0
        deviance = 2.0 * max(expected - observed + logarithm, 0.0)
        raw = math.sqrt(deviance)
        if expected < observed:
            raw = -raw
        absolute = abs(raw)
        if absolute > COUNT_FLOOR:
            residual_jacobian_scale = abs(expected - observed) / (
                expected * max(absolute, COUNT_FLOOR)
            )
        else:
            residual_jacobian_scale = 1.0 / math.sqrt(expected)
    else:
        raw = predicted - observed
        residual_jacobian_scale = weight if use_weights else 1.0
        if use_weights:
            raw *= weight
    if not math.isfinite(raw):
        return raw, math.inf, math.inf, 0.0, 0.0, False
    squared = raw * raw
    rho0, rho1, rho2 = _rho(squared, loss_code)
    gradient_factor = raw * rho1 * residual_jacobian_scale
    information_factor = residual_jacobian_scale * residual_jacobian_scale * max(
        rho1 + 2.0 * rho2 * squared,
        EPSILON,
    )
    return (
        raw,
        squared,
        0.5 * rho0,
        gradient_factor,
        information_factor,
        True,
    )


@njit(cache=True, inline="always")
def compiled_reset_accumulators(
    gradient: np.ndarray,
    information: np.ndarray,
) -> None:
    for row in range(gradient.size):
        gradient[row] = 0.0
        for column in range(gradient.size):
            information[row, column] = 0.0


@njit(cache=True, inline="always")
def compiled_accumulate(
    jacobian_row: np.ndarray,
    gradient: np.ndarray,
    information: np.ndarray,
    gradient_factor: float,
    information_factor: float,
) -> None:
    for row in range(jacobian_row.size):
        value = jacobian_row[row]
        gradient[row] += value * gradient_factor
        for column in range(row + 1):
            information[row, column] += (
                value * jacobian_row[column] * information_factor
            )


@njit(cache=True, inline="always")
def compiled_finish_information(information: np.ndarray) -> None:
    for row in range(information.shape[0]):
        for column in range(row):
            information[column, row] = information[row, column]


@njit(cache=True, inline="always")
def _vector_norm(values: np.ndarray) -> float:
    total = 0.0
    for index in range(values.size):
        total += values[index] * values[index]
    return math.sqrt(max(total, 0.0))


@njit(cache=True, inline="always")
def _quadratic(information: np.ndarray, gradient: np.ndarray, step: np.ndarray) -> float:
    quadratic = 0.0
    linear = 0.0
    for row in range(step.size):
        linear += gradient[row] * step[row]
        product = 0.0
        for column in range(step.size):
            product += information[row, column] * step[column]
        quadratic += step[row] * product
    return 0.5 * quadratic + linear


@njit(cache=True, inline="always")
def _quadratic_line(
    information: np.ndarray,
    gradient: np.ndarray,
    direction: np.ndarray,
    origin: np.ndarray,
    has_origin: bool,
) -> tuple[float, float, float]:
    product = np.empty(direction.size, dtype=np.float64)
    second = 0.0
    first = 0.0
    for row in range(direction.size):
        value = 0.0
        for column in range(direction.size):
            value += information[row, column] * direction[column]
        product[row] = value
        second += direction[row] * value
        first += gradient[row] * direction[row]
    constant = 0.0
    if has_origin:
        for index in range(direction.size):
            first += origin[index] * product[index]
        constant = _quadratic(information, gradient, origin)
    return 0.5 * second, first, constant


@njit(cache=True, inline="always")
def _minimize_quadratic(
    second: float,
    first: float,
    lower: float,
    upper: float,
    constant: float,
) -> tuple[float, float]:
    best_step = lower
    best_value = lower * (second * lower + first) + constant
    value = upper * (second * upper + first) + constant
    if value < best_value:
        best_step = upper
        best_value = value
    if second != 0.0:
        candidate = -0.5 * first / second
        if lower < candidate < upper:
            value = candidate * (second * candidate + first) + constant
            if value < best_value:
                best_step = candidate
                best_value = value
    return best_step, best_value


@njit(cache=True, inline="always")
def _step_to_bound(
    values: np.ndarray,
    step: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    hits: np.ndarray,
) -> float:
    minimum = math.inf
    for index in range(values.size):
        hits[index] = 0
        if step[index] == 0.0:
            continue
        low = (lower[index] - values[index]) / step[index]
        high = (upper[index] - values[index]) / step[index]
        forward = max(low, high)
        if forward < minimum:
            minimum = forward
    for index in range(values.size):
        if step[index] == 0.0:
            continue
        low = (lower[index] - values[index]) / step[index]
        high = (upper[index] - values[index]) / step[index]
        if max(low, high) == minimum:
            hits[index] = 1 if step[index] > 0.0 else -1
    return minimum


@njit(cache=True, inline="always")
def _positive_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    radius: float,
) -> float:
    quadratic = 0.0
    linear = 0.0
    constant = -radius * radius
    for index in range(origin.size):
        quadratic += direction[index] * direction[index]
        linear += origin[index] * direction[index]
        constant += origin[index] * origin[index]
    discriminant = max(linear * linear - quadratic * constant, 0.0)
    root = math.sqrt(discriminant)
    qvalue = -(linear + math.copysign(root, linear))
    if qvalue == 0.0:
        return -linear / quadratic
    first = qvalue / quadratic
    second = constant / qvalue
    return min(first, second)


@njit(cache=True, inline="always")
def _inside_bounds(
    values: np.ndarray,
    step: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    for index in range(values.size):
        candidate = values[index] + step[index]
        if candidate < lower[index] or candidate > upper[index]:
            return False
    return True


@njit(cache=True, inline="always")
def _select_reflective_step(
    values: np.ndarray,
    information: np.ndarray,
    gradient: np.ndarray,
    unconstrained: np.ndarray,
    scaling: np.ndarray,
    radius: float,
    lower: np.ndarray,
    upper: np.ndarray,
    theta: float,
    step: np.ndarray,
    scaled_step: np.ndarray,
) -> float:
    count = values.size
    direct = np.empty(count, dtype=np.float64)
    for index in range(count):
        direct[index] = scaling[index] * unconstrained[index]
    if _inside_bounds(values, direct, lower, upper):
        for index in range(count):
            step[index] = direct[index]
            scaled_step[index] = unconstrained[index]
        return -_quadratic(information, gradient, unconstrained)

    hits = np.empty(count, dtype=np.int32)
    direct_stride = _step_to_bound(values, direct, lower, upper, hits)
    reflected_scaled = np.empty(count, dtype=np.float64)
    reflected = np.empty(count, dtype=np.float64)
    at_bound = np.empty(count, dtype=np.float64)
    for index in range(count):
        reflected_scaled[index] = (
            -unconstrained[index] if hits[index] != 0 else unconstrained[index]
        )
        direct[index] *= direct_stride
        unconstrained[index] *= direct_stride
        at_bound[index] = values[index] + direct[index]
        reflected[index] = scaling[index] * reflected_scaled[index]

    to_radius = _positive_intersection(unconstrained, reflected_scaled, radius)
    dummy_hits = np.empty(count, dtype=np.int32)
    to_bound = _step_to_bound(at_bound, reflected, lower, upper, dummy_hits)
    reflected_stride = min(to_bound, to_radius)
    if reflected_stride > 0.0:
        reflected_lower = (1.0 - theta) * direct_stride / reflected_stride
        reflected_upper = theta * to_bound if reflected_stride == to_bound else to_radius
    else:
        reflected_lower, reflected_upper = 0.0, -1.0
    if reflected_lower <= reflected_upper:
        second, first, constant = _quadratic_line(
            information,
            gradient,
            reflected_scaled,
            unconstrained,
            True,
        )
        stride, reflected_value = _minimize_quadratic(
            second,
            first,
            reflected_lower,
            reflected_upper,
            constant,
        )
        for index in range(count):
            reflected_scaled[index] = unconstrained[index] + stride * reflected_scaled[index]
            reflected[index] = scaling[index] * reflected_scaled[index]
    else:
        reflected_value = math.inf

    for index in range(count):
        direct[index] *= theta
        unconstrained[index] *= theta
    direct_value = _quadratic(information, gradient, unconstrained)

    anti_gradient_scaled = np.empty(count, dtype=np.float64)
    anti_gradient = np.empty(count, dtype=np.float64)
    norm_squared = 0.0
    for index in range(count):
        anti_gradient_scaled[index] = -gradient[index]
        anti_gradient[index] = scaling[index] * anti_gradient_scaled[index]
        norm_squared += anti_gradient_scaled[index] * anti_gradient_scaled[index]
    if norm_squared == 0.0:
        anti_gradient_stride = 0.0
        anti_gradient_value = 0.0
    else:
        to_radius = radius / math.sqrt(norm_squared)
        to_bound = _step_to_bound(
            values,
            anti_gradient,
            lower,
            upper,
            dummy_hits,
        )
        maximum = theta * to_bound if to_bound < to_radius else to_radius
        second, first, _constant = _quadratic_line(
            information,
            gradient,
            anti_gradient_scaled,
            anti_gradient_scaled,
            False,
        )
        anti_gradient_stride, anti_gradient_value = _minimize_quadratic(
            second,
            first,
            0.0,
            maximum,
            0.0,
        )
    for index in range(count):
        anti_gradient_scaled[index] *= anti_gradient_stride
        anti_gradient[index] *= anti_gradient_stride

    if direct_value < reflected_value and direct_value < anti_gradient_value:
        for index in range(count):
            step[index] = direct[index]
            scaled_step[index] = unconstrained[index]
        return -direct_value
    if reflected_value < direct_value and reflected_value < anti_gradient_value:
        for index in range(count):
            step[index] = reflected[index]
            scaled_step[index] = reflected_scaled[index]
        return -reflected_value
    for index in range(count):
        step[index] = anti_gradient[index]
        scaled_step[index] = anti_gradient_scaled[index]
    return -anti_gradient_value


@njit(cache=True, inline="always")
def _make_feasible(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    relative_step: float,
) -> None:
    for index in range(values.size):
        lower_distance = values[index] - lower[index]
        upper_distance = upper[index] - values[index]
        lower_active = math.isfinite(lower[index]) and lower_distance <= min(
            upper_distance,
            relative_step * max(1.0, abs(lower[index])),
        )
        upper_active = math.isfinite(upper[index]) and upper_distance <= min(
            lower_distance,
            relative_step * max(1.0, abs(upper[index])),
        )
        if lower_active:
            values[index] = (
                np.nextafter(lower[index], upper[index])
                if relative_step == 0.0
                else lower[index] + relative_step * max(1.0, abs(lower[index]))
            )
        elif upper_active:
            values[index] = (
                np.nextafter(upper[index], lower[index])
                if relative_step == 0.0
                else upper[index] - relative_step * max(1.0, abs(upper[index]))
            )
        if values[index] < lower[index] or values[index] > upper[index]:
            values[index] = 0.5 * (lower[index] + upper[index])


@njit(cache=True, inline="always")
def _cholesky_inside(
    information: np.ndarray,
    gradient: np.ndarray,
    radius: float,
    lower_factor: np.ndarray,
    intermediate: np.ndarray,
    result: np.ndarray,
) -> bool:
    count = gradient.size
    maximum_diagonal = 0.0
    for index in range(count):
        maximum_diagonal = max(maximum_diagonal, abs(information[index, index]))
    threshold = EPSILON * count * max(1.0, maximum_diagonal)
    for row in range(count):
        for column in range(row + 1):
            value = information[row, column]
            for prior in range(column):
                value -= lower_factor[row, prior] * lower_factor[column, prior]
            if row == column:
                if not math.isfinite(value) or value <= threshold:
                    return False
                lower_factor[row, column] = math.sqrt(value)
            else:
                lower_factor[row, column] = value / lower_factor[column, column]
        for column in range(row + 1, count):
            lower_factor[row, column] = 0.0
    for row in range(count):
        value = -gradient[row]
        for prior in range(row):
            value -= lower_factor[row, prior] * intermediate[prior]
        intermediate[row] = value / lower_factor[row, row]
    norm_squared = 0.0
    for reverse in range(count):
        row = count - 1 - reverse
        value = intermediate[row]
        for following in range(row + 1, count):
            value -= lower_factor[following, row] * result[following]
        result[row] = value / lower_factor[row, row]
        norm_squared += result[row] * result[row]
    return math.sqrt(max(norm_squared, 0.0)) <= radius


@njit(cache=True, inline="always")
def _trust_eigh(
    observations: int,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    projected_gradient: np.ndarray,
    radius: float,
    alpha: float,
    output: np.ndarray,
) -> float:
    count = eigenvalues.size
    full_rank = False
    if observations >= count and eigenvalues[count - 1] > 0.0:
        threshold = EPSILON * observations * math.sqrt(eigenvalues[count - 1])
        full_rank = math.sqrt(max(eigenvalues[0], 0.0)) > threshold
    if full_rank:
        norm_squared = 0.0
        for row in range(count):
            value = 0.0
            for column in range(count):
                value += eigenvectors[row, column] * (
                    projected_gradient[column] / eigenvalues[column]
                )
            output[row] = -value
            norm_squared += value * value
        if math.sqrt(norm_squared) <= radius:
            return 0.0

    gradient_squared = 0.0
    for index in range(count):
        gradient_squared += projected_gradient[index] * projected_gradient[index]
    alpha_upper = math.sqrt(gradient_squared) / radius
    alpha_lower = 0.0
    if full_rank:
        norm_squared = 0.0
        derivative_numerator = 0.0
        for index in range(count):
            value = projected_gradient[index] / eigenvalues[index]
            norm_squared += value * value
            derivative_numerator += (
                projected_gradient[index]
                * projected_gradient[index]
                / (eigenvalues[index] ** 3)
            )
        norm = math.sqrt(norm_squared)
        phi = norm - radius
        derivative = -derivative_numerator / norm
        alpha_lower = -phi / derivative
    if (
        (not full_rank and alpha == 0.0)
        or alpha < alpha_lower
        or alpha > alpha_upper
    ):
        alpha = max(
            0.001 * alpha_upper,
            math.sqrt(max(alpha_lower * alpha_upper, 0.0)),
        )
    for _iteration in range(10):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(
                0.001 * alpha_upper,
                math.sqrt(max(alpha_lower * alpha_upper, 0.0)),
            )
        norm_squared = 0.0
        derivative_numerator = 0.0
        for index in range(count):
            denominator = eigenvalues[index] + alpha
            value = projected_gradient[index] / denominator
            norm_squared += value * value
            derivative_numerator += (
                projected_gradient[index]
                * projected_gradient[index]
                / (denominator * denominator * denominator)
            )
        norm = math.sqrt(norm_squared)
        phi = norm - radius
        derivative = -derivative_numerator / norm
        if phi < 0.0:
            alpha_upper = alpha
        ratio = phi / derivative
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + radius) * ratio / radius
        if abs(phi) < 0.01 * radius:
            break
    norm_squared = 0.0
    for row in range(count):
        value = 0.0
        for column in range(count):
            value += eigenvectors[row, column] * (
                projected_gradient[column] / (eigenvalues[column] + alpha)
            )
        output[row] = -value
        norm_squared += value * value
    if norm_squared > 0.0:
        scale = radius / math.sqrt(norm_squared)
        for index in range(count):
            output[index] *= scale
    return alpha


@njit(cache=True, inline="always")
def _update_radius(
    radius: float,
    actual: float,
    predicted: float,
    step_norm: float,
) -> tuple[float, float]:
    ratio = (
        actual / predicted
        if predicted > 0.0
        else 1.0 if predicted == 0.0 and actual == 0.0 else 0.0
    )
    if ratio < 0.25:
        return 0.25 * step_norm, ratio
    if ratio > 0.75 and step_norm > 0.95 * radius:
        return 2.0 * radius, ratio
    return radius, ratio


@njit(cache=True, inline="always")
def _termination(
    actual: float,
    cost: float,
    step_norm: float,
    value_norm: float,
    ratio: float,
    ftol: float,
    xtol: float,
) -> int:
    function_done = actual < ftol * cost and ratio > 0.25
    parameter_done = step_norm < xtol * (xtol + value_norm)
    if function_done and parameter_done:
        return STATUS_FTOL_XTOL
    if function_done:
        return STATUS_FTOL
    if parameter_done:
        return STATUS_XTOL
    return STATUS_MAX_NFEV


@njit(cache=True, inline="always")
def _seed_finite(seed: np.ndarray) -> bool:
    for index in range(seed.size):
        if not math.isfinite(seed[index]):
            return False
    return True


@njit(cache=True, inline="always")
def _clip_full_seed(
    seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    free_mask: np.ndarray,
) -> bool:
    for index in range(seed.size):
        if not math.isfinite(seed[index]):
            return False
        if free_mask[index]:
            low_inside = (
                np.nextafter(lower[index], upper[index])
                if math.isfinite(lower[index])
                else lower[index]
            )
            high_inside = (
                np.nextafter(upper[index], lower[index])
                if math.isfinite(upper[index])
                else upper[index]
            )
            seed[index] = min(max(seed[index], low_inside), high_inside)
        else:
            seed[index] = lower[index]
    return _seed_finite(seed)


@njit(cache=True, inline="always")
def _same_seed(left: np.ndarray, right: np.ndarray) -> bool:
    for index in range(left.size):
        if left[index] != right[index]:
            return False
    return True


@njit(cache=True, inline="always")
def _append_seed(
    source: np.ndarray,
    seeds: np.ndarray,
    count: int,
    lower: np.ndarray,
    upper: np.ndarray,
    free_mask: np.ndarray,
) -> int:
    candidate = np.empty(source.size, dtype=np.float64)
    for index in range(source.size):
        candidate[index] = source[index]
    if not _clip_full_seed(candidate, lower, upper, free_mask):
        return count
    for prior in range(count):
        if _same_seed(candidate, seeds[prior]):
            return count
    if count >= seeds.shape[0]:
        return count
    for index in range(source.size):
        seeds[count, index] = candidate[index]
    return count + 1


@njit(cache=True)
def _prepare_one(
    prepare_callback: Any,
    coordinates: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    context: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
    requested_lower: np.ndarray,
    requested_upper: np.ndarray,
    requested_mask: np.ndarray,
    warm_seed: np.ndarray,
    use_warm: bool,
    authored_seeds: np.ndarray,
    use_authored: bool,
    free_mask: np.ndarray,
    max_cold_candidates: int,
    output_seeds: np.ndarray,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
) -> tuple[int, int]:
    parameter_count = base_lower.size
    for index in range(parameter_count):
        output_lower[index] = base_lower[index]
        output_upper[index] = base_upper[index]
    cold = np.full(
        (max_cold_candidates, parameter_count),
        np.nan,
        dtype=np.float64,
    )
    cold_count = prepare_callback(
        coordinates,
        observations,
        valid,
        cold,
        output_lower,
        output_upper,
        context,
    )
    if cold_count < 0 or cold_count > cold.shape[0]:
        return 0, STATUS_INVALID

    # Explicit authored bounds have the same authority as the scalar path:
    # they replace the model-derived bound for that parameter rather than
    # intersecting it.  Exact equal endpoints are the fixed-parameter marker.
    for index in range(parameter_count):
        if requested_mask[index]:
            output_lower[index] = requested_lower[index]
            output_upper[index] = requested_upper[index]
        if not math.isfinite(output_lower[index]) and not math.isinf(output_lower[index]):
            return 0, STATUS_INVALID
        if not math.isfinite(output_upper[index]) and not math.isinf(output_upper[index]):
            return 0, STATUS_INVALID
        if free_mask[index]:
            if not output_lower[index] < output_upper[index]:
                return 0, STATUS_INVALID
        elif output_lower[index] != output_upper[index]:
            return 0, STATUS_INVALID

    for seed_index in range(output_seeds.shape[0]):
        for parameter in range(parameter_count):
            output_seeds[seed_index, parameter] = np.nan

    output_count = 0
    if use_warm:
        output_count = _append_seed(
            warm_seed,
            output_seeds,
            output_count,
            output_lower,
            output_upper,
            free_mask,
        )
        # Invalid warm values are rejected by the Python entry before this
        # kernel.  Failure to append here can only mean a violated bound shape.
        if output_count == 0:
            return 0, STATUS_INVALID

    if use_authored:
        for seed_index in range(authored_seeds.shape[0]):
            output_count = _append_seed(
                authored_seeds[seed_index],
                output_seeds,
                output_count,
                output_lower,
                output_upper,
                free_mask,
            )
    else:
        for seed_index in range(cold_count):
            output_count = _append_seed(
                cold[seed_index],
                output_seeds,
                output_count,
                output_lower,
                output_upper,
                free_mask,
            )
    if output_count == 0:
        return 0, STATUS_NO_CANDIDATE
    return output_count, STATUS_MAX_NFEV


@njit(cache=True, nogil=True)
def _prepare_serial(
    prepare_callback: Any,
    coordinates: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    contexts: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
    requested_lower: np.ndarray,
    requested_upper: np.ndarray,
    requested_mask: np.ndarray,
    warm_seeds: np.ndarray,
    use_warm: np.ndarray,
    authored_seeds: np.ndarray,
    use_authored: np.ndarray,
    free_mask: np.ndarray,
    max_cold_candidates: int,
    output_seeds: np.ndarray,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    counts: np.ndarray,
    statuses: np.ndarray,
) -> None:
    for cell in range(observations.shape[0]):
        counts[cell], statuses[cell] = _prepare_one(
            prepare_callback,
            coordinates[0],
            observations[cell],
            valid[cell],
            contexts[cell],
            base_lower[cell],
            base_upper[cell],
            requested_lower[cell],
            requested_upper[cell],
            requested_mask[cell],
            warm_seeds[cell],
            use_warm[cell],
            authored_seeds[cell],
            use_authored[cell],
            free_mask,
            max_cold_candidates,
            output_seeds[cell],
            output_lower[cell],
            output_upper[cell],
        )


@njit(cache=True, nogil=True, parallel=True)
def _prepare_parallel(
    prepare_callback: Any,
    coordinates: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    contexts: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
    requested_lower: np.ndarray,
    requested_upper: np.ndarray,
    requested_mask: np.ndarray,
    warm_seeds: np.ndarray,
    use_warm: np.ndarray,
    authored_seeds: np.ndarray,
    use_authored: np.ndarray,
    free_mask: np.ndarray,
    max_cold_candidates: int,
    output_seeds: np.ndarray,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    counts: np.ndarray,
    statuses: np.ndarray,
) -> None:
    for cell in prange(observations.shape[0]):
        counts[cell], statuses[cell] = _prepare_one(
            prepare_callback,
            coordinates[0],
            observations[cell],
            valid[cell],
            contexts[cell],
            base_lower[cell],
            base_upper[cell],
            requested_lower[cell],
            requested_upper[cell],
            requested_mask[cell],
            warm_seeds[cell],
            use_warm[cell],
            authored_seeds[cell],
            use_authored[cell],
            free_mask,
            max_cold_candidates,
            output_seeds[cell],
            output_lower[cell],
            output_upper[cell],
        )


@njit(cache=True)
def _solve_seed(
    objective_callback: Any,
    coordinates: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    full_template: np.ndarray,
    free_indices: np.ndarray,
    seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: np.ndarray,
    use_weights: bool,
    poisson: bool,
    loss_code: int,
    max_nfev: int,
    ftol: float,
    xtol: float,
    gtol: float,
) -> tuple[np.ndarray, float, float, int, int, int, int]:
    free_count = seed.size
    values = seed.copy()
    for index in range(free_count):
        if not math.isfinite(values[index]):
            return values, math.inf, math.inf, STATUS_INVALID, 0, 0, 0
        low_inside = (
            np.nextafter(lower[index], upper[index])
            if math.isfinite(lower[index])
            else lower[index]
        )
        high_inside = (
            np.nextafter(upper[index], lower[index])
            if math.isfinite(upper[index])
            else upper[index]
        )
        values[index] = min(max(values[index], low_inside), high_inside)
    _make_feasible(values, lower, upper, 1.0e-10)

    selected = 0
    for index in range(observations.size):
        if valid[index]:
            selected += 1
    if selected <= free_count:
        return values, math.inf, math.inf, STATUS_INVALID, 0, 0, 0

    full_parameters = full_template.copy()
    gradient = np.empty(free_count, dtype=np.float64)
    information = np.empty((free_count, free_count), dtype=np.float64)
    jacobian_row = np.empty(free_count, dtype=np.float64)
    scale_inverse = np.zeros(free_count, dtype=np.float64)
    coleman_li = np.empty(free_count, dtype=np.float64)
    coleman_li_derivative = np.empty(free_count, dtype=np.float64)
    scaling = np.empty(free_count, dtype=np.float64)
    scaled_gradient = np.empty(free_count, dtype=np.float64)
    trust_information = np.empty((free_count, free_count), dtype=np.float64)
    diagonal = np.empty(free_count, dtype=np.float64)
    unconstrained = np.empty(free_count, dtype=np.float64)
    step = np.empty(free_count, dtype=np.float64)
    scaled_step = np.empty(free_count, dtype=np.float64)
    trial = np.empty(free_count, dtype=np.float64)
    projected_gradient = np.empty(free_count, dtype=np.float64)
    lower_factor = np.empty((free_count, free_count), dtype=np.float64)
    intermediate = np.empty(free_count, dtype=np.float64)

    initialized = False
    radius = 1.0
    alpha = 0.0
    nfev = 0
    njev = 0
    iteration = 0
    status = STATUS_MAX_NFEV
    cost = math.inf
    raw_rss = math.inf

    while True:
        for index in range(free_count):
            full_parameters[free_indices[index]] = values[index]
        cost, raw_rss, finite = objective_callback(
            coordinates,
            observations,
            valid,
            full_parameters,
            free_indices,
            weights,
            use_weights,
            poisson,
            loss_code,
            gradient,
            information,
            jacobian_row,
            True,
        )
        if iteration == 0:
            nfev = 1
        njev += 1
        if not finite or not math.isfinite(cost) or not math.isfinite(raw_rss):
            status = STATUS_INVALID
            break

        gradient_norm = 0.0
        for index in range(free_count):
            norm = math.sqrt(max(information[index, index], 0.0))
            scale_inverse[index] = (
                max(norm, scale_inverse[index])
                if initialized
                else 1.0 if norm == 0.0 else norm
            )
            value = gradient[index]
            if value < 0.0 and math.isfinite(upper[index]):
                coleman_li[index] = upper[index] - values[index]
                coleman_li_derivative[index] = -1.0
            elif value > 0.0 and math.isfinite(lower[index]):
                coleman_li[index] = values[index] - lower[index]
                coleman_li_derivative[index] = 1.0
            else:
                coleman_li[index] = 1.0
                coleman_li_derivative[index] = 0.0
            gradient_norm = max(
                gradient_norm,
                abs(value * coleman_li[index]),
            )
        initialized = True
        if gradient_norm < gtol:
            status = STATUS_GTOL
            break
        if nfev >= max_nfev:
            status = STATUS_MAX_NFEV
            break

        for index in range(free_count):
            if coleman_li_derivative[index] != 0.0:
                coleman_li[index] *= scale_inverse[index]
            inverse = 1.0 / scale_inverse[index]
            scaling[index] = math.sqrt(coleman_li[index]) * inverse
            scaled_gradient[index] = scaling[index] * gradient[index]
            diagonal[index] = max(
                gradient[index]
                * coleman_li_derivative[index]
                * inverse,
                0.0,
            )
        for row in range(free_count):
            for column in range(free_count):
                trust_information[row, column] = (
                    scaling[row]
                    * information[row, column]
                    * scaling[column]
                )
            trust_information[row, row] += diagonal[row]

        if iteration == 0:
            norm_squared = 0.0
            for index in range(free_count):
                value = (
                    values[index]
                    * scale_inverse[index]
                    / math.sqrt(coleman_li[index])
                )
                norm_squared += value * value
            radius = math.sqrt(norm_squared)
            if radius == 0.0:
                radius = 1.0

        cholesky_ready = _cholesky_inside(
            trust_information,
            scaled_gradient,
            radius,
            lower_factor,
            intermediate,
            unconstrained,
        )
        eigenvalues = np.empty(free_count, dtype=np.float64)
        eigenvectors = np.empty((free_count, free_count), dtype=np.float64)
        eigen_ready = False
        if not cholesky_ready:
            eigenvalues, eigenvectors = np.linalg.eigh(trust_information)
            eigen_ready = True
            for index in range(free_count):
                if (
                    eigenvalues[index] < 0.0
                    and eigenvalues[index]
                    > -1.0e-12 * max(1.0, eigenvalues[free_count - 1])
                ):
                    eigenvalues[index] = 0.0
            for column in range(free_count):
                value = 0.0
                for row in range(free_count):
                    value += eigenvectors[row, column] * scaled_gradient[row]
                projected_gradient[column] = value
        else:
            alpha = 0.0

        theta = max(0.995, 1.0 - gradient_norm)
        accepted = False
        terminal = STATUS_MAX_NFEV
        first = True
        while nfev < max_nfev:
            if not (first and cholesky_ready):
                if not eigen_ready:
                    eigenvalues, eigenvectors = np.linalg.eigh(trust_information)
                    eigen_ready = True
                    for index in range(free_count):
                        if (
                            eigenvalues[index] < 0.0
                            and eigenvalues[index]
                            > -1.0e-12 * max(1.0, eigenvalues[free_count - 1])
                        ):
                            eigenvalues[index] = 0.0
                    for column in range(free_count):
                        value = 0.0
                        for row in range(free_count):
                            value += (
                                eigenvectors[row, column]
                                * scaled_gradient[row]
                            )
                        projected_gradient[column] = value
                alpha = _trust_eigh(
                    selected,
                    eigenvalues,
                    eigenvectors,
                    projected_gradient,
                    radius,
                    alpha,
                    unconstrained,
                )
            first = False
            predicted = _select_reflective_step(
                values,
                trust_information,
                scaled_gradient,
                unconstrained,
                scaling,
                radius,
                lower,
                upper,
                theta,
                step,
                scaled_step,
            )
            for index in range(free_count):
                trial[index] = values[index] + step[index]
            _make_feasible(trial, lower, upper, 0.0)
            for index in range(free_count):
                full_parameters[free_indices[index]] = trial[index]
            trial_cost, trial_rss, trial_finite = objective_callback(
                coordinates,
                observations,
                valid,
                full_parameters,
                free_indices,
                weights,
                use_weights,
                poisson,
                loss_code,
                gradient,
                information,
                jacobian_row,
                False,
            )
            nfev += 1
            scaled_norm = _vector_norm(scaled_step)
            if not trial_finite or not math.isfinite(trial_cost):
                radius = 0.25 * scaled_norm
                continue
            actual = cost - trial_cost
            old_radius = radius
            radius, ratio = _update_radius(
                old_radius,
                actual,
                predicted,
                scaled_norm,
            )
            terminal = _termination(
                actual,
                cost,
                _vector_norm(step),
                _vector_norm(values),
                ratio,
                ftol,
                xtol,
            )
            if radius != 0.0:
                alpha *= old_radius / radius
            if actual > 0.0:
                for index in range(free_count):
                    values[index] = trial[index]
                cost = trial_cost
                raw_rss = trial_rss
                accepted = True
                break
            if terminal != STATUS_MAX_NFEV:
                break

        iteration += 1
        if terminal != STATUS_MAX_NFEV:
            status = terminal
            break
        if nfev >= max_nfev:
            status = STATUS_MAX_NFEV
            break
        if not accepted:
            status = STATUS_INVALID
            break

    return values, cost, raw_rss, status, iteration, nfev, njev


@njit(cache=True)
def _solve_cell(
    objective_callback: Any,
    coordinates: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    full_seeds: np.ndarray,
    seed_count: int,
    full_lower: np.ndarray,
    full_upper: np.ndarray,
    free_indices: np.ndarray,
    weights: np.ndarray,
    use_weights: bool,
    poisson: bool,
    loss_code: int,
    max_nfev: int,
    ftol: float,
    xtol: float,
    gtol: float,
    warm_first: bool,
    prepare_status: int,
    lane_status: np.ndarray,
    lane_nfev: np.ndarray,
    lane_njev: np.ndarray,
    lane_iterations: np.ndarray,
) -> tuple[np.ndarray, float, float, int, int, int, int, int]:
    full_count = full_lower.size
    free_count = free_indices.size
    best_full = np.full(full_count, np.nan, dtype=np.float64)
    if prepare_status < STATUS_MAX_NFEV or seed_count <= 0:
        return best_full, math.inf, math.inf, prepare_status, 0, 0, 0, -1

    full_template = full_lower.copy()
    free_lower = np.empty(free_count, dtype=np.float64)
    free_upper = np.empty(free_count, dtype=np.float64)
    free_seed = np.empty(free_count, dtype=np.float64)
    for index in range(free_count):
        parameter = free_indices[index]
        free_lower[index] = full_lower[parameter]
        free_upper[index] = full_upper[parameter]

    best_cost = math.inf
    best_rss = math.inf
    best_status = STATUS_NO_CANDIDATE
    best_iterations = 0
    best_nfev = 0
    best_njev = 0
    best_seed = -1
    have = False
    best_success = False

    for seed_index in range(seed_count):
        for index in range(free_count):
            free_seed[index] = full_seeds[seed_index, free_indices[index]]
        (
            solved,
            cost,
            raw_rss,
            status,
            iterations,
            nfev,
            njev,
        ) = _solve_seed(
            objective_callback,
            coordinates,
            observations,
            valid,
            full_template,
            free_indices,
            free_seed,
            free_lower,
            free_upper,
            weights,
            use_weights,
            poisson,
            loss_code,
            max_nfev,
            ftol,
            xtol,
            gtol,
        )
        lane_status[seed_index] = status
        lane_nfev[seed_index] = nfev
        lane_njev[seed_index] = njev
        lane_iterations[seed_index] = iterations
        successful = status > STATUS_MAX_NFEV
        choose = False
        if not have:
            choose = True
        elif successful and not best_success:
            choose = True
        elif successful == best_success and raw_rss < (
            best_rss - RSS_TIE_RELATIVE * max(1.0, abs(best_rss))
        ):
            choose = True
        if choose:
            for index in range(full_count):
                best_full[index] = full_template[index]
            for index in range(free_count):
                best_full[free_indices[index]] = solved[index]
            best_cost = cost
            best_rss = raw_rss
            best_status = status
            best_iterations = iterations
            best_nfev = nfev
            best_njev = njev
            best_seed = seed_index
            best_success = successful
            have = True
        # Preserve the scalar warm shortcut: only an already successful,
        # effectively exact warm solution can suppress cold competition.
        if (
            seed_index == 0
            and warm_first
            and successful
            and raw_rss <= RSS_TIE_RELATIVE
        ):
            break
    if not have:
        return best_full, math.inf, math.inf, STATUS_NO_CANDIDATE, 0, 0, 0, -1
    return (
        best_full,
        best_cost,
        best_rss,
        best_status,
        best_iterations,
        best_nfev,
        best_njev,
        best_seed,
    )


@njit(cache=True, nogil=True)
def _solve_serial(
    objective_callback: Any,
    coordinates: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    full_seeds: np.ndarray,
    seed_counts: np.ndarray,
    full_lower: np.ndarray,
    full_upper: np.ndarray,
    free_indices: np.ndarray,
    weights: np.ndarray,
    use_weights: bool,
    poisson: bool,
    loss_code: int,
    max_nfev: int,
    ftol: float,
    xtol: float,
    gtol: float,
    warm_first: np.ndarray,
    prepare_status: np.ndarray,
    parameters: np.ndarray,
    costs: np.ndarray,
    raw_rss: np.ndarray,
    statuses: np.ndarray,
    iterations: np.ndarray,
    nfev: np.ndarray,
    njev: np.ndarray,
    winner_seed: np.ndarray,
    lane_status: np.ndarray,
    lane_nfev: np.ndarray,
    lane_njev: np.ndarray,
    lane_iterations: np.ndarray,
) -> None:
    for cell in range(observations.shape[0]):
        (
            parameters[cell],
            costs[cell],
            raw_rss[cell],
            statuses[cell],
            iterations[cell],
            nfev[cell],
            njev[cell],
            winner_seed[cell],
        ) = _solve_cell(
            objective_callback,
            coordinates[0],
            observations[cell],
            valid[cell],
            full_seeds[cell],
            seed_counts[cell],
            full_lower[cell],
            full_upper[cell],
            free_indices,
            weights[cell],
            use_weights,
            poisson,
            loss_code,
            max_nfev,
            ftol,
            xtol,
            gtol,
            warm_first[cell],
            prepare_status[cell],
            lane_status[cell],
            lane_nfev[cell],
            lane_njev[cell],
            lane_iterations[cell],
        )


@njit(cache=True, nogil=True, parallel=True)
def _solve_parallel(
    objective_callback: Any,
    coordinates: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    full_seeds: np.ndarray,
    seed_counts: np.ndarray,
    full_lower: np.ndarray,
    full_upper: np.ndarray,
    free_indices: np.ndarray,
    weights: np.ndarray,
    use_weights: bool,
    poisson: bool,
    loss_code: int,
    max_nfev: int,
    ftol: float,
    xtol: float,
    gtol: float,
    warm_first: np.ndarray,
    prepare_status: np.ndarray,
    parameters: np.ndarray,
    costs: np.ndarray,
    raw_rss: np.ndarray,
    statuses: np.ndarray,
    iterations: np.ndarray,
    nfev: np.ndarray,
    njev: np.ndarray,
    winner_seed: np.ndarray,
    lane_status: np.ndarray,
    lane_nfev: np.ndarray,
    lane_njev: np.ndarray,
    lane_iterations: np.ndarray,
) -> None:
    for cell in prange(observations.shape[0]):
        (
            parameters[cell],
            costs[cell],
            raw_rss[cell],
            statuses[cell],
            iterations[cell],
            nfev[cell],
            njev[cell],
            winner_seed[cell],
        ) = _solve_cell(
            objective_callback,
            coordinates[0],
            observations[cell],
            valid[cell],
            full_seeds[cell],
            seed_counts[cell],
            full_lower[cell],
            full_upper[cell],
            free_indices,
            weights[cell],
            use_weights,
            poisson,
            loss_code,
            max_nfev,
            ftol,
            xtol,
            gtol,
            warm_first[cell],
            prepare_status[cell],
            lane_status[cell],
            lane_nfev[cell],
            lane_njev[cell],
            lane_iterations[cell],
        )


@njit(cache=True)
def _finalize_one(
    value_jacobian_callback: Any,
    coordinates: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    parameters: np.ndarray,
    free_indices: np.ndarray,
    weights: np.ndarray,
    use_weights: bool,
    poisson: bool,
    loss_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
    point_count = observations.size
    parameter_count = parameters.size
    free_count = free_indices.size
    fitted = np.full(point_count, np.nan, dtype=np.float64)
    residuals = np.full(point_count, np.nan, dtype=np.float64)
    covariance = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    errors = np.zeros(parameter_count, dtype=np.float64)
    if not _seed_finite(parameters):
        for row in range(parameter_count):
            errors[row] = math.nan
            for column in range(parameter_count):
                covariance[row, column] = math.nan
        return fitted, residuals, covariance, errors, math.inf, False

    predicted, full_jacobian = value_jacobian_callback(coordinates, parameters)
    if predicted.size != point_count:
        for row in range(parameter_count):
            errors[row] = math.nan
            for column in range(parameter_count):
                covariance[row, column] = math.nan
        return fitted, residuals, covariance, errors, math.inf, False

    selected = 0
    raw_rss = 0.0
    finite = True
    scaled_jacobian = np.zeros((point_count, free_count), dtype=np.float64)
    for point in range(point_count):
        fitted[point] = predicted[point]
        residuals[point] = observations[point] - predicted[point]
        if not math.isfinite(fitted[point]) or not math.isfinite(residuals[point]):
            finite = False
        if not valid[point]:
            continue
        selected += 1
        (
            raw,
            squared,
            _point_cost,
            _gradient_factor,
            _information_factor,
            point_finite,
        ) = compiled_point_terms(
            predicted[point],
            observations[point],
            poisson,
            weights[point],
            use_weights,
            loss_code,
        )
        raw_rss += squared
        _rho0, rho1, rho2 = _rho(raw * raw, loss_code)
        robust_scale = math.sqrt(max(rho1 + 2.0 * rho2 * raw * raw, EPSILON))
        if poisson:
            expected = max(predicted[point], COUNT_FLOOR)
            absolute = abs(raw)
            residual_scale = (
                abs(expected - observations[point])
                / (expected * max(absolute, COUNT_FLOOR))
                if absolute > COUNT_FLOOR
                else 1.0 / math.sqrt(expected)
            )
        else:
            residual_scale = weights[point] if use_weights else 1.0
        if not point_finite:
            finite = False
        for free in range(free_count):
            value = (
                full_jacobian[point, free_indices[free]]
                * residual_scale
                * robust_scale
            )
            scaled_jacobian[point, free] = value
            if not math.isfinite(value):
                finite = False

    reduced = raw_rss / max(selected - free_count, 1)
    if free_count == 0:
        return fitted, residuals, covariance, errors, reduced, finite
    covariance_valid = (
        finite
        and selected > free_count
        and math.isfinite(reduced)
        and reduced >= 0.0
    )
    free_covariance = np.empty((free_count, free_count), dtype=np.float64)
    if covariance_valid:
        _left, singular_values, right = np.linalg.svd(
            scaled_jacobian,
            full_matrices=False,
        )
        covariance_valid = (
            singular_values.size == free_count
            and math.isfinite(singular_values[0])
            and math.isfinite(singular_values[free_count - 1])
        )
        tolerance = (
            EPSILON * max(selected, free_count) * singular_values[0]
            if covariance_valid
            else math.inf
        )
        covariance_valid = (
            covariance_valid and singular_values[free_count - 1] > tolerance
        )
        if covariance_valid:
            for row in range(free_count):
                for column in range(free_count):
                    value = 0.0
                    for singular in range(free_count):
                        value += (
                            right[singular, row]
                            * right[singular, column]
                            / (singular_values[singular] * singular_values[singular])
                        )
                    free_covariance[row, column] = value * reduced
            for row in range(free_count):
                for column in range(row):
                    value = 0.5 * (
                        free_covariance[row, column]
                        + free_covariance[column, row]
                    )
                    free_covariance[row, column] = value
                    free_covariance[column, row] = value
                if (
                    not math.isfinite(free_covariance[row, row])
                    or free_covariance[row, row] < 0.0
                ):
                    covariance_valid = False
    free_mask = np.zeros(parameter_count, dtype=np.bool_)
    for free in range(free_count):
        free_mask[free_indices[free]] = True
    if covariance_valid:
        for row in range(free_count):
            full_row = free_indices[row]
            errors[full_row] = math.sqrt(max(free_covariance[row, row], 0.0))
            for column in range(free_count):
                covariance[full_row, free_indices[column]] = free_covariance[row, column]
    else:
        for row in range(parameter_count):
            if free_mask[row]:
                errors[row] = math.nan
            for column in range(parameter_count):
                if free_mask[row] and free_mask[column]:
                    covariance[row, column] = math.nan
    return fitted, residuals, covariance, errors, reduced, covariance_valid


@njit(cache=True, nogil=True)
def _finalize_serial(
    value_jacobian_callback: Any,
    coordinates: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    parameters: np.ndarray,
    free_indices: np.ndarray,
    weights: np.ndarray,
    use_weights: bool,
    poisson: bool,
    loss_code: int,
    fitted: np.ndarray,
    residuals: np.ndarray,
    covariance: np.ndarray,
    errors: np.ndarray,
    reduced: np.ndarray,
    covariance_valid: np.ndarray,
) -> None:
    for cell in range(observations.shape[0]):
        (
            fitted[cell],
            residuals[cell],
            covariance[cell],
            errors[cell],
            reduced[cell],
            covariance_valid[cell],
        ) = _finalize_one(
            value_jacobian_callback,
            coordinates[0],
            observations[cell],
            valid[cell],
            parameters[cell],
            free_indices,
            weights[cell],
            use_weights,
            poisson,
            loss_code,
        )


@njit(cache=True, nogil=True, parallel=True)
def _finalize_parallel(
    value_jacobian_callback: Any,
    coordinates: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    parameters: np.ndarray,
    free_indices: np.ndarray,
    weights: np.ndarray,
    use_weights: bool,
    poisson: bool,
    loss_code: int,
    fitted: np.ndarray,
    residuals: np.ndarray,
    covariance: np.ndarray,
    errors: np.ndarray,
    reduced: np.ndarray,
    covariance_valid: np.ndarray,
) -> None:
    for cell in prange(observations.shape[0]):
        (
            fitted[cell],
            residuals[cell],
            covariance[cell],
            errors[cell],
            reduced[cell],
            covariance_valid[cell],
        ) = _finalize_one(
            value_jacobian_callback,
            coordinates[0],
            observations[cell],
            valid[cell],
            parameters[cell],
            free_indices,
            weights[cell],
            use_weights,
            poisson,
            loss_code,
        )


def _row_matrix(
    value: np.ndarray | Sequence[float],
    cells: int,
    width: int,
    *,
    dtype: np.dtype[Any],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 1:
        if array.shape != (width,):
            raise ValueError(f"{name} must have shape ({width},) or ({cells}, {width})")
        array = np.broadcast_to(array, (cells, width))
    elif array.shape != (cells, width):
        raise ValueError(f"{name} must have shape ({width},) or ({cells}, {width})")
    return np.array(array, dtype=dtype, order="C", copy=True)


def _flag_vector(value: bool | Sequence[bool], cells: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.bool_)
    if array.ndim == 0:
        array = np.full(cells, bool(array), dtype=np.bool_)
    elif array.shape != (cells,):
        raise ValueError(f"{name} must be bool or have shape ({cells},)")
    return np.array(array, dtype=np.bool_, order="C", copy=True)


def _seed_cube(
    value: np.ndarray | Sequence[float] | None,
    cells: int,
    parameters: int,
    *,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.zeros((cells, 1, parameters), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        if array.shape != (parameters,):
            raise ValueError(f"{name} has the wrong parameter count")
        array = np.broadcast_to(array.reshape(1, 1, parameters), (cells, 1, parameters))
    elif array.ndim == 2:
        if array.shape == (cells, parameters):
            array = array.reshape(cells, 1, parameters)
        elif array.shape[1:] == (parameters,):
            array = np.broadcast_to(
                array.reshape(1, array.shape[0], parameters),
                (cells, array.shape[0], parameters),
            )
        else:
            raise ValueError(
                f"{name} must have shape (P,), (B,P), (K,P), or (B,K,P)"
            )
    elif array.ndim == 3:
        if array.shape[0] != cells or array.shape[2] != parameters:
            raise ValueError(
                f"{name} must have shape (P,), (B,P), (K,P), or (B,K,P)"
            )
    else:
        raise ValueError(
            f"{name} must have shape (P,), (B,P), (K,P), or (B,K,P)"
        )
    if array.shape[1] == 0:
        raise ValueError(f"{name} cannot contain zero candidates")
    return np.array(array, dtype=np.float64, order="C", copy=True)


def _coordinate_stack(
    coordinates: Sequence[np.ndarray],
    points: int,
) -> np.ndarray:
    if not coordinates:
        raise ValueError("compiled fit requires at least one coordinate axis")
    stack = np.empty((1, len(coordinates), points), dtype=np.float64)
    for axis, values in enumerate(coordinates):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (points,):
            raise ValueError("compiled fit coordinates must be shared 1D axes")
        stack[0, axis, :] = array
    return np.ascontiguousarray(stack)


def _canonicalize_coordinates(
    descriptor: CompiledFitDescriptor,
    coordinates: np.ndarray,
    valid: np.ndarray,
    *,
    already_canonical: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cells, axes, _points = coordinates.shape
    origins = np.zeros((cells, axes), dtype=np.float64)
    origin_axis = descriptor.coordinate_origin
    if origin_axis is None or already_canonical:
        return coordinates, origins
    canonical = np.array(coordinates, dtype=np.float64, order="C", copy=True)
    if origin_axis >= axes:
        raise ValueError("compiled fit coordinate_origin exceeds coordinate arity")
    for cell in range(cells):
        selected = canonical[cell, origin_axis, valid[cell]]
        if not selected.size:
            continue
        origin = float(np.min(selected))
        if not math.isfinite(origin):
            continue
        canonical[cell, origin_axis] -= origin
        origins[cell, origin_axis] = origin
    return canonical, origins


def _context_stack(
    descriptor: CompiledFitDescriptor,
    coordinates: np.ndarray,
    valid: np.ndarray,
    context: np.ndarray | Sequence[np.ndarray] | None,
    *,
    cells: int,
) -> np.ndarray:
    if context is not None:
        array = np.asarray(context, dtype=np.float64)
        if array.ndim == 2:
            array = np.broadcast_to(array, (cells, *array.shape))
        elif array.ndim != 3 or array.shape[0] != cells:
            raise ValueError("compiled fit context must be shared 2D or per-cell 3D")
        return np.array(array, dtype=np.float64, order="C", copy=True)
    built: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for cell in range(cells):
        compact = tuple(
            np.asarray(
                coordinates[0, axis, valid[cell]],
                dtype=np.float64,
            )
            for axis in range(coordinates.shape[1])
        )
        item = np.asarray(descriptor.context_builder(compact), dtype=np.float64)
        if item.ndim != 2:
            raise ValueError("compiled fit context_builder must return a 2D array")
        if shape is None:
            shape = item.shape
        elif item.shape != shape:
            raise ValueError(
                "compiled fit cells produced different context shapes; bucket them "
                "by coordinate plan before solving"
            )
        built.append(item)
    return np.ascontiguousarray(np.stack(built, axis=0), dtype=np.float64)


def _compile_exact(dispatcher: Any, signature: Any, name: str) -> None:
    compile_method = getattr(dispatcher, "compile", None)
    if compile_method is None:
        raise TypeError(f"compiled fit {name} must be a Numba dispatcher")
    argument_signature = tuple(signature.args)
    if argument_signature not in tuple(getattr(dispatcher, "signatures", ())):
        compile_method(signature)
    # This callback has one ABI by contract.  Refusing an accidental new
    # layout keeps a readonly/strided input from silently compiling another
    # full model and invalidating the disk-cache accounting.
    dispatcher.disable_compile()


def _ensure_compiled_abi(
    descriptor: CompiledFitDescriptor,
    *,
    parallel: bool,
) -> None:
    """Lazily compile callbacks and generic kernels through stable ABI types."""

    global _SERIAL_ABI_READY, _PARALLEL_ABI_READY
    with _ABI_LOCK:
        _compile_exact(
            descriptor.prepare,
            _PREPARE_CALLBACK_SIGNATURE,
            "prepare callback",
        )
        _compile_exact(
            descriptor.objective,
            _OBJECTIVE_CALLBACK_SIGNATURE,
            "objective callback",
        )
        _compile_exact(
            descriptor.value_jacobian,
            _VALUE_JACOBIAN_CALLBACK_SIGNATURE,
            "value/Jacobian callback",
        )
        if parallel:
            if not _PARALLEL_ABI_READY:
                _prepare_parallel.compile(_PREPARE_KERNEL_SIGNATURE)
                _solve_parallel.compile(_SOLVE_KERNEL_SIGNATURE)
                _finalize_parallel.compile(_FINALIZE_KERNEL_SIGNATURE)
                _prepare_parallel.disable_compile()
                _solve_parallel.disable_compile()
                _finalize_parallel.disable_compile()
                _PARALLEL_ABI_READY = True
        elif not _SERIAL_ABI_READY:
            _prepare_serial.compile(_PREPARE_KERNEL_SIGNATURE)
            _solve_serial.compile(_SOLVE_KERNEL_SIGNATURE)
            _finalize_serial.compile(_FINALIZE_KERNEL_SIGNATURE)
            _prepare_serial.disable_compile()
            _solve_serial.disable_compile()
            _finalize_serial.disable_compile()
            _SERIAL_ABI_READY = True


def _solve_compiled(
    descriptor: CompiledFitDescriptor,
    coordinates: Sequence[np.ndarray],
    observations: np.ndarray,
    *,
    base_lower: np.ndarray | Sequence[float],
    base_upper: np.ndarray | Sequence[float],
    valid: np.ndarray | None,
    context: np.ndarray | Sequence[np.ndarray] | None,
    requested_lower: np.ndarray | Sequence[float] | None,
    requested_upper: np.ndarray | Sequence[float] | None,
    requested_mask: np.ndarray | Sequence[bool] | None,
    free_indices: np.ndarray | Sequence[int] | None,
    weights: np.ndarray | None,
    poisson: bool,
    loss: str,
    max_nfev: int,
    ftol: float,
    xtol: float,
    gtol: float,
    authored_seeds: np.ndarray | Sequence[float] | None,
    use_authored: bool | Sequence[bool],
    warm_seeds: np.ndarray | Sequence[float] | None,
    use_warm: bool | Sequence[bool],
    coordinates_are_canonical: bool,
    parallel: bool,
    finalize: bool,
) -> CompiledFitOutput:
    if not isinstance(descriptor, CompiledFitDescriptor):
        raise TypeError("descriptor must be CompiledFitDescriptor")
    _ensure_compiled_abi(descriptor, parallel=parallel)
    values = np.asarray(observations)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim != 2:
        raise ValueError("compiled fit observations must have shape (N,) or (B,N)")
    if (
        values.dtype != np.float64
        or not values.flags.c_contiguous
        or not values.flags.writeable
    ):
        values = np.array(values, dtype=np.float64, order="C", copy=True)
    cells, points = values.shape
    if cells == 0 or points == 0:
        raise ValueError("compiled fit observations cannot be empty")
    coordinate_values = _coordinate_stack(coordinates, points)
    if valid is None:
        valid_values = np.ones((cells, points), dtype=np.bool_)
    else:
        valid_values = np.asarray(valid, dtype=np.bool_)
        if valid_values.ndim == 1 and cells == 1:
            valid_values = valid_values.reshape(1, -1)
        if valid_values.shape != (cells, points):
            raise ValueError("compiled fit valid mask must match observations")
        valid_values = np.array(valid_values, dtype=np.bool_, order="C", copy=True)
    valid_values &= np.isfinite(values)
    for axis in range(coordinate_values.shape[1]):
        valid_values &= np.isfinite(coordinate_values[:, axis, :])

    coordinate_values, coordinate_origins = _canonicalize_coordinates(
        descriptor,
        coordinate_values,
        valid_values,
        already_canonical=bool(coordinates_are_canonical),
    )
    if coordinate_origins.shape[0] == 1 and cells > 1:
        coordinate_origins = np.zeros(
            (cells, coordinate_origins.shape[1]), dtype=np.float64
        )
    contexts = _context_stack(
        descriptor,
        coordinate_values,
        valid_values,
        context,
        cells=cells,
    )

    lower_input = np.asarray(base_lower, dtype=np.float64)
    if lower_input.ndim == 1:
        parameter_count = lower_input.size
    elif lower_input.ndim == 2 and lower_input.shape[0] == cells:
        parameter_count = lower_input.shape[1]
    else:
        raise ValueError("base_lower must be (P,) or (B,P)")
    if parameter_count == 0:
        raise ValueError("compiled fit requires at least one parameter")
    base_lower_values = _row_matrix(
        base_lower,
        cells,
        parameter_count,
        dtype=np.dtype(np.float64),
        name="base_lower",
    )
    base_upper_values = _row_matrix(
        base_upper,
        cells,
        parameter_count,
        dtype=np.dtype(np.float64),
        name="base_upper",
    )

    if requested_mask is None:
        requested_mask_values = np.zeros((cells, parameter_count), dtype=np.bool_)
    else:
        requested_mask_values = _row_matrix(
            requested_mask,
            cells,
            parameter_count,
            dtype=np.dtype(np.bool_),
            name="requested_mask",
        )
    requested_lower_values = _row_matrix(
        base_lower if requested_lower is None else requested_lower,
        cells,
        parameter_count,
        dtype=np.dtype(np.float64),
        name="requested_lower",
    )
    requested_upper_values = _row_matrix(
        base_upper if requested_upper is None else requested_upper,
        cells,
        parameter_count,
        dtype=np.dtype(np.float64),
        name="requested_upper",
    )
    if np.any(
        requested_mask_values
        & (requested_lower_values > requested_upper_values)
    ):
        raise ValueError("requested fit bounds are empty")

    if free_indices is None:
        fixed = requested_mask_values & (
            requested_lower_values == requested_upper_values
        )
        if np.any(fixed != fixed[0]):
            raise ValueError("all compiled fit cells must share one fixed/free pattern")
        free_index_values = np.flatnonzero(~fixed[0]).astype(np.int64)
    else:
        free_index_values = np.asarray(free_indices, dtype=np.int64).reshape(-1)
        if (
            np.any(free_index_values < 0)
            or np.any(free_index_values >= parameter_count)
            or np.unique(free_index_values).size != free_index_values.size
        ):
            raise ValueError("free_indices must be unique valid parameter indices")
    free_index_values = np.ascontiguousarray(free_index_values, dtype=np.int64)
    free_mask = np.zeros(parameter_count, dtype=np.bool_)
    free_mask[free_index_values] = True
    fixed_indices = np.flatnonzero(~free_mask)
    if fixed_indices.size:
        fixed_ok = (
            requested_mask_values[:, fixed_indices]
            & (
                requested_lower_values[:, fixed_indices]
                == requested_upper_values[:, fixed_indices]
            )
        )
        if not np.all(fixed_ok):
            raise ValueError("every non-free parameter needs an exact requested value")

    authored = _seed_cube(
        authored_seeds,
        cells,
        parameter_count,
        name="authored_seeds",
    )
    authored_flags = _flag_vector(use_authored, cells, "use_authored")
    if np.any(authored_flags) and authored_seeds is None:
        raise ValueError("use_authored requires authored_seeds")
    for cell in np.flatnonzero(authored_flags):
        if not np.all(np.isfinite(authored[cell])):
            raise ValueError("authored fit initializer returned invalid parameter values")

    warm_cube = _seed_cube(
        warm_seeds,
        cells,
        parameter_count,
        name="warm_seeds",
    )
    if warm_cube.shape[1] != 1:
        raise ValueError("warm_seeds accepts exactly one seed per cell")
    warm = np.ascontiguousarray(warm_cube[:, 0, :])
    warm_flags = _flag_vector(use_warm, cells, "use_warm")
    if np.any(warm_flags) and warm_seeds is None:
        raise ValueError("use_warm requires warm_seeds")
    for cell in np.flatnonzero(warm_flags):
        if not np.all(np.isfinite(warm[cell])):
            raise ValueError("warm fit initializer returned invalid parameter values")

    if weights is None:
        weight_values = np.ones((cells, points), dtype=np.float64)
        use_weights_value = False
    else:
        weight_values = np.asarray(weights, dtype=np.float64)
        if weight_values.ndim == 1 and cells == 1:
            weight_values = weight_values.reshape(1, -1)
        elif weight_values.ndim == 1 and weight_values.shape == (points,):
            weight_values = np.broadcast_to(weight_values, (cells, points))
        if weight_values.shape != (cells, points):
            raise ValueError("compiled fit weights must match observations")
        weight_values = np.array(
            weight_values,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if not np.all(np.isfinite(weight_values[valid_values])):
            raise ValueError("compiled fit weights must be finite on valid points")
        use_weights_value = True

    try:
        loss_code = LOSS_CODES[str(loss)]
    except KeyError as error:
        raise ValueError(f"unsupported compiled fit loss: {loss!r}") from error
    max_evaluations = int(max_nfev)
    if max_evaluations <= 0:
        raise ValueError("max_nfev must be positive")
    tolerances = (float(ftol), float(xtol), float(gtol))
    if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError("compiled fit tolerances must be finite and positive")

    seed_capacity = 1 + max(descriptor.max_candidates, authored.shape[1])
    full_seeds = np.full(
        (cells, seed_capacity, parameter_count),
        np.nan,
        dtype=np.float64,
    )
    full_lower = np.empty((cells, parameter_count), dtype=np.float64)
    full_upper = np.empty((cells, parameter_count), dtype=np.float64)
    seed_counts = np.zeros(cells, dtype=np.int32)
    prepare_status = np.full(cells, STATUS_NO_CANDIDATE, dtype=np.int32)
    prepare_kernel = _prepare_parallel if parallel else _prepare_serial
    prepare_kernel(
        descriptor.prepare,
        coordinate_values,
        values,
        valid_values,
        contexts,
        base_lower_values,
        base_upper_values,
        requested_lower_values,
        requested_upper_values,
        requested_mask_values,
        warm,
        warm_flags,
        authored,
        authored_flags,
        free_mask,
        descriptor.max_candidates,
        full_seeds,
        full_lower,
        full_upper,
        seed_counts,
        prepare_status,
    )

    parameters = np.full((cells, parameter_count), np.nan, dtype=np.float64)
    costs = np.full(cells, math.inf, dtype=np.float64)
    raw_rss = np.full(cells, math.inf, dtype=np.float64)
    statuses = prepare_status.copy()
    iterations = np.zeros(cells, dtype=np.int32)
    nfev = np.zeros(cells, dtype=np.int32)
    njev = np.zeros(cells, dtype=np.int32)
    winner_seed = np.full(cells, -1, dtype=np.int32)
    lane_status = np.full(
        (cells, seed_capacity),
        STATUS_NO_CANDIDATE,
        dtype=np.int32,
    )
    lane_nfev = np.zeros((cells, seed_capacity), dtype=np.int32)
    lane_njev = np.zeros((cells, seed_capacity), dtype=np.int32)
    lane_iterations = np.zeros((cells, seed_capacity), dtype=np.int32)
    solve_kernel = _solve_parallel if parallel else _solve_serial
    solve_kernel(
        descriptor.objective,
        coordinate_values,
        values,
        valid_values,
        full_seeds,
        seed_counts,
        full_lower,
        full_upper,
        free_index_values,
        weight_values,
        use_weights_value,
        bool(poisson),
        loss_code,
        max_evaluations,
        tolerances[0],
        tolerances[1],
        tolerances[2],
        warm_flags,
        prepare_status,
        parameters,
        costs,
        raw_rss,
        statuses,
        iterations,
        nfev,
        njev,
        winner_seed,
        lane_status,
        lane_nfev,
        lane_njev,
        lane_iterations,
    )

    covariance = np.full(
        (cells, parameter_count, parameter_count),
        np.nan,
        dtype=np.float64,
    )
    standard_errors = np.full((cells, parameter_count), np.nan, dtype=np.float64)
    reduced = np.full(cells, math.inf, dtype=np.float64)
    covariance_valid = np.zeros(cells, dtype=np.bool_)
    if finalize:
        fitted = np.full((cells, points), np.nan, dtype=np.float64)
        residuals = np.full((cells, points), np.nan, dtype=np.float64)
        finalize_kernel = _finalize_parallel if parallel else _finalize_serial
        finalize_kernel(
            descriptor.value_jacobian,
            coordinate_values,
            values,
            valid_values,
            parameters,
            free_index_values,
            weight_values,
            use_weights_value,
            bool(poisson),
            loss_code,
            fitted,
            residuals,
            covariance,
            standard_errors,
            reduced,
            covariance_valid,
        )
    else:
        # Regular-image fits retain the source image and materialize fitted
        # values only when a consumer asks.  They need the common independent
        # TRF result, not an eager B x N x P Jacobian and two B x N planes.
        fitted = np.empty((cells, 0), dtype=np.float64)
        residuals = np.empty((cells, 0), dtype=np.float64)
    success = statuses > STATUS_MAX_NFEV
    covariance_valid &= statuses >= STATUS_MAX_NFEV
    return CompiledFitOutput(
        parameters=parameters,
        standard_errors=standard_errors,
        covariance=covariance,
        fitted_values=fitted,
        residuals=residuals,
        reduced_chi_square=reduced,
        covariance_valid=covariance_valid,
        success=success,
        status=statuses,
        cost=costs,
        raw_rss=raw_rss,
        nfev=nfev,
        njev=njev,
        iterations=iterations,
        winner_seed=winner_seed,
        lane_status=lane_status,
        lane_nfev=lane_nfev,
        lane_njev=lane_njev,
        lane_iterations=lane_iterations,
        coordinate_origins=coordinate_origins,
    )


def solve_compiled_batch(
    descriptor: CompiledFitDescriptor,
    coordinates: Sequence[np.ndarray],
    observations: np.ndarray,
    *,
    base_lower: np.ndarray | Sequence[float],
    base_upper: np.ndarray | Sequence[float],
    valid: np.ndarray | None = None,
    context: np.ndarray | Sequence[np.ndarray] | None = None,
    requested_lower: np.ndarray | Sequence[float] | None = None,
    requested_upper: np.ndarray | Sequence[float] | None = None,
    requested_mask: np.ndarray | Sequence[bool] | None = None,
    free_indices: np.ndarray | Sequence[int] | None = None,
    weights: np.ndarray | None = None,
    poisson: bool = False,
    loss: str = "linear",
    max_nfev: int = 5000,
    ftol: float = 1.0e-8,
    xtol: float = 1.0e-8,
    gtol: float = 1.0e-8,
    authored_seeds: np.ndarray | Sequence[float] | None = None,
    use_authored: bool | Sequence[bool] = False,
    warm_seeds: np.ndarray | Sequence[float] | None = None,
    use_warm: bool | Sequence[bool] = False,
    coordinates_are_canonical: bool = False,
    finalize: bool = True,
) -> CompiledFitOutput:
    """Solve independent cells in one ``prange`` compiled invocation."""

    return _solve_compiled(
        descriptor,
        coordinates,
        observations,
        base_lower=base_lower,
        base_upper=base_upper,
        valid=valid,
        context=context,
        requested_lower=requested_lower,
        requested_upper=requested_upper,
        requested_mask=requested_mask,
        free_indices=free_indices,
        weights=weights,
        poisson=poisson,
        loss=loss,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        authored_seeds=authored_seeds,
        use_authored=use_authored,
        warm_seeds=warm_seeds,
        use_warm=use_warm,
        coordinates_are_canonical=coordinates_are_canonical,
        parallel=True,
        finalize=bool(finalize),
    )


def solve_compiled_single(
    descriptor: CompiledFitDescriptor,
    coordinates: Sequence[np.ndarray],
    observations: np.ndarray,
    *,
    base_lower: np.ndarray | Sequence[float],
    base_upper: np.ndarray | Sequence[float],
    valid: np.ndarray | None = None,
    context: np.ndarray | Sequence[np.ndarray] | None = None,
    requested_lower: np.ndarray | Sequence[float] | None = None,
    requested_upper: np.ndarray | Sequence[float] | None = None,
    requested_mask: np.ndarray | Sequence[bool] | None = None,
    free_indices: np.ndarray | Sequence[int] | None = None,
    weights: np.ndarray | None = None,
    poisson: bool = False,
    loss: str = "linear",
    max_nfev: int = 5000,
    ftol: float = 1.0e-8,
    xtol: float = 1.0e-8,
    gtol: float = 1.0e-8,
    authored_seeds: np.ndarray | Sequence[float] | None = None,
    use_authored: bool | Sequence[bool] = False,
    warm_seeds: np.ndarray | Sequence[float] | None = None,
    use_warm: bool | Sequence[bool] = False,
    coordinates_are_canonical: bool = False,
    finalize: bool = True,
) -> CompiledFitOutput:
    """Solve exactly one cell through the serial form of the compiled core."""

    values = np.asarray(observations)
    if values.ndim == 2 and values.shape[0] == 1:
        pass
    elif values.ndim != 1:
        raise ValueError("solve_compiled_single requires one observation vector")
    return _solve_compiled(
        descriptor,
        coordinates,
        observations,
        base_lower=base_lower,
        base_upper=base_upper,
        valid=valid,
        context=context,
        requested_lower=requested_lower,
        requested_upper=requested_upper,
        requested_mask=requested_mask,
        free_indices=free_indices,
        weights=weights,
        poisson=poisson,
        loss=loss,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        authored_seeds=authored_seeds,
        use_authored=use_authored,
        warm_seeds=warm_seeds,
        use_warm=use_warm,
        coordinates_are_canonical=coordinates_are_canonical,
        parallel=False,
        finalize=bool(finalize),
    )


def _readonly_context(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _context_1d(coordinates: tuple[np.ndarray, ...], *, trigonometric: bool) -> np.ndarray:
    x = np.asarray(coordinates[0], dtype=np.float64).reshape(-1)
    if not x.size:
        return _readonly_context(np.zeros((3, 1), dtype=np.float64))
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    span = max(float(np.ptp(x)), EPSILON)
    differences = np.abs(np.diff(sorted_x))
    differences = differences[differences != 0.0]
    step = max(
        float(np.median(differences)) if differences.size else span,
        EPSILON,
    )
    rows = 3 + (2 * (x.size // 2) if trigonometric else 0)
    context = np.zeros((rows, x.size), dtype=np.float64)
    context[0] = order
    context[1] = sorted_x
    context[2, 0] = float(np.min(x))
    if x.size > 1:
        context[2, 1] = float(np.max(x))
    if x.size > 2:
        context[2, 2] = span
    if x.size > 3:
        context[2, 3] = float(np.mean(x))
    if x.size > 4:
        context[2, 4] = step
    if trigonometric:
        indices = np.arange(x.size, dtype=np.float64)
        for harmonic in range(1, x.size // 2 + 1):
            angle = 2.0 * np.pi * harmonic * indices / x.size
            context[3 + 2 * (harmonic - 1)] = np.cos(angle)
            context[4 + 2 * (harmonic - 1)] = np.sin(angle)
    return _readonly_context(context)


def series_context_builder(coordinates: tuple[np.ndarray, ...]) -> np.ndarray:
    return _context_1d(coordinates, trigonometric=False)


def damped_context_builder(coordinates: tuple[np.ndarray, ...]) -> np.ndarray:
    return _context_1d(coordinates, trigonometric=True)


def image_context_builder(coordinates: tuple[np.ndarray, ...]) -> np.ndarray:
    x = np.asarray(coordinates[0], dtype=np.float64).reshape(-1)
    y = np.asarray(coordinates[1], dtype=np.float64).reshape(-1)
    context = np.zeros((1, 8), dtype=np.float64)
    if x.size:
        context[0, 0] = np.min(x)
        context[0, 1] = np.max(x)
        context[0, 2] = max(float(np.ptp(x)), EPSILON)
        context[0, 4] = np.mean(x)
    if y.size:
        context[0, 3] = max(float(np.ptp(y)), EPSILON)
        context[0, 5] = np.mean(y)
        context[0, 6] = np.min(y)
        context[0, 7] = np.max(y)
    return _readonly_context(context)


@njit(cache=True, inline="always")
def _project_jacobian_row(
    full_row: np.ndarray,
    free_indices: np.ndarray,
    projected: np.ndarray,
) -> bool:
    for index in range(free_indices.size):
        value = full_row[free_indices[index]]
        projected[index] = value
        if not math.isfinite(value):
            return False
    return True


@njit(cache=True, inline="always")
def _accumulate_model_point(
    predicted: float,
    observed: float,
    full_row: np.ndarray,
    free_indices: np.ndarray,
    weight: float,
    use_weights: bool,
    poisson: bool,
    loss_code: int,
    gradient: np.ndarray,
    information: np.ndarray,
    projected: np.ndarray,
    with_derivatives: bool,
) -> tuple[float, float, bool]:
    (
        _raw,
        squared,
        cost,
        gradient_factor,
        information_factor,
        finite,
    ) = compiled_point_terms(
        predicted,
        observed,
        poisson,
        weight,
        use_weights,
        loss_code,
    )
    if not finite or not math.isfinite(predicted):
        return math.inf, math.inf, False
    if with_derivatives:
        if not _project_jacobian_row(full_row, free_indices, projected):
            return math.inf, math.inf, False
        compiled_accumulate(
            projected,
            gradient,
            information,
            gradient_factor,
            information_factor,
        )
    return cost, squared, True


@njit(cache=True)
def _value_jacobian_lorentzian(coords: np.ndarray, parameters: np.ndarray):
    x = coords[0]
    output = np.empty(x.size, dtype=np.float64)
    jacobian = np.empty((x.size, 4), dtype=np.float64)
    center, width, amplitude, offset = parameters
    half = 0.5 * width
    half_squared = half * half
    for point in range(x.size):
        delta = x[point] - center
        denominator = delta * delta + half_squared
        squared = denominator * denominator
        shape = half_squared / denominator
        output[point] = amplitude * shape + offset
        jacobian[point, 0] = amplitude * half_squared * 2.0 * delta / squared
        jacobian[point, 1] = amplitude * half * delta * delta / squared
        jacobian[point, 2] = shape
        jacobian[point, 3] = 1.0
    return output, jacobian


@njit(cache=True)
def _value_jacobian_gaussian(coords: np.ndarray, parameters: np.ndarray):
    x = coords[0]
    output = np.empty(x.size, dtype=np.float64)
    jacobian = np.empty((x.size, 4), dtype=np.float64)
    amplitude, offset, sigma, center = parameters
    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    for point in range(x.size):
        delta = x[point] - center
        shape = math.exp(-0.5 * delta * delta / sigma2)
        output[point] = amplitude * shape + offset
        jacobian[point, 0] = shape
        jacobian[point, 1] = 1.0
        jacobian[point, 2] = amplitude * shape * delta * delta / sigma3
        jacobian[point, 3] = amplitude * shape * delta / sigma2
    return output, jacobian


@njit(cache=True)
def _value_jacobian_histogram(coords: np.ndarray, parameters: np.ndarray):
    x = coords[0]
    output = np.empty(x.size, dtype=np.float64)
    jacobian = np.empty((x.size, 3), dtype=np.float64)
    amplitude, center, sigma = parameters
    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    for point in range(x.size):
        delta = x[point] - center
        shape = math.exp(-0.5 * delta * delta / sigma2)
        output[point] = amplitude * shape
        jacobian[point, 0] = shape
        jacobian[point, 1] = amplitude * shape * delta / sigma2
        jacobian[point, 2] = amplitude * shape * delta * delta / sigma3
    return output, jacobian


@njit(cache=True)
def _value_jacobian_bimodal(coords: np.ndarray, parameters: np.ndarray):
    x = coords[0]
    output = np.empty(x.size, dtype=np.float64)
    jacobian = np.empty((x.size, 6), dtype=np.float64)
    center, splitting, left_amp, left_sigma, right_amp, right_sigma = parameters
    left_center = center - 0.5 * splitting
    right_center = center + 0.5 * splitting
    left2 = left_sigma * left_sigma
    left3 = left2 * left_sigma
    right2 = right_sigma * right_sigma
    right3 = right2 * right_sigma
    for point in range(x.size):
        left_delta = x[point] - left_center
        right_delta = x[point] - right_center
        left_shape = math.exp(-0.5 * left_delta * left_delta / left2)
        right_shape = math.exp(-0.5 * right_delta * right_delta / right2)
        left_center_d = left_amp * left_shape * left_delta / left2
        right_center_d = right_amp * right_shape * right_delta / right2
        output[point] = left_amp * left_shape + right_amp * right_shape
        jacobian[point, 0] = left_center_d + right_center_d
        jacobian[point, 1] = -0.5 * left_center_d + 0.5 * right_center_d
        jacobian[point, 2] = left_shape
        jacobian[point, 3] = left_amp * left_shape * left_delta * left_delta / left3
        jacobian[point, 4] = right_shape
        jacobian[point, 5] = right_amp * right_shape * right_delta * right_delta / right3
    return output, jacobian


@njit(cache=True)
def _value_jacobian_poisson(coords: np.ndarray, parameters: np.ndarray):
    x = coords[0]
    output = np.empty(x.size, dtype=np.float64)
    jacobian = np.empty((x.size, 3), dtype=np.float64)
    valid = np.ones(x.size, dtype=np.bool_)
    table = np.empty(
        _lattice_extent(coords, valid, parameters[1], parameters[2]), dtype=np.float64
    )
    _poisson_table(parameters[1], table)
    floor, _ceiling = _poisson_rate_window(parameters[1])
    for point in range(x.size):
        output[point] = _point_poisson(
            coords, point, parameters, jacobian[point], table, floor
        )
    return output, jacobian


@njit(cache=True)
def _value_jacobian_poisson_bimodal(coords: np.ndarray, parameters: np.ndarray):
    x = coords[0]
    output = np.empty(x.size, dtype=np.float64)
    jacobian = np.empty((x.size, 6), dtype=np.float64)
    valid = np.ones(x.size, dtype=np.bool_)
    right_rate = parameters[0] + parameters[1]
    left = np.empty(
        _lattice_extent(coords, valid, parameters[0], parameters[3]), dtype=np.float64
    )
    right = np.empty(
        _lattice_extent(coords, valid, right_rate, parameters[5]), dtype=np.float64
    )
    _poisson_table(parameters[0], left)
    _poisson_table(right_rate, right)
    left_floor, _l = _poisson_rate_window(parameters[0])
    right_floor, _r = _poisson_rate_window(right_rate)
    for point in range(x.size):
        output[point] = _point_poisson_bimodal(
            coords, point, parameters, jacobian[point],
            left, left_floor, right, right_floor,
        )
    return output, jacobian


@njit(cache=True)
def _value_jacobian_doublet(coords: np.ndarray, parameters: np.ndarray):
    x = coords[0]
    output = np.empty(x.size, dtype=np.float64)
    jacobian = np.empty((x.size, 5), dtype=np.float64)
    center, width, amplitude, offset, splitting = parameters
    half = 0.5 * width
    half_squared = half * half
    left_center = center - 0.5 * splitting
    right_center = center + 0.5 * splitting
    for point in range(x.size):
        left_delta = x[point] - left_center
        right_delta = x[point] - right_center
        left_den = left_delta * left_delta + half_squared
        right_den = right_delta * right_delta + half_squared
        left_shape = half_squared / left_den
        right_shape = half_squared / right_den
        left_center_d = amplitude * half_squared * 2.0 * left_delta / (left_den * left_den)
        right_center_d = amplitude * half_squared * 2.0 * right_delta / (right_den * right_den)
        output[point] = amplitude * (left_shape + right_shape) + offset
        jacobian[point, 0] = left_center_d + right_center_d
        jacobian[point, 1] = amplitude * half * (
            left_delta * left_delta / (left_den * left_den)
            + right_delta * right_delta / (right_den * right_den)
        )
        jacobian[point, 2] = left_shape + right_shape
        jacobian[point, 3] = 1.0
        jacobian[point, 4] = -0.5 * left_center_d + 0.5 * right_center_d
    return output, jacobian


@njit(cache=True)
def _value_jacobian_damped(coords: np.ndarray, parameters: np.ndarray):
    x = coords[0]
    output = np.empty(x.size, dtype=np.float64)
    jacobian = np.empty((x.size, 5), dtype=np.float64)
    amplitude, offset, frequency, decay, phase = parameters
    decay2 = decay * decay
    for point in range(x.size):
        coordinate = x[point]
        exponential = math.exp(-coordinate / decay)
        argument = 2.0 * math.pi * frequency * coordinate + phase
        sine = math.sin(argument)
        cosine = math.cos(argument)
        exp_sine = exponential * sine
        exp_cosine = exponential * cosine
        output[point] = offset + amplitude * exp_sine
        jacobian[point, 0] = exp_sine
        jacobian[point, 1] = 1.0
        jacobian[point, 2] = amplitude * exp_cosine * 2.0 * math.pi * coordinate
        jacobian[point, 3] = amplitude * exp_sine * coordinate / decay2
        jacobian[point, 4] = amplitude * exp_cosine
    return output, jacobian


@njit(cache=True)
def _value_jacobian_exponential(coords: np.ndarray, parameters: np.ndarray):
    x = coords[0]
    output = np.empty(x.size, dtype=np.float64)
    jacobian = np.empty((x.size, 3), dtype=np.float64)
    amplitude, offset, decay = parameters
    decay2 = decay * decay
    for point in range(x.size):
        exponential = math.exp(-x[point] / decay)
        output[point] = offset + amplitude * exponential
        jacobian[point, 0] = exponential
        jacobian[point, 1] = 1.0
        jacobian[point, 2] = amplitude * exponential * x[point] / decay2
    return output, jacobian


@njit(cache=True)
def _value_jacobian_radial(coords: np.ndarray, parameters: np.ndarray):
    x, y = coords[0], coords[1]
    output = np.empty(x.size, dtype=np.float64)
    jacobian = np.empty((x.size, 5), dtype=np.float64)
    amplitude, offset, radius, center_x, center_y = parameters
    radius2 = radius * radius
    radius3 = radius2 * radius
    for point in range(x.size):
        delta_x = x[point] - center_x
        delta_y = y[point] - center_y
        squared = delta_x * delta_x + delta_y * delta_y
        shape = math.exp(-squared / radius2)
        output[point] = offset + amplitude * shape
        jacobian[point, 0] = shape
        jacobian[point, 1] = 1.0
        jacobian[point, 2] = amplitude * shape * 2.0 * squared / radius3
        jacobian[point, 3] = amplitude * shape * 2.0 * delta_x / radius2
        jacobian[point, 4] = amplitude * shape * 2.0 * delta_y / radius2
    return output, jacobian


@njit(cache=True)
def _value_jacobian_anisotropic(coords: np.ndarray, parameters: np.ndarray):
    x, y = coords[0], coords[1]
    output = np.empty(x.size, dtype=np.float64)
    jacobian = np.empty((x.size, 6), dtype=np.float64)
    amplitude, offset, radius_x, radius_y, center_x, center_y = parameters
    radius_x2 = radius_x * radius_x
    radius_y2 = radius_y * radius_y
    radius_x3 = radius_x2 * radius_x
    radius_y3 = radius_y2 * radius_y
    for point in range(x.size):
        delta_x = x[point] - center_x
        delta_y = y[point] - center_y
        delta_x2 = delta_x * delta_x
        delta_y2 = delta_y * delta_y
        shape = math.exp(-(delta_x2 / radius_x2 + delta_y2 / radius_y2))
        output[point] = offset + amplitude * shape
        jacobian[point, 0] = shape
        jacobian[point, 1] = 1.0
        jacobian[point, 2] = amplitude * shape * 2.0 * delta_x2 / radius_x3
        jacobian[point, 3] = amplitude * shape * 2.0 * delta_y2 / radius_y3
        jacobian[point, 4] = amplitude * shape * 2.0 * delta_x / radius_x2
        jacobian[point, 5] = amplitude * shape * 2.0 * delta_y / radius_y2
    return output, jacobian


@njit(cache=True, inline="always")
def _point_lorentzian(coords, point, parameters, row):
    center, width, amplitude, offset = parameters
    delta = coords[0, point] - center
    half = 0.5 * width
    half_squared = half * half
    denominator = delta * delta + half_squared
    denominator_squared = denominator * denominator
    shape = half_squared / denominator
    row[0] = amplitude * half_squared * 2.0 * delta / denominator_squared
    row[1] = amplitude * half * delta * delta / denominator_squared
    row[2] = shape
    row[3] = 1.0
    return amplitude * shape + offset


@njit(cache=True, inline="always")
def _point_gaussian(coords, point, parameters, row):
    amplitude, offset, sigma, center = parameters
    delta = coords[0, point] - center
    sigma2 = sigma * sigma
    shape = math.exp(-0.5 * delta * delta / sigma2)
    row[0] = shape
    row[1] = 1.0
    row[2] = amplitude * shape * delta * delta / (sigma2 * sigma)
    row[3] = amplitude * shape * delta / sigma2
    return amplitude * shape + offset


@njit(cache=True, inline="always")
def _point_histogram(coords, point, parameters, row):
    amplitude, center, sigma = parameters
    delta = coords[0, point] - center
    sigma2 = sigma * sigma
    shape = math.exp(-0.5 * delta * delta / sigma2)
    row[0] = shape
    row[1] = amplitude * shape * delta / sigma2
    row[2] = amplitude * shape * delta * delta / (sigma2 * sigma)
    return amplitude * shape


@njit(cache=True, inline="always")
def _point_bimodal(coords, point, parameters, row):
    center, splitting, left_amp, left_sigma, right_amp, right_sigma = parameters
    x = coords[0, point]
    left_delta = x - (center - 0.5 * splitting)
    right_delta = x - (center + 0.5 * splitting)
    left2 = left_sigma * left_sigma
    right2 = right_sigma * right_sigma
    left_shape = math.exp(-0.5 * left_delta * left_delta / left2)
    right_shape = math.exp(-0.5 * right_delta * right_delta / right2)
    left_center_d = left_amp * left_shape * left_delta / left2
    right_center_d = right_amp * right_shape * right_delta / right2
    row[0] = left_center_d + right_center_d
    row[1] = -0.5 * left_center_d + 0.5 * right_center_d
    row[2] = left_shape
    row[3] = left_amp * left_shape * left_delta * left_delta / (left2 * left_sigma)
    row[4] = right_shape
    row[5] = right_amp * right_shape * right_delta * right_delta / (right2 * right_sigma)
    return left_amp * left_shape + right_amp * right_shape


#: How many read-noise widths a Poisson-Gaussian lattice sum reaches from the
#: bin: the dropped tail is under 2e-14 of the peak term, which keeps the
#: model exact to the frozen anchors' 2e-12.
POISSON_GAUSSIAN_WINDOW = 8.0
#: ...and never reaches lattice terms the Poisson factor itself has left:
#: below ``rate - 10 sqrt(rate) - 10`` and above ``rate + 12 sqrt(rate) + 20``
#: the term is under e^-50 of the mode at every rate, so a wide read noise
#: (a pixel-value histogram, a misfit) does not sum hundreds of zeros.
POISSON_RATE_WINDOW_BELOW = 10.0
POISSON_RATE_WINDOW_BELOW_MARGIN = 10.0
POISSON_RATE_WINDOW_ABOVE = 12.0
POISSON_RATE_WINDOW_ABOVE_MARGIN = 20.0
TINY = float(np.finfo(np.float64).tiny)


@njit(cache=True, inline="always")
def _poisson_rate_window(rate):
    root = math.sqrt(max(rate, 0.0))
    lowest = int(math.floor(rate - POISSON_RATE_WINDOW_BELOW * root - POISSON_RATE_WINDOW_BELOW_MARGIN))
    highest = int(math.ceil(rate + POISSON_RATE_WINDOW_ABOVE * root + POISSON_RATE_WINDOW_ABOVE_MARGIN))
    return max(lowest, 0), highest


@njit(cache=True, inline="always")
def _lattice_extent(coords, valid, rate, sigma):
    """How many lattice terms ``0..K`` the bins can reach: the largest bin
    plus the window, and no further than the rate's own window.  At least
    one term, so an empty cell still has a table."""

    half = int(math.ceil(POISSON_GAUSSIAN_WINDOW * sigma))
    largest = -math.inf
    for point in range(coords.shape[1]):
        if valid[point] and coords[0, point] > largest:
            largest = coords[0, point]
    if not math.isfinite(largest):
        return 1
    _lowest, highest = _poisson_rate_window(rate)
    return max(min(int(np.rint(largest)) + half, highest) + 1, 1)


@njit(cache=True, inline="always")
def _poisson_table(rate, table):
    """``table[k] = P(k | rate)`` for every lattice term the bins can reach.

    Exact at the mode, walked outward on both sides by the ratio of
    neighbouring terms -- each step is a multiply and the terms only shrink,
    so nothing overflows and a far tail underflows to the zero it is.  Filled
    once per objective evaluation; the per-bin lattice sums then only read it.
    """

    size = table.size
    if rate <= 0.0:
        for k in range(size):
            table[k] = 0.0
        table[0] = 1.0
        return
    mode = min(int(rate), size - 1)
    centre = math.exp(mode * math.log(rate) - rate - math.lgamma(mode + 1.0))
    table[mode] = centre
    value = centre
    for k in range(mode, size - 1):
        value *= rate / (k + 1.0)
        table[k + 1] = value
    value = centre
    for k in range(mode, 0, -1):
        value *= k / rate
        table[k - 1] = value


@njit(cache=True, inline="always")
def _poisson_lattice(x, sigma, table, k_floor):
    """``sum_k P_k g_k``, ``sum_k P_k g_k k`` and ``sum_k P_k g_k (x-k)^2``
    over the lattice window around one bin, ``P_k`` read from ``table``;
    ``k_floor`` is the rate window's lower edge, its upper edge the table's.

    Three exponentials per bin and none per lattice term: the Gaussian
    factor walks the lattice as ``g(k+1) = g(k) exp((d_k - 1/2)/sigma^2)``,
    whose own ratio is the constant ``exp(-1/sigma^2)``.  Per term that is
    five multiplies and a table read, which keeps this model within a small
    factor of the plain Gaussian at read-noise widths.  A width so small
    that the running ratio would overflow falls back to the direct
    exponential.
    """

    half = int(math.ceil(POISSON_GAUSSIAN_WINDOW * sigma))
    centre = int(np.rint(x))
    k_low = centre - half
    if k_low < k_floor:
        k_low = k_floor
    k_high = centre + half
    if k_high > table.size - 1:
        k_high = table.size - 1
    if k_high < k_low:
        return 0.0, 0.0, 0.0
    inverse = 1.0 / (sigma * sigma)
    delta = x - k_low
    direct = inverse * (half + 1.0) > 300.0
    shape = math.exp(-0.5 * delta * delta * inverse)
    ratio = math.exp((delta - 0.5) * inverse)
    decay = math.exp(-inverse)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for k in range(k_low, k_high + 1):
        weight = table[k] * shape
        s0 += weight
        s1 += weight * k
        s2 += weight * delta * delta
        delta -= 1.0
        if direct:
            shape = math.exp(-0.5 * delta * delta * inverse)
        else:
            shape *= ratio
            ratio *= decay
    return s0, s1, s2


@njit(cache=True, inline="always")
def _point_poisson(coords, point, parameters, row, table, k_floor):
    amplitude, rate, sigma = parameters
    s0, s1, s2 = _poisson_lattice(coords[0, point], sigma, table, k_floor)
    row[0] = s0
    row[1] = amplitude * (s1 / max(rate, TINY) - s0)
    row[2] = amplitude * s2 / (sigma * sigma * sigma)
    return amplitude * s0


@njit(cache=True, inline="always")
def _point_poisson_bimodal(
    coords, point, parameters, row, left_table, left_floor, right_table, right_floor
):
    left_rate, splitting, left_amp, left_sigma, right_amp, right_sigma = parameters
    x = coords[0, point]
    right_rate = left_rate + splitting
    l0, l1, l2 = _poisson_lattice(x, left_sigma, left_table, left_floor)
    r0, r1, r2 = _poisson_lattice(x, right_sigma, right_table, right_floor)
    left_rate_d = left_amp * (l1 / max(left_rate, TINY) - l0)
    right_rate_d = right_amp * (r1 / max(right_rate, TINY) - r0)
    row[0] = left_rate_d + right_rate_d
    row[1] = right_rate_d
    row[2] = l0
    row[3] = left_amp * l2 / (left_sigma * left_sigma * left_sigma)
    row[4] = r0
    row[5] = right_amp * r2 / (right_sigma * right_sigma * right_sigma)
    return left_amp * l0 + right_amp * r0


@njit(cache=True, inline="always")
def _point_doublet(coords, point, parameters, row):
    center, width, amplitude, offset, splitting = parameters
    x = coords[0, point]
    half = 0.5 * width
    half_squared = half * half
    left_delta = x - (center - 0.5 * splitting)
    right_delta = x - (center + 0.5 * splitting)
    left_den = left_delta * left_delta + half_squared
    right_den = right_delta * right_delta + half_squared
    left_den2 = left_den * left_den
    right_den2 = right_den * right_den
    left_shape = half_squared / left_den
    right_shape = half_squared / right_den
    left_center_d = amplitude * half_squared * 2.0 * left_delta / left_den2
    right_center_d = amplitude * half_squared * 2.0 * right_delta / right_den2
    row[0] = left_center_d + right_center_d
    row[1] = amplitude * half * (
        left_delta * left_delta / left_den2 + right_delta * right_delta / right_den2
    )
    row[2] = left_shape + right_shape
    row[3] = 1.0
    row[4] = -0.5 * left_center_d + 0.5 * right_center_d
    return amplitude * (left_shape + right_shape) + offset


@njit(cache=True, inline="always")
def _point_damped(coords, point, parameters, row):
    amplitude, offset, frequency, decay, phase = parameters
    x = coords[0, point]
    exponential = math.exp(-x / decay)
    argument = 2.0 * math.pi * frequency * x + phase
    sine = math.sin(argument)
    cosine = math.cos(argument)
    exp_sine = exponential * sine
    exp_cosine = exponential * cosine
    row[0] = exp_sine
    row[1] = 1.0
    row[2] = amplitude * exp_cosine * 2.0 * math.pi * x
    row[3] = amplitude * exp_sine * x / (decay * decay)
    row[4] = amplitude * exp_cosine
    return offset + amplitude * exp_sine


@njit(cache=True, inline="always")
def _point_exponential(coords, point, parameters, row):
    amplitude, offset, decay = parameters
    x = coords[0, point]
    exponential = math.exp(-x / decay)
    row[0] = exponential
    row[1] = 1.0
    row[2] = amplitude * exponential * x / (decay * decay)
    return offset + amplitude * exponential


@njit(cache=True, inline="always")
def _point_radial(coords, point, parameters, row):
    amplitude, offset, radius, center_x, center_y = parameters
    delta_x = coords[0, point] - center_x
    delta_y = coords[1, point] - center_y
    squared = delta_x * delta_x + delta_y * delta_y
    radius2 = radius * radius
    shape = math.exp(-squared / radius2)
    row[0] = shape
    row[1] = 1.0
    row[2] = amplitude * shape * 2.0 * squared / (radius2 * radius)
    row[3] = amplitude * shape * 2.0 * delta_x / radius2
    row[4] = amplitude * shape * 2.0 * delta_y / radius2
    return offset + amplitude * shape


@njit(cache=True, inline="always")
def _point_anisotropic(coords, point, parameters, row):
    amplitude, offset, radius_x, radius_y, center_x, center_y = parameters
    delta_x = coords[0, point] - center_x
    delta_y = coords[1, point] - center_y
    delta_x2 = delta_x * delta_x
    delta_y2 = delta_y * delta_y
    radius_x2 = radius_x * radius_x
    radius_y2 = radius_y * radius_y
    shape = math.exp(-(delta_x2 / radius_x2 + delta_y2 / radius_y2))
    row[0] = shape
    row[1] = 1.0
    row[2] = amplitude * shape * 2.0 * delta_x2 / (radius_x2 * radius_x)
    row[3] = amplitude * shape * 2.0 * delta_y2 / (radius_y2 * radius_y)
    row[4] = amplitude * shape * 2.0 * delta_x / radius_x2
    row[5] = amplitude * shape * 2.0 * delta_y / radius_y2
    return offset + amplitude * shape


@njit(cache=True)
def _objective_lorentzian(coords, obs, valid, params, free, weights, use_w, poisson, loss, gradient, info, row, derivatives):
    if derivatives: compiled_reset_accumulators(gradient, info)
    cost=0.0; rss=0.0; full=np.empty(params.size, dtype=np.float64)
    for point in range(obs.size):
        if not valid[point]: continue
        predicted=_point_lorentzian(coords, point, params, full)
        pc, pr, ok=_accumulate_model_point(predicted, obs[point], full, free, weights[point], use_w, poisson, loss, gradient, info, row, derivatives)
        if not ok: return math.inf, math.inf, False
        cost+=pc; rss+=pr
    if derivatives: compiled_finish_information(info)
    return cost, rss, True


@njit(cache=True)
def _objective_gaussian(coords, obs, valid, params, free, weights, use_w, poisson, loss, gradient, info, row, derivatives):
    if derivatives: compiled_reset_accumulators(gradient, info)
    cost=0.0; rss=0.0; full=np.empty(params.size, dtype=np.float64)
    for point in range(obs.size):
        if not valid[point]: continue
        predicted=_point_gaussian(coords, point, params, full)
        pc, pr, ok=_accumulate_model_point(predicted, obs[point], full, free, weights[point], use_w, poisson, loss, gradient, info, row, derivatives)
        if not ok: return math.inf, math.inf, False
        cost+=pc; rss+=pr
    if derivatives: compiled_finish_information(info)
    return cost, rss, True


@njit(cache=True)
def _objective_histogram(coords, obs, valid, params, free, weights, use_w, poisson, loss, gradient, info, row, derivatives):
    if derivatives: compiled_reset_accumulators(gradient, info)
    cost=0.0; rss=0.0; full=np.empty(params.size, dtype=np.float64)
    for point in range(obs.size):
        if not valid[point]: continue
        predicted=_point_histogram(coords, point, params, full)
        pc, pr, ok=_accumulate_model_point(predicted, obs[point], full, free, weights[point], use_w, poisson, loss, gradient, info, row, derivatives)
        if not ok: return math.inf, math.inf, False
        cost+=pc; rss+=pr
    if derivatives: compiled_finish_information(info)
    return cost, rss, True


@njit(cache=True)
def _objective_bimodal(coords, obs, valid, params, free, weights, use_w, poisson, loss, gradient, info, row, derivatives):
    if derivatives: compiled_reset_accumulators(gradient, info)
    cost=0.0; rss=0.0; full=np.empty(params.size, dtype=np.float64)
    for point in range(obs.size):
        if not valid[point]: continue
        predicted=_point_bimodal(coords, point, params, full)
        pc, pr, ok=_accumulate_model_point(predicted, obs[point], full, free, weights[point], use_w, poisson, loss, gradient, info, row, derivatives)
        if not ok: return math.inf, math.inf, False
        cost+=pc; rss+=pr
    if derivatives: compiled_finish_information(info)
    return cost, rss, True


@njit(cache=True)
def _objective_poisson(coords, obs, valid, params, free, weights, use_w, poisson, loss, gradient, info, row, derivatives):
    if derivatives: compiled_reset_accumulators(gradient, info)
    cost=0.0; rss=0.0; full=np.empty(params.size, dtype=np.float64)
    table=np.empty(_lattice_extent(coords, valid, params[1], params[2]), dtype=np.float64)
    _poisson_table(params[1], table)
    floor, _ceiling=_poisson_rate_window(params[1])
    for point in range(obs.size):
        if not valid[point]: continue
        predicted=_point_poisson(coords, point, params, full, table, floor)
        pc, pr, ok=_accumulate_model_point(predicted, obs[point], full, free, weights[point], use_w, poisson, loss, gradient, info, row, derivatives)
        if not ok: return math.inf, math.inf, False
        cost+=pc; rss+=pr
    if derivatives: compiled_finish_information(info)
    return cost, rss, True


@njit(cache=True)
def _objective_poisson_bimodal(coords, obs, valid, params, free, weights, use_w, poisson, loss, gradient, info, row, derivatives):
    if derivatives: compiled_reset_accumulators(gradient, info)
    cost=0.0; rss=0.0; full=np.empty(params.size, dtype=np.float64)
    left=np.empty(_lattice_extent(coords, valid, params[0], params[3]), dtype=np.float64)
    right=np.empty(_lattice_extent(coords, valid, params[0] + params[1], params[5]), dtype=np.float64)
    _poisson_table(params[0], left)
    _poisson_table(params[0] + params[1], right)
    left_floor, _l=_poisson_rate_window(params[0])
    right_floor, _r=_poisson_rate_window(params[0] + params[1])
    for point in range(obs.size):
        if not valid[point]: continue
        predicted=_point_poisson_bimodal(coords, point, params, full, left, left_floor, right, right_floor)
        pc, pr, ok=_accumulate_model_point(predicted, obs[point], full, free, weights[point], use_w, poisson, loss, gradient, info, row, derivatives)
        if not ok: return math.inf, math.inf, False
        cost+=pc; rss+=pr
    if derivatives: compiled_finish_information(info)
    return cost, rss, True


@njit(cache=True)
def _objective_doublet(coords, obs, valid, params, free, weights, use_w, poisson, loss, gradient, info, row, derivatives):
    if derivatives: compiled_reset_accumulators(gradient, info)
    cost=0.0; rss=0.0; full=np.empty(params.size, dtype=np.float64)
    for point in range(obs.size):
        if not valid[point]: continue
        predicted=_point_doublet(coords, point, params, full)
        pc, pr, ok=_accumulate_model_point(predicted, obs[point], full, free, weights[point], use_w, poisson, loss, gradient, info, row, derivatives)
        if not ok: return math.inf, math.inf, False
        cost+=pc; rss+=pr
    if derivatives: compiled_finish_information(info)
    return cost, rss, True


@njit(cache=True)
def _objective_damped(coords, obs, valid, params, free, weights, use_w, poisson, loss, gradient, info, row, derivatives):
    if derivatives: compiled_reset_accumulators(gradient, info)
    cost=0.0; rss=0.0; full=np.empty(params.size, dtype=np.float64)
    for point in range(obs.size):
        if not valid[point]: continue
        predicted=_point_damped(coords, point, params, full)
        pc, pr, ok=_accumulate_model_point(predicted, obs[point], full, free, weights[point], use_w, poisson, loss, gradient, info, row, derivatives)
        if not ok: return math.inf, math.inf, False
        cost+=pc; rss+=pr
    if derivatives: compiled_finish_information(info)
    return cost, rss, True


@njit(cache=True)
def _objective_exponential(coords, obs, valid, params, free, weights, use_w, poisson, loss, gradient, info, row, derivatives):
    if derivatives: compiled_reset_accumulators(gradient, info)
    cost=0.0; rss=0.0; full=np.empty(params.size, dtype=np.float64)
    for point in range(obs.size):
        if not valid[point]: continue
        predicted=_point_exponential(coords, point, params, full)
        pc, pr, ok=_accumulate_model_point(predicted, obs[point], full, free, weights[point], use_w, poisson, loss, gradient, info, row, derivatives)
        if not ok: return math.inf, math.inf, False
        cost+=pc; rss+=pr
    if derivatives: compiled_finish_information(info)
    return cost, rss, True


@njit(cache=True)
def _objective_radial(coords, obs, valid, params, free, weights, use_w, poisson, loss, gradient, info, row, derivatives):
    if derivatives: compiled_reset_accumulators(gradient, info)
    cost=0.0; rss=0.0; full=np.empty(params.size, dtype=np.float64)
    for point in range(obs.size):
        if not valid[point]: continue
        predicted=_point_radial(coords, point, params, full)
        pc, pr, ok=_accumulate_model_point(predicted, obs[point], full, free, weights[point], use_w, poisson, loss, gradient, info, row, derivatives)
        if not ok: return math.inf, math.inf, False
        cost+=pc; rss+=pr
    if derivatives: compiled_finish_information(info)
    return cost, rss, True


@njit(cache=True)
def _objective_anisotropic(coords, obs, valid, params, free, weights, use_w, poisson, loss, gradient, info, row, derivatives):
    if derivatives: compiled_reset_accumulators(gradient, info)
    cost=0.0; rss=0.0; full=np.empty(params.size, dtype=np.float64)
    for point in range(obs.size):
        if not valid[point]: continue
        predicted=_point_anisotropic(coords, point, params, full)
        pc, pr, ok=_accumulate_model_point(predicted, obs[point], full, free, weights[point], use_w, poisson, loss, gradient, info, row, derivatives)
        if not ok: return math.inf, math.inf, False
        cost+=pc; rss+=pr
    if derivatives: compiled_finish_information(info)
    return cost, rss, True


@njit(cache=True, inline="always")
def _compact_observations(coords, observations, valid):
    selected = 0
    for point in range(observations.size):
        if valid[point]:
            selected += 1
    compact_coords = np.empty((coords.shape[0], selected), dtype=np.float64)
    compact_values = np.empty(selected, dtype=np.float64)
    output = 0
    for point in range(observations.size):
        if not valid[point]:
            continue
        compact_values[output] = observations[point]
        for axis in range(coords.shape[0]):
            compact_coords[axis, output] = coords[axis, point]
        output += 1
    return compact_coords, compact_values


@njit(cache=True, inline="always")
def _minimum_maximum(values):
    low = values[0]
    high = values[0]
    for index in range(1, values.size):
        low = min(low, values[index])
        high = max(high, values[index])
    if low < high:
        return low, high
    padding = math.sqrt(EPSILON) * max(abs(low), 1.0)
    return low - padding, high + padding


@njit(cache=True, inline="always")
def _array_span(values):
    low, high = _minimum_maximum(values)
    return max(high - low, EPSILON)


@njit(cache=True, inline="always")
def _array_argmax(values):
    result = 0
    for index in range(1, values.size):
        if values[index] > values[result]:
            result = index
    return result


@njit(cache=True, inline="always")
def _array_argmin(values):
    result = 0
    for index in range(1, values.size):
        if values[index] < values[result]:
            result = index
    return result


@njit(cache=True, inline="always")
def _median(values):
    ordered = np.sort(values.copy())
    count = ordered.size
    if count & 1:
        return ordered[count // 2]
    return 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])


@njit(cache=True, inline="always")
def _unique_step(values):
    ordered = np.sort(values.copy())
    differences = np.empty(max(ordered.size - 1, 1), dtype=np.float64)
    count = 0
    for index in range(1, ordered.size):
        difference = abs(ordered[index] - ordered[index - 1])
        if difference > 0.0:
            differences[count] = difference
            count += 1
    if count == 0:
        return _array_span(ordered)
    return max(_median(differences[:count]), EPSILON)


@njit(cache=True)
def _prepare_lorentzian(coords, observations, valid, seeds, lower, upper, context):
    compact, values = _compact_observations(coords, observations, valid)
    if values.size == 0 or seeds.shape[0] < 2:
        return 0
    x = compact[0]
    xlow, xhigh = _minimum_maximum(x)
    ylow, yhigh = _minimum_maximum(values)
    xspan = max(xhigh - xlow, EPSILON)
    value_range = yhigh - ylow
    width = xspan / 4.0
    seeds[0, 0] = x[_array_argmax(values)]
    seeds[0, 1] = width
    seeds[0, 2] = value_range
    seeds[0, 3] = ylow
    seeds[1, 0] = x[_array_argmin(values)]
    seeds[1, 1] = width
    seeds[1, 2] = -value_range
    seeds[1, 3] = yhigh
    lower[0] = max(lower[0], xlow); upper[0] = min(upper[0], xhigh)
    lower[1] = max(lower[1], width / 10.0); upper[1] = min(upper[1], width * 10.0)
    lower[2] = max(lower[2], -10.0 * value_range); upper[2] = min(upper[2], 10.0 * value_range)
    lower[3] = max(lower[3], ylow - 10.0 * value_range); upper[3] = min(upper[3], yhigh + 10.0 * value_range)
    return 2


@njit(cache=True)
def _prepare_gaussian(coords, observations, valid, seeds, lower, upper, context):
    compact, values = _compact_observations(coords, observations, valid)
    if values.size == 0:
        return 0
    x = compact[0]
    count = max(1, min(values.size // 10, 20))
    edge = np.empty(2 * count, dtype=np.float64)
    for index in range(count):
        edge[index] = values[index]
        edge[count + index] = values[values.size - count + index]
    offset = _median(edge)
    peak = 0
    largest = abs(values[0] - offset)
    for index in range(1, values.size):
        candidate = abs(values[index] - offset)
        if candidate > largest:
            largest = candidate
            peak = index
    amplitude = values[peak] - offset
    if amplitude == 0.0:
        low, high = _minimum_maximum(values)
        amplitude = high - low
        if amplitude == 0.0:
            amplitude = 1.0
    seeds[0, 0] = amplitude
    seeds[0, 1] = offset
    seeds[0, 2] = _array_span(x) / 6.0
    seeds[0, 3] = x[peak]
    return 1


@njit(cache=True)
def _prepare_histogram(coords, observations, valid, seeds, lower, upper, context):
    compact, values = _compact_observations(coords, observations, valid)
    if values.size == 0:
        return 0
    x = compact[0]
    total = 0.0
    first_moment = 0.0
    maximum = max(values[0], 0.0)
    for index in range(values.size):
        weight = max(values[index], 0.0)
        total += weight
        first_moment += x[index] * weight
        maximum = max(maximum, values[index])
    span = _array_span(x)
    if total <= 0.0:
        center = np.mean(x)
        sigma = span / 6.0
    else:
        center = first_moment / total
        variance = 0.0
        for index in range(values.size):
            weight = max(values[index], 0.0)
            difference = x[index] - center
            variance += weight * difference * difference
        sigma = max(math.sqrt(variance / total), span / 1000.0)
    seeds[0, 0] = max(maximum, 0.0)
    seeds[0, 1] = center
    seeds[0, 2] = sigma
    lower[2] = max(lower[2], 0.5 * _unique_step(x))
    return 1


@njit(cache=True, inline="always")
def _bimodal_score(x, counts, center, splitting, left_amp, left_sigma, right_amp, right_sigma):
    total = 0.0
    left_center = center - 0.5 * splitting
    right_center = center + 0.5 * splitting
    for index in range(x.size):
        left = (x[index] - left_center) / left_sigma
        right = (x[index] - right_center) / right_sigma
        predicted = left_amp * math.exp(-0.5 * left * left) + right_amp * math.exp(-0.5 * right * right)
        difference = predicted - counts[index]
        total += difference * difference
    return total


@njit(cache=True, inline="always")
def _try_bimodal_split(x, counts, split_value, step, output):
    left_mass = 0.0
    right_mass = 0.0
    left_moment = 0.0
    right_moment = 0.0
    left_maximum = 0.0
    right_maximum = 0.0
    for index in range(x.size):
        count = max(counts[index], 0.0)
        if x[index] <= split_value:
            left_mass += count
            left_moment += x[index] * count
            left_maximum = max(left_maximum, count)
        else:
            right_mass += count
            right_moment += x[index] * count
            right_maximum = max(right_maximum, count)
    if left_mass <= 0.0 or right_mass <= 0.0:
        return math.inf
    left_center = left_moment / left_mass
    right_center = right_moment / right_mass
    if not right_center > left_center:
        return math.inf
    left_variance = 0.0
    right_variance = 0.0
    for index in range(x.size):
        count = max(counts[index], 0.0)
        if x[index] <= split_value:
            difference = x[index] - left_center
            left_variance += count * difference * difference
        else:
            difference = x[index] - right_center
            right_variance += count * difference * difference
    left_sigma = max(math.sqrt(left_variance / left_mass), step)
    right_sigma = max(math.sqrt(right_variance / right_mass), step)
    center = 0.5 * (left_center + right_center)
    splitting = right_center - left_center
    output[0] = center
    output[1] = splitting
    output[2] = left_maximum
    output[3] = left_sigma
    output[4] = right_maximum
    output[5] = right_sigma
    return _bimodal_score(
        x,
        counts,
        center,
        splitting,
        left_maximum,
        left_sigma,
        right_maximum,
        right_sigma,
    )


@njit(cache=True, inline="always")
def _two_state_cuts(x, values, total, split_values):
    """Fill the cuts a two-peak seed is tried from (sorted ``x``): deciles of
    the distribution, then the cut that most separates its two halves.
    Returns how many were written; ``split_values`` holds at least ten."""

    count = x.size
    split_count = 0
    cumulative = 0.0
    target = 1
    for index in range(count):
        cumulative += max(values[index], 0.0) / total
        while target <= 9 and cumulative >= 0.1 * target:
            candidate = x[index]
            duplicate = False
            for prior in range(split_count):
                if split_values[prior] == candidate:
                    duplicate = True
                    break
            if not duplicate:
                split_values[split_count] = candidate
                split_count += 1
            target += 1
    left_mass = 0.0
    left_moment = 0.0
    total_moment = 0.0
    for index in range(count):
        total_moment += max(values[index], 0.0) * x[index]
    best_between = -math.inf
    best_index = 0
    tiny = np.finfo(np.float64).tiny
    for index in range(count - 1):
        value = max(values[index], 0.0)
        left_mass += value
        left_moment += value * x[index]
        right_mass = total - left_mass
        safe_left = max(left_mass, tiny)
        safe_right = max(right_mass, tiny)
        left_mean = left_moment / safe_left
        right_mean = (total_moment - left_moment) / safe_right
        between = safe_left * safe_right * (right_mean - left_mean) ** 2
        if math.isfinite(between) and between > best_between:
            best_between = between
            best_index = index
    if best_between > -math.inf and split_count < 10:
        candidate = x[best_index]
        duplicate = False
        for prior in range(split_count):
            if split_values[prior] == candidate:
                duplicate = True
                break
        if not duplicate:
            split_values[split_count] = candidate
            split_count += 1
    return split_count


@njit(cache=True)
def _prepare_bimodal(coords, observations, valid, seeds, lower, upper, context):
    compact, raw_values = _compact_observations(coords, observations, valid)
    count = raw_values.size
    if count == 0:
        return 0
    order = np.argsort(compact[0])
    x = compact[0, order]
    values = raw_values[order]
    span = _array_span(x)
    step = _unique_step(x)
    total = 0.0
    maximum = 0.0
    for index in range(count):
        value = max(values[index], 0.0)
        total += value
        maximum = max(maximum, value)
    if count < 3 or total <= 0.0:
        midpoint = 0.5 * (x[0] + x[count - 1])
        seeds[0, 0] = midpoint
        seeds[0, 1] = span / 2.0
        seeds[0, 2] = maximum
        seeds[0, 3] = span / 10.0
        seeds[0, 4] = maximum
        seeds[0, 5] = span / 10.0
        lower[3] = max(lower[3], 0.5 * step)
        lower[5] = max(lower[5], 0.5 * step)
        return 1
    split_values = np.empty(10, dtype=np.float64)
    split_count = _two_state_cuts(x, values, total, split_values)
    trial = np.empty(6, dtype=np.float64)
    best = np.empty(6, dtype=np.float64)
    best_score = math.inf
    found = False
    for split in range(split_count):
        score = _try_bimodal_split(x, values, split_values[split], step, trial)
        if score < best_score:
            best_score = score
            for parameter in range(6):
                best[parameter] = trial[parameter]
            found = True
    if not found:
        best[0] = 0.5 * (x[0] + x[count - 1])
        best[1] = span / 2.0
        best[2] = maximum
        best[3] = span / 10.0
        best[4] = maximum
        best[5] = span / 10.0
    for parameter in range(6):
        seeds[0, parameter] = best[parameter]
    lower[3] = max(lower[3], 0.5 * step)
    lower[5] = max(lower[5], 0.5 * step)
    return 1


@njit(cache=True, inline="always")
def _poisson_moments(x, values, split, side, step):
    """(amplitude, rate, sigma, mass) of one Poisson-Gaussian component from
    the quartiles of the bins on one ``side`` of ``split`` (0: every bin);
    ``x`` is sorted.  Mirrors ``fit._poisson_moments``."""

    total = 0.0
    maximum = 0.0
    for index in range(x.size):
        if side < 0 and x[index] > split:
            continue
        if side > 0 and x[index] <= split:
            continue
        value = max(values[index], 0.0)
        total += value
        maximum = max(maximum, value)
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    cumulative = 0.0
    lower = x[0]
    median = x[0]
    upper = x[0]
    quartile = 0
    for index in range(x.size):
        if side < 0 and x[index] > split:
            continue
        if side > 0 and x[index] <= split:
            continue
        cumulative += max(values[index], 0.0)
        while quartile < 3 and cumulative >= 0.25 * (quartile + 1) * total:
            if quartile == 0:
                lower = x[index]
            elif quartile == 1:
                median = x[index]
            else:
                upper = x[index]
            quartile += 1
    rate = max(median, 0.0)
    width = (upper - lower) / 1.349
    sigma = math.sqrt(max(width * width - rate, 0.25 * step * step))
    amplitude = maximum * math.sqrt(rate + sigma * sigma) / sigma
    return amplitude, rate, sigma, total


@njit(cache=True)
def _prepare_poisson_histogram(coords, observations, valid, seeds, lower, upper, context):
    compact, raw_values = _compact_observations(coords, observations, valid)
    if raw_values.size == 0:
        return 0
    order = np.argsort(compact[0])
    x = compact[0, order]
    values = raw_values[order]
    step = _unique_step(x)
    span = _array_span(x)
    amplitude, rate, sigma, total = _poisson_moments(x, values, 0.0, 0, step)
    if total <= 0.0:
        amplitude = 0.0
        rate = max(np.mean(x), 0.0)
        sigma = max(span / 6.0, 0.5 * step)
    seeds[0, 0] = amplitude
    seeds[0, 1] = rate
    seeds[0, 2] = sigma
    lower[2] = max(lower[2], 0.5 * step); upper[2] = min(upper[2], span)
    return 1


@njit(cache=True, inline="always")
def _poisson_bimodal_score(x, counts, seed):
    """Distance of a two-state seed from the histogram by each component's
    Gaussian approximation (variance ``rate + sigma^2``): one exponential
    per bin ranks the cuts.  ``fit._poisson_seed_distance`` is the same
    arithmetic for the SciPy path."""

    total = 0.0
    left_rate = seed[0]
    right_rate = seed[0] + seed[1]
    left_variance = left_rate + seed[3] * seed[3]
    right_variance = right_rate + seed[5] * seed[5]
    left_height = seed[2] * seed[3] / math.sqrt(left_variance)
    right_height = seed[4] * seed[5] / math.sqrt(right_variance)
    for index in range(x.size):
        left_delta = x[index] - left_rate
        right_delta = x[index] - right_rate
        predicted = (
            left_height * math.exp(-0.5 * left_delta * left_delta / left_variance)
            + right_height * math.exp(-0.5 * right_delta * right_delta / right_variance)
        )
        difference = predicted - counts[index]
        total += difference * difference
    return total


@njit(cache=True, inline="always")
def _try_poisson_split(x, counts, split_value, step, output):
    left_amplitude, left_rate, left_sigma, left_mass = _poisson_moments(
        x, counts, split_value, -1, step
    )
    right_amplitude, right_rate, right_sigma, right_mass = _poisson_moments(
        x, counts, split_value, 1, step
    )
    if left_mass <= 0.0 or right_mass <= 0.0 or not right_rate > left_rate:
        return math.inf
    output[0] = left_rate
    output[1] = right_rate - left_rate
    output[2] = left_amplitude
    output[3] = left_sigma
    output[4] = right_amplitude
    output[5] = right_sigma
    return _poisson_bimodal_score(x, counts, output)


@njit(cache=True)
def _prepare_poisson_bimodal(coords, observations, valid, seeds, lower, upper, context):
    compact, raw_values = _compact_observations(coords, observations, valid)
    count = raw_values.size
    if count == 0:
        return 0
    order = np.argsort(compact[0])
    x = compact[0, order]
    values = raw_values[order]
    span = _array_span(x)
    step = _unique_step(x)
    total = 0.0
    maximum = 0.0
    for index in range(count):
        value = max(values[index], 0.0)
        total += value
        maximum = max(maximum, value)
    lower[3] = max(lower[3], 0.5 * step); upper[3] = min(upper[3], span)
    lower[5] = max(lower[5], 0.5 * step); upper[5] = min(upper[5], span)
    midpoint = 0.5 * (x[0] + x[count - 1])
    if count < 3 or total <= 0.0:
        seeds[0, 0] = max(midpoint - span / 4.0, 0.0)
        seeds[0, 1] = span / 2.0
        seeds[0, 2] = maximum
        seeds[0, 3] = span / 10.0
        seeds[0, 4] = maximum
        seeds[0, 5] = span / 10.0
        return 1
    split_values = np.empty(10, dtype=np.float64)
    split_count = _two_state_cuts(x, values, total, split_values)
    trial = np.empty(6, dtype=np.float64)
    best = np.empty(6, dtype=np.float64)
    best_score = math.inf
    found = False
    for split in range(split_count):
        score = _try_poisson_split(x, values, split_values[split], step, trial)
        if score < best_score:
            best_score = score
            for parameter in range(6):
                best[parameter] = trial[parameter]
            found = True
    if not found:
        best[0] = max(midpoint - span / 4.0, 0.0)
        best[1] = span / 2.0
        best[2] = maximum
        best[3] = span / 10.0
        best[4] = maximum
        best[5] = span / 10.0
    for parameter in range(6):
        seeds[0, parameter] = best[parameter]
    return 1


@njit(cache=True, inline="always")
def _peak_properties(values, threshold, peaks, widths):
    count = 0
    index = 1
    while index < values.size - 1:
        if values[index] > values[index - 1]:
            end = index
            while end + 1 < values.size and values[end + 1] == values[index]:
                end += 1
            if end < values.size - 1 and values[end] > values[end + 1]:
                peak = (index + end) // 2
                height = values[peak]
                left_minimum = height
                scan = peak
                while scan > 0:
                    scan -= 1
                    left_minimum = min(left_minimum, values[scan])
                    if values[scan] > height:
                        break
                right_minimum = height
                scan = peak
                while scan < values.size - 1:
                    scan += 1
                    right_minimum = min(right_minimum, values[scan])
                    if values[scan] > height:
                        break
                prominence = height - max(left_minimum, right_minimum)
                if prominence >= threshold:
                    level = height - 0.5 * prominence
                    left = peak
                    while left > 0 and values[left] > level:
                        left -= 1
                    left_position = float(left) if values[left + 1] == values[left] else left + (level - values[left]) / (values[left + 1] - values[left])
                    right = peak
                    while right < values.size - 1 and values[right] > level:
                        right += 1
                    right_position = float(right) if values[right - 1] == values[right] else (right - 1) + (level - values[right - 1]) / (values[right] - values[right - 1])
                    width = right_position - left_position
                    if width >= 1.0:
                        peaks[count] = peak
                        widths[count] = width
                        count += 1
            index = end + 1
        else:
            index += 1
    return count


@njit(cache=True, inline="always")
def _append_doublet_sign(x, values, sign, value_range, step, seeds, start):
    signed = sign * values
    peaks = np.empty(values.size, dtype=np.int64)
    widths = np.empty(values.size, dtype=np.float64)
    count = _peak_properties(signed, value_range / 8.0, peaks, widths)
    if count == 0:
        return start
    chosen_count = min(4, count)
    chosen = np.empty(chosen_count, dtype=np.int64)
    used = np.zeros(count, dtype=np.bool_)
    for output in range(chosen_count):
        best = -1
        best_value = -math.inf
        for candidate in range(count):
            if not used[candidate] and (
                signed[peaks[candidate]] > best_value
                or (signed[peaks[candidate]] == best_value and candidate > best)
            ):
                best = candidate
                best_value = signed[peaks[candidate]]
        used[best] = True
        chosen[output] = best
    first = peaks[chosen[0]]
    width = max(widths[chosen[0]] * step, step)
    low, high = _minimum_maximum(values)
    offset = low if sign > 0.0 else high
    for output in range(chosen_count):
        second = peaks[chosen[output]]
        row = start + output
        if row >= seeds.shape[0]:
            break
        seeds[row, 0] = 0.5 * (x[first] + x[second])
        seeds[row, 1] = width
        seeds[row, 2] = sign * value_range
        seeds[row, 3] = offset
        seeds[row, 4] = abs(x[second] - x[first])
    return min(start + chosen_count, seeds.shape[0])


@njit(cache=True)
def _prepare_doublet(coords, observations, valid, seeds, lower, upper, context):
    compact, raw_values = _compact_observations(coords, observations, valid)
    if raw_values.size == 0:
        return 0
    order = np.argsort(compact[0])
    x = compact[0, order]
    values = raw_values[order]
    xlow, xhigh = _minimum_maximum(x)
    ylow, yhigh = _minimum_maximum(values)
    xspan = max(xhigh - xlow, EPSILON)
    value_range = yhigh - ylow
    step = _unique_step(x)
    count = _append_doublet_sign(x, values, 1.0, value_range, step, seeds, 0)
    count = _append_doublet_sign(x, values, -1.0, value_range, step, seeds, count)
    if count == 0:
        if seeds.shape[0] < 2:
            return 0
        width = xspan / 8.0
        high_index = _array_argmax(values)
        low_index = _array_argmin(values)
        seeds[0, 0] = x[high_index]; seeds[0, 1] = width; seeds[0, 2] = value_range; seeds[0, 3] = ylow; seeds[0, 4] = 2.0 * width
        seeds[1, 0] = x[low_index]; seeds[1, 1] = width; seeds[1, 2] = -value_range; seeds[1, 3] = yhigh; seeds[1, 4] = 2.0 * width
        count = 2
    width = seeds[0, 1]
    lower[0] = max(lower[0], xlow); upper[0] = min(upper[0], xhigh)
    lower[1] = max(lower[1], width / 10.0); upper[1] = min(upper[1], width * 10.0)
    lower[2] = max(lower[2], -10.0 * value_range); upper[2] = min(upper[2], 10.0 * value_range)
    lower[3] = max(lower[3], ylow - 10.0 * value_range); upper[3] = min(upper[3], yhigh + 10.0 * value_range)
    lower[4] = max(lower[4], 0.0); upper[4] = min(upper[4], 2.0 * xspan)
    return count


@njit(cache=True)
def _prepare_damped(coords, observations, valid, seeds, lower, upper, context):
    compact, raw_values = _compact_observations(coords, observations, valid)
    if raw_values.size == 0 or seeds.shape[0] < 3:
        return 0
    order = np.argsort(compact[0])
    x = compact[0, order]
    values = raw_values[order]
    count = values.size
    offset = np.mean(values)
    low, high = _minimum_maximum(values)
    amplitude = max(0.5 * (high - low), EPSILON)
    spacing = _unique_step(x) if count > 1 else 1.0
    best_harmonic = 1
    best_power = -1.0
    # Goertzel per harmonic: the same |DFT|^2 the naive scan computed,
    # via one multiply-add recurrence per sample instead of a sine and a
    # cosine call each -- the scan is O(N^2) either way, but the naive
    # form spent ~60 ms of a damped-sine solve inside the trig calls at
    # two thousand points, ten times the whole solve of every other
    # curve model.
    for harmonic in range(1, count // 2 + 1):
        angle = 2.0 * math.pi * harmonic / count
        coefficient = 2.0 * math.cos(angle)
        previous = 0.0
        before = 0.0
        for index in range(count):
            current = (
                (values[index] - offset) + coefficient * previous - before
            )
            before = previous
            previous = current
        power = (
            previous * previous
            + before * before
            - coefficient * previous * before
        )
        if power > best_power:
            best_power = power
            best_harmonic = harmonic
    frequency = (
        best_harmonic / (count * spacing)
        if count // 2 >= 1
        else 1.0 / _array_span(x)
    )
    frequency = max(frequency, EPSILON)
    decay = _array_span(x)
    # Three blind phases on purpose: a measured DFT phase was tried and
    # made the solve basin-sensitive (batch and single diverged on
    # ordinary data); the third lane is robustness, priced in.
    phases = (-0.5 * math.pi, 0.0, 0.5 * math.pi)
    for seed in range(3):
        seeds[seed, 0] = amplitude
        seeds[seed, 1] = offset
        seeds[seed, 2] = frequency
        seeds[seed, 3] = decay
        seeds[seed, 4] = phases[seed]
    lower[0] = max(lower[0], amplitude / 5.0); upper[0] = min(upper[0], amplitude * 5.0)
    lower[1] = max(lower[1], low); upper[1] = min(upper[1], high)
    lower[2] = max(lower[2], frequency / 5.0); upper[2] = min(upper[2], frequency * 5.0)
    lower[3] = max(lower[3], decay / 5.0); upper[3] = min(upper[3], decay * 5.0)
    return 3


@njit(cache=True)
def _prepare_exponential(coords, observations, valid, seeds, lower, upper, context):
    compact, raw_values = _compact_observations(coords, observations, valid)
    if raw_values.size == 0 or seeds.shape[0] < 2:
        return 0
    order = np.argsort(compact[0])
    x = compact[0, order]
    values = raw_values[order]
    tail_count = max(1, values.size // 10)
    offset = _median(values[values.size - tail_count :])
    decay = _array_span(x) / 3.0
    amplitude = values[0] - offset
    if amplitude == 0.0:
        low, high = _minimum_maximum(values)
        amplitude = high - low
        if amplitude == 0.0:
            amplitude = 1.0
    seeds[0, 0] = amplitude; seeds[0, 1] = offset; seeds[0, 2] = decay
    seeds[1, 0] = -amplitude; seeds[1, 1] = offset; seeds[1, 2] = decay
    low, high = _minimum_maximum(values)
    value_range = high - low
    limit = max(4.0 * value_range, 10.0 * abs(amplitude))
    lower[0] = max(lower[0], -limit); upper[0] = min(upper[0], limit)
    lower[1] = max(lower[1], low - 10.0 * value_range); upper[1] = min(upper[1], high + 10.0 * value_range)
    lower[2] = max(lower[2], decay / 10.0); upper[2] = min(upper[2], decay * 10.0)
    return 2


@njit(cache=True, inline="always")
def _radial_seed(compact, values, sign, output):
    x, y = compact[0], compact[1]
    offset = _median(values)
    total = 0.0
    x_moment = 0.0
    y_moment = 0.0
    for index in range(values.size):
        weight = max(sign * (values[index] - offset), 0.0)
        total += weight
        x_moment += x[index] * weight
        y_moment += y[index] * weight
    if total <= 0.0:
        center_x = np.mean(x)
        center_y = np.mean(y)
        radius = max(_array_span(x), _array_span(y)) / 4.0
    else:
        center_x = x_moment / total
        center_y = y_moment / total
        moment = 0.0
        for index in range(values.size):
            weight = max(sign * (values[index] - offset), 0.0)
            delta_x = x[index] - center_x
            delta_y = y[index] - center_y
            moment += weight * (delta_x * delta_x + delta_y * delta_y)
        radius = max(math.sqrt(moment / total), EPSILON)
    low, high = _minimum_maximum(values)
    amplitude = high - offset if sign > 0.0 else low - offset
    if amplitude == 0.0:
        amplitude = sign * (high - low)
    output[0] = amplitude
    output[1] = offset
    output[2] = radius
    output[3] = center_x
    output[4] = center_y


@njit(cache=True)
def _prepare_radial(coords, observations, valid, seeds, lower, upper, context):
    compact, values = _compact_observations(coords, observations, valid)
    if values.size == 0 or seeds.shape[0] < 2:
        return 0
    _radial_seed(compact, values, 1.0, seeds[0])
    _radial_seed(compact, values, -1.0, seeds[1])
    xlow, xhigh = _minimum_maximum(compact[0])
    ylow, yhigh = _minimum_maximum(compact[1])
    low, high = _minimum_maximum(values)
    value_range = high - low
    radius_low = max(min(seeds[0, 2], seeds[1, 2]) / 10.0, EPSILON)
    radius_high = max(seeds[0, 2], seeds[1, 2]) * 10.0
    lower[0] = max(lower[0], -4.0 * value_range); upper[0] = min(upper[0], 4.0 * value_range)
    lower[1] = max(lower[1], low - value_range); upper[1] = min(upper[1], high + value_range)
    lower[2] = max(lower[2], radius_low); upper[2] = min(upper[2], radius_high)
    lower[3] = max(lower[3], xlow); upper[3] = min(upper[3], xhigh)
    lower[4] = max(lower[4], ylow); upper[4] = min(upper[4], yhigh)
    return 2


@njit(cache=True, inline="always")
def _anisotropic_seed(compact, values, sign, output):
    x, y = compact[0], compact[1]
    offset = _median(values)
    total = 0.0
    x_moment = 0.0
    y_moment = 0.0
    for index in range(values.size):
        weight = max(sign * (values[index] - offset), 0.0)
        total += weight
        x_moment += x[index] * weight
        y_moment += y[index] * weight
    if total <= 0.0:
        center_x = np.mean(x)
        center_y = np.mean(y)
        radius_x = _array_span(x) / 4.0
        radius_y = _array_span(y) / 4.0
    else:
        center_x = x_moment / total
        center_y = y_moment / total
        variance_x = 0.0
        variance_y = 0.0
        for index in range(values.size):
            weight = max(sign * (values[index] - offset), 0.0)
            variance_x += weight * (x[index] - center_x) ** 2
            variance_y += weight * (y[index] - center_y) ** 2
        radius_x = math.sqrt(variance_x / total)
        radius_y = math.sqrt(variance_y / total)
    radius_x = max(radius_x, EPSILON)
    radius_y = max(radius_y, EPSILON)
    low, high = _minimum_maximum(values)
    amplitude = high - offset if sign > 0.0 else low - offset
    if amplitude == 0.0:
        amplitude = sign * (high - low)
    output[0] = amplitude
    output[1] = offset
    output[2] = radius_x
    output[3] = radius_y
    output[4] = center_x
    output[5] = center_y


@njit(cache=True)
def _prepare_anisotropic(coords, observations, valid, seeds, lower, upper, context):
    compact, values = _compact_observations(coords, observations, valid)
    if values.size == 0 or seeds.shape[0] < 2:
        return 0
    _anisotropic_seed(compact, values, 1.0, seeds[0])
    _anisotropic_seed(compact, values, -1.0, seeds[1])
    xlow, xhigh = _minimum_maximum(compact[0])
    ylow, yhigh = _minimum_maximum(compact[1])
    low, high = _minimum_maximum(values)
    value_range = high - low
    radius_x = max(seeds[0, 2], seeds[1, 2])
    radius_y = max(seeds[0, 3], seeds[1, 3])
    lower[0] = max(lower[0], -4.0 * value_range); upper[0] = min(upper[0], 4.0 * value_range)
    lower[1] = max(lower[1], low - value_range); upper[1] = min(upper[1], high + value_range)
    lower[2] = max(lower[2], max(radius_x / 10.0, EPSILON)); upper[2] = min(upper[2], radius_x * 10.0)
    lower[3] = max(lower[3], max(radius_y / 10.0, EPSILON)); upper[3] = min(upper[3], radius_y * 10.0)
    lower[4] = max(lower[4], xlow); upper[4] = min(upper[4], xhigh)
    lower[5] = max(lower[5], ylow); upper[5] = min(upper[5], yhigh)
    return 2


def lorentzian_descriptor() -> CompiledFitDescriptor:
    return CompiledFitDescriptor(
        prepare=_prepare_lorentzian,
        objective=_objective_lorentzian,
        value_jacobian=_value_jacobian_lorentzian,
        context_builder=series_context_builder,
        max_candidates=2,
        cache_key="lorentzian-v1",
    )


def gaussian_offset_descriptor() -> CompiledFitDescriptor:
    return CompiledFitDescriptor(
        prepare=_prepare_gaussian,
        objective=_objective_gaussian,
        value_jacobian=_value_jacobian_gaussian,
        context_builder=series_context_builder,
        max_candidates=1,
        cache_key="gaussian-offset-v1",
    )


def histogram_gaussian_descriptor() -> CompiledFitDescriptor:
    return CompiledFitDescriptor(
        prepare=_prepare_histogram,
        objective=_objective_histogram,
        value_jacobian=_value_jacobian_histogram,
        context_builder=series_context_builder,
        max_candidates=1,
        cache_key="histogram-gaussian-v1",
    )


def bimodal_gaussian_descriptor() -> CompiledFitDescriptor:
    return CompiledFitDescriptor(
        prepare=_prepare_bimodal,
        objective=_objective_bimodal,
        value_jacobian=_value_jacobian_bimodal,
        context_builder=series_context_builder,
        max_candidates=1,
        cache_key="bimodal-gaussian-v1",
    )


def histogram_poisson_gaussian_descriptor() -> CompiledFitDescriptor:
    return CompiledFitDescriptor(
        prepare=_prepare_poisson_histogram,
        objective=_objective_poisson,
        value_jacobian=_value_jacobian_poisson,
        context_builder=series_context_builder,
        max_candidates=1,
        cache_key="histogram-poisson-gaussian-v1",
    )


def bimodal_poisson_gaussian_descriptor() -> CompiledFitDescriptor:
    return CompiledFitDescriptor(
        prepare=_prepare_poisson_bimodal,
        objective=_objective_poisson_bimodal,
        value_jacobian=_value_jacobian_poisson_bimodal,
        context_builder=series_context_builder,
        max_candidates=1,
        cache_key="bimodal-poisson-gaussian-v1",
    )


def symmetric_lorentzian_doublet_descriptor() -> CompiledFitDescriptor:
    return CompiledFitDescriptor(
        prepare=_prepare_doublet,
        objective=_objective_doublet,
        value_jacobian=_value_jacobian_doublet,
        context_builder=series_context_builder,
        max_candidates=8,
        cache_key="symmetric-lorentzian-doublet-v1",
    )


def damped_sine_descriptor() -> CompiledFitDescriptor:
    return CompiledFitDescriptor(
        prepare=_prepare_damped,
        objective=_objective_damped,
        value_jacobian=_value_jacobian_damped,
        context_builder=damped_context_builder,
        max_candidates=3,
        coordinate_origin=0,
        cache_key="damped-sine-v1",
    )


def exponential_decay_descriptor() -> CompiledFitDescriptor:
    return CompiledFitDescriptor(
        prepare=_prepare_exponential,
        objective=_objective_exponential,
        value_jacobian=_value_jacobian_exponential,
        context_builder=series_context_builder,
        max_candidates=2,
        coordinate_origin=0,
        cache_key="exponential-decay-v1",
    )


def radial_gaussian_center_descriptor() -> CompiledFitDescriptor:
    return CompiledFitDescriptor(
        prepare=_prepare_radial,
        objective=_objective_radial,
        value_jacobian=_value_jacobian_radial,
        context_builder=image_context_builder,
        max_candidates=2,
        cache_key="radial-gaussian-center-v1",
    )


def anisotropic_gaussian_center_descriptor() -> CompiledFitDescriptor:
    return CompiledFitDescriptor(
        prepare=_prepare_anisotropic,
        objective=_objective_anisotropic,
        value_jacobian=_value_jacobian_anisotropic,
        context_builder=image_context_builder,
        max_candidates=2,
        cache_key="anisotropic-gaussian-center-v1",
    )


def production_dispatchers() -> tuple[Any, ...]:
    """The exact 39 dispatchers whose machine code belongs in production cache.

    Inline algebra helpers compile as dependencies of these roots and are not
    independent warmer responsibilities.  Keeping this list explicit prevents
    a module scan from treating every tiny inlined function as a separate
    public cache target.
    """

    return (
        _prepare_lorentzian,
        _prepare_gaussian,
        _prepare_histogram,
        _prepare_bimodal,
        _prepare_poisson_histogram,
        _prepare_poisson_bimodal,
        _prepare_doublet,
        _prepare_damped,
        _prepare_exponential,
        _prepare_radial,
        _prepare_anisotropic,
        _objective_lorentzian,
        _objective_gaussian,
        _objective_histogram,
        _objective_bimodal,
        _objective_poisson,
        _objective_poisson_bimodal,
        _objective_doublet,
        _objective_damped,
        _objective_exponential,
        _objective_radial,
        _objective_anisotropic,
        _value_jacobian_lorentzian,
        _value_jacobian_gaussian,
        _value_jacobian_histogram,
        _value_jacobian_bimodal,
        _value_jacobian_poisson,
        _value_jacobian_poisson_bimodal,
        _value_jacobian_doublet,
        _value_jacobian_damped,
        _value_jacobian_exponential,
        _value_jacobian_radial,
        _value_jacobian_anisotropic,
        _prepare_serial,
        _prepare_parallel,
        _solve_serial,
        _solve_parallel,
        _finalize_serial,
        _finalize_parallel,
    )


def warm_production_cache() -> dict[str, Any]:
    """Compile every production descriptor through representative real solves.

    Each model gets one finite, identifiable normal input and an exact authored
    initializer.  Histogram models exercise Poisson likelihood, Gaussian uses
    ``soft_l1`` so the robust derivatives compile, and the remaining models
    exercise linear loss.  A final four-cell Gaussian solve warms the common
    ``prange`` kernels.  Loss code is a runtime integer, so these branches do
    not create additional Numba signatures.
    """

    infinity = math.inf
    positive = EPSILON
    statuses: dict[str, int] = {}

    def run_single(
        name: str,
        descriptor: CompiledFitDescriptor,
        coordinates: tuple[np.ndarray, ...],
        observations: np.ndarray,
        parameters: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        poisson: bool = False,
        loss: str = "linear",
    ) -> None:
        result = solve_compiled_single(
            descriptor,
            coordinates,
            observations,
            base_lower=lower,
            base_upper=upper,
            authored_seeds=parameters,
            use_authored=True,
            poisson=poisson,
            loss=loss,
        )
        status = int(result.status[0])
        if not bool(result.success[0]):
            raise RuntimeError(f"cache warm failed for {name}: {termination_message(status)}")
        statuses[name] = status

    x = np.linspace(-3.0, 3.0, 97, dtype=np.float64)

    lorentzian = np.asarray((0.25, 0.7, 3.0, 0.2), dtype=np.float64)
    lorentzian_values = _value_jacobian_lorentzian.py_func(
        np.ascontiguousarray(x.reshape(1, -1)), lorentzian
    )[0]
    run_single(
        "lorentzian",
        lorentzian_descriptor(),
        (x,),
        lorentzian_values,
        lorentzian,
        np.asarray((-infinity, positive, -infinity, -infinity)),
        np.asarray((infinity, infinity, infinity, infinity)),
    )

    gaussian = np.asarray((3.0, 0.25, 0.6, -0.2), dtype=np.float64)
    gaussian_values = _value_jacobian_gaussian.py_func(
        np.ascontiguousarray(x.reshape(1, -1)), gaussian
    )[0]
    run_single(
        "gaussian_offset_robust",
        gaussian_offset_descriptor(),
        (x,),
        gaussian_values,
        gaussian,
        np.asarray((-infinity, -infinity, positive, -infinity)),
        np.asarray((infinity, infinity, infinity, infinity)),
        loss="soft_l1",
    )

    histogram = np.asarray((70.0, 0.15, 0.75), dtype=np.float64)
    histogram_values = _value_jacobian_histogram.py_func(
        np.ascontiguousarray(x.reshape(1, -1)), histogram
    )[0]
    run_single(
        "histogram_gaussian_poisson",
        histogram_gaussian_descriptor(),
        (x,),
        histogram_values,
        histogram,
        np.asarray((0.0, -infinity, positive)),
        np.asarray((infinity, infinity, infinity)),
        poisson=True,
    )

    bimodal = np.asarray((0.0, 2.0, 55.0, 0.45, 40.0, 0.65), dtype=np.float64)
    bimodal_values = _value_jacobian_bimodal.py_func(
        np.ascontiguousarray(x.reshape(1, -1)), bimodal
    )[0]
    run_single(
        "bimodal_gaussian_poisson",
        bimodal_gaussian_descriptor(),
        (x,),
        bimodal_values,
        bimodal,
        np.asarray((-infinity, 0.0, 0.0, positive, 0.0, positive)),
        np.asarray((infinity, infinity, infinity, infinity, infinity, infinity)),
        poisson=True,
    )

    # Photon-count histograms: the values come from the compiled kernels
    # themselves (their pure-Python twins would compile the inlined lattice
    # helper on its own, which is not a production dispatcher).
    photons = np.linspace(-2.0, 14.0, 97, dtype=np.float64)
    photon_coords = np.ascontiguousarray(photons.reshape(1, -1))
    poisson_single = np.asarray((60.0, 1.5, 0.6), dtype=np.float64)
    run_single(
        "histogram_poisson_gaussian_poisson",
        histogram_poisson_gaussian_descriptor(),
        (photons,),
        _value_jacobian_poisson(photon_coords, poisson_single)[0],
        poisson_single,
        np.asarray((0.0, 0.0, positive)),
        np.asarray((infinity, infinity, infinity)),
        poisson=True,
    )

    poisson_bimodal = np.asarray((0.8, 5.2, 55.0, 0.5, 40.0, 0.7), dtype=np.float64)
    run_single(
        "bimodal_poisson_gaussian_poisson",
        bimodal_poisson_gaussian_descriptor(),
        (photons,),
        _value_jacobian_poisson_bimodal(photon_coords, poisson_bimodal)[0],
        poisson_bimodal,
        np.asarray((0.0, 0.0, 0.0, positive, 0.0, positive)),
        np.asarray((infinity, infinity, infinity, infinity, infinity, infinity)),
        poisson=True,
    )

    doublet = np.asarray((0.1, 0.45, 2.0, 0.15, 1.5), dtype=np.float64)
    doublet_values = _value_jacobian_doublet.py_func(
        np.ascontiguousarray(x.reshape(1, -1)), doublet
    )[0]
    run_single(
        "symmetric_lorentzian_doublet",
        symmetric_lorentzian_doublet_descriptor(),
        (x,),
        doublet_values,
        doublet,
        np.asarray((-infinity, positive, -infinity, -infinity, 0.0)),
        np.asarray((infinity, infinity, infinity, infinity, infinity)),
    )

    world_time = np.linspace(10.0, 14.0, 97, dtype=np.float64)
    relative_time = world_time - world_time[0]
    damped = np.asarray((2.0, 0.2, 0.8, 2.5, 0.3), dtype=np.float64)
    damped_values = _value_jacobian_damped.py_func(
        np.ascontiguousarray(relative_time.reshape(1, -1)), damped
    )[0]
    run_single(
        "damped_sine",
        damped_sine_descriptor(),
        (world_time,),
        damped_values,
        damped,
        np.asarray((0.0, -infinity, positive, positive, -math.pi)),
        np.asarray((infinity, infinity, infinity, infinity, math.pi)),
    )

    exponential = np.asarray((3.0, 0.2, 0.9), dtype=np.float64)
    exponential_values = _value_jacobian_exponential.py_func(
        np.ascontiguousarray(relative_time.reshape(1, -1)), exponential
    )[0]
    run_single(
        "exponential_decay",
        exponential_decay_descriptor(),
        (world_time,),
        exponential_values,
        exponential,
        np.asarray((-infinity, -infinity, positive)),
        np.asarray((infinity, infinity, infinity)),
    )

    axis = np.linspace(-2.0, 2.0, 17, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    image_x = np.ascontiguousarray(grid_x.reshape(-1))
    image_y = np.ascontiguousarray(grid_y.reshape(-1))
    image_coordinates = np.ascontiguousarray(np.stack((image_x, image_y)))

    radial = np.asarray((3.0, 0.2, 0.8, 0.15, -0.1), dtype=np.float64)
    radial_values = _value_jacobian_radial.py_func(image_coordinates, radial)[0]
    run_single(
        "radial_gaussian_center",
        radial_gaussian_center_descriptor(),
        (image_x, image_y),
        radial_values,
        radial,
        np.asarray((-infinity, -infinity, positive, -infinity, -infinity)),
        np.asarray((infinity, infinity, infinity, infinity, infinity)),
    )

    anisotropic = np.asarray((3.0, 0.2, 0.7, 1.0, 0.15, -0.1), dtype=np.float64)
    anisotropic_values = _value_jacobian_anisotropic.py_func(
        image_coordinates, anisotropic
    )[0]
    run_single(
        "anisotropic_gaussian_center",
        anisotropic_gaussian_center_descriptor(),
        (image_x, image_y),
        anisotropic_values,
        anisotropic,
        np.asarray((-infinity, -infinity, positive, positive, -infinity, -infinity)),
        np.asarray((infinity, infinity, infinity, infinity, infinity, infinity)),
    )

    batch = solve_compiled_batch(
        gaussian_offset_descriptor(),
        (x,),
        np.ascontiguousarray(np.stack((gaussian_values,) * 4)),
        base_lower=np.asarray((-infinity, -infinity, positive, -infinity)),
        base_upper=np.asarray((infinity, infinity, infinity, infinity)),
        authored_seeds=gaussian,
        use_authored=True,
    )
    if not bool(np.all(batch.success)):
        raise RuntimeError(f"cache warm failed for Gaussian batch: {batch.status.tolist()}")
    return {
        "dispatchers": len(production_dispatchers()),
        "single_models": statuses,
        "batch_status": tuple(int(value) for value in batch.status),
        "overloads": compiled_overload_counts(),
    }


def compiled_overload_counts() -> dict[str, int]:
    """Small diagnostic for lazy-compilation/cache acceptance checks."""

    dispatchers = {
        "prepare_serial": _prepare_serial,
        "prepare_parallel": _prepare_parallel,
        "solve_serial": _solve_serial,
        "solve_parallel": _solve_parallel,
        "finalize_serial": _finalize_serial,
        "finalize_parallel": _finalize_parallel,
    }
    return {
        name: len(getattr(dispatcher, "overloads", ()))
        for name, dispatcher in dispatchers.items()
    }


def self_check() -> dict[str, Any]:
    """Compile one serial descriptor and verify a deterministic Gaussian fit.

    This is intentionally opt-in: importing :mod:`zlc_plot` must never pay JIT
    cost.  The repository warmer or a focused developer check may call it.
    """

    x = np.linspace(-2.0, 2.0, 81, dtype=np.float64)
    expected = np.asarray((3.0, 0.25, 0.55, -0.2), dtype=np.float64)
    y = expected[0] * np.exp(-0.5 * ((x - expected[3]) / expected[2]) ** 2) + expected[1]
    before = compiled_overload_counts()
    result = solve_compiled_single(
        gaussian_offset_descriptor(),
        (x,),
        y,
        base_lower=np.asarray((-math.inf, -math.inf, EPSILON, -math.inf)),
        base_upper=np.asarray((math.inf, math.inf, math.inf, math.inf)),
    )
    if not bool(result.success[0]):
        raise AssertionError(termination_message(int(result.status[0])))
    np.testing.assert_allclose(result.parameters[0], expected, rtol=1.0e-7, atol=1.0e-9)
    np.testing.assert_allclose(result.fitted_values[0], y, rtol=1.0e-9, atol=1.0e-11)
    return {
        "status": int(result.status[0]),
        "nfev": int(result.nfev[0]),
        "before": before,
        "after": compiled_overload_counts(),
    }


__all__ = [
    "CompiledFitDescriptor",
    "CompiledFitOutput",
    "LOSS_CODES",
    "STATUS_FTOL",
    "STATUS_FTOL_XTOL",
    "STATUS_GTOL",
    "STATUS_INVALID",
    "STATUS_MAX_NFEV",
    "STATUS_NO_CANDIDATE",
    "STATUS_XTOL",
    "anisotropic_gaussian_center_descriptor",
    "bimodal_gaussian_descriptor",
    "bimodal_poisson_gaussian_descriptor",
    "compiled_accumulate",
    "compiled_finish_information",
    "compiled_overload_counts",
    "compiled_point_terms",
    "compiled_reset_accumulators",
    "damped_sine_descriptor",
    "exponential_decay_descriptor",
    "gaussian_offset_descriptor",
    "histogram_gaussian_descriptor",
    "histogram_poisson_gaussian_descriptor",
    "lorentzian_descriptor",
    "production_dispatchers",
    "radial_gaussian_center_descriptor",
    "self_check",
    "solve_compiled_batch",
    "solve_compiled_single",
    "symmetric_lorentzian_doublet_descriptor",
    "termination_message",
    "warm_production_cache",
]
