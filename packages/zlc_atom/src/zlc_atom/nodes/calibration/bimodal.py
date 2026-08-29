"""Dependency-light Gaussian and two-state readout mathematics.

This module is the single owner of the normal CDF, Gaussian overlap, and
threshold classification primitives used by calibration and runtime readout.
It intentionally has no device, runtime, plotting, or GUI imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, isfinite, log, pi, sqrt

import numpy as np

_SIGMA_FLOOR = 1e-12

#: How much wider the bright readout state may be than the dark one.  Both are
#: the same sum over the same pixels, differing by the shot noise of one atom's
#: photons; a few is already generous, and the bound is what keeps a fitted
#: "state" from collapsing onto a handful of samples.
_MAX_WIDTH_RATIO = 5.0

# A weak, symmetric prior keeps a finite sample's tail from winning merely by
# making one component several times narrower than the other.  It does not
# prescribe which state is wider, and raw likelihood dominates it as the shot
# count grows.  sigma=0.5 in log(width ratio) corresponds to a factor 1.65.
_LOG_WIDTH_RATIO_PRIOR_SIGMA = 0.5


def _erf_array(values: np.ndarray) -> np.ndarray:
    """Evaluate :func:`math.erf` over an array without a SciPy dependency."""

    flat = np.asarray(values, dtype=float).reshape(-1)
    result = np.fromiter((erf(float(value)) for value in flat), dtype=float)
    return result.reshape(np.asarray(values).shape)


def normal_cdf(x: object, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray | float:
    """Return the Gaussian CDF, preserving scalar-in/scalar-out behavior."""

    width = max(abs(float(sigma)), _SIGMA_FLOOR)
    values = np.asarray(x, dtype=float)
    result = 0.5 * (
        1.0
        + _erf_array((values - float(mu)) / (width * sqrt(2.0)))
    )
    return float(result) if result.ndim == 0 else result


def finite_mean(values: object, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Mean finite values without warnings for all-invalid slices."""

    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array)
    count = np.count_nonzero(valid, axis=axis)
    total = np.sum(np.where(valid, array, 0.0), axis=axis)
    output = np.full(np.shape(count), np.nan, dtype=float)
    np.divide(total, count, out=output, where=count > 0)
    return output


def _threshold_error(
    threshold: float,
    dark_mean: float,
    dark_sigma: float,
    bright_mean: float,
    bright_sigma: float,
    bright_above: bool,
    dark_weight: float,
    bright_weight: float,
) -> float:
    if bright_above:
        dark_error = 1.0 - float(normal_cdf(threshold, dark_mean, dark_sigma))
        bright_error = float(normal_cdf(threshold, bright_mean, bright_sigma))
    else:
        dark_error = float(normal_cdf(threshold, dark_mean, dark_sigma))
        bright_error = 1.0 - float(normal_cdf(threshold, bright_mean, bright_sigma))
    return dark_weight * dark_error + bright_weight * bright_error


def optimal_gaussian_threshold(
    dark_mean: float,
    dark_sigma: float,
    bright_mean: float,
    bright_sigma: float,
    dark_weight: float = 0.5,
    bright_weight: float = 0.5,
) -> tuple[float, bool]:
    """Return the relevant Bayes crossing for two weighted Gaussians."""

    dark_mean = float(dark_mean)
    bright_mean = float(bright_mean)
    dark_sigma = abs(float(dark_sigma))
    bright_sigma = abs(float(bright_sigma))
    dark_weight = float(dark_weight)
    bright_weight = float(bright_weight)
    bright_above = bright_mean >= dark_mean
    if not all(
        isfinite(value)
        for value in (dark_mean, dark_sigma, bright_mean, bright_sigma)
    ) or min(dark_sigma, bright_sigma) <= 0.0:
        return float("nan"), bright_above
    if (
        not isfinite(dark_weight)
        or not isfinite(bright_weight)
        or min(dark_weight, bright_weight) <= 0.0
    ):
        return float("nan"), bright_above
    weight_sum = dark_weight + bright_weight
    dark_weight /= weight_sum
    bright_weight /= weight_sum

    if bright_above:
        low_mean, low_sigma = dark_mean, dark_sigma
        high_mean, high_sigma = bright_mean, bright_sigma
        low_weight, high_weight = dark_weight, bright_weight
    else:
        low_mean, low_sigma = bright_mean, bright_sigma
        high_mean, high_sigma = dark_mean, dark_sigma
        low_weight, high_weight = bright_weight, dark_weight
    separation = high_mean - low_mean
    if separation <= 0.0:
        return float("nan"), bright_above

    # In x=(threshold-midpoint)/separation coordinates the component means
    # are exactly -1/2 and +1/2.  Equating their log densities gives one
    # linear or quadratic equation without the large count-scale offsets of
    # the raw camera values.
    low_width = max(low_sigma / separation, _SIGMA_FLOOR)
    high_width = max(high_sigma / separation, _SIGMA_FLOOR)
    low_inverse_variance = 1.0 / (low_width * low_width)
    high_inverse_variance = 1.0 / (high_width * high_width)
    a = 0.5 * (low_inverse_variance - high_inverse_variance)
    b = 0.5 * (low_inverse_variance + high_inverse_variance)
    c = (
        0.125 * (low_inverse_variance - high_inverse_variance)
        - log(high_width / low_width)
        - log(low_weight / high_weight)
    )
    scale = max(abs(a), abs(b), abs(c), 1.0)
    roots: tuple[float, ...]
    if abs(a) <= np.finfo(float).eps * scale:
        roots = () if b == 0.0 else (-c / b,)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return float("nan"), bright_above
        root = sqrt(max(discriminant, 0.0))
        roots = ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
    candidates = tuple(
        0.5 * (low_mean + high_mean) + value * separation
        for value in roots
        if isfinite(value) and -0.5 <= value <= 0.5
    )
    if not candidates:
        return float("nan"), bright_above
    threshold = min(
        candidates,
        key=lambda value: _threshold_error(
            value,
            dark_mean,
            dark_sigma,
            bright_mean,
            bright_sigma,
            bright_above,
            dark_weight,
            bright_weight,
        ),
    )
    return float(threshold), bright_above


def gaussian_fidelity(
    dark_mean: float,
    dark_sigma: float,
    bright_mean: float,
    bright_sigma: float,
    threshold: float,
    bright_above: bool = True,
    dark_weight: float = 0.5,
    bright_weight: float = 0.5,
) -> tuple[float, float, float]:
    """Return dark, bright, and weighted classification fidelity."""

    values = (
        dark_mean,
        dark_sigma,
        bright_mean,
        bright_sigma,
        threshold,
        dark_weight,
        bright_weight,
    )
    if not np.isfinite(values).all():
        return float("nan"), float("nan"), float("nan")
    dark_weight = float(dark_weight)
    bright_weight = float(bright_weight)
    if min(dark_weight, bright_weight) <= 0.0:
        return float("nan"), float("nan"), float("nan")
    total = dark_weight + bright_weight
    dark_weight /= total
    bright_weight /= total
    if bright_above:
        dark = float(normal_cdf(threshold, dark_mean, dark_sigma))
        bright = 1.0 - float(normal_cdf(threshold, bright_mean, bright_sigma))
    else:
        dark = 1.0 - float(normal_cdf(threshold, dark_mean, dark_sigma))
        bright = float(normal_cdf(threshold, bright_mean, bright_sigma))
    return dark, bright, dark_weight * dark + bright_weight * bright


def _exact_otsu_threshold(values: np.ndarray, min_fraction: float = 0.02) -> float:
    samples = np.sort(np.asarray(values, dtype=float).reshape(-1))
    samples = samples[np.isfinite(samples)]
    count = int(samples.size)
    if count < 4:
        return float("nan")
    minimum = max(2, int(np.ceil(min_fraction * count)))
    if count < 2 * minimum + 1:
        minimum = max(1, count // 4)
    positions = np.arange(1, count, dtype=float)
    cumulative = np.cumsum(samples)
    left_count = positions
    right_count = float(count) - positions
    valid = (
        (left_count >= minimum)
        & (right_count >= minimum)
        & (samples[:-1] < samples[1:])
    )
    if not np.any(valid):
        return float(np.median(samples))
    left_mean = cumulative[:-1] / left_count
    right_mean = (float(cumulative[-1]) - cumulative[:-1]) / right_count
    score = left_count * right_count * (right_mean - left_mean) ** 2
    score[~valid] = -np.inf
    index = int(np.argmax(score))
    return float(np.median(samples)) if not np.isfinite(score[index]) else float(
        0.5 * (samples[index] + samples[index + 1])
    )


def _em_two_state(
    values: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray,
    *,
    sigma_min: float,
    weight_min: float,
    iterations: int = 400,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run EM from one starting point and report where it got and how well.

    The physics of the two states is imposed on every step rather than hoped
    for afterwards.  A readout state cannot occupy none of the shots, so each
    weight has a floor; a state cannot be infinitely sharp, so each width has
    one (without it, one component walks onto a single sample, its width goes
    to zero and its likelihood to infinity -- the classic way an unconstrained
    mixture "wins" while explaining nothing).  Neither population is forced
    wider: real technical noise can make either conditional distribution the
    narrower one.  Only their width ratio is bounded symmetrically.
    """

    likelihood = -np.inf
    for _ in range(int(iterations)):
        scaled = (values[:, None] - means[None, :]) / sigmas[None, :]
        densities = np.exp(-0.5 * scaled**2) / (sigmas[None, :] * sqrt(2.0 * pi))
        joint = densities * weights[None, :]
        total = joint.sum(axis=1)
        if not np.all(np.isfinite(total)) or np.any(total <= 0.0):
            return means, sigmas, weights, -np.inf
        responsibility = joint / total[:, None]
        counts = responsibility.sum(axis=0)
        weights = np.maximum(counts / values.size, weight_min)
        weights = weights / weights.sum()
        counts = np.maximum(counts, np.finfo(float).tiny)
        means = (responsibility * values[:, None]).sum(axis=0) / counts
        variances = (
            responsibility * (values[:, None] - means[None, :]) ** 2
        ).sum(axis=0) / counts
        sigmas = np.maximum(np.sqrt(np.maximum(variances, 0.0)), sigma_min)
        narrow = int(np.argmin(sigmas))
        wide = 1 - narrow
        sigmas[narrow] = max(
            sigmas[narrow], sigmas[wide] / _MAX_WIDTH_RATIO
        )
        updated_scaled = (values[:, None] - means[None, :]) / sigmas[None, :]
        updated_total = np.sum(
            weights[None, :]
            * np.exp(-0.5 * updated_scaled**2)
            / (sigmas[None, :] * sqrt(2.0 * pi)),
            axis=1,
        )
        if not np.all(np.isfinite(updated_total)) or np.any(updated_total <= 0.0):
            return means, sigmas, weights, -np.inf
        current = float(np.sum(np.log(updated_total)))
        if abs(current - likelihood) <= tolerance * max(1.0, abs(current)):
            likelihood = current
            break
        likelihood = current
    return means, sigmas, weights, likelihood


def _split_start(
    values: np.ndarray, split: float, *, sigma_min: float, weight_min: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Turn a cut through the sample into a starting pair of states."""

    low, high = values[values <= split], values[values > split]
    if low.size < 1 or high.size < 1:
        return None
    spread = float(np.std(values)) or 1.0
    means = np.array([float(np.mean(low)), float(np.mean(high))])
    if means[1] <= means[0]:
        return None
    sigmas = np.array(
        [
            max(float(np.std(low)) if low.size > 1 else 0.5 * spread, sigma_min),
            max(float(np.std(high)) if high.size > 1 else 0.5 * spread, sigma_min),
        ]
    )
    weights = np.array([low.size / values.size, high.size / values.size])
    weights = np.maximum(weights, weight_min)
    return means, sigmas, weights / weights.sum()


@dataclass(frozen=True)
class BimodalFit:
    threshold: float
    fidelity: float
    dark_mean: float
    dark_sigma: float
    bright_mean: float
    bright_sigma: float
    bright_fraction: float
    dark_fidelity: float
    bright_fidelity: float
    bright_above: bool
    ok: bool


def fit_bimodal(values: object, *, min_component_fraction: float = 0.01) -> BimodalFit:
    """Fit the two readout states of one sample, whatever the sample looks like.

    Two Gaussians are always what is fitted, because two states are always
    what is there: a site either held an atom in that shot or it did not, and
    a sample where the two are hard to tell apart is a sample with poor
    separation, not a sample with one state.  So separation is reported, and
    it never withholds the fit -- an operator who put a threshold classifier
    on a site asked to see the two states, and "the peaks are close" is an
    answer about the data, not a reason to show nothing.

    Cuts across the sample provide both hard-partition moment candidates and
    EM-refined candidates.  A weak symmetric prior on their width ratio keeps
    a handful of tail samples from winning solely through an extremely narrow
    Gaussian, while enough real shots still dominate that prior.  This also
    permits a genuinely narrower bright peak; neither component owns the wide
    side.  A winner pinned to the artificial width-ratio boundary is reported
    descriptively but is not a valid two-population classifier.
    """

    samples = np.asarray(values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size < 4:
        split = _exact_otsu_threshold(samples) if samples.size else float("nan")
        return BimodalFit(
            split, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            True, False,
        )
    spread = float(np.std(samples)) or 1.0
    sigma_min = max(0.01 * spread, _SIGMA_FLOOR)
    weight_min = min(
        0.25, max(float(min_component_fraction), 2.0 / float(samples.size))
    )
    starts = [_exact_otsu_threshold(samples)]
    starts.extend(
        float(value)
        for value in np.quantile(samples, (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9))
    )

    best: tuple[
        bool,
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ] | None = None
    for split in starts:
        start = _split_start(
            samples, split, sigma_min=sigma_min, weight_min=weight_min
        )
        if start is None:
            continue
        start_means, start_sigmas, start_weights = (
            np.array(value, copy=True) for value in start
        )
        narrow = int(np.argmin(start_sigmas))
        wide = 1 - narrow
        start_sigmas[narrow] = max(
            start_sigmas[narrow],
            start_sigmas[wide] / _MAX_WIDTH_RATIO,
        )
        refined_means, refined_sigmas, refined_weights, _likelihood = (
            _em_two_state(
                samples,
                *(np.array(value, copy=True) for value in start),
                sigma_min=sigma_min,
                weight_min=weight_min,
            )
        )
        for means, sigmas, weights in (
            (start_means, start_sigmas, start_weights),
            (refined_means, refined_sigmas, refined_weights),
        ):
            scaled = (samples[:, None] - means[None, :]) / sigmas[None, :]
            total = np.sum(
                weights[None, :]
                * np.exp(-0.5 * scaled**2)
                / (sigmas[None, :] * sqrt(2.0 * pi)),
                axis=1,
            )
            likelihood = (
                float(np.sum(np.log(total)))
                if np.all(np.isfinite(total)) and np.all(total > 0.0)
                else -np.inf
            )
            if not isfinite(likelihood):
                continue
            ratio = float(np.max(sigmas) / max(np.min(sigmas), _SIGMA_FLOOR))
            interior = ratio < _MAX_WIDTH_RATIO * (1.0 - 1e-9)
            score = likelihood - 0.5 * (
                log(ratio) / _LOG_WIDTH_RATIO_PRIOR_SIGMA
            ) ** 2
            if best is None or (interior, score) > (best[0], best[1]):
                best = (interior, score, likelihood, means, sigmas, weights)

    if best is None:
        split = _exact_otsu_threshold(samples)
        low, high = samples[samples <= split], samples[samples > split]
        dark_mean = float(np.mean(low)) if low.size else float(np.mean(samples))
        bright_mean = float(np.mean(high)) if high.size else dark_mean + spread
        width = max(0.5 * spread, sigma_min)
        best = (
            False,
            float("nan"),
            float("nan"),
            np.array([dark_mean, bright_mean]),
            np.array([width, width]),
            np.array([0.5, 0.5]),
        )

    _interior, _score, _likelihood, means, sigmas, weights = best
    order = np.argsort(means)
    dark, bright = int(order[0]), int(order[1])
    dark_mean, dark_sigma = float(means[dark]), float(sigmas[dark])
    bright_mean, bright_sigma = float(means[bright]), float(sigmas[bright])
    fraction = float(weights[bright])
    threshold, bright_above = optimal_gaussian_threshold(
        dark_mean, dark_sigma, bright_mean, bright_sigma
    )
    dark_fidelity, bright_fidelity, fidelity = gaussian_fidelity(
        dark_mean, dark_sigma, bright_mean, bright_sigma, threshold, bright_above
    )
    separation = (bright_mean - dark_mean) / max(
        dark_sigma + bright_sigma, _SIGMA_FLOOR
    )
    minimum = max(4.0, float(min_component_fraction) * samples.size) / samples.size
    # Descriptive: whether the two states are far enough apart, and populated
    # enough, for a shot to be assigned to one of them.  Every number above is
    # returned either way.
    #
    # A CLAMP MAY NOT JUDGE THE FIT IT SHAPED.  This also required the fitted
    # width ratio to stay strictly inside _MAX_WIDTH_RATIO -- but that
    # constant is the estimator's own floor on the narrow sigma, applied in
    # _em_two_state, so a site whose TRUE ratio reaches it comes back pinned
    # at exactly the bound and was then rejected for standing there.  In this
    # readout the expected ratio is bright shot noise over dark read noise,
    # sqrt(130/6) ~ 4.7, so the population sits astride the bound: six sites
    # of a thirty-five site lattice -- dark 5.9 +/- 0.8, bright 130 +/- 4.0,
    # fifty sigma apart, BIC gain +550 to +666 -- were reported to the
    # feedback loop as sites that DID NOT LOAD.  Width is not what this
    # sentence is about; distance and population are, and both are asked
    # above.  The bound keeps its other two jobs: it stops a component
    # collapsing onto a handful of samples, and ``interior`` still prefers a
    # start that never had to touch it.
    separated = bool(
        np.isfinite(threshold)
        and separation > 0.5
        and minimum <= fraction <= 1.0 - minimum
    )
    return BimodalFit(
        threshold,
        fidelity,
        dark_mean,
        dark_sigma,
        bright_mean,
        bright_sigma,
        fraction,
        dark_fidelity,
        bright_fidelity,
        bright_above,
        separated,
    )


def per_site_fidelity(
    signals: object,
    labels: object,
    thresholds: object,
    *,
    valid_mask: object | None = None,
) -> "PerSiteConfusion":
    """Return overall accuracy and both class-conditional accuracies per site."""

    values = np.asarray(signals, dtype=float)
    truth = np.asarray(labels, dtype=bool)
    boundary = np.asarray(thresholds, dtype=float).reshape(-1)
    if values.ndim != 2 or truth.shape != values.shape or boundary.shape != (values.shape[1],):
        raise ValueError("signals/labels must be (shots, sites) and thresholds must be (sites,)")
    prediction = values > boundary[None, :]
    valid = np.isfinite(values) & np.isfinite(boundary)[None, :]
    if valid_mask is not None:
        measured = np.asarray(valid_mask, dtype=bool)
        if measured.shape != values.shape:
            raise ValueError("valid_mask must match signals shape")
        valid &= measured
    overall = np.full(values.shape[1], np.nan, dtype="<f8")
    dark_out = np.full(values.shape[1], np.nan, dtype="<f8")
    bright_out = np.full(values.shape[1], np.nan, dtype="<f8")
    evaluated = np.zeros(values.shape[1], dtype=int)
    for site in range(values.shape[1]):
        selected = valid[:, site]
        evaluated[site] = int(np.count_nonzero(selected))
        if not np.any(selected):
            continue
        actual = truth[selected, site]
        predicted = prediction[selected, site]
        overall[site] = float(np.count_nonzero(predicted == actual)) / len(actual)
        dark = int(np.count_nonzero(~actual))
        bright = int(np.count_nonzero(actual))
        if dark:
            dark_out[site] = float(np.count_nonzero(~predicted & ~actual)) / dark
        if bright:
            bright_out[site] = float(np.count_nonzero(predicted & actual)) / bright
    return PerSiteConfusion(overall, dark_out, bright_out, evaluated)


@dataclass(frozen=True)
class PerSiteConfusion:
    """One measured confusion per site: overall and its two class conditionals."""

    overall: np.ndarray
    dark: np.ndarray
    bright: np.ndarray
    #: How many labelled shots each site was evaluated on.
    evaluated: np.ndarray


__all__ = [
    "BimodalFit",
    "PerSiteConfusion",
    "finite_mean",
    "fit_bimodal",
    "gaussian_fidelity",
    "normal_cdf",
    "optimal_gaussian_threshold",
    "per_site_fidelity",
]
