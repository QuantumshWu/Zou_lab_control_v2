"""Dependency-light Gaussian and two-state readout mathematics.

This module is the single owner of the normal CDF, Gaussian overlap, and
threshold classification primitives used by calibration and runtime readout.
It intentionally has no device, runtime, plotting, or GUI imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, isfinite, pi, sqrt

import numpy as np

_SIGMA_FLOOR = 1e-12


def _erf_array(values: np.ndarray) -> np.ndarray:
    """Evaluate :func:`math.erf` over an array without a SciPy dependency."""

    flat = np.asarray(values, dtype=float).reshape(-1)
    result = np.fromiter((erf(float(value)) for value in flat), dtype=float)
    return result.reshape(np.asarray(values).shape)


def gaussian(x: object, amp: float, mu: float, sigma: float) -> np.ndarray | float:
    """Return ``amp * exp(-(x-mu)^2/(2*sigma^2))``."""

    width = max(abs(float(sigma)), _SIGMA_FLOOR)
    result = float(amp) * np.exp(
        -((np.asarray(x, dtype=float) - float(mu)) ** 2) / (2.0 * width * width)
    )
    return float(result) if result.ndim == 0 else result


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


def confidence_weighted_fidelity(
    threshold: float,
    mu0: float,
    sigma0: float,
    weight0: float,
    mu1: float,
    sigma1: float,
    weight1: float,
) -> tuple[float, float, float]:
    """Return confidence-weighted, raw, and normalized peak separation."""

    total = float(weight0) + float(weight1)
    if total <= 0.0:
        return float("nan"), float("nan"), float("nan")
    dark_ok = float(normal_cdf(threshold, mu0, sigma0))
    bright_ok = 1.0 - float(normal_cdf(threshold, mu1, sigma1))
    raw = (float(weight0) * dark_ok + float(weight1) * bright_ok) / total
    separation = abs(float(mu1) - float(mu0)) / max(
        sqrt(float(sigma0) ** 2 + float(sigma1) ** 2), _SIGMA_FLOOR
    )
    balance = 2.0 * min(float(weight0), float(weight1)) / total
    effective = max(0.0, separation - 2.0)
    confidence = float(
        np.clip(balance * (1.0 - exp(-0.5 * effective * effective)), 0.0, 1.0)
    )
    return 0.5 + (raw - 0.5) * confidence, raw, separation


def _threshold_error(
    threshold: float,
    dark_mean: float,
    dark_sigma: float,
    bright_mean: float,
    bright_sigma: float,
    bright_above: bool,
) -> float:
    if bright_above:
        dark_error = 1.0 - float(normal_cdf(threshold, dark_mean, dark_sigma))
        bright_error = float(normal_cdf(threshold, bright_mean, bright_sigma))
    else:
        dark_error = float(normal_cdf(threshold, dark_mean, dark_sigma))
        bright_error = 1.0 - float(normal_cdf(threshold, bright_mean, bright_sigma))
    return 0.5 * (dark_error + bright_error)


def optimal_gaussian_threshold(
    dark_mean: float,
    dark_sigma: float,
    bright_mean: float,
    bright_sigma: float,
) -> tuple[float, bool]:
    """Find the equal-prior threshold between two Gaussian components."""

    dark_mean = float(dark_mean)
    bright_mean = float(bright_mean)
    dark_sigma = max(abs(float(dark_sigma)), _SIGMA_FLOOR)
    bright_sigma = max(abs(float(bright_sigma)), _SIGMA_FLOOR)
    bright_above = bright_mean >= dark_mean
    lower, upper = sorted((dark_mean, bright_mean))
    if not all(isfinite(value) for value in (lower, upper, dark_sigma, bright_sigma)):
        return 0.5 * (dark_mean + bright_mean), bright_above
    if upper <= lower:
        return 0.5 * (lower + upper), bright_above

    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        lambda threshold: _threshold_error(
            threshold,
            dark_mean,
            dark_sigma,
            bright_mean,
            bright_sigma,
            bright_above,
        ),
        bounds=(lower, upper),
        method="bounded",
    )
    threshold = result.x if result.success else 0.5 * (lower + upper)
    return float(threshold), bright_above


def gaussian_fidelity(
    dark_mean: float,
    dark_sigma: float,
    bright_mean: float,
    bright_sigma: float,
    threshold: float,
    bright_above: bool = True,
) -> tuple[float, float, float]:
    """Return dark, bright, and balanced classification fidelity."""

    values = (dark_mean, dark_sigma, bright_mean, bright_sigma, threshold)
    if not np.isfinite(values).all():
        return float("nan"), float("nan"), float("nan")
    if bright_above:
        dark = float(normal_cdf(threshold, dark_mean, dark_sigma))
        bright = 1.0 - float(normal_cdf(threshold, bright_mean, bright_sigma))
    else:
        dark = 1.0 - float(normal_cdf(threshold, dark_mean, dark_sigma))
        bright = float(normal_cdf(threshold, bright_mean, bright_sigma))
    return dark, bright, 0.5 * (dark + bright)


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


def _one_sided_core_stats(
    samples: np.ndarray,
    side: str,
    sigma_floor: float,
) -> tuple[float, float, bool]:
    finite = np.asarray(samples, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size < 4:
        return float("nan"), float("nan"), False
    q16, q50, q84 = np.percentile(finite, [15.865525393145708, 50.0, 84.1344746068543])
    sigma = float(q50 - q16) if side == "low" else float(q84 - q50)
    alternative = 0.5 * float(q84 - q16)
    if not isfinite(sigma) or sigma <= 0.0:
        sigma = alternative
    if not isfinite(sigma) or sigma <= 0.0:
        median = float(np.median(finite))
        sigma = 1.482602218505602 * float(np.median(np.abs(finite - median)))
    return float(q50), max(float(sigma), sigma_floor, _SIGMA_FLOOR), True


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
    """Fit robust one-sided Gaussian summaries to a two-state sample."""

    samples = np.asarray(values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    split = _exact_otsu_threshold(samples)
    if samples.size < 8 or not isfinite(split):
        return BimodalFit(split, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, True, False)
    low, high = samples[samples <= split], samples[samples > split]
    minimum = max(4, int(np.ceil(min_component_fraction * samples.size)))
    if low.size < minimum or high.size < minimum:
        return BimodalFit(split, np.nan, np.nan, np.nan, np.nan, np.nan, high.size / samples.size, np.nan, np.nan, True, False)
    floor = max(1e-6 * float(np.std(samples)), 1e-12)
    dark_mean, dark_sigma, dark_ok = _one_sided_core_stats(low, "low", floor)
    bright_mean, bright_sigma, bright_ok = _one_sided_core_stats(high, "high", floor)
    fraction = float(high.size / samples.size)
    if not dark_ok or not bright_ok or not np.isfinite((dark_mean, dark_sigma, bright_mean, bright_sigma)).all() or bright_mean <= dark_mean:
        return BimodalFit(split, np.nan, dark_mean, dark_sigma, bright_mean, bright_sigma, fraction, np.nan, np.nan, True, False)
    threshold, bright_above = optimal_gaussian_threshold(dark_mean, dark_sigma, bright_mean, bright_sigma)
    dark_fidelity, bright_fidelity, fidelity = gaussian_fidelity(dark_mean, dark_sigma, bright_mean, bright_sigma, threshold, bright_above)
    separation = (bright_mean - dark_mean) / max(dark_sigma + bright_sigma, _SIGMA_FLOOR)
    return BimodalFit(threshold, fidelity, dark_mean, dark_sigma, bright_mean, bright_sigma, fraction, dark_fidelity, bright_fidelity, bright_above, bool(separation > 0.5))


def per_site_fidelity(
    signals: object,
    labels: object,
    thresholds: object,
    *,
    test_mask: object | None = None,
    valid_mask: object | None = None,
) -> "PerSiteConfusion":
    """Return the held-out confusion, per site, as all three of its numbers.

    Balanced, dark and bright together, because they are one measurement.  The
    caller used to compute the dark and bright halves with a second
    implementation that applied a different validity mask, so the reported
    balanced fidelity was not the mean of the reported halves -- two answers to
    one question, on the number a readout is judged by.
    """

    values = np.asarray(signals, dtype=float)
    truth = np.asarray(labels, dtype=bool)
    boundary = np.asarray(thresholds, dtype=float).reshape(-1)
    if values.ndim != 2 or truth.shape != values.shape or boundary.shape != (values.shape[1],):
        raise ValueError("signals/labels must be (shots, sites) and thresholds must be (sites,)")
    prediction = values > boundary[None, :]
    valid = np.isfinite(values) & np.isfinite(boundary)[None, :]
    if test_mask is not None:
        test = np.asarray(test_mask, dtype=bool)
        if test.shape != values.shape:
            raise ValueError("test_mask must match signals shape")
        valid &= test
    if valid_mask is not None:
        measured = np.asarray(valid_mask, dtype=bool)
        if measured.shape != values.shape:
            raise ValueError("valid_mask must match signals shape")
        valid &= measured
    balanced = np.full(values.shape[1], np.nan, dtype="<f8")
    dark_out = np.full(values.shape[1], np.nan, dtype="<f8")
    bright_out = np.full(values.shape[1], np.nan, dtype="<f8")
    tested = np.zeros(values.shape[1], dtype=int)
    for site in range(values.shape[1]):
        selected = valid[:, site]
        tested[site] = int(np.count_nonzero(selected))
        if not np.any(selected):
            continue
        actual = truth[selected, site]
        predicted = prediction[selected, site]
        dark = int(np.count_nonzero(~actual))
        bright = int(np.count_nonzero(actual))
        if not dark or not bright:
            continue
        dark_out[site] = float(np.count_nonzero(~predicted & ~actual)) / dark
        bright_out[site] = float(np.count_nonzero(predicted & actual)) / bright
        balanced[site] = 0.5 * (dark_out[site] + bright_out[site])
    return PerSiteConfusion(balanced, dark_out, bright_out, tested)


@dataclass(frozen=True)
class PerSiteConfusion:
    """One held-out confusion per site: the balanced figure and its two halves."""

    balanced: np.ndarray
    dark: np.ndarray
    bright: np.ndarray
    #: How many held-out shots each site was judged on.
    tested: np.ndarray


__all__ = [
    "BimodalFit",
    "PerSiteConfusion",
    "confidence_weighted_fidelity",
    "finite_mean",
    "fit_bimodal",
    "gaussian",
    "gaussian_fidelity",
    "normal_cdf",
    "optimal_gaussian_threshold",
    "per_site_fidelity",
]
