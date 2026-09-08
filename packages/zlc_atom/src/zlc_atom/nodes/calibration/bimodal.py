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

#: Two populations are accepted only on DECISIVE evidence: a BIC gain over
#: ten against one Gaussian (Kass and Raftery's "very strong").  A loaded
#: site clears it by hundreds -- four bright shots in a hundred, the fitter's
#: own population floor, at an archived run's contrast already do -- while a
#: dark site whose one Gaussian the fitter split in two, 1.7 sigma apart,
#: came in at +4.6 and was reported to the SLM feedback loop as loaded with
#: a contrast of 10.9 photoelectrons, one hundred times under its
#: neighbours: the uniformity ratio read 116 and the observable count 33.
#: The calibration asks the same question of a site's reference frames: one
#: that never loaded has no two states to label them by.
_DECISIVE_BIC_GAIN = 10.0

#: What keeps a "state" off a handful of samples.  A Gaussian mixture's
#: likelihood is unbounded: a component narrowed onto a few shots that
#: happen to lie close together gains about log(r) per shot, r being how
#: much narrower it is than its partner, and on a sixty-shot sample whose
#: two states overlap, six shots within half a unit did exactly that --
#: a state of width 0.1 beside one of width 7, threshold on the spike.
#: Ranking the candidate pairs by likelihood less 2 (log r)^2 (a Gaussian
#: of this width on log r) makes a ratio r cost more than fewer than
#: 2 log r shots can buy -- nine at r = 72, six at r = 20 -- and less than
#: any population pays back: sixty dark shots that ARE ten times narrower
#: than the bright ones gain hundreds.  It is a cost on the ratio, not a
#: bound: the widths remain the sample's own.
_LOG_WIDTH_RATIO_SCALE = 0.5


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
        # Cancellation-free roots.  ``b`` is half the sum of two inverse
        # variances and always positive, so ``-b - sqrt(D)`` is a sum of
        # like signs; the other root is the product of roots, ``c/a``,
        # divided by it.  Forming that root as ``(-b + sqrt(D)) / 2a``
        # subtracted two nearly equal numbers whenever the two widths
        # nearly agreed (``a`` -> 0) and handed back a threshold the two
        # weighted curves do not cross at -- 1.80 for 1.8986, on widths
        # differing by one part in 1e15 -- while the exactly-equal case
        # took the linear branch and was right.
        q = -0.5 * (b + sqrt(max(discriminant, 0.0)))
        roots = (q / a, c / q)
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
    wider or narrower than the other: real technical noise can make either
    conditional distribution the narrower one, and how far apart their widths
    sit is whatever the shots say -- bright shot noise over dark read noise
    is a factor of ten on one camera and two on another.
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
        current = _log_likelihood(values, means, sigmas, weights)
        if not isfinite(current):
            return means, sigmas, weights, -np.inf
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


def _log_likelihood(
    values: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray,
) -> float:
    """The log-likelihood of ``values`` under a weighted pair of Gaussians;
    minus infinity where the pair gives some value no density at all."""

    scaled = (values[:, None] - means[None, :]) / sigmas[None, :]
    total = np.sum(
        weights[None, :]
        * np.exp(-0.5 * scaled**2)
        / (sigmas[None, :] * sqrt(2.0 * pi)),
        axis=1,
    )
    if not np.all(np.isfinite(total)) or np.any(total <= 0.0):
        return -np.inf
    return float(np.sum(np.log(total)))


def _bic_gain(values: np.ndarray, log_two: float) -> float:
    """The BIC gain of the fitted pair (log-likelihood ``log_two``, five
    parameters) over one Gaussian on the same values (two parameters).
    Positive favours two populations."""

    one_sigma = max(float(np.std(values)), np.finfo(float).tiny)
    one_mean = float(np.mean(values))
    log_one = float(
        np.sum(
            -0.5 * np.square((values - one_mean) / one_sigma)
            - np.log(one_sigma * sqrt(2.0 * pi))
        )
    )
    penalty = (5.0 - 2.0) * log(values.size)
    return float(2.0 * (log_two - log_one) - penalty)


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
    #: The evidence for two populations over one: the BIC gain of the fitted
    #: pair against a single Gaussian on the same shots.
    bic_gain: float
    #: Two states far enough apart, and both populated enough, for a shot to
    #: be assigned to one of them.
    ok: bool

    @property
    def decisive(self) -> bool:
        """``ok`` on decisive evidence that two Gaussians beat one: the site
        loaded.  What a control action, or a label, may rest on; a threshold
        for a site known to load needs only ``ok``, since two populations
        that overlap can be as real as the evidence for them is weak."""

        return bool(
            self.ok and isfinite(self.bic_gain) and self.bic_gain > _DECISIVE_BIC_GAIN
        )


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
    EM-refined candidates; the likeliest pair wins, less the cost on their
    width ratio that keeps a state off a handful of samples
    (``_LOG_WIDTH_RATIO_SCALE``).  The widths are otherwise the sample's
    own: a dark state that is read noise beside a bright state that is shot
    noise sit a factor of ten apart on a qCMOS, and a rule that preferred
    any pair inside a fixed width ratio to the likelier one outside it
    handed the calibration figure a forty-wide "dark" Gaussian over a
    three-wide spike, and the feedback loop the threshold that went with
    it.  A state also holds at least ``min_component_fraction`` of the
    shots, and two, and is at least one percent of the spread wide.

    ``ok`` says whether the shots are two states far enough apart, and both
    populated enough, for a shot to be assigned to one of them; ``decisive``
    adds the evidence that two Gaussians beat one at all (``bic_gain`` over
    ``_DECISIVE_BIC_GAIN``).  A site that never loaded has its one Gaussian
    split in two like any other sample, and the evidence is what says it
    did not; two populations that overlap at sixty shots are ``ok`` and not
    ``decisive``, and their crossing is still the best threshold there is.
    Every number is returned either way.
    """

    samples = np.asarray(values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size < 4:
        split = _exact_otsu_threshold(samples) if samples.size else float("nan")
        return BimodalFit(
            threshold=split,
            fidelity=np.nan,
            dark_mean=np.nan,
            dark_sigma=np.nan,
            bright_mean=np.nan,
            bright_sigma=np.nan,
            bright_fraction=np.nan,
            dark_fidelity=np.nan,
            bright_fidelity=np.nan,
            bright_above=True,
            bic_gain=np.nan,
            ok=False,
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

    best: tuple[float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for split in starts:
        start = _split_start(
            samples, split, sigma_min=sigma_min, weight_min=weight_min
        )
        if start is None:
            continue
        refined = _em_two_state(
            samples,
            *(np.array(value, copy=True) for value in start),
            sigma_min=sigma_min,
            weight_min=weight_min,
        )
        for likelihood, means, sigmas, weights in (
            (_log_likelihood(samples, *start), *start),
            (refined[3], *refined[:3]),
        ):
            if not isfinite(likelihood):
                continue
            ratio = float(np.max(sigmas)) / max(float(np.min(sigmas)), _SIGMA_FLOOR)
            score = likelihood - 0.5 * (log(ratio) / _LOG_WIDTH_RATIO_SCALE) ** 2
            if best is None or score > best[0]:
                best = (score, likelihood, means, sigmas, weights)

    if best is None:
        split = _exact_otsu_threshold(samples)
        low, high = samples[samples <= split], samples[samples > split]
        dark_mean = float(np.mean(low)) if low.size else float(np.mean(samples))
        bright_mean = float(np.mean(high)) if high.size else dark_mean + spread
        width = max(0.5 * spread, sigma_min)
        best = (
            float("nan"),
            float("nan"),
            np.array([dark_mean, bright_mean]),
            np.array([width, width]),
            np.array([0.5, 0.5]),
        )

    _score, likelihood, means, sigmas, weights = best
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
    bic_gain = _bic_gain(samples, likelihood) if isfinite(likelihood) else float("nan")
    separation = (bright_mean - dark_mean) / max(
        dark_sigma + bright_sigma, _SIGMA_FLOOR
    )
    minimum = max(4.0, float(min_component_fraction) * samples.size) / samples.size
    # Two states far enough apart, and both populated enough, for a shot to
    # be assigned to one of them.  Width is not what this sentence is about.
    separated = bool(
        np.isfinite(threshold)
        and separation > 0.5
        and minimum <= fraction <= 1.0 - minimum
    )
    return BimodalFit(
        threshold=threshold,
        fidelity=fidelity,
        dark_mean=dark_mean,
        dark_sigma=dark_sigma,
        bright_mean=bright_mean,
        bright_sigma=bright_sigma,
        bright_fraction=fraction,
        dark_fidelity=dark_fidelity,
        bright_fidelity=bright_fidelity,
        bright_above=bright_above,
        bic_gain=bic_gain,
        ok=separated,
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
