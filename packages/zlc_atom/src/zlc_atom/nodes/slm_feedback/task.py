"""Single-frame, multi-shot bright-dark fluorescence feedback."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import special
from zlc_data import (
    COMPONENT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
)
from zlc_data.snapshot_projection import (
    restricted_values,
    selection_indices,
    value_selection,
)
from zlc_durable import atomic_write_file, atomic_write_text, write_readable_json
from zlc_pulse import PulseSequence
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    IMAGE_POINT_OVERLAY_GEOMETRY_RECORD,
    ImageFrame,
    ImagePointOverlay,
    ImagePlot,
    PlotLabels,
    PointStatus,
    image_point_overlay_geometry,
)
from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput, MonitorCoverage

from zlc_atom.data import snapshot_from_array
from zlc_atom.devices.slm import SlmAdapter, canonical_phase
from zlc_atom.devices.sequencer import sequencer_archive_snapshot
from zlc_atom.devices.slm.solver import (
    SCIENCE_CONTEXT_ARTIFACT_CONTRACT,
    compose_science_phase,
    freeze_pattern_phase,
    save_science_context,
    solve_phase,
    validate_target,
)
from zlc_atom.nodes.calibration import (
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
    extract_box_signals,
    extract_psf_signals,
    fit_bimodal,
)
from zlc_atom.nodes.calibration.calibration import (
    _register_target_sites,
    validate_target_registration,
)
from zlc_atom.nodes.calibration.pulse import arm_sequencer, resolve_pulse
from zlc_atom.nodes.camera_measurement.measurement import (
    CAMERA_FRAMES_OUTPUT,
    CameraMeasurementNode,
    CameraMeasurementRequest,
    _finite_cycle_output,
)
from zlc_atom.nodes.scan.source import wait_for_report


SLM_PHASE_ARTIFACT_CONTRACT = SCIENCE_CONTEXT_ARTIFACT_CONTRACT
_FEEDBACK_MEASUREMENT_CHECKPOINT_CONTRACT = "zlc.slm.feedback-measurement"
CANDIDATE_PHASE_OUTPUT = DatasetOutputDeclaration(
    "candidate_phase", "slm-feedback.candidate-phase"
)
UNIFORMITY_HISTORY_OUTPUT = DatasetOutputDeclaration(
    "uniformity_history", "slm-feedback.uniformity-history"
)
OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT = DatasetOutputDeclaration(
    "observable_uniformity_history",
    "slm-feedback.observable-uniformity-history",
)
SITE_SIGNAL_HISTORY_OUTPUT = DatasetOutputDeclaration(
    "site_signal_history", "slm-feedback.site-signal-history"
)
TARGET_SHARE_HISTORY_OUTPUT = DatasetOutputDeclaration(
    "target_share_history", "slm-feedback.target-share-history"
)
_CONTROLLER_CONTRACT = "slm-feedback.qcmos-bright-dark"
READOUT_FRAME_COORDINATE = 0
#: Formal candidates in a row whose split-half dispersion is within its own
#: standard error of zero before the run declares itself converged.
_CONVERGENCE_CANDIDATES = 3
#: The feedback re-solve is a small perturbation of a converged phase, so it
#: is held to a far tighter support gate than an editor solve: the default
#: 1% early stop left a fresh ~0.2% rms site-intensity pattern each candidate,
#: three times the loop's own step (see ``solve_phase``).
_FEEDBACK_SOLVE_SUPPORT_TOLERANCE = 1.002
_FEEDBACK_SOLVE_MINIMUM_ITERATIONS = 5


def _check_cancelled(context: object) -> None:
    if context.cancel_requested():
        raise RuntimeError("SLM feedback was cancelled")


def _accepted_pulse_fault(
    report: object, *, delivered_cycles: int, requested_cycles: int
) -> bool:
    """Whether a shot batch the board REPORTED as faulted is still a good batch.

    The rule: keep the batch when the only thing that failed is the host's
    OBSERVATION of the board -- the pulse observer's own UART poll -- and the
    camera delivered every requested cycle.  Every cycle plays exactly one
    trigger edge and yields exactly one frame, so a full frame count is the
    board's own proof that the whole batch played; an underflow the observer
    saw before it died is a board fact and keeps the batch refused.

    The failure this fixes: one CURSOR poll lost its last byte at shot 85 of
    200, the observer thread died and stamped the report with a fabricated
    ERROR status, the board played the remaining 115 shots on its own and the
    camera handed over all 200 frames -- and the whole batch, then the whole
    four-hour run, was thrown away on that one poll.

    Only the fault text and the ``observer_error``/``underflow`` fields are
    read: the observer clause must be the whole fault (no second ``;``-joined
    reason next to it), so a report that learns to say more is refused rather
    than misread.
    """

    observer_error = str(getattr(report, "observer_error", "") or "")
    fault = str(report.fault or "")
    if not observer_error or observer_error not in fault:
        return False
    if ";" in fault.replace(observer_error, ""):
        return False
    if bool(getattr(report, "underflow", False)):
        return False
    return int(delivered_cycles) == int(requested_cycles)


def _readout_frames(snapshot: object, *, shots: int) -> np.ndarray:
    """Select the sole frame from one sealed single-frame Camera dataset."""

    selection = value_selection(
        snapshot.block.schema,
        {"frame": READOUT_FRAME_COORDINATE},
    )
    repeat_indices, point_indices, data_indices = selection_indices(
        snapshot.block.schema,
        selection,
    )
    selected = restricted_values(
        snapshot.block.values,
        snapshot.block.schema,
        repeat_indices,
        point_indices,
        data_indices,
    )
    validity = restricted_values(
        snapshot.expanded_validity(),
        snapshot.block.schema,
        repeat_indices,
        point_indices,
        data_indices,
    )
    if selected.shape[:2] != (int(shots), 1):
        raise RuntimeError("Camera Measurement readout projection changed shape")
    if not bool(np.all(validity[:, 0])):
        raise RuntimeError("Camera Measurement readout projection is incomplete")
    return selected[:, 0]


def _ratio_interval(
    values: np.ndarray,
    standard_error: np.ndarray,
) -> tuple[float, float, float, float]:
    measured = np.asarray(values, dtype=float)
    error = np.asarray(standard_error, dtype=float)
    if (
        measured.ndim != 1
        or error.shape != measured.shape
        or not np.all(np.isfinite(measured))
        or not np.all(np.isfinite(error))
        or np.any(measured <= 0.0)
        or np.any(error < 0.0)
    ):
        raise ValueError("bright-dark contrast and uncertainty must be finite and positive")
    relative = error / measured
    z = float(
        special.ndtri(1.0 - 0.05 / (2.0 * len(measured)))
    )
    logarithm = np.log(measured)
    estimate = float(np.exp(np.max(logarithm) - np.min(logarithm)))
    lower = float(
        np.exp(np.max(logarithm - z * relative) - np.min(logarithm + z * relative))
    )
    upper = float(
        np.exp(np.max(logarithm + z * relative) - np.min(logarithm - z * relative))
    )
    return estimate, max(1.0, lower), upper, float(np.max(relative))


def _half_contrasts(
    samples: object, threshold: object
) -> tuple[np.ndarray, np.ndarray]:
    """Bright-minus-dark of the odd shots and of the even shots, per site.

    Both halves classify their shots with the WHOLE batch's fitted threshold,
    so they are two independent readings of the same quantity the full fit
    reports.  Their agreement is the only thing in a shot batch that can tell
    real site-to-site dispersion from estimator noise: noise is independent
    between the halves, true dispersion is shared by both.  A half with fewer
    than two shots on either side of the threshold gives NaN.
    """

    values = np.asarray(samples, dtype=float)
    cut = np.asarray(threshold, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != cut.shape[0]:
        raise ValueError("half-batch contrasts need (shots, sites) samples")
    halves = []
    for start in (0, 1):
        half = values[start::2]
        finite = np.isfinite(half)
        bright = finite & (half > cut[None, :])
        dark = finite & ~bright
        bright_count = np.count_nonzero(bright, axis=0)
        dark_count = np.count_nonzero(dark, axis=0)
        usable = (bright_count >= 2) & (dark_count >= 2) & np.isfinite(cut)
        contrast = np.full(cut.shape, np.nan, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            bright_mean = np.sum(np.where(bright, half, 0.0), axis=0) / bright_count
            dark_mean = np.sum(np.where(dark, half, 0.0), axis=0) / dark_count
        contrast[usable] = bright_mean[usable] - dark_mean[usable]
        halves.append(contrast)
    return halves[0], halves[1]


def _split_half_dispersion(
    odd_contrast: object, even_contrast: object, valid: object
) -> tuple[float, float]:
    """The TRUE between-site variance of log contrast, and its standard error.

    ``max/min`` of N noisy estimates never reaches 1: with 35 sites at 1.2%
    relative error its expectation is 1.054 when the array is perfectly
    uniform, so a controller judged by it keeps chasing noise for ever.  The
    cross-covariance of the two half-batch residuals is unbiased for the real
    dispersion, because the halves share the truth and not the noise; it can
    come out negative, and a value within its own standard error of zero is
    the honest statement "no dispersion is resolved by this batch".
    """

    odd = np.asarray(odd_contrast, dtype=float)
    even = np.asarray(even_contrast, dtype=float)
    usable = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(odd) & np.isfinite(even)
        & (odd > 0.0) & (even > 0.0)
    )
    count = int(np.count_nonzero(usable))
    if count < 3:
        return float("nan"), float("nan")
    first = np.log(odd[usable])
    second = np.log(even[usable])
    products = (first - np.mean(first)) * (second - np.mean(second))
    variance = float(np.sum(products) / (count - 1))
    error = float(np.std(products, ddof=1) * np.sqrt(count) / (count - 1))
    return variance, error


def _expected_noise_ratio(
    relative_error: object, valid: object, *, draws: int = 4096
) -> float:
    """The max/min a perfectly uniform array would show with these errors."""

    sigma = np.asarray(relative_error, dtype=float)[np.asarray(valid, dtype=bool)]
    sigma = sigma[np.isfinite(sigma) & (sigma >= 0.0)]
    if sigma.size < 2:
        return float("nan")
    noise = np.random.default_rng(0).standard_normal((int(draws), sigma.size)) * sigma
    return float(np.mean(np.exp(np.max(noise, axis=1) - np.min(noise, axis=1))))


_PLANT_SLOPE_LAGS = 3
_PLANT_SLOPE_MINIMUM_CANDIDATES = 3
_PLANT_SLOPE_RELATIVE_ERROR = 0.3
_PLANT_SLOPE_BOUNDS = (0.3, 5.0)
#: The identification excitation: the first this many formal updates after
#: the baseline carry a fresh zero-sum +-2% log-weight pattern on top of the
#: controller's own step, and that pattern is the only instrument the plant
#: slope is read through.  Six at 2% resolve a unit-slope plant to under 30%
#: in 95% of simulated runs at the archived run's 1.2% read noise; the
#: archived plant itself (-3.3) needs three.  A fixed count, never "until
#: trusted": stopping on the estimate's own size selects the runs where it
#: came out large (12-16% bias in simulation).
_PLANT_EXCITATION_CANDIDATES = 6
_PLANT_EXCITATION_LOG_STEP = 0.02


def _excitation_pattern(
    rng: np.random.Generator, excitable: np.ndarray
) -> np.ndarray:
    """One excited candidate's zero-sum +-δ log-weight pattern on the
    ``excitable`` sites; a held site (invalid, unobservable) gets none, as it
    would answer with no usable row and its share is to stay where it is.

    Fresh random signs every candidate: the three lag columns are then driven
    at every frequency and can be told apart.  A pattern that merely flipped
    each candidate would excite only the alternating response and measure
    ``s0 - s1 + s2`` -- 0.7 for the archived plant whose static slope is 3.3.
    """

    mask = np.asarray(excitable, dtype=bool)
    pattern = np.zeros(mask.shape, dtype=float)
    count = int(np.count_nonzero(mask))
    if count >= 2:
        signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
        values = _PLANT_EXCITATION_LOG_STEP * rng.permutation(signs)
        pattern[mask] = values - float(np.mean(values))
    return pattern


def _excited_target(
    target: np.ndarray,
    excitation: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """The Target the phase is solved for: the control Target times ``exp``
    of this candidate's excitation on its sites.

    The control Target stays the integrator the controller updates; the
    excitation is applied on top for one candidate and is gone from the next
    one's Target unless a new pattern is drawn, so it is never integrated.
    """

    updated = np.array(target, dtype=np.float32, copy=True)
    updated[rows, columns] = (
        np.asarray(target[rows, columns], dtype=float)
        * np.exp(np.asarray(excitation, dtype=float))
    ).astype(np.float32)
    return validate_target(updated)


def _plant_slope(
    log_weights: list[np.ndarray],
    log_contrast: list[np.ndarray],
    excitation: list[np.ndarray],
) -> tuple[float, float, int]:
    """Pool every site's response to the EXCITATION into ONE plant slope.

    Model: ``Δlog C_t = s0 Δlog w_t + s1 Δlog w_{t-1} + s2 Δlog w_{t-2}``
    per site, with common-mode drift removed by centring each candidate's
    rows across sites; the static slope is ``s0 + s1 + s2``.  The old
    per-site rule needed a single site to move by 2% before it would look,
    which a 0.35% typical step never did, and read a slope of -1 into a plant
    that actually answers with about -3.3 spread over two candidates.

    The three lag regressors are instrumented by the excitation differences
    at the same lags.  Nothing else in a closed loop is exogenous: the
    controller's step is its reaction to the previous batch, and that batch
    carries not only estimator noise but the plant's own per-candidate wander
    (1.1-1.5% between archived candidates, gone again the next one).  The
    half-batch instrument this replaces was blind to exactly that wander --
    both halves share it -- and in simulation read -2.6 into a unit plant
    and -5.4 into the archived one while reporting a standard error of 0.2.
    The excitation is drawn by the task itself, so it shares nothing with
    the plant but the response.

    A transition no excitation touches contributes nothing and is left out,
    so the estimate is frozen once the excitation ends and the residual
    variance is that of the excited rows.  Returns the signed slope, its
    standard error and the row count; NaN slope when nothing is estimable.
    Steps before the first candidate are zero, which is the physical truth
    of a settled baseline, not a fill.
    """

    count = len(log_weights)
    if count < 2:
        return float("nan"), float("nan"), 0
    outcomes: list[np.ndarray] = []
    regressors: list[np.ndarray] = []
    instruments: list[np.ndarray] = []
    for index in range(1, count):
        outcome = log_contrast[index] - log_contrast[index - 1]
        lagged = [
            log_weights[index - lag] - log_weights[index - lag - 1]
            if index - lag >= 1
            else np.zeros_like(outcome)
            for lag in range(_PLANT_SLOPE_LAGS)
        ]
        excited = [
            excitation[index - lag] - excitation[index - lag - 1]
            if index - lag >= 1
            else np.zeros_like(outcome)
            for lag in range(_PLANT_SLOPE_LAGS)
        ]
        usable = np.isfinite(outcome) & np.all(
            np.isfinite(np.stack(lagged)), axis=0
        )
        if np.count_nonzero(usable) < 2 or not np.any(
            np.stack(excited)[:, usable] != 0.0
        ):
            continue
        design = np.column_stack([column[usable] for column in lagged])
        design -= np.mean(design, axis=0, keepdims=True)
        rows = np.column_stack([column[usable] for column in excited])
        rows -= np.mean(rows, axis=0, keepdims=True)
        outcomes.append(outcome[usable] - np.mean(outcome[usable]))
        regressors.append(design)
        instruments.append(rows)
    if not outcomes:
        return float("nan"), float("nan"), 0
    y = np.concatenate(outcomes)
    x = np.concatenate(regressors)
    z = np.concatenate(instruments)
    rows_used = int(len(y))
    if rows_used <= _PLANT_SLOPE_LAGS:
        return float("nan"), float("nan"), rows_used
    # The pseudo-inverse keeps the estimate defined while a lag column is
    # still identically zero (no step that old has been applied yet): that
    # coefficient is zero and the static sum is the estimable part.  The
    # cut-off is far above float noise so a direction the data has not
    # excited reads as zero rather than as a 1e29 with a 1e42 error.
    cross = z.T @ x
    inverse = np.linalg.pinv(cross, rcond=1e-6)
    beta = inverse @ (z.T @ y)
    residual = y - x @ beta
    sigma_squared = float(residual @ residual) / (rows_used - _PLANT_SLOPE_LAGS)
    covariance = sigma_squared * inverse @ (z.T @ z) @ inverse.T
    ones = np.ones(_PLANT_SLOPE_LAGS)
    slope = float(ones @ beta)
    error = float(np.sqrt(max(float(ones @ covariance @ ones), 0.0)))
    if not np.isfinite(slope) or not np.isfinite(error):
        return float("nan"), float("nan"), rows_used
    return slope, error, rows_used


def _usable_plant_slope(
    candidates: int, slope: float, error: float
) -> float | None:
    """The step divisor the controller may use, or None for the assumed plant.

    An estimate is trusted once at least three candidates exist, it has the
    physical sign (more weight, deeper trap, less contrast) and its standard
    error is under 30% of it; the magnitude is then held to [0.3, 5].  Without
    one the controller assumes unit slope at HALF the authored loop gain --
    the archived run showed the real plant answering three times harder than
    assumed, and the safe side of not knowing is the small step.
    """

    if (
        candidates < _PLANT_SLOPE_MINIMUM_CANDIDATES
        or not np.isfinite(slope)
        or not np.isfinite(error)
        or slope >= 0.0
        or error >= _PLANT_SLOPE_RELATIVE_ERROR * abs(slope)
    ):
        return None
    lower, upper = _PLANT_SLOPE_BOUNDS
    return float(np.clip(abs(slope), lower, upper))


def _bic_gain(samples: object, fit: object) -> float:
    column = np.asarray(samples, dtype=float).reshape(-1)
    if column.size < 4 or not np.all(np.isfinite(column)):
        return float("nan")
    one_sigma = max(float(np.std(column)), np.finfo(float).tiny)
    one_mean = float(np.mean(column))
    log_one = float(np.sum(
        -0.5 * np.square((column - one_mean) / one_sigma)
        - np.log(one_sigma * np.sqrt(2.0 * np.pi))
    ))
    try:
        dark_mean = float(fit.dark_mean)
        dark_sigma = float(fit.dark_sigma)
        bright_mean = float(fit.bright_mean)
        bright_sigma = float(fit.bright_sigma)
        fraction = float(fit.bright_fraction)
    except (AttributeError, TypeError, ValueError):
        return float("nan")
    if (
        not all(np.isfinite(value) for value in (
            dark_mean, dark_sigma, bright_mean, bright_sigma, fraction
        ))
        or dark_sigma <= 0.0
        or bright_sigma <= 0.0
        or not 0.0 < fraction < 1.0
    ):
        return float("nan")
    dark_density = (
        np.exp(-0.5 * np.square((column - dark_mean) / dark_sigma))
        / (dark_sigma * np.sqrt(2.0 * np.pi))
    )
    bright_density = (
        np.exp(-0.5 * np.square((column - bright_mean) / bright_sigma))
        / (bright_sigma * np.sqrt(2.0 * np.pi))
    )
    mixture = (1.0 - fraction) * dark_density + fraction * bright_density
    if not np.all(np.isfinite(mixture)) or np.any(mixture <= 0.0):
        return float("nan")
    log_two = float(np.sum(np.log(mixture)))
    return float(
        2.0 * log_two
        - 5.0 * np.log(column.size)
        - (2.0 * log_one - 2.0 * np.log(column.size))
    )


#: Two populations are accepted only on DECISIVE evidence: a BIC gain over
#: ten (Kass and Raftery's "very strong").  A loaded site clears it by
#: hundreds -- four bright shots in a hundred, the fitter's own population
#: floor, at the archived run's contrast already do -- while a dark site
#: whose one Gaussian the fitter split in two
#: 1.7 sigma apart came in at +4.6 and was reported to the loop as loaded
#: with a contrast of 10.9 photoelectrons, one hundred times under its
#: neighbours: the uniformity ratio read 116 and the observable count 33.
_DECISIVE_BIC_GAIN = 10.0

#: A loaded site whose bright fraction is under this share of the lattice's
#: typical one is ON ITS LOADING RAMP: loading probability rises from zero
#: over the last few percent of depth above the loading threshold, so half
#: the typical loading means the site's edge is within about one resolution
#: below it.  The bright fraction is the loading-margin observable the
#: binary loaded/dark verdict throws away; it is what tells the controller
#: which loaded sites have no share to give.
_LOADING_EDGE_FRACTION = 0.5


def _loading_edge(bright_fraction: object, observable: object) -> np.ndarray:
    """The loaded sites on their loading ramp (see ``_LOADING_EDGE_FRACTION``)."""

    fraction = np.asarray(bright_fraction, dtype=float)
    loaded = np.asarray(observable, dtype=bool)
    if fraction.shape != loaded.shape:
        raise ValueError("bright fraction and observable shapes differ")
    known = loaded & np.isfinite(fraction)
    if not np.any(known):
        return np.zeros(loaded.shape, dtype=bool)
    typical = float(np.median(fraction[known]))
    return known & (fraction < _LOADING_EDGE_FRACTION * typical)


def _fit_contrasts(samples: object) -> dict[str, np.ndarray]:
    """Classify each site's one user-authored shot batch.

    A decisively resolved two-population fit (see ``_DECISIVE_BIC_GAIN``)
    supplies the bright-minus-dark feedback observable.  Evidence for one
    Gaussian is a different, useful physical result: this feedback mode
    treats it as a site which did not load.  Bad samples or a numerically
    undecidable model remain invalid and therefore cannot create a control
    action.
    """

    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 10 or values.shape[1] < 1:
        raise ValueError("feedback box samples must have shape (shots, sites)")
    sites = values.shape[1]
    contrast = np.full(sites, np.nan, dtype=float)
    error = np.full(sites, np.nan, dtype=float)
    dark_mean = np.full(sites, np.nan, dtype=float)
    dark_sigma = np.full(sites, np.nan, dtype=float)
    dark_standard_error = np.full(sites, np.nan, dtype=float)
    bright_mean = np.full(sites, np.nan, dtype=float)
    bright_sigma = np.full(sites, np.nan, dtype=float)
    bright_fraction = np.full(sites, np.nan, dtype=float)
    threshold = np.full(sites, np.nan, dtype=float)
    fidelity = np.full(sites, np.nan, dtype=float)
    bic_gain = np.full(sites, np.nan, dtype=float)
    single_mean = np.full(sites, np.nan, dtype=float)
    single_sigma = np.full(sites, np.nan, dtype=float)
    separated = np.zeros(sites, dtype=bool)
    single_population = np.zeros(sites, dtype=bool)
    for site in range(sites):
        column = np.asarray(values[:, site], dtype=float)
        if column.size < 4 or not np.all(np.isfinite(column)):
            continue
        one_mean = float(np.mean(column))
        one_sigma = float(np.std(column))
        if not np.isfinite(one_mean) or not np.isfinite(one_sigma):
            continue
        single_mean[site] = one_mean
        single_sigma[site] = one_sigma
        fit = fit_bimodal(column, min_component_fraction=0.01)
        dark_mean[site] = fit.dark_mean
        dark_sigma[site] = fit.dark_sigma
        bright_mean[site] = fit.bright_mean
        bright_sigma[site] = fit.bright_sigma
        bright_fraction[site] = fit.bright_fraction
        threshold[site] = fit.threshold
        fidelity[site] = fit.fidelity
        if not all(
            np.isfinite(value)
            for value in (
                fit.dark_mean,
                fit.dark_sigma,
                fit.bright_mean,
                fit.bright_sigma,
                fit.bright_fraction,
            )
        ):
            continue
        fraction = float(fit.bright_fraction)
        count_bright = max(fraction * column.size, 1.0)
        count_dark = max((1.0 - fraction) * column.size, 1.0)
        dark_standard_error[site] = float(fit.dark_sigma / np.sqrt(count_dark))
        estimate = float(fit.bright_mean - fit.dark_mean)
        sem = float(
            np.sqrt(
                fit.bright_sigma**2 / count_bright
                + fit.dark_sigma**2 / count_dark
            )
        )
        bic_gain[site] = _bic_gain(column, fit)
        finite_pair = bool(
            estimate > 0.0
            and np.isfinite(sem)
            and sem >= 0.0
            and np.isfinite(bic_gain[site])
        )
        if not finite_pair:
            continue
        if fit.ok and bic_gain[site] > _DECISIVE_BIC_GAIN:
            contrast[site], error[site] = estimate, sem
            separated[site] = True
        else:
            single_population[site] = True
    odd_contrast, even_contrast = _half_contrasts(values, threshold)
    odd_contrast[~separated] = np.nan
    even_contrast[~separated] = np.nan
    return {
        "contrast": contrast,
        "standard_error": error,
        "odd_contrast": odd_contrast,
        "even_contrast": even_contrast,
        "dark_mean": dark_mean,
        "dark_sigma": dark_sigma,
        "dark_standard_error": dark_standard_error,
        "bright_mean": bright_mean,
        "bright_sigma": bright_sigma,
        "bright_fraction": bright_fraction,
        "threshold": threshold,
        "fidelity": fidelity,
        "bic_gain": bic_gain,
        "single_mean": single_mean,
        "single_sigma": single_sigma,
        "valid": separated,
        "single_population": single_population,
        "invalid": ~(separated | single_population),
    }


def _json_floats(values: object) -> list[float | None]:
    """Keep strict JSON artifacts readable when a rejected shot was missing."""

    return [float(value) if np.isfinite(value) else None for value in np.asarray(values)]


def _plain_json(value: object) -> object:
    """Freeze the small metadata side of a feedback checkpoint."""

    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("feedback metadata keys must be text")
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    raise TypeError(f"feedback metadata contains unsupported {type(value).__name__}")


def _write_npz(
    path: str | Path,
    *,
    arrays: Mapping[str, object],
    metadata: Mapping[str, object],
) -> Path:
    """Atomically write one current, pickle-free feedback data checkpoint."""

    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("feedback checkpoint array names must be non-empty text")
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise TypeError(f"feedback checkpoint {name!r} cannot use object dtype")
        payload[name] = array
    encoded = json.dumps(
        _plain_json(dict(metadata)),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload["metadata"] = np.asarray(encoded)
    return atomic_write_file(
        selected,
        lambda stream: np.savez_compressed(stream, **payload),
    )


_CANDIDATE_VECTOR_FIELDS = (
    "target_weight",
    "control_weight",
    "excitation_log_step",
    "dark_mean",
    "dark_sigma",
    "dark_standard_error",
    "bright_mean",
    "bright_sigma",
    "fit_threshold",
    "bright_minus_dark",
    "contrast_standard_error",
    "bright_fraction",
    "fit_fidelity",
    "bic_gain",
    "fit_valid",
    "observable_valid",
    "loading_edge",
    "single_population",
    "fit_invalid",
    "single_mean",
    "single_sigma",
    "odd_shot_bright_minus_dark",
    "even_shot_bright_minus_dark",
    "decision",
    "requested_log_correction",
    "previous_double_control_weight",
    "previous_double_bright_minus_dark",
    "probe_effective_factor",
    "probe_selected_formal_factor",
    "probe_decision",
    "probe_single_bound",
    "probe_observable_bound",
    "probe_control_boundary",
)


def _control_weights(values: object) -> np.ndarray:
    """Put positive Target intensities in the solver's relative log gauge."""

    weights = np.asarray(values, dtype=float)
    if (
        weights.ndim != 1
        or not len(weights)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("feedback control weights must be finite and positive")
    return weights * (len(weights) / float(np.sum(weights)))


def _allocate_requested_shares(
    shares: object,
    requested_log_step: object,
    *,
    lower: object | None = None,
    upper: object | None = None,
) -> tuple[np.ndarray, float, float, float]:
    current = np.asarray(shares, dtype=float)
    requested = np.asarray(requested_log_step, dtype=float)
    if (
        current.ndim != 1
        or requested.shape != current.shape
        or not np.all(np.isfinite(current))
        or np.any(current <= 0.0)
        or not np.isclose(float(np.sum(current)), 1.0)
        or not np.all(np.isfinite(requested))
    ):
        raise ValueError("requested share allocation inputs are invalid")
    minimum = (
        np.zeros_like(current) if lower is None else np.asarray(lower, dtype=float)
    )
    maximum = (
        np.full_like(current, np.inf)
        if upper is None else np.asarray(upper, dtype=float)
    )
    if minimum.shape != current.shape or maximum.shape != current.shape:
        raise ValueError("requested share allocation bounds differ")
    desired = current * np.exp(requested)
    increase = np.minimum(
        np.maximum(desired - current, 0.0),
        np.maximum(maximum - current, 0.0),
    )
    decrease = np.minimum(
        np.maximum(current - desired, 0.0),
        np.maximum(current - minimum, 0.0),
    )
    increase_total = float(np.sum(increase))
    decrease_total = float(np.sum(decrease))
    transfer = min(increase_total, decrease_total)
    increase_scale = 0.0 if increase_total == 0.0 else transfer / increase_total
    decrease_scale = 0.0 if decrease_total == 0.0 else transfer / decrease_total
    allocated = current + increase_scale * increase - decrease_scale * decrease
    return allocated, transfer, increase_scale, decrease_scale


def _funded_shares(
    shares: np.ndarray,
    allocated: np.ndarray,
    directed: np.ndarray,
    desired: np.ndarray,
    compensators: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Fund the directed sites' unmet share from the loaded sites in common.

    ``allocated`` is the ordinary allocation; what the directed sites still
    lack of ``desired`` is taken from (or, for a directed decrease, given
    to) every compensating site by ONE common factor, each compensator held
    inside ONE RESOLUTION STEP (``_PLANT_EXCITATION_LOG_STEP``) of its
    current share and inside its own bracket bound.  When the compensators
    cannot cover the whole request within those bounds they give
    everything the bounds allow and the directed sites receive that, pro
    rata, in their own direction.

    The resolution, not ``maximum_weight_change``, bounds a compensator: a
    loaded site's own loading margin is unknown until it has shown an edge,
    and the clamp meant for a site's OWN correction (a third of its share)
    is many times the margin a marginally loaded array has.  Measured on
    the virtual lattice, where that margin is ~4%: funding ten dark sites
    under the clamp pressed twenty-two loaded ones dark in one candidate
    and the run crawled back at one bracket bisection per candidate; at
    the resolution twenty-five loaded sites still hand over half a site's
    share per candidate -- a dark site is lifted in a few candidates --
    and none of them crosses its edge.
    """

    current = np.asarray(shares, dtype=float)
    result = np.array(allocated, dtype=float, copy=True)
    wants = np.asarray(directed, dtype=bool)
    pays = np.asarray(compensators, dtype=bool)
    unmet = np.where(wants, np.asarray(desired, dtype=float) - result, 0.0)
    need = float(np.sum(unmet))
    if need == 0.0 or not np.any(pays):
        return result
    cap = _PLANT_EXCITATION_LOG_STEP
    floor = np.maximum(np.asarray(lower, dtype=float), current * np.exp(-cap))[pays]
    ceiling = np.minimum(np.asarray(upper, dtype=float), current * np.exp(cap))[pays]
    base = result[pays]
    goal = float(np.sum(base)) - need

    def total(log_factor: float) -> float:
        return float(np.sum(np.clip(base * np.exp(log_factor), floor, ceiling)))

    if need > 0.0:
        reachable = float(np.sum(floor))
        low, high = float(np.min(np.log(floor / base))), 0.0
        short = reachable > goal
        limit = floor
    else:
        reachable = float(np.sum(ceiling))
        low, high = 0.0, float(np.max(np.log(ceiling / base)))
        short = reachable < goal
        limit = ceiling
    if short:
        paid = limit
        scale = (float(np.sum(base)) - reachable) / need
    else:
        for _ in range(200):
            middle = 0.5 * (low + high)
            if total(middle) < goal:
                low = middle
            else:
                high = middle
        paid = np.clip(base * np.exp(0.5 * (low + high)), floor, ceiling)
        scale = 1.0
    result[pays] = paid
    result[wants] += scale * unmet[wants]
    return result


def _support(
    target: np.ndarray,
    calibration: TrapCalibration,
    model: ReadoutModel,
    *,
    science_context_path: str | Path,
    command_receipt: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, SiteMap, np.ndarray]:
    """Register the Calibration's sites to this Feedback Target.

    Returns the Target rows and columns, the registered roster, and for every
    roster site the index of the Calibration site it was matched to (``-1``
    for a site the Calibration never observed, whose centre is predicted).
    """

    usable = np.asarray(calibration.site_map.valid_sites, dtype=bool)
    if not np.any(usable):
        raise ValueError("SLM Feedback requires at least one calibrated site")
    source_indices = np.flatnonzero(usable)
    source_map = SiteMap(
        tuple(calibration.site_map.site_ids[index] for index in source_indices),
        np.asarray(calibration.site_map.centers_xy)[source_indices],
        np.ones(len(source_indices), dtype=bool),
        np.asarray(calibration.site_map.quality)[source_indices],
        calibration.site_map.coordinate_frame,
        {},
    )
    registered = _register_target_sites(
        source_map,
        target,
        {
            "science_context_path": str(
                Path(science_context_path).expanduser().resolve()
            ),
            "command_receipt": dict(command_receipt),
        },
        frame_shape=calibration.frame_contract.image_shape,
        measurement_radius=_readout_window_half_width(model),
    )
    support, provenance = validate_target_registration(
        registered,
        frame_shape=calibration.frame_contract.image_shape,
        box_half_width=_readout_window_half_width(model),
    )
    rows, columns = support.T
    if not np.array_equal(support, np.column_stack(np.nonzero(target > 0.0))):
        raise ValueError("registered Calibration support differs from Science Context")
    if provenance["command_receipt"] != dict(command_receipt):
        raise RuntimeError("Feedback registration lost its Science Context receipt")
    observed = np.asarray(registered.topology["observed_sites"], dtype=bool)
    lookup = {
        tuple(center): int(source)
        for center, source in zip(
            np.asarray(source_map.centers_xy, dtype=float).tolist(),
            source_indices.tolist(),
            strict=True,
        )
    }
    source_index = np.full(len(rows), -1, dtype=int)
    for site in np.flatnonzero(observed):
        source_index[site] = lookup[
            tuple(np.asarray(registered.centers_xy, dtype=float)[site].tolist())
        ]
    return rows, columns, registered, source_index


def _readout_window_half_width(model: ReadoutModel) -> int:
    """Every pixel the model's readout touches around a site centre."""

    if model.kind is ReadoutModelKind.BOX or model.background != "annulus":
        return int(model.integration_half_width)
    return int(model.integration_half_width) + int(model.psf_padding)


def _relative_probe_target(
    target: np.ndarray,
    requested_factors: object,
    probe_sites: object,
    rows: np.ndarray,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    factors = np.asarray(requested_factors, dtype=float)
    probe = np.asarray(probe_sites, dtype=bool)
    raw = np.asarray(target[rows, columns], dtype=float)
    shares = raw / float(np.sum(raw))
    effective = np.ones(len(rows), dtype=float)
    if not np.any(probe) or np.all(probe):
        return np.array(target, dtype=np.float32, copy=True), effective
    effective[probe] = factors[probe]
    probe_total = float(np.sum(shares[probe] * effective[probe]))
    while probe_total >= 1.0:
        effective[probe] = 1.0 + 0.5 * (effective[probe] - 1.0)
        probe_total = float(np.sum(shares[probe] * effective[probe]))
    remaining = 1.0 - probe_total
    nonprobe_total = float(np.sum(shares[~probe]))
    next_shares = np.empty_like(shares)
    next_shares[probe] = shares[probe] * effective[probe]
    next_shares[~probe] = shares[~probe] * (remaining / nonprobe_total)
    updated = np.array(target, dtype=np.float32, copy=True)
    updated[rows, columns] = (next_shares * float(np.sum(raw))).astype(np.float32)
    return validate_target(updated), next_shares / shares


def _probe_verdict(
    probe_sites: np.ndarray,
    baseline_weights: np.ndarray,
    baseline_contrast: np.ndarray,
    baseline_valid: np.ndarray,
    measurements: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    maximum_weight_change: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn a two-sided probe episode into each probe site's direction.

    THE DIRECTION OF A DARK SITE IS DECIDED BY EVIDENCE.  The side of the
    episode on which the site loaded is the side it goes to; when both sides
    loaded, the one whose contrast is closer to the loaded sites' reference
    (closer to the baseline share when there is no reference).  When neither
    side loaded the physical prior takes over: loading needs a deeper trap,
    so the site extrapolates one ``maximum_weight_change`` step beyond the
    deepest share the episode has already shown dark, instead of standing
    still at a baseline that is known not to load.  Standing still was a
    stall ("baseline restored") that left a dark site dark forever.

    Returns, for every site (zeros/NaN off the probe sites): the requested
    log step from the baseline share, the bracket direction, the share known
    single, the share known loaded, and the decision text.
    """

    probe = np.asarray(probe_sites, dtype=bool)
    baseline = np.asarray(baseline_weights, dtype=float)
    values = np.asarray(baseline_contrast, dtype=float)
    valid = np.asarray(baseline_valid, dtype=bool)
    shape = baseline.shape
    reasonable = valid & np.isfinite(values) & (values > 0.0)
    reference = (
        float(np.exp(np.mean(np.log(values[reasonable]))))
        if np.any(reasonable) else float("nan")
    )
    cap = np.log1p(float(maximum_weight_change))
    step = np.zeros(shape, dtype=float)
    direction = np.zeros(shape, dtype=float)
    single_bound = np.full(shape, np.nan, dtype=float)
    observable_bound = np.full(shape, np.nan, dtype=float)
    decisions = np.full(shape, "not_probed", dtype="<U32")
    for site in np.flatnonzero(probe):
        options: list[tuple[bool, float, float, float]] = []
        for factor, contrasts, standard_errors, observable_values in measurements:
            observable = bool(observable_values[site])
            value = float(contrasts[site])
            uncertainty = float(standard_errors[site])
            if (
                not observable or not np.isfinite(value) or value <= 0.0
                or not np.isfinite(uncertainty) or uncertainty < 0.0
                or uncertainty > 0.25 * value
            ):
                continue
            options.append(
                (
                    factor < 1.0,
                    abs(float(np.log(value / reference)))
                    if np.isfinite(reference)
                    else abs(float(np.log(factor))),
                    uncertainty / value,
                    factor,
                )
            )
        sides = {option[0] for option in options}
        if not options:
            deepest = max(
                [1.0] + [float(factor) for factor, *_rest in measurements]
            )
            direction[site] = 1.0
            single_bound[site] = baseline[site] * deepest
            step[site] = float(np.log(deepest)) + cap
            decisions[site] = "probe_extrapolate_upper"
            continue
        chosen = min(options, key=lambda item: (item[1], item[2]))
        factor = chosen[3]
        direction[site] = float(np.sign(np.log(factor)))
        single_bound[site] = baseline[site]
        observable_bound[site] = baseline[site] * factor
        step[site] = float(np.log(factor))
        side = "lower" if chosen[0] else "upper"
        decisions[site] = (
            f"probe_choose_{side}_{'closest' if len(sides) == 2 else 'only'}"
        )
    return step, direction, single_bound, observable_bound, decisions


def _probe_boundary(
    single: np.ndarray, observable: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    bracket = (
        np.isfinite(single) & np.isfinite(observable)
        & (((direction > 0.0) & (single < observable))
           | ((direction < 0.0) & (observable < single)))
    )
    boundary = np.full(single.shape, np.nan, dtype=float)
    boundary[bracket] = np.sqrt(single[bracket] * observable[bracket])
    return boundary


def _updated_brackets(
    current: object,
    dark: object,
    observable: object,
    previous_weights: object,
    single_bound: object,
    observable_bound: object,
    direction: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Record what this candidate proved about every site's loading edge.

    A bracket is one site's evidence about where its loading edge lies: the
    share it was last seen single at, the share it was last seen loaded at,
    and the direction from the first to the second.  EVERY measured
    candidate is evidence -- a diagnostic probe as much as a formal one --
    and THE LATEST OBSERVATION WINS: the edge drifts with the rest of the
    lattice (the solver redistributes, the reference mean moves; on the
    virtual lattice by ~5% over ten candidates), so a bound is the most
    recent share the site showed that state at, never the tightest ever
    seen.  A loaded site pressed dark by the compensation for somebody
    else's share is exactly a site that has just shown where its edge is:
    that goes into its bracket and its next step goes back towards the
    share it was loaded at.

    The direction is the probe verdict's (or, seeded here, the prior's:
    deeper loads) and ONLY a verdict changes it.  A dark observation cannot
    say which side of its loading window a site is on, and the rule that
    turned a site round whenever it was dark beyond its loaded bound sent
    three virtual sites ping-ponging between two shares for the rest of
    the run: "loaded at 1.05" was a marginal loading the ramp had since
    moved past, and the flip made it a ceiling.

    Rules, in order:
    - a loaded site records the share as its loaded bound; a single bound
      on the loaded side of it is stale and dropped;
    - a dark site with no bracket seeds one from the last share it was
      loaded at (see ``previous_weights``);
    - a dark site beyond its loaded bound by more than the resolution has
      contradicted that bound: it is dropped, and the site records the
      share as its single bound and keeps creeping in its direction
      (``_single_bracket_step``);
    - otherwise a dark site records the share as its single bound; dark
      within a resolution of the loaded share it has FOUND the edge and
      steps one resolution past it.

    The resolution is the identification excitation
    (``_PLANT_EXCITATION_LOG_STEP``): the SLM Target is wiggled by that much
    while the plant is identified, so no bracket can be resolved finer.
    """

    present = np.asarray(current, dtype=float)
    single_now = np.asarray(dark, dtype=bool)
    loaded_now = np.asarray(observable, dtype=bool)
    previous = np.asarray(previous_weights, dtype=float)
    single = np.asarray(single_bound, dtype=float).copy()
    loaded = np.asarray(observable_bound, dtype=float).copy()
    sign = np.asarray(direction, dtype=float).copy()
    if not all(
        value.shape == present.shape
        for value in (single_now, loaded_now, previous, single, loaded, sign)
    ):
        raise ValueError("single bracket shapes differ")
    resolution = _PLANT_EXCITATION_LOG_STEP

    loaded[loaded_now] = present[loaded_now]
    with np.errstate(divide="ignore", invalid="ignore"):
        toward_single = np.where(
            loaded_now & np.isfinite(single), sign * np.log(single / present), 0.0
        )
    stale = loaded_now & np.isfinite(single) & (toward_single >= 0.0)
    single[stale] = np.nan

    seed = single_now & (sign == 0.0) & np.isfinite(previous) & (previous > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.where(seed, np.log(previous / present), 0.0)
    sign[seed] = np.where(np.abs(gap[seed]) <= resolution, 1.0, np.sign(gap[seed]))
    single[seed] = present[seed]
    loaded[seed] = previous[seed]

    bracketed = single_now & ~seed & (sign != 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        beyond = np.where(
            bracketed & np.isfinite(loaded), sign * np.log(present / loaded), 0.0
        )
    contradicted = bracketed & np.isfinite(loaded) & (beyond > resolution)
    loaded[contradicted] = np.nan
    single[bracketed] = present[bracketed]
    return single, loaded, sign


def _single_bracket_step(
    current: np.ndarray,
    dark: np.ndarray,
    single_bound: np.ndarray,
    observable_bound: np.ndarray,
    direction: np.ndarray,
    maximum_weight_change: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The step a dark site asks for from its bracket, and its name.

    With a bracket wider than the resolution the site bisects towards its
    loaded share (the geometric midpoint, capped at
    ``maximum_weight_change``): ``single_bracket_midpoint``.  Otherwise it
    CREEPS: one resolution past its single bound in its direction --
    ``single_edge_step`` with a loaded share on record (the edge is found,
    or the bracket is inverted), ``single_extrapolate`` without one.  The
    prior is applied one resolution at a time, never a whole
    ``maximum_weight_change``: a site jumping a third of its share is funded
    by every loaded site at once, and on the virtual lattice, at a 4%
    loading margin, that pressed the neighbours dark in turn
    (29->23->26->25->32->22 observable sites).  The one large step a
    never-loaded site gets is the probe verdict's own, informed by both
    probe sides (``_probe_verdict``).
    """

    present = np.asarray(current, dtype=float)
    single_now = np.asarray(dark, dtype=bool)
    single = np.asarray(single_bound, dtype=float)
    loaded = np.asarray(observable_bound, dtype=float)
    sign = np.asarray(direction, dtype=float)
    cap = np.log1p(float(maximum_weight_change))
    resolution = _PLANT_EXCITATION_LOG_STEP
    step = np.zeros(present.shape, dtype=float)
    kind = np.full(present.shape, "", dtype="<U32")
    active = (
        single_now & (sign != 0.0) & np.isfinite(present) & (present > 0.0)
        & np.isfinite(single) & (single > 0.0)
    )
    known = active & np.isfinite(loaded) & (loaded > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        span = np.log(loaded / single)
        midpoint = np.sqrt(single * loaded)
        toward = np.log(midpoint / present)
        past_single = np.log(single / present)
    bisect = known & (np.sign(span) == sign) & (np.abs(span) > resolution)
    step[bisect] = np.clip(toward[bisect], -cap, cap)
    kind[bisect] = "single_bracket_midpoint"
    creep = active & ~bisect
    step[creep] = past_single[creep] + sign[creep] * resolution
    kind[creep & known] = "single_edge_step"
    kind[creep & ~known] = "single_extrapolate"
    return step, kind


def _needs_probe(
    single: np.ndarray,
    observable: np.ndarray,
    acquisition_invalid: np.ndarray,
    direction: np.ndarray,
    previous_weights: np.ndarray,
    previous_contrast: np.ndarray,
) -> np.ndarray:
    """A dark site with no evidence at all -- no bracket, never loaded."""

    has_history = (
        np.isfinite(previous_weights) & (previous_weights > 0.0)
        & np.isfinite(previous_contrast) & (previous_contrast > 0.0)
    )
    return (
        _unobservable_single(single, observable, acquisition_invalid)
        & (np.asarray(direction, dtype=float) == 0.0)
        & ~has_history
    )


def _unobservable_single(
    single: object,
    observable: object,
    acquisition_invalid: object,
) -> np.ndarray:
    return (
        np.asarray(single, dtype=bool)
        & ~np.asarray(observable, dtype=bool)
        & ~np.asarray(acquisition_invalid, dtype=bool)
    )


def _updated_target(
    target: np.ndarray,
    contrast: np.ndarray,
    standard_error: np.ndarray,
    valid: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    reference_valid: np.ndarray,
    feedback_gain: float,
    plant_slope: float | None,
    maximum_weight_change: float,
    directed_log_step: np.ndarray | None = None,
    control_boundary: np.ndarray | None = None,
    control_direction: np.ndarray | None = None,
    loading_edge: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update observable sites relatively; fund the dark sites' directions.

    A loaded site on its loading ramp (``loading_edge``, see
    ``_loading_edge``) has no share to give: its own step is held at zero
    when it points shallower (``hold_loading_edge``) and neither the
    ordinary trades nor the funding may take it below its current share.
    The site that ping-ponged for twenty candidates on the virtual lattice
    was one loaded at a 15% bright fraction and then handed 2% to a dark
    neighbour: dark the next candidate, and back on the probe treadmill.

    ``feedback_gain`` is the LOOP gain: the fraction of each site's log
    residual the next candidate is asked to remove.  Dividing by the
    measured ``plant_slope`` magnitude turns that into the weight step which
    does it; with no trusted slope the step assumes unit slope at half gain
    (see ``_usable_plant_slope``).  Every step is then scaled by the fit
    quality, clamped to ``maximum_weight_change`` and passed through the
    share-conserving allocator, so the recorded correction is what was
    actually applied.

    A dark site with a direction (``directed_log_step``: a probe verdict, a
    bracket bisection or an extrapolation) asks for that share outright --
    the request is not the loop's, so neither gain nor clamp applies to it.
    THE LOADED SITES FUND IT, TOGETHER AND BOUNDED: whatever the ordinary
    trades leave unmet is drawn from every loaded site by one common factor,
    each site giving at most one resolution step in this candidate and never
    past its own bracket boundary (``control_boundary``; see
    ``_funded_shares``).  Total power
    is the hard constraint; how fast a dark site gets there is not.  Direct
    adoption of a winning probe share used to rescale the loaded sites by
    whatever it took (-40% at once), which pressed every one of them past
    its loading edge at 4% margin; and a bracket step with nobody assigned
    to pay for it moved ~1% per candidate, the crumbs of the loop's own
    residual trades.  What cannot be funded this candidate is scaled back
    towards the current share, never reversed, and asked for again next
    candidate.  Without a directed site this is exactly the ordinary
    allocation, arithmetic untouched.
    """

    values = np.asarray(contrast, dtype=float)
    errors = np.asarray(standard_error, dtype=float)
    control_valid = np.asarray(valid, dtype=bool)
    references = np.asarray(reference_valid, dtype=bool)
    gain = float(feedback_gain)
    if plant_slope is None:
        step_gain = 0.5 * gain
        feedback_decision = "feedback_assumed_slope"
    else:
        slope = float(plant_slope)
        if not np.isfinite(slope) or slope <= 0.0:
            raise ValueError("plant_slope must be a positive magnitude or None")
        step_gain = gain / slope
        feedback_decision = "feedback_estimated_slope"
    maximum_change = float(maximum_weight_change)
    site_shape = (len(rows),)
    direction = (
        np.zeros(site_shape, dtype=float)
        if directed_log_step is None
        else np.asarray(directed_log_step, dtype=float)
    )
    boundary = (
        np.full(site_shape, np.nan, dtype=float)
        if control_boundary is None else np.asarray(control_boundary, dtype=float)
    )
    boundary_direction = (
        np.zeros(site_shape, dtype=float)
        if control_direction is None else np.asarray(control_direction, dtype=float)
    )
    edge = (
        np.zeros(site_shape, dtype=bool)
        if loading_edge is None else np.asarray(loading_edge, dtype=bool)
    )
    if (
        values.shape != site_shape
        or errors.shape != site_shape
        or control_valid.shape != site_shape
        or references.shape != site_shape
        or direction.shape != site_shape
        or boundary.shape != site_shape
        or boundary_direction.shape != site_shape
        or edge.shape != site_shape
        or np.any(edge & ~control_valid)
        or np.any(references & ~control_valid)
        or np.any(~np.isfinite(direction))
        or not np.isfinite(gain)
        or gain < 0.0
        or not np.isfinite(maximum_change)
        or maximum_change < 0.0
    ):
        raise ValueError("feedback site history shapes differ")
    if np.any(
        ~np.isfinite(values[control_valid]) | (values[control_valid] <= 0.0)
    ):
        raise ValueError("valid feedback contrasts must be finite and positive")
    if not np.any(references):
        reference = float("nan")
    else:
        reference = float(np.exp(np.mean(np.log(values[references]))))
    raw_weights = np.asarray(target[rows, columns], dtype=float)
    log_correction = np.zeros(site_shape, dtype=float)
    decision = np.full(site_shape, "hold_invalid", dtype="<U32")
    for site in range(len(rows)):
        if control_valid[site] and np.isfinite(reference):
            residual = float(np.log(values[site] / reference))
            relative_error = max(float(errors[site]), 0.0) / values[site]
            quality = float(np.clip(1.0 - 4.0 * relative_error, 0.1, 1.0))
            decision[site] = feedback_decision
            log_correction[site] = float(
                np.clip(
                    step_gain * quality * residual,
                    -np.log1p(maximum_change),
                    np.log1p(maximum_change),
                )
            )
            if edge[site] and log_correction[site] < 0.0:
                log_correction[site] = 0.0
                decision[site] = "hold_loading_edge"
        elif direction[site] != 0.0:
            log_correction[site] = float(direction[site])
            decision[site] = (
                "probe_direction_lower"
                if log_correction[site] < 0.0
                else "probe_direction_upper"
            )
        elif control_valid[site]:
            decision[site] = "hold_no_double_reference"
    shares = raw_weights / float(np.sum(raw_weights))
    boundary_share = boundary / float(len(rows))
    lower = np.where(
        (boundary_direction > 0.0) & np.isfinite(boundary_share),
        boundary_share,
        0.0,
    )
    lower = np.where(edge, np.maximum(lower, shares), lower)
    upper = np.where(
        (boundary_direction < 0.0) & np.isfinite(boundary_share),
        boundary_share,
        np.inf,
    )
    next_shares, _transfer, _increase_scale, _decrease_scale = (
        _allocate_requested_shares(
            shares,
            log_correction,
            lower=lower,
            upper=upper,
        )
    )
    directed = (direction != 0.0) & ~control_valid
    if np.any(directed):
        next_shares = _funded_shares(
            shares,
            next_shares,
            directed,
            shares * np.exp(log_correction),
            control_valid & ~directed,
            lower=lower,
            upper=upper,
        )
    log_correction = np.log(next_shares / shares)
    updated = np.array(target, dtype=np.float32, copy=True)
    updated[rows, columns] = (
        next_shares * float(np.sum(raw_weights))
    ).astype(np.float32)
    return validate_target(updated), log_correction, decision


class SlmFeedbackTask:
    """Apply candidates, measure exact qCMOS cycles, and retain a valid phase."""

    instance_id = "slm_feedback"

    def __init__(
        self,
        *,
        camera: object,
        camera_key: str,
        sequencer: object,
        sequencer_key: str,
        slm: object,
        slm_key: str,
        signal_plane: object,
        calibration: TrapCalibration,
        calibration_path: str | Path,
        science_context: Mapping[str, object],
        science_context_path: str | Path,
        pulse_sequence: PulseSequence,
        pulse_path: str | Path,
        feedback_mode: str,
        exposure_seconds: float,
        shots_per_candidate: int,
        probe_factors: tuple[float, ...],
        feedback_gain: float,
        maximum_weight_change: float,
        max_updates: int,
        save_figure_artifact: object = None,
    ) -> None:
        if not isinstance(slm, SlmAdapter):
            raise TypeError("slm must implement SlmAdapter")
        if not isinstance(calibration, TrapCalibration) or not isinstance(pulse_sequence, PulseSequence):
            raise TypeError("feedback requires TrapCalibration and PulseSequence")
        if not isinstance(science_context, Mapping):
            raise TypeError("science_context must be a loaded Science Context mapping")
        if save_figure_artifact is not None and not callable(save_figure_artifact):
            raise TypeError("save_figure_artifact must be callable or None")
        if science_context.get("objective_kind") != "spots":
            raise ValueError("SLM feedback Science Context must use the spots objective")
        self._save_figure_artifact = save_figure_artifact
        context_target = science_context.get("target_intensity")
        if context_target is None:
            raise ValueError("Science Context has no frozen Target")
        frozen_target = validate_target(context_target)
        if frozen_target.shape != slm.shape_yx:
            raise ValueError("Science Context Target shape differs from the selected SLM")
        pattern = freeze_pattern_phase(
            science_context.get("pattern_phase"), slm.shape_yx
        )
        operator = canonical_phase(
            science_context.get("operator_wavefront"), slm.shape_yx
        )
        incoming = compose_science_phase(pattern, operator)
        pupil = np.asarray(science_context.get("pupil_amplitude"), dtype=np.float32)
        if (
            pupil.shape != slm.shape_yx
            or not np.all(np.isfinite(pupil))
            or np.any(pupil < 0.0)
            or not np.any(pupil > 0.0)
        ):
            raise ValueError("Science Context has an invalid pupil")
        receipt = science_context.get("command_receipt")
        if not isinstance(receipt, Mapping):
            raise TypeError("Science Context command receipt must be a mapping")
        # The readout is the Calibration's DEFAULT model -- the same matched
        # filter (or box) that occupancy reads with, at the registered site
        # centres.  The task no longer sums its own 3x3 box: that box caught
        # 66% of a site's light and lost 1.25% of signal per quarter pixel
        # of drift, the same size as the residuals the loop was correcting.
        model = calibration.select_model()
        context_path = Path(science_context_path).expanduser().resolve()
        (
            self._rows,
            self._columns,
            self._registered_site_map,
            source_index,
        ) = _support(
            frozen_target,
            calibration,
            model,
            science_context_path=context_path,
            command_receipt=receipt,
        )
        self._site_count = len(self._rows)
        self._site_centers_xy = np.asarray(
            self._registered_site_map.centers_xy, dtype=float
        ).copy()
        self._site_centers_xy.setflags(write=False)
        self._site_kernels: np.ndarray | None = None
        if model.kind is not ReadoutModelKind.BOX:
            observed = source_index >= 0
            kernels = np.empty(
                (self._site_count, *model.psf_weights.shape[1:]), dtype=float
            )
            kernels[observed] = model.psf_weights[source_index[observed]]
            if not np.all(observed):
                # A site the Calibration never saw has no measured shape; it
                # reads with the shape the other sites agreed on, exactly as
                # the Calibration itself treats a site whose atoms it could
                # not measure.
                try:
                    uniform = calibration.select_model(ReadoutModelKind.UNIFORM_PSF)
                except KeyError:
                    raise ValueError(
                        "predicted Target sites need the Calibration's uniform "
                        "PSF model to read with"
                    ) from None
                if (
                    uniform.psf_weights.shape[1:] != kernels.shape[1:]
                    or uniform.integration_half_width != model.integration_half_width
                ):
                    raise ValueError(
                        "Calibration uniform PSF differs in size from its default model"
                    )
                kernels[~observed] = uniform.psf_weights[0]
            kernels.setflags(write=False)
            self._site_kernels = kernels
        self.camera, self.sequencer, self.slm = camera, sequencer, slm
        self.camera_key, self.sequencer_key, self.slm_key = camera_key, sequencer_key, slm_key
        self.signal_plane, self.calibration, self.model = signal_plane, calibration, model
        self.target, self.sequence = frozen_target, pulse_sequence
        self.calibration_path = Path(calibration_path).expanduser().resolve()
        self.pulse_path = Path(pulse_path).expanduser().resolve()
        self.science_context_path = context_path
        self._incoming_phase = incoming
        self._pattern_phase = pattern
        self._operator_wavefront = operator
        self._pupil_amplitude = np.array(pupil, copy=True)
        self._pupil_amplitude.setflags(write=False)
        self._pupil = dict(science_context.get("pupil", {}))
        self._system_correction = science_context.get("system_correction")
        incoming_pattern_metadata = dict(
            science_context.get("pattern_metadata", {})
        )
        self._prior_pattern_metadata = incoming_pattern_metadata
        runtime_metadata = {
            "candidate",
            "status",
            "measurement",
            "updates",
            "solver",
            "outcome",
        }
        self._pattern_metadata = {
            key: value
            for key, value in incoming_pattern_metadata.items()
            if key not in runtime_metadata
        }
        self._operator_metadata = dict(science_context.get("operator_metadata", {}))
        self._mapping_revision = int(slm.mapping_revision)
        self.feedback_mode = str(feedback_mode)
        if self.feedback_mode != "qcmos_bright_dark":
            raise ValueError("unsupported SLM feedback mode")
        self.exposure_seconds = float(exposure_seconds)
        if not np.isfinite(self.exposure_seconds) or self.exposure_seconds <= 0.0:
            raise ValueError("feedback exposure_seconds must be finite and positive")
        self.shots = int(shots_per_candidate)
        self.feedback_gain = float(feedback_gain)
        self.maximum_weight_change = float(maximum_weight_change)
        self.max_updates = int(max_updates)
        self._publication_revision = 0
        self._actual_exposure_seconds: float | None = None
        self._effective_photoelectrons: bool | None = None
        self._effective_count_unit: str | None = None
        self._last_measured_phase: np.ndarray | None = None
        self._actual_device_snapshots: dict[str, Mapping[str, object]] = {}
        factors = tuple(float(value) for value in probe_factors)
        if (
            not factors
            or any(
                not np.isfinite(value) or value <= 0.0 or value == 1.0
                for value in factors
            )
            or len(set(factors)) != len(factors)
        ):
            raise ValueError(
                "probe_factors must be unique positive numbers excluding 1"
            )
        self.probe_factors = factors
        self._candidate_capacity = (
            1
            + self.max_updates
            + self.max_updates * len(factors)
        )
        if (
            self.shots < 10
            or self.max_updates < 1
            or not np.isfinite(self.feedback_gain)
            or self.feedback_gain < 0.0
            or not np.isfinite(self.maximum_weight_change)
            or self.maximum_weight_change < 0.0
        ):
            raise ValueError("feedback needs at least 10 shots and one update")
        contract = calibration.frame_contract
        height, width = contract.image_shape
        site_mask = np.zeros((height, width), dtype=bool)
        windows: list[tuple[slice, slice]] = []
        radius = int(model.integration_half_width)
        for center_x, center_y in self._site_centers_xy:
            x, y = int(round(float(center_x))), int(round(float(center_y)))
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            site_mask[y0:y1, x0:x1] = True
            windows.append((slice(y0, y1), slice(x0, x1)))
        self._site_mask, self._site_windows = site_mask, tuple(windows)

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return (
            CANDIDATE_PHASE_OUTPUT,
            UNIFORMITY_HISTORY_OUTPUT,
            OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT,
            SITE_SIGNAL_HISTORY_OUTPUT,
            TARGET_SHARE_HISTORY_OUTPUT,
        )

    def _run_record(self) -> dict[str, object]:
        return {
            "node": self.instance_id,
            "calibration_path": str(self.calibration_path),
            "science_context_path": str(self.science_context_path),
            "pulse_path": str(self.pulse_path),
            "named_devices": {
                "camera": self.camera_key,
                "sequencer": self.sequencer_key,
                "slm": self.slm_key,
            },
            "max_updates": self.max_updates,
            "feedback_controller": _CONTROLLER_CONTRACT,
            "feedback_mode": self.feedback_mode,
            "exposure_seconds": self.exposure_seconds,
            "shots_per_candidate": self.shots,
            "probe_factors": list(self.probe_factors),
            "feedback_gain": self.feedback_gain,
            "maximum_weight_change": self.maximum_weight_change,
            "readout_model_kind": self.model.kind.value,
            "readout_half_width": int(self.model.integration_half_width),
        }

    def _site_signals(self, image: np.ndarray) -> np.ndarray:
        """One frame read with the Calibration's default model at every site."""

        model = self.model
        if self._site_kernels is None:
            return extract_box_signals(
                image,
                self._site_centers_xy,
                radius=model.integration_half_width,
            )
        return extract_psf_signals(
            image,
            self._site_centers_xy,
            kernels=self._site_kernels,
            background=model.background,
            radius=model.integration_half_width,
            padding=model.psf_padding,
        )

    def _device_event_record(
        self,
        *,
        include_measurement: bool,
        candidate: int,
    ) -> dict[str, object]:
        """The exact working points for the candidate being published."""

        if type(include_measurement) is not bool:
            raise TypeError("include_measurement must be bool")
        if type(candidate) is not int or candidate < 1:
            raise ValueError("feedback device snapshot candidate must be positive")
        snapshots: dict[str, object] = {}
        if include_measurement:
            missing = {"camera", "sequencer"} - set(
                self._actual_device_snapshots
            )
            if missing:
                raise RuntimeError(
                    "measured feedback candidate has no exact device snapshot for "
                    + ", ".join(sorted(missing))
                )
            snapshots.update(
                {
                    role: dict(self._actual_device_snapshots[role])
                    for role in ("camera", "sequencer")
                }
            )
        snapshots["slm"] = {
            "identity": str(self.slm.identity),
            "shape_yx": [int(value) for value in self.slm.shape_yx],
            "command_revision": int(self.slm.command_revision),
            "mapping_revision": int(self.slm.mapping_revision),
            "command_receipt": dict(self.slm.last_command_receipt),
        }
        plain = _plain_json(
            {
                "device_snapshots": snapshots,
                "device_snapshot_context": {
                    "candidate": candidate,
                    "measurement_completed": include_measurement,
                },
            }
        )
        if not isinstance(plain, dict):
            raise TypeError("feedback device event record must remain an object")
        return plain

    def _candidate_metadata(
        self,
        *,
        candidate: int,
        status: str,
        history: list[dict[str, object]],
        solver: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "calibration_path": str(self.calibration_path),
            "science_context_path": str(self.science_context_path),
            "pulse_path": str(self.pulse_path),
            "named_devices": {
                "camera": self.camera_key,
                "sequencer": self.sequencer_key,
                "slm": self.slm_key,
            },
            "candidate": int(candidate),
            "status": str(status),
            "feedback_controller": _CONTROLLER_CONTRACT,
            "feedback_mode": self.feedback_mode,
            "exposure_seconds": self.exposure_seconds,
            "shots_per_candidate": self.shots,
            "probe_factors": list(self.probe_factors),
            "feedback_gain": self.feedback_gain,
            "maximum_weight_change": self.maximum_weight_change,
            "actual_exposure_seconds": self._actual_exposure_seconds,
            "effective_photoelectrons": self._effective_photoelectrons,
            "effective_count_unit": self._effective_count_unit,
            "measurement": next(
                (
                    item
                    for item in reversed(history)
                    if item["iteration"] == candidate
                ),
                None,
            ),
            "updates": len(history),
            "solver": None if solver is None else dict(solver),
        }

    def _save_candidate(
        self,
        path: str | Path,
        pattern: object,
        target: object,
        metadata: Mapping[str, object],
    ) -> Path:
        receipt = dict(self.slm.last_command_receipt)
        if receipt.get("outcome") not in {"known-old", "known-new"}:
            raise RuntimeError("SLM candidate command outcome is unknown")
        pattern_metadata = _plain_json(
            {**self._pattern_metadata, **dict(metadata)}
        )
        if not isinstance(pattern_metadata, Mapping):
            raise TypeError("SLM candidate metadata must remain a mapping")
        return save_science_context(
            path,
            pattern,
            target_intensity=target,
            objective_kind="spots",
            pupil=self._pupil,
            system_correction=self._system_correction,
            command_receipt=receipt,
            pattern_metadata=pattern_metadata,
            operator_metadata=self._operator_metadata,
        )

    def _publish_candidate(
        self,
        context: object,
        *,
        phase: np.ndarray,
        candidate: int,
        history: list[dict[str, object]],
        device_event_record: Mapping[str, object],
    ) -> None:
        candidate = int(candidate)
        if not 1 <= candidate <= self._candidate_capacity:
            raise ValueError("feedback candidate lies outside its authored history")
        self._publication_revision += 1
        publication_revision = self._publication_revision
        generation = str(getattr(context.generation, "value", context.generation))
        canonical = canonical_phase(phase, self.slm.shape_yx)
        phase_event = snapshot_from_array(
            canonical[None],
            producer=self.instance_id,
            signal=CANDIDATE_PHASE_OUTPUT.name,
            cell_axes=(SPATIAL_Y, SPATIAL_X),
            value_unit="rad",
            generation=generation,
            revision=publication_revision,
        )
        coordinate_id = AxisId("slm_feedback.candidate")
        curve = np.full(self._candidate_capacity, np.nan, dtype="<f8")
        observable_curve = np.full_like(curve, np.nan)
        site_signal = np.full(
            (self._candidate_capacity, self._site_count), np.nan, dtype="<f8"
        )
        target_share = np.full_like(site_signal, np.nan)
        for item in history:
            index = int(item["iteration"]) - 1
            full_ratio = item.get("uniformity_ratio")
            observable_ratio = item.get("observable_uniformity_ratio")
            if full_ratio is not None:
                curve[index] = float(full_ratio)
            if observable_ratio is not None:
                observable_curve[index] = float(observable_ratio)
            for field, destination in (
                ("bright_minus_dark", site_signal),
                ("control_weight", target_share),
            ):
                destination[index] = np.asarray(
                    [np.nan if value is None else value for value in item[field]],
                    dtype=float,
                )
        point_axes = (
            AxisSpec(
                coordinate_id,
                "candidate",
                SCAN_POINT,
                self._candidate_capacity,
                tuple(
                    float(index)
                    for index in range(1, self._candidate_capacity + 1)
                ),
            ),
        )
        history_event = snapshot_from_array(
            curve[None],
            producer=self.instance_id,
            signal=UNIFORMITY_HISTORY_OUTPUT.name,
            point_axes=point_axes,
            generation=generation,
            revision=publication_revision,
            validity=np.isfinite(curve)[None],
        )
        observable_history_event = snapshot_from_array(
            observable_curve[None],
            producer=self.instance_id,
            signal=OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT.name,
            point_axes=point_axes,
            generation=generation,
            revision=publication_revision,
            validity=np.isfinite(observable_curve)[None],
        )
        site_axis = self._registered_site_map.site_axis
        site_signal_event = snapshot_from_array(
            site_signal[None],
            producer=self.instance_id,
            signal=SITE_SIGNAL_HISTORY_OUTPUT.name,
            point_axes=point_axes,
            cell_axes=(site_axis,),
            generation=generation,
            revision=publication_revision,
            validity=np.isfinite(site_signal)[None],
        )
        target_share_event = snapshot_from_array(
            target_share[None],
            producer=self.instance_id,
            signal=TARGET_SHARE_HISTORY_OUTPUT.name,
            point_axes=point_axes,
            cell_axes=(site_axis,),
            generation=generation,
            revision=publication_revision,
            validity=np.isfinite(target_share)[None],
        )
        record = self._run_record()
        if not isinstance(device_event_record, Mapping):
            raise TypeError("candidate device_event_record must be a mapping")
        event_record = dict(device_event_record)
        outputs = {
            CANDIDATE_PHASE_OUTPUT.name: LiveDatasetOutput(
                CANDIDATE_PHASE_OUTPUT,
                phase_event,
                MonitorCoverage(1, 1),
                record,
                event_record=event_record,
            ),
            UNIFORMITY_HISTORY_OUTPUT.name: LiveDatasetOutput(
                UNIFORMITY_HISTORY_OUTPUT,
                history_event,
                MonitorCoverage(
                    self._candidate_capacity,
                    self._candidate_capacity,
                ),
                record,
                event_record=event_record,
            ),
            OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT.name: LiveDatasetOutput(
                OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT,
                observable_history_event,
                MonitorCoverage(
                    self._candidate_capacity,
                    self._candidate_capacity,
                ),
                record,
                event_record=event_record,
            ),
            SITE_SIGNAL_HISTORY_OUTPUT.name: LiveDatasetOutput(
                SITE_SIGNAL_HISTORY_OUTPUT,
                site_signal_event,
                MonitorCoverage(
                    self._candidate_capacity,
                    self._candidate_capacity,
                ),
                record,
                event_record=event_record,
            ),
            TARGET_SHARE_HISTORY_OUTPUT.name: LiveDatasetOutput(
                TARGET_SHARE_HISTORY_OUTPUT,
                target_share_event,
                MonitorCoverage(
                    self._candidate_capacity,
                    self._candidate_capacity,
                ),
                record,
                event_record=event_record,
            ),
        }
        context.commit_live(outputs)

    def _incoming_candidate(
        self,
        *,
        observed: Mapping[str, object] | None,
        phase: np.ndarray,
        pattern: np.ndarray,
    ) -> dict[str, object]:
        candidate = 1 if observed is None else int(observed["candidate"])
        return {
            "candidate": candidate,
            "phase": np.array(phase, copy=True),
            "pattern_phase": np.array(pattern, copy=True),
            "target": np.array(self.target, copy=True),
            "solver": None,
            "history": None if observed is None else observed["history"],
            "samples": None if observed is None else observed.get("samples"),
            "mean_frame": None if observed is None else observed.get("mean_frame"),
        }

    def _apply_exact(self, phase: object) -> np.ndarray:
        if self.slm.mapping_revision != self._mapping_revision:
            raise RuntimeError("SLM correction mapping changed during feedback")
        expected = canonical_phase(phase, self.slm.shape_yx)
        applied = self.slm.apply_phase(expected)
        observed = self.slm.last_commanded_phase
        receipt = dict(self.slm.last_command_receipt)
        if (
            not np.array_equal(applied, expected)
            or observed is None
            or not np.array_equal(observed, expected)
            or receipt.get("outcome") != "known-new"
            or receipt.get("mapping_revision") != self._mapping_revision
        ):
            raise RuntimeError("SLM did not confirm the commanded canonical phase")
        return applied

    def _assert_camera_contract(self, actual: object) -> None:
        """The Calibration constrains box coordinates, not the camera physics."""

        contract = self.calibration.frame_contract
        roi = contract.roi_xywh
        expected_roi = None if roi is None else (roi[1], roi[0], roi[3], roi[2])
        actual_roi = (
            *tuple(actual.roi_origin_yx),
            *tuple(actual.roi_shape_yx),
        )
        matches = (
            actual.acquisition_mode == "EXTERNAL_TRIGGERED"
            and tuple(actual.frame_shape_yx) == contract.image_shape
            and tuple(actual.binning_yx) == contract.binning_yx
            and (expected_roi is None or actual_roi == expected_roi)
        )
        if not matches:
            raise ValueError("selected camera geometry differs from the calibrated site boxes")

    def _saturated_sites(
        self, image: np.ndarray, saturation_value: object
    ) -> tuple[int, ...]:
        saturated = np.asarray(image) == saturation_value
        if not bool(np.any(saturated[self._site_mask])):
            return ()
        return tuple(
            index
            for index, window in enumerate(self._site_windows)
            if bool(np.any(saturated[window]))
        )

    def _measure(
        self,
        pulse: object,
        context: object,
        iteration: int,
    ) -> tuple[
        np.ndarray,
        tuple[int, ...],
        tuple[int, ...],
        np.ndarray,
        str,
    ]:
        """Shoot one candidate and return its samples plus the pulse verdict.

        The fifth value is the pulse warning the candidate carries: "" for a
        clean shot, else the fault the batch was ACCEPTED with (see
        ``_accepted_pulse_fault``) and/or the fault a first batch was REPEATED
        after -- both spelled out, because a run whose first attempt's
        failure is not written down cannot be understood afterwards (the
        UART transport already lost its first two attempts' shortfalls that
        way).  A fault the batch cannot be accepted with -- the board
        reporting an error or an underflow, whether it played every trigger
        or stopped so that the camera timed out (``_shoot`` asks the board
        then) -- repeats the whole batch once (safe, arm, fire, collect); a
        second fault is the candidate's, and names both.  The unchanged-phase
        guard below is checked once per CANDIDATE: a repeat inside this call
        is the same candidate being measured, not a second batch at the same
        phase.
        """

        contract = self.calibration.frame_contract
        requested = self.shots
        if requested < 4:
            raise ValueError("qCMOS bright-dark statistics require at least four shots")
        commanded = self.slm.last_commanded_phase
        if commanded is None:
            raise RuntimeError("SLM has no confirmed phase for feedback measurement")
        phase = canonical_phase(commanded, self.slm.shape_yx)
        if self._last_measured_phase is not None and np.array_equal(
            phase, self._last_measured_phase
        ):
            raise RuntimeError(
                "SLM feedback refuses a second shot batch at an unchanged phase"
            )
        self._last_measured_phase = np.array(phase, copy=True)
        camera_owner = f"{context.instance_id}/camera"
        repeated_after = ""
        while True:
            result, saturation_value, report = self._shoot(
                pulse, context, iteration, camera_owner=camera_owner
            )
            delivered = 0 if result is None else int(result.cycle_count)
            fault = str(report.fault or "")
            if not fault:
                accepted_with = ""
            elif _accepted_pulse_fault(
                report, delivered_cycles=delivered, requested_cycles=requested
            ):
                accepted_with = fault
            elif not repeated_after:
                repeated_after = fault
                context.report_progress(
                    f"Repeating the shot batch of candidate {iteration + 1} "
                    f"after a pulse fault: {fault}"
                )
                continue
            else:
                raise RuntimeError(
                    "the pulse failed twice for one candidate: first "
                    f"{repeated_after}; then {fault}"
                )
            break
        pulse_warning = "; ".join(
            text
            for text in (
                f"batch repeated after a pulse fault: {repeated_after}"
                if repeated_after else "",
                f"batch accepted with a pulse fault: {accepted_with}"
                if accepted_with else "",
            )
            if text
        )
        context.report_progress(
            f"Reading mean qCMOS brightness for candidate {iteration + 1}",
            current=requested,
            total=requested,
        )

        frames = _readout_frames(result.snapshot, shots=requested)

        saturated_sites: set[int] = set()
        for image in frames:
            saturated_sites.update(
                self._saturated_sites(image, saturation_value)
            )
        site_samples = np.asarray(
            [self._site_signals(image) for image in frames],
            dtype=float,
        )
        complete = np.all(np.isfinite(site_samples), axis=0)
        missing_sites = set(int(index) for index in np.flatnonzero(~complete))
        mean_frame = np.asarray(np.mean(frames, axis=0), dtype=np.float32)
        return (
            site_samples,
            tuple(sorted(saturated_sites)),
            tuple(sorted(missing_sites)),
            mean_frame,
            pulse_warning,
        )

    def _shoot(
        self,
        pulse: object,
        context: object,
        iteration: int,
        *,
        camera_owner: str,
    ) -> tuple[object, object, object]:
        """One shot batch: safe, arm, fire, collect, and the board's report.

        The program is loaded once per run, not once per candidate: after
        DONE or SAFE the board keeps the program resident and ``fire`` replays
        its own mini-loader, so a LOAD per candidate only re-sent the whole
        image over a lossy line 68 times for nothing.  What decides is the
        board's own answer -- the digest it reports holding -- never a memory
        of having loaded, so a program somebody else loaded in between is
        replaced and a reopened board is loaded afresh.  ``safe()`` before
        arming the camera stays: the camera must be armed while no trigger
        edge can play, and on a board already safe it costs no traffic.
        """

        contract = self.calibration.frame_contract
        requested = self.shots
        node = CameraMeasurementNode(
            camera=self.camera,
            request=CameraMeasurementRequest(
                camera_key=self.camera_key,
                exposure_seconds=self.exposure_seconds,
                roi_xywh=contract.roi_xywh,
                repeat=requested,
                frames_per_cycle=1,
                photoelectrons=True,
            ),
            signal_plane=self.signal_plane,
            producer=camera_owner,
        )
        capture = None
        try:
            self.sequencer.safe()
            board_state = self.sequencer.snapshot()
            if not isinstance(board_state, Mapping):
                raise TypeError("sequencer snapshot must be a mapping")
            if board_state.get("applied_digest") != pulse.program.digest:
                arm_sequencer(self.sequencer, pulse)
                board_state = self.sequencer.snapshot()
                if not isinstance(board_state, Mapping):
                    raise TypeError("sequencer snapshot must be a mapping")
            capture = node.prepare(should_stop=context.cancel_requested)
            actual = node.actual_working_point
            if actual is None:
                raise RuntimeError("camera did not freeze its actual working point")
            camera_snapshots = node.run_record.get("device_snapshots")
            camera_snapshot = (
                camera_snapshots.get("camera")
                if isinstance(camera_snapshots, Mapping)
                else None
            )
            if not isinstance(camera_snapshot, Mapping):
                raise RuntimeError("Camera Measurement did not freeze its working point")
            self._actual_device_snapshots["camera"] = dict(camera_snapshot)
            self._assert_camera_contract(actual)
            self._actual_exposure_seconds = float(actual.exposure_seconds)
            self._effective_photoelectrons = bool(node.reads_photoelectrons)
            self._effective_count_unit = (
                "photoelectron" if node.reads_photoelectrons else actual.count_unit
            )
            raw_dtype = np.dtype(actual.dtype)
            if raw_dtype.kind not in "iu":
                raise ValueError(
                    "Feedback saturation requires an integer raw camera dtype"
                )
            if not isinstance(actual.count_unit, str) or not actual.count_unit:
                raise ValueError("Feedback camera count_unit is invalid")
            raw_maximum = np.iinfo(raw_dtype).max
            if node.reads_photoelectrons:
                current_offset = actual.offset_counts
                current_scale = actual.electrons_per_count
                if current_offset is None or current_scale is None:
                    raise RuntimeError("camera lost its effective photoelectron conversion")
                saturation_value = (
                    np.float32(raw_maximum) - np.float32(current_offset)
                ) * np.float32(current_scale)
            else:
                saturation_value = raw_maximum
            _check_cancelled(context)
            context.report_progress(
                f"Reading mean qCMOS brightness for candidate {iteration + 1}",
                current=0,
                total=requested,
            )
            self.sequencer.fire(run_repeats=requested, scan_repeats=1)
            if "sequencer" not in self._actual_device_snapshots:
                execution_state = self.sequencer.snapshot()
                if not isinstance(execution_state, Mapping):
                    raise TypeError("sequencer snapshot must be a mapping")
                self._actual_device_snapshots["sequencer"] = (
                    sequencer_archive_snapshot(state=execution_state)
                )

            def commit_camera_cycle(cycle: object, index: int) -> None:
                output = _finite_cycle_output(node, cycle, index)
                record = node.run_record
                if IMAGE_POINT_OVERLAY_GEOMETRY_RECORD not in record:
                    record[IMAGE_POINT_OVERLAY_GEOMETRY_RECORD] = (
                        image_point_overlay_geometry(
                            output.snapshot,
                            self._registered_site_map.centers_xy,
                            self._registered_site_map.site_ids,
                            status_axis=self._registered_site_map.site_axis,
                            labels=tuple(
                                str(site)
                                for site in range(1, self._site_count + 1)
                            ),
                            coordinates_are_indices=True,
                        )
                    )
                node._commit_direct_outputs({CAMERA_FRAMES_OUTPUT.name: output})

            try:
                result = capture.collect(
                    commit_cycle=commit_camera_cycle,
                    retain_cycles=False,
                )
            except Exception:
                # THE BOARD IS ASKED BEFORE THE CAMERA IS BLAMED.  A board
                # that errs or underruns mid-batch stops playing triggers,
                # and the first thing to notice is the camera's frame
                # timeout -- which used to leave here as the candidate's
                # failure before the board's report was ever read, so the
                # one-repeat rule of ``_measure`` never saw a mid-batch
                # board fault.  One poll, no wait: a board still playing
                # (a genuine camera fault) reports nothing and the camera's
                # complaint stands; a board that has faulted hands its
                # report back and the batch is the board's fault, with no
                # frames to keep.
                report = self.sequencer.wait_done(0.0)
                if report is None or not report.fault:
                    raise
                result = None
            else:
                _check_cancelled(context)
                report = wait_for_report(self.sequencer, context)
        finally:
            try:
                self.sequencer.safe()
            finally:
                if capture is not None and not capture.closed:
                    capture.close()
        return result, saturation_value, report

    def _prepare_artifacts(self, context: object) -> dict[str, Path]:
        root = Path(context.run_directory).expanduser().resolve()
        paths = {
            "root": root,
            "data": root / "data",
            "measurements": root / "data" / "measurements",
            "candidate_contexts": root / "candidates",
            "figures": root / "figures",
            "final": root / "final",
        }
        for name in (
            "data",
            "measurements",
            "candidate_contexts",
            "figures",
            "final",
        ):
            paths[name].mkdir(parents=True, exist_ok=True)
        site_path = _write_npz(
            paths["data"] / "sites.npz",
            arrays={
                "target_row": np.asarray(self._rows, dtype="<i8"),
                "target_column": np.asarray(self._columns, dtype="<i8"),
                "camera_center_xy": np.asarray(self._site_centers_xy, dtype="<f8"),
                "initial_target_weight": np.asarray(
                    self.target[self._rows, self._columns], dtype="<f4"
                ),
            },
            metadata={
                "format": "zlc.slm.feedback-sites",
                "readout_model_kind": self.model.kind.value,
                "readout_half_width": int(self.model.integration_half_width),
                "calibration_path": str(self.calibration_path),
                "science_context_path": str(self.science_context_path),
            },
        )
        context.register_artifact("site_geometry", site_path, role="process")
        return paths

    def _save_candidate_checkpoint(
        self,
        context: object,
        paths: Mapping[str, Path],
        *,
        candidate: int,
        samples: np.ndarray,
        measurement: Mapping[str, object],
        solver: Mapping[str, object] | None,
        phase: np.ndarray,
        pattern: np.ndarray,
        target: np.ndarray,
        history: list[dict[str, object]],
    ) -> Path:
        arrays: dict[str, object] = {
            "site_samples": np.asarray(samples, dtype="<f8"),
        }
        bool_fields = {
            "fit_valid",
            "observable_valid",
            "loading_edge",
            "single_population",
            "fit_invalid",
        }
        for name in _CANDIDATE_VECTOR_FIELDS:
            values = measurement[name]
            if name in {"decision", "probe_decision"}:
                arrays[name] = np.asarray(values, dtype="U32")
            elif name in bool_fields:
                arrays[name] = np.asarray(values, dtype=np.bool_)
            else:
                arrays[name] = np.asarray(
                    [np.nan if value is None else value for value in values],
                    dtype="<f8",
                )
        metadata = {
            key: value
            for key, value in measurement.items()
            if key not in _CANDIDATE_VECTOR_FIELDS and key != "artifact_path"
        }
        metadata.update(
            {
                "format": _FEEDBACK_MEASUREMENT_CHECKPOINT_CONTRACT,
                "solver": None if solver is None else dict(solver),
                "slm_command_receipt": dict(self.slm.last_command_receipt),
            }
        )
        data_path = _write_npz(
            paths["measurements"] / f"measurement-{int(candidate):04d}.npz",
            arrays=arrays,
            metadata=metadata,
        )
        context.register_artifact(
            f"measurement_{int(candidate):04d}",
            data_path,
            role="checkpoint",
        )
        candidate_metadata = self._candidate_metadata(
            candidate=int(candidate),
            status="checkpoint",
            history=history,
            solver=solver,
        )
        candidate_metadata.update(
            {
                "measurement_checkpoint": str(
                    data_path.relative_to(paths["root"]).as_posix()
                ),
            }
        )
        context_path = self._save_candidate(
            paths["candidate_contexts"] / f"candidate-{int(candidate):04d}.npz",
            pattern,
            target,
            candidate_metadata,
        )
        context.register_artifact(
            f"candidate_{int(candidate):04d}",
            context_path,
            role="checkpoint",
            contract_id=SLM_PHASE_ARTIFACT_CONTRACT,
        )
        return context_path

    def _save_figure(
        self,
        context: object,
        paths: Mapping[str, Path],
        name: str,
        *,
        snapshot: object,
        spec: object,
        parameters: Mapping[str, object] | None = None,
        size: str = "4x4",
        artifact_name: str | None = None,
        image_role: str = "preview",
        fit: Mapping[str, object] | None = None,
        device_event_record: Mapping[str, object],
    ) -> tuple[Path, Path]:
        if not isinstance(device_event_record, Mapping):
            raise TypeError("Figure device_event_record must be a mapping")
        registered_name = str(name if artifact_name is None else artifact_name)
        base = paths["figures"] / f"{name}.png"
        try:
            writer = self._save_figure_artifact
            if writer is None:
                from zlc_plot import save_figure_artifact as writer
            written = writer(
                base,
                plot_input=snapshot,
                spec=spec,
                parameters={} if parameters is None else parameters,
                size=size,
                fit=fit,
                source={
                    "task": self.instance_id,
                    "report": name,
                    "calibration_path": str(self.calibration_path),
                    "science_context_path": str(self.science_context_path),
                    "run_record": {
                        **self._run_record(),
                        **dict(device_event_record),
                    },
                },
            )
            if hasattr(written, "result"):
                written = written.result()
            image, archive = written
        except BaseException:
            archive = base.with_suffix(".npz")
            if archive.is_file():
                context.register_artifact(
                    f"{registered_name}_figure",
                    archive,
                    role="figure",
                    contract_id="zlc.figure",
                )
            raise
        context.register_artifact(
            f"{registered_name}_figure",
            archive,
            role="figure",
            contract_id="zlc.figure",
        )
        image_suffix = "preview" if image_role == "preview" else "image"
        context.register_artifact(
            f"{registered_name}_{image_suffix}", image, role=image_role
        )
        return image, archive

    def _candidate_fit_figure(
        self,
        measurement: Mapping[str, object],
        samples: np.ndarray,
        *,
        generation: str,
    ) -> tuple[object, FacetGridPlot]:
        """One candidate's true per-site Histogram Figure input."""

        values = np.asarray(samples, dtype=float)
        if values.ndim != 2 or values.shape[1] != self._site_count:
            raise ValueError("candidate histogram samples have the wrong shape")
        candidate = int(measurement["iteration"])
        shot_axis = AxisSpec(
            AxisId("slm_feedback.shot"),
            "shot",
            COMPONENT,
            values.shape[0],
        )
        site_axis = self._registered_site_map.site_axis
        snapshot = snapshot_from_array(
            values.T[None],
            producer=self.instance_id,
            signal=f"candidate_{candidate:04d}_site_histogram_figure",
            cell_axes=(site_axis, shot_axis),
            generation=generation,
            revision=candidate,
            validity=np.isfinite(values.T)[None],
        )
        kind = str(measurement.get("candidate_kind", "candidate"))
        return snapshot, FacetGridPlot(
            AxisRef.cell_data(str(site_axis.axis_id)),
            HistogramPlot(
                labels=PlotLabels(
                    title=f"Candidate {candidate} ({kind}) site histograms and fits",
                    x="site signal",
                    y="shots",
                ),
            ),
        )

    def _save_candidate_fit_figures(
        self,
        context: object,
        paths: Mapping[str, Path],
        reports: list[tuple[Mapping[str, object], np.ndarray]],
        *,
        generation: str,
    ) -> None:
        for measurement, samples in reports:
            candidate = int(measurement["iteration"])
            snapshot, spec = self._candidate_fit_figure(
                measurement,
                samples,
                generation=generation,
            )
            self._save_figure(
                context,
                paths,
                f"candidate_site_fits/candidate-{candidate:04d}",
                artifact_name=f"candidate_{candidate:04d}_site_fits",
                snapshot=snapshot,
                spec=spec,
                parameters={
                    "bin_count": min(60, max(10, samples.shape[0] // 2))
                },
                size="8x8",
                image_role="figure",
                fit={"model": "bimodal_gaussian", "fit_all_facets": True},
                device_event_record=measurement["device_event_record"],
            )

    def _save_figures(
        self,
        context: object,
        paths: Mapping[str, Path],
        *,
        history: list[dict[str, object]],
        selected: Mapping[str, object],
        initial_phase: np.ndarray,
        initial_mean_frame: np.ndarray,
        candidate_reports: list[tuple[Mapping[str, object], np.ndarray]],
    ) -> None:
        count = len(history)
        if count < 1:
            return
        generation = str(getattr(context.generation, "value", context.generation))
        self._save_candidate_fit_figures(
            context,
            paths,
            candidate_reports,
            generation=generation,
        )
        selected_history = selected.get("history")
        selected_device_record = (
            selected_history.get("device_event_record")
            if isinstance(selected_history, Mapping)
            else None
        )
        if not isinstance(selected_device_record, Mapping):
            raise RuntimeError("selected feedback candidate lost its device snapshot")
        candidate_id = AxisId("slm_feedback.candidate")
        candidate_axis = AxisSpec(
            candidate_id,
            "candidate",
            SCAN_POINT,
            count,
            tuple(range(1, count + 1)),
        )
        site_axis = self._registered_site_map.site_axis

        def site_history_snapshot(field: str, signal: str) -> object:
            values = np.asarray(
                [
                    [np.nan if value is None else value for value in item[field]]
                    for item in history
                ],
                dtype="<f8",
            )
            return snapshot_from_array(
                values[None],
                producer=self.instance_id,
                signal=signal,
                point_axes=(candidate_axis,),
                cell_axes=(site_axis,),
                generation=generation,
                revision=count,
                validity=np.isfinite(values)[None],
            )

        uniformity_axis = AxisSpec(
            AxisId("slm_feedback.uniformity.metric"),
            "metric",
            COMPONENT,
            3,
            (0, 1, 2),
            coordinate_labels=(
                "all sites", "observable sites", "expected noise floor"
            ),
        )
        uniformity = np.asarray(
            [
                [
                    np.nan if item[field] is None else item[field]
                    for field in (
                        "uniformity_ratio",
                        "observable_uniformity_ratio",
                        "expected_noise_ratio",
                    )
                ]
                for item in history
            ],
            dtype="<f8",
        )
        uniformity_snapshot = snapshot_from_array(
            uniformity[None],
            producer=self.instance_id,
            signal="uniformity_figure",
            point_axes=(candidate_axis,),
            cell_axes=(uniformity_axis,),
            generation=generation,
            revision=count,
            validity=np.isfinite(uniformity)[None],
        )
        self._save_figure(
            context,
            paths,
            "uniformity_history",
            snapshot=uniformity_snapshot,
            spec=CurvePlot(
                AxisRef.point(str(candidate_id)),
                group=AxisRef.cell_data(str(uniformity_axis.axis_id)),
                labels=PlotLabels(
                    title="SLM feedback uniformity",
                    x="candidate",
                    y="max / min bright-dark",
                ),
            ),
            device_event_record=selected_device_record,
        )

        self._save_figure(
            context,
            paths,
            "site_signal_evolution",
            snapshot=site_history_snapshot(
                "bright_minus_dark", "site_signal_figure"
            ),
            spec=CurvePlot(
                AxisRef.point(str(candidate_id)),
                group=AxisRef.cell_data(str(site_axis.axis_id)),
                labels=PlotLabels(
                    title="Per-site bright-dark evolution",
                    x="candidate",
                    y="bright - dark",
                ),
            ),
            device_event_record=selected_device_record,
        )

        self._save_figure(
            context,
            paths,
            "weight_evolution",
            snapshot=site_history_snapshot("control_weight", "weight_figure"),
            spec=CurvePlot(
                AxisRef.point(str(candidate_id)),
                group=AxisRef.cell_data(str(site_axis.axis_id)),
                labels=PlotLabels(
                    title="Per-site target weight evolution",
                    x="candidate",
                    y="normalized weight",
                ),
            ),
            device_event_record=selected_device_record,
        )

        selected_samples = np.asarray(selected["samples"], dtype="<f8")
        shot_axis = AxisSpec(
            AxisId("slm_feedback.shot"),
            "shot",
            COMPONENT,
            selected_samples.shape[0],
        )
        histogram_snapshot = snapshot_from_array(
            selected_samples.T[None],
            producer=self.instance_id,
            signal="selected_histogram_figure",
            cell_axes=(site_axis, shot_axis),
            generation=generation,
            revision=count,
        )
        self._save_figure(
            context,
            paths,
            "selected_site_histograms",
            snapshot=histogram_snapshot,
            spec=FacetGridPlot(
                AxisRef.cell_data(str(site_axis.axis_id)),
                HistogramPlot(
                    labels=PlotLabels(
                        title="Selected candidate site distributions",
                        x="site signal",
                        y="shots",
                    )
                ),
            ),
            parameters={"bin_count": min(60, max(10, self.shots // 2))},
            device_event_record=selected_device_record,
        )

        selected_number = int(selected["candidate"])
        comparison_id = AxisId("slm_feedback.comparison")
        comparison_axis = AxisSpec(
            comparison_id,
            "state",
            SCAN_POINT,
            2,
            (0, selected_number),
            coordinate_labels=("initial", f"selected {selected_number}"),
        )

        def comparison_snapshot(
            name: str,
            initial: object,
            retained: object,
            *,
            unit: str | None = None,
        ) -> tuple[object, AxisSpec, AxisSpec]:
            first = np.asarray(initial, dtype="<f4")
            second = np.asarray(retained, dtype="<f4")
            if first.ndim != 2 or second.shape != first.shape:
                raise ValueError(f"{name} comparison images differ in shape")
            y_axis = AxisSpec(
                AxisId(f"slm_feedback.{name}.y"),
                f"{name} y",
                SPATIAL_Y,
                first.shape[0],
            )
            x_axis = AxisSpec(
                AxisId(f"slm_feedback.{name}.x"),
                f"{name} x",
                SPATIAL_X,
                first.shape[1],
            )
            return (
                snapshot_from_array(
                    np.stack((first, second), axis=0)[None],
                    producer=self.instance_id,
                    signal=f"{name}_comparison_figure",
                    point_axes=(comparison_axis,),
                    cell_axes=(y_axis, x_axis),
                    value_unit=unit,
                    generation=generation,
                    revision=count,
                ),
                x_axis,
                y_axis,
            )

        camera_snapshot, image_x, image_y = comparison_snapshot(
            "camera", initial_mean_frame, selected["mean_frame"]
        )
        camera_overlay_geometry = image_point_overlay_geometry(
            camera_snapshot,
            self._registered_site_map.centers_xy,
            self._registered_site_map.site_ids,
            status_axis=site_axis,
            labels=tuple(str(site) for site in range(1, self._site_count + 1)),
            coordinates_are_indices=True,
        )
        self._save_figure(
            context,
            paths,
            "camera_initial_selected",
            snapshot=ImageFrame(
                camera_snapshot,
                ImagePointOverlay(
                    revision=count,
                    coordinates=np.asarray(
                        camera_overlay_geometry["coordinates_xy"], dtype=float
                    ),
                    point_ids=tuple(camera_overlay_geometry["point_ids"]),
                    labels=tuple(camera_overlay_geometry["labels"]),
                    static_statuses=tuple(
                        PointStatus.UNKNOWN for _ in range(self._site_count)
                    ),
                ),
            ),
            spec=FacetGridPlot(
                AxisRef.point(str(comparison_id)),
                ImagePlot(
                    AxisRef.cell_data(str(image_x.axis_id)),
                    AxisRef.cell_data(str(image_y.axis_id)),
                ),
                labels=PlotLabels(title="Initial and selected camera mean"),
            ),
            device_event_record=selected_device_record,
        )

        phase_snapshot, phase_x, phase_y = comparison_snapshot(
            "phase", initial_phase, selected["phase"], unit="rad"
        )
        self._save_figure(
            context,
            paths,
            "phase_initial_selected",
            snapshot=phase_snapshot,
            spec=FacetGridPlot(
                AxisRef.point(str(comparison_id)),
                ImagePlot(
                    AxisRef.cell_data(str(phase_x.axis_id)),
                    AxisRef.cell_data(str(phase_y.axis_id)),
                ),
                labels=PlotLabels(title="Initial and selected SLM phase"),
            ),
            device_event_record=selected_device_record,
        )

    def _write_summary(
        self,
        context: object,
        paths: Mapping[str, Path],
        *,
        status: str,
        history: list[dict[str, object]],
        selected_candidate: int | None,
        outcome: Mapping[str, object] | None = None,
        error: BaseException | None = None,
        rollback: Mapping[str, object] | None = None,
        figures_error: BaseException | None = None,
    ) -> None:
        initial = None if not history else history[0]
        formal_history = [
            item for item in history if item["candidate_kind"] != "probe"
        ]
        selected = next(
            (
                item
                for item in history
                if int(item["iteration"]) == selected_candidate
            ),
            None,
        )
        common_mask = (
            np.logical_and.reduce(
                [
                    np.asarray(item["observable_valid"], dtype=bool)
                    for item in formal_history
                ]
            )
            if formal_history
            else np.zeros(self._site_count, dtype=bool)
        )
        common_site_count = int(np.count_nonzero(common_mask))
        common_totals: list[float | None] = []
        for item in history:
            contrast = np.asarray(
                [np.nan if value is None else value for value in item["bright_minus_dark"]],
                dtype=float,
            )
            values = contrast[common_mask]
            common_totals.append(
                float(np.sum(values))
                if len(values) and np.all(np.isfinite(values))
                else None
            )
        selected_common_total = next(
            (
                total
                for item, total in zip(history, common_totals, strict=True)
                if int(item["iteration"]) == selected_candidate
            ),
            None,
        )
        document = {
            "format": "zlc.slm.feedback-summary",
            "status": str(status),
            "selected_candidate": selected_candidate,
            "candidate_count": len(history),
            "settings": self._run_record(),
            "outcome": None if outcome is None else dict(outcome),
            "plant_slope_history": [
                {
                    "candidate": item["iteration"],
                    "estimate": item["plant_slope_estimate"],
                    "standard_error": item["plant_slope_se"],
                    "source": item["plant_slope_source"],
                }
                for item in formal_history
            ],
            "probe_candidates": [
                {
                    "candidate": item["iteration"],
                    "requested_factor": item["probe_requested_factor"],
                    "group_effective_factor": item[
                        "probe_group_effective_factor"
                    ],
                    "site_effective_factor": item[
                        "probe_effective_factor"
                    ],
                    "sites": item["probe_sites"],
                    "observable_sites": item["observable_sites"],
                }
                for item in history
                if item["candidate_kind"] == "probe"
            ],
            "probe_combination": next(
                (
                    {
                        "candidate": item["iteration"],
                        "selected_formal_factor": item[
                            "probe_selected_formal_factor"
                        ],
                        "decision": item["probe_decision"],
                    }
                    for item in history
                    if item["candidate_kind"] == "probe_combined"
                ),
                None,
            ),
            "final_probe_control_boundary": (
                None if not history else history[-1]["probe_control_boundary"]
            ),
            "initial_uniformity_ratio": (
                None if initial is None else initial["uniformity_ratio"]
            ),
            "initial_observable_uniformity_ratio": (
                None if initial is None else initial["observable_uniformity_ratio"]
            ),
            "selected_uniformity_ratio": (
                None if selected is None else selected["uniformity_ratio"]
            ),
            "selected_observable_uniformity_ratio": (
                None if selected is None else selected["observable_uniformity_ratio"]
            ),
            "selected_observable_sites": (
                None if selected is None else selected["observable_sites"]
            ),
            "selected_total_observable_bright_minus_dark": (
                None
                if selected is None
                else selected["total_observable_bright_minus_dark"]
            ),
            "actual_exposure_seconds": self._actual_exposure_seconds,
            "effective_photoelectrons": self._effective_photoelectrons,
            "effective_count_unit": self._effective_count_unit,
            "selected_uniformity_confidence_lower": (
                None if selected is None else selected["uniformity_confidence_lower"]
            ),
            "selected_uniformity_confidence_upper": (
                None if selected is None else selected["uniformity_confidence_upper"]
            ),
            "selected_maximum_relative_standard_error": (
                None
                if selected is None
                else selected["maximum_relative_standard_error"]
            ),
            "selected_true_uniformity_cv": (
                None if selected is None else selected["true_uniformity_cv"]
            ),
            "selected_expected_noise_ratio": (
                None if selected is None else selected["expected_noise_ratio"]
            ),
            "common_observable_sites": common_site_count,
            "selected_common_site_total_bright_minus_dark": selected_common_total,
            "uniformity_history": [
                {
                    "candidate": item["iteration"],
                    "all_sites": item["uniformity_ratio"],
                    "observable_sites": item["observable_uniformity_ratio"],
                    "observable_site_count": item["observable_sites"],
                    "total_observable_bright_minus_dark": item[
                        "total_observable_bright_minus_dark"
                    ],
                    "common_site_total_bright_minus_dark": common_total,
                    "true_uniformity_cv": item["true_uniformity_cv"],
                    "expected_noise_ratio": item["expected_noise_ratio"],
                    "converged": item["converged"],
                    "plant_slope_estimate": item["plant_slope_estimate"],
                    "plant_slope_source": item["plant_slope_source"],
                }
                for item, common_total in zip(history, common_totals, strict=True)
            ],
            "rollback": None if rollback is None else dict(rollback),
            "error": (
                None
                if error is None
                else {
                    "type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": str(error),
                }
            ),
            "figures_error": (
                None
                if figures_error is None
                else {
                    "type": (
                        f"{type(figures_error).__module__}."
                        f"{type(figures_error).__qualname__}"
                    ),
                    "message": str(figures_error),
                }
            ),
        }
        json_path = write_readable_json(
            paths["root"] / "summary.json", _plain_json(document)
        )
        final_slope = (
            document["plant_slope_history"][-1]
            if document["plant_slope_history"]
            else None
        )
        lines = [
            f"SLM feedback status: {status}",
            f"Candidates measured: {len(history)}",
            f"Selected candidate: {selected_candidate if selected_candidate is not None else 'none'}",
            "Final plant slope: "
            + (
                "none"
                if final_slope is None
                else f"{final_slope['estimate']} +/- {final_slope['standard_error']} "
                f"({final_slope['source']})"
            ),
        ]
        if outcome is not None and outcome.get("reason"):
            lines.append(f"Outcome: {outcome['reason']}")
        for item in document["probe_candidates"]:
            lines.append(
                "Probe candidate "
                f"{item['candidate']}: {item['requested_factor']}x requested, "
                f"{item['group_effective_factor']}x effective, "
                f"{len(item['sites'])} sites"
            )
        if initial is not None:
            lines.append(
                "Initial observable-site uniformity ratio: "
                f"{initial['observable_uniformity_ratio']}"
            )
        if selected is not None:
            lines.extend(
                (
                    f"All-site uniformity ratio: {selected['uniformity_ratio']}",
                    "Observable-site uniformity ratio: "
                    f"{selected['observable_uniformity_ratio']}",
                    f"Observable sites: {selected['observable_sites']}/{self._site_count}",
                    f"Common observable sites: {common_site_count}/{self._site_count}",
                    "Selected common-site total bright-dark: "
                    f"{selected_common_total}",
                    "Simultaneous 95% interval: "
                    f"[{selected['uniformity_confidence_lower']}, "
                    f"{selected['uniformity_confidence_upper']}]",
                    "True between-site contrast CV (split-half): "
                    f"{selected['true_uniformity_cv']}",
                    "Expected max/min of a uniform array at this noise: "
                    f"{selected['expected_noise_ratio']}",
                )
            )
        if rollback is not None:
            lines.append(f"Rollback: {rollback.get('status')}")
        if error is not None:
            lines.append(f"Error: {type(error).__name__}: {error}")
        if figures_error is not None:
            lines.append(
                "Figures not written: "
                f"{type(figures_error).__name__}: {figures_error}"
            )
        text_path = atomic_write_text(
            paths["root"] / "summary.txt", "\n".join(lines) + "\n"
        )
        context.register_artifact("summary_json", json_path, role="summary")
        context.register_artifact("summary_text", text_path, role="summary")

    def _finish_candidate(
        self,
        context: object,
        candidate: dict[str, object],
        history: list[dict[str, object]],
        *,
        paths: Mapping[str, Path],
        initial_phase: np.ndarray,
        initial_mean_frame: np.ndarray | None,
        candidate_reports: list[tuple[Mapping[str, object], np.ndarray]],
        status: str,
        republish: bool,
        error: BaseException | None = None,
    ) -> dict[str, object]:
        """Seal one candidate as the run's result: SLM, artifact, figures, summary.

        Every way a run ends -- completed, stalled, stopped, FAILED -- passes
        through here with the candidate it keeps, so ``final/`` and the SLM
        never disagree about what was chosen.  The failure path used to skip
        this and restore the incoming phase instead: a run that had selected
        candidate 66 of 68 died on one pulse poll with an empty ``final/``
        and the SLM showing the phase it started from.

        The artifact is the deliverable, the figures are the report, and a
        report that fails must not take the deliverable with it: the figures
        are written after the artifact, and a figure writer that raises does
        not raise out of the seal.  It is recorded in the summary, told to
        the operator, and noted on the run's error when there is one.
        Letting it out of here made the failure handler seal the same
        candidate a second time, fail at the same figures, and then restore
        the incoming phase -- ``final/`` claiming candidate 3 with the SLM on
        the phase the run started from, and a summary saying "restored".
        """

        expected = canonical_phase(candidate["phase"], self.slm.shape_yx)
        observed = self.slm.last_commanded_phase
        applied = (
            expected
            if observed is not None and np.array_equal(observed, expected)
            else self._apply_exact(expected)
        )
        candidate_number = int(candidate["candidate"])
        if republish:
            candidate_history = candidate.get("history")
            device_record = (
                candidate_history.get("device_event_record")
                if isinstance(candidate_history, Mapping)
                else None
            )
            if not isinstance(device_record, Mapping):
                # NEVER MEASURED IS NOT LOST.  A candidate that has been
                # applied but not yet shot carries no history at all, which
                # is exactly what an operator Stop before the first batch
                # leaves behind -- and reading that as a missing snapshot
                # turned a graceful stop into a failed node, so the seal
                # never happened and the original cancellation came back out
                # of the handler.  A candidate that HAS a history and lost
                # its record is still a real defect and still raises.
                if isinstance(candidate_history, Mapping):
                    raise RuntimeError(
                        "selected feedback candidate lost its device snapshot"
                    )
                device_record = self._device_event_record(
                    include_measurement=False,
                    candidate=candidate_number,
                )
            self._publish_candidate(
                context,
                phase=applied,
                candidate=candidate_number,
                history=history,
                device_event_record=device_record,
            )
        retained_history = candidate.get("history")
        artifact_path = paths["final"] / "science-context.npz"
        metadata = self._candidate_metadata(
            candidate=(candidate_number if isinstance(retained_history, Mapping) else 0),
            status=status,
            history=history,
            solver=candidate.get("solver"),
        )
        outcome = candidate.get("outcome")
        metadata["outcome"] = (
            dict(outcome)
            if isinstance(outcome, Mapping)
            else {
                "status": str(status),
                "selected_candidate": (
                    candidate_number if isinstance(retained_history, Mapping) else None
                ),
                "candidates_measured": len(history),
            }
        )
        self._save_candidate(
            artifact_path,
            candidate["pattern_phase"],
            candidate["target"],
            metadata,
        )
        context.register_artifact(
            "artifact_path",
            artifact_path,
            role="final",
            contract_id=SLM_PHASE_ARTIFACT_CONTRACT,
        )
        figures_error: BaseException | None = None
        if (
            history
            and isinstance(retained_history, Mapping)
            and candidate.get("samples") is not None
            and candidate.get("mean_frame") is not None
            and initial_mean_frame is not None
        ):
            try:
                self._save_figures(
                    context,
                    paths,
                    history=history,
                    selected=candidate,
                    initial_phase=initial_phase,
                    initial_mean_frame=initial_mean_frame,
                    candidate_reports=candidate_reports,
                )
            except Exception as figure_error:
                figures_error = figure_error
                context.report_progress(
                    "Feedback figures were not written: "
                    f"{type(figure_error).__name__}: {figure_error}"
                )
                if error is not None:
                    error.add_note(
                        "Feedback figures were not written: "
                        f"{type(figure_error).__name__}: {figure_error}"
                    )
        self._write_summary(
            context,
            paths,
            status=status,
            history=history,
            selected_candidate=(
                candidate_number if isinstance(retained_history, Mapping) else None
            ),
            outcome=metadata["outcome"],
            error=error,
            figures_error=figures_error,
        )
        terminal_uniformity = (
            retained_history.get("uniformity_ratio")
            if isinstance(retained_history, Mapping)
            and bool(retained_history.get("uniformity_complete"))
            else None
        )
        return {
            "artifact_path": artifact_path,
            "terminal_uniformity": terminal_uniformity,
            "feedback_status": str(status),
            "uniformity_confidence_lower": (
                retained_history.get("uniformity_confidence_lower")
                if isinstance(retained_history, Mapping)
                else None
            ),
            "uniformity_confidence_upper": (
                retained_history.get("uniformity_confidence_upper")
                if isinstance(retained_history, Mapping)
                else None
            ),
            "maximum_relative_standard_error": (
                retained_history.get("maximum_relative_standard_error")
                if isinstance(retained_history, Mapping)
                else None
            ),
            "true_uniformity_cv": (
                retained_history.get("true_uniformity_cv")
                if isinstance(retained_history, Mapping)
                else None
            ),
            "expected_noise_ratio": (
                retained_history.get("expected_noise_ratio")
                if isinstance(retained_history, Mapping)
                else None
            ),
            "updates": len(history),
            "feedback_mode": self.feedback_mode,
            "requested_exposure_seconds": self.exposure_seconds,
            "actual_exposure_seconds": self._actual_exposure_seconds,
        }

    def execute(self, context: object) -> dict[str, object]:
        self._actual_device_snapshots = {}
        self._actual_exposure_seconds = None
        self._effective_photoelectrons = None
        self._effective_count_unit = None
        incoming = self._incoming_phase
        incoming_pattern = self._pattern_phase
        paths = self._prepare_artifacts(context)
        history: list[dict[str, object]] = []
        candidate_reports: list[
            tuple[Mapping[str, object], np.ndarray]
        ] = []
        retained_valid: dict[str, object] | None = None
        most_visible_observed: dict[str, object] | None = None
        last_completed_candidate: dict[str, object] | None = None
        initial_mean_frame: np.ndarray | None = None
        stalled = False
        termination_reason = "all authored feedback updates completed"
        try:
            _check_cancelled(context)
            # Science Context is the requested starting CONTENT, not proof of
            # what a previous process happens to have commanded. This Task owns
            # the SLM now, so establish and confirm that starting state itself.
            self._mapping_revision = int(self.slm.mapping_revision)
            self._last_measured_phase = None
            incoming = self._apply_exact(incoming)
            pulse = resolve_pulse(
                self.sequence,
                path=self.pulse_path,
                sequencer=self.sequencer,
                api_values={},
            )
            _check_cancelled(context)
            current_target = self.target
            # The Target on the SLM is the control Target with this
            # candidate's identification excitation on its sites (see
            # ``_excited_target``); the baseline is the Context's own.
            current_solved_target = self.target
            current_excitation = np.zeros(self._site_count, dtype=float)
            excitation_rng = np.random.default_rng(0)
            spot_optimizer_state: dict[str, object] = {}
            current_pattern = incoming_pattern
            current_phase = incoming
            solver_metadata: Mapping[str, object] | None = None
            previous_weights = np.full(self._site_count, np.nan, dtype=float)
            previous_contrast = np.full(self._site_count, np.nan, dtype=float)
            prior_measurement = self._prior_pattern_metadata.get("measurement")
            prior_probe_factors = self._prior_pattern_metadata.get("probe_factors")
            comparable_history = bool(
                self._prior_pattern_metadata.get("feedback_controller")
                == _CONTROLLER_CONTRACT
                and self._prior_pattern_metadata.get("feedback_mode") == self.feedback_mode
                and self._prior_pattern_metadata.get("pulse_path") == str(self.pulse_path)
                and type(self._prior_pattern_metadata.get("exposure_seconds"))
                in (int, float)
                and float(self._prior_pattern_metadata["exposure_seconds"])
                == self.exposure_seconds
                and isinstance(prior_probe_factors, (tuple, list))
                and tuple(float(value) for value in prior_probe_factors)
                == self.probe_factors
                and float(self._prior_pattern_metadata.get("feedback_gain", -1.0))
                == self.feedback_gain
                and float(
                    self._prior_pattern_metadata.get("maximum_weight_change", -1.0)
                )
                == self.maximum_weight_change
                and isinstance(prior_measurement, Mapping)
            )
            if comparable_history:
                prior = prior_measurement
                assert isinstance(prior, Mapping)
                def restored(name: str, dtype: object) -> np.ndarray | None:
                    try:
                        return np.asarray(prior[name], dtype=dtype).reshape(
                            self._site_count
                        )
                    except (KeyError, TypeError, ValueError):
                        return None

                restored_control = restored(
                    "previous_double_control_weight", float
                )
                if restored_control is not None:
                    previous_weights[:] = restored_control
                restored_contrast = restored(
                    "previous_double_bright_minus_dark", float
                )
                if restored_contrast is not None:
                    previous_contrast[:] = restored_contrast

                prior_weights = restored("control_weight", float)
                prior_contrast = restored("bright_minus_dark", float)
                prior_fit_valid = restored("fit_valid", bool)
                if (
                    prior_weights is not None
                    and prior_contrast is not None
                    and prior_fit_valid is not None
                ):
                    usable = (
                        prior_fit_valid
                        & np.isfinite(prior_weights)
                        & np.isfinite(prior_contrast)
                    )
                    previous_weights[usable] = prior_weights[usable]
                    previous_contrast[usable] = prior_contrast[usable]

            candidate_number = 0
            candidate_kind = "baseline"
            formal_updates = 0
            # The plant record: every candidate's weights as the SLM saw them
            # (excitation included), the excitation itself, and the contrasts
            # they produced.  The slope estimate pools all of it; a new run
            # starts without one.
            plant_log_weights: list[np.ndarray] = []
            plant_log_contrast: list[np.ndarray] = []
            plant_excitation: list[np.ndarray] = []
            convergence_streak = 0
            probe_sites = np.zeros(self._site_count, dtype=bool)
            probe_episode_used = np.zeros(self._site_count, dtype=bool)
            probe_baseline_target: np.ndarray | None = None
            probe_baseline_pattern: np.ndarray | None = None
            probe_baseline_optimizer_state: dict[str, object] | None = None
            probe_baseline_contrast = np.full(
                self._site_count, np.nan, dtype=float
            )
            probe_baseline_error = np.full(
                self._site_count, np.nan, dtype=float
            )
            probe_baseline_valid = np.zeros(self._site_count, dtype=bool)
            probe_baseline_reference_valid = np.zeros(
                self._site_count, dtype=bool
            )
            probe_baseline_dark = np.zeros(self._site_count, dtype=bool)
            probe_baseline_edge = np.zeros(self._site_count, dtype=bool)
            pending_probes: list[
                tuple[float, float, np.ndarray, np.ndarray]
            ] = []
            probe_measurements: list[
                tuple[float, np.ndarray, np.ndarray, np.ndarray]
            ] = []
            active_probe_requested: float | None = None
            active_probe_effective: float | None = None
            active_probe_site_factors: np.ndarray | None = None
            probe_selected_factors = np.ones(self._site_count, dtype=float)
            # Every site's bracket: the direction its loading edge lies in,
            # a share it was single at and a share it was loaded at (see
            # ``_updated_brackets``).  Kept across the whole run; a probe
            # episode only ever seeds it for sites that had none.
            bracket_direction = np.zeros(self._site_count, dtype=float)
            probe_single_bound = np.full(
                self._site_count, np.nan, dtype=float
            )
            probe_observable_bound = np.full(
                self._site_count, np.nan, dtype=float
            )
            probe_control_boundary = np.full(
                self._site_count, np.nan, dtype=float
            )
            probe_decisions = np.full(
                self._site_count, "not_probed", dtype="<U32"
            )
            probe_baseline_candidate: dict[str, object] | None = None
            forced_terminal_candidate: dict[str, object] | None = None
            def take_probe() -> tuple[np.ndarray, float, float, np.ndarray]:
                requested, effective, target, site_factors = pending_probes.pop(0)
                return target, requested, effective, site_factors

            while candidate_number < self._candidate_capacity:
                _check_cancelled(context)
                applied = (
                    current_phase
                    if candidate_number == 0
                    else self._apply_exact(current_phase)
                )
                candidate_number += 1
                iteration = candidate_number - 1
                candidate_solver = solver_metadata
                if candidate_kind == "probe":
                    context.report_progress(
                        "Measuring SLM feedback probe "
                        f"{active_probe_requested:g}x requested / "
                        f"{active_probe_effective:g}x effective for "
                        f"{int(np.count_nonzero(probe_sites))} sites"
                    )
                elif candidate_kind == "probe_combined":
                    context.report_progress(
                        "Measuring combined SLM probe decision for "
                        f"{int(np.count_nonzero(probe_sites))} sites"
                    )
                else:
                    context.report_progress(
                        f"Measuring SLM feedback candidate {candidate_number}"
                    )
                # The operator must see the phase which is already on the SLM
                # before this candidate's camera exposure begins.  Publishing
                # only after the shot batch made Monitor lag one candidate and
                # falsely look like an unchanged phase was being remeasured.
                self._publish_candidate(
                    context,
                    phase=applied,
                    candidate=candidate_number,
                    history=history,
                    device_event_record=self._device_event_record(
                        include_measurement=False,
                        candidate=candidate_number,
                    ),
                )
                (
                    samples,
                    saturated_sites,
                    missing_sites,
                    mean_frame,
                    pulse_warning,
                ) = self._measure(pulse, context, iteration)
                if initial_mean_frame is None:
                    initial_mean_frame = np.array(mean_frame, copy=True)
                fitted = _fit_contrasts(samples)
                fit_valid = np.asarray(fitted["valid"], dtype=bool).copy()
                fit_single = np.asarray(
                    fitted["single_population"], dtype=bool
                ).copy()
                fit_invalid = np.asarray(fitted["invalid"], dtype=bool).copy()
                acquisition_invalid = np.zeros(self._site_count, dtype=bool)
                acquisition_invalid[list(saturated_sites)] = True
                acquisition_invalid[list(missing_sites)] = True
                fit_valid[acquisition_invalid] = False
                fit_single[acquisition_invalid] = False
                fit_invalid |= acquisition_invalid
                fitted["valid"] = fit_valid
                fitted["single_population"] = fit_single
                fitted["invalid"] = fit_invalid
                contrast = np.asarray(fitted["contrast"], dtype=float)
                error = np.asarray(fitted["standard_error"], dtype=float)
                observable_valid = fit_valid
                loading_edge = _loading_edge(
                    fitted["bright_fraction"], observable_valid
                )
                odd_contrast = np.asarray(fitted["odd_contrast"], dtype=float).copy()
                even_contrast = np.asarray(fitted["even_contrast"], dtype=float).copy()
                odd_contrast[~observable_valid] = np.nan
                even_contrast[~observable_valid] = np.nan
                true_variance, true_variance_error = _split_half_dispersion(
                    odd_contrast, even_contrast, observable_valid
                )
                true_uniformity_cv = (
                    float(np.sqrt(max(true_variance, 0.0)))
                    if np.isfinite(true_variance)
                    else float("nan")
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    expected_noise_ratio = _expected_noise_ratio(
                        error / contrast, observable_valid
                    )
                unobservable_single = _unobservable_single(
                    fit_single,
                    observable_valid,
                    acquisition_invalid,
                )
                valid = bool(np.all(observable_valid))
                observable_contrast = contrast[observable_valid]
                total_observable_contrast = (
                    float(np.sum(observable_contrast))
                    if len(observable_contrast)
                    and np.all(np.isfinite(observable_contrast))
                    else float("nan")
                )
                observed_score = (
                    float(np.max(observable_contrast) / np.min(observable_contrast))
                    if len(observable_contrast)
                    and np.all(np.isfinite(observable_contrast))
                    and np.all(observable_contrast > 0.0)
                    else float("nan")
                )
                if valid:
                    score, confidence_lower, confidence_upper, relative_sem = (
                        _ratio_interval(contrast, error)
                    )
                else:
                    score = confidence_lower = confidence_upper = relative_sem = float("inf")
                visibility = int(np.count_nonzero(observable_valid))
                visible_margin = contrast[observable_valid] - float(
                    special.ndtri(
                        1.0
                        - 0.05 / (2.0 * self._site_count)
                    )
                ) * error[observable_valid]
                visibility_margin = (
                    None
                    if not len(visible_margin)
                    else float(np.min(visible_margin))
                )
                current_weights = np.asarray(
                    current_solved_target[self._rows, self._columns], dtype=float
                )
                current_control_weights = _control_weights(
                    np.asarray(current_target[self._rows, self._columns], dtype=float)
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    plant_log_weights.append(
                        np.log(_control_weights(current_weights))
                    )
                    plant_log_contrast.append(
                        np.where(observable_valid, np.log(contrast), np.nan)
                    )
                    plant_excitation.append(np.array(current_excitation, copy=True))
                plant_slope_estimate, plant_slope_se, plant_slope_rows = _plant_slope(
                    plant_log_weights,
                    plant_log_contrast,
                    plant_excitation,
                )
                plant_slope_magnitude = _usable_plant_slope(
                    len(plant_log_weights), plant_slope_estimate, plant_slope_se
                )
                plant_slope_source = (
                    "assumed" if plant_slope_magnitude is None else "estimated"
                )
                # CONVERGED means: every site loaded and resolved, and the two
                # half batches agree that whatever dispersion is left is
                # within this batch's own power to see.  Three formal
                # candidates in a row saying so end the run; a diagnostic
                # probe neither counts nor breaks the streak.
                converged = bool(
                    valid
                    and np.isfinite(true_variance)
                    and np.isfinite(true_variance_error)
                    and true_variance <= true_variance_error
                )
                if candidate_kind != "probe":
                    convergence_streak = convergence_streak + 1 if converged else 0
                # Bracket bookkeeping runs on every measured candidate: a
                # probe's non-probe sites are evidence too.  The probe sites
                # of an episode in flight are the verdict's to seed.
                bookkept = (
                    ~probe_sites if candidate_kind == "probe" else
                    np.ones(self._site_count, dtype=bool)
                )
                (
                    probe_single_bound,
                    probe_observable_bound,
                    bracket_direction,
                ) = _updated_brackets(
                    current_control_weights,
                    unobservable_single & bookkept,
                    observable_valid & bookkept,
                    previous_weights,
                    probe_single_bound,
                    probe_observable_bound,
                    bracket_direction,
                )
                probe_control_boundary[:] = _probe_boundary(
                    probe_single_bound,
                    probe_observable_bound,
                    bracket_direction,
                )
                bracket_step, bracket_kind = _single_bracket_step(
                    current_control_weights,
                    unobservable_single,
                    probe_single_bound,
                    probe_observable_bound,
                    bracket_direction,
                    self.maximum_weight_change,
                )
                episode_probe_sites = np.zeros(self._site_count, dtype=bool)
                starts_probe_episode = False
                if candidate_kind != "probe":
                    episode_probe_sites = (
                        _needs_probe(
                            fit_single,
                            observable_valid,
                            acquisition_invalid,
                            bracket_direction,
                            previous_weights,
                            previous_contrast,
                        )
                        & ~probe_episode_used
                        & (formal_updates < self.max_updates)
                    )
                    starts_probe_episode = bool(np.any(episode_probe_sites))
                feedback_valid = observable_valid
                if starts_probe_episode:
                    probe_sites = np.array(episode_probe_sites, copy=True)
                    probe_episode_used[probe_sites] = True
                    pending_probes.clear()
                    probe_measurements.clear()
                    probe_selected_factors[probe_sites] = 1.0
                    probe_decisions[probe_sites] = "not_probed"
                    probe_baseline_candidate = None
                    probe_baseline_target = np.array(
                        current_target, dtype=np.float32, copy=True
                    )
                    probe_baseline_pattern = np.array(
                        current_pattern, dtype=np.float32, copy=True
                    )
                    probe_baseline_optimizer_state = deepcopy(
                        spot_optimizer_state
                    )
                    probe_baseline_contrast[:] = contrast
                    probe_baseline_error[:] = error
                    probe_baseline_valid[:] = observable_valid
                    probe_baseline_reference_valid[:] = fit_valid
                    probe_baseline_dark[:] = unobservable_single
                    probe_baseline_edge[:] = loading_edge
                    for requested_factor in self.probe_factors:
                        requested = np.ones(self._site_count, dtype=float)
                        requested[probe_sites] = requested_factor
                        probe_target, effective = _relative_probe_target(
                            probe_baseline_target,
                            requested,
                            probe_sites,
                            self._rows,
                            self._columns,
                        )
                        if np.array_equal(probe_target, probe_baseline_target):
                            continue
                        if any(
                            np.array_equal(probe_target, item[2])
                            for item in pending_probes
                        ):
                            continue
                        pending_probes.append((
                            requested_factor,
                            float(effective[probe_sites][0]),
                            probe_target,
                            np.array(effective, copy=True),
                        ))
                record_probe_effective = np.full(
                    self._site_count, np.nan, dtype=float
                )
                record_probe_selection = np.full(
                    self._site_count, np.nan, dtype=float
                )
                record_probe_decisions = np.full(
                    self._site_count, "not_probed", dtype="<U32"
                )
                if starts_probe_episode:
                    record_probe_decisions[probe_sites] = "probe_required"
                elif candidate_kind == "probe":
                    assert active_probe_site_factors is not None
                    record_probe_effective[:] = active_probe_site_factors
                    record_probe_decisions[probe_sites] = "probe_measurement"
                elif candidate_kind in {"probe_combined", "ordinary"}:
                    record_probe_selection[probe_sites] = (
                        probe_selected_factors[probe_sites]
                    )
                    record_probe_decisions[:] = probe_decisions
                if candidate_kind == "probe":
                    proposed_target = np.array(
                        current_target, dtype=np.float32, copy=True
                    )
                    log_correction = np.zeros(self._site_count, dtype=float)
                    decisions = np.full(
                        self._site_count, "hold_for_probe", dtype="<U32"
                    )
                else:
                    directed_step = np.array(bracket_step, copy=True)
                    if starts_probe_episode:
                        directed_step[probe_sites] = 0.0
                    proposed_target, log_correction, decisions = (
                        _updated_target(
                            current_target,
                            contrast,
                            error,
                            feedback_valid,
                            self._rows,
                            self._columns,
                            reference_valid=fit_valid,
                            feedback_gain=self.feedback_gain,
                            plant_slope=plant_slope_magnitude,
                            maximum_weight_change=self.maximum_weight_change,
                            directed_log_step=directed_step,
                            control_boundary=probe_control_boundary,
                            control_direction=bracket_direction,
                            loading_edge=loading_edge,
                        )
                    )
                    stepped = unobservable_single & (directed_step != 0.0)
                    decisions[stepped] = bracket_kind[stepped]
                    if starts_probe_episode:
                        decisions[probe_sites] = "hold_for_probe"
                measured_device_record = self._device_event_record(
                    include_measurement=True,
                    candidate=candidate_number,
                )
                history.append({
                    "iteration": candidate_number,
                    "shots": self.shots,
                    "valid": valid,
                    "uniformity_complete": valid,
                    "observable_sites": visibility,
                    "total_sites": self._site_count,
                    "saturated": bool(saturated_sites),
                    "saturated_sites": list(saturated_sites),
                    "uniformity_ratio": (
                        observed_score
                        if valid and np.isfinite(observed_score)
                        else None
                    ),
                    "observable_uniformity_ratio": (
                        None if not np.isfinite(observed_score) else observed_score
                    ),
                    "total_observable_bright_minus_dark": (
                        None
                        if not np.isfinite(total_observable_contrast)
                        else total_observable_contrast
                    ),
                    "uniformity_confidence_lower": (
                        None if not valid else confidence_lower
                    ),
                    "uniformity_confidence_upper": (
                        None if not valid else confidence_upper
                    ),
                    "maximum_relative_standard_error": (
                        None if not valid else relative_sem
                    ),
                    "true_uniformity_variance": (
                        None if not np.isfinite(true_variance) else true_variance
                    ),
                    "true_uniformity_variance_error": (
                        None
                        if not np.isfinite(true_variance_error)
                        else true_variance_error
                    ),
                    "true_uniformity_cv": (
                        None if not np.isfinite(true_uniformity_cv) else true_uniformity_cv
                    ),
                    "expected_noise_ratio": (
                        None
                        if not np.isfinite(expected_noise_ratio)
                        else expected_noise_ratio
                    ),
                    "converged": converged,
                    "convergence_streak": convergence_streak,
                    "feedback_gain": self.feedback_gain,
                    "plant_slope_estimate": (
                        None if not np.isfinite(plant_slope_estimate) else plant_slope_estimate
                    ),
                    "plant_slope_se": (
                        None if not np.isfinite(plant_slope_se) else plant_slope_se
                    ),
                    "plant_slope_rows": plant_slope_rows,
                    "plant_slope_source": plant_slope_source,
                    "formal_updates_applied": formal_updates,
                    "maximum_weight_change": self.maximum_weight_change,
                    "candidate_kind": candidate_kind,
                    "probe_requested_factor": active_probe_requested,
                    "probe_group_effective_factor": active_probe_effective,
                    "probe_sites": [
                        int(value) for value in np.flatnonzero(probe_sites)
                    ],
                    "feedback_mode": self.feedback_mode,
                    "requested_exposure_seconds": self.exposure_seconds,
                    "actual_exposure_seconds": self._actual_exposure_seconds,
                    "effective_photoelectrons": self._effective_photoelectrons,
                    "effective_count_unit": self._effective_count_unit,
                    "target_weight": _json_floats(current_weights),
                    "control_weight": _json_floats(current_control_weights),
                    "excitation_log_step": _json_floats(current_excitation),
                    "dark_mean": _json_floats(fitted["dark_mean"]),
                    "dark_sigma": _json_floats(fitted["dark_sigma"]),
                    "dark_standard_error": _json_floats(
                        fitted["dark_standard_error"]
                    ),
                    "bright_mean": _json_floats(fitted["bright_mean"]),
                    "bright_sigma": _json_floats(fitted["bright_sigma"]),
                    "fit_threshold": _json_floats(fitted["threshold"]),
                    "bright_minus_dark": _json_floats(contrast),
                    "contrast_standard_error": _json_floats(error),
                    "odd_shot_bright_minus_dark": _json_floats(odd_contrast),
                    "even_shot_bright_minus_dark": _json_floats(even_contrast),
                    "bright_fraction": _json_floats(fitted["bright_fraction"]),
                    "fit_fidelity": _json_floats(fitted["fidelity"]),
                    "bic_gain": _json_floats(fitted["bic_gain"]),
                    "fit_valid": [bool(value) for value in fit_valid],
                    "observable_valid": [
                        bool(value) for value in observable_valid
                    ],
                    "loading_edge": [bool(value) for value in loading_edge],
                    "single_population": [bool(value) for value in fit_single],
                    "fit_invalid": [bool(value) for value in fit_invalid],
                    "single_mean": _json_floats(fitted["single_mean"]),
                    "single_sigma": _json_floats(fitted["single_sigma"]),
                    "decision": [str(value) for value in decisions],
                    "requested_log_correction": _json_floats(log_correction),
                    "previous_double_control_weight": _json_floats(
                        previous_weights
                    ),
                    "previous_double_bright_minus_dark": _json_floats(
                        previous_contrast
                    ),
                    "probe_effective_factor": _json_floats(
                        record_probe_effective
                    ),
                    "probe_selected_formal_factor": _json_floats(
                        record_probe_selection
                    ),
                    "probe_decision": [
                        str(value) for value in record_probe_decisions
                    ],
                    "probe_single_bound": _json_floats(probe_single_bound),
                    "probe_observable_bound": _json_floats(
                        probe_observable_bound
                    ),
                    "probe_control_boundary": _json_floats(
                        probe_control_boundary
                    ),
                    "missing_sites": list(missing_sites),
                    "pulse_warning": pulse_warning or None,
                    "single_population_sites": [
                        int(value) for value in np.flatnonzero(fit_single)
                    ],
                    "invalid_sites": [
                        int(value) for value in np.flatnonzero(fit_invalid)
                    ],
                    "minimum_visibility_confidence": visibility_margin,
                    "device_event_record": measured_device_record,
                })
                history[-1]["checkpoint_path"] = (
                    f"candidates/candidate-{candidate_number:04d}.npz"
                )
                history[-1]["measurement_checkpoint_path"] = (
                    f"data/measurements/measurement-{candidate_number:04d}.npz"
                )
                self._publish_candidate(
                    context,
                    phase=applied,
                    candidate=candidate_number,
                    history=history,
                    device_event_record=measured_device_record,
                )
                completed: dict[str, object] = {
                    "candidate": candidate_number,
                    "phase": np.array(applied, copy=True),
                    "pattern_phase": np.array(current_pattern, copy=True),
                    "target": np.array(current_solved_target, copy=True),
                    "solver": candidate_solver,
                    "history": history[-1],
                    # The candidate's rank is the dispersion it can PROVE it
                    # is under -- the split-half variance plus its own
                    # standard error -- never the max/min of one noisy batch:
                    # that number's minimum over 68 candidates was a sampling
                    # extreme, and picking it was the winner's curse.  Nor
                    # the variance clipped at zero: a single low draw of a
                    # noisy estimate then tied at "zero" with the genuinely
                    # uniform candidates (archived candidate 41, whose
                    # neighbours both resolved 1.7% CV, tied with 62 and 65).
                    # Ties go to the most recent candidate.
                    "score": (
                        true_variance + true_variance_error
                        if np.isfinite(true_variance)
                        and np.isfinite(true_variance_error)
                        else float("inf")
                    ),
                    "contrast": np.array(contrast, copy=True),
                    "standard_error": np.array(error, copy=True),
                    "samples": np.array(samples, copy=True),
                    "mean_frame": np.array(mean_frame, copy=True),
                }
                candidate_reports.append((history[-1], np.asarray(samples)))
                last_completed_candidate = completed
                completed["visibility_rank"] = (
                    visibility,
                    -int(np.count_nonzero(fit_invalid)),
                    float("-inf") if visibility_margin is None else visibility_margin,
                )
                if candidate_kind != "probe" and (
                    most_visible_observed is None
                    or completed["visibility_rank"]
                    >= most_visible_observed["visibility_rank"]
                ):
                    most_visible_observed = completed
                if valid and candidate_kind != "probe":
                    if (
                        retained_valid is None
                        or float(completed["score"]) <= float(retained_valid["score"])
                    ):
                        retained_valid = completed
                    context.report_progress(
                        f"qCMOS bright-dark ratio {score:.5f} "
                        f"(uniform-array floor {expected_noise_ratio:.5f}); "
                        f"true site CV {true_uniformity_cv:.4f}; "
                        f"plant slope {plant_slope_source}"
                    )
                elif candidate_kind == "probe":
                    context.report_progress(
                        "SLM probe "
                        f"{active_probe_requested:g}x requested / "
                        f"{active_probe_effective:g}x effective measured; "
                        f"{visibility}/{self._site_count} sites observable"
                    )
                else:
                    context.report_progress(
                        f"Candidate {candidate_number}: {visibility}/"
                        f"{self._site_count} site fits valid; applying only "
                        "observable-site updates"
                    )
                if starts_probe_episode:
                    probe_baseline_candidate = completed
                elif candidate_kind == "probe":
                    assert active_probe_requested is not None
                    assert active_probe_effective is not None
                    probe_measurements.append((
                        active_probe_effective,
                        np.array(contrast, copy=True),
                        np.array(error, copy=True),
                        np.array(observable_valid, copy=True),
                    ))
                if candidate_kind != "probe":
                    previous_weights[fit_valid] = current_control_weights[
                        fit_valid
                    ]
                    previous_contrast[fit_valid] = contrast[fit_valid]
                continue_feedback = True
                next_target: np.ndarray | None = None
                next_kind = "ordinary"
                next_probe_requested: float | None = None
                next_probe_effective: float | None = None
                next_probe_site_factors: np.ndarray | None = None
                if starts_probe_episode:
                    if pending_probes:
                        (
                            next_target,
                            next_probe_requested,
                            next_probe_effective,
                            next_probe_site_factors,
                        ) = take_probe()
                        next_kind = "probe"
                    else:
                        stalled = True
                        continue_feedback = False
                        history[-1]["next_phase_changed"] = False
                        termination_reason = (
                            "relative SLM probe has no non-probe power reservoir"
                        )
                        forced_terminal_candidate = probe_baseline_candidate
                elif candidate_kind == "probe":
                    if pending_probes:
                        (
                            next_target,
                            next_probe_requested,
                            next_probe_effective,
                            next_probe_site_factors,
                        ) = take_probe()
                        next_kind = "probe"
                    else:
                        assert probe_baseline_target is not None
                        baseline_control = _control_weights(
                            probe_baseline_target[self._rows, self._columns]
                        )
                        (
                            verdict_step,
                            verdict_direction,
                            verdict_single,
                            verdict_observable,
                            verdict_decisions,
                        ) = _probe_verdict(
                            probe_sites,
                            baseline_control,
                            probe_baseline_contrast,
                            probe_baseline_valid,
                            probe_measurements,
                            self.maximum_weight_change,
                        )
                        probe_decisions[probe_sites] = verdict_decisions[probe_sites]
                        bracket_direction[probe_sites] = verdict_direction[probe_sites]
                        probe_single_bound[probe_sites] = verdict_single[probe_sites]
                        probe_observable_bound[probe_sites] = verdict_observable[
                            probe_sites
                        ]
                        probe_control_boundary[:] = _probe_boundary(
                            probe_single_bound,
                            probe_observable_bound,
                            bracket_direction,
                        )
                        # The combined Target is the baseline's own formal
                        # update plus every dark site's direction -- the
                        # probe verdict for the probed sites, the bracket
                        # step (now holding what the probes showed) for the
                        # rest -- funded by the loaded sites within a
                        # resolution step and their bracket floors (see
                        # ``_updated_target``).
                        combined_directed_step, _combined_kind = _single_bracket_step(
                            baseline_control,
                            probe_baseline_dark & ~probe_sites,
                            probe_single_bound,
                            probe_observable_bound,
                            bracket_direction,
                            self.maximum_weight_change,
                        )
                        combined_directed_step[probe_sites] = verdict_step[
                            probe_sites
                        ]
                        combined_target, *_formal_details = _updated_target(
                            probe_baseline_target,
                            probe_baseline_contrast,
                            probe_baseline_error,
                            probe_baseline_valid,
                            self._rows,
                            self._columns,
                            reference_valid=probe_baseline_reference_valid,
                            feedback_gain=self.feedback_gain,
                            plant_slope=plant_slope_magnitude,
                            maximum_weight_change=self.maximum_weight_change,
                            directed_log_step=combined_directed_step,
                            control_boundary=probe_control_boundary,
                            control_direction=bracket_direction,
                            loading_edge=probe_baseline_edge,
                        )
                        combined_control = _control_weights(
                            combined_target[self._rows, self._columns]
                        )
                        probe_selected_factors[probe_sites] = (
                            combined_control[probe_sites]
                            / baseline_control[probe_sites]
                        )
                        if np.array_equal(
                            combined_target, probe_baseline_target
                        ):
                            stalled = True
                            continue_feedback = False
                            history[-1]["next_phase_changed"] = False
                            termination_reason = (
                                "the loaded sites could fund no share for the "
                                "probed sites' direction; baseline restored"
                            )
                            forced_terminal_candidate = probe_baseline_candidate
                        else:
                            next_target = combined_target
                            next_kind = "probe_combined"
                            formal_updates += 1
                else:
                    if convergence_streak >= _CONVERGENCE_CANDIDATES:
                        continue_feedback = False
                        history[-1]["next_phase_changed"] = None
                        termination_reason = (
                            "true between-site contrast dispersion indistinguishable "
                            f"from zero for {_CONVERGENCE_CANDIDATES} consecutive "
                            "candidates"
                        )
                    elif formal_updates >= self.max_updates:
                        continue_feedback = False
                        history[-1]["next_phase_changed"] = None
                    elif np.array_equal(proposed_target, current_target):
                        stalled = True
                        continue_feedback = False
                        history[-1]["next_phase_changed"] = False
                        termination_reason = (
                            "all sites held because this shot batch supplied no "
                            "actionable update"
                        )
                    else:
                        next_target = proposed_target
                        next_kind = "ordinary"
                        formal_updates += 1
                if continue_feedback:
                    assert next_target is not None
                    # The identification excitation rides on the first
                    # ordinary updates only; a probe's Target is the probe's
                    # own statement and carries none, and a site on its
                    # loading ramp is not wiggled 2% towards dark.
                    next_excitation = (
                        _excitation_pattern(
                            excitation_rng, feedback_valid & ~loading_edge
                        )
                        if next_kind == "ordinary"
                        and formal_updates <= _PLANT_EXCITATION_CANDIDATES
                        else np.zeros(self._site_count, dtype=float)
                    )
                    next_solved_target = (
                        _excited_target(
                            next_target,
                            next_excitation,
                            self._rows,
                            self._columns,
                        )
                        if np.any(next_excitation != 0.0)
                        else next_target
                    )
                    if next_kind == "probe":
                        context.report_progress(
                            "Solving SLM probe "
                            f"{next_probe_requested:g}x requested / "
                            f"{next_probe_effective:g}x effective for "
                            f"{int(np.count_nonzero(probe_sites))} sites"
                        )
                    try:
                        probe_solve = next_kind in {"probe", "probe_combined"}
                        solve_state = (
                            deepcopy(probe_baseline_optimizer_state or {})
                            if probe_solve else spot_optimizer_state
                        )
                        next_pattern, solver_metadata = solve_phase(
                            next_solved_target,
                            pupil_amplitude=self._pupil_amplitude,
                            initial_phase=(
                                probe_baseline_pattern
                                if probe_solve else current_pattern
                            ),
                            objective_kind="spots",
                            iterations=None,
                            stop_requested=context.cancel_requested,
                            spot_optimizer_state=solve_state,
                            support_tolerance=_FEEDBACK_SOLVE_SUPPORT_TOLERANCE,
                            minimum_iterations=_FEEDBACK_SOLVE_MINIMUM_ITERATIONS,
                        )
                        next_pattern = freeze_pattern_phase(
                            next_pattern, self.slm.shape_yx
                        )
                    except BaseException:
                        history[-1]["next_phase_changed"] = None
                        self._save_candidate_checkpoint(
                            context,
                            paths,
                            candidate=candidate_number,
                            samples=samples,
                            measurement=history[-1],
                            solver=candidate_solver,
                            phase=np.asarray(completed["phase"]),
                            pattern=np.asarray(completed["pattern_phase"]),
                            target=np.asarray(completed["target"]),
                            history=history,
                        )
                        raise
                    next_phase = compose_science_phase(
                        next_pattern, self._operator_wavefront
                    )
                    if np.array_equal(next_phase, applied):
                        stalled = True
                        continue_feedback = False
                        history[-1]["next_phase_changed"] = False
                        termination_reason = (
                            "the requested SLM probe/update produced no different "
                            "phase; no unchanged-phase shot batch was taken"
                        )
                        if next_kind == "probe":
                            forced_terminal_candidate = probe_baseline_candidate
                    else:
                        if next_kind == "probe_combined":
                            spot_optimizer_state.clear()
                            spot_optimizer_state.update(solve_state)
                        history[-1]["next_phase_changed"] = True
                        current_target = next_target
                        current_solved_target = next_solved_target
                        current_excitation = next_excitation
                        current_pattern = next_pattern
                        current_phase = next_phase
                        candidate_kind = next_kind
                        active_probe_requested = next_probe_requested
                        active_probe_effective = next_probe_effective
                        active_probe_site_factors = next_probe_site_factors
                elif "next_phase_changed" not in history[-1]:
                    history[-1]["next_phase_changed"] = None
                self._save_candidate_checkpoint(
                    context,
                    paths,
                    candidate=candidate_number,
                    samples=samples,
                    measurement=history[-1],
                    solver=candidate_solver,
                    phase=np.asarray(completed["phase"]),
                    pattern=np.asarray(completed["pattern_phase"]),
                    target=np.asarray(completed["target"]),
                    history=history,
                )
                if not continue_feedback:
                    break
            selected = (
                forced_terminal_candidate
                or retained_valid
                or most_visible_observed
            )
            if selected is None:
                raise RuntimeError("qCMOS feedback produced no completed candidate")
            status = "stalled" if stalled else "completed"
            selected["outcome"] = {
                "status": status,
                "reason": termination_reason,
                "selected_candidate": int(selected["candidate"]),
                "shots_per_candidate": self.shots,
                "candidates_measured": len(history),
                "formal_updates_completed": formal_updates,
                "probe_candidates_measured": sum(
                    item["candidate_kind"] == "probe" for item in history
                ),
                "probe_sites": [
                    int(value) for value in np.flatnonzero(probe_sites)
                ],
            }
            context.seal_terminal()
            return self._finish_candidate(
                context,
                selected,
                history,
                paths=paths,
                initial_phase=incoming,
                initial_mean_frame=initial_mean_frame,
                candidate_reports=candidate_reports,
                status=status,
                republish=True,
            )
        except BaseException as error:
            if context.cancel_requested():
                try:
                    context.seal_terminal(accept_stop=True)
                    retained = retained_valid or most_visible_observed
                    if retained is None:
                        retained = self._incoming_candidate(
                            observed=most_visible_observed,
                            phase=incoming,
                            pattern=incoming_pattern,
                        )
                    retained["outcome"] = {
                        "status": "stopped",
                        "reason": "operator Stop",
                        "selected_candidate": (
                            int(retained["candidate"])
                            if isinstance(retained.get("history"), Mapping)
                            else None
                        ),
                        "shots_per_candidate": self.shots,
                        "candidates_measured": len(history),
                    }
                    return self._finish_candidate(
                        context,
                        retained,
                        history,
                        paths=paths,
                        initial_phase=incoming,
                        initial_mean_frame=initial_mean_frame,
                        candidate_reports=candidate_reports,
                        status="stopped",
                        republish=True,
                    )
                except BaseException as stop_error:
                    error = stop_error
            failure_selected = (
                retained_valid
                or most_visible_observed
                or last_completed_candidate
            )
            if failure_selected is not None:
                # A FAILED RUN KEEPS WHAT IT MEASURED.  The candidate the
                # run would have selected is sealed exactly as a completed
                # run seals it -- on the SLM, in final/, in the summary --
                # and the failure travels with it in the outcome and the
                # summary.  Only when even that cannot be done is the
                # incoming phase the one known-good state left.
                failure_selected["outcome"] = {
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                    "selected_candidate": int(failure_selected["candidate"]),
                    "shots_per_candidate": self.shots,
                    "candidates_measured": len(history),
                }
                try:
                    self._finish_candidate(
                        context,
                        failure_selected,
                        history,
                        paths=paths,
                        initial_phase=incoming,
                        initial_mean_frame=initial_mean_frame,
                        candidate_reports=candidate_reports,
                        status="failed",
                        republish=True,
                        error=error,
                    )
                except BaseException as seal_error:
                    error.add_note(
                        "Feedback could not seal the selected candidate after "
                        f"failure: {type(seal_error).__name__}: {seal_error}"
                    )
                else:
                    raise
            try:
                self._apply_exact(incoming)
            except BaseException as restore_error:
                try:
                    self._write_summary(
                        context,
                        paths,
                        status="failed",
                        history=history,
                        selected_candidate=(
                            None
                            if failure_selected is None
                            else int(failure_selected["candidate"])
                        ),
                        error=error,
                        rollback={
                            "status": "failed",
                            "error": (
                                f"{type(restore_error).__name__}: {restore_error}"
                            ),
                        },
                    )
                except BaseException as summary_error:
                    error.add_note(
                        "Feedback failure summary could not be saved: "
                        f"{type(summary_error).__name__}: {summary_error}"
                    )
                raise RuntimeError(
                    "SLM feedback failed and the incoming phase could not be "
                    f"restored: {error}"
                ) from restore_error
            try:
                self._write_summary(
                    context,
                    paths,
                    status="failed",
                    history=history,
                    selected_candidate=(
                        None
                        if failure_selected is None
                        else int(failure_selected["candidate"])
                    ),
                    error=error,
                    rollback={
                        "status": "restored",
                        "slm_command_receipt": dict(self.slm.last_command_receipt),
                    },
                )
            except BaseException as summary_error:
                error.add_note(
                    "Feedback failure summary could not be saved: "
                    f"{type(summary_error).__name__}: {summary_error}"
                )
            raise

__all__ = [
    "CANDIDATE_PHASE_OUTPUT",
    "OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT",
    "READOUT_FRAME_COORDINATE",
    "SITE_SIGNAL_HISTORY_OUTPUT",
    "SLM_PHASE_ARTIFACT_CONTRACT",
    "SlmFeedbackTask",
    "TARGET_SHARE_HISTORY_OUTPUT",
    "UNIFORMITY_HISTORY_OUTPUT",
]
