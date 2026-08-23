"""Single-frame, multi-shot bright-dark fluorescence feedback."""

from __future__ import annotations

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
    PointColumn,
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
    ImagePlot,
    PlotLabels,
    save_figure_artifact,
)
from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput, MonitorCoverage

from zlc_atom.data import snapshot_from_array
from zlc_atom.devices.slm import SlmAdapter, canonical_phase
from zlc_atom.devices.slm.solver import (
    SCIENCE_CONTEXT_ARTIFACT_CONTRACT,
    save_science_context,
    solve_phase,
    validate_target,
)
from zlc_atom.nodes.calibration import (
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
    extract_box_signals,
    fit_bimodal,
)
from zlc_atom.nodes.calibration.calibration import (
    _register_target_sites,
    validate_target_registration,
)
from zlc_atom.nodes.calibration.pulse import arm_sequencer, resolve_pulse
from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_atom.nodes.scan.source import wait_for_board


SLM_PHASE_ARTIFACT_CONTRACT = SCIENCE_CONTEXT_ARTIFACT_CONTRACT
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
_CONTROLLER_CONTRACT = "slm-feedback.qcmos-bright-dark"
READOUT_FRAME_COORDINATE = 0


def _check_cancelled(context: object) -> None:
    if context.cancel_requested():
        raise RuntimeError("SLM feedback was cancelled")


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


def _fit_contrasts(samples: object) -> dict[str, np.ndarray]:
    """Classify each site's one user-authored shot batch.

    A resolved two-population fit supplies the bright-minus-dark feedback
    observable.  Evidence for one Gaussian is a different, useful physical
    result: this feedback mode treats it as a site which did not load.  Bad
    samples or a numerically undecidable model remain invalid and therefore
    cannot create a control action.
    """

    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 1:
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
            single_population[site] = True
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
        sigma_one = max(one_sigma, np.finfo(float).tiny)
        log_one = float(
            np.sum(
                -0.5 * np.square((column - np.mean(column)) / sigma_one)
                - np.log(sigma_one * np.sqrt(2.0 * np.pi))
            )
        )
        dark_density = (
            np.exp(-0.5 * np.square((column - fit.dark_mean) / fit.dark_sigma))
            / (fit.dark_sigma * np.sqrt(2.0 * np.pi))
        )
        bright_density = (
            np.exp(-0.5 * np.square((column - fit.bright_mean) / fit.bright_sigma))
            / (fit.bright_sigma * np.sqrt(2.0 * np.pi))
        )
        mixture = (1.0 - fraction) * dark_density + fraction * bright_density
        if np.all(np.isfinite(mixture)) and np.all(mixture > 0.0):
            log_two = float(np.sum(np.log(mixture)))
            bic_gain[site] = (
                2.0 * log_two
                - 5.0 * np.log(column.size)
                - (2.0 * log_one - 2.0 * np.log(column.size))
            )
        finite_pair = bool(
            estimate > 0.0
            and np.isfinite(sem)
            and sem >= 0.0
            and np.isfinite(bic_gain[site])
        )
        if finite_pair and bic_gain[site] > 0.0:
            contrast[site], error[site] = estimate, sem
            separated[site] = True
        else:
            single_population[site] = True
    return {
        "contrast": contrast,
        "standard_error": error,
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


def _single_population_observables(
    fitted: Mapping[str, np.ndarray],
    *,
    shots: int,
    previous_dark_mean: np.ndarray,
    previous_dark_standard_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use the last resolved mixture to disambiguate a present single peak.

    With no per-site history the experiment's authored prior is that one peak
    is the dark/no-loading population.  Once a trustworthy mixture exists,
    the current batch's resolved dark components (or that site's historical
    dark component) decide whether the new single population is dark-only or
    bright-only.  A bright-only population still supplies a contrast by
    subtracting that dark reference.
    """

    count = int(shots)
    if count < 4:
        raise ValueError("single-population classification needs at least four shots")
    single = np.asarray(fitted["single_population"], dtype=bool)
    mean = np.asarray(fitted["single_mean"], dtype=float)
    sigma = np.asarray(fitted["single_sigma"], dtype=float)
    prior_dark = np.asarray(previous_dark_mean, dtype=float)
    prior_dark_error = np.asarray(previous_dark_standard_error, dtype=float)
    shape = single.shape
    if any(
        value.shape != shape
        for value in (
            mean,
            sigma,
            prior_dark,
            prior_dark_error,
        )
    ):
        raise ValueError("single-population history shapes differ")

    contrast = np.asarray(fitted["contrast"], dtype=float).copy()
    error = np.asarray(fitted["standard_error"], dtype=float).copy()
    dark_only = np.zeros(shape, dtype=bool)
    bright_only = np.zeros(shape, dtype=bool)
    single_error = sigma / np.sqrt(float(count))
    direct_valid = np.asarray(fitted["valid"], dtype=bool)
    direct_dark = np.asarray(fitted["dark_mean"], dtype=float)
    direct_dark_error = np.asarray(
        fitted["dark_standard_error"], dtype=float
    )
    fallback = direct_valid & np.isfinite(direct_dark) & np.isfinite(direct_dark_error)
    fallback_dark = (
        float(np.median(direct_dark[fallback])) if np.any(fallback) else float("nan")
    )
    fallback_dark_error = (
        float(np.median(direct_dark_error[fallback]))
        if np.any(fallback)
        else float("nan")
    )
    for site in np.flatnonzero(single):
        has_history = bool(
            np.isfinite(prior_dark[site])
            and np.isfinite(prior_dark_error[site])
            and prior_dark_error[site] >= 0.0
        )
        reference = (
            float(prior_dark[site]) if has_history else fallback_dark
        )
        reference_error = (
            float(prior_dark_error[site])
            if has_history
            else fallback_dark_error
        )
        if not np.isfinite(reference) or not np.isfinite(reference_error):
            # With no bright/dark reference at all, use the experiment's
            # authored prior: a one-population site starts as no-loading.
            dark_only[site] = True
            continue
        estimate = float(mean[site] - reference)
        estimate_error = float(np.hypot(single_error[site], reference_error))
        if estimate > 3.0 * estimate_error:
            bright_only[site] = True
            contrast[site] = estimate
            error[site] = estimate_error
        else:
            dark_only[site] = True
    return contrast, error, dark_only, bright_only


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
    "fit_bic_gain",
    "fit_valid",
    "observable_valid",
    "single_population",
    "single_dark_only",
    "single_bright_only",
    "fit_invalid",
    "single_mean",
    "single_sigma",
    "decision",
    "requested_log_correction",
    "response_log_slope",
    "previous_valid_control_weight",
    "previous_valid_bright_minus_dark",
    "previous_valid_dark_mean",
    "previous_valid_dark_standard_error",
    "loading_dark_bound",
    "loading_loaded_bound",
    "loading_floor",
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


def _support(
    target: np.ndarray,
    calibration: TrapCalibration,
    *,
    science_context_path: str | Path,
    command_receipt: Mapping[str, object],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Register only the Calibration's site boxes to this Feedback Target."""

    model = calibration.select_model(ReadoutModelKind.BOX)
    usable = np.asarray(calibration.site_map.valid_sites, dtype=bool)
    if not np.any(usable):
        raise ValueError("SLM Feedback requires at least one calibrated BOX site")
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
        measurement_radius=model.integration_half_width,
    )
    support, provenance = validate_target_registration(
        registered,
        frame_shape=calibration.frame_contract.image_shape,
        box_half_width=model.integration_half_width,
    )
    rows, columns = support.T
    if not np.array_equal(support, np.column_stack(np.nonzero(target > 0.0))):
        raise ValueError("registered Calibration support differs from Science Context")
    if provenance["command_receipt"] != dict(command_receipt):
        raise RuntimeError("Feedback registration lost its Science Context receipt")

    centers = np.asarray(registered.centers_xy, dtype=float)
    return rows, columns, centers


def _updated_target(
    target: np.ndarray,
    contrast: np.ndarray,
    standard_error: np.ndarray,
    valid: np.ndarray,
    dark_only: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    previous_weights: np.ndarray,
    previous_contrast: np.ndarray,
    feedback_gain: float,
    single_gaussian_boost: float,
    maximum_weight_change: float,
    minimum_control_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update loaded sites relatively and dark-only sites absolutely."""

    values = np.asarray(contrast, dtype=float)
    errors = np.asarray(standard_error, dtype=float)
    fit_valid = np.asarray(valid, dtype=bool)
    fit_dark_only = np.asarray(dark_only, dtype=bool)
    prior_weights = np.asarray(previous_weights, dtype=float)
    prior_values = np.asarray(previous_contrast, dtype=float)
    gain = float(feedback_gain)
    dark_boost = float(single_gaussian_boost)
    maximum_change = float(maximum_weight_change)
    minimum_control = np.asarray(minimum_control_weight, dtype=float)
    site_shape = (len(rows),)
    if (
        values.shape != site_shape
        or errors.shape != site_shape
        or fit_valid.shape != site_shape
        or fit_dark_only.shape != site_shape
        or prior_weights.shape != site_shape
        or prior_values.shape != site_shape
        or minimum_control.shape != site_shape
        or np.any(np.isfinite(minimum_control) & (minimum_control <= 0.0))
        or np.any(fit_valid & fit_dark_only)
        or not np.isfinite(gain)
        or gain < 0.0
        or not np.isfinite(dark_boost)
        or dark_boost < 0.0
        or not np.isfinite(maximum_change)
        or maximum_change < 0.0
    ):
        raise ValueError("feedback site history shapes differ")
    if not np.any(fit_valid):
        reference = float("nan")
    else:
        if np.any(~np.isfinite(values[fit_valid]) | (values[fit_valid] <= 0.0)):
            raise ValueError("valid feedback contrasts must be finite and positive")
        reference = float(np.exp(np.mean(np.log(values[fit_valid]))))
    raw_weights = np.asarray(target[rows, columns], dtype=float)
    current_weights = _control_weights(raw_weights)
    log_correction = np.zeros(site_shape, dtype=float)
    response_slope = np.full(site_shape, np.nan, dtype=float)
    decision = np.full(site_shape, "hold_invalid", dtype="<U32")
    for site in range(len(rows)):
        if fit_valid[site]:
            residual = float(np.log(values[site] / reference))
            relative_error = max(float(errors[site]), 0.0) / values[site]
            quality = float(np.clip(1.0 - 4.0 * relative_error, 0.1, 1.0))
            correction = quality * residual
            if (
                np.isfinite(prior_weights[site])
                and prior_weights[site] > 0.0
                and np.isfinite(prior_values[site])
                and prior_values[site] > 0.0
            ):
                moved = float(np.log(current_weights[site] / prior_weights[site]))
                if abs(moved) >= 0.02:
                    slope = float(np.log(values[site] / prior_values[site]) / moved)
                    if -4.0 <= slope <= -0.20:
                        response_slope[site] = slope
                        correction = -quality * residual / slope
                        decision[site] = "feedback_history_slope"
                    else:
                        decision[site] = "feedback_assumed_slope"
                else:
                    decision[site] = "feedback_assumed_slope"
            else:
                decision[site] = "feedback_assumed_slope"
            log_correction[site] = float(
                np.clip(
                    gain * correction,
                    -np.log1p(maximum_change),
                    np.log1p(maximum_change),
                )
            )
        elif fit_dark_only[site]:
            log_correction[site] = float(np.log1p(dark_boost))
            decision[site] = "boost_dark_single_gaussian"

    # The solver consumes relative intensity shares.  Applying independent raw
    # multipliers and normalizing afterwards can make a no-loading site's
    # actual share DECREASE when the loaded sites collectively grow more.  Set
    # the constrained shares directly: dark sites receive the exact authored
    # increase, invalid sites keep their physical share, and loaded sites own
    # only the relative distribution of the remaining power.
    shares = raw_weights / float(np.sum(raw_weights))
    invalid = ~(fit_valid | fit_dark_only)
    next_shares = np.zeros_like(shares)
    next_shares[invalid] = shares[invalid]
    invalid_total = float(np.sum(next_shares[invalid]))
    loaded = np.flatnonzero(fit_valid)
    floors = np.where(
        np.isfinite(minimum_control),
        minimum_control / float(len(rows)),
        0.0,
    )
    loaded_floors = floors[loaded]
    dark = np.flatnonzero(fit_dark_only)
    dark_base = np.maximum(shares[dark], floors[dark])
    dark_requested = np.maximum(
        shares[dark] * (1.0 + dark_boost), floors[dark]
    )
    dark_budget = max(
        0.0,
        1.0 - invalid_total - float(np.sum(loaded_floors)),
    )
    base_total = float(np.sum(dark_base))
    requested_extra = dark_requested - dark_base
    if not len(loaded):
        next_shares = np.array(shares, copy=True)
    elif base_total >= dark_budget:
        available = max(
            0.0, dark_budget - float(np.sum(shares[dark]))
        )
        needed = float(np.sum(np.maximum(dark_base - shares[dark], 0.0)))
        fraction = 0.0 if needed <= 0.0 else min(1.0, available / needed)
        next_shares[dark] = shares[dark] + fraction * (
            dark_base - shares[dark]
        )
    else:
        available = dark_budget - base_total
        needed = float(np.sum(requested_extra))
        fraction = 1.0 if needed <= 0.0 else min(1.0, available / needed)
        next_shares[dark] = dark_base + fraction * requested_extra
    remaining = 1.0 - float(
        np.sum(next_shares[fit_dark_only]) + np.sum(next_shares[invalid])
    )
    loaded_requested = shares[loaded] * np.exp(log_correction[loaded])
    loaded_total = float(np.sum(loaded_requested))
    if len(loaded) and (not np.isfinite(loaded_total) or loaded_total <= 0.0):
        raise RuntimeError("loaded-site feedback produced no positive power share")
    if len(loaded):
        free = np.ones(len(loaded), dtype=bool)
        allocated = np.zeros(len(loaded), dtype=float)
        while np.any(free):
            available = remaining - float(np.sum(allocated[~free]))
            requested_total = float(np.sum(loaded_requested[free]))
            scale = available / requested_total
            candidate = scale * loaded_requested[free]
            below_floor = candidate < loaded_floors[free]
            if not np.any(below_floor):
                allocated[free] = candidate
                break
            free_indices = np.flatnonzero(free)
            fixed_indices = free_indices[below_floor]
            allocated[fixed_indices] = loaded_floors[fixed_indices]
            free[fixed_indices] = False
        next_shares[loaded] = allocated
    log_correction = np.log(next_shares / shares)
    updated = np.array(target, dtype=np.float32, copy=True)
    updated[rows, columns] = (
        next_shares * float(np.sum(raw_weights))
    ).astype(np.float32)
    return validate_target(updated), log_correction, response_slope, decision


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
        single_gaussian_boost: float,
        feedback_gain: float,
        maximum_weight_change: float,
        max_updates: int,
    ) -> None:
        if not isinstance(slm, SlmAdapter):
            raise TypeError("slm must implement SlmAdapter")
        if not isinstance(calibration, TrapCalibration) or not isinstance(pulse_sequence, PulseSequence):
            raise TypeError("feedback requires TrapCalibration and PulseSequence")
        if not isinstance(science_context, Mapping):
            raise TypeError("science_context must be a loaded Science Context mapping")
        if science_context.get("objective_kind") != "spots":
            raise ValueError("SLM feedback Science Context must use the spots objective")
        context_target = science_context.get("target_intensity")
        if context_target is None:
            raise ValueError("Science Context has no frozen Target")
        frozen_target = validate_target(context_target)
        if frozen_target.shape != slm.shape_yx:
            raise ValueError("Science Context Target shape differs from the selected SLM")
        incoming = canonical_phase(science_context.get("phase"), slm.shape_yx)
        pattern = canonical_phase(
            science_context.get("pattern_phase"), slm.shape_yx
        )
        operator = canonical_phase(
            science_context.get("operator_wavefront"), slm.shape_yx
        )
        pupil = np.asarray(science_context.get("pupil_amplitude"), dtype=np.float32)
        support = np.asarray(science_context.get("pupil_support"))
        if (
            pupil.shape != slm.shape_yx
            or not np.all(np.isfinite(pupil))
            or np.any(pupil < 0.0)
            or not np.any(pupil > 0.0)
            or support.shape != slm.shape_yx
            or support.dtype != np.dtype(bool)
        ):
            raise ValueError("Science Context has invalid pupil arrays")
        receipt = science_context.get("command_receipt")
        if not isinstance(receipt, Mapping):
            raise TypeError("Science Context command receipt must be a mapping")
        model = calibration.select_model(ReadoutModelKind.BOX)
        context_path = Path(science_context_path).expanduser().resolve()
        (
            self._rows,
            self._columns,
            self._site_centers_xy,
        ) = _support(
            frozen_target,
            calibration,
            science_context_path=context_path,
            command_receipt=receipt,
        )
        self._site_count = len(self._rows)
        self._site_centers_xy.setflags(write=False)
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
        self._pupil_support = np.array(support, copy=True)
        self._pupil_support.setflags(write=False)
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
        self.single_gaussian_boost = float(single_gaussian_boost)
        self.feedback_gain = float(feedback_gain)
        self.maximum_weight_change = float(maximum_weight_change)
        self.max_updates = int(max_updates)
        self._candidate_capacity = self.max_updates + 1
        self._publication_revision = 0
        self._actual_exposure_seconds: float | None = None
        self._effective_photoelectrons: bool | None = None
        self._effective_count_unit: str | None = None
        self._last_measured_phase: np.ndarray | None = None
        if (
            self.shots < 10
            or self.max_updates < 1
            or not np.isfinite(self.single_gaussian_boost)
            or self.single_gaussian_boost < 0.0
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
        )

    def _run_record(self) -> dict[str, object]:
        return {
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
            "single_gaussian_boost": self.single_gaussian_boost,
            "feedback_gain": self.feedback_gain,
            "maximum_weight_change": self.maximum_weight_change,
        }

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
            "single_gaussian_boost": self.single_gaussian_boost,
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

    def _science_phase(self, pattern: object) -> np.ndarray:
        return canonical_phase(
            canonical_phase(pattern, self.slm.shape_yx).astype(float)
            + self._operator_wavefront.astype(float),
            self.slm.shape_yx,
        )

    def _save_candidate(
        self,
        path: str | Path,
        phase: object,
        pattern: object,
        target: object,
        metadata: Mapping[str, object],
    ) -> Path:
        receipt = dict(self.slm.last_command_receipt)
        if receipt.get("outcome") not in {"known-old", "known-new"}:
            raise RuntimeError("SLM candidate command outcome is unknown")
        return save_science_context(
            path,
            phase,
            pattern_phase=pattern,
            operator_wavefront=self._operator_wavefront,
            pupil_amplitude=self._pupil_amplitude,
            pupil_support=self._pupil_support,
            target_intensity=target,
            objective_kind="spots",
            pupil=self._pupil,
            system_correction=self._system_correction,
            command_receipt=receipt,
            pattern_metadata={**self._pattern_metadata, **dict(metadata)},
            operator_metadata=self._operator_metadata,
        )

    def _publish_candidate(
        self,
        context: object,
        *,
        phase: np.ndarray,
        candidate: int,
        history: list[dict[str, object]],
    ) -> None:
        candidate = int(candidate)
        if not 1 <= candidate <= self._candidate_capacity:
            raise ValueError("feedback candidate lies outside its authored history")
        self._publication_revision += 1
        publication_revision = self._publication_revision
        generation = str(getattr(context.generation, "value", context.generation))
        record = self._run_record()
        canonical = canonical_phase(phase, self.slm.shape_yx)
        phase_event = snapshot_from_array(
            canonical[None],
            producer=self.instance_id,
            signal=CANDIDATE_PHASE_OUTPUT.name,
            roles=(SPATIAL_Y, SPATIAL_X),
            value_unit="rad",
            generation=generation,
            revision=publication_revision,
        )
        coordinate_id = AxisId("slm_feedback.candidate")
        curve = np.full(self._candidate_capacity, np.nan, dtype="<f8")
        observable_curve = np.full_like(curve, np.nan)
        for item in history:
            index = int(item["iteration"]) - 1
            full_ratio = item.get("uniformity_ratio")
            observable_ratio = item.get("observable_uniformity_ratio")
            if full_ratio is not None:
                curve[index] = float(full_ratio)
            if observable_ratio is not None:
                observable_curve[index] = float(observable_ratio)
        point_columns = {
            SCAN_POINT: PointColumn(
                coordinate_id,
                "candidate",
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(
                    float(index)
                    for index in range(1, self._candidate_capacity + 1)
                ),
            )
        }
        history_event = snapshot_from_array(
            curve[None],
            producer=self.instance_id,
            signal=UNIFORMITY_HISTORY_OUTPUT.name,
            roles=(SCAN_POINT,),
            point_columns=point_columns,
            generation=generation,
            revision=publication_revision,
            validity=np.isfinite(curve)[None],
        )
        observable_history_event = snapshot_from_array(
            observable_curve[None],
            producer=self.instance_id,
            signal=OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT.name,
            roles=(SCAN_POINT,),
            point_columns=point_columns,
            generation=generation,
            revision=publication_revision,
            validity=np.isfinite(observable_curve)[None],
        )
        context.commit_live(
            {
                CANDIDATE_PHASE_OUTPUT.name: LiveDatasetOutput(
                    CANDIDATE_PHASE_OUTPUT,
                    phase_event,
                    MonitorCoverage(1, 1, retain_at_terminal=True),
                    record,
                ),
                UNIFORMITY_HISTORY_OUTPUT.name: LiveDatasetOutput(
                    UNIFORMITY_HISTORY_OUTPUT,
                    history_event,
                    MonitorCoverage(
                        self._candidate_capacity,
                        self._candidate_capacity,
                        retain_at_terminal=True,
                    ),
                    record,
                ),
                OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT.name: LiveDatasetOutput(
                    OBSERVABLE_UNIFORMITY_HISTORY_OUTPUT,
                    observable_history_event,
                    MonitorCoverage(
                        self._candidate_capacity,
                        self._candidate_capacity,
                        retain_at_terminal=True,
                    ),
                    record,
                ),
            }
        )

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
    ]:
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
            arm_sequencer(self.sequencer, pulse)
            capture = node.prepare(should_stop=context.cancel_requested)
            actual = node.actual_working_point
            if actual is None:
                raise RuntimeError("camera did not freeze its actual working point")
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
            self.sequencer.fire(cycles=requested)
            result = capture.collect(retain_cycles=False)
            _check_cancelled(context)
            wait_for_board(self.sequencer, context)
            if result is None or result.cycle_count != requested:
                raise RuntimeError("canonical Camera Measurement ended before all shots")
            context.report_progress(
                f"Reading mean qCMOS brightness for candidate {iteration + 1}",
                current=requested,
                total=requested,
            )
        finally:
            try:
                self.sequencer.safe()
            finally:
                if capture is not None and not capture.closed:
                    capture.close()

        frames = _readout_frames(result.snapshot, shots=requested)

        saturated_sites: set[int] = set()
        for image in frames:
            saturated_sites.update(
                self._saturated_sites(image, saturation_value)
            )
        box_samples = np.asarray(
            [
                extract_box_signals(
                    image,
                    self._site_centers_xy,
                    radius=self.model.integration_half_width,
                    reducer=self.model.reducer,  # type: ignore[arg-type]
                )
                for image in frames
            ],
            dtype=float,
        )
        complete = np.all(np.isfinite(box_samples), axis=0)
        missing_sites = set(int(index) for index in np.flatnonzero(~complete))
        return (
            box_samples,
            tuple(sorted(saturated_sites)),
            tuple(sorted(missing_sites)),
            np.asarray(np.mean(frames, axis=0), dtype=np.float32),
        )

    def _prepare_artifacts(self, context: object) -> dict[str, Path]:
        root = Path(context.run_directory).expanduser().resolve()
        paths = {
            "root": root,
            "data": root / "data",
            "candidates": root / "data" / "candidates",
            "figures": root / "figures",
            "final": root / "final",
        }
        for name in ("data", "candidates", "figures", "final"):
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
                "box_half_width": int(self.model.integration_half_width),
                "box_reducer": str(self.model.reducer),
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
    ) -> Path:
        arrays: dict[str, object] = {
            "box_samples": np.asarray(samples, dtype="<f8"),
        }
        bool_fields = {
            "fit_valid",
            "observable_valid",
            "single_population",
            "single_dark_only",
            "single_bright_only",
            "fit_invalid",
        }
        for name in _CANDIDATE_VECTOR_FIELDS:
            values = measurement[name]
            if name == "decision":
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
                "format": "zlc.slm.feedback-candidate",
                "solver": None if solver is None else dict(solver),
                "slm_command_receipt": dict(self.slm.last_command_receipt),
            }
        )
        path = _write_npz(
            paths["candidates"] / f"candidate-{int(candidate):04d}.npz",
            arrays=arrays,
            metadata=metadata,
        )
        context.register_artifact(
            f"candidate_{int(candidate):04d}", path, role="checkpoint"
        )
        return path

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
    ) -> tuple[Path, Path]:
        base = paths["figures"] / f"{name}.png"
        try:
            image, archive = save_figure_artifact(
                base,
                plot_input=snapshot,
                spec=spec,
                parameters={} if parameters is None else parameters,
                size=size,
                source={
                    "task": self.instance_id,
                    "calibration_path": str(self.calibration_path),
                    "science_context_path": str(self.science_context_path),
                },
            )
        except BaseException:
            archive = base.with_suffix(".npz")
            if archive.is_file():
                context.register_artifact(
                    f"{name}_figure", archive, role="figure", contract_id="zlc.figure"
                )
            raise
        context.register_artifact(
            f"{name}_figure", archive, role="figure", contract_id="zlc.figure"
        )
        context.register_artifact(f"{name}_preview", image, role="preview")
        return image, archive

    def _save_figures(
        self,
        context: object,
        paths: Mapping[str, Path],
        *,
        history: list[dict[str, object]],
        selected: Mapping[str, object],
        initial_phase: np.ndarray,
        initial_mean_frame: np.ndarray,
    ) -> None:
        count = len(history)
        if count < 1:
            return
        generation = str(getattr(context.generation, "value", context.generation))
        candidate_id = AxisId("slm_feedback.candidate")
        candidate_column = PointColumn(
            candidate_id,
            "candidate",
            SCAN_POINT,
            PointColumn.NUMERIC,
            tuple(range(1, count + 1)),
        )
        site_axis = AxisSpec(
            AxisId("slm_feedback.site"),
            "site",
            SITE,
            self._site_count,
            tuple(range(self._site_count)),
            coordinate_labels=tuple(
                f"({int(row)}, {int(column)})"
                for row, column in zip(self._rows, self._columns, strict=True)
            ),
        )

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
                roles=(SCAN_POINT, SITE),
                axis_specs={SITE: site_axis},
                point_columns={SCAN_POINT: candidate_column},
                generation=generation,
                revision=count,
                validity=np.isfinite(values)[None],
            )

        uniformity_axis = AxisSpec(
            AxisId("slm_feedback.uniformity.metric"),
            "metric",
            COMPONENT,
            2,
            (0, 1),
            coordinate_labels=("all sites", "observable sites"),
        )
        uniformity = np.asarray(
            [
                [
                    np.nan if item["uniformity_ratio"] is None else item["uniformity_ratio"],
                    np.nan
                    if item["observable_uniformity_ratio"] is None
                    else item["observable_uniformity_ratio"],
                ]
                for item in history
            ],
            dtype="<f8",
        )
        uniformity_snapshot = snapshot_from_array(
            uniformity[None],
            producer=self.instance_id,
            signal="uniformity_figure",
            roles=(SCAN_POINT, COMPONENT),
            axis_specs={COMPONENT: uniformity_axis},
            point_columns={SCAN_POINT: candidate_column},
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
                group=AxisRef.data(str(uniformity_axis.axis_id)),
                labels=PlotLabels(
                    title="SLM feedback uniformity",
                    x="candidate",
                    y="max / min bright-dark",
                ),
            ),
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
                group=AxisRef.data(str(site_axis.axis_id)),
                labels=PlotLabels(
                    title="Per-site bright-dark evolution",
                    x="candidate",
                    y="bright - dark",
                ),
            ),
        )

        self._save_figure(
            context,
            paths,
            "weight_evolution",
            snapshot=site_history_snapshot("control_weight", "weight_figure"),
            spec=CurvePlot(
                AxisRef.point(str(candidate_id)),
                group=AxisRef.data(str(site_axis.axis_id)),
                labels=PlotLabels(
                    title="Per-site target weight evolution",
                    x="candidate",
                    y="normalized weight",
                ),
            ),
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
            roles=(SITE, COMPONENT),
            axis_specs={SITE: site_axis, COMPONENT: shot_axis},
            generation=generation,
            revision=count,
        )
        self._save_figure(
            context,
            paths,
            "selected_site_histograms",
            snapshot=histogram_snapshot,
            spec=FacetGridPlot(
                AxisRef.data(str(site_axis.axis_id)),
                HistogramPlot(
                    PlotLabels(
                        title="Selected candidate site distributions",
                        x="box signal",
                        y="shots",
                    )
                ),
            ),
            parameters={"bin_count": min(60, max(10, self.shots // 2))},
        )

        selected_number = int(selected["candidate"])
        comparison_id = AxisId("slm_feedback.comparison")
        comparison_column = PointColumn(
            comparison_id,
            "state",
            SCAN_POINT,
            PointColumn.NUMERIC,
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
                    roles=(SCAN_POINT, SPATIAL_Y, SPATIAL_X),
                    axis_specs={SPATIAL_Y: y_axis, SPATIAL_X: x_axis},
                    point_columns={SCAN_POINT: comparison_column},
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
        self._save_figure(
            context,
            paths,
            "camera_initial_selected",
            snapshot=camera_snapshot,
            spec=FacetGridPlot(
                AxisRef.point(str(comparison_id)),
                ImagePlot(
                    AxisRef.data(str(image_x.axis_id)),
                    AxisRef.data(str(image_y.axis_id)),
                ),
                labels=PlotLabels(title="Initial and selected camera mean"),
            ),
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
                    AxisRef.data(str(phase_x.axis_id)),
                    AxisRef.data(str(phase_y.axis_id)),
                ),
                labels=PlotLabels(title="Initial and selected SLM phase"),
            ),
        )

    def _write_summary(
        self,
        context: object,
        paths: Mapping[str, Path],
        *,
        status: str,
        history: list[dict[str, object]],
        selected_candidate: int | None,
        error: BaseException | None = None,
        rollback: Mapping[str, object] | None = None,
    ) -> None:
        initial = None if not history else history[0]
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
                [np.asarray(item["observable_valid"], dtype=bool) for item in history]
            )
            if history
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
        }
        json_path = write_readable_json(
            paths["root"] / "summary.json", _plain_json(document)
        )
        lines = [
            f"SLM feedback status: {status}",
            f"Candidates measured: {len(history)}",
            f"Selected candidate: {selected_candidate if selected_candidate is not None else 'none'}",
        ]
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
                )
            )
        if rollback is not None:
            lines.append(f"Rollback: {rollback.get('status')}")
        if error is not None:
            lines.append(f"Error: {type(error).__name__}: {error}")
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
        status: str,
        republish: bool,
    ) -> dict[str, object]:
        expected = canonical_phase(candidate["phase"], self.slm.shape_yx)
        observed = self.slm.last_commanded_phase
        applied = (
            expected
            if observed is not None and np.array_equal(observed, expected)
            else self._apply_exact(expected)
        )
        candidate_number = int(candidate["candidate"])
        if republish:
            self._publish_candidate(
                context,
                phase=applied,
                candidate=candidate_number,
                history=history,
            )
        retained_history = candidate.get("history")
        if (
            history
            and isinstance(retained_history, Mapping)
            and candidate.get("samples") is not None
            and candidate.get("mean_frame") is not None
            and initial_mean_frame is not None
        ):
            self._save_figures(
                context,
                paths,
                history=history,
                selected=candidate,
                initial_phase=initial_phase,
                initial_mean_frame=initial_mean_frame,
            )
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
            applied,
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
        self._write_summary(
            context,
            paths,
            status=status,
            history=history,
            selected_candidate=(
                candidate_number if isinstance(retained_history, Mapping) else None
            ),
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
            "updates": len(history),
            "feedback_mode": self.feedback_mode,
            "requested_exposure_seconds": self.exposure_seconds,
            "actual_exposure_seconds": self._actual_exposure_seconds,
        }

    def execute(self, context: object) -> dict[str, object]:
        incoming = self._incoming_phase
        incoming_pattern = self._pattern_phase
        paths = self._prepare_artifacts(context)
        history: list[dict[str, object]] = []
        retained_valid: dict[str, object] | None = None
        most_visible_observed: dict[str, object] | None = None
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
                board=self.sequencer.describe(),
                api_values={},
            )
            _check_cancelled(context)
            current_target = self.target
            spot_optimizer_state: dict[str, object] = {}
            current_pattern = incoming_pattern
            current_phase = incoming
            solver_metadata: Mapping[str, object] | None = None
            previous_weights = np.full(self._site_count, np.nan, dtype=float)
            previous_contrast = np.full(self._site_count, np.nan, dtype=float)
            previous_dark_mean = np.full(self._site_count, np.nan, dtype=float)
            previous_dark_standard_error = np.full(
                self._site_count, np.nan, dtype=float
            )
            loading_dark_bound = np.full(
                self._site_count, np.nan, dtype=float
            )
            loading_loaded_bound = np.full(
                self._site_count, np.nan, dtype=float
            )
            prior_measurement = self._prior_pattern_metadata.get("measurement")
            comparable_history = bool(
                self._prior_pattern_metadata.get("feedback_controller")
                == _CONTROLLER_CONTRACT
                and self._prior_pattern_metadata.get("feedback_mode") == self.feedback_mode
                and self._prior_pattern_metadata.get("pulse_path") == str(self.pulse_path)
                and type(self._prior_pattern_metadata.get("exposure_seconds"))
                in (int, float)
                and float(self._prior_pattern_metadata["exposure_seconds"])
                == self.exposure_seconds
                and float(
                    self._prior_pattern_metadata.get("single_gaussian_boost", -1.0)
                )
                == self.single_gaussian_boost
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
                    "previous_valid_control_weight", float
                )
                if restored_control is not None:
                    previous_weights[:] = restored_control
                for name, destination in (
                    ("previous_valid_bright_minus_dark", previous_contrast),
                    ("previous_valid_dark_mean", previous_dark_mean),
                    (
                        "previous_valid_dark_standard_error",
                        previous_dark_standard_error,
                    ),
                    ("loading_dark_bound", loading_dark_bound),
                    ("loading_loaded_bound", loading_loaded_bound),
                ):
                    restored_values = restored(name, float)
                    if restored_values is not None:
                        destination[:] = restored_values

                prior_weights = restored("control_weight", float)
                if prior_weights is None:
                    raw_prior_weights = restored("target_weight", float)
                    if raw_prior_weights is not None:
                        prior_weights = _control_weights(raw_prior_weights)
                prior_contrast = restored("bright_minus_dark", float)
                prior_observable = restored("observable_valid", bool)
                if prior_observable is None:
                    prior_observable = restored("fit_valid", bool)
                if (
                    prior_weights is not None
                    and prior_contrast is not None
                    and prior_observable is not None
                ):
                    usable = (
                        prior_observable
                        & np.isfinite(prior_weights)
                        & np.isfinite(prior_contrast)
                    )
                    previous_weights[usable] = prior_weights[usable]
                    previous_contrast[usable] = prior_contrast[usable]

                prior_fit_valid = restored("fit_valid", bool)
                prior_dark = restored("dark_mean", float)
                prior_dark_error = restored("dark_standard_error", float)
                if all(
                    value is not None
                    for value in (
                        prior_fit_valid,
                        prior_dark,
                        prior_dark_error,
                    )
                ):
                    gaussian_usable = (
                        prior_fit_valid
                        & np.isfinite(prior_dark)
                        & np.isfinite(prior_dark_error)
                    )
                    previous_dark_mean[gaussian_usable] = prior_dark[gaussian_usable]
                    previous_dark_standard_error[gaussian_usable] = (
                        prior_dark_error[gaussian_usable]
                    )
            for iteration in range(self._candidate_capacity):
                _check_cancelled(context)
                applied = (
                    current_phase
                    if iteration == 0
                    else self._apply_exact(current_phase)
                )
                candidate_number = iteration + 1
                candidate_solver = solver_metadata
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
                )
                samples, saturated_sites, missing_sites, mean_frame = self._measure(
                    pulse, context, iteration
                )
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
                (
                    contrast,
                    error,
                    dark_only,
                    bright_only,
                ) = _single_population_observables(
                    fitted,
                    shots=self.shots,
                    previous_dark_mean=previous_dark_mean,
                    previous_dark_standard_error=previous_dark_standard_error,
                )
                observable_valid = fit_valid | bright_only
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
                    current_target[self._rows, self._columns], dtype=float
                )
                current_control_weights = _control_weights(current_weights)
                for site in np.flatnonzero(dark_only):
                    loaded_bound = loading_loaded_bound[site]
                    if not np.isfinite(loaded_bound) or (
                        current_control_weights[site] < loaded_bound
                    ):
                        prior_dark = loading_dark_bound[site]
                        loading_dark_bound[site] = (
                            current_control_weights[site]
                            if not np.isfinite(prior_dark)
                            else max(prior_dark, current_control_weights[site])
                        )
                for site in np.flatnonzero(observable_valid):
                    dark_bound = loading_dark_bound[site]
                    if not np.isfinite(dark_bound) or (
                        current_control_weights[site] > dark_bound
                    ):
                        prior_loaded = loading_loaded_bound[site]
                        loading_loaded_bound[site] = (
                            current_control_weights[site]
                            if not np.isfinite(prior_loaded)
                            else min(prior_loaded, current_control_weights[site])
                        )
                loading_floor = np.full(
                    self._site_count, np.nan, dtype=float
                )
                bracketed = (
                    np.isfinite(loading_dark_bound)
                    & np.isfinite(loading_loaded_bound)
                    & (loading_dark_bound < loading_loaded_bound)
                )
                loading_floor[bracketed] = np.sqrt(
                    loading_dark_bound[bracketed]
                    * loading_loaded_bound[bracketed]
                )
                feedback_valid = observable_valid
                proposed_target, log_correction, response_slope, decisions = (
                    _updated_target(
                        current_target,
                        contrast,
                        error,
                        feedback_valid,
                        dark_only,
                        self._rows,
                        self._columns,
                        previous_weights=previous_weights,
                        previous_contrast=previous_contrast,
                        feedback_gain=self.feedback_gain,
                        single_gaussian_boost=self.single_gaussian_boost,
                        maximum_weight_change=self.maximum_weight_change,
                        minimum_control_weight=loading_floor,
                    )
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
                    "feedback_gain": self.feedback_gain,
                    "single_gaussian_boost": self.single_gaussian_boost,
                    "maximum_weight_change": self.maximum_weight_change,
                    "feedback_mode": self.feedback_mode,
                    "requested_exposure_seconds": self.exposure_seconds,
                    "actual_exposure_seconds": self._actual_exposure_seconds,
                    "effective_photoelectrons": self._effective_photoelectrons,
                    "effective_count_unit": self._effective_count_unit,
                    "target_weight": _json_floats(current_weights),
                    "control_weight": _json_floats(current_control_weights),
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
                    "bright_fraction": _json_floats(fitted["bright_fraction"]),
                    "fit_fidelity": _json_floats(fitted["fidelity"]),
                    "fit_bic_gain": _json_floats(fitted["bic_gain"]),
                    "fit_valid": [bool(value) for value in fit_valid],
                    "observable_valid": [
                        bool(value) for value in observable_valid
                    ],
                    "single_population": [bool(value) for value in fit_single],
                    "single_dark_only": [bool(value) for value in dark_only],
                    "single_bright_only": [bool(value) for value in bright_only],
                    "fit_invalid": [bool(value) for value in fit_invalid],
                    "single_mean": _json_floats(fitted["single_mean"]),
                    "single_sigma": _json_floats(fitted["single_sigma"]),
                    "decision": [str(value) for value in decisions],
                    "requested_log_correction": _json_floats(log_correction),
                    "response_log_slope": _json_floats(response_slope),
                    "previous_valid_control_weight": _json_floats(
                        previous_weights
                    ),
                    "previous_valid_bright_minus_dark": _json_floats(
                        previous_contrast
                    ),
                    "previous_valid_dark_mean": _json_floats(
                        previous_dark_mean
                    ),
                    "previous_valid_dark_standard_error": _json_floats(
                        previous_dark_standard_error
                    ),
                    "loading_dark_bound": _json_floats(
                        loading_dark_bound
                    ),
                    "loading_loaded_bound": _json_floats(
                        loading_loaded_bound
                    ),
                    "loading_floor": _json_floats(loading_floor),
                    "missing_sites": list(missing_sites),
                    "single_gaussian_sites": [
                        int(value) for value in np.flatnonzero(fit_single)
                    ],
                    "invalid_sites": [
                        int(value) for value in np.flatnonzero(fit_invalid)
                    ],
                    "minimum_visibility_confidence": visibility_margin,
                })
                history[-1]["checkpoint_path"] = (
                    f"data/candidates/candidate-{candidate_number:04d}.npz"
                )
                self._publish_candidate(
                    context,
                    phase=applied,
                    candidate=candidate_number,
                    history=history,
                )
                completed: dict[str, object] = {
                    "candidate": candidate_number,
                    "phase": np.array(applied, copy=True),
                    "pattern_phase": np.array(current_pattern, copy=True),
                    "target": np.array(current_target, copy=True),
                    "solver": candidate_solver,
                    "history": history[-1],
                    "score": observed_score,
                    "contrast": np.array(contrast, copy=True),
                    "standard_error": np.array(error, copy=True),
                    "samples": np.array(samples, copy=True),
                    "mean_frame": np.array(mean_frame, copy=True),
                }
                completed["visibility_rank"] = (
                    visibility,
                    -int(np.count_nonzero(fit_invalid)),
                    float("-inf") if visibility_margin is None else visibility_margin,
                )
                if (
                    most_visible_observed is None
                    or completed["visibility_rank"]
                    >= most_visible_observed["visibility_rank"]
                ):
                    most_visible_observed = completed
                if valid:
                    if (
                        retained_valid is None
                        or float(completed["score"]) < float(retained_valid["score"])
                    ):
                        retained_valid = completed
                    context.report_progress(
                        f"qCMOS bright-dark ratio {score:.5f}; "
                        f"simultaneous 95% upper {confidence_upper:.5f}"
                    )
                else:
                    context.report_progress(
                        f"Candidate {candidate_number}: {visibility}/"
                        f"{self._site_count} site fits valid; applying only "
                        "history-supported site updates"
                    )
                continue_feedback = candidate_number < self._candidate_capacity
                if continue_feedback:
                    previous_weights[observable_valid] = current_control_weights[
                        observable_valid
                    ]
                    previous_contrast[observable_valid] = contrast[observable_valid]
                    previous_dark_mean[fit_valid] = np.asarray(
                        fitted["dark_mean"], dtype=float
                    )[fit_valid]
                    previous_dark_standard_error[fit_valid] = np.asarray(
                        fitted["dark_standard_error"], dtype=float
                    )[fit_valid]
                    if np.array_equal(proposed_target, current_target):
                        stalled = True
                        continue_feedback = False
                        history[-1]["next_phase_changed"] = False
                        termination_reason = (
                            "all sites held because this shot batch supplied no "
                            "actionable update"
                        )
                    else:
                        try:
                            next_pattern, solver_metadata = solve_phase(
                                proposed_target,
                                pupil_amplitude=self._pupil_amplitude,
                                initial_phase=current_pattern,
                                objective_kind="spots",
                                iterations=None,
                                stop_requested=context.cancel_requested,
                                spot_optimizer_state=spot_optimizer_state,
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
                            )
                            raise
                        next_phase = self._science_phase(next_pattern)
                        if np.array_equal(next_phase, applied):
                            stalled = True
                            continue_feedback = False
                            history[-1]["next_phase_changed"] = False
                            termination_reason = (
                                "the target correction produced no different SLM phase; "
                                "no second shot batch was taken"
                            )
                        else:
                            history[-1]["next_phase_changed"] = True
                            current_target = proposed_target
                            current_pattern = next_pattern
                            current_phase = next_phase
                else:
                    history[-1]["next_phase_changed"] = None
                self._save_candidate_checkpoint(
                    context,
                    paths,
                    candidate=candidate_number,
                    samples=samples,
                    measurement=history[-1],
                    solver=candidate_solver,
                )
                if not continue_feedback:
                    break
            selected = retained_valid or most_visible_observed
            if selected is None:
                raise RuntimeError("qCMOS feedback produced no completed candidate")
            status = "stalled" if stalled else "completed"
            selected["outcome"] = {
                "status": status,
                "reason": termination_reason,
                "selected_candidate": int(selected["candidate"]),
                "shots_per_candidate": self.shots,
                "candidates_measured": len(history),
            }
            context.seal_terminal()
            return self._finish_candidate(
                context,
                selected,
                history,
                paths=paths,
                initial_phase=incoming,
                initial_mean_frame=initial_mean_frame,
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
                        status="stopped",
                        republish=True,
                    )
                except BaseException as stop_error:
                    try:
                        self._apply_exact(incoming)
                    except BaseException as restore_error:
                        raise RuntimeError(
                            "SLM feedback Stop failed and the incoming phase "
                            "could not be restored"
                        ) from restore_error
                    raise stop_error
            try:
                self._apply_exact(incoming)
            except BaseException as restore_error:
                try:
                    self._write_summary(
                        context,
                        paths,
                        status="failed",
                        history=history,
                        selected_candidate=None,
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
                    selected_candidate=None,
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
    "SLM_PHASE_ARTIFACT_CONTRACT",
    "SlmFeedbackTask",
    "UNIFORMITY_HISTORY_OUTPUT",
]
