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
    IMAGE_POINT_OVERLAY_GEOMETRY_RECORD,
    ImageFrame,
    ImagePointOverlay,
    ImagePlot,
    PlotLabels,
    PointStatus,
    image_point_overlay_geometry,
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
    CAMERA_FRAMES_OUTPUT,
    CameraMeasurementNode,
    CameraMeasurementRequest,
    _finite_cycle_output,
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
SITE_SIGNAL_HISTORY_OUTPUT = DatasetOutputDeclaration(
    "site_signal_history", "slm-feedback.site-signal-history"
)
TARGET_SHARE_HISTORY_OUTPUT = DatasetOutputDeclaration(
    "target_share_history", "slm-feedback.target-share-history"
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


def _adapt_double_gain(
    previous_contrast: object,
    previous_error: object,
    previous_valid: object,
    current_contrast: object,
    current_error: object,
    current_valid: object,
    *,
    gain: float,
    improvement_streak: int,
) -> tuple[float, int, str, int, float | None, float | None]:
    """Adapt one scalar double-site gain from comparable formal candidates."""

    before = np.asarray(previous_contrast, dtype=float)
    before_error = np.asarray(previous_error, dtype=float)
    before_valid = np.asarray(previous_valid, dtype=bool)
    after = np.asarray(current_contrast, dtype=float)
    after_error = np.asarray(current_error, dtype=float)
    after_valid = np.asarray(current_valid, dtype=bool)
    if not (
        before.shape
        == before_error.shape
        == before_valid.shape
        == after.shape
        == after_error.shape
        == after_valid.shape
    ):
        raise ValueError("adaptive double-gain histories differ in shape")
    common = (
        before_valid
        & after_valid
        & np.isfinite(before)
        & np.isfinite(before_error)
        & np.isfinite(after)
        & np.isfinite(after_error)
        & (before > 0.0)
        & (after > 0.0)
        & (before_error >= 0.0)
        & (after_error >= 0.0)
    )
    common_count = int(np.count_nonzero(common))
    if common_count < 2:
        return float(gain), 0, "hold_insufficient_common_double", common_count, None, None

    previous_ratio, previous_lower, previous_upper, _ = _ratio_interval(
        before[common], before_error[common]
    )
    current_ratio, current_lower, current_upper, _ = _ratio_interval(
        after[common], after_error[common]
    )
    previous_uncertainty = max(
        np.log(previous_ratio / previous_lower),
        np.log(previous_upper / previous_ratio),
    )
    current_uncertainty = max(
        np.log(current_ratio / current_lower),
        np.log(current_upper / current_ratio),
    )
    noise = float(np.hypot(previous_uncertainty, current_uncertainty))
    improvement = float(np.log(previous_ratio / current_ratio))
    selected_gain = float(gain)
    streak = int(improvement_streak)
    if improvement > noise:
        streak += 1
        if streak >= 2:
            selected_gain *= 1.25
            streak = 1
            action = "increase_after_continuous_improvement"
        else:
            action = "hold_first_significant_improvement"
    elif improvement < -noise:
        selected_gain *= 0.5
        streak = 0
        action = "decrease_after_worsening"
    else:
        streak = 0
        action = "hold_within_uncertainty"
    return (
        selected_gain,
        streak,
        action,
        common_count,
        previous_ratio,
        current_ratio,
    )


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


def _fit_contrasts(samples: object) -> dict[str, np.ndarray]:
    """Classify each site's one user-authored shot batch.

    A resolved two-population fit supplies the bright-minus-dark feedback
    observable.  Evidence for one Gaussian is a different, useful physical
    result: this feedback mode treats it as a site which did not load.  Bad
    samples or a numerically undecidable model remain invalid and therefore
    cannot create a control action.
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
    bic_gain_even = np.full(sites, np.nan, dtype=float)
    bic_gain_odd = np.full(sites, np.nan, dtype=float)
    bic_stable = np.zeros(sites, dtype=bool)
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
        fit_even = fit_bimodal(column[0::2], min_component_fraction=0.01)
        fit_odd = fit_bimodal(column[1::2], min_component_fraction=0.01)
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
        bic_gain_even[site] = _bic_gain(column[0::2], fit_even)
        bic_gain_odd[site] = _bic_gain(column[1::2], fit_odd)
        finite_pair = bool(
            estimate > 0.0
            and np.isfinite(sem)
            and sem >= 0.0
            and np.isfinite(bic_gain[site])
            and np.isfinite(bic_gain_even[site])
            and np.isfinite(bic_gain_odd[site])
        )
        if not finite_pair:
            continue
        bic_stable[site] = bool(
            fit.ok
            and bic_gain[site] > 0.0
            and bic_gain_even[site] > 0.0
            and bic_gain_odd[site] > 0.0
        )
        if bic_stable[site]:
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
        "bic_gain_even": bic_gain_even,
        "bic_gain_odd": bic_gain_odd,
        "bic_stable": bic_stable,
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
    "bic_gain_even",
    "bic_gain_odd",
    "bic_stable",
    "fit_valid",
    "observable_valid",
    "single_population",
    "fit_invalid",
    "single_mean",
    "single_sigma",
    "decision",
    "requested_log_correction",
    "response_log_slope",
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


def _support(
    target: np.ndarray,
    calibration: TrapCalibration,
    *,
    science_context_path: str | Path,
    command_receipt: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, SiteMap]:
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

    return rows, columns, registered


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


def _selected_probe_target(
    target: np.ndarray,
    probe_sites: np.ndarray,
    baseline_contrast: np.ndarray,
    baseline_valid: np.ndarray,
    measurements: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    feedback_gain: float,
    maximum_weight_change: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probe = np.asarray(probe_sites, dtype=bool)
    values = np.asarray(baseline_contrast, dtype=float)
    valid = np.asarray(baseline_valid, dtype=bool)
    shape = (len(rows),)
    reasonable = (
        valid & np.isfinite(values) & (values > 0.0)
    )
    reference = (
        float(np.exp(np.mean(np.log(values[reasonable]))))
        if np.any(reasonable) else float("nan")
    )
    selected = np.ones(shape, dtype=float)
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
                    if np.isfinite(reference) else float("inf"),
                    uncertainty / value,
                    factor,
                )
            )
        sides = {option[0] for option in options}
        if not options:
            decisions[site] = "probe_hold_unobservable"
        elif len(sides) == 2 and not np.isfinite(reference):
            decisions[site] = "probe_hold_no_reference"
        else:
            chosen = min(options, key=lambda item: (item[1], item[2]))
            selected[site] = chosen[3]
            side = "lower" if chosen[0] else "upper"
            decisions[site] = f"probe_choose_{side}_{'closest' if len(sides) == 2 else 'only'}"
    gain = float(feedback_gain)
    limit = float(maximum_weight_change)
    cap = np.log1p(limit)
    requested_log = np.log(selected[probe])
    observed_factors = np.array(selected, copy=True)
    selected[probe] = np.exp(
        np.sign(requested_log) * np.minimum(gain * np.abs(requested_log), cap)
    )
    updated, effective = _relative_probe_target(
        target, selected, probe, rows, columns
    )
    selected_effective = np.ones(shape, dtype=float)
    selected_effective[probe] = effective[probe]
    return updated, selected_effective, decisions, observed_factors


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


def _updated_single_bracket(
    current: object,
    active_single: object,
    single_bound: object,
    observable_bound: object,
    direction: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    present = np.asarray(current, dtype=float)
    active = np.asarray(active_single, dtype=bool)
    single = np.asarray(single_bound, dtype=float).copy()
    observable = np.asarray(observable_bound, dtype=float).copy()
    sign = np.asarray(direction, dtype=float).copy()
    if not all(
        value.shape == present.shape
        for value in (active, single, observable, sign)
    ):
        raise ValueError("single bracket shapes differ")
    crossed = (
        active
        & np.isfinite(observable)
        & (((sign > 0.0) & (present > observable))
           | ((sign < 0.0) & (present < observable)))
    )
    sign[crossed] = np.sign(observable[crossed] - present[crossed])
    single[crossed] = present[crossed]
    up = sign > 0.0
    down = sign < 0.0
    single[active & up] = np.fmax(single[active & up], present[active & up])
    single[active & down] = np.fmin(single[active & down], present[active & down])
    return single, observable, sign, crossed


def _single_bracket_step(
    current: np.ndarray,
    boundary: np.ndarray,
    direction: np.ndarray,
    maximum_weight_change: float,
) -> np.ndarray:
    present = np.asarray(current, dtype=float)
    midpoint = np.asarray(boundary, dtype=float)
    sign = np.asarray(direction, dtype=float)
    active = (
        np.isfinite(present) & (present > 0.0)
        & np.isfinite(midpoint) & (midpoint > 0.0)
        & (((sign > 0.0) & (midpoint > present))
           | ((sign < 0.0) & (midpoint < present)))
    )
    step = np.zeros(present.shape, dtype=float)
    cap = np.log1p(float(maximum_weight_change))
    step[active] = np.clip(
        np.log(midpoint[active] / present[active]), -cap, cap
    )
    return step


def _needs_probe(
    single: np.ndarray,
    observable: np.ndarray,
    acquisition_invalid: np.ndarray,
    previous_weights: np.ndarray,
    previous_contrast: np.ndarray,
) -> np.ndarray:
    has_history = (
        np.isfinite(previous_weights) & (previous_weights > 0.0)
        & np.isfinite(previous_contrast) & (previous_contrast > 0.0)
    )
    return _unobservable_single(single, observable, acquisition_invalid) & ~has_history


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
    previous_weights: np.ndarray,
    previous_contrast: np.ndarray,
    feedback_gain: float,
    maximum_weight_change: float,
    directed_log_step: np.ndarray | None = None,
    control_boundary: np.ndarray | None = None,
    control_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update observable sites relatively while holding all others."""

    values = np.asarray(contrast, dtype=float)
    errors = np.asarray(standard_error, dtype=float)
    control_valid = np.asarray(valid, dtype=bool)
    references = np.asarray(reference_valid, dtype=bool)
    prior_weights = np.asarray(previous_weights, dtype=float)
    prior_values = np.asarray(previous_contrast, dtype=float)
    gain = float(feedback_gain)
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
    if (
        values.shape != site_shape
        or errors.shape != site_shape
        or control_valid.shape != site_shape
        or references.shape != site_shape
        or prior_weights.shape != site_shape
        or prior_values.shape != site_shape
        or direction.shape != site_shape
        or boundary.shape != site_shape
        or boundary_direction.shape != site_shape
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
    current_weights = _control_weights(raw_weights)
    log_correction = np.zeros(site_shape, dtype=float)
    response_slope = np.full(site_shape, np.nan, dtype=float)
    decision = np.full(site_shape, "hold_invalid", dtype="<U32")
    for site in range(len(rows)):
        if control_valid[site] and np.isfinite(reference):
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
        elif direction[site] != 0.0:
            log_correction[site] = float(
                np.clip(
                    direction[site],
                    -np.log1p(maximum_change),
                    np.log1p(maximum_change),
                )
            )
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
        probe_factors: tuple[float, ...],
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
            self._registered_site_map,
        ) = _support(
            frozen_target,
            calibration,
            science_context_path=context_path,
            command_receipt=receipt,
        )
        self._site_count = len(self._rows)
        self._site_centers_xy = np.asarray(
            self._registered_site_map.centers_xy, dtype=float
        ).copy()
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
        self.feedback_gain = float(feedback_gain)
        self.maximum_weight_change = float(maximum_weight_change)
        self.max_updates = int(max_updates)
        self._publication_revision = 0
        self._actual_exposure_seconds: float | None = None
        self._effective_photoelectrons: bool | None = None
        self._effective_count_unit: str | None = None
        self._last_measured_phase: np.ndarray | None = None
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
            "probe_factors": list(self.probe_factors),
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
        site_axis = self._registered_site_map.site_axis
        site_signal_event = snapshot_from_array(
            site_signal[None],
            producer=self.instance_id,
            signal=SITE_SIGNAL_HISTORY_OUTPUT.name,
            roles=(SCAN_POINT, SITE),
            axis_specs={SITE: site_axis},
            point_columns=point_columns,
            generation=generation,
            revision=publication_revision,
            validity=np.isfinite(site_signal)[None],
        )
        target_share_event = snapshot_from_array(
            target_share[None],
            producer=self.instance_id,
            signal=TARGET_SHARE_HISTORY_OUTPUT.name,
            roles=(SCAN_POINT, SITE),
            axis_specs={SITE: site_axis},
            point_columns=point_columns,
            generation=generation,
            revision=publication_revision,
            validity=np.isfinite(target_share)[None],
        )
        record = self._run_record()
        outputs = {
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
            SITE_SIGNAL_HISTORY_OUTPUT.name: LiveDatasetOutput(
                SITE_SIGNAL_HISTORY_OUTPUT,
                site_signal_event,
                MonitorCoverage(
                    self._candidate_capacity,
                    self._candidate_capacity,
                    retain_at_terminal=True,
                ),
                record,
            ),
            TARGET_SHARE_HISTORY_OUTPUT.name: LiveDatasetOutput(
                TARGET_SHARE_HISTORY_OUTPUT,
                target_share_event,
                MonitorCoverage(
                    self._candidate_capacity,
                    self._candidate_capacity,
                    retain_at_terminal=True,
                ),
                record,
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

            result = capture.collect(
                commit_cycle=commit_camera_cycle,
                retain_cycles=False,
            )
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
        mean_frame = np.asarray(np.mean(frames, axis=0), dtype=np.float32)
        return (
            box_samples,
            tuple(sorted(saturated_sites)),
            tuple(sorted(missing_sites)),
            mean_frame,
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
            "fit_invalid",
            "bic_stable",
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
        outcome: Mapping[str, object] | None = None,
        error: BaseException | None = None,
        rollback: Mapping[str, object] | None = None,
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
            "final_double_feedback_gain": (
                None
                if not formal_history
                else formal_history[-1]["double_feedback_gain"]
            ),
            "double_gain_history": [
                {
                    "candidate": item["iteration"],
                    "gain": item["double_feedback_gain"],
                    "action": item["adaptive_gain_action"],
                    "common_double_sites": item["adaptive_gain_common_sites"],
                    "previous_ratio": item["adaptive_gain_previous_ratio"],
                    "current_ratio": item["adaptive_gain_current_ratio"],
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
                    "double_feedback_gain": item["double_feedback_gain"],
                    "adaptive_gain_action": item["adaptive_gain_action"],
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
            f"Final double feedback gain: {document['final_double_feedback_gain']}",
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
            outcome=metadata["outcome"],
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
            double_feedback_gain = self.feedback_gain
            adaptive_improvement_streak = 0
            previous_formal_contrast: np.ndarray | None = None
            previous_formal_error: np.ndarray | None = None
            previous_formal_valid: np.ndarray | None = None
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
            probe_baseline_previous_weights = np.full(
                self._site_count, np.nan, dtype=float
            )
            probe_baseline_previous_contrast = np.full(
                self._site_count, np.nan, dtype=float
            )
            probe_baseline_directed_step = np.zeros(
                self._site_count, dtype=float
            )
            probe_baseline_control_boundary = np.full(
                self._site_count, np.nan, dtype=float
            )
            probe_baseline_control_direction = np.zeros(
                self._site_count, dtype=float
            )
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
            probe_direction_log_step = np.zeros(self._site_count, dtype=float)
            probe_direction_sign = np.zeros(self._site_count, dtype=float)
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
                contrast = np.asarray(fitted["contrast"], dtype=float)
                error = np.asarray(fitted["standard_error"], dtype=float)
                observable_valid = fit_valid
                adaptive_gain_action = "ignored_diagnostic_probe"
                adaptive_gain_common_sites = 0
                adaptive_gain_previous_ratio: float | None = None
                adaptive_gain_current_ratio: float | None = None
                if candidate_kind != "probe":
                    if (
                        previous_formal_contrast is None
                        or previous_formal_error is None
                        or previous_formal_valid is None
                    ):
                        adaptive_gain_action = "initialize_formal_double_history"
                    else:
                        (
                            double_feedback_gain,
                            adaptive_improvement_streak,
                            adaptive_gain_action,
                            adaptive_gain_common_sites,
                            adaptive_gain_previous_ratio,
                            adaptive_gain_current_ratio,
                        ) = _adapt_double_gain(
                            previous_formal_contrast,
                            previous_formal_error,
                            previous_formal_valid,
                            contrast,
                            error,
                            observable_valid,
                            gain=double_feedback_gain,
                            improvement_streak=adaptive_improvement_streak,
                        )
                    previous_formal_contrast = np.array(contrast, copy=True)
                    previous_formal_error = np.array(error, copy=True)
                    previous_formal_valid = np.array(observable_valid, copy=True)
                needs_probe_sites = _needs_probe(
                    fit_single,
                    observable_valid,
                    acquisition_invalid,
                    previous_weights,
                    previous_contrast,
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
                    current_target[self._rows, self._columns], dtype=float
                )
                current_control_weights = _control_weights(current_weights)
                bracket_recovery = np.zeros(self._site_count, dtype=bool)
                episode_probe_sites = np.zeros(self._site_count, dtype=bool)
                starts_probe_episode = False
                if candidate_kind != "probe":
                    probe_direction_log_step[fit_valid] = 0.0
                    unobservable_single = _unobservable_single(
                        fit_single,
                        observable_valid,
                        acquisition_invalid,
                    )
                    has_double_history = (
                        unobservable_single
                        & np.isfinite(previous_weights)
                        & (previous_weights > 0.0)
                        & np.isfinite(previous_contrast)
                        & (previous_contrast > 0.0)
                    )
                    new_history_branch = (
                        has_double_history & (probe_direction_sign == 0.0)
                    )
                    history_direction = np.zeros(self._site_count, dtype=float)
                    history_direction[new_history_branch] = np.sign(np.log(
                        previous_weights[new_history_branch]
                        / current_control_weights[new_history_branch]
                    ))
                    new_history_branch &= history_direction != 0.0
                    probe_direction_sign[new_history_branch] = history_direction[
                        new_history_branch
                    ]
                    probe_single_bound[new_history_branch] = (
                        current_control_weights[new_history_branch]
                    )
                    probe_observable_bound[new_history_branch] = previous_weights[
                        new_history_branch
                    ]
                    (
                        probe_single_bound,
                        probe_observable_bound,
                        probe_direction_sign,
                        _crossed_bracket,
                    ) = _updated_single_bracket(
                        current_control_weights,
                        unobservable_single,
                        probe_single_bound,
                        probe_observable_bound,
                        probe_direction_sign,
                    )
                    up = probe_direction_sign > 0.0
                    down = probe_direction_sign < 0.0
                    probe_observable_bound[observable_valid & up] = np.fmin(
                        probe_observable_bound[observable_valid & up], current_control_weights[observable_valid & up]
                    )
                    probe_observable_bound[observable_valid & down] = np.fmax(
                        probe_observable_bound[observable_valid & down], current_control_weights[observable_valid & down]
                    )
                    probe_direction_log_step[observable_valid] = 0.0
                    probe_control_boundary[:] = _probe_boundary(
                        probe_single_bound,
                        probe_observable_bound,
                        probe_direction_sign,
                    )
                    midpoint_step = _single_bracket_step(
                        current_control_weights,
                        probe_control_boundary,
                        probe_direction_sign,
                        self.maximum_weight_change,
                    )
                    bracket_recovery = unobservable_single & (midpoint_step != 0.0)
                    probe_direction_log_step[bracket_recovery] = midpoint_step[
                        bracket_recovery
                    ]
                    relative_move = np.full(
                        self._site_count, np.inf, dtype=float
                    )
                    relative_move[has_double_history] = np.abs(np.log(
                        current_control_weights[has_double_history]
                        / previous_weights[has_double_history]
                    ))
                    stored_probe_step = np.log(probe_selected_factors)
                    has_probe_direction = (
                        np.isfinite(stored_probe_step)
                        & (stored_probe_step != 0.0)
                    )
                    reuse_probe_direction = (
                        unobservable_single
                        & has_probe_direction
                        & (
                            ~bracket_recovery
                            | (
                                has_double_history
                                & (relative_move < 0.02)
                            )
                        )
                    )
                    probe_direction_log_step[reuse_probe_direction] = np.clip(
                        stored_probe_step[reuse_probe_direction],
                        -np.log1p(self.maximum_weight_change),
                        np.log1p(self.maximum_weight_change),
                    )
                    probe_direction_sign[reuse_probe_direction] = np.sign(
                        stored_probe_step[reuse_probe_direction]
                    )
                    recovery_probe = (
                        has_double_history
                        & ~_crossed_bracket
                        & ~has_probe_direction
                        & (
                            (relative_move < 0.02)
                            | (probe_direction_sign == 0.0)
                            | (
                                np.isfinite(probe_observable_bound)
                                & ~np.isfinite(probe_control_boundary)
                            )
                        )
                    )
                    episode_probe_sites = (
                        (
                            (needs_probe_sites & ~has_probe_direction)
                            | recovery_probe
                        )
                        & ~probe_episode_used
                        & (formal_updates < self.max_updates)
                    )
                    starts_probe_episode = bool(np.any(episode_probe_sites))
                    if starts_probe_episode:
                        probe_direction_log_step[episode_probe_sites] = 0.0
                        probe_direction_sign[episode_probe_sites] = 0.0
                        probe_single_bound[episode_probe_sites] = np.nan
                        probe_observable_bound[episode_probe_sites] = np.nan
                        probe_control_boundary[episode_probe_sites] = np.nan
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
                    probe_baseline_previous_weights[:] = previous_weights
                    probe_baseline_previous_contrast[:] = previous_contrast
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
                    response_slope = np.full(
                        self._site_count, np.nan, dtype=float
                    )
                    decisions = np.full(
                        self._site_count, "hold_for_probe", dtype="<U32"
                    )
                else:
                    directed_step = np.array(
                        probe_direction_log_step, copy=True
                    )
                    directed_step[acquisition_invalid] = 0.0
                    allocation_boundary = np.array(
                        probe_control_boundary, copy=True
                    )
                    proposed_target, log_correction, response_slope, decisions = (
                        _updated_target(
                            current_target,
                            contrast,
                            error,
                            feedback_valid,
                            self._rows,
                            self._columns,
                            reference_valid=fit_valid,
                            previous_weights=previous_weights,
                            previous_contrast=previous_contrast,
                            feedback_gain=double_feedback_gain,
                            maximum_weight_change=self.maximum_weight_change,
                            directed_log_step=directed_step,
                            control_boundary=allocation_boundary,
                            control_direction=probe_direction_sign,
                        )
                    )
                    decisions[bracket_recovery] = "single_bracket_midpoint"
                    if starts_probe_episode:
                        decisions[probe_sites] = "hold_for_probe"
                        probe_baseline_directed_step[:] = directed_step
                        probe_baseline_control_boundary[:] = allocation_boundary
                        probe_baseline_control_direction[:] = (
                            probe_direction_sign
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
                    "authored_feedback_gain": self.feedback_gain,
                    "double_feedback_gain": double_feedback_gain,
                    "adaptive_gain_action": adaptive_gain_action,
                    "adaptive_gain_common_sites": adaptive_gain_common_sites,
                    "adaptive_gain_previous_ratio": adaptive_gain_previous_ratio,
                    "adaptive_gain_current_ratio": adaptive_gain_current_ratio,
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
                    "bic_gain": _json_floats(fitted["bic_gain"]),
                    "bic_gain_even": _json_floats(fitted["bic_gain_even"]),
                    "bic_gain_odd": _json_floats(fitted["bic_gain_odd"]),
                    "bic_stable": [bool(value) for value in fitted["bic_stable"]],
                    "fit_valid": [bool(value) for value in fit_valid],
                    "observable_valid": [
                        bool(value) for value in observable_valid
                    ],
                    "single_population": [bool(value) for value in fit_single],
                    "fit_invalid": [bool(value) for value in fit_invalid],
                    "single_mean": _json_floats(fitted["single_mean"]),
                    "single_sigma": _json_floats(fitted["single_sigma"]),
                    "decision": [str(value) for value in decisions],
                    "requested_log_correction": _json_floats(log_correction),
                    "response_log_slope": _json_floats(response_slope),
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
                    "single_population_sites": [
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
                if candidate_kind != "probe" and (
                    most_visible_observed is None
                    or completed["visibility_rank"]
                    >= most_visible_observed["visibility_rank"]
                ):
                    most_visible_observed = completed
                if valid and candidate_kind != "probe":
                    if (
                        retained_valid is None
                        or float(completed["score"]) < float(retained_valid["score"])
                    ):
                        retained_valid = completed
                    context.report_progress(
                        f"qCMOS bright-dark ratio {score:.5f}; "
                        f"simultaneous 95% upper {confidence_upper:.5f}"
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
                        (
                            _diagnostic_target,
                            selected_factors,
                            selected_decisions,
                            probe_observed_factors,
                        ) = _selected_probe_target(
                            probe_baseline_target,
                            probe_sites,
                            probe_baseline_contrast,
                            probe_baseline_valid,
                            probe_measurements,
                            self._rows,
                            self._columns,
                            feedback_gain=self.feedback_gain,
                            maximum_weight_change=self.maximum_weight_change,
                        )
                        probe_direction_log_step[probe_sites] = 0.0
                        probe_selected_factors[probe_sites] = selected_factors[
                            probe_sites
                        ]
                        probe_decisions[probe_sites] = selected_decisions[probe_sites]
                        probe_direction_log_step[probe_sites] = np.log(
                            probe_selected_factors[probe_sites]
                        )
                        probe_direction_sign[probe_sites] = np.sign(
                            np.log(probe_observed_factors[probe_sites])
                        )
                        baseline_control = _control_weights(
                            probe_baseline_target[self._rows, self._columns]
                        )
                        directed_probe = probe_sites & (
                            probe_direction_sign != 0.0
                        )
                        probe_single_bound[directed_probe] = baseline_control[
                            directed_probe
                        ]
                        combined_directed_step = np.array(
                            probe_baseline_directed_step, copy=True
                        )
                        combined_directed_step[probe_sites] = (
                            probe_direction_log_step[probe_sites]
                        )
                        combined_target, *_formal_details = _updated_target(
                            probe_baseline_target,
                            probe_baseline_contrast,
                            probe_baseline_error,
                            probe_baseline_valid,
                            self._rows,
                            self._columns,
                            reference_valid=probe_baseline_reference_valid,
                            previous_weights=probe_baseline_previous_weights,
                            previous_contrast=probe_baseline_previous_contrast,
                            feedback_gain=double_feedback_gain,
                            maximum_weight_change=self.maximum_weight_change,
                            directed_log_step=combined_directed_step,
                            control_boundary=(
                                probe_baseline_control_boundary
                            ),
                            control_direction=(
                                probe_baseline_control_direction
                            ),
                        )
                        if np.array_equal(
                            combined_target, probe_baseline_target
                        ):
                            stalled = True
                            continue_feedback = False
                            history[-1]["next_phase_changed"] = False
                            termination_reason = (
                                "two-sided SLM probes supplied no observable "
                                "direction; baseline restored"
                            )
                            forced_terminal_candidate = probe_baseline_candidate
                        else:
                            next_target = combined_target
                            next_kind = "probe_combined"
                            formal_updates += 1
                else:
                    if formal_updates >= self.max_updates:
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
                            next_target,
                            pupil_amplitude=self._pupil_amplitude,
                            initial_phase=(
                                probe_baseline_pattern
                                if probe_solve else current_pattern
                            ),
                            objective_kind="spots",
                            iterations=None,
                            stop_requested=context.cancel_requested,
                            spot_optimizer_state=solve_state,
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
    "SITE_SIGNAL_HISTORY_OUTPUT",
    "SLM_PHASE_ARTIFACT_CONTRACT",
    "SlmFeedbackTask",
    "TARGET_SHARE_HISTORY_OUTPUT",
    "UNIFORMITY_HISTORY_OUTPUT",
]
