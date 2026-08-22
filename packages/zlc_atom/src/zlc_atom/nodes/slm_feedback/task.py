"""Single-frame, multi-shot bright-dark fluorescence feedback."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Mapping

import numpy as np
from scipy import special
from zlc_data import (
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    PointColumn,
)
from zlc_data.snapshot_projection import (
    restricted_values,
    selection_indices,
    value_selection,
)
from zlc_durable import unique_path
from zlc_pulse import PulseSequence
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
    "candidate_phase", "slm-feedback.candidate-phase.v1"
)
UNIFORMITY_HISTORY_OUTPUT = DatasetOutputDeclaration(
    "uniformity_history", "slm-feedback.uniformity-history.v1"
)
_TARGET_RATIO = 1.10
_FEEDBACK_EXPONENT = 0.25
_BOOTSTRAP_LOG_STEP = float(np.log(1.4))
_MAX_FEEDBACK_LOG_STEP = float(np.log(1.5))
_COARSE_MAX_BATCHES = 3
_MAX_BOOTSTRAP_UPDATES = 3
_VALIDATION_BATCH_SHOTS = 100
_VALIDATION_MAX_SECONDS = 300.0
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
    *,
    looks: int = 1,
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
        or type(looks) is not int
        or looks < 1
    ):
        raise ValueError("bright-dark contrast and uncertainty must be finite and positive")
    relative = error / measured
    z = float(
        special.ndtri(1.0 - 0.05 / (2.0 * len(measured) * int(looks)))
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


def _fit_contrasts(samples: object, *, looks: int = 1) -> dict[str, np.ndarray]:
    """Fit this run's two shot populations without Calibration statistics."""

    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 1:
        raise ValueError("feedback box samples must have shape (shots, sites)")
    if type(looks) is not int or looks < 1:
        raise ValueError("feedback looks must be a positive integer")
    sites = values.shape[1]
    contrast = np.full(sites, np.nan, dtype=float)
    error = np.full(sites, np.nan, dtype=float)
    dark_mean = np.full(sites, np.nan, dtype=float)
    bright_mean = np.full(sites, np.nan, dtype=float)
    bright_fraction = np.full(sites, np.nan, dtype=float)
    fidelity = np.full(sites, np.nan, dtype=float)
    bic_gain = np.full(sites, np.nan, dtype=float)
    separated = np.zeros(sites, dtype=bool)
    uncertain = np.zeros(sites, dtype=bool)
    z = float(special.ndtri(1.0 - 0.05 / (2.0 * sites * int(looks))))
    for site in range(sites):
        column = values[:, site]
        column = column[np.isfinite(column)]
        if column.size < 4:
            continue
        fit = fit_bimodal(column, min_component_fraction=0.01)
        dark_mean[site] = fit.dark_mean
        bright_mean[site] = fit.bright_mean
        bright_fraction[site] = fit.bright_fraction
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
        estimate = float(fit.bright_mean - fit.dark_mean)
        sem = float(
            np.sqrt(
                fit.bright_sigma**2 / count_bright
                + fit.dark_sigma**2 / count_dark
            )
        )
        contrast[site], error[site] = estimate, sem
        sigma_one = max(float(np.std(column)), np.finfo(float).tiny)
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
        credible_pair = bool(
            fit.ok
            and fit.bright_above
            and estimate > 0.0
            and np.isfinite(sem)
            and sem >= 0.0
            and bic_gain[site] > 10.0
        )
        separated[site] = credible_pair and estimate > z * sem
        uncertain[site] = credible_pair and not separated[site]
    return {
        "contrast": contrast,
        "standard_error": error,
        "dark_mean": dark_mean,
        "bright_mean": bright_mean,
        "bright_fraction": bright_fraction,
        "fidelity": fidelity,
        "bic_gain": bic_gain,
        "valid": separated,
        "uncertain": uncertain,
        "censored": ~(separated | uncertain),
    }


def _json_floats(values: object) -> list[float | None]:
    """Keep strict JSON artifacts readable when a rejected shot was missing."""

    return [float(value) if np.isfinite(value) else None for value in np.asarray(values)]


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
    censored: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    previous_weights: np.ndarray,
    previous_contrast: np.ndarray,
    bootstrap_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update one Target using both the current fit and each site's history."""

    values = np.asarray(contrast, dtype=float)
    errors = np.asarray(standard_error, dtype=float)
    fit_valid = np.asarray(valid, dtype=bool)
    fit_censored = np.asarray(censored, dtype=bool)
    prior_weights = np.asarray(previous_weights, dtype=float)
    prior_values = np.asarray(previous_contrast, dtype=float)
    bootstraps = np.asarray(bootstrap_counts, dtype=int)
    site_shape = (len(rows),)
    if (
        values.shape != site_shape
        or errors.shape != site_shape
        or fit_valid.shape != site_shape
        or fit_censored.shape != site_shape
        or prior_weights.shape != site_shape
        or prior_values.shape != site_shape
        or bootstraps.shape != site_shape
        or np.any(fit_valid & fit_censored)
    ):
        raise ValueError("feedback site history shapes differ")
    if not np.any(fit_valid):
        reference = float("nan")
    else:
        if np.any(~np.isfinite(values[fit_valid]) | (values[fit_valid] <= 0.0)):
            raise ValueError("valid feedback contrasts must be finite and positive")
        reference = float(np.exp(np.mean(np.log(values[fit_valid]))))
    current_weights = np.asarray(target[rows, columns], dtype=float)
    log_correction = np.zeros(site_shape, dtype=float)
    response_slope = np.full(site_shape, np.nan, dtype=float)
    decision = np.full(site_shape, "hold_uncertain", dtype="<U32")
    for site in range(len(rows)):
        if fit_valid[site]:
            residual = float(np.log(values[site] / reference))
            relative_error = max(float(errors[site]), 0.0) / values[site]
            quality = float(np.clip(1.0 - 4.0 * relative_error, 0.1, 1.0))
            correction = _FEEDBACK_EXPONENT * quality * residual
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
                        correction = -_FEEDBACK_EXPONENT * quality * residual / slope
                        decision[site] = "feedback_history_slope"
                    else:
                        decision[site] = "feedback_assumed_slope"
                else:
                    decision[site] = "feedback_assumed_slope"
            else:
                decision[site] = "feedback_assumed_slope"
            log_correction[site] = float(
                np.clip(correction, -_MAX_FEEDBACK_LOG_STEP, _MAX_FEEDBACK_LOG_STEP)
            )
        elif fit_censored[site] and not np.isfinite(prior_values[site]):
            if bootstraps[site] < _MAX_BOOTSTRAP_UPDATES:
                log_correction[site] = _BOOTSTRAP_LOG_STEP
                decision[site] = "bootstrap_shallow"
            else:
                decision[site] = "hold_bootstrap_limit"
        elif (
            fit_censored[site]
            and np.isfinite(prior_weights[site])
            and prior_weights[site] > 0.0
            and current_weights[site] > 1.01 * prior_weights[site]
        ):
            log_correction[site] = float(
                np.clip(
                    np.log(prior_weights[site] / current_weights[site]),
                    -_MAX_FEEDBACK_LOG_STEP,
                    0.0,
                )
            )
            decision[site] = "rollback_after_disappearance"
    updated = np.array(target, dtype=np.float32, copy=True)
    updated[rows, columns] *= np.exp(log_correction).astype(np.float32)
    updated *= float(np.sum(target)) / float(np.sum(updated))
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
        validation_shots: int,
        max_updates: int,
        artifact_directory: str | Path,
    ) -> None:
        if not isinstance(slm, SlmAdapter):
            raise TypeError("slm must implement SlmAdapter")
        if not isinstance(calibration, TrapCalibration) or not isinstance(pulse_sequence, PulseSequence):
            raise TypeError("feedback requires TrapCalibration and PulseSequence")
        directory = Path(artifact_directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("artifact_directory must be an existing directory")
        if not isinstance(science_context, Mapping):
            raise TypeError("science_context must be a loaded Science Context mapping")
        if science_context.get("objective_kind") != "spots":
            raise ValueError("SLM feedback Science Context must use the spots objective")
        context_target = science_context.get("target_intensity")
        if context_target is None:
            raise ValueError(
                "legacy Science Context has no frozen Target; load it in the "
                "SLM Editor with the intended Target and resave"
            )
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
        self._pattern_metadata = dict(science_context.get("pattern_metadata", {}))
        self._operator_metadata = dict(science_context.get("operator_metadata", {}))
        self._mapping_revision = int(slm.mapping_revision)
        self.feedback_mode = str(feedback_mode)
        if self.feedback_mode != "qcmos_bright_dark":
            raise ValueError("unsupported SLM feedback mode")
        self.exposure_seconds = float(exposure_seconds)
        if not np.isfinite(self.exposure_seconds) or self.exposure_seconds <= 0.0:
            raise ValueError("feedback exposure_seconds must be finite and positive")
        self.shots = int(shots_per_candidate)
        self.validation_shots = int(validation_shots)
        self.max_updates = int(max_updates)
        self._candidate_capacity = self.max_updates + 1
        self._publication_revision = 0
        self._actual_exposure_seconds: float | None = None
        self._effective_photoelectrons: bool | None = None
        self._effective_count_unit: str | None = None
        if self.shots < 10 or self.validation_shots < 10 or self.max_updates < 1:
            raise ValueError("feedback needs at least 10 coarse/validation shots and one update")
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
        self.artifact_directory = directory

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return (
            CANDIDATE_PHASE_OUTPUT,
            UNIFORMITY_HISTORY_OUTPUT,
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
            "target_uniformity_ratio": _TARGET_RATIO,
            "feedback_mode": self.feedback_mode,
            "exposure_seconds": self.exposure_seconds,
            "actual_exposure_seconds": self._actual_exposure_seconds,
            "effective_photoelectrons": self._effective_photoelectrons,
            "effective_count_unit": self._effective_count_unit,
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
            "target_uniformity_ratio": _TARGET_RATIO,
            "feedback_mode": self.feedback_mode,
            "exposure_seconds": self.exposure_seconds,
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
        for item in history:
            validation = item.get("validation")
            ratio = (
                validation.get("uniformity_ratio")
                if isinstance(validation, Mapping)
                else item.get("uniformity_ratio")
            )
            if ratio is not None:
                curve[int(item["iteration"]) - 1] = float(ratio)
        history_event = snapshot_from_array(
            curve[None],
            producer=self.instance_id,
            signal=UNIFORMITY_HISTORY_OUTPUT.name,
            roles=(SCAN_POINT,),
            point_columns={
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
            },
            generation=generation,
            revision=publication_revision,
            validity=np.isfinite(curve)[None],
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
            }
        )

    def _incoming_candidate(
        self,
        *,
        stem: str,
        status: str,
        history: list[dict[str, object]],
        observed: Mapping[str, object] | None,
        phase: np.ndarray,
        pattern: np.ndarray,
    ) -> dict[str, object]:
        candidate = 1 if observed is None else int(observed["candidate"])
        metadata = self._candidate_metadata(
            candidate=0,
            status=status,
            history=history,
        )
        artifact_path = unique_path(
            self.artifact_directory,
            stem,
            ".npz",
            writer=lambda temporary: self._save_candidate(
                temporary,
                phase,
                pattern,
                self.target,
                metadata,
            ),
        )
        return {
            "candidate": candidate,
            "artifact_candidate": 0,
            "phase": np.array(phase, copy=True),
            "pattern_phase": np.array(pattern, copy=True),
            "artifact_path": artifact_path,
            "target": np.array(self.target, copy=True),
            "solver": None,
            "history": None if observed is None else observed["history"],
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
        *,
        shots: int | None = None,
    ) -> tuple[
        np.ndarray,
        tuple[int, ...],
        tuple[int, ...],
    ]:
        contract = self.calibration.frame_contract
        requested = self.shots if shots is None else int(shots)
        if requested < 4:
            raise ValueError("qCMOS bright-dark statistics require at least four shots")
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
        )

    def _coarse_measure(
        self,
        pulse: object,
        context: object,
        iteration: int,
    ) -> tuple[
        dict[str, np.ndarray],
        tuple[int, ...],
        tuple[int, ...],
        int,
        int,
    ]:
        for attempt in range(2):
            batches: list[np.ndarray] = []
            saturated: tuple[int, ...] = ()
            missing: tuple[int, ...] = ()
            for batch in range(_COARSE_MAX_BATCHES):
                samples, saturated, missing = self._measure(
                    pulse,
                    context,
                    iteration,
                    shots=None if batch == 0 else self.shots,
                )
                if saturated or missing:
                    break
                batches.append(np.asarray(samples, dtype=float))
                fitted = _fit_contrasts(
                    np.concatenate(batches, axis=0),
                    looks=_COARSE_MAX_BATCHES,
                )
                if bool(np.all(fitted["valid"])):
                    return (
                        fitted,
                        (),
                        (),
                        attempt + 1,
                        sum(item.shape[0] for item in batches),
                    )
            if not saturated and not missing and batches:
                fitted = _fit_contrasts(
                    np.concatenate(batches, axis=0),
                    looks=_COARSE_MAX_BATCHES,
                )
                return (
                    fitted,
                    (),
                    (),
                    attempt + 1,
                    sum(item.shape[0] for item in batches),
                )
            if attempt == 0:
                context.report_progress(
                    f"Candidate {iteration + 1} camera samples invalid; "
                    "retrying the same applied phase"
                )
        invalid = _fit_contrasts(
            np.full((4, self._site_count), np.nan),
            looks=_COARSE_MAX_BATCHES,
        )
        return invalid, saturated, missing, 2, 0

    def _finish_candidate(
        self,
        context: object,
        candidate: dict[str, object],
        history: list[dict[str, object]],
        *,
        status: str,
        republish: bool,
    ) -> dict[str, object]:
        applied = self._apply_exact(candidate["phase"])
        candidate_number = int(candidate["candidate"])
        if republish:
            self._publish_candidate(
                context,
                phase=applied,
                candidate=candidate_number,
                history=history,
            )
        artifact_path = Path(candidate["artifact_path"])
        metadata = self._candidate_metadata(
            candidate=int(candidate.get("artifact_candidate", candidate_number)),
            status=status,
            history=history,
            solver=candidate.get("solver"),
        )
        metadata["retained"] = candidate.get("history")
        metadata["history"] = history
        self._save_candidate(
            artifact_path,
            applied,
            candidate["pattern_phase"],
            candidate["target"],
            metadata,
        )
        terminal_uniformity = candidate.get("validation_score")
        if terminal_uniformity is None and isinstance(candidate.get("history"), Mapping):
            terminal_uniformity = candidate["history"].get("uniformity_ratio")
        return {
            "artifact_path": artifact_path,
            "terminal_uniformity": terminal_uniformity,
            "validation_status": candidate.get("validation_status"),
            "validation_confidence_lower": candidate.get(
                "validation_confidence_lower"
            ),
            "validation_confidence_upper": candidate.get(
                "validation_confidence_upper"
            ),
            "validation_max_relative_standard_error": candidate.get(
                "validation_max_relative_standard_error"
            ),
            "updates": len(history),
            "target_uniformity_ratio": _TARGET_RATIO,
            "feedback_mode": self.feedback_mode,
            "requested_exposure_seconds": self.exposure_seconds,
            "actual_exposure_seconds": self._actual_exposure_seconds,
        }

    def execute(self, context: object) -> dict[str, object]:
        incoming = self._incoming_phase
        incoming_pattern = self._pattern_phase
        history: list[dict[str, object]] = []
        retained_valid: dict[str, object] | None = None
        most_visible_observed: dict[str, object] | None = None
        try:
            _check_cancelled(context)
            # Science Context is the requested starting CONTENT, not proof of
            # what a previous process happens to have commanded. This Task owns
            # the SLM now, so establish and confirm that starting state itself.
            self._mapping_revision = int(self.slm.mapping_revision)
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
            bootstrap_counts = np.zeros(self._site_count, dtype=int)
            prior_history = self._pattern_metadata.get("history")
            comparable_history = bool(
                self._pattern_metadata.get("feedback_mode") == self.feedback_mode
                and self._pattern_metadata.get("pulse_path") == str(self.pulse_path)
                and type(self._pattern_metadata.get("exposure_seconds"))
                in (int, float)
                and float(self._pattern_metadata["exposure_seconds"])
                == self.exposure_seconds
                and isinstance(prior_history, list)
                and prior_history
                and isinstance(prior_history[-1], Mapping)
            )
            if comparable_history:
                prior = prior_history[-1]
                try:
                    previous_weights = np.asarray(
                        prior["previous_valid_weight"], dtype=float
                    ).reshape(self._site_count)
                    previous_contrast = np.asarray(
                        prior["previous_valid_bright_minus_dark"], dtype=float
                    ).reshape(self._site_count)
                    bootstrap_counts = np.asarray(
                        prior["bootstrap_count"], dtype=int
                    ).reshape(self._site_count)
                    decisions = np.asarray(prior["decision"], dtype=str).reshape(
                        self._site_count
                    )
                    prior_weights = np.asarray(
                        prior["target_weight"], dtype=float
                    ).reshape(self._site_count)
                    prior_contrast = np.asarray(
                        prior["bright_minus_dark"], dtype=float
                    ).reshape(self._site_count)
                    prior_valid = np.asarray(
                        prior["fit_valid"], dtype=bool
                    ).reshape(self._site_count)
                except (KeyError, TypeError, ValueError):
                    previous_weights[:] = np.nan
                    previous_contrast[:] = np.nan
                    bootstrap_counts[:] = 0
                else:
                    usable = (
                        prior_valid
                        & np.isfinite(prior_weights)
                        & np.isfinite(prior_contrast)
                    )
                    previous_weights[usable] = prior_weights[usable]
                    previous_contrast[usable] = prior_contrast[usable]
                    bootstrap_counts += decisions == "bootstrap_shallow"
                    bootstrap_counts = np.maximum(bootstrap_counts, 0)
            for iteration in range(self._candidate_capacity):
                _check_cancelled(context)
                applied = (
                    current_phase
                    if iteration == 0
                    else self._apply_exact(current_phase)
                )
                candidate_number = iteration + 1
                applied_metadata = self._candidate_metadata(
                    candidate=candidate_number,
                    status="applied",
                    history=history,
                    solver=solver_metadata,
                )
                artifact_path = unique_path(
                    self.artifact_directory,
                    f"slm_feedback_candidate_{candidate_number:04d}",
                    ".npz",
                    writer=lambda temporary: self._save_candidate(
                        temporary,
                        applied,
                        current_pattern,
                        current_target,
                        applied_metadata,
                    ),
                )
                context.report_progress(
                    f"Candidate {candidate_number} phase saved to {artifact_path}"
                )
                (
                    fitted,
                    saturated_sites,
                    missing_sites,
                    attempts,
                    coarse_shots,
                ) = self._coarse_measure(pulse, context, iteration)
                contrast = fitted["contrast"]
                error = fitted["standard_error"]
                fit_valid = fitted["valid"]
                fit_uncertain = fitted["uncertain"]
                fit_censored = fitted["censored"]
                saturated = bool(saturated_sites)
                missing = bool(missing_sites)
                valid = bool(not saturated and not missing and np.all(fit_valid))
                observed_score = (
                    float(np.max(contrast) / np.min(contrast))
                    if valid
                    else float("nan")
                )
                if valid:
                    score, confidence_lower, confidence_upper, relative_sem = (
                        _ratio_interval(
                            contrast,
                            error,
                            looks=_COARSE_MAX_BATCHES,
                        )
                    )
                else:
                    score = confidence_lower = confidence_upper = relative_sem = float("inf")
                visibility = int(np.count_nonzero(fit_valid))
                visible_margin = contrast[fit_valid] - float(
                    special.ndtri(
                        1.0
                        - 0.05
                        / (2.0 * self._site_count * _COARSE_MAX_BATCHES)
                    )
                ) * error[fit_valid]
                visibility_margin = (
                    None
                    if saturated or missing or not len(visible_margin)
                    else float(np.min(visible_margin))
                )
                proposed_target, log_correction, response_slope, decisions = (
                    _updated_target(
                        current_target,
                        contrast,
                        error,
                        fit_valid,
                        fit_censored,
                        self._rows,
                        self._columns,
                        previous_weights=previous_weights,
                        previous_contrast=previous_contrast,
                        bootstrap_counts=bootstrap_counts,
                    )
                )
                if valid and score <= _TARGET_RATIO:
                    log_correction[:] = 0.0
                    decisions[:] = "converged"
                    proposed_target = current_target
                current_weights = np.asarray(
                    current_target[self._rows, self._columns], dtype=float
                )
                history.append({
                    "iteration": candidate_number,
                    "shots": coarse_shots,
                    "attempts": attempts,
                    "valid": valid,
                    "saturated": saturated,
                    "saturated_sites": list(saturated_sites),
                    "uniformity_ratio": (
                        None if not np.isfinite(observed_score) else observed_score
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
                    "feedback_exponent": _FEEDBACK_EXPONENT,
                    "feedback_mode": self.feedback_mode,
                    "requested_exposure_seconds": self.exposure_seconds,
                    "actual_exposure_seconds": self._actual_exposure_seconds,
                    "effective_photoelectrons": self._effective_photoelectrons,
                    "effective_count_unit": self._effective_count_unit,
                    "target_weight": _json_floats(current_weights),
                    "dark_mean": _json_floats(fitted["dark_mean"]),
                    "bright_mean": _json_floats(fitted["bright_mean"]),
                    "bright_minus_dark": _json_floats(contrast),
                    "contrast_standard_error": _json_floats(error),
                    "bright_fraction": _json_floats(fitted["bright_fraction"]),
                    "fit_fidelity": _json_floats(fitted["fidelity"]),
                    "fit_bic_gain": _json_floats(fitted["bic_gain"]),
                    "fit_valid": [bool(value) for value in fit_valid],
                    "fit_uncertain": [bool(value) for value in fit_uncertain],
                    "decision": [str(value) for value in decisions],
                    "requested_log_correction": _json_floats(log_correction),
                    "response_log_slope": _json_floats(response_slope),
                    "previous_valid_weight": _json_floats(previous_weights),
                    "previous_valid_bright_minus_dark": _json_floats(
                        previous_contrast
                    ),
                    "bootstrap_count": [int(value) for value in bootstrap_counts],
                    "missing_sites": list(missing_sites),
                    "censored_sites": [
                        int(value) for value in np.flatnonzero(fit_censored)
                    ],
                    "uncertain_sites": [
                        int(value) for value in np.flatnonzero(fit_uncertain)
                    ],
                    "minimum_visibility_confidence": visibility_margin,
                    "artifact_path": str(artifact_path),
                })
                self._save_candidate(
                    artifact_path,
                    applied,
                    current_pattern,
                    current_target,
                    self._candidate_metadata(
                        candidate=candidate_number,
                        status="measured",
                        history=history,
                        solver=solver_metadata,
                    ),
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
                    "artifact_path": artifact_path,
                    "target": np.array(current_target, copy=True),
                    "solver": solver_metadata,
                    "history": history[-1],
                    "score": observed_score,
                    "contrast": np.array(contrast, copy=True),
                    "standard_error": np.array(error, copy=True),
                }
                if saturated or missing:
                    raise RuntimeError(
                        "qCMOS site samples remained invalid after two measurements "
                        "of the same candidate"
                    )
                completed["visibility_rank"] = (
                    visibility,
                    float("-inf") if visibility_margin is None else visibility_margin,
                )
                if (
                    most_visible_observed is None
                    or completed["visibility_rank"]
                    > most_visible_observed["visibility_rank"]
                ):
                    most_visible_observed = completed
                if valid:
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
                if score <= _TARGET_RATIO:
                    break
                if candidate_number == self._candidate_capacity:
                    break
                visible_ratio = (
                    float(np.max(contrast[fit_valid]) / np.min(contrast[fit_valid]))
                    if np.any(fit_valid)
                    else float("inf")
                )
                site_action = np.isin(
                    decisions,
                    ("bootstrap_shallow", "rollback_after_disappearance"),
                )
                if not valid and visible_ratio <= _TARGET_RATIO and not np.any(site_action):
                    break
                previous_weights[fit_valid] = current_weights[fit_valid]
                previous_contrast[fit_valid] = contrast[fit_valid]
                bootstrap_counts[
                    np.asarray(decisions) == "bootstrap_shallow"
                ] += 1
                if np.allclose(proposed_target, current_target, rtol=0.0, atol=0.0):
                    break
                current_target = proposed_target
                current_pattern, solver_metadata = solve_phase(
                    current_target,
                    pupil_amplitude=self._pupil_amplitude,
                    initial_phase=current_pattern,
                    objective_kind="spots",
                    iterations=None,
                    stop_requested=context.cancel_requested,
                    spot_optimizer_state=spot_optimizer_state,
                )
                current_phase = self._science_phase(current_pattern)
            if retained_valid is None:
                if most_visible_observed is None:
                    raise RuntimeError("qCMOS feedback produced no observable candidate")
                most_visible_observed["history"]["validation"] = {
                    "status": "inconclusive",
                    "reason": "censored sites remained after bounded bootstrap",
                    "shots": 0,
                    "maximum_shots": self.validation_shots,
                    "maximum_seconds": _VALIDATION_MAX_SECONDS,
                    "maximum_looks": (
                        self.validation_shots + _VALIDATION_BATCH_SHOTS - 1
                    ) // _VALIDATION_BATCH_SHOTS,
                    "confidence_family_alpha": 0.05,
                    "elapsed_seconds": 0.0,
                    "uniformity_ratio": None,
                    "uniformity_confidence_lower": None,
                    "uniformity_confidence_upper": None,
                    "maximum_relative_standard_error": None,
                    "bright_minus_dark": _json_floats(
                        most_visible_observed["contrast"]
                    ),
                    "contrast_standard_error": _json_floats(
                        most_visible_observed["standard_error"]
                    ),
                    "censored_sites": most_visible_observed["history"][
                        "censored_sites"
                    ],
                }
                inconclusive = {
                    **self._incoming_candidate(
                        stem="slm_feedback_inconclusive",
                        status="inconclusive",
                        history=history,
                        observed=most_visible_observed,
                        phase=incoming,
                        pattern=incoming_pattern,
                    ),
                    "validation_status": "inconclusive",
                    "validation_score": None,
                    "validation_confidence_lower": None,
                    "validation_confidence_upper": None,
                    "validation_max_relative_standard_error": None,
                }
                context.seal_terminal()
                return self._finish_candidate(
                    context,
                    inconclusive,
                    history,
                    status="inconclusive",
                    republish=False,
                )
            _check_cancelled(context)
            self._apply_exact(retained_valid["phase"])
            validation_batches: list[np.ndarray] = []
            validation_count = 0
            validation_contrast = np.full(self._site_count, np.nan, dtype=float)
            validation_error = np.full(self._site_count, np.nan, dtype=float)
            validation_estimate = validation_lower = validation_upper = float("inf")
            validation_relative_sem = float("inf")
            validation_censored = tuple(range(self._site_count))
            validation_status = "inconclusive"
            validation_reason = "maximum validation shots reached"
            validation_started = time.monotonic()
            deadline = validation_started + _VALIDATION_MAX_SECONDS
            validation_max_looks = (
                self.validation_shots + _VALIDATION_BATCH_SHOTS - 1
            ) // _VALIDATION_BATCH_SHOTS
            while (
                validation_count < self.validation_shots
                and time.monotonic() < deadline
            ):
                _check_cancelled(context)
                remaining_shots = self.validation_shots - validation_count
                batch_shots = (
                    _VALIDATION_BATCH_SHOTS - 1
                    if remaining_shots == _VALIDATION_BATCH_SHOTS + 1
                    else min(_VALIDATION_BATCH_SHOTS, remaining_shots)
                )
                (
                    batch_samples,
                    saturated_sites,
                    missing_sites,
                ) = self._measure(
                    pulse,
                    context,
                    int(retained_valid["candidate"]) - 1,
                    shots=batch_shots,
                )
                if saturated_sites or missing_sites:
                    validation_reason = "independent validation data were invalid"
                    break
                validation_batches.append(np.asarray(batch_samples, dtype=float))
                validation_count += int(batch_samples.shape[0])
                validation_fitted = _fit_contrasts(
                    np.concatenate(validation_batches, axis=0),
                    looks=validation_max_looks,
                )
                validation_contrast = validation_fitted["contrast"]
                validation_error = validation_fitted["standard_error"]
                validation_censored = tuple(
                    int(value)
                    for value in np.flatnonzero(~validation_fitted["valid"])
                )
                if validation_censored:
                    validation_reason = (
                        "independent validation sites remained unresolved"
                    )
                    context.report_progress(
                        f"Independent validation {validation_count}/"
                        f"{self.validation_shots}: "
                        f"{len(validation_censored)} unresolved site fit(s)"
                    )
                    continue
                (
                    validation_estimate,
                    validation_lower,
                    validation_upper,
                    validation_relative_sem,
                ) = _ratio_interval(
                    validation_contrast,
                    validation_error,
                    looks=validation_max_looks,
                )
                context.report_progress(
                    f"Independent validation {validation_count}/"
                    f"{self.validation_shots}: bright-dark ratio "
                    f"{validation_estimate:.5f}, 95% upper "
                    f"{validation_upper:.5f}"
                )
                if validation_upper <= _TARGET_RATIO:
                    validation_status = "accepted"
                    validation_reason = "simultaneous confidence bound passed"
                    break
                if validation_lower > _TARGET_RATIO:
                    validation_reason = (
                        "simultaneous confidence bound excludes 1.10"
                    )
                    break
            if time.monotonic() >= deadline and validation_status != "accepted":
                validation_reason = "validation time budget reached"
            retained_valid["history"]["validation"] = {
                "status": validation_status,
                "reason": validation_reason,
                "shots": validation_count,
                "maximum_shots": self.validation_shots,
                "maximum_seconds": _VALIDATION_MAX_SECONDS,
                "maximum_looks": validation_max_looks,
                "confidence_family_alpha": 0.05,
                "elapsed_seconds": time.monotonic() - validation_started,
                "uniformity_ratio": (
                    None
                    if not np.isfinite(validation_estimate)
                    else validation_estimate
                ),
                "uniformity_confidence_lower": (
                    None if not np.isfinite(validation_lower) else validation_lower
                ),
                "uniformity_confidence_upper": (
                    None if not np.isfinite(validation_upper) else validation_upper
                ),
                "maximum_relative_standard_error": (
                    None
                    if not np.isfinite(validation_relative_sem)
                    else validation_relative_sem
                ),
                "bright_minus_dark": _json_floats(validation_contrast),
                "contrast_standard_error": _json_floats(validation_error),
                "censored_sites": list(validation_censored),
            }
            accepted = {
                **retained_valid,
                "history": retained_valid["history"],
                "validation_status": validation_status,
                "validation_score": (
                    None
                    if not np.isfinite(validation_estimate)
                    else validation_estimate
                ),
                "validation_confidence_lower": (
                    None if not np.isfinite(validation_lower) else validation_lower
                ),
                "validation_confidence_upper": (
                    None if not np.isfinite(validation_upper) else validation_upper
                ),
                "validation_max_relative_standard_error": (
                    None
                    if not np.isfinite(validation_relative_sem)
                    else validation_relative_sem
                ),
            }
            context.seal_terminal()
            return self._finish_candidate(
                context,
                accepted,
                history,
                status=validation_status,
                republish=True,
            )
        except BaseException as error:
            if context.cancel_requested():
                try:
                    context.seal_terminal(accept_stop=True)
                    retained = retained_valid
                    if retained is None:
                        retained = self._incoming_candidate(
                            stem="slm_feedback_stopped",
                            status="stopped-before-measurement",
                            history=history,
                            observed=most_visible_observed,
                            phase=incoming,
                            pattern=incoming_pattern,
                        )
                    return self._finish_candidate(
                        context,
                        retained,
                        history,
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
                raise RuntimeError(
                    "SLM feedback failed and the incoming phase could not be "
                    f"restored: {error}"
                ) from restore_error
            raise

__all__ = [
    "CANDIDATE_PHASE_OUTPUT",
    "READOUT_FRAME_COORDINATE",
    "SLM_PHASE_ARTIFACT_CONTRACT",
    "SlmFeedbackTask",
    "UNIFORMITY_HISTORY_OUTPUT",
]
