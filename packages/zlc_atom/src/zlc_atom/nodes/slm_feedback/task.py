"""Occupied-shot site-brightness feedback; no hidden-plant inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from zlc_durable import unique_path
from zlc_pulse import PulseSequence

from zlc_atom.devices.slm import SlmAdapter, canonical_phase
from zlc_atom.devices.slm.solver import save_phase, solve_phase, validate_target
from zlc_atom.nodes.calibration import ReadoutModelKind, TrapCalibration
from zlc_atom.nodes.calibration.calibration import reads_photoelectrons
from zlc_atom.nodes.calibration.pulse import arm_sequencer, resolve_pulse
from zlc_atom.nodes.camera_measurement.measurement import (
    CameraCycleSource,
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_atom.nodes.occupancy import OccupancyProcessor
from zlc_atom.nodes.scan.source import wait_for_board


SLM_PHASE_ARTIFACT_CONTRACT = "zlc.slm.phase.v1"
_TARGET_RATIO = 1.01
_VALIDATION_RELATIVE_SEM = 0.005
_SOLVE_ITERATIONS = 8
_FEEDBACK_EXPONENT = 0.45
_READOUT_FRAME = 1
_SHOT_CHUNK = 128


def _check_cancelled(context: object) -> None:
    if context.cancel_requested():
        raise RuntimeError("SLM feedback was cancelled")


def _ratio(values: np.ndarray) -> float:
    measured = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(measured)) or np.any(measured <= 0.0):
        return float("inf")
    return float(np.max(measured) / np.min(measured))


def _json_floats(values: object) -> list[float | None]:
    """Keep strict JSON artifacts readable when a rejected shot was missing."""

    return [float(value) if np.isfinite(value) else None for value in np.asarray(values)]


def _support(target: np.ndarray, calibration: TrapCalibration) -> tuple[np.ndarray, np.ndarray]:
    """The only unambiguous first feedback geometry: a complete 5 x 7 grid."""

    rows, columns = np.nonzero(target > 0.0)
    unique_rows, unique_columns = np.unique(rows), np.unique(columns)
    if (
        calibration.n_sites != 35
        or len(rows) != 35
        or len(unique_rows) != 5
        or len(unique_columns) != 7
        or set(zip(rows, columns, strict=True))
        != set((int(row), int(column)) for row in unique_rows for column in unique_columns)
    ):
        raise ValueError("SLM feedback requires one complete 5 x 7 sparse target aligned to 35 calibrated sites")
    centers = calibration.site_map.centers_xy.reshape(5, 7, 2)
    if not (
        np.all(np.diff(centers[:, :, 0], axis=1) > 0.0)
        and np.all(np.diff(np.mean(centers[:, :, 1], axis=1)) > 0.0)
    ):
        raise ValueError("calibration sites do not declare a stable 5 x 7 row-major order")
    return rows, columns


def _updated_target(target: np.ndarray, fluorescence: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    values = np.asarray(fluorescence, dtype=float)
    if values.shape != (len(rows),) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("feedback fluorescence must be finite and positive at every site")
    geometric_mean = float(np.exp(np.mean(np.log(values))))
    updated = np.array(target, dtype=np.float32, copy=True)
    updated[rows, columns] *= np.asarray(
        (geometric_mean / values) ** _FEEDBACK_EXPONENT, dtype=np.float32
    )
    updated *= float(np.sum(target)) / float(np.sum(updated))
    return validate_target(updated)


class SlmFeedbackTask:
    """Apply candidates, measure exact qCMOS cycles, and retain the best phase."""

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
        target: object,
        target_path: str | Path,
        pulse_sequence: PulseSequence,
        pulse_path: str | Path,
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
        frozen_target = validate_target(target)
        if frozen_target.shape != slm.shape_yx:
            raise ValueError("target shape differs from the selected SLM")
        model = calibration.select_model(ReadoutModelKind.BOX)
        response = np.asarray(model.bright_mean) - np.asarray(model.dark_mean)
        valid = calibration.site_map.valid_sites & model.usable_sites
        if not np.all(valid) or not np.all(np.isfinite(response)) or np.any(response <= 0.0):
            raise ValueError("all 35 sites require finite usable dark/bright PSF calibration")
        self._rows, self._columns = _support(frozen_target, calibration)
        self.camera, self.sequencer, self.slm = camera, sequencer, slm
        self.camera_key, self.sequencer_key, self.slm_key = camera_key, sequencer_key, slm_key
        self.signal_plane, self.calibration, self.model = signal_plane, calibration, model
        self._occupancy = OccupancyProcessor(
            calibration,
            calibration_path=calibration_path,
            producer=self.instance_id,
            model_kind=model.kind,
        )
        self.target, self.sequence = frozen_target, pulse_sequence
        self.calibration_path = Path(calibration_path).expanduser().resolve()
        self.target_path, self.pulse_path = Path(target_path).expanduser().resolve(), Path(pulse_path).expanduser().resolve()
        self.shots = int(shots_per_candidate)
        self.validation_shots = int(validation_shots)
        self.max_updates = int(max_updates)
        if self.shots < 10 or self.validation_shots < 10 or self.max_updates < 1:
            raise ValueError("feedback needs at least 10 coarse/validation shots and one update")
        contract = calibration.frame_contract
        if contract.camera_id is not None and self.camera_key != contract.camera_id:
            raise ValueError(
                f"calibration belongs to camera {contract.camera_id!r}, not {self.camera_key!r}"
            )
        height, width = contract.image_shape
        site_mask = np.zeros((height, width), dtype=bool)
        windows: list[tuple[slice, slice]] = []
        radius = int(model.integration_half_width)
        for center_x, center_y in np.asarray(
            calibration.site_map.centers_xy, dtype=float
        ):
            x, y = int(round(float(center_x))), int(round(float(center_y)))
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            site_mask[y0:y1, x0:x1] = True
            windows.append((slice(y0, y1), slice(x0, x1)))
        self._site_mask, self._site_windows = site_mask, tuple(windows)
        self.artifact_directory = directory

    def _apply_exact(self, phase: object) -> np.ndarray:
        expected = canonical_phase(phase, self.slm.shape_yx)
        applied = self.slm.apply_phase(expected)
        if not np.array_equal(applied, expected) or not np.array_equal(self.slm.last_commanded_phase, expected):
            raise RuntimeError("SLM did not confirm the commanded canonical phase")
        return applied

    def _assert_camera_contract(self, actual: object) -> None:
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
            and (contract.sensor_shape is None or tuple(actual.sensor_shape_yx) == contract.sensor_shape)
            and (expected_roi is None or actual_roi == expected_roi)
            and np.isclose(
                float(actual.exposure_seconds),
                float(contract.exposure_seconds),
                rtol=1e-9,
                atol=0.0,
            )
            and (contract.readout_mode is None or actual.readout_mode == contract.readout_mode)
        )
        if not matches:
            raise ValueError("selected camera working point differs from the frozen calibration")

    def _saturated_sites(self, image: np.ndarray) -> tuple[int, ...]:
        if not np.issubdtype(image.dtype, np.integer):
            return ()
        saturated = image == np.iinfo(image.dtype).max
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
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
        contract = self.calibration.frame_contract
        if contract.exposure_seconds is None:
            raise ValueError("calibration does not record its readout exposure")
        requested = self.shots if shots is None else int(shots)
        if requested < 2:
            raise ValueError("qCMOS fluorescence statistics require at least two shots")
        mean = np.zeros(self.calibration.n_sites, dtype=float)
        sum_squared_deviations = np.zeros_like(mean)
        occupied_counts = np.zeros(self.calibration.n_sites, dtype=np.int64)
        saturated_sites: set[int] = set()
        missing_sites: set[int] = set()
        measured = 0
        while measured < requested:
            _check_cancelled(context)
            chunk = min(_SHOT_CHUNK, requested - measured)
            node = CameraMeasurementNode(
                camera=self.camera,
                request=CameraMeasurementRequest(
                    camera_key=self.camera_key,
                    exposure_seconds=float(contract.exposure_seconds),
                    roi_xywh=contract.roi_xywh,
                    repeat=chunk,
                    frames_per_cycle=3,
                    # The calibration's own numbers: its thresholds apply to
                    # frames in the unit they were fitted on.
                    photoelectrons=reads_photoelectrons(self.calibration),
                ),
                signal_plane=self.signal_plane,
                producer=self.instance_id,
            )
            source = CameraCycleSource(node)
            source.open(context, cycles=chunk)
            try:
                self.sequencer.safe()
                arm_sequencer(self.sequencer, pulse)
                # One empty scan row is one complete authored pulse.  Sweeping
                # that row gives every shot its own cooling/load event on the
                # real board and the virtual world, without a simulation-only
                # path or a nested timeline repeat.
                self.sequencer.write_scan_table(((),), sweeps=chunk)
                source.arm(pulse.program)
                actual = node.actual_working_point
                if actual is None:
                    raise RuntimeError("camera did not freeze its actual working point")
                self._assert_camera_contract(actual)
                _check_cancelled(context)
                self.sequencer.fire()
                for _local_shot in range(chunk):
                    _check_cancelled(context)
                    value = source.next_value(context)
                    image = np.asarray(value.snapshot.block.values)[0, _READOUT_FRAME]
                    saturated_sites.update(self._saturated_sites(image))
                    occupancy = self._occupancy.evaluate(value)
                    counts = np.asarray(occupancy["counts"].snapshot.block.values)[
                        0, _READOUT_FRAME
                    ]
                    valid = np.asarray(
                        occupancy["valid"].snapshot.block.values, dtype=bool
                    )[0, _READOUT_FRAME]
                    occupied = np.asarray(
                        occupancy["occupied"].snapshot.block.values, dtype=bool
                    )[0, _READOUT_FRAME]
                    missing_sites.update(
                        int(index)
                        for index in np.flatnonzero(~valid | ~np.isfinite(counts))
                    )
                    # Calibration defines both facts used here: ``occupied``
                    # says which readouts contain one atom, and ``counts`` is
                    # the same per-site feature whose occupied training mean
                    # is persisted as ``bright_mean``.  The feedback observable
                    # is that feature's mean over occupied shots only.  Dividing
                    # each site by its own bright response would erase exactly
                    # the brightness non-uniformity the operator asked to
                    # correct; treating empty shots as zero would instead turn
                    # this into a loading-probability measurement.
                    sample = np.asarray(counts, dtype=float)
                    usable = valid & np.isfinite(sample) & occupied
                    if np.any(usable):
                        next_counts = occupied_counts[usable] + 1
                        delta = sample[usable] - mean[usable]
                        mean[usable] += delta / next_counts
                        sum_squared_deviations[usable] += delta * (
                            sample[usable] - mean[usable]
                        )
                        occupied_counts[usable] = next_counts
                    measured += 1
                    context.report_progress(
                        f"Reading occupied-shot qCMOS brightness for candidate {iteration + 1}",
                        current=measured,
                        total=requested,
                    )
                wait_for_board(self.sequencer, context)
            finally:
                try:
                    self.sequencer.safe()
                finally:
                    source.close()
        insufficient = occupied_counts < 2
        missing_sites.update(int(index) for index in np.flatnonzero(insufficient))
        variance = np.full_like(mean, np.nan)
        enough = ~insufficient
        variance[enough] = (
            sum_squared_deviations[enough] / (occupied_counts[enough] - 1)
        )
        standard_error = np.full_like(mean, np.nan)
        standard_error[enough] = np.sqrt(
            variance[enough] / occupied_counts[enough]
        )
        missing = np.fromiter(missing_sites, dtype=int)
        if missing.size:
            mean[missing] = np.nan
            standard_error[missing] = np.nan
        return (
            mean,
            standard_error,
            tuple(sorted(saturated_sites)),
            tuple(sorted(missing_sites)),
        )

    def execute(self, context: object) -> dict[str, object]:
        incoming = np.array(self.slm.last_commanded_phase, copy=True)
        try:
            exposure = self.calibration.frame_contract.exposure_seconds
            if exposure is None:
                raise ValueError("calibration does not record its readout exposure")
            pulse = resolve_pulse(
                self.sequence,
                path=self.pulse_path,
                board=self.sequencer.describe(),
                # Pulse timing is authored by the operator.  The calibration's
                # exposure is a sensor integration readback, not a probe gate.
                api_values={},
            )
            current_target = self.target
            current_phase, _ = solve_phase(
                current_target,
                initial_phase=incoming,
                iterations=_SOLVE_ITERATIONS,
                stop_requested=context.cancel_requested,
            )
            baseline = None
            coarse_best = float("inf")
            history: list[dict[str, object]] = []
            accepted = None
            for iteration in range(self.max_updates):
                _check_cancelled(context)
                self._apply_exact(current_phase)
                fluorescence, error, saturated_sites, missing_sites = self._measure(
                    pulse, context, iteration
                )
                saturated = bool(saturated_sites)
                missing = bool(missing_sites)
                total = float(np.sum(fluorescence))
                if baseline is None and np.all(fluorescence > 0.0) and not saturated:
                    baseline = total
                valid = bool(
                    baseline is not None
                    and not saturated
                    and not missing
                    and np.all(np.isfinite(fluorescence))
                    and np.all(fluorescence > 0.0)
                    and total >= 0.9 * baseline
                )
                score = _ratio(fluorescence) if valid else float("inf")
                history.append({
                    "iteration": iteration + 1,
                    "shots": self.shots,
                    "valid": valid,
                    "saturated": saturated,
                    "saturated_sites": list(saturated_sites),
                    "uniformity_ratio": None if not valid else score,
                    "fluorescence": _json_floats(fluorescence),
                    "standard_error": _json_floats(error),
                    "missing_sites": list(missing_sites),
                    "total_relative_to_baseline": (
                        total / baseline
                        if baseline is not None and np.isfinite(total)
                        else None
                    ),
                })
                if valid:
                    coarse_best = min(coarse_best, score)
                context.report_progress(
                    f"qCMOS fluorescence ratio {score:.5f}; "
                    f"best {coarse_best:.5f}"
                    if np.isfinite(coarse_best)
                    else "qCMOS fluorescence invalid"
                )
                # A coarse point estimate is not a retained best.  Every
                # threshold-crossing candidate remains eligible for its own
                # independent validation until one actually passes.
                if valid and score <= _TARGET_RATIO:
                    _check_cancelled(context)
                    self._apply_exact(current_phase)
                    (
                        validation,
                        validation_error,
                        validation_saturated_sites,
                        validation_missing_sites,
                    ) = self._measure(
                        pulse, context, iteration, shots=self.validation_shots
                    )
                    validation_saturated = bool(validation_saturated_sites)
                    validation_total = float(np.sum(validation))
                    relative_error = validation_error / validation
                    max_relative_error = (
                        float(np.max(relative_error))
                        if np.all(np.isfinite(relative_error))
                        else float("inf")
                    )
                    validation_valid = bool(
                        not validation_saturated
                        and not validation_missing_sites
                        and np.all(np.isfinite(validation))
                        and np.all(validation > 0.0)
                        and max_relative_error <= _VALIDATION_RELATIVE_SEM
                        and validation_total >= 0.9 * baseline
                    )
                    validation_score = _ratio(validation) if validation_valid else float("inf")
                    history[-1]["validation"] = {
                        "shots": self.validation_shots,
                        "valid": validation_valid,
                        "saturated": validation_saturated,
                        "saturated_sites": list(validation_saturated_sites),
                        "uniformity_ratio": None if not validation_valid else validation_score,
                        "maximum_relative_standard_error": (
                            None if not np.isfinite(max_relative_error) else max_relative_error
                        ),
                        "fluorescence": _json_floats(validation),
                        "standard_error": _json_floats(validation_error),
                        "missing_sites": list(validation_missing_sites),
                    }
                    context.report_progress(
                        f"qCMOS validation ratio {validation_score:.5f}; "
                        f"max relative SEM {max_relative_error:.5f}"
                    )
                    if validation_score <= _TARGET_RATIO:
                        accepted = (
                            np.array(current_phase, copy=True),
                            history[-1],
                            validation_score,
                            max_relative_error,
                        )
                        break
                if valid:
                    current_target = _updated_target(current_target, fluorescence, self._rows, self._columns)
                    current_phase, _ = solve_phase(
                        current_target,
                        initial_phase=current_phase,
                        iterations=_SOLVE_ITERATIONS,
                        stop_requested=context.cancel_requested,
                    )
            if accepted is None:
                raise RuntimeError("qCMOS feedback did not reach 1.01 site uniformity")
            # Atomically order Stop against the irreversible terminal pair.
            # A Stop that won restores ``incoming`` through the exception
            # path; after this seal, apply/save either both lead to success or
            # an exception is a real failure rather than a cancellation.
            context.seal_terminal()
            applied = self._apply_exact(accepted[0])
            artifact_path = unique_path(self.artifact_directory, "slm_feedback", ".npz")
            save_phase(artifact_path, applied, {
                "calibration_path": str(self.calibration_path),
                "target_path": str(self.target_path),
                "pulse_path": str(self.pulse_path),
                "named_devices": {"camera": self.camera_key, "sequencer": self.sequencer_key, "slm": self.slm_key},
                "best": accepted[1],
                "history": history,
                "updates": len(history),
            })
            return {
                "artifact_path": artifact_path,
                "best_uniformity": accepted[2],
                "validation_max_relative_standard_error": accepted[3],
                "updates": len(history),
            }
        except BaseException as error:
            try:
                self._apply_exact(incoming)
            except BaseException as restore_error:
                raise RuntimeError(
                    f"SLM feedback failed and the incoming phase could not be restored: {error}"
                ) from restore_error
            raise


__all__ = ["SLM_PHASE_ARTIFACT_CONTRACT", "SlmFeedbackTask"]
